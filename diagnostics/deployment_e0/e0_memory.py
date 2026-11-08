"""E0 analytic memory model for a U-shaped split of a Qwen3 checkpoint (prediction only).

The run decides feasibility by measurement (e0_f.py); this model only predicts where to start the
U-max(M) search and documents the components. Parameter counts are exact from config.json; the
runtime transients follow the transformers 4.51.3 SDPA path, which materialises the GQA-expanded
K/V (all query heads x context) for every layer call (integrations/sdpa_attention.repeat_kv).

  python e0_memory.py            # markdown table for Qwen3-32B at L_F, all memory classes
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e0_spec as S  # noqa: E402

MIB = 1 << 20
BYTES_BF16 = 2
CUDA_CONTEXT_MIB = 400       # assumed CUDA context + cuBLAS handles (measured in F, not assumed there)
ALLOCATOR_SLACK_MIB = 256    # assumed caching-allocator fragmentation allowance


def param_counts(cfg):
    H, I, V = cfg["hidden_size"], cfg["intermediate_size"], cfg["vocab_size"]
    nh, nkv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    layer = (H * nh * hd + 2 * H * nkv * hd + nh * hd * H    # q, k, v, o
             + 2 * hd                                         # q_norm, k_norm
             + 3 * H * I                                      # gate, up, down
             + 2 * H)                                         # input / post-attention RMSNorm
    embed = V * H
    head = 0 if cfg["tie_word_embeddings"] else V * H
    return {"embed": embed, "layer": layer, "norm": H, "lm_head": head,
            "total": embed + head + H + cfg["num_hidden_layers"] * layer}


def kv_bytes_per_token_layer(cfg, elem=BYTES_BF16):
    return 2 * cfg["num_key_value_heads"] * cfg["head_dim"] * elem


def transient_bytes(cfg, n_chunk, ctx, elem=BYTES_BF16):
    """Peak per-layer transient (bytes) for a forward step of n_chunk tokens at total context ctx."""
    H, I = cfg["hidden_size"], cfg["intermediate_size"]
    nh, nkv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    n = n_chunk
    expanded_kv = 2 * nh * ctx * hd                            # repeat_kv(K), repeat_kv(V)
    mask = n * (ctx + 1) if (n > 1 and ctx > n) else 0         # 4D additive mask for chunk > 0
    attn = 2 * n * H + 4 * n * nh * hd + 2 * n * nkv * hd + expanded_kv + mask
    mlp = 2 * n * H + 3 * n * I
    return max(attn, mlp) * elem


def edge_breakdown(k, j, cfg=None, ctx=S.L_F, chunk=S.PREFILL_CHUNK):
    cfg = cfg or S.QWEN3_32B_CONFIG
    p = param_counts(cfg)
    n_layers = k + j
    weights = (p["embed"] + p["lm_head"] + p["norm"] + n_layers * p["layer"]) * BYTES_BF16
    kv = n_layers * kv_bytes_per_token_layer(cfg) * ctx
    # worst prefill chunk: a full chunk ending at the largest multiple of `chunk` below ctx
    last_full_end = ((ctx - S.GEN_TOKENS) // chunk) * chunk
    pre = transient_bytes(cfg, chunk, last_full_end) if last_full_end >= chunk else 0
    dec = transient_bytes(cfg, 1, ctx)
    logits = cfg["vocab_size"] * 4 * 2                         # last-position logits (bf16 + fp32 copy)
    act = max(pre, dec) + logits
    fixed = (CUDA_CONTEXT_MIB + ALLOCATOR_SLACK_MIB) * MIB
    return {"k": k, "j": j, "weights": weights, "kv": kv, "activations": act, "fixed": fixed,
            "total": weights + kv + act + fixed}


def server_breakdown(k, j, cfg=None, ctx=S.L_F, chunk=S.PREFILL_CHUNK):
    cfg = cfg or S.QWEN3_32B_CONFIG
    p = param_counts(cfg)
    n_layers = cfg["num_hidden_layers"] - k - j
    weights = n_layers * p["layer"] * BYTES_BF16
    kv = n_layers * kv_bytes_per_token_layer(cfg) * ctx
    act = max(transient_bytes(cfg, chunk, ctx - chunk), transient_bytes(cfg, 1, ctx))
    fixed = (CUDA_CONTEXT_MIB + ALLOCATOR_SLACK_MIB) * MIB
    return {"layers": n_layers, "weights": weights, "kv": kv, "activations": act, "fixed": fixed,
            "total": weights + kv + act + fixed}


def predict_umax(m, cfg=None, ctx=S.L_F):
    """Largest symmetric (k, k) whose predicted edge total fits budget_mib(m); None if (1,1) does not."""
    best = None
    for k in range(1, cfg["num_hidden_layers"] // 2 if cfg else 32):
        if edge_breakdown(k, k, cfg, ctx)["total"] <= S.budget_mib(m) * MIB:
            best = (k, k)
        else:
            break
    return best


def shards_needed(index_path, layer_ids, extra=("model.embed_tokens.weight", "model.norm.weight",
                                                  "lm_head.weight")):
    wm = json.loads(Path(index_path).read_text())["weight_map"]
    need = set()
    for name, shard in wm.items():
        parts = name.split(".")
        if name in extra or (parts[1] == "layers" and int(parts[2]) in layer_ids):
            need.add(shard)
    return sorted(need)


def edge_layer_ids(k, j, n_layers):
    return list(range(k)) + list(range(n_layers - j, n_layers))


EDGE_COMPETITORS = {   # weights bytes from the HF tree API (S1 PREREG §3); per-token KV bytes, all layers
    "Qwen3-8B-AWQ": (6.1e9, 36 * 4096), "Qwen3-14B-AWQ": (10.0e9, 40 * 4096),
    "Qwen3-32B-AWQ": (19.3e9, 64 * 4096), "Qwen3-8B": (16.4e9, 36 * 4096),
}
VLLM_OVERHEAD_MIB = 1536   # assumed activation profile + CUDA graphs + context (desk check: 1.5 GB)


def edge_competitor_mib(model, ctx=S.L_F, fp8_kv=False):
    w, kvt = EDGE_COMPETITORS[model]
    kv = kvt * ctx * (0.5 if fp8_kv else 1.0)
    return (w + kv) / MIB + VLLM_OVERHEAD_MIB


def markdown():
    cfg = S.QWEN3_32B_CONFIG
    p = param_counts(cfg)
    lines = ["# E0 analytic memory model (prediction; F decides by measurement)", "",
             f"Qwen3-32B exact parameter counts: embedding {p['embed']:,}; per layer {p['layer']:,}; "
             f"LM head {p['lm_head']:,}; total {p['total']:,} = {p['total'] * 2:,} bytes bf16 "
             f"(index total_size {S.QWEN3_32B_TOTAL_BYTES:,}).", "",
             f"Context L_F = {S.L_F} tokens, prefill chunk {S.PREFILL_CHUNK}, KV bf16 "
             f"({kv_bytes_per_token_layer(cfg):,} B per token per layer). Assumed CUDA context "
             f"{CUDA_CONTEXT_MIB} MiB + allocator slack {ALLOCATOR_SLACK_MIB} MiB.", "",
             "| (k, j) | edge layers | weights MiB | KV MiB | activations MiB | fixed MiB | total MiB | "
             "fits 12 (11776) | fits 16 (15872) | fits 24 (24064) | server layers | server total MiB |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in range(1, 12):
        e = edge_breakdown(k, k)
        s = server_breakdown(k, k)
        fits = ["yes" if e["total"] <= S.budget_mib(m) * MIB else "no" for m in S.MEMORY_CLASSES]
        lines.append(f"| ({k}, {k}) | {2 * k} | {e['weights'] / MIB:,.0f} | {e['kv'] / MIB:,.0f} | "
                     f"{e['activations'] / MIB:,.0f} | {e['fixed'] / MIB:,.0f} | {e['total'] / MIB:,.0f} | "
                     + " | ".join(fits) + f" | {s['layers']} | {s['total'] / MIB:,.0f} |")
    lines += ["", "Predicted U-max(M): " + ", ".join(f"{m} GB -> {predict_umax(m, cfg)}" for m in S.MEMORY_CLASSES),
              "", "Edge-resident competitors (vLLM; weights + KV at L_F + assumed 1.5 GiB engine overhead):", "",
              "| model | KV | predicted MiB | fits 12 | fits 16 | fits 24 |", "|---|---|---|---|---|---|"]
    for mname in EDGE_COMPETITORS:
        for fp8 in (False, True):
            if fp8 and mname != "Qwen3-32B-AWQ":
                continue
            v = edge_competitor_mib(mname, fp8_kv=fp8)
            fits = ["yes" if v <= S.budget_mib(m) else "no" for m in S.MEMORY_CLASSES]
            lines.append(f"| {mname} | {'fp8' if fp8 else 'bf16/fp16'} | {v:,.0f} | " + " | ".join(fits) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(markdown())
