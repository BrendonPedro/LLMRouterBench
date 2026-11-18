"""K16-HWFIT expected memory per host, from the checkpoint's actual tensors.

Weights are summed from frozen/tensor_table.json: every tensor of Qwen/Qwen3-32B at the pinned revision with its
dtype and shape, read from the safetensors headers. On a host, `local_weights()` re-reads the headers of the
local checkpoint, so the preflight compares the prediction with the files that will be loaded.
KV, activation transients and the fixed overhead follow E0's analytic model (deployment_e0/e0_memory.py), which
E0 Stage 1 checked on an RTX 3090 (predicted 22,377 MiB, measured 22,554 MiB at k = 9).

  python -I k16_memory.py            # markdown table for the k = 16 cut
"""
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # python -I: the script dir is not on sys.path
import k16_spec as S  # noqa: E402

sys.path.insert(0, str(S.E0_DIR))
import e0_memory as M0  # noqa: E402

MIB = 1 << 20
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
STAGES = ("server", "edge_twohost", "edge_reference", "edge_inhost")
HOST_RAM_MIN_MIB = 8192            # discrete-GPU host: staging buffers, pinned wire buffers, Python; checked by the preflight


def tensor_table():
    return json.loads((S.K16_DIR / "frozen" / "tensor_table.json").read_text())


def owner(name, k=S.K, j=S.J, n_layers=S.N_LAYERS):
    """'edge' or 'server' for a checkpoint tensor name under the U-shaped cut (k, j)."""
    parts = name.split(".")
    if len(parts) > 2 and parts[0] == "model" and parts[1] == "layers":
        i = int(parts[2])
        return "edge" if (i < k or i >= n_layers - j) else "server"
    return "edge"          # model.embed_tokens, model.norm, lm_head


def role_tensors(role, table, k=S.K, j=S.J, n_layers=S.N_LAYERS):
    return [n for n in table if role == "all" or owner(n, k, j, n_layers) == role]


def read_safetensors_header(path):
    """(header_len, {tensor name: {dtype, shape, data_offsets}}) without reading any weight data."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > 100 * MIB:
            raise ValueError(f"{path}: implausible safetensors header length {n}")
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return n, h


def local_weights(model_dir, k=S.K, j=S.J, n_layers=S.N_LAYERS):
    """Bytes of the tensors each role owns, summed from the headers of the checkpoint in model_dir."""
    d = Path(model_dir)
    idx = d / "model.safetensors.index.json"
    shards = sorted(set(json.loads(idx.read_text())["weight_map"].values())) if idx.exists() else ["model.safetensors"]
    out = {"edge": 0, "server": 0, "largest": {"edge": 0, "server": 0}, "n_tensors": 0, "missing_shards": []}
    for s in shards:
        if not (d / s).exists():
            out["missing_shards"].append(s)
            continue
        for name, t in read_safetensors_header(d / s)[1].items():
            n = DTYPE_BYTES[t["dtype"]]
            for dim in t["shape"]:
                n *= dim
            o = owner(name, k, j, n_layers)
            out[o] += n
            out["largest"][o] = max(out["largest"][o], n)
            out["n_tensors"] += 1
    out["all"] = out["edge"] + out["server"]
    return out


def frozen_weights():
    t = tensor_table()
    by = {r: sum(t[n]["nbytes"] for n in role_tensors(r, t)) for r in ("edge", "server")}
    by["largest"] = {r: max(t[n]["nbytes"] for n in role_tensors(r, t)) for r in ("edge", "server")}
    by["all"] = by["edge"] + by["server"]
    return by


def expected(stage, weights=None, cfg=None, max_len=S.MAX_LEN, chunk=S.PREFILL_CHUNK, static=True, longest_seq=None,
             n_edge_layers=len(S.EDGE_LAYERS), n_server_layers=len(S.SERVER_LAYERS)):
    """Expected peak device memory (bytes) of one process, by component.
    stage: server | edge_twohost | edge_reference (unsplit HF model) | edge_inhost (both segments on the edge host)."""
    assert stage in STAGES, stage
    cfg = cfg or S.E0.QWEN3_32B_CONFIG
    w = weights or frozen_weights()
    kv_tl = M0.kv_bytes_per_token_layer(cfg)
    H, V = cfg["hidden_size"], cfg["vocab_size"]
    longest_seq = longest_seq or (chunk + S.CRITERIA["greedy_tokens"])
    step = max(M0.transient_bytes(cfg, chunk, chunk), M0.transient_bytes(cfg, 1, max_len))
    head_eval = min(S.E0.HEAD_CHUNK, longest_seq) * V * (2 + 4 + 4)      # teacher-forced LM-head chunk: bf16 logits, fp32, log-softmax
    wire = chunk * H * 2
    caches = 2 if static else 1                                         # prefix cache, plus the optional static cache
    # Loading transients. Every loader reads one tensor at a time straight onto the device (split_runtime.Slice and
    # k16_runtime.hf_model): a host staging buffer of that tensor, and for Slice a second device copy while it is
    # written into the parameter. On the GB10 host and device share one pool, so both count (2 x the largest
    # tensor). On a discrete GPU only the device copy counts here; the host side is HOST_RAM_MIN_MIB in the preflight.
    load_edge, load_server = 2 * w["largest"]["edge"], w["largest"]["server"]
    # static mode runs SDPA with an explicit [n, max_len] mask, which materialises the attention scores of a prefill chunk
    scores = 2 * cfg["num_attention_heads"] * chunk * max_len * 2 if static else 0
    c = {"stage": stage}
    if stage == "server":
        c.update(weights=w["server"], kv=n_server_layers * kv_tl * max_len * caches, activations=int(step * S.ACTIVATION_ALLOWANCE) + scores,
                 load_transient=load_server, wire=wire)
    elif stage == "edge_twohost":
        c.update(weights=w["edge"], kv=n_edge_layers * kv_tl * max_len * caches, activations=int(step * S.ACTIVATION_ALLOWANCE) + max(head_eval, scores),
                 load_transient=load_edge, wire=wire)
    elif stage == "edge_reference":
        full = M0.transient_bytes(cfg, longest_seq, longest_seq)
        scores = cfg["num_attention_heads"] * longest_seq * longest_seq * 2      # if SDPA falls back to the math kernel
        c.update(weights=w["all"], kv=(n_edge_layers + n_server_layers) * kv_tl * longest_seq,
                 activations=int((full + scores) * S.ACTIVATION_ALLOWANCE) + head_eval + longest_seq * H * 2,
                 load_transient=load_edge, wire=0)
    else:
        c.update(weights=w["all"], kv=(n_edge_layers + n_server_layers) * kv_tl * max_len,
                 activations=int(step * S.ACTIVATION_ALLOWANCE) + head_eval, load_transient=load_edge, wire=2 * wire)
    c["fixed"] = (M0.CUDA_CONTEXT_MIB + M0.ALLOCATOR_SLACK_MIB) * MIB
    c["total"] = sum(c[x] for x in ("weights", "kv", "activations", "load_transient", "wire", "fixed"))
    margin = S.SERVER_MARGIN_MIB if stage == "server" else S.EDGE_MARGIN_MIB
    c["margin"] = margin * MIB
    c["required_free"] = c["total"] + c["margin"]
    return c


def required_free_mib(stage, **kw):
    return -(-expected(stage, **kw)["required_free"] // MIB)


def markdown():
    w = frozen_weights()
    g = lambda b: f"{b / 2**30:,.2f}"
    lines = [f"# K16-HWFIT expected memory (k = {S.K}; KV preallocated for {S.MAX_LEN} tokens, prefix + static caches)", "",
             f"Checkpoint tensors (frozen/tensor_table.json): edge-owned {w['edge']:,} B, server-owned {w['server']:,} B, "
             f"total {w['all']:,} B.", "",
             "| process | weights GiB | KV GiB | activations GiB | load transient GiB | wire GiB | fixed GiB | expected peak GiB | margin GiB | required free GiB |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    names = {"server": "ASUS server (layers 16-47)", "edge_twohost": "DGX edge, two-host stages", "edge_reference": "DGX, unsplit reference",
             "edge_inhost": "DGX, in-host split check"}
    for st in STAGES:
        c = expected(st)
        lines.append(f"| {names[st]} | " + " | ".join(g(c[x]) for x in ("weights", "kv", "activations", "load_transient", "wire", "fixed", "total", "margin", "required_free")) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(markdown())
