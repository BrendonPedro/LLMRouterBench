"""Static memory calculation for E1-OPT (edge and server), reusing E0's analytic memory model.

E1-OPT preallocates the same KV (L_F tokens for the owned layers) as E0 and adds only fixed-size buffers:
the wire staging buffers (2 x max_tokens x hidden x 2 B host-pinned, one device copy) and the graph's static
input/output ([1,1,hidden] x 2). So the per-class feasibility of E0 Stage 1 carries over, up to those bytes.
"""
import sys
import e1_spec as S
sys.path.insert(0, str(S.E0_DIR))
import e0_memory as M  # noqa: E402


def extra_bytes_e1(hidden=S.E0.QWEN3_32B_CONFIG["hidden_size"], max_tokens=S.E0.PREFILL_CHUNK):
    wire_dev = max_tokens * hidden * 2
    graph_bufs = 2 * hidden * 2
    return {"wire_device_bytes": wire_dev, "graph_static_bytes": graph_bufs, "total_device_bytes": wire_dev + graph_bufs}


if __name__ == "__main__":
    cfg = S.E0.QWEN3_32B_CONFIG
    p = M.param_counts(cfg)
    kv_tok_layer = M.kv_bytes_per_token_layer(cfg)
    L = cfg["num_hidden_layers"]
    for k in (1, 3, 5, 9):
        edge_w = (2 * k * p["layer"] + p["embed"] + p["lm_head"]) * 2 if "embed" in p else None
        srv_w = (L - 2 * k) * p["layer"] * 2
        srv_kv = (L - 2 * k) * kv_tok_layer * S.E0.L_F
        print(f"k={k}: server weights {srv_w/2**30:.2f} GiB + KV(L_F) {srv_kv/2**30:.2f} GiB; extra E1 device bytes {extra_bytes_e1()['total_device_bytes']/2**20:.1f} MiB")
