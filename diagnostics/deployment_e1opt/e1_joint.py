"""E1-OPT Stage 4: joint edge/server cells on the physical lab pair (PREREG_E1OPT §8, stage 4). NOT YET RUN.

  serve           server side: the middle segment [k, L-j) of the frozen cut on ONE GPU, E1 runtime (static cache,
                  frozen branch), HELLO verified against the freeze hash, config hash, cut, model id and protocol
  run-cell        edge side: one frozen E0 cell (e0_spec.cells()) with the E1 runtime; refuses any other cell,
                  any cut that is not the frozen one for the cell's class, a missing load state, or a shaping
                  state that does not match the cell
  run-remote-cell edge side: a REMOTE cell through E0's run_openai (vLLM, unchanged) + provenance stamp
  linkcheck       G-link over the E1 protocol against an echo server (E0 §9 thresholds)

Wire rule per forward step (E0 PREREG §2.1): edge -> server: the bf16 residual stream of n tokens + the 72-byte
header; server -> edge: the bf16 residual stream + the header. Nothing else. Sampling stays on the edge. The
server never receives text, token ids, logits, sampled tokens or KV entries; the only positions it learns are
the step's start position and token count, which E0 declared as control metadata.
"""
import json
import re
import socket
import statistics
import sys
import time
from pathlib import Path

import torch

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))   # python -I: the script dir is not on sys.path
import e1_spec as S
sys.path.insert(0, str(S.E0_DIR))
import e0_instrument as I  # noqa: E402
import e0_latency as L  # noqa: E402
import split_runtime as E0RT  # noqa: E402
import e1_attention  # noqa: E402
import e1_freeze  # noqa: E402
import e1_protocol as P  # noqa: E402
import e1_provenance as PV  # noqa: E402
from e1_equivalence import device_identity, GraphCaptureError  # noqa: E402
from e1_segment import USplitEdge, LocalMiddle, edge_slice, server_slice  # noqa: E402


def resolve_cell_cut(cell, f_results):
    """(k, j, cap_class) for a SPLIT cell of the frozen grid; the cut is E0's fixed one, never an argument."""
    path = cell["path"]
    if not path.startswith("SPLIT"):
        raise ValueError(f"cell {cell['id']} is not a SPLIT cell ({path}); REMOTE cells use run-remote-cell, EDGE cells are Stage-3 profiles")
    if path == "SPLIT U-min":
        return S.U_MIN[0], S.U_MIN[1], min(S.E0.MEMORY_CLASSES)
    m = int(re.search(r"U-max\((\d+)\)", path).group(1))
    k, j = S.FIXED_CUTS[m]
    if f_results is not None:
        um = f_results["u_max"].get(str(m))
        if um is None:
            raise ValueError(f"class {m} is not evaluable on this edge (its frozen cut did not fit): cell {cell['id']} is dropped")
        if tuple(um) != (k, j):
            raise ValueError(f"f_results u_max for {m} GB is {um}, not the frozen cut {(k, j)}")
    return k, j, m


def _freeze(a, role):
    if a.host and not a.dev_mode:
        sys.exit("--host overrides the machine identity and is permitted only with --dev-mode (a stage runs as the machine it is on)")
    rec = e1_freeze.check(a.freeze, role=role, require_tag=not a.dev_mode, host=a.host)
    m = e1_freeze.machine_entry(rec, a.host)
    e1_attention.set_branch(m["branch"]["attention"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return rec, m["branch"]


def serve(a):
    rec, branch = _freeze(a, "server")
    gpu = device_identity(a.device)
    if not a.dev_mode and rec["roles"]["server"].get("gpu_uuid") and gpu["uuid"] != rec["roles"]["server"]["gpu_uuid"]:
        sys.exit(f"this GPU {gpu['uuid']} is not the inventoried server GPU; refusing (one-server-GPU rule)")
    expected = None if a.dev_mode else S.E0.QWEN3_32B_CONFIG
    sl = server_slice(a.model_dir, a.k, a.j, a.device, torch.bfloat16, expected)
    sl.cfg._attn_implementation = "e1_static"
    mid = LocalMiddle(sl, sl.layer_ids, a.max_len, "static")
    if branch["cuda_graphs"] == "on":
        try:
            mid.enable_graph()
        except Exception as e:
            raise GraphCaptureError(f"{type(e).__name__}: {e}") from e
    model_id = f"{S.E0.SPLIT_REPO}@{S.E0.SPLIT_REVISION}" if not a.dev_mode else "dev"
    ident = {"runtime": S.NAME, "freeze_sha256": rec["freeze_sha256"], "config_hash": E0RT.config_hash(sl.cfg), "k": a.k, "j": a.j,
             "model_id": model_id, "protocol_version": S.PROTOCOL_VERSION, "host": socket.gethostname(), "gpu": gpu, "branch": branch,
             "active": [a.k, sl.n_layers - a.j], "mode": "static", "one_server_gpu": True, "profile_role": "screening" if a.dev_mode else "target"}
    log_path = Path(a.log) if a.log else None

    def log(ev):
        if log_path:
            I.append_jsonl(log_path, dict(ev, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
    srv = P.SegmentServer(mid.seg, ident, sl.cfg.hidden_size, S.E0.PREFILL_CHUNK, log=log)
    print(f"E1 segment server ready on {a.host_bind}:{a.port} {json.dumps({k: ident[k] for k in ('freeze_sha256', 'k', 'j', 'active', 'gpu')}, default=str)}", flush=True)
    srv.serve_forever(a.host_bind, a.port, once=a.once, ready_file=a.ready_file)


def run_cell(a):
    rec, branch = _freeze(a, "edge")
    cell = L.cell_by_id(a.cell)
    fres = json.loads(Path(a.f_results).read_text()) if a.f_results else None
    k, j, cap_class = resolve_cell_cut(cell, fres)
    S.validate_cut(cap_class, k, j)   # raises for anything but the frozen cut
    ok, st = L.check_load(a.load_status, cell["load"])
    if not ok:
        sys.exit(f"cell {cell['id']} needs load {cell['load']}: {st}")
    if (cell["rtt_ms"] or (cell["bw_mbps"] or 1000) < 1000) and a.shaping == "none":
        sys.exit(f"cell {cell['id']} needs shaping (netem or app)")
    if a.shaping == "netem" and not L.netem_matches(L.netem_show(a.dev), cell["rtt_ms"], cell["bw_mbps"]):
        sys.exit(f"netem state does not match cell {cell['id']}:\n{L.netem_show(a.dev)}")
    gpu = device_identity(a.device)
    if not a.dev_mode and rec["roles"]["edge"].get("gpu_uuid") and gpu["uuid"] != rec["roles"]["edge"]["gpu_uuid"]:
        sys.exit(f"this GPU {gpu['uuid']} is not the inventoried edge GPU; refusing")
    profile_role = "screening" if a.dev_mode else "target"
    prov = PV.make(rec, "4", "edge", gpu, profile_role, (k, j), cap_class, a.host, extra={"cell": cell["id"]})
    nv = I.NVML(gpu["index"] or 0) if str(a.device).startswith("cuda") else None
    baseline = I.idle_baseline(nv) if nv and nv.available else 0
    cap = I.apply_cap(torch, a.device, S.E0.budget_mib(cap_class), nv, baseline) if nv and nv.available else None
    expected = None if a.dev_mode else S.E0.QWEN3_32B_CONFIG
    sl = edge_slice(a.model_dir, k, j, a.device, torch.bfloat16, expected)
    sl.cfg._attn_implementation = "e1_static"
    host, port = a.server.split(":")
    model_id = f"{S.E0.SPLIT_REPO}@{S.E0.SPLIT_REVISION}" if not a.dev_mode else "dev"
    shaper = P.AppShaper(cell.get("rtt_ms"), cell.get("bw_mbps")) if a.shaping == "app" else None
    mid = P.RemoteMiddle(host, int(port), P.hello_for(sl, model_id, k, j, a.max_len, rec["freeze_sha256"]), a.device, sl.cfg.hidden_size,
                         S.E0.PREFILL_CHUNK, shaper=shaper)
    si = mid.server_identity
    if si.get("freeze_sha256") != rec["freeze_sha256"] or si.get("k") != k or si.get("j") != j or si.get("branch") != rec["machines"].get(rec["roles"]["server"]["host"], {}).get("branch"):
        mid.close(); sys.exit(f"server identity does not match the freeze record: {si}")
    e1 = USplitEdge(sl, k, j, mid, a.max_len, "static")
    if branch["cuda_graphs"] == "on":
        e1.enable_graph()
    wl = L.load_workload(a.workload)
    out = Path(a.out_dir) / f"{cell['id']}.jsonl"
    done = L.done_keys(out)
    sampler = None
    if nv and nv.available:
        sampler = I.Sampler(nv, S.E0.POWER_SAMPLE_S, baseline); sampler.start()
    if a.shaping == "netem":
        I.append_jsonl(out, {"cell": cell["id"], "netem_before": L.netem_show(a.dev), "netem_ok": True})
    ident = {"k": k, "j": j, "cap": cap, "server_identity": si, "edge_software": e1_freeze.versions(), "model_id": model_id, "runtime": S.NAME,
             "mode": "static", "branch": branch, "edge_device": gpu, "freeze_sha256": rec["freeze_sha256"], "provenance": prov}
    n_err = n_req = 0
    for req in L.requests_for(cell["id"], cell["block"], wl, a.with_depth):
        if a.dev_max_requests and n_req >= a.dev_max_requests:
            break
        if (cell["id"], req["prompt"]["id"], req["rep"]) in done:
            continue
        n_req += 1
        r = L.base_record(a, cell, req, "SPLIT", ident)
        b0, b1 = mid.bytes_sent, mid.bytes_recv
        t0 = time.perf_counter()
        try:
            if nv and nv.available:
                torch.cuda.reset_peak_memory_stats(a.device)
            toks = e1.generate(req["prompt"]["ids"], S.E0.GEN_TOKENS, S.E0.PREFILL_CHUNK, force_len=True)
            steps = e1.steps
            for s in steps:
                s["phase"] = "prefill" if (s["n"] > 1 or s["start"] < req["prompt"]["n_in"]) else "decode"
            link_ns = sum(s.get("net_ns", 0) for s in steps)
            r.update(ok=len(toks) == S.E0.GEN_TOKENS, token_ns=e1.token_ns, steps=steps, bytes_out=mid.bytes_sent - b0, bytes_in=mid.bytes_recv - b1,
                     link_ns_total=link_ns, server_compute_ns_total=sum(s.get("server_compute_ns", 0) for s in steps),
                     edge_compute_ns_total=sum(s.get("edge_head_ns", 0) + s.get("edge_tail_ns", 0) + s.get("lm_head_ns", 0) for s in steps))
            if nv and nv.available:
                r["alloc"] = I.allocator_stats(torch, a.device)
        except Exception as e:
            n_err += 1
            r.update(ok=False, error=f"{type(e).__name__}: {e}")
        if sampler:
            r["nvml"] = sampler.summary(t0, time.perf_counter())
        I.append_jsonl(out, r)
        if not r["ok"] and "TransportError" in str(r.get("error", "")):
            break   # fail closed: a dropped or malformed exchange ends the cell; the failed row stays (E0 §5.5)
    if sampler:
        sampler.stop()
    if a.shaping == "netem":
        I.append_jsonl(out, {"cell": cell["id"], "netem_after": L.netem_show(a.dev)})
    mid.close()
    print(json.dumps({"cell": cell["id"], "requests": n_req, "errors": n_err, "profile_role": profile_role}))
    sys.exit(1 if n_err else 0)


def run_remote_cell(a):
    rec, branch = _freeze(a, "edge")
    cell = L.cell_by_id(a.cell)
    if cell["path"] != "REMOTE":
        sys.exit(f"{cell['id']} is not a REMOTE cell")
    gpu = device_identity(a.device)
    prov = PV.make(rec, "4", "edge", gpu, "screening" if a.dev_mode else "target", None, None, a.host, extra={"cell": cell["id"], "engine": "vllm"})
    ok, st = L.check_load(a.load_status, cell["load"])
    if not ok:
        sys.exit(f"cell {cell['id']} needs load {cell['load']}: {st}")
    out = Path(a.out_dir) / f"{cell['id']}.jsonl"
    a.nvml = False; a.served_model = a.served_model; a.edge_class = None
    n_err = L.run_openai(a, cell, out, "REMOTE")
    PV.stamp_jsonl(out, prov)
    sys.exit(1 if n_err else 0)


def linkcheck(a):
    rec, branch = _freeze(a, "edge")
    H = S.E0.QWEN3_32B_CONFIG["hidden_size"] if not a.dev_mode else a.dev_hidden
    host, port = a.server.split(":")
    hello = {"model_id": None, "config_hash": None, "freeze_sha256": rec["freeze_sha256"], "k": 1, "j": 1, "n_layers": 64, "hidden": H,
             "max_len": a.max_len, "wire_dtype": "bfloat16", "protocol_version": S.PROTOCOL_VERSION, "runtime": S.NAME}
    mid = P.RemoteMiddle(host, int(port), hello, "cpu", H, S.E0.PREFILL_CHUNK)
    mid.reset()
    x1 = torch.zeros((1, 1, H), dtype=torch.bfloat16); xb = torch.zeros((1, S.E0.PREFILL_CHUNK, H), dtype=torch.bfloat16)
    rt = []
    for i in range(a.n):
        mid.forward(x1, i); rt.append(mid.last["rt_ns"] / 1e6)
    mid.reset()
    big = []
    for i in range(a.n_big):
        mid.forward(xb, i * S.E0.PREFILL_CHUNK); big.append(mid.last["rt_ns"] / 1e9)
    mid.close()
    bits = 2 * (P.HDR.size + S.E0.PREFILL_CHUNK * H * 2) * 8
    q = sorted(rt)
    res = {"experiment": S.NAME, "stage": "4-linkcheck", "server": a.server, "rt_ms_p50": statistics.median(rt), "rt_ms_p95": q[int(0.95 * (len(q) - 1))],
           "bulk_mbps_median": bits / statistics.median(big) / 1e6, "n": a.n, "n_big": a.n_big, "freeze_sha256": rec["freeze_sha256"],
           "edge": socket.gethostname(), "server_identity": mid.server_identity}
    res["joint_allowed"] = res["rt_ms_p50"] <= S.E0.LINK_MAX_RTT_P50_MS and res["bulk_mbps_median"] >= S.E0.LINK_MIN_BULK_MBPS
    if a.out:
        I.write_json(a.out, res)
    print(json.dumps({k: res[k] for k in ("rt_ms_p50", "rt_ms_p95", "bulk_mbps_median", "joint_allowed")}, indent=1))
    return res


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    wl = str(S.E0_DIR / "frozen" / "workload.json")

    def common(p):
        p.add_argument("--freeze", required=True); p.add_argument("--host", default=None); p.add_argument("--device", default="cuda:0")
        p.add_argument("--dev-mode", action="store_true"); p.add_argument("--max-len", type=int, default=S.E0.L_F)
    s = sub.add_parser("serve"); common(s)
    s.add_argument("--model-dir", required=True); s.add_argument("--class", dest="mem_class", type=int, choices=S.E0.MEMORY_CLASSES, required=True)
    s.add_argument("--k", type=int, default=None); s.add_argument("--j", type=int, default=None)
    s.add_argument("--host-bind", default="0.0.0.0"); s.add_argument("--port", type=int, default=29500); s.add_argument("--once", action="store_true")
    s.add_argument("--ready-file", default=None); s.add_argument("--log", default=None)
    r = sub.add_parser("run-cell"); common(r)
    r.add_argument("--cell", required=True); r.add_argument("--model-dir", required=True); r.add_argument("--server", required=True)
    r.add_argument("--f-results", default=None); r.add_argument("--out-dir", required=True); r.add_argument("--workload", default=wl)
    r.add_argument("--shaping", choices=["none", "netem", "app"], default="none"); r.add_argument("--dev", default="eth0")
    r.add_argument("--load-status", default=None); r.add_argument("--with-depth", type=lambda s_: s_ == "1", default=None)
    r.add_argument("--dev-max-requests", type=int, default=0)
    q = sub.add_parser("run-remote-cell"); common(q)
    q.add_argument("--cell", required=True); q.add_argument("--base-url", required=True); q.add_argument("--served-model", default="Qwen3-32B")
    q.add_argument("--out-dir", required=True); q.add_argument("--workload", default=wl); q.add_argument("--shaping", choices=["none", "netem", "app"], default="none")
    q.add_argument("--dev", default="eth0"); q.add_argument("--load-status", default=None); q.add_argument("--with-depth", type=lambda s_: s_ == "1", default=None)
    lc = sub.add_parser("linkcheck"); common(lc)
    lc.add_argument("--server", required=True); lc.add_argument("--n", type=int, default=1000); lc.add_argument("--n-big", type=int, default=20)
    lc.add_argument("--out", default=None); lc.add_argument("--dev-hidden", type=int, default=64)
    a = ap.parse_args(argv)
    if a.cmd == "serve":
        if a.k is None:
            a.k, a.j = S.FIXED_CUTS[a.mem_class]
        if (a.k, a.j) != S.FIXED_CUTS[a.mem_class] and (a.k, a.j) != S.U_MIN and not a.dev_mode:
            sys.exit("cut override is not permitted in the experiment path (PREREG §3)")
    {"serve": serve, "run-cell": run_cell, "run-remote-cell": run_remote_cell, "linkcheck": linkcheck}[a.cmd](a)


if __name__ == "__main__":
    main()
