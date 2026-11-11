"""E1-OPT Stage 2 and Stage 3 profilers: fresh measurements on the physical target machines (PREREG_E1OPT §8).

Every command requires --freeze (checked first) and writes rows in E0's schema with an E1 `provenance` object
(e1_provenance). Rows are "target" only when the freeze check passed with the tag, the cut is the frozen cut and
no dev flag is set; otherwise they are "screening" and the decision gates reject them.

  fit-split               Stage 3: E0 §5.2 memory-fit attempt of the edge slice at the frozen cut under the class cap,
                          E1 runtime (static cache, frozen branch), depth prompt + 256 decode tokens, echo middle;
                          child process, exit status recorded (E0 H1 semantics)
  fit-competitor          Stage 3: E0's `e0_f.py edge-fit` unchanged (vLLM competitor under the cap) + provenance stamp
  profile-edge-segment    Stage 3: E0 run-split semantics with the E1 runtime against an echo stub -> P-edge-U(k,k)-cap<M>.jsonl
  profile-server-segment  Stage 2: server-segment compute profile with synthetic activations -> P-server-U(k,k)-load0.jsonl
  summarize-fit           Stage 3: f_results.json in E0's schema (u_max = frozen cut if it fits; eligibility; defects)
"""
import json
import os
import socket
import subprocess
import sys
import threading
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
import e0_f as F  # noqa: E402  (read-only: failure_kind, edge-fit via subprocess)
import e1_attention  # noqa: E402
import e1_freeze  # noqa: E402
import e1_protocol as P  # noqa: E402
import e1_provenance as PV  # noqa: E402
from e1_equivalence import device_identity, GraphCaptureError  # noqa: E402
from e1_segment import USplitEdge, LocalMiddle, edge_slice, server_slice  # noqa: E402

MIB = I.MIB


def _prep(a, role, stage):
    """Freeze check, branch, device identity, cut validation -> (freeze, branch, gpu, profile_role, provenance)."""
    if a.host and not a.dev_mode:
        sys.exit("--host overrides the machine identity and is permitted only with --dev-mode (a stage runs as the machine it is on)")
    rec = e1_freeze.check(a.freeze, role=role, require_tag=not a.dev_mode, host=a.host)
    m = e1_freeze.machine_entry(rec, a.host)
    branch = m["branch"]
    e1_attention.set_branch(branch["attention"])
    gpu = device_identity(a.device)
    k, j = a.k, a.j
    role_of_cut = S.validate_cut(a.mem_class, k, j, dev_mode=a.dev_mode) if a.mem_class else ("target" if (k, j) in S.FIXED_CUTS.values() else S.validate_cut(0, k, j, a.dev_mode))
    profile_role = "screening" if (a.dev_mode or getattr(a, "screening", False) or role_of_cut != "target") else "target"
    roles = rec["roles"]
    if profile_role == "target" and roles.get(role, {}).get("gpu_uuid") and gpu["uuid"] != roles[role]["gpu_uuid"]:
        sys.exit(f"this GPU {gpu['uuid']} is not the inventoried {role} GPU {roles[role]['gpu_uuid']}; refusing a target profile")
    prov = PV.make(rec, stage, role, gpu, profile_role, (k, j), a.mem_class, a.host,
                   extra={"reason_screening": "dev_mode" if a.dev_mode else ("flag" if getattr(a, "screening", False) else None)} if profile_role == "screening" else None)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return rec, branch, gpu, profile_role, prov


def _echo_server(freeze_sha, k, j, hidden, chunk):
    ident = {"freeze_sha256": freeze_sha, "config_hash": None, "k": k, "j": j, "model_id": None, "runtime": S.NAME, "echo": True}
    srv = P.SegmentServer(None, ident, hidden, chunk)
    s0 = socket.socket(); s0.bind(("127.0.0.1", 0)); port = s0.getsockname()[1]; s0.close()
    ready = threading.Event()
    t = threading.Thread(target=srv.serve_forever, args=("127.0.0.1", port, ready, True), daemon=True, name="e1-echo"); t.start(); ready.wait(10)
    return srv, t, port


def _edge_runtime(a, rec, branch, nv_cap=None):
    expected = None if a.dev_mode else S.E0.QWEN3_32B_CONFIG
    sl = edge_slice(a.model_dir, a.k, a.j, a.device, torch.bfloat16, expected)
    sl.cfg._attn_implementation = "e1_static"
    H = sl.cfg.hidden_size
    srv, t, port = _echo_server(rec["freeze_sha256"], a.k, a.j, H, S.E0.PREFILL_CHUNK)
    model_id = f"{S.E0.SPLIT_REPO}@{S.E0.SPLIT_REVISION}" if not a.dev_mode else "dev"
    mid = P.RemoteMiddle("127.0.0.1", port, P.hello_for(sl, model_id, a.k, a.j, a.max_len, rec["freeze_sha256"]), a.device, H, S.E0.PREFILL_CHUNK)
    e1 = USplitEdge(sl, a.k, a.j, mid, a.max_len, "static")
    if branch["cuda_graphs"] == "on":
        try:
            e1.enable_graph()
        except Exception as e:
            raise GraphCaptureError(f"{type(e).__name__}: {e}") from e
    return sl, mid, e1, srv, t


def _phase_steps(steps, n_in):
    for s in steps:
        s["phase"] = "prefill" if (s["n"] > 1 or s["start"] < n_in) else "decode"
    return steps


# --------------------------------------------------------------------------- #
# Stage 3: memory fit of the edge slice (E0 §5.2 with the E1 runtime)         #
# --------------------------------------------------------------------------- #
def fit_split_child(a):
    import faulthandler
    faulthandler.enable(all_threads=True)
    rec, branch, gpu, profile_role, prov = _prep(a, "edge", "3")
    nv = F._DryNVML() if a.cpu_dry else I.NVML(gpu["index"] or 0)
    if not nv.available:
        sys.exit(f"NVML unavailable: {getattr(nv, 'error', '')}")
    baseline = I.idle_baseline(nv)
    budget = S.E0.budget_mib(a.mem_class)
    astats = (lambda: {}) if a.cpu_dry else (lambda: I.allocator_stats(torch, a.device))
    reset_peak = (lambda: None) if a.cpu_dry else (lambda: torch.cuda.reset_peak_memory_stats(a.device))
    out = {"kind": "split-fit", "experiment": S.NAME, "dry": bool(a.cpu_dry), "class": a.mem_class, "k": a.k, "j": a.j, "budget_mib": budget,
           "L_F": a.max_len, "gpu": nv.name(), "device": gpu, "branch": branch, "mode": "static", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "provenance": prov}
    sampler = I.Sampler(nv, S.E0.MEM_SAMPLE_S, baseline); sampler.start()
    phase, mid, t = "cap", None, None
    try:
        out["cap"] = {} if a.cpu_dry else I.apply_cap(torch, a.device, budget, nv, baseline)
        phase = "load"
        sl, mid, e1, srv, t = _edge_runtime(a, rec, branch)
        out["weights_mib"] = sl.param_bytes / MIB
        out["kv_mib"] = (e1.head.cache.nbytes() + e1.tail.cache.nbytes()) / MIB
        out["after_load"] = astats(); reset_peak()
        wl = L.load_workload(a.workload)
        ids = next(p for p in wl["prompts"] if p["group"] == "depth")["ids"]
        if a.dev_depth_tokens:
            ids = ids[: a.dev_depth_tokens]
        e1.reset()
        phase = "prefill"
        t0 = time.perf_counter()
        ids_t = torch.tensor([ids], device=a.device)
        for s in range(0, len(ids), S.E0.PREFILL_CHUNK):
            h = e1.forward_hidden(ids_t[:, s:s + S.E0.PREFILL_CHUNK], s)
        t1 = time.perf_counter()
        out["prefill"] = dict(astats(), nvml=sampler.summary(t0, t1), seconds=t1 - t0)
        reset_peak()
        phase = "decode"
        pos = len(ids)
        with torch.inference_mode():
            for _ in range(S.E0.GEN_TOKENS if not a.dev_depth_tokens else 8):
                tok = int(torch.argmax(e1.sl.lm_head(h[:, -1:, :])[0, -1].float()))
                h = e1.forward_hidden(torch.tensor([[tok]], device=a.device), pos); pos += 1
        t2 = time.perf_counter()
        out["decode"] = dict(astats(), nvml=sampler.summary(t1, t2), seconds=t2 - t1)
        out["final_context"] = pos
        mid.close(); mid = None
        phase = "done"
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"; out["error_phase"] = phase
        out["error_kind"] = "memory" if F.failure_kind(out["error"]) == "memory" else "harness"
        try:
            out["at_error"] = astats(); out["memory_summary"] = torch.cuda.memory_summary(a.device, abbreviated=True)[-3000:]
        except Exception:
            pass
    out["nvml_total"] = sampler.stop()
    if t is not None:
        if mid is not None:
            mid.close()
        t.join(30)
    live = [th.name for th in threading.enumerate() if th is not threading.main_thread()]
    out["shutdown"] = {"echo_thread_joined": t is None or not t.is_alive(), "live_threads": live}
    nontorch = out.get("cap", {}).get("nontorch_mib", 0)
    reserved = max(out.get(p, {}).get("reserved_peak_mib", 0) for p in ("after_load", "prefill", "decode"))
    out["peak_device_view_mib"] = out["nvml_total"].get("peak_used_mib")
    out["peak_allocator_view_mib"] = nontorch + reserved
    out["pass"] = bool(phase == "done" and (a.cpu_dry or out["peak_device_view_mib"] is not None)
                       and (a.cpu_dry or (out["peak_device_view_mib"] <= budget and out["peak_allocator_view_mib"] <= budget)))
    I.write_json(a.out, out)
    print(json.dumps({k: out.get(k) for k in ("class", "k", "pass", "error", "peak_device_view_mib", "peak_allocator_view_mib", "budget_mib")}), flush=True)
    sys.exit(0 if out["pass"] else 3)


def fit_split(a):
    """Parent: one child attempt, exit status recorded (E0 HARNESS_FIXES H1)."""
    f = Path(a.out_dir) / f"split_fit_c{a.mem_class}_k{a.k}.json"
    if f.exists():
        sys.exit(f"{f} exists; Stage 3 re-runs start from an empty directory")
    f.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, __file__, "_fit-split-child", "--freeze", a.freeze, "--model-dir", a.model_dir, "--class", str(a.mem_class),
           "--k", str(a.k), "--j", str(a.j), "--device", a.device, "--out", str(f), "--workload", a.workload, "--max-len", str(a.max_len)]
    if a.dev_mode: cmd.append("--dev-mode")
    if a.cpu_dry: cmd.append("--cpu-dry")
    if a.host: cmd += ["--host", a.host]
    if a.dev_depth_tokens: cmd += ["--dev-depth-tokens", str(a.dev_depth_tokens)]
    rc = subprocess.run(cmd).returncode
    rec = json.loads(f.read_text()) if f.exists() else {"kind": "split-fit", "class": a.mem_class, "k": a.k, "j": a.j, "pass": False,
                                                           "error": "HarnessError: the child wrote no record"}
    rec["exit_status"] = rc
    expected = 0 if rec.get("pass") else 3
    if rc != expected or not f.exists():
        rec["error_kind"] = "harness"; rec["harness_error"] = f"child exit status {rc} (expected {expected} for pass={rec.get('pass')})"
    I.write_json(f, rec)
    print(json.dumps({k: rec.get(k) for k in ("class", "k", "pass", "exit_status", "error_kind")}))
    if rec.get("error_kind") == "harness":
        sys.exit(f"HARNESS DEFECT: {rec.get('harness_error') or rec.get('error')} -> Stage 3 INVALID")


# --------------------------------------------------------------------------- #
# Stage 3: competitor fit and profile (E0's e0_f.py edge-fit, unchanged)      #
# --------------------------------------------------------------------------- #
def fit_competitor(a):
    a.k, a.j = S.FIXED_CUTS[a.mem_class]
    rec, branch, gpu, profile_role, prov = _prep(a, "edge", "3")
    prov = dict(prov, cut=None, competitor=a.model, engine="vllm")
    if (gpu["index"] or 0) != 0 and torch.cuda.device_count() > 1:
        sys.exit("E0's edge-fit samples NVML index 0; the inventoried edge GPU is not index 0 on a multi-GPU machine. "
                 "Run with the edge GPU as the only visible device at the system level (not CUDA_VISIBLE_DEVICES) or stop.")
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(S.E0_DIR / "e0_f.py"), "edge-fit", "--model", a.model, "--class", str(a.mem_class), "--vllm-bin", a.vllm_bin,
           "--out-dir", str(out), "--profile-out", str(out), "--workload", a.workload]
    rc = subprocess.run(cmd).returncode
    j = out / f"edge_fit_{a.model}_c{a.mem_class}.json"
    if j.exists():
        PV.stamp_json(j, dict(prov, edge_fit_exit_status=rc))
    p = out / f"P-edge-{a.model}-cap{a.mem_class}.jsonl"
    n = PV.stamp_jsonl(p, prov) if p.exists() else 0
    with open(out / "exit_status.log", "a") as f:
        f.write(f"rc={rc} e1_profile fit-competitor --model {a.model} --class {a.mem_class}\n")
    print(json.dumps({"model": a.model, "class": a.mem_class, "edge_fit_rc": rc, "profile_rows_stamped": n, "profile_role": profile_role}))
    sys.exit(0 if rc == 0 else rc)


# --------------------------------------------------------------------------- #
# Stage 3: edge-segment profile (E0 run-split semantics, E1 runtime)          #
# --------------------------------------------------------------------------- #
def profile_edge_segment(a):
    rec, branch, gpu, profile_role, prov = _prep(a, "edge", "3")
    nv = I.NVML(gpu["index"] or 0) if str(a.device).startswith("cuda") else None
    baseline = I.idle_baseline(nv) if nv and nv.available else 0
    cap = I.apply_cap(torch, a.device, S.E0.budget_mib(a.mem_class), nv, baseline) if nv and nv.available else None
    sl, mid, e1, srv, t = _edge_runtime(a, rec, branch)
    wl = L.load_workload(a.workload)
    cell = {"id": a.label or f"P-edge-U({a.k},{a.j})-cap{a.mem_class}", "block": "profile", "rtt_ms": None, "bw_mbps": None, "load": 0}
    out = Path(a.out_dir) / f"{cell['id']}.jsonl"
    done = L.done_keys(out)
    sampler = None
    if nv and nv.available:
        sampler = I.Sampler(nv, S.E0.POWER_SAMPLE_S, baseline); sampler.start()
    ident = {"k": a.k, "j": a.j, "cap": cap, "server_identity": mid.server_identity, "edge_software": e1_freeze.versions(),
             "model_id": f"{S.E0.SPLIT_REPO}@{S.E0.SPLIT_REVISION}", "runtime": S.NAME, "mode": "static", "branch": branch, "device": gpu,
             "freeze_sha256": rec["freeze_sha256"], "provenance": prov}
    n_err = n_req = 0
    for req in L.requests_for(cell["id"], cell["block"], wl, True if a.with_depth is None else a.with_depth):
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
            r.update(ok=len(toks) == S.E0.GEN_TOKENS, token_ns=e1.token_ns, steps=_phase_steps(e1.steps, req["prompt"]["n_in"]),
                     bytes_out=mid.bytes_sent - b0, bytes_in=mid.bytes_recv - b1)
            if nv and nv.available:
                r["alloc"] = I.allocator_stats(torch, a.device)
        except Exception as e:   # OOM and transport errors are failures; recorded, never retried
            n_err += 1
            r.update(ok=False, error=f"{type(e).__name__}: {e}")
        if sampler:
            r["nvml"] = sampler.summary(t0, time.perf_counter())
        I.append_jsonl(out, r)
        if not r["ok"] and "TransportError" in str(r.get("error", "")):
            break
    if sampler:
        sampler.stop()
    mid.close(); t.join(30)
    print(json.dumps({"out": str(out), "requests": n_req, "errors": n_err, "profile_role": profile_role}))
    sys.exit(1 if n_err else 0)


# --------------------------------------------------------------------------- #
# Stage 2: server-segment profile (synthetic activations)                     #
# --------------------------------------------------------------------------- #
def profile_server_segment(a):
    rec, branch, gpu, profile_role, prov = _prep(a, "server", "2")
    if S.ONE_SERVER_GPU and torch.cuda.is_available() and torch.device(a.device).type == "cuda":
        pass   # the segment lives on one device by construction (server_slice takes one device)
    expected = None if a.dev_mode else S.E0.QWEN3_32B_CONFIG
    sl = server_slice(a.model_dir, a.k, a.j, a.device, torch.bfloat16, expected)
    sl.cfg._attn_implementation = "e1_static"
    mid = LocalMiddle(sl, sl.layer_ids, a.max_len, "static")
    if branch["cuda_graphs"] == "on":
        try:
            mid.enable_graph()
        except Exception as e:
            raise GraphCaptureError(f"{type(e).__name__}: {e}") from e
    H = sl.cfg.hidden_size
    wl = L.load_workload(a.workload)
    g = torch.Generator().manual_seed(0)
    cell = {"id": f"P-server-U({a.k},{a.j})-load0", "block": "profile", "load": 0}
    out = Path(a.out_dir) / f"{cell['id']}.jsonl"
    n_req = 0
    for req in L.requests_for(cell["id"], "core", wl, True):
        if a.dev_max_requests and n_req >= a.dev_max_requests:
            break
        n_req += 1
        mid.reset()
        Pn = req["prompt"]["n_in"]
        steps = []
        for s in range(0, Pn, S.E0.PREFILL_CHUNK):
            n = min(S.E0.PREFILL_CHUNK, Pn - s)
            mid.forward(torch.randn((1, n, H), generator=g).to(torch.bfloat16).to(a.device), s)
            steps.append(dict(mid.last, n=n, start=s, phase="prefill"))
        for i in range(S.E0.GEN_TOKENS - 1):
            mid.forward(torch.randn((1, 1, H), generator=g).to(torch.bfloat16).to(a.device), Pn + i)
            steps.append(dict(mid.last, n=1, start=Pn + i, phase="decode"))
        r = L.base_record(a, cell, req, "PROFILE-SERVER", {"k": a.k, "j": a.j, "runtime": S.NAME, "mode": "static", "branch": branch,
                                                          "server_identity": {"gpu": gpu, "host": socket.gethostname(), "freeze_sha256": rec["freeze_sha256"]},
                                                          "provenance": prov})
        r.update(ok=True, steps=steps)
        I.append_jsonl(out, r)
    print(json.dumps({"out": str(out), "requests": n_req, "profile_role": profile_role}))


# --------------------------------------------------------------------------- #
# Stage 3: f_results.json (E0 schema)                                          #
# --------------------------------------------------------------------------- #
def summarize_fit(a):
    if a.host and not a.dev_mode:
        sys.exit("--host is permitted only with --dev-mode")
    rec = e1_freeze.check(a.freeze, require_tag=not a.dev_mode, host=a.host, role="edge")
    d = Path(a.dir)
    u_max, eligible, defects, engine = {}, {}, [], {}
    for m in S.E0.MEMORY_CLASSES:
        k = S.FIXED_CUTS[m][0]
        f = d / f"split_fit_c{m}_k{k}.json"
        if f.exists():
            r = json.loads(f.read_text())
            if r.get("error_kind") == "harness" or r.get("exit_status") != (0 if r.get("pass") else 3):
                defects.append(f.name)
            u_max[str(m)] = [k, k] if r.get("pass") else None
        else:
            u_max[str(m)] = None
        eligible[str(m)] = []
        for model in S.E0.EDGE_MODELS:
            g = d / f"edge_fit_{model}_c{m}.json"
            if g.exists():
                rr = json.loads(g.read_text())
                if rr.get("pass"):
                    eligible[str(m)].append(model)
                elif rr.get("fail_kind") == "engine" and model in S.E0.EDGE_ELIGIBLE_DESK[m]:
                    engine.setdefault(str(m), []).append(model)
                for at in rr.get("attempts", []):
                    if at.get("error_kind") == "harness":
                        defects.append(f"{g.name}: {at.get('harness_error')}")
            elif m in S.PRIMARY_CLASSES and model in S.E0.EDGE_ELIGIBLE_DESK[m]:
                defects.append(f"edge_fit_{model}_c{m}.json missing (desk-eligible at a primary class)")
    prov = PV.make(rec, "3", "edge", json.loads((d / "device.json").read_text()) if (d / "device.json").exists() else {"name": None, "uuid": None, "index": None},
                   "screening" if a.dev_mode else "target", None, None, a.host)
    res = {"experiment": S.NAME, "u_max": u_max, "u_min_fits_12": None, "edge_eligible": eligible,
           "F_class_pass_static": {m: v is not None for m, v in u_max.items()}, "F_pass_static": any(v is not None for v in u_max.values()),
           "harness_defects": defects, "valid": not defects, "edge_engine_failures": engine,
           "no_split_pass_classes": sorted(int(m) for m in engine), "cuts_fixed": {str(m): list(v) for m, v in S.FIXED_CUTS.items()},
           "note": "E1-OPT: U-max is the frozen E0 cut if it fits (no search); a class whose cut does not fit is not evaluable.",
           "provenance": prov}
    I.write_json(d / "f_results.json", res)
    I.write_json(d / "edge_fit.json", eligible)
    print(json.dumps({k: res[k] for k in ("u_max", "edge_eligible", "valid", "no_split_pass_classes")}, indent=1))


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    wl = str(S.E0_DIR / "frozen" / "workload.json")

    def common(p, cut=True):
        p.add_argument("--freeze", required=True); p.add_argument("--host", default=None); p.add_argument("--device", default="cuda:0")
        p.add_argument("--dev-mode", action="store_true", help="tests only: no tag check, non-frozen cuts allowed, rows marked screening")
        p.add_argument("--screening", action="store_true", help="label rows screening (rented hardware); never decision inputs")
        p.add_argument("--workload", default=wl); p.add_argument("--max-len", type=int, default=S.E0.L_F)
        p.add_argument("--shaping", default="none")
        if cut:
            p.add_argument("--class", dest="mem_class", type=int, choices=S.E0.MEMORY_CLASSES, required=True)
            p.add_argument("--k", type=int, default=None); p.add_argument("--j", type=int, default=None)
    for name in ("fit-split", "_fit-split-child"):
        p = sub.add_parser(name); common(p)
        p.add_argument("--model-dir", required=True); p.add_argument("--cpu-dry", action="store_true"); p.add_argument("--dev-depth-tokens", type=int, default=0)
        if name == "fit-split":
            p.add_argument("--out-dir", required=True)
        else:
            p.add_argument("--out", required=True)
    c = sub.add_parser("fit-competitor"); common(c)
    c.add_argument("--model", choices=S.E0.EDGE_MODELS, required=True); c.add_argument("--vllm-bin", required=True); c.add_argument("--out-dir", required=True)
    e = sub.add_parser("profile-edge-segment"); common(e)
    e.add_argument("--model-dir", required=True); e.add_argument("--out-dir", required=True); e.add_argument("--label", default=None)
    e.add_argument("--with-depth", type=lambda s: s == "1", default=None); e.add_argument("--dev-max-requests", type=int, default=0)
    s_ = sub.add_parser("profile-server-segment"); common(s_)
    s_.add_argument("--model-dir", required=True); s_.add_argument("--out-dir", required=True); s_.add_argument("--dev-max-requests", type=int, default=0)
    m = sub.add_parser("summarize-fit"); common(m, cut=False); m.add_argument("--dir", required=True)
    a = ap.parse_args(argv)
    if getattr(a, "mem_class", None) is not None and getattr(a, "k", None) is None:
        a.k, a.j = S.FIXED_CUTS[a.mem_class]       # the frozen cut; an explicit --k/--j is allowed only with --dev-mode
    if getattr(a, "k", None) is not None and (a.k, a.j) != S.FIXED_CUTS.get(getattr(a, "mem_class", 0)) and not a.dev_mode:
        sys.exit(f"cut override ({a.k},{a.j}) is not permitted in the experiment path (PREREG §3); use --dev-mode for screening")
    {"fit-split": fit_split, "_fit-split-child": fit_split_child, "fit-competitor": fit_competitor, "profile-edge-segment": profile_edge_segment,
     "profile-server-segment": profile_server_segment, "summarize-fit": summarize_fit}[a.cmd](a)


if __name__ == "__main__":
    main()
