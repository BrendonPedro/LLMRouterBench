"""E0-F: memory feasibility on the edge (E0 §8 F; PREREG §5).

Subcommands (edge host; the GPU must be exclusive)
  split-fit   one capped attempt: edge slice U(k, k) + KV for L_F, depth prompt (L_F-256 tokens,
              chunked prefill) + 256 decode tokens against an in-process echo segment
  split-attempt  one split-fit attempt as a child process, its exit status recorded (U-min at 12 GB)
  split-umax  the frozen U-max(M) search: fresh process per attempt, start at the prediction,
              step up while PASS, down while FAIL; U-max(M) = largest passing symmetric k
  edge-fit    EDGE competitor fit under the same budget: vLLM with gpu_memory_utilization = budget/total,
              max_model_len = L_F, one depth request; optional S5 latency profile on the same server
              (kept as P-edge-<model>-cap<M>.jsonl only from the attempt that passes the fit)
  summarize   collect attempts -> f_results.json (U-max per class, EDGE eligibility, F verdict)

PASS for an attempt: no exception (OOM, cache overflow, transport) AND device-view peak <= budget AND
allocator-view peak (non-torch overhead + max reserved) <= budget.

Clean exit (HARNESS_FIXES.md H1): every split-fit attempt runs as a child of `split-umax` / `split-attempt`,
which records the child's exit status in the attempt record. Exit 0 must go with pass = true and exit 3 with
pass = false; anything else (a signal such as SIGABRT, a missing record) is a harness defect, so the attempt is
not F evidence and Stage 1 is INVALID (PREREG §5.6). split-fit itself joins its echo-server thread before it
exits and records any thread still alive.
"""

import argparse
import faulthandler
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e0_instrument as I  # noqa: E402
import e0_memory as MEM  # noqa: E402
import e0_spec as S  # noqa: E402

MIB = I.MIB


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


# Failure classification (PREREG §5.6). Only memory failures are F evidence; anything else in Stage 1 (the
# first CUDA execution of this harness) is a harness/engine defect. Unrecognised text never counts as memory.
MEMORY_ERROR_PATTERNS = ("out of memory", "outofmemoryerror", "alloc_failed", "cudaerrormemoryallocation",
                         "no available memory", "kv cache memory", "kv cache is needed")


def failure_kind(text):
    t = (text or "").lower()
    return "memory" if any(x in t for x in MEMORY_ERROR_PATTERNS) else "other"


class _DryNVML:
    """--cpu-dry: exercises every phase of split-fit on CPU; memory checks are not evaluated."""
    available = True

    def used_bytes(self):
        return 0

    def power_w(self):
        return 0.0

    def sm_clock_mhz(self):
        return 0

    def name(self):
        return "cpu-dry"


def _stop_echo(thread, port, mid, timeout_s=30):
    """Shut the in-process echo server down and join its thread (HARNESS_FIXES.md H1).

    A daemon thread that is still inside a torch call when the interpreter finalises makes CPython 3.12 call
    pthread_exit() through a noexcept pybind11 frame: libstdc++ prints "terminate called without an active
    exception" and the process aborts after the record is written. The thread therefore has to be finished,
    not abandoned. Closing the client ends handle(); a throwaway connection unblocks accept() if the client
    never connected. Returns True iff the thread has ended."""
    if thread is None:
        return True
    if mid is not None:
        mid.close()
    if thread.is_alive() and mid is None:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=5).close()
        except OSError:
            pass
    thread.join(timeout_s)
    return not thread.is_alive()


def split_fit(a):
    faulthandler.enable(all_threads=True)   # a fatal signal dumps every thread's Python stack to stderr
    nv = _DryNVML() if a.cpu_dry else I.NVML()
    if not nv.available:
        sys.exit(f"NVML unavailable: {getattr(nv, 'error', '')}")
    baseline = I.idle_baseline(nv)            # before this process touches CUDA
    import torch
    import split_runtime as R
    from e0_latency import load_workload
    budget = S.budget_mib(a.mem_class)
    astats = (lambda: {}) if a.cpu_dry else (lambda: I.allocator_stats(torch, a.device))
    reset_peak = (lambda: None) if a.cpu_dry else (lambda: torch.cuda.reset_peak_memory_stats(a.device))
    rec = {"kind": "split-fit", "dry": bool(a.cpu_dry), "class": a.mem_class, "k": a.k, "j": a.k, "budget_mib": budget, "L_F": S.L_F,
           "gpu": nv.name(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "predicted": {k: v / MIB for k, v in MEM.edge_breakdown(a.k, a.k).items() if k not in ("k", "j")}}
    sampler = I.Sampler(nv, S.MEM_SAMPLE_S, baseline)
    sampler.start()
    phase = "cap"
    srv = srv_thread = mid = port = None
    try:
        rec["cap"] = {} if a.cpu_dry else I.apply_cap(torch, a.device, budget, nv, baseline)
        phase = "load"
        sl = R.edge_slice(a.model_dir, a.k, a.k, a.device, torch.bfloat16,
                          S.QWEN3_32B_CONFIG if a.expect_32b else None)
        rec["weights_mib"] = sl.param_bytes / MIB
        port = _free_port()
        srv = R.SegmentServer(None, "echo", S.L_F, echo=True)
        srv_thread = threading.Thread(target=srv.serve_forever, args=("127.0.0.1", port),
                                      kwargs={"once": True}, daemon=True, name="echo-server")
        srv_thread.start()
        time.sleep(0.3)
        mid = R.RemoteMiddle("127.0.0.1", port, R.hello_for(sl, "echo", a.k, a.k, S.L_F), a.device,
                             sl.cfg.hidden_size)
        runner = R.EdgeRunner(sl, a.k, a.k, mid, max_len=S.L_F)
        rec["kv_mib"] = runner.cache.nbytes() / MIB
        rec["after_load"] = astats()
        reset_peak()
        ids = next(p for p in load_workload(a.workload)["prompts"] if p["group"] == "depth")["ids"]
        runner.reset()
        phase = "prefill"
        t0 = time.perf_counter()
        ids_t = torch.tensor([ids], device=a.device)
        for s in range(0, len(ids), S.PREFILL_CHUNK):
            h = runner.forward_hidden(ids_t[:, s:s + S.PREFILL_CHUNK], s, True)
        t1 = time.perf_counter()
        rec["prefill"] = dict(astats(), nvml=sampler.summary(t0, t1), seconds=t1 - t0)
        reset_peak()
        phase = "decode"
        pos = len(ids)
        for _ in range(S.GEN_TOKENS):
            tok = int(torch.argmax(runner.last_logits(h).float()))
            h = runner.forward_hidden(torch.tensor([[tok]], device=a.device), pos, True)
            pos += 1
        t2 = time.perf_counter()
        rec["decode"] = dict(astats(), nvml=sampler.summary(t1, t2), seconds=t2 - t1)
        rec["final_context"] = pos
        mid.close()
        mid = None
        phase = "done"
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["error_phase"] = phase
        rec["error_kind"] = "memory" if failure_kind(rec["error"]) == "memory" else "harness"
        try:
            rec["at_error"] = astats()
            rec["memory_summary"] = torch.cuda.memory_summary(a.device, abbreviated=True)[-3000:]
        except Exception:
            pass
    rec["nvml_total"] = sampler.stop()
    # after the last memory sample: the shutdown below cannot change any measurement
    echo_joined = _stop_echo(srv_thread, port, mid)
    live = [t.name for t in threading.enumerate() if t is not threading.main_thread()]
    rec["shutdown"] = {"echo_thread_joined": echo_joined, "live_threads": live}
    if (not echo_joined or live) and phase == "done":
        rec["error"] = f"HarnessError: threads alive at exit: {live}"
        rec["error_phase"] = phase = "shutdown"
        rec["error_kind"] = "harness"
    nontorch = rec.get("cap", {}).get("nontorch_mib", 0)
    reserved = max(rec.get(p, {}).get("reserved_peak_mib", 0) for p in ("after_load", "prefill", "decode"))
    rec["peak_device_view_mib"] = rec["nvml_total"].get("peak_used_mib")
    rec["peak_allocator_view_mib"] = nontorch + reserved
    if "weights_mib" in rec and "kv_mib" in rec:
        for p in ("prefill", "decode"):
            if p in rec and not a.cpu_dry:
                rec[p]["activation_peak_mib"] = rec[p]["allocated_peak_mib"] - rec["weights_mib"] - rec["kv_mib"]
    rec["pass"] = bool(phase == "done" and (a.cpu_dry or rec["peak_device_view_mib"] is not None) and
                       rec["peak_device_view_mib"] <= budget and rec["peak_allocator_view_mib"] <= budget)
    I.write_json(a.out, rec)
    print(json.dumps({k: rec.get(k) for k in ("class", "k", "pass", "error", "peak_device_view_mib",
                                                "peak_allocator_view_mib", "budget_mib")}), flush=True)
    sys.exit(0 if rec["pass"] else 3)


def run_attempt(cmd, f, mem_class, k):
    """Run one split-fit child and record its exit status in its record (HARNESS_FIXES.md H1). Exit 0 must
    go with pass = true and exit 3 with pass = false. Anything else, or no record, is a harness defect: the
    record keeps what the child measured, but error_kind = "harness" makes Stage 1 INVALID (PREREG §5.6)."""
    rc = subprocess.run(cmd).returncode
    rec = json.loads(f.read_text()) if f.exists() else {
        "kind": "split-fit", "class": mem_class, "k": k, "j": k, "pass": False,
        "error": "HarnessError: the child wrote no record"}
    rec["exit_status"] = rc
    expected = 0 if rec.get("pass") else 3
    if rc != expected or not f.exists():
        rec["error_kind"] = "harness"
        rec["harness_error"] = (f"child exit status {rc} (expected {expected} for pass={rec.get('pass')})"
                                + ("" if rc >= 0 else f", signal {signal.Signals(-rc).name}"))
    I.write_json(f, rec)
    return rec


class HarnessDefect(RuntimeError):
    pass


def umax_search(k0, attempt, k_max=31):
    """Largest symmetric k with attempt(k) True: from k0 step up while passing, else down until one passes."""
    k = k0
    if attempt(k):
        while k + 1 <= k_max and attempt(k + 1):
            k += 1
        return k
    while k > 1:
        k -= 1
        if attempt(k):
            return k
    return None


def split_umax(a):
    """Frozen search; each attempt is a fresh process so allocator state never carries over."""
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    k0 = MEM.predict_umax(a.mem_class, S.QWEN3_32B_CONFIG)[0]
    results = {}

    def attempt(k):
        f = out / f"split_fit_c{a.mem_class}_k{k}.json"
        rec = json.loads(f.read_text()) if f.exists() else run_attempt(split_fit_cmd(a, k, f), f, a.mem_class, k)
        if rec.get("error_kind") == "harness" or "exit_status" not in rec:
            raise HarnessDefect(f"{f.name}: {rec.get('harness_error') or rec.get('error') or 'no exit status'}")
        results[k] = rec["pass"]
        return results[k]

    try:
        umax = umax_search(k0, attempt)
    except HarnessDefect as e:   # Stage 1 is INVALID; further attempts would not be F evidence
        s = {"class": a.mem_class, "predicted_k": k0, "attempts": results, "u_max": None, "harness_defect": str(e)}
        I.write_json(out / f"umax_c{a.mem_class}.json", s)
        print(json.dumps(s))
        sys.exit(f"HARNESS DEFECT: {e} -> Stage 1 INVALID (PREREG §5.6)")
    s = {"class": a.mem_class, "predicted_k": k0, "attempts": results, "u_max": [umax, umax] if umax else None}
    I.write_json(out / f"umax_c{a.mem_class}.json", s)
    print(json.dumps(s))


def split_fit_cmd(a, k, f):
    return [sys.executable, __file__, "split-fit", "--model-dir", a.model_dir, "--class", str(a.mem_class),
            "--k", str(k), "--device", a.device, "--out", str(f), "--workload", a.workload]


def split_attempt(a):
    """One split-fit attempt as a child with its exit status recorded (the U-min run at 12 GB, k = 1)."""
    f = Path(a.out_dir) / f"split_fit_c{a.mem_class}_k{a.k}.json"
    if f.exists():
        sys.exit(f"{f} exists; Stage 1 re-runs start from an empty directory")
    rec = run_attempt(split_fit_cmd(a, a.k, f), f, a.mem_class, a.k)
    print(json.dumps({k: rec.get(k) for k in ("class", "k", "pass", "exit_status", "error_kind")}))
    if rec.get("error_kind") == "harness":
        sys.exit(f"HARNESS DEFECT: {rec.get('harness_error') or rec.get('error')} -> Stage 1 INVALID (PREREG §5.6)")


def _wait_ready(base, proc, timeout_s=1200):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(base + "/models", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(3)
    return False


def settle_profiles(profile_dir, model, mem_class, n_attempts, passed):
    """Keep the S5 profile only from the last attempt and only if it passed the fit; set the others aside
    (`.rejected`, outside every *.jsonl glob) so R never uses a profile from an over-budget server."""
    for i in range(n_attempts):
        tagged = Path(profile_dir) / f"attempt{i}-P-edge-{model}-cap{mem_class}.jsonl"
        if tagged.exists():
            keep = passed and i == n_attempts - 1
            tagged.rename(Path(profile_dir) / (f"P-edge-{model}-cap{mem_class}.jsonl" if keep
                                               else f"{tagged.name}.rejected"))


def edge_fit(a):
    s1 = S.load_s1()
    spec = s1.MODELS[a.model]
    nv = I.NVML()
    if not nv.available:
        sys.exit("NVML unavailable")
    budget = S.budget_mib(a.mem_class)
    total = nv.total_bytes()
    fp8 = (a.model, a.mem_class) in S.FP8_KV_AT
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    util = budget * MIB / total
    rec = {"kind": "edge-fit", "model": a.model, "class": a.mem_class, "budget_mib": budget, "fp8_kv": fp8,
           "gpu": nv.name(), "attempts": []}
    from e0_latency import _stream_completion, load_workload
    depth = next(p for p in load_workload(a.workload)["prompts"] if p["group"] == "depth")
    for attempt in range(2):
        baseline = I.idle_baseline(nv)
        port = _free_port()
        cmd = [a.vllm_bin, "serve", spec["repo"], "--revision", spec["revision"], "--tokenizer-revision",
               spec["revision"], "--served-model-name", a.model, "--dtype", spec["dtype"], "--max-model-len",
               str(S.L_F), "--gpu-memory-utilization", f"{util:.4f}", "--seed", "0", "--host", "127.0.0.1",
               "--port", str(port)] + (["--kv-cache-dtype", "fp8"] if fp8 else [])
        log = out / f"vllm_{a.model}_c{a.mem_class}_a{attempt}.log"
        sampler = I.Sampler(nv, S.MEM_SAMPLE_S, baseline)
        sampler.start()
        with open(log, "w") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        base = f"http://127.0.0.1:{port}/v1"
        at = {"attempt": attempt, "util": util, "cmd": cmd, "log": str(log)}
        try:
            if not _wait_ready(base, proc):
                at["error"] = "server did not start: " + log.read_text()[-2000:]
            else:
                body = {"model": a.model, "prompt": depth["ids"], "max_tokens": S.GEN_TOKENS,
                        "min_tokens": S.GEN_TOKENS, "ignore_eos": True, "temperature": 0.0, "stream": True,
                        "stream_options": {"include_usage": True, "continuous_usage_stats": True}}
                stamps, usage, _, _ = _stream_completion(base, body)
                at["depth_ok"] = len(stamps) == S.GEN_TOKENS
                at["usage"] = usage
                if a.profile_out and at["depth_ok"]:
                    # attempt-tagged label; renamed to the canonical P-edge-* name only if this attempt passes
                    prof = [sys.executable, str(S.E0_DIR / "e0_latency.py"), "run-openai", "--out-dir", a.profile_out,
                            "--base-url", base, "--served-model", a.model, "--path", "EDGE", "--edge-class",
                            str(a.mem_class), "--label", f"attempt{attempt}-P-edge-{a.model}-cap{a.mem_class}",
                            "--nvml", "--with-depth", "1", "--workload", a.workload]
                    at["profile_rc"] = subprocess.run(prof).returncode
        except Exception as e:
            at["error"] = f"{type(e).__name__}: {e}"
        finally:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
        at["nvml"] = sampler.stop()
        peak = at["nvml"].get("peak_used_mib")
        at["pass"] = bool(at.get("depth_ok") and peak is not None and peak <= budget)
        if not at["pass"]:
            over = peak is not None and peak > budget
            at["fail_kind"] = "memory" if over or failure_kind(at.get("error")) == "memory" else "engine"
        rec["attempts"].append(at)
        if at["pass"] or not at.get("depth_ok") or peak is None:
            break
        util = util - (peak - budget) * MIB / total - 0.01      # the single pre-declared retry
        time.sleep(10)
    rec["pass"] = rec["attempts"][-1]["pass"]
    rec["fail_kind"] = None if rec["pass"] else rec["attempts"][-1]["fail_kind"]
    if a.profile_out:
        settle_profiles(a.profile_out, a.model, a.mem_class, len(rec["attempts"]), rec["pass"])
    I.write_json(out / f"edge_fit_{a.model}_c{a.mem_class}.json", rec)
    print(json.dumps({"model": a.model, "class": a.mem_class, "pass": rec["pass"]}))


def summarize(a):
    d = Path(a.dir)
    u_max, eligible = {}, {}
    for m in S.MEMORY_CLASSES:
        f = d / f"umax_c{m}.json"
        u_max[str(m)] = json.loads(f.read_text())["u_max"] if f.exists() else None
        eligible[str(m)] = []
        for model in S.EDGE_MODELS:
            g = d / f"edge_fit_{model}_c{m}.json"
            if g.exists() and json.loads(g.read_text())["pass"]:
                eligible[str(m)].append(model)
    umin = d / "split_fit_c12_k1.json"
    # PREREG §5.6: a non-memory exception in any split-fit attempt invalidates Stage 1 (fix, re-run all of it).
    # HARNESS_FIXES.md H1: so does an attempt whose exit status was not recorded or did not match its verdict,
    # and a U-max search stopped by a harness defect.
    def defective(rec):
        return rec.get("error_kind") == "harness" or rec.get("exit_status") != (0 if rec.get("pass") else 3)
    defects = sorted(g.name for g in d.glob("split_fit_c*_k*.json") if defective(json.loads(g.read_text())))
    defects += sorted(g.name for g in d.glob("umax_c*.json") if json.loads(g.read_text()).get("harness_defect"))
    # The other Stage-1 children: exit 1 from a latency client means failed requests (the F error rule, in the
    # records); any other non-zero status, e.g. a signal, is a harness defect. RUNBOOK §3 logs each status.
    for g in sorted(d.glob("edge_fit_*_c*.json")):
        for at in json.loads(g.read_text()).get("attempts", []):
            if at.get("profile_rc") not in (None, 0, 1):
                defects.append(f"{g.name}: profile exit status {at['profile_rc']}")
    log = d / "exit_status.log"
    if log.exists():
        for line in log.read_text().splitlines():
            rc, _, cmd = line.partition(" ")
            allowed = (0, 1) if " run-split " in f" {cmd} " else (0,)
            if int(rc.removeprefix("rc=")) not in allowed:
                defects.append(f"exit_status.log: {line}")
    else:
        defects.append("exit_status.log missing (RUNBOOK §3 runs every Stage-1 command through st)")
    # PREREG §8.4 (b): a desk-eligible EDGE competitor that failed its fit for a non-memory reason blocks every
    # SPLIT pass at that class (the competitor set would be incomplete)
    engine = {}
    for m in S.MEMORY_CLASSES:
        for model in S.EDGE_ELIGIBLE_DESK[m]:
            g = d / f"edge_fit_{model}_c{m}.json"
            if g.exists() and json.loads(g.read_text()).get("fail_kind") == "engine":
                engine.setdefault(str(m), []).append(model)
    res = {"u_max": u_max, "u_min_fits_12": json.loads(umin.read_text())["pass"] if umin.exists() else None,
           "edge_eligible": eligible, "F_class_pass_static": {m: v is not None for m, v in u_max.items()},
           "F_pass_static": any(v is not None for v in u_max.values()),
           "harness_defects": defects, "valid": not defects,
           "edge_engine_failures": engine, "no_split_pass_classes": sorted(int(m) for m in engine),
           "note": "G-F is evaluated only if valid. F also requires zero failed E0-L requests (e0_r decide)."}
    I.write_json(d / "f_results.json", res)
    I.write_json(d / "edge_fit.json", eligible)   # summarize_s1.py --fit-json format
    print(json.dumps(res, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    wl = str(S.E0_DIR / "frozen" / "workload.json")
    f = sub.add_parser("split-fit")
    f.add_argument("--model-dir", required=True)
    f.add_argument("--class", dest="mem_class", type=int, choices=S.MEMORY_CLASSES, required=True)
    f.add_argument("--k", type=int, required=True)
    f.add_argument("--device", default="cuda:0")
    f.add_argument("--out", required=True)
    f.add_argument("--workload", default=wl)
    f.add_argument("--expect-32b", type=lambda s: s != "0", default=True)
    f.add_argument("--cpu-dry", action="store_true", help="local code-path check only (no NVML, no cap)")
    t = sub.add_parser("split-attempt", help="one split-fit child with its exit status recorded")
    t.add_argument("--model-dir", required=True)
    t.add_argument("--class", dest="mem_class", type=int, choices=S.MEMORY_CLASSES, required=True)
    t.add_argument("--k", type=int, required=True)
    t.add_argument("--device", default="cuda:0")
    t.add_argument("--out-dir", required=True)
    t.add_argument("--workload", default=wl)
    u = sub.add_parser("split-umax")
    u.add_argument("--model-dir", required=True)
    u.add_argument("--class", dest="mem_class", type=int, choices=S.MEMORY_CLASSES, required=True)
    u.add_argument("--device", default="cuda:0")
    u.add_argument("--out-dir", required=True)
    u.add_argument("--workload", default=wl)
    e = sub.add_parser("edge-fit")
    e.add_argument("--model", choices=S.EDGE_MODELS, required=True)
    e.add_argument("--class", dest="mem_class", type=int, choices=S.MEMORY_CLASSES, required=True)
    e.add_argument("--vllm-bin", required=True)
    e.add_argument("--out-dir", required=True)
    e.add_argument("--profile-out")
    e.add_argument("--workload", default=wl)
    s = sub.add_parser("summarize")
    s.add_argument("--dir", required=True)
    a = ap.parse_args(argv)
    {"split-fit": split_fit, "split-attempt": split_attempt, "split-umax": split_umax, "edge-fit": edge_fit,
     "summarize": summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
