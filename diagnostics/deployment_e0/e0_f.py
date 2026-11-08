"""E0-F: memory feasibility on the edge (E0 §8 F; PREREG §5).

Subcommands (edge host; the GPU must be exclusive)
  split-fit   one capped attempt: edge slice U(k, k) + KV for L_F, depth prompt (L_F-256 tokens,
              chunked prefill) + 256 decode tokens against an in-process echo segment
  split-umax  the frozen U-max(M) search: fresh process per attempt, start at the prediction,
              step up while PASS, down while FAIL; U-max(M) = largest passing symmetric k
  edge-fit    EDGE competitor fit under the same budget: vLLM with gpu_memory_utilization = budget/total,
              max_model_len = L_F, one depth request; optional S5 latency profile on the same server
  summarize   collect attempts -> f_results.json (U-max per class, EDGE eligibility, F verdict)

PASS for an attempt: no exception (OOM, cache overflow, transport) AND device-view peak <= budget AND
allocator-view peak (non-torch overhead + max reserved) <= budget.
"""

import argparse
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


def split_fit(a):
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
    srv = None
    try:
        rec["cap"] = {} if a.cpu_dry else I.apply_cap(torch, a.device, budget, nv, baseline)
        phase = "load"
        sl = R.edge_slice(a.model_dir, a.k, a.k, a.device, torch.bfloat16,
                          S.QWEN3_32B_CONFIG if a.expect_32b else None)
        rec["weights_mib"] = sl.param_bytes / MIB
        port = _free_port()
        srv = R.SegmentServer(None, "echo", S.L_F, echo=True)
        threading.Thread(target=srv.serve_forever, args=("127.0.0.1", port), kwargs={"once": True},
                         daemon=True).start()
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
        phase = "done"
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["error_phase"] = phase
        try:
            rec["at_error"] = astats()
            rec["memory_summary"] = torch.cuda.memory_summary(a.device, abbreviated=True)[-3000:]
        except Exception:
            pass
    rec["nvml_total"] = sampler.stop()
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
                                                "peak_allocator_view_mib", "budget_mib")}))
    sys.exit(0 if rec["pass"] else 3)


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
        if not f.exists():
            cmd = [sys.executable, __file__, "split-fit", "--model-dir", a.model_dir, "--class", str(a.mem_class),
                   "--k", str(k), "--device", a.device, "--out", str(f), "--workload", a.workload]
            subprocess.run(cmd)
        results[k] = json.loads(f.read_text())["pass"] if f.exists() else False
        return results[k]

    umax = umax_search(k0, attempt)
    s = {"class": a.mem_class, "predicted_k": k0, "attempts": results, "u_max": [umax, umax] if umax else None}
    I.write_json(out / f"umax_c{a.mem_class}.json", s)
    print(json.dumps(s))


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
                    prof = [sys.executable, str(S.E0_DIR / "e0_latency.py"), "run-openai", "--out-dir", a.profile_out,
                            "--base-url", base, "--served-model", a.model, "--path", "EDGE", "--edge-class",
                            str(a.mem_class), "--label", f"P-edge-{a.model}-cap{a.mem_class}", "--nvml",
                            "--with-depth", "1", "--workload", a.workload]
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
        rec["attempts"].append(at)
        if at["pass"] or not at.get("depth_ok") or peak is None:
            break
        util = util - (peak - budget) * MIB / total - 0.01      # the single pre-declared retry
        time.sleep(10)
    rec["pass"] = rec["attempts"][-1]["pass"]
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
    res = {"u_max": u_max, "u_min_fits_12": json.loads(umin.read_text())["pass"] if umin.exists() else None,
           "edge_eligible": eligible, "F_class_pass_static": {m: v is not None for m, v in u_max.items()},
           "F_pass_static": any(v is not None for v in u_max.values()),
           "note": "F also requires zero OOM/transport errors over every E0-L request (checked in e0_r decide)."}
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
    {"split-fit": split_fit, "split-umax": split_umax, "edge-fit": edge_fit, "summarize": summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
