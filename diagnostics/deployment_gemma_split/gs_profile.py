"""GEMMA-SPLIT Stage 4 (component runtime) and Stage 5 (end-to-end runtime) drivers (PREREG §10, §11). A driver file:
the summaries, the efficiency endpoint and the validity rules live in gs_rules, a decision file.

Both stages refuse to run unless the previous gate is open (Stage 3 PASS for Stage 4; Stage 4 COMPLETE for Stage 5):
a fast runtime can never be measured, let alone reported, for a split that failed correctness.

Stage 4, isolated (adapter capability segment_step): on each host, its own segment runs one prefill and gen_tokens-1
decode steps per prompt (10 measured + 2 warm-up prompts per timing group), each call timed with time.perf_counter_ns
between CUDA synchronisations. In-path (step_trace, when segment_step is absent): the edge records the adapter's per-step
decomposition during a dedicated instrumented run with the server half serving. Neither is an end-to-end number.

Stage 5: the blocks of gs_spec.block_order in that order, each from freshly started processes, each over the same
2 x (2 warm-up + 20 measured) prompts, greedy, exactly 256 new tokens when the deployment supports ignore_eos. The
host running the block records perf_counter_ns at submission and at every token's arrival; no synchronisation is added.
TTFT = first token - submission; TPOT = (last - first) / (n - 1).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # python -I: the script dir is not on sys.path
import gs_adapter as AD  # noqa: E402
import gs_common as C  # noqa: E402
import gs_memory as M  # noqa: E402
import gs_results as R  # noqa: E402
import gs_spec as S  # noqa: E402
import gs_workload as WL  # noqa: E402


def sync_for(devices):
    try:
        import torch
    except Exception:
        return lambda: None
    cuda = [torch.device(d) for d in devices if str(d).startswith("cuda")]
    if not cuda or not torch.cuda.is_available():
        return lambda: None
    return lambda: [torch.cuda.synchronize(d) for d in cuda]


def timed(fn, sync):
    sync()
    t0 = time.perf_counter_ns()
    fn()
    sync()
    return time.perf_counter_ns() - t0


def workload(freeze, gs_dir):
    return C.read_json(Path(gs_dir) / freeze["workload"]["tokens_file"])


def component_items(w):
    out = []
    for g in S.TIMING_GROUPS:
        out += WL.items(w, "timing", g, "warmup")[:S.COMPONENT_WARMUP]
        out += WL.items(w, "timing", g, "measured")[:S.COMPONENT_REQUESTS]
    return out


def timing_items(w):
    out = []
    for g in S.TIMING_GROUPS:
        out += WL.items(w, "timing", g, "warmup") + WL.items(w, "timing", g, "measured")
    return out


def contention(devices_uuids):
    snap = M.smi_snapshot()
    return {"utc": C.utc(), "foreign": M.foreign_processes(snap, devices_uuids), "smi_available": snap["available"]}


# --------------------------------------------------------------------------- #
# Stage 4                                                                     #
# --------------------------------------------------------------------------- #
def run_components(freeze, gs_dir, adapter, role, uuids, profile_role, check=None):
    caps = adapter.capabilities()
    w = workload(freeze, gs_dir)
    sd = R.stage_dir(R.run_dir(gs_dir, freeze), "stage4") / role
    before = contention(uuids)
    rows, mode, devices, out, extra, failure = [], None, [], None, {}, None
    try:
        if "segment_step" in caps:
            mode, out = "isolated", "components.jsonl"
            devices = adapter.edge_devices() if role == "edge" else adapter.server_devices()
            if check:
                check(devices)
            sync = sync_for(devices)
            for it in component_items(w):
                runner = adapter.segment_runner(it["ids"])
                extra["boundary_inputs"] = getattr(runner, "inputs", None)
                rows.append({"group": it["group"], "prompt_id": it["id"], "kind": it["kind"], "phase": "prefill", "step": 0,
                             "duration_ns": timed(runner.prefill, sync)})
                for s in range(1, it["gen_tokens"]):
                    rows.append({"group": it["group"], "prompt_id": it["id"], "kind": it["kind"], "phase": "decode", "step": s,
                                 "duration_ns": timed(runner.decode_step, sync)})
        elif "step_trace" in caps and role == "edge":
            mode, out = "in_path", "trace.jsonl"
            adapter.connect()
            try:
                devices = adapter.edge_devices()
                if check:
                    check(devices)
                adapter.set_trace(True)
                eos = freeze["workload"]["eos_token_ids"]
                for it in component_items(w):
                    utc = C.utc_ms()
                    adapter.stream(it["ids"], it["gen_tokens"], eos, it["ignore_eos"] and "ignore_eos" in caps, lambda i, t: None)
                    for s, tr in enumerate(adapter.step_trace()):
                        rows.append({"group": it["group"], "prompt_id": it["id"], "kind": it["kind"], "step": s, "utc": utc,
                                     **{k: tr[k] for k in ("phase", "edge_compute_ns", "server_compute_ns", "roundtrip_ns")}})
            finally:
                AD.safe_close(adapter)
    except Exception as e:      # recorded; the gate makes the stage INVALID
        failure = f"{type(e).__name__}: {e}"
    after = contention(uuids)
    if out:
        R.write_rows(sd / out, rows, timing_allowed=True)
    R.write(sd / "meta.json", {"mode": mode or "not_available", "contention_before": before, "contention_after": after, "failure": failure,
                               **extra, "provenance": R.provenance(freeze, "stage4", role, [str(d) for d in devices], profile_role)},
            timing_allowed=True)
    return rows


# --------------------------------------------------------------------------- #
# Stage 5                                                                     #
# --------------------------------------------------------------------------- #
def run_block(freeze, gs_dir, block, runner, role, devices, uuids, runtime_label, can_ignore_eos, profile_role):
    if block not in S.block_order(freeze["adapter"]["info"]["capabilities"]):
        raise ValueError(f"block {block} is not in this run's block order")
    w = workload(freeze, gs_dir)
    eos = freeze["workload"]["eos_token_ids"]
    started = C.utc_ms()
    before = contention(uuids)
    rows = []
    for it in timing_items(w):
        stamps = []
        row = {"group": it["group"], "prompt_id": it["id"], "kind": it["kind"]}
        t0 = time.perf_counter_ns()
        try:
            res = runner.stream(it["ids"], it["gen_tokens"], eos, it["ignore_eos"] and can_ignore_eos, lambda i, t: stamps.append(time.perf_counter_ns()))
            t_end = time.perf_counter_ns()
            n = len(stamps)
            row.update(n_tokens=n, stopped_on_eos=bool(res.get("stopped_on_eos")), total_ns=t_end - t0,
                       ttft_ns=stamps[0] - t0 if n else None, tpot_ns=(stamps[-1] - stamps[0]) / (n - 1) if n >= 2 else None,
                       token_ns=[s - t0 for s in stamps], tokens_sha256=C.sha256_obj(list(res.get("tokens", []))))
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
    after = contention(uuids)
    bd = R.stage_dir(R.run_dir(gs_dir, freeze), "stage5") / block
    R.write_rows(bd / "requests.jsonl", rows, timing_allowed=True)
    R.write(bd / "meta.json", {"block": block, "runtime": runtime_label, "ignore_eos_used": bool(can_ignore_eos), "started_utc": started,
                               "ended_utc": C.utc_ms(), "contention_before": before, "contention_after": after,
                               "provenance": R.provenance(freeze, "stage5", role, [str(d) for d in devices], profile_role)}, timing_allowed=True)
    return rows
