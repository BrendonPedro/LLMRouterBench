"""GEMMA-SPLIT Stage 3 drivers: computation equivalence of the split deployment (PREREG §9). A correctness test, NOT a
benchmark. A driver file: the metrics and the decision live in gs_rules (decide_stage3), a decision file.

No latency, duration, rate or clock reading is produced, stored or printed here; `gs_results.write*` rejects any
timing-like key before a file is written. The split run's start and end (UTC, seconds; provenance that ties the run to
the server's serve window, as coarse as the exit-status log) are recorded once in its meta.json; rows carry no time.

Items: the 30 natural prompts + 4 coverage items of the frozen workload (34). Order on the hosts:
  reference-server  (SERVER, unsplit HF on the deployment's server GPU): R_server greedy continuation with the
                    top1-top2 margin per step, and the top-K log-probs at every continuation position
  reference-edge    (EDGE, unsplit HF on the DGX Spark): R_edge teacher-forced on R_server's sequences -> hardware floor
  split             (EDGE, server half running): greedy through the split; teacher-forced log-probs (tier T1);
                    boundary audit of the generation path
  pipeline          (SERVER, optional, in_host_split): both segments in one process -> a diagnostic, never a floor
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # python -I: the script dir is not on sys.path
import gs_adapter as AD  # noqa: E402
import gs_common as C  # noqa: E402
import gs_results as R  # noqa: E402
import gs_rules as RU  # noqa: E402
import gs_spec as S  # noqa: E402

STAGE = "stage3"


def reference_dtype(freeze):
    cap = freeze["split"]["capture"]["model"]
    de, ds = cap.get("dtype_edge"), cap.get("dtype_server")
    return (de if de == ds else (freeze["checkpoint"]["summary"] or {}).get("dtype")) or "bfloat16"


def _meta(freeze, role, devices, profile_role, extra):
    return dict(extra, provenance=R.provenance(freeze, STAGE, role, [str(d) for d in devices], profile_role))


def run_reference_server(freeze, gs_dir, device, model_dir, profile_role, items=None):
    import gs_reference as REF
    model, info = REF.load_model(model_dir, reference_dtype(freeze), device, freeze["split"]["capture"]["model"].get("attn_implementation"))
    eos = freeze["workload"]["eos_token_ids"]
    rows = []
    for it in items or RU.corr_items(freeze, gs_dir):
        g = REF.greedy(model, it["ids"], it["gen_tokens"], eos, it["ignore_eos"], device)
        seq = it["ids"] + g["tokens"]
        tids, tlps, nan = RU.topk_rows(REF.teacher_forced(model, seq, it["n"] - 1, device), S.CORR_TOPK)
        rows.append({"id": it["id"], "set": it["set"], "prompt_n": it["n"], "tokens": g["tokens"], "margins": g["margins"],
                     "stopped_on_eos": g["stopped_on_eos"], "nan": g["nan"] or nan, "tf_start": it["n"] - 1, "topk_ids": tids, "topk_lps": tlps})
    sd = R.stage_dir(R.run_dir(gs_dir, freeze), STAGE) / "reference_server"
    R.write_rows(sd / "reference.jsonl", rows, timing_allowed=False)
    R.write(sd / "meta.json", _meta(freeze, "server", [device], profile_role, {"model": info, "k": S.CORR_TOPK,
                                                                               "dtype_rule": "deployment dtype if equal on both hosts, else config dtype"}),
            timing_allowed=False)
    return rows


def run_reference_edge(freeze, gs_dir, device, model_dir, profile_role):
    import gs_reference as REF
    rd = R.run_dir(gs_dir, freeze)
    ref = {r["id"]: r for r in C.read_jsonl(R.stage_dir(rd, STAGE) / "reference_server" / "reference.jsonl")}
    model, info = REF.load_model(model_dir, reference_dtype(freeze), device, freeze["split"]["capture"]["model"].get("attn_implementation"))
    eos = freeze["workload"]["eos_token_ids"]
    rows = []
    for it in RU.corr_items(freeze, gs_dir):
        r = ref[it["id"]]
        cmp_ = RU.compare_rows(REF.teacher_forced(model, it["ids"] + r["tokens"], r["tf_start"], device), r["topk_ids"], r["topk_lps"], r["tf_start"])
        g = REF.greedy(model, it["ids"], it["gen_tokens"], eos, it["ignore_eos"], device)
        rows.append({"id": it["id"], "tokens": g["tokens"], "stopped_on_eos": g["stopped_on_eos"], **cmp_, "nan": cmp_["nan"] or g["nan"]})
    sd = R.stage_dir(rd, STAGE) / "reference_edge"
    R.write_rows(sd / "compare.jsonl", rows, timing_allowed=False)
    R.write(sd / "meta.json", _meta(freeze, "edge", [device], profile_role, {"model": info}), timing_allowed=False)
    return rows


def _run_split_like(freeze, gs_dir, runner, tier, can_ignore_eos):
    rd = R.run_dir(gs_dir, freeze)
    ref = {r["id"]: r for r in C.read_jsonl(R.stage_dir(rd, STAGE) / "reference_server" / "reference.jsonl")}
    eos = freeze["workload"]["eos_token_ids"]
    rows, messages, tf_messages = [], [], []
    for it in RU.corr_items(freeze, gs_dir):
        r = ref[it["id"]]
        row = {"id": it["id"]}
        try:
            ign = it["ignore_eos"] and can_ignore_eos
            g = runner.greedy(it["ids"], it["gen_tokens"], eos, ign)
            row.update(tokens=list(g["tokens"]), stopped_on_eos=bool(g["stopped_on_eos"]), eos_truncated=bool(it["ignore_eos"] and not ign and g["stopped_on_eos"]))
            if hasattr(runner, "boundary_audit"):
                messages += runner.boundary_audit()          # the deployment's own generation path: audited
            if tier == S.TIER_FULL:
                row.update(RU.compare_rows(runner.teacher_forced_logprobs(it["ids"] + r["tokens"], r["tf_start"]), r["topk_ids"], r["topk_lps"], r["tf_start"]))
                if hasattr(runner, "boundary_audit"):
                    tf_messages += runner.boundary_audit()   # test-only path (may return log-probs): recorded, not audited
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
            row.setdefault("tokens", [])
        rows.append(row)
    return rows, {"messages": RU.aggregate_audit(messages), "teacher_forced_messages": RU.aggregate_audit(tf_messages)}


def run_split(freeze, gs_dir, adapter, profile_role, check=None):
    started = C.utc()
    adapter.connect()
    try:
        devices = adapter.edge_devices()
        if check:
            check(devices)
        rows, audit = _run_split_like(freeze, gs_dir, adapter, freeze["adapter"]["tier"], "ignore_eos" in adapter.capabilities())
    finally:
        AD.safe_close(adapter)
    sd = R.stage_dir(R.run_dir(gs_dir, freeze), STAGE) / "split"
    R.write_rows(sd / "compare.jsonl", rows, timing_allowed=False)
    R.write(sd / "boundary_audit.json", audit, timing_allowed=False)
    R.write(sd / "meta.json", _meta(freeze, "edge", devices, profile_role, {"adapter": freeze["adapter"]["info"], "started_utc": started,
                                                                         "ended_utc": C.utc()}), timing_allowed=False)
    return rows


def run_pipeline(freeze, gs_dir, adapter, profile_role, check=None):
    runner = adapter.in_host_runner()
    devices = adapter.server_devices()
    if check:
        check(devices)
    rows, _ = _run_split_like(freeze, gs_dir, runner, freeze["adapter"]["tier"], "ignore_eos" in adapter.capabilities())
    sd = R.stage_dir(R.run_dir(gs_dir, freeze), STAGE) / "pipeline"
    R.write_rows(sd / "compare.jsonl", rows, timing_allowed=False)
    R.write(sd / "meta.json", _meta(freeze, "server", devices, profile_role, {"runner": "adapter.in_host_runner (diagnostic)"}),
            timing_allowed=False)
    return rows
