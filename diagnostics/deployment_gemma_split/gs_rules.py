"""GEMMA-SPLIT decision rules: every metric and every mapping from recorded outputs to a verdict (PREREG §8-§11).

This is a DECISION file (gs_spec.DECISION_FILES): it is byte-identical to the tag gemma-split-prereg-v1 at every
freeze check, in every freeze record. Nothing here runs a model or a timer; the drivers (gs_correctness, gs_memory,
gs_profile) record outputs and the gates (gs_gate) call these functions. torch is imported lazily, only by the metric
functions that receive log-prob tensors on the host where they were produced.

Every problem found is returned as {"part", "cause", "reason"}: `part` is the sub-directory of the stage that would be
repeated and `cause` the key of gs_spec.REPEAT_CAPS that limits how often (gs_gate applies the caps).
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # python -I: the script dir is not on sys.path
import gs_common as C  # noqa: E402
import gs_results as R  # noqa: E402
import gs_spec as S  # noqa: E402
import gs_stats as ST  # noqa: E402


def problem(part, cause, reason):
    return {"part": part, "cause": cause, "reason": reason}


# --------------------------------------------------------------------------- #
# Correctness metrics (PREREG §9.4)                                           #
# --------------------------------------------------------------------------- #
def coarse_kl(ref_lps, x_lps):
    """KL(ref || x) on the partition {each reference top-K id, rest}; one position; natural log (E0 PREREG §4.1 form)."""
    head = sum(math.exp(r) * (r - x) for r, x in zip(ref_lps, x_lps))
    rr = max(1.0 - sum(math.exp(r) for r in ref_lps), 1e-12)
    rx = max(1.0 - sum(math.exp(x) for x in x_lps), 1e-12)
    return max(head + rr * (math.log(rr) - math.log(rx)), 0.0)


def topk_rows(lp_iter, k):
    import torch
    ids, lps, nan = [], [], False
    for _, lp in lp_iter:
        nan = nan or bool(torch.isnan(lp).any())
        v, i = torch.topk(lp, k, dim=-1)
        ids += i.cpu().tolist()
        lps += v.cpu().tolist()
    return ids, lps, nan


def compare_rows(lp_iter, ref_ids, ref_lps, start):
    """Per position: agree (argmax == reference top-1), coarse KL vs the reference top-K, x margin; NaN flag. The chunks
    must be contiguous from `start`; the decision checks that every reference position is covered."""
    import torch
    agree, kl, xm, nan, expect = [], [], [], False, start
    for a, lp in lp_iter:
        if a != expect:
            raise ValueError(f"teacher-forced chunk starts at {a}, expected {expect} (positions skipped or repeated)")
        n = lp.shape[0]
        expect += n
        rid = ref_ids[a - start:a - start + n]
        if len(rid) != n:
            raise ValueError(f"reference covers {len(rid)} positions at {a}, split chunk has {n}")
        nan = nan or bool(torch.isnan(lp).any())
        top = torch.topk(lp, 2, dim=-1)
        arg = top.indices[:, 0].cpu().tolist()
        gx = torch.gather(lp, 1, torch.tensor(rid, device=lp.device, dtype=torch.long)).double().cpu().tolist()
        for j in range(n):
            agree.append(int(arg[j] == rid[j][0]))
            kl.append(coarse_kl(ref_lps[a - start + j], gx[j]) if not math.isnan(sum(gx[j])) else float("inf"))
        xm += (top.values[:, 0] - top.values[:, 1]).cpu().tolist()
    return {"agree": agree, "kl": kl, "x_margin": xm, "nan": nan}


# --------------------------------------------------------------------------- #
# Correctness rules                                                           #
# --------------------------------------------------------------------------- #
def first_divergence(ref, x):
    for i, (a, b) in enumerate(zip(ref, x)):
        if a != b:
            return i
    return None if len(ref) == len(x) else min(len(ref), len(x))


def strip_eos(tokens, eos_ids):
    """Drop one trailing EOS id: an implementation may or may not return the stopping EOS token (PREREG §9.4)."""
    return list(tokens[:-1]) if tokens and tokens[-1] in eos_ids else list(tokens)


def is_degenerate(tokens, vocab_len, window=S.DEGEN_WINDOW, max_period=S.DEGEN_MAX_PERIOD):
    if any(t < 0 or (vocab_len is not None and t >= vocab_len) for t in tokens):
        return True, "invalid token id"
    if len(tokens) < window:
        return False, ""
    tail = tokens[-window:]
    for p in range(1, max_period + 1):
        if all(tail[i] == tail[i - p] for i in range(p, window)):
            return True, f"period {p}"
    return False, ""


def tf_summary(rows):
    ag = [a for r in rows for a in r["agree"]]
    kl = [k for r in rows for k in r["kl"]]
    if not ag:
        return None
    return {"n_items": len(rows), "n_positions": len(ag), "D_pp": 100.0 * (1.0 - sum(ag) / len(ag)), "KL": sum(kl) / len(kl)}


def tf_rule(split, hardware):
    """T1 thresholds from the HARDWARE floor only (R_edge vs R_server). The deployment's own in-host pipeline is never a
    floor: it shares the deployment's code, so a defect would raise floor and split together (PREREG §9.2)."""
    kl_thr = S.TF_KL_FACTOR * hardware["KL"] if hardware["KL"] > 0 else S.TF_KL_EPS
    ok_d = split["D_pp"] <= hardware["D_pp"] + S.TF_D_MARGIN_PP
    ok_kl = split["KL"] <= kl_thr and not math.isinf(split["KL"])
    return {"floor": "hardware", "D_split_pp": split["D_pp"], "D_threshold_pp": hardware["D_pp"] + S.TF_D_MARGIN_PP, "KL_split": split["KL"],
            "KL_threshold": kl_thr, "pass_D": bool(ok_d), "pass_KL": bool(ok_kl), "pass": bool(ok_d and ok_kl)}


def gen_rule(ref_rows, x_rows, vocab_len, n_expected, eos_ids):
    ref = {r["id"]: r for r in ref_rows}
    eos = set(eos_ids)
    per, unexplained, bad = [], 0, []
    for x in x_rows:
        r = ref[x["id"]]
        rt = r["tokens"][:len(x["tokens"])] if x.get("eos_truncated") else r["tokens"]
        i = first_divergence(strip_eos(rt, eos), strip_eos(x["tokens"], eos))
        m = None if i is None or i >= len(r["margins"]) else r["margins"][i]
        explained = i is None or (m is not None and m < S.GEN_MARGIN_NATS)
        unexplained += not explained
        dx, why = is_degenerate(x["tokens"], vocab_len)
        dr, _ = is_degenerate(r["tokens"], vocab_len)
        dx = dx or bool(x.get("nan"))
        if dx and not dr:
            bad.append(x["id"])
        per.append({"id": x["id"], "first_div": i, "ref_margin": m, "explained": bool(explained), "x_degenerate": dx, "why": why,
                    "ref_degenerate": dr, "eos_truncated": bool(x.get("eos_truncated"))})
    ok = len(x_rows) == n_expected and unexplained <= S.GEN_MAX_UNEXPLAINED and not bad
    return {"n": len(x_rows), "n_expected": n_expected, "unexplained": unexplained, "max_unexplained": S.GEN_MAX_UNEXPLAINED,
            "degenerate_only_split": bad, "pass": bool(ok), "per_item": per}


def aggregate_audit(messages):
    """Collapse per-step messages to unique (direction, tensor names+dtypes+ranks) entries with counts and total bytes."""
    agg = {}
    for m in messages:
        key = (m["direction"], tuple((t["name"], str(t["dtype"]), len(t.get("shape") or [])) for t in m.get("tensors", [])))
        a = agg.setdefault(key, {"direction": m["direction"], "tensors": [dict(t) for t in m.get("tensors", [])], "bytes": 0, "count": 0})
        a["bytes"] += int(m.get("bytes", 0))
        a["count"] += 1
    return list(agg.values())


def corr_items(freeze, gs_dir):
    w = C.read_json(Path(gs_dir) / freeze["workload"]["tokens_file"])
    return [it for it in w["items"] if it["set"] in ("natural", "coverage")]


def devices_ok(meta_devices, freeze, role):
    frozen = set(freeze["split"]["capture"]["gpu_placement"][role]["devices"])
    return set(meta_devices or []) <= frozen


def within_window(times, start, end, tol=S.CLOCK_SKEW_TOLERANCE_S):
    ts = [C.parse_utc(t) for t in times if t]
    return bool(ts) and min(ts) >= C.parse_utc(start) - tol and max(ts) <= C.parse_utc(end) + tol


def contention_seen(recs):
    """Foreign compute processes (start, during, end) or memory held by others before the run, in Stage-3 records."""
    out = []
    for name, rec in recs.items():
        c = rec.get("contention") or {}
        lists = [(rec.get("contention_before") or {}).get("foreign") or [], (rec.get("contention_after") or {}).get("foreign") or [],
                 rec.get("foreign_during") or [], c.get("foreign_before") or [], c.get("foreign_during") or [], c.get("foreign_after") or []]
        if any(len(x) > S.FOREIGN_PROCESSES_ALLOWED for x in lists):
            out.append(f"{name}: foreign compute process on an experiment device")
        if (rec.get("occupied_before") or {}).get("busy") or (c.get("occupied_before") or {}).get("busy"):
            out.append(f"{name}: memory held by others before the run")
    return out


def would_be(rows, tier, vocab, eos):
    """What the completed split items alone give, reported with an INVALID attempt (PREREG §9.5). The generation rule is
    monotone (more items can only add unexplained divergences or degenerate outputs), so its failure over the completed
    items is decisive; the teacher-forced rule is decisive only when every item completed (a mean can move either way)."""
    done = [r for r in rows["split"] if not r.get("error")]
    g = gen_rule(rows["reference_server"], done, vocab, len(done), eos)
    out = {"items_completed": len(done), "items": len(rows["split"]), "generation": g, "teacher_forced": None,
           "verdict": "FAIL" if not g["pass"] else "UNDETERMINED"}
    if tier == S.TIER_FULL and done and tf_summary(done) and tf_summary(rows["reference_edge"]):
        out["teacher_forced"] = tf_rule(tf_summary(done), tf_summary(rows["reference_edge"]))
        if len(done) == len(rows["split"]) and not out["teacher_forced"]["pass"]:
            out["verdict"] = "FAIL"           # all items completed: the teacher-forced rule is decisive too
    return out


def decide_stage3(rd, freeze, reference_feasible=True, dev=False):
    """Stage-3 decision from runs/<run_id>/stage3_correctness (PREREG §9.5). Returns verdict + problems + rules."""
    import gs_split as SP
    gs_dir = Path(rd).resolve().parents[1]
    sd = R.stage_dir(rd, "stage3")
    caps = freeze["adapter"]["info"]["capabilities"]
    tier = freeze["adapter"]["tier"]
    eos = freeze["workload"]["eos_token_ids"]
    dec = {"stage": "stage3", "tier": tier, "verdict": None, "problems": [], "rules": {}, "diagnostics": {}}
    if not reference_feasible:
        dec.update(verdict="NOT_EVALUABLE", reasons=["an unsplit reference does not fit on its host (Stage 2)"])
        return dec
    P = dec["problems"]
    for f in R.expected("stage3", caps):
        if not (sd / f).exists():
            P.append(problem(f.split("/")[0], "invalid", f"missing output {f}"))
    if P:
        dec["verdict"] = "INVALID"
        return dec
    parts = {"reference_server": "server", "reference_edge": "edge", "split": "edge"}
    if "in_host_split" in caps:
        parts["pipeline"] = "server"
    rows = {}
    for sub, role in parts.items():
        meta = C.read_json(sd / sub / "meta.json")
        P += [problem(sub, "invalid", p) for p in R.check_provenance(meta.get("provenance"), freeze, "stage3", role, require_target=not dev,
                                                                       what=f"{sub}/meta.json")]
        if not devices_ok((meta.get("provenance") or {}).get("devices"), freeze, role):
            P.append(problem(sub, "invalid", f"{sub} ran on {meta['provenance'].get('devices')}, not the frozen {role} placement"))
        rows[sub] = C.read_jsonl(sd / sub / ("reference.jsonl" if sub == "reference_server" else "compare.jsonl"))
        try:
            for r in rows[sub]:
                R.assert_no_timing_fields(r)
        except R.TimingFieldError as e:
            P.append(problem(sub, "invalid", f"timing field {e}"))
    want = sorted(i["id"] for i in corr_items(freeze, gs_dir))
    for sub, rr in rows.items():
        if sorted(r["id"] for r in rr) != want:
            P.append(problem(sub, "invalid", f"{sub}: items differ from the {len(want)} frozen correctness items"))
    if P:
        dec["verdict"] = "INVALID"
        return dec
    ref = {r["id"]: r for r in rows["reference_server"]}
    vocab = (freeze["checkpoint"]["summary"] or {}).get("vocab_size")
    serve, smeta = C.read_json(sd / "split" / "server_serve.json"), C.read_json(sd / "split" / "meta.json")
    until = C.parse_utc(smeta["ended_utc"]) - S.CLOCK_SKEW_TOLERANCE_S if smeta.get("ended_utc") else None   # before the result existed
    srv = dict(serve, contention_after={"foreign": []},       # only what the server saw before the split run ended counts
               foreign_during=[x for x in serve.get("foreign_during") or [] if until is None or not x.get("first_seen_utc")
                               or C.parse_utc(x["first_seen_utc"]) <= until])
    cont = contention_seen({"server half": srv, "edge (split run)": smeta})
    if cont:                 # an attempt with contention is repeated; its exceptions never count toward FAIL
        P.append(problem("split", "contention", "contention during the split run: " + "; ".join(cont)))
    errs = [r["id"] for r in rows["split"] if r.get("error")]
    if errs and not cont:
        P.append(problem("split", "exceptions", f"exceptions in split runs: {errs[:10]}"))
    if "pipeline" in rows:   # a diagnostic, never a floor: its errors are reported and never affect the verdict (PREREG §9.2)
        dec["diagnostics"]["pipeline_errors"] = [r["id"] for r in rows["pipeline"] if r.get("error")]
    if tier == S.TIER_FULL:
        for sub in [s for s in ("reference_edge", "split", "pipeline") if s in rows]:
            short = [r["id"] for r in rows[sub] if not r.get("error") and len(r.get("agree", [])) != len(ref[r["id"]]["tokens"])]
            if short:
                P.append(problem(sub, "invalid", f"teacher-forced coverage incomplete in {sub}: {short[:10]}"))
    P += [problem("split", "invalid", p) for p in R.check_provenance(serve.get("provenance"), freeze, "stage3", "server",
                                                                       require_target=not dev, what="split/server_serve.json")]
    if not devices_ok((serve.get("provenance") or {}).get("devices"), freeze, "server"):
        P.append(problem("split", "invalid", "the server half served on devices outside the frozen server placement"))
    if not within_window([smeta.get("started_utc"), smeta.get("ended_utc")], serve["started_utc"], serve["ended_utc"]):
        P.append(problem("split", "invalid", "the split run falls outside the server's serve window for this run"))
    audit = SP.audit_boundary(freeze["split"]["capture"], C.read_json(sd / "split" / "boundary_audit.json")["messages"])
    dec["boundary_audit"] = audit
    if audit["problems"]:
        P.append(problem("split", "boundary", "boundary content differs from the frozen split capture: " + "; ".join(audit["problems"][:5])))
    if P:
        dec["would_be"] = would_be(rows, tier, vocab, eos)
        if all(p["cause"] in ("exceptions", "contention") for p in P) and dec["would_be"]["verdict"] == "FAIL":
            dec["rules"]["generation"] = dec["would_be"]["generation"]
            dec.update(verdict="FAIL", reasons=[f"the {dec['would_be']['items_completed']} completed items already fail the rules, which "
                                                "neither more items nor contention can reverse: FAIL (PREREG §9.5)"])
            return dec
        dec["verdict"] = "INVALID"
        return dec
    g = gen_rule(rows["reference_server"], rows["split"], vocab, len(want), eos)
    dec["rules"]["generation"] = g
    dec["diagnostics"]["hardware_generation"] = gen_rule(rows["reference_server"], rows["reference_edge"], vocab, len(want), eos)
    ok = g["pass"]
    if tier == S.TIER_FULL:
        hw = tf_summary(rows["reference_edge"])
        split_sum = tf_summary(rows["split"])
        t = tf_rule(split_sum, hw)
        t.update(summary=split_sum, hardware=hw)
        dec["rules"]["teacher_forced"] = t
        ok = ok and t["pass"]
    if "pipeline" in rows:   # descriptive only: localises a divergence (transport vs segment code); never a floor
        ok_rows = [r for r in rows["pipeline"] if not r.get("error")]
        dec["diagnostics"]["pipeline"] = {"teacher_forced": tf_summary(ok_rows) if tier == S.TIER_FULL and ok_rows else None,
                                          "generation": gen_rule(rows["reference_server"], ok_rows, vocab, len(ok_rows), eos)}
    if any(r.get("nan") for r in rows["split"]):
        dec.setdefault("reasons", []).append("NaN in split log-probs")
        ok = False
    dec["verdict"] = "PASS" if ok else "FAIL"
    return dec


# --------------------------------------------------------------------------- #
# Memory (PREREG §8)                                                          #
# --------------------------------------------------------------------------- #
def classify(text):
    t = (text or "").lower()
    if any(p in t for p in S.OOM_DEVICE_PATTERNS):
        return "oom_device"
    if any(p in t for p in S.OOM_HOST_PATTERNS):
        return "oom_host"
    return "other"


def status_of(failure, foreign_before, foreign_after, unified=False, ooms=0, during=(), occupied=False):
    """FITS / INFEASIBLE / BLOCKED_CONTENTION / INVALID for one probe. `ooms`: allocator OOM events in the process.
    A failure counts as INFEASIBLE only on an otherwise idle device: no foreign compute process at the start, during the
    probe or at the end, and no memory held by others before the probe loaded anything (PREREG §8)."""
    if len(foreign_before) > S.FOREIGN_PROCESSES_ALLOWED:
        return "BLOCKED_CONTENTION"
    if failure is None:
        return "FITS"
    if len(foreign_after) > S.FOREIGN_PROCESSES_ALLOWED or len(during) > S.FOREIGN_PROCESSES_ALLOWED or occupied:
        return "INVALID"
    c = classify(failure)
    return "INFEASIBLE" if (ooms > 0 or c == "oom_device" or (c == "oom_host" and unified)) else "INVALID"


def combine_split(edge, server):
    """Split feasibility from the edge probe and the server record. INFEASIBLE > BLOCKED_CONTENTION > INVALID > FITS.
    An edge failure that coincides with allocator OOM events in the server process is the server's infeasibility."""
    st = {"edge": edge["status"], "server": server["status"]}
    if st["edge"] == "INVALID" and server.get("ooms", 0) > 0 and st["server"] == "FITS" and not server.get("foreign_after") and \
            not edge.get("foreign_after") and \
            not server.get("foreign_during") and not (server.get("occupied_before") or {}).get("busy") and \
            not edge.get("foreign_during") and not (edge.get("occupied_before") or {}).get("busy"):
        st["server"] = "INFEASIBLE"
    for v in ("INFEASIBLE", "BLOCKED_CONTENTION", "INVALID"):
        if v in st.values():
            return v, st
    return "FITS", st


# --------------------------------------------------------------------------- #
# Runtime (PREREG §10, §11)                                                   #
# --------------------------------------------------------------------------- #
def summarize_components(rd, freeze):
    sd = R.stage_dir(rd, "stage4")
    caps = freeze["adapter"]["info"]["capabilities"]
    out = {"mode": "isolated" if "segment_step" in caps else ("in_path" if "step_trace" in caps else "not_available"), "decode_step_ns": {}}
    if out["mode"] == "isolated":
        for role in S.ROLES:
            rows = [r for r in C.read_jsonl(sd / role / "components.jsonl") if r["kind"] == "measured"]
            out[role] = {g: {ph: ST.summarize([r["duration_ns"] for r in rows if r["group"] == g and r["phase"] == ph]) for ph in ("prefill", "decode")}
                         for g in S.TIMING_GROUPS}
            out["decode_step_ns"][role] = {g: out[role][g]["decode"].get("median") for g in S.TIMING_GROUPS}
    elif out["mode"] == "in_path":
        rows = [r for r in C.read_jsonl(sd / "edge" / "trace.jsonl") if r["kind"] == "measured"]
        out["in_path"] = {}
        for g in S.TIMING_GROUPS:
            dec = [r for r in rows if r["group"] == g and r["phase"] == "decode"]
            out["in_path"][g] = {"edge_compute": ST.summarize([r["edge_compute_ns"] for r in dec]),
                                 "server_compute": ST.summarize([r["server_compute_ns"] for r in dec]),
                                 "transport": ST.summarize([r["roundtrip_ns"] - r["server_compute_ns"] for r in dec])}
        out["decode_step_ns"] = {"edge": {g: out["in_path"][g]["edge_compute"].get("median") for g in S.TIMING_GROUPS},
                                 "server": {g: out["in_path"][g]["server_compute"].get("median") for g in S.TIMING_GROUPS}}
    return out


def request_failure(r):
    """Why a Stage-5 request counts as failed (PREREG §11.3), or None: an exception, or a stream that ended before
    gen_tokens without stopping at EOS (with ignore_eos it must deliver exactly gen_tokens)."""
    if r.get("error"):
        return str(r["error"])[:120]
    n, gen = r.get("n_tokens"), r.get("gen_tokens")
    if n is None or gen is None:
        return "no token count recorded"
    if n == gen or (not r.get("ignore_eos") and r.get("stopped_on_eos") and 1 <= n <= gen):
        return None
    return f"truncated: {n} of {gen} tokens" + ("" if r.get("ignore_eos") else " without stopping at EOS")


def partial_primary(rd, freeze):
    """What a crashed Stage-5 attempt would have shown from the blocks it completed (split_A against edge_only), kept with
    the repeat (PREREG §11.7); None when either block is missing."""
    sd = R.stage_dir(rd, "stage5")
    if not all((sd / b / "requests.jsonl").exists() for b in ("split_A", "edge_only")):
        return None
    a, e = ({r["prompt_id"]: r for r in _measured(rd, b)} for b in ("split_A", "edge_only"))
    pairs = {g: [(r["tpot_ns"], e[pid]["tpot_ns"]) for pid, r in a.items() if r["group"] == g and pid in e and
                 r.get("tpot_ns") is not None and e[pid].get("tpot_ns") is not None] for g in S.TIMING_GROUPS}
    prim = ST.paired_bootstrap_ratio(pairs)
    return {"statistic": "PARTIAL (crashed attempt): median TPOT(split_A) / median TPOT(edge_only)", **prim,
            "category": ST.efficiency_category(prim["ci"])}


def _measured(rd, block):
    return [r for r in C.read_jsonl(R.stage_dir(rd, "stage5") / block / "requests.jsonl") if r["kind"] == "measured"]


def summarize_e2e(rd, freeze, components, dev=False):
    sd = R.stage_dir(rd, "stage5")
    caps = freeze["adapter"]["info"]["capabilities"]
    blocks = S.block_order(caps)
    out = {"verdict": None, "problems": [], "blocks": {}, "block_order": list(blocks)}
    P = out["problems"]
    for f in R.expected("stage5", caps):
        if not (sd / f).exists():
            P.append(problem(f.split("/")[0], "invalid", f"missing output {f}"))
    if P:
        out["verdict"] = "INVALID"
        return out
    metas = {b: C.read_json(sd / b / "meta.json") for b in blocks}
    for prev, nxt in zip(blocks, blocks[1:]):
        if C.parse_utc(metas[nxt]["started_utc"]) < C.parse_utc(metas[prev]["ended_utc"]) - S.CLOCK_SKEW_TOLERANCE_S:
            P.append(problem(nxt, "invalid", f"{nxt} started before {prev} ended: blocks run in the frozen order {blocks}"))
    for b in blocks:
        role = S.BLOCK_HOST[b]
        meta = metas[b]
        P += [problem(b, "invalid", p) for p in R.check_provenance(meta["provenance"], freeze, "stage5", role, require_target=not dev,
                                                                     what=f"{b}/meta.json")]
        if not devices_ok(meta["provenance"].get("devices"), freeze, role):
            P.append(problem(b, "invalid", f"{b} ran on {meta['provenance'].get('devices')}, not the frozen {role} placement"))
        if b in ("edge_only", "server_only") and not str(meta.get("runtime", "")).startswith("reference runtime"):
            P.append(problem(b, "invalid", f"{b} must run the reference runtime (PREREG §11.2), ran {meta.get('runtime')!r}"))
        recs = [meta]
        if b.startswith("split"):
            srv = C.read_json(sd / b / "server_meta.json")
            recs.append(srv)
            P += [problem(b, "invalid", p) for p in R.check_provenance(srv.get("provenance"), freeze, "stage5", "server", require_target=not dev,
                                                                         what=f"{b}/server_meta.json")]
            if not devices_ok((srv.get("provenance") or {}).get("devices"), freeze, "server"):
                P.append(problem(b, "invalid", f"{b}: the server half served on devices outside the frozen server placement"))
            if srv.get("failure"):
                P.append(problem(b, "failed_requests", f"{b}: the server half failed: {srv['failure'][:200]}"))
            if not within_window([meta["started_utc"], meta["ended_utc"]], srv["started_utc"], srv["ended_utc"]):
                P.append(problem(b, "invalid", f"{b}: the edge block falls outside the server's serve window for this block"))
        for m in recs:
            for k in ("contention_before", "contention_after"):
                if len(m[k]["foreign"]) > S.FOREIGN_PROCESSES_ALLOWED:
                    P.append(problem(b, "contention", f"{b}: foreign compute process on an experiment device ({k})"))
        rows = _measured(rd, b)
        errs = [f"{r['prompt_id']}: {request_failure(r)}" for r in C.read_jsonl(sd / b / "requests.jsonl") if request_failure(r)]
        if errs:                                   # warm-up requests included: a failing system is not timed around
            P.append(problem(b, "failed_requests", f"{b}: failed requests {errs[:5]} (never retried or dropped within a block)"))
        out["blocks"][b] = {"runtime": meta["runtime"], **{g: {"ttft_ns": ST.summarize([r["ttft_ns"] for r in rows if r["group"] == g and r.get("ttft_ns") is not None]),
                                                              "tpot_ns": ST.summarize([r["tpot_ns"] for r in rows if r["group"] == g and r.get("tpot_ns") is not None]),
                                                              "excluded_tpot_n_lt_2": sum(1 for r in rows if r["group"] == g and r.get("tpot_ns") is None and not r.get("error"))}
                                                          for g in S.TIMING_GROUPS}}
    by = {b: {r["prompt_id"]: r for r in _measured(rd, b)} for b in blocks}   # computed even for an INVALID attempt (reported)
    a = [r["tpot_ns"] for r in by["split_A"].values() if r.get("tpot_ns") is not None]
    bb = [r["tpot_ns"] for r in by["split_B"].values() if r.get("tpot_ns") is not None]
    drift = abs(ST.median(bb) - ST.median(a)) / ST.median(a) if a and bb and ST.median(a) else None
    out["drift"] = {"split_A_median_tpot_ns": ST.median(a), "split_B_median_tpot_ns": ST.median(bb), "rel": drift, "max_rel": S.DRIFT_MAX_REL,
                    "ok": drift is not None and drift <= S.DRIFT_MAX_REL}

    def pairs(metric, other):
        p = {}
        for g in S.TIMING_GROUPS:
            p[g] = []
            for pid, ra in by["split_A"].items():
                rb, ro = by["split_B"].get(pid), by[other].get(pid)
                if ra["group"] != g or not rb or not ro or None in (ra.get(metric), rb.get(metric), ro.get(metric)):
                    continue
                p[g].append(((ra[metric] + rb[metric]) / 2.0, ro[metric]))
        return p

    prim = ST.paired_bootstrap_ratio(pairs("tpot_ns", "edge_only"))
    out["primary"] = {"statistic": "median TPOT(split, mean of A and B) / median TPOT(edge_only, reference runtime), pooled over groups",
                      **prim, "category": ST.efficiency_category(prim["ci"])}
    out["secondary"] = {"tpot_split_over_server_only": ST.paired_bootstrap_ratio(pairs("tpot_ns", "server_only")),
                        "ttft_split_over_edge_only": {g: ST.paired_bootstrap_ratio({g: pairs("ttft_ns", "edge_only")[g]}) for g in S.TIMING_GROUPS},
                        "ttft_split_over_server_only": {g: ST.paired_bootstrap_ratio({g: pairs("ttft_ns", "server_only")[g]}) for g in S.TIMING_GROUPS}}
    for nb in S.NATIVE_BLOCKS:
        if nb in by:
            out["secondary"][f"tpot_split_over_{nb}"] = ST.paired_bootstrap_ratio(pairs("tpot_ns", nb))
            out["secondary"][f"ttft_split_over_{nb}"] = {g: ST.paired_bootstrap_ratio({g: pairs("ttft_ns", nb)[g]}) for g in S.TIMING_GROUPS}

    def agree(x, y):
        common = [p for p in by[x] if p in by[y] and by[x][p].get("tokens_sha256") and by[y][p].get("tokens_sha256")]
        return {"n": len(common), "identical": sum(by[x][p]["tokens_sha256"] == by[y][p]["tokens_sha256"] for p in common)}
    out["token_agreement"] = {f"{x}~{y}": agree(x, y) for x, y in [("split_A", "split_B")] + [("split_A", b) for b in blocks if not b.startswith("split")]}
    dec = {}
    for g in S.TIMING_GROUPS:
        tp = ST.median([(by["split_A"][p]["tpot_ns"] + by["split_B"][p]["tpot_ns"]) / 2.0 for p in by["split_A"]
                        if by["split_A"][p]["group"] == g and p in by["split_B"]
                        and None not in (by["split_A"][p].get("tpot_ns"), by["split_B"][p].get("tpot_ns"))])
        e = (components.get("decode_step_ns", {}).get("edge") or {}).get(g)
        s = (components.get("decode_step_ns", {}).get("server") or {}).get(g)
        if tp is None or e is None or s is None:
            dec[g] = {"available": False}
            continue
        dec[g] = {"available": True, "split_tpot_ns": tp, "edge_step_ns": e, "server_step_ns": s, "transport_and_overhead_ns": tp - (e + s),
                  "closure_ratio": tp / (e + s), "closure_ok": tp / (e + s) >= S.CLOSURE_MIN_RATIO,
                  "shares": {"edge": e / tp, "server": s / tp, "transport_and_overhead": (tp - e - s) / tp}}
    out["decomposition"] = {"source": components.get("mode"), "per_group": dec}
    out["verdict"] = "INVALID" if P else ("COMPLETE" if out["drift"]["ok"] else "UNSTABLE")
    if P:
        out["primary"]["note"] = "INVALID attempt: descriptive only, never a result (kept so that a repeat cannot hide it)"
    return out
