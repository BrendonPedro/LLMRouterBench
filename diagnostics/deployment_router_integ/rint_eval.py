"""ROUTER-INTEG: the record-based policy evaluator and the live-pipeline comparison (analysis machine, CPU only).

Three things are reported separately and never merged into one number:
  1. Semantic router quality     Macro-F1 (primary) and AUC of the frozen router against (A) the original released
                                 one-shot winner labels and (B) the winner labels observed in this protocol's executions.
  2. Execution placement         completion time and correct-by-deadline rate of each model under each execution mode.
  3. End-to-end integrated use   each model-selection policy followed by the same frozen placement rule, per data policy.

Policies, all evaluated on the same queries from the same shared execution records (protocol, section 9):
  always_DS, always_QW    one fixed model
  router                  the frozen router's pinned held-out decision; its measured latency on the edge is added
  lookup_D2               task-level lookup from training folds only, benchmark PREDICTED by a training-fold recogniser
  lookup_D1_diagnostic    the same table indexed by the TRUE benchmark: a non-deployable diagnostic
  hindsight_model         retrospective reference: the better model of each query under the same placement
  hindsight_config        retrospective reference: the best feasible configuration of each query
The hindsight rows are computed from the held-out outcomes themselves. They are not deployable and they are not a
ceiling on future systems.

Record-based evaluation replays recorded outcomes. It says nothing about queueing, concurrency, caching or changing
load, and it does not replace the live pass, which is compared with it here.

Standard library only.
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rint_common as C  # noqa: E402
import rint_policy as P  # noqa: E402
import rint_spec as S  # noqa: E402

FIXED = ("always_DS", "always_QW")
POLICIES = ("always_DS", "always_QW", "router", "lookup_D2", "lookup_D1_diagnostic", "hindsight_model", "hindsight_config")
LABELS = {"always_DS": "fixed model", "always_QW": "fixed model", "router": "frozen router (deployable)",
          "lookup_D2": "training-derived task lookup, predicted benchmark (predictive)",
          "lookup_D1_diagnostic": "task lookup with the TRUE benchmark (non-deployable diagnostic)",
          "hindsight_model": "retrospective reference, not deployable", "hindsight_config": "retrospective reference, not deployable"}
OTHER = {S.DS: S.QW, S.QW: S.DS}


# --------------------------------------------------------------------------- #
# Small statistics                                                             #
# --------------------------------------------------------------------------- #
def quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def macro_f1(y_true, y_pred, classes=(S.DS, S.QW)):
    """Mean of the per-class F1 scores (the router study's primary metric), on the rows given."""
    fs = []
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        fs.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return sum(fs) / len(fs) if y_true else None


def auc(y_true, scores, positive=S.QW):
    """Area under the ROC curve of the router score for 'Qwen3-8B wins'; ties in the score count one half."""
    pos = [s for t, s in zip(y_true, scores) if t == positive]
    neg = [s for t, s in zip(y_true, scores) if t != positive]
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def winner(c_ds, c_qw):
    """Winner label of one query from two correctness values: the model that alone is correct, otherwise a tie."""
    return S.DS if c_ds > c_qw else S.QW if c_qw > c_ds else "tie"


def boot_weights(strata, n_boot, seed):
    """Paired bootstrap, stratified by benchmark: for every replicate, how often each query is drawn."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for i, s in enumerate(strata):
        groups[s].append(i)
    order = [groups[k] for k in sorted(groups)]
    out = []
    for _ in range(n_boot):
        w = [0] * len(strata)
        for g in order:
            for _ in g:
                w[g[rng.randrange(len(g))]] += 1
        out.append(w)
    return out


def ci(values, level=S.CI_LEVEL):
    a = (1 - level) / 2
    return [quantile(values, a), quantile(values, 1 - a)]


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #
def load(edge_run, scores_path, workload_path=None, dev=False):
    rd = Path(edge_run)
    wl = C.read_json(workload_path or C.RI_DIR / "frozen" / "workload.json")
    rows = {}
    for name in ("rows_shared_edge.jsonl", "rows_shared_remote.jsonl", "rows_live_A.jsonl", "rows_live_B.jsonl"):
        rs, _ = C.read_jsonl(rd / name)
        rows[name[5:-6]] = [r for r in rs if r["phase"] != "warmup"]
    flat = [r for v in rows.values() for r in v]
    if any(r.get("dev") or r.get("mock") for r in flat) and not dev:
        raise SystemExit("STOP: dev or mock rows are refused as results")
    sc, _ = C.read_jsonl(scores_path)
    scores = {s["id"]: s for s in sc}
    rt, _ = C.read_jsonl(rd / "router_timing.jsonl")
    events, _ = C.read_jsonl(rd / "events.jsonl")
    return {"wl": wl, "rows": rows, "scores": scores, "router_timing": {r["key"]: r for r in rt}, "events": events,
            "done": {p.name: C.read_json(p) for p in rd.glob("*.done.json")}}


def cell(row, scores):
    """What one measurement row contributes: correctness (0/1) and completion time (None if the request failed)."""
    s = scores.get(row["id"])
    if row["status"] != "ok":
        return {"correct": 0, "total_s": None, "status": "error", "stop": "error", "n_out": 0, "unscored": False}
    if s is None or s.get("score") is None:
        return {"correct": 0, "total_s": row["total_s"], "status": "ok", "stop": row["stop"], "n_out": row["n_out"], "unscored": True}
    return {"correct": int(s["score"] >= 1.0), "total_s": row["total_s"], "status": "ok", "stop": row["stop"], "n_out": row["n_out"], "unscored": False}


def shared_table(data):
    """{key: {config: cell}} from the last recorded row of every (query, configuration) of the shared pass."""
    tab = defaultdict(dict)
    for lane in ("shared_edge", "shared_remote"):
        for r in data["rows"][lane]:
            if r["phase"] == "shared" and not (r["status"] == "error" and r.get("transport_error")):
                tab[r["key"]][r["config"]] = cell(r, data["scores"])
    return tab


def transport_failures(data):
    """Requests that failed because the server or the wire was lost. They are kept as rows, repeated on resumption, and
    never read as an outcome of a model."""
    return sum(1 for lane in ("shared_edge", "shared_remote") for r in data["rows"][lane]
               if r["phase"] == "shared" and r["status"] == "error" and r.get("transport_error"))


# --------------------------------------------------------------------------- #
# Policies                                                                     #
# --------------------------------------------------------------------------- #
def on_time_correct(c, overhead, deadline):
    return int(bool(c["correct"]) and c["total_s"] is not None and c["total_s"] + overhead <= deadline)


def choose(policy, q, tab_q, mode, feasible_modes):
    """(model, config) a policy uses for one query under a placement mode."""
    if policy == "always_DS":
        m = S.DS
    elif policy == "always_QW":
        m = S.QW
    elif policy == "router":
        m = q["router_pinned"]["choice"]
    elif policy == "lookup_D2":
        m = q["task_lookup_pinned"]["D2_predicted_benchmark_choice"]
    elif policy == "lookup_D1_diagnostic":
        m = q["task_lookup_pinned"]["D1_true_benchmark_choice"]
    elif policy in ("hindsight_model", "hindsight_config"):
        modes = (mode,) if policy == "hindsight_model" else feasible_modes
        cands = [(mm, P.config_of(mm, x)) for x in modes for mm in S.MODEL_ORDER if P.config_of(mm, x) in tab_q]
        best = max(cands, key=lambda mc: (tab_q[mc[1]]["correct"], -(tab_q[mc[1]]["total_s"] if tab_q[mc[1]]["total_s"] is not None else 1e18)))
        return best
    else:
        raise ValueError(policy)
    return m, P.config_of(m, mode)


def evaluate(data, n_boot=S.BOOT_N, seed=S.BOOT_SEED):
    wl = data["wl"]
    ev = [q for q in wl["queries"] if q["role"] == "eval"]
    tab = shared_table(data)
    complete = [q for q in ev if all(c in tab.get(q["key"], {}) for c in S.REQUIRED_CONFIGS)]
    missing = [q["key"] for q in ev if q not in complete]
    rt = data["router_timing"]
    out = {"schema": S.SCHEMA, "protocol": S.PROTOCOL, "n_sample": len(ev), "n_evaluated": len(complete), "queries_without_all_configurations": missing,
           "transport_failures_recorded_and_repeated": transport_failures(data),
           "single_draw": S.REPLICATES_PER_CONFIG == 1,
           "caveat": "Record-based counterfactual evaluation on single stochastic draws. Recorded outcomes do not establish behaviour "
                     "under other queueing, concurrency, caching or load. Hindsight rows are retrospective and not deployable."}
    if not complete:
        out["status"] = "NO_COMPLETE_QUERY"
        return out
    out["status"] = "COMPLETE" if not missing else "PARTIAL_PREFIX"
    strata = [q["dataset"] for q in complete]
    W = boot_weights(strata, n_boot, seed)
    n = len(complete)
    router_s = [rt[q["key"]]["router_s"] if q["key"] in rt else None for q in complete]
    out["router_latency_available"] = all(x is not None for x in router_s)
    med_router = quantile([x for x in router_s if x is not None], 0.5) or 0.0

    # ---- 2. execution placement: every configuration by itself ----
    out["configurations"] = {}
    for cfg in S.REQUIRED_CONFIGS:
        cells = [tab[q["key"]][cfg] for q in complete]
        ts = [c["total_s"] for c in cells if c["total_s"] is not None]
        out["configurations"][cfg] = {
            "model": S.MODELS[S.CONFIGS[cfg][0]]["name"], "mode": S.CONFIGS[cfg][1],
            "feasible": {pol: P.feasible(S.CONFIGS[cfg][1], pol)[0] for pol in (S.POLICY_A, S.POLICY_B)},
            "correct_within_cutoff_pct": 100.0 * sum(c["correct"] for c in cells) / n,
            "correct_by_deadline_pct": {str(d): 100.0 * sum(on_time_correct(c, 0.0, d) for c in cells) / n for d in S.DEADLINES_S},
            "completion_s": {"median": quantile(ts, 0.5), "p90": quantile(ts, 0.9), "mean": sum(ts) / len(ts) if ts else None},
            "stops": dict(Counter(c["stop"] for c in cells)), "failures": sum(c["status"] == "error" for c in cells),
            "unscored": sum(c["unscored"] for c in cells), "output_tokens": sum(c["n_out"] for c in cells)}

    # ---- labels B (new executions) and label comparisons ----
    lab = {"A_original": [q["labels_A_original"]["class4"] for q in complete]}
    labA = [winner(q["labels_A_original"]["score"][S.DS], q["labels_A_original"]["score"][S.QW]) for q in complete]
    labB = {mode: [winner(tab[q["key"]][P.config_of(S.DS, mode)]["correct"], tab[q["key"]][P.config_of(S.QW, mode)]["correct"]) for q in complete]
            for mode in (S.EDGE, S.REMOTE)}

    def cross(a, b):
        return {f"{x}->{y}": c for (x, y), c in sorted(Counter(zip(a, b)).items())}
    out["labels"] = {
        "definition_B": "winner = the model that alone is correct within the cutoff under that execution mode; otherwise a tie "
                        "(both correct or both wrong); one stochastic draw per model",
        "A_original_counts": dict(Counter(labA)), "B_edge_counts": dict(Counter(labB[S.EDGE])), "B_remote_counts": dict(Counter(labB[S.REMOTE])),
        "A_vs_B_edge": cross(labA, labB[S.EDGE]), "A_vs_B_remote": cross(labA, labB[S.REMOTE]), "B_edge_vs_B_remote": cross(labB[S.EDGE], labB[S.REMOTE]),
        "agreement_A_B_edge": sum(a == b for a, b in zip(labA, labB[S.EDGE])) / n,
        "agreement_A_B_remote": sum(a == b for a, b in zip(labA, labB[S.REMOTE])) / n,
        "agreement_B_edge_B_remote": sum(a == b for a, b in zip(labB[S.EDGE], labB[S.REMOTE])) / n,
        "per_model_correctness_agreement_between_the_two_draws": {
            m: sum(tab[q["key"]][P.config_of(m, S.EDGE)]["correct"] == tab[q["key"]][P.config_of(m, S.REMOTE)]["correct"] for q in complete) / n
            for m in S.MODEL_ORDER},
        "per_model_correctness_agreement_with_original": {
            f"{m}_{mode}": sum(tab[q["key"]][P.config_of(m, mode)]["correct"] == q["labels_A_original"]["score"][m] for q in complete) / n
            for m in S.MODEL_ORDER for mode in (S.EDGE, S.REMOTE)}}

    # ---- 1. semantic router quality ----
    score = [q["router_pinned"]["score"] for q in complete]
    choice = [q["router_pinned"]["choice"] for q in complete]
    sem = {}
    for name, y in (("A_original", labA), ("B_edge", labB[S.EDGE]), ("B_remote", labB[S.REMOTE])):
        idx = [i for i, t in enumerate(y) if t != "tie"]
        yt, yp, ys = [y[i] for i in idx], [choice[i] for i in idx], [score[i] for i in idx]
        both = len(set(yt)) == 2
        f1s, aucs = [], []
        for w in W:
            rep = [i for i in idx for _ in range(w[i])]
            if len({y[i] for i in rep}) == 2:       # a resample without one winner class has no Macro-F1 or AUC: left out, counted
                f1s.append(macro_f1([y[i] for i in rep], [choice[i] for i in rep]))
                aucs.append(auc([y[i] for i in rep], [score[i] for i in rep]))
        sem[name] = {"n_with_winner": len(idx), "n_ties": n - len(idx), "winner_counts": dict(Counter(yt)),
                     "both_winner_classes_present": both,
                     "macro_f1": macro_f1(yt, yp) if both else None, "macro_f1_ci95": ci(f1s) if both and f1s else None,
                     "auc": auc(yt, ys), "auc_ci95": ci(aucs) if both and aucs else None,
                     "bootstrap_resamples_with_both_classes": len(f1s) / len(W) if W else None,
                     "accuracy_on_winner_queries": sum(a == b for a, b in zip(yt, yp)) / len(idx) if idx else None}
    sem["note"] = ("Macro-F1 is the primary semantic metric, on queries with a winner, as in the router study. Labels A and B are "
                   "different targets: A is the released one-shot label, B a new single draw under this protocol. A change of Macro-F1 "
                   "between them says nothing about training; no router was trained or tuned here.")
    sem["router_choice_counts"] = dict(Counter(choice))
    out["semantic_router"] = sem

    # ---- 3. end-to-end integrated use, per data policy ----
    out["data_policies"] = {}
    for pol in (S.POLICY_A, S.POLICY_B):
        mode = S.PLACEMENT[pol]
        feas = S.MODE_FEASIBLE[pol]
        y = labB[mode]
        res, vec = {}, {}
        for policy in POLICIES:
            picks = [choose(policy, q, tab[q["key"]], mode, feas) for q in complete]
            cells = [tab[q["key"]][cfg] for q, (_, cfg) in zip(complete, picks)]
            if policy in ("router", "lookup_D2"):
                over = [x if x is not None else med_router for x in router_s]
            else:
                over = [0.0] * n
            lat = [c["total_s"] + o for c, o in zip(cells, over) if c["total_s"] is not None]
            v = {"acc": [c["correct"] for c in cells]}
            for d in S.DEADLINES_S:
                v[f"u{d}"] = [on_time_correct(c, o, d) for c, o in zip(cells, over)]
            vec[policy] = v
            idx = [i for i, t in enumerate(y) if t != "tie"]
            models = [m for m, _ in picks]
            res[policy] = {
                "kind": LABELS[policy], "placement_mode": mode if policy != "hindsight_config" else "any feasible mode",
                "selection_counts": dict(Counter(S.MODELS[m]["name"] for m in models)),
                "configuration_counts": dict(Counter(cfg for _, cfg in picks)),
                "model_choice_vs_new_winner": {"target": f"labels B under {mode}", "n_with_winner": len(idx),
                                               "chose_the_winner": sum(models[i] == y[i] for i in idx),
                                               "macro_f1": macro_f1([y[i] for i in idx], [models[i] for i in idx])
                                               if len({y[i] for i in idx}) == 2 else None},
                "correct_within_cutoff_pct": 100.0 * sum(v["acc"]) / n,
                "correct_by_deadline_pct": {str(d): 100.0 * sum(v[f"u{d}"]) / n for d in S.DEADLINES_S},
                "correct_by_deadline_ci95": {str(d): [100.0 * x for x in ci([sum(a * b for a, b in zip(w, v[f"u{d}"])) / n for w in W])] for d in S.DEADLINES_S},
                "end_to_end_latency_s": {"median": quantile(lat, 0.5), "p90": quantile(lat, 0.9), "mean": sum(lat) / len(lat) if lat else None,
                                         "includes_router_latency": policy in ("router", "lookup_D2"),
                                         "overhead_s_median": quantile(over, 0.5)},
                "failures": sum(c["status"] == "error" for c in cells), "unscored": sum(c["unscored"] for c in cells),
                "data_policy_violations": sum(not P.feasible(S.CONFIGS[cfg][1], pol)[0] for _, cfg in picks)}
        diffs = {}
        for a_, b_ in (("router", "always_QW"), ("router", "always_DS"), ("router", "lookup_D2"), ("lookup_D2", "always_QW"),
                       ("always_QW", "always_DS"), ("hindsight_model", "router")):
            d = {}
            for k in ["acc"] + [f"u{x}" for x in S.DEADLINES_S]:
                dv = [x - y_ for x, y_ in zip(vec[a_][k], vec[b_][k])]
                bs = [100.0 * sum(w_ * x for w_, x in zip(w, dv)) / n for w in W]
                d[k] = {"delta_pt": 100.0 * sum(dv) / n, "ci95": ci(bs), "discordant": [sum(x > 0 for x in dv), sum(x < 0 for x in dv)]}
            diffs[f"{a_} minus {b_}"] = d
        out["data_policies"][pol] = {"placement_mode": mode, "feasible_modes": list(feas), "policies": res, "paired_differences": diffs,
                                     "interval": f"paired bootstrap, {n_boot} resamples, seed {seed}, stratified by benchmark, percentile"}
    out["live"] = {ph: live_compare(data, ph, tab, rt) for ph in ("live_A", "live_B")}
    return out


# --------------------------------------------------------------------------- #
# Live pass against the record-based router policy                             #
# --------------------------------------------------------------------------- #
def live_compare(data, phase, tab, rt):
    """The live pass against the record-based result of policy C. The record of a query is the one of the configuration
    policy C plans for it (the pinned router choice under the frozen placement), whatever the live pass executed. Both
    sides of every comparison use the same queries: those with a live row and a shared record of that configuration."""
    pol = S.POLICY_A if phase == "live_A" else S.POLICY_B
    mode = S.PLACEMENT[pol]
    by_key = {q["key"]: q for q in data["wl"]["queries"]}
    rows = {r["key"]: r for r in data["rows"][phase] if r["phase"] == phase}
    if not rows:
        return {"status": "NOT_RUN"}
    n = len(rows)
    planned_cfg = {k: P.config_of(by_key[k]["router_pinned"]["choice"], mode) for k in rows}
    sel_agree = sum(r.get("router", {}).get("choice") == by_key[k]["router_pinned"]["choice"] for k, r in rows.items())
    planned = sum(r["config"] == planned_cfg[k] for k, r in rows.items())
    pairs, corr, d_abs, d_rel, e2e_vs_rec = [], Counter(), [], [], []
    util_live, util_rec = Counter(), Counter()
    for k, r in rows.items():
        live = cell(r, data["scores"])
        rec = tab.get(k, {}).get(planned_cfg[k])
        if rec is None:
            continue
        pairs.append(k)
        e2e = r.get("end_to_end_s")
        for d in S.DEADLINES_S:
            util_live[d] += int(bool(live["correct"]) and e2e is not None and e2e <= d)
            util_rec[d] += on_time_correct(rec, rt[k]["router_s"] if k in rt else 0.0, d)
        corr[(rec["correct"], live["correct"])] += 1
        if r["config"] == planned_cfg[k] and rec["total_s"] is not None and live["total_s"] is not None:
            d_abs.append(live["total_s"] - rec["total_s"])
            d_rel.append((live["total_s"] - rec["total_s"]) / rec["total_s"] if rec["total_s"] else 0.0)
            if e2e is not None and k in rt:
                e2e_vs_rec.append(e2e - (rec["total_s"] + rt[k]["router_s"]))
    ok = [r for r in rows.values() if r["status"] == "ok"]
    m = max(1, len(pairs))
    stat = lambda xs: {"n": len(xs), "median": quantile(xs, 0.5), "p90": quantile(xs, 0.9), "mean": sum(xs) / len(xs) if xs else None,  # noqa: E731
                       "max": max(xs) if xs else None}
    return {"status": "COMPLETE" if f"lane_{phase}.done.json" in data["done"] else "PARTIAL", "data_policy": pol, "placement_mode": mode, "n": n,
            "reduced": bool(data["done"].get(f"lane_{phase}.done.json", {}).get("reduced")),
            "selection_agreement_with_pinned_router": sel_agree, "executed_the_planned_configuration": planned,
            "ids_match_pinned": sum(bool(r.get("ids_match_pinned")) for r in rows.values()),
            "the_backend_ran_the_selected_model": sum(r.get("backend_model") == r["model"] for r in ok),
            "n_compared_with_record": len(pairs),
            "correctness_record_vs_live": {"both_correct": corr[(1, 1)], "both_wrong": corr[(0, 0)], "record_only": corr[(1, 0)], "live_only": corr[(0, 1)],
                                           "note": "different stochastic draws (different seeds): equality is not expected"},
            "generation_time_live_minus_record_s": stat(d_abs), "generation_time_relative_difference": stat(d_rel),
            "end_to_end_live_minus_record_plus_router_s": stat(e2e_vs_rec),
            "router_s": stat([r["router_s"] for r in rows.values() if r.get("router_s") is not None]),
            "dispatch_overhead_s": stat([r["dispatch_overhead_s"] for r in rows.values() if r.get("dispatch_overhead_s") is not None]),
            "bytes_out": stat([r["bytes_out"] for r in ok if r.get("bytes_out") is not None]),
            "bytes_in": stat([r["bytes_in"] for r in ok if r.get("bytes_in") is not None]),
            "wire_and_edge_s": stat([r["wire_and_edge_s"] for r in ok if r.get("wire_and_edge_s") is not None]),
            "failures": sum(r["status"] == "error" for r in rows.values()), "fallbacks": sum(bool(r.get("fallback")) for r in rows.values()),
            "data_policy_violations": sum(not P.feasible(r["mode"], pol)[0] for r in rows.values()),
            "correct_by_deadline_pct_live": {str(d): 100.0 * util_live[d] / m for d in S.DEADLINES_S},
            "correct_by_deadline_pct_record_same_queries": {str(d): 100.0 * util_rec[d] / m for d in S.DEADLINES_S},
            "caveat": "One live pass shows that the pipeline works. It does not establish statistical stability."}


# --------------------------------------------------------------------------- #
# Report                                                                       #
# --------------------------------------------------------------------------- #
def markdown(o):
    f = lambda x, nd=1: "n/a" if x is None else f"{x:.{nd}f}"  # noqa: E731
    L = [f"# {S.NAME} summary ({o['protocol']})", "", f"Status: {o['status']}. Queries evaluated: {o['n_evaluated']} of {o['n_sample']}. "
         f"Single stochastic draw per configuration: {o['single_draw']}.", "", o["caveat"], ""]
    if o["status"] == "NO_COMPLETE_QUERY":
        return "\n".join(L)
    L += ["## Execution placement (each configuration by itself)", "",
          "| configuration | correct within cutoff % | " + " | ".join(f"correct by {d} s %" for d in S.DEADLINES_S) + " | median s | p90 s | failures |",
          "|---|---|" + "---|" * (len(S.DEADLINES_S) + 3)]
    for c, v in o["configurations"].items():
        L.append(f"| {c} | {f(v['correct_within_cutoff_pct'])} | " + " | ".join(f(v["correct_by_deadline_pct"][str(d)]) for d in S.DEADLINES_S)
                 + f" | {f(v['completion_s']['median'])} | {f(v['completion_s']['p90'])} | {v['failures']} |")
    L += ["", "## Semantic router quality", "", "| labels | queries with a winner | ties | Macro-F1 | 95% interval | AUC | 95% interval |",
          "|---|---|---|---|---|---|---|"]
    iv_ = lambda iv: f"[{f(iv[0], 3)}, {f(iv[1], 3)}]" if iv else "n/a"  # noqa: E731
    for k in ("A_original", "B_edge", "B_remote"):
        v = o["semantic_router"][k]
        L.append(f"| {k} | {v['n_with_winner']} | {v['n_ties']} | {f(v['macro_f1'], 3)} | {iv_(v['macro_f1_ci95'])} | {f(v['auc'], 3)} | {iv_(v['auc_ci95'])} |")
    L += ["", o["semantic_router"]["note"], ""]
    for pol, v in o["data_policies"].items():
        L += [f"## End-to-end, data policy {pol} (placement: {v['placement_mode']})", "",
              "| policy | kind | selections | correct within cutoff % | " + " | ".join(f"by {d} s %" for d in S.DEADLINES_S) + " | median s | failures | violations |",
              "|---|---|---|---|" + "---|" * (len(S.DEADLINES_S) + 3)]
        for p, r in v["policies"].items():
            L.append(f"| {p} | {r['kind']} | {r['selection_counts']} | {f(r['correct_within_cutoff_pct'])} | "
                     + " | ".join(f(r["correct_by_deadline_pct"][str(d)]) for d in S.DEADLINES_S)
                     + f" | {f(r['end_to_end_latency_s']['median'])} | {r['failures']} | {r['data_policy_violations']} |")
        L += ["", "| paired difference | correct within cutoff, points [95% interval] | " + " | ".join(f"by {d} s" for d in S.DEADLINES_S) + " |",
              "|---|---|" + "---|" * len(S.DEADLINES_S)]
        for name, d in v["paired_differences"].items():
            L.append(f"| {name} | " + " | ".join(f"{d[k]['delta_pt']:+.1f} [{d[k]['ci95'][0]:+.1f}, {d[k]['ci95'][1]:+.1f}]" for k in ["acc"] + [f"u{x}" for x in S.DEADLINES_S]) + " |")
        L.append("")
    for ph, v in o["live"].items():
        L += [f"## Live pipeline, {ph}", ""]
        if v["status"] == "NOT_RUN":
            L += ["Not run.", ""]
            continue
        L += [f"- status {v['status']}, {v['n']} queries (compared with a record: {v['n_compared_with_record']}), failures {v['failures']}, "
              f"fallbacks {v['fallbacks']}, data-policy violations {v['data_policy_violations']}",
              f"- router selection equal to the pinned decision: {v['selection_agreement_with_pinned_router']} of {v['n']}; "
              f"backend ran the selected model: {v['the_backend_ran_the_selected_model']}",
              f"- correctness, record against live: {v['correctness_record_vs_live']}",
              f"- generation time, live minus record (s): {v['generation_time_live_minus_record_s']}",
              f"- router latency (s): {v['router_s']}; dispatch overhead (s): {v['dispatch_overhead_s']}",
              f"- correct by deadline %, live: {v['correct_by_deadline_pct_live']}; record, same queries: {v['correct_by_deadline_pct_record_same_queries']}",
              f"- {v['caveat']}", ""]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edge-run", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workload")
    ap.add_argument("--n-boot", type=int, default=S.BOOT_N)
    ap.add_argument("--dev", action="store_true")
    a = ap.parse_args(argv)
    if (a.workload or a.n_boot != S.BOOT_N) and not a.dev:
        raise SystemExit("STOP: --workload and --n-boot are for --dev only")
    data = load(a.edge_run, a.scores, a.workload, a.dev)
    o = evaluate(data, a.n_boot)
    o["dev"] = bool(a.dev)
    out = Path(a.out_dir)
    C.write_json(out / "summary.json", o)
    (out / "SUMMARY.md").write_text(("**DEV FIXTURE: not measurements.**\n\n" if a.dev else "") + markdown(o), encoding="utf-8")
    print(json.dumps({"status": o["status"], "n_evaluated": o["n_evaluated"], "out": str(out)}))


if __name__ == "__main__":
    main()
