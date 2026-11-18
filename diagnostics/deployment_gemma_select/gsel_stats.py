"""GEMMA-SELECT rules: correctness gate, deadline utility, the decision rule, the timing bound and the outcome.

Pure functions on recorded data (standard library only), so every verdict can be recomputed from the raw outputs.
A decision file: never edited after the tag.
"""
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gsel_spec as S  # noqa: E402


def quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    h = (len(xs) - 1) * q
    lo = math.floor(h)
    hi = min(lo + 1, len(xs) - 1)
    return float(xs[lo] + (h - lo) * (xs[hi] - xs[lo]))


def median(xs):
    return quantile(xs, 0.5)


# --------------------------------------------------------------------------- #
# Correctness gate (split against the whole float32 model, no timers)          #
# --------------------------------------------------------------------------- #
def correctness_gate(positions, greedy):
    """positions: [{"ref_top1", "split_top1", "ref_margin", "kl", "finite"}] over all teacher-forced positions.
    greedy: [{"identical", "first_divergence", "ref_margin_there"}] per prompt. Returns the verdict with its inputs."""
    decisive = [p for p in positions if p["ref_margin"] >= S.MARGIN_NATS]
    agree = sum(p["ref_top1"] == p["split_top1"] for p in decisive)
    share = agree / len(decisive) if decisive else None
    finite = all(p["finite"] for p in positions) and bool(positions)
    kls = [p["kl"] for p in positions if p["finite"]]
    mean_kl = sum(kls) / len(kls) if kls else None
    ok_greedy = sum(1 for g in greedy if g["identical"] or (g.get("ref_margin_there") is not None and g["ref_margin_there"] < S.MARGIN_NATS))
    checks = {"finite": finite,
              "decisive_agreement": share is not None and share >= S.MIN_DECISIVE_AGREEMENT,
              "mean_kl": mean_kl is not None and mean_kl <= S.MAX_MEAN_KL_NATS,
              "greedy": len(greedy) == S.N_CORRECTNESS_SHORT + S.N_CORRECTNESS_LONG and ok_greedy >= S.MIN_GREEDY_OK}
    return {"verdict": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "n_positions": len(positions),
            "n_decisive": len(decisive), "decisive_disagreements": len(decisive) - agree, "decisive_agreement": share,
            "mean_kl_nats": mean_kl, "max_kl_nats": max(kls) if kls else None, "greedy_ok": ok_greedy, "n_prompts": len(greedy),
            "greedy_identical": sum(1 for g in greedy if g["identical"]),
            "thresholds": {"margin_nats": S.MARGIN_NATS, "min_decisive_agreement": S.MIN_DECISIVE_AGREEMENT,
                           "max_mean_kl_nats": S.MAX_MEAN_KL_NATS, "min_greedy_ok": S.MIN_GREEDY_OK}}


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


# --------------------------------------------------------------------------- #
# Deadline utility and the decision rule                                       #
# --------------------------------------------------------------------------- #
def utility(rows, config, deadline):
    """Share of queries answered correctly within the deadline. rows: [{"bench", "correct": {c: 0/1}, "time": {c: s}}]."""
    return sum(r["correct"][config] * (r["time"][config] <= deadline) for r in rows) / len(rows)


def delta(rows, deadline):
    """U(split) - max over the edge configurations, in points, with the edge configuration that attains the maximum."""
    us = {c: utility(rows, c, deadline) for c in S.CONFIGS}
    best = max(S.EDGE_CONFIGS, key=lambda c: (us[c], c))
    return 100.0 * (us["S"] - us[best]), us, best


def bootstrap_delta(rows, deadline, n_boot=None, seed=None, level=None):
    """Paired bootstrap, stratified by benchmark, the edge maximum re-selected inside every replicate (E0's rule)."""
    n_boot, seed, level = n_boot or S.BOOT_N, S.BOOT_SEED if seed is None else seed, level or S.CI_LEVEL
    strata = {}
    for r in rows:
        strata.setdefault(r["bench"], []).append(r)
    groups = [strata[b] for b in sorted(strata)]
    rng = random.Random(f"{seed}:{deadline}")
    stats = []
    for _ in range(n_boot):
        sample = [g[rng.randrange(len(g))] for g in groups for _ in range(len(g))]
        stats.append(delta(sample, deadline)[0])
    a = (1.0 - level) / 2.0
    return [quantile(stats, a), quantile(stats, 1.0 - a)]


def decide(rows, deadlines=None, n_boot=None):
    """One cell per deadline. Split is selected only if delta >= 3.0 points and the CI lower end is above zero."""
    cells = []
    for d in deadlines or S.DEADLINES_S:
        dl, us, best = delta(rows, d)
        evaluated = max(us.values()) >= S.MIN_CELL_UTILITY
        ci = bootstrap_delta(rows, d, n_boot=n_boot) if evaluated else [None, None]
        split = bool(evaluated and dl >= S.MARGIN_PT and ci[0] is not None and ci[0] > 0)
        cells.append({"deadline_s": d, "utility": us, "best_edge": best, "delta_pt": dl, "ci95": ci, "evaluated": evaluated,
                      "selected": "SPLIT" if split else ("EDGE" if evaluated else None)})
    return cells


def stability(s1_rows, s2_rows):
    """Median completion time of the second split pass over the first, on the sentinel queries."""
    t1 = {r["key"]: r["total_s"] for r in s1_rows}
    pairs = [(t1[r["key"]], r["total_s"]) for r in s2_rows if r["key"] in t1]
    if not pairs:
        return {"ratio": None, "stable": False, "n": 0}
    ratio = median([b for _, b in pairs]) / median([a for a, _ in pairs])
    return {"ratio": ratio, "stable": S.STABILITY_BAND[0] <= ratio <= S.STABILITY_BAND[1], "n": len(pairs)}


# --------------------------------------------------------------------------- #
# Secondary: the screen's timing bound from isolated segment profiles          #
# --------------------------------------------------------------------------- #
def fit_line(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
    return my - b * mx, b


def segment_model(profile):
    """profile: [{"q", "prefill_s", "decode_s": [...]}]. Prefill time is a least-squares line in q, clipped at zero
    slope and intercept; a decode step is the median over every recorded step."""
    a, b = fit_line([p["q"] for p in profile], [p["prefill_s"] for p in profile])
    return {"prefill_intercept_s": max(a, 0.0), "prefill_slope_s": max(b, 0.0),
            "decode_s": median([t for p in profile for t in p["decode_s"]])}


def bound_time(model_edge, model_server, n_prompt, n_out):
    seg = lambda m: m["prefill_intercept_s"] + m["prefill_slope_s"] * n_prompt + max(n_out - 1, 0) * m["decode_s"]  # noqa: E731
    return S.BOUND_DISCOUNT * (seg(model_edge) + seg(model_server))


def bound_analysis(rows, split_meta, model_edge, model_server, n_boot=None):
    """rows as in `decide`; split_meta: {key: (n_prompt, n_out)}. The bound replaces the split's measured time."""
    b_rows, ge = [], 0
    for r in rows:
        tb = bound_time(model_edge, model_server, *split_meta[r["key"]])
        ge += r["time"]["S"] >= tb
        b_rows.append(dict(r, time=dict(r["time"], S=tb)))
    cells = decide(b_rows, n_boot=n_boot)
    share = ge / len(rows)
    return {"share_measured_at_least_bound": share, "optimistic": share >= S.BOUND_OPTIMISTIC_SHARE,
            "would_stop": not any(c["selected"] == "SPLIT" for c in cells), "cells": cells}


# --------------------------------------------------------------------------- #
# Outcome                                                                      #
# --------------------------------------------------------------------------- #
def outcome(correctness_verdict, cells, stable, complete, split_ran=True, matched_edge_ran=True):
    if not matched_edge_ran:
        return "NOT_EVALUABLE"
    if not split_ran:
        return "SPLIT_INFEASIBLE"
    if correctness_verdict != "PASS":
        return "CORRECTNESS_FAIL" if correctness_verdict == "FAIL" else "INCOMPLETE"
    if not complete:
        return "INCOMPLETE"
    if not stable:
        return "UNSTABLE"
    if not any(c["evaluated"] for c in cells):
        return "NOT_EVALUABLE"
    return "SCREEN_CONTRADICTED" if any(c["selected"] == "SPLIT" for c in cells) else "SCREEN_CONFIRMED"
