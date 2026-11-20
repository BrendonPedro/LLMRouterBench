"""QWEN-SELECT rules: deadline utility, the decision rule under each data rule, the same-model comparison, stability,
the zero-network counterfactual and the outcome.

Pure functions on recorded data (standard library only), so every verdict can be recomputed from the raw outputs.
A decision file: never edited after the tag.
"""
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qsel_spec as S  # noqa: E402


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
# Deadline utility and the decision rule                                       #
# --------------------------------------------------------------------------- #
def utility(rows, config, deadline):
    """Share of queries answered correctly within the deadline. rows: [{"bench", "correct": {c: 0/1}, "time": {c: s}}]."""
    return sum(r["correct"][config] * (r["time"][config] <= deadline) for r in rows) / len(rows)


def best_of(us, configs):
    """The configuration of `configs` with the highest utility; ties go to the earlier entry of qsel_spec.TIE_ORDER."""
    return max(configs, key=lambda c: (us[c], -S.TIE_ORDER.index(c)))


def delta(rows, deadline, competitors):
    """U(S) minus the highest utility among `competitors`, in points; the utilities; the competitor attaining the maximum."""
    us = {c: utility(rows, c, deadline) for c in ("S",) + tuple(competitors)}
    best = best_of(us, competitors)
    return 100.0 * (us["S"] - us[best]), us, best


def bootstrap_delta(rows, deadline, competitors, label, n_boot=None, seed=None, level=None):
    """Paired bootstrap, stratified by benchmark, the competitor maximum re-selected inside every replicate (E0's rule)."""
    n_boot, seed, level = n_boot or S.BOOT_N, S.BOOT_SEED if seed is None else seed, level or S.CI_LEVEL
    strata = {}
    for r in rows:
        strata.setdefault(r["bench"], []).append(r)
    groups = [strata[b] for b in sorted(strata)]
    rng = random.Random(f"{seed}:{label}:{deadline}")
    stats = []
    for _ in range(n_boot):
        sample = [g[rng.randrange(len(g))] for g in groups for _ in range(len(g))]
        stats.append(delta(sample, deadline, competitors)[0])
    a = (1.0 - level) / 2.0
    return [quantile(stats, a), quantile(stats, 1.0 - a)]


def passes(delta_pt, ci_low):
    """Split is selected only if delta, rounded to 1e-6 points, is at least the margin and the interval's lower end is above zero."""
    return bool(ci_low is not None and round(delta_pt, S.DELTA_DECIMALS) >= S.MARGIN_PT and ci_low > 0)


def decide(rows, competitors, label, deadlines=None, n_boot=None):
    """One cell per deadline: Split against the best of `competitors`."""
    cells = []
    for d in deadlines or S.DEADLINES_S:
        dl, us, best = delta(rows, d, competitors)
        evaluated = max(us.values()) >= S.MIN_CELL_UTILITY
        ci = bootstrap_delta(rows, d, competitors, label, n_boot=n_boot) if evaluated else [None, None]
        split = evaluated and passes(dl, ci[0])
        cells.append({"deadline_s": d, "utility": us, "best_competitor": best, "delta_pt": dl, "ci95": ci, "evaluated": evaluated,
                      "selected": "S" if split else (best if evaluated else None)})
    return cells


def tables(rows, have_remote, n_boot=None):
    """The decision under each data rule and the same-model comparison. The open rule is evaluated only if REMOTE ran."""
    out = {"no-raw-text": decide(rows, S.POLICIES["no-raw-text"], "no-raw-text", n_boot=n_boot),
           "same-model": decide(rows, (S.SAME_MODEL_EDGE,), "same-model", n_boot=n_boot)}
    if have_remote:
        out["open"] = decide(rows, S.POLICIES["open"], "open", n_boot=n_boot)
    return out


def stability(s1_rows, s2_rows):
    """Median completion time of the second split pass over the first, on the sentinel queries."""
    t1 = {r["key"]: r["total_s"] for r in s1_rows}
    pairs = [(t1[r["key"]], r["total_s"]) for r in s2_rows if r["key"] in t1]
    if not pairs:
        return {"ratio": None, "stable": False, "n": 0}
    ratio = median([b for _, b in pairs]) / median([a for a, _ in pairs])
    return {"ratio": ratio, "stable": S.STABILITY_BAND[0] <= ratio <= S.STABILITY_BAND[1], "n": len(pairs)}


def zero_network(rows, split_wire_s, n_boot=None):
    """Counterfactual of the screen's network-free bound, from the split run's own counters: the split's completion
    time minus everything of each round trip that the server did not report as compute. It can only shorten the
    split's time, so it is an upper bound on the split's utility. Reported, never a decision."""
    b_rows = []
    for r in rows:
        t = r["time"]["S"]
        b_rows.append(dict(r, time=dict(r["time"], S=t - split_wire_s.get(r["key"], 0.0) if t != float("inf") else t)))
    cells = decide(b_rows, S.POLICIES[S.PRIMARY_POLICY], "zero-network", n_boot=n_boot)
    return {"cells": cells, "would_select_split": any(c["selected"] == "S" for c in cells)}


# --------------------------------------------------------------------------- #
# Outcome                                                                      #
# --------------------------------------------------------------------------- #
def outcome(correctness_pass, primary_cells, stable, complete, split_ran=True, reference_ran=True):
    """The primary outcome, from the no-raw-text cells (PREREG section 13). correctness_pass: True, False or None (no verdict)."""
    if not reference_ran:
        return "NOT_EVALUABLE"
    if not split_ran:
        return "SPLIT_INFEASIBLE"
    if correctness_pass is not True:
        return "CORRECTNESS_FAIL" if correctness_pass is False else "INCOMPLETE"
    if not complete:
        return "INCOMPLETE"
    if not stable:
        return "UNSTABLE"
    if not any(c["evaluated"] for c in primary_cells):
        return "NOT_EVALUABLE"
    return "SCREEN_CONTRADICTED" if any(c["selected"] == "S" for c in primary_cells) else "SCREEN_CONFIRMED"
