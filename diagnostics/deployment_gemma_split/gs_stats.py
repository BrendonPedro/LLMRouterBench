"""GEMMA-SPLIT statistics (standard library only; PREREG §11.5).

Quantiles use linear interpolation between order statistics (Hyndman-Fan type 7, the numpy default). The
efficiency CI is a paired, group-stratified percentile bootstrap over prompts (B = 10,000, seed 20261116):
prompts are resampled with replacement within each timing group, the statistic is recomputed on the resample.
"""
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # python -I: the script dir is not on sys.path
import gs_spec as S  # noqa: E402


def quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    if len(xs) == 1:
        return float(xs[0])
    h = (len(xs) - 1) * q
    lo = math.floor(h)
    hi = min(lo + 1, len(xs) - 1)
    return float(xs[lo] + (h - lo) * (xs[hi] - xs[lo]))


def median(xs):
    return quantile(xs, 0.5)


def summarize(xs):
    xs = [float(x) for x in xs]
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "median": median(xs), "p10": quantile(xs, 0.10), "p90": quantile(xs, 0.90), "mean": sum(xs) / len(xs),
            "min": min(xs), "max": max(xs)}


def ratio_of_medians(a, b):
    mb = median(b)
    return None if not mb else median(a) / mb


def paired_bootstrap_ratio(pairs, n_boot=S.BOOT_N, seed=S.BOOT_SEED, level=S.CI_LEVEL):
    """pairs: {group: [(a_i, b_i), ...]} paired by prompt. Statistic: median(a) / median(b) over all prompts.
    Returns {"point", "ci": [lo, hi], "n_pairs", "n_boot", "seed"}. Deterministic for a given input order."""
    groups = [list(pairs[g]) for g in sorted(pairs)]
    flat = [p for g in groups for p in g]
    if not flat:
        return {"point": None, "ci": [None, None], "n_pairs": 0, "n_boot": n_boot, "seed": seed}
    point = ratio_of_medians([a for a, _ in flat], [b for _, b in flat])
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        sa, sb = [], []
        for g in groups:
            for _ in range(len(g)):
                a, b = g[rng.randrange(len(g))]
                sa.append(a)
                sb.append(b)
        r = ratio_of_medians(sa, sb)
        if r is not None:
            stats.append(r)
    alpha = (1.0 - level) / 2.0
    return {"point": point, "ci": [quantile(stats, alpha), quantile(stats, 1.0 - alpha)], "n_pairs": len(flat), "n_boot": n_boot,
            "seed": seed, "level": level}


def efficiency_category(ci):
    """PREREG §11.4: from the CI of median TPOT(split) / median TPOT(edge-only)."""
    lo, hi = ci
    if lo is None or hi is None:
        return None
    if hi < 1.0:
        return "SPLIT_FASTER_THAN_EDGE"
    if lo > 1.0:
        return "SPLIT_SLOWER_THAN_EDGE"
    return "NOT_DISTINGUISHABLE_FROM_EDGE"
