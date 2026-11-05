#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "diagnostics" / "winner_label_stability"

SCORES = DIR / "main_scores.csv"
STABILITY = DIR / "query_stability.csv"

DS = "DeepSeek-R1-0528-Qwen3-8B"
QW = "Qwen3-8B"

SIGNAL = {"bbh", "korbench", "mbpp"}
CHANCE = {"meld", "medqa", "finqa"}


def repeated_preference(x):
    if x > 0:
        return DS
    if x < 0:
        return QW
    return "tie"


def summarize(g):
    nonzero = g[g["utility_margin"] != 0]
    return pd.Series({
        "n_queries": len(g),
        "mean_abs_margin": g["abs_margin"].mean(),
        "median_abs_margin": g["abs_margin"].median(),
        "zero_margin_queries": int((g["utility_margin"] == 0).sum()),
        "zero_margin_rate": (g["utility_margin"] == 0).mean(),
        "weak_margin_le_0.2_rate": (g["abs_margin"] <= 0.2 + 1e-12).mean(),
        "strong_margin_ge_0.4_rate": (g["abs_margin"] >= 0.4 - 1e-12).mean(),
        "ds_preference_queries": int((g["repeated_preference"] == DS).sum()),
        "qwen_preference_queries": int((g["repeated_preference"] == QW).sum()),
        "original_agreement_nonzero_n": len(nonzero),
        "original_agreement_nonzero_rate":
            nonzero["agrees_with_original"].mean() if len(nonzero) else float("nan"),
        "winner_stable_rate": g["winner_stable"].mean(),
    })


scores = pd.read_csv(SCORES)
stability = pd.read_csv(STABILITY)

if len(scores) != 1200:
    raise SystemExit(f"expected 1200 score rows, found {len(scores)}")

if scores["eval_error"].notna().any():
    raise SystemExit("evaluation errors present")

counts = (
    scores.groupby(["dataset", "index", "model"])
    .size()
)

if not (counts == 5).all():
    raise SystemExit("not every query/model has exactly 5 scored repeats")

rates = (
    scores.groupby(["dataset", "index", "model"])["fresh_score"]
    .mean()
    .unstack("model")
    .reset_index()
)

for model in (DS, QW):
    if model not in rates:
        raise SystemExit(f"missing model: {model}")

rates = rates.rename(columns={
    DS: "deepseek_success_rate",
    QW: "qwen_success_rate",
})

rates["utility_margin"] = (
    rates["deepseek_success_rate"] -
    rates["qwen_success_rate"]
)

rates["utility_margin"] = rates["utility_margin"].round(10)
rates["abs_margin"] = rates["utility_margin"].abs()
rates["repeated_preference"] = rates["utility_margin"].map(repeated_preference)

keep = stability[
    [
        "dataset",
        "index",
        "original_winner",
        "winner_stable",
        "original_winner_repeat_wins",
        "opposite_winner_repeat_wins",
        "tie_repeats",
    ]
]

out = rates.merge(
    keep,
    on=["dataset", "index"],
    how="left",
    validate="one_to_one",
)

if out["original_winner"].isna().any():
    raise SystemExit("failed to join query stability data")

out["agrees_with_original"] = out.apply(
    lambda r: (
        pd.NA
        if r["repeated_preference"] == "tie"
        else r["repeated_preference"] == r["original_winner"]
    ),
    axis=1,
)

out["group"] = out["dataset"].map(
    lambda x: "signal" if x in SIGNAL else "chance" if x in CHANCE else "unexpected"
)

if (out["group"] == "unexpected").any():
    raise SystemExit("unexpected benchmark")

out.to_csv(DIR / "utility_margin_diagnostic.csv", index=False)

bench = (
    out.groupby("dataset", sort=True)
    .apply(summarize, include_groups=False)
    .reset_index()
)
bench.to_csv(DIR / "utility_margin_benchmark.csv", index=False)

groups = (
    out.groupby("group", sort=True)
    .apply(summarize, include_groups=False)
    .reset_index()
)
groups.to_csv(DIR / "utility_margin_groups.csv", index=False)

stable_compare = (
    out.groupby("winner_stable", sort=True)
    .agg(
        n_queries=("index", "size"),
        mean_abs_margin=("abs_margin", "mean"),
        median_abs_margin=("abs_margin", "median"),
        zero_margin_rate=("utility_margin", lambda x: (x == 0).mean()),
        weak_margin_le_0_2_rate=("abs_margin", lambda x: (x <= 0.2 + 1e-12).mean()),
        strong_margin_ge_0_4_rate=("abs_margin", lambda x: (x >= 0.4 - 1e-12).mean()),
    )
    .reset_index()
)
stable_compare.to_csv(DIR / "utility_margin_stable_vs_unstable.csv", index=False)

dist = (
    out.groupby(["group", "abs_margin"])
    .size()
    .rename("n_queries")
    .reset_index()
)
dist.to_csv(DIR / "utility_margin_distribution.csv", index=False)

print("=== OVERALL ===")
print("queries:", len(out))
print("mean |margin|:", round(out["abs_margin"].mean(), 4))
print("median |margin|:", round(out["abs_margin"].median(), 4))
print("zero-margin rate:", round((out["utility_margin"] == 0).mean(), 4))
print("weak |margin| <= 0.2:", round((out["abs_margin"] <= 0.2 + 1e-12).mean(), 4))
print("strong |margin| >= 0.4:", round((out["abs_margin"] >= 0.4 - 1e-12).mean(), 4))

nonzero = out[out["utility_margin"] != 0]
print(
    "original-winner agreement among nonzero margins:",
    round(nonzero["agrees_with_original"].mean(), 4),
    f"({len(nonzero)} queries)"
)

print("\n=== BENCHMARK ===")
print(
    bench[
        [
            "dataset",
            "n_queries",
            "mean_abs_margin",
            "zero_margin_rate",
            "weak_margin_le_0.2_rate",
            "strong_margin_ge_0.4_rate",
            "original_agreement_nonzero_rate",
            "winner_stable_rate",
        ]
    ].to_string(index=False)
)

print("\n=== SIGNAL VS CHANCE ===")
print(
    groups[
        [
            "group",
            "n_queries",
            "mean_abs_margin",
            "zero_margin_rate",
            "weak_margin_le_0.2_rate",
            "strong_margin_ge_0.4_rate",
            "original_agreement_nonzero_rate",
            "winner_stable_rate",
        ]
    ].to_string(index=False)
)

print("\n=== FROZEN STABLE VS UNSTABLE LABELS ===")
print(stable_compare.to_string(index=False))

print("\n=== ABSOLUTE MARGIN DISTRIBUTION ===")
print(dist.to_string(index=False))
