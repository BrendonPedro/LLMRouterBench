#!/usr/bin/env python3
"""Target-only diagnostic for winner-label stability.

Uses the EXISTING held-out MiniLM MLP-64 B_student_score from the previous
5-fold experiment. No model is retrained.

Compare on exactly the same nonzero repeated-margin queries:
  A: original single-draw winner
  B: sign of 5-repeat utility margin

Classes:
  0 = DeepSeek
  1 = Qwen
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
WLS = ROOT / "diagnostics" / "winner_label_stability"

PRED_PATH = ROOT / "diagnostics" / "capability_profile" / "heldout_predictions.csv"
MARGIN_PATH = WLS / "utility_margin_diagnostic.csv"

OUT_ROWS = WLS / "target_only_eval_rows.csv"
OUT_BENCH = WLS / "target_only_eval_benchmark.csv"
OUT_FOLDS = WLS / "target_only_eval_folds.csv"
OUT_SUMMARY = WLS / "target_only_eval_summary.csv"


def auc_safe(y, score):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, score))


def metric_row(df, target_col):
    y = df[target_col].astype(int).to_numpy()
    score = df["B_student_score"].to_numpy()
    pred = (score > 0).astype(int)

    return {
        "n": len(df),
        "deepseek_n": int((y == 0).sum()),
        "qwen_n": int((y == 1).sum()),
        "macro_f1": float(
            f1_score(
                y,
                pred,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
        "auc": auc_safe(y, score),
        "pred_qwen_rate": float((pred == 1).mean()),
    }


pred = pd.read_csv(PRED_PATH)
margin = pd.read_csv(MARGIN_PATH)

pred["index"] = pred["index"].astype(int)
margin["index"] = margin["index"].astype(int)

# Binary repeated-preference comparison only.
margin = margin[margin["utility_margin"] != 0].copy()

margin["repeated_target"] = np.where(
    margin["utility_margin"] > 0,
    0,  # DeepSeek preferred
    1,  # Qwen preferred
)

margin["original_target_from_wls"] = np.where(
    margin["original_winner"] == "DeepSeek-R1-0528-Qwen3-8B",
    0,
    1,
)

keep_pred = pred[
    [
        "fold",
        "dataset",
        "index",
        "target",
        "B_student_score",
        "B_pred",
    ]
].copy()

rows = margin.merge(
    keep_pred,
    on=["dataset", "index"],
    how="inner",
    validate="one_to_one",
)

if len(rows) != 103:
    raise SystemExit(
        f"expected exactly 103 nonzero-margin queries, joined {len(rows)}"
    )

if rows.duplicated(["dataset", "index"]).any():
    raise SystemExit("duplicate query after join")

if not (
    rows["target"].astype(int)
    == rows["original_target_from_wls"].astype(int)
).all():
    bad = rows[
        rows["target"].astype(int)
        != rows["original_target_from_wls"].astype(int)
    ]
    raise SystemExit(
        f"original-target mismatch on {len(bad)} queries"
    )

if not (
    rows["B_pred"].astype(int)
    == (rows["B_student_score"] > 0).astype(int)
).all():
    raise SystemExit("B_pred != B_student_score > 0")

rows["original_target"] = rows["target"].astype(int)
rows["target_agrees"] = (
    rows["original_target"] == rows["repeated_target"]
)

rows.to_csv(OUT_ROWS, index=False)

# ------------------------------------------------------------------
# Fold-level comparison
# ------------------------------------------------------------------
fold_rows = []

for fold, sub in rows.groupby("fold", sort=True):
    for target_name, col in [
        ("original_single_draw", "original_target"),
        ("repeated_5run_preference", "repeated_target"),
    ]:
        fold_rows.append(
            {
                "fold": int(fold),
                "target": target_name,
                **metric_row(sub, col),
            }
        )

folds = pd.DataFrame(fold_rows)
folds.to_csv(OUT_FOLDS, index=False)

# ------------------------------------------------------------------
# Within-benchmark comparison
# ------------------------------------------------------------------
bench_rows = []

for dataset, sub in rows.groupby("dataset", sort=True):
    a = metric_row(sub, "original_target")
    b = metric_row(sub, "repeated_target")

    bench_rows.append(
        {
            "dataset": dataset,
            "n": len(sub),

            "original_ds_n": a["deepseek_n"],
            "original_qwen_n": a["qwen_n"],
            "original_macro_f1": a["macro_f1"],
            "original_balanced_acc": a["balanced_acc"],
            "original_auc": a["auc"],

            "repeated_ds_n": b["deepseek_n"],
            "repeated_qwen_n": b["qwen_n"],
            "repeated_macro_f1": b["macro_f1"],
            "repeated_balanced_acc": b["balanced_acc"],
            "repeated_auc": b["auc"],

            "delta_macro_f1": b["macro_f1"] - a["macro_f1"],
            "delta_auc": (
                b["auc"] - a["auc"]
                if not np.isnan(a["auc"]) and not np.isnan(b["auc"])
                else np.nan
            ),
        }
    )

bench = pd.DataFrame(bench_rows)
bench.to_csv(OUT_BENCH, index=False)

valid_auc = bench[
    bench["original_auc"].notna()
    & bench["repeated_auc"].notna()
]

# ------------------------------------------------------------------
# Summary on exact same 103 held-out queries
# ------------------------------------------------------------------
oa = metric_row(rows, "original_target")
rb = metric_row(rows, "repeated_target")

fold_orig = folds[folds["target"] == "original_single_draw"]
fold_rep = folds[folds["target"] == "repeated_5run_preference"]

summary = pd.DataFrame(
    [
        {
            "comparison": "original_single_draw",
            "n_queries": len(rows),
            "coverage_of_120": len(rows) / 120,
            "overall_macro_f1": oa["macro_f1"],
            "overall_balanced_acc": oa["balanced_acc"],
            "overall_auc": oa["auc"],
            "fold_mean_macro_f1": fold_orig["macro_f1"].mean(),
            "fold_std_macro_f1": fold_orig["macro_f1"].std(ddof=1),
            "fold_mean_auc": fold_orig["auc"].mean(),
            "within_macro_f1": bench["original_macro_f1"].mean(),
            "within_macro_auc": valid_auc["original_auc"].mean(),
            "within_median_auc": valid_auc["original_auc"].median(),
            "n_valid_auc_benchmarks": len(valid_auc),
        },
        {
            "comparison": "repeated_5run_preference",
            "n_queries": len(rows),
            "coverage_of_120": len(rows) / 120,
            "overall_macro_f1": rb["macro_f1"],
            "overall_balanced_acc": rb["balanced_acc"],
            "overall_auc": rb["auc"],
            "fold_mean_macro_f1": fold_rep["macro_f1"].mean(),
            "fold_std_macro_f1": fold_rep["macro_f1"].std(ddof=1),
            "fold_mean_auc": fold_rep["auc"].mean(),
            "within_macro_f1": bench["repeated_macro_f1"].mean(),
            "within_macro_auc": valid_auc["repeated_auc"].mean(),
            "within_median_auc": valid_auc["repeated_auc"].median(),
            "n_valid_auc_benchmarks": len(valid_auc),
        },
    ]
)

summary.to_csv(OUT_SUMMARY, index=False)

print("=== COVERAGE / LABEL AGREEMENT ===")
print(f"nonzero repeated-margin queries: {len(rows)}/120")
print(
    "original vs repeated target agreement:",
    f"{int(rows['target_agrees'].sum())}/{len(rows)}",
    f"= {rows['target_agrees'].mean():.4f}",
)

print("\n=== SAME HELD-OUT B SCORES, TARGET ONLY CHANGED ===")
print(
    summary[
        [
            "comparison",
            "overall_macro_f1",
            "overall_auc",
            "fold_mean_macro_f1",
            "fold_mean_auc",
            "within_macro_f1",
            "within_macro_auc",
            "within_median_auc",
            "n_valid_auc_benchmarks",
        ]
    ].to_string(index=False)
)

a = summary.iloc[0]
b = summary.iloc[1]

print("\n=== DELTA: REPEATED - ORIGINAL ===")
print(
    "overall Macro-F1:",
    round(b["overall_macro_f1"] - a["overall_macro_f1"], 4),
)
print(
    "overall AUC:",
    round(b["overall_auc"] - a["overall_auc"], 4),
)
print(
    "fold-mean Macro-F1:",
    round(b["fold_mean_macro_f1"] - a["fold_mean_macro_f1"], 4),
)
print(
    "within macro Macro-F1:",
    round(b["within_macro_f1"] - a["within_macro_f1"], 4),
)
print(
    "within macro AUC:",
    round(b["within_macro_auc"] - a["within_macro_auc"], 4),
)

print("\n=== PER BENCHMARK ===")
print(
    bench[
        [
            "dataset",
            "n",
            "original_macro_f1",
            "repeated_macro_f1",
            "delta_macro_f1",
            "original_auc",
            "repeated_auc",
            "delta_auc",
        ]
    ].to_string(index=False)
)
