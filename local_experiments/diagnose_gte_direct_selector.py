from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


OUT = Path("local_experiments/results")

rows = []


for seed in range(42, 52):

    path = OUT / f"gte_direct_selector_seed{seed}_predictions.csv"
    df = pd.read_csv(path)

    # Ties have no unique correct model, so exclude them
    # from winner-classification diagnostics.
    df = df[df["selector_target"] != -1].copy()

    for dataset, g in df.groupby("dataset"):

        y_true = g["selector_target"].astype(int).to_numpy()
        y_pred = g["selector_class"].astype(int).to_numpy()
        y_prob = g["qwen_probability"].to_numpy()

        two_true_classes = len(np.unique(y_true)) == 2

        if two_true_classes:
            macro_f1 = f1_score(
                y_true,
                y_pred,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )

            balanced_accuracy = balanced_accuracy_score(
                y_true,
                y_pred,
            )

            auc = roc_auc_score(
                y_true,
                y_prob,
            )

            mcc = matthews_corrcoef(
                y_true,
                y_pred,
            )
        else:
            macro_f1 = np.nan
            balanced_accuracy = np.nan
            auc = np.nan
            mcc = np.nan

        true_qwen_pct = (y_true == 1).mean() * 100
        pred_qwen_pct = (y_pred == 1).mean() * 100

        rows.append({
            "seed": seed,
            "dataset": dataset,
            "non_tie_n": len(g),
            "two_true_classes": two_true_classes,
            "macro_f1": macro_f1,
            "balanced_accuracy": balanced_accuracy,
            "auc": auc,
            "mcc": mcc,
            "true_qwen_pct": true_qwen_pct,
            "pred_qwen_pct": pred_qwen_pct,
            "qwen_bias_pp": pred_qwen_pct - true_qwen_pct,
        })


per_seed = pd.DataFrame(rows)

per_seed.to_csv(
    OUT / "gte_direct_selector_diagnostics_by_seed.csv",
    index=False,
)


summary = (
    per_seed.groupby("dataset")
    .agg(
        seeds=("seed", "nunique"),
        metric_seeds=("macro_f1", "count"),
        avg_non_tie_n=("non_tie_n", "mean"),

        macro_f1=("macro_f1", "mean"),
        f1_std=("macro_f1", "std"),

        balanced_accuracy=(
            "balanced_accuracy",
            "mean",
        ),

        auc=("auc", "mean"),
        mcc=("mcc", "mean"),

        true_qwen_pct=(
            "true_qwen_pct",
            "mean",
        ),

        pred_qwen_pct=(
            "pred_qwen_pct",
            "mean",
        ),

        qwen_bias_pp=(
            "qwen_bias_pp",
            "mean",
        ),
    )
    .reset_index()
    .sort_values(
        "macro_f1",
        ascending=False,
    )
)


summary.to_csv(
    OUT / "gte_direct_selector_diagnostics_summary.csv",
    index=False,
)


print()
print("=" * 120)
print("GTE DIRECT SELECTOR PER-DATASET DIAGNOSTICS")
print("=" * 120)

print(
    summary.round(3).to_string(index=False)
)

print()
print("Notes:")
print("  metric_seeds = seeds where both winner classes occur in the test subset.")
print("  positive qwen_bias_pp = router chooses Qwen too often.")
print("  negative qwen_bias_pp = router chooses DeepSeek too often.")

print()
print("Saved:")
print(
    OUT /
    "gte_direct_selector_diagnostics_by_seed.csv"
)
print(
    OUT /
    "gte_direct_selector_diagnostics_summary.csv"
)
