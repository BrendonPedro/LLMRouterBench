import json
from pathlib import Path

import pandas as pd


RESULTS = Path("local_experiments/results")

tfidf = pd.read_csv(
    RESULTS / "tfidf_exact_multiseed_by_dataset.csv"
)

rows = []

for seed in range(42, 52):

    path = Path(
        f"baselines/AvengersPro/results/"
        f"pair_seed{seed}_split0.7/results.json"
    )

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    perf = data["results"]["dataset_performance"]

    for dataset, values in perf.items():
        rows.append({
            "seed": seed,
            "dataset": dataset,
            "avengerspro_accuracy_pct":
                float(values["accuracy"]),
        })

av = pd.DataFrame(rows)

merged = tfidf.merge(
    av,
    on=["seed", "dataset"],
    how="inner",
)

merged["tfidf_gain_vs_fixed_pp"] = (
    merged["tfidf_accuracy_pct"]
    - merged["best_fixed_accuracy_pct"]
)

merged["avengerspro_gain_vs_fixed_pp"] = (
    merged["avengerspro_accuracy_pct"]
    - merged["best_fixed_accuracy_pct"]
)

merged["oracle_headroom_vs_fixed_pp"] = (
    merged["pair_oracle_accuracy_pct"]
    - merged["best_fixed_accuracy_pct"]
)

merged["remaining_gap_after_tfidf_pp"] = (
    merged["pair_oracle_accuracy_pct"]
    - merged["tfidf_accuracy_pct"]
)

merged["remaining_gap_after_avengerspro_pp"] = (
    merged["pair_oracle_accuracy_pct"]
    - merged["avengerspro_accuracy_pct"]
)

merged["avengerspro_minus_tfidf_pp"] = (
    merged["avengerspro_accuracy_pct"]
    - merged["tfidf_accuracy_pct"]
)

summary = (
    merged.groupby("dataset")
    .agg(
        mean_test_queries=("n_test", "mean"),

        best_fixed_pct=(
            "best_fixed_accuracy_pct", "mean"
        ),

        tfidf_pct=(
            "tfidf_accuracy_pct", "mean"
        ),

        avengerspro_pct=(
            "avengerspro_accuracy_pct", "mean"
        ),

        oracle_pct=(
            "pair_oracle_accuracy_pct", "mean"
        ),

        oracle_headroom_pp=(
            "oracle_headroom_vs_fixed_pp", "mean"
        ),

        tfidf_gain_pp=(
            "tfidf_gain_vs_fixed_pp", "mean"
        ),

        avengerspro_gain_pp=(
            "avengerspro_gain_vs_fixed_pp", "mean"
        ),

        remaining_gap_tfidf_pp=(
            "remaining_gap_after_tfidf_pp", "mean"
        ),

        remaining_gap_avengerspro_pp=(
            "remaining_gap_after_avengerspro_pp", "mean"
        ),

        avengerspro_minus_tfidf_pp=(
            "avengerspro_minus_tfidf_pp", "mean"
        ),
    )
    .reset_index()
)

summary["tfidf_headroom_captured_pct"] = (
    summary["tfidf_gain_pp"]
    / summary["oracle_headroom_pp"]
    * 100
)

summary["avengerspro_headroom_captured_pct"] = (
    summary["avengerspro_gain_pp"]
    / summary["oracle_headroom_pp"]
    * 100
)

summary = summary.sort_values(
    "remaining_gap_avengerspro_pp",
    ascending=False,
)

merged.to_csv(
    RESULTS / "exact_multiseed_dataset_gap_details.csv",
    index=False,
)

summary.to_csv(
    RESULTS / "exact_multiseed_dataset_gap_summary.csv",
    index=False,
)

print("\n" + "=" * 100)
print("WHERE IS THE ROUTING GAP?")
print("=" * 100)

cols = [
    "dataset",
    "best_fixed_pct",
    "tfidf_pct",
    "avengerspro_pct",
    "oracle_pct",
    "oracle_headroom_pp",
    "tfidf_gain_pp",
    "avengerspro_gain_pp",
    "remaining_gap_avengerspro_pp",
]

print(
    summary[cols]
    .round(2)
    .to_string(index=False)
)

print("\nSaved:")
print(
    RESULTS /
    "exact_multiseed_dataset_gap_details.csv"
)
print(
    RESULTS /
    "exact_multiseed_dataset_gap_summary.csv"
)
