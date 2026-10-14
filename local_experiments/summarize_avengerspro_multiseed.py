import json
from pathlib import Path
import pandas as pd

rows = []

for seed in range(42, 52):
    path = Path(
        f"baselines/AvengersPro/results/"
        f"pair_seed{seed}_split0.7/results.json"
    )

    if not path.exists():
        print(f"Missing seed {seed}: {path}")
        continue

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    r = data["results"]
    baseline = r["baseline_analysis"]["best_overall_baseline"]

    rows.append({
        "seed": seed,
        "sample_avg_pct": r["all_sample_avg"],
        "dataset_avg_pct": r["accuracy"],
        "best_fixed_model": baseline["model"],
        "best_fixed_dataset_avg_pct": baseline["avg_score"] * 100,
        "dataset_gain_vs_best_fixed_pp":
            r["accuracy"] - baseline["avg_score"] * 100,
        "test_queries": r["total_queries"],
    })

df = pd.DataFrame(rows)

out = Path(
    "local_experiments/results/"
    "avengerspro_pair_multiseed_results.csv"
)
df.to_csv(out, index=False)

summary = pd.DataFrame([
    {
        "metric": "sample_avg_pct",
        "mean": df["sample_avg_pct"].mean(),
        "std": df["sample_avg_pct"].std(),
        "min": df["sample_avg_pct"].min(),
        "max": df["sample_avg_pct"].max(),
    },
    {
        "metric": "dataset_avg_pct",
        "mean": df["dataset_avg_pct"].mean(),
        "std": df["dataset_avg_pct"].std(),
        "min": df["dataset_avg_pct"].min(),
        "max": df["dataset_avg_pct"].max(),
    },
    {
        "metric": "dataset_gain_vs_best_fixed_pp",
        "mean": df["dataset_gain_vs_best_fixed_pp"].mean(),
        "std": df["dataset_gain_vs_best_fixed_pp"].std(),
        "min": df["dataset_gain_vs_best_fixed_pp"].min(),
        "max": df["dataset_gain_vs_best_fixed_pp"].max(),
    },
])

summary_out = Path(
    "local_experiments/results/"
    "avengerspro_pair_multiseed_summary.csv"
)
summary.to_csv(summary_out, index=False)

print("\nPer-seed results:")
print(df.to_string(index=False))

print("\nSummary:")
print(summary.to_string(index=False))

print("\nSaved:")
print(out)
print(summary_out)
