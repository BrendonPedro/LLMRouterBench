from itertools import combinations

import pandas as pd

from baselines import BaselineDataLoader


print("Loading benchmark records...")

loader = BaselineDataLoader("config/baseline_config.yaml")
records = loader.load_all_records()

rows = []

for r in records:
    try:
        score = float(r.score)
    except (TypeError, ValueError):
        continue

    rows.append(
        {
            "dataset": r.dataset_id,
            "split": r.split,
            "index": r.record_index,
            "model": r.model_name,
            "score": score,
        }
    )

df = pd.DataFrame(rows)

pivot = df.pivot_table(
    index=["dataset", "split", "index"],
    columns="model",
    values="score",
    aggfunc="first",
)

print("\nUnique query-level examples:", len(pivot))
print("Models:", len(pivot.columns))

print("\nUnique queries per dataset:")
query_counts = (
    df[["dataset", "split", "index"]]
    .drop_duplicates()
    .groupby("dataset")
    .size()
    .sort_index()
)
print(query_counts.to_string())

results = []

for m1, m2 in combinations(pivot.columns, 2):
    pair = pivot[[m1, m2]].dropna()

    if pair.empty:
        continue

    s1 = pair[m1]
    s2 = pair[m2]

    avg1 = s1.mean()
    avg2 = s2.mean()

    best_single = max(avg1, avg2)
    pair_oracle = pd.concat([s1, s2], axis=1).max(axis=1).mean()
    headroom = pair_oracle - best_single

    results.append(
        {
            "model_1": m1,
            "model_2": m2,
            "model_1_score": avg1,
            "model_2_score": avg2,
            "best_single": best_single,
            "pair_oracle": pair_oracle,
            "headroom": headroom,
            "m1_better_pct": (s1 > s2).mean(),
            "m2_better_pct": (s2 > s1).mean(),
            "tie_pct": (s1 == s2).mean(),
            "n": len(pair),
        }
    )

result_df = pd.DataFrame(results).sort_values(
    ["headroom", "pair_oracle"],
    ascending=False,
)

pct_cols = [
    "model_1_score",
    "model_2_score",
    "best_single",
    "pair_oracle",
    "headroom",
    "m1_better_pct",
    "m2_better_pct",
    "tie_pct",
]

result_df[pct_cols] = result_df[pct_cols] * 100

print("\nTop 15 model pairs by routing headroom:")
print(
    result_df.head(15).to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)

print("\nQwen3-8B pairs:")
qwen_pairs = result_df[
    (result_df["model_1"] == "Qwen3-8B")
    | (result_df["model_2"] == "Qwen3-8B")
]

print(
    qwen_pairs.head(10).to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)

# Save full pair analysis for reproducibility
from pathlib import Path

output_dir = Path("local_experiments/results")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "pair_headroom.csv"
result_df.to_csv(output_file, index=False)

print(f"\nFull pair-headroom results saved to: {output_file}")
