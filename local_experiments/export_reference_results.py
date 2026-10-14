from pathlib import Path
from collections import defaultdict
import subprocess
import sys

import pandas as pd

from baselines import BaselineDataLoader, BaselineAggregator


OUTPUT_DIR = Path("local_experiments/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading benchmark records...")

loader = BaselineDataLoader("config/baseline_config.yaml")
records = loader.load_all_records()

print(f"Loaded {len(records)} records.")

agg = BaselineAggregator(records, data_loader=loader)


# --------------------------------------------------
# 1. Benchmark performance and cost tables
# --------------------------------------------------

performance_df, cost_df = agg.to_summary_table(
    cost_metric="total_cost",
    test_mode=False,
    random_seed=42,
)

# Keep the stored performance values numeric, but express them as percentages
performance_percent = performance_df.copy()

for col in performance_percent.columns:
    numeric = pd.to_numeric(performance_percent[col], errors="coerce")
    performance_percent[col] = numeric * 100

performance_percent.index.name = "model"
cost_df.index.name = "model"

performance_percent.to_csv(
    OUTPUT_DIR / "benchmark_performance_percent.csv"
)

cost_df.to_csv(
    OUTPUT_DIR / "benchmark_cost.csv"
)


# --------------------------------------------------
# 2. Unique query counts by dataset
# --------------------------------------------------

queries_by_dataset = defaultdict(set)

for r in records:
    queries_by_dataset[(r.dataset_id, r.split)].add(r.record_index)

dataset_rows = []

for (dataset, split), indices in sorted(queries_by_dataset.items()):
    dataset_rows.append({
        "dataset": dataset,
        "split": split,
        "unique_queries": len(indices),
    })

dataset_counts_df = pd.DataFrame(dataset_rows)

dataset_counts_df.to_csv(
    OUTPUT_DIR / "dataset_query_counts.csv",
    index=False,
)


# --------------------------------------------------
# 3. Model list
# --------------------------------------------------

models = sorted(set(r.model_name for r in records))

models_df = pd.DataFrame({
    "model": models
})

models_df.to_csv(
    OUTPUT_DIR / "models.csv",
    index=False,
)


# --------------------------------------------------
# 4. Dataset list
# --------------------------------------------------

datasets = sorted(set(r.dataset_id for r in records))

datasets_df = pd.DataFrame({
    "dataset": datasets
})

datasets_df.to_csv(
    OUTPUT_DIR / "datasets.csv",
    index=False,
)


# --------------------------------------------------
# 5. Reproducibility / run metadata
# --------------------------------------------------

unique_queries = {
    (r.dataset_id, r.split, r.record_index)
    for r in records
}

unique_combinations = {
    (r.dataset_id, r.split, r.model_name)
    for r in records
}

try:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True
    ).strip()
except Exception:
    git_commit = "unknown"

raw_json_files = len(
    list(Path("results/bench").rglob("*.json"))
)

metadata = [
    ["git_commit", git_commit],
    ["python_version", sys.version.split()[0]],
    ["pandas_version", pd.__version__],
    ["config", "config/baseline_config.yaml"],
    ["random_seed", 42],
    ["raw_json_files", raw_json_files],
    ["loaded_records", len(records)],
    ["unique_dataset_split_model_combinations", len(unique_combinations)],
    ["unique_queries", len(unique_queries)],
    ["datasets", len(datasets)],
    ["models", len(models)],
]

metadata_df = pd.DataFrame(
    metadata,
    columns=["field", "value"]
)

metadata_df.to_csv(
    OUTPUT_DIR / "run_metadata.csv",
    index=False,
)


print("\nSaved:")
for path in sorted(OUTPUT_DIR.glob("*.csv")):
    print(" -", path)
