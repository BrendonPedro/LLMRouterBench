import json
from pathlib import Path

import pandas as pd


INPUT = Path(
    "baselines/AvengersPro/results/"
    "seed42_split0.7/results.json"
)

OUTPUT_DIR = Path("local_experiments/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


with INPUT.open("r", encoding="utf-8") as f:
    data = json.load(f)

config = data["config"]
results = data["results"]

dataset_perf = results["dataset_performance"]
selection_stats = results["model_selection_stats"]
cost = results["cost_analysis"]
baseline = results["baseline_analysis"]


# ============================================================
# 1. OVERALL METRICS
# ============================================================

best_baseline = baseline["best_overall_baseline"]

overall = pd.DataFrame([
    {
        "metric": "AvengersPro Dataset-Avg",
        "value": results["accuracy"],
        "unit": "percent",
        "meaning": (
            "Average router accuracy across datasets, with each "
            "dataset weighted equally."
        ),
    },
    {
        "metric": "AvengersPro Sample-Avg",
        "value": results["all_sample_avg"],
        "unit": "percent",
        "meaning": (
            "Average router accuracy across all individual test "
            "queries, so larger datasets contribute more."
        ),
    },
    {
        "metric": "Correct routed predictions",
        "value": results["correct_routes"],
        "unit": "count",
        "meaning": "Total successful routed predictions on the test set.",
    },
    {
        "metric": "Total test queries",
        "value": results["total_queries"],
        "unit": "count",
        "meaning": "Total number of query-level test examples.",
    },
    {
        "metric": "Best overall fixed model",
        "value": best_baseline["model"],
        "unit": "model",
        "meaning": (
            "Single fixed model with the highest average benchmark "
            "performance."
        ),
    },
    {
        "metric": "Best overall fixed model score",
        "value": best_baseline["avg_score"] * 100,
        "unit": "percent",
        "meaning": (
            "Average score of the strongest single fixed model "
            "across datasets."
        ),
    },
    {
        "metric": "Router gain vs best overall fixed model",
        "value": (
            results["accuracy"]
            - best_baseline["avg_score"] * 100
        ),
        "unit": "percentage points",
        "meaning": (
            "Dataset-Avg improvement of AvengersPro over the "
            "strongest single fixed model."
        ),
    },
    {
        "metric": "Total recorded routing cost",
        "value": cost["total_cost"],
        "unit": "USD",
        "meaning": (
            "Recorded inference cost of models selected by the router. "
            "Interpret cautiously because some candidate models have "
            "missing or zero recorded costs."
        ),
    },
    {
        "metric": "Average recorded cost per query",
        "value": cost["avg_cost_per_query"],
        "unit": "USD/query",
        "meaning": "Mean recorded selected-model cost per routed query.",
    },
])

overall.to_csv(
    OUTPUT_DIR / "avengerspro_overall_metrics.csv",
    index=False,
)


# ============================================================
# 2. DATASET-LEVEL RESULTS
# ============================================================

comparisons = {
    row["dataset"]: row
    for row in baseline["dataset_comparisons"]
}

dataset_costs = cost.get("dataset_costs", {})

dataset_rows = []

for dataset, perf in dataset_perf.items():

    comparison = comparisons.get(dataset, {})

    dataset_rows.append({
        "dataset": dataset,
        "correct": perf.get("correct"),
        "n_test": perf.get("total"),
        "router_accuracy_pct": perf.get("accuracy"),
        "best_baseline_model":
            comparison.get("best_baseline_model"),
        "best_baseline_score_pct":
            comparison.get("best_baseline_score", 0) * 100,
        "router_improvement_vs_dataset_best_pp":
            comparison.get("improvement", 0) * 100,
        "recorded_router_cost_usd":
            dataset_costs.get(dataset),
    })

dataset_df = pd.DataFrame(dataset_rows)

dataset_df = dataset_df.sort_values(
    "router_improvement_vs_dataset_best_pp",
    ascending=False,
)

dataset_df.to_csv(
    OUTPUT_DIR / "avengerspro_by_dataset.csv",
    index=False,
)


# ============================================================
# 3. MODEL SELECTION FREQUENCY
# ============================================================

total_queries = results["total_queries"]

selection_rows = []

for model, count in selection_stats.items():
    selection_rows.append({
        "model": model,
        "selection_count": count,
        "selection_rate_pct": count / total_queries * 100,
    })

selection_df = pd.DataFrame(selection_rows)

selection_df = selection_df.sort_values(
    "selection_count",
    ascending=False,
)

selection_df.to_csv(
    OUTPUT_DIR / "avengerspro_model_selection.csv",
    index=False,
)


# ============================================================
# 4. BASELINE MODEL SUMMARY
# ============================================================

baseline_models = pd.DataFrame(
    baseline["model_summaries"]
)

baseline_models["avg_score_pct"] = (
    baseline_models["avg_score"] * 100
)

baseline_models = baseline_models[
    [
        "model",
        "avg_score_pct",
        "total_cost",
        "dataset_coverage",
        "valid_datasets",
    ]
]

baseline_models = baseline_models.sort_values(
    "avg_score_pct",
    ascending=False,
)

baseline_models.to_csv(
    OUTPUT_DIR / "avengerspro_baseline_models.csv",
    index=False,
)


# ============================================================
# 5. COST SUMMARY
# ============================================================

cost_summary = pd.DataFrame([
    ["total_cost_usd", cost.get("total_cost")],
    ["avg_cost_per_query_usd", cost.get("avg_cost_per_query")],
    [
        "cost_per_correct_prediction_usd",
        cost.get("cost_per_correct_prediction")
    ],
    ["cost_efficiency", cost.get("cost_efficiency")],
    ["ood_total_cost_usd", cost.get("ood_total_cost")],
    ["non_ood_total_cost_usd", cost.get("non_ood_total_cost")],
], columns=["metric", "value"])

cost_summary.to_csv(
    OUTPUT_DIR / "avengerspro_cost_summary.csv",
    index=False,
)


# ============================================================
# 6. MODEL COSTS
# ============================================================

model_costs = pd.DataFrame([
    {
        "model": model,
        "recorded_cost_usd": value,
    }
    for model, value in cost.get("model_costs", {}).items()
])

if not model_costs.empty:
    model_costs = model_costs.sort_values(
        "recorded_cost_usd",
        ascending=False,
    )

model_costs.to_csv(
    OUTPUT_DIR / "avengerspro_model_costs.csv",
    index=False,
)


# ============================================================
# 7. DATASET COSTS
# ============================================================

dataset_cost_df = pd.DataFrame([
    {
        "dataset": dataset,
        "recorded_cost_usd": value,
    }
    for dataset, value in cost.get("dataset_costs", {}).items()
])

if not dataset_cost_df.empty:
    dataset_cost_df = dataset_cost_df.sort_values(
        "recorded_cost_usd",
        ascending=False,
    )

dataset_cost_df.to_csv(
    OUTPUT_DIR / "avengerspro_dataset_costs.csv",
    index=False,
)


# ============================================================
# 8. CONFIGURATION / REPRODUCIBILITY
# ============================================================

config_rows = [
    {
        "field": "timestamp",
        "value": data.get("timestamp"),
    }
]

for key, value in config.items():

    if isinstance(value, (list, dict)):
        value = json.dumps(value)

    config_rows.append({
        "field": key,
        "value": value,
    })

config_df = pd.DataFrame(config_rows)

config_df.to_csv(
    OUTPUT_DIR / "avengerspro_config.csv",
    index=False,
)


# ============================================================
# 9. DATASET COMPARISONS AS PROVIDED BY AVENGERSPRO
# ============================================================

comparison_df = pd.DataFrame(
    baseline["dataset_comparisons"]
)

for col in [
    "router_accuracy",
    "best_baseline_score",
    "improvement",
]:
    if col in comparison_df.columns:
        comparison_df[col + "_pct"] = (
            comparison_df[col] * 100
        )

comparison_df.to_csv(
    OUTPUT_DIR / "avengerspro_dataset_comparisons_raw.csv",
    index=False,
)


# ============================================================
# DONE
# ============================================================

print("AvengersPro structured results exported.\n")

for path in sorted(
    OUTPUT_DIR.glob("avengerspro_*.csv")
):
    print(" -", path)
