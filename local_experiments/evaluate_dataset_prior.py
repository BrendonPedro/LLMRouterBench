import json
from pathlib import Path

import pandas as pd


DS = "DeepSeek-R1-0528-Qwen3-8B"
QWEN = "Qwen3-8B"

OUT = Path("local_experiments/results")
rows = []
dataset_rows = []


def load_jsonl(path):
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)

            data.append({
                "dataset": x["dataset"],
                "index": x.get("index"),
                "deepseek_score": float(x["records"][DS]),
                "qwen_score": float(x["records"][QWEN]),
            })

    return pd.DataFrame(data)


comparison = pd.read_csv(
    OUT / "exact_multiseed_router_comparison.csv"
)


for seed in range(42, 52):

    base = Path(
        f"baselines/AvengersPro/data/"
        f"small_models_seed_{seed}/seed{seed}_split0.7"
    )

    train = load_jsonl(base / "train.jsonl")
    test = load_jsonl(base / "test.jsonl")

    # -----------------------------------------------
    # Pick the best model for EACH DATASET using
    # TRAIN DATA ONLY.
    # -----------------------------------------------

    train_means = (
        train.groupby("dataset")
        [["deepseek_score", "qwen_score"]]
        .mean()
    )

    dataset_choice = {}

    for dataset, r in train_means.iterrows():

        if r["qwen_score"] >= r["deepseek_score"]:
            dataset_choice[dataset] = QWEN
        else:
            dataset_choice[dataset] = DS

    test["dataset_prior_choice"] = (
        test["dataset"].map(dataset_choice)
    )

    test["dataset_prior_score"] = test.apply(
        lambda r:
            r["qwen_score"]
            if r["dataset_prior_choice"] == QWEN
            else r["deepseek_score"],
        axis=1,
    )

    # -----------------------------------------------
    # Sample average
    # -----------------------------------------------

    sample_avg = (
        test["dataset_prior_score"].mean() * 100
    )

    # -----------------------------------------------
    # Dataset average
    # -----------------------------------------------

    per_dataset = (
        test.groupby("dataset")
        .agg(
            n_test=("dataset_prior_score", "size"),
            dataset_prior_accuracy=(
                "dataset_prior_score", "mean"
            ),
        )
        .reset_index()
    )

    per_dataset["dataset_prior_accuracy_pct"] = (
        per_dataset["dataset_prior_accuracy"] * 100
    )

    dataset_avg = (
        per_dataset[
            "dataset_prior_accuracy_pct"
        ].mean()
    )

    original = comparison[
        comparison["seed"] == seed
    ].iloc[0]

    rows.append({
        "seed": seed,

        "dataset_prior_sample_avg_pct":
            sample_avg,

        "tfidf_sample_avg_pct":
            original["tfidf_sample_avg_pct"],

        "avengerspro_sample_avg_pct":
            original["avengerspro_sample_avg_pct"],

        "dataset_prior_dataset_avg_pct":
            dataset_avg,

        "tfidf_dataset_avg_pct":
            original["tfidf_dataset_avg_pct"],

        "avengerspro_dataset_avg_pct":
            original["avengerspro_dataset_avg_pct"],

        "tfidf_minus_dataset_prior_sample_pp":
            original["tfidf_sample_avg_pct"]
            - sample_avg,

        "avengerspro_minus_dataset_prior_sample_pp":
            original["avengerspro_sample_avg_pct"]
            - sample_avg,

        "tfidf_minus_dataset_prior_dataset_pp":
            original["tfidf_dataset_avg_pct"]
            - dataset_avg,

        "avengerspro_minus_dataset_prior_dataset_pp":
            original["avengerspro_dataset_avg_pct"]
            - dataset_avg,
    })

    for _, r in per_dataset.iterrows():

        dataset_rows.append({
            "seed": seed,
            "dataset": r["dataset"],
            "dataset_prior_model":
                dataset_choice[r["dataset"]],
            "dataset_prior_accuracy_pct":
                r["dataset_prior_accuracy_pct"],
        })


results = pd.DataFrame(rows)

results.to_csv(
    OUT / "dataset_prior_multiseed_comparison.csv",
    index=False,
)

dataset_results = pd.DataFrame(dataset_rows)

dataset_results.to_csv(
    OUT / "dataset_prior_by_dataset.csv",
    index=False,
)


print()
print("=" * 80)
print("DATASET-PRIOR BASELINE")
print("=" * 80)

print(
    f"Dataset Prior Sample-Avg : "
    f"{results['dataset_prior_sample_avg_pct'].mean():.4f}%"
)

print(
    f"TF-IDF Sample-Avg        : "
    f"{results['tfidf_sample_avg_pct'].mean():.4f}%"
)

print(
    f"AvengersPro Sample-Avg   : "
    f"{results['avengerspro_sample_avg_pct'].mean():.4f}%"
)

print()

print(
    f"TF-IDF - Dataset Prior   : "
    f"{results['tfidf_minus_dataset_prior_sample_pp'].mean():+.4f} pp"
)

print(
    f"AvengersPro - Dataset Prior: "
    f"{results['avengerspro_minus_dataset_prior_sample_pp'].mean():+.4f} pp"
)

print()

print(
    f"Dataset Prior Dataset-Avg: "
    f"{results['dataset_prior_dataset_avg_pct'].mean():.4f}%"
)

print(
    f"TF-IDF Dataset-Avg       : "
    f"{results['tfidf_dataset_avg_pct'].mean():.4f}%"
)

print(
    f"AvengersPro Dataset-Avg  : "
    f"{results['avengerspro_dataset_avg_pct'].mean():.4f}%"
)

print()

print(
    f"TF-IDF - Dataset Prior   : "
    f"{results['tfidf_minus_dataset_prior_dataset_pp'].mean():+.4f} pp"
)

print(
    f"AvengersPro - Dataset Prior: "
    f"{results['avengerspro_minus_dataset_prior_dataset_pp'].mean():+.4f} pp"
)

print("=" * 80)

print("\nSaved:")
print(OUT / "dataset_prior_multiseed_comparison.csv")
print(OUT / "dataset_prior_by_dataset.csv")
