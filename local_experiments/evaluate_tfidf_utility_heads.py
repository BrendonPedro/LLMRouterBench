import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


DS = "DeepSeek-R1-0528-Qwen3-8B"
QWEN = "Qwen3-8B"

OUT = Path("local_experiments/results")

seed_rows = []
dataset_rows = []


def load_jsonl(path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)

            rows.append({
                "dataset": x["dataset"],
                "index": x.get("index"),
                "query": x["query"],
                "deepseek_score": float(x["records"][DS]),
                "qwen_score": float(x["records"][QWEN]),
            })

    return pd.DataFrame(rows)


# Existing exact-split results for comparison.
existing = pd.read_csv(
    OUT / "exact_multiseed_router_comparison.csv"
)

dataset_prior = pd.read_csv(
    OUT / "dataset_prior_multiseed_comparison.csv"
)


for seed in range(42, 52):

    print()
    print("=" * 60)
    print(f"SEED {seed}")
    print("=" * 60)

    base = Path(
        f"baselines/AvengersPro/data/"
        f"small_models_seed_{seed}/seed{seed}_split0.7"
    )

    train = load_jsonl(base / "train.jsonl")
    test = load_jsonl(base / "test.jsonl")

    # --------------------------------------------------
    # ONE SHARED TF-IDF REPRESENTATION
    # --------------------------------------------------

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=20000,
    )

    X_train = vectorizer.fit_transform(train["query"])
    X_test = vectorizer.transform(test["query"])

    # --------------------------------------------------
    # INDEPENDENT MODEL-UTILITY HEADS
    # P(model succeeds | query)
    # --------------------------------------------------

    ds_head = LogisticRegression(
        max_iter=2000,
        random_state=42,
    )

    qwen_head = LogisticRegression(
        max_iter=2000,
        random_state=42,
    )

    ds_head.fit(
        X_train,
        train["deepseek_score"].astype(int),
    )

    qwen_head.fit(
        X_train,
        train["qwen_score"].astype(int),
    )

    test["deepseek_utility"] = (
        ds_head.predict_proba(X_test)[:, 1]
    )

    test["qwen_utility"] = (
        qwen_head.predict_proba(X_test)[:, 1]
    )

    test["utility_choice"] = np.where(
        test["qwen_utility"]
        >= test["deepseek_utility"],
        QWEN,
        DS,
    )

    test["utility_score"] = np.where(
        test["utility_choice"] == QWEN,
        test["qwen_score"],
        test["deepseek_score"],
    )

    test["oracle_score"] = test[
        ["deepseek_score", "qwen_score"]
    ].max(axis=1)

    # --------------------------------------------------
    # SAMPLE-AVERAGE
    # --------------------------------------------------

    utility_sample = (
        test["utility_score"].mean() * 100
    )

    oracle_sample = (
        test["oracle_score"].mean() * 100
    )

    # --------------------------------------------------
    # DATASET-AVERAGE
    # --------------------------------------------------

    by_dataset = (
        test.groupby("dataset")
        .agg(
            n_test=("query", "size"),
            utility_accuracy=("utility_score", "mean"),
            oracle_accuracy=("oracle_score", "mean"),
            deepseek_selection=(
                "utility_choice",
                lambda s: (s == DS).mean()
            ),
            qwen_selection=(
                "utility_choice",
                lambda s: (s == QWEN).mean()
            ),
        )
        .reset_index()
    )

    by_dataset["utility_accuracy_pct"] = (
        by_dataset["utility_accuracy"] * 100
    )

    by_dataset["oracle_accuracy_pct"] = (
        by_dataset["oracle_accuracy"] * 100
    )

    by_dataset["deepseek_selection_pct"] = (
        by_dataset["deepseek_selection"] * 100
    )

    by_dataset["qwen_selection_pct"] = (
        by_dataset["qwen_selection"] * 100
    )

    utility_dataset = (
        by_dataset["utility_accuracy_pct"].mean()
    )

    oracle_dataset = (
        by_dataset["oracle_accuracy_pct"].mean()
    )

    old = existing[
        existing["seed"] == seed
    ].iloc[0]

    prior = dataset_prior[
        dataset_prior["seed"] == seed
    ].iloc[0]

    seed_rows.append({
        "seed": seed,

        "dataset_prior_sample_avg_pct":
            prior["dataset_prior_sample_avg_pct"],

        "winner_tfidf_sample_avg_pct":
            old["tfidf_sample_avg_pct"],

        "utility_heads_sample_avg_pct":
            utility_sample,

        "avengerspro_sample_avg_pct":
            old["avengerspro_sample_avg_pct"],

        "oracle_sample_avg_pct":
            oracle_sample,

        "utility_minus_winner_tfidf_sample_pp":
            utility_sample
            - old["tfidf_sample_avg_pct"],

        "utility_minus_dataset_prior_sample_pp":
            utility_sample
            - prior["dataset_prior_sample_avg_pct"],

        "utility_minus_avengerspro_sample_pp":
            utility_sample
            - old["avengerspro_sample_avg_pct"],

        "dataset_prior_dataset_avg_pct":
            prior["dataset_prior_dataset_avg_pct"],

        "winner_tfidf_dataset_avg_pct":
            old["tfidf_dataset_avg_pct"],

        "utility_heads_dataset_avg_pct":
            utility_dataset,

        "avengerspro_dataset_avg_pct":
            old["avengerspro_dataset_avg_pct"],

        "oracle_dataset_avg_pct":
            oracle_dataset,

        "utility_minus_winner_tfidf_dataset_pp":
            utility_dataset
            - old["tfidf_dataset_avg_pct"],

        "utility_minus_dataset_prior_dataset_pp":
            utility_dataset
            - prior["dataset_prior_dataset_avg_pct"],

        "utility_minus_avengerspro_dataset_pp":
            utility_dataset
            - old["avengerspro_dataset_avg_pct"],
    })

    for _, row in by_dataset.iterrows():

        dataset_rows.append({
            "seed": seed,
            "dataset": row["dataset"],
            "n_test": row["n_test"],

            "utility_accuracy_pct":
                row["utility_accuracy_pct"],

            "oracle_accuracy_pct":
                row["oracle_accuracy_pct"],

            "deepseek_selection_pct":
                row["deepseek_selection_pct"],

            "qwen_selection_pct":
                row["qwen_selection_pct"],
        })

    print(
        f"Sample-Avg: "
        f"Utility={utility_sample:.4f} "
        f"Winner-TFIDF={old['tfidf_sample_avg_pct']:.4f} "
        f"Prior={prior['dataset_prior_sample_avg_pct']:.4f} "
        f"AvengersPro={old['avengerspro_sample_avg_pct']:.4f} "
        f"Oracle={oracle_sample:.4f}"
    )


# ======================================================
# SAVE MULTI-SEED RESULTS
# ======================================================

results = pd.DataFrame(seed_rows)

results.to_csv(
    OUT / "tfidf_utility_heads_multiseed.csv",
    index=False,
)

dataset_results = pd.DataFrame(dataset_rows)

dataset_results.to_csv(
    OUT / "tfidf_utility_heads_by_dataset_seed.csv",
    index=False,
)


dataset_summary = (
    dataset_results.groupby("dataset")
    .agg(
        utility_heads_pct=(
            "utility_accuracy_pct", "mean"
        ),
        oracle_pct=(
            "oracle_accuracy_pct", "mean"
        ),
        deepseek_selection_pct=(
            "deepseek_selection_pct", "mean"
        ),
        qwen_selection_pct=(
            "qwen_selection_pct", "mean"
        ),
    )
    .reset_index()
)

dataset_summary["remaining_oracle_gap_pp"] = (
    dataset_summary["oracle_pct"]
    - dataset_summary["utility_heads_pct"]
)

dataset_summary = dataset_summary.sort_values(
    "remaining_oracle_gap_pp",
    ascending=False,
)

dataset_summary.to_csv(
    OUT / "tfidf_utility_heads_dataset_summary.csv",
    index=False,
)


print()
print("=" * 90)
print("TF-IDF MODEL-UTILITY HEADS")
print("=" * 90)

print(
    f"Dataset Prior Sample-Avg  : "
    f"{results['dataset_prior_sample_avg_pct'].mean():.4f}%"
)

print(
    f"Winner TF-IDF Sample-Avg  : "
    f"{results['winner_tfidf_sample_avg_pct'].mean():.4f}%"
)

print(
    f"Utility Heads Sample-Avg  : "
    f"{results['utility_heads_sample_avg_pct'].mean():.4f}%"
)

print(
    f"AvengersPro Sample-Avg    : "
    f"{results['avengerspro_sample_avg_pct'].mean():.4f}%"
)

print(
    f"Oracle Sample-Avg         : "
    f"{results['oracle_sample_avg_pct'].mean():.4f}%"
)

print()

print(
    f"Utility - Winner TF-IDF   : "
    f"{results['utility_minus_winner_tfidf_sample_pp'].mean():+.4f} pp"
)

print(
    f"Utility - Dataset Prior   : "
    f"{results['utility_minus_dataset_prior_sample_pp'].mean():+.4f} pp"
)

print(
    f"Utility - AvengersPro     : "
    f"{results['utility_minus_avengerspro_sample_pp'].mean():+.4f} pp"
)

print()

print(
    f"Dataset Prior Dataset-Avg : "
    f"{results['dataset_prior_dataset_avg_pct'].mean():.4f}%"
)

print(
    f"Winner TF-IDF Dataset-Avg : "
    f"{results['winner_tfidf_dataset_avg_pct'].mean():.4f}%"
)

print(
    f"Utility Heads Dataset-Avg : "
    f"{results['utility_heads_dataset_avg_pct'].mean():.4f}%"
)

print(
    f"AvengersPro Dataset-Avg   : "
    f"{results['avengerspro_dataset_avg_pct'].mean():.4f}%"
)

print(
    f"Oracle Dataset-Avg        : "
    f"{results['oracle_dataset_avg_pct'].mean():.4f}%"
)

print()

print(
    f"Utility - Winner TF-IDF   : "
    f"{results['utility_minus_winner_tfidf_dataset_pp'].mean():+.4f} pp"
)

print(
    f"Utility - Dataset Prior   : "
    f"{results['utility_minus_dataset_prior_dataset_pp'].mean():+.4f} pp"
)

print(
    f"Utility - AvengersPro     : "
    f"{results['utility_minus_avengerspro_dataset_pp'].mean():+.4f} pp"
)

print()
print("PER-DATASET UTILITY-HEAD RESULTS")
print()

print(
    dataset_summary.round(2).to_string(index=False)
)

print()
print("Saved:")
print(OUT / "tfidf_utility_heads_multiseed.csv")
print(OUT / "tfidf_utility_heads_by_dataset_seed.csv")
print(OUT / "tfidf_utility_heads_dataset_summary.csv")
