import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


DS = "DeepSeek-R1-0528-Qwen3-8B"
QWEN = "Qwen3-8B"

OUT = Path("local_experiments/results")

all_dataset_rows = []
all_seed_rows = []


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
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

    seed_predictions = []

    for dataset in sorted(test["dataset"].unique()):

        tr = train[train["dataset"] == dataset].copy()
        te = test[test["dataset"] == dataset].copy()

        ds_train_mean = tr["deepseek_score"].mean()
        qw_train_mean = tr["qwen_score"].mean()

        prior_model = (
            QWEN if qw_train_mean >= ds_train_mean else DS
        )

        te["dataset_prior_score"] = np.where(
            prior_model == QWEN,
            te["qwen_score"],
            te["deepseek_score"],
        )

        tr_nt = tr[
            tr["deepseek_score"] != tr["qwen_score"]
        ].copy()

        if len(tr_nt) == 0:
            te["within_tfidf_choice"] = prior_model
        else:
            tr_nt["label"] = np.where(
                tr_nt["qwen_score"] > tr_nt["deepseek_score"],
                1,
                0,
            )

            if tr_nt["label"].nunique() < 2:
                te["within_tfidf_choice"] = prior_model
            else:
                vec = TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=20000,
                )

                try:
                    X_train = vec.fit_transform(tr_nt["query"])
                    X_test = vec.transform(te["query"])

                    clf = LogisticRegression(
                        max_iter=2000,
                        random_state=42,
                    )

                    clf.fit(X_train, tr_nt["label"])

                    pred = clf.predict(X_test)

                    te["within_tfidf_choice"] = np.where(
                        pred == 1,
                        QWEN,
                        DS,
                    )

                except ValueError:
                    te["within_tfidf_choice"] = prior_model

        te["within_tfidf_score"] = np.where(
            te["within_tfidf_choice"] == QWEN,
            te["qwen_score"],
            te["deepseek_score"],
        )

        te["oracle_score"] = te[
            ["deepseek_score", "qwen_score"]
        ].max(axis=1)

        prior_acc = te["dataset_prior_score"].mean() * 100
        tfidf_acc = te["within_tfidf_score"].mean() * 100
        oracle_acc = te["oracle_score"].mean() * 100

        all_dataset_rows.append({
            "seed": seed,
            "dataset": dataset,
            "train_queries": len(tr),
            "test_queries": len(te),
            "non_tied_train_queries": len(tr_nt),
            "dataset_prior_model": prior_model,
            "dataset_prior_accuracy_pct": prior_acc,
            "within_dataset_tfidf_accuracy_pct": tfidf_acc,
            "oracle_accuracy_pct": oracle_acc,
            "tfidf_minus_prior_pp": tfidf_acc - prior_acc,
            "oracle_headroom_over_prior_pp":
                oracle_acc - prior_acc,
            "headroom_captured_pct":
                (
                    (tfidf_acc - prior_acc)
                    / (oracle_acc - prior_acc)
                    * 100
                    if oracle_acc > prior_acc
                    else np.nan
                ),
        })

        te["seed"] = seed
        seed_predictions.append(te)

    seed_df = pd.concat(
        seed_predictions,
        ignore_index=True,
    )

    prior_sample = (
        seed_df["dataset_prior_score"].mean() * 100
    )

    tfidf_sample = (
        seed_df["within_tfidf_score"].mean() * 100
    )

    oracle_sample = (
        seed_df["oracle_score"].mean() * 100
    )

    per_dataset = pd.DataFrame(
        [
            r for r in all_dataset_rows
            if r["seed"] == seed
        ]
    )

    prior_dataset = (
        per_dataset["dataset_prior_accuracy_pct"].mean()
    )

    tfidf_dataset = (
        per_dataset["within_dataset_tfidf_accuracy_pct"].mean()
    )

    oracle_dataset = (
        per_dataset["oracle_accuracy_pct"].mean()
    )

    all_seed_rows.append({
        "seed": seed,
        "dataset_prior_sample_avg_pct": prior_sample,
        "within_dataset_tfidf_sample_avg_pct": tfidf_sample,
        "oracle_sample_avg_pct": oracle_sample,
        "tfidf_minus_prior_sample_pp":
            tfidf_sample - prior_sample,
        "dataset_prior_dataset_avg_pct": prior_dataset,
        "within_dataset_tfidf_dataset_avg_pct":
            tfidf_dataset,
        "oracle_dataset_avg_pct": oracle_dataset,
        "tfidf_minus_prior_dataset_pp":
            tfidf_dataset - prior_dataset,
    })

    print(
        f"Sample: Prior={prior_sample:.4f} "
        f"Within-TFIDF={tfidf_sample:.4f} "
        f"Oracle={oracle_sample:.4f}"
    )


dataset_results = pd.DataFrame(all_dataset_rows)

dataset_results.to_csv(
    OUT / "within_dataset_tfidf_by_dataset_seed.csv",
    index=False,
)

seed_results = pd.DataFrame(all_seed_rows)

seed_results.to_csv(
    OUT / "within_dataset_tfidf_multiseed.csv",
    index=False,
)


summary = (
    dataset_results.groupby("dataset")
    .agg(
        dataset_prior_pct=(
            "dataset_prior_accuracy_pct", "mean"
        ),
        within_tfidf_pct=(
            "within_dataset_tfidf_accuracy_pct", "mean"
        ),
        oracle_pct=(
            "oracle_accuracy_pct", "mean"
        ),
        tfidf_minus_prior_pp=(
            "tfidf_minus_prior_pp", "mean"
        ),
        oracle_headroom_pp=(
            "oracle_headroom_over_prior_pp", "mean"
        ),
    )
    .reset_index()
)

summary["remaining_oracle_gap_pp"] = (
    summary["oracle_pct"]
    - summary["within_tfidf_pct"]
)

summary = summary.sort_values(
    "tfidf_minus_prior_pp",
    ascending=False,
)

summary.to_csv(
    OUT / "within_dataset_tfidf_summary.csv",
    index=False,
)


print()
print("=" * 95)
print("WITHIN-DATASET QUERY ROUTING")
print("=" * 95)

print(
    f"Dataset Prior Sample-Avg    : "
    f"{seed_results['dataset_prior_sample_avg_pct'].mean():.4f}%"
)

print(
    f"Within-Dataset TF-IDF      : "
    f"{seed_results['within_dataset_tfidf_sample_avg_pct'].mean():.4f}%"
)

print(
    f"Difference                 : "
    f"{seed_results['tfidf_minus_prior_sample_pp'].mean():+.4f} pp"
)

print()

print(
    f"Dataset Prior Dataset-Avg  : "
    f"{seed_results['dataset_prior_dataset_avg_pct'].mean():.4f}%"
)

print(
    f"Within-Dataset TF-IDF      : "
    f"{seed_results['within_dataset_tfidf_dataset_avg_pct'].mean():.4f}%"
)

print(
    f"Difference                 : "
    f"{seed_results['tfidf_minus_prior_dataset_pp'].mean():+.4f} pp"
)

print()
print("PER-DATASET MEANS")
print()

print(
    summary[
        [
            "dataset",
            "dataset_prior_pct",
            "within_tfidf_pct",
            "tfidf_minus_prior_pp",
            "oracle_pct",
            "remaining_oracle_gap_pp",
        ]
    ]
    .round(2)
    .to_string(index=False)
)

print()
print("Saved:")
print(OUT / "within_dataset_tfidf_multiseed.csv")
print(OUT / "within_dataset_tfidf_by_dataset_seed.csv")
print(OUT / "within_dataset_tfidf_summary.csv")
