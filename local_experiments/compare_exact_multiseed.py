import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


MODEL_DS = "DeepSeek-R1-0528-Qwen3-8B"
MODEL_QWEN = "Qwen3-8B"

OUT = Path("local_experiments/results")
OUT.mkdir(parents=True, exist_ok=True)


def load_jsonl(path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            records = item["records"]

            rows.append({
                "query": item["query"],
                "dataset": item["dataset"],
                "index": item.get("index"),
                "deepseek_score": float(records.get(MODEL_DS, 0.0)),
                "qwen_score": float(records.get(MODEL_QWEN, 0.0)),
            })

    return pd.DataFrame(rows)


all_rows = []
all_dataset_rows = []


for seed in range(42, 52):

    print("\n" + "=" * 60)
    print(f"SEED {seed}")
    print("=" * 60)

    base = Path(
        f"baselines/AvengersPro/data/"
        f"small_models_seed_{seed}/"
        f"seed{seed}_split0.7"
    )

    train_path = base / "train.jsonl"
    test_path = base / "test.jsonl"

    avengers_path = Path(
        f"baselines/AvengersPro/results/"
        f"pair_seed{seed}_split0.7/results.json"
    )

    if not train_path.exists():
        raise FileNotFoundError(train_path)

    if not test_path.exists():
        raise FileNotFoundError(test_path)

    if not avengers_path.exists():
        raise FileNotFoundError(avengers_path)

    train = load_jsonl(train_path)
    test = load_jsonl(test_path)

    print("Train queries:", len(train))
    print("Test queries :", len(test))

    # Train only on queries where one model strictly wins.
    train_non_tie = train[
        train["deepseek_score"] != train["qwen_score"]
    ].copy()

    train_non_tie["label"] = np.where(
        train_non_tie["qwen_score"] > train_non_tie["deepseek_score"],
        1,
        0,
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=20000,
    )

    X_train = vectorizer.fit_transform(train_non_tie["query"])
    X_test = vectorizer.transform(test["query"])

    clf = LogisticRegression(
        max_iter=2000,
        random_state=42,
    )

    clf.fit(X_train, train_non_tie["label"])

    pred = clf.predict(X_test)

    test["router_choice"] = np.where(
        pred == 1,
        MODEL_QWEN,
        MODEL_DS,
    )

    test["tfidf_score"] = np.where(
        pred == 1,
        test["qwen_score"],
        test["deepseek_score"],
    )

    # Best fixed model chosen using TRAIN only.
    train_ds_mean = train["deepseek_score"].mean()
    train_qwen_mean = train["qwen_score"].mean()

    best_fixed_model = (
        MODEL_QWEN
        if train_qwen_mean >= train_ds_mean
        else MODEL_DS
    )

    if best_fixed_model == MODEL_QWEN:
        test["best_fixed_score"] = test["qwen_score"]
    else:
        test["best_fixed_score"] = test["deepseek_score"]

    # Oracle chooses the better of the two models for each test query.
    test["pair_oracle_score"] = test[
        ["deepseek_score", "qwen_score"]
    ].max(axis=1)

    test["is_tie"] = (
        test["deepseek_score"] == test["qwen_score"]
    )

    test["true_winner"] = np.where(
        test["qwen_score"] > test["deepseek_score"],
        MODEL_QWEN,
        np.where(
            test["deepseek_score"] > test["qwen_score"],
            MODEL_DS,
            "tie",
        ),
    )

    test["winner_correct"] = (
        test["router_choice"] == test["true_winner"]
    )

    # Sample-average metrics.
    tfidf_sample = test["tfidf_score"].mean() * 100
    fixed_sample = test["best_fixed_score"].mean() * 100
    oracle_sample = test["pair_oracle_score"].mean() * 100

    non_tie_test = test[~test["is_tie"]]

    winner_accuracy = (
        non_tie_test["winner_correct"].mean() * 100
    )

    # Dataset-average metrics.
    by_dataset = (
        test.groupby("dataset")
        .agg(
            n_test=("query", "size"),
            tfidf_accuracy=("tfidf_score", "mean"),
            best_fixed_accuracy=("best_fixed_score", "mean"),
            oracle_accuracy=("pair_oracle_score", "mean"),
        )
        .reset_index()
    )

    by_dataset["tfidf_accuracy_pct"] = (
        by_dataset["tfidf_accuracy"] * 100
    )

    by_dataset["best_fixed_accuracy_pct"] = (
        by_dataset["best_fixed_accuracy"] * 100
    )

    by_dataset["oracle_accuracy_pct"] = (
        by_dataset["oracle_accuracy"] * 100
    )

    tfidf_dataset = by_dataset["tfidf_accuracy_pct"].mean()
    fixed_dataset = by_dataset["best_fixed_accuracy_pct"].mean()
    oracle_dataset = by_dataset["oracle_accuracy_pct"].mean()

    # Load AvengersPro result from exact same split.
    with avengers_path.open("r", encoding="utf-8") as f:
        av_data = json.load(f)

    av = av_data["results"]

    av_sample = float(av["all_sample_avg"])
    av_dataset = float(av["accuracy"])

    for _, row in by_dataset.iterrows():
        all_dataset_rows.append({
            "seed": seed,
            "dataset": row["dataset"],
            "n_test": row["n_test"],
            "tfidf_accuracy_pct": row["tfidf_accuracy_pct"],
            "best_fixed_accuracy_pct": row["best_fixed_accuracy_pct"],
            "pair_oracle_accuracy_pct": row["oracle_accuracy_pct"],
        })

    all_rows.append({
        "seed": seed,
        "train_queries": len(train),
        "test_queries": len(test),
        "non_tied_train_queries": len(train_non_tie),
        "best_fixed_model": best_fixed_model,

        "best_fixed_sample_avg_pct": fixed_sample,
        "tfidf_sample_avg_pct": tfidf_sample,
        "avengerspro_sample_avg_pct": av_sample,
        "pair_oracle_sample_avg_pct": oracle_sample,

        "tfidf_gain_vs_fixed_sample_pp":
            tfidf_sample - fixed_sample,

        "avengerspro_gain_vs_fixed_sample_pp":
            av_sample - fixed_sample,

        "avengerspro_minus_tfidf_sample_pp":
            av_sample - tfidf_sample,

        "best_fixed_dataset_avg_pct": fixed_dataset,
        "tfidf_dataset_avg_pct": tfidf_dataset,
        "avengerspro_dataset_avg_pct": av_dataset,
        "pair_oracle_dataset_avg_pct": oracle_dataset,

        "tfidf_gain_vs_fixed_dataset_pp":
            tfidf_dataset - fixed_dataset,

        "avengerspro_gain_vs_fixed_dataset_pp":
            av_dataset - fixed_dataset,

        "avengerspro_minus_tfidf_dataset_pp":
            av_dataset - tfidf_dataset,

        "tfidf_winner_accuracy_non_ties_pct":
            winner_accuracy,
    })

    print(
        f"Sample-Avg: Fixed={fixed_sample:.4f}  "
        f"TF-IDF={tfidf_sample:.4f}  "
        f"AvengersPro={av_sample:.4f}  "
        f"Oracle={oracle_sample:.4f}"
    )

    print(
        f"Dataset-Avg: Fixed={fixed_dataset:.4f}  "
        f"TF-IDF={tfidf_dataset:.4f}  "
        f"AvengersPro={av_dataset:.4f}  "
        f"Oracle={oracle_dataset:.4f}"
    )

    print(
        f"AvengersPro - TF-IDF: "
        f"Sample={av_sample-tfidf_sample:+.4f} pp, "
        f"Dataset={av_dataset-tfidf_dataset:+.4f} pp"
    )


results = pd.DataFrame(all_rows)

results_path = OUT / "exact_multiseed_router_comparison.csv"
results.to_csv(results_path, index=False)

dataset_results = pd.DataFrame(all_dataset_rows)

dataset_path = OUT / "tfidf_exact_multiseed_by_dataset.csv"
dataset_results.to_csv(dataset_path, index=False)


summary_metrics = [
    "best_fixed_sample_avg_pct",
    "tfidf_sample_avg_pct",
    "avengerspro_sample_avg_pct",
    "pair_oracle_sample_avg_pct",
    "tfidf_gain_vs_fixed_sample_pp",
    "avengerspro_gain_vs_fixed_sample_pp",
    "avengerspro_minus_tfidf_sample_pp",

    "best_fixed_dataset_avg_pct",
    "tfidf_dataset_avg_pct",
    "avengerspro_dataset_avg_pct",
    "pair_oracle_dataset_avg_pct",
    "tfidf_gain_vs_fixed_dataset_pp",
    "avengerspro_gain_vs_fixed_dataset_pp",
    "avengerspro_minus_tfidf_dataset_pp",

    "tfidf_winner_accuracy_non_ties_pct",
]

summary_rows = []

for metric in summary_metrics:
    series = results[metric]

    summary_rows.append({
        "metric": metric,
        "mean": series.mean(),
        "std": series.std(),
        "min": series.min(),
        "max": series.max(),
    })

summary = pd.DataFrame(summary_rows)

summary_path = (
    OUT / "exact_multiseed_router_comparison_summary.csv"
)

summary.to_csv(summary_path, index=False)


print("\n" + "=" * 72)
print("EXACT APPLES-TO-APPLES MULTI-SEED COMPARISON")
print("=" * 72)

print(
    f"Best Fixed Sample-Avg  : "
    f"{results['best_fixed_sample_avg_pct'].mean():.4f}%"
)

print(
    f"TF-IDF Sample-Avg      : "
    f"{results['tfidf_sample_avg_pct'].mean():.4f}%"
)

print(
    f"AvengersPro Sample-Avg : "
    f"{results['avengerspro_sample_avg_pct'].mean():.4f}%"
)

print(
    f"Pair Oracle Sample-Avg : "
    f"{results['pair_oracle_sample_avg_pct'].mean():.4f}%"
)

print(
    f"\nAvengersPro - TF-IDF Sample difference: "
    f"{results['avengerspro_minus_tfidf_sample_pp'].mean():+.4f} pp"
)

print(
    f"\nBest Fixed Dataset-Avg  : "
    f"{results['best_fixed_dataset_avg_pct'].mean():.4f}%"
)

print(
    f"TF-IDF Dataset-Avg      : "
    f"{results['tfidf_dataset_avg_pct'].mean():.4f}%"
)

print(
    f"AvengersPro Dataset-Avg : "
    f"{results['avengerspro_dataset_avg_pct'].mean():.4f}%"
)

print(
    f"Pair Oracle Dataset-Avg : "
    f"{results['pair_oracle_dataset_avg_pct'].mean():.4f}%"
)

print(
    f"\nAvengersPro - TF-IDF Dataset difference: "
    f"{results['avengerspro_minus_tfidf_dataset_pp'].mean():+.4f} pp"
)

av_sample_wins = (
    results["avengerspro_minus_tfidf_sample_pp"] > 0
).sum()

tfidf_sample_wins = (
    results["avengerspro_minus_tfidf_sample_pp"] < 0
).sum()

sample_ties = (
    results["avengerspro_minus_tfidf_sample_pp"] == 0
).sum()

print(
    f"\nSample-Avg seed wins: "
    f"AvengersPro {av_sample_wins}/10, "
    f"TF-IDF {tfidf_sample_wins}/10, "
    f"ties {sample_ties}/10"
)

av_dataset_wins = (
    results["avengerspro_minus_tfidf_dataset_pp"] > 0
).sum()

tfidf_dataset_wins = (
    results["avengerspro_minus_tfidf_dataset_pp"] < 0
).sum()

dataset_ties = (
    results["avengerspro_minus_tfidf_dataset_pp"] == 0
).sum()

print(
    f"Dataset-Avg seed wins: "
    f"AvengersPro {av_dataset_wins}/10, "
    f"TF-IDF {tfidf_dataset_wins}/10, "
    f"ties {dataset_ties}/10"
)

print("=" * 72)

print("\nSaved:")
print(results_path)
print(summary_path)
print(dataset_path)
