import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


TRAIN_PATH = Path(
    "baselines/AvengersPro/data/small_models_seed_42/"
    "seed42_split0.7/train.jsonl"
)

TEST_PATH = Path(
    "baselines/AvengersPro/data/small_models_seed_42/"
    "seed42_split0.7/test.jsonl"
)

OUT = Path("local_experiments/results")
OUT.mkdir(parents=True, exist_ok=True)

MODEL_DS = "DeepSeek-R1-0528-Qwen3-8B"
MODEL_QWEN = "Qwen3-8B"


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
                "deepseek_score": float(
                    records.get(MODEL_DS, 0.0)
                ),
                "qwen_score": float(
                    records.get(MODEL_QWEN, 0.0)
                ),
            })

    return pd.DataFrame(rows)


train = load_jsonl(TRAIN_PATH)
test = load_jsonl(TEST_PATH)

print("Train queries:", len(train))
print("Test queries :", len(test))


# ---------------------------------------------------------
# TRAINING LABELS
# ---------------------------------------------------------
# Train classifier only on queries where one model
# is strictly better than the other.

train_non_tie = train[
    train["deepseek_score"] != train["qwen_score"]
].copy()

train_non_tie["label"] = np.where(
    train_non_tie["qwen_score"]
    > train_non_tie["deepseek_score"],
    1,  # Qwen wins
    0,  # DeepSeek wins
)

print("Non-tied train examples:", len(train_non_tie))
print(
    "DeepSeek wins in train:",
    int((train_non_tie["label"] == 0).sum())
)
print(
    "Qwen wins in train    :",
    int((train_non_tie["label"] == 1).sum())
)


# ---------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=20000,
)

X_train = vectorizer.fit_transform(
    train_non_tie["query"]
)

X_test = vectorizer.transform(
    test["query"]
)

print("TF-IDF features:", X_train.shape[1])


# ---------------------------------------------------------
# LOGISTIC REGRESSION
# ---------------------------------------------------------

clf = LogisticRegression(
    max_iter=2000,
    random_state=42,
)

clf.fit(
    X_train,
    train_non_tie["label"],
)

pred = clf.predict(X_test)

test["router_choice"] = np.where(
    pred == 1,
    MODEL_QWEN,
    MODEL_DS,
)

test["router_score"] = np.where(
    pred == 1,
    test["qwen_score"],
    test["deepseek_score"],
)


# ---------------------------------------------------------
# BASELINES / ORACLE
# ---------------------------------------------------------

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

test["pair_oracle_score"] = test[
    ["deepseek_score", "qwen_score"]
].max(axis=1)

test["is_tie"] = (
    test["deepseek_score"]
    == test["qwen_score"]
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

test["router_correct_winner"] = (
    test["router_choice"]
    == test["true_winner"]
)


# ---------------------------------------------------------
# SAMPLE-AVERAGE METRICS
# ---------------------------------------------------------

router_sample = test["router_score"].mean() * 100
fixed_sample = test["best_fixed_score"].mean() * 100
oracle_sample = test["pair_oracle_score"].mean() * 100

non_tie = test[~test["is_tie"]]

winner_accuracy = (
    non_tie["router_correct_winner"].mean() * 100
    if len(non_tie)
    else np.nan
)

gain_pp = router_sample - fixed_sample
headroom_pp = oracle_sample - fixed_sample

captured_pct = (
    gain_pp / headroom_pp * 100
    if headroom_pp > 0
    else np.nan
)


# ---------------------------------------------------------
# DATASET-AVERAGE METRICS
# ---------------------------------------------------------

by_dataset = (
    test.groupby("dataset")
    .agg(
        n_test=("query", "size"),
        router_accuracy=("router_score", "mean"),
        best_fixed_accuracy=("best_fixed_score", "mean"),
        pair_oracle_accuracy=("pair_oracle_score", "mean"),
    )
    .reset_index()
)

by_dataset["router_accuracy_pct"] = (
    by_dataset["router_accuracy"] * 100
)
by_dataset["best_fixed_accuracy_pct"] = (
    by_dataset["best_fixed_accuracy"] * 100
)
by_dataset["pair_oracle_accuracy_pct"] = (
    by_dataset["pair_oracle_accuracy"] * 100
)

by_dataset["gain_vs_fixed_pp"] = (
    by_dataset["router_accuracy_pct"]
    - by_dataset["best_fixed_accuracy_pct"]
)

router_dataset_avg = (
    by_dataset["router_accuracy_pct"].mean()
)

fixed_dataset_avg = (
    by_dataset["best_fixed_accuracy_pct"].mean()
)

oracle_dataset_avg = (
    by_dataset["pair_oracle_accuracy_pct"].mean()
)


# ---------------------------------------------------------
# SAVE METRICS
# ---------------------------------------------------------

metrics = pd.DataFrame([
    {
        "method": "TF-IDF + Logistic Regression",
        "split": "AvengersPro seed42 split0.7",
        "candidate_models": 2,
        "train_queries": len(train),
        "test_queries": len(test),
        "non_tied_train_queries": len(train_non_tie),
        "best_fixed_model": best_fixed_model,
        "sample_avg_pct": router_sample,
        "best_fixed_sample_avg_pct": fixed_sample,
        "pair_oracle_sample_avg_pct": oracle_sample,
        "gain_vs_best_fixed_sample_pp": gain_pp,
        "available_sample_headroom_pp": headroom_pp,
        "sample_headroom_captured_pct": captured_pct,
        "winner_accuracy_non_ties_pct": winner_accuracy,
        "dataset_avg_pct": router_dataset_avg,
        "best_fixed_dataset_avg_pct": fixed_dataset_avg,
        "pair_oracle_dataset_avg_pct": oracle_dataset_avg,
        "gain_vs_best_fixed_dataset_pp":
            router_dataset_avg - fixed_dataset_avg,
    }
])

metrics.to_csv(
    OUT / "tfidf_exact_avengers_split_metrics.csv",
    index=False,
)

by_dataset.to_csv(
    OUT / "tfidf_exact_avengers_split_by_dataset.csv",
    index=False,
)

test.to_csv(
    OUT / "tfidf_exact_avengers_split_predictions.csv",
    index=False,
)

joblib.dump(
    {
        "vectorizer": vectorizer,
        "classifier": clf,
        "models": [MODEL_DS, MODEL_QWEN],
    },
    OUT / "tfidf_exact_avengers_split_router.joblib",
)


print("\n========================================")
print("EXACT-SPLIT TF-IDF RESULTS")
print("========================================")
print(f"Best fixed model        : {best_fixed_model}")
print(f"Sample-Avg")
print(f"  Best fixed            : {fixed_sample:.4f}%")
print(f"  TF-IDF router         : {router_sample:.4f}%")
print(f"  Pair Oracle           : {oracle_sample:.4f}%")
print(f"  Router gain           : {gain_pp:+.4f} pp")
print(f"  Headroom captured     : {captured_pct:.2f}%")
print()
print(f"Dataset-Avg")
print(f"  Best fixed            : {fixed_dataset_avg:.4f}%")
print(f"  TF-IDF router         : {router_dataset_avg:.4f}%")
print(f"  Pair Oracle           : {oracle_dataset_avg:.4f}%")
print(
    f"  Router gain           : "
    f"{router_dataset_avg-fixed_dataset_avg:+.4f} pp"
)
print()
print(
    f"Winner accuracy non-ties: "
    f"{winner_accuracy:.2f}%"
)
print("========================================")
