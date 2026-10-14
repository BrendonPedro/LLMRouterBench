from pathlib import Path
from datetime import datetime
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from baselines import BaselineDataLoader


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_1 = "DeepSeek-R1-0528-Qwen3-8B"
MODEL_2 = "Qwen3-8B"

TEST_SIZE = 0.30
RANDOM_SEED = 42

OUTPUT_DIR = Path("local_experiments/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. LOAD LLMROUTERBENCH
# ============================================================

print("Loading LLMRouterBench records...")

loader = BaselineDataLoader("config/baseline_config.yaml")
records = loader.load_all_records()

print(f"Loaded {len(records):,} total benchmark records.")


# ============================================================
# 2. KEEP ONLY OUR TWO BASELINE MODELS
# ============================================================

rows = []

for r in records:
    if r.model_name not in {MODEL_1, MODEL_2}:
        continue

    try:
        score = float(r.score)
    except (TypeError, ValueError):
        continue

    rows.append({
        "dataset": r.dataset_id,
        "split": r.split,
        "index": r.record_index,
        "model": r.model_name,
        "prompt": r.prompt,
        "score": score,
    })

df = pd.DataFrame(rows)

key_cols = ["dataset", "split", "index"]

print(f"Two-model records: {len(df):,}")


# ============================================================
# 3. ALIGN THE TWO MODELS ON THE SAME QUERIES
# ============================================================

score_wide = df.pivot_table(
    index=key_cols,
    columns="model",
    values="score",
    aggfunc="first",
)

prompt_series = (
    df.groupby(key_cols)["prompt"]
    .first()
    .rename("prompt")
)

prompt_variants = df.groupby(key_cols)["prompt"].nunique(dropna=False)
prompt_mismatch_count = int((prompt_variants > 1).sum())

pair = (
    score_wide
    .join(prompt_series)
    .reset_index()
    .dropna(subset=[MODEL_1, MODEL_2])
)

pair = pair.rename(columns={
    MODEL_1: "model_1_score",
    MODEL_2: "model_2_score",
})

pair["prompt"] = pair["prompt"].fillna("")

print(f"Aligned query pairs: {len(pair):,}")
print(f"Prompt mismatches between models: {prompt_mismatch_count}")


# ============================================================
# 4. CREATE QUERY-LEVEL TRAIN / TEST SPLIT
#
# We split BEFORE removing ties.
# Stratifying by dataset keeps roughly the same dataset
# proportions in train and test.
# ============================================================

train_df, test_df = train_test_split(
    pair,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=pair["dataset"],
)

train_df = train_df.copy()
test_df = test_df.copy()

print(f"Training queries: {len(train_df):,}")
print(f"Test queries:     {len(test_df):,}")


# ============================================================
# 5. CREATE TRAINING LABELS
#
# A label means:
# "Which model scored higher on this historical query?"
#
# Ties are removed ONLY from classifier training because they
# do not tell us which model should be preferred.
# ============================================================

def winner(row):
    if row["model_1_score"] > row["model_2_score"]:
        return MODEL_1
    elif row["model_2_score"] > row["model_1_score"]:
        return MODEL_2
    return "Tie"


train_df["winner"] = train_df.apply(winner, axis=1)
test_df["actual_winner"] = test_df.apply(winner, axis=1)

train_pref = train_df[train_df["winner"] != "Tie"].copy()

print(f"Non-tied training examples: {len(train_pref):,}")
print("\nTraining-label counts:")
print(train_pref["winner"].value_counts().to_string())


# ============================================================
# 6. TRAIN SIMPLE ROUTER
#
# TF-IDF:
# converts query text into numerical word/phrase features.
#
# Logistic Regression:
# learns which of the two models is more likely to win.
# ============================================================

router = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=20000,
        ),
    ),
    (
        "logreg",
        LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_SEED,
        ),
    ),
])

print("\nTraining TF-IDF + Logistic Regression router...")

router.fit(
    train_pref["prompt"],
    train_pref["winner"],
)

feature_count = len(
    router.named_steps["tfidf"].vocabulary_
)

print(f"Training complete.")
print(f"TF-IDF features learned: {feature_count:,}")


# ============================================================
# 7. ROUTE THE UNSEEN TEST QUERIES
# ============================================================

test_df["predicted_model"] = router.predict(
    test_df["prompt"]
)

test_df["router_score"] = np.where(
    test_df["predicted_model"] == MODEL_1,
    test_df["model_1_score"],
    test_df["model_2_score"],
)


# ============================================================
# 8. BEST SINGLE BASELINE
#
# IMPORTANT:
# The best fixed model is chosen using TRAINING data only.
# It is then used unchanged on the unseen test set.
# ============================================================

train_model_1_avg = train_df["model_1_score"].mean()
train_model_2_avg = train_df["model_2_score"].mean()

if train_model_1_avg >= train_model_2_avg:
    best_single_model = MODEL_1
    test_df["best_single_score"] = test_df["model_1_score"]
else:
    best_single_model = MODEL_2
    test_df["best_single_score"] = test_df["model_2_score"]


# ============================================================
# 9. RANDOM BASELINE
# ============================================================

rng = np.random.default_rng(RANDOM_SEED)

test_df["random_model"] = rng.choice(
    [MODEL_1, MODEL_2],
    size=len(test_df),
)

test_df["random_score"] = np.where(
    test_df["random_model"] == MODEL_1,
    test_df["model_1_score"],
    test_df["model_2_score"],
)


# ============================================================
# 10. PAIR ORACLE
#
# This is NOT a real router.
#
# It looks at both outcomes after the fact and always takes
# the higher score. It is the theoretical ceiling for this
# two-model pool.
# ============================================================

test_df["pair_oracle_score"] = test_df[
    ["model_1_score", "model_2_score"]
].max(axis=1)


# ============================================================
# 11. CALCULATE METRICS
# ============================================================

random_avg = test_df["random_score"].mean()
best_single_avg = test_df["best_single_score"].mean()
router_avg = test_df["router_score"].mean()
oracle_avg = test_df["pair_oracle_score"].mean()

test_non_tie = test_df[
    test_df["actual_winner"] != "Tie"
].copy()


def winner_accuracy(choice_series):
    if len(test_non_tie) == 0:
        return np.nan

    return (
        choice_series.loc[test_non_tie.index]
        == test_non_tie["actual_winner"]
    ).mean()


random_winner_accuracy = winner_accuracy(
    test_df["random_model"]
)

best_single_choices = pd.Series(
    best_single_model,
    index=test_df.index,
)

best_single_winner_accuracy = winner_accuracy(
    best_single_choices
)

router_winner_accuracy = winner_accuracy(
    test_df["predicted_model"]
)


metrics = pd.DataFrame([
    {
        "method": "Random",
        "test_avg_score_pct": random_avg * 100,
        "gain_vs_best_single_pp":
            (random_avg - best_single_avg) * 100,
        "gap_to_pair_oracle_pp":
            (oracle_avg - random_avg) * 100,
        "winner_accuracy_non_ties_pct":
            random_winner_accuracy * 100,
    },
    {
        "method": f"Best Single ({best_single_model})",
        "test_avg_score_pct": best_single_avg * 100,
        "gain_vs_best_single_pp": 0.0,
        "gap_to_pair_oracle_pp":
            (oracle_avg - best_single_avg) * 100,
        "winner_accuracy_non_ties_pct":
            best_single_winner_accuracy * 100,
    },
    {
        "method": "TF-IDF + Logistic Regression",
        "test_avg_score_pct": router_avg * 100,
        "gain_vs_best_single_pp":
            (router_avg - best_single_avg) * 100,
        "gap_to_pair_oracle_pp":
            (oracle_avg - router_avg) * 100,
        "winner_accuracy_non_ties_pct":
            router_winner_accuracy * 100,
    },
    {
        "method": "Pair Oracle",
        "test_avg_score_pct": oracle_avg * 100,
        "gain_vs_best_single_pp":
            (oracle_avg - best_single_avg) * 100,
        "gap_to_pair_oracle_pp": 0.0,
        "winner_accuracy_non_ties_pct": 100.0,
    },
])


# ============================================================
# 12. DATASET-LEVEL RESULTS
# ============================================================

dataset_results = (
    test_df.groupby("dataset")
    .agg(
        n_test=("index", "size"),
        random_score=("random_score", "mean"),
        best_single_score=("best_single_score", "mean"),
        router_score=("router_score", "mean"),
        pair_oracle_score=("pair_oracle_score", "mean"),
    )
    .reset_index()
)

for col in [
    "random_score",
    "best_single_score",
    "router_score",
    "pair_oracle_score",
]:
    dataset_results[col] *= 100

dataset_results["router_gain_vs_best_single_pp"] = (
    dataset_results["router_score"]
    - dataset_results["best_single_score"]
)


# ============================================================
# 13. SAVE EXACT TRAIN / TEST ASSIGNMENTS
# ============================================================

train_assignments = train_df[
    ["dataset", "split", "index", "winner"]
].copy()

train_assignments["assignment"] = "train"

test_assignments = test_df[
    ["dataset", "split", "index", "actual_winner"]
].copy()

test_assignments = test_assignments.rename(
    columns={"actual_winner": "winner"}
)

test_assignments["assignment"] = "test"

split_assignments = pd.concat(
    [train_assignments, test_assignments],
    ignore_index=True,
)


# ============================================================
# 14. REPRODUCIBILITY METADATA
# ============================================================

try:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()
except Exception:
    git_commit = "unknown"

metadata = pd.DataFrame([
    ["generated_at", datetime.now().astimezone().isoformat(timespec="seconds")],
    ["git_commit", git_commit],
    ["python_version", sys.version.split()[0]],
    ["pandas_version", pd.__version__],
    ["scikit_learn_version", sklearn.__version__],
    ["model_1", MODEL_1],
    ["model_2", MODEL_2],
    ["random_seed", RANDOM_SEED],
    ["test_size", TEST_SIZE],
    ["total_aligned_queries", len(pair)],
    ["training_queries", len(train_df)],
    ["test_queries", len(test_df)],
    ["non_tied_training_queries", len(train_pref)],
    ["non_tied_test_queries", len(test_non_tie)],
    ["tfidf_feature_count", feature_count],
    ["tfidf_ngram_range", "1,2"],
    ["tfidf_min_df", 2],
    ["tfidf_max_features", 20000],
    ["classifier", "LogisticRegression"],
    ["best_single_selected_from_training", best_single_model],
    ["prompt_mismatch_count", prompt_mismatch_count],
], columns=["field", "value"])


# ============================================================
# 15. SAVE EVERYTHING
# ============================================================

metrics.to_csv(
    OUTPUT_DIR / "tfidf_logreg_metrics.csv",
    index=False,
)

dataset_results.to_csv(
    OUTPUT_DIR / "tfidf_logreg_by_dataset.csv",
    index=False,
)

test_df.to_csv(
    OUTPUT_DIR / "tfidf_logreg_test_predictions.csv",
    index=False,
)

split_assignments.to_csv(
    OUTPUT_DIR / "tfidf_logreg_split_assignments.csv",
    index=False,
)

metadata.to_csv(
    OUTPUT_DIR / "tfidf_logreg_config.csv",
    index=False,
)

joblib.dump(
    router,
    OUTPUT_DIR / "tfidf_logreg_router.joblib",
)


# ============================================================
# 16. TERMINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FIRST TRAINED ROUTER RESULTS")
print("=" * 70)

print(
    metrics.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)

print("\nBest Single selected from TRAINING data:")
print(best_single_model)

print("\nFiles saved:")
for filename in [
    "tfidf_logreg_metrics.csv",
    "tfidf_logreg_by_dataset.csv",
    "tfidf_logreg_test_predictions.csv",
    "tfidf_logreg_split_assignments.csv",
    "tfidf_logreg_config.csv",
    "tfidf_logreg_router.joblib",
]:
    print(f" - {OUTPUT_DIR / filename}")
