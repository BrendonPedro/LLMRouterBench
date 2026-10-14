from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from baselines import BaselineDataLoader


MODEL_1 = "DeepSeek-R1-0528-Qwen3-8B"
MODEL_2 = "Qwen3-8B"

SEEDS = list(range(42, 52))
TEST_SIZE = 0.30

OUTPUT_DIR = Path("local_experiments/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("Loading benchmark once...")

loader = BaselineDataLoader("config/baseline_config.yaml")
records = loader.load_all_records()


# ------------------------------------------------------------
# Build aligned two-model query table
# ------------------------------------------------------------

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

keys = ["dataset", "split", "index"]

scores = df.pivot_table(
    index=keys,
    columns="model",
    values="score",
    aggfunc="first",
)

prompts = (
    df.groupby(keys)["prompt"]
    .first()
    .rename("prompt")
)

pair = (
    scores
    .join(prompts)
    .reset_index()
    .dropna(subset=[MODEL_1, MODEL_2])
    .rename(columns={
        MODEL_1: "model_1_score",
        MODEL_2: "model_2_score",
    })
)

pair["prompt"] = pair["prompt"].fillna("")

print(f"Aligned queries: {len(pair):,}")
print(f"Running {len(SEEDS)} seeds...\n")


# ------------------------------------------------------------
# Helper
# ------------------------------------------------------------

def make_winner(frame):
    return np.where(
        frame["model_1_score"] > frame["model_2_score"],
        MODEL_1,
        np.where(
            frame["model_2_score"] > frame["model_1_score"],
            MODEL_2,
            "Tie",
        ),
    )


# ------------------------------------------------------------
# Repeat experiment
# ------------------------------------------------------------

results = []

for seed in SEEDS:

    train_df, test_df = train_test_split(
        pair,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=pair["dataset"],
    )

    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df["winner"] = make_winner(train_df)
    test_df["actual_winner"] = make_winner(test_df)

    train_pref = train_df[
        train_df["winner"] != "Tie"
    ].copy()

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
                random_state=seed,
            ),
        ),
    ])

    router.fit(
        train_pref["prompt"],
        train_pref["winner"],
    )

    test_df["predicted_model"] = router.predict(
        test_df["prompt"]
    )

    test_df["router_score"] = np.where(
        test_df["predicted_model"] == MODEL_1,
        test_df["model_1_score"],
        test_df["model_2_score"],
    )


    # Best Single selected ONLY from training set
    train_m1 = train_df["model_1_score"].mean()
    train_m2 = train_df["model_2_score"].mean()

    if train_m1 >= train_m2:
        best_model = MODEL_1
        test_df["best_single_score"] = test_df["model_1_score"]
    else:
        best_model = MODEL_2
        test_df["best_single_score"] = test_df["model_2_score"]


    # Random baseline
    rng = np.random.default_rng(seed)

    test_df["random_model"] = rng.choice(
        [MODEL_1, MODEL_2],
        size=len(test_df),
    )

    test_df["random_score"] = np.where(
        test_df["random_model"] == MODEL_1,
        test_df["model_1_score"],
        test_df["model_2_score"],
    )


    # Pair Oracle
    test_df["oracle_score"] = test_df[
        ["model_1_score", "model_2_score"]
    ].max(axis=1)


    # Main metrics
    random_score = test_df["random_score"].mean()
    best_score = test_df["best_single_score"].mean()
    router_score = test_df["router_score"].mean()
    oracle_score = test_df["oracle_score"].mean()

    gain = router_score - best_score
    headroom = oracle_score - best_score

    headroom_capture = (
        gain / headroom
        if headroom > 0
        else np.nan
    )


    # Winner accuracy on non-ties
    non_tie = test_df[
        test_df["actual_winner"] != "Tie"
    ]

    router_winner_accuracy = (
        (
            non_tie["predicted_model"]
            == non_tie["actual_winner"]
        ).mean()
        if len(non_tie)
        else np.nan
    )


    results.append({
        "seed": seed,
        "train_queries": len(train_df),
        "test_queries": len(test_df),
        "non_tied_train_queries": len(train_pref),
        "non_tied_test_queries": len(non_tie),
        "best_single_model": best_model,
        "random_score_pct": random_score * 100,
        "best_single_score_pct": best_score * 100,
        "router_score_pct": router_score * 100,
        "pair_oracle_score_pct": oracle_score * 100,
        "router_gain_vs_best_single_pp": gain * 100,
        "available_headroom_pp": headroom * 100,
        "headroom_captured_pct": headroom_capture * 100,
        "router_winner_accuracy_non_ties_pct":
            router_winner_accuracy * 100,
    })

    print(
        f"Seed {seed}: "
        f"Best={best_score * 100:.2f}% | "
        f"Router={router_score * 100:.2f}% | "
        f"Gain={gain * 100:+.2f} pp | "
        f"Oracle={oracle_score * 100:.2f}%"
    )


# ------------------------------------------------------------
# Save per-seed results
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

results_file = (
    OUTPUT_DIR / "tfidf_logreg_multiseed_results.csv"
)

results_df.to_csv(
    results_file,
    index=False,
)


# ------------------------------------------------------------
# Summary across seeds
# ------------------------------------------------------------

numeric_cols = [
    "random_score_pct",
    "best_single_score_pct",
    "router_score_pct",
    "pair_oracle_score_pct",
    "router_gain_vs_best_single_pp",
    "available_headroom_pp",
    "headroom_captured_pct",
    "router_winner_accuracy_non_ties_pct",
]

summary = (
    results_df[numeric_cols]
    .agg(["mean", "std", "min", "max"])
    .T
    .reset_index()
    .rename(columns={"index": "metric"})
)

summary_file = (
    OUTPUT_DIR / "tfidf_logreg_multiseed_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False,
)


# ------------------------------------------------------------
# Terminal summary
# ------------------------------------------------------------

wins = int(
    (results_df["router_gain_vs_best_single_pp"] > 0).sum()
)

ties = int(
    (results_df["router_gain_vs_best_single_pp"] == 0).sum()
)

losses = int(
    (results_df["router_gain_vs_best_single_pp"] < 0).sum()
)


print("\n" + "=" * 70)
print("MULTI-SEED SUMMARY")
print("=" * 70)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)

print(
    f"\nRouter beat Best Single in "
    f"{wins}/{len(SEEDS)} seeds."
)

print(
    f"Router tied Best Single in "
    f"{ties}/{len(SEEDS)} seeds."
)

print(
    f"Router lost to Best Single in "
    f"{losses}/{len(SEEDS)} seeds."
)

print("\nSaved:")
print(f" - {results_file}")
print(f" - {summary_file}")
