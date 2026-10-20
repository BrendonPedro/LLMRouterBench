import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold


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


existing = pd.read_csv(
    OUT / "exact_multiseed_router_comparison.csv"
)

dataset_prior = pd.read_csv(
    OUT / "dataset_prior_multiseed_comparison.csv"
)

uncalibrated = pd.read_csv(
    OUT / "tfidf_utility_heads_multiseed.csv"
)


for seed in range(42, 52):

    print()
    print("=" * 70)
    print(f"SEED {seed}")
    print("=" * 70)

    base = Path(
        f"baselines/AvengersPro/data/"
        f"small_models_seed_{seed}/seed{seed}_split0.7"
    )

    train = load_jsonl(base / "train.jsonl")
    test = load_jsonl(base / "test.jsonl")

    # ==================================================
    # SAME TF-IDF REPRESENTATION AS BEFORE
    # ==================================================

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=20000,
    )

    X_train = vectorizer.fit_transform(train["query"])
    X_test = vectorizer.transform(test["query"])

    y_ds_train = train["deepseek_score"].astype(int)
    y_qw_train = train["qwen_score"].astype(int)

    y_ds_test = test["deepseek_score"].astype(int)
    y_qw_test = test["qwen_score"].astype(int)

    # ==================================================
    # UNCALIBRATED HEADS
    # for probability-quality comparison
    # ==================================================

    raw_ds = LogisticRegression(
        max_iter=2000,
        random_state=seed,
    )

    raw_qw = LogisticRegression(
        max_iter=2000,
        random_state=seed,
    )

    raw_ds.fit(X_train, y_ds_train)
    raw_qw.fit(X_train, y_qw_train)

    raw_ds_prob = raw_ds.predict_proba(X_test)[:, 1]
    raw_qw_prob = raw_qw.predict_proba(X_test)[:, 1]

    # ==================================================
    # CALIBRATED UTILITY HEADS
    #
    # 5-fold training-only calibration.
    # Test data is never used for fitting/calibration.
    # ==================================================

    ds_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed,
    )

    qw_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed,
    )

    ds_head = CalibratedClassifierCV(
        estimator=LogisticRegression(
            max_iter=2000,
            random_state=seed,
        ),
        method="sigmoid",
        cv=ds_cv,
    )

    qwen_head = CalibratedClassifierCV(
        estimator=LogisticRegression(
            max_iter=2000,
            random_state=seed,
        ),
        method="sigmoid",
        cv=qw_cv,
    )

    ds_head.fit(X_train, y_ds_train)
    qwen_head.fit(X_train, y_qw_train)

    ds_prob = ds_head.predict_proba(X_test)[:, 1]
    qw_prob = qwen_head.predict_proba(X_test)[:, 1]

    test["deepseek_calibrated_utility"] = ds_prob
    test["qwen_calibrated_utility"] = qw_prob

    test["calibrated_choice"] = np.where(
        test["qwen_calibrated_utility"]
        >= test["deepseek_calibrated_utility"],
        QWEN,
        DS,
    )

    test["calibrated_score"] = np.where(
        test["calibrated_choice"] == QWEN,
        test["qwen_score"],
        test["deepseek_score"],
    )

    test["oracle_score"] = test[
        ["deepseek_score", "qwen_score"]
    ].max(axis=1)

    # ==================================================
    # CALIBRATION QUALITY
    # lower Brier score = better probability estimates
    # ==================================================

    raw_ds_brier = brier_score_loss(
        y_ds_test,
        raw_ds_prob,
    )

    calibrated_ds_brier = brier_score_loss(
        y_ds_test,
        ds_prob,
    )

    raw_qw_brier = brier_score_loss(
        y_qw_test,
        raw_qw_prob,
    )

    calibrated_qw_brier = brier_score_loss(
        y_qw_test,
        qw_prob,
    )

    # ==================================================
    # SAMPLE-AVERAGE
    # ==================================================

    calibrated_sample = (
        test["calibrated_score"].mean() * 100
    )

    oracle_sample = (
        test["oracle_score"].mean() * 100
    )

    # ==================================================
    # DATASET-AVERAGE
    # ==================================================

    by_dataset = (
        test.groupby("dataset")
        .agg(
            n_test=("query", "size"),

            calibrated_accuracy=(
                "calibrated_score", "mean"
            ),

            oracle_accuracy=(
                "oracle_score", "mean"
            ),

            deepseek_selection=(
                "calibrated_choice",
                lambda s: (s == DS).mean()
            ),

            qwen_selection=(
                "calibrated_choice",
                lambda s: (s == QWEN).mean()
            ),
        )
        .reset_index()
    )

    by_dataset["calibrated_accuracy_pct"] = (
        by_dataset["calibrated_accuracy"] * 100
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

    calibrated_dataset = (
        by_dataset["calibrated_accuracy_pct"].mean()
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

    raw = uncalibrated[
        uncalibrated["seed"] == seed
    ].iloc[0]

    seed_rows.append({
        "seed": seed,

        "dataset_prior_sample_avg_pct":
            prior["dataset_prior_sample_avg_pct"],

        "winner_tfidf_sample_avg_pct":
            old["tfidf_sample_avg_pct"],

        "uncalibrated_utility_sample_avg_pct":
            raw["utility_heads_sample_avg_pct"],

        "calibrated_utility_sample_avg_pct":
            calibrated_sample,

        "avengerspro_sample_avg_pct":
            old["avengerspro_sample_avg_pct"],

        "oracle_sample_avg_pct":
            oracle_sample,

        "calibrated_minus_uncalibrated_sample_pp":
            calibrated_sample
            - raw["utility_heads_sample_avg_pct"],

        "calibrated_minus_winner_tfidf_sample_pp":
            calibrated_sample
            - old["tfidf_sample_avg_pct"],

        "calibrated_minus_dataset_prior_sample_pp":
            calibrated_sample
            - prior["dataset_prior_sample_avg_pct"],

        "calibrated_minus_avengerspro_sample_pp":
            calibrated_sample
            - old["avengerspro_sample_avg_pct"],

        "dataset_prior_dataset_avg_pct":
            prior["dataset_prior_dataset_avg_pct"],

        "winner_tfidf_dataset_avg_pct":
            old["tfidf_dataset_avg_pct"],

        "uncalibrated_utility_dataset_avg_pct":
            raw["utility_heads_dataset_avg_pct"],

        "calibrated_utility_dataset_avg_pct":
            calibrated_dataset,

        "avengerspro_dataset_avg_pct":
            old["avengerspro_dataset_avg_pct"],

        "oracle_dataset_avg_pct":
            oracle_dataset,

        "calibrated_minus_uncalibrated_dataset_pp":
            calibrated_dataset
            - raw["utility_heads_dataset_avg_pct"],

        "calibrated_minus_dataset_prior_dataset_pp":
            calibrated_dataset
            - prior["dataset_prior_dataset_avg_pct"],

        "raw_deepseek_brier":
            raw_ds_brier,

        "calibrated_deepseek_brier":
            calibrated_ds_brier,

        "raw_qwen_brier":
            raw_qw_brier,

        "calibrated_qwen_brier":
            calibrated_qw_brier,
    })

    for _, row in by_dataset.iterrows():

        dataset_rows.append({
            "seed": seed,
            "dataset": row["dataset"],
            "n_test": row["n_test"],

            "calibrated_accuracy_pct":
                row["calibrated_accuracy_pct"],

            "oracle_accuracy_pct":
                row["oracle_accuracy_pct"],

            "deepseek_selection_pct":
                row["deepseek_selection_pct"],

            "qwen_selection_pct":
                row["qwen_selection_pct"],
        })

    print(
        f"Sample: "
        f"Calibrated={calibrated_sample:.4f} "
        f"RawUtility={raw['utility_heads_sample_avg_pct']:.4f} "
        f"WinnerTFIDF={old['tfidf_sample_avg_pct']:.4f} "
        f"Prior={prior['dataset_prior_sample_avg_pct']:.4f} "
        f"AvengersPro={old['avengerspro_sample_avg_pct']:.4f} "
        f"Oracle={oracle_sample:.4f}"
    )

    print(
        f"Brier DS: "
        f"{raw_ds_brier:.4f} -> "
        f"{calibrated_ds_brier:.4f} | "
        f"Qwen: "
        f"{raw_qw_brier:.4f} -> "
        f"{calibrated_qw_brier:.4f}"
    )


# ======================================================
# SAVE
# ======================================================

results = pd.DataFrame(seed_rows)

results.to_csv(
    OUT / "calibrated_tfidf_utility_heads_multiseed.csv",
    index=False,
)

dataset_results = pd.DataFrame(dataset_rows)

dataset_results.to_csv(
    OUT / "calibrated_tfidf_utility_heads_by_dataset_seed.csv",
    index=False,
)


dataset_summary = (
    dataset_results.groupby("dataset")
    .agg(
        calibrated_utility_pct=(
            "calibrated_accuracy_pct", "mean"
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
    - dataset_summary["calibrated_utility_pct"]
)

dataset_summary = dataset_summary.sort_values(
    "remaining_oracle_gap_pp",
    ascending=False,
)

dataset_summary.to_csv(
    OUT / "calibrated_tfidf_utility_heads_dataset_summary.csv",
    index=False,
)


# ======================================================
# PRINT SUMMARY
# ======================================================

print()
print("=" * 95)
print("CALIBRATED TF-IDF MODEL-UTILITY HEADS")
print("=" * 95)

print(
    f"Dataset Prior Sample-Avg       : "
    f"{results['dataset_prior_sample_avg_pct'].mean():.4f}%"
)

print(
    f"Winner TF-IDF Sample-Avg       : "
    f"{results['winner_tfidf_sample_avg_pct'].mean():.4f}%"
)

print(
    f"Raw Utility Heads Sample-Avg   : "
    f"{results['uncalibrated_utility_sample_avg_pct'].mean():.4f}%"
)

print(
    f"Calibrated Utility Sample-Avg  : "
    f"{results['calibrated_utility_sample_avg_pct'].mean():.4f}%"
)

print(
    f"AvengersPro Sample-Avg         : "
    f"{results['avengerspro_sample_avg_pct'].mean():.4f}%"
)

print(
    f"Oracle Sample-Avg              : "
    f"{results['oracle_sample_avg_pct'].mean():.4f}%"
)

print()

print(
    f"Calibration vs Raw Utility     : "
    f"{results['calibrated_minus_uncalibrated_sample_pp'].mean():+.4f} pp"
)

print(
    f"Calibration vs Winner TF-IDF   : "
    f"{results['calibrated_minus_winner_tfidf_sample_pp'].mean():+.4f} pp"
)

print(
    f"Calibration vs Dataset Prior   : "
    f"{results['calibrated_minus_dataset_prior_sample_pp'].mean():+.4f} pp"
)

print(
    f"Calibration vs AvengersPro     : "
    f"{results['calibrated_minus_avengerspro_sample_pp'].mean():+.4f} pp"
)

print()

print(
    f"Dataset Prior Dataset-Avg      : "
    f"{results['dataset_prior_dataset_avg_pct'].mean():.4f}%"
)

print(
    f"Winner TF-IDF Dataset-Avg      : "
    f"{results['winner_tfidf_dataset_avg_pct'].mean():.4f}%"
)

print(
    f"Raw Utility Heads Dataset-Avg  : "
    f"{results['uncalibrated_utility_dataset_avg_pct'].mean():.4f}%"
)

print(
    f"Calibrated Utility Dataset-Avg : "
    f"{results['calibrated_utility_dataset_avg_pct'].mean():.4f}%"
)

print(
    f"AvengersPro Dataset-Avg        : "
    f"{results['avengerspro_dataset_avg_pct'].mean():.4f}%"
)

print(
    f"Oracle Dataset-Avg             : "
    f"{results['oracle_dataset_avg_pct'].mean():.4f}%"
)

print()

print("MEAN BRIER SCORES (LOWER IS BETTER)")
print(
    f"DeepSeek: "
    f"{results['raw_deepseek_brier'].mean():.4f} "
    f"-> "
    f"{results['calibrated_deepseek_brier'].mean():.4f}"
)

print(
    f"Qwen    : "
    f"{results['raw_qwen_brier'].mean():.4f} "
    f"-> "
    f"{results['calibrated_qwen_brier'].mean():.4f}"
)

print()
print("PER-DATASET CALIBRATED RESULTS")
print()

print(
    dataset_summary.round(2).to_string(index=False)
)

print()
print("Saved:")
print(
    OUT / "calibrated_tfidf_utility_heads_multiseed.csv"
)
print(
    OUT / "calibrated_tfidf_utility_heads_by_dataset_seed.csv"
)
print(
    OUT / "calibrated_tfidf_utility_heads_dataset_summary.csv"
)
