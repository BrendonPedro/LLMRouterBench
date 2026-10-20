import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict


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
                "query": x["query"],
                "deepseek_score": float(x["records"][DS]),
                "qwen_score": float(x["records"][QWEN]),
            })

    return pd.DataFrame(rows)


def routing_score(df, prob, threshold):
    choose_qwen = prob >= threshold

    selected = np.where(
        choose_qwen,
        df["qwen_score"].to_numpy(),
        df["deepseek_score"].to_numpy(),
    )

    return selected.mean()


prior_results = pd.read_csv(
    OUT / "dataset_prior_multiseed_comparison.csv"
)

existing = pd.read_csv(
    OUT / "exact_multiseed_router_comparison.csv"
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

    seed_test_parts = []

    for dataset in sorted(test["dataset"].unique()):

        tr = train[train["dataset"] == dataset].copy()
        te = test[test["dataset"] == dataset].copy()

        # --------------------------------------------
        # Dataset prior from training data
        # --------------------------------------------

        prior_model = (
            QWEN
            if tr["qwen_score"].mean()
            >= tr["deepseek_score"].mean()
            else DS
        )

        te["prior_score"] = np.where(
            prior_model == QWEN,
            te["qwen_score"],
            te["deepseek_score"],
        )

        # --------------------------------------------
        # Winner classifier uses NON-TIED examples
        # --------------------------------------------

        tr_nt = tr[
            tr["deepseek_score"] != tr["qwen_score"]
        ].copy()

        if len(tr_nt) < 10:
            te["tuned_score"] = te["prior_score"]
            threshold = np.nan

        else:
            tr_nt["label"] = np.where(
                tr_nt["qwen_score"]
                > tr_nt["deepseek_score"],
                1,
                0,
            )

            if tr_nt["label"].nunique() < 2:
                te["tuned_score"] = te["prior_score"]
                threshold = np.nan

            else:

                # ------------------------------------
                # Shared vectorizer for this dataset
                # ------------------------------------

                vec = TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=20000,
                )

                X_nt = vec.fit_transform(
                    tr_nt["query"]
                )

                y_nt = tr_nt["label"].to_numpy()

                # ------------------------------------
                # Out-of-fold TRAIN probabilities
                # No test data used.
                # ------------------------------------

                min_class = np.bincount(y_nt).min()

                n_splits = min(5, int(min_class))

                if n_splits < 2:
                    te["tuned_score"] = te["prior_score"]
                    threshold = np.nan

                else:
                    cv = StratifiedKFold(
                        n_splits=n_splits,
                        shuffle=True,
                        random_state=seed,
                    )

                    clf_cv = LogisticRegression(
                        max_iter=2000,
                        random_state=seed,
                    )

                    oof_prob = cross_val_predict(
                        clf_cv,
                        X_nt,
                        y_nt,
                        cv=cv,
                        method="predict_proba",
                    )[:, 1]

                    # --------------------------------
                    # Threshold chosen ONLY from
                    # non-tied training cases.
                    #
                    # This directly optimizes routing
                    # winner accuracy/utility.
                    # --------------------------------

                    thresholds = np.linspace(
                        0.05,
                        0.95,
                        181,
                    )

                    best_threshold = None
                    best_score = -1

                    for t in thresholds:

                        pred = (
                            oof_prob >= t
                        ).astype(int)

                        score = (
                            pred == y_nt
                        ).mean()

                        if score > best_score:
                            best_score = score
                            best_threshold = t

                    threshold = best_threshold

                    # --------------------------------
                    # Refit on all non-tied training
                    # --------------------------------

                    clf = LogisticRegression(
                        max_iter=2000,
                        random_state=seed,
                    )

                    clf.fit(X_nt, y_nt)

                    X_test = vec.transform(
                        te["query"]
                    )

                    test_prob = clf.predict_proba(
                        X_test
                    )[:, 1]

                    choose_qwen = (
                        test_prob >= threshold
                    )

                    te["tuned_score"] = np.where(
                        choose_qwen,
                        te["qwen_score"],
                        te["deepseek_score"],
                    )

        te["oracle_score"] = te[
            ["deepseek_score", "qwen_score"]
        ].max(axis=1)

        prior_pct = (
            te["prior_score"].mean() * 100
        )

        tuned_pct = (
            te["tuned_score"].mean() * 100
        )

        oracle_pct = (
            te["oracle_score"].mean() * 100
        )

        dataset_rows.append({
            "seed": seed,
            "dataset": dataset,
            "dataset_prior_model": prior_model,
            "learned_threshold": threshold,
            "dataset_prior_pct": prior_pct,
            "tuned_tfidf_pct": tuned_pct,
            "oracle_pct": oracle_pct,
            "tuned_minus_prior_pp":
                tuned_pct - prior_pct,
            "remaining_oracle_gap_pp":
                oracle_pct - tuned_pct,
        })

        te["seed"] = seed
        te["dataset"] = dataset

        seed_test_parts.append(te)

    seed_test = pd.concat(
        seed_test_parts,
        ignore_index=True,
    )

    tuned_sample = (
        seed_test["tuned_score"].mean() * 100
    )

    prior_sample = (
        seed_test["prior_score"].mean() * 100
    )

    oracle_sample = (
        seed_test["oracle_score"].mean() * 100
    )

    current_dataset = pd.DataFrame(
        [
            r for r in dataset_rows
            if r["seed"] == seed
        ]
    )

    tuned_dataset = (
        current_dataset["tuned_tfidf_pct"].mean()
    )

    prior_dataset = (
        current_dataset["dataset_prior_pct"].mean()
    )

    oracle_dataset = (
        current_dataset["oracle_pct"].mean()
    )

    old = existing[
        existing["seed"] == seed
    ].iloc[0]

    seed_rows.append({
        "seed": seed,

        "dataset_prior_sample_pct":
            prior_sample,

        "standard_tfidf_sample_pct":
            old["tfidf_sample_avg_pct"],

        "tuned_tfidf_sample_pct":
            tuned_sample,

        "avengerspro_sample_pct":
            old["avengerspro_sample_avg_pct"],

        "oracle_sample_pct":
            oracle_sample,

        "tuned_minus_prior_sample_pp":
            tuned_sample - prior_sample,

        "tuned_minus_standard_tfidf_sample_pp":
            tuned_sample
            - old["tfidf_sample_avg_pct"],

        "tuned_minus_avengerspro_sample_pp":
            tuned_sample
            - old["avengerspro_sample_avg_pct"],

        "dataset_prior_dataset_pct":
            prior_dataset,

        "tuned_tfidf_dataset_pct":
            tuned_dataset,

        "oracle_dataset_pct":
            oracle_dataset,

        "tuned_minus_prior_dataset_pp":
            tuned_dataset - prior_dataset,
    })

    print(
        f"Prior={prior_sample:.4f} "
        f"StandardTFIDF={old['tfidf_sample_avg_pct']:.4f} "
        f"TunedTFIDF={tuned_sample:.4f} "
        f"AvengersPro={old['avengerspro_sample_avg_pct']:.4f} "
        f"Oracle={oracle_sample:.4f}"
    )


# ======================================================
# SAVE
# ======================================================

seed_df = pd.DataFrame(seed_rows)

seed_df.to_csv(
    OUT / "utility_tuned_threshold_multiseed.csv",
    index=False,
)

dataset_df = pd.DataFrame(dataset_rows)

dataset_df.to_csv(
    OUT / "utility_tuned_threshold_by_dataset_seed.csv",
    index=False,
)


summary = (
    dataset_df.groupby("dataset")
    .agg(
        prior_pct=("dataset_prior_pct", "mean"),
        tuned_tfidf_pct=("tuned_tfidf_pct", "mean"),
        oracle_pct=("oracle_pct", "mean"),
        tuned_minus_prior_pp=(
            "tuned_minus_prior_pp", "mean"
        ),
        mean_threshold=(
            "learned_threshold", "mean"
        ),
    )
    .reset_index()
)

summary["remaining_oracle_gap_pp"] = (
    summary["oracle_pct"]
    - summary["tuned_tfidf_pct"]
)

summary = summary.sort_values(
    "tuned_minus_prior_pp",
    ascending=False,
)

summary.to_csv(
    OUT / "utility_tuned_threshold_summary.csv",
    index=False,
)


print()
print("=" * 95)
print("UTILITY-TUNED WITHIN-DATASET TF-IDF")
print("=" * 95)

print(
    f"Dataset Prior Sample-Avg : "
    f"{seed_df['dataset_prior_sample_pct'].mean():.4f}%"
)

print(
    f"Standard TF-IDF          : "
    f"{seed_df['standard_tfidf_sample_pct'].mean():.4f}%"
)

print(
    f"Tuned TF-IDF             : "
    f"{seed_df['tuned_tfidf_sample_pct'].mean():.4f}%"
)

print(
    f"AvengersPro              : "
    f"{seed_df['avengerspro_sample_pct'].mean():.4f}%"
)

print(
    f"Oracle                    : "
    f"{seed_df['oracle_sample_pct'].mean():.4f}%"
)

print()

print(
    f"Tuned - Prior            : "
    f"{seed_df['tuned_minus_prior_sample_pp'].mean():+.4f} pp"
)

print(
    f"Tuned - Standard TF-IDF  : "
    f"{seed_df['tuned_minus_standard_tfidf_sample_pp'].mean():+.4f} pp"
)

print(
    f"Tuned - AvengersPro      : "
    f"{seed_df['tuned_minus_avengerspro_sample_pp'].mean():+.4f} pp"
)

print()

print(
    f"Dataset Prior Dataset-Avg: "
    f"{seed_df['dataset_prior_dataset_pct'].mean():.4f}%"
)

print(
    f"Tuned TF-IDF Dataset-Avg : "
    f"{seed_df['tuned_tfidf_dataset_pct'].mean():.4f}%"
)

print(
    f"Oracle Dataset-Avg       : "
    f"{seed_df['oracle_dataset_pct'].mean():.4f}%"
)

print()

print("PER-DATASET RESULTS")
print(
    summary.round(3).to_string(index=False)
)

print()
print("Saved:")
print(
    OUT / "utility_tuned_threshold_multiseed.csv"
)
print(
    OUT / "utility_tuned_threshold_by_dataset_seed.csv"
)
print(
    OUT / "utility_tuned_threshold_summary.csv"
)
