import argparse
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import Normalizer


DS = "DeepSeek-R1-0528-Qwen3-8B"
QWEN = "Qwen3-8B"

EMBEDDING_MODEL = "Alibaba-NLP/gte-Qwen2-7B-instruct"

CACHE = Path(
    "baselines/AvengersPro/cache/"
    "gte_qwen2_7b_embeddings.sqlite"
)

OUT = Path("local_experiments/results")
OUT.mkdir(parents=True, exist_ok=True)


def load_jsonl(path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)

            rows.append({
                "dataset": x["dataset"],
                "index": x.get("index"),
                "query": x["query"],
                "deepseek_score": float(
                    x["records"][DS]
                ),
                "qwen_score": float(
                    x["records"][QWEN]
                ),
            })

    return pd.DataFrame(rows)


def load_embedding_cache():
    print(f"Loading embedding cache: {CACHE}")

    conn = sqlite3.connect(CACHE)

    rows = conn.execute(
        """
        SELECT text, embedding
        FROM embeddings
        WHERE model = ?
        """,
        (EMBEDDING_MODEL,),
    ).fetchall()

    conn.close()

    cache = {}

    for text, embedding_json in rows:
        cache[text] = np.asarray(
            json.loads(embedding_json),
            dtype=np.float32,
        )

    if not cache:
        raise RuntimeError(
            "No cached GTE embeddings found."
        )

    first = next(iter(cache.values()))

    print(
        f"Cached query embeddings : "
        f"{len(cache)}"
    )
    print(
        f"Embedding dimension     : "
        f"{len(first)}"
    )

    return cache


def embedding_matrix(queries, cache):
    missing = [
        q for q in queries
        if q not in cache
    ]

    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} query embeddings."
        )

    return np.stack(
        [cache[q] for q in queries],
        axis=0,
    )


def train_one_seed(
    seed,
    args,
    embedding_cache,
    baseline_exact,
    baseline_prior,
    baseline_micro,
    baseline_direct,
):
    print()
    print("=" * 90)
    print(f"GTE FOUR-OUTCOME SELECTOR SEED {seed}")
    print("=" * 90)

    base = Path(
        f"baselines/AvengersPro/data/"
        f"small_models_seed_{seed}/"
        f"seed{seed}_split0.7"
    )

    train = load_jsonl(
        base / "train.jsonl"
    )

    test = load_jsonl(
        base / "test.jsonl"
    )

    # Direct model selection has no unique winner on ties.
    # Remove ties from training only.
    train_selector = train.copy()

    train_selector["outcome_class"] = np.select(
        [
            (train_selector["deepseek_score"] == 0)
            & (train_selector["qwen_score"] == 0),

            (train_selector["deepseek_score"] == 1)
            & (train_selector["qwen_score"] == 0),

            (train_selector["deepseek_score"] == 0)
            & (train_selector["qwen_score"] == 1),

            (train_selector["deepseek_score"] == 1)
            & (train_selector["qwen_score"] == 1),
        ],
        [0, 1, 2, 3],
        default=-1,
    )

    if (train_selector["outcome_class"] == -1).any():
        raise RuntimeError(
            "Unexpected non-binary model scores in training data"
        )

    print(
        f"Train queries: {len(train)} | "
        f"Four-outcome train: {len(train_selector)} | "
        f"Test queries: {len(test)}"
    )

    # --------------------------------------------------
    # FIXED SEMANTIC REPRESENTATION
    # Same GTE embedding family used by AvengersPro.
    # --------------------------------------------------

    X_train = embedding_matrix(
        train_selector["query"].tolist(),
        embedding_cache,
    )

    X_test = embedding_matrix(
        test["query"].tolist(),
        embedding_cache,
    )

    # Match AvengersPro's L2 normalization.
    normalizer = Normalizer(norm="l2")

    X_train = normalizer.fit_transform(
        X_train
    )

    X_test = normalizer.transform(
        X_test
    )

    y_train = (
        train_selector["outcome_class"]
        .astype(int)
        .to_numpy()
    )

    print(
        "Training class counts: "
        f"both_wrong={int((y_train == 0).sum())} | "
        f"deepseek_only={int((y_train == 1).sum())} | "
        f"qwen_only={int((y_train == 2).sum())} | "
        f"both_correct={int((y_train == 3).sum())}"
    )

    # --------------------------------------------------
    # SIMPLE SUPERVISED ROUTER
    # No class weighting or hyperparameter tuning.
    # --------------------------------------------------

    classifier = LogisticRegression(
        C=args.C,
        solver="lbfgs",
        max_iter=args.max_iter,
        random_state=seed,
        class_weight=None,
    )

    train_start = time.perf_counter()

    classifier.fit(
        X_train,
        y_train,
    )

    train_seconds = (
        time.perf_counter()
        - train_start
    )

    if list(classifier.classes_) != [0, 1, 2, 3]:
        raise RuntimeError(
            f"Unexpected classes: "
            f"{classifier.classes_}"
        )

    inference_start = time.perf_counter()

    probabilities = classifier.predict_proba(
        X_test
    )

    predicted_outcome_class = classifier.predict(
        X_test
    )

    # Binary routing decision derived only from the
    # two classes where model choice changes quality.
    deepseek_only_probability = probabilities[:, 1]
    qwen_only_probability = probabilities[:, 2]

    selector_class = (
        qwen_only_probability
        > deepseek_only_probability
    ).astype(int)

    inference_seconds = (
        time.perf_counter()
        - inference_start
    )

    test["both_wrong_probability"] = (
        probabilities[:, 0]
    )

    test["deepseek_only_probability"] = (
        deepseek_only_probability
    )

    test["qwen_only_probability"] = (
        qwen_only_probability
    )

    test["both_correct_probability"] = (
        probabilities[:, 3]
    )

    test["predicted_outcome_class"] = (
        predicted_outcome_class.astype(int)
    )

    # Preserve these binary fields so the existing
    # evaluation code remains directly comparable.
    test["deepseek_probability"] = (
        deepseek_only_probability
    )

    test["qwen_probability"] = (
        qwen_only_probability
    )

    test["selector_class"] = (
        selector_class.astype(int)
    )

    test["direct_selector_choice"] = np.where(
        test["selector_class"] == 1,
        QWEN,
        DS,
    )

    test["gte_direct_score"] = np.where(
        test["direct_selector_choice"] == QWEN,
        test["qwen_score"],
        test["deepseek_score"],
    )

    # --------------------------------------------------
    # ROUTER CLASSIFICATION METRICS
    # Evaluate only non-tie test examples.
    # --------------------------------------------------

    test["selector_target"] = np.select(
        [
            test["deepseek_score"]
            > test["qwen_score"],

            test["qwen_score"]
            > test["deepseek_score"],
        ],
        [0, 1],
        default=-1,
    )

    selector_test = test[
        test["selector_target"] != -1
    ].copy()

    y_true = (
        selector_test["selector_target"]
        .astype(int)
        .to_numpy()
    )

    y_pred = (
        selector_test["selector_class"]
        .astype(int)
        .to_numpy()
    )

    y_prob_qwen = (
        selector_test["qwen_only_probability"]
        - selector_test["deepseek_only_probability"]
    ).to_numpy()

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            y_pred,
        )
    )

    mcc = matthews_corrcoef(
        y_true,
        y_pred,
    )

    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(
            y_true,
            y_prob_qwen,
        )
    else:
        auc = np.nan

    # --------------------------------------------------
    # DOWNSTREAM ROUTED TASK PERFORMANCE
    # --------------------------------------------------

    test["oracle_score"] = test[
        [
            "deepseek_score",
            "qwen_score",
        ]
    ].max(axis=1)

    gte_sample = (
        test["gte_direct_score"].mean()
        * 100
    )

    oracle_sample = (
        test["oracle_score"].mean()
        * 100
    )

    by_dataset = (
        test.groupby("dataset")
        .agg(
            n_test=("query", "size"),

            gte_task_score=(
                "gte_direct_score",
                "mean",
            ),

            oracle_task_score=(
                "oracle_score",
                "mean",
            ),

            deepseek_selection=(
                "direct_selector_choice",
                lambda s: (s == DS).mean(),
            ),

            qwen_selection=(
                "direct_selector_choice",
                lambda s: (s == QWEN).mean(),
            ),
        )
        .reset_index()
    )

    by_dataset[
        "gte_task_score_pct"
    ] = (
        by_dataset["gte_task_score"]
        * 100
    )

    by_dataset[
        "oracle_task_score_pct"
    ] = (
        by_dataset["oracle_task_score"]
        * 100
    )

    by_dataset[
        "deepseek_selection_pct"
    ] = (
        by_dataset["deepseek_selection"]
        * 100
    )

    by_dataset[
        "qwen_selection_pct"
    ] = (
        by_dataset["qwen_selection"]
        * 100
    )

    gte_dataset = (
        by_dataset[
            "gte_task_score_pct"
        ].mean()
    )

    oracle_dataset = (
        by_dataset[
            "oracle_task_score_pct"
        ].mean()
    )

    # --------------------------------------------------
    # EXISTING CONTROLLED BASELINES
    # --------------------------------------------------

    old = baseline_exact[
        baseline_exact["seed"] == seed
    ].iloc[0]

    prior = baseline_prior[
        baseline_prior["seed"] == seed
    ].iloc[0]

    independent_micro = baseline_micro[
        baseline_micro["seed"] == seed
    ].iloc[0]

    tiny_direct = baseline_direct[
        baseline_direct["seed"] == seed
    ].iloc[0]

    classifier_parameters = (
        classifier.coef_.size
        + classifier.intercept_.size
    )

    row = {
        "seed": seed,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": X_train.shape[1],
        "classifier":
            "multinomial_logistic_regression",

        "training_target":
            "four_joint_outcomes",

        "routing_rule":
            "qwen_only_prob_gt_deepseek_only_prob",

        "C": args.C,
        "class_weight": "none",
        "classifier_parameters":
            classifier_parameters,

        "train_queries":
            len(train),

        "non_tie_train_queries":
            int(
                (
                    train["deepseek_score"]
                    != train["qwen_score"]
                ).sum()
            ),

        "four_outcome_train_queries":
            len(train_selector),

        "non_tie_test_queries":
            len(selector_test),

        "training_seconds":
            train_seconds,

        "inference_seconds":
            inference_seconds,

        # Primary router metric.
        "router_macro_f1":
            macro_f1,

        "router_macro_precision":
            macro_precision,

        "router_macro_recall":
            macro_recall,

        "router_balanced_accuracy":
            balanced_accuracy,

        "router_auc":
            auc,

        "router_mcc":
            mcc,

        # Direct comparison to tiny BERT selector.
        "tiny_direct_router_macro_f1":
            tiny_direct[
                "router_macro_f1"
            ],

        "gte_minus_tiny_macro_f1":
            macro_f1
            - tiny_direct[
                "router_macro_f1"
            ],

        # Downstream task performance.
        "gte_sample_avg_pct":
            gte_sample,

        "tiny_direct_sample_avg_pct":
            tiny_direct[
                "direct_selector_sample_avg_pct"
            ],

        "gte_minus_tiny_sample_pp":
            gte_sample
            - tiny_direct[
                "direct_selector_sample_avg_pct"
            ],

        "independent_microencoder_sample_avg_pct":
            independent_micro[
                "microencoder_sample_avg_pct"
            ],

        "dataset_prior_sample_avg_pct":
            prior[
                "dataset_prior_sample_avg_pct"
            ],

        "tfidf_sample_avg_pct":
            old[
                "tfidf_sample_avg_pct"
            ],

        "avengerspro_sample_avg_pct":
            old[
                "avengerspro_sample_avg_pct"
            ],

        "oracle_sample_avg_pct":
            oracle_sample,

        "gte_dataset_avg_pct":
            gte_dataset,

        "tiny_direct_dataset_avg_pct":
            tiny_direct[
                "direct_selector_dataset_avg_pct"
            ],

        "gte_minus_tiny_dataset_pp":
            gte_dataset
            - tiny_direct[
                "direct_selector_dataset_avg_pct"
            ],

        "dataset_prior_dataset_avg_pct":
            prior[
                "dataset_prior_dataset_avg_pct"
            ],

        "tfidf_dataset_avg_pct":
            old[
                "tfidf_dataset_avg_pct"
            ],

        "avengerspro_dataset_avg_pct":
            old[
                "avengerspro_dataset_avg_pct"
            ],

        "oracle_dataset_avg_pct":
            oracle_dataset,
    }

    # --------------------------------------------------
    # SAVE SEED EVIDENCE
    # --------------------------------------------------

    prediction_columns = [
        "dataset",
        "index",
        "deepseek_score",
        "qwen_score",
        "both_wrong_probability",
        "deepseek_only_probability",
        "qwen_only_probability",
        "both_correct_probability",
        "predicted_outcome_class",
        "deepseek_probability",
        "qwen_probability",
        "selector_target",
        "selector_class",
        "direct_selector_choice",
        "gte_direct_score",
        "oracle_score",
    ]

    test[
        prediction_columns
    ].to_csv(
        OUT
        / f"gte_four_outcome_selector_seed{seed}_predictions.csv",
        index=False,
    )

    by_dataset.to_csv(
        OUT
        / f"gte_four_outcome_selector_seed{seed}_by_dataset.csv",
        index=False,
    )

    # --------------------------------------------------
    # TERMINAL SUMMARY
    # --------------------------------------------------

    print()
    print("-" * 90)
    print("ROUTER MODEL-SELECTION PERFORMANCE")

    print(
        f"Macro-F1 [primary]        : "
        f"{macro_f1:.4f}"
    )

    print(
        f"Tiny direct Macro-F1      : "
        f"{tiny_direct['router_macro_f1']:.4f}"
    )

    print(
        f"GTE - tiny F1             : "
        f"{macro_f1 - tiny_direct['router_macro_f1']:+.4f}"
    )

    print(
        f"Macro Precision           : "
        f"{macro_precision:.4f}"
    )

    print(
        f"Macro Recall              : "
        f"{macro_recall:.4f}"
    )

    print(
        f"Balanced Accuracy         : "
        f"{balanced_accuracy:.4f}"
    )

    print(
        f"AUC                       : "
        f"{auc:.4f}"
    )

    print(
        f"MCC                       : "
        f"{mcc:.4f}"
    )

    print(
        f"Non-tie test queries      : "
        f"{len(selector_test)}/{len(test)}"
    )

    print()
    print("DOWNSTREAM ROUTED TASK PERFORMANCE")

    print(
        f"GTE four-outcome Sample-Avg     : "
        f"{gte_sample:.4f}%"
    )

    print(
        f"Tiny direct selector      : "
        f"{tiny_direct['direct_selector_sample_avg_pct']:.4f}%"
    )

    print(
        f"GTE four-outcome - tiny         : "
        f"{gte_sample - tiny_direct['direct_selector_sample_avg_pct']:+.4f} pp"
    )

    print(
        f"Independent-head baseline: "
        f"{independent_micro['microencoder_sample_avg_pct']:.4f}%"
    )

    print(
        f"Dataset Prior             : "
        f"{prior['dataset_prior_sample_avg_pct']:.4f}%"
    )

    print(
        f"TF-IDF                    : "
        f"{old['tfidf_sample_avg_pct']:.4f}%"
    )

    print(
        f"AvengersPro               : "
        f"{old['avengerspro_sample_avg_pct']:.4f}%"
    )

    print(
        f"Oracle                    : "
        f"{oracle_sample:.4f}%"
    )

    print()

    print(
        f"GTE four-outcome Dataset-Avg           : "
        f"{gte_dataset:.4f}%"
    )

    print(
        f"Tiny direct Dataset-Avg   : "
        f"{tiny_direct['direct_selector_dataset_avg_pct']:.4f}%"
    )

    print(
        f"AvengersPro Dataset-Avg   : "
        f"{old['avengerspro_dataset_avg_pct']:.4f}%"
    )

    print(
        f"Oracle Dataset-Avg        : "
        f"{oracle_dataset:.4f}%"
    )

    print()

    print(
        f"Classifier parameters     : "
        f"{classifier_parameters:,}"
    )

    print(
        f"Training time             : "
        f"{train_seconds:.3f}s"
    )

    print(
        f"Inference time            : "
        f"{inference_seconds:.3f}s"
    )

    print("-" * 90)

    return row, by_dataset


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
    )

    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=2000,
    )

    args = parser.parse_args()

    embedding_cache = (
        load_embedding_cache()
    )

    baseline_exact = pd.read_csv(
        OUT
        / "exact_multiseed_router_comparison.csv"
    )

    baseline_prior = pd.read_csv(
        OUT
        / "dataset_prior_multiseed_comparison.csv"
    )

    baseline_micro = pd.read_csv(
        OUT
        / "microencoder_multiseed.csv"
    )

    baseline_direct = pd.read_csv(
        OUT
        / "direct_selector_multiseed.csv"
    )

    summary_rows = []
    all_dataset_rows = []

    overall_start = time.perf_counter()

    for seed in args.seeds:
        row, dataset_df = train_one_seed(
            seed,
            args,
            embedding_cache,
            baseline_exact,
            baseline_prior,
            baseline_micro,
            baseline_direct,
        )

        summary_rows.append(row)

        dataset_df = dataset_df.copy()
        dataset_df.insert(
            0,
            "seed",
            seed,
        )

        all_dataset_rows.append(
            dataset_df
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        OUT
        / "gte_four_outcome_selector_multiseed.csv",
        index=False,
    )

    pd.concat(
        all_dataset_rows,
        ignore_index=True,
    ).to_csv(
        OUT
        / "gte_four_outcome_selector_by_dataset_all.csv",
        index=False,
    )

    overall_seconds = (
        time.perf_counter()
        - overall_start
    )

    print()
    print("=" * 95)
    print("GTE FOUR-OUTCOME SELECTOR CONTROLLED COMPARISON")
    print("=" * 95)

    print(
        f"Seeds completed           : "
        f"{len(summary)}"
    )

    print()
    print("ROUTER MODEL-SELECTION PERFORMANCE")

    print(
        f"Macro-F1 [primary]        : "
        f"{summary['router_macro_f1'].mean():.4f}"
    )

    print(
        f"Tiny direct Macro-F1      : "
        f"{summary['tiny_direct_router_macro_f1'].mean():.4f}"
    )

    print(
        f"GTE - tiny F1             : "
        f"{summary['gte_minus_tiny_macro_f1'].mean():+.4f}"
    )

    print(
        f"Balanced Accuracy         : "
        f"{summary['router_balanced_accuracy'].mean():.4f}"
    )

    print(
        f"AUC                       : "
        f"{summary['router_auc'].mean():.4f}"
    )

    print(
        f"MCC                       : "
        f"{summary['router_mcc'].mean():.4f}"
    )

    print()
    print("DOWNSTREAM ROUTED TASK PERFORMANCE")

    print(
        f"GTE four-outcome Sample-Avg     : "
        f"{summary['gte_sample_avg_pct'].mean():.4f}%"
    )

    print(
        f"Tiny direct selector      : "
        f"{summary['tiny_direct_sample_avg_pct'].mean():.4f}%"
    )

    print(
        f"GTE four-outcome - tiny         : "
        f"{summary['gte_minus_tiny_sample_pp'].mean():+.4f} pp"
    )

    print(
        f"Dataset Prior             : "
        f"{summary['dataset_prior_sample_avg_pct'].mean():.4f}%"
    )

    print(
        f"TF-IDF                    : "
        f"{summary['tfidf_sample_avg_pct'].mean():.4f}%"
    )

    print(
        f"AvengersPro               : "
        f"{summary['avengerspro_sample_avg_pct'].mean():.4f}%"
    )

    print(
        f"Oracle                    : "
        f"{summary['oracle_sample_avg_pct'].mean():.4f}%"
    )

    print()

    print(
        f"GTE four-outcome Dataset-Avg           : "
        f"{summary['gte_dataset_avg_pct'].mean():.4f}%"
    )

    print(
        f"Tiny direct Dataset-Avg   : "
        f"{summary['tiny_direct_dataset_avg_pct'].mean():.4f}%"
    )

    print(
        f"AvengersPro Dataset-Avg   : "
        f"{summary['avengerspro_dataset_avg_pct'].mean():.4f}%"
    )

    print(
        f"Oracle Dataset-Avg        : "
        f"{summary['oracle_dataset_avg_pct'].mean():.4f}%"
    )

    print()

    print(
        f"Total wall time           : "
        f"{overall_seconds:.2f}s"
    )

    print()
    print("Saved:")
    print(
        OUT
        / "gte_four_outcome_selector_multiseed.csv"
    )
    print(
        OUT
        / "gte_four_outcome_selector_by_dataset_all.csv"
    )


if __name__ == "__main__":
    main()
