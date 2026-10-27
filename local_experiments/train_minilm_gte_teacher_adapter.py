import argparse
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
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

GTE_MODEL = "Alibaba-NLP/gte-Qwen2-7B-instruct"
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

GTE_CACHE = Path(
    "baselines/AvengersPro/cache/"
    "gte_qwen2_7b_embeddings.sqlite"
)

MINILM_CACHE = Path(
    "local_experiments/cache/"
    "minilm_l6_v2_embeddings.sqlite"
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
                "query": x["query"],
                "deepseek_score": float(
                    x["records"][DS]
                ),
                "qwen_score": float(
                    x["records"][QWEN]
                ),
            })

    return pd.DataFrame(rows)


def load_cache(path, model_name):
    conn = sqlite3.connect(path)

    rows = conn.execute(
        """
        SELECT text, embedding
        FROM embeddings
        WHERE model = ?
        """,
        (model_name,),
    ).fetchall()

    conn.close()

    cache = {
        text: np.asarray(
            json.loads(embedding),
            dtype=np.float32,
        )
        for text, embedding in rows
    }

    if not cache:
        raise RuntimeError(
            f"No embeddings found for {model_name}"
        )

    first = next(iter(cache.values()))

    print(
        f"{model_name}: "
        f"{len(cache)} embeddings, "
        f"{len(first)} dimensions"
    )

    return cache


def embedding_matrix(queries, cache):
    missing = [
        q for q in queries
        if q not in cache
    ]

    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} embeddings."
        )

    return np.stack(
        [cache[q] for q in queries],
        axis=0,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
    )

    parser.add_argument(
        "--target-dim",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--adapter-alpha",
        type=float,
        default=1.0,
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

    print("Loading embedding caches...")

    gte_cache = load_cache(
        GTE_CACHE,
        GTE_MODEL,
    )

    minilm_cache = load_cache(
        MINILM_CACHE,
        MINILM_MODEL,
    )

    baseline_gte = pd.read_csv(
        OUT / "gte_direct_selector_multiseed.csv"
    )

    baseline_minilm = pd.read_csv(
        OUT / "minilm_direct_selector_multiseed.csv"
    )

    baseline_pca = pd.read_csv(
        OUT / "gte_pca_dimension_sweep.csv"
    )

    rows = []
    dataset_rows = []

    total_start = time.perf_counter()

    for seed in args.seeds:
        print()
        print("=" * 90)
        print(
            f"MINILM -> GTE TEACHER ADAPTER "
            f"SEED {seed}"
        )
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

        # --------------------------------------------------
        # ROUTER LABELS
        # --------------------------------------------------

        train_mask = (
            train["deepseek_score"]
            != train["qwen_score"]
        ).to_numpy()

        y_train = (
            train.loc[
                train_mask,
                "qwen_score",
            ]
            > train.loc[
                train_mask,
                "deepseek_score",
            ]
        ).astype(int).to_numpy()

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

        test_mask = (
            test["selector_target"] != -1
        ).to_numpy()

        y_true = (
            test.loc[
                test_mask,
                "selector_target",
            ]
            .astype(int)
            .to_numpy()
        )

        # --------------------------------------------------
        # LOAD BOTH REPRESENTATIONS
        # --------------------------------------------------

        queries_train = train[
            "query"
        ].tolist()

        queries_test = test[
            "query"
        ].tolist()

        X_gte_train = embedding_matrix(
            queries_train,
            gte_cache,
        )

        X_gte_test = embedding_matrix(
            queries_test,
            gte_cache,
        )

        X_mini_train = embedding_matrix(
            queries_train,
            minilm_cache,
        )

        X_mini_test = embedding_matrix(
            queries_test,
            minilm_cache,
        )

        # --------------------------------------------------
        # BUILD 256-D GTE TEACHER SPACE
        #
        # Fit normalization and PCA on TRAIN only.
        # Test data never influences the teacher space.
        # --------------------------------------------------

        gte_norm = Normalizer(norm="l2")

        X_gte_train = (
            gte_norm.fit_transform(
                X_gte_train
            )
        )

        X_gte_test = (
            gte_norm.transform(
                X_gte_test
            )
        )

        pca = PCA(
            n_components=args.target_dim,
            svd_solver="randomized",
            random_state=seed,
        )

        teacher_train = pca.fit_transform(
            X_gte_train
        )

        teacher_test = pca.transform(
            X_gte_test
        )

        teacher_norm = Normalizer(norm="l2")

        teacher_train = (
            teacher_norm.fit_transform(
                teacher_train
            )
        )

        teacher_test = (
            teacher_norm.transform(
                teacher_test
            )
        )

        # --------------------------------------------------
        # MINILM STUDENT INPUT
        # --------------------------------------------------

        mini_norm = Normalizer(norm="l2")

        X_mini_train = (
            mini_norm.fit_transform(
                X_mini_train
            )
        )

        X_mini_test = (
            mini_norm.transform(
                X_mini_test
            )
        )

        # --------------------------------------------------
        # TEACHER ADAPTER
        #
        # Learn from ALL training queries.
        # No model-winner labels are used here.
        #
        # MiniLM 384-D -> GTE-PCA 256-D
        # --------------------------------------------------

        adapter = Ridge(
            alpha=args.adapter_alpha,
            fit_intercept=True,
        )

        adapter_start = time.perf_counter()

        adapter.fit(
            X_mini_train,
            teacher_train,
        )

        adapted_train = adapter.predict(
            X_mini_train
        )

        adapted_test = adapter.predict(
            X_mini_test
        )

        adapter_seconds = (
            time.perf_counter()
            - adapter_start
        )

        adapted_norm = Normalizer(norm="l2")

        adapted_train = (
            adapted_norm.fit_transform(
                adapted_train
            )
        )

        adapted_test = (
            adapted_norm.transform(
                adapted_test
            )
        )

        # How closely does the student reproduce the
        # teacher representation on unseen test queries?
        teacher_cosine = np.mean(
            np.sum(
                adapted_test
                * teacher_test,
                axis=1,
            )
        )

        teacher_mse = np.mean(
            (
                adapted_test
                - teacher_test
            ) ** 2
        )

        # --------------------------------------------------
        # SAME DIRECT LOGISTIC ROUTER
        # --------------------------------------------------

        classifier = LogisticRegression(
            C=args.C,
            penalty="l2",
            solver="liblinear",
            max_iter=args.max_iter,
            random_state=seed,
            class_weight=None,
        )

        router_start = time.perf_counter()

        classifier.fit(
            adapted_train[train_mask],
            y_train,
        )

        router_training_seconds = (
            time.perf_counter()
            - router_start
        )

        inference_start = time.perf_counter()

        selector_class = classifier.predict(
            adapted_test
        )

        probabilities = (
            classifier.predict_proba(
                adapted_test
            )
        )

        router_inference_seconds = (
            time.perf_counter()
            - inference_start
        )

        y_pred = selector_class[
            test_mask
        ]

        y_prob = probabilities[
            test_mask,
            1,
        ]

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

        balanced = balanced_accuracy_score(
            y_true,
            y_pred,
        )

        auc = roc_auc_score(
            y_true,
            y_prob,
        )

        mcc = matthews_corrcoef(
            y_true,
            y_pred,
        )

        # --------------------------------------------------
        # DOWNSTREAM PERFORMANCE
        # --------------------------------------------------

        choices = np.where(
            selector_class == 1,
            QWEN,
            DS,
        )

        routed_score = np.where(
            selector_class == 1,
            test["qwen_score"].to_numpy(),
            test["deepseek_score"].to_numpy(),
        )

        sample_avg = (
            routed_score.mean() * 100
        )

        temp = test[
            ["dataset"]
        ].copy()

        temp["routed_score"] = (
            routed_score
        )

        dataset_scores = (
            temp.groupby("dataset")[
                "routed_score"
            ]
            .mean()
            * 100
        )

        dataset_avg = (
            dataset_scores.mean()
        )

        adapter_parameters = (
            adapter.coef_.size
            + adapter.intercept_.size
        )

        router_parameters = (
            classifier.coef_.size
            + classifier.intercept_.size
        )

        gte_row = baseline_gte[
            baseline_gte["seed"] == seed
        ].iloc[0]

        mini_row = baseline_minilm[
            baseline_minilm["seed"] == seed
        ].iloc[0]

        pca_row = baseline_pca[
            (
                baseline_pca["seed"]
                == seed
            )
            & (
                baseline_pca[
                    "pca_dimension"
                ]
                == args.target_dim
            )
        ].iloc[0]

        print()
        print("-" * 75)
        print("ROUTER MODEL-SELECTION PERFORMANCE")
        print(
            f"Macro-F1 [primary]        : "
            f"{macro_f1:.4f}"
        )
        print(
            f"Balanced Accuracy         : "
            f"{balanced:.4f}"
        )
        print(
            f"AUC                       : "
            f"{auc:.4f}"
        )
        print(
            f"MCC                       : "
            f"{mcc:.4f}"
        )

        print()
        print("REPRESENTATION TRANSFER")
        print(
            f"Teacher test cosine       : "
            f"{teacher_cosine:.4f}"
        )
        print(
            f"Teacher test MSE          : "
            f"{teacher_mse:.6f}"
        )

        print()
        print("DOWNSTREAM PERFORMANCE")
        print(
            f"Student Sample-Avg        : "
            f"{sample_avg:.4f}%"
        )
        print(
            f"Student Dataset-Avg       : "
            f"{dataset_avg:.4f}%"
        )

        print()
        print("SEED REFERENCES")
        print(
            f"MiniLM F1                 : "
            f"{mini_row['router_macro_f1']:.4f}"
        )
        print(
            f"GTE-PCA-{args.target_dim} F1          : "
            f"{pca_row['router_macro_f1']:.4f}"
        )
        print(
            f"Full GTE F1               : "
            f"{gte_row['router_macro_f1']:.4f}"
        )

        print()
        print("LIGHTWEIGHT HEAD")
        print(
            f"Adapter parameters        : "
            f"{adapter_parameters:,}"
        )
        print(
            f"Router parameters         : "
            f"{router_parameters:,}"
        )
        print(
            f"Total head parameters     : "
            f"{adapter_parameters + router_parameters:,}"
        )
        print(
            f"Adapter fit time          : "
            f"{adapter_seconds:.3f}s"
        )
        print("-" * 75)

        rows.append({
            "seed": seed,
            "student_encoder": MINILM_MODEL,
            "teacher_encoder": GTE_MODEL,
            "teacher_dimension":
                args.target_dim,
            "adapter":
                "ridge_linear",
            "adapter_alpha":
                args.adapter_alpha,
            "adapter_parameters":
                adapter_parameters,
            "router_parameters":
                router_parameters,
            "total_head_parameters":
                adapter_parameters
                + router_parameters,
            "adapter_seconds":
                adapter_seconds,
            "router_training_seconds":
                router_training_seconds,
            "router_inference_seconds":
                router_inference_seconds,
            "teacher_test_cosine":
                teacher_cosine,
            "teacher_test_mse":
                teacher_mse,
            "router_macro_f1":
                macro_f1,
            "router_macro_precision":
                macro_precision,
            "router_macro_recall":
                macro_recall,
            "router_balanced_accuracy":
                balanced,
            "router_auc":
                auc,
            "router_mcc":
                mcc,
            "sample_avg_pct":
                sample_avg,
            "dataset_avg_pct":
                dataset_avg,
            "minilm_macro_f1":
                mini_row[
                    "router_macro_f1"
                ],
            "gte_pca_macro_f1":
                pca_row[
                    "router_macro_f1"
                ],
            "full_gte_macro_f1":
                gte_row[
                    "router_macro_f1"
                ],
            "minilm_sample_avg_pct":
                mini_row[
                    "minilm_sample_avg_pct"
                ],
            "gte_pca_sample_avg_pct":
                pca_row[
                    "sample_avg_pct"
                ],
            "full_gte_sample_avg_pct":
                gte_row[
                    "gte_sample_avg_pct"
                ],
        })

        for dataset, score in (
            dataset_scores.items()
        ):
            mask = (
                test["dataset"]
                == dataset
            ).to_numpy()

            dataset_rows.append({
                "seed": seed,
                "dataset": dataset,
                "task_score_pct": score,
                "qwen_selection_pct":
                    (
                        choices[mask]
                        == QWEN
                    ).mean()
                    * 100,
                "deepseek_selection_pct":
                    (
                        choices[mask]
                        == DS
                    ).mean()
                    * 100,
            })

    results = pd.DataFrame(rows)

    by_dataset = pd.DataFrame(
        dataset_rows
    )

    results.to_csv(
        OUT
        / "minilm_gte_teacher_adapter_multiseed.csv",
        index=False,
    )

    by_dataset.to_csv(
        OUT
        / "minilm_gte_teacher_adapter_by_dataset.csv",
        index=False,
    )

    print()
    print("=" * 90)
    print("MINILM -> GTE TEACHER ADAPTER SUMMARY")
    print("=" * 90)
    print(
        f"Seeds completed           : "
        f"{len(results)}"
    )
    print(
        f"Macro-F1                  : "
        f"{results['router_macro_f1'].mean():.4f}"
    )
    print(
        f"Balanced Accuracy         : "
        f"{results['router_balanced_accuracy'].mean():.4f}"
    )
    print(
        f"AUC                       : "
        f"{results['router_auc'].mean():.4f}"
    )
    print(
        f"MCC                       : "
        f"{results['router_mcc'].mean():.4f}"
    )
    print(
        f"Teacher cosine            : "
        f"{results['teacher_test_cosine'].mean():.4f}"
    )
    print(
        f"Sample Avg                : "
        f"{results['sample_avg_pct'].mean():.4f}%"
    )
    print(
        f"Dataset Avg               : "
        f"{results['dataset_avg_pct'].mean():.4f}%"
    )

    print()
    print("REFERENCES")
    print(
        f"MiniLM F1                 : "
        f"{results['minilm_macro_f1'].mean():.4f}"
    )
    print(
        f"GTE-PCA-{args.target_dim} F1          : "
        f"{results['gte_pca_macro_f1'].mean():.4f}"
    )
    print(
        f"Full GTE F1               : "
        f"{results['full_gte_macro_f1'].mean():.4f}"
    )

    print()
    print(
        f"Total wall time           : "
        f"{time.perf_counter() - total_start:.2f}s"
    )

    print()
    print("Saved:")
    print(
        OUT
        / "minilm_gte_teacher_adapter_multiseed.csv"
    )
    print(
        OUT
        / "minilm_gte_teacher_adapter_by_dataset.csv"
    )


if __name__ == "__main__":
    main()
