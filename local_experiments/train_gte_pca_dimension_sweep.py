import argparse
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
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
    print("Loading GTE cache:", CACHE)

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

    cache = {
        text: np.asarray(
            json.loads(embedding),
            dtype=np.float32,
        )
        for text, embedding in rows
    }

    if not cache:
        raise RuntimeError(
            "No GTE embeddings found."
        )

    first = next(iter(cache.values()))

    print("Cached embeddings :", len(cache))
    print("Full dimension    :", len(first))

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
        "--dimensions",
        type=int,
        nargs="+",
        default=[384, 256, 128, 64, 32],
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

    dimensions = sorted(
        set(args.dimensions),
        reverse=True,
    )

    max_dim = max(dimensions)

    cache = load_embedding_cache()

    baseline_gte = pd.read_csv(
        OUT / "gte_direct_selector_multiseed.csv"
    )

    baseline_minilm = pd.read_csv(
        OUT / "minilm_direct_selector_multiseed.csv"
    )

    result_rows = []
    dataset_rows = []

    total_start = time.perf_counter()

    for seed in args.seeds:
        print()
        print("=" * 90)
        print(f"GTE PCA DIMENSION SWEEP SEED {seed}")
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

        X_train_all = embedding_matrix(
            train["query"].tolist(),
            cache,
        )

        X_test = embedding_matrix(
            test["query"].tolist(),
            cache,
        )

        # Match the original GTE baseline input
        # before learning the compression.
        pre_normalizer = Normalizer(norm="l2")

        X_train_all = (
            pre_normalizer.fit_transform(
                X_train_all
            )
        )

        X_test = pre_normalizer.transform(
            X_test
        )

        print(
            f"Train queries       : {len(train)}"
        )
        print(
            f"Non-tie train       : {train_mask.sum()}"
        )
        print(
            f"Non-tie test        : {test_mask.sum()}"
        )
        print(
            f"Fitting PCA to      : {max_dim} dimensions"
        )

        pca_start = time.perf_counter()

        pca = PCA(
            n_components=max_dim,
            svd_solver="randomized",
            random_state=seed,
        )

        X_train_pca_full = pca.fit_transform(
            X_train_all
        )

        X_test_pca_full = pca.transform(
            X_test
        )

        pca_seconds = (
            time.perf_counter() - pca_start
        )

        print(
            f"PCA fit/transform   : "
            f"{pca_seconds:.3f}s"
        )

        gte_row = baseline_gte[
            baseline_gte["seed"] == seed
        ].iloc[0]

        minilm_row = baseline_minilm[
            baseline_minilm["seed"] == seed
        ].iloc[0]

        for dim in dimensions:
            print()
            print("-" * 70)
            print(f"PCA DIMENSION {dim}")

            X_train = (
                X_train_pca_full[
                    train_mask,
                    :dim,
                ]
            )

            X_test_dim = (
                X_test_pca_full[:, :dim]
            )

            # Preserve the L2-normalized geometry
            # used by the original GTE selector.
            post_normalizer = Normalizer(norm="l2")

            X_train = (
                post_normalizer.fit_transform(
                    X_train
                )
            )

            X_test_dim = (
                post_normalizer.transform(
                    X_test_dim
                )
            )

            classifier = LogisticRegression(
                C=args.C,
                penalty="l2",
                solver="liblinear",
                max_iter=args.max_iter,
                random_state=seed,
                class_weight=None,
            )

            train_start = time.perf_counter()

            classifier.fit(
                X_train,
                y_train,
            )

            training_seconds = (
                time.perf_counter()
                - train_start
            )

            inference_start = time.perf_counter()

            selector_class = classifier.predict(
                X_test_dim
            )

            probabilities = (
                classifier.predict_proba(
                    X_test_dim
                )
            )

            inference_seconds = (
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

            temp["routed_score"] = routed_score

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

            explained = (
                pca.explained_variance_ratio_[
                    :dim
                ].sum()
            )

            print(
                f"Macro-F1            : "
                f"{macro_f1:.4f}"
            )
            print(
                f"Balanced Accuracy   : "
                f"{balanced:.4f}"
            )
            print(
                f"AUC                 : "
                f"{auc:.4f}"
            )
            print(
                f"Sample Avg          : "
                f"{sample_avg:.4f}%"
            )
            print(
                f"Dataset Avg         : "
                f"{dataset_avg:.4f}%"
            )
            print(
                f"Explained variance  : "
                f"{explained * 100:.2f}%"
            )

            result_rows.append({
                "seed": seed,
                "pca_dimension": dim,
                "explained_variance_ratio":
                    explained,
                "classifier_parameters":
                    dim + 1,
                "pca_seconds":
                    pca_seconds,
                "training_seconds":
                    training_seconds,
                "inference_seconds":
                    inference_seconds,
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
                "full_gte_macro_f1":
                    gte_row[
                        "router_macro_f1"
                    ],
                "full_gte_sample_avg_pct":
                    gte_row[
                        "gte_sample_avg_pct"
                    ],
                "full_gte_dataset_avg_pct":
                    gte_row[
                        "gte_dataset_avg_pct"
                    ],
                "minilm_macro_f1":
                    minilm_row[
                        "router_macro_f1"
                    ],
                "minilm_sample_avg_pct":
                    minilm_row[
                        "minilm_sample_avg_pct"
                    ],
                "minilm_dataset_avg_pct":
                    minilm_row[
                        "minilm_dataset_avg_pct"
                    ],
            })

            choices = np.where(
                selector_class == 1,
                QWEN,
                DS,
            )

            for dataset, score in (
                dataset_scores.items()
            ):
                mask = (
                    test["dataset"]
                    == dataset
                ).to_numpy()

                dataset_rows.append({
                    "seed": seed,
                    "pca_dimension": dim,
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

    results = pd.DataFrame(result_rows)

    by_dataset = pd.DataFrame(
        dataset_rows
    )

    results.to_csv(
        OUT / "gte_pca_dimension_sweep.csv",
        index=False,
    )

    by_dataset.to_csv(
        OUT
        / "gte_pca_dimension_sweep_by_dataset.csv",
        index=False,
    )

    print()
    print("=" * 90)
    print("GTE PCA DIMENSION SWEEP SUMMARY")
    print("=" * 90)

    summary = (
        results.groupby("pca_dimension")
        .agg(
            macro_f1=(
                "router_macro_f1",
                "mean",
            ),
            balanced_accuracy=(
                "router_balanced_accuracy",
                "mean",
            ),
            auc=(
                "router_auc",
                "mean",
            ),
            mcc=(
                "router_mcc",
                "mean",
            ),
            sample_avg_pct=(
                "sample_avg_pct",
                "mean",
            ),
            dataset_avg_pct=(
                "dataset_avg_pct",
                "mean",
            ),
            explained_variance_pct=(
                "explained_variance_ratio",
                lambda x: x.mean() * 100,
            ),
        )
        .sort_index(
            ascending=False
        )
    )

    print(summary.to_string())

    print()
    print("REFERENCE")
    print(
        "Full GTE Macro-F1 :",
        f"{baseline_gte['router_macro_f1'].mean():.4f}",
    )
    print(
        "MiniLM Macro-F1   :",
        f"{baseline_minilm['router_macro_f1'].mean():.4f}",
    )

    print()
    print(
        "Total wall time   :",
        f"{time.perf_counter() - total_start:.2f}s",
    )

    print()
    print("Saved:")
    print(
        OUT / "gte_pca_dimension_sweep.csv"
    )
    print(
        OUT
        / "gte_pca_dimension_sweep_by_dataset.csv"
    )


if __name__ == "__main__":
    main()
