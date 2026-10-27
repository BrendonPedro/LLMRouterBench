import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import Normalizer

import train_minilm_routing_score_distillation as base


OUT = Path("local_experiments/results")
OUT.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=300,
    )

    args = parser.parse_args()

    print("Loading MiniLM embedding cache...")

    minilm_cache = base.load_cache(
        base.MINILM_CACHE,
        base.MINILM_MODEL,
    )

    baseline_minilm = pd.read_csv(
        OUT / "minilm_direct_selector_multiseed.csv"
    )

    baseline_nonlinear_distill = pd.read_csv(
        OUT
        / "minilm_nonlinear_score_distillation_multiseed.csv"
    )

    rows = []
    dataset_rows = []

    total_start = time.perf_counter()

    for seed in args.seeds:
        print()
        print("=" * 90)
        print(
            f"MINILM NONLINEAR HARD-LABEL "
            f"SEED {seed}"
        )
        print("=" * 90)

        data_dir = Path(
            f"baselines/AvengersPro/data/"
            f"small_models_seed_{seed}/"
            f"seed{seed}_split0.7"
        )

        train = base.load_jsonl(
            data_dir / "train.jsonl"
        )

        test = base.load_jsonl(
            data_dir / "test.jsonl"
        )

        # --------------------------------------------------
        # REAL BENCHMARK WINNER LABELS
        # --------------------------------------------------

        train_mask = (
            train["deepseek_score"]
            != train["qwen_score"]
        ).to_numpy()

        # Keep the same sign convention as teacher
        # routing scores:
        #
        # -1 -> DeepSeek
        # +1 -> Qwen
        y_train = np.where(
            train.loc[
                train_mask,
                "qwen_score",
            ].to_numpy()
            >
            train.loc[
                train_mask,
                "deepseek_score",
            ].to_numpy(),
            1.0,
            -1.0,
        )

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

        train_queries = train[
            "query"
        ].tolist()

        test_queries = test[
            "query"
        ].tolist()

        # --------------------------------------------------
        # FROZEN MINILM FEATURES
        # --------------------------------------------------

        X_train = base.embedding_matrix(
            train_queries,
            minilm_cache,
        )

        X_test = base.embedding_matrix(
            test_queries,
            minilm_cache,
        )

        normalizer = Normalizer(norm="l2")

        X_train = normalizer.fit_transform(
            X_train
        )

        X_test = normalizer.transform(
            X_test
        )

        # --------------------------------------------------
        # SAME NONLINEAR ARCHITECTURE
        #
        # 384-D MiniLM
        #      ->
        # 64-unit ReLU
        #      ->
        # scalar
        #
        # Difference from distilled model:
        # only real non-tie benchmark labels are used.
        # --------------------------------------------------

        student = MLPRegressor(
            hidden_layer_sizes=(
                args.hidden_size,
            ),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=args.max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=seed,
        )

        training_start = time.perf_counter()

        student.fit(
            X_train[train_mask],
            y_train,
        )

        training_seconds = (
            time.perf_counter()
            - training_start
        )

        inference_start = time.perf_counter()

        student_score = student.predict(
            X_test
        )

        student_class = (
            student_score > 0
        ).astype(int)

        inference_seconds = (
            time.perf_counter()
            - inference_start
        )

        # --------------------------------------------------
        # ROUTER CLASSIFICATION METRICS
        # --------------------------------------------------

        y_pred = student_class[
            test_mask
        ]

        y_score = student_score[
            test_mask
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
            y_score,
        )

        mcc = matthews_corrcoef(
            y_true,
            y_pred,
        )

        # --------------------------------------------------
        # DOWNSTREAM ROUTED PERFORMANCE
        # --------------------------------------------------

        choices = np.where(
            student_class == 1,
            base.QWEN,
            base.DS,
        )

        routed_score = np.where(
            student_class == 1,
            test["qwen_score"].to_numpy(),
            test["deepseek_score"].to_numpy(),
        )

        sample_avg = (
            routed_score.mean()
            * 100
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

        student_parameters = (
            384 * args.hidden_size
            + args.hidden_size
            + args.hidden_size
            + 1
        )

        mini_row = baseline_minilm[
            baseline_minilm["seed"] == seed
        ].iloc[0]

        distill_row = (
            baseline_nonlinear_distill[
                baseline_nonlinear_distill[
                    "seed"
                ] == seed
            ].iloc[0]
        )

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
            f"Plain MiniLM F1           : "
            f"{mini_row['router_macro_f1']:.4f}"
        )
        print(
            f"Nonlinear hard-label F1   : "
            f"{macro_f1:.4f}"
        )
        print(
            f"Nonlinear distilled F1    : "
            f"{distill_row['router_macro_f1']:.4f}"
        )

        print()
        print("NONLINEAR STUDENT")
        print(
            f"Training examples         : "
            f"{train_mask.sum()}"
        )
        print(
            f"Hidden size               : "
            f"{args.hidden_size}"
        )
        print(
            f"Student parameters        : "
            f"{student_parameters:,}"
        )
        print(
            f"Epochs used               : "
            f"{student.n_iter_}"
        )
        print(
            f"Training time             : "
            f"{training_seconds:.4f}s"
        )
        print(
            f"Inference time            : "
            f"{inference_seconds:.4f}s"
        )
        print("-" * 75)

        rows.append({
            "seed": seed,
            "student_encoder":
                base.MINILM_MODEL,
            "student_model":
                "mlp_regressor",
            "supervision":
                "hard_non_tie_labels_only",
            "hidden_size":
                args.hidden_size,
            "student_parameters":
                student_parameters,
            "train_examples":
                int(train_mask.sum()),
            "epochs_used":
                student.n_iter_,
            "student_training_seconds":
                training_seconds,
            "student_inference_seconds":
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
            "plain_minilm_macro_f1":
                mini_row[
                    "router_macro_f1"
                ],
            "nonlinear_distilled_macro_f1":
                distill_row[
                    "router_macro_f1"
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
                        == base.QWEN
                    ).mean()
                    * 100,
                "deepseek_selection_pct":
                    (
                        choices[mask]
                        == base.DS
                    ).mean()
                    * 100,
            })

    results = pd.DataFrame(rows)
    by_dataset = pd.DataFrame(
        dataset_rows
    )

    results.to_csv(
        OUT
        / "minilm_nonlinear_hard_label_multiseed.csv",
        index=False,
    )

    by_dataset.to_csv(
        OUT
        / "minilm_nonlinear_hard_label_by_dataset.csv",
        index=False,
    )

    print()
    print("=" * 90)
    print("MINILM NONLINEAR HARD-LABEL SUMMARY")
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
        f"Plain MiniLM F1           : "
        f"{results['plain_minilm_macro_f1'].mean():.4f}"
    )
    print(
        f"Nonlinear hard-label F1   : "
        f"{results['router_macro_f1'].mean():.4f}"
    )
    print(
        f"Nonlinear distilled F1    : "
        f"{results['nonlinear_distilled_macro_f1'].mean():.4f}"
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
        / "minilm_nonlinear_hard_label_multiseed.csv"
    )
    print(
        OUT
        / "minilm_nonlinear_hard_label_by_dataset.csv"
    )


if __name__ == "__main__":
    main()
