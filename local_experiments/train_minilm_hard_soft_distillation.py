import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

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

import train_minilm_routing_score_distillation as base


OUT = Path("local_experiments/results")
OUT.mkdir(parents=True, exist_ok=True)


def sigmoid(x):
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
    )

    parser.add_argument(
        "--student-alpha",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--teacher-C",
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

    gte_cache = base.load_cache(
        base.GTE_CACHE,
        base.GTE_MODEL,
    )

    minilm_cache = base.load_cache(
        base.MINILM_CACHE,
        base.MINILM_MODEL,
    )

    baseline_gte = pd.read_csv(
        OUT / "gte_direct_selector_multiseed.csv"
    )

    baseline_minilm = pd.read_csv(
        OUT / "minilm_direct_selector_multiseed.csv"
    )

    baseline_score_distill = pd.read_csv(
        OUT
        / "minilm_routing_score_distillation_multiseed.csv"
    )

    rows = []
    dataset_rows = []

    total_start = time.perf_counter()

    for seed in args.seeds:
        print()
        print("=" * 90)
        print(
            f"MINILM HARD + SOFT DISTILLATION "
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
        # TRUE WINNER LABELS
        # --------------------------------------------------

        train_mask = (
            train["deepseek_score"]
            != train["qwen_score"]
        ).to_numpy()

        y_train_hard = (
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

        train_queries = train[
            "query"
        ].tolist()

        test_queries = test[
            "query"
        ].tolist()

        # --------------------------------------------------
        # GTE TEACHER
        #
        # Same teacher as routing-score distillation:
        # train only on real non-tie winner labels.
        # --------------------------------------------------

        X_gte_train = base.embedding_matrix(
            train_queries,
            gte_cache,
        )

        X_gte_test = base.embedding_matrix(
            test_queries,
            gte_cache,
        )

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

        teacher = LogisticRegression(
            C=args.teacher_C,
            penalty="l2",
            solver="liblinear",
            max_iter=args.max_iter,
            random_state=seed,
            class_weight=None,
        )

        teacher.fit(
            X_gte_train[train_mask],
            y_train_hard,
        )

        teacher_train_score = (
            teacher.decision_function(
                X_gte_train
            )
        )

        teacher_test_score = (
            teacher.decision_function(
                X_gte_test
            )
        )

        # Put soft teacher information onto the
        # same 0..1 scale as the hard labels.
        teacher_train_prob = sigmoid(
            teacher_train_score
        )

        teacher_test_prob = sigmoid(
            teacher_test_score
        )

        teacher_test_class = (
            teacher_test_prob >= 0.5
        ).astype(int)

        teacher_f1 = f1_score(
            y_true,
            teacher_test_class[
                test_mask
            ],
            average="macro",
            zero_division=0,
        )

        # --------------------------------------------------
        # MINILM STUDENT FEATURES
        # --------------------------------------------------

        X_mini_train = base.embedding_matrix(
            train_queries,
            minilm_cache,
        )

        X_mini_test = base.embedding_matrix(
            test_queries,
            minilm_cache,
        )

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
        # HARD + SOFT TRAINING SET
        #
        # SOFT:
        # all 8,021 training queries receive the
        # GTE teacher probability.
        #
        # HARD:
        # non-tie queries additionally receive the
        # real benchmark winner label.
        #
        # Total soft loss weight = 50%
        # Total hard loss weight = 50%.
        #
        # We normalize sample weights so their
        # overall mean remains 1.0. This keeps the
        # Ridge alpha comparable to the previous
        # experiment.
        # --------------------------------------------------

        X_soft = X_mini_train
        y_soft = teacher_train_prob

        X_hard = X_mini_train[
            train_mask
        ]

        y_hard = (
            y_train_hard.astype(
                np.float64
            )
        )

        X_student = np.vstack([
            X_soft,
            X_hard,
        ])

        y_student = np.concatenate([
            y_soft,
            y_hard,
        ])

        n_soft = len(X_soft)
        n_hard = len(X_hard)
        n_total = n_soft + n_hard

        soft_weight = (
            0.5
            * n_total
            / n_soft
        )

        hard_weight = (
            0.5
            * n_total
            / n_hard
        )

        sample_weight = np.concatenate([
            np.full(
                n_soft,
                soft_weight,
                dtype=np.float64,
            ),
            np.full(
                n_hard,
                hard_weight,
                dtype=np.float64,
            ),
        ])

        print(
            f"Soft examples              : "
            f"{n_soft}"
        )
        print(
            f"Hard examples              : "
            f"{n_hard}"
        )
        print(
            f"Soft total weight          : "
            f"{sample_weight[:n_soft].sum():.2f}"
        )
        print(
            f"Hard total weight          : "
            f"{sample_weight[n_soft:].sum():.2f}"
        )

        # --------------------------------------------------
        # ONE 385-PARAMETER STUDENT HEAD
        # --------------------------------------------------

        student = Ridge(
            alpha=args.student_alpha,
            fit_intercept=True,
        )

        training_start = time.perf_counter()

        student.fit(
            X_student,
            y_student,
            sample_weight=sample_weight,
        )

        student_training_seconds = (
            time.perf_counter()
            - training_start
        )

        inference_start = time.perf_counter()

        student_test_score = (
            student.predict(
                X_mini_test
            )
        )

        student_test_class = (
            student_test_score >= 0.5
        ).astype(int)

        inference_seconds = (
            time.perf_counter()
            - inference_start
        )

        # --------------------------------------------------
        # TEACHER-STUDENT TRANSFER
        # --------------------------------------------------

        probability_correlation = np.corrcoef(
            student_test_score,
            teacher_test_prob,
        )[0, 1]

        teacher_sign_agreement = np.mean(
            student_test_class
            == teacher_test_class
        )

        # --------------------------------------------------
        # ROUTER CLASSIFICATION
        # --------------------------------------------------

        y_pred = student_test_class[
            test_mask
        ]

        y_score = student_test_score[
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
        # DOWNSTREAM TASK PERFORMANCE
        # --------------------------------------------------

        choices = np.where(
            student_test_class == 1,
            base.QWEN,
            base.DS,
        )

        routed_score = np.where(
            student_test_class == 1,
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
            student.coef_.size
            + 1
        )

        gte_row = baseline_gte[
            baseline_gte["seed"] == seed
        ].iloc[0]

        mini_row = baseline_minilm[
            baseline_minilm["seed"] == seed
        ].iloc[0]

        score_row = baseline_score_distill[
            baseline_score_distill[
                "seed"
            ] == seed
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
        print("TEACHER-STUDENT TRANSFER")
        print(
            f"Probability correlation   : "
            f"{probability_correlation:.4f}"
        )
        print(
            f"Teacher choice agreement  : "
            f"{teacher_sign_agreement:.4f}"
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
            f"Score distillation F1     : "
            f"{score_row['router_macro_f1']:.4f}"
        )
        print(
            f"Hard+soft student F1      : "
            f"{macro_f1:.4f}"
        )
        print(
            f"GTE teacher F1            : "
            f"{teacher_f1:.4f}"
        )

        print()
        print("LIGHTWEIGHT STUDENT")
        print(
            f"Student parameters        : "
            f"{student_parameters:,}"
        )
        print(
            f"Training time             : "
            f"{student_training_seconds:.4f}s"
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
            "teacher_encoder":
                base.GTE_MODEL,
            "training_method":
                "equal_hard_soft_probability_distillation",
            "soft_loss_weight":
                0.5,
            "hard_loss_weight":
                0.5,
            "student_alpha":
                args.student_alpha,
            "student_parameters":
                student_parameters,
            "student_training_seconds":
                student_training_seconds,
            "student_inference_seconds":
                inference_seconds,
            "teacher_student_probability_correlation":
                probability_correlation,
            "teacher_student_choice_agreement":
                teacher_sign_agreement,
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
            "score_distillation_macro_f1":
                score_row[
                    "router_macro_f1"
                ],
            "teacher_macro_f1":
                teacher_f1,
            "full_gte_macro_f1":
                gte_row[
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
        / "minilm_hard_soft_distillation_multiseed.csv",
        index=False,
    )

    by_dataset.to_csv(
        OUT
        / "minilm_hard_soft_distillation_by_dataset.csv",
        index=False,
    )

    print()
    print("=" * 90)
    print("MINILM HARD + SOFT DISTILLATION SUMMARY")
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
        f"Teacher probability corr  : "
        f"{results['teacher_student_probability_correlation'].mean():.4f}"
    )
    print(
        f"Teacher choice agreement  : "
        f"{results['teacher_student_choice_agreement'].mean():.4f}"
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
        f"Score distillation F1     : "
        f"{results['score_distillation_macro_f1'].mean():.4f}"
    )
    print(
        f"Hard+soft student F1      : "
        f"{results['router_macro_f1'].mean():.4f}"
    )
    print(
        f"GTE teacher F1            : "
        f"{results['teacher_macro_f1'].mean():.4f}"
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
        / "minilm_hard_soft_distillation_multiseed.csv"
    )
    print(
        OUT
        / "minilm_hard_soft_distillation_by_dataset.csv"
    )


if __name__ == "__main__":
    main()
