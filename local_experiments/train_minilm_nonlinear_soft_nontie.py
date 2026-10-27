import argparse
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
        "--teacher-C",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=300,
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

    baseline_hard = pd.read_csv(
        OUT / "minilm_nonlinear_hard_label_multiseed.csv"
    )

    baseline_all_soft = pd.read_csv(
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
            f"MINILM NONLINEAR SOFT NON-TIE "
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
        # REAL NON-TIE WINNER LABELS
        # Used only to train the GTE teacher.
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

        train_queries = train[
            "query"
        ].tolist()

        test_queries = test[
            "query"
        ].tolist()

        # --------------------------------------------------
        # GTE TEACHER
        # Same teacher as the full soft-distillation run.
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
            max_iter=2000,
            random_state=seed,
            class_weight=None,
        )

        teacher.fit(
            X_gte_train[train_mask],
            y_train,
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

        teacher_test_class = (
            teacher_test_score > 0
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
        # FROZEN MINILM FEATURES
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
        # NONLINEAR STUDENT
        #
        # IMPORTANT CONTROL:
        #
        # Train only on the SAME non-tie examples used
        # by the hard-label nonlinear model.
        #
        # The only supervision change is:
        #
        # hard +/-1 winner labels
        #          versus
        # continuous GTE routing scores.
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
            X_mini_train[train_mask],
            teacher_train_score[train_mask],
        )

        training_seconds = (
            time.perf_counter()
            - training_start
        )

        inference_start = time.perf_counter()

        student_test_score = student.predict(
            X_mini_test
        )

        student_test_class = (
            student_test_score > 0
        ).astype(int)

        inference_seconds = (
            time.perf_counter()
            - inference_start
        )

        # --------------------------------------------------
        # TEACHER-STUDENT TRANSFER
        # --------------------------------------------------

        score_correlation = np.corrcoef(
            student_test_score,
            teacher_test_score,
        )[0, 1]

        choice_agreement = np.mean(
            student_test_class
            == teacher_test_class
        )

        # --------------------------------------------------
        # ROUTER CLASSIFICATION METRICS
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
        # DOWNSTREAM ROUTED PERFORMANCE
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
            384 * args.hidden_size
            + args.hidden_size
            + args.hidden_size
            + 1
        )

        hard_row = baseline_hard[
            baseline_hard["seed"] == seed
        ].iloc[0]

        all_soft_row = baseline_all_soft[
            baseline_all_soft[
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
            f"Score correlation         : "
            f"{score_correlation:.4f}"
        )
        print(
            f"Teacher choice agreement  : "
            f"{choice_agreement:.4f}"
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
            f"Hard-label MLP F1         : "
            f"{hard_row['router_macro_f1']:.4f}"
        )
        print(
            f"Soft non-tie MLP F1       : "
            f"{macro_f1:.4f}"
        )
        print(
            f"Soft all-query MLP F1     : "
            f"{all_soft_row['router_macro_f1']:.4f}"
        )
        print(
            f"GTE teacher F1            : "
            f"{teacher_f1:.4f}"
        )

        print()
        print("CONTROLLED TRAINING SET")
        print(
            f"Non-tie examples          : "
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
        print("-" * 75)

        rows.append({
            "seed": seed,
            "student_encoder":
                base.MINILM_MODEL,
            "teacher_encoder":
                base.GTE_MODEL,
            "student_model":
                "mlp_regressor",
            "supervision":
                "gte_soft_scores_non_tie_only",
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
            "teacher_student_score_correlation":
                score_correlation,
            "teacher_student_choice_agreement":
                choice_agreement,
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
            "hard_label_macro_f1":
                hard_row[
                    "router_macro_f1"
                ],
            "all_query_soft_macro_f1":
                all_soft_row[
                    "router_macro_f1"
                ],
            "teacher_macro_f1":
                teacher_f1,
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
        / "minilm_nonlinear_soft_nontie_multiseed.csv",
        index=False,
    )

    by_dataset.to_csv(
        OUT
        / "minilm_nonlinear_soft_nontie_by_dataset.csv",
        index=False,
    )

    print()
    print("=" * 90)
    print("MINILM NONLINEAR SOFT NON-TIE SUMMARY")
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
        f"Teacher score correlation : "
        f"{results['teacher_student_score_correlation'].mean():.4f}"
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
    print("KEY ABLATION")
    print(
        f"Hard labels, non-ties     : "
        f"{results['hard_label_macro_f1'].mean():.4f}"
    )
    print(
        f"Soft scores, non-ties     : "
        f"{results['router_macro_f1'].mean():.4f}"
    )
    print(
        f"Soft scores, all queries  : "
        f"{results['all_query_soft_macro_f1'].mean():.4f}"
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
        / "minilm_nonlinear_soft_nontie_multiseed.csv"
    )
    print(
        OUT
        / "minilm_nonlinear_soft_nontie_by_dataset.csv"
    )


if __name__ == "__main__":
    main()
