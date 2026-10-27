import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

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
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

import train_minilm_routing_score_distillation as base


OUT = Path("local_experiments/results")
OUT.mkdir(parents=True, exist_ok=True)


class MiniLMRoutingStudent(nn.Module):
    def __init__(self, model_name, hidden_size=64):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            model_name
        )

        encoder_dim = (
            self.encoder.config.hidden_size
        )

        # Same routing-head shape used in the
        # frozen nonlinear student:
        #
        # 384 -> 64 ReLU -> 1
        self.routing_head = nn.Sequential(
            nn.Linear(
                encoder_dim,
                hidden_size,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_size,
                1,
            ),
        )

    def forward(
        self,
        input_ids,
        attention_mask,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        token_embeddings = (
            outputs.last_hidden_state
        )

        mask = (
            attention_mask
            .unsqueeze(-1)
            .expand(
                token_embeddings.size()
            )
            .float()
        )

        pooled = (
            (
                token_embeddings
                * mask
            ).sum(dim=1)
            /
            mask.sum(dim=1).clamp(
                min=1e-9
            )
        )

        # Match the normalized MiniLM
        # representation used previously.
        pooled = F.normalize(
            pooled,
            p=2,
            dim=1,
        )

        return (
            self.routing_head(
                pooled
            )
            .squeeze(-1)
        )


def make_loader(
    tokenizer,
    queries,
    targets,
    max_length,
    batch_size,
    shuffle,
    seed,
):
    encoded = tokenizer(
        queries,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    target_tensor = torch.tensor(
        targets,
        dtype=torch.float32,
    )

    dataset = TensorDataset(
        encoded["input_ids"],
        encoded["attention_mask"],
        target_tensor,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


@torch.no_grad()
def predict_scores(
    model,
    loader,
    device,
):
    model.eval()

    scores = []

    for (
        input_ids,
        attention_mask,
        _,
    ) in loader:

        input_ids = (
            input_ids.to(device)
        )

        attention_mask = (
            attention_mask.to(device)
        )

        batch_scores = model(
            input_ids,
            attention_mask,
        )

        scores.append(
            batch_scores
            .cpu()
            .numpy()
        )

    return np.concatenate(
        scores
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
        "--epochs",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    torch.set_num_threads(
        args.threads
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    print(
        f"PyTorch threads: "
        f"{torch.get_num_threads()}"
    )

    print("Loading GTE embedding cache...")

    gte_cache = base.load_cache(
        base.GTE_CACHE,
        base.GTE_MODEL,
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            base.MINILM_MODEL
        )
    )

    frozen_reference = pd.read_csv(
        OUT
        / "minilm_nonlinear_score_distillation_multiseed.csv"
    )

    rows = []
    dataset_rows = []

    total_start = (
        time.perf_counter()
    )

    for seed in args.seeds:
        print()
        print("=" * 90)
        print(
            f"MINILM ENCODER SCORE "
            f"DISTILLATION SEED {seed}"
        )
        print("=" * 90)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                seed
            )

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

        train_mask = (
            train["deepseek_score"]
            != train["qwen_score"]
        ).to_numpy()

        y_teacher_train = (
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
            test["selector_target"]
            != -1
        ).to_numpy()

        y_true = (
            test.loc[
                test_mask,
                "selector_target",
            ]
            .astype(int)
            .to_numpy()
        )

        train_queries = (
            train["query"].tolist()
        )

        test_queries = (
            test["query"].tolist()
        )

        # --------------------------------------------------
        # GTE TEACHER
        # --------------------------------------------------

        X_gte_train = (
            base.embedding_matrix(
                train_queries,
                gte_cache,
            )
        )

        X_gte_test = (
            base.embedding_matrix(
                test_queries,
                gte_cache,
            )
        )

        normalizer = Normalizer(
            norm="l2"
        )

        X_gte_train = (
            normalizer.fit_transform(
                X_gte_train
            )
        )

        X_gte_test = (
            normalizer.transform(
                X_gte_test
            )
        )

        teacher = LogisticRegression(
            C=args.teacher_C,
            solver="liblinear",
            max_iter=2000,
            random_state=seed,
            class_weight=None,
        )

        teacher.fit(
            X_gte_train[
                train_mask
            ],
            y_teacher_train,
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
        # END-TO-END MINILM STUDENT
        #
        # All 8,021 training queries receive the
        # continuous GTE teacher routing score.
        # --------------------------------------------------

        train_loader = make_loader(
            tokenizer=tokenizer,
            queries=train_queries,
            targets=teacher_train_score,
            max_length=args.max_length,
            batch_size=args.batch_size,
            shuffle=True,
            seed=seed,
        )

        test_loader = make_loader(
            tokenizer=tokenizer,
            queries=test_queries,
            targets=np.zeros(
                len(test_queries),
                dtype=np.float32,
            ),
            max_length=args.max_length,
            batch_size=args.batch_size,
            shuffle=False,
            seed=seed,
        )

        model = MiniLMRoutingStudent(
            model_name=base.MINILM_MODEL,
            hidden_size=args.hidden_size,
        ).to(device)

        total_parameters = sum(
            p.numel()
            for p in model.parameters()
        )

        trainable_parameters = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        criterion = nn.MSELoss()

        training_start = (
            time.perf_counter()
        )

        for epoch in range(
            1,
            args.epochs + 1,
        ):
            model.train()

            total_loss = 0.0
            total_examples = 0

            for (
                input_ids,
                attention_mask,
                targets,
            ) in train_loader:

                input_ids = (
                    input_ids.to(device)
                )

                attention_mask = (
                    attention_mask.to(
                        device
                    )
                )

                targets = (
                    targets.to(device)
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                scores = model(
                    input_ids,
                    attention_mask,
                )

                loss = criterion(
                    scores,
                    targets,
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

                batch_n = (
                    input_ids.size(0)
                )

                total_loss += (
                    loss.item()
                    * batch_n
                )

                total_examples += (
                    batch_n
                )

            epoch_mse = (
                total_loss
                / total_examples
            )

            print(
                f"Epoch {epoch}/"
                f"{args.epochs} "
                f"train MSE="
                f"{epoch_mse:.6f}"
            )

        training_seconds = (
            time.perf_counter()
            - training_start
        )

        inference_start = (
            time.perf_counter()
        )

        student_test_score = (
            predict_scores(
                model,
                test_loader,
                device,
            )
        )

        inference_seconds = (
            time.perf_counter()
            - inference_start
        )

        student_test_class = (
            student_test_score > 0
        ).astype(int)

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

        macro_precision = (
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        )

        macro_recall = recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )

        balanced = (
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
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
        # DOWNSTREAM PERFORMANCE
        # --------------------------------------------------

        choices = np.where(
            student_test_class == 1,
            base.QWEN,
            base.DS,
        )

        routed_score = np.where(
            student_test_class == 1,
            test[
                "qwen_score"
            ].to_numpy(),
            test[
                "deepseek_score"
            ].to_numpy(),
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
            temp.groupby(
                "dataset"
            )["routed_score"]
            .mean()
            * 100
        )

        dataset_avg = (
            dataset_scores.mean()
        )

        frozen_row = (
            frozen_reference[
                frozen_reference[
                    "seed"
                ] == seed
            ].iloc[0]
        )

        frozen_f1 = (
            frozen_row[
                "router_macro_f1"
            ]
        )

        print()
        print("-" * 75)
        print(
            "ROUTER MODEL-SELECTION PERFORMANCE"
        )
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
        print(
            "TEACHER-STUDENT TRANSFER"
        )
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
            f"Frozen nonlinear F1       : "
            f"{frozen_f1:.4f}"
        )
        print(
            f"Fine-tuned encoder F1     : "
            f"{macro_f1:.4f}"
        )
        print(
            f"GTE teacher F1            : "
            f"{teacher_f1:.4f}"
        )

        print()
        print("TRAINING")
        print(
            f"Train queries             : "
            f"{len(train)}"
        )
        print(
            f"Epochs                    : "
            f"{args.epochs}"
        )
        print(
            f"Batch size                : "
            f"{args.batch_size}"
        )
        print(
            f"Learning rate             : "
            f"{args.learning_rate}"
        )
        print(
            f"Max length                : "
            f"{args.max_length}"
        )
        print(
            f"Total parameters          : "
            f"{total_parameters:,}"
        )
        print(
            f"Trainable parameters      : "
            f"{trainable_parameters:,}"
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
                "finetuned_encoder_mlp",
            "supervision":
                "gte_routing_score_all_queries",
            "hidden_size":
                args.hidden_size,
            "epochs":
                args.epochs,
            "batch_size":
                args.batch_size,
            "learning_rate":
                args.learning_rate,
            "weight_decay":
                args.weight_decay,
            "max_length":
                args.max_length,
            "total_parameters":
                total_parameters,
            "trainable_parameters":
                trainable_parameters,
            "training_seconds":
                training_seconds,
            "inference_seconds":
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
            "frozen_nonlinear_macro_f1":
                frozen_f1,
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
                "task_score_pct":
                    score,
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

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = pd.DataFrame(rows)

    by_dataset = pd.DataFrame(
        dataset_rows
    )

    results.to_csv(
        OUT
        / "minilm_encoder_score_distillation_multiseed.csv",
        index=False,
    )

    by_dataset.to_csv(
        OUT
        / "minilm_encoder_score_distillation_by_dataset.csv",
        index=False,
    )

    print()
    print("=" * 90)
    print(
        "MINILM ENCODER SCORE DISTILLATION SUMMARY"
    )
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
    print("REFERENCES")
    print(
        f"Frozen nonlinear F1       : "
        f"{results['frozen_nonlinear_macro_f1'].mean():.4f}"
    )
    print(
        f"Fine-tuned encoder F1     : "
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
        / "minilm_encoder_score_distillation_multiseed.csv"
    )
    print(
        OUT
        / "minilm_encoder_score_distillation_by_dataset.csv"
    )


if __name__ == "__main__":
    main()
