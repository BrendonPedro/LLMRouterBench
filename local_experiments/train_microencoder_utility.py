import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer


DS = "DeepSeek-R1-0528-Qwen3-8B"
QWEN = "Qwen3-8B"

MODEL_NAME = "prajjwal1/bert-tiny"

OUT = Path("local_experiments/results")
OUT.mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


class MicroEncoderUtilityRouter(nn.Module):

    def __init__(self, model_name):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            model_name
        )

        hidden = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(0.1)

        # Two independent model-utility logits:
        # [DeepSeek success, Qwen success]
        self.utility_head = nn.Linear(
            hidden,
            2,
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

        # CLS representation
        representation = (
            outputs.last_hidden_state[:, 0, :]
        )

        representation = self.dropout(
            representation
        )

        return self.utility_head(
            representation
        )


def encode_dataframe(
    df,
    tokenizer,
    max_length,
    include_labels=True,
):

    encoded = tokenizer(
        df["query"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    ids = encoded["input_ids"]
    masks = encoded["attention_mask"]

    if include_labels:
        labels = torch.tensor(
            df[
                [
                    "deepseek_score",
                    "qwen_score",
                ]
            ].to_numpy(),
            dtype=torch.float32,
        )

        return TensorDataset(
            ids,
            masks,
            labels,
        )

    return TensorDataset(
        ids,
        masks,
    )


def train_one_seed(
    seed,
    args,
    tokenizer,
    baseline_exact,
    baseline_prior,
):

    print()
    print("=" * 90)
    print(f"MICRO-ENCODER SEED {seed}")
    print("=" * 90)

    set_seed(seed)

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

    print(
        f"Train queries: {len(train)} | "
        f"Test queries: {len(test)}"
    )

    train_dataset = encode_dataframe(
        train,
        tokenizer,
        args.max_length,
        include_labels=True,
    )

    test_dataset = encode_dataframe(
        test,
        tokenizer,
        args.max_length,
        include_labels=False,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = MicroEncoderUtilityRouter(
        args.model_name
    )

    device = torch.device("cpu")
    model.to(device)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    approx_fp32_mb = (
        total_params * 4 / (1024 ** 2)
    )

    print(
        f"Parameters: {total_params:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_params:,}"
    )

    print(
        f"Approx FP32 parameter size: "
        f"{approx_fp32_mb:.2f} MiB"
    )

    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    epoch_logs = []

    training_start = time.perf_counter()

    # ==================================================
    # TRAIN
    # ==================================================

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()

        epoch_start = time.perf_counter()

        total_loss = 0.0
        total_examples = 0

        for (
            input_ids,
            attention_mask,
            labels,
        ) in train_loader:

            input_ids = input_ids.to(device)
            attention_mask = (
                attention_mask.to(device)
            )
            labels = labels.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                input_ids,
                attention_mask,
            )

            loss = loss_fn(
                logits,
                labels,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            batch_n = labels.shape[0]

            total_loss += (
                loss.item() * batch_n
            )

            total_examples += batch_n

        epoch_seconds = (
            time.perf_counter()
            - epoch_start
        )

        mean_loss = (
            total_loss
            / total_examples
        )

        epoch_logs.append({
            "seed": seed,
            "epoch": epoch,
            "loss": mean_loss,
            "seconds": epoch_seconds,
        })

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"loss={mean_loss:.6f} | "
            f"time={epoch_seconds:.1f}s"
        )

    train_seconds = (
        time.perf_counter()
        - training_start
    )

    # ==================================================
    # TEST
    # ==================================================

    model.eval()

    all_probabilities = []

    inference_start = time.perf_counter()

    with torch.no_grad():

        for (
            input_ids,
            attention_mask,
        ) in test_loader:

            input_ids = input_ids.to(device)
            attention_mask = (
                attention_mask.to(device)
            )

            logits = model(
                input_ids,
                attention_mask,
            )

            probabilities = torch.sigmoid(
                logits
            )

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

    inference_seconds = (
        time.perf_counter()
        - inference_start
    )

    probabilities = np.concatenate(
        all_probabilities,
        axis=0,
    )

    test["deepseek_utility"] = (
        probabilities[:, 0]
    )

    test["qwen_utility"] = (
        probabilities[:, 1]
    )

    test["microencoder_choice"] = (
        np.where(
            test["qwen_utility"]
            >= test["deepseek_utility"],
            QWEN,
            DS,
        )
    )

    test["microencoder_score"] = np.where(
        test["microencoder_choice"]
        == QWEN,
        test["qwen_score"],
        test["deepseek_score"],
    )

    test["oracle_score"] = test[
        [
            "deepseek_score",
            "qwen_score",
        ]
    ].max(axis=1)

    # ==================================================
    # SAMPLE AVERAGE
    # ==================================================

    micro_sample = (
        test["microencoder_score"].mean()
        * 100
    )

    oracle_sample = (
        test["oracle_score"].mean()
        * 100
    )

    # ==================================================
    # DATASET AVERAGE
    # ==================================================

    by_dataset = (
        test.groupby("dataset")
        .agg(
            n_test=(
                "query",
                "size",
            ),

            microencoder_accuracy=(
                "microencoder_score",
                "mean",
            ),

            oracle_accuracy=(
                "oracle_score",
                "mean",
            ),

            deepseek_selection=(
                "microencoder_choice",
                lambda s:
                    (s == DS).mean(),
            ),

            qwen_selection=(
                "microencoder_choice",
                lambda s:
                    (s == QWEN).mean(),
            ),
        )
        .reset_index()
    )

    by_dataset[
        "microencoder_accuracy_pct"
    ] = (
        by_dataset[
            "microencoder_accuracy"
        ]
        * 100
    )

    by_dataset[
        "oracle_accuracy_pct"
    ] = (
        by_dataset[
            "oracle_accuracy"
        ]
        * 100
    )

    by_dataset[
        "deepseek_selection_pct"
    ] = (
        by_dataset[
            "deepseek_selection"
        ]
        * 100
    )

    by_dataset[
        "qwen_selection_pct"
    ] = (
        by_dataset[
            "qwen_selection"
        ]
        * 100
    )

    micro_dataset = (
        by_dataset[
            "microencoder_accuracy_pct"
        ].mean()
    )

    oracle_dataset = (
        by_dataset[
            "oracle_accuracy_pct"
        ].mean()
    )

    # ==================================================
    # EXISTING CONTROLLED BASELINES
    # ==================================================

    old = baseline_exact[
        baseline_exact["seed"] == seed
    ].iloc[0]

    prior = baseline_prior[
        baseline_prior["seed"] == seed
    ].iloc[0]

    row = {
        "seed": seed,
        "model_name":
            args.model_name,
        "epochs":
            args.epochs,
        "batch_size":
            args.batch_size,
        "max_length":
            args.max_length,
        "learning_rate":
            args.lr,

        "total_parameters":
            total_params,

        "trainable_parameters":
            trainable_params,

        "approx_fp32_parameter_mb":
            approx_fp32_mb,

        "training_seconds":
            train_seconds,

        "inference_seconds":
            inference_seconds,

        "microencoder_sample_avg_pct":
            micro_sample,

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

        "microencoder_dataset_avg_pct":
            micro_dataset,

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

        "micro_minus_prior_sample_pp":
            micro_sample
            - prior[
                "dataset_prior_sample_avg_pct"
            ],

        "micro_minus_tfidf_sample_pp":
            micro_sample
            - old[
                "tfidf_sample_avg_pct"
            ],

        "micro_minus_avengerspro_sample_pp":
            micro_sample
            - old[
                "avengerspro_sample_avg_pct"
            ],

        "micro_minus_prior_dataset_pp":
            micro_dataset
            - prior[
                "dataset_prior_dataset_avg_pct"
            ],

        "micro_minus_tfidf_dataset_pp":
            micro_dataset
            - old[
                "tfidf_dataset_avg_pct"
            ],

        "micro_minus_avengerspro_dataset_pp":
            micro_dataset
            - old[
                "avengerspro_dataset_avg_pct"
            ],
    }

    # ==================================================
    # SAVE EVIDENCE
    # ==================================================

    prediction_columns = [
        "dataset",
        "index",
        "deepseek_score",
        "qwen_score",
        "deepseek_utility",
        "qwen_utility",
        "microencoder_choice",
        "microencoder_score",
        "oracle_score",
    ]

    test[
        prediction_columns
    ].to_csv(
        OUT
        / f"microencoder_seed{seed}_predictions.csv",
        index=False,
    )

    by_dataset.to_csv(
        OUT
        / f"microencoder_seed{seed}_by_dataset.csv",
        index=False,
    )

    pd.DataFrame(
        epoch_logs
    ).to_csv(
        OUT
        / f"microencoder_seed{seed}_training.csv",
        index=False,
    )

    print()
    print("-" * 90)

    print(
        f"Micro-encoder Sample-Avg : "
        f"{micro_sample:.4f}%"
    )

    print(
        f"Dataset Prior            : "
        f"{prior['dataset_prior_sample_avg_pct']:.4f}%"
    )

    print(
        f"TF-IDF                   : "
        f"{old['tfidf_sample_avg_pct']:.4f}%"
    )

    print(
        f"AvengersPro              : "
        f"{old['avengerspro_sample_avg_pct']:.4f}%"
    )

    print(
        f"Oracle                    : "
        f"{oracle_sample:.4f}%"
    )

    print()

    print(
        f"Micro-encoder Dataset-Avg: "
        f"{micro_dataset:.4f}%"
    )

    print(
        f"Prior Dataset-Avg        : "
        f"{prior['dataset_prior_dataset_avg_pct']:.4f}%"
    )

    print(
        f"TF-IDF Dataset-Avg       : "
        f"{old['tfidf_dataset_avg_pct']:.4f}%"
    )

    print(
        f"AvengersPro Dataset-Avg  : "
        f"{old['avengerspro_dataset_avg_pct']:.4f}%"
    )

    print(
        f"Oracle Dataset-Avg       : "
        f"{oracle_dataset:.4f}%"
    )

    print()

    print(
        f"Training time            : "
        f"{train_seconds:.1f}s"
    )

    print(
        f"Inference time           : "
        f"{inference_seconds:.1f}s"
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
        "--model-name",
        default=MODEL_NAME,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    os.environ[
        "TOKENIZERS_PARALLELISM"
    ] = "false"

    torch.set_num_threads(
        args.threads
    )

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    print(
        f"PyTorch threads: "
        f"{torch.get_num_threads()}"
    )

    print(
        f"Device: CPU"
    )

    print(
        f"Loading tokenizer: "
        f"{args.model_name}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name
    )

    baseline_exact = pd.read_csv(
        OUT
        / "exact_multiseed_router_comparison.csv"
    )

    baseline_prior = pd.read_csv(
        OUT
        / "dataset_prior_multiseed_comparison.csv"
    )

    summary_rows = []
    all_dataset_rows = []

    overall_start = time.perf_counter()

    for seed in args.seeds:

        row, dataset_df = train_one_seed(
            seed,
            args,
            tokenizer,
            baseline_exact,
            baseline_prior,
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
        / "microencoder_multiseed.csv",
        index=False,
    )

    if all_dataset_rows:

        pd.concat(
            all_dataset_rows,
            ignore_index=True,
        ).to_csv(
            OUT
            / "microencoder_by_dataset_all.csv",
            index=False,
        )

    overall_seconds = (
        time.perf_counter()
        - overall_start
    )

    print()
    print("=" * 95)
    print("MICRO-ENCODER CONTROLLED COMPARISON")
    print("=" * 95)

    print(
        f"Seeds completed           : "
        f"{len(summary)}"
    )

    print(
        f"Micro-encoder Sample-Avg : "
        f"{summary['microencoder_sample_avg_pct'].mean():.4f}%"
    )

    print(
        f"Dataset Prior            : "
        f"{summary['dataset_prior_sample_avg_pct'].mean():.4f}%"
    )

    print(
        f"TF-IDF                   : "
        f"{summary['tfidf_sample_avg_pct'].mean():.4f}%"
    )

    print(
        f"AvengersPro              : "
        f"{summary['avengerspro_sample_avg_pct'].mean():.4f}%"
    )

    print(
        f"Oracle                    : "
        f"{summary['oracle_sample_avg_pct'].mean():.4f}%"
    )

    print()

    print(
        f"Micro - Prior            : "
        f"{summary['micro_minus_prior_sample_pp'].mean():+.4f} pp"
    )

    print(
        f"Micro - TF-IDF           : "
        f"{summary['micro_minus_tfidf_sample_pp'].mean():+.4f} pp"
    )

    print(
        f"Micro - AvengersPro      : "
        f"{summary['micro_minus_avengerspro_sample_pp'].mean():+.4f} pp"
    )

    print()

    print(
        f"Micro Dataset-Avg        : "
        f"{summary['microencoder_dataset_avg_pct'].mean():.4f}%"
    )

    print(
        f"Prior Dataset-Avg        : "
        f"{summary['dataset_prior_dataset_avg_pct'].mean():.4f}%"
    )

    print(
        f"TF-IDF Dataset-Avg       : "
        f"{summary['tfidf_dataset_avg_pct'].mean():.4f}%"
    )

    print(
        f"AvengersPro Dataset-Avg  : "
        f"{summary['avengerspro_dataset_avg_pct'].mean():.4f}%"
    )

    print(
        f"Oracle Dataset-Avg       : "
        f"{summary['oracle_dataset_avg_pct'].mean():.4f}%"
    )

    print()

    print(
        f"Total wall time          : "
        f"{overall_seconds:.1f}s"
    )

    print()

    print("Saved:")
    print(
        OUT / "microencoder_multiseed.csv"
    )
    print(
        OUT / "microencoder_by_dataset_all.csv"
    )


if __name__ == "__main__":
    main()
