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
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


DS = "DeepSeek-R1-0528-Qwen3-8B"
QWEN = "Qwen3-8B"

MODEL_NAME = "google/bert_uncased_L-4_H-256_A-4"

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


class MicroEncoderDirectSelector(nn.Module):

    def __init__(self, model_name):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            model_name
        )

        hidden = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(0.1)

        # Direct model-selection logits:
        # class 0 = DeepSeek better
        # class 1 = Qwen better
        self.selector_head = nn.Linear(
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

        return self.selector_head(
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
        # Direct selector target:
        # 0 = DeepSeek performs better
        # 1 = Qwen performs better
        labels = torch.tensor(
            (
                df["qwen_score"]
                > df["deepseek_score"]
            ).astype(np.int64).to_numpy(),
            dtype=torch.long,
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
    baseline_micro,
    baseline_direct,
):

    print()
    print("=" * 90)
    print(f"BERT-MINI DIRECT SELECTOR SEED {seed}")
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

    # For direct model selection, ties do not have
    # a unique "better model" label.
    # Remove ties only from training.
    train_selector = train[
        train["deepseek_score"]
        != train["qwen_score"]
    ].copy()

    print(
        f"Train queries: {len(train)} | "
        f"Non-tie selector train: {len(train_selector)} | "
        f"Test queries: {len(test)}"
    )

    train_dataset = encode_dataframe(
        train_selector,
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

    model = MicroEncoderDirectSelector(
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

    loss_fn = nn.CrossEntropyLoss()

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

            probabilities = torch.softmax(
                logits,
                dim=1,
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

    test["deepseek_probability"] = (
        probabilities[:, 0]
    )

    test["qwen_probability"] = (
        probabilities[:, 1]
    )

    test["selector_class"] = (
        probabilities.argmax(axis=1)
    )

    test["direct_selector_choice"] = np.where(
        test["selector_class"] == 1,
        QWEN,
        DS,
    )

    test["direct_selector_score"] = np.where(
        test["direct_selector_choice"] == QWEN,
        test["qwen_score"],
        test["deepseek_score"],
    )

    # --------------------------------------------------
    # ROUTER CLASSIFICATION METRICS
    # Evaluate only queries where one model is actually
    # better than the other. Ties have no unique label.
    # --------------------------------------------------

    test["selector_target"] = np.select(
        [
            test["deepseek_score"] > test["qwen_score"],
            test["qwen_score"] > test["deepseek_score"],
        ],
        [
            0,
            1,
        ],
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
        selector_test["qwen_probability"]
        .to_numpy()
    )

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

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred,
    )

    mcc = matthews_corrcoef(
        y_true,
        y_pred,
    )

    auc = roc_auc_score(
        y_true,
        y_prob_qwen,
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
        test["direct_selector_score"].mean()
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

            direct_selector_task_score=(
                "direct_selector_score",
                "mean",
            ),

            oracle_accuracy=(
                "oracle_score",
                "mean",
            ),

            deepseek_selection=(
                "direct_selector_choice",
                lambda s:
                    (s == DS).mean(),
            ),

            qwen_selection=(
                "direct_selector_choice",
                lambda s:
                    (s == QWEN).mean(),
            ),
        )
        .reset_index()
    )

    by_dataset[
        "direct_selector_task_score_pct"
    ] = (
        by_dataset[
            "direct_selector_task_score"
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
            "direct_selector_task_score_pct"
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

    independent_micro = baseline_micro[
        baseline_micro["seed"] == seed
    ].iloc[0]

    tiny_direct = baseline_direct[
        baseline_direct["seed"] == seed
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

        # Router model-selection evaluation.
        # Primary metric requested by Film: Macro-F1.
        "selector_non_tie_test_n":
            len(selector_test),

        "selector_non_tie_test_pct":
            len(selector_test) / len(test) * 100,

        "router_macro_f1":
            macro_f1,

        "tiny_direct_router_macro_f1":
            tiny_direct[
                "router_macro_f1"
            ],

        "bertmini_minus_tiny_macro_f1":
            macro_f1
            - tiny_direct[
                "router_macro_f1"
            ],

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

        # Downstream routed task performance.
        "direct_selector_sample_avg_pct":
            micro_sample,

        "tiny_direct_sample_avg_pct":
            tiny_direct[
                "direct_selector_sample_avg_pct"
            ],

        "bertmini_minus_tiny_sample_pp":
            micro_sample
            - tiny_direct[
                "direct_selector_sample_avg_pct"
            ],

        "independent_microencoder_sample_avg_pct":
            independent_micro[
                "microencoder_sample_avg_pct"
            ],

        "direct_minus_independent_micro_sample_pp":
            micro_sample
            - independent_micro[
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

        "direct_selector_dataset_avg_pct":
            micro_dataset,

        "tiny_direct_dataset_avg_pct":
            tiny_direct[
                "direct_selector_dataset_avg_pct"
            ],

        "bertmini_minus_tiny_dataset_pp":
            micro_dataset
            - tiny_direct[
                "direct_selector_dataset_avg_pct"
            ],

        "independent_microencoder_dataset_avg_pct":
            independent_micro[
                "microencoder_dataset_avg_pct"
            ],

        "direct_minus_independent_micro_dataset_pp":
            micro_dataset
            - independent_micro[
                "microencoder_dataset_avg_pct"
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

        "direct_minus_prior_sample_pp":
            micro_sample
            - prior[
                "dataset_prior_sample_avg_pct"
            ],

        "direct_minus_tfidf_sample_pp":
            micro_sample
            - old[
                "tfidf_sample_avg_pct"
            ],

        "direct_minus_avengerspro_sample_pp":
            micro_sample
            - old[
                "avengerspro_sample_avg_pct"
            ],

        "direct_minus_prior_dataset_pp":
            micro_dataset
            - prior[
                "dataset_prior_dataset_avg_pct"
            ],

        "direct_minus_tfidf_dataset_pp":
            micro_dataset
            - old[
                "tfidf_dataset_avg_pct"
            ],

        "direct_minus_avengerspro_dataset_pp":
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
        "deepseek_probability",
        "qwen_probability",
        "selector_target",
        "selector_class",
        "direct_selector_choice",
        "direct_selector_score",
        "oracle_score",
    ]

    test[
        prediction_columns
    ].to_csv(
        OUT
        / f"bertmini_direct_selector_seed{seed}_predictions.csv",
        index=False,
    )

    by_dataset.to_csv(
        OUT
        / f"bertmini_direct_selector_seed{seed}_by_dataset.csv",
        index=False,
    )

    pd.DataFrame(
        epoch_logs
    ).to_csv(
        OUT
        / f"bertmini_direct_selector_seed{seed}_training.csv",
        index=False,
    )

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
        f"BERT-Mini - tiny F1       : "
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
        f"{len(selector_test)}/{len(test)} "
        f"({len(selector_test) / len(test) * 100:.2f}%)"
    )

    print()
    print("DOWNSTREAM ROUTED TASK PERFORMANCE")

    print(
        f"BERT-Mini Sample-Avg      : "
        f"{micro_sample:.4f}%"
    )

    print(
        f"Tiny direct selector      : "
        f"{tiny_direct['direct_selector_sample_avg_pct']:.4f}%"
    )

    print(
        f"BERT-Mini - tiny direct   : "
        f"{micro_sample - tiny_direct['direct_selector_sample_avg_pct']:+.4f} pp"
    )

    print(
        f"Independent-head baseline: "
        f"{independent_micro['microencoder_sample_avg_pct']:.4f}%"
    )

    print(
        f"Direct - independent     : "
        f"{micro_sample - independent_micro['microencoder_sample_avg_pct']:+.4f} pp"
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
        f"Direct selector Dataset-Avg: "
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
            tokenizer,
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
        / "bertmini_direct_selector_multiseed.csv",
        index=False,
    )

    if all_dataset_rows:

        pd.concat(
            all_dataset_rows,
            ignore_index=True,
        ).to_csv(
            OUT
            / "bertmini_direct_selector_by_dataset_all.csv",
            index=False,
        )

    overall_seconds = (
        time.perf_counter()
        - overall_start
    )

    print()
    print("=" * 95)
    print("BERT-MINI DIRECT SELECTOR CONTROLLED COMPARISON")
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
        f"BERT-Mini - tiny F1       : "
        f"{summary['bertmini_minus_tiny_macro_f1'].mean():+.4f}"
    )

    print(
        f"Macro Precision           : "
        f"{summary['router_macro_precision'].mean():.4f}"
    )

    print(
        f"Macro Recall              : "
        f"{summary['router_macro_recall'].mean():.4f}"
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
        f"BERT-Mini Sample-Avg      : "
        f"{summary['direct_selector_sample_avg_pct'].mean():.4f}%"
    )

    print(
        f"Tiny direct selector      : "
        f"{summary['tiny_direct_sample_avg_pct'].mean():.4f}%"
    )

    print(
        f"BERT-Mini - tiny direct   : "
        f"{summary['bertmini_minus_tiny_sample_pp'].mean():+.4f} pp"
    )

    print(
        f"Independent-head baseline: "
        f"{summary['independent_microencoder_sample_avg_pct'].mean():.4f}%"
    )

    print(
        f"Direct - independent     : "
        f"{summary['direct_minus_independent_micro_sample_pp'].mean():+.4f} pp"
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
        f"Direct - Prior           : "
        f"{summary['direct_minus_prior_sample_pp'].mean():+.4f} pp"
    )

    print(
        f"Direct - TF-IDF          : "
        f"{summary['direct_minus_tfidf_sample_pp'].mean():+.4f} pp"
    )

    print(
        f"Direct - AvengersPro     : "
        f"{summary['direct_minus_avengerspro_sample_pp'].mean():+.4f} pp"
    )

    print()

    print(
        f"Direct Dataset-Avg       : "
        f"{summary['direct_selector_dataset_avg_pct'].mean():.4f}%"
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
        OUT / "bertmini_direct_selector_multiseed.csv"
    )
    print(
        OUT / "bertmini_direct_selector_by_dataset_all.csv"
    )


if __name__ == "__main__":
    main()
