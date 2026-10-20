import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score


DS = "DeepSeek-R1-0528-Qwen3-8B"
QWEN = "Qwen3-8B"

OUT = Path("local_experiments/results")
rows = []


def load_jsonl(path):
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)

            data.append({
                "dataset": x["dataset"],
                "query": x["query"],
                "deepseek_score": float(x["records"][DS]),
                "qwen_score": float(x["records"][QWEN]),
            })

    return pd.DataFrame(data)


for seed in range(42, 52):

    base = Path(
        f"baselines/AvengersPro/data/"
        f"small_models_seed_{seed}/seed{seed}_split0.7"
    )

    train = load_jsonl(base / "train.jsonl")
    test = load_jsonl(base / "test.jsonl")

    for dataset in sorted(test["dataset"].unique()):

        tr = train[train["dataset"] == dataset].copy()
        te = test[test["dataset"] == dataset].copy()

        tr = tr[
            tr["deepseek_score"] != tr["qwen_score"]
        ].copy()

        te = te[
            te["deepseek_score"] != te["qwen_score"]
        ].copy()

        if len(tr) < 10 or len(te) < 5:
            continue

        tr["label"] = np.where(
            tr["qwen_score"] > tr["deepseek_score"], 1, 0
        )

        te["label"] = np.where(
            te["qwen_score"] > te["deepseek_score"], 1, 0
        )

        if (
            tr["label"].nunique() < 2
            or te["label"].nunique() < 2
        ):
            continue

        vec = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=20000,
        )

        try:
            X_train = vec.fit_transform(tr["query"])
            X_test = vec.transform(te["query"])
        except ValueError:
            continue

        clf = LogisticRegression(
            max_iter=2000,
            random_state=42,
            class_weight="balanced",
        )

        clf.fit(X_train, tr["label"])

        prob = clf.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)

        auc = roc_auc_score(te["label"], prob)

        bal_acc = balanced_accuracy_score(
            te["label"], pred
        )

        majority_acc = max(
            (te["label"] == 0).mean(),
            (te["label"] == 1).mean(),
        )

        rows.append({
            "seed": seed,
            "dataset": dataset,
            "non_tied_train": len(tr),
            "non_tied_test": len(te),
            "qwen_win_rate_test_pct":
                te["label"].mean() * 100,
            "roc_auc": auc,
            "balanced_accuracy": bal_acc,
            "majority_accuracy": majority_acc,
        })


df = pd.DataFrame(rows)

df.to_csv(
    OUT / "within_dataset_signal_diagnostics_by_seed.csv",
    index=False,
)

summary = (
    df.groupby("dataset")
    .agg(
        seeds=("seed", "nunique"),
        mean_non_tied_test=("non_tied_test", "mean"),
        roc_auc=("roc_auc", "mean"),
        roc_auc_std=("roc_auc", "std"),
        balanced_accuracy=("balanced_accuracy", "mean"),
        majority_accuracy=("majority_accuracy", "mean"),
    )
    .reset_index()
)

summary["auc_above_random"] = summary["roc_auc"] - 0.5

summary = summary.sort_values(
    "roc_auc",
    ascending=False,
)

summary.to_csv(
    OUT / "within_dataset_signal_diagnostics_summary.csv",
    index=False,
)

print()
print("=" * 95)
print("WITHIN-DATASET SIGNAL DIAGNOSTICS")
print("=" * 95)

print(
    summary.round(3).to_string(index=False)
)

print()
print("Saved:")
print(
    OUT /
    "within_dataset_signal_diagnostics_by_seed.csv"
)
print(
    OUT /
    "within_dataset_signal_diagnostics_summary.csv"
)
