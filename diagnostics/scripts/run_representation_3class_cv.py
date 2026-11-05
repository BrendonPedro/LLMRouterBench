#!/usr/bin/env python3
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cv_eval as cv

OUT = cv.OUT / "representation_3class"
OUT.mkdir(parents=True, exist_ok=True)

MAP3 = {0: 0, -1: 1, 1: 2}
LABELS = [0, 1, 2]


def make_clf(seed):
    return LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
    )


def metrics(y, p):
    per = f1_score(
        y, p,
        labels=LABELS,
        average=None,
        zero_division=0,
    )
    return {
        "macro_f1": f1_score(
            y, p,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),
        "balanced_accuracy":
            balanced_accuracy_score(y, p),
        "accuracy": accuracy_score(y, p),
        "f1_deepseek": per[0],
        "f1_tie": per[1],
        "f1_qwen": per[2],
    }


def run_representation(pool, cache, name):
    fold_rows = []
    predictions = []

    for fold in sorted(pool["fold"].unique()):
        train = pool[pool["fold"] != fold].copy()
        test = pool[pool["fold"] == fold].copy()

        norm = Normalizer(norm="l2")

        Xtr = norm.fit_transform(
            cv.embedding_matrix(
                train["query"].tolist(),
                cache,
            )
        )

        Xte = norm.transform(
            cv.embedding_matrix(
                test["query"].tolist(),
                cache,
            )
        )

        ytr = train["y3"].to_numpy()
        yte = test["y3"].to_numpy()

        clf = make_clf(42 + int(fold))

        t0 = time.perf_counter()
        clf.fit(Xtr, ytr)
        train_sec = time.perf_counter() - t0

        t0 = time.perf_counter()
        pred = clf.predict(Xte)
        infer_sec = time.perf_counter() - t0

        m = metrics(yte, pred)

        fold_rows.append({
            "representation": name,
            "fold": fold,
            **m,
            "train_sec": train_sec,
            "infer_sec": infer_sec,
            "dim": Xtr.shape[1],
        })

        print(
            f"{name:10s} fold {fold} "
            f"F1={m['macro_f1']:.4f} "
            f"DS={m['f1_deepseek']:.4f} "
            f"Tie={m['f1_tie']:.4f} "
            f"Qwen={m['f1_qwen']:.4f}"
        )

        for j, idx in enumerate(test.index):
            predictions.append({
                "representation": name,
                "row_index": idx,
                "fold": fold,
                "dataset":
                    test.loc[idx, "dataset"],
                "target": int(yte[j]),
                "prediction": int(pred[j]),
            })

    return (
        pd.DataFrame(fold_rows),
        pd.DataFrame(predictions),
    )


def benchmark_table(pred_df):
    rows = []

    for rep in sorted(
        pred_df["representation"].unique()
    ):
        r = pred_df[
            pred_df["representation"] == rep
        ]

        for dataset, d in r.groupby("dataset"):
            y = d["target"].to_numpy()
            p = d["prediction"].to_numpy()

            present = set(np.unique(y))

            # Require all three outcomes so benchmark
            # Macro-F1 means the same thing everywhere.
            if present != {0, 1, 2}:
                continue

            m = metrics(y, p)

            rows.append({
                "representation": rep,
                "dataset": dataset,
                "n": len(d),
                **m,
            })

    return pd.DataFrame(rows)


def main():
    pool = cv.load_full_pool_with_folds().copy()
    pool["y3"] = (
        pool["target"]
        .map(MAP3)
        .astype(int)
    )

    print(f"pool={len(pool)}")
    print(f"folds={sorted(pool['fold'].unique())}")
    print(
        "class counts:",
        pool["y3"]
        .value_counts()
        .sort_index()
        .to_dict(),
    )

    print("\nLoading MiniLM cache...")
    mini = cv.load_embedding_cache(
        cv.MINILM_CACHE,
        cv.MINILM_MODEL,
    )
    print(
        f"MiniLM cached={len(mini)} "
        f"dim={len(next(iter(mini.values())))}"
    )

    print("\nLoading GTE cache...")
    gte = cv.load_embedding_cache(
        cv.GTE_CACHE,
        cv.GTE_MODEL,
    )
    print(
        f"GTE cached={len(gte)} "
        f"dim={len(next(iter(gte.values())))}"
    )

    print("\nRunning MiniLM...")
    mini_folds, mini_preds = run_representation(
        pool, mini, "MiniLM"
    )

    print("\nRunning GTE...")
    gte_folds, gte_preds = run_representation(
        pool, gte, "GTE"
    )

    folds = pd.concat(
        [mini_folds, gte_folds],
        ignore_index=True,
    )

    preds = pd.concat(
        [mini_preds, gte_preds],
        ignore_index=True,
    )

    bench = benchmark_table(preds)

    folds.to_csv(
        OUT / "folds.csv",
        index=False,
    )
    preds.to_csv(
        OUT / "predictions.csv",
        index=False,
    )
    bench.to_csv(
        OUT / "benchmark_metrics.csv",
        index=False,
    )

    print()
    print("=" * 76)
    print("OVERALL POOLED HELD-OUT RESULTS")
    print("=" * 76)

    summary = []

    for rep in ["MiniLM", "GTE"]:
        d = preds[
            preds["representation"] == rep
        ]

        y = d["target"].to_numpy()
        p = d["prediction"].to_numpy()

        m = metrics(y, p)

        fd = folds[
            folds["representation"] == rep
        ]

        bd = bench[
            bench["representation"] == rep
        ]

        within_macro = (
            bd["macro_f1"].mean()
            if len(bd)
            else np.nan
        )

        winner_mean = (
            m["f1_deepseek"]
            + m["f1_qwen"]
        ) / 2.0

        summary.append({
            "representation": rep,
            **m,
            "winner_f1_mean": winner_mean,
            "fold_macro_f1_mean":
                fd["macro_f1"].mean(),
            "fold_macro_f1_std":
                fd["macro_f1"].std(ddof=1),
            "within_benchmark_macro_f1":
                within_macro,
            "n_valid_benchmarks":
                len(bd),
            "train_sec_total":
                fd["train_sec"].sum(),
        })

        print()
        print(rep)
        print(
            f"  Overall Macro-F1:        "
            f"{m['macro_f1']:.4f}"
        )
        print(
            f"  Balanced accuracy:       "
            f"{m['balanced_accuracy']:.4f}"
        )
        print(
            f"  F1 DeepSeek:             "
            f"{m['f1_deepseek']:.4f}"
        )
        print(
            f"  F1 Tie:                  "
            f"{m['f1_tie']:.4f}"
        )
        print(
            f"  F1 Qwen:                 "
            f"{m['f1_qwen']:.4f}"
        )
        print(
            f"  Winner-class F1 mean:    "
            f"{winner_mean:.4f}"
        )
        print(
            f"  Fold Macro-F1:           "
            f"{fd['macro_f1'].mean():.4f} +/- "
            f"{fd['macro_f1'].std(ddof=1):.4f}"
        )
        print(
            f"  Within-benchmark Macro-F1: "
            f"{within_macro:.4f} "
            f"({len(bd)} benchmarks)"
        )
        print("  Confusion matrix:")
        print(
            confusion_matrix(
                y, p,
                labels=LABELS,
            )
        )

    sdf = pd.DataFrame(summary)
    sdf.to_csv(
        OUT / "summary.csv",
        index=False,
    )

    mini_s = sdf[
        sdf["representation"] == "MiniLM"
    ].iloc[0]

    gte_s = sdf[
        sdf["representation"] == "GTE"
    ].iloc[0]

    overall_delta = (
        gte_s["macro_f1"]
        - mini_s["macro_f1"]
    )

    within_delta = (
        gte_s["within_benchmark_macro_f1"]
        - mini_s["within_benchmark_macro_f1"]
    )

    # Benchmark-by-benchmark paired comparison.
    bm = bench.pivot(
        index="dataset",
        columns="representation",
        values="macro_f1",
    ).dropna()

    bm["delta_GTE_minus_MiniLM"] = (
        bm["GTE"] - bm["MiniLM"]
    )

    print()
    print("=" * 76)
    print("REPRESENTATION COMPARISON")
    print("=" * 76)

    print(
        f"Overall Macro-F1 delta: "
        f"{overall_delta:+.4f}"
    )
    print(
        f"Within-benchmark Macro-F1 delta: "
        f"{within_delta:+.4f}"
    )

    if len(bm):
        print(
            f"GTE better on "
            f"{int((bm['delta_GTE_minus_MiniLM'] > 0).sum())}"
            f"/{len(bm)} valid benchmarks"
        )

        print("\nPer-benchmark:")
        print(
            bm.sort_values(
                "delta_GTE_minus_MiniLM",
                ascending=False,
            ).to_string()
        )

    print()
    print(
        "Criterion: GTE improves within-benchmark "
        "macro Macro-F1 by >= +0.03 and overall "
        "Macro-F1 does not worsen."
    )

    promising = (
        within_delta >= 0.03
        and overall_delta >= 0
    )

    print(
        f"REPRESENTATION_SIGNAL = "
        f"{bool(promising)}"
    )

    print(f"\nOutputs: {OUT}")


if __name__ == "__main__":
    main()
