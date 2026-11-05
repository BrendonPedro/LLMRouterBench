#!/usr/bin/env python3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cv_eval as cv

OUT = cv.OUT / "gte_benchmark_balanced"
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


def ordered_proba(model, X):
    raw = model.predict_proba(X)
    out = np.zeros((len(X), 3), dtype=float)

    for j, c in enumerate(model.classes_):
        out[:, int(c)] = raw[:, j]

    return out


def benchmark_weights(datasets):
    """
    Every benchmark contributes equal total weight.

    Weight of each row = 1 / number of training rows
    from that benchmark, then normalize mean weight to 1.
    """
    s = pd.Series(datasets)
    counts = s.value_counts()

    w = s.map(
        lambda d: 1.0 / counts[d]
    ).to_numpy(dtype=float)

    return w / w.mean()


def add_predictions(rows, test, probs, method):
    pred = probs.argmax(axis=1)

    for j, idx in enumerate(test.index):
        rows.append({
            "row_index": idx,
            "fold": int(test.loc[idx, "fold"]),
            "dataset": test.loc[idx, "dataset"],
            "target": int(test.loc[idx, "y3"]),
            "method": method,
            "prediction": int(pred[j]),
            "p_ds": float(probs[j, 0]),
            "p_tie": float(probs[j, 1]),
            "p_qwen": float(probs[j, 2]),
        })


def benchmark_metrics(preds):
    rows = []

    for method in sorted(preds["method"].unique()):
        md = preds[preds["method"] == method]

        for dataset, d in md.groupby("dataset"):
            y = d["target"].to_numpy()
            p = d["prediction"].to_numpy()

            if set(np.unique(y)) != {0, 1, 2}:
                continue

            macro_f1 = f1_score(
                y,
                p,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )

            # Winner-only AUC on real non-ties.
            nt = y != 1
            y_nt = y[nt]
            y_win = (y_nt == 2).astype(int)

            winner_auc = np.nan

            if len(np.unique(y_win)) == 2:
                ds = d.loc[nt, "p_ds"].to_numpy()
                qw = d.loc[nt, "p_qwen"].to_numpy()

                denom = ds + qw
                score = np.divide(
                    qw,
                    denom,
                    out=np.full_like(qw, 0.5),
                    where=denom > 0,
                )

                winner_auc = roc_auc_score(
                    y_win,
                    score,
                )

            rows.append({
                "method": method,
                "dataset": dataset,
                "n": len(d),
                "macro_f1": macro_f1,
                "winner_auc": winner_auc,
            })

    return pd.DataFrame(rows)


def main():
    pool = cv.load_full_pool_with_folds().copy()
    pool["y3"] = pool["target"].map(MAP3).astype(int)

    cache = cv.load_embedding_cache(
        cv.GTE_CACHE,
        cv.GTE_MODEL,
    )

    print(f"pool={len(pool)}")
    print(f"benchmarks={pool['dataset'].nunique()}")
    print(f"folds={sorted(pool['fold'].unique())}")
    print(
        f"GTE cached={len(cache)} "
        f"dim={len(next(iter(cache.values())))}"
    )

    print("\nBenchmark sizes:")
    print(
        pool["dataset"]
        .value_counts()
        .sort_values(ascending=False)
        .to_string()
    )
    print()

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

        # -------------------------------------
        # Existing ordinary global GTE router.
        # -------------------------------------
        base = make_clf(42 + int(fold))
        base.fit(Xtr, ytr)

        base_probs = ordered_proba(
            base,
            Xte,
        )

        add_predictions(
            predictions,
            test,
            base_probs,
            "Global GTE",
        )

        # -------------------------------------
        # Benchmark-balanced training.
        # No benchmark identity at inference.
        # -------------------------------------
        weights = benchmark_weights(
            train["dataset"].to_numpy()
        )

        balanced = make_clf(
            142 + int(fold)
        )

        balanced.fit(
            Xtr,
            ytr,
            sample_weight=weights,
        )

        balanced_probs = ordered_proba(
            balanced,
            Xte,
        )

        add_predictions(
            predictions,
            test,
            balanced_probs,
            "Benchmark-balanced GTE",
        )

        print(
            f"fold {fold}: "
            f"train weight range="
            f"{weights.min():.3f}..{weights.max():.3f}"
        )

    preds = pd.DataFrame(predictions)
    bench = benchmark_metrics(preds)

    preds.to_csv(
        OUT / "predictions.csv",
        index=False,
    )

    bench.to_csv(
        OUT / "benchmark_metrics.csv",
        index=False,
    )

    print()
    print("=" * 78)
    print("HELD-OUT RESULTS")
    print("=" * 78)

    summary = []

    for method in [
        "Global GTE",
        "Benchmark-balanced GTE",
    ]:
        d = preds[preds["method"] == method]

        overall_f1 = f1_score(
            d["target"],
            d["prediction"],
            labels=LABELS,
            average="macro",
            zero_division=0,
        )

        b = bench[
            bench["method"] == method
        ]

        within_f1 = b["macro_f1"].mean()
        winner_auc = b["winner_auc"].mean()

        summary.append({
            "method": method,
            "overall_macro_f1": overall_f1,
            "within_benchmark_macro_f1":
                within_f1,
            "winner_auc": winner_auc,
            "n_benchmarks": len(b),
        })

        print()
        print(method)
        print(
            f"  Overall Macro-F1:          "
            f"{overall_f1:.4f}"
        )
        print(
            f"  Within-benchmark Macro-F1: "
            f"{within_f1:.4f}"
        )
        print(
            f"  Winner-only AUC:           "
            f"{winner_auc:.4f}"
        )

    sdf = pd.DataFrame(summary)

    sdf.to_csv(
        OUT / "summary.csv",
        index=False,
    )

    base = sdf[
        sdf["method"] == "Global GTE"
    ].iloc[0]

    bal = sdf[
        sdf["method"] == "Benchmark-balanced GTE"
    ].iloc[0]

    f1_delta = (
        bal["within_benchmark_macro_f1"]
        - base["within_benchmark_macro_f1"]
    )

    auc_delta = (
        bal["winner_auc"]
        - base["winner_auc"]
    )

    overall_delta = (
        bal["overall_macro_f1"]
        - base["overall_macro_f1"]
    )

    paired = bench.pivot(
        index="dataset",
        columns="method",
        values=["macro_f1", "winner_auc"],
    )

    paired["f1_delta"] = (
        paired[
            ("macro_f1", "Benchmark-balanced GTE")
        ]
        - paired[
            ("macro_f1", "Global GTE")
        ]
    )

    paired["auc_delta"] = (
        paired[
            ("winner_auc", "Benchmark-balanced GTE")
        ]
        - paired[
            ("winner_auc", "Global GTE")
        ]
    )

    print()
    print("=" * 78)
    print("COMPARISON")
    print("=" * 78)

    print(
        f"Within-benchmark Macro-F1 delta: "
        f"{f1_delta:+.4f}"
    )
    print(
        f"Winner AUC delta: "
        f"{auc_delta:+.4f}"
    )
    print(
        f"Overall Macro-F1 delta: "
        f"{overall_delta:+.4f}"
    )

    print(
        f"Balanced training improves F1 on "
        f"{int((paired['f1_delta'] > 0).sum())}"
        f"/{len(paired)} benchmarks"
    )

    print("\nPer-benchmark deltas:")
    print(
        paired[
            ["f1_delta", "auc_delta"]
        ]
        .sort_values(
            "f1_delta",
            ascending=False,
        )
        .to_string()
    )

    print()
    print(
        "Criterion: benchmark-balanced global training "
        "improves within-benchmark Macro-F1 by >= +0.02 "
        "and winner AUC does not worsen."
    )

    promising = (
        f1_delta >= 0.02
        and auc_delta >= 0
    )

    print(
        f"GLOBAL_RECOVERY = "
        f"{bool(promising)}"
    )

    print(f"\nOutputs: {OUT}")


if __name__ == "__main__":
    main()
