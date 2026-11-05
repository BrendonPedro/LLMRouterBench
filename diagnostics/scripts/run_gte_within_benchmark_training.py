#!/usr/bin/env python3
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cv_eval as cv

OUT = cv.OUT / "gte_within_benchmark_training"
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
                y, p,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )

            nt = y != 1
            y_nt = y[nt]

            # Convert ordered 3-class target:
            # DS=0 -> binary 0
            # Qwen=2 -> binary 1
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
    print()

    predictions = []
    timing = []

    for fold in sorted(pool["fold"].unique()):
        train = pool[pool["fold"] != fold].copy()
        test = pool[pool["fold"] == fold].copy()

        # Embeddings are L2 normalized exactly as before.
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

        # -----------------------------------------
        # A: existing global GTE router
        # -----------------------------------------
        global_clf = make_clf(42 + int(fold))

        t0 = time.perf_counter()
        global_clf.fit(Xtr, ytr)
        global_sec = time.perf_counter() - t0

        global_probs = ordered_proba(
            global_clf,
            Xte,
        )

        add_predictions(
            predictions,
            test,
            global_probs,
            "Global GTE",
        )

        # -----------------------------------------
        # B: benchmark-specific diagnostic
        # Train only on same benchmark.
        # -----------------------------------------
        local_sec = 0.0

        for dataset in sorted(test["dataset"].unique()):
            tr_mask = (
                train["dataset"].to_numpy()
                == dataset
            )
            te_mask = (
                test["dataset"].to_numpy()
                == dataset
            )

            y_local = train.loc[
                tr_mask, "y3"
            ].to_numpy()

            classes = set(np.unique(y_local))

            if classes != {0, 1, 2}:
                raise RuntimeError(
                    f"fold={fold} dataset={dataset}: "
                    f"training classes={classes}"
                )

            local_clf = make_clf(
                1000
                + int(fold) * 100
                + sorted(
                    test["dataset"].unique()
                ).index(dataset)
            )

            t0 = time.perf_counter()

            local_clf.fit(
                Xtr[tr_mask],
                y_local,
            )

            local_sec += (
                time.perf_counter() - t0
            )

            local_probs = ordered_proba(
                local_clf,
                Xte[te_mask],
            )

            add_predictions(
                predictions,
                test.loc[te_mask],
                local_probs,
                "Within-benchmark GTE",
            )

        timing.append({
            "fold": fold,
            "global_train_sec": global_sec,
            "within_benchmark_train_sec":
                local_sec,
        })

        print(
            f"fold {fold}: "
            f"global={global_sec:.1f}s "
            f"within-benchmark={local_sec:.1f}s"
        )

    preds = pd.DataFrame(predictions)
    timing = pd.DataFrame(timing)

    bench = benchmark_metrics(preds)

    preds.to_csv(
        OUT / "predictions.csv",
        index=False,
    )
    bench.to_csv(
        OUT / "benchmark_metrics.csv",
        index=False,
    )
    timing.to_csv(
        OUT / "timing.csv",
        index=False,
    )

    print()
    print("=" * 78)
    print("WITHIN-BENCHMARK HELD-OUT RESULTS")
    print("=" * 78)

    summary = []

    for method in [
        "Global GTE",
        "Within-benchmark GTE",
    ]:
        b = bench[
            bench["method"] == method
        ]

        macro_f1 = b["macro_f1"].mean()
        macro_auc = b["winner_auc"].mean()

        summary.append({
            "method": method,
            "within_benchmark_macro_f1":
                macro_f1,
            "within_benchmark_winner_auc":
                macro_auc,
            "n_benchmarks": len(b),
        })

        print()
        print(method)
        print(
            f"  Mean benchmark Macro-F1: "
            f"{macro_f1:.4f}"
        )
        print(
            f"  Mean winner-only AUC:     "
            f"{macro_auc:.4f}"
        )

    s = pd.DataFrame(summary)
    s.to_csv(
        OUT / "summary.csv",
        index=False,
    )

    g = s[
        s["method"] == "Global GTE"
    ].iloc[0]

    w = s[
        s["method"] == "Within-benchmark GTE"
    ].iloc[0]

    f1_delta = (
        w["within_benchmark_macro_f1"]
        - g["within_benchmark_macro_f1"]
    )

    auc_delta = (
        w["within_benchmark_winner_auc"]
        - g["within_benchmark_winner_auc"]
    )

    paired = bench.pivot(
        index="dataset",
        columns="method",
        values=["macro_f1", "winner_auc"],
    )

    paired["f1_delta"] = (
        paired[
            ("macro_f1", "Within-benchmark GTE")
        ]
        - paired[
            ("macro_f1", "Global GTE")
        ]
    )

    paired["auc_delta"] = (
        paired[
            ("winner_auc", "Within-benchmark GTE")
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
        f"Macro-F1 delta: {f1_delta:+.4f}"
    )
    print(
        f"Winner AUC delta: {auc_delta:+.4f}"
    )

    print(
        f"Within-benchmark training improves F1 on "
        f"{int((paired['f1_delta'] > 0).sum())}"
        f"/{len(paired)} benchmarks"
    )

    print()
    print("Per-benchmark deltas:")
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
        "Criterion: within-benchmark training improves "
        "mean benchmark Macro-F1 by >= +0.03 and "
        "winner-only AUC does not worsen."
    )

    promising = (
        f1_delta >= 0.03
        and auc_delta >= 0
    )

    print(
        f"WITHIN_TASK_SIGNAL = "
        f"{bool(promising)}"
    )

    print(f"\nOutputs: {OUT}")


if __name__ == "__main__":
    main()
