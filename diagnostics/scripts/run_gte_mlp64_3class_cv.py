#!/usr/bin/env python3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Normalizer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cv_eval as cv

OUT = cv.OUT / "gte_mlp64_3class"
OUT.mkdir(parents=True, exist_ok=True)

MAP3 = {0: 0, -1: 1, 1: 2}
LABELS = [0, 1, 2]


def make_logistic(seed):
    return LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
    )


def make_mlp(seed):
    return MLPClassifier(
        hidden_layer_sizes=(64,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=seed,
    )


def ordered_proba(model, X):
    raw = model.predict_proba(X)
    out = np.zeros((len(X), 3), dtype=float)

    for j, c in enumerate(model.classes_):
        out[:, int(c)] = raw[:, j]

    return out


def add_rows(rows, test, probs, method):
    pred = probs.argmax(axis=1)

    for j, idx in enumerate(test.index):
        rows.append({
            "row_index": int(idx),
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

            mf1 = f1_score(
                y,
                p,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )

            nt = y != 1
            ywin = (y[nt] == 2).astype(int)

            ds = d.loc[nt, "p_ds"].to_numpy()
            qw = d.loc[nt, "p_qwen"].to_numpy()

            denom = ds + qw
            winner_score = np.divide(
                qw,
                denom,
                out=np.full_like(qw, 0.5),
                where=denom > 0,
            )

            auc = roc_auc_score(
                ywin,
                winner_score,
            )

            rows.append({
                "method": method,
                "dataset": dataset,
                "n": len(d),
                "macro_f1": mf1,
                "winner_auc": auc,
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
    print(
        "class counts:",
        pool["y3"].value_counts().sort_index().to_dict()
    )
    print()

    rows = []
    timing = []

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

        # ------------------------------------
        # Linear baseline.
        # ------------------------------------
        logistic = make_logistic(
            42 + int(fold)
        )

        t0 = time.perf_counter()

        logistic.fit(
            Xtr,
            ytr,
        )

        logistic_sec = (
            time.perf_counter() - t0
        )

        logistic_probs = ordered_proba(
            logistic,
            Xte,
        )

        add_rows(
            rows,
            test,
            logistic_probs,
            "GTE Logistic",
        )

        # ------------------------------------
        # Nonlinear global MLP-64.
        #
        # Same original rows. No oversampling.
        # Class balancing is training-only
        # sample weighting.
        # ------------------------------------
        weights = compute_sample_weight(
            class_weight="balanced",
            y=ytr,
        )

        mlp = make_mlp(
            42 + int(fold)
        )

        t0 = time.perf_counter()

        mlp.fit(
            Xtr,
            ytr,
            sample_weight=weights,
        )

        mlp_sec = (
            time.perf_counter() - t0
        )

        mlp_probs = ordered_proba(
            mlp,
            Xte,
        )

        add_rows(
            rows,
            test,
            mlp_probs,
            "GTE MLP-64",
        )

        timing.append({
            "fold": fold,
            "logistic_sec": logistic_sec,
            "mlp_sec": mlp_sec,
            "mlp_iterations": mlp.n_iter_,
            "mlp_loss": mlp.loss_,
        })

        print(
            f"fold {fold}: "
            f"logistic={logistic_sec:.1f}s "
            f"MLP={mlp_sec:.1f}s "
            f"MLP_iter={mlp.n_iter_}"
        )

    preds = pd.DataFrame(rows)
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
    print("HELD-OUT RESULTS")
    print("=" * 78)

    summary = []

    for method in [
        "GTE Logistic",
        "GTE MLP-64",
    ]:
        d = preds[
            preds["method"] == method
        ]

        y = d["target"].to_numpy()
        p = d["prediction"].to_numpy()

        overall = f1_score(
            y,
            p,
            labels=LABELS,
            average="macro",
            zero_division=0,
        )

        per = f1_score(
            y,
            p,
            labels=LABELS,
            average=None,
            zero_division=0,
        )

        b = bench[
            bench["method"] == method
        ]

        within = b["macro_f1"].mean()
        auc = b["winner_auc"].mean()

        summary.append({
            "method": method,
            "overall_macro_f1": overall,
            "within_benchmark_macro_f1": within,
            "winner_auc": auc,
            "f1_deepseek": per[0],
            "f1_tie": per[1],
            "f1_qwen": per[2],
        })

        print()
        print(method)
        print(
            f"  Overall Macro-F1:          "
            f"{overall:.4f}"
        )
        print(
            f"  Within-benchmark Macro-F1: "
            f"{within:.4f}"
        )
        print(
            f"  Winner-only AUC:           "
            f"{auc:.4f}"
        )
        print(
            f"  F1 DeepSeek:               "
            f"{per[0]:.4f}"
        )
        print(
            f"  F1 Tie:                    "
            f"{per[1]:.4f}"
        )
        print(
            f"  F1 Qwen:                   "
            f"{per[2]:.4f}"
        )

    sdf = pd.DataFrame(summary)

    sdf.to_csv(
        OUT / "summary.csv",
        index=False,
    )

    base = sdf[
        sdf["method"] == "GTE Logistic"
    ].iloc[0]

    mlp = sdf[
        sdf["method"] == "GTE MLP-64"
    ].iloc[0]

    within_delta = (
        mlp["within_benchmark_macro_f1"]
        - base["within_benchmark_macro_f1"]
    )

    overall_delta = (
        mlp["overall_macro_f1"]
        - base["overall_macro_f1"]
    )

    auc_delta = (
        mlp["winner_auc"]
        - base["winner_auc"]
    )

    paired = bench.pivot(
        index="dataset",
        columns="method",
        values=["macro_f1", "winner_auc"],
    )

    paired["f1_delta"] = (
        paired[
            ("macro_f1", "GTE MLP-64")
        ]
        - paired[
            ("macro_f1", "GTE Logistic")
        ]
    )

    paired["auc_delta"] = (
        paired[
            ("winner_auc", "GTE MLP-64")
        ]
        - paired[
            ("winner_auc", "GTE Logistic")
        ]
    )

    print()
    print("=" * 78)
    print("COMPARISON")
    print("=" * 78)

    print(
        f"Within-benchmark Macro-F1 delta: "
        f"{within_delta:+.4f}"
    )

    print(
        f"Overall Macro-F1 delta: "
        f"{overall_delta:+.4f}"
    )

    print(
        f"Winner AUC delta: "
        f"{auc_delta:+.4f}"
    )

    print(
        f"MLP improves benchmark F1 on "
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
        "Criterion: GTE MLP-64 improves "
        "within-benchmark Macro-F1 by >= +0.02, "
        "overall Macro-F1 does not worsen, and "
        "winner-only AUC does not worsen."
    )

    promising = (
        within_delta >= 0.02
        and overall_delta >= 0
        and auc_delta >= 0
    )

    print(
        f"NONLINEAR_GLOBAL_SIGNAL = "
        f"{bool(promising)}"
    )

    print()
    print("Timing:")
    print(timing.to_string(index=False))

    print(f"\nOutputs: {OUT}")


if __name__ == "__main__":
    main()
