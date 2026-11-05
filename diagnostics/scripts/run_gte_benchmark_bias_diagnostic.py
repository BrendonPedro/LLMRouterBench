#!/usr/bin/env python3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import Normalizer
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cv_eval as cv

OUT = cv.OUT / "gte_benchmark_bias_diagnostic"
OUT.mkdir(parents=True, exist_ok=True)

MAP3 = {0: 0, -1: 1, 1: 2}
LABELS = [0, 1, 2]

INNER_K = 3
BIAS_GRID = np.arange(-1.5, 1.5001, 0.1)


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


def predict_bias(probs, b_ds, b_qw):
    logp = np.log(np.clip(probs, 1e-12, 1.0))
    s = logp.copy()
    s[:, 0] += b_ds
    s[:, 2] += b_qw
    return s.argmax(axis=1)


def choose_bias(y, probs):
    best = (-1.0, 0.0, 0.0)

    for b_ds in BIAS_GRID:
        for b_qw in BIAS_GRID:
            pred = predict_bias(
                probs, b_ds, b_qw
            )

            f = f1_score(
                y,
                pred,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )

            if f > best[0]:
                best = (
                    float(f),
                    float(b_ds),
                    float(b_qw),
                )

    return best


def add_rows(rows, test, probs, method, biases=None):
    pred = np.zeros(len(test), dtype=int)

    if biases is None:
        pred = probs.argmax(axis=1)

    else:
        datasets = test["dataset"].to_numpy()

        for ds in np.unique(datasets):
            m = datasets == ds
            b_ds, b_qw = biases[ds]

            pred[m] = predict_bias(
                probs[m],
                b_ds,
                b_qw,
            )

    for j, idx in enumerate(test.index):
        rows.append({
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

    for method in preds["method"].unique():
        md = preds[preds["method"] == method]

        for dataset, d in md.groupby("dataset"):
            y = d["target"].to_numpy()
            p = d["prediction"].to_numpy()

            if set(np.unique(y)) != {0, 1, 2}:
                continue

            mf1 = f1_score(
                y, p,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )

            nt = y != 1
            ywin = (y[nt] == 2).astype(int)

            ds = d.loc[nt, "p_ds"].to_numpy()
            qw = d.loc[nt, "p_qwen"].to_numpy()

            denom = ds + qw
            score = np.divide(
                qw,
                denom,
                out=np.full_like(qw, 0.5),
                where=denom > 0,
            )

            auc = roc_auc_score(
                ywin,
                score,
            )

            rows.append({
                "method": method,
                "dataset": dataset,
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
    print()

    rows = []
    bias_rows = []

    for outer in sorted(pool["fold"].unique()):
        train = pool[pool["fold"] != outer].copy()
        test = pool[pool["fold"] == outer].copy()

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

        # ---------------------------------------------
        # Training-only OOF predictions.
        # Outer test fold is never used for bias choice.
        # ---------------------------------------------
        inner = StratifiedKFold(
            n_splits=INNER_K,
            shuffle=True,
            random_state=1000 + int(outer),
        )

        oof = np.zeros(
            (len(train), 3),
            dtype=float,
        )

        for inner_fold, (itr, iva) in enumerate(
            inner.split(Xtr, ytr)
        ):
            clf = make_clf(
                10000
                + int(outer) * 100
                + inner_fold
            )

            clf.fit(
                Xtr[itr],
                ytr[itr],
            )

            oof[iva] = ordered_proba(
                clf,
                Xtr[iva],
            )

        biases = {}

        for dataset in sorted(train["dataset"].unique()):
            m = (
                train["dataset"].to_numpy()
                == dataset
            )

            best_f1, b_ds, b_qw = choose_bias(
                ytr[m],
                oof[m],
            )

            biases[dataset] = (
                b_ds,
                b_qw,
            )

            bias_rows.append({
                "fold": outer,
                "dataset": dataset,
                "bias_ds": b_ds,
                "bias_qwen": b_qw,
                "train_oof_macro_f1": best_f1,
            })

        # Final global model on all outer-train rows.
        final = make_clf(
            42 + int(outer)
        )

        final.fit(
            Xtr,
            ytr,
        )

        probs = ordered_proba(
            final,
            Xte,
        )

        add_rows(
            rows,
            test,
            probs,
            "Global GTE",
        )

        add_rows(
            rows,
            test,
            probs,
            "Benchmark-bias diagnostic",
            biases,
        )

        print(
            f"fold {outer}: fitted "
            f"{len(biases)} training-only benchmark biases"
        )

    preds = pd.DataFrame(rows)
    bias_df = pd.DataFrame(bias_rows)

    bench = benchmark_metrics(preds)

    preds.to_csv(
        OUT / "predictions.csv",
        index=False,
    )

    bench.to_csv(
        OUT / "benchmark_metrics.csv",
        index=False,
    )

    bias_df.to_csv(
        OUT / "biases.csv",
        index=False,
    )

    print()
    print("=" * 78)
    print("HELD-OUT RESULTS")
    print("=" * 78)

    summary = []

    for method in [
        "Global GTE",
        "Benchmark-bias diagnostic",
    ]:
        b = bench[bench["method"] == method]

        f1 = b["macro_f1"].mean()
        auc = b["winner_auc"].mean()

        summary.append({
            "method": method,
            "within_macro_f1": f1,
            "winner_auc": auc,
        })

        print()
        print(method)
        print(
            f"  Within-benchmark Macro-F1: "
            f"{f1:.4f}"
        )
        print(
            f"  Winner-only AUC:           "
            f"{auc:.4f}"
        )

    s = pd.DataFrame(summary)
    s.to_csv(
        OUT / "summary.csv",
        index=False,
    )

    base = s[
        s["method"] == "Global GTE"
    ].iloc[0]

    bias = s[
        s["method"] == "Benchmark-bias diagnostic"
    ].iloc[0]

    delta = (
        bias["within_macro_f1"]
        - base["within_macro_f1"]
    )

    print()
    print("=" * 78)
    print("COMPARISON")
    print("=" * 78)

    print(
        f"Within-benchmark Macro-F1 delta: "
        f"{delta:+.4f}"
    )

    print(
        f"Winner AUC delta: "
        f"{bias['winner_auc'] - base['winner_auc']:+.4f}"
    )

    print()
    print(
        "Criterion: benchmark-specific training-only "
        "biases recover >= +0.02 Macro-F1."
    )

    prior_shift = delta >= 0.02

    print(
        f"PRIOR_SHIFT_EXPLAINS_GAIN = "
        f"{bool(prior_shift)}"
    )

    if prior_shift:
        print(
            "Interpretation: different class tendencies "
            "explain a substantial part of the task-specific gain."
        )
    else:
        print(
            "Interpretation: simple class-prior differences are "
            "not enough; task-specific feature boundaries are needed."
        )

    print(f"\nOutputs: {OUT}")


if __name__ == "__main__":
    main()
