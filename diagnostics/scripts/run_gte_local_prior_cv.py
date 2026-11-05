#!/usr/bin/env python3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import Normalizer
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cv_eval as cv

OUT = cv.OUT / "gte_local_prior"
OUT.mkdir(parents=True, exist_ok=True)

MAP3 = {0: 0, -1: 1, 1: 2}
LABELS = [0, 1, 2]

# Frozen before results. No k sweep.
K_NEIGHBORS = 100

# Laplace smoothing for local class counts.
SMOOTH = 1.0


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


def local_prior_correct(
    base_probs,
    neighbor_labels,
    global_prior,
):
    """
    Prior-shift correction:

      corrected_c proportional to
      base_prob_c * local_prior_c / global_prior_c

    No tunable strength parameter.
    """
    counts = np.zeros(
        (len(neighbor_labels), 3),
        dtype=float,
    )

    for c in LABELS:
        counts[:, c] = (
            neighbor_labels == c
        ).sum(axis=1)

    local_prior = (
        counts + SMOOTH
    ) / (
        counts.sum(axis=1, keepdims=True)
        + 3 * SMOOTH
    )

    corrected = (
        base_probs
        * local_prior
        / global_prior[None, :]
    )

    corrected /= corrected.sum(
        axis=1,
        keepdims=True,
    )

    return corrected, local_prior


def add_rows(
    rows,
    test,
    probs,
    method,
    local_prior=None,
):
    pred = probs.argmax(axis=1)

    for j, idx in enumerate(test.index):
        row = {
            "fold": int(test.loc[idx, "fold"]),
            "dataset": test.loc[idx, "dataset"],
            "target": int(test.loc[idx, "y3"]),
            "method": method,
            "prediction": int(pred[j]),
            "p_ds": float(probs[j, 0]),
            "p_tie": float(probs[j, 1]),
            "p_qwen": float(probs[j, 2]),
        }

        if local_prior is not None:
            row.update({
                "local_prior_ds":
                    float(local_prior[j, 0]),
                "local_prior_tie":
                    float(local_prior[j, 1]),
                "local_prior_qwen":
                    float(local_prior[j, 2]),
            })

        rows.append(row)


def benchmark_metrics(preds):
    rows = []

    for method in preds["method"].unique():
        md = preds[
            preds["method"] == method
        ]

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
                "macro_f1": mf1,
                "winner_auc": auc,
            })

    return pd.DataFrame(rows)


def main():
    pool = cv.load_full_pool_with_folds().copy()
    pool["y3"] = (
        pool["target"]
        .map(MAP3)
        .astype(int)
    )

    cache = cv.load_embedding_cache(
        cv.GTE_CACHE,
        cv.GTE_MODEL,
    )

    print(f"pool={len(pool)}")
    print(
        f"folds="
        f"{sorted(pool['fold'].unique())}"
    )
    print(
        f"GTE dim="
        f"{len(next(iter(cache.values())))}"
    )
    print(f"K_NEIGHBORS={K_NEIGHBORS}")
    print()

    rows = []

    for fold in sorted(
        pool["fold"].unique()
    ):
        train = pool[
            pool["fold"] != fold
        ].copy()

        test = pool[
            pool["fold"] == fold
        ].copy()

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

        # ----------------------------------
        # Standard global classifier.
        # ----------------------------------
        clf = make_clf(
            42 + int(fold)
        )

        clf.fit(
            Xtr,
            ytr,
        )

        base_probs = ordered_proba(
            clf,
            Xte,
        )

        add_rows(
            rows,
            test,
            base_probs,
            "Global GTE",
        )

        # ----------------------------------
        # Query-local class prior.
        # Uses ONLY outer-training queries.
        # ----------------------------------
        nn = NearestNeighbors(
            n_neighbors=K_NEIGHBORS,
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )

        nn.fit(Xtr)

        _, idx = nn.kneighbors(
            Xte,
            return_distance=True,
        )

        neighbor_labels = ytr[idx]

        counts = np.bincount(
            ytr,
            minlength=3,
        ).astype(float)

        global_prior = (
            counts / counts.sum()
        )

        corrected, local_prior = (
            local_prior_correct(
                base_probs,
                neighbor_labels,
                global_prior,
            )
        )

        add_rows(
            rows,
            test,
            corrected,
            "Query-local prior GTE",
            local_prior,
        )

        print(
            f"fold {fold}: "
            f"train={len(train)} "
            f"test={len(test)} "
            f"neighbors={K_NEIGHBORS}"
        )

    preds = pd.DataFrame(rows)

    bench = benchmark_metrics(
        preds
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
    print("=" * 78)
    print("HELD-OUT RESULTS")
    print("=" * 78)

    summary = []

    for method in [
        "Global GTE",
        "Query-local prior GTE",
    ]:
        d = preds[
            preds["method"] == method
        ]

        overall = f1_score(
            d["target"],
            d["prediction"],
            labels=LABELS,
            average="macro",
            zero_division=0,
        )

        b = bench[
            bench["method"] == method
        ]

        within = (
            b["macro_f1"].mean()
        )

        auc = (
            b["winner_auc"].mean()
        )

        summary.append({
            "method": method,
            "overall_macro_f1":
                overall,
            "within_benchmark_macro_f1":
                within,
            "winner_auc":
                auc,
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

    s = pd.DataFrame(summary)

    s.to_csv(
        OUT / "summary.csv",
        index=False,
    )

    base = s[
        s["method"] == "Global GTE"
    ].iloc[0]

    local = s[
        s["method"] == "Query-local prior GTE"
    ].iloc[0]

    within_delta = (
        local["within_benchmark_macro_f1"]
        - base["within_benchmark_macro_f1"]
    )

    overall_delta = (
        local["overall_macro_f1"]
        - base["overall_macro_f1"]
    )

    auc_delta = (
        local["winner_auc"]
        - base["winner_auc"]
    )

    paired = bench.pivot(
        index="dataset",
        columns="method",
        values=["macro_f1", "winner_auc"],
    )

    paired["f1_delta"] = (
        paired[
            ("macro_f1", "Query-local prior GTE")
        ]
        - paired[
            ("macro_f1", "Global GTE")
        ]
    )

    paired["auc_delta"] = (
        paired[
            ("winner_auc", "Query-local prior GTE")
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
        f"Local prior improves F1 on "
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
        "Criterion: query-local prior improves "
        "within-benchmark Macro-F1 by >= +0.02 "
        "and overall Macro-F1 does not worsen."
    )

    promising = (
        within_delta >= 0.02
        and overall_delta >= 0
    )

    print(
        f"QUERY_LOCAL_PRIOR = "
        f"{bool(promising)}"
    )

    print(f"\nOutputs: {OUT}")


if __name__ == "__main__":
    main()
