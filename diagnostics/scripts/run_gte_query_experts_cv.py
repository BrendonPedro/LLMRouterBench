#!/usr/bin/env python3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cv_eval as cv

OUT = cv.OUT / "gte_query_experts"
OUT.mkdir(parents=True, exist_ok=True)

MAP3 = {0: 0, -1: 1, 1: 2}
LABELS = [0, 1, 2]

# Frozen before results. No K sweep.
K_EXPERTS = 4


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


def add_predictions(rows, test, probs, method, clusters=None):
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
            "cluster": (
                int(clusters[j])
                if clusters is not None
                else -1
            ),
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
                y, p,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )

            nt = y != 1
            ywin = (y[nt] == 2).astype(int)

            auc = np.nan

            if len(np.unique(ywin)) == 2:
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
    print(f"K_EXPERTS={K_EXPERTS}")
    print()

    predictions = []
    cluster_rows = []

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

        # ------------------------------------------------
        # Baseline: one global GTE classifier.
        # ------------------------------------------------
        global_clf = make_clf(42 + int(fold))
        global_clf.fit(Xtr, ytr)

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

        # ------------------------------------------------
        # Query-only gating.
        # No benchmark labels enter clustering or heads.
        # ------------------------------------------------
        gate = MiniBatchKMeans(
            n_clusters=K_EXPERTS,
            random_state=142 + int(fold),
            batch_size=1024,
            n_init=10,
        )

        train_cluster = gate.fit_predict(Xtr)
        test_cluster = gate.predict(Xte)

        expert_probs = np.zeros(
            (len(test), 3),
            dtype=float,
        )

        for c in range(K_EXPERTS):
            tr = train_cluster == c
            te = test_cluster == c

            ntr = int(tr.sum())
            nte = int(te.sum())
            classes = sorted(np.unique(ytr[tr]).tolist())

            cluster_rows.append({
                "fold": fold,
                "cluster": c,
                "n_train": ntr,
                "n_test": nte,
                "classes": ",".join(map(str, classes)),
            })

            print(
                f"fold {fold} cluster {c}: "
                f"train={ntr} test={nte} "
                f"classes={classes}"
            )

            if nte == 0:
                continue

            # Defensive fallback if a cluster lacks a class.
            if set(classes) != {0, 1, 2}:
                expert_probs[te] = global_probs[te]
                continue

            expert = make_clf(
                1000
                + int(fold) * 10
                + c
            )

            expert.fit(
                Xtr[tr],
                ytr[tr],
            )

            expert_probs[te] = ordered_proba(
                expert,
                Xte[te],
            )

        add_predictions(
            predictions,
            test,
            expert_probs,
            "Query-cluster experts",
            test_cluster,
        )

        print()

    preds = pd.DataFrame(predictions)
    clusters = pd.DataFrame(cluster_rows)

    bench = benchmark_metrics(preds)

    preds.to_csv(
        OUT / "predictions.csv",
        index=False,
    )

    bench.to_csv(
        OUT / "benchmark_metrics.csv",
        index=False,
    )

    clusters.to_csv(
        OUT / "cluster_sizes.csv",
        index=False,
    )

    print("=" * 78)
    print("HELD-OUT RESULTS")
    print("=" * 78)

    summary = []

    for method in [
        "Global GTE",
        "Query-cluster experts",
    ]:
        d = preds[preds["method"] == method]

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

        within = b["macro_f1"].mean()
        auc = b["winner_auc"].mean()

        summary.append({
            "method": method,
            "overall_macro_f1": overall,
            "within_benchmark_macro_f1": within,
            "winner_auc": auc,
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

    sdf = pd.DataFrame(summary)
    sdf.to_csv(
        OUT / "summary.csv",
        index=False,
    )

    base = sdf[
        sdf["method"] == "Global GTE"
    ].iloc[0]

    exp = sdf[
        sdf["method"] == "Query-cluster experts"
    ].iloc[0]

    within_delta = (
        exp["within_benchmark_macro_f1"]
        - base["within_benchmark_macro_f1"]
    )

    auc_delta = (
        exp["winner_auc"]
        - base["winner_auc"]
    )

    overall_delta = (
        exp["overall_macro_f1"]
        - base["overall_macro_f1"]
    )

    paired = bench.pivot(
        index="dataset",
        columns="method",
        values=["macro_f1", "winner_auc"],
    )

    paired["f1_delta"] = (
        paired[
            ("macro_f1", "Query-cluster experts")
        ]
        - paired[
            ("macro_f1", "Global GTE")
        ]
    )

    paired["auc_delta"] = (
        paired[
            ("winner_auc", "Query-cluster experts")
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
        f"Winner AUC delta: "
        f"{auc_delta:+.4f}"
    )
    print(
        f"Overall Macro-F1 delta: "
        f"{overall_delta:+.4f}"
    )

    print(
        f"Query experts improve F1 on "
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
        "Criterion: query-cluster experts improve "
        "within-benchmark Macro-F1 by >= +0.02, "
        "winner AUC does not worsen, and overall "
        "Macro-F1 does not worsen."
    )

    promising = (
        within_delta >= 0.02
        and auc_delta >= 0
        and overall_delta >= 0
    )

    print(
        f"QUERY_EXPERT_SIGNAL = "
        f"{bool(promising)}"
    )

    # Diagnostic only: inspect what clusters contain.
    # Benchmark labels are NOT used to train or route.
    q = preds[
        preds["method"] == "Query-cluster experts"
    ]

    composition = (
        q.groupby(["fold", "cluster", "dataset"])
        .size()
        .rename("n")
        .reset_index()
    )

    composition.to_csv(
        OUT / "cluster_composition_diagnostic.csv",
        index=False,
    )

    print(f"\nOutputs: {OUT}")


if __name__ == "__main__":
    main()
