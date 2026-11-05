#!/usr/bin/env python3
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)
from sklearn.preprocessing import Normalizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cv_eval as cv

OUT = cv.OUT / "ordinal_router_mlp64_sklearn"
OUT.mkdir(parents=True, exist_ok=True)

# Existing labels:
#  0 = DeepSeek win
# -1 = tie
#  1 = Qwen win
#
# Ordered:
#  0 = DeepSeek better
#  1 = tie
#  2 = Qwen better
ORD_MAP = {0: 0, -1: 1, 1: 2}
LABELS = [0, 1, 2]


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


def balanced_resample(X, y, seed):
    """
    Training-fold-only random oversampling.
    Test folds are never changed.
    """
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    n = counts.max()

    idxs = []
    for c in classes:
        base = np.where(y == c)[0]
        chosen = rng.choice(base, size=n, replace=True)
        idxs.append(chosen)

    idx = np.concatenate(idxs)
    rng.shuffle(idx)
    return X[idx], y[idx]


def metrics(y, p):
    per = f1_score(
        y, p, labels=LABELS,
        average=None, zero_division=0
    )
    return {
        "macro_f1": f1_score(
            y, p, labels=LABELS,
            average="macro", zero_division=0
        ),
        "balanced_accuracy": balanced_accuracy_score(y, p),
        "accuracy": accuracy_score(y, p),
        "ordinal_mae": mean_absolute_error(y, p),
        "f1_deepseek": per[0],
        "f1_tie": per[1],
        "f1_qwen": per[2],
    }


def fit_nominal(Xtr, ytr, Xte, seed):
    Xb, yb = balanced_resample(Xtr, ytr, seed)

    model = make_mlp(seed)

    t0 = time.perf_counter()
    model.fit(Xb, yb)
    train_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = model.predict(Xte)
    infer_sec = time.perf_counter() - t0

    return pred, model.n_iter_, train_sec, infer_sec


def fit_binary_boundary(Xtr, ybin, Xte, seed):
    Xb, yb = balanced_resample(Xtr, ybin, seed)

    model = make_mlp(seed)

    t0 = time.perf_counter()
    model.fit(Xb, yb)
    train_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    prob = model.predict_proba(Xte)[:, 1]
    infer_sec = time.perf_counter() - t0

    return prob, model.n_iter_, train_sec, infer_sec


def fit_ordinal(Xtr, ytr, Xte, seed):
    # Boundary 1:
    # DeepSeek vs {Tie, Qwen}
    y_gt0 = (ytr > 0).astype(int)

    # Boundary 2:
    # {DeepSeek, Tie} vs Qwen
    y_gt1 = (ytr > 1).astype(int)

    p_gt0, e0, t0, i0 = fit_binary_boundary(
        Xtr, y_gt0, Xte, seed
    )
    p_gt1, e1, t1, i1 = fit_binary_boundary(
        Xtr, y_gt1, Xte, seed + 100
    )

    # Enforce P(Y>1) <= P(Y>0)
    bad = p_gt1 > p_gt0
    if bad.any():
        avg = (p_gt0[bad] + p_gt1[bad]) / 2.0
        p_gt0[bad] = avg
        p_gt1[bad] = avg

    probs = np.column_stack([
        1.0 - p_gt0,
        p_gt0 - p_gt1,
        p_gt1,
    ])

    pred = probs.argmax(axis=1)

    return pred, max(e0, e1), t0 + t1, i0 + i1


def main():
    pool = cv.load_full_pool_with_folds().copy()
    pool["y3"] = pool["target"].map(ORD_MAP).astype(int)

    cache = cv.load_embedding_cache(
        cv.MINILM_CACHE, cv.MINILM_MODEL
    )

    print(f"pool={len(pool)}")
    print(f"folds={sorted(pool['fold'].unique())}")
    print(f"MiniLM cached={len(cache)}")
    print(
        "class counts:",
        pool["y3"].value_counts().sort_index().to_dict()
    )
    print()

    fold_rows = []
    all_preds = []

    for fold in sorted(pool["fold"].unique()):
        train = pool[pool["fold"] != fold].copy()
        test = pool[pool["fold"] == fold].copy()

        norm = Normalizer(norm="l2")

        Xtr = norm.fit_transform(
            cv.embedding_matrix(
                train["query"].tolist(), cache
            )
        )

        Xte = norm.transform(
            cv.embedding_matrix(
                test["query"].tolist(), cache
            )
        )

        ytr = train["y3"].to_numpy()
        yte = test["y3"].to_numpy()

        pred_nom, e_nom, t_nom, i_nom = fit_nominal(
            Xtr, ytr, Xte, 42 + int(fold)
        )

        pred_ord, e_ord, t_ord, i_ord = fit_ordinal(
            Xtr, ytr, Xte, 142 + int(fold)
        )

        methods = [
            (
                "Always Tie",
                np.ones(len(test), dtype=int),
                0, 0.0, 0.0
            ),
            (
                "Nominal MLP-64",
                pred_nom,
                e_nom, t_nom, i_nom
            ),
            (
                "Ordinal MLP-64",
                pred_ord,
                e_ord, t_ord, i_ord
            ),
        ]

        for name, pred, epochs, train_sec, infer_sec in methods:
            m = metrics(yte, pred)

            fold_rows.append({
                "fold": fold,
                "method": name,
                **m,
                "epochs": epochs,
                "train_sec": train_sec,
                "infer_sec": infer_sec,
                "n_test": len(test),
            })

            print(
                f"fold {fold} | {name:16s} "
                f"F1={m['macro_f1']:.4f} "
                f"DS={m['f1_deepseek']:.4f} "
                f"Tie={m['f1_tie']:.4f} "
                f"Qwen={m['f1_qwen']:.4f} "
                f"epochs={epochs}"
            )

            for j, idx in enumerate(test.index):
                all_preds.append({
                    "row_index": idx,
                    "fold": fold,
                    "dataset": test.loc[idx, "dataset"],
                    "target": int(yte[j]),
                    "method": name,
                    "prediction": int(pred[j]),
                })

        print()

    fdf = pd.DataFrame(fold_rows)
    pdf = pd.DataFrame(all_preds)

    fdf.to_csv(OUT / "folds.csv", index=False)
    pdf.to_csv(OUT / "predictions.csv", index=False)

    summary = []

    print("=" * 72)
    print("POOLED HELD-OUT RESULTS")
    print("=" * 72)

    for method in [
        "Always Tie",
        "Nominal MLP-64",
        "Ordinal MLP-64"
    ]:
        d = pdf[pdf["method"] == method]

        y = d["target"].to_numpy()
        p = d["prediction"].to_numpy()

        m = metrics(y, p)
        fd = fdf[fdf["method"] == method]

        summary.append({
            "method": method,
            **m,
            "fold_macro_f1_mean": fd["macro_f1"].mean(),
            "fold_macro_f1_std": fd["macro_f1"].std(ddof=1),
            "train_sec_total": fd["train_sec"].sum(),
        })

        print()
        print(method)
        print(f"  Macro-F1:          {m['macro_f1']:.4f}")
        print(
            f"  Balanced accuracy: "
            f"{m['balanced_accuracy']:.4f}"
        )
        print(f"  Accuracy:          {m['accuracy']:.4f}")
        print(f"  Ordinal MAE:       {m['ordinal_mae']:.4f}")
        print(f"  F1 DeepSeek:       {m['f1_deepseek']:.4f}")
        print(f"  F1 Tie:            {m['f1_tie']:.4f}")
        print(f"  F1 Qwen:           {m['f1_qwen']:.4f}")
        print(
            f"  Fold Macro-F1:     "
            f"{fd['macro_f1'].mean():.4f} +/- "
            f"{fd['macro_f1'].std(ddof=1):.4f}"
        )
        print("  Confusion matrix [DS, Tie, Qwen]:")
        print(confusion_matrix(y, p, labels=LABELS))

    sdf = pd.DataFrame(summary)
    sdf.to_csv(OUT / "summary.csv", index=False)

    nom = sdf[
        sdf["method"] == "Nominal MLP-64"
    ].iloc[0]

    ordm = sdf[
        sdf["method"] == "Ordinal MLP-64"
    ].iloc[0]

    delta = ordm["macro_f1"] - nom["macro_f1"]

    print()
    print("=" * 72)
    print("PREDECLARED COMPARISON")
    print("=" * 72)

    print(
        f"Ordinal - nominal Macro-F1 delta: "
        f"{delta:+.4f}"
    )

    print(
        "Criterion: ordinal >= nominal +0.02 Macro-F1 "
        "and non-zero F1 for both winner classes."
    )

    promising = (
        delta >= 0.02
        and ordm["f1_deepseek"] > 0
        and ordm["f1_qwen"] > 0
    )

    print(f"PROMISING = {bool(promising)}")
    print(f"Outputs: {OUT}")


if __name__ == "__main__":
    main()
