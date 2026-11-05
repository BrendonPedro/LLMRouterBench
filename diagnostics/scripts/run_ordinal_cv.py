#!/usr/bin/env python3

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
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


OUT_DIR = cv.OUT / "ordinal_router"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Original target:
#   0  = DeepSeek wins
#   1  = Qwen wins
#  -1  = tie
#
# Ordered target:
#   0 = DeepSeek better
#   1 = tie
#   2 = Qwen better
ORD_MAP = {0: 0, -1: 1, 1: 2}
LABELS = [0, 1, 2]
LABEL_NAMES = ["DeepSeek_better", "Tie", "Qwen_better"]


def metrics(y_true, y_pred):
    per = f1_score(
        y_true, y_pred,
        labels=LABELS,
        average=None,
        zero_division=0,
    )
    return {
        "macro_f1": f1_score(
            y_true, y_pred,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "ordinal_mae": mean_absolute_error(y_true, y_pred),
        "f1_deepseek": per[0],
        "f1_tie": per[1],
        "f1_qwen": per[2],
    }


def fit_nominal(X_tr, y_tr, X_te, fold):
    clf = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        class_weight="balanced",
        max_iter=2000,
        random_state=42 + fold,
    )
    t0 = time.perf_counter()
    clf.fit(X_tr, y_tr)
    train_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = clf.predict(X_te)
    infer_sec = time.perf_counter() - t0

    return pred, train_sec, infer_sec


def fit_ordinal(X_tr, y_tr, X_te, fold):
    """
    Cumulative ordinal reduction.

    Boundary 1 learns:
        DeepSeek  vs  Tie-or-Qwen

    Boundary 2 learns:
        DeepSeek-or-Tie  vs  Qwen

    Convert the two cumulative probabilities into probabilities
    for the three ordered classes.
    """

    # P(Y > 0): Tie or Qwen versus DeepSeek
    y_gt0 = (y_tr > 0).astype(int)

    # P(Y > 1): Qwen versus DeepSeek or Tie
    y_gt1 = (y_tr > 1).astype(int)

    m0 = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        max_iter=2000,
        random_state=42 + fold,
    )
    m1 = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        max_iter=2000,
        random_state=142 + fold,
    )

    t0 = time.perf_counter()
    m0.fit(X_tr, y_gt0)
    m1.fit(X_tr, y_gt1)
    train_sec = time.perf_counter() - t0

    t0 = time.perf_counter()

    p_gt0 = m0.predict_proba(X_te)[:, 1]
    p_gt1 = m1.predict_proba(X_te)[:, 1]

    # Enforce ordinal consistency: P(Y>1) <= P(Y>0).
    bad = p_gt1 > p_gt0
    if np.any(bad):
        avg = (p_gt0[bad] + p_gt1[bad]) / 2.0
        p_gt0[bad] = avg
        p_gt1[bad] = avg

    probs = np.column_stack([
        1.0 - p_gt0,        # DeepSeek better
        p_gt0 - p_gt1,      # Tie
        p_gt1,               # Qwen better
    ])

    pred = probs.argmax(axis=1)
    infer_sec = time.perf_counter() - t0

    return pred, train_sec, infer_sec


def main():
    pool = cv.load_full_pool_with_folds()
    cache = cv.load_embedding_cache(cv.MINILM_CACHE, cv.MINILM_MODEL)

    pool = pool.copy()
    pool["ordinal_target"] = pool["target"].map(ORD_MAP)

    if pool["ordinal_target"].isna().any():
        raise RuntimeError("Unexpected target value.")

    pool["ordinal_target"] = pool["ordinal_target"].astype(int)

    print(f"pool={len(pool)}")
    print(f"folds={sorted(pool['fold'].unique())}")
    print(f"MiniLM cached={len(cache)}")
    print()
    print("Overall class counts:")
    print(pool["ordinal_target"].value_counts().sort_index())
    print()

    all_predictions = []
    fold_rows = []

    for fold in sorted(pool["fold"].unique()):
        train = pool[pool["fold"] != fold].copy()
        test = pool[pool["fold"] == fold].copy()

        norm = Normalizer(norm="l2")

        X_tr = norm.fit_transform(
            cv.embedding_matrix(train["query"].tolist(), cache)
        )
        X_te = norm.transform(
            cv.embedding_matrix(test["query"].tolist(), cache)
        )

        y_tr = train["ordinal_target"].to_numpy()
        y_te = test["ordinal_target"].to_numpy()

        # Baseline: always predict the majority class (Tie).
        pred_tie = np.ones(len(test), dtype=int)

        # Ordinary 3-class classifier.
        pred_nom, nom_train, nom_infer = fit_nominal(
            X_tr, y_tr, X_te, fold
        )

        # Ordered classifier.
        pred_ord, ord_train, ord_infer = fit_ordinal(
            X_tr, y_tr, X_te, fold
        )

        models = [
            ("Always Tie", pred_tie, 0.0, 0.0),
            ("Nominal 3-class", pred_nom, nom_train, nom_infer),
            ("Ordinal cumulative", pred_ord, ord_train, ord_infer),
        ]

        for name, pred, train_sec, infer_sec in models:
            m = metrics(y_te, pred)

            fold_rows.append({
                "fold": fold,
                "method": name,
                **m,
                "train_sec": train_sec,
                "infer_sec": infer_sec,
                "n_test": len(test),
            })

            print(
                f"fold {fold} | {name:20s} "
                f"MacroF1={m['macro_f1']:.4f} "
                f"DS={m['f1_deepseek']:.4f} "
                f"Tie={m['f1_tie']:.4f} "
                f"Qwen={m['f1_qwen']:.4f}"
            )

            for i, row_idx in enumerate(test.index):
                all_predictions.append({
                    "row_index": row_idx,
                    "fold": fold,
                    "dataset": test.loc[row_idx, "dataset"],
                    "target": int(y_te[i]),
                    "method": name,
                    "prediction": int(pred[i]),
                })

        print()

    pred_df = pd.DataFrame(all_predictions)
    fold_df = pd.DataFrame(fold_rows)

    pred_df.to_csv(OUT_DIR / "ordinal_predictions.csv", index=False)
    fold_df.to_csv(OUT_DIR / "ordinal_folds.csv", index=False)

    print("=" * 72)
    print("POOLED HELD-OUT RESULTS")
    print("=" * 72)

    summary_rows = []

    for name in ["Always Tie", "Nominal 3-class", "Ordinal cumulative"]:
        d = pred_df[pred_df["method"] == name]

        y = d["target"].to_numpy()
        p = d["prediction"].to_numpy()

        m = metrics(y, p)

        folds = fold_df[fold_df["method"] == name]

        summary_rows.append({
            "method": name,
            **m,
            "fold_macro_f1_mean": folds["macro_f1"].mean(),
            "fold_macro_f1_std": folds["macro_f1"].std(ddof=1),
            "train_sec_total": folds["train_sec"].sum(),
            "infer_sec_total": folds["infer_sec"].sum(),
        })

        print()
        print(name)
        print(
            f"  Macro-F1:          {m['macro_f1']:.4f}"
        )
        print(
            f"  Balanced accuracy: {m['balanced_accuracy']:.4f}"
        )
        print(
            f"  Accuracy:          {m['accuracy']:.4f}"
        )
        print(
            f"  Ordinal MAE:       {m['ordinal_mae']:.4f}"
        )
        print(
            f"  F1 DeepSeek:       {m['f1_deepseek']:.4f}"
        )
        print(
            f"  F1 Tie:            {m['f1_tie']:.4f}"
        )
        print(
            f"  F1 Qwen:           {m['f1_qwen']:.4f}"
        )
        print(
            f"  Fold Macro-F1:     "
            f"{folds['macro_f1'].mean():.4f} "
            f"+/- {folds['macro_f1'].std(ddof=1):.4f}"
        )

        print("  Confusion matrix [DS, Tie, Qwen]:")
        print(confusion_matrix(y, p, labels=LABELS))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        OUT_DIR / "ordinal_summary.csv",
        index=False
    )

    nom = summary_df[
        summary_df["method"] == "Nominal 3-class"
    ].iloc[0]

    ord_ = summary_df[
        summary_df["method"] == "Ordinal cumulative"
    ].iloc[0]

    delta = ord_["macro_f1"] - nom["macro_f1"]

    print()
    print("=" * 72)
    print("PREDECLARED COMPARISON")
    print("=" * 72)
    print(
        f"Ordinal - nominal Macro-F1 delta: {delta:+.4f}"
    )

    promising = (
        delta >= 0.02
        and ord_["f1_deepseek"] > 0
        and ord_["f1_qwen"] > 0
    )

    print(
        "Criterion: ordinal >= nominal +0.02 Macro-F1 "
        "and non-zero F1 for both winner classes."
    )
    print(f"PROMISING = {promising}")
    print()
    print(f"Outputs: {OUT_DIR}")


if __name__ == "__main__":
    main()
