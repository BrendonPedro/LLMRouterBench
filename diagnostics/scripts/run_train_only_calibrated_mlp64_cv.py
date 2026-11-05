#!/usr/bin/env python3
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Normalizer
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cv_eval as cv

OUT = cv.OUT / "train_only_calibrated_mlp64"
OUT.mkdir(parents=True, exist_ok=True)

MAP3 = {0: 0, -1: 1, 1: 2}
LABELS = [0, 1, 2]

# Frozen before results.
INNER_K = 3
BIAS_GRID = np.arange(-1.5, 1.5001, 0.1)


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
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    n = counts.max()

    parts = []
    for c in classes:
        idx = np.where(y == c)[0]
        parts.append(rng.choice(idx, n, replace=True))

    idx = np.concatenate(parts)
    rng.shuffle(idx)
    return X[idx], y[idx]


def ordered_proba(model, X):
    raw = model.predict_proba(X)
    out = np.zeros((len(X), 3), dtype=float)
    for j, c in enumerate(model.classes_):
        out[:, int(c)] = raw[:, j]
    return out


def train_model(X, y, seed):
    Xb, yb = balanced_resample(X, y, seed)
    model = make_mlp(seed)
    model.fit(Xb, yb)
    return model


def metrics(y, pred):
    per = f1_score(
        y, pred,
        labels=LABELS,
        average=None,
        zero_division=0,
    )
    return {
        "macro_f1": f1_score(
            y, pred,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),
        "balanced_accuracy":
            balanced_accuracy_score(y, pred),
        "accuracy": accuracy_score(y, pred),
        "f1_deepseek": per[0],
        "f1_tie": per[1],
        "f1_qwen": per[2],
    }


def predict_with_bias(probs, b_ds, b_qw):
    # Tie bias fixed at zero because adding the same constant
    # to all classes changes nothing.
    logp = np.log(np.clip(probs, 1e-12, 1.0))
    scores = logp.copy()
    scores[:, 0] += b_ds
    scores[:, 2] += b_qw
    return scores.argmax(axis=1)


def choose_biases(y, probs):
    best_f1 = -1.0
    best = (0.0, 0.0)

    for b_ds in BIAS_GRID:
        for b_qw in BIAS_GRID:
            pred = predict_with_bias(
                probs, b_ds, b_qw
            )
            score = f1_score(
                y, pred,
                labels=LABELS,
                average="macro",
                zero_division=0,
            )

            if score > best_f1 + 1e-12:
                best_f1 = score
                best = (float(b_ds), float(b_qw))

    return best[0], best[1], best_f1


def main():
    pool = cv.load_full_pool_with_folds().copy()
    pool["y3"] = pool["target"].map(MAP3).astype(int)

    cache = cv.load_embedding_cache(
        cv.MINILM_CACHE,
        cv.MINILM_MODEL,
    )

    print(f"pool={len(pool)}")
    print(f"folds={sorted(pool['fold'].unique())}")
    print(f"MiniLM cached={len(cache)}")
    print(
        "class counts:",
        pool["y3"].value_counts().sort_index().to_dict()
    )
    print()
    print(
        "Calibration rule: 3-fold inner CV on outer-train only; "
        "fixed bias grid [-1.5,1.5] step 0.1."
    )
    print()

    rows = []
    preds = []

    for outer_fold in sorted(pool["fold"].unique()):
        train = pool[pool["fold"] != outer_fold].copy()
        test = pool[pool["fold"] == outer_fold].copy()

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

        # ----------------------------------------------------
        # TRAIN-ONLY OOF probabilities for decision calibration
        # ----------------------------------------------------
        inner = StratifiedKFold(
            n_splits=INNER_K,
            shuffle=True,
            random_state=1000 + int(outer_fold),
        )

        oof_probs = np.zeros(
            (len(train), 3),
            dtype=float
        )

        t0 = time.perf_counter()

        for inner_fold, (itr, iva) in enumerate(
            inner.split(Xtr, ytr)
        ):
            model = train_model(
                Xtr[itr],
                ytr[itr],
                seed=10000
                + int(outer_fold) * 100
                + inner_fold,
            )

            oof_probs[iva] = ordered_proba(
                model,
                Xtr[iva],
            )

        b_ds, b_qw, calibration_f1 = choose_biases(
            ytr,
            oof_probs,
        )

        calibration_sec = time.perf_counter() - t0

        # ----------------------------------------------------
        # Final model: entire outer-training set.
        # Outer test fold remains untouched.
        # ----------------------------------------------------
        t0 = time.perf_counter()

        final_model = train_model(
            Xtr,
            ytr,
            seed=42 + int(outer_fold),
        )

        final_train_sec = time.perf_counter() - t0

        probs = ordered_proba(
            final_model,
            Xte,
        )

        pred_default = probs.argmax(axis=1)

        pred_cal = predict_with_bias(
            probs,
            b_ds,
            b_qw,
        )

        print(
            f"fold {outer_fold}: "
            f"training-only biases "
            f"DS={b_ds:+.1f} Qwen={b_qw:+.1f} "
            f"innerF1={calibration_f1:.4f}"
        )

        for method, pred in [
            ("Default argmax", pred_default),
            ("Train-only calibrated", pred_cal),
        ]:
            m = metrics(yte, pred)

            rows.append({
                "fold": outer_fold,
                "method": method,
                **m,
                "bias_deepseek": b_ds,
                "bias_qwen": b_qw,
                "inner_calibration_f1":
                    calibration_f1,
                "calibration_sec":
                    calibration_sec,
                "final_train_sec":
                    final_train_sec,
            })

            print(
                f"  {method:22s} "
                f"F1={m['macro_f1']:.4f} "
                f"DS={m['f1_deepseek']:.4f} "
                f"Tie={m['f1_tie']:.4f} "
                f"Qwen={m['f1_qwen']:.4f}"
            )

            for j, idx in enumerate(test.index):
                preds.append({
                    "row_index": idx,
                    "fold": outer_fold,
                    "dataset":
                        test.loc[idx, "dataset"],
                    "target": int(yte[j]),
                    "method": method,
                    "prediction": int(pred[j]),
                })

        print()

    fdf = pd.DataFrame(rows)
    pdf = pd.DataFrame(preds)

    fdf.to_csv(
        OUT / "folds.csv",
        index=False,
    )
    pdf.to_csv(
        OUT / "predictions.csv",
        index=False,
    )

    print("=" * 76)
    print("POOLED HELD-OUT RESULTS")
    print("=" * 76)

    summary = []

    for method in [
        "Default argmax",
        "Train-only calibrated",
    ]:
        d = pdf[pdf["method"] == method]

        y = d["target"].to_numpy()
        p = d["prediction"].to_numpy()

        m = metrics(y, p)

        fd = fdf[fdf["method"] == method]

        summary.append({
            "method": method,
            **m,
            "fold_macro_f1_mean":
                fd["macro_f1"].mean(),
            "fold_macro_f1_std":
                fd["macro_f1"].std(ddof=1),
        })

        print()
        print(method)
        print(
            f"  Macro-F1:          "
            f"{m['macro_f1']:.4f}"
        )
        print(
            f"  Balanced accuracy: "
            f"{m['balanced_accuracy']:.4f}"
        )
        print(
            f"  Accuracy:          "
            f"{m['accuracy']:.4f}"
        )
        print(
            f"  F1 DeepSeek:       "
            f"{m['f1_deepseek']:.4f}"
        )
        print(
            f"  F1 Tie:            "
            f"{m['f1_tie']:.4f}"
        )
        print(
            f"  F1 Qwen:           "
            f"{m['f1_qwen']:.4f}"
        )
        print(
            f"  Fold Macro-F1:     "
            f"{fd['macro_f1'].mean():.4f} +/- "
            f"{fd['macro_f1'].std(ddof=1):.4f}"
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

    base = sdf[
        sdf["method"] == "Default argmax"
    ].iloc[0]

    cal = sdf[
        sdf["method"] == "Train-only calibrated"
    ].iloc[0]

    delta = (
        cal["macro_f1"]
        - base["macro_f1"]
    )

    print()
    print("=" * 76)
    print("PREDECLARED COMPARISON")
    print("=" * 76)

    print(
        f"Calibrated - default Macro-F1 delta: "
        f"{delta:+.4f}"
    )

    print(
        "Criterion: train-only calibration >= default "
        "+0.02 Macro-F1 and both winner classes retain "
        "non-zero F1."
    )

    promising = (
        delta >= 0.02
        and cal["f1_deepseek"] > 0
        and cal["f1_qwen"] > 0
    )

    print(f"PROMISING = {bool(promising)}")
    print()
    print("Chosen training-only biases by fold:")
    print(
        fdf[
            fdf["method"] == "Train-only calibrated"
        ][[
            "fold",
            "bias_deepseek",
            "bias_qwen",
            "inner_calibration_f1",
        ]].to_string(index=False)
    )

    print(f"\nOutputs: {OUT}")


if __name__ == "__main__":
    main()
