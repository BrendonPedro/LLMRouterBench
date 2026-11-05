#!/usr/bin/env python3
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd

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

OUT = cv.OUT / "four_state_router_mlp64"
OUT.mkdir(parents=True, exist_ok=True)

# Collapsed routing target:
# 0 = DeepSeek better
# 1 = Tie
# 2 = Qwen better
THREE_MAP = {0: 0, -1: 1, 1: 2}
THREE_LABELS = [0, 1, 2]

# Four utility states:
# 0 = both wrong
# 1 = DeepSeek only correct
# 2 = Qwen only correct
# 3 = both correct
FOUR_LABELS = [0, 1, 2, 3]
FOUR_NAMES = [
    "Both_wrong",
    "DeepSeek_only",
    "Qwen_only",
    "Both_correct",
]


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
    # Training fold only. Held-out fold is untouched.
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    n = counts.max()

    parts = []
    for c in classes:
        idx = np.where(y == c)[0]
        parts.append(rng.choice(idx, size=n, replace=True))

    idx = np.concatenate(parts)
    rng.shuffle(idx)
    return X[idx], y[idx]


def three_metrics(y, pred):
    per = f1_score(
        y, pred,
        labels=THREE_LABELS,
        average=None,
        zero_division=0,
    )
    return {
        "macro_f1": f1_score(
            y, pred,
            labels=THREE_LABELS,
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


def four_metrics(y, pred):
    per = f1_score(
        y, pred,
        labels=FOUR_LABELS,
        average=None,
        zero_division=0,
    )
    return {
        "four_macro_f1": f1_score(
            y, pred,
            labels=FOUR_LABELS,
            average="macro",
            zero_division=0,
        ),
        "f1_both_wrong": per[0],
        "f1_deepseek_only": per[1],
        "f1_qwen_only": per[2],
        "f1_both_correct": per[3],
    }


def make_four_state(ds, qw):
    ds = int(round(float(ds)))
    qw = int(round(float(qw)))

    if ds == 0 and qw == 0:
        return 0
    if ds == 1 and qw == 0:
        return 1
    if ds == 0 and qw == 1:
        return 2
    if ds == 1 and qw == 1:
        return 3

    raise ValueError(f"Unexpected score pair: ds={ds}, qwen={qw}")


def collapse_four(pred4):
    # both wrong / both correct -> tie
    out = np.ones(len(pred4), dtype=int)
    out[pred4 == 1] = 0  # DeepSeek only
    out[pred4 == 2] = 2  # Qwen only
    return out


def fit_mlp(Xtr, ytr, Xte, seed):
    Xb, yb = balanced_resample(Xtr, ytr, seed)
    model = make_mlp(seed)

    t0 = time.perf_counter()
    model.fit(Xb, yb)
    train_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = model.predict(Xte)
    infer_sec = time.perf_counter() - t0

    return pred, model.n_iter_, train_sec, infer_sec


def main():
    pool = cv.load_full_pool_with_folds().copy()

    # Verify the evaluator scores are binary correctness values.
    for col in ["deepseek_score", "qwen_score"]:
        vals = sorted(pool[col].dropna().unique().tolist())
        print(f"{col} unique values: {vals}")
        if not set(vals).issubset({0, 0.0, 1, 1.0}):
            raise RuntimeError(
                f"{col} is not binary correctness: {vals}"
            )

    pool["y3"] = pool["target"].map(THREE_MAP).astype(int)

    pool["y4"] = [
        make_four_state(ds, qw)
        for ds, qw in zip(
            pool["deepseek_score"],
            pool["qwen_score"],
        )
    ]

    print()
    print(f"pool={len(pool)}")
    print(f"folds={sorted(pool['fold'].unique())}")

    print("\nFOUR-STATE COUNTS")
    for k, name in enumerate(FOUR_NAMES):
        n = int((pool["y4"] == k).sum())
        print(f"{name:16s} {n:5d}  {n/len(pool):.1%}")

    cache = cv.load_embedding_cache(
        cv.MINILM_CACHE,
        cv.MINILM_MODEL,
    )
    print(f"\nMiniLM cached={len(cache)}")
    print()

    fold_rows = []
    preds = []
    four_rows = []

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

        y3tr = train["y3"].to_numpy()
        y3te = test["y3"].to_numpy()

        y4tr = train["y4"].to_numpy()
        y4te = test["y4"].to_numpy()

        # Existing 3-class formulation.
        pred3, e3, t3, i3 = fit_mlp(
            Xtr, y3tr, Xte,
            42 + int(fold),
        )

        # New 4-state formulation.
        pred4, e4, t4, i4 = fit_mlp(
            Xtr, y4tr, Xte,
            142 + int(fold),
        )

        collapsed4 = collapse_four(pred4)

        methods = [
            (
                "Direct 3-class MLP-64",
                pred3, e3, t3, i3
            ),
            (
                "4-state -> 3-class MLP-64",
                collapsed4, e4, t4, i4
            ),
        ]

        for name, pred, epochs, tr_sec, inf_sec in methods:
            m = three_metrics(y3te, pred)

            fold_rows.append({
                "fold": fold,
                "method": name,
                **m,
                "epochs": epochs,
                "train_sec": tr_sec,
                "infer_sec": inf_sec,
            })

            print(
                f"fold {fold} | {name:25s} "
                f"F1={m['macro_f1']:.4f} "
                f"DS={m['f1_deepseek']:.4f} "
                f"Tie={m['f1_tie']:.4f} "
                f"Qwen={m['f1_qwen']:.4f}"
            )

            for j, idx in enumerate(test.index):
                preds.append({
                    "row_index": idx,
                    "fold": fold,
                    "dataset": test.loc[idx, "dataset"],
                    "target": int(y3te[j]),
                    "method": name,
                    "prediction": int(pred[j]),
                })

        fm = four_metrics(y4te, pred4)

        four_rows.append({
            "fold": fold,
            **fm,
        })

        print(
            f"         4-state F1={fm['four_macro_f1']:.4f} "
            f"both-wrong={fm['f1_both_wrong']:.4f} "
            f"DS-only={fm['f1_deepseek_only']:.4f} "
            f"Qwen-only={fm['f1_qwen_only']:.4f} "
            f"both-correct={fm['f1_both_correct']:.4f}"
        )
        print()

    fdf = pd.DataFrame(fold_rows)
    pdf = pd.DataFrame(preds)
    four_df = pd.DataFrame(four_rows)

    fdf.to_csv(OUT / "folds.csv", index=False)
    pdf.to_csv(OUT / "predictions.csv", index=False)
    four_df.to_csv(OUT / "four_state_folds.csv", index=False)

    print("=" * 76)
    print("POOLED HELD-OUT ROUTING RESULTS")
    print("=" * 76)

    summary = []

    for method in [
        "Direct 3-class MLP-64",
        "4-state -> 3-class MLP-64",
    ]:
        d = pdf[pdf["method"] == method]

        y = d["target"].to_numpy()
        p = d["prediction"].to_numpy()

        m = three_metrics(y, p)
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
        print(f"  Macro-F1:          {m['macro_f1']:.4f}")
        print(
            f"  Balanced accuracy: "
            f"{m['balanced_accuracy']:.4f}"
        )
        print(f"  Accuracy:          {m['accuracy']:.4f}")
        print(f"  F1 DeepSeek:       {m['f1_deepseek']:.4f}")
        print(f"  F1 Tie:            {m['f1_tie']:.4f}")
        print(f"  F1 Qwen:           {m['f1_qwen']:.4f}")
        print(
            f"  Fold Macro-F1:     "
            f"{fd['macro_f1'].mean():.4f} +/- "
            f"{fd['macro_f1'].std(ddof=1):.4f}"
        )
        print("  Confusion matrix:")
        print(
            confusion_matrix(
                y, p,
                labels=THREE_LABELS
            )
        )

    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "summary.csv", index=False)

    base = summary[
        summary["method"] == "Direct 3-class MLP-64"
    ].iloc[0]

    four = summary[
        summary["method"] == "4-state -> 3-class MLP-64"
    ].iloc[0]

    delta = four["macro_f1"] - base["macro_f1"]

    print()
    print("=" * 76)
    print("FOUR-STATE DIAGNOSTIC")
    print("=" * 76)

    print(
        f"4-state Macro-F1 mean: "
        f"{four_df['four_macro_f1'].mean():.4f}"
    )
    print(
        f"Both-wrong F1 mean:    "
        f"{four_df['f1_both_wrong'].mean():.4f}"
    )
    print(
        f"DeepSeek-only F1 mean: "
        f"{four_df['f1_deepseek_only'].mean():.4f}"
    )
    print(
        f"Qwen-only F1 mean:     "
        f"{four_df['f1_qwen_only'].mean():.4f}"
    )
    print(
        f"Both-correct F1 mean:  "
        f"{four_df['f1_both_correct'].mean():.4f}"
    )

    print()
    print("=" * 76)
    print("PREDECLARED COMPARISON")
    print("=" * 76)

    print(
        f"4-state collapsed - direct 3-class "
        f"Macro-F1 delta: {delta:+.4f}"
    )

    print(
        "Criterion: collapsed 4-state >= direct 3-class "
        "+0.02 Macro-F1 and non-zero F1 for both winner classes."
    )

    promising = (
        delta >= 0.02
        and four["f1_deepseek"] > 0
        and four["f1_qwen"] > 0
    )

    print(f"PROMISING = {bool(promising)}")
    print(f"Outputs: {OUT}")


if __name__ == "__main__":
    main()
