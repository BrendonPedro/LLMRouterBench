#!/usr/bin/env python3
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Normalizer
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cv_eval as cv

OUT = cv.OUT / "hierarchical_router_mlp64"
OUT.mkdir(parents=True, exist_ok=True)

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


def three_class_metrics(y, p):
    per = f1_score(
        y, p,
        labels=LABELS,
        average=None,
        zero_division=0
    )
    return {
        "macro_f1": f1_score(
            y, p,
            labels=LABELS,
            average="macro",
            zero_division=0
        ),
        "balanced_accuracy":
            balanced_accuracy_score(y, p),
        "accuracy": accuracy_score(y, p),
        "f1_deepseek": per[0],
        "f1_tie": per[1],
        "f1_qwen": per[2],
    }


def train_flat(Xtr, ytr, Xte, seed):
    Xb, yb = balanced_resample(Xtr, ytr, seed)
    clf = make_mlp(seed)

    t0 = time.perf_counter()
    clf.fit(Xb, yb)
    train_sec = time.perf_counter() - t0

    return clf.predict(Xte), train_sec, clf.n_iter_


def train_hierarchical(Xtr, target_tr, Xte, seed):
    # Stage 1:
    # 0 = tie
    # 1 = non-tie
    y_non = (target_tr != -1).astype(int)

    X1, y1 = balanced_resample(
        Xtr, y_non, seed
    )

    gate = make_mlp(seed)

    t0 = time.perf_counter()
    gate.fit(X1, y1)

    # Stage 2:
    # train ONLY on real non-tie rows
    nt = target_tr != -1
    X2raw = Xtr[nt]

    # Existing labels are already:
    # 0 = DeepSeek
    # 1 = Qwen
    y2raw = target_tr[nt].astype(int)

    X2, y2 = balanced_resample(
        X2raw, y2raw, seed + 100
    )

    winner = make_mlp(seed + 100)
    winner.fit(X2, y2)

    train_sec = time.perf_counter() - t0

    # Soft hierarchical prediction.
    p_non = gate.predict_proba(Xte)[:, 1]
    p_qwen_given_non = winner.predict_proba(Xte)[:, 1]

    p_ds = p_non * (1.0 - p_qwen_given_non)
    p_tie = 1.0 - p_non
    p_qwen = p_non * p_qwen_given_non

    probs = np.column_stack([
        p_ds, p_tie, p_qwen
    ])

    pred = probs.argmax(axis=1)

    return (
        pred,
        p_non,
        p_qwen_given_non,
        train_sec,
        gate.n_iter_,
        winner.n_iter_,
    )


def main():
    pool = cv.load_full_pool_with_folds().copy()
    pool["y3"] = pool["target"].map(ORD_MAP).astype(int)

    cache = cv.load_embedding_cache(
        cv.MINILM_CACHE,
        cv.MINILM_MODEL
    )

    print(f"pool={len(pool)}")
    print(f"folds={sorted(pool.fold.unique())}")
    print(f"MiniLM cached={len(cache)}")
    print()

    final_rows = []
    stage_rows = []
    all_preds = []

    for fold in sorted(pool.fold.unique()):
        train = pool[pool.fold != fold].copy()
        test = pool[pool.fold == fold].copy()

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

        target_tr = train["target"].to_numpy()
        target_te = test["target"].to_numpy()

        y3tr = train["y3"].to_numpy()
        y3te = test["y3"].to_numpy()

        # Flat comparison.
        p_flat, flat_sec, flat_epochs = train_flat(
            Xtr, y3tr, Xte,
            42 + int(fold)
        )

        # Hierarchical router.
        (
            p_hier,
            p_non,
            p_qwen_non,
            hier_sec,
            gate_epochs,
            winner_epochs,
        ) = train_hierarchical(
            Xtr,
            target_tr,
            Xte,
            142 + int(fold)
        )

        for name, pred, sec in [
            ("Flat MLP-64", p_flat, flat_sec),
            ("Hierarchical MLP-64", p_hier, hier_sec),
        ]:
            m = three_class_metrics(y3te, pred)

            final_rows.append({
                "fold": fold,
                "method": name,
                **m,
                "train_sec": sec,
            })

            print(
                f"fold {fold} | {name:19s} "
                f"F1={m['macro_f1']:.4f} "
                f"DS={m['f1_deepseek']:.4f} "
                f"Tie={m['f1_tie']:.4f} "
                f"Qwen={m['f1_qwen']:.4f}"
            )

            for j, idx in enumerate(test.index):
                all_preds.append({
                    "row_index": idx,
                    "fold": fold,
                    "dataset": test.loc[idx, "dataset"],
                    "target": int(y3te[j]),
                    "method": name,
                    "prediction": int(pred[j]),
                })

        # Stage 1 held-out quality.
        true_non = (target_te != -1).astype(int)
        pred_non = (p_non >= 0.5).astype(int)

        gate_f1 = f1_score(
            true_non, pred_non,
            average="macro",
            zero_division=0
        )
        gate_auc = roc_auc_score(
            true_non, p_non
        )

        # Stage 2 quality only on true held-out non-ties.
        nt = target_te != -1

        true_winner = target_te[nt].astype(int)
        winner_score = p_qwen_non[nt]
        winner_pred = (
            winner_score >= 0.5
        ).astype(int)

        winner_f1 = f1_score(
            true_winner,
            winner_pred,
            average="macro",
            zero_division=0
        )

        winner_auc = roc_auc_score(
            true_winner,
            winner_score
        )

        stage_rows.append({
            "fold": fold,
            "gate_macro_f1": gate_f1,
            "gate_auc": gate_auc,
            "winner_macro_f1": winner_f1,
            "winner_auc": winner_auc,
            "gate_epochs": gate_epochs,
            "winner_epochs": winner_epochs,
        })

        print(
            f"         Stage1 tie/non-tie: "
            f"F1={gate_f1:.4f} AUC={gate_auc:.4f}"
        )
        print(
            f"         Stage2 DS/Qwen:     "
            f"F1={winner_f1:.4f} "
            f"AUC={winner_auc:.4f}"
        )
        print()

    fdf = pd.DataFrame(final_rows)
    sdf = pd.DataFrame(stage_rows)
    pdf = pd.DataFrame(all_preds)

    fdf.to_csv(OUT / "folds.csv", index=False)
    sdf.to_csv(OUT / "stages.csv", index=False)
    pdf.to_csv(OUT / "predictions.csv", index=False)

    print("=" * 72)
    print("POOLED HELD-OUT RESULTS")
    print("=" * 72)

    summary = []

    for method in [
        "Flat MLP-64",
        "Hierarchical MLP-64"
    ]:
        d = pdf[pdf.method == method]

        y = d.target.to_numpy()
        p = d.prediction.to_numpy()

        m = three_class_metrics(y, p)

        fd = fdf[fdf.method == method]

        summary.append({
            "method": method,
            **m,
            "fold_macro_f1_mean":
                fd.macro_f1.mean(),
            "fold_macro_f1_std":
                fd.macro_f1.std(ddof=1),
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
            f"{fd.macro_f1.mean():.4f} "
            f"+/- {fd.macro_f1.std(ddof=1):.4f}"
        )
        print("  Confusion matrix:")
        print(
            confusion_matrix(
                y, p,
                labels=LABELS
            )
        )

    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "summary.csv", index=False)

    flat = summary[
        summary.method == "Flat MLP-64"
    ].iloc[0]

    hier = summary[
        summary.method == "Hierarchical MLP-64"
    ].iloc[0]

    delta = (
        hier.macro_f1 -
        flat.macro_f1
    )

    print()
    print("=" * 72)
    print("STAGE DIAGNOSTICS")
    print("=" * 72)

    print(
        f"Tie/non-tie gate Macro-F1: "
        f"{sdf.gate_macro_f1.mean():.4f}"
    )
    print(
        f"Tie/non-tie gate AUC:      "
        f"{sdf.gate_auc.mean():.4f}"
    )
    print(
        f"Winner stage Macro-F1:     "
        f"{sdf.winner_macro_f1.mean():.4f}"
    )
    print(
        f"Winner stage AUC:          "
        f"{sdf.winner_auc.mean():.4f}"
    )

    print()
    print("=" * 72)
    print("PREDECLARED COMPARISON")
    print("=" * 72)

    print(
        f"Hierarchical - flat "
        f"Macro-F1 delta: {delta:+.4f}"
    )

    print(
        "Criterion: hierarchical >= flat "
        "+0.02 Macro-F1."
    )

    print(
        f"PROMISING = "
        f"{bool(delta >= 0.02)}"
    )

    print(f"Outputs: {OUT}")


if __name__ == "__main__":
    main()
