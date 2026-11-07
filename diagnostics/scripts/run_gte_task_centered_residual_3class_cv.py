#!/usr/bin/env python3
"""Synthesis §9: task-centred residual head + predicted-task prior (3-class).

B0 = run_gte_predicted_task_bias_cv.py arm "Predicted-task bias", rerun verbatim.
C  = identical, except the router input: outer-train rows are centred by their
     true-benchmark centroid (outer-train rows only) and L2-renormalised; test
     rows are centred by the centroid of their RidgeClassifier-predicted
     benchmark (recipe of run_gte_predicted_centered_winner_cv.py).

All model/bias/metric logic is imported from the two reference scripts.
CPU only, cached GTE embeddings only.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import Normalizer
from sklearn.metrics import roc_auc_score, accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cv_eval as cv
import run_gte_predicted_task_bias_cv as b0  # router, probs3, adjust, choose_bias, add, evaluate, MAP3, BIAS_GRID
import run_gte_predicted_centered_winner_cv as cw  # center

OUT = cv.OUT / "gte_task_centered_residual_3class"
OUT.mkdir(parents=True, exist_ok=True)

B0 = "B0 Predicted-task bias"
C = "C Task-centred residual + predicted-task bias"
METHODS = [B0, C]

REF_WITHIN, REF_OVERALL, TOL = 0.3680, 0.4150, 1e-4


def fit_arm(Xtr, ytr, tr_ds, Xte, test_ds_for_bias, outer):
    """Exactly B0's per-fold procedure, on whatever representation is passed."""
    inner = StratifiedKFold(3, shuffle=True, random_state=1000 + int(outer))
    oof = np.zeros((len(Xtr), 3))
    for k, (itr, iva) in enumerate(inner.split(Xtr, ytr)):
        m = b0.router(10000 + outer * 10 + k)
        m.fit(Xtr[itr], ytr[itr])
        oof[iva] = b0.probs3(m, Xtr[iva])

    biases = {}
    for ds in pd.unique(tr_ds):
        msk = tr_ds == ds
        biases[ds] = b0.choose_bias(ytr[msk], oof[msk])

    r = b0.router(42 + int(outer))
    r.fit(Xtr, ytr)
    base = b0.probs3(r, Xte)
    return b0.adjust(base, test_ds_for_bias, biases), biases


def cell_auc(df):
    """Winner AUC inside each (benchmark, fold) cell with both winners present."""
    vals = []
    for (_, _), d in df.groupby(["dataset", "fold"]):
        nt = d.target != 1
        y = (d.loc[nt, "target"] == 2).astype(int).to_numpy()
        if len(np.unique(y)) != 2:
            continue
        ds_, qw = d.loc[nt, "p_ds"].to_numpy(), d.loc[nt, "p_qwen"].to_numpy()
        vals.append(roc_auc_score(y, qw / np.maximum(ds_ + qw, 1e-12)))
    return float(np.mean(vals)), len(vals)


def main():
    pool = cv.load_full_pool_with_folds().copy()
    pool["y3"] = pool.target.map(b0.MAP3).astype(int)
    assert len(pool) == 11467, len(pool)
    cache = cv.load_embedding_cache(cv.GTE_CACHE, cv.GTE_MODEL)

    rows = []
    for outer in sorted(pool.fold.unique()):
        train = pool[pool.fold != outer].copy()
        test = pool[pool.fold == outer].copy()

        norm = Normalizer()
        Xtr = norm.fit_transform(cv.embedding_matrix(train["query"].tolist(), cache))
        Xte = norm.transform(cv.embedding_matrix(test["query"].tolist(), cache))
        ytr = train.y3.to_numpy()
        tr_ds = train.dataset.to_numpy()

        # Same task predictor as B0 (raw normalised GTE, outer-train only).
        task = RidgeClassifier(alpha=1.0, class_weight="balanced")
        task.fit(Xtr, tr_ds)
        pred_ds = task.predict(Xte)

        # ---- B0 (unchanged representation) ----
        p_b0, _ = fit_arm(Xtr, ytr, tr_ds, Xte, pred_ds, outer)

        # ---- C (only the router input changes) ----
        centroids = {ds: Xtr[tr_ds == ds].mean(axis=0) for ds in np.unique(tr_ds)}
        Xtr_c = cw.center(Xtr, tr_ds, centroids)      # true train benchmark
        Xte_c = cw.center(Xte, pred_ds, centroids)    # predicted test benchmark
        p_c, _ = fit_arm(Xtr_c, ytr, tr_ds, Xte_c, pred_ds, outer)

        for name, p in [(B0, p_b0), (C, p_c)]:
            tmp = []
            b0.add(tmp, test, p, name)
            for r in tmp:
                r["fold"] = int(outer)
            rows.extend(tmp)

        print(f"fold {outer}: task accuracy={accuracy_score(test.dataset, pred_ds):.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "predictions.csv", index=False)

    # ---- metrics ----
    res = {}
    for m in METHODS:
        d = df[df.method == m]
        overall, within, pooled_auc = b0.evaluate(d)
        cauc, ncell = cell_auc(d)
        per_fold = {k: b0.evaluate(d[d.fold == k])[1] for k in sorted(d.fold.unique())}
        res[m] = dict(overall=overall, within=within, pooled_auc=pooled_auc,
                      cell_auc=cauc, n_cells=ncell, per_fold=per_fold)

    print("\n" + "=" * 72)
    for m in METHODS:
        r = res[m]
        print(m)
        print(f"  Overall Macro-F1:            {r['overall']:.4f}")
        print(f"  Within-benchmark Macro-F1:   {r['within']:.4f}")
        print(f"  Cell AUC (bench x fold):     {r['cell_auc']:.4f}  (n_cells={r['n_cells']})")
        print(f"  Pooled-per-bench AUC (desc): {r['pooled_auc']:.4f}")

    b, c = res[B0], res[C]
    reproduced = abs(b["within"] - REF_WITHIN) <= TOL and abs(b["overall"] - REF_OVERALL) <= TOL
    print("\nB0 REPRODUCTION")
    print(f"  within {b['within']:.6f} vs {REF_WITHIN}  overall {b['overall']:.6f} vs {REF_OVERALL}  -> {'OK' if reproduced else 'VOID'}")

    print("\nPER-FOLD WITHIN-F1")
    deltas = []
    for k in b["per_fold"]:
        dlt = c["per_fold"][k] - b["per_fold"][k]
        deltas.append(dlt)
        print(f"  fold {k}: B0={b['per_fold'][k]:.4f}  C={c['per_fold'][k]:.4f}  delta={dlt:+.4f}")

    d_within = c["within"] - b["within"]
    n_pos = sum(x > 0 for x in deltas)
    d_cell = c["cell_auc"] - b["cell_auc"]
    crit1 = d_within >= 0.020
    crit2 = n_pos >= 4
    crit3 = c["overall"] >= 0.4155
    f1_pass = crit1 and crit2 and crit3

    print("\nCRITERIA")
    print(f"  1. within delta {d_within:+.4f} >= +0.020 : {crit1}")
    print(f"  2. positive folds {n_pos}/5 >= 4      : {crit2}")
    print(f"  3. overall(C) {c['overall']:.4f} >= 0.4155 : {crit3}")
    print(f"  cell-AUC delta {d_cell:+.4f} (> 0: {d_cell > 0})")
    print(f"  pooled AUC delta (descriptive) {c['pooled_auc'] - b['pooled_auc']:+.4f}")

    if not reproduced:
        verdict = "VOID"
    elif f1_pass and d_cell > 0:
        verdict = "PASS"
    elif f1_pass:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "FAIL"
    print(f"\nVERDICT = {verdict}")
    print(f"Outputs: {OUT}")


if __name__ == "__main__":
    main()
