"""Fairness correction: MiniLM MLP-64 student (condition B) + training-only threshold.

Condition C's secondary row uses a training-only threshold, while the B
reference was reported only at its fixed score > 0 cutoff. This script adds
"B-threshold" as a calibration comparator. The MiniLM model is unchanged; only
the decision cutoff is selected, from outer-training data only.

Protocol (mirrors A/C's training-only threshold):
* Outer-train out-of-fold student scores are produced EXACTLY as for condition
  C in run_capability_profile_cv.py: inner StratifiedKFold(5, shuffle,
  random_state=42+fold) over outer-train rows stratified on target, refitting
  the GTE teacher (inner-train non-ties) and MLP-64 student (inner-train
  queries) per inner split with seed 1000 + 10*fold + j.
  They are checked against the OOF-student Macro-F1/AUC saved for C.
* Cutoff = the value maximising Macro-F1 of those OOF scores on outer-train
  non-tie rows. The student emits an unbounded regression score rather than a
  probability, so the probability grid 0.05..0.95 used for A/C does not apply;
  candidates are all midpoints between consecutive distinct OOF non-tie scores
  (an exhaustive search; first maximum wins).
* Outer-test labels are used only for the final metrics.

Before anything is written, the regenerated held-out B scores are verified
against heldout_predictions.csv and the saved corrected_nonlinear_cv.csv.

Appends B-threshold rows (per fold, mean, std, pooled) to
diagnostics/capability_profile/fold_results.csv; existing rows are untouched.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import Normalizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_capability_profile_cv as rcp  # noqa: E402
import run_corrected_nonlinear_cv as rc  # noqa: E402
import run_cv_eval as cv  # noqa: E402

B_THR_RULE = "training_only_threshold"
C_THR_RULE = "training_only_threshold"


def select_threshold(scores, y):
    u = np.unique(scores)
    cands = np.concatenate([[0.0], (u[:-1] + u[1:]) / 2])
    best_t, best_f1 = 0.0, -1.0
    for t in cands:
        f = f1_score(y, (scores > t).astype(int), average="macro", zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    return best_t, best_f1


def main():
    fr = pd.read_csv(rcp.FOLD_CSV)
    assert not ((fr["method"] == rcp.B_NAME) & (fr["decision_rule"] == B_THR_RULE)).any(), \
        "B-threshold rows already present"
    preds = pd.read_csv(rcp.PRED_CSV)
    ref_b = pd.read_csv(rcp.REF_B_CSV)
    ref_b = ref_b[ref_b["method"] == rc.NONLINEAR_NAME].set_index("fold")
    c_rows = fr[(fr["method"] == rcp.C_NAME) & (fr["decision_rule"] == C_THR_RULE)
                & fr["fold"].isin([str(k) for k in range(cv.K)])].set_index("fold")

    pool = cv.load_full_pool_with_folds()
    gte_cache = cv.load_embedding_cache(cv.GTE_CACHE, cv.GTE_MODEL)
    minilm_cache = cv.load_embedding_cache(cv.MINILM_CACHE, cv.MINILM_MODEL)
    Xg_all = Normalizer().fit_transform(cv.embedding_matrix(pool["query"].tolist(), gte_cache))
    Xm_all = Normalizer().fit_transform(cv.embedding_matrix(pool["query"].tolist(), minilm_cache))
    del gte_cache, minilm_cache
    target = pool["target"].to_numpy()

    rows, heldout = [], []
    for fold in range(cv.K):
        seed = 42 + fold
        tr_idx = np.flatnonzero(pool["fold"].to_numpy() != fold)
        te_idx = np.flatnonzero(pool["fold"].to_numpy() == fold)
        tr_nt, te_nt = target[tr_idx] != -1, target[te_idx] != -1
        y_tr, y_te = target[tr_idx][tr_nt], target[te_idx][te_nt]

        # --- held-out B scores: regenerate and verify ---
        student = rcp.fit_teacher_student(Xg_all[tr_idx], Xm_all[tr_idx], target[tr_idx], seed)
        s_te = student.predict(Xm_all[te_idx])
        saved = preds[preds["fold"] == fold]
        assert (saved["dataset"].to_numpy() == pool["dataset"].to_numpy()[te_idx]).all()
        assert (saved["index"].to_numpy() == pool["index"].to_numpy()[te_idx]).all()
        max_diff = float(np.max(np.abs(saved["B_student_score"].to_numpy() - s_te)))
        f1_def = f1_score(y_te, (s_te[te_nt] > 0).astype(int), average="macro")
        assert max_diff < 1e-6, f"fold {fold}: held-out B scores differ by {max_diff}"
        assert abs(f1_def - ref_b.loc[str(fold), "macro_f1"]) < 1e-9

        # --- outer-train OOF student scores (identical to condition C) ---
        oof = np.full(len(tr_idx), np.nan)
        inner = StratifiedKFold(n_splits=rcp.INNER_K, shuffle=True, random_state=seed)
        for j, (a, b) in enumerate(inner.split(np.zeros(len(tr_idx)), target[tr_idx])):
            st = rcp.fit_teacher_student(Xg_all[tr_idx[a]], Xm_all[tr_idx[a]], target[tr_idx[a]],
                                         seed=1000 + 10 * fold + j)
            oof[b] = st.predict(Xm_all[tr_idx[b]])
        oof_f1 = f1_score(y_tr, (oof[tr_nt] > 0).astype(int), average="macro")
        oof_auc = roc_auc_score(y_tr, oof[tr_nt])
        assert abs(oof_f1 - c_rows.loc[str(fold), "oof_student_train_macro_f1"]) < 1e-9
        assert abs(oof_auc - c_rows.loc[str(fold), "oof_student_train_auc"]) < 1e-9

        thr, thr_oof_f1 = select_threshold(oof[tr_nt], y_tr)
        pred = (s_te > thr).astype(int)
        row = rcp.metric_row(rcp.B_NAME, B_THR_RULE, fold, y_te, pred[te_nt], s_te[te_nt], {
            "n_train_nontie": int(tr_nt.sum()), "threshold": thr,
            "train_oof_macro_f1_at_threshold": thr_oof_f1,
            "oof_student_train_macro_f1": oof_f1, "oof_student_train_auc": oof_auc,
        })
        row["delta_macro_f1_vs_B"] = row["macro_f1"] - f1_def
        row["delta_auc_vs_B"] = 0.0
        rows.append(row)
        heldout.append(pd.DataFrame({"target": target[te_idx][te_nt], "pred": pred[te_nt], "score": s_te[te_nt]}))
        c = c_rows.loc[str(fold)]
        print(f"fold {fold}: verified (max |dscore|={max_diff:.1e}, default F1={f1_def:.4f})  "
              f"thr={thr:+.4f}  B-thr F1={row['macro_f1']:.4f}  C-thr F1={c['macro_f1']:.4f}  "
              f"C-B_thr={c['macro_f1'] - row['macro_f1']:+.4f}")

    df = pd.DataFrame(rows)
    num = [c for c in df.columns if c not in ("method", "decision_rule", "fold")]
    agg = [{"method": rcp.B_NAME, "decision_rule": B_THR_RULE, "fold": stat,
            **{c: float(fn(df[c].to_numpy(dtype=float))) for c in num}}
           for stat, fn in (("mean", np.mean), ("std", lambda v: np.std(v, ddof=1)))]
    H = pd.concat(heldout, ignore_index=True)
    pooled = rcp.metric_row(rcp.B_NAME, B_THR_RULE, "pooled", H["target"], H["pred"], H["score"])

    new = pd.concat([df, pd.DataFrame(agg), pd.DataFrame([pooled])], ignore_index=True)
    pd.concat([fr, new], ignore_index=True).to_csv(rcp.FOLD_CSV, index=False)

    cm = c_rows["macro_f1"].astype(float).to_numpy()
    bm = df["macro_f1"].to_numpy()
    print(f"\nB default   F1={ref_b.loc[[str(k) for k in range(cv.K)], 'macro_f1'].astype(float).mean():.4f}")
    print(f"B-threshold F1={bm.mean():.4f} +/- {bm.std(ddof=1):.4f}   AUC={df['auc'].mean():.4f}")
    print(f"C-threshold F1={cm.mean():.4f}   C - B_thr mean={np.mean(cm - bm):+.4f}   "
          f"C wins {int((cm > bm).sum())}/5")
    print(f"Appended {len(new)} rows to {rcp.FOLD_CSV}")


if __name__ == "__main__":
    main()
