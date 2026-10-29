"""Corrected CV Mission -- Task 1: nonlinear MiniLM score distillation, 5-fold CV.

The "Frozen-MiniLM distillation" row in router_cv_results.csv (0.5922 mean
Macro-F1) was produced by run_cv_eval.run_distillation, whose student is
Ridge(alpha=1.0): the OLD LINEAR score-distillation student. This script runs
the actual previously-defined best lightweight student on the SAME frozen folds:

    frozen all-MiniLM-L6-v2 (384-D, L2-norm)
      -> MLPRegressor(hidden=64, ReLU)
      -> scalar GTE teacher routing score (decision_function)

with the hyperparameters copied verbatim from
local_experiments/train_minilm_nonlinear_score_distillation.py (defaults):
    hidden_layer_sizes=(64,), activation="relu", solver="adam", alpha=1e-4,
    batch_size=256, learning_rate_init=1e-3, max_iter=300, early_stopping=True,
    validation_fraction=0.1, n_iter_no_change=20
    teacher: LogisticRegression(C=1.0, l2, liblinear, max_iter=2000)
    decision rule: student score > 0 -> Qwen  (fixed, never tuned)
random_state = 42 + fold, matching the run_cv_eval.py per-fold convention.

The linear Ridge student is re-run alongside on the identical teacher scores so
the two students are compared like-for-like (and the 0.5922 is reproduced).

Anti-leakage:
  * Teacher fit on outer-TRAIN non-tie rows only; it then scores ALL outer-train
    queries (ties included) as the regression target.
  * Student fit on outer-train queries only. Early stopping uses a 10% split of
    the outer-train portion (MLPRegressor internal), never the outer test fold.
  * No threshold / epoch / hyperparameter is selected from outer-test labels.
  * Outer-test labels are used only for the final Macro-F1 on non-tie rows.

Timings recorded here are HEAD-ONLY on precomputed (cached) embeddings; they are
NOT end-to-end latency (no tokenizer / encoder forward pass is timed).

Writes diagnostics/stratified_cv/corrected_nonlinear_cv.csv (new file; no
existing result is overwritten).
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import Normalizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cv_eval as cv  # noqa: E402

OUT_CSV = cv.OUT / "corrected_nonlinear_cv.csv"

HIDDEN = 64
MINILM_DIM = 384

LINEAR_NAME = "Frozen-MiniLM Ridge student (linear; previous 'distillation' row)"
NONLINEAR_NAME = "Frozen-MiniLM MLP-64 student (nonlinear; corrected)"


def make_student_mlp(fold):
    return MLPRegressor(
        hidden_layer_sizes=(HIDDEN,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42 + fold,
    )


def run_fold(train, test, gte_cache, minilm_cache, fold):
    tr_nt_mask = (train["target"] != -1).to_numpy()
    te_nt_mask = (test["target"] != -1).to_numpy()
    y_true = test.loc[te_nt_mask, "target"].to_numpy()

    # ---- teacher: GTE direct LogReg, outer-train non-tie only ----
    gte_norm = Normalizer(norm="l2")
    Xg_tr = gte_norm.fit_transform(cv.embedding_matrix(train["query"].tolist(), gte_cache))
    Xg_te = gte_norm.transform(cv.embedding_matrix(test["query"].tolist(), gte_cache))
    teacher = LogisticRegression(C=1.0, penalty="l2", solver="liblinear",
                                 max_iter=2000, random_state=42 + fold, class_weight=None)
    teacher.fit(Xg_tr[tr_nt_mask], train.loc[tr_nt_mask, "target"].to_numpy())
    teacher_train_score = teacher.decision_function(Xg_tr)  # ALL outer-train rows
    # teacher test score is only used for transfer diagnostics, never for fitting
    teacher_test_score = teacher.decision_function(Xg_te)
    teacher_test_cls = (teacher_test_score > 0).astype(int)
    teacher_metrics = cv.classification_metrics(y_true, teacher_test_cls[te_nt_mask])

    # ---- frozen MiniLM features ----
    mini_norm = Normalizer(norm="l2")
    Xm_tr = mini_norm.fit_transform(cv.embedding_matrix(train["query"].tolist(), minilm_cache))
    Xm_te = mini_norm.transform(cv.embedding_matrix(test["query"].tolist(), minilm_cache))

    students = {
        LINEAR_NAME: (Ridge(alpha=1.0, fit_intercept=True), MINILM_DIM + 1),
        NONLINEAR_NAME: (make_student_mlp(fold), MINILM_DIM * HIDDEN + HIDDEN + HIDDEN + 1),
    }
    rows = []
    for name, (student, n_params) in students.items():
        t0 = time.perf_counter()
        student.fit(Xm_tr, teacher_train_score)
        head_train_sec = time.perf_counter() - t0

        t0 = time.perf_counter()
        s_te = student.predict(Xm_te)
        head_infer_sec = time.perf_counter() - t0

        cls = (s_te > 0).astype(int)
        m = cv.classification_metrics(y_true, cls[te_nt_mask])
        rows.append({
            "method": name,
            "fold": fold,
            **m,
            "auc": float(roc_auc_score(y_true, s_te[te_nt_mask])),
            "teacher_macro_f1": teacher_metrics["macro_f1"],
            "teacher_student_score_corr": float(np.corrcoef(s_te, teacher_test_score)[0, 1]),
            "teacher_student_choice_agreement": float((cls == teacher_test_cls).mean()),
            "epochs_used": (student.n_iter_ if isinstance(student, MLPRegressor) else np.nan),
            "head_params": n_params,
            "head_train_sec_cached_emb": head_train_sec,
            "head_infer_sec_cached_emb_full_test_fold": head_infer_sec,
            "n_test_all_rows": int(len(test)),
        })
    return rows


def main():
    pool = cv.load_full_pool_with_folds()
    print(f"pool={len(pool)} folds={sorted(pool['fold'].unique())}")
    gte_cache = cv.load_embedding_cache(cv.GTE_CACHE, cv.GTE_MODEL)
    minilm_cache = cv.load_embedding_cache(cv.MINILM_CACHE, cv.MINILM_MODEL)
    print(f"GTE cached={len(gte_cache)}  MiniLM cached={len(minilm_cache)}")

    rows = []
    for fold in range(cv.K):
        test = pool[pool["fold"] == fold].copy()
        train = pool[pool["fold"] != fold].copy()
        fold_rows = run_fold(train, test, gte_cache, minilm_cache, fold)
        for r in fold_rows:
            print(f"fold {fold} {r['method'][:40]:40s} F1={r['macro_f1']:.4f} "
                  f"AUC={r['auc']:.4f} teacherF1={r['teacher_macro_f1']:.4f} "
                  f"epochs={r['epochs_used']}")
        rows.extend(fold_rows)

    df = pd.DataFrame(rows)
    num_cols = [c for c in df.columns if c not in ("method", "fold")]
    agg = []
    for name in (LINEAR_NAME, NONLINEAR_NAME):
        sub = df[df["method"] == name]
        for stat, fn in (("mean", np.mean), ("std", lambda v: np.std(v, ddof=1))):
            agg.append({"method": name, "fold": stat,
                        **{c: float(fn(sub[c].to_numpy(dtype=float))) for c in num_cols}})
    out = pd.concat([df, pd.DataFrame(agg)], ignore_index=True)

    assert not OUT_CSV.exists(), f"refusing to overwrite {OUT_CSV}"
    out.to_csv(OUT_CSV, index=False)

    print("\n5-fold Macro-F1 (held-out non-tie)")
    for name in (LINEAR_NAME, NONLINEAR_NAME):
        sub = df[df["method"] == name]["macro_f1"]
        print(f"  {name:70s} {sub.mean():.4f} +/- {sub.std(ddof=1):.4f}")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
