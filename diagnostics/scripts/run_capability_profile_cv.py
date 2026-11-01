"""Capability-profile routing experiment (CAPABILITY_PROFILE_MISSION.md).

Frozen protocol: the existing stratified 5-fold assignment
(diagnostics/stratified_cv/fold_assignments.csv) -- never regenerated.
Metrics are computed on held-out NON-TIE rows only (0 = DeepSeek wins,
1 = Qwen wins).

Conditions
----------
A. capability-profile-v1 (15 features) -> LogisticRegression
B. frozen MiniLM -> MLP-64 routing-score distillation student (reference).
   The fold-level held-out scores are regenerated with the exact
   run_corrected_nonlinear_cv.py configuration (GTE LogReg teacher fit on
   outer-train non-ties, MLP-64 student fit on all outer-train queries,
   random_state = 42 + fold, decision: score > 0) so they can be aligned
   row-by-row with C. The per-fold Macro-F1 is asserted to match the saved
   diagnostics/stratified_cv/corrected_nonlinear_cv.csv (0.6003 +/- 0.0154).
C. capability-profile-v1 + 1 MiniLM student score -> LogisticRegression.

Leakage rules for C
-------------------
* Outer-train student scores used to fit the combiner are OUT-OF-FOLD inside
  the outer-train data: an inner StratifiedKFold(5) over outer-train rows; in
  each inner split BOTH the GTE teacher (inner-train non-ties) and the MLP-64
  student (inner-train queries) are refit, then score the inner held-out rows.
  No outer-train row's score comes from a teacher or student that saw it.
* Outer-test student scores come from the condition-B teacher/student fit on
  the full outer-train fold only.
* Scaling statistics, LogReg coefficients and the training-only threshold all
  use outer-train data only.

Preprocessing / model (fixed, no search)
----------------------------------------
* Continuous/count features (capability_profile_v1.CONTINUOUS_FEATURES, plus
  the student score in C) -> StandardScaler fit on outer-train non-tie rows.
  Binary features -> passed through as 0/1.
* LogisticRegression(C=1.0, L2, lbfgs, max_iter=5000, class_weight=None),
  trained on outer-train non-tie rows (ties carry no winner label).

Decision rules (pre-declared before running)
--------------------------------------------
* PRIMARY rows for A and C: default cutoff P(Qwen) >= 0.5 (untuned, like the
  B reference's fixed score > 0 rule).
* SECONDARY rows: training-only threshold -- the cutoff maximising Macro-F1 of
  out-of-fold cross_val_predict probabilities on outer-train non-tie rows
  (StratifiedKFold(5), grid 0.05..0.95 step 0.005; same procedure as
  run_cv_eval.run_gte_threshold). For C the OOF combiner probabilities use the
  OOF student scores, so the threshold never sees an in-sample score.
* Mission success criterion is evaluated on both rows and reported as such.

Truncation diagnostic (<=256 vs >256 MiniLM tokens) is computed on held-out
predictions after all fitting is finished; nothing is tuned on it.

Outputs (diagnostics/capability_profile/): fold_results.csv,
coefficients_by_fold.csv, truncation_analysis.csv, efficiency.csv,
heldout_predictions.csv. Existing files are never overwritten.
"""

import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capability_profile_v1 as cp  # noqa: E402
import run_corrected_nonlinear_cv as rc  # noqa: E402
import run_cv_eval as cv  # noqa: E402

OUT_DIR = cv.REPO / "diagnostics" / "capability_profile"
FEATURES_CSV = OUT_DIR / "capability_features_v1.csv"
FOLD_CSV = OUT_DIR / "fold_results.csv"
COEF_CSV = OUT_DIR / "coefficients_by_fold.csv"
TRUNC_CSV = OUT_DIR / "truncation_analysis.csv"
EFF_CSV = OUT_DIR / "efficiency.csv"
PRED_CSV = OUT_DIR / "heldout_predictions.csv"
REF_B_CSV = cv.OUT / "corrected_nonlinear_cv.csv"

STUDENT = "minilm_student_score"
A_NAME = "A: capability-profile-v1 LogReg"
B_NAME = "B: MiniLM MLP-64 student (reference, regenerated fold scores)"
C_NAME = "C: capability-profile-v1 + MiniLM student score LogReg"
INNER_K = 5
MINILM_E2E_MS = 38.0  # existing measured end-to-end CPU reference (minilm_end_to_end_timing.csv)
MINILM_ENCODER_PARAMS = 22_565_376
STUDENT_HEAD_PARAMS = rc.MINILM_DIM * rc.HIDDEN + rc.HIDDEN + rc.HIDDEN + 1  # 24,705


# --------------------------------------------------------------------------- #
# Models                                                                       #
# --------------------------------------------------------------------------- #
def make_logreg(continuous, binary, seed):
    pre = ColumnTransformer(
        [("scale", StandardScaler(), continuous), ("bin", "passthrough", binary)],
        remainder="drop",
    )
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000,
                             class_weight=None, random_state=seed)
    return Pipeline([("pre", pre), ("clf", clf)])


def training_only_threshold(make_model, X_tr, y_tr, seed):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = cross_val_predict(make_model(), X_tr, y_tr, cv=skf, method="predict_proba")[:, 1]
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        f = f1_score(y_tr, (oof >= t).astype(int), average="macro", zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    return best_t, best_f1


def fit_teacher_student(Xg_tr, Xm_tr, target_tr, seed):
    """GTE LogReg teacher on non-ties -> MLP-64 student on all rows (rc config)."""
    nt = target_tr != -1
    teacher = LogisticRegression(C=1.0, penalty="l2", solver="liblinear",
                                 max_iter=2000, random_state=seed, class_weight=None)
    teacher.fit(Xg_tr[nt], target_tr[nt])
    student = rc.make_student_mlp(seed - 42)  # make_student_mlp uses random_state = 42 + arg
    student.fit(Xm_tr, teacher.decision_function(Xg_tr))
    return student


def coef_rows(method, fold, pipe, columns):
    clf = pipe.named_steps["clf"]
    names = list(pipe.named_steps["pre"].transformers_[0][2]) + list(pipe.named_steps["pre"].transformers_[1][2])
    assert sorted(names) == sorted(columns)
    rows = [{"method": method, "fold": fold, "feature": n,
             "scale": "standardized" if i < len(pipe.named_steps["pre"].transformers_[0][2]) else "binary_0_1",
             "coef": float(c)} for i, (n, c) in enumerate(zip(names, clf.coef_[0]))]
    rows.append({"method": method, "fold": fold, "feature": "(intercept)", "scale": "",
                 "coef": float(clf.intercept_[0])})
    return rows


def metric_row(method, decision, fold, y_true, y_pred, score, extra=None):
    m = cv.classification_metrics(y_true, y_pred)
    auc = float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) == 2 else np.nan
    return {"method": method, "decision_rule": decision, "fold": fold, **m, "auc": auc, **(extra or {})}


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    for p in (FOLD_CSV, COEF_CSV, TRUNC_CSV, EFF_CSV, PRED_CSV):
        assert not p.exists(), f"refusing to overwrite {p}"

    pool = cv.load_full_pool_with_folds()
    feats = pd.read_csv(FEATURES_CSV)
    pool = pool.merge(feats, on=["dataset", "index"], how="inner", validate="one_to_one")
    assert len(pool) == len(feats) == 11467
    print(f"pool={len(pool)}  non-tie={(pool['target'] != -1).sum()}")

    gte_cache = cv.load_embedding_cache(cv.GTE_CACHE, cv.GTE_MODEL)
    minilm_cache = cv.load_embedding_cache(cv.MINILM_CACHE, cv.MINILM_MODEL)
    Xg_all = Normalizer().fit_transform(cv.embedding_matrix(pool["query"].tolist(), gte_cache))
    Xm_all = Normalizer().fit_transform(cv.embedding_matrix(pool["query"].tolist(), minilm_cache))
    del gte_cache, minilm_cache  # row-wise L2 norm is stateless, so pre-normalising is fold-safe

    cont_A, bin_A = cp.CONTINUOUS_FEATURES, cp.BINARY_FEATURES
    cont_C, bin_C = cp.CONTINUOUS_FEATURES + [STUDENT], cp.BINARY_FEATURES
    cols_A, cols_C = cp.FEATURES, cp.FEATURES + [STUDENT]

    ref_b = pd.read_csv(REF_B_CSV)
    ref_b = ref_b[ref_b["method"] == rc.NONLINEAR_NAME].set_index("fold")

    fold_rows, coef, preds, models, train_times = [], [], [], {}, []
    target = pool["target"].to_numpy()

    for fold in range(cv.K):
        seed = 42 + fold
        tr_idx = np.flatnonzero(pool["fold"].to_numpy() != fold)
        te_idx = np.flatnonzero(pool["fold"].to_numpy() == fold)
        train, test = pool.iloc[tr_idx].copy(), pool.iloc[te_idx].copy()
        tr_nt = train["target"].to_numpy() != -1
        te_nt = test["target"].to_numpy() != -1
        y_tr = train.loc[tr_nt, "target"].to_numpy()
        y_te = test.loc[te_nt, "target"].to_numpy()

        # ---------------- A: capability-only ----------------
        t0 = time.perf_counter()
        mk_A = lambda: make_logreg(cont_A, bin_A, seed)  # noqa: E731
        pipe_A = mk_A().fit(train.loc[tr_nt, cols_A], y_tr)
        thr_A, oof_f1_A = training_only_threshold(mk_A, train.loc[tr_nt, cols_A], y_tr, seed)
        t_A = time.perf_counter() - t0
        p_A = pipe_A.predict_proba(test[cols_A])[:, 1]

        # ---------------- B: MiniLM student on full outer-train ----------------
        t0 = time.perf_counter()
        student_full = fit_teacher_student(Xg_all[tr_idx], Xm_all[tr_idx], target[tr_idx], seed)
        t_B = time.perf_counter() - t0
        s_te = student_full.predict(Xm_all[te_idx])
        f1_B = f1_score(y_te, (s_te[te_nt] > 0).astype(int), average="macro")
        assert abs(f1_B - ref_b.loc[str(fold), "macro_f1"]) < 1e-9, \
            f"fold {fold}: regenerated B {f1_B:.6f} != saved {ref_b.loc[str(fold), 'macro_f1']:.6f}"

        # ---------------- C: inner OOF student scores on outer-train ----------------
        t0 = time.perf_counter()
        oof_s = np.full(len(tr_idx), np.nan)
        inner = StratifiedKFold(n_splits=INNER_K, shuffle=True, random_state=seed)
        for j, (in_tr, in_ho) in enumerate(inner.split(np.zeros(len(tr_idx)), target[tr_idx])):
            st = fit_teacher_student(Xg_all[tr_idx[in_tr]], Xm_all[tr_idx[in_tr]],
                                     target[tr_idx[in_tr]], seed=1000 + 10 * fold + j)
            oof_s[in_ho] = st.predict(Xm_all[tr_idx[in_ho]])
        assert not np.isnan(oof_s).any()
        train[STUDENT] = oof_s
        test[STUDENT] = s_te
        mk_C = lambda: make_logreg(cont_C, bin_C, seed)  # noqa: E731
        pipe_C = mk_C().fit(train.loc[tr_nt, cols_C], y_tr)
        thr_C, oof_f1_C = training_only_threshold(mk_C, train.loc[tr_nt, cols_C], y_tr, seed)
        t_C = time.perf_counter() - t0 + t_B  # C deploys the full-train student too
        p_C = pipe_C.predict_proba(test[cols_C])[:, 1]

        oof_student_f1 = f1_score(y_tr, (oof_s[tr_nt] > 0).astype(int), average="macro")
        oof_student_auc = roc_auc_score(y_tr, oof_s[tr_nt])
        print(f"fold {fold}: A thr={thr_A:.3f}  C thr={thr_C:.3f}  B F1={f1_B:.4f} (matches saved)  "
              f"OOF-student(outer-train) F1={oof_student_f1:.4f} AUC={oof_student_auc:.4f}")

        # ---------------- metrics ----------------
        cmn = {"n_train_nontie": int(tr_nt.sum())}
        fold_rows += [
            metric_row(A_NAME, "default_p>=0.5", fold, y_te, (p_A[te_nt] >= 0.5).astype(int), p_A[te_nt],
                       {**cmn, "threshold": 0.5, "train_sec": t_A}),
            metric_row(A_NAME, "training_only_threshold", fold, y_te, (p_A[te_nt] >= thr_A).astype(int),
                       p_A[te_nt], {**cmn, "threshold": thr_A, "train_oof_macro_f1_at_threshold": oof_f1_A,
                                    "train_sec": t_A}),
            metric_row(B_NAME, "fixed_score>0", fold, y_te, (s_te[te_nt] > 0).astype(int), s_te[te_nt],
                       {**cmn, "threshold": 0.0, "train_sec": t_B}),
            metric_row(C_NAME, "default_p>=0.5", fold, y_te, (p_C[te_nt] >= 0.5).astype(int), p_C[te_nt],
                       {**cmn, "threshold": 0.5, "train_sec": t_C,
                        "oof_student_train_macro_f1": oof_student_f1, "oof_student_train_auc": oof_student_auc}),
            metric_row(C_NAME, "training_only_threshold", fold, y_te, (p_C[te_nt] >= thr_C).astype(int),
                       p_C[te_nt], {**cmn, "threshold": thr_C, "train_oof_macro_f1_at_threshold": oof_f1_C,
                                    "train_sec": t_C, "oof_student_train_macro_f1": oof_student_f1,
                                    "oof_student_train_auc": oof_student_auc}),
        ]
        coef += coef_rows(A_NAME, fold, pipe_A, cols_A) + coef_rows(C_NAME, fold, pipe_C, cols_C)

        preds.append(pd.DataFrame({
            "fold": fold, "dataset": test["dataset"].to_numpy(), "index": test["index"].to_numpy(),
            "query_sha1": test["query_sha1"].to_numpy(), "target": test["target"].to_numpy(),
            "is_nontie": te_nt.astype(int),
            "minilm_token_count": test["minilm_token_count"].to_numpy(),
            "exceeds_256_tokens": test["exceeds_256_tokens"].to_numpy(),
            "A_prob_qwen": p_A, "A_pred_default": (p_A >= 0.5).astype(int),
            "A_pred_threshold": (p_A >= thr_A).astype(int),
            "B_student_score": s_te, "B_pred": (s_te > 0).astype(int),
            "C_prob_qwen": p_C, "C_pred_default": (p_C >= 0.5).astype(int),
            "C_pred_threshold": (p_C >= thr_C).astype(int),
        }))
        models[fold] = (pipe_A, pipe_C, student_full)
        train_times.append({"fold": fold, "A": t_A, "B": t_B, "C": t_C})

    # ---------------- fold_results.csv ----------------
    df = pd.DataFrame(fold_rows)
    num = [c for c in df.columns if c not in ("method", "decision_rule", "fold")]
    agg = []
    for (mth, rule), sub in df.groupby(["method", "decision_rule"], sort=False):
        for stat, fn in (("mean", np.nanmean), ("std", lambda v: np.nanstd(v, ddof=1))):
            agg.append({"method": mth, "decision_rule": rule, "fold": stat,
                        **{c: float(fn(sub[c].to_numpy(dtype=float))) for c in num}})
    # pooled over all held-out non-tie predictions
    P = pd.concat(preds, ignore_index=True)
    Pn = P[P["is_nontie"] == 1]
    pooled_specs = [(A_NAME, "default_p>=0.5", "A_pred_default", "A_prob_qwen"),
                    (A_NAME, "training_only_threshold", "A_pred_threshold", "A_prob_qwen"),
                    (B_NAME, "fixed_score>0", "B_pred", "B_student_score"),
                    (C_NAME, "default_p>=0.5", "C_pred_default", "C_prob_qwen"),
                    (C_NAME, "training_only_threshold", "C_pred_threshold", "C_prob_qwen")]
    pooled = [metric_row(m, r, "pooled", Pn["target"], Pn[pc], Pn[sc]) for m, r, pc, sc in pooled_specs]
    # paired per-fold deltas vs B (reported on C and A rows)
    b_f1 = df[df["method"] == B_NAME].set_index("fold")["macro_f1"]
    b_auc = df[df["method"] == B_NAME].set_index("fold")["auc"]
    df["delta_macro_f1_vs_B"] = [r.macro_f1 - b_f1[r.fold] for r in df.itertuples()]
    df["delta_auc_vs_B"] = [r.auc - b_auc[r.fold] for r in df.itertuples()]
    out = pd.concat([df, pd.DataFrame(agg), pd.DataFrame(pooled)], ignore_index=True)
    out.to_csv(FOLD_CSV, index=False)

    # ---------------- coefficients_by_fold.csv ----------------
    cf = pd.DataFrame(coef)
    summ = []
    for (mth, feat), sub in cf.groupby(["method", "feature"], sort=False):
        v = sub["coef"].to_numpy()
        summ.append({"method": mth, "fold": "summary", "feature": feat, "scale": sub["scale"].iloc[0],
                     "coef": float(v.mean()), "coef_std": float(v.std(ddof=1)),
                     "n_folds_positive": int((v > 0).sum()), "n_folds_negative": int((v < 0).sum()),
                     "sign_stable_5_of_5": bool((v > 0).all() or (v < 0).all()),
                     "abs_mean_over_std": float(abs(v.mean()) / v.std(ddof=1)) if v.std(ddof=1) > 0 else np.inf})
    pd.concat([cf, pd.DataFrame(summ)], ignore_index=True).to_csv(COEF_CSV, index=False)

    # ---------------- truncation_analysis.csv ----------------
    trows = []
    for grp_name, grp_mask in (("<=256_tokens", Pn["exceeds_256_tokens"] == 0),
                               (">256_tokens", Pn["exceeds_256_tokens"] == 1)):
        for fold in list(range(cv.K)) + ["pooled"]:
            sub = Pn[grp_mask] if fold == "pooled" else Pn[grp_mask & (Pn["fold"] == fold)]
            all_sub = P[(P["exceeds_256_tokens"] == (grp_name == ">256_tokens"))]
            if fold != "pooled":
                all_sub = all_sub[all_sub["fold"] == fold]
            for m, r, pc, sc in pooled_specs:
                row = metric_row(m, r, fold, sub["target"], sub[pc], sub[sc])
                row.update({"token_group": grp_name, "n_test_all_rows": int(len(all_sub)),
                            "tie_rate": float(1 - all_sub["is_nontie"].mean()),
                            "qwen_share_nontie": float(sub["target"].mean())})
                trows.append(row)
    tr = pd.DataFrame(trows)
    tr = tr[["token_group"] + [c for c in tr.columns if c != "token_group"]]
    # fold mean/std per group/method
    tagg = []
    tnum = [c for c in tr.columns if c not in ("token_group", "method", "decision_rule", "fold")]
    per_fold = tr[tr["fold"] != "pooled"]
    for (g, mth, rule), sub in per_fold.groupby(["token_group", "method", "decision_rule"], sort=False):
        for stat, fn in (("mean", np.nanmean), ("std", lambda v: np.nanstd(v, ddof=1))):
            tagg.append({"token_group": g, "method": mth, "decision_rule": rule, "fold": stat,
                         **{c: float(fn(sub[c].to_numpy(dtype=float))) for c in tnum}})
    pd.concat([tr, pd.DataFrame(tagg)], ignore_index=True).to_csv(TRUNC_CSV, index=False)

    # ---------------- heldout_predictions.csv ----------------
    P.to_csv(PRED_CSV, index=False)

    # ---------------- efficiency.csv ----------------
    measure_efficiency(pool, models, train_times)

    # ---------------- console summary ----------------
    print("\n5-fold held-out non-tie Macro-F1 / AUC (mean +/- std)")
    for (mth, rule), sub in df.groupby(["method", "decision_rule"], sort=False):
        print(f"  {mth[:60]:60s} {rule:24s} F1={sub.macro_f1.mean():.4f} +/- {sub.macro_f1.std(ddof=1):.4f}"
              f"  AUC={sub.auc.mean():.4f}  folds>B={(sub.delta_macro_f1_vs_B > 0).sum()}/5")


def measure_efficiency(pool, models, train_times):
    """Single-process CPU timings on fold-0 held-out queries (raw text in)."""
    tok = cp.load_tokenizer()
    q = pool.loc[pool["fold"] == 0, "query"].tolist()
    n = len(q)
    pipe_A, pipe_C, _ = models[0]

    # warm-up
    for t in q[:50]:
        cp.extract(t, len(tok(t, truncation=False)["input_ids"]))

    t0 = time.perf_counter()
    counts = [len(tok(t, truncation=False)["input_ids"]) for t in q]
    tok_ms = (time.perf_counter() - t0) / n * 1e3
    t0 = time.perf_counter()
    feats = [cp.extract(t, c) for t, c in zip(q, counts)]
    regex_ms = (time.perf_counter() - t0) / n * 1e3

    F = pd.DataFrame(feats)[cp.FEATURES]
    # single-query sklearn pipeline inference
    rowsA = [F.iloc[[i]] for i in range(n)]
    t0 = time.perf_counter()
    for r in rowsA:
        pipe_A.predict_proba(r)
    a_single_ms = (time.perf_counter() - t0) / n * 1e3
    t0 = time.perf_counter()
    pipe_A.predict_proba(F)
    a_batch_ms = (time.perf_counter() - t0) / n * 1e3

    # same model as a plain numpy dot product (what a deployed router would run)
    def as_numpy(pipe, cols):
        sc = pipe.named_steps["pre"].named_transformers_["scale"]
        cont = list(pipe.named_steps["pre"].transformers_[0][2])
        binc = list(pipe.named_steps["pre"].transformers_[1][2])
        w = pipe.named_steps["clf"].coef_[0]
        b = pipe.named_steps["clf"].intercept_[0]
        mu = np.concatenate([sc.mean_, np.zeros(len(binc))])
        sd = np.concatenate([sc.scale_, np.ones(len(binc))])
        order = [cols.index(c) for c in cont + binc]
        return w, b, mu, sd, order

    wA, bA, muA, sdA, oA = as_numpy(pipe_A, cp.FEATURES)
    Fa = F.to_numpy(dtype=float)
    t0 = time.perf_counter()
    for i in range(n):
        x = (Fa[i, oA] - muA) / sdA
        1.0 / (1.0 + np.exp(-(x @ wA + bA)))
    a_numpy_ms = (time.perf_counter() - t0) / n * 1e3

    FC = F.copy()
    FC[STUDENT] = 0.0
    colsC = cp.FEATURES + [STUDENT]
    rowsC = [FC.iloc[[i]] for i in range(n)]
    t0 = time.perf_counter()
    for r in rowsC:
        pipe_C.predict_proba(r)
    c_single_ms = (time.perf_counter() - t0) / n * 1e3
    wC, bC, muC, sdC, oC = as_numpy(pipe_C, colsC)
    Fc = FC.to_numpy(dtype=float)
    t0 = time.perf_counter()
    for i in range(n):
        x = (Fc[i, oC] - muC) / sdC
        1.0 / (1.0 + np.exp(-(x @ wC + bC)))
    c_numpy_ms = (time.perf_counter() - t0) / n * 1e3

    nA = int(pipe_A.named_steps["clf"].coef_.size + 1)
    nC = int(pipe_C.named_steps["clf"].coef_.size + 1)
    scA = 2 * len(cp.CONTINUOUS_FEATURES)
    scC = 2 * (len(cp.CONTINUOUS_FEATURES) + 1)
    tt = pd.DataFrame(train_times)

    rows = [
        ("timing_scope", "single process, CPU, fold-0 held-out queries, raw text in, one query per call", "", ""),
        ("n_timed_queries", n, "queries", ""),
        ("capability_extraction_regex_ms_per_query", regex_ms, "ms/query",
         "14 regex/count features given the token count"),
        ("capability_extraction_tokenizer_ms_per_query", tok_ms, "ms/query",
         "MiniLM fast tokenizer, untruncated, for minilm_token_count; free in C (MiniLM pipeline already tokenizes)"),
        ("capability_extraction_total_ms_per_query", regex_ms + tok_ms, "ms/query", "standalone extractor (A)"),
        ("A_router_inference_sklearn_single_ms_per_query", a_single_ms, "ms/query", "sklearn Pipeline.predict_proba, 1 row/call"),
        ("A_router_inference_sklearn_batched_ms_per_query", a_batch_ms, "ms/query", "one call over all timed queries"),
        ("A_router_inference_numpy_single_ms_per_query", a_numpy_ms, "ms/query", "same weights as numpy dot product"),
        ("A_end_to_end_ms_per_query_numpy", regex_ms + tok_ms + a_numpy_ms, "ms/query", "extractor + numpy LogReg"),
        ("C_combiner_inference_sklearn_single_ms_per_query", c_single_ms, "ms/query", "sklearn Pipeline.predict_proba, 1 row/call"),
        ("C_combiner_inference_numpy_single_ms_per_query", c_numpy_ms, "ms/query", "same weights as numpy dot product"),
        ("C_additional_overhead_over_MiniLM_ms_per_query_numpy", regex_ms + c_numpy_ms, "ms/query",
         "regex features + combiner; token count reused from the MiniLM tokenizer pass"),
        ("C_additional_overhead_over_MiniLM_ms_per_query_numpy_with_separate_tokenizer", regex_ms + tok_ms + c_numpy_ms,
         "ms/query", "upper bound if the untruncated token count is recomputed"),
        ("MiniLM_end_to_end_reference_ms_per_query", MINILM_E2E_MS, "ms/query",
         "EXISTING measurement (diagnostics/stratified_cv/minilm_end_to_end_timing.csv); not rerun"),
        ("C_estimated_end_to_end_ms_per_query", MINILM_E2E_MS + regex_ms + c_numpy_ms, "ms/query",
         "reference MiniLM e2e + measured C overhead (estimate, not a single timed pipeline)"),
        ("A_learned_params", nA, "params", "15 coef + intercept"),
        ("A_scaler_stats", scA, "params", "mean/std of 6 continuous features (not learned by gradient)"),
        ("C_combiner_learned_params", nC, "params", "16 coef + intercept"),
        ("C_scaler_stats", scC, "params", "mean/std of 6 continuous features + student score"),
        ("MiniLM_student_head_params", STUDENT_HEAD_PARAMS, "params", "MLP-64 head"),
        ("MiniLM_encoder_params", MINILM_ENCODER_PARAMS, "params", "frozen all-MiniLM-L6-v2"),
        ("C_total_deployed_params", MINILM_ENCODER_PARAMS + STUDENT_HEAD_PARAMS + nC + scC, "params",
         "encoder + student head + combiner + scaler stats"),
        ("B_total_deployed_params", MINILM_ENCODER_PARAMS + STUDENT_HEAD_PARAMS, "params", "encoder + student head"),
        ("A_pipeline_pickle_bytes", len(pickle.dumps(pipe_A)), "bytes", "fold-0 sklearn pipeline"),
        ("C_pipeline_pickle_bytes", len(pickle.dumps(pipe_C)), "bytes", "fold-0 sklearn combiner pipeline (excl. MiniLM)"),
        ("A_train_sec_mean_per_fold", tt["A"].mean(), "sec", "fit + training-only threshold CV"),
        ("B_train_sec_mean_per_fold", tt["B"].mean(), "sec", "GTE teacher + MLP-64 student on outer-train (cached embeddings)"),
        ("C_train_sec_mean_per_fold", tt["C"].mean(), "sec",
         "B + 5 inner teacher/student refits for OOF scores + combiner + threshold CV"),
    ]
    pd.DataFrame(rows, columns=["metric", "value", "unit", "notes"]).to_csv(EFF_CSV, index=False)


if __name__ == "__main__":
    main()
