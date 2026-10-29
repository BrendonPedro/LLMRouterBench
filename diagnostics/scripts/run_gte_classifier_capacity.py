"""Corrected CV Mission -- Task 2: GTE classifier-capacity diagnostic, 5-fold CV.

Question: does the frozen GTE-Qwen2-7B representation contain substantially more
winner-predictive signal than the ~0.615 Macro-F1 the linear router extracts?

Same fixed GTE embeddings (L2-normalised, as in run_cv_eval.py), same frozen
fold_assignments.csv, same held-out non-tie Macro-F1. Classifiers, each with ONE
fixed configuration (no hyperparameter sweep):

  1. LogReg, default cutoff            C=1.0, l2, liblinear  (== run_cv_eval)
  2. LogReg + training-only threshold  (== run_cv_eval "GTE + train-only threshold")
  3. MLP-64 classifier                 MLPClassifier mirroring the fixed MLP-64
                                       student config of
                                       train_minilm_nonlinear_score_distillation.py:
                                       hidden=(64,), relu, adam, alpha=1e-4,
                                       batch 256, lr 1e-3, max_iter 300,
                                       early_stopping (10% of outer-train),
                                       n_iter_no_change 20
  4. RBF-kernel SVM                    SVC(kernel="rbf", C=1.0, gamma="scale"),
                                       sklearn defaults

For 3 and 4 we report both the default cutoff (proba 0.5 / decision 0) and the
same training-only threshold protocol as method 2, so linear vs nonlinear is
compared like-for-like with and without a cutoff adjustment.

Anti-leakage:
  * Every classifier is fit on outer-train non-tie rows only.
  * Training-only threshold = argmax Macro-F1 over out-of-fold scores from an
    inner StratifiedKFold(5) INSIDE the outer-train portion (never outer test).
  * MLP early stopping uses an internal 10% split of the outer-train portion.
  * Outer-test labels are used only for the final metrics.

Also recorded (diagnostic only, never used for selection):
  * AUC on held-out non-tie rows (threshold-free ranking signal)
  * outer-train Macro-F1 at the default cutoff (fit capacity / overfitting gap)

Timings are HEAD-ONLY on precomputed (cached) GTE embeddings; they are NOT
end-to-end latency (no tokenizer / 7B encoder forward pass is timed).

Writes diagnostics/stratified_cv/gte_classifier_capacity.csv (new file; no
existing result is overwritten).
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cv_eval as cv  # noqa: E402

OUT_CSV = cv.OUT / "gte_classifier_capacity.csv"

HIDDEN = 64
PROB_GRID = np.linspace(0.05, 0.95, 181)  # identical to run_cv_eval.run_gte_threshold
QUANTILE_GRID = np.linspace(0.05, 0.95, 181)  # for unbounded SVM decision scores


def make_logreg(fold):
    return LogisticRegression(C=1.0, penalty="l2", solver="liblinear",
                              max_iter=2000, random_state=42 + fold, class_weight=None)


def make_mlp(fold):
    return MLPClassifier(
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


def make_svm(fold):
    return SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42 + fold)


# name, factory, score kind ("proba" -> P(Qwen), default cutoff 0.5;
# "decision" -> decision_function, default cutoff 0)
CLASSIFIERS = [
    ("LogReg", make_logreg, "proba"),
    ("MLP-64", make_mlp, "proba"),
    ("RBF-SVM", make_svm, "decision"),
]


def scores(model, X, kind):
    return model.predict_proba(X)[:, 1] if kind == "proba" else model.decision_function(X)


def head_params(model):
    if isinstance(model, LogisticRegression):
        return int(model.coef_.size + model.intercept_.size)
    if isinstance(model, MLPClassifier):
        return int(sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_))
    if isinstance(model, SVC):
        # stored support vectors + dual coefs + intercept
        return int(model.support_vectors_.size + model.dual_coef_.size + model.intercept_.size)
    raise TypeError(type(model))


def training_only_threshold(base, X_tr, y_tr, kind, fold):
    """Inner-OOF threshold within the outer-train portion (run_cv_eval protocol)."""
    n_splits = min(5, int(np.bincount(y_tr).min()))
    inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42 + fold)
    method = "predict_proba" if kind == "proba" else "decision_function"
    oof = cross_val_predict(clone(base), X_tr, y_tr, cv=inner, method=method)
    if kind == "proba":
        oof = oof[:, 1]
        grid = PROB_GRID
    else:
        grid = np.quantile(oof, QUANTILE_GRID)
    best_t, best_f1 = (0.5 if kind == "proba" else 0.0), -1.0
    for t in grid:
        f = f1_score(y_tr, (oof >= t).astype(int), average="macro", zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    return best_t, best_f1


def run_fold(train, test, gte_cache, fold):
    tr_nt = train[train["target"] != -1]
    te_nt = test[test["target"] != -1]
    norm = Normalizer(norm="l2")
    X_tr = norm.fit_transform(cv.embedding_matrix(tr_nt["query"].tolist(), gte_cache))
    X_te = norm.transform(cv.embedding_matrix(te_nt["query"].tolist(), gte_cache))
    y_tr = tr_nt["target"].to_numpy()
    y_te = te_nt["target"].to_numpy()

    rows = []
    for name, factory, kind in CLASSIFIERS:
        base = factory(fold)
        model = clone(base)
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        fit_sec = time.perf_counter() - t0

        t0 = time.perf_counter()
        s_te = scores(model, X_te, kind)
        infer_sec = time.perf_counter() - t0
        s_tr = scores(model, X_tr, kind)

        default_t = 0.5 if kind == "proba" else 0.0
        t0 = time.perf_counter()
        thr, inner_f1 = training_only_threshold(base, X_tr, y_tr, kind, fold)
        thr_sec = time.perf_counter() - t0

        common = {
            "classifier": name,
            "fold": fold,
            "auc": float(roc_auc_score(y_te, s_te)),
            "train_macro_f1_default_cutoff": float(
                f1_score(y_tr, (s_tr >= default_t).astype(int), average="macro")),
            "epochs_used": getattr(model, "n_iter_", np.nan) if name == "MLP-64" else np.nan,
            "head_params": head_params(model),
            "head_infer_sec_cached_emb_test_nontie": infer_sec,
            "n_train_nontie": int(len(y_tr)),
        }
        for variant, t, extra_sec, inner in (
            ("default cutoff", default_t, 0.0, np.nan),
            ("training-only threshold", thr, thr_sec, inner_f1),
        ):
            m = cv.classification_metrics(y_te, (s_te >= t).astype(int))
            rows.append({
                "method": f"GTE {name} + {variant}",
                **common,
                **m,
                "cutoff": t,
                "inner_oof_macro_f1": inner,
                "head_train_sec_cached_emb": fit_sec + extra_sec,
            })
        print(f"fold {fold} {name:8s} default F1={rows[-2]['macro_f1']:.4f} "
              f"thr F1={rows[-1]['macro_f1']:.4f} (t={thr:.3f}) AUC={common['auc']:.4f} "
              f"trainF1={common['train_macro_f1_default_cutoff']:.4f} "
              f"fit={fit_sec:.1f}s thr={thr_sec:.1f}s", flush=True)
    return rows


def main():
    assert not OUT_CSV.exists(), f"refusing to overwrite {OUT_CSV}"
    pool = cv.load_full_pool_with_folds()
    gte_cache = cv.load_embedding_cache(cv.GTE_CACHE, cv.GTE_MODEL)
    print(f"pool={len(pool)} GTE cached={len(gte_cache)}", flush=True)

    rows = []
    for fold in range(cv.K):
        test = pool[pool["fold"] == fold]
        train = pool[pool["fold"] != fold]
        rows.extend(run_fold(train, test, gte_cache, fold))

    df = pd.DataFrame(rows)
    order = list(dict.fromkeys(df["method"]))
    num_cols = [c for c in df.columns if c not in ("method", "classifier", "fold")]
    agg = []
    for name in order:
        sub = df[df["method"] == name]
        for stat, fn in (("mean", np.mean), ("std", lambda v: np.std(v, ddof=1))):
            agg.append({"method": name, "classifier": sub["classifier"].iloc[0], "fold": stat,
                        **{c: float(fn(sub[c].to_numpy(dtype=float))) for c in num_cols}})
    out = pd.concat([df, pd.DataFrame(agg)], ignore_index=True)
    out.to_csv(OUT_CSV, index=False)

    print("\n5-fold held-out non-tie Macro-F1 / AUC")
    for name in order:
        sub = df[df["method"] == name]
        print(f"  {name:45s} F1 {sub['macro_f1'].mean():.4f} +/- {sub['macro_f1'].std(ddof=1):.4f}"
              f"  AUC {sub['auc'].mean():.4f}")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
