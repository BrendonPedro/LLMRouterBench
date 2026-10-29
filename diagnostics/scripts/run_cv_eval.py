"""Stratified K-Fold Router Evaluation Mission -- Step 2 + 3.

Runs the frozen stratified 5-fold partition (built by build_cv_folds.py) through
every fairly reproducible required baseline, with NO test-fold leakage, and
writes:

  * diagnostics/stratified_cv/router_cv_results.csv
        per-fold + mean + std classification metrics for every method, computed
        ONLY on held-out NON-TIE examples (target 0 = DeepSeek wins,
        1 = Qwen wins). Reports Macro-F1 (primary), balanced accuracy, per-class
        precision / recall / F1 and the 2x2 confusion matrix.
  * diagnostics/stratified_cv/efficiency_results.csv
        measured efficiency evidence per method (parameter counts, serialized
        model size, per-fold and total train time, held-out inference time,
        ms/query, queries/sec, approx peak training memory, and the
        representation model required at deployment -- with offline teacher vs
        online inference requirements clearly separated).

Baselines (exactly the mission's required five, with the configs frozen from the
existing repeated-holdout scripts so the two protocols are comparable):

  1. TF-IDF + Logistic Regression   (train_tfidf_logreg config / f1 ladder)
  2. GTE Direct Logistic Regression (train_gte_direct_selector.py)
  3. GTE + training-only threshold  (gte_threshold_selector; global OOF thresh)
  4. MiniLM Direct                  (train_minilm_direct_selector.py)
  5. Frozen MiniLM nonlinear routing-score distillation
        (train_minilm_routing_score_distillation.py: GTE LogReg teacher ->
         Ridge student on frozen MiniLM embeddings)

AvengersPro is intentionally deferred: its fold-specific K-means router would
need a bespoke retrain and is handled (if feasible) in a later step; the five
above are the required reproducible baselines.

Anti-leakage guarantees:
  * The held-out fold is never seen during fit or any model/threshold selection.
  * The GTE-threshold cutoff is chosen by out-of-fold cross_val_predict INSIDE
    the four training folds only (never the outer held-out fold).
  * Per-fold fixed random_state for determinism.

Read-only w.r.t. all existing data and prior result files.
"""

import io
import json
import pickle
import sqlite3
import time
import tracemalloc
import warnings
from pathlib import Path

import numpy as np

# The baseline configs are frozen from the existing repeated-holdout scripts
# (penalty="l2", solver="liblinear") so the two protocols stay comparable;
# sklearn>=1.8 emits a deprecation FutureWarning for the explicit penalty arg,
# which is cosmetic here.
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import Normalizer

DS = "DeepSeek-R1-0528-Qwen3-8B"
QWEN = "Qwen3-8B"

GTE_MODEL = "Alibaba-NLP/gte-Qwen2-7B-instruct"
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SOURCE_SEED = 42  # canonical seed whose train+test union = full pool
K = 5

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "baselines" / "AvengersPro" / "data"
GTE_CACHE = REPO / "baselines" / "AvengersPro" / "cache" / "gte_qwen2_7b_embeddings.sqlite"
MINILM_CACHE = REPO / "local_experiments" / "cache" / "minilm_l6_v2_embeddings.sqlite"
OUT = REPO / "diagnostics" / "stratified_cv"


# --------------------------------------------------------------------------- #
# Data loading                                                                #
# --------------------------------------------------------------------------- #
def split_dir(seed):
    return DATA_ROOT / f"small_models_seed_{seed}" / f"seed{seed}_split0.7"


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            rows.append(
                {
                    "dataset": x["dataset"],
                    "index": x.get("index"),
                    "query": x["query"],
                    "deepseek_score": float(x["records"][DS]),
                    "qwen_score": float(x["records"][QWEN]),
                }
            )
    return pd.DataFrame(rows)


def load_full_pool_with_folds():
    """Full pool (query text + scores) joined to the frozen fold assignment."""
    d = split_dir(SOURCE_SEED)
    full = pd.concat(
        [load_jsonl(d / "train.jsonl"), load_jsonl(d / "test.jsonl")],
        ignore_index=True,
    ).sort_values(["dataset", "index"]).reset_index(drop=True)

    folds = pd.read_csv(OUT / "fold_assignments.csv")
    merged = full.merge(folds[["dataset", "index", "fold"]], on=["dataset", "index"], how="inner")
    assert len(merged) == len(full) == len(folds), "fold/pool join lost rows"

    # target: 0 = DeepSeek wins, 1 = Qwen wins, -1 = tie (identical to audit)
    merged["target"] = np.select(
        [merged["deepseek_score"] > merged["qwen_score"],
         merged["qwen_score"] > merged["deepseek_score"]],
        [0, 1],
        default=-1,
    )
    return merged


def load_embedding_cache(path, model):
    conn = sqlite3.connect(path)
    rows = conn.execute(
        "SELECT text, embedding FROM embeddings WHERE model = ?", (model,)
    ).fetchall()
    conn.close()
    cache = {t: np.asarray(json.loads(e), dtype=np.float32) for t, e in rows}
    if not cache:
        raise RuntimeError(f"No cached embeddings for {model}")
    return cache


def embedding_matrix(queries, cache):
    missing = [q for q in queries if q not in cache]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} embeddings")
    return np.stack([cache[q] for q in queries], axis=0)


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
def classification_metrics(y_true, y_pred):
    """Full metric bundle on held-out NON-TIE rows. Classes: 0=DeepSeek,1=Qwen."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "n_test_nontie": int(len(y_true)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_deepseek": float(p[0]),
        "precision_qwen": float(p[1]),
        "recall_deepseek": float(r[0]),
        "recall_qwen": float(r[1]),
        "f1_deepseek": float(f[0]),
        "f1_qwen": float(f[1]),
        # confusion matrix rows = true class, cols = predicted class
        "cm_true_ds_pred_ds": int(cm[0, 0]),
        "cm_true_ds_pred_qwen": int(cm[0, 1]),
        "cm_true_qwen_pred_ds": int(cm[1, 0]),
        "cm_true_qwen_pred_qwen": int(cm[1, 1]),
        "pred_qwen_frac": float((y_pred == 1).mean()),
    }


def model_size_bytes(model):
    buf = io.BytesIO()
    pickle.dump(model, buf)
    return buf.tell()


# --------------------------------------------------------------------------- #
# Baseline fold runners                                                        #
# Each returns (y_true, y_pred, eff) where eff holds raw efficiency numbers.   #
# --------------------------------------------------------------------------- #
def run_tfidf(train, test_nt, fold):
    tr_nt = train[train["target"] != -1]
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_features=20000)
    clf = LogisticRegression(max_iter=2000, random_state=42 + fold)

    tracemalloc.start()
    t0 = time.perf_counter()
    X_tr = vec.fit_transform(tr_nt["query"])
    clf.fit(X_tr, tr_nt["target"].to_numpy())
    train_sec = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    t0 = time.perf_counter()
    X_te = vec.transform(test_nt["query"])
    y_pred = clf.predict(X_te)
    infer_sec = time.perf_counter() - t0

    params = int(clf.coef_.size + clf.intercept_.size)
    eff = {
        "total_params": params,
        "trainable_params": params,
        "feature_dim": int(X_tr.shape[1]),
        "model_disk_bytes": model_size_bytes((vec, clf)),
        "train_sec": train_sec,
        "infer_sec": infer_sec,
        "peak_train_mem_mb": peak / 1e6,
    }
    return test_nt["target"].to_numpy(), y_pred, eff


def _gte_test_proba(train, test, cache, fold):
    """Shared GTE LogReg fit; returns (clf, qwen_prob_on_test, train_sec, infer_sec,
    peak, X_tr_nontie, y_tr_nontie)."""
    tr_nt = train[train["target"] != -1]
    norm = Normalizer(norm="l2")
    X_tr = norm.fit_transform(embedding_matrix(tr_nt["query"].tolist(), cache))
    X_te = norm.transform(embedding_matrix(test["query"].tolist(), cache))
    y_tr = tr_nt["target"].to_numpy()

    clf = LogisticRegression(C=1.0, penalty="l2", solver="liblinear",
                             max_iter=2000, random_state=42 + fold, class_weight=None)
    tracemalloc.start()
    t0 = time.perf_counter()
    clf.fit(X_tr, y_tr)
    train_sec = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    t0 = time.perf_counter()
    qwen_prob = clf.predict_proba(X_te)[:, 1]
    infer_sec = time.perf_counter() - t0
    return clf, qwen_prob, train_sec, infer_sec, peak, X_tr, y_tr


def run_gte_direct(train, test, cache, fold):
    clf, qwen_prob, train_sec, infer_sec, peak, _, _ = _gte_test_proba(train, test, cache, fold)
    test = test.copy()
    test["pred"] = (qwen_prob >= 0.5).astype(int)
    test_nt = test[test["target"] != -1]
    params = int(clf.coef_.size + clf.intercept_.size)
    eff = {
        "total_params": params, "trainable_params": params,
        "feature_dim": int(clf.coef_.shape[1]),
        "model_disk_bytes": model_size_bytes(clf),
        "train_sec": train_sec, "infer_sec": infer_sec,
        "peak_train_mem_mb": peak / 1e6,
    }
    return test_nt["target"].to_numpy(), test_nt["pred"].to_numpy(), eff


def run_gte_threshold(train, test, cache, fold):
    clf, qwen_prob, train_sec, infer_sec, peak, X_tr, y_tr = _gte_test_proba(train, test, cache, fold)
    # training-only global threshold via OOF cross_val_predict within train folds
    sel_start = time.perf_counter()
    n_splits = min(5, int(np.bincount(y_tr).min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42 + fold)
    oof = cross_val_predict(
        LogisticRegression(C=1.0, penalty="l2", solver="liblinear",
                           max_iter=2000, random_state=42 + fold, class_weight=None),
        X_tr, y_tr, cv=cv, method="predict_proba",
    )[:, 1]
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        f = f1_score(y_tr, (oof >= t).astype(int), average="macro", zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    sel_sec = time.perf_counter() - sel_start

    test = test.copy()
    test["pred"] = (qwen_prob >= best_t).astype(int)
    test_nt = test[test["target"] != -1]
    params = int(clf.coef_.size + clf.intercept_.size)
    eff = {
        "total_params": params, "trainable_params": params,
        "feature_dim": int(clf.coef_.shape[1]),
        "model_disk_bytes": model_size_bytes(clf),
        # threshold selection is part of (offline) training cost
        "train_sec": train_sec + sel_sec, "infer_sec": infer_sec,
        "peak_train_mem_mb": peak / 1e6,
        "learned_threshold": best_t,
    }
    return test_nt["target"].to_numpy(), test_nt["pred"].to_numpy(), eff


def run_minilm_direct(train, test, cache, fold):
    tr_nt = train[train["target"] != -1]
    norm = Normalizer(norm="l2")
    X_tr = norm.fit_transform(embedding_matrix(tr_nt["query"].tolist(), cache))
    X_te = norm.transform(embedding_matrix(test["query"].tolist(), cache))
    y_tr = tr_nt["target"].to_numpy()
    clf = LogisticRegression(C=1.0, penalty="l2", solver="liblinear",
                             max_iter=2000, random_state=42 + fold, class_weight=None)
    tracemalloc.start()
    t0 = time.perf_counter()
    clf.fit(X_tr, y_tr)
    train_sec = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    t0 = time.perf_counter()
    y_pred_all = clf.predict(X_te)
    infer_sec = time.perf_counter() - t0

    test = test.copy()
    test["pred"] = y_pred_all.astype(int)
    test_nt = test[test["target"] != -1]
    params = int(clf.coef_.size + clf.intercept_.size)
    eff = {
        "total_params": params, "trainable_params": params,
        "feature_dim": int(clf.coef_.shape[1]),
        "model_disk_bytes": model_size_bytes(clf),
        "train_sec": train_sec, "infer_sec": infer_sec,
        "peak_train_mem_mb": peak / 1e6,
    }
    return test_nt["target"].to_numpy(), test_nt["pred"].to_numpy(), eff


def run_distillation(train, test, gte_cache, minilm_cache, fold):
    """Frozen-MiniLM student distilling the GTE LogReg teacher's routing score."""
    tr_nt_mask = train["target"] != -1
    # ---- offline teacher: GTE LogReg on non-tie train only ----
    gte_norm = Normalizer(norm="l2")
    Xg_tr_all = gte_norm.fit_transform(embedding_matrix(train["query"].tolist(), gte_cache))
    y_tr_nt = train.loc[tr_nt_mask, "target"].to_numpy()
    teacher = LogisticRegression(C=1.0, penalty="l2", solver="liblinear",
                                 max_iter=2000, random_state=42 + fold, class_weight=None)
    t0 = time.perf_counter()
    teacher.fit(Xg_tr_all[tr_nt_mask.to_numpy()], y_tr_nt)
    # soft routing preference for EVERY train query (incl. ties)
    teacher_train_score = teacher.decision_function(Xg_tr_all)
    teacher_sec = time.perf_counter() - t0

    # ---- online student: Ridge on frozen MiniLM embeddings ----
    mini_norm = Normalizer(norm="l2")
    Xm_tr = mini_norm.fit_transform(embedding_matrix(train["query"].tolist(), minilm_cache))
    Xm_te = mini_norm.transform(embedding_matrix(test["query"].tolist(), minilm_cache))
    student = Ridge(alpha=1.0, fit_intercept=True)

    tracemalloc.start()
    t0 = time.perf_counter()
    student.fit(Xm_tr, teacher_train_score)
    student_sec = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    t0 = time.perf_counter()
    student_test_score = student.predict(Xm_te)
    infer_sec = time.perf_counter() - t0

    test = test.copy()
    test["pred"] = (student_test_score > 0).astype(int)
    test_nt = test[test["target"] != -1]

    student_params = int(student.coef_.size + 1)
    teacher_params = int(teacher.coef_.size + teacher.intercept_.size)
    eff = {
        # ONLINE inference cost = student only
        "total_params": student_params + teacher_params,
        "trainable_params": student_params,
        "feature_dim": int(student.coef_.size),
        "model_disk_bytes": model_size_bytes(student),
        "teacher_disk_bytes": model_size_bytes(teacher),
        "teacher_params": teacher_params,
        # total training = offline teacher + online student fit
        "train_sec": teacher_sec + student_sec,
        "teacher_train_sec": teacher_sec,
        "student_train_sec": student_sec,
        "infer_sec": infer_sec,
        "peak_train_mem_mb": peak / 1e6,
    }
    return test_nt["target"].to_numpy(), test_nt["pred"].to_numpy(), eff


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #
METHOD_META = {
    "TF-IDF + LogReg": {
        "representation": "TF-IDF (1-2)gram vectorizer (no neural encoder)",
        "deploy_rep": "TF-IDF vectorizer (fit on train queries)",
        "offline_requirements": "none (lexical)",
        "online_requirements": "TF-IDF transform + logistic regression",
    },
    "GTE Direct LogReg": {
        "representation": "GTE-Qwen2-7B-instruct embeddings (3584-D, L2-norm)",
        "deploy_rep": "GTE-Qwen2-7B-instruct (7B param encoder) at inference",
        "offline_requirements": "GTE encoder to embed train queries",
        "online_requirements": "GTE 7B encoder + logistic regression per query",
    },
    "GTE + train-only threshold": {
        "representation": "GTE-Qwen2-7B-instruct embeddings (3584-D, L2-norm)",
        "deploy_rep": "GTE-Qwen2-7B-instruct (7B param encoder) at inference",
        "offline_requirements": "GTE encoder + OOF threshold search on train",
        "online_requirements": "GTE 7B encoder + logistic regression + fixed threshold",
    },
    "MiniLM Direct": {
        "representation": "all-MiniLM-L6-v2 embeddings (384-D, L2-norm)",
        "deploy_rep": "all-MiniLM-L6-v2 (22.7M param encoder) at inference",
        "offline_requirements": "MiniLM encoder to embed train queries",
        "online_requirements": "MiniLM encoder + logistic regression per query",
    },
    "Frozen-MiniLM distillation": {
        "representation": "frozen all-MiniLM-L6-v2 (384-D) student; GTE teacher offline",
        "deploy_rep": "all-MiniLM-L6-v2 only (GTE NOT needed online)",
        "offline_requirements": "GTE 7B teacher embeddings + LogReg teacher (train only)",
        "online_requirements": "MiniLM encoder + Ridge student per query (no GTE)",
    },
}


def main():
    print("Loading full pool + folds ...")
    pool = load_full_pool_with_folds()
    print(f"  pool={len(pool)} rows, folds={sorted(pool['fold'].unique())}")

    print("Loading GTE embedding cache (large) ...")
    gte_cache = load_embedding_cache(GTE_CACHE, GTE_MODEL)
    print(f"  GTE cached={len(gte_cache)} dim={len(next(iter(gte_cache.values())))}")
    print("Loading MiniLM embedding cache ...")
    minilm_cache = load_embedding_cache(MINILM_CACHE, MINILM_MODEL)
    print(f"  MiniLM cached={len(minilm_cache)} dim={len(next(iter(minilm_cache.values())))}")

    methods = [
        "TF-IDF + LogReg",
        "GTE Direct LogReg",
        "GTE + train-only threshold",
        "MiniLM Direct",
        "Frozen-MiniLM distillation",
    ]

    per_fold = []     # metric rows
    eff_rows = []     # efficiency raw rows (per method per fold)

    for fold in range(K):
        test = pool[pool["fold"] == fold].copy()
        train = pool[pool["fold"] != fold].copy()
        test_nt = test[test["target"] != -1].copy()
        print(f"\n=== Fold {fold}: train={len(train)} test={len(test)} "
              f"test_nontie={len(test_nt)} ===")

        runners = {
            "TF-IDF + LogReg": lambda: run_tfidf(train, test_nt, fold),
            "GTE Direct LogReg": lambda: run_gte_direct(train, test, gte_cache, fold),
            "GTE + train-only threshold": lambda: run_gte_threshold(train, test, gte_cache, fold),
            "MiniLM Direct": lambda: run_minilm_direct(train, test, minilm_cache, fold),
            "Frozen-MiniLM distillation": lambda: run_distillation(train, test, gte_cache, minilm_cache, fold),
        }

        for name in methods:
            y_true, y_pred, eff = runners[name]()
            metrics = classification_metrics(y_true, y_pred)
            per_fold.append({"method": name, "fold": fold, **metrics})
            eff_rows.append({"method": name, "fold": fold, **eff})
            qps = metrics["n_test_nontie"] / eff["infer_sec"] if eff["infer_sec"] > 0 else float("nan")
            print(f"  {name:28s} F1={metrics['macro_f1']:.4f} "
                  f"bAcc={metrics['balanced_acc']:.4f} "
                  f"train={eff['train_sec']:.3f}s infer={eff['infer_sec']:.4f}s")

    metrics_df = pd.DataFrame(per_fold)

    # ---- per-fold rows + mean/std rows into router_cv_results.csv ----
    metric_cols = [c for c in metrics_df.columns if c not in ("method", "fold")]
    agg_rows = []
    for name in methods:
        sub = metrics_df[metrics_df["method"] == name]
        for stat, fn in (("mean", np.mean), ("std", lambda v: np.std(v, ddof=1))):
            row = {"method": name, "fold": stat}
            for c in metric_cols:
                row[c] = float(fn(sub[c].to_numpy()))
            agg_rows.append(row)
    out_df = pd.concat([metrics_df, pd.DataFrame(agg_rows)], ignore_index=True)
    out_df.to_csv(OUT / "router_cv_results.csv", index=False)

    # ---- efficiency_results.csv: per-method aggregate ----
    eff_df = pd.DataFrame(eff_rows)
    eff_summary = []
    for name in methods:
        sub = eff_df[eff_df["method"] == name]
        meta = METHOD_META[name]
        n_test_mean = metrics_df[metrics_df["method"] == name]["n_test_nontie"].mean()
        infer_mean = sub["infer_sec"].mean()
        ms_per_query = (infer_mean / n_test_mean) * 1000 if n_test_mean else float("nan")
        qps = n_test_mean / infer_mean if infer_mean > 0 else float("nan")
        row = {
            "method": name,
            "total_params": int(sub["total_params"].iloc[0]),
            "trainable_params": int(sub["trainable_params"].iloc[0]),
            "feature_dim": int(sub["feature_dim"].iloc[0]),
            "model_disk_bytes": int(round(sub["model_disk_bytes"].mean())),
            "train_sec_per_fold_mean": float(sub["train_sec"].mean()),
            "train_sec_per_fold_std": float(sub["train_sec"].std(ddof=1)),
            "total_5fold_train_sec": float(sub["train_sec"].sum()),
            "infer_sec_per_fold_mean": float(infer_mean),
            "ms_per_query": float(ms_per_query),
            "queries_per_sec": float(qps),
            "peak_train_mem_mb_mean": float(sub["peak_train_mem_mb"].mean()),
            "deploy_representation": meta["deploy_rep"],
            "offline_requirements": meta["offline_requirements"],
            "online_requirements": meta["online_requirements"],
        }
        # distillation extras (offline teacher split)
        if "teacher_params" in sub.columns and sub["teacher_params"].notna().any():
            row["teacher_params"] = int(sub["teacher_params"].iloc[0])
            row["teacher_disk_bytes"] = int(round(sub["teacher_disk_bytes"].mean()))
            row["teacher_train_sec_per_fold_mean"] = float(sub["teacher_train_sec"].mean())
            row["student_train_sec_per_fold_mean"] = float(sub["student_train_sec"].mean())
        if "learned_threshold" in sub.columns and sub["learned_threshold"].notna().any():
            row["mean_learned_threshold"] = float(sub["learned_threshold"].mean())
        eff_summary.append(row)
    eff_summary_df = pd.DataFrame(eff_summary)
    eff_summary_df.to_csv(OUT / "efficiency_results.csv", index=False)

    # ---- console summary ----
    print("\n" + "=" * 78)
    print("5-FOLD STRATIFIED CV -- Macro-F1 (non-tie held-out)")
    print("=" * 78)
    mean_rows = out_df[out_df["fold"] == "mean"].set_index("method")
    std_rows = out_df[out_df["fold"] == "std"].set_index("method")
    for name in methods:
        print(f"  {name:28s} F1 = {mean_rows.loc[name,'macro_f1']:.4f} "
              f"+/- {std_rows.loc[name,'macro_f1']:.4f}   "
              f"bAcc = {mean_rows.loc[name,'balanced_acc']:.4f}")
    print("\nWrote:")
    print(" ", OUT / "router_cv_results.csv", f"({len(out_df)} rows)")
    print(" ", OUT / "efficiency_results.csv", f"({len(eff_summary_df)} rows)")


if __name__ == "__main__":
    main()
