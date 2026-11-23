"""ROUTER-INTEG: regenerate, verify and export the frozen lightweight router (desktop, CPU only).

No serialised router checkpoint was stored by the router study. The router of the paper (Section VIII, "MiniLM
router", Macro-F1 0.600) is five cross-validation fold models. Each is

    sentence-transformers/all-MiniLM-L6-v2 (kept fixed)
      -> attention-mask mean pooling, L2 normalisation (384-D)
      -> MLPRegressor(64, ReLU) trained to regress the decision score of a fold-local
         LogisticRegression teacher on GTE-Qwen2-7B embeddings
      -> decision: score > 0 -> Qwen3-8B, otherwise DeepSeek-R1-0528-Qwen3-8B (fixed, never tuned)

This script does NOT train a new router. It calls the original, unmodified training code
(diagnostics/scripts/run_corrected_nonlinear_cv.py: make_student_mlp, same teacher, same seeds 42 + fold) on the
original cached embeddings and the frozen folds, and accepts the result only if

  1. the epochs used by every fold head equal the epochs stored in corrected_nonlinear_cv.csv, and
  2. the held-out score of every one of the 11,467 pool queries equals the stored per-query score
     (capability_profile/heldout_predictions.csv, column B_student_score, the file behind the paper's numbers)
     within TOL, with identical decisions.

Only then are the head weights exported as plain arrays in their own dtype (no pickle, no scikit-learn needed at inference) and
pinned by SHA-256. If either check fails, nothing is exported and the exit status is 2: that is a blocker, not a
reason to fit another model.

Run with the router study's own environment:
  ~/research/LLMRouterBench/.venv/bin/python tools/rint_router_build.py --repo ~/research/LLMRouterBench
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

TOL = 1e-6          # stored scores are printed with 8 significant digits (float32)
PLAIN_TOL = 1e-5    # plain forward pass against MLPRegressor.predict (float32 arithmetic, different operation order)
OUT = Path(__file__).resolve().parent.parent / "router"


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def task_lookup(repo, cv, pool, mini, out):
    """Policy D of the protocol: a task-level lookup, derived from training folds only.

    Table (per outer fold): a benchmark is assigned Qwen3-8B if Qwen3-8B wins at least half of that benchmark's
    outer-TRAIN queries with a winner, otherwise DeepSeek. This is the rule of the stored "benchmark-identity
    diagnostic reference" (within_task_routing/benchmark_identity_reference.csv), which is reproduced and checked here.
      D1 (diagnostic, NOT deployable): the table is indexed by the TRUE benchmark of the held-out query.
      D2 (predictive): the table is indexed by the benchmark PREDICTED for the held-out query by a recogniser fitted on
         the outer-train queries only: LogisticRegression(C=1.0, max_iter=2000, random_state=42 + fold) on the same
         L2-normalised MiniLM embeddings the router uses. One fixed configuration, no tuning, no held-out label used.
    Written per query: fold, predicted benchmark, D1 choice, D2 choice (0 = DeepSeek, 1 = Qwen3-8B)."""
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import Normalizer
    ref = pd.read_csv(repo / "diagnostics" / "within_task_routing" / "benchmark_identity_reference.csv")
    ref_choice = {(int(r.fold), r.dataset): int(r.ref_predicted_winner == "Qwen") for r in ref.itertuples()}
    rows, acc, mismatch = {}, {}, 0
    for fold in range(cv.K):
        test, train = pool[pool["fold"] == fold], pool[pool["fold"] != fold]
        nt = train[train["target"] != -1]
        glob_share = float((nt["target"] == 1).mean())
        table = {}
        for ds in sorted(pool["dataset"].unique()):
            sub = nt[nt["dataset"] == ds]
            share = float((sub["target"] == 1).mean()) if len(sub) else glob_share
            table[ds] = int(share >= 0.5)
            if (fold, ds) in ref_choice and ref_choice[(fold, ds)] != table[ds]:
                mismatch += 1
        Xtr = Normalizer(norm="l2").fit_transform(cv.embedding_matrix(train["query"].tolist(), mini))
        Xte = Normalizer(norm="l2").fit_transform(cv.embedding_matrix(test["query"].tolist(), mini))
        rec = LogisticRegression(C=1.0, max_iter=2000, random_state=42 + fold)
        rec.fit(Xtr, train["dataset"].to_numpy())
        pred = rec.predict(Xte)
        acc[str(fold)] = float((pred == test["dataset"].to_numpy()).mean())
        for ds, idx, p in zip(test["dataset"], test["index"].astype(int), pred):
            rows[f"{ds}:{idx}"] = [fold, str(p), table[ds], table[str(p)]]
        print(f"fold {fold}: task recogniser held-out accuracy {acc[str(fold)]:.4f}", flush=True)
    if mismatch:
        print(f"BLOCKER: {mismatch} lookup-table entries differ from the stored benchmark-identity reference", file=sys.stderr)
        sys.exit(2)
    (out / "task_lookup.json").write_text(json.dumps(rows, sort_keys=True, separators=(",", ":")) + "\n")
    return {"recogniser": "LogisticRegression(C=1.0, max_iter=2000, random_state=42+fold) on L2-normalised MiniLM embeddings, outer-train only",
            "recogniser_heldout_accuracy_by_fold": acc, "table_rule": "outer-train Qwen share among queries with a winner >= 0.5 -> Qwen3-8B",
            "table_entries_checked_against_stored_reference": len(ref_choice), "table_mismatches": mismatch,
            "columns": ["fold", "predicted_benchmark", "D1_choice_true_benchmark", "D2_choice_predicted_benchmark"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="the router study's repository (read-only)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    repo = Path(args.repo).expanduser().resolve()
    out = Path(args.out)
    sys.path.insert(0, str(repo / "diagnostics" / "scripts"))
    import pandas as pd
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import Normalizer
    import run_cv_eval as cv
    import run_corrected_nonlinear_cv as rc

    pool = cv.load_full_pool_with_folds()
    stored = pd.read_csv(repo / "diagnostics" / "capability_profile" / "heldout_predictions.csv")
    saved = pd.read_csv(repo / "diagnostics" / "stratified_cv" / "corrected_nonlinear_cv.csv")
    saved = saved[saved["method"] == rc.NONLINEAR_NAME]
    epochs_stored = {int(r.fold): int(r.epochs_used) for r in saved.itertuples() if str(r.fold).isdigit()}
    gte = cv.load_embedding_cache(cv.GTE_CACHE, cv.GTE_MODEL)
    mini = cv.load_embedding_cache(cv.MINILM_CACHE, cv.MINILM_MODEL)

    key = lambda df: list(zip(df["dataset"], df["index"].astype(int)))  # noqa: E731
    stored_score = dict(zip(key(stored), stored["B_student_score"].astype(float)))
    stored_fold = dict(zip(key(stored), stored["fold"].astype(int)))
    stored_pred = dict(zip(key(stored), stored["B_pred"].astype(int)))

    heads, report, problems = {}, {"folds": {}}, []
    max_diff, n_checked, n_decision_diff = 0.0, 0, 0
    for fold in range(cv.K):
        test, train = pool[pool["fold"] == fold].copy(), pool[pool["fold"] != fold].copy()
        nt = (train["target"] != -1).to_numpy()
        Xg = Normalizer(norm="l2").fit_transform(cv.embedding_matrix(train["query"].tolist(), gte))
        teacher = LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=2000, random_state=42 + fold, class_weight=None)
        teacher.fit(Xg[nt], train.loc[nt, "target"].to_numpy())
        Xm = Normalizer(norm="l2").fit_transform(cv.embedding_matrix(train["query"].tolist(), mini))
        head = rc.make_student_mlp(fold)
        head.fit(Xm, teacher.decision_function(Xg))
        Xt = Normalizer(norm="l2").fit_transform(cv.embedding_matrix(test["query"].tolist(), mini))
        s = head.predict(Xt)
        ks = key(test)
        d = np.array([abs(float(a) - stored_score[k]) for a, k in zip(s, ks)])
        dec = sum(int(a > 0) != stored_pred[k] for a, k in zip(s, ks))
        if any(stored_fold[k] != fold for k in ks):
            problems.append(f"fold {fold}: a held-out query has another stored fold")
        if int(head.n_iter_) != epochs_stored.get(fold):
            problems.append(f"fold {fold}: {head.n_iter_} epochs, stored {epochs_stored.get(fold)}")
        if d.max() > TOL:
            problems.append(f"fold {fold}: max |score - stored| = {d.max():.3e} > {TOL}")
        if dec:
            problems.append(f"fold {fold}: {dec} decisions differ from the stored predictions")
        max_diff, n_checked, n_decision_diff = max(max_diff, float(d.max())), n_checked + len(ks), n_decision_diff + dec
        W1, W2 = head.coefs_
        b1, b2 = head.intercepts_
        assert W1.shape == (384, 64) and W2.shape == (64, 1) and head.activation == "relu" and head.out_activation_ == "identity"
        # plain forward pass, as the inference adapter will compute it
        plain = (np.maximum(Xt.astype(W1.dtype) @ W1 + b1, 0.0) @ W2 + b2).ravel()
        if np.abs(plain - s).max() > PLAIN_TOL or int(((plain > 0) != (s > 0)).sum()):
            problems.append(f"fold {fold}: exported forward pass differs from MLPRegressor.predict by {np.abs(plain - s).max():.3e}")
        heads[fold] = {"W1": np.asarray(W1), "b1": np.asarray(b1), "W2": np.asarray(W2), "b2": np.asarray(b2)}   # native dtype
        min_abs = float(np.abs(s).min())
        report["folds"][str(fold)] = {"epochs": int(head.n_iter_), "epochs_stored": epochs_stored.get(fold), "n_train": int(len(train)),
                                      "n_train_nontie_teacher": int(nt.sum()), "n_heldout": int(len(test)),
                                      "max_abs_diff_vs_stored_score": float(d.max()), "decisions_differing": int(dec),
                                      "params": int(W1.size + b1.size + W2.size + b2.size),
                                      "weights_dtype": str(W1.dtype), "max_abs_diff_plain_forward": float(np.abs(plain - s).max()),
                                      "min_abs_heldout_score": min_abs}
        print(f"fold {fold}: epochs {head.n_iter_} (stored {epochs_stored.get(fold)}), held-out {len(test)}, "
              f"max |diff| {d.max():.2e}, decisions differing {dec}", flush=True)

    report.update({"n_pool": int(len(pool)), "n_checked": n_checked, "max_abs_diff_vs_stored_score": max_diff,
                   "decisions_differing": n_decision_diff, "tolerance": TOL, "problems": problems,
                   "sklearn": sklearn.__version__, "numpy": np.__version__,
                   "training_code": {p: sha256_file(repo / "diagnostics" / "scripts" / p)
                                     for p in ("run_corrected_nonlinear_cv.py", "run_cv_eval.py")},
                   "inputs": {"fold_assignments.csv": sha256_file(repo / "diagnostics" / "stratified_cv" / "fold_assignments.csv"),
                              "heldout_predictions.csv": sha256_file(repo / "diagnostics" / "capability_profile" / "heldout_predictions.csv"),
                              "corrected_nonlinear_cv.csv": sha256_file(repo / "diagnostics" / "stratified_cv" / "corrected_nonlinear_cv.csv"),
                              "minilm_cache_sqlite": sha256_file(cv.MINILM_CACHE), "gte_cache_sqlite": sha256_file(cv.GTE_CACHE)}})
    if problems:
        print("BLOCKER: the regenerated fold heads do not reproduce the stored router. Nothing exported.", file=sys.stderr)
        for p in problems:
            print("  " + p, file=sys.stderr)
        sys.exit(2)

    out.mkdir(parents=True, exist_ok=True)
    arrays = {f"fold{f}_{k}": v for f, h in heads.items() for k, v in h.items()}
    np.savez(out / "router_heads.npz", **arrays)
    # a canonical hash that does not depend on the zip container
    canon = hashlib.sha256()
    for name in sorted(arrays):
        canon.update(name.encode() + b"\0" + str(arrays[name].shape).encode() + b"\0" + np.ascontiguousarray(arrays[name]).tobytes())
    report["heads_canonical_sha256"] = canon.hexdigest()
    report["heads_file_sha256"] = sha256_file(out / "router_heads.npz")
    # the fold of every pool query, by the 16-hex query hash the folds file already uses (no query text is written)
    folds = pd.read_csv(repo / "diagnostics" / "stratified_cv" / "fold_assignments.csv")
    assert not folds.duplicated(["dataset", "index"]).any()
    fmap = {f"{r.dataset}:{int(r.index)}": [int(r.fold), str(r.query_sha1)] for r in folds.itertuples()}
    (out / "fold_map.json").write_text(json.dumps(fmap, sort_keys=True, separators=(",", ":")) + "\n")
    report["fold_map_sha256"] = sha256_file(out / "fold_map.json")
    report["task_lookup"] = task_lookup(repo, cv, pool, mini, out)
    report["task_lookup_sha256"] = sha256_file(out / "task_lookup.json")
    (out / "router_build_report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(f"OK: {n_checked} held-out scores reproduced, max |diff| {max_diff:.2e}; heads {report['heads_canonical_sha256'][:16]}")


if __name__ == "__main__":
    main()
