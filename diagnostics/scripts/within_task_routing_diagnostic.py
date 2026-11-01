"""Within-task routing diagnostic (analysis only -- no router is trained).

Question: do MiniLM (B) and Capability+MiniLM (C) predict which model wins for
individual queries, or mostly recognise which benchmark a query comes from?

Inputs (read-only):
  diagnostics/capability_profile/heldout_predictions.csv  (frozen 5-fold OOF predictions)
  diagnostics/capability_profile/fold_results.csv         (saved metrics, for verification)
  diagnostics/stratified_cv/fold_assignments.csv          (frozen folds, for verification)

Benchmark identity (the `dataset` column) is used ONLY for this diagnostic
analysis. It is never a router feature, and the "benchmark-identity diagnostic
reference" is an explanatory reference, not a model or routing method.

Scores / predictions used (all saved, none recomputed by refitting):
  B: B_student_score (AUC); B_pred = score > 0 (Macro-F1, the saved reference rule)
  C: C_prob_qwen     (AUC); C_pred_threshold = training-only threshold (headline rule)

Pre-declared definitions (fixed before looking at per-benchmark results):
  * valid benchmark: both winner classes present among its pooled held-out
    non-tie rows; otherwise AUC is reported as undefined with the reason.
  * within-benchmark AUC: AUC over that benchmark's held-out non-tie rows pooled
    across the 5 outer folds (each row scored by the model of its own fold).
  * near chance: 0.45 <= AUC <= 0.55.
  * clear within-task signal: AUC > 0.55 AND lower bound of the 95% stratified
    bootstrap CI > 0.50.
  * C vs B on a benchmark: the paired bootstrap 95% CI of (C AUC - B AUC);
    "C better"/"B better" only if the CI excludes 0.
  * bootstrap: 2000 resamples, class-stratified within each benchmark,
    numpy default_rng(0). Percentile intervals. These are uncertainty
    estimates, not multiple-comparison-corrected significance tests.
  * weighted mean AUC: weights = number of non-tie rows in the benchmark.
  * benchmark-identity reference: per outer fold, score = outer-TRAIN non-tie
    Qwen share of the row's benchmark; predict Qwen if share >= 0.5. Fallback
    for a benchmark with zero outer-train non-tie rows: the global outer-train
    non-tie Qwen share (count reported). Its within-benchmark AUC is 0.5 by
    construction (constant score within a benchmark, per fold).

Outputs (diagnostics/within_task_routing/, never overwritten):
  benchmark_auc.csv, benchmark_identity_reference.csv, fold_results.csv,
  comparison_summary.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cv_eval as cv  # noqa: E402

REPO = cv.REPO
PRED_CSV = REPO / "diagnostics" / "capability_profile" / "heldout_predictions.csv"
CP_FOLD_CSV = REPO / "diagnostics" / "capability_profile" / "fold_results.csv"
FOLDS_CSV = REPO / "diagnostics" / "stratified_cv" / "fold_assignments.csv"
OUT = REPO / "diagnostics" / "within_task_routing"
BENCH_CSV = OUT / "benchmark_auc.csv"
REF_CSV = OUT / "benchmark_identity_reference.csv"
FOLD_OUT = OUT / "fold_results.csv"
SUMMARY_CSV = OUT / "comparison_summary.csv"

N_BOOT = 2000
REF = "benchmark-identity diagnostic reference"
B = "B: MiniLM MLP-64 student"
C = "C: capability-profile-v1 + MiniLM student LogReg"
METHODS = {  # name -> (score column, prediction column, decision rule)
    B: ("B_student_score", "B_pred", "fixed_score>0"),
    C: ("C_prob_qwen", "C_pred_threshold", "training_only_threshold"),
}


def auc_or_nan(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else np.nan


def boot_indices(y, rng):
    """Class-stratified bootstrap resample indices for one benchmark."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    return np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])


def band_counts(aucs):
    a = np.asarray([x for x in aucs if not np.isnan(x)])
    return {"n_auc_lt_0.45": int((a < 0.45).sum()),
            "n_auc_0.45_to_0.55": int(((a >= 0.45) & (a <= 0.55)).sum()),
            "n_auc_gt_0.55": int((a > 0.55).sum()),
            "n_auc_ge_0.60": int((a >= 0.60).sum())}


# --------------------------------------------------------------------------- #
# Part 1: verification                                                         #
# --------------------------------------------------------------------------- #
def verify(P):
    folds = pd.read_csv(FOLDS_CSV)
    assert len(P) == len(folds) == 11467
    assert not P.duplicated(["dataset", "index"]).any(), "duplicate held-out predictions"
    m = P.merge(folds[["dataset", "index", "query_sha1", "fold"]], on=["dataset", "index"],
                suffixes=("", "_frozen"), validate="one_to_one")
    assert len(m) == len(folds)
    assert (m["fold"] == m["fold_frozen"]).all(), "held-out fold != frozen fold"
    assert (m["query_sha1"] == m["query_sha1_frozen"]).all()
    assert ((P["target"] != -1).astype(int) == P["is_nontie"]).all()
    assert (P["B_pred"] == (P["B_student_score"] > 0).astype(int)).all()

    saved = pd.read_csv(CP_FOLD_CSV)
    N = P[P["is_nontie"] == 1]
    rows = []
    for name, (sc, pc, rule) in METHODS.items():
        prefix = name[0]
        sv = saved[saved["method"].str.startswith(prefix + ":") & (saved["decision_rule"] == rule)]
        per_fold = []
        for k in range(cv.K):
            sub = N[N["fold"] == k]
            mm = cv.classification_metrics(sub["target"], sub[pc])
            per_fold.append((mm["macro_f1"], auc_or_nan(sub["target"].to_numpy(), sub[sc].to_numpy())))
            s = sv[sv["fold"] == str(k)].iloc[0]
            assert abs(mm["macro_f1"] - s["macro_f1"]) < 1e-9 and abs(per_fold[-1][1] - s["auc"]) < 1e-9
        f1m, aucm = np.mean([x[0] for x in per_fold]), np.mean([x[1] for x in per_fold])
        rows.append({"method": name, "decision_rule": rule, "fold_mean_macro_f1": f1m,
                     "fold_mean_auc": aucm, "matches_saved_fold_results": True})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Part 3: benchmark-identity diagnostic reference                              #
# --------------------------------------------------------------------------- #
def benchmark_identity_reference(P):
    N = P[P["is_nontie"] == 1].copy()
    N["ref_score"] = np.nan
    table = []
    fallback_rows = 0
    for k in range(cv.K):
        tr, te = N[N["fold"] != k], N[N["fold"] == k]
        share = tr.groupby("dataset")["target"].agg(["size", "sum"])
        global_share = float(tr["target"].mean())
        for ds in sorted(N["dataset"].unique()):
            n_tr = int(share.loc[ds, "size"]) if ds in share.index else 0
            fb = n_tr == 0
            q = global_share if fb else float(share.loc[ds, "sum"] / n_tr)
            idx = te.index[te["dataset"] == ds]
            N.loc[idx, "ref_score"] = q
            fallback_rows += len(idx) if fb else 0
            heldout = N.loc[idx, "target"]
            table.append({"fold": k, "dataset": ds, "n_train_nontie": n_tr,
                          "n_train_qwen_wins": int(share.loc[ds, "sum"]) if n_tr else 0,
                          "train_qwen_share": q, "fallback_global_share_used": fb,
                          "global_train_qwen_share": global_share,
                          "ref_predicted_winner": "Qwen" if q >= 0.5 else "DeepSeek",
                          "n_heldout_nontie": int(len(idx)),
                          "heldout_qwen_share": float(heldout.mean()) if len(idx) else np.nan})
    N["ref_pred"] = (N["ref_score"] >= 0.5).astype(int)
    assert not N["ref_score"].isna().any()
    return N, pd.DataFrame(table), fallback_rows


# --------------------------------------------------------------------------- #
# Part 2 + 4: within-benchmark AUC, bootstrap, comparison                       #
# --------------------------------------------------------------------------- #
def within_benchmark(N, rng):
    rows, boot = [], {}
    for ds, sub in N.groupby("dataset", sort=True):
        y = sub["target"].to_numpy()
        n_q, n_d = int((y == 1).sum()), int((y == 0).sum())
        r = {"dataset": ds, "n_nontie": len(y), "n_deepseek_wins": n_d, "n_qwen_wins": n_q,
             "qwen_share": n_q / len(y), "valid_for_auc": bool(n_q and n_d),
             "auc_undefined_reason": "" if (n_q and n_d) else
             f"only one winner class present ({'Qwen' if n_q else 'DeepSeek'} wins only)",
             "low_minority_n_lt_10": bool(min(n_q, n_d) < 10)}
        if not r["valid_for_auc"]:
            rows.append(r)
            continue
        sB, sC = sub["B_student_score"].to_numpy(), sub["C_prob_qwen"].to_numpy()
        r["B_auc"], r["C_auc"] = auc_or_nan(y, sB), auc_or_nan(y, sC)
        r["C_minus_B_auc"] = r["C_auc"] - r["B_auc"]
        bb, bc = np.empty(N_BOOT), np.empty(N_BOOT)
        for i in range(N_BOOT):
            ix = boot_indices(y, rng)
            bb[i], bc[i] = roc_auc_score(y[ix], sB[ix]), roc_auc_score(y[ix], sC[ix])
        boot[ds] = (bb, bc, len(y))
        for nm, arr in (("B", bb), ("C", bc), ("C_minus_B", bc - bb)):
            r[f"{nm}_auc_ci95_lo"], r[f"{nm}_auc_ci95_hi"] = np.percentile(arr, [2.5, 97.5])
        for nm in ("B", "C"):
            a, lo = r[f"{nm}_auc"], r[f"{nm}_auc_ci95_lo"]
            r[f"{nm}_near_chance"] = bool(0.45 <= a <= 0.55)
            r[f"{nm}_clear_within_task_signal"] = bool(a > 0.55 and lo > 0.50)
        # sensitivity: pooling scores from 5 fold models can distort AUC, so also
        # report (a) mean of per-fold AUCs, (b) AUC of within-fold percentile ranks
        for nm, sc in (("B", "B_student_score"), ("C", "C_prob_qwen")):
            per = [roc_auc_score(g["target"], g[sc]) for _, g in sub.groupby("fold") if g["target"].nunique() == 2]
            r[f"{nm}_auc_mean_of_folds"] = float(np.mean(per))
            r[f"{nm}_auc_fold_rank_pooled"] = auc_or_nan(y, sub[f"{sc}_foldrank"].to_numpy())
        r["n_folds_with_both_classes"] = len(per)
        lo, hi = r["C_minus_B_auc_ci95_lo"], r["C_minus_B_auc_ci95_hi"]
        r["C_vs_B_paired_bootstrap"] = "C better" if lo > 0 else ("B better" if hi < 0 else "no clear difference")
        rows.append(r)
    return pd.DataFrame(rows), boot


def aggregate(bench, boot, col):
    v = bench[bench["valid_for_auc"]]
    a, w = v[col].to_numpy(), v["n_nontie"].to_numpy()
    return {"macro": float(a.mean()), "weighted": float(np.average(a, weights=w)),
            "median": float(np.median(a)), **band_counts(a)}


def boot_aggregates(boot):
    """Bootstrap CIs for macro/weighted within-benchmark AUC (B, C, C-B)."""
    names = sorted(boot)
    w = np.array([boot[d][2] for d in names], dtype=float)
    Bm = np.stack([boot[d][0] for d in names])  # (n_bench, N_BOOT)
    Cm = np.stack([boot[d][1] for d in names])
    out = {}
    for tag, M in (("B", Bm), ("C", Cm), ("C_minus_B", Cm - Bm)):
        out[f"{tag}_macro"] = np.percentile(M.mean(0), [2.5, 97.5])
        out[f"{tag}_weighted"] = np.percentile((M * w[:, None]).sum(0) / w.sum(), [2.5, 97.5])
    return out


def overall_metrics(N, score_col, pred_col, name, rule):
    rows = []
    for k in list(range(cv.K)) + ["pooled"]:
        sub = N if k == "pooled" else N[N["fold"] == k]
        m = cv.classification_metrics(sub["target"], sub[pred_col])
        rows.append({"method": name, "decision_rule": rule, "fold": k, **m,
                     "auc": auc_or_nan(sub["target"].to_numpy(), sub[score_col].to_numpy())})
    df = pd.DataFrame(rows)
    pf = df[df["fold"] != "pooled"]
    num = [c for c in df.columns if c not in ("method", "decision_rule", "fold")]
    agg = [{"method": name, "decision_rule": rule, "fold": stat,
            **{c: float(fn(pf[c].to_numpy(dtype=float))) for c in num}}
           for stat, fn in (("mean", np.mean), ("std", lambda v: np.std(v, ddof=1)))]
    return pd.concat([df, pd.DataFrame(agg)], ignore_index=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for p in (BENCH_CSV, REF_CSV, FOLD_OUT, SUMMARY_CSV):
        assert not p.exists(), f"refusing to overwrite {p}"

    P = pd.read_csv(PRED_CSV)
    ver = verify(P)
    print("Part 1 verification (fold-mean, matches saved fold_results exactly):")
    print(ver.round(4).to_string(index=False))

    N, ref_table, n_fallback = benchmark_identity_reference(P)
    print(f"\nbenchmark-identity reference: fallback rows = {n_fallback}")

    for sc in ("B_student_score", "C_prob_qwen"):
        N[f"{sc}_foldrank"] = N.groupby("fold")[sc].rank(pct=True)
    rng = np.random.default_rng(0)
    bench, boot = within_benchmark(N, rng)
    ci = boot_aggregates(boot)

    folds = pd.concat([
        overall_metrics(N, "ref_score", "ref_pred", REF, "train_benchmark_qwen_share>=0.5"),
        overall_metrics(N, "B_student_score", "B_pred", B, "fixed_score>0"),
        overall_metrics(N, "C_prob_qwen", "C_pred_threshold", C, "training_only_threshold"),
    ], ignore_index=True)

    summary = []
    for name, col in ((REF, None), (B, "B_auc"), (C, "C_auc")):
        f = folds[folds["method"] == name].set_index("fold")
        row = {"method": name,
               "overall_macro_f1_fold_mean": f.loc["mean", "macro_f1"],
               "overall_macro_f1_fold_std": f.loc["std", "macro_f1"],
               "overall_macro_f1_pooled": f.loc["pooled", "macro_f1"],
               "overall_auc_fold_mean": f.loc["mean", "auc"],
               "overall_auc_fold_std": f.loc["std", "auc"],
               "overall_auc_pooled": f.loc["pooled", "auc"],
               "overall_balanced_acc_pooled": f.loc["pooled", "balanced_acc"],
               "n_valid_benchmarks": int(bench["valid_for_auc"].sum())}
        if col is None:
            row.update({"within_macro_auc": 0.5, "within_weighted_auc": 0.5, "within_median_auc": 0.5,
                        "note": "within-benchmark AUC = 0.5 by construction (constant score per benchmark "
                                "and fold); explanatory reference only, not a model or routing method"})
        else:
            a = aggregate(bench, boot, col)
            tag = col[0]
            row.update({"within_macro_auc": a["macro"],
                        "within_macro_auc_ci95_lo": ci[f"{tag}_macro"][0],
                        "within_macro_auc_ci95_hi": ci[f"{tag}_macro"][1],
                        "within_weighted_auc": a["weighted"],
                        "within_weighted_auc_ci95_lo": ci[f"{tag}_weighted"][0],
                        "within_weighted_auc_ci95_hi": ci[f"{tag}_weighted"][1],
                        "within_median_auc": a["median"],
                        **{k: v for k, v in a.items() if k.startswith("n_auc")},
                        "n_clear_within_task_signal": int(bench[f"{tag}_clear_within_task_signal"].sum()),
                        "n_near_chance": int(bench[f"{tag}_near_chance"].sum())})
            for sens in ("auc_mean_of_folds", "auc_fold_rank_pooled"):
                a2 = aggregate(bench, boot, f"{tag}_{sens}")
                row[f"sensitivity_within_macro_{sens}"] = a2["macro"]
                row[f"sensitivity_within_weighted_{sens}"] = a2["weighted"]
        summary.append(row)
    summary.append({"method": "C minus B",
                    "overall_auc_fold_mean": summary[2]["overall_auc_fold_mean"] - summary[1]["overall_auc_fold_mean"],
                    "overall_auc_pooled": summary[2]["overall_auc_pooled"] - summary[1]["overall_auc_pooled"],
                    "overall_macro_f1_fold_mean": summary[2]["overall_macro_f1_fold_mean"] - summary[1]["overall_macro_f1_fold_mean"],
                    "within_macro_auc": summary[2]["within_macro_auc"] - summary[1]["within_macro_auc"],
                    "within_macro_auc_ci95_lo": ci["C_minus_B_macro"][0],
                    "within_macro_auc_ci95_hi": ci["C_minus_B_macro"][1],
                    "within_weighted_auc": summary[2]["within_weighted_auc"] - summary[1]["within_weighted_auc"],
                    "within_weighted_auc_ci95_lo": ci["C_minus_B_weighted"][0],
                    "within_weighted_auc_ci95_hi": ci["C_minus_B_weighted"][1],
                    "within_median_auc": summary[2]["within_median_auc"] - summary[1]["within_median_auc"],
                    "note": "paired class-stratified bootstrap CIs (2000 resamples)"})
    summary = pd.DataFrame(summary)

    bench.to_csv(BENCH_CSV, index=False)
    ref_table.assign(total_fallback_heldout_rows=n_fallback).to_csv(REF_CSV, index=False)
    folds.to_csv(FOLD_OUT, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)

    pd.set_option("display.width", 250)
    print("\nPer-benchmark (pooled OOF held-out non-tie):")
    cols = ["dataset", "n_nontie", "n_deepseek_wins", "n_qwen_wins", "qwen_share", "B_auc", "B_auc_ci95_lo",
            "B_auc_ci95_hi", "C_auc", "C_auc_ci95_lo", "C_auc_ci95_hi", "C_minus_B_auc", "C_vs_B_paired_bootstrap"]
    print(bench[cols].round(3).to_string(index=False))
    print("\nSummary:")
    print(summary.round(4).T.to_string())


if __name__ == "__main__":
    main()
