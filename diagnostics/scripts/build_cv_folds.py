"""Stratified K-Fold Router Evaluation Mission -- Step 1: fold construction.

Builds a reproducible STRATIFIED 5-FOLD partition of the *full* controlled
DeepSeek-R1-0528-Qwen3-8B vs Qwen3-8B dataset and writes the fold assignment
plus fold statistics, so every later evaluation step consumes the identical,
frozen folds.

Design decisions (all evidence-checked, see notes):
  * The "full dataset" is the union of any seed's train+test split. Every seed
    42-51 partitions the SAME 11,467 unique (dataset, index) queries (verified:
    pairwise symmetric difference of the union key sets is 0). We rebuild it
    from seed 42's train+test union and key every row by (dataset, index),
    which is unique across the pool (11,467 rows, 11,467 unique keys, 0 None
    indices, 0 train/test overlap).
  * Stratification uses the RAW THREE-WAY outcome category required by the
    mission, derived from the strictly-binary per-model correctness scores:
        0 = DeepSeek-only correct (ds > qwen)
        1 = tie                   (ds == qwen; both right OR both wrong)
        2 = Qwen-only correct     (qwen > ds)
  * K = 5, fixed seed, shuffle=True, using sklearn StratifiedKFold so each fold
    preserves the global class proportions. Because this is a partition, no
    query occurs in more than one outer test fold and every query is held out
    exactly once (asserted).

Outputs (diagnostics/stratified_cv/):
  * fold_assignments.csv  -- one row per query: dataset, index, query_sha1,
                             strat_category, strat_label, fold
  * fold_statistics.csv   -- per-fold (and per-fold-per-dataset) sample counts,
                             DeepSeek-win / Qwen-win / tie counts and
                             proportions, plus a whole-dataset reference row.

Read-only w.r.t. all existing data and results. Deterministic (fixed seed).
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

DS = "DeepSeek-R1-0528-Qwen3-8B"
QWEN = "Qwen3-8B"

# Canonical source seed whose train+test union reconstitutes the full pool.
SOURCE_SEED = 42
K = 5
FOLD_SEED = 42  # fixed random seed for reproducible fold assignment

STRAT_LABELS = {0: "DeepSeek-only", 1: "tie", 2: "Qwen-only"}

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "baselines" / "AvengersPro" / "data"
OUT = REPO / "diagnostics" / "stratified_cv"
OUT.mkdir(parents=True, exist_ok=True)


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


def load_full_pool():
    """Union of the canonical seed's train+test = the full dataset."""
    d = split_dir(SOURCE_SEED)
    full = pd.concat(
        [load_jsonl(d / "train.jsonl"), load_jsonl(d / "test.jsonl")],
        ignore_index=True,
    )
    # deterministic canonical ordering independent of file order
    full = full.sort_values(["dataset", "index"]).reset_index(drop=True)
    assert full[["dataset", "index"]].duplicated().sum() == 0, "non-unique key"
    assert full["index"].notna().all(), "found None index"
    return full


def strat_category(df):
    """Raw three-way outcome category required for stratification.

    0 = DeepSeek-only correct, 1 = tie, 2 = Qwen-only correct.
    """
    return np.select(
        [
            df["deepseek_score"] > df["qwen_score"],
            df["qwen_score"] > df["deepseek_score"],
        ],
        [0, 2],
        default=1,
    ).astype(int)


def sha1(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def main():
    full = load_full_pool()
    full["strat_category"] = strat_category(full)

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=FOLD_SEED)
    y = full["strat_category"].to_numpy()
    fold = np.full(len(full), -1, dtype=int)
    for f, (_, test_idx) in enumerate(skf.split(np.zeros(len(full)), y)):
        fold[test_idx] = f
    full["fold"] = fold

    # ---- partition integrity: each query held out exactly once ----
    assert (full["fold"] >= 0).all(), "a query was never assigned to a fold"
    assert (full["fold"] < K).all(), "fold index out of range"
    vc = full["fold"].value_counts()
    assert len(vc) == K, f"expected {K} folds, got {len(vc)}"

    # ---- fold_assignments.csv ----
    assign = full[["dataset", "index", "strat_category", "fold"]].copy()
    assign["strat_label"] = assign["strat_category"].map(STRAT_LABELS)
    assign["query_sha1"] = full["query"].map(sha1)
    assign = assign[
        ["dataset", "index", "query_sha1", "strat_category", "strat_label", "fold"]
    ].sort_values(["fold", "dataset", "index"]).reset_index(drop=True)
    assign.to_csv(OUT / "fold_assignments.csv", index=False)

    # ---- fold_statistics.csv ----
    def bundle(sub, fold_label, scope):
        n = len(sub)
        ds_win = int((sub["strat_category"] == 0).sum())
        tie = int((sub["strat_category"] == 1).sum())
        qw_win = int((sub["strat_category"] == 2).sum())
        return {
            "fold": fold_label,
            "scope": scope,
            "n_samples": n,
            "ds_win": ds_win,
            "tie": tie,
            "qwen_win": qw_win,
            "nontie": ds_win + qw_win,
            "ds_win_prop": ds_win / n if n else 0.0,
            "tie_prop": tie / n if n else 0.0,
            "qwen_win_prop": qw_win / n if n else 0.0,
            "nontie_prop": (ds_win + qw_win) / n if n else 0.0,
        }

    stats = []
    # whole-dataset reference row
    stats.append(bundle(full, "ALL", "ALL"))
    # per-fold overall
    for f in range(K):
        stats.append(bundle(full[full["fold"] == f], f, "ALL"))
    # per-fold per-dataset composition
    for ds_name in sorted(full["dataset"].unique()):
        sub_ds = full[full["dataset"] == ds_name]
        stats.append(bundle(sub_ds, "ALL", ds_name))
        for f in range(K):
            stats.append(bundle(sub_ds[sub_ds["fold"] == f], f, ds_name))

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(OUT / "fold_statistics.csv", index=False)

    # ---- console summary ----
    print(f"Full pool: {len(full)} queries (source seed {SOURCE_SEED} union)")
    print(f"Stratified {K}-fold, seed={FOLD_SEED}, shuffle=True\n")
    overall = stats_df[(stats_df["scope"] == "ALL")]
    cols = ["fold", "n_samples", "ds_win", "tie", "qwen_win",
            "ds_win_prop", "tie_prop", "qwen_win_prop"]
    print(overall[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print("Wrote:")
    print(" ", OUT / "fold_assignments.csv", f"({len(assign)} rows)")
    print(" ", OUT / "fold_statistics.csv", f"({len(stats_df)} rows)")


if __name__ == "__main__":
    main()
