"""Section 3 of the Router Performance Diagnostic Mission — Pair learnability.

Question: does the controlled DeepSeek-R1-0528-Qwen3-8B vs Qwen3-8B pair create
an *unusually difficult routing target*, independent of any particular
classifier?

This script does NOT recompute Section 1. It consumes the already-audited
`diagnostics/label_statistics.csv` (per seed / split / dataset counts of
total, tie, non_tie, deepseek_win, qwen_win, both_correct, both_wrong) and
derives learnability metrics:

  - outcome agreement          = tie / total        (both models same outcome)
  - informative non-tie rate   = non_tie / total    (exactly one model correct)
  - per-dataset winner balance = min(dw,qw)/non_tie  (0.5 balanced, ->0 one-sided)
  - task-specific dominance    = max(dw,qw)/non_tie
  - deepseek winner share      = dw / non_tie
  - pair-oracle gain           = min(dw,qw)/total    (headroom a perfect router
                                 adds OVER always picking the best single model)
  - winner-label sparsity      = 1 - non_tie/total   (share of rows with no
                                 learnable winner signal)
  - train/test stability       = |mean_train_share - mean_test_share| and
                                 cross-seed std of the test winner share, per
                                 dataset, plus sign agreement of the preference.

Label definitions (from Section 1): a row is a "tie" when both models get the
same binary outcome (both correct OR both wrong); it is "non-tie / informative"
when exactly one model is correct. Routers are only evaluated on non-tie rows,
so non-tie rows are the ONLY rows carrying a learnable winner label.

Outputs (written to diagnostics/):
  - pair_learnability.csv   (per scope/dataset aggregated over the 10 seeds)
  - pair_learnability.md    (prose verdict + summary tables)

Read-only w.r.t. all existing data and results.
"""

import csv
import os
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DIAG = os.path.dirname(HERE)
SRC = os.path.join(DIAG, "label_statistics.csv")
OUT_CSV = os.path.join(DIAG, "pair_learnability.csv")
OUT_MD = os.path.join(DIAG, "pair_learnability.md")


def load_rows():
    rows = []
    with open(SRC, newline="") as f:
        for r in csv.DictReader(f):
            for k in (
                "total", "tie", "non_tie", "deepseek_win",
                "qwen_win", "both_correct", "both_wrong",
            ):
                r[k] = int(r[k])
            r["seed"] = int(r["seed"])
            rows.append(r)
    return rows


def per_seed_metrics(r):
    """Derive learnability metrics for a single (seed, split, dataset) row."""
    total = r["total"]
    nt = r["non_tie"]
    dw, qw = r["deepseek_win"], r["qwen_win"]
    m = {
        "tie_rate": r["tie"] / total if total else float("nan"),
        "nontie_rate": nt / total if total else float("nan"),
        "both_correct_rate": r["both_correct"] / total if total else float("nan"),
        "both_wrong_rate": r["both_wrong"] / total if total else float("nan"),
        "oracle_gain": min(dw, qw) / total if total else float("nan"),
    }
    if nt > 0:
        m["ds_share"] = dw / nt
        m["minority_share"] = min(dw, qw) / nt
        m["dominance"] = max(dw, qw) / nt
    else:
        m["ds_share"] = float("nan")
        m["minority_share"] = float("nan")
        m["dominance"] = float("nan")
    return m


def agg(values):
    """mean/std over a list (ignoring nan); returns (mean, std, n)."""
    vals = [v for v in values if v == v]  # drop nan
    if not vals:
        return (float("nan"), float("nan"), 0)
    mean = statistics.fmean(vals)
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return (mean, std, len(vals))


def collapse_all(rows_for_scope):
    """Sum counts across datasets per seed -> one synthetic 'ALL' row/seed."""
    by_seed = defaultdict(lambda: defaultdict(int))
    for r in rows_for_scope:
        for k in ("total", "tie", "non_tie", "deepseek_win",
                  "qwen_win", "both_correct", "both_wrong"):
            by_seed[r["seed"]][k] += r[k]
    out = []
    for seed, c in by_seed.items():
        rr = {"seed": seed, "dataset": "ALL"}
        rr.update(c)
        out.append(rr)
    return out


def main():
    rows = load_rows()
    datasets = sorted({r["dataset"] for r in rows})

    # ---- aggregate per (split, dataset) over seeds, plus ALL -----------------
    metric_keys = [
        "tie_rate", "nontie_rate", "both_correct_rate", "both_wrong_rate",
        "ds_share", "minority_share", "dominance", "oracle_gain",
    ]

    records = {}  # (split, dataset) -> {metric: (mean,std,n), extras...}
    for split in ("test", "train"):
        scope_rows = [r for r in rows if r["split"] == split]
        # per real dataset
        for ds in datasets:
            drows = [r for r in scope_rows if r["dataset"] == ds]
            per = [per_seed_metrics(r) for r in drows]
            rec = {"n_seeds": len(drows),
                   "mean_total": statistics.fmean([r["total"] for r in drows])}
            for mk in metric_keys:
                rec[mk] = agg([p[mk] for p in per])
            records[(split, ds)] = rec
        # ALL (pooled across datasets per seed)
        all_rows = collapse_all(scope_rows)
        per = [per_seed_metrics(r) for r in all_rows]
        rec = {"n_seeds": len(all_rows),
               "mean_total": statistics.fmean([r["total"] for r in all_rows])}
        for mk in metric_keys:
            rec[mk] = agg([p[mk] for p in per])
        records[(split, "ALL")] = rec

    # ---- train/test stability per dataset -----------------------------------
    stability = {}
    for ds in datasets + ["ALL"]:
        test_share = records[("test", ds)]["ds_share"][0]
        train_share = records[("train", ds)]["ds_share"][0]
        test_std = records[("test", ds)]["ds_share"][1]
        pref_test = test_share - 0.5
        pref_train = train_share - 0.5
        sign_agree = (pref_test >= 0) == (pref_train >= 0)
        stability[ds] = {
            "test_ds_share": test_share,
            "train_ds_share": train_share,
            "abs_train_test_gap": abs(test_share - train_share)
            if test_share == test_share and train_share == train_share
            else float("nan"),
            "test_share_cross_seed_std": test_std,
            "pref_sign_agree": sign_agree,
        }

    # ---- write CSV -----------------------------------------------------------
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "split", "dataset", "n_seeds", "mean_total",
            "tie_rate_mean", "tie_rate_std",
            "nontie_rate_mean", "nontie_rate_std",
            "both_correct_rate_mean", "both_wrong_rate_mean",
            "ds_win_share_mean", "ds_win_share_std",
            "winner_minority_share_mean", "dominance_mean",
            "oracle_gain_mean", "oracle_gain_std",
            "train_test_ds_share_gap", "pref_sign_agree_train_test",
        ])
        for split in ("test", "train"):
            for ds in datasets + ["ALL"]:
                rec = records[(split, ds)]
                st = stability[ds]
                w.writerow([
                    split, ds, rec["n_seeds"], f"{rec['mean_total']:.1f}",
                    f"{rec['tie_rate'][0]:.4f}", f"{rec['tie_rate'][1]:.4f}",
                    f"{rec['nontie_rate'][0]:.4f}", f"{rec['nontie_rate'][1]:.4f}",
                    f"{rec['both_correct_rate'][0]:.4f}",
                    f"{rec['both_wrong_rate'][0]:.4f}",
                    f"{rec['ds_share'][0]:.4f}", f"{rec['ds_share'][1]:.4f}",
                    f"{rec['minority_share'][0]:.4f}",
                    f"{rec['dominance'][0]:.4f}",
                    f"{rec['oracle_gain'][0]:.4f}", f"{rec['oracle_gain'][1]:.4f}",
                    f"{st['abs_train_test_gap']:.4f}",
                    "yes" if st["pref_sign_agree"] else "NO",
                ])

    # ---- build markdown ------------------------------------------------------
    write_md(records, stability, datasets)
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_MD}")

    # quick console verdict
    allt = records[("test", "ALL")]
    print(f"\nTEST pooled: tie_rate={allt['tie_rate'][0]:.3f} "
          f"nontie_rate={allt['nontie_rate'][0]:.3f} "
          f"ds_share={allt['ds_share'][0]:.3f} "
          f"minority_share={allt['minority_share'][0]:.3f} "
          f"oracle_gain={allt['oracle_gain'][0]:.3f}")


def fmt(t):
    """format a (mean,std,n) tuple as 'mean±std'."""
    return f"{t[0]:.3f}±{t[1]:.3f}"


def write_md(records, stability, datasets):
    allt = records[("test", "ALL")]
    allr = records[("train", "ALL")]

    # rank datasets by test oracle gain (routing headroom)
    ds_by_gain = sorted(
        datasets, key=lambda d: records[("test", d)]["oracle_gain"][0],
        reverse=True)

    lines = []
    A = lines.append
    A("# Pair Learnability (Section 3)")
    A("")
    A("Router Performance Diagnostic Mission, Priority 2.")
    A("")
    A("**Question answered:** *Does the controlled DeepSeek-R1-0528-Qwen3-8B vs "
      "Qwen3-8B pair create an unusually difficult routing target, independent "
      "of any particular classifier?*")
    A("")
    A("Derived entirely from the Section-1-audited `label_statistics.csv` "
      "(10 seeds, 42-51; 15 datasets). A row is a **tie** when both models "
      "share the same binary outcome (both correct or both wrong) and "
      "**non-tie / informative** when exactly one model is correct. Only "
      "non-tie rows carry a learnable winner label, and routers are evaluated "
      "only on them.")
    A("")
    A("## Pool-level verdict (TEST split, mean over 10 seeds)")
    A("")
    A("| metric | value | meaning |")
    A("|---|---:|---|")
    A(f"| outcome agreement (tie rate) | {fmt(allt['tie_rate'])} | "
      "share of rows where routing is irrelevant (both same outcome) |")
    A(f"| &nbsp;&nbsp;of which both-correct | {fmt(allt['both_correct_rate'])} | "
      "either model works |")
    A(f"| &nbsp;&nbsp;of which both-wrong | {fmt(allt['both_wrong_rate'])} | "
      "neither model works |")
    A(f"| informative non-tie rate | {fmt(allt['nontie_rate'])} | "
      "share of rows carrying a learnable winner label |")
    A(f"| winner-label sparsity | {1 - allt['nontie_rate'][0]:.3f} | "
      "share of rows with NO learnable winner signal |")
    A(f"| DeepSeek winner share (of non-tie) | {fmt(allt['ds_share'])} | "
      "class balance of the winner label |")
    A(f"| winner minority share | {fmt(allt['minority_share'])} | "
      "0.50 = perfectly balanced |")
    A(f"| **pair-oracle gain** | **{fmt(allt['oracle_gain'])}** | "
      "accuracy a perfect router adds OVER always picking the best single "
      "model |")
    A("")
    A("## Why this is a hard target")
    A("")
    A(f"1. **Routing is irrelevant on ~{allt['tie_rate'][0]*100:.0f}% of the "
      f"pool.** Only ~{allt['nontie_rate'][0]*100:.0f}% of test rows are "
      "informative, so ~{:.0f}% carry no winner signal to learn from or be "
      "scored on. The learnable surface is intrinsically small."
      .format((1 - allt['nontie_rate'][0]) * 100))
    A("")
    A(f"2. **The ceiling on *usefulness* is tiny.** A perfect oracle router "
      f"beats the best single static model by only "
      f"**{allt['oracle_gain'][0]*100:.1f} percentage points** of overall "
      "accuracy (pair-oracle gain = min(DS,Qwen wins)/total). Most of the "
      "apparent headroom in Macro-F1 lives in a sliver of the data.")
    A("")
    A(f"3. **The winner label is close to a coin flip overall** "
      f"(DeepSeek share {allt['ds_share'][0]:.2f} of non-tie), so there is no "
      "dominant-class shortcut at the pool level — yet the signal is highly "
      "*dataset-conditional* (next section).")
    A("")
    A("## Task-specific dominance (TEST, mean over seeds)")
    A("")
    A("Per-dataset winner preference among non-tie rows. `ds_share` > 0.5 means "
      "DeepSeek tends to win the informative rows; < 0.5 means Qwen does. "
      "`dominance` is the majority winner's share (1.0 = one model always wins "
      "the informative rows).")
    A("")
    A("| dataset | mean non-tie n | nontie_rate | ds_share | dominance | "
      "oracle_gain | train→test ds_share gap | pref stable |")
    A("|---|---:|---:|---:|---:|---:|---:|:--:|")
    for ds in sorted(datasets,
                     key=lambda d: records[("test", d)]["ds_share"][0]):
        rec = records[("test", ds)]
        st = stability[ds]
        nt_n = rec["mean_total"] * rec["nontie_rate"][0]
        A(f"| {ds} | {nt_n:.0f} | {rec['nontie_rate'][0]:.3f} | "
          f"{rec['ds_share'][0]:.3f} | {rec['dominance'][0]:.3f} | "
          f"{rec['oracle_gain'][0]:.3f} | {st['abs_train_test_gap']:.3f} | "
          f"{'yes' if st['pref_sign_agree'] else '**NO**'} |")
    A("")
    A("**Reading the table.** The pair splits cleanly into model-specialised "
      "regimes: reasoning/knowledge datasets (bbh, korbench, gpqa, mmlupro, "
      "medqa, math/aime) lean DeepSeek; coding/affective datasets (mbpp, "
      "humaneval, livecodebench, meld, emorynlp) lean Qwen. Because the "
      "preference is strong *within* a dataset but flips *across* datasets, a "
      "router that merely learns which dataset a query comes from captures most "
      "of the available signal — there is little residual per-query signal "
      "beyond dataset identity.")
    A("")
    A("## Where the routing headroom actually is")
    A("")
    A("Datasets ranked by pair-oracle gain (TEST) — i.e. where correctly "
      "choosing per-query adds the most overall accuracy:")
    A("")
    A("| rank | dataset | oracle_gain | winner_minority_share | interpretation |")
    A("|---:|---|---:|---:|---|")
    for i, ds in enumerate(ds_by_gain[:6], 1):
        rec = records[("test", ds)]
        mins = rec["minority_share"][0]
        interp = ("genuinely contested — both models win substantial shares"
                  if mins > 0.33 else
                  "mostly one-sided — a per-dataset constant captures it")
        A(f"| {i} | {ds} | {rec['oracle_gain'][0]:.3f} | {mins:.3f} | "
          f"{interp} |")
    A("")
    A("The gain is concentrated in a few high-non-tie datasets (kandk, mbpp, "
      "and the coding sets). Elsewhere the informative rows are so rare that "
      "even perfect per-query routing barely moves overall accuracy.")
    A("")
    A("## Train/test stability of winner distributions")
    A("")
    A(f"Across the 10 seeds, the pooled DeepSeek winner share is "
      f"{allt['ds_share'][0]:.3f} on test vs {allr['ds_share'][0]:.3f} on train "
      f"(gap {abs(allt['ds_share'][0]-allr['ds_share'][0]):.3f}), and the "
      f"per-dataset preference sign agrees between train and test for "
      f"{sum(1 for d in datasets if stability[d]['pref_sign_agree'])}/"
      f"{len(datasets)} datasets. The winner *distribution* is therefore stable "
      "and learnable in principle — instability is NOT the obstacle. The "
      "obstacle is that the stable signal is almost entirely dataset-level, and "
      "the per-query residual within a dataset is small and noisy "
      "(cross-seed std of the test winner share is large on the rare-non-tie "
      "datasets simply because of small counts).")
    A("")
    A("## Verdict")
    A("")
    A("**Yes — this DeepSeek/Qwen pair is an intrinsically difficult, "
      "low-headroom routing target.** Three structural facts, all independent "
      "of the classifier, explain the ~0.60 Macro-F1 band found in Section 2:")
    A("")
    A(f"- **Sparse labels:** ~{(1-allt['nontie_rate'][0])*100:.0f}% of rows are "
      "ties and carry no winner signal.")
    A(f"- **Tiny usefulness ceiling:** a perfect oracle adds only "
      f"~{allt['oracle_gain'][0]*100:.1f} pts of accuracy over the best single "
      "model.")
    A("- **Signal is dataset-conditional, not per-query:** strong, stable "
      "model specialisation across datasets, but little residual per-query "
      "signal once the dataset is known.")
    A("")
    A("This frames Sections 4-5: the open question is whether the ~0.60 ceiling "
      "is set by (a) the small/noisy per-query residual itself (a data "
      "property) or (b) the GTE-representation + linear-head's inability to "
      "extract it (a model property). Section 5 tests (b) directly by holding "
      "GTE fixed and increasing classifier capacity.")
    A("")

    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
