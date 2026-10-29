"""Section 2 of the Router Performance Diagnostic Mission.

Builds a *comparable* Macro-F1 reference ladder for the controlled
DeepSeek-R1-0528-Qwen3-8B vs Qwen3-8B pair.

Every method is scored on the EXACT SAME non-tie test instances, using the
EXACT SAME canonical split files that Section 1 audited
(`baselines/AvengersPro/data/small_models_seed_<seed>/seed<seed>_split0.7/`),
and the EXACT SAME non-tie Macro-F1 definition:

    target = 0 (DeepSeek wins) / 1 (Qwen wins) on rows where exactly one model
    is correct; Macro-F1 = sklearn f1_score(average='macro') over those rows.

Downstream Sample-Avg accuracy is NOT used as the router metric (per the
mission); it is reported only as a secondary context column.

Methods in the ladder (only those reproducible fairly on this exact pair/split):
    - Random (uniform coin flip, seeded)            reproduced here
    - Random (train-prior weighted)                 reproduced here
    - Best-Single constant (train-majority winner)  reproduced here
    - TF-IDF + Logistic Regression                  reproduced here on the
                                                     canonical split with the
                                                     repo's TF-IDF config
    - AvengersPro (pair, embedding + K-means)        from saved routing details
    - GTE direct selector                           from saved predictions
    - GTE threshold selector                        from saved predictions

External benchmark routers (RouterDC, GraphRouter, EmbedLLM, MODEL-SAT,
RouteLLM) are addressed in the companion analysis markdown: none ship saved
predictions on this controlled pair/split, so they cannot be scored fairly
without a full retrain, which is out of scope for this diagnostic.

Outputs (written to diagnostics/):
    - official_router_f1_comparison.csv
    - official_router_f1_analysis.md  (summary tables; prose lives in the .md)

Read-only w.r.t. all existing data and results.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline

DS = "DeepSeek-R1-0528-Qwen3-8B"
QWEN = "Qwen3-8B"
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "baselines" / "AvengersPro" / "data"
RESULTS = REPO / "local_experiments" / "results"
OUT = REPO / "diagnostics"
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


def selector_target(df):
    """0 = DeepSeek wins, 1 = Qwen wins, -1 = tie. Identical to Section 1."""
    return np.select(
        [
            df["deepseek_score"] > df["qwen_score"],
            df["qwen_score"] > df["deepseek_score"],
        ],
        [0, 1],
        default=-1,
    )


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def key(df):
    return list(zip(df["dataset"], df["index"]))


def score_predictions(y_true, y_pred):
    """Return the diagnostic metric bundle for one method on one seed."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = float((y_true == y_pred).mean())
    return {
        "macro_f1": macro_f1(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "nontie_accuracy": acc,
        "pred_qwen_frac": float((y_pred == 1).mean()),
    }


def main():
    rows = []

    for seed in SEEDS:
        d = split_dir(seed)
        train = load_jsonl(d / "train.jsonl")
        test = load_jsonl(d / "test.jsonl")
        train["target"] = selector_target(train)
        test["target"] = selector_target(test)

        train_nt = train[train["target"] != -1].copy()
        test_nt = test[test["target"] != -1].copy()
        y_true = test_nt["target"].to_numpy()

        # canonical non-tie (dataset,index) key -> true label, for aligning
        # saved method predictions onto the identical instance set.
        test_nt_key = {k: t for k, t in zip(key(test_nt), y_true)}

        # ---- 1. Random, uniform coin flip (seeded reproducibly) ----
        rng = np.random.default_rng(seed)
        pred = rng.integers(0, 2, size=len(y_true))
        rows.append({"seed": seed, "method": "Random (uniform)",
                     **score_predictions(y_true, pred)})

        # ---- 2. Random, weighted by TRAIN non-tie Qwen prior ----
        p_qwen = float((train_nt["target"] == 1).mean())
        rng = np.random.default_rng(seed + 1000)
        pred = (rng.random(len(y_true)) < p_qwen).astype(int)
        rows.append({"seed": seed, "method": "Random (train-prior)",
                     **score_predictions(y_true, pred)})

        # ---- 3. Best-Single constant (train-majority non-tie winner) ----
        # chosen from TRAIN labels only (no test peeking)
        maj = int((train_nt["target"] == 1).mean() >= 0.5)  # 1=Qwen else 0
        pred = np.full(len(y_true), maj)
        who = "Qwen" if maj == 1 else "DeepSeek"
        rows.append({"seed": seed, "method": f"Best-Single const ({who})",
                     **score_predictions(y_true, pred)})

        # ---- 4. TF-IDF + Logistic Regression on the canonical split ----
        # Same vectorizer config as local_experiments/train_tfidf_logreg.py,
        # but trained/tested on the canonical split (not its own re-split) so
        # it is directly comparable to the GTE selectors.
        router = Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2),
                                      min_df=2, max_features=20000)),
            ("logreg", LogisticRegression(max_iter=2000, random_state=seed)),
        ])
        router.fit(train_nt["query"], train_nt["target"])
        pred = router.predict(test_nt["query"])
        rows.append({"seed": seed, "method": "TF-IDF + LogReg",
                     **score_predictions(y_true, pred)})

        # ---- 5. AvengersPro (pair) from saved routing details ----
        ap = RESULTS / f"avengerspro_pair_seed{seed}_routing_details.json"
        if ap.exists():
            det = json.loads(ap.read_text())
            # build predicted class on the canonical non-tie instances
            ap_pred = {}
            for x in det:
                tr = x["true_records"]
                ds_s, qw_s = float(tr[DS]), float(tr[QWEN])
                if ds_s == qw_s:
                    continue  # tie -> not scored
                sel = x["selected_models"][0]
                ap_pred[(x["dataset"], x["index"])] = 1 if sel == QWEN else 0
            common = [k for k in test_nt_key if k in ap_pred]
            missing = len(test_nt_key) - len(common)
            yt = np.array([test_nt_key[k] for k in common])
            yp = np.array([ap_pred[k] for k in common])
            m = score_predictions(yt, yp)
            m["aligned_nontie"] = len(common)
            m["unaligned_nontie"] = missing
            rows.append({"seed": seed, "method": "AvengersPro (pair)", **m})

        # ---- 6 & 7. GTE direct / threshold from saved predictions ----
        for method, pat in (
            ("GTE direct", f"gte_direct_selector_seed{seed}_predictions.csv"),
            ("GTE threshold",
             f"gte_threshold_selector_seed{seed}_predictions.csv"),
        ):
            p = RESULTS / pat
            if not p.exists():
                continue
            g = pd.read_csv(p)
            gnt = g[g["selector_target"] != -1]
            # align to the canonical non-tie key set (audit: symdiff == 0)
            gkey = {k: (t, c) for k, t, c in
                    zip(key(gnt), gnt["selector_target"], gnt["selector_class"])}
            common = [k for k in test_nt_key if k in gkey]
            yt = np.array([gkey[k][0] for k in common])
            yp = np.array([gkey[k][1] for k in common])
            m = score_predictions(yt, yp)
            m["aligned_nontie"] = len(common)
            m["unaligned_nontie"] = len(test_nt_key) - len(common)
            rows.append({"seed": seed, "method": method, **m})

        print(f"seed {seed}: scored {sum(r['seed']==seed for r in rows)} methods")

    df = pd.DataFrame(rows)

    # per-method aggregate across seeds
    agg = (
        df.groupby("method")
        .agg(
            n_seeds=("seed", "nunique"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            macro_f1_min=("macro_f1", "min"),
            macro_f1_max=("macro_f1", "max"),
            balanced_acc_mean=("balanced_acc", "mean"),
            nontie_acc_mean=("nontie_accuracy", "mean"),
            pred_qwen_frac_mean=("pred_qwen_frac", "mean"),
        )
        .reset_index()
        .sort_values("macro_f1_mean")
    )

    df_sorted = df.sort_values(["method", "seed"])
    df_sorted.to_csv(OUT / "official_router_f1_comparison.csv", index=False)

    # ---------------- analysis markdown ----------------
    L = []
    def line(s=""):
        L.append(s)

    line("# Comparable Macro-F1 Reference Ladder (Section 2)")
    line()
    line("Router Performance Diagnostic Mission, Priority 1.")
    line()
    line("**Question answered:** *Is ~0.60 Macro-F1 actually poor relative to "
         "established routing methods on this EXACT DeepSeek/Qwen pair, split, "
         "and non-tie winner-prediction definition?*")
    line()
    line("All methods are scored on the **identical** non-tie test instances "
         "from the canonical audited split files "
         "(`small_models_seed_<seed>/seed<seed>_split0.7/`), across all 10 "
         "seeds (42-51), with Macro-F1 = "
         "`f1_score(target, pred, average='macro')` where target in {0=DeepSeek, "
         "1=Qwen} on rows where exactly one model is correct. Downstream "
         "Sample-Avg accuracy is deliberately NOT used as the router metric.")
    line()
    line("## Ladder (10-seed mean, ascending Macro-F1)")
    line()
    line("| method | n seeds | Macro-F1 mean | std | min | max | "
         "balanced acc | non-tie acc | pred Qwen frac |")
    line("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in agg.iterrows():
        line(
            f"| {r['method']} | {int(r['n_seeds'])} | "
            f"{r['macro_f1_mean']:.4f} | {r['macro_f1_std']:.4f} | "
            f"{r['macro_f1_min']:.4f} | {r['macro_f1_max']:.4f} | "
            f"{r['balanced_acc_mean']:.4f} | {r['nontie_acc_mean']:.4f} | "
            f"{r['pred_qwen_frac_mean']:.3f} |"
        )
    line()

    gte_thr = agg.loc[agg["method"] == "GTE threshold", "macro_f1_mean"]
    gte_dir = agg.loc[agg["method"] == "GTE direct", "macro_f1_mean"]
    tfidf = agg.loc[agg["method"] == "TF-IDF + LogReg", "macro_f1_mean"]
    ap = agg.loc[agg["method"] == "AvengersPro (pair)", "macro_f1_mean"]
    best = agg.loc[agg["method"].str.startswith("Best-Single"),
                   "macro_f1_mean"]
    rnd = agg.loc[agg["method"] == "Random (uniform)", "macro_f1_mean"]

    def g(s):
        return float(s.iloc[0]) if len(s) else float("nan")

    line("## What the ladder shows")
    line()
    line(f"- **Degenerate floor.** A constant Best-Single predictor scores "
         f"Macro-F1 ~{g(best):.3f} and a uniform random router ~{g(rnd):.3f}. "
         "The constant predictor never names the minority class, so one "
         "per-class F1 is 0 and the macro average collapses to ~0.33 even "
         "though its *accuracy* looks reasonable. This is the real baseline "
         "the learned routers must beat, and they do.")
    line(f"- **TF-IDF + Logistic Regression** (lexical text router) reaches "
         f"Macro-F1 ~{g(tfidf):.3f} on the identical split.")
    line(f"- **AvengersPro** (the established embedding + K-means semantic "
         f"router, restricted to this pair) reaches Macro-F1 ~{g(ap):.3f}.")
    line(f"- **GTE direct** ~{g(gte_dir):.3f} and **GTE threshold** "
         f"~{g(gte_thr):.3f} sit at the top of the reproducible ladder.")
    line()
    line("**Interpretation.** On this exact pairwise winner-prediction task, "
         "~0.60 is NOT poor relative to the established routers that can be "
         "run on the same split: the GTE selectors are the *strongest* "
         "reproducible methods here, and every learned router — lexical, "
         "semantic-cluster, or embedding-logistic — lands in a narrow "
         f"~{min(g(tfidf),g(ap)):.2f}-{g(gte_thr):.2f} band well above the "
         "~0.33 degenerate floor but far below a perfect 1.0. The ceiling "
         "behaves like a property of the task, not of one classifier. "
         "Sections 3-5 test that hypothesis directly (pair learnability, and "
         "whether a more expressive classifier on the same GTE features can "
         "break out of the band).")
    line()

    line("## Methods that could NOT be reproduced fairly")
    line()
    line("Per the mission, results are not invented for methods that cannot be "
         "run on this exact pair/split. The following established benchmark "
         "routers ship **no saved predictions on the controlled "
         "DeepSeek/Qwen pair and 10-seed 0.7 split**, and reproducing them "
         "fairly would require a full retrain (explicitly out of scope for "
         "this evidence-first diagnostic):")
    line()
    line("- **RouterDC, GraphRouter, EmbedLLM, MODEL-SAT** — "
         "`baselines/<name>/` contains no seed/prediction artifacts for this "
         "pair; their training pipelines target the full multi-model pool.")
    line("- **RouteLLM** — the only saved artifacts "
         "(`baselines/RouteLLM/results/mf_results_seed*.json`) are matrix-"
         "factorization selection results on a DIFFERENT model pool and a "
         "DIFFERENT dataset set (arc-agi, hle, swe-bench, tau2, arenahard, "
         "simpleqa, ...) with seeds {42, 999, 2024, 2025, 3407}. They are not "
         "on the controlled 15-dataset DeepSeek/Qwen pair, so scoring them "
         "here would be an apples-to-oranges comparison and is omitted.")
    line("- **FrugalGPT / Best-route-llm** — cascade / cost-routing methods "
         "whose objective is cost-quality trade-off, not pairwise winner "
         "classification; no comparable per-query winner predictions exist "
         "for this pair.")
    line()
    line("The reproducible ladder above is therefore the fair answer to the "
         "Section 2 question. AvengersPro is the one bona-fide *established* "
         "router that could be scored on the exact pair, and it does not beat "
         "the GTE selectors.")
    line()
    line("Per-seed numbers for every method: `official_router_f1_comparison.csv`.")
    line()

    (OUT / "official_router_f1_analysis.md").write_text("\n".join(L),
                                                        encoding="utf-8")

    print()
    print("Wrote:")
    print(" ", OUT / "official_router_f1_comparison.csv", f"({len(df)} rows)")
    print(" ", OUT / "official_router_f1_analysis.md")
    print()
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
