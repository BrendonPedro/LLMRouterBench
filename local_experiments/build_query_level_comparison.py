import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


DS = "DeepSeek-R1-0528-Qwen3-8B"
QWEN = "Qwen3-8B"

OUT = Path("local_experiments/results")
all_rows = []


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            rows.append({
                "dataset": x["dataset"],
                "index": x.get("index"),
                "query": x["query"],
                "deepseek_score": float(x["records"][DS]),
                "qwen_score": float(x["records"][QWEN]),
            })
    return pd.DataFrame(rows)


for seed in range(42, 52):
    print(f"Processing seed {seed}...")

    base = Path(
        f"baselines/AvengersPro/data/"
        f"small_models_seed_{seed}/seed{seed}_split0.7"
    )

    train = load_jsonl(base / "train.jsonl")
    test = load_jsonl(base / "test.jsonl")

    # --------------------------------------------------
    # Reproduce exact TF-IDF router
    # --------------------------------------------------
    train_nt = train[
        train["deepseek_score"] != train["qwen_score"]
    ].copy()

    train_nt["label"] = np.where(
        train_nt["qwen_score"] > train_nt["deepseek_score"],
        1,
        0,
    )

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=20000,
    )

    X_train = vec.fit_transform(train_nt["query"])
    X_test = vec.transform(test["query"])

    clf = LogisticRegression(
        max_iter=2000,
        random_state=42,
    )
    clf.fit(X_train, train_nt["label"])

    pred = clf.predict(X_test)

    test["tfidf_choice"] = np.where(pred == 1, QWEN, DS)

    # --------------------------------------------------
    # Load AvengersPro per-query decisions
    # --------------------------------------------------
    details_path = OUT / (
        f"avengerspro_pair_seed{seed}_routing_details.json"
    )

    with open(details_path, "r", encoding="utf-8") as f:
        details = json.load(f)

    av_map = {}

    for d in details:
        key = (d["dataset"], str(d["index"]))
        selected = d.get("selected_models", [])

        av_map[key] = selected[0] if selected else None

    test["avengerspro_choice"] = [
        av_map.get((r.dataset, str(r.index)))
        for r in test.itertuples()
    ]

    if test["avengerspro_choice"].isna().any():
        missing = test["avengerspro_choice"].isna().sum()
        raise RuntimeError(
            f"Seed {seed}: {missing} AvengersPro decisions missing"
        )

    # --------------------------------------------------
    # Oracle / scores / regret
    # --------------------------------------------------
    test["oracle_score"] = test[
        ["deepseek_score", "qwen_score"]
    ].max(axis=1)

    test["oracle_winner"] = np.where(
        test["deepseek_score"] > test["qwen_score"],
        DS,
        np.where(
            test["qwen_score"] > test["deepseek_score"],
            QWEN,
            "tie",
        ),
    )

    test["tfidf_score"] = np.where(
        test["tfidf_choice"] == QWEN,
        test["qwen_score"],
        test["deepseek_score"],
    )

    test["avengerspro_score"] = np.where(
        test["avengerspro_choice"] == QWEN,
        test["qwen_score"],
        test["deepseek_score"],
    )

    test["tfidf_regret"] = (
        test["oracle_score"] - test["tfidf_score"]
    )

    test["avengerspro_regret"] = (
        test["oracle_score"] - test["avengerspro_score"]
    )

    test["oracle_margin"] = (
        test["deepseek_score"] - test["qwen_score"]
    ).abs()

    test["non_tie"] = test["oracle_winner"] != "tie"

    test["tfidf_captures_oracle"] = (
        test["tfidf_regret"] == 0
    )

    test["avengerspro_captures_oracle"] = (
        test["avengerspro_regret"] == 0
    )

    test["both_miss"] = (
        (test["tfidf_regret"] > 0)
        & (test["avengerspro_regret"] > 0)
    )

    test["tfidf_only_capture"] = (
        test["tfidf_captures_oracle"]
        & ~test["avengerspro_captures_oracle"]
    )

    test["avengerspro_only_capture"] = (
        test["avengerspro_captures_oracle"]
        & ~test["tfidf_captures_oracle"]
    )

    test["both_capture"] = (
        test["tfidf_captures_oracle"]
        & test["avengerspro_captures_oracle"]
    )

    test["query_words"] = (
        test["query"].fillna("").str.split().str.len()
    )

    test.insert(0, "seed", seed)

    all_rows.append(test)


# ======================================================
# MASTER QUERY-LEVEL FILE
# ======================================================
master = pd.concat(all_rows, ignore_index=True)

master_path = OUT / "query_level_router_comparison.csv"
master.to_csv(master_path, index=False)


# ======================================================
# DATASET SUMMARY, NON-TIE QUERIES ONLY
# ======================================================
nt = master[master["non_tie"]].copy()

summary = (
    nt.groupby("dataset")
    .agg(
        non_tie_queries=("query", "size"),

        deepseek_wins=(
            "oracle_winner",
            lambda s: (s == DS).sum()
        ),

        qwen_wins=(
            "oracle_winner",
            lambda s: (s == QWEN).sum()
        ),

        tfidf_capture_pct=(
            "tfidf_captures_oracle",
            lambda s: s.mean() * 100
        ),

        avengerspro_capture_pct=(
            "avengerspro_captures_oracle",
            lambda s: s.mean() * 100
        ),

        both_miss_queries=("both_miss", "sum"),
        tfidf_only_queries=("tfidf_only_capture", "sum"),
        avengerspro_only_queries=(
            "avengerspro_only_capture", "sum"
        ),
        both_capture_queries=("both_capture", "sum"),

        tfidf_mean_regret_pp=(
            "tfidf_regret",
            lambda s: s.mean() * 100
        ),

        avengerspro_mean_regret_pp=(
            "avengerspro_regret",
            lambda s: s.mean() * 100
        ),
    )
    .reset_index()
)

summary["both_miss_pct"] = (
    summary["both_miss_queries"]
    / summary["non_tie_queries"]
    * 100
)

summary = summary.sort_values(
    "both_miss_pct",
    ascending=False,
)

summary_path = OUT / "query_level_failure_summary.csv"
summary.to_csv(summary_path, index=False)


# ======================================================
# FOCUS DATASETS: EXAMPLES BOTH ROUTERS MISSED
# ======================================================
focus = ["kandk", "mbpp", "gpqa", "korbench", "bbh"]

examples = (
    master[
        master["dataset"].isin(focus)
        & master["both_miss"]
    ]
    .sort_values(
        ["dataset", "oracle_margin"],
        ascending=[True, False],
    )
    .groupby("dataset", group_keys=False)
    .head(20)
)

examples_path = OUT / "both_missed_focus_examples.csv"
examples.to_csv(examples_path, index=False)


print()
print("=" * 100)
print("QUERY-LEVEL ROUTING FAILURE SUMMARY")
print("=" * 100)

display_cols = [
    "dataset",
    "non_tie_queries",
    "deepseek_wins",
    "qwen_wins",
    "tfidf_capture_pct",
    "avengerspro_capture_pct",
    "both_miss_pct",
    "tfidf_mean_regret_pp",
    "avengerspro_mean_regret_pp",
]

print(
    summary[display_cols]
    .round(2)
    .to_string(index=False)
)

print()
print("Saved:")
print(master_path)
print(summary_path)
print(examples_path)
