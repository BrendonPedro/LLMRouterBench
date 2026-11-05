#!/usr/bin/env python3
"""Score the frozen 5-repeat winner-label-stability main run.

Reuses wls_preflight_score.score_records() unchanged for each repeat.
Therefore benchmark evaluators and MBPP Bubblewrap execution are identical
to the already-validated pre-flight scoring path.

Scientific definition is frozen:
  original winner is the original stored non-tie winner;
  winner-stable iff that same winner wins >= 4 of 5 fresh repeats.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

import wls_common as C
import wls_preflight_score as P


DS = "DeepSeek-R1-0528-Qwen3-8B"
QW = "Qwen3-8B"
MODELS = [DS, QW]
N_REPEATS = 5

ROOT = Path(__file__).resolve().parents[2]
MAIN_DIR = ROOT / "diagnostics" / "winner_label_stability"
SAMPLE = MAIN_DIR / "frozen_sample.csv"
DEFAULT_GENS = MAIN_DIR / "generations.jsonl"

SIGNAL = {"bbh", "korbench", "mbpp"}
CHANCE = {"meld", "medqa", "finqa"}


def load_main_generations(path, sample):
    wanted = {
        (r.dataset, int(r.index))
        for r in sample.itertuples(index=False)
    }

    recs = [
        json.loads(x)
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]

    ok = {}
    errors = []

    for r in recs:
        if r.get("phase") != "main":
            sys.exit(f"non-main record found in {path}")

        q = (r["dataset"], int(r["index"]))
        if q not in wanted:
            sys.exit(f"generation outside frozen sample: {q}")

        if r["model"] not in MODELS:
            sys.exit(f"unexpected model: {r['model']}")

        if r.get("status") != "ok":
            errors.append(r)
            continue

        repeat = r.get("repeat")
        if repeat not in range(1, N_REPEATS + 1):
            sys.exit(f"invalid repeat number: {repeat}")

        k = (r["dataset"], int(r["index"]), r["model"], repeat)

        if k in ok:
            sys.exit(f"duplicate successful generation: {k}")

        ok[k] = r

    expected = len(wanted) * len(MODELS) * N_REPEATS

    if len(ok) != expected:
        sys.exit(
            f"expected {expected} successful generations, found {len(ok)}"
        )

    missing = []
    for dataset, index in sorted(wanted):
        for model in MODELS:
            for repeat in range(1, N_REPEATS + 1):
                k = (dataset, index, model, repeat)
                if k not in ok:
                    missing.append(k)

    if missing:
        sys.exit(f"missing generations: {missing[:20]}")

    print(
        f"validated {len(ok)} successful generations; "
        f"{len(errors)} error records"
    )

    return ok


def score_all(sample, gens):
    frames = []

    for repeat in range(1, N_REPEATS + 1):
        print(
            f"\n=== SCORING REPEAT {repeat}/{N_REPEATS} ===",
            flush=True,
        )

        one = {
            (dataset, index, model): rec
            for (dataset, index, model, rep), rec in gens.items()
            if rep == repeat
        }

        expected = len(sample) * len(MODELS)
        if len(one) != expected:
            sys.exit(
                f"repeat {repeat}: expected {expected} generations, "
                f"found {len(one)}"
            )

        scores, sandbox_ok = P.score_records(sample, one)
        scores["repeat"] = repeat

        if not sandbox_ok:
            sys.exit(f"repeat {repeat}: MBPP sandbox failed")

        errors = scores[scores["eval_error"].notna()]
        if len(errors):
            errors.to_csv(
                MAIN_DIR / f"scoring_errors_repeat_{repeat}.csv",
                index=False,
            )
            sys.exit(
                f"repeat {repeat}: {len(errors)} evaluation errors"
            )

        print(
            f"repeat {repeat}: {len(scores)}/{expected} scored; "
            f"MBPP sandbox OK",
            flush=True,
        )

        frames.append(scores)

    out = pd.concat(frames, ignore_index=True)

    if len(out) != len(sample) * len(MODELS) * N_REPEATS:
        sys.exit("unexpected final score-row count")

    return out


def repeated_outcome(ds, qw):
    ds = int(ds)
    qw = int(qw)

    if ds == 1 and qw == 0:
        return DS
    if ds == 0 and qw == 1:
        return QW
    return "tie"


def make_query_stability(sample, scores):
    sample_lookup = {
        (r.dataset, int(r.index)): r
        for r in sample.itertuples(index=False)
    }

    rows = []

    for (dataset, index), g in scores.groupby(["dataset", "index"]):
        stored = (
            g.groupby("model")["stored_score"]
            .first()
            .astype(int)
            .to_dict()
        )

        if stored[DS] == stored[QW]:
            sys.exit(
                f"frozen main sample unexpectedly contains original tie: "
                f"{dataset}/{index}"
            )

        original = DS if stored[DS] == 1 else QW
        opposite = QW if original == DS else DS

        outcomes = []
        for repeat in range(1, N_REPEATS + 1):
            rg = g[g["repeat"] == repeat]

            vals = (
                rg.set_index("model")["fresh_score"]
                .astype(int)
                .to_dict()
            )

            if set(vals) != set(MODELS):
                sys.exit(
                    f"incomplete pair at {dataset}/{index} "
                    f"repeat {repeat}"
                )

            outcomes.append(
                repeated_outcome(vals[DS], vals[QW])
            )

        original_wins = outcomes.count(original)
        opposite_wins = outcomes.count(opposite)
        ties = outcomes.count("tie")

        sr = sample_lookup[(dataset, int(index))]

        row = {
            "dataset": dataset,
            "index": int(index),
            "original_winner": original,
            "original_winner_repeat_wins": original_wins,
            "opposite_winner_repeat_wins": opposite_wins,
            "tie_repeats": ties,
            "winner_stable": original_wins >= 4,
            "repeat_outcomes": "|".join(outcomes),
        }

        for optional in ("query_sha1", "fold"):
            if hasattr(sr, optional):
                row[optional] = getattr(sr, optional)

        rows.append(row)

    out = pd.DataFrame(rows)

    if len(out) != len(sample):
        sys.exit(
            f"expected {len(sample)} query rows, found {len(out)}"
        )

    return out


def summarize(group):
    n = len(group)
    repeat_total = n * N_REPEATS

    return pd.Series({
        "n_queries": n,
        "stable_queries": int(group["winner_stable"].sum()),
        "stable_rate": float(group["winner_stable"].mean()),
        "original_winner_repeat_wins": int(
            group["original_winner_repeat_wins"].sum()
        ),
        "original_winner_repeat_win_rate": (
            group["original_winner_repeat_wins"].sum() / repeat_total
        ),
        "opposite_winner_repeat_wins": int(
            group["opposite_winner_repeat_wins"].sum()
        ),
        "opposite_winner_repeat_win_rate": (
            group["opposite_winner_repeat_wins"].sum() / repeat_total
        ),
        "tie_repeats": int(group["tie_repeats"].sum()),
        "tie_rate": group["tie_repeats"].sum() / repeat_total,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--generations",
        type=Path,
        default=DEFAULT_GENS,
    )
    args = ap.parse_args()

    sample = C.load_sample(SAMPLE)
    C.assert_sample_eligible(sample)

    print(f"frozen sample: {len(sample)} queries")

    gens = load_main_generations(args.generations, sample)

    scores = score_all(sample, gens)
    scores_path = MAIN_DIR / "main_scores.csv"
    scores.to_csv(scores_path, index=False)

    query = make_query_stability(sample, scores)
    query_path = MAIN_DIR / "query_stability.csv"
    query.to_csv(query_path, index=False)

    benchmark = (
        query.groupby("dataset", sort=True)
        .apply(summarize, include_groups=False)
        .reset_index()
    )
    benchmark_path = MAIN_DIR / "benchmark_stability.csv"
    benchmark.to_csv(benchmark_path, index=False)

    grouped = query.copy()
    grouped["group"] = grouped["dataset"].map(
        lambda x: (
            "signal"
            if x in SIGNAL
            else "chance"
            if x in CHANCE
            else "unexpected"
        )
    )

    if (grouped["group"] == "unexpected").any():
        sys.exit("unexpected benchmark in group comparison")

    groups = (
        grouped.groupby("group", sort=True)
        .apply(summarize, include_groups=False)
        .reset_index()
    )
    group_path = MAIN_DIR / "group_comparison.csv"
    groups.to_csv(group_path, index=False)

    print("\n=== MAIN SCORING COMPLETE ===")
    print(f"scores:    {scores_path}")
    print(f"queries:   {query_path}")
    print(f"benchmark: {benchmark_path}")
    print(f"groups:    {group_path}")

    print("\n=== BENCHMARK STABILITY ===")
    print(
        benchmark[
            ["dataset", "n_queries", "stable_queries", "stable_rate",
             "original_winner_repeat_win_rate", "tie_rate"]
        ].to_string(index=False)
    )

    print("\n=== GROUP COMPARISON ===")
    print(
        groups[
            ["group", "n_queries", "stable_queries", "stable_rate",
             "original_winner_repeat_win_rate", "tie_rate"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
