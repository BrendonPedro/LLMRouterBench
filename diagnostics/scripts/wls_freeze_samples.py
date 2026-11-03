"""Freeze the winner-label stability main sample, then the disjoint pre-flight sample.

Implements the mission's "Frozen Sample Strategy" exactly:
  frame     : FinQA rows are eligible only if the `answer` field read by the unchanged
              FinQAEvaluator is non-empty after stripping whitespace (data-validity rule,
              declared before any generation; no other benchmark is filtered).
  main      : eligible original non-ties only, rows whose pool scores equal the stored scores,
              sorted by (dataset, index); ONE rng = default_rng(20261101); benchmarks in
              the fixed order; class 0 then class 1; rng.choice(n_class, 10, replace=False).
  pre-flight: all eligible rows (ties included) minus the main sample, sorted by (dataset, index);
              ONE rng = default_rng(20261102); same benchmark order;
              rng.choice(n, 20, replace=False).

Uses only the routing pool and the stored original outputs -- no fresh model output.
Refuses to overwrite an existing sample file with different content, and fails loudly if
an ineligible FinQA row ends up in either sample.

Usage: python diagnostics/scripts/wls_freeze_samples.py [--check | --redraw-preflight]
  --check             regenerate in memory and compare with the committed files; write nothing.
  --redraw-preflight  allow replacing the pre-flight sample and manifest (only while no
                      pre-flight generation exists); the main sample must still be identical.
"""

import argparse
import io
import json
import sys

import numpy as np
import pandas as pd

import wls_common as C

COLUMNS = ["draw_order", "dataset", "group", "index", "query_sha1", "prompt_sha256", "fold",
           "target", "original_outcome", "stored_score_ds", "stored_score_qwen"]
OUTCOME = {0: "deepseek_only", 1: "qwen_only"}


def stored_outcome(ds, qw):
    if ds == 1 and qw == 0:
        return "deepseek_only"
    if ds == 0 and qw == 1:
        return "qwen_only"
    return "both_correct" if ds == 1 else "both_wrong"


def build_frame():
    pool = C.load_pool()
    pool = pool[pool["dataset"].isin(C.BENCHMARKS)].copy()
    stored = {(b, m): C.load_stored(b, m) for b in C.BENCHMARKS for m in C.MODELS}
    rows = []
    for r in pool.itertuples(index=False):
        s_ds, s_qw = stored[(r.dataset, C.DS)], stored[(r.dataset, C.QWEN)]
        if r.index not in s_ds or r.index not in s_qw:
            raise RuntimeError(f"pool row {r.dataset}/{r.index} has no stored record")
        p_ds, p_qw = s_ds[r.index]["prompt"], s_qw[r.index]["prompt"]
        if not (p_ds == p_qw == r.query):
            raise RuntimeError(f"prompt mismatch at {r.dataset}/{r.index}")
        rows.append({
            "dataset": r.dataset, "group": C.GROUP[r.dataset], "index": int(r.index),
            "query_sha1": r.query_sha1, "prompt_sha256": C.sha256_text(r.query),
            "fold": int(r.fold), "target": int(r.target),
            "pool_score_ds": float(r.deepseek_score), "pool_score_qwen": float(r.qwen_score),
            "stored_score_ds": float(s_ds[r.index]["score"]),
            "stored_score_qwen": float(s_qw[r.index]["score"]),
        })
    frame = pd.DataFrame(rows).sort_values(["dataset", "index"]).reset_index(drop=True)
    assert frame["query_sha1"].notna().all()
    empty_finqa = C.finqa_empty_answer_indices()
    frame["eligible"] = ~((frame["dataset"] == "finqa") & frame["index"].isin(empty_finqa))
    frame["stored_agrees"] = ((frame["pool_score_ds"] == frame["stored_score_ds"])
                              & (frame["pool_score_qwen"] == frame["stored_score_qwen"]))
    return frame


def draw_main(frame):
    rng = np.random.default_rng(C.MAIN_SEED)
    eligible = frame[frame["eligible"] & frame["target"].isin([0, 1]) & frame["stored_agrees"]]
    picks, avail = [], {}
    for b in C.BENCHMARKS:
        for cls in (0, 1):
            rows = (eligible[(eligible["dataset"] == b) & (eligible["target"] == cls)]
                    .sort_values(["dataset", "index"]).reset_index(drop=True))
            avail[(b, cls)] = len(rows)
            if len(rows) < C.MAIN_PER_CLASS:
                # The mission's fallback rule would apply here; it is not expected to trigger.
                raise RuntimeError(f"{b} class {cls}: only {len(rows)} rows; fallback rule needed")
            idx = rng.choice(len(rows), size=C.MAIN_PER_CLASS, replace=False)
            picks.append(rows.iloc[idx])
    main = pd.concat(picks, ignore_index=True)
    main["original_outcome"] = main["target"].map(OUTCOME)
    main["draw_order"] = np.arange(1, len(main) + 1)
    return main[COLUMNS], avail


def draw_preflight(frame, main):
    rng = np.random.default_rng(C.PREFLIGHT_SEED)
    taken = set(zip(main["dataset"], main["index"]))
    rest = frame[frame["eligible"]
                 & [(d, i) not in taken for d, i in zip(frame["dataset"], frame["index"])]]
    picks, avail = [], {}
    for b in C.BENCHMARKS:
        rows = rest[rest["dataset"] == b].sort_values(["dataset", "index"]).reset_index(drop=True)
        avail[b] = len(rows)
        idx = rng.choice(len(rows), size=C.PREFLIGHT_PER_BENCHMARK, replace=False)
        picks.append(rows.iloc[idx])
    pre = pd.concat(picks, ignore_index=True)
    # Outcome against the stored scores, the pre-flight reference (mission: gross-mismatch test).
    pre["original_outcome"] = [stored_outcome(a, b) for a, b in
                               zip(pre["stored_score_ds"], pre["stored_score_qwen"])]
    pre["draw_order"] = np.arange(1, len(pre) + 1)
    return pre[COLUMNS], avail


def to_csv_text(df):
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator="\n")
    return buf.getvalue()


def write_once(path, text):
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            sys.exit(f"REFUSING to overwrite {path}: regenerated content differs")
        print(f"unchanged: {path.relative_to(C.REPO)}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"wrote: {path.relative_to(C.REPO)}")


def build_all():
    frame = build_frame()
    main, main_avail = draw_main(frame)
    pre, pre_avail = draw_preflight(frame, main)
    empty_finqa = set(frame.loc[~frame["eligible"], "index"].astype(int))
    C.assert_sample_eligible(main, empty_finqa)
    C.assert_sample_eligible(pre, empty_finqa)
    manifest = {
        "description": "Winner-label stability frozen samples (mission: Frozen Sample Strategy). "
                       "Chosen from the routing pool and stored original outputs only; "
                       "no fresh model output was used.",
        "script": "diagnostics/scripts/wls_freeze_samples.py",
        "main": {"seed": C.MAIN_SEED, "per_class": C.MAIN_PER_CLASS,
                 "benchmark_order": C.BENCHMARKS, "class_order": [0, 1],
                 "available_per_benchmark_class": {f"{b}|{c}": n for (b, c), n in main_avail.items()},
                 "rows": len(main), "file": str(C.MAIN_SAMPLE_CSV.relative_to(C.REPO))},
        "preflight": {"seed": C.PREFLIGHT_SEED, "per_benchmark": C.PREFLIGHT_PER_BENCHMARK,
                      "frame": "all pool rows incl. ties, minus the main sample",
                      "available_per_benchmark": pre_avail,
                      "rows": len(pre), "file": str(C.PREFLIGHT_SAMPLE_CSV.relative_to(C.REPO))},
        "eligibility_rule": "FinQA only: eligible iff the `answer` field read by the unchanged "
                            "FinQAEvaluator is non-empty after stripping whitespace. Data-validity "
                            "rule declared before any generation; not performance-based.",
        "finqa_excluded_empty_answer": sorted(empty_finqa),
        "frame_rows_not_matching_stored": frame.loc[~frame["stored_agrees"], ["dataset", "index"]]
                                               .to_dict("records"),
        "stored_files": {f"{b}|{m}": {"path": str(C.stored_path(b, m).relative_to(C.REPO)),
                                      "sha256": C.sha256_file(C.stored_path(b, m))}
                         for b in C.BENCHMARKS for m in C.MODELS},
        "identifiers": "(dataset, index) = 1-based position in the evaluator's load_data('test') "
                       "order and the key of results/bench records; query_sha1 = sha1(prompt)[:16] "
                       "as in diagnostics/stratified_cv/fold_assignments.csv; prompt_sha256 = "
                       "sha256 of the exact stored prompt.",
    }
    return main, pre, manifest


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true")
    g.add_argument("--redraw-preflight", action="store_true")
    args = ap.parse_args()
    main_df, pre_df, manifest = build_all()
    texts = {C.MAIN_SAMPLE_CSV: to_csv_text(main_df), C.PREFLIGHT_SAMPLE_CSV: to_csv_text(pre_df),
             C.SAMPLING_MANIFEST: json.dumps(manifest, indent=2) + "\n"}
    if args.check:
        bad = [p for p, t in texts.items() if not p.exists() or p.read_text(encoding="utf-8") != t]
        for p in texts:
            print(("MISMATCH " if p in bad else "OK       ") + str(p.relative_to(C.REPO)))
        sys.exit(1 if bad else 0)
    # Main sample is fixed first; the pre-flight frame depends on it.
    write_once(C.MAIN_SAMPLE_CSV, texts[C.MAIN_SAMPLE_CSV])
    if args.redraw_preflight:
        produced = [p.name for p in C.PREFLIGHT_DIR.glob("*") if p.name != C.PREFLIGHT_SAMPLE_CSV.name]
        if produced:
            sys.exit(f"REFUSING to redraw: pre-flight outputs already exist: {produced}")
        for p in (C.PREFLIGHT_SAMPLE_CSV, C.SAMPLING_MANIFEST):
            if p.exists() and p.read_text(encoding="utf-8") != texts[p]:
                p.unlink()
                print(f"redrawing: {p.relative_to(C.REPO)}")
    write_once(C.PREFLIGHT_SAMPLE_CSV, texts[C.PREFLIGHT_SAMPLE_CSV])
    write_once(C.SAMPLING_MANIFEST, texts[C.SAMPLING_MANIFEST])


if __name__ == "__main__":
    main()
