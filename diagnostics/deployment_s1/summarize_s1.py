"""S1 summary and pre-registered early stop (E0 §8; WEEK-EXECUTION-PLAN gate G1).

For every benchmark and mode:
    gap(M) = acc(Qwen3-32B bf16, mode) - max over edge-eligible m at class M of acc(m, mode)
in percentage points, for M in {12, 16, 24} GB. Accuracy is the mean score over the frozen
queries and the two seeds. Comparisons stay within a mode (thinking vs thinking,
non-thinking vs non-thinking), as E0 §8 states.

Early stop (decision on point estimates, threshold strictly < 3.0 pt):
  rule16: gap(16) < 3.0 on every benchmark in both modes -> remove the 16/24 GB split regimes
          (only the 12 GB class and the 6 core E0-L cells remain);
  rule12: rule16 and gap(12) < 3.0 on every benchmark in both modes -> SPLIT has no quality
          headroom: R fails by the S1 bound; stop before split-runtime work.
Paired-bootstrap 95% CIs (10,000 resamples over queries) are reported as description only.

The memory-class eligibility defaults to the E0 §1 desk check; S6's measured fit results may
replace it with --fit-json '{"12": [...], "16": [...], "24": [...]}'.

Usage:
  python diagnostics/deployment_s1/summarize_s1.py [--runs-dir DIR] [--fit-json FILE] [--allow-incomplete]
  python diagnostics/deployment_s1/summarize_s1.py --selftest
"""

import argparse
import datetime
import json
import random
import sys
from pathlib import Path

import run_s1 as R

THRESHOLD_PT = 3.0
CLASSES = [12, 16, 24]
# E0 §1 desk check (checkpoint + bf16 KV for 16k tokens + 1.5 GB): 8B-AWQ 10.0 GB, 14B-AWQ 14.2,
# 8B bf16 20.3, 32B-AWQ 25.1 with bf16 KV / 22.9 with FP8 KV (E0 counts it at 24 GB with FP8 KV).
DEFAULT_ELIGIBLE = {
    12: ["Qwen3-8B-AWQ"],
    16: ["Qwen3-8B-AWQ", "Qwen3-14B-AWQ"],
    24: ["Qwen3-8B-AWQ", "Qwen3-14B-AWQ", "Qwen3-8B", "Qwen3-32B-AWQ"],
}
EDGE_MODELS = ["Qwen3-8B-AWQ", "Qwen3-14B-AWQ", "Qwen3-32B-AWQ", "Qwen3-8B"]
MODES = ["thinking", "nonthinking"]
BOOT = 10000


def per_query(scores, sample, model, mode, seeds=R.SEEDS):
    """{query key: mean score over seeds}, only for queries with every seed scored."""
    out = {}
    for q in sample:
        vals = [scores.get(R.job_key(model, mode, s, q["dataset"], q["index"])) for s in seeds]
        if all(v is not None and v.get("score") is not None for v in vals):
            out[q["key"]] = sum(v["score"] for v in vals) / len(vals)
    return out


def boot_ci(diffs, rng, n=BOOT):
    if not diffs:
        return None
    k = len(diffs)
    stats = sorted(sum(diffs[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return [100 * stats[int(0.025 * n)], 100 * stats[int(0.975 * n) - 1]]


def summarize(scores, sample, eligible=DEFAULT_ELIGIBLE, allow_incomplete=False, n_boot=BOOT):
    benches = [d for d, _, _ in R.BENCHMARKS]
    qs_by_bench = {d: [q for q in sample if q["dataset"] == d] for d in benches}
    pq = {(m, mode): per_query(scores, sample, m, mode) for m in [R.REFERENCE_MODEL] + EDGE_MODELS for mode in MODES}
    expected = len(sample) * len(R.SEEDS)
    coverage = {f"{m}|{mode}": sum(1 for q in sample for s in R.SEEDS
                                   if (scores.get(R.job_key(m, mode, s, q["dataset"], q["index"])) or {}).get("score")
                                   is not None)
                for (m, mode) in pq}
    complete = all(v == expected for v in coverage.values())
    rng = random.Random(R.SEED)
    acc, gaps = {}, {}
    for d in benches:
        keys = [q["key"] for q in qs_by_bench[d]]
        for mode in MODES:
            # Paired over the queries every model in this mode has; = all queries when complete.
            common = [k for k in keys if all(k in pq[(m, mode)] for m in [R.REFERENCE_MODEL] + EDGE_MODELS)]
            for m in [R.REFERENCE_MODEL] + EDGE_MODELS:
                vals = [pq[(m, mode)][k] for k in common]
                acc[(m, mode, d)] = 100 * sum(vals) / len(vals) if vals else None
            ref = acc[(R.REFERENCE_MODEL, mode, d)]
            for M in CLASSES:
                cands = [(acc[(m, mode, d)], -i, m) for i, m in enumerate(eligible[M]) if acc[(m, mode, d)] is not None]
                if ref is None or not cands:
                    gaps[(d, mode, M)] = None
                    continue
                best_acc, _, best = max(cands)   # ties -> earlier (smaller) model in the eligibility list
                diffs = [pq[(R.REFERENCE_MODEL, mode)][k] - pq[(best, mode)][k] for k in common]
                gaps[(d, mode, M)] = {"best": best, "ref_acc": ref, "best_acc": best_acc, "gap_pt": ref - best_acc,
                                      "ci95_pt": boot_ci(diffs, rng, n_boot), "n": len(common),
                                      "per_model_gap_pt": {m: ref - acc[(m, mode, d)] for m in eligible[M]
                                                           if acc[(m, mode, d)] is not None}}

    def all_below(M):
        vals = [gaps[(d, mode, M)] for d in benches for mode in MODES]
        return None if any(v is None for v in vals) else all(v["gap_pt"] < THRESHOLD_PT for v in vals)

    below = {M: all_below(M) for M in CLASSES}
    rule16 = below[16]
    rule12 = None if rule16 is None or below[12] is None else (rule16 and below[12])
    if not complete and not allow_incomplete:
        decision = "INCOMPLETE"
    elif rule16 is None or rule12 is None:
        decision = "UNDETERMINED"
    elif rule12:
        decision = "STOP_NO_HEADROOM"
    elif rule16:
        decision = "DROP_16_24"
    else:
        decision = "CONTINUE"
    if not complete and decision != "INCOMPLETE":
        decision = "PROVISIONAL_" + decision
    return {"complete": complete, "coverage": coverage, "expected_per_model_mode": expected,
            "acc": acc, "gaps": gaps, "all_below": below, "rule16": rule16, "rule12": rule12,
            "decision": decision, "eligible": eligible}


DECISION_TEXT = {
    "CONTINUE": ("32B-bf16 is >= 3.0 pt better than EDGE-best(16) on at least one benchmark in at least one mode: "
                 "the 16/24 GB split regimes stay. Proceed with the full E0 (G1 passes)."),
    "DROP_16_24": ("32B-bf16 is < 3.0 pt better than EDGE-best(16) on every benchmark in both modes: remove the 16/24 GB "
                   "split regimes. Run only the 12 GB class and the 6 core E0-L cells."),
    "STOP_NO_HEADROOM": ("32B-bf16 is < 3.0 pt better than EDGE-best(16) and than EDGE-best(12) on every benchmark in both "
                         "modes: SPLIT has no quality headroom. R fails by the S1 bound; stop before split-runtime work "
                         "and write e0/DECISION.md (R: FAIL by S1 bound), branch B."),
    "INCOMPLETE": "Not every (model, mode, seed, query) is scored; no decision. Finish generation/scoring first.",
    "UNDETERMINED": "A required accuracy is missing; no decision.",
}


def fmt(v, nd=1):
    return "-" if v is None else f"{v:+.{nd}f}"


def write_report(res, scores, sample, out_dir, fit_source):
    out_dir.mkdir(parents=True, exist_ok=True)
    benches = [d for d, _, _ in R.BENCHMARKS]
    dec = res["decision"]
    base = dec.replace("PROVISIONAL_", "")
    L = [f"# S1 quality gate: {dec}", "",
         f"Generated {datetime.datetime.now(datetime.timezone.utc).isoformat()} from {len(scores)} scored generations. "
         f"Eligibility: {fit_source}. Threshold: gap < {THRESHOLD_PT} pt (point estimate).", "",
         f"**Decision.** {DECISION_TEXT.get(base, base)}", ""]
    if dec.startswith("PROVISIONAL_"):
        L += ["**PROVISIONAL:** computed on incomplete data (`--allow-incomplete`); not a G1 decision.", ""]
    L += [f"rule16 (all gaps at 16 GB < {THRESHOLD_PT}): {res['rule16']}; rule12 (rule16 and all gaps at 12 GB < "
          f"{THRESHOLD_PT}): {res['rule12']}. Informational, not a pre-registered rule: all gaps at 24 GB < "
          f"{THRESHOLD_PT}: {res['all_below'][24]}.", "",
          "## Gap: Qwen3-32B bf16 minus EDGE-best(M), percentage points (within mode)", "",
          "| benchmark | mode | 32B acc | 12 GB: best, gap [95% CI] | 16 GB: best, gap [95% CI] | 24 GB: best, gap [95% CI] |",
          "|---|---|---|---|---|---|"]
    for d in benches:
        for mode in MODES:
            cells = []
            for M in CLASSES:
                g = res["gaps"][(d, mode, M)]
                cells.append("-" if g is None else
                             f"{g['best']}, **{fmt(g['gap_pt'])}** [{fmt(g['ci95_pt'][0])}, {fmt(g['ci95_pt'][1])}]")
            ref = res["acc"][(R.REFERENCE_MODEL, mode, d)]
            L.append(f"| {d} | {mode} | {'-' if ref is None else f'{ref:.1f}'} | " + " | ".join(cells) + " |")
    L += ["", "## Accuracy (%), mean over queries and seeds", "",
          "| benchmark | mode | " + " | ".join([R.REFERENCE_MODEL] + EDGE_MODELS) + " |",
          "|---|---|" + "---|" * (1 + len(EDGE_MODELS))]
    for d in benches:
        for mode in MODES:
            L.append(f"| {d} | {mode} | " + " | ".join(
                "-" if res["acc"][(m, mode, d)] is None else f"{res['acc'][(m, mode, d)]:.1f}"
                for m in [R.REFERENCE_MODEL] + EDGE_MODELS) + " |")
    # Truncation and length (E0 §4.3 asks for the truncation rate)
    L += ["", f"## Truncation rate at max_new_tokens={R.MAX_NEW_TOKENS} and mean completion tokens", "",
          "| model | mode | " + " | ".join(benches) + " |", "|---|---|" + "---|" * len(benches)]
    for m in [R.REFERENCE_MODEL] + EDGE_MODELS:
        for mode in MODES:
            cells = []
            for d in benches:
                rs = [s for s in scores.values() if s["model"] == m and s["mode"] == mode and s["dataset"] == d]
                cells.append("-" if not rs else f"{sum(bool(s['truncated']) for s in rs) / len(rs):.2f} / "
                             f"{sum(s['completion_tokens'] for s in rs) / len(rs):.0f}")
            L.append(f"| {m} | {mode} | " + " | ".join(cells) + " |")
    # Chat-template sanity: thinking outputs open a think block, non-thinking ones never do.
    bad_t = sum(1 for s in scores.values() if s["mode"] == "thinking" and not s["think_open"])
    bad_n = sum(1 for s in scores.values() if s["mode"] == "nonthinking" and s["think_open"])
    L += ["", f"Template sanity: thinking outputs without `<think>`: {bad_t}; non-thinking outputs with `<think>`: {bad_n}.",
          "", "## Coverage", "", f"Expected scored generations per (model, mode): {res['expected_per_model_mode']}.", ""]
    L += [f"- {k}: {v}" for k, v in res["coverage"].items()]
    L += ["", "## Eligibility per memory class", ""] + [f"- {M} GB: {', '.join(res['eligible'][M])}" for M in CLASSES]
    L += ["", "Notes: EDGE-best is chosen per benchmark and mode on the same queries it is scored on (a small "
          "selection effect that favours the edge, i.e. the early stop). The 24 GB class counts Qwen3-32B-AWQ under the "
          "E0 FP8-KV assumption; S1 serves it with a bf16/fp16 KV cache. The CI is a paired bootstrap over queries "
          "against the selected EDGE-best, descriptive only."]
    (out_dir / "S1_SUMMARY.md").write_text("\n".join(L) + "\n")
    js = {"decision": dec, "complete": res["complete"], "rule16": res["rule16"], "rule12": res["rule12"],
          "all_below": {str(k): v for k, v in res["all_below"].items()}, "threshold_pt": THRESHOLD_PT,
          "eligible": {str(k): v for k, v in res["eligible"].items()}, "eligibility_source": fit_source,
          "coverage": res["coverage"],
          "acc": [{"model": m, "mode": mode, "dataset": d, "acc_pct": v} for (m, mode, d), v in res["acc"].items()],
          "gaps": [{"dataset": d, "mode": mode, "class_gb": M, **(g or {})} for (d, mode, M), g in res["gaps"].items()]}
    (out_dir / "s1_summary.json").write_text(json.dumps(js, indent=2) + "\n")
    # E0 §10 artifact: per query, model, mode, seed, score and tokens.
    with open(out_dir / "q3_scores.jsonl", "w", encoding="utf-8") as f:
        for k in sorted(scores):
            s = scores[k]
            f.write(json.dumps({x: s.get(x) for x in ("dataset", "index", "model", "mode", "seed", "score",
                                                      "completion_tokens", "prompt_tokens", "finish_reason",
                                                      "truncated")}) + "\n")
    print("\n".join(L[:8]))
    print(f"\nwrote {out_dir / 'S1_SUMMARY.md'}, s1_summary.json, q3_scores.jsonl")


def load_eligibility(path):
    if not path:
        return DEFAULT_ELIGIBLE, "E0 §1 desk check (default)"
    raw = json.loads(Path(path).read_text())
    el = {int(k): list(v) for k, v in raw.items()}
    if sorted(el) != CLASSES or any(set(v) - set(EDGE_MODELS) for v in el.values()):
        sys.exit(f"--fit-json must map exactly {CLASSES} to subsets of {EDGE_MODELS}")
    return el, f"S6 fit results ({path})"


def selftest():
    """Synthetic scores that must produce each decision."""
    sample = [{"key": f"{d}:{i}", "dataset": d, "index": i} for d, _, n in R.BENCHMARKS for i in range(1, n + 1)]

    def make(acc):   # acc[model] -> fraction correct, identical across benchmarks/modes/seeds
        sc = {}
        for m in [R.REFERENCE_MODEL] + EDGE_MODELS:
            for mode in MODES:
                for s in R.SEEDS:
                    for d, _, n in R.BENCHMARKS:
                        k_ok = round(acc[m] * n)
                        for i in range(1, n + 1):
                            sc[R.job_key(m, mode, s, d, i)] = {
                                "model": m, "mode": mode, "seed": s, "dataset": d, "index": i,
                                "score": 1.0 if i <= k_ok else 0.0, "truncated": False, "completion_tokens": 10,
                                "think_open": mode == "thinking"}
        return sc

    base = {"Qwen3-8B-AWQ": 0.50, "Qwen3-14B-AWQ": 0.55, "Qwen3-32B-AWQ": 0.60, "Qwen3-8B": 0.55}
    cases = [({**base, R.REFERENCE_MODEL: 0.70}, "CONTINUE"),         # +15 over 14B-AWQ at 16 GB
             ({**base, R.REFERENCE_MODEL: 0.575}, "DROP_16_24"),      # +2.5 over 14B, +7.5 over 8B-AWQ
             ({**base, R.REFERENCE_MODEL: 0.52}, "STOP_NO_HEADROOM")]   # +1.7 / +2.5 over 8B-AWQ
    for acc, want in cases:
        res = summarize(make(acc), sample, n_boot=200)
        assert res["decision"] == want, (want, res["decision"])
    sc = make({**base, R.REFERENCE_MODEL: 0.70})
    sc.pop(next(iter(sc)))
    assert summarize(sc, sample, n_boot=200)["decision"] == "INCOMPLETE"
    assert summarize(sc, sample, allow_incomplete=True, n_boot=200)["decision"].startswith("PROVISIONAL_")
    # One benchmark/mode at >= 3.0 keeps the 16/24 GB regimes (rule needs every task in both modes).
    sc = make({**base, R.REFERENCE_MODEL: 0.575})
    for k, v in sc.items():
        if v["model"] == R.REFERENCE_MODEL and v["mode"] == "nonthinking" and v["dataset"] == "gpqa":
            v["score"] = 1.0
    res = summarize(sc, sample, n_boot=200)
    assert res["decision"] == "CONTINUE", res["decision"]
    g = res["gaps"][("gpqa", "nonthinking", 16)]
    assert g["best"] == "Qwen3-14B-AWQ" and abs(g["gap_pt"] - 45.0) < 1e-9 and g["ci95_pt"][0] > 0
    print("summarize_s1 selftest: ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(R.RUNS_DIR))
    ap.add_argument("--fit-json", default=None)
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    sample = R.load_sample()
    run_dir = Path(args.runs_dir) / "s1"
    scores = R.latest_scores(run_dir)
    eligible, src = load_eligibility(args.fit_json)
    res = summarize(scores, sample, eligible, args.allow_incomplete)
    write_report(res, scores, sample, Path(args.out_dir) if args.out_dir else run_dir / "summary", src)
    sys.exit(0 if not res["decision"].startswith(("INCOMPLETE", "UNDETERMINED")) else 2)


if __name__ == "__main__":
    main()
