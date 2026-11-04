"""Score the pre-flight generations and run the frozen pre-flight checks.

THE PRE-FLIGHT IS A VALIDITY CHECK OF THE RECONSTRUCTED SETUP. IT IS NOT EVIDENCE OF
LABEL STABILITY, AND PASSING IT DOES NOT SHOW EQUIVALENCE WITH THE ORIGINAL SETUP.

Mission (Pre-flight validation):
  Operational checks, hard stop on any failure:
    1 served model identifiers exact; revision, dtype, engine recorded; generation config auto
      and the server log shows the V1 0.8.4 engine with the expected model sampling defaults
    2 thinking active: every output contains <think>
    3 no finish_reason=length below the 32,768-token total
    4 generation cache disabled, no cache reads reported
    5 the unchanged evaluator scores 100% of outputs without error; MBPP only in the sandbox
  Gross-mismatch warning: exact two-sided McNemar (binomial on discordant pairs), fresh vs
  stored results/bench score, alpha = 0.05 separately for each model, no correction.
  Descriptive: per-model accuracy with Wilson CIs and a paired (Newcombe) CI on the difference.

Scoring: EvaluatorFactory().get_evaluator(b).evaluate(item, raw_output), unchanged.
  BBH/MELD/MedQA/FinQA: item = {target|answer: stored ground_truth}
  KorBench: item = load_data("test")[index-1] from the verified LFS copy (host)
  MBPP:     same, but evaluated only inside wls_mbpp_sandbox.sh

Outputs (pre-flight directory only): preflight_scores.csv, preflight_check.csv.

Usage:
  python diagnostics/scripts/wls_preflight_score.py
  python diagnostics/scripts/wls_preflight_score.py --self-test-stored DIR
      Pipeline self-test with NO model call: the STORED original outputs of the pre-flight
      queries are fed through the same scoring path; results go to DIR (outside the
      winner_label_stability tree) and must reproduce the stored scores exactly.
"""

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

import wls_common as C

ALPHA = 0.05
DISCLAIMER = ("Pre-flight = reconstruction validity check only; not evidence of label "
              "stability; passing is not an equivalence certificate.")


# --------------------------------------------------------------------------- #
# Statistics                                                                  #
# --------------------------------------------------------------------------- #
def wilson(k, n):
    ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
    return ci.low, ci.high


def mcnemar_exact(stored, fresh):
    """Exact two-sided McNemar: binomial test on the discordant pairs."""
    b = sum(1 for s, f in zip(stored, fresh) if s == 1 and f == 0)
    c = sum(1 for s, f in zip(stored, fresh) if s == 0 and f == 1)
    p = 1.0 if b + c == 0 else binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue
    return b, c, min(1.0, p)


def newcombe_paired(stored, fresh):
    """Newcombe (1998) method 10: CI for p_fresh - p_stored from paired binary data."""
    n = len(stored)
    a = sum(1 for s, f in zip(stored, fresh) if f == 1 and s == 1)
    b = sum(1 for s, f in zip(stored, fresh) if f == 1 and s == 0)
    c = sum(1 for s, f in zip(stored, fresh) if f == 0 and s == 1)
    d = n - a - b - c
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n)
    l2, u2 = wilson(a + c, n)
    denom = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    if denom == 0:
        phi = 0.0
    elif a * d > b * c:
        phi = max(a * d - b * c - n / 2, 0) / denom
    else:
        phi = (a * d - b * c) / denom
    dlt = math.sqrt(max((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2, 0))
    eps = math.sqrt(max((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2, 0))
    return p1 - p2, p1 - p2 - dlt, p1 - p2 + eps


# --------------------------------------------------------------------------- #
# Scoring                                                                     #
# --------------------------------------------------------------------------- #
def evaluator_items(sample):
    """item passed to evaluate() for every sample row (identical for both models)."""
    kb, mb = C.point_evaluators_at_lfs()
    from evaluation import EvaluatorFactory
    fac = EvaluatorFactory()
    evs = {b: fac.get_evaluator(b) for b in C.BENCHMARKS}
    data = {b: evs[b].load_data("test") for b in ("korbench", "mbpp")}
    stored = {(b, m): C.load_stored(b, m) for b in C.BENCHMARKS for m in C.MODELS}
    items = {}
    for r in sample.itertuples(index=False):
        k = (r.dataset, int(r.index))
        prompt = stored[(r.dataset, C.DS)][r.index]["prompt"]
        if r.dataset in ("korbench", "mbpp"):
            item = data[r.dataset][r.index - 1]
            if item["prompt"] != prompt:
                raise RuntimeError(f"load_data prompt mismatch {k}")
        else:
            gts = {stored[(r.dataset, m)][r.index]["ground_truth"] for m in C.MODELS}
            if len(gts) != 1:
                raise RuntimeError(f"stored ground_truth differs between models at {k}")
            item = {C.GT_FIELD[r.dataset]: gts.pop()}
        items[k] = item
    return evs, items, stored


def score_mbpp_in_sandbox(rows):
    """rows: [(key, index, prompt, raw_output)] -> {key: result dict}."""
    with tempfile.TemporaryDirectory(prefix="wls_mbpp_in_") as d:
        with open(Path(d) / "inputs.jsonl", "w", encoding="utf-8") as f:
            for key, index, prompt, raw in rows:
                f.write(json.dumps({"key": key, "index": index, "prompt": prompt, "raw_output": raw}) + "\n")
        proc = subprocess.run(["bash", str(C.MBPP_SANDBOX_SH), d, "mbpp_harness.py"],
                              capture_output=True, text=True)
    out = {}
    lines = proc.stdout.splitlines()
    for line in lines:
        if line.startswith("RESULT\t"):
            r = json.loads(line.split("\t", 1)[1])
            out[r["key"]] = r
    ok = proc.returncode == 0 and lines[-1:] == ["DONE"] and "START_METHOD\tspawn" in lines
    if not ok:
        print(proc.stderr[-4000:], file=sys.stderr)
    return out, ok


def score_records(sample, gens):
    """gens: {(dataset, index, model): record with raw_output}. Returns a scores DataFrame."""
    evs, items, stored = evaluator_items(sample)
    rows, mbpp_rows = [], []
    for (dataset, index, model), g in sorted(gens.items()):
        row = {"dataset": dataset, "index": index, "model": model,
               "stored_score": float(stored[(dataset, model)][index]["score"]),
               "fresh_score": None, "prediction": None, "eval_error": None, "scored_in": None}
        if dataset == "mbpp":
            mbpp_rows.append((f"{model}|{index}", index, stored[(dataset, model)][index]["prompt"],
                              g["raw_output"]))
        else:
            try:
                out = evs[dataset].evaluate(items[(dataset, index)], g["raw_output"])
                row.update(fresh_score=float(out["is_correct"]), prediction=str(out["prediction"]),
                           scored_in="host")
            except Exception as e:
                row["eval_error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
    sandbox_ok = True
    if mbpp_rows:
        res, sandbox_ok = score_mbpp_in_sandbox(mbpp_rows)
        for row in rows:
            if row["dataset"] != "mbpp":
                continue
            r = res.get(f"{row['model']}|{row['index']}")
            if r is None or not sandbox_ok:
                row["eval_error"] = "no sandbox result"
            elif "error" in r:
                row["eval_error"] = r["error"]
            else:
                row.update(fresh_score=r["score"], prediction=r["prediction"], scored_in="bwrap_sandbox",
                           sandbox_result=r["result"])
    return pd.DataFrame(rows), sandbox_ok


# --------------------------------------------------------------------------- #
# Checks                                                                      #
# --------------------------------------------------------------------------- #
def add(checks, check, model, status, value, detail=""):
    checks.append({"check": check, "model": model, "status": status, "value": value, "detail": detail})


def load_generations(path, sample):
    """Exactly one ok record per (query, model) is required; error records are never scored."""
    want = {(r.dataset, int(r.index)) for r in sample.itertuples(index=False)}
    recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ok, errs, dup = {}, {}, []
    for r in recs:
        if r.get("phase") != "preflight":
            sys.exit(f"non-preflight record in {path}")
        k = (r["dataset"], r["index"], r["model"])
        if (r["dataset"], r["index"]) not in want:
            sys.exit(f"record for a query outside the pre-flight sample: {k}")
        if r["status"] == "ok":
            if k in ok:
                dup.append(k)
            ok[k] = r
        else:
            errs[k] = errs.get(k, 0) + 1
    return recs, ok, errs, dup


def run_checks(sample, recs, ok, errs, dup, scores, sandbox_ok, runs, self_test):
    checks = []
    n_q = len(sample)
    for m in C.MODELS:
        mine = {k: r for k, r in ok.items() if k[2] == m}
        n_err = sum(v for k, v in errs.items() if k[2] == m)
        add(checks, "0_completeness", m, "PASS" if len(mine) == n_q and not [d for d in dup if d[2] == m] else "FAIL",
            f"{len(mine)}/{n_q} ok", f"error attempts logged: {n_err}; duplicate ok: {len([d for d in dup if d[2] == m])}")

        # 1 model identity
        if self_test:
            add(checks, "1_model_identity", m, "N/A", "self-test uses stored outputs")
        else:
            bad = [k for k, r in mine.items() if r.get("response_model") != m or r.get("served_model_name") != m]
            mruns = [r for r in runs if r["model"] == m]
            man = [r.get("server_manifest") or {} for r in mruns]
            man_ok = bool(man) and all(x.get("revision") == C.HF_REVISION[m] and x.get("dtype") == "bfloat16"
                                       and x.get("engine") == "vllm" and x.get("vllm_version") == C.VLLM_VERSION
                                       and x.get("generation_config") == C.GENERATION_CONFIG
                                       for x in man)
            # Runtime evidence recorded by the client from the server's own startup log.
            logs = [(r.get("server") or {}).get("log") or {} for r in mruns]
            want = C.MODEL_SAMPLING_DEFAULTS[m]
            log_ok = bool(logs) and all(
                lg.get("engines") == [["V1", C.VLLM_VERSION]]
                and lg.get("default_sampling_lines") == ([["model", want]] if want else [])
                and lg.get("effective_sampling_expected") == C.EXPECTED_EFFECTIVE_SAMPLING[m]
                for lg in logs)
            add(checks, "1_model_identity", m, "PASS" if not bad and man_ok and log_ok else "FAIL",
                f"{len(mine) - len(bad)}/{len(mine)} exact name",
                f"revision={[x.get('revision') for x in man]} dtype={[x.get('dtype') for x in man]} "
                f"engine=vllm {[x.get('vllm_version') for x in man]} "
                f"generation_config={[x.get('generation_config') for x in man]} "
                f"log engines={[lg.get('engines') for lg in logs]} "
                f"log sampling defaults={[lg.get('default_sampling_lines') for lg in logs]} "
                f"gpus={[x.get('gpus') for x in man]}")

        # 2 thinking active
        no_think = [k for k, r in mine.items() if "<think>" not in (r["raw_output"] or "")]
        rf = [k for k, r in mine.items() if r.get("reasoning_field_present")]
        add(checks, "2_thinking_active", m, "PASS" if not no_think and not rf else "FAIL",
            f"{len(mine) - len(no_think)}/{len(mine)} contain <think>",
            f"reasoning-parser fields present: {len(rf)}")

        # 3 length policy
        if self_test:
            add(checks, "3_length_policy", m, "N/A", "self-test uses stored outputs")
        else:
            short = [k for k, r in mine.items() if r.get("finish_reason") == "length"
                     and r["usage"]["prompt_tokens"] + r["usage"]["completion_tokens"] < C.MAX_MODEL_LEN]
            n_len = sum(r.get("finish_reason") == "length" for r in mine.values())
            add(checks, "3_length_policy", m, "PASS" if not short else "FAIL",
                f"{len(short)} truncated below {C.MAX_MODEL_LEN}", f"finish_reason=length at cap: {n_len - len(short)}")

        # 4 cache disabled
        if self_test:
            add(checks, "4_cache_disabled", m, "N/A", "self-test uses stored outputs")
        else:
            mruns = [r for r in runs if r["model"] == m]
            problems = []
            for r in mruns:
                if not str(r.get("repo_response_cache", "")).startswith("not used"):
                    problems.append("repo cache")
                info = (r.get("server") or {}).get("cache_config_info") or {}
                if info.get("enable_prefix_caching") != "False":
                    problems.append("prefix caching")
                before = (r.get("server") or {}).get("prefix_cache_counters_before") or {}
                after = r.get("prefix_cache_counters_after") or {}
                h0, h1 = before.get(C.PREFIX_CACHE_HITS_METRIC), after.get(C.PREFIX_CACHE_HITS_METRIC)
                if h0 is None or h1 is None:
                    problems.append(f"{C.PREFIX_CACHE_HITS_METRIC} not recorded before/after")
                elif h1 - h0 > 0:
                    problems.append(f"prefix cache hits {h1 - h0}")
            cached = [k for k, r in mine.items()
                      if ((r.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")]
            if cached:
                problems.append(f"{len(cached)} responses report cached tokens")
            add(checks, "4_cache_disabled", m, "PASS" if mruns and not problems else "FAIL",
                f"{len(mruns)} run(s)", "; ".join(problems))

        # 5 evaluator
        sm = scores[scores["model"] == m]
        nerr = int(sm["eval_error"].notna().sum())
        mbpp = sm[sm["dataset"] == "mbpp"]
        in_box = bool((mbpp["scored_in"] == "bwrap_sandbox").all()) and sandbox_ok
        add(checks, "5_evaluator_scored_all", m,
            "PASS" if nerr == 0 and len(sm) == n_q and in_box else "FAIL",
            f"{len(sm) - nerr}/{n_q} scored", f"MBPP in sandbox: {in_box} ({len(mbpp)} items)")

    # McNemar warning test and descriptive accuracy, per model
    for m in C.MODELS:
        sm = scores[(scores["model"] == m) & scores["fresh_score"].notna()]
        s, f = sm["stored_score"].astype(int).tolist(), sm["fresh_score"].astype(int).tolist()
        if len(sm) != n_q:
            add(checks, "mcnemar_exact_two_sided", m, "NOT_RUN", "", "incomplete scoring")
            continue
        b, c, p = mcnemar_exact(s, f)
        add(checks, "mcnemar_exact_two_sided", m, "WARN_STOP" if p < ALPHA else "NO_GROSS_MISMATCH_DETECTED",
            f"p={p:.4g}", f"stored1_fresh0={b} stored0_fresh1={c} alpha={ALPHA} (operational warning only)")
        ls, us = wilson(sum(s), len(s))
        lf, uf = wilson(sum(f), len(f))
        dlt, lo, hi = newcombe_paired(s, f)
        add(checks, "accuracy_descriptive", m, "INFO",
            f"stored={sum(s) / len(s):.3f} fresh={sum(f) / len(f):.3f} diff={dlt:+.3f}",
            f"stored Wilson95=[{ls:.3f},{us:.3f}] fresh Wilson95=[{lf:.3f},{uf:.3f}] "
            f"paired Newcombe95 diff=[{lo:+.3f},{hi:+.3f}]")
        for bmk in C.BENCHMARKS:
            x = sm[sm["dataset"] == bmk]
            add(checks, "accuracy_by_benchmark_descriptive", m, "INFO",
                f"{bmk}: stored={x['stored_score'].mean():.2f} fresh={x['fresh_score'].mean():.2f}", f"n={len(x)}")
    add(checks, "disclaimer", "", "INFO", DISCLAIMER)
    return pd.DataFrame(checks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=Path, default=C.PREFLIGHT_GENERATIONS)
    ap.add_argument("--self-test-stored", type=Path, default=None, metavar="DIR")
    args = ap.parse_args()
    sample = C.load_sample(C.PREFLIGHT_SAMPLE_CSV)
    if not all(C.verify_lfs_copy().values()):
        sys.exit("verified LFS copy missing; run diagnostics/scripts/wls_fetch_lfs.py")
    C.assert_sample_eligible(sample)

    runs = []
    if args.self_test_stored:
        out_dir = args.self_test_stored.resolve()
        if C.WLS_DIR.resolve() in out_dir.parents or out_dir == C.WLS_DIR.resolve():
            sys.exit("self-test output must stay outside diagnostics/winner_label_stability/")
        recs, errs, dup = [], {}, []
        ok = {}
        for r in sample.itertuples(index=False):
            for m in C.MODELS:
                st = C.load_stored(r.dataset, m)[r.index]
                ok[(r.dataset, int(r.index), m)] = {"raw_output": st["raw_output"], "source": "stored_original"}
    else:
        out_dir = C.PREFLIGHT_DIR
        recs, ok, errs, dup = load_generations(args.generations, sample)
        if any(r.get("source") for r in ok.values()):
            sys.exit("generation file contains non-fresh records")
        runs_path = C.PREFLIGHT_DIR / "preflight_runs.jsonl"
        runs = [json.loads(x) for x in runs_path.read_text().splitlines()] if runs_path.exists() else []

    scores, sandbox_ok = score_records(sample, ok)
    checks = run_checks(sample, recs, ok, errs, dup, scores, sandbox_ok, runs, bool(args.self_test_stored))
    out_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out_dir / C.PREFLIGHT_SCORES_CSV.name, index=False)
    checks.to_csv(out_dir / C.PREFLIGHT_CHECK_CSV.name, index=False)

    with pd.option_context("display.max_colwidth", 120, "display.width", 250):
        print(checks.to_string(index=False))
    hard_fail = checks["status"].eq("FAIL").any() or checks["status"].eq("NOT_RUN").any()
    warn = checks["status"].eq("WARN_STOP").any()
    if args.self_test_stored:
        agree = (scores["fresh_score"] == scores["stored_score"]).sum()
        print(f"\nSELF-TEST (stored outputs, no model call): rescored == stored {agree}/{len(scores)}")
        sys.exit(0 if agree == len(scores) and not hard_fail else 1)
    print("\n" + ("STOP: operational check failed" if hard_fail else
                  "STOP FOR REVIEW: McNemar warning triggered" if warn else
                  "Operational checks passed; no gross mismatch detected.") + "\n" + DISCLAIMER)
    sys.exit(1 if hard_fail or warn else 0)


if __name__ == "__main__":
    main()
