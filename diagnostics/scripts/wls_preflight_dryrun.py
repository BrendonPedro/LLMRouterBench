"""Dry-run validation of the winner-label stability pre-flight pipeline. Calls NO model.

Checks: samples regenerate exactly and are disjoint; row counts; prompts reconstruct
exactly; model names exact; generation request matches the mission; no cache use; every
sample row resolves to its evaluator; MBPP sandbox available and isolated; no path depends
on a Claude scratchpad.

Usage: python diagnostics/scripts/wls_preflight_dryrun.py [--with-selftest]
  --with-selftest  also run wls_preflight_score.py --self-test-stored (scores the STORED
                   original outputs through the full scoring path, MBPP in the sandbox;
                   a few minutes; still no model call)
"""

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

import wls_common as C

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""), flush=True)


def py(*args, **kw):
    return subprocess.run([sys.executable, *args], cwd=C.REPO / "diagnostics" / "scripts",
                          capture_output=True, text=True, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-selftest", action="store_true")
    args = ap.parse_args()
    scripts = C.REPO / "diagnostics" / "scripts"
    new_files = sorted(f for f in [*scripts.glob("wls_*"), *(scripts / "wls_sandbox").glob("*.py")] if f.is_file())

    # ---- samples --------------------------------------------------------------------------
    r = py("wls_freeze_samples.py", "--check")
    check("samples regenerate byte-identically from the recorded seeds", r.returncode == 0, r.stdout.strip().replace("\n", "; "))
    main_s, pre = C.load_sample(C.MAIN_SAMPLE_CSV), C.load_sample(C.PREFLIGHT_SAMPLE_CSV)
    check("main sample byte-identical to the originally frozen file",
          C.sha256_file(C.MAIN_SAMPLE_CSV) == C.MAIN_SAMPLE_SHA256, C.MAIN_SAMPLE_SHA256[:16])
    mk = set(zip(main_s["dataset"], main_s["index"]))
    pk = set(zip(pre["dataset"], pre["index"]))
    check("main sample: 120 rows, unique keys", len(main_s) == 120 and len(mk) == 120)
    cnt = main_s.groupby(["dataset", "target"]).size()
    check("main sample: 10 DeepSeek-win + 10 Qwen-win per benchmark",
          set(main_s["dataset"]) == set(C.BENCHMARKS) and (cnt == 10).all() and len(cnt) == 12)
    check("main sample: original non-ties only, stored scores agree",
          main_s["target"].isin([0, 1]).all()
          and ((main_s["target"] == 0) == ((main_s["stored_score_ds"] == 1) & (main_s["stored_score_qwen"] == 0))).all()
          and ((main_s["target"] == 1) == ((main_s["stored_score_ds"] == 0) & (main_s["stored_score_qwen"] == 1))).all())
    check("pre-flight sample: 120 rows, 20 per benchmark, unique keys",
          len(pre) == 120 and len(pk) == 120 and (pre.groupby("dataset").size() == 20).all()
          and set(pre["dataset"]) == set(C.BENCHMARKS))
    check("pre-flight sample includes ties (frame = all rows)", pre["target"].eq(-1).any(),
          f"pool ties in sample: {int(pre['target'].eq(-1).sum())}")
    check("main and pre-flight samples are disjoint (dataset,index)", not (mk & pk), f"overlap={len(mk & pk)}")
    check("main and pre-flight samples are disjoint (prompt hash)",
          not (set(main_s["prompt_sha256"]) & set(pre["prompt_sha256"])))

    # ---- prompts reconstruct --------------------------------------------------------------
    lfs = C.verify_lfs_copy()
    check(f"LFS copy: {len(C.LFS_FILES)}/{len(C.LFS_FILES)} objects match committed pointer sha256+size",
          all(lfs.values()), f"{sum(lfs.values())}/{len(lfs)} at {C.lfs_dir()}")
    empty_finqa = C.finqa_empty_answer_indices()
    hit = sorted({(d, int(i)) for df in (main_s, pre) for d, i in zip(df["dataset"], df["index"])
                  if d == "finqa" and int(i) in empty_finqa})
    check("no FinQA row with an empty evaluator `answer` in either sample", not hit,
          f"{len(empty_finqa)} ineligible FinQA rows {sorted(empty_finqa)}; in samples: {hit}")
    try:
        C.assert_sample_eligible(pre.assign(index=[365] + list(pre["index"][1:]),
                                            dataset=["finqa"] + list(pre["dataset"][1:])), empty_finqa)
        guard = False
    except RuntimeError:
        guard = True
    check("eligibility guard fails loudly on an injected ineligible row", guard)
    pool = C.load_pool().set_index(["dataset", "index"])
    kb, mb = C.point_evaluators_at_lfs()
    from evaluation import EvaluatorFactory
    fac = EvaluatorFactory()
    data = {"korbench": fac.get_evaluator("korbench").load_data("test"),
            "mbpp": fac.get_evaluator("mbpp").load_data("test")}
    templates = {b: getattr(sys.modules[type(fac.get_evaluator(b)).__module__], "PROMPT")
                 for b in ("bbh", "meld", "medqa", "finqa")}
    bad = []
    stored = {(b, m): C.load_stored(b, m) for b in C.BENCHMARKS for m in C.MODELS}
    for df in (main_s, pre):
        for r in df.itertuples(index=False):
            p_ds, p_qw = stored[(r.dataset, C.DS)][r.index]["prompt"], stored[(r.dataset, C.QWEN)][r.index]["prompt"]
            ok = (p_ds == p_qw == pool.loc[(r.dataset, r.index), "query"]
                  and C.sha256_text(p_ds) == r.prompt_sha256 and C.sha1_16(p_ds) == r.query_sha1)
            if r.dataset in data:
                ok = ok and data[r.dataset][r.index - 1]["prompt"] == p_ds
            else:
                ok = ok and p_ds.startswith(templates[r.dataset].split("{", 1)[0])
            if not ok:
                bad.append((r.dataset, r.index))
    check("prompts reconstruct exactly for all 240 sample rows", not bad,
          "stored DS==QW==pool query; sha256/sha1 match; KorBench/MBPP == load_data('test')[index-1]; "
          f"others start with evaluator PROMPT; mismatches={bad[:5]}")

    # ---- evaluators -----------------------------------------------------------------------
    res = {b: type(fac.get_evaluator(b)).__name__ for b in C.BENCHMARKS}
    check("evaluator resolves for every benchmark", res == C.EXPECTED_EVALUATOR, json.dumps(res))
    missing, empty_gt = [], []
    for df in (main_s, pre):
        for r in df.itertuples(index=False):
            if r.dataset == "korbench":
                it = data["korbench"][r.index - 1]
                ok = all(k in it for k in ("question_type", "rule_id", "idx"))
            elif r.dataset == "mbpp":
                it = data["mbpp"][r.index - 1]
                ok = all(k in it for k in ("test_list", "task_id"))
            else:
                # evaluate() reads item[target|answer] only to derive its own ground_truth, and the
                # stored ground_truth IS that derived value; re-passing it is exact unless it
                # still contains \boxed (it would be re-extracted).
                gts = {stored[(r.dataset, m)][r.index].get("ground_truth") for m in C.MODELS}
                ok = len(gts) == 1 and all(isinstance(g, str) and "\\boxed" not in g for g in gts)
                if gts == {""}:
                    empty_gt.append((r.dataset, r.index))
                    ok = False
            if not ok:
                missing.append((r.dataset, r.index))
    check("every sample row has the fields its evaluator needs", not missing,
          f"missing={missing[:5]}; rows with empty stored ground_truth: {empty_gt}")

    # ---- model names and generation settings ----------------------------------------------
    cfg = yaml.safe_load((C.REPO / "config" / "data_collector_small_model_config.yaml").read_text())
    orig = {m["api_model_name"]: m for m in cfg["models"] if m["api_model_name"] in C.MODELS}
    check("model names exact (collector config, stored dirs, constants)",
          set(orig) == set(C.MODELS) and all((C.REPO / "results/bench/bbh/test" / m).is_dir() for m in C.MODELS))
    check("request sampling equals recorded collector settings",
          all(orig[m]["temperature"] == C.TEMPERATURE and orig[m]["top_p"] == C.TOP_P
              and "reasoning_effort" not in orig[m] and "extra_body" not in orig[m] for m in C.MODELS),
          f"temperature={C.TEMPERATURE} top_p={C.TOP_P}")
    for m in C.MODELS:
        body = C.build_request(m, "PROMPT")
        check(f"request body fields exact for {m}",
              tuple(body) == C.REQUEST_KEYS and body["model"] == m
              and body["messages"] == [{"role": "user", "content": "PROMPT"}],
              json.dumps({k: v for k, v in body.items() if k != "messages"}) + " + 1 user message, no system")
    r = py("wls_preflight_generate.py", "--model", C.QWEN, "--dry-run")
    lines = [json.loads(x) for x in r.stdout.splitlines()]
    check("client --dry-run builds 120 requests and sends nothing",
          r.returncode == 0 and len(lines) == 120
          and all(set(x["request"]) == set(C.REQUEST_KEYS) for x in lines)
          and all(len(x["request"]["messages"]) == 1 and x["request"]["messages"][0]["role"] == "user" for x in lines),
          r.stderr.strip())

    # ---- server launch settings -----------------------------------------------------------
    for key, m in (("deepseek", C.DS), ("qwen", C.QWEN)):
        r = subprocess.run(["bash", str(C.SERVE_VLLM_SH), key, "--print-cmd"], capture_output=True, text=True)
        argv = r.stdout.split("\n")[:-1]
        def val(f):
            return argv[argv.index(f) + 1] if f in argv else None
        ok = (argv[:3] == ["vllm", "serve", C.HF_REPO[m]] and val("--revision") == C.HF_REVISION[m]
              and val("--tokenizer-revision") == C.HF_REVISION[m] and val("--served-model-name") == m
              and val("--dtype") == "bfloat16" and val("--max-model-len") == str(C.MAX_MODEL_LEN)
              and val("--generation-config") == "vllm" and "--no-enable-prefix-caching" in argv
              and not any(a.startswith(("--reasoning-parser", "--enable-prefix-caching", "--seed",
                                        "--override-generation-config", "--quantization", "--enable-reasoning"))
                          for a in argv))
        check(f"vLLM launch settings exact for {m}", ok and r.returncode == 0, " ".join(argv))
    pin = re.search(r'^VLLM_VERSION="([^"]+)"', C.SERVE_VLLM_SH.read_text(), re.M)
    check("vLLM version pinned identically in launcher and client", pin and pin.group(1) == C.VLLM_VERSION,
          C.VLLM_VERSION)

    # ---- cache ----------------------------------------------------------------------------
    src = (scripts / "wls_preflight_generate.py").read_text()
    imports = {n.module if isinstance(n, ast.ImportFrom) else a.name
               for n in ast.walk(ast.parse(src)) if isinstance(n, (ast.Import, ast.ImportFrom))
               for a in (n.names if isinstance(n, ast.Import) else [n])}
    check("client does not import the repo generator or cache",
          not any(str(i).startswith(("generators", "common", "openai", "data_collector")) for i in imports),
          f"imports={sorted(map(str, imports))}")

    # ---- MBPP sandbox ---------------------------------------------------------------------
    v = subprocess.run(["bwrap", "--version"], capture_output=True, text=True).stdout.strip()
    check("bubblewrap 0.9.0 and systemd-run available",
          v == "bubblewrap 0.9.0" and subprocess.run(["systemd-run", "--version"], capture_output=True).returncode == 0, v)
    with tempfile.TemporaryDirectory(prefix="wls_probe_in_") as d:
        r = subprocess.run(["bash", str(C.MBPP_SANDBOX_SH), d, "probe.py",
                            f"{Path.home()}:{C.REPO}:/tmp/claude-{os.getuid()}"],
                           capture_output=True, text=True)
    try:
        p = json.loads(r.stdout)
        oid = C.lfs_pointer("data/MBPP/test.json")[0]
        iso = (p["net_ifaces"] == ["lo"] and all(s.startswith("blocked") for s in p["tcp"].values())
               and p["dns"].startswith("blocked") and not any(p["host_paths_visible"].values())
               and all(w.startswith("denied") for k, w in p["write"].items() if k not in ("/tmp/x", "/dev/x"))
               and p["append_test_json"].startswith("denied") and p["replace_test_json"].startswith("denied")
               and p["nested_userns"].startswith("denied") and p["new_mount_ns"].startswith("denied")
               and set(p["caps"][k] for k in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")) == {"0000000000000000"}
               and p["uid_gid"] == [65534, 65534, 65534] and p["test_json_sha256"] == oid
               and p["start_method"] == "spawn" and p["multiprocessing_ok"])
        check("MBPP sandbox runs from repo paths and is isolated", iso,
              f"python {p['python']} numpy {p['numpy']} ifaces={p['net_ifaces']} test.json sha ok={p['test_json_sha256'] == oid}")
    except Exception as e:
        check("MBPP sandbox runs from repo paths and is isolated", False, f"{e}; {r.stderr[-500:]}")

    # ---- scratchpad independence ----------------------------------------------------------
    hits = [f.name for f in new_files if re.search(r"/tmp/claude-[0-9]|b9de8309", f.read_text())
            and f.name != "wls_preflight_dryrun.py"]
    check("no new script references a Claude scratchpad path", not hits, f"files={hits}")
    check("LFS copy lives outside the working tree and any scratchpad",
          C.REPO not in C.lfs_dir().parents and not str(C.lfs_dir()).startswith("/tmp/claude-"), str(C.lfs_dir()))
    tracked = subprocess.run(["git", "status", "--porcelain", "data/"], cwd=C.REPO, capture_output=True, text=True).stdout
    check("LFS pointer stubs under data/ are unmodified", tracked.strip() == "")

    # ---- no outputs exist yet -------------------------------------------------------------
    gen = [p.name for p in C.PREFLIGHT_DIR.glob("*") if p.name != "preflight_sample.csv"]
    check("no pre-flight generations/results exist yet", not gen, f"other files={gen}")

    if args.with_selftest:
        with tempfile.TemporaryDirectory(prefix="wls_selftest_") as d:
            r = py("wls_preflight_score.py", "--self-test-stored", d, timeout=3600)
        tail = [x for x in r.stdout.splitlines() if x.startswith("SELF-TEST")]
        check("scoring self-test on stored outputs reproduces stored scores (MBPP in sandbox)",
              r.returncode == 0, tail[0] if tail else r.stderr[-800:])

    n_fail = sum(not ok for _, ok, _ in RESULTS)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed; no model was called.")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
