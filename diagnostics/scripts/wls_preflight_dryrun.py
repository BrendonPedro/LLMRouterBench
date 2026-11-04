"""Dry-run validation of the winner-label stability pre-flight pipeline. Calls NO model.

Checks: samples regenerate exactly and are disjoint; row counts; prompts reconstruct
exactly; model names exact; generation request matches the mission; serving settings
(vLLM 0.8.4, --generation-config auto, expected Qwen top_k=20); no cache use; every sample
row resolves to its evaluator; MBPP sandbox available and isolated; no path depends on a
Claude scratchpad.

Checks marked [static] validate configuration and source-derived expectations only. vLLM is not
installed here and no server runs, so what the real 0.8.4 server does is verified on the GPU
host by wls_preflight_generate.py / wls_preflight_score.py; those are listed as DEFERRED at
the end and never counted as passes.

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
import wls_preflight_generate as G

RESULTS = []

# Verified only against a running vLLM 0.8.4 server on the GPU host (never passed here).
DEFERRED = [
    "server /version == 0.8.4 and server log shows exactly one 'Initializing a V1 LLM engine (v0.8.4)'",
    "Qwen3-8B server log: 'Using default chat sampling params from model: "
    "{temperature 0.6, top_k 20, top_p 0.95}'; DeepSeek: no such line",
    "/v1/models serves exactly the one expected name with max_model_len 32768",
    "/metrics: cache_config_info enable_prefix_caching=\"False\"; V1 counters "
    "vllm:gpu_prefix_cache_{hits,queries}_total present; zero hits during the run",
    "server manifest: revision, bf16, generation_config auto, no quantization, GPU recorded",
    "per output: <think> present, no reasoning field, no finish_reason=length below 32768",
]


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
    banned = ("top_k", "max_tokens", "max_completion_tokens", "seed", "min_p", "extra_body",
              "reasoning_effort", "chat_template_kwargs")
    for m in C.MODELS:
        r = py("wls_preflight_generate.py", "--model", m, "--dry-run")
        lines = [json.loads(x) for x in r.stdout.splitlines()]
        reqs = [x["request"] for x in lines]
        check(f"client --dry-run builds 120 requests for {m} and sends nothing",
              r.returncode == 0 and len(lines) == 120
              and all(set(q) == set(C.REQUEST_KEYS) and q["model"] == m for q in reqs)
              and all(len(q["messages"]) == 1 and q["messages"][0]["role"] == "user" for q in reqs),
              r.stderr.strip())
        check(f"requests for {m}: temperature=0.2 and top_p=1.0 explicit; no top_k/max_tokens/seed",
              all(q["temperature"] == 0.2 and q["top_p"] == 1.0 for q in reqs)
              and not any(k in q for q in reqs for k in banned), f"banned keys checked: {banned}")

    # ---- server launch settings -----------------------------------------------------------
    for key, m in (("deepseek", C.DS), ("qwen", C.QWEN)):
        r = subprocess.run(["bash", str(C.SERVE_VLLM_SH), key, "--print-cmd"], capture_output=True, text=True)
        argv = r.stdout.split("\n")[:-1]
        def val(f):
            return argv[argv.index(f) + 1] if f in argv else None
        ok = (argv[:3] == ["vllm", "serve", C.HF_REPO[m]] and val("--revision") == C.HF_REVISION[m]
              and val("--tokenizer-revision") == C.HF_REVISION[m] and val("--served-model-name") == m
              and val("--dtype") == "bfloat16" and val("--max-model-len") == str(C.MAX_MODEL_LEN)
              and val("--generation-config") == C.GENERATION_CONFIG and "--no-enable-prefix-caching" in argv
              and not any(a.startswith(("--reasoning-parser", "--enable-prefix-caching", "--seed",
                                        "--override-generation-config", "--quantization", "--enable-reasoning"))
                          for a in argv))
        check(f"vLLM launch settings exact for {m}", ok and r.returncode == 0, " ".join(argv))
    launcher = C.SERVE_VLLM_SH.read_text()
    pin = re.search(r'^VLLM_VERSION="([^"]+)"', launcher, re.M)
    check("vLLM version pinned identically in launcher and client", pin and pin.group(1) == C.VLLM_VERSION,
          C.VLLM_VERSION)
    check("[static] vLLM version pin is 0.8.4 (paper-reported version; reconstruction choice)",
          C.VLLM_VERSION == "0.8.4" and pin and pin.group(1) == "0.8.4")
    check("[static] generation config is auto in constants, launch command and launcher manifest",
          C.GENERATION_CONFIG == "auto" and len(re.findall(r"^\s+--generation-config auto$", launcher, re.M)) == 1
          and '"generation_config": "auto"' in launcher and "--generation-config vllm" not in launcher)
    check("[static] max model length 32768; bf16; no reasoning parser; prefix caching off (launcher manifest)",
          C.MAX_MODEL_LEN == 32768 and C.DTYPE == "bfloat16"
          and '"max_model_len": 32768' in launcher and '"reasoning_parser": None' in launcher
          and '"prefix_caching": False' in launcher)
    check("[static] launcher captures a fresh server log for the runtime log checks",
          'LOG="$RUN_DIR/server_${SERVED}.log"' in launcher and 'exec "${CMD[@]}" > >(tee "$LOG") 2>&1' in launcher
          and '"server_log": log' in launcher)

    # ---- effective sampling expected under vLLM 0.8.4 --generation-config auto -------------
    eff = {m: C.effective_sampling(C.build_request(m, "PROMPT"), C.MODEL_SAMPLING_DEFAULTS[m]) for m in C.MODELS}
    check("[static] Qwen3-8B: model default top_k=20 applies when the request omits top_k; "
          "request temperature/top_p override 0.6/0.95",
          C.MODEL_SAMPLING_DEFAULTS[C.QWEN] == {"temperature": 0.6, "top_k": 20, "top_p": 0.95}
          and eff[C.QWEN] == C.EXPECTED_EFFECTIVE_SAMPLING[C.QWEN] and eff[C.QWEN]["top_k"] == 20
          and eff[C.QWEN]["temperature"] == 0.2 and eff[C.QWEN]["top_p"] == 1.0, json.dumps(eff[C.QWEN]))
    check("[static] DeepSeek: no generation_config sampling defaults, so top_k stays disabled (-1)",
          C.MODEL_SAMPLING_DEFAULTS[C.DS] == {} and eff[C.DS] == C.EXPECTED_EFFECTIVE_SAMPLING[C.DS]
          and eff[C.DS]["top_k"] == -1, json.dumps(eff[C.DS]))

    # ---- runtime check logic, exercised on synthetic 0.8.4-format text (not a server) ------
    v1 = "INFO 11-04 10:00:00 [core.py:61] Initializing a V1 LLM engine (v0.8.4) with config: model='x'\n"
    qline = ("INFO 11-04 10:00:09 [serving_chat.py:118] Using default chat sampling params from model: "
             "{'temperature': 0.6, 'top_k': 20, 'top_p': 0.95}\n")
    cases = []
    with tempfile.TemporaryDirectory(prefix="wls_logprobe_") as d:
        def log_errs(text, m):
            f = Path(d) / "server.log"
            f.write_text(text)
            return G.check_server_log(f, m)[1]
        cases += [("qwen expected", not log_errs(v1 + qline, C.QWEN)),
                  ("deepseek expected", not log_errs(v1, C.DS)),
                  ("qwen without defaults line", bool(log_errs(v1, C.QWEN))),
                  ("qwen top_k missing", bool(log_errs(v1 + qline.replace("'top_k': 20, ", ""), C.QWEN))),
                  ("deepseek with defaults line", bool(log_errs(v1 + qline, C.DS))),
                  ("V0 engine", bool(log_errs(v1.replace("V1", "V0") + qline, C.QWEN))),
                  ("wrong version", bool(log_errs(v1.replace("0.8.4", "0.33.0") + qline, C.QWEN))),
                  ("missing log", bool(G.check_server_log(Path(d) / "absent.log", C.QWEN)[1]))]
    check("[static] server-log check accepts the expected 0.8.4 lines and rejects 7 wrong cases",
          all(ok for _, ok in cases), "; ".join(f"{n}={'ok' if ok else 'WRONG'}" for n, ok in cases))
    ok_metrics = ('vllm:cache_config_info{block_size="16",enable_prefix_caching="False"} 1.0\n'
                  'vllm:gpu_prefix_cache_queries_total{model_name="Qwen3-8B"} 0.0\n'
                  'vllm:gpu_prefix_cache_hits_total{model_name="Qwen3-8B"} 0.0\n')
    info, cnt = G.parse_metrics(ok_metrics)
    info33, cnt33 = G.parse_metrics(ok_metrics.replace("gpu_prefix_cache", "prefix_cache"))
    check("[static] /metrics parser reads the vLLM 0.8.4 V1 names and flags the 0.33.0 names as absent",
          info == {"block_size": "16", "enable_prefix_caching": "False"}
          and cnt == {C.PREFIX_CACHE_HITS_METRIC: 0.0, C.PREFIX_CACHE_QUERIES_METRIC: 0.0}
          and all(v is None for v in cnt33.values()), f"names={list(cnt)}")

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
    frozen = ["evaluation/", "diagnostics/scripts/wls_mbpp_sandbox.sh", "diagnostics/scripts/wls_sandbox/",
              str(C.MAIN_SAMPLE_CSV.relative_to(C.REPO)), str(C.PREFLIGHT_SAMPLE_CSV.relative_to(C.REPO)),
              str(C.SAMPLING_MANIFEST.relative_to(C.REPO))]
    st = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", *frozen],
                        cwd=C.REPO, capture_output=True, text=True).stdout
    check("evaluator code, MBPP sandbox and frozen sample files unchanged vs HEAD", st.strip() == "",
          st.strip().replace("\n", "; ") or ", ".join(frozen))

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
    n_static = sum(name.startswith("[static]") for name, _, _ in RESULTS)
    print("\nDEFERRED to the GPU host (need a running vLLM 0.8.4 server; NOT checked here):")
    for d in DEFERRED:
        print(f"  DEFERRED  {d}")
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed ({n_static} of them [static]); "
          f"{len(DEFERRED)} runtime checks deferred; no model was called.")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
