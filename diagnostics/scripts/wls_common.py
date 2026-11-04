"""Shared constants and loaders for the winner-label stability (WLS) pipeline.

Everything that the mission (WINNER_LABEL_STABILITY_MISSION.md) freezes lives here
once, so the sample freezer, server launcher, generation client, scorer and
dry-run validator cannot drift apart.

No function in this module contacts a model server.
"""

import glob
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
WLS_DIR = REPO / "diagnostics" / "winner_label_stability"
PREFLIGHT_DIR = WLS_DIR / "preflight"
MAIN_SAMPLE_CSV = WLS_DIR / "frozen_sample.csv"
PREFLIGHT_SAMPLE_CSV = PREFLIGHT_DIR / "preflight_sample.csv"
SAMPLING_MANIFEST = WLS_DIR / "sampling_manifest.json"
PREFLIGHT_GENERATIONS = PREFLIGHT_DIR / "preflight_generations.jsonl"
PREFLIGHT_CHECK_CSV = PREFLIGHT_DIR / "preflight_check.csv"
PREFLIGHT_SCORES_CSV = PREFLIGHT_DIR / "preflight_scores.csv"
SANDBOX_SCRIPT_DIR = REPO / "diagnostics" / "scripts" / "wls_sandbox"
MBPP_SANDBOX_SH = REPO / "diagnostics" / "scripts" / "wls_mbpp_sandbox.sh"
SERVE_VLLM_SH = REPO / "diagnostics" / "scripts" / "wls_serve_vllm.sh"

# --------------------------------------------------------------------------- #
# Frozen design (mission: "Frozen Sample Strategy")                           #
# --------------------------------------------------------------------------- #
DS = "DeepSeek-R1-0528-Qwen3-8B"   # class 0
QWEN = "Qwen3-8B"                  # class 1
MODELS = [DS, QWEN]

BENCHMARKS = ["bbh", "korbench", "mbpp", "meld", "medqa", "finqa"]   # fixed order
GROUP = {"bbh": "signal", "korbench": "signal", "mbpp": "signal",
         "meld": "chance", "medqa": "chance", "finqa": "chance"}
MAIN_SEED = 20261101
MAIN_PER_CLASS = 10            # 10 DeepSeek wins + 10 Qwen wins per benchmark
# sha256 of frozen_sample.csv as first frozen (2026-11-03); the FinQA eligibility rule left it unchanged.
MAIN_SAMPLE_SHA256 = "09b0cda6e4c5161e15aff0df7c41079eb455dddf90ef3d5e4686f8a500cc79c9"
PREFLIGHT_SEED = 20261102
PREFLIGHT_PER_BENCHMARK = 20   # simple random sample, ties included

# Stored original outputs (mission Audit §2); the `prompt` field is the exact text sent.
STORED_FILES = {
    ("bbh", DS): "bbh-test-DeepSeek-R1-0528-Qwen3-8B-20251004_152248.json",
    ("bbh", QWEN): "bbh-test-Qwen3-8B-20250919_071923.json",
    ("korbench", DS): "korbench-test-DeepSeek-R1-0528-Qwen3-8B-20251004_184641.json",
    ("korbench", QWEN): "korbench-test-Qwen3-8B-20250919_102827.json",
    ("mbpp", DS): "mbpp-test-DeepSeek-R1-0528-Qwen3-8B-20251004_235947.json",
    ("mbpp", QWEN): "mbpp-test-Qwen3-8B-20250919_132029.json",
    ("meld", DS): "meld-test-DeepSeek-R1-0528-Qwen3-8B-20251019_064225.json",
    ("meld", QWEN): "meld-test-Qwen3-8B-20251019_064159.json",
    ("medqa", DS): "medqa-test-DeepSeek-R1-0528-Qwen3-8B-20251019_064141.json",
    ("medqa", QWEN): "medqa-test-Qwen3-8B-20251019_064130.json",
    ("finqa", DS): "finqa-test-DeepSeek-R1-0528-Qwen3-8B-20251004_163738.json",
    ("finqa", QWEN): "finqa-test-Qwen3-8B-20250919_081013.json",
}

# Evaluator class the factory must return for each benchmark (mission: Evaluator-Reuse Policy).
EXPECTED_EVALUATOR = {"bbh": "BBHEvaluator", "korbench": "KORBenchEvaluator",
                      "mbpp": "MBPPEvaluator", "meld": "MELDEvaluator",
                      "medqa": "MedQAEvaluator", "finqa": "FinQAEvaluator"}
# Field the ground-truth-only evaluators read from `item`.
GT_FIELD = {"bbh": "target", "meld": "answer", "medqa": "answer", "finqa": "answer"}

# --------------------------------------------------------------------------- #
# Reconstructed serving setup (reconstruction choices, not original values)   #
# --------------------------------------------------------------------------- #
HF_REPO = {DS: "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", QWEN: "Qwen/Qwen3-8B"}
HF_REVISION = {DS: "6e8885a6ff5c1dc5201574c8fd700323f23c25fa",
               QWEN: "b968826d9c46dd6066d109eabc6255188de91218"}
# The version the LLMRouterBench paper (App. B.1) reports for its ~7B pool; still a
# reconstruction choice, not proof of what served these two stored files.
VLLM_VERSION = "0.8.4"
MAX_MODEL_LEN = 32768
DTYPE = "bfloat16"
# vLLM 0.8.4's default: the model's generation_config.json fills sampling fields the request omits.
GENERATION_CONFIG = "auto"

# Recorded original request sampling (collector config); nothing else is sent.
TEMPERATURE = 0.2
TOP_P = 1.0
REQUEST_KEYS = ("model", "messages", "temperature", "top_p")

# What vLLM 0.8.4 ModelConfig.get_diff_sampling_param() yields under --generation-config auto
# at the pinned revisions (source-audited 2026-11-03, not observed on a server yet):
#   Qwen3-8B generation_config.json (sha256 2325da0f15bb848e...): temperature 0.6, top_k 20, top_p 0.95
#   DeepSeek: no generation_config.json -> GenerationConfig.from_model_config -> no sampling fields
# The 0.8.4 server logs these at startup ("Using default chat sampling params from model: {...}");
# wls_preflight_generate.py checks that log line on the GPU host before sending anything.
MODEL_SAMPLING_DEFAULTS = {DS: {}, QWEN: {"temperature": 0.6, "top_k": 20, "top_p": 0.95}}
# vLLM 0.8.4 ChatCompletionRequest._DEFAULT_SAMPLING_PARAMS (used when neither side sets a field).
VLLM_NEUTRAL_SAMPLING = {"repetition_penalty": 1.0, "temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0}


def effective_sampling(request, model_defaults):
    """Mirror of vLLM 0.8.4 ChatCompletionRequest.to_sampling_params precedence for the sampling
    fields: request value if given, else the model's generation_config default, else vLLM's
    neutral default. (max_tokens: none from request or model, so max_model_len - prompt_tokens.)"""
    return {k: request[k] if request.get(k) is not None else model_defaults.get(k, v)
            for k, v in VLLM_NEUTRAL_SAMPLING.items()}


# Expected effective sampling per model under the reconstruction (temperature/top_p from the
# request override Qwen's 0.6/0.95; Qwen's top_k=20 applies; DeepSeek has no top_k).
EXPECTED_EFFECTIVE_SAMPLING = {
    DS: {"repetition_penalty": 1.0, "temperature": TEMPERATURE, "top_p": TOP_P, "top_k": -1, "min_p": 0.0},
    QWEN: {"repetition_penalty": 1.0, "temperature": TEMPERATURE, "top_p": TOP_P, "top_k": 20, "min_p": 0.0},
}

# vLLM 0.8.4 V1 Prometheus names (vllm/v1/metrics/loggers.py). The counters are created even
# with prefix caching off; V0 exposes only vllm:gpu_prefix_cache_hit_rate, so their absence
# means the server is not the expected V1 engine.
PREFIX_CACHE_HITS_METRIC = "vllm:gpu_prefix_cache_hits_total"
PREFIX_CACHE_QUERIES_METRIC = "vllm:gpu_prefix_cache_queries_total"


def build_request(model, prompt):
    """The exact chat-completions body: one user message, no system prompt,
    temperature/top_p only (no max_tokens, seed, top_k, extra_body)."""
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}")
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
    }


# --------------------------------------------------------------------------- #
# Loaders                                                                     #
# --------------------------------------------------------------------------- #
def sha1_16(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stored_path(dataset, model):
    p = REPO / "results" / "bench" / dataset / "test" / model / STORED_FILES[(dataset, model)]
    # The mission names exactly one stored file per (benchmark, model).
    found = glob.glob(str(REPO / "results" / "bench" / dataset / "test" / model / "*.json"))
    if found != [str(p)]:
        raise RuntimeError(f"expected exactly {p}, found {found}")
    return p


def load_stored(dataset, model):
    with open(stored_path(dataset, model), encoding="utf-8") as f:
        return {int(r["index"]): r for r in json.load(f)["records"]}


def load_pool():
    """The full routing pool joined to the frozen folds (mission sampling frame step 1)."""
    import sys
    sys.path.insert(0, str(REPO / "diagnostics" / "scripts"))
    from run_cv_eval import load_full_pool_with_folds
    pool = load_full_pool_with_folds()
    folds = pd.read_csv(REPO / "diagnostics" / "stratified_cv" / "fold_assignments.csv")
    pool = pool.merge(folds[["dataset", "index", "query_sha1"]], on=["dataset", "index"],
                      how="left", validate="one_to_one")
    return pool


def load_sample(path):
    return pd.read_csv(path, dtype={"query_sha1": str, "prompt_sha256": str})


def git_head():
    def run(*a):
        return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "branch": run("branch", "--show-current"),
            "dirty": bool(run("status", "--porcelain"))}


# --------------------------------------------------------------------------- #
# Git LFS benchmark data (kept outside the working tree; see wls_fetch_lfs.py) #
# --------------------------------------------------------------------------- #
LFS_FILES = (["data/KORBench/three-shot.yaml"]
             + [f"data/KORBench/{t}/{f}"
                for t in ["cipher", "operation", "puzzle", "counterfactual", "logic"]
                for f in ["sample.jsonl", "three-shot.jsonl", "rule.jsonl"]]
             + ["data/MBPP/test.json"]
             # Not read by any evaluator here; read only by the FinQA eligibility rule below.
             + ["data/FinQA/test.json"])
DEFAULT_LFS_DIR = Path.home() / ".cache" / "LLMRouterBench" / "wls_lfs_data"


def lfs_dir():
    """Directory holding the verified LFS copy (env WLS_LFS_DIR overrides)."""
    return Path(os.environ.get("WLS_LFS_DIR", DEFAULT_LFS_DIR)).resolve()


def lfs_pointer(rel):
    kv = dict(line.split(" ", 1) for line in (REPO / rel).read_text().splitlines() if " " in line)
    return kv["oid"].split(":", 1)[1], int(kv["size"])


def verify_lfs_copy(root=None):
    """Return {rel: ok} comparing each copied object with its committed pointer."""
    root = Path(root or lfs_dir())
    out = {}
    for rel in LFS_FILES:
        oid, size = lfs_pointer(rel)
        p = root / rel
        out[rel] = p.is_file() and p.stat().st_size == size and sha256_file(p) == oid
    return out


def point_evaluators_at_lfs(root=None):
    """Redirect only the module-level DATA_DIR of the KorBench/MBPP evaluators (mission §7a)."""
    import sys
    sys.path.insert(0, str(REPO))
    import evaluation.KORBench.korbench as kb
    import evaluation.MBPP.mbpp as mb
    root = Path(root or lfs_dir())
    kb.DATA_DIR = str(root / "data" / "KORBench")
    mb.DATA_DIR = str(root / "data" / "MBPP")
    return kb, mb


# --------------------------------------------------------------------------- #
# FinQA data-validity rule (mission: Frozen Sample Strategy, frame step 5)    #
# --------------------------------------------------------------------------- #
def finqa_empty_answer_indices(root=None):
    """1-based FinQA test indices whose `answer` field -- the field the unchanged
    FinQAEvaluator reads -- is empty after stripping whitespace.

    The data are loaded with the evaluator's own load_data (DATA_DIR redirected), and every
    rebuilt prompt must equal the stored prompt, so the index mapping is the evaluator's."""
    import sys
    sys.path.insert(0, str(REPO))
    import evaluation.FinQA.finqa as fq
    root = Path(root or lfs_dir())
    if not verify_lfs_copy(root)["data/FinQA/test.json"]:
        raise RuntimeError("verified data/FinQA/test.json missing; run wls_fetch_lfs.py")
    fq.DATA_DIR = str(root / "data" / "FinQA")
    data = fq.FinQAEvaluator().load_data("test")
    for m in MODELS:
        stored = load_stored("finqa", m)
        if len(stored) != len(data) or any(data[i - 1]["prompt"] != r["prompt"] for i, r in stored.items()):
            raise RuntimeError(f"FinQA load_data order does not reproduce the stored prompts ({m})")
    return {i for i, item in enumerate(data, 1) if not str(item["answer"]).strip()}


def assert_sample_eligible(df, empty_finqa=None):
    """Fail loudly if a sample contains a FinQA row with an empty evaluator reference."""
    empty_finqa = finqa_empty_answer_indices() if empty_finqa is None else empty_finqa
    bad = sorted(int(i) for d, i in zip(df["dataset"], df["index"]) if d == "finqa" and int(i) in empty_finqa)
    if bad:
        raise RuntimeError(f"ineligible FinQA rows (empty `answer`) in sample: {bad}")
