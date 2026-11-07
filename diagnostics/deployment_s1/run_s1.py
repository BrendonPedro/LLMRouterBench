"""S1 / E0-Q3 server-only quality gate: frozen sample, vLLM generation, scoring.

Frozen design: ~/dev/cloud_slm-router/deployment-pivot/E0-SPLIT-VIABILITY.md §4.3-§4.4, §8
(early stop) and WEEK-EXECUTION-PLAN.md (S1, harness parity, G1). Pre-registration:
diagnostics/deployment_s1/PREREG.md. Pod commands: diagnostics/deployment_s1/RUNPOD.md.

Subcommands (all resumable; nothing is ever overwritten):
  sample      draw / re-verify the frozen 200-query sample (CPU)
  fetch-data  download LiveCodeBench test data + cached Qwen3-8B outputs, sha256-verified (CPU)
  check       offline sample / evaluator / data / request checks (CPU)
  preflight   GPU, disk, engine and dependency checks on the pod
  smoke       tiny end-to-end run (one small model, 8 queries, short outputs) + plumbing checks
  parity      harness parity: Qwen3-8B bf16 at the cached run's settings vs the cached scores
  run         the full S1 matrix (5 models x 2 modes x 2 seeds x 200 queries), unattended
  score       score generations with the unchanged LLMRouterBench evaluators (CPU)
  status      progress per model / mode / seed
  selftest    CPU-only unit tests (fake OpenAI-compatible server, interruption + resume)

Generation uses only the standard library (plain HTTP to a vLLM OpenAI server, no SDK, no
repository response cache). Prompts are the exact cached LLMRouterBench prompt strings; scoring
calls each benchmark's unchanged evaluator `evaluate()` method.
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

S1_DIR = REPO / "diagnostics" / "deployment_s1"
SAMPLE_JSONL = S1_DIR / "sample_200.jsonl"
SAMPLE_MANIFEST = S1_DIR / "sample_manifest.json"
DATA_DIR = Path(os.environ.get("S1_DATA_DIR", Path.home() / ".cache" / "LLMRouterBench" / "s1_data"))
RUNS_DIR = Path(os.environ.get("S1_RUNS_DIR", S1_DIR / "runs"))

# --------------------------------------------------------------------------- #
# Frozen sample (E0 §4.4)                                                     #
# --------------------------------------------------------------------------- #
SEED = 20261107
# (dataset id, cached split directory, n); fixed order.
BENCHMARKS = [("mmlupro", "test_1000", 60), ("gpqa", "test", 60),
              ("livemathbench", "test", 40), ("livecodebench", "test", 40)]
N_PER_BENCH = {d: n for d, _, n in BENCHMARKS}
SPLIT_DIR = {d: s for d, s, _ in BENCHMARKS}
SMOKE_PER_BENCH = 2

# Sampling frame = the cached LLMRouterBench Qwen3-8B record files (the `prompt` field is the
# exact text the collector sent; `index` is the 1-based position in the evaluator's load order).
CACHED_FILES = {
    "mmlupro": ("mmlupro-test-Qwen3-8B-20251019_064245.json",
                "c7a4219859615ffefef13ff8894ee85abea4e0a032b9129a6bac42b2880a6056"),
    "gpqa": ("gpqa-test-Qwen3-8B-20251019_064106.json",
             "87c8f9226fa944eecbb4ae6335b593dfc3837606b5b58b5a37df5cddda711207"),
    "livemathbench": ("livemathbench-test-Qwen3-8B-20250919_112608.json",
                      "6a34d4d51d91fdc295a1ec1d8490fa4b330945df10253b79c878c41dad12e7c9"),
    "livecodebench": ("livecodebench-test-Qwen3-8B-20251002_155958.json",
                      "ff871d9818d7ac2cfdf91113bee648534f2908b0dda33da15c3f1c24805258eb"),
}
CACHED_COUNTS = {"mmlupro": 1001, "gpqa": 198, "livemathbench": 121, "livecodebench": 1055}
# The released results bundle those files come from (byte-identical members).
BENCH_TARBALL_URL = ("https://huggingface.co/datasets/NPULH/LLMRouterBench/resolve/"
                     "b95870f9e60c589e82f7972c91ffa8de9ca89f85/bench-release.tar.gz")
BENCH_TARBALL_SHA256 = "b79f8cde1a6f029c2efa663a3a3b6f7748defb22341fe59f328cebef6648c8f1"

# LiveCodeBench test data: the repo's data/LiveCodeBench is an LFS stub whose LFS budget is
# exhausted (HTTP 403 on origin and upstream), so the evaluator's inputs come from the public
# release. release_v6 = test.jsonl..test6.jsonl = 1055 problems; the evaluator concatenates
# sorted(*.jsonl), and `check` verifies every rebuilt prompt equals the cached prompt.
LCB_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
LCB_FILES = {
    "test.jsonl": (1252609773, "2bd02b38beb48e8c46b5b9987095d999ff38cd8efc255ea5d58974317c48f63f"),
    "test2.jsonl": (713377060, "095df7c5daf15f882c51a9deb84085cff1e073495a5dbcf95015a564d485f3a3"),
    "test3.jsonl": (623360766, "28ed26cc83363ce3f1fe2d5fad9f8393077beb1907b167a31bd3b32f80801b79"),
    "test4.jsonl": (1204644685, "d711138ddaebfcf5f8ec6a4283ee677298c0f5c5d374a235af92aaf0584510da"),
    "test5.jsonl": (557699297, "7f77571c2a6df0c2a72a3277650309f67e01e0008e18117e624633df53f81214"),
    "test6.jsonl": (134303240, "bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5"),
}
LCB_URL = "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/{rev}/{name}"

# sha256 of sample_200.jsonl as frozen on 2026-11-07 (filled in once, then never changed).
SAMPLE_SHA256 = "2c4df49360f997c3e35593030da40111c2826f068f03b1401284a394d9716955"

# --------------------------------------------------------------------------- #
# Models and decoding (E0 §3.1, §4.3)                                         #
# --------------------------------------------------------------------------- #
# Official Qwen repos named on the Qwen3 AWQ model cards (and the E0 desk check); revisions are
# the HF main SHAs on 2026-11-07. AWQ checkpoints declare torch_dtype float16, so they are served
# with --dtype auto (= float16); bf16 checkpoints with --dtype bfloat16.
# presence_penalty (pre-data amendment A1, 2026-11-07, PREREG §10): the AWQ cards' recommended 1.5
# for quantised models is sent with every AWQ request; None = not sent (vLLM default 0), unchanged
# for the bf16 models.
AWQ_PRESENCE_PENALTY = 1.5
MODELS = {
    "Qwen3-32B": {"repo": "Qwen/Qwen3-32B", "revision": "9216db5781bf21249d130ec9da846c4624c16137",
                  "dtype": "bfloat16", "weights_gb": 65.5, "presence_penalty": None},
    "Qwen3-8B": {"repo": "Qwen/Qwen3-8B", "revision": "b968826d9c46dd6066d109eabc6255188de91218",
                 "dtype": "bfloat16", "weights_gb": 16.4, "presence_penalty": None},
    "Qwen3-8B-AWQ": {"repo": "Qwen/Qwen3-8B-AWQ", "revision": "4da05a8edb55c6046cce958586c33b61da07bb79",
                     "dtype": "auto", "weights_gb": 6.1, "presence_penalty": AWQ_PRESENCE_PENALTY},
    "Qwen3-14B-AWQ": {"repo": "Qwen/Qwen3-14B-AWQ", "revision": "31c69efc29464b6bb0aee1398b5a7b50a99340c3",
                      "dtype": "auto", "weights_gb": 10.0, "presence_penalty": AWQ_PRESENCE_PENALTY},
    "Qwen3-32B-AWQ": {"repo": "Qwen/Qwen3-32B-AWQ", "revision": "0499c3ac83fdef8810b907a23894ba91e95eddd8",
                      "dtype": "auto", "weights_gb": 19.3, "presence_penalty": AWQ_PRESENCE_PENALTY},
}
AWQ_MODELS = [m for m in MODELS if m.endswith("-AWQ")]
REFERENCE_MODEL = "Qwen3-32B"
# Reference first, then the 12/16 GB competitors that decide the early stop, then 24 GB ones.
RUN_ORDER = ["Qwen3-32B", "Qwen3-8B-AWQ", "Qwen3-14B-AWQ", "Qwen3-32B-AWQ", "Qwen3-8B"]

VLLM_VERSION = "0.8.5.post1"   # first release the Qwen3 AWQ cards support (vllm>=0.8.5)
MAX_MODEL_LEN = 32768          # Qwen3 native context; also the cached run's total cap
MAX_NEW_TOKENS = 16384         # E0 §4.3
SEEDS = [0, 1]

# One entry per decoding setting. `mode` is the chat-template switch; `sampling` names the
# parameter set. min_p 0 as on the model cards. presence_penalty is per model (MODELS), not per
# setting: 1.5 on AWQ requests, not sent on bf16 requests.
SETTINGS = {
    "thinking": {"mode": "thinking", "sampling": "vendor_thinking", "enable_thinking": True,
                 "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
    "nonthinking": {"mode": "nonthinking", "sampling": "vendor_nonthinking", "enable_thinking": False,
                    "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0},
    # Harness parity: the cached run's settings (collector config: T 0.2, top_p 1.0; thinking on;
    # top_k 20 = the model generation_config default that vLLM applied; no max_tokens, so the
    # 32768-token total cap applies as in the cached outputs).
    "parity": {"mode": "thinking", "sampling": "cached_run", "enable_thinking": True,
               "temperature": 0.2, "top_p": 1.0, "top_k": 20, "min_p": 0.0},
}

RUN_SPECS = {
    "s1": {"models": RUN_ORDER, "settings": ["thinking", "nonthinking"], "seeds": SEEDS,
           "max_tokens": MAX_NEW_TOKENS, "per_bench": None, "concurrency": 64},
    "parity": {"models": ["Qwen3-8B"], "settings": ["parity"], "seeds": [0],
               "max_tokens": None, "per_bench": None, "concurrency": 64},
    "smoke": {"models": ["Qwen3-8B-AWQ"], "settings": ["thinking", "nonthinking"], "seeds": [0],
              "max_tokens": 1024, "per_bench": SMOKE_PER_BENCH, "concurrency": 16},
}

# Harness-parity rule (WEEK-EXECUTION-PLAN Mon 11:00): validation only, not a SPLIT criterion.
PARITY_AGREEMENT_MIN = 0.80
PARITY_AGREEMENT_BENCHES = ["mmlupro", "gpqa"]

TEMPLATE_MARGIN_TOKENS = 16    # non-thinking template adds "<think>\n\n</think>\n\n" to the prompt


# --------------------------------------------------------------------------- #
# Small helpers                                                               #
# --------------------------------------------------------------------------- #
def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def qkey(dataset, index):
    return f"{dataset}:{int(index)}"


def job_key(model, setting, seed, dataset, index):
    return f"{model}|{setting}|{seed}|{qkey(dataset, index)}"


def git_head():
    def run(*a):
        try:
            return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception as e:
            return f"unavailable: {type(e).__name__}"
    return {"commit": run("rev-parse", "HEAD"), "branch": run("branch", "--show-current"),
            "dirty": bool(run("status", "--porcelain", "--", "diagnostics/deployment_s1"))}


def sh(*cmd, timeout=60):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception as e:
        return f"unavailable: {type(e).__name__}: {e}"


def append_jsonl(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def repair_tail(path):
    """An interrupted write can leave a final line without its newline. Move that fragment to a
    side file so the next append starts on a fresh line. Returns the side file or None."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    with open(path, "rb+") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - 1))
        if f.read(1) == b"\n":
            return None
        f.seek(0)
        data = f.read()
        cut = data.rfind(b"\n") + 1
        side = path.with_name(f"{path.name}.torn-{datetime.datetime.now():%Y%m%dT%H%M%S}")
        side.write_bytes(data[cut:])
        f.truncate(cut)
    print(f"repaired torn final line of {path} -> {side}", file=sys.stderr)
    return side


def read_jsonl(path):
    """Every complete record; an unparsable line is fatal unless it is a torn final line."""
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").split("\n")
    out = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                print(f"ignoring torn final line of {path}", file=sys.stderr)
                continue
            raise
    return out


def wilson(k, n, z=1.959963984540054):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# --------------------------------------------------------------------------- #
# Cached LLMRouterBench outputs and the frozen sample                         #
# --------------------------------------------------------------------------- #
def bench_roots():
    return [REPO / "results" / "bench", DATA_DIR / "bench-release"]


def cached_path(dataset, model="Qwen3-8B", root=None):
    fname, _ = CACHED_FILES[dataset]
    roots = [Path(root)] if root else bench_roots()
    for r in roots:
        p = r / dataset / SPLIT_DIR[dataset] / model / fname
        if p.exists():
            return p
    return None


def load_cached(dataset, root=None):
    """Cached Qwen3-8B records in index order, hash-verified."""
    p = cached_path(dataset, root=root)
    if p is None:
        raise FileNotFoundError(f"cached {dataset} file not found under {bench_roots()}; "
                                f"run `run_s1.py fetch-data`")
    if sha256_file(p) != CACHED_FILES[dataset][1]:
        raise RuntimeError(f"{p}: sha256 differs from the pinned cached file")
    recs = json.loads(p.read_text(encoding="utf-8"))["records"]
    idx = [int(r["index"]) for r in recs]
    if idx != list(range(1, CACHED_COUNTS[dataset] + 1)):
        raise RuntimeError(f"{p}: indices are not 1..{CACHED_COUNTS[dataset]}")
    return recs


def sampling_frame(recs):
    """All records, minus later copies of a prompt that occurs more than once in the benchmark
    (MMLU-Pro test_1000 has one: index 338 repeats index 16)."""
    seen, frame, dropped = set(), [], []
    for r in recs:
        if r["prompt"] in seen:
            dropped.append(int(r["index"]))
            continue
        seen.add(r["prompt"])
        frame.append(r)
    return frame, dropped


def permutation(dataset, frame):
    """Seeded permutation of the frame. The sample is a prefix of it, so the pre-declared
    extension to 400 (E0 §4.4) uses `scale=2` and contains the 200."""
    order = sorted(frame, key=lambda r: int(r["index"]))
    random.Random(f"deployment-s1:{SEED}:{dataset}").shuffle(order)
    return order


def draw_sample(scale=1, root=None):
    rows, frames = [], {}
    for dataset, split, n in BENCHMARKS:
        recs = load_cached(dataset, root=root)
        frame, dropped = sampling_frame(recs)
        frames[dataset] = {"records": len(recs), "frame": len(frame), "dropped_duplicate_prompts": dropped}
        for rank, r in enumerate(permutation(dataset, frame)[: n * scale]):
            rows.append({
                "key": qkey(dataset, r["index"]), "dataset": dataset, "split": split,
                "index": int(r["index"]), "rank": rank,
                "prompt": r["prompt"], "prompt_sha256": sha256_text(r["prompt"]),
                # The non-code evaluators read only item["answer"]; the cached ground_truth is that
                # value (verified by re-scoring, `check`). LiveCodeBench reads its tests from data.
                "answer": None if dataset == "livecodebench" else r["ground_truth"],
                "ref_qwen3_8b": {"score": r["score"], "prompt_tokens": r["prompt_tokens"],
                                 "completion_tokens": r["completion_tokens"],
                                 "raw_output_sha256": sha256_text(r["raw_output"])},
            })
    order = {d: i for i, (d, _, _) in enumerate(BENCHMARKS)}
    rows.sort(key=lambda x: (order[x["dataset"]], x["index"]))
    return rows, frames


def sample_bytes(rows):
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode("utf-8")


def load_sample(path=SAMPLE_JSONL, verify=True):
    data = Path(path).read_bytes()
    if verify and SAMPLE_SHA256 != "PENDING" and hashlib.sha256(data).hexdigest() != SAMPLE_SHA256:
        raise RuntimeError(f"{path} sha256 differs from the frozen SAMPLE_SHA256")
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


def cmd_sample(args):
    rows, frames = draw_sample()
    data = sample_bytes(rows)
    digest = hashlib.sha256(data).hexdigest()
    if SAMPLE_JSONL.exists():
        old = SAMPLE_JSONL.read_bytes()
        if old != data:
            sys.exit(f"FAIL: {SAMPLE_JSONL} exists and differs from a fresh draw; it is frozen and "
                     f"never overwritten")
        print(f"OK: {SAMPLE_JSONL} re-drawn byte-identically (sha256 {digest})")
    else:
        SAMPLE_JSONL.write_bytes(data)
        print(f"wrote {SAMPLE_JSONL} ({len(rows)} rows, sha256 {digest})")
    manifest = {
        "created_utc": utcnow(), "seed": SEED,
        "procedure": ("per benchmark: frame = cached LLMRouterBench Qwen3-8B records (later copies of a "
                      "duplicated prompt dropped), sorted by index, shuffled with "
                      "random.Random('deployment-s1:<seed>:<dataset>'); the first n are taken. "
                      "Extension to 400 = first 2n of the same permutation."),
        "n_per_benchmark": N_PER_BENCH, "frames": frames,
        "cached_files": {d: {"path": f"{d}/{SPLIT_DIR[d]}/Qwen3-8B/{f}", "sha256": s}
                         for d, (f, s) in CACHED_FILES.items()},
        "bench_tarball": {"url": BENCH_TARBALL_URL, "sha256": BENCH_TARBALL_SHA256},
        "sample_sha256": digest, "python": sys.version.split()[0],
    }
    if SAMPLE_MANIFEST.exists():
        old = json.loads(SAMPLE_MANIFEST.read_text())
        if old.get("sample_sha256") != digest:
            sys.exit(f"FAIL: {SAMPLE_MANIFEST} names a different sample")
        print(f"OK: {SAMPLE_MANIFEST} matches")
    else:
        SAMPLE_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"wrote {SAMPLE_MANIFEST}")
    if SAMPLE_SHA256 == "PENDING":
        print(f"NOTE: pin SAMPLE_SHA256 = \"{digest}\" in run_s1.py")


# --------------------------------------------------------------------------- #
# Data download (pod or local), outside the working tree                      #
# --------------------------------------------------------------------------- #
def download(url, dst, size, sha256):
    dst = Path(dst)
    if dst.exists() and dst.stat().st_size == size and sha256_file(dst) == sha256:
        print(f"OK   {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    h = hashlib.sha256()
    with urllib.request.urlopen(urllib.request.Request(url), timeout=120) as r, open(tmp, "wb") as f:
        for chunk in iter(lambda: r.read(1 << 22), b""):
            h.update(chunk)
            f.write(chunk)
    if h.hexdigest() != sha256 or tmp.stat().st_size != size:
        tmp.unlink()
        sys.exit(f"BAD  {url}: sha256/size mismatch")
    tmp.replace(dst)
    print(f"GOT  {dst} ({size} bytes)")


def lcb_dir():
    return DATA_DIR / "LiveCodeBench"


def verify_lcb(hash_files=True):
    d = lcb_dir()
    out = {}
    for name, (size, sha) in LCB_FILES.items():
        p = d / name
        out[name] = p.exists() and p.stat().st_size == size and (not hash_files or sha256_file(p) == sha)
    return out


def cmd_fetch_data(args):
    if REPO in DATA_DIR.resolve().parents:
        sys.exit(f"S1_DATA_DIR must be outside the working tree: {DATA_DIR}")
    if not args.skip_lcb:
        for name, (size, sha) in LCB_FILES.items():
            download(LCB_URL.format(rev=LCB_REVISION, name=name), lcb_dir() / name, size, sha)
    if args.skip_bench:
        return
    if all(cached_path(d, root=DATA_DIR / "bench-release") for d in CACHED_FILES):
        print("OK   cached Qwen3-8B files already extracted")
        return
    tarball = Path(args.bench_tarball) if args.bench_tarball else DATA_DIR / "bench-release.tar.gz"
    if not args.bench_tarball:
        download(BENCH_TARBALL_URL, tarball, 1283503080, BENCH_TARBALL_SHA256)
    elif sha256_file(tarball) != BENCH_TARBALL_SHA256:
        sys.exit(f"{tarball}: sha256 differs from the released bench tarball")
    want = {f"bench-release/{d}/{SPLIT_DIR[d]}/Qwen3-8B/{f}" for d, (f, _) in CACHED_FILES.items()}
    with tarfile.open(tarball, "r:gz") as tf:
        for m in tf:
            if m.name in want and m.isfile():
                dst = DATA_DIR / m.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                with tf.extractfile(m) as src, open(dst, "wb") as f:
                    shutil.copyfileobj(src, f)
                print(f"GOT  {dst}")
    for d, (f, sha) in CACHED_FILES.items():
        p = cached_path(d, root=DATA_DIR / "bench-release")
        if p is None or sha256_file(p) != sha:
            sys.exit(f"BAD  cached {d} file missing or wrong hash after extraction")


def iter_lcb_rows():
    """(0-based position, row) in the evaluator's order: sorted(*.jsonl), line by line."""
    files = sorted(lcb_dir().glob("*.jsonl"))
    if [f.name for f in files] != sorted(LCB_FILES):
        raise FileNotFoundError(f"LiveCodeBench files missing in {lcb_dir()}; run `run_s1.py fetch-data`")
    pos = 0
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                yield pos, json.loads(line)
                pos += 1


def load_lcb_rows(positions):
    """Full raw rows for the needed 0-based positions only (the full set is ~4.5 GB)."""
    need, out = set(positions), {}
    for pos, row in iter_lcb_rows():
        if pos in need:
            out[pos] = row
            if len(out) == len(need):
                break
    if len(out) != len(need):
        raise RuntimeError(f"LiveCodeBench rows missing: {sorted(need - set(out))[:10]}")
    return out


# --------------------------------------------------------------------------- #
# Evaluators (unchanged LLMRouterBench code)                                  #
# --------------------------------------------------------------------------- #
class Scorer:
    """Calls each benchmark's unchanged `evaluate(data, output_text)`. Only the `data` item is
    built here: {"answer": ...} for MMLU-Pro / GPQA / LiveMathBench (the only field they read)
    and {"__idx__": pos} with `_raw_data` holding the needed LiveCodeBench rows."""

    def __init__(self, lcb_positions=()):
        from evaluation.GPQA.gpqa import GPQAEvaluator
        from evaluation.LiveCodeBench.livecodebench import LiveCodeBenchEvaluator
        from evaluation.LiveMathBench.livemathbench import LiveMathBenchEvaluator
        from evaluation.MMLUPro.mmlupro import MMLUProEvaluator
        self.ev = {"mmlupro": MMLUProEvaluator(), "gpqa": GPQAEvaluator(),
                   "livemathbench": LiveMathBenchEvaluator(), "livecodebench": LiveCodeBenchEvaluator()}
        if lcb_positions:
            self.ev["livecodebench"]._raw_data = load_lcb_rows(lcb_positions)

    def score(self, q, output_text):
        d = q["dataset"]
        item = {"__idx__": q["index"] - 1} if d == "livecodebench" else {"answer": q["answer"]}
        res = self.ev[d].evaluate(item, output_text)
        v = res.get("is_correct", False)   # same conversion as data_collector/runner.py
        score = None if v is None else (1.0 if v is True else 0.0 if v is False else float(v))
        return score, res.get("prediction"), res.get("ground_truth")


def rebuilt_lcb_prompt(row):
    from evaluation.LiveCodeBench.livecodebench import LiveCodeBenchEvaluator
    light = {"question_content": row.get("question_content", ""), "starter_code": row.get("starter_code", ""),
             "metadata": row.get("metadata", "{}")}
    return LiveCodeBenchEvaluator.format_prompt(None, light)["prompt"]


# --------------------------------------------------------------------------- #
# Requests and the vLLM server                                                #
# --------------------------------------------------------------------------- #
def build_request(model, prompt, setting, seed, max_tokens):
    s = SETTINGS[setting]
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": s["temperature"], "top_p": s["top_p"], "top_k": s["top_k"],
            "min_p": s["min_p"], "seed": seed,
            "chat_template_kwargs": {"enable_thinking": s["enable_thinking"]}}
    if MODELS[model]["presence_penalty"] is not None:
        body["presence_penalty"] = MODELS[model]["presence_penalty"]
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    return body


def request_record(body):
    rec = {k: v for k, v in body.items() if k != "messages"}
    rec["messages"] = [{"role": m["role"], "content_sha256": sha256_text(m["content"])} for m in body["messages"]]
    return rec


def plan_jobs(spec, sample, models=None, settings=None):
    queries = sample
    if spec["per_bench"]:
        queries = [q for q in sample if q["rank"] < spec["per_bench"]]
    jobs = []
    for model in models or spec["models"]:
        for setting in settings or spec["settings"]:
            for seed in spec["seeds"]:
                for q in queries:
                    jobs.append({"key": job_key(model, setting, seed, q["dataset"], q["index"]),
                                 "model": model, "setting": setting, "seed": seed, "q": q})
    return jobs


def http(method, url, body=None, timeout=60):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode("utf-8")


def server_checks(base, model):
    rec, errors = {}, []
    _, body = http("GET", f"{base}/v1/models")
    models = json.loads(body)["data"]
    rec["v1_models"] = [{k: m.get(k) for k in ("id", "root", "max_model_len")} for m in models]
    if [m["id"] for m in models] != [model]:
        errors.append(f"server serves {[m['id'] for m in models]}, expected [{model!r}]")
    elif models[0].get("max_model_len") != MAX_MODEL_LEN:
        errors.append(f"max_model_len {models[0].get('max_model_len')} != {MAX_MODEL_LEN}")
    _, body = http("GET", f"{base}/version")
    rec["vllm_version"] = json.loads(body).get("version")
    if rec["vllm_version"] != VLLM_VERSION:
        errors.append(f"vllm {rec['vllm_version']} != {VLLM_VERSION}")
    return rec, errors


def vllm_python(vllm_bin):
    p = Path(shutil.which(vllm_bin) or vllm_bin).with_name("python")
    return str(p) if p.exists() else None


def serve_cmd(args, model):
    m = MODELS[model]
    return [args.vllm_bin, "serve", m["repo"], "--revision", m["revision"], "--tokenizer-revision", m["revision"],
            "--served-model-name", model, "--dtype", m["dtype"], "--max-model-len", str(MAX_MODEL_LEN),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--tensor-parallel-size", str(args.tp), "--seed", "0",
            "--host", "127.0.0.1", "--port", str(args.port)]


def pip_freeze(py):
    """Package list of the vLLM interpreter (uv venvs have no pip module)."""
    if not py:
        return None
    cmd = ["uv", "pip", "freeze", "--python", py] if shutil.which("uv") else [py, "-m", "pip", "freeze"]
    return sh(*cmd, timeout=120).splitlines()


def vllm_running():
    """PIDs whose argv is `<...>/vllm serve ...` (or the self-test stand-in). Exact argv match, so a
    shell or log viewer whose command line merely mentions "vllm serve" does not count."""
    pids = []
    for d in Path("/proc").glob("[0-9]*"):
        try:
            argv = (d / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        names = [os.path.basename(x.decode("utf-8", "replace")) for x in argv]
        if int(d.name) != os.getpid() and any(n in ("vllm", "_fake-vllm") and i + 1 < len(names)
                                               and names[i + 1] == "serve" for i, n in enumerate(names)):
            pids.append(int(d.name))
    return pids


def port_in_use(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


class VLLMServer:
    """One `vllm serve` process per model; stopped (whole process group) on exit."""

    def __init__(self, args, model, run_dir):
        self.args, self.model, self.run_dir = args, model, Path(run_dir)
        self.base = f"http://127.0.0.1:{args.port}"
        self.proc = None

    def __enter__(self):
        if vllm_running():
            raise RuntimeError(f"a 'vllm serve' process is already running ({vllm_running()}); stop it first")
        if port_in_use(self.args.port):
            raise RuntimeError(f"port {self.args.port} is in use")
        sdir = self.run_dir / "servers"
        sdir.mkdir(parents=True, exist_ok=True)
        stamp = f"{self.model}-{datetime.datetime.now(datetime.timezone.utc):%Y%m%dT%H%M%SZ}"
        self.log_path = sdir / f"{stamp}.log"
        cmd = serve_cmd(self.args, self.model)
        py = vllm_python(self.args.vllm_bin)
        versions = sh(py, "-c", "import json, vllm, torch, transformers; print(json.dumps({'vllm': vllm.__version__, "
                      "'torch': torch.__version__, 'cuda': torch.version.cuda, "
                      "'transformers': transformers.__version__}))", timeout=300) if py else None
        manifest = {"launched_utc": utcnow(), "model": self.model, **MODELS[self.model], "command": cmd,
                    "versions": versions, "gpus": sh("nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                                     "--format=csv,noheader"),
                    "hostname": socket.gethostname(), "git": git_head(), "log": str(self.log_path),
                    "pip_freeze": pip_freeze(py)}
        (sdir / f"{stamp}.json").write_text(json.dumps(manifest, indent=2) + "\n")
        self.log = open(self.log_path, "w")
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        t0 = time.time()
        while time.time() - t0 < self.args.startup_timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"vllm exited with {self.proc.returncode}; see {self.log_path}:\n"
                                   + "".join(open(self.log_path).readlines()[-30:]))
            try:
                if http("GET", f"{self.base}/health", timeout=5)[0] == 200:
                    print(f"[{self.model}] server ready after {time.time() - t0:.0f}s", flush=True)
                    return self
            except Exception:
                pass
            time.sleep(5)
        self.__exit__(None, None, None)
        raise RuntimeError(f"vllm not healthy after {self.args.startup_timeout}s; see {self.log_path}")

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            for sig, wait in ((signal.SIGINT, 90), (signal.SIGTERM, 30), (signal.SIGKILL, 30)):
                try:
                    os.killpg(self.proc.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    self.proc.wait(wait)
                    break
                except subprocess.TimeoutExpired:
                    continue
        if getattr(self, "log", None):
            self.log.close()
        return False


def generate_one(base, job, max_tokens, timeout, run_id, attempt):
    q = job["q"]
    body = build_request(job["model"], q["prompt"], job["setting"], job["seed"], max_tokens)
    s = SETTINGS[job["setting"]]
    rec = {"key": job["key"], "dataset": q["dataset"], "index": q["index"], "model": job["model"],
           "hf_repo": MODELS[job["model"]]["repo"], "revision": MODELS[job["model"]]["revision"],
           "mode": s["mode"], "setting": job["setting"], "sampling": s["sampling"], "seed": job["seed"],
           "prompt_sha256": q["prompt_sha256"], "request": request_record(body), "run_id": run_id,
           "attempt": attempt, "status": "error", "raw_output": None, "request_sent_utc": utcnow()}
    t0 = time.time()
    try:
        st, text = http("POST", f"{base}/v1/chat/completions", body, timeout=timeout)
        rec["http_status"] = st
        resp = json.loads(text)
        choices = resp.get("choices") or []
        if len(choices) != 1:
            raise ValueError(f"{len(choices)} choices")
        ch, usage = choices[0], resp.get("usage") or {}
        msg = ch.get("message") or {}
        rec.update({"response_id": resp.get("id"), "response_model": resp.get("model"),
                    "finish_reason": ch.get("finish_reason"), "stop_reason": ch.get("stop_reason"),
                    "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                    "reasoning_field_present": any(msg.get(k) for k in ("reasoning_content", "reasoning"))})
        if resp.get("model") != job["model"]:
            raise ValueError(f"response model {resp.get('model')!r}")
        if not isinstance(msg.get("content"), str):
            raise ValueError("message.content missing")
        if ch.get("finish_reason") not in ("stop", "length"):
            raise ValueError(f"finish_reason {ch.get('finish_reason')!r}")
        if not isinstance(rec["prompt_tokens"], int) or not isinstance(rec["completion_tokens"], int):
            raise ValueError("usage token counts missing")
        rec["raw_output"] = msg["content"]
        rec["truncated"] = ch.get("finish_reason") == "length"
        rec["status"] = "ok"
    except urllib.error.HTTPError as e:
        rec["http_status"] = e.code
        rec["error"] = f"HTTPError {e.code}: {e.read().decode('utf-8', 'replace')[:2000]}"
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["response_received_utc"] = utcnow()
    rec["latency_s"] = round(time.time() - t0, 3)
    return rec


def generation_state(path):
    """(ok keys, attempts per key). A key is done once it has one ok record."""
    ok, attempts = set(), {}
    for r in read_jsonl(path):
        attempts[r["key"]] = attempts.get(r["key"], 0) + 1
        if r["status"] == "ok":
            ok.add(r["key"])
    return ok, attempts


def stale_requests(path, jobs, max_tokens):
    """Planned keys whose existing ok record was sent with a different request than the one that
    would be sent now (e.g. generated before amendment A1). Resume would silently reuse them."""
    want = {j["key"]: request_record(build_request(j["model"], j["q"]["prompt"], j["setting"], j["seed"],
                                                   max_tokens)) for j in jobs}
    return sorted({r["key"] for r in read_jsonl(path)
                   if r["status"] == "ok" and r["key"] in want and r.get("request") != want[r["key"]]})


def run_jobs(base, jobs, out_path, max_tokens, args, run_id):
    """Send every job without an ok record (resumable). Returns (n_ok, n_err, n_skipped)."""
    n_ok = n_err = 0
    for p in range(1 + args.retry_passes):
        repair_tail(out_path)
        ok, attempts = generation_state(out_path)
        todo = [j for j in jobs if j["key"] not in ok and attempts.get(j["key"], 0) < args.max_attempts]
        if not todo:
            break
        if p:
            print(f"retry pass {p}: {len(todo)} errored requests", flush=True)
            time.sleep(args.retry_wait)
        # Longest expected outputs first (thinking; code and math), so the tail is short.
        rank = {"livecodebench": 0, "livemathbench": 1, "gpqa": 2, "mmlupro": 3}
        todo.sort(key=lambda j: (SETTINGS[j["setting"]]["mode"] != "thinking", rank[j["q"]["dataset"]]))
        with ThreadPoolExecutor(args.concurrency) as pool:
            futs = [pool.submit(generate_one, base, j, max_tokens, args.request_timeout, run_id,
                                attempts.get(j["key"], 0) + 1) for j in todo]
            for i, fut in enumerate(as_completed(futs), 1):
                rec = fut.result()
                append_jsonl(out_path, rec)
                n_ok += rec["status"] == "ok"
                n_err += rec["status"] != "ok"
                if rec["status"] != "ok" or i % 25 == 0 or i == len(futs):
                    print(f"  [{rec['model']} {i}/{len(futs)}] {rec['status']} {rec['key']} "
                          f"{rec.get('finish_reason')} {rec.get('completion_tokens')} {rec.get('error', '')[:200]}",
                          flush=True)
    ok, attempts = generation_state(out_path)
    stuck = [j["key"] for j in jobs if j["key"] not in ok]
    return n_ok, n_err, stuck


def generate(args, run, sample, models=None, settings=None, max_tokens="spec"):
    spec = RUN_SPECS[run]
    run_dir = Path(args.runs_dir) / run
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "generations.jsonl"
    mt = spec["max_tokens"] if max_tokens == "spec" else max_tokens
    run_id = f"{run}-{datetime.datetime.now(datetime.timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
    all_stuck = []
    for model in models or spec["models"]:
        jobs = plan_jobs(spec, sample, [model], settings)
        stale = stale_requests(out, jobs, mt)
        if stale:
            raise RuntimeError(f"{len(stale)} existing ok records in {out} were sent with a different request "
                               f"than the current settings (first: {stale[:3]}); nothing sent. Use a fresh "
                               f"--runs-dir; do not mix pre- and post-amendment generations.")
        ok, _ = generation_state(out)
        pending = [j for j in jobs if j["key"] not in ok]
        print(f"[{model}] {len(jobs)} jobs, {len(jobs) - len(pending)} already ok, {len(pending)} to send", flush=True)
        if not pending:
            continue
        inv = {"event": "start", "run": run, "run_id": run_id, "model": model, "utc": utcnow(),
               "max_tokens": mt, "concurrency": args.concurrency, "n_pending": len(pending),
               "repo": git_head(), "argv": sys.argv, "accepted_limitations": getattr(args, "accept", None)}
        append_jsonl(run_dir / "invocations.jsonl", inv)
        if args.external_server:
            base = args.external_server.rstrip("/")
            ctx = None
        else:
            ctx = VLLMServer(args, model, run_dir)
            base = ctx.base
        try:
            if ctx:
                ctx.__enter__()
            srv, errors = server_checks(base, model)
            if errors:
                raise RuntimeError("server checks failed; nothing sent: " + "; ".join(errors))
            n_ok, n_err, stuck = run_jobs(base, jobs, out, mt, args, run_id)
            if run == "smoke" and args.smoke_probe:
                smoke_probes(base, model, run_dir, sample)
        finally:
            if ctx:
                ctx.__exit__(None, None, None)
        append_jsonl(run_dir / "invocations.jsonl", {"event": "end", "run_id": run_id, "model": model,
                                                      "utc": utcnow(), "server": srv, "n_ok": n_ok,
                                                      "n_error": n_err, "unfinished": stuck})
        print(f"[{model}] sent: {n_ok} ok, {n_err} error; unfinished keys: {len(stuck)}", flush=True)
        all_stuck += stuck
    return all_stuck


# --------------------------------------------------------------------------- #
# Scoring                                                                     #
# --------------------------------------------------------------------------- #
def score_run(run_dir, sample, workers=8, skip_lcb=False):
    run_dir = Path(run_dir)
    gens = [r for r in read_jsonl(run_dir / "generations.jsonl") if r["status"] == "ok"]
    first = {}
    for r in gens:
        first.setdefault(r["key"], r)   # resumption never re-requests an ok key; first one wins
    out = run_dir / "scores.jsonl"
    repair_tail(out)
    done = {(r["key"], r["generation_sha256"]) for r in read_jsonl(out)}
    qs = {q["key"]: q for q in sample}
    todo = []
    for key, g in first.items():
        gsha = sha256_text(g["raw_output"])
        if (key, gsha) in done or (skip_lcb and g["dataset"] == "livecodebench"):
            continue
        todo.append((g, gsha))
    if not todo:
        print(f"{run_dir.name}: nothing to score ({len(done)} scored)")
        return
    lcb_pos = sorted({g["index"] - 1 for g, _ in todo if g["dataset"] == "livecodebench"})
    scorer = Scorer(lcb_pos)

    def one(item):
        g, gsha = item
        q = qs[qkey(g["dataset"], g["index"])]
        rec = {k: g[k] for k in ("key", "dataset", "index", "model", "mode", "setting", "sampling", "seed",
                                 "prompt_tokens", "completion_tokens", "finish_reason", "truncated")}
        rec.update(generation_sha256=gsha, think_open="<think>" in g["raw_output"],
                   think_closed="</think>" in g["raw_output"], scored_utc=utcnow())
        try:
            rec["score"], pred, gt = scorer.score(q, g["raw_output"])
            rec["prediction"] = pred if not isinstance(pred, str) or len(pred) < 4000 else pred[:4000] + "...[cut]"
            rec["ground_truth"] = gt
        except Exception as e:
            rec["score"], rec["eval_error"] = None, f"{type(e).__name__}: {e}"
        return rec

    n = 0
    with ThreadPoolExecutor(workers) as pool:
        for rec in pool.map(one, todo):
            append_jsonl(out, rec)
            n += 1
            if n % 100 == 0 or n == len(todo):
                print(f"  scored {n}/{len(todo)}", flush=True)
    print(f"{run_dir.name}: scored {n} new generations -> {out}")


def latest_scores(run_dir):
    """Score per job key: the record for the generation that is (still) the first ok one."""
    gens = {}
    for r in read_jsonl(Path(run_dir) / "generations.jsonl"):
        if r["status"] == "ok":
            gens.setdefault(r["key"], sha256_text(r["raw_output"]))
    out = {}
    for r in read_jsonl(Path(run_dir) / "scores.jsonl"):
        if gens.get(r["key"]) == r["generation_sha256"]:
            out[r["key"]] = r
    return out


# --------------------------------------------------------------------------- #
# Parity and smoke reports                                                    #
# --------------------------------------------------------------------------- #
def parity_report(run_dir, sample):
    run_dir = Path(run_dir)
    scores = latest_scores(run_dir)
    jobs = plan_jobs(RUN_SPECS["parity"], sample)
    missing = [j["key"] for j in jobs if j["key"] not in scores or scores[j["key"]]["score"] is None]
    res = {"created_utc": utcnow(), "model": "Qwen3-8B", "setting": SETTINGS["parity"],
           "vllm_version": VLLM_VERSION, "rule": (f"per benchmark: new accuracy inside the Wilson 95% CI of the "
                                                  f"cached accuracy on the same queries; per-query agreement >= "
                                                  f"{PARITY_AGREEMENT_MIN:.0%} on {PARITY_AGREEMENT_BENCHES}"),
           "missing": missing, "benchmarks": {}}
    ok = not missing
    for d, _, _ in BENCHMARKS:
        rows = [(q, scores.get(job_key("Qwen3-8B", "parity", 0, d, q["index"]))) for q in sample if q["dataset"] == d]
        rows = [(q, s) for q, s in rows if s and s["score"] is not None]
        n = len(rows)
        if not n:
            ok = False
            continue
        k_ref = sum(q["ref_qwen3_8b"]["score"] for q, _ in rows)
        k_new = sum(s["score"] for _, s in rows)
        lo, hi = wilson(k_ref, n)
        agree = sum(q["ref_qwen3_8b"]["score"] == s["score"] for q, s in rows) / n
        in_ci = lo <= k_new / n <= hi
        agree_ok = agree >= PARITY_AGREEMENT_MIN if d in PARITY_AGREEMENT_BENCHES else None
        ok &= in_ci and agree_ok is not False
        res["benchmarks"][d] = {
            "n": n, "cached_acc": k_ref / n, "cached_ci95": [lo, hi], "new_acc": k_new / n, "in_ci": in_ci,
            "agreement": agree, "agreement_ok": agree_ok,
            "truncated_rate": sum(bool(s["truncated"]) for _, s in rows) / n,
            "new_mean_completion_tokens": sum(s["completion_tokens"] for _, s in rows) / n,
            "cached_mean_completion_tokens": sum(q["ref_qwen3_8b"]["completion_tokens"] for q, _ in rows) / n}
    res["verdict"] = "INCOMPLETE" if missing else ("PASS" if ok else "FAIL")
    (run_dir / "parity.json").write_text(json.dumps(res, indent=2) + "\n")
    lines = [f"# S1 harness parity: {res['verdict']}", "",
             f"Qwen3-8B bf16, vLLM {VLLM_VERSION}, cached-run settings {SETTINGS['parity']}; no max_tokens "
             f"(cap = {MAX_MODEL_LEN} total). Rule: {res['rule']}. Validation only; decides nothing about SPLIT.", "",
             "| benchmark | n | cached acc | cached 95% CI | new acc | in CI | agreement | agreement ok | "
             "new trunc @32k | mean tokens new / cached |", "|---|---|---|---|---|---|---|---|---|---|"]
    for d, b in res["benchmarks"].items():
        lines.append(f"| {d} | {b['n']} | {b['cached_acc']:.3f} | [{b['cached_ci95'][0]:.3f}, {b['cached_ci95'][1]:.3f}] "
                     f"| {b['new_acc']:.3f} | {b['in_ci']} | {b['agreement']:.3f} | {b['agreement_ok']} | "
                     f"{b['truncated_rate']:.3f} | {b['new_mean_completion_tokens']:.0f} / "
                     f"{b['cached_mean_completion_tokens']:.0f} |")
    if missing:
        lines += ["", f"Missing or unscored: {len(missing)} jobs (first: {missing[:5]})"]
    lines += ["", "If FAIL: fix the harness (prompt, chat template, answer extraction) before trusting S1, or record "
              "the residual gap as a stated limitation (`run --accept-parity-limitation`). The cached run's engine and "
              "sampling are reconstructions, not recorded facts."]
    (run_dir / "parity.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return res


def smoke_probes(base, model, run_dir, sample):
    """Seed-determinism probe: the same short request twice with the same seed, concurrently and
    then alone. Informational (vLLM does not guarantee bit-identical batched sampling)."""
    q = next(q for q in sample if q["dataset"] == "gpqa")
    body = build_request(model, q["prompt"], "thinking", 0, 64)
    outs = []
    with ThreadPoolExecutor(2) as pool:
        for st, text in pool.map(lambda _: http("POST", f"{base}/v1/chat/completions", body, 600), range(2)):
            outs.append(json.loads(text)["choices"][0]["message"]["content"])
    st, text = http("POST", f"{base}/v1/chat/completions", body, 600)
    outs.append(json.loads(text)["choices"][0]["message"]["content"])
    probe = {"utc": utcnow(), "model": model, "same_seed_identical": len(set(outs)) == 1,
             "n_distinct": len(set(outs))}
    (Path(run_dir) / "seed_probe.json").write_text(json.dumps(probe, indent=2) + "\n")
    print(f"seed probe: {probe}")


def smoke_report(run_dir, sample, model, skip_lcb=False):
    run_dir = Path(run_dir)
    spec = RUN_SPECS["smoke"]
    jobs = plan_jobs(spec, sample, [model])
    gens = {}
    for r in read_jsonl(run_dir / "generations.jsonl"):
        if r["status"] == "ok":
            gens.setdefault(r["key"], r)
    scores = latest_scores(run_dir)
    checks = {}
    checks["all_jobs_ok"] = all(j["key"] in gens for j in jobs)
    checks["all_scored"] = all(j["key"] in scores and scores[j["key"]]["score"] is not None for j in jobs
                               if not (skip_lcb and j["q"]["dataset"] == "livecodebench"))
    think = [g for k, g in gens.items() if g["mode"] == "thinking" and g["model"] == model]
    nothink = [g for k, g in gens.items() if g["mode"] == "nonthinking" and g["model"] == model]
    checks["thinking_outputs_open_think"] = bool(think) and all(g["raw_output"].lstrip().startswith("<think>") for g in think)
    checks["nonthinking_outputs_have_no_think"] = bool(nothink) and all("<think>" not in g["raw_output"] for g in nothink)
    checks["lcb_scored"] = None if skip_lcb else any(s["dataset"] == "livecodebench" and s["score"] is not None
                                                     for s in scores.values())
    checks["max_tokens_respected"] = all(g["completion_tokens"] <= spec["max_tokens"] for g in gens.values())
    checks["no_duplicate_ok_records"] = (sum(1 for r in read_jsonl(run_dir / "generations.jsonl")
                                             if r["status"] == "ok") == len(gens))
    probe = run_dir / "seed_probe.json"
    res = {"created_utc": utcnow(), "model": model, "checks": checks,
           "seed_probe": json.loads(probe.read_text()) if probe.exists() else None,
           "verdict": "PASS" if all(v is not False for v in checks.values()) else "FAIL",
           "scores": {k: s["score"] for k, s in sorted(scores.items())}}
    (run_dir / "smoke.json").write_text(json.dumps(res, indent=2) + "\n")
    for k, v in checks.items():
        print(f"  {'SKIP' if v is None else 'OK  ' if v else 'FAIL'} {k}")
    print(f"smoke verdict: {res['verdict']} (seed probe: {res['seed_probe']})")
    return res


# --------------------------------------------------------------------------- #
# Commands                                                                    #
# --------------------------------------------------------------------------- #
def cmd_check(args):
    """Offline checks. Hard failures exit non-zero; skipped checks are reported as SKIP."""
    results = []

    def rep(name, ok, detail=""):
        results.append((name, ok))
        print(f"{'OK  ' if ok is True else 'SKIP' if ok is None else 'FAIL'} {name} {detail}")

    sample = load_sample(verify=False)
    digest = sha256_file(SAMPLE_JSONL)
    rep("sample sha256 pinned", SAMPLE_SHA256 == digest, f"{digest} vs pinned {SAMPLE_SHA256}")
    counts = {d: sum(q["dataset"] == d for q in sample) for d in N_PER_BENCH}
    rep("per-benchmark counts 60/60/40/40", counts == N_PER_BENCH, str(counts))
    rep("keys unique", len({q["key"] for q in sample}) == len(sample))
    rep("prompts unique", len({q["prompt"] for q in sample}) == len(sample))
    rep("prompt hashes", all(sha256_text(q["prompt"]) == q["prompt_sha256"] for q in sample))
    rep("ranks are the permutation prefix", all(
        sorted(q["rank"] for q in sample if q["dataset"] == d) == list(range(n)) for d, n in N_PER_BENCH.items()))

    have_cached = all(cached_path(d) for d in CACHED_FILES)
    if have_cached:
        rows, _ = draw_sample()
        rep("sample re-draws byte-identically", sample_bytes(rows) == SAMPLE_JSONL.read_bytes())
        rows2, _ = draw_sample(scale=2)
        rep("400 extension contains the 200", {q["key"] for q in sample} <= {r["key"] for r in rows2}
            and len(rows2) == 2 * len(sample))
        # Same index -> same prompt and answer in every cached model file (stable keys across the pool).
        bad, n_files = 0, 0
        for d in CACHED_FILES:
            if d == "livecodebench":
                continue
            base = cached_path(d).parents[1]
            want = {q["index"]: q for q in sample if q["dataset"] == d}
            for f in sorted(base.glob("*/*.json")):
                try:
                    recs = {int(r["index"]): r for r in json.loads(f.read_text(encoding="utf-8"))["records"]}
                except Exception:
                    continue
                n_files += 1
                bad += sum(i in recs and (recs[i]["prompt"] != q["prompt"] or recs[i]["ground_truth"] != q["answer"])
                           for i, q in want.items())
        rep("prompt + answer identical across cached model files", bad == 0, f"{n_files} files, {bad} mismatches")
        # The unchanged evaluators reproduce every cached Qwen3-8B score from the stored answer.
        scorer = Scorer()
        mism, n = [], 0
        for d in ("mmlupro", "gpqa", "livemathbench"):
            recs = {int(r["index"]): r for r in load_cached(d)}
            for q in (q for q in sample if q["dataset"] == d):
                r = recs[q["index"]]
                score, _, _ = scorer.score(q, r["raw_output"])
                n += 1
                if score != r["score"]:
                    mism.append(q["key"])
        rep("evaluator reproduces cached scores (MMLU-Pro/GPQA/LiveMathBench)", not mism, f"{n - len(mism)}/{n}")
    else:
        rep("cached-output checks", None, "(no cached files; run fetch-data)")

    lcb = verify_lcb(hash_files=args.hash_lcb)
    if all(lcb.values()):
        want = {q["index"] - 1: q for q in sample if q["dataset"] == "livecodebench"}
        cached = {int(r["index"]) - 1: r["prompt"] for r in load_cached("livecodebench")} if have_cached else {}
        n_rows, bad_s, bad_all = 0, 0, 0
        for pos, row in iter_lcb_rows():
            n_rows += 1
            if pos in want or pos in cached:
                p = rebuilt_lcb_prompt(row)
                bad_s += pos in want and p != want[pos]["prompt"]
                bad_all += pos in cached and p != cached[pos]
        rep("LiveCodeBench rows", n_rows == CACHED_COUNTS["livecodebench"], str(n_rows))
        rep("LiveCodeBench rebuilt prompts == sample prompts", bad_s == 0, f"{bad_s} mismatches of {len(want)}")
        if cached:
            rep("LiveCodeBench rebuilt prompts == all cached prompts", bad_all == 0, f"{bad_all} of {len(cached)}")
        if args.execute_lcb:
            if not have_cached:
                rep("LiveCodeBench evaluator reproduction", None, "(needs cached outputs)")
            else:
                recs = {int(r["index"]): r for r in load_cached("livecodebench")}
                qs = [q for q in sample if q["dataset"] == "livecodebench"]
                sc = Scorer([q["index"] - 1 for q in qs])
                with ThreadPoolExecutor(args.workers) as pool:
                    got = list(pool.map(lambda q: sc.score(q, recs[q["index"]]["raw_output"])[0], qs))
                agree = sum(g == recs[q["index"]]["score"] for g, q in zip(got, qs))
                # Execution timing can differ by host; reported, not a gate (PREREG §4).
                print(f"INFO LiveCodeBench evaluator reproduction: {agree}/{len(qs)} cached scores reproduced")
        else:
            rep("LiveCodeBench evaluator reproduction", None, "(executes model code; pod only: --execute-lcb)")
    else:
        rep("LiveCodeBench data", None, f"(missing/unverified in {lcb_dir()}: "
                                        f"{[k for k, v in lcb.items() if not v]}; run fetch-data)")

    # Every request the three runs would send is buildable and fits the context.
    too_long, bad_sampling = [], []
    for run, spec in RUN_SPECS.items():
        for j in plan_jobs(spec, sample):
            body = build_request(j["model"], j["q"]["prompt"], j["setting"], j["seed"], spec["max_tokens"])
            json.dumps(body)
            s = SETTINGS[j["setting"]]
            # Amendment A1: 1.5 on every AWQ request, absent on every bf16 request; all other
            # sampling fields exactly the frozen per-setting values.
            pp_ok = (body.get("presence_penalty") == 1.5) if j["model"] in AWQ_MODELS else ("presence_penalty" not in body)
            rest_ok = all(body[k] == s[k] for k in ("temperature", "top_p", "top_k", "min_p")) and \
                body["chat_template_kwargs"] == {"enable_thinking": s["enable_thinking"]} and \
                body["seed"] == j["seed"] and body.get("max_tokens") == spec["max_tokens"]
            if not (pp_ok and rest_ok):
                bad_sampling.append((run, j["key"]))
            need = j["q"]["ref_qwen3_8b"]["prompt_tokens"] + TEMPLATE_MARGIN_TOKENS + (spec["max_tokens"] or 0)
            if need > MAX_MODEL_LEN:
                too_long.append((run, j["key"]))
    n_s1 = len(plan_jobs(RUN_SPECS["s1"], sample))
    rep("request plan", n_s1 == 5 * 2 * 2 * 200, f"s1 jobs {n_s1}")
    rep("prompt + max_tokens fits max_model_len", not too_long, f"{len(too_long)} too long")
    n_awq = sum(j["model"] in AWQ_MODELS for spec in RUN_SPECS.values() for j in plan_jobs(spec, sample))
    rep("sampling: presence_penalty 1.5 on AWQ only (A1), other fields frozen", not bad_sampling,
        f"{n_awq} AWQ requests; {len(bad_sampling)} mismatched")
    failed = [n for n, ok in results if ok is False]
    print(f"\n{sum(ok is True for _, ok in results)} OK, {sum(ok is None for _, ok in results)} SKIP, {len(failed)} FAIL")
    sys.exit(1 if failed else 0)


def cmd_preflight(args):
    """Pod checks before any download or run. Writes runs/preflight.json."""
    res, errors = {"utc": utcnow(), "hostname": socket.gethostname(), "repo": git_head()}, []
    q = sh("nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader,nounits")
    res["nvidia_smi"] = q
    gpus = []
    for line in q.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4 and parts[1].isdigit():
            gpus.append({"name": parts[0], "mem_mib": int(parts[1]), "used_mib": int(parts[2]), "driver": parts[3]})
    res["gpus"] = gpus
    if len(gpus) < args.tp:
        errors.append(f"need {args.tp} NVIDIA GPU(s), found {len(gpus)}")
    else:
        total_gb = sum(g["mem_mib"] for g in gpus[: args.tp]) * 1.048576e6 / 1e9
        kv_gb = total_gb * args.gpu_memory_utilization - MODELS[REFERENCE_MODEL]["weights_gb"] - 4.0 * args.tp
        one_seq_gb = 64 * 8 * 128 * 2 * 2 * MAX_MODEL_LEN / 1e9   # Qwen3-32B bf16 KV for one full sequence
        res["kv_headroom_gb_32b"] = round(kv_gb, 1)
        if kv_gb < one_seq_gb:
            errors.append(f"Qwen3-32B bf16 leaves ~{kv_gb:.1f} GB KV (< {one_seq_gb:.1f} GB for one {MAX_MODEL_LEN}-token "
                          f"sequence): use 1x H200 141GB or --tp 2 on 2x80GB")
        elif kv_gb < 30:
            print(f"WARN only ~{kv_gb:.1f} GB KV for Qwen3-32B: few concurrent long sequences, slow run")
        if any(g["used_mib"] > 2048 for g in gpus[: args.tp]):
            errors.append("GPU memory already in use (another process?)")
        if any("B200" in g["name"] or "B100" in g["name"] or "RTX 50" in g["name"] for g in gpus):
            errors.append("Blackwell GPU: vLLM 0.8.5.post1 wheels (CUDA 12.4) do not support it")
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    missing_gb = 0.0
    for name, m in MODELS.items():
        snap = hf_home / "hub" / f"models--{m['repo'].replace('/', '--')}" / "snapshots" / m["revision"]
        if not (snap.exists() and any(snap.glob("*.safetensors"))):
            missing_gb += m["weights_gb"]
    lcb_missing = sum(s for n, (s, _) in LCB_FILES.items() if not verify_lcb(False)[n]) / 1e9
    need = missing_gb + lcb_missing + (1.3 if not all(cached_path(d) for d in CACHED_FILES) else 0) + 20
    for label, p in (("HF_HOME", hf_home), ("S1_DATA_DIR", DATA_DIR), ("S1_RUNS_DIR", Path(args.runs_dir))):
        p.mkdir(parents=True, exist_ok=True)
        res[f"free_gb_{label}"] = round(shutil.disk_usage(p).free / 1e9, 1)
    res.update(hf_home=str(hf_home), weights_to_download_gb=round(missing_gb, 1), need_free_gb=round(need, 1))
    if shutil.disk_usage(hf_home).free / 1e9 < need:
        errors.append(f"free disk at HF_HOME {res['free_gb_HF_HOME']} GB < needed {need:.0f} GB")
    res["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    if res["ram_gb"] < 32:
        errors.append(f"RAM {res['ram_gb']} GB < 32 GB")
    py = vllm_python(args.vllm_bin)
    res["vllm_python"] = py
    got = sh(py, "-c", "import vllm; print(vllm.__version__)", timeout=300) if py else "not found"
    res["vllm_version"] = got
    if got != VLLM_VERSION:
        errors.append(f"vllm {VLLM_VERSION} required at {args.vllm_bin}, found {got!r}")
    try:
        Scorer()
        res["evaluators_import"] = True
    except Exception as e:
        errors.append(f"evaluators do not import in this interpreter: {type(e).__name__}: {e}")
    try:
        res["hf_reachable"] = http("GET", "https://huggingface.co/api/models/Qwen/Qwen3-8B", timeout=20)[0] == 200
    except Exception as e:
        res["hf_reachable"] = f"{type(e).__name__}: {e}"
    if vllm_running():
        errors.append("a 'vllm serve' process is running")
    res["errors"], res["verdict"] = errors, "FAIL" if errors else "PASS"
    Path(args.runs_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.runs_dir) / "preflight.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("repo",)}, indent=2))
    sys.exit(1 if errors else 0)


def gate(args, run):
    """Full run gates: smoke PASS and parity PASS (or an explicitly recorded limitation)."""
    rd = Path(args.runs_dir)
    problems = []
    smoke = rd / "smoke" / "smoke.json"
    if not (smoke.exists() and json.loads(smoke.read_text())["verdict"] == "PASS"):
        problems.append(f"smoke test has not passed ({smoke})")
    parity = rd / "parity" / "parity.json"
    pv = json.loads(parity.read_text())["verdict"] if parity.exists() else "MISSING"
    if pv != "PASS" and not args.accept:
        problems.append(f"harness parity verdict is {pv} ({parity}); fix the harness or pass "
                        f"--accept-parity-limitation 'reason'")
    if problems and not args.no_gates:
        sys.exit("refusing to start the full run:\n  " + "\n  ".join(problems))


def cmd_smoke(args):
    sample = load_sample()
    model = args.model or RUN_SPECS["smoke"]["models"][0]
    args.concurrency = args.concurrency or RUN_SPECS["smoke"]["concurrency"]
    args.smoke_probe = not args.external_server or args.probe_external
    stuck = generate(args, "smoke", sample, models=[model])
    # A second pass must find nothing to send (resume path).
    args.smoke_probe = False
    ok_before = len(generation_state(Path(args.runs_dir) / "smoke" / "generations.jsonl")[0])
    if not stuck:
        generate(args, "smoke", sample, models=[model])
    ok_after = len(generation_state(Path(args.runs_dir) / "smoke" / "generations.jsonl")[0])
    print(f"resume pass: ok records {ok_before} -> {ok_after} (must be unchanged)")
    score_run(Path(args.runs_dir) / "smoke", sample, args.workers, skip_lcb=args.skip_lcb)
    res = smoke_report(Path(args.runs_dir) / "smoke", sample, model, skip_lcb=args.skip_lcb)
    sys.exit(0 if res["verdict"] == "PASS" and ok_before == ok_after else 1)


def cmd_parity(args):
    sample = load_sample()
    args.concurrency = args.concurrency or RUN_SPECS["parity"]["concurrency"]
    args.smoke_probe = False
    generate(args, "parity", sample)
    score_run(Path(args.runs_dir) / "parity", sample, args.workers)
    res = parity_report(Path(args.runs_dir) / "parity", sample)
    sys.exit(0 if res["verdict"] == "PASS" else 1)


def cmd_run(args):
    sample = load_sample()
    gate(args, "s1")
    args.concurrency = args.concurrency or RUN_SPECS["s1"]["concurrency"]
    args.smoke_probe = False
    models = args.models.split(",") if args.models else None
    if models and set(models) - set(MODELS):
        sys.exit(f"unknown models {set(models) - set(MODELS)}")
    stuck = generate(args, "s1", sample, models=models)
    if args.score:
        score_run(Path(args.runs_dir) / "s1", sample, args.workers)
    print(f"S1 generation finished; unfinished jobs: {len(stuck)}")
    sys.exit(1 if stuck else 0)


def cmd_score(args):
    sample = load_sample()
    score_run(Path(args.runs_dir) / args.run, sample, args.workers, skip_lcb=args.skip_lcb)
    if args.run == "parity":
        parity_report(Path(args.runs_dir) / "parity", sample)


def cmd_status(args):
    sample = load_sample()
    for run, spec in RUN_SPECS.items():
        rd = Path(args.runs_dir) / run
        if not (rd / "generations.jsonl").exists():
            continue
        ok, attempts = generation_state(rd / "generations.jsonl")
        scores = latest_scores(rd)
        print(f"== {run}")
        for model in spec["models"]:
            for setting in spec["settings"]:
                for seed in spec["seeds"]:
                    js = [j for j in plan_jobs(spec, sample, [model], [setting]) if j["seed"] == seed]
                    n_ok = sum(j["key"] in ok for j in js)
                    n_sc = sum(j["key"] in scores for j in js)
                    n_err = sum(j["key"] not in ok and j["key"] in attempts for j in js)
                    if n_ok or n_err:
                        print(f"  {model:14} {setting:11} seed {seed}: {n_ok}/{len(js)} generated, "
                              f"{n_sc} scored, {n_err} errored-not-done")


# --------------------------------------------------------------------------- #
# Self-test (CPU only; fake OpenAI-compatible server)                         #
# --------------------------------------------------------------------------- #
def fake_server(model, delay=0.0, fail_first=1, port=0):
    """Minimal stand-in for `vllm serve` (/health, /version, /v1/models, /v1/chat/completions)."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    state = {"fails_left": fail_first, "lock": threading.Lock(), "bodies": []}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def reply(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                return self.reply(200, {})
            if self.path == "/version":
                return self.reply(200, {"version": VLLM_VERSION})
            if self.path == "/v1/models":
                return self.reply(200, {"data": [{"id": model, "root": model, "max_model_len": MAX_MODEL_LEN}]})
            self.reply(404, {})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["bodies"].append(body)
            with state["lock"]:
                if state["fails_left"] > 0:
                    state["fails_left"] -= 1
                    return self.reply(500, {"error": "injected failure"})
            time.sleep(delay)
            p = body["messages"][0]["content"]
            ans = ("\\boxed{4049}" if p.startswith("Solve the following math problem") else
                   "```python\nprint(1)\n```" if p.startswith("You are an expert Python programmer") else "Answer: A")
            think = body.get("chat_template_kwargs", {}).get("enable_thinking", True)
            text = ("<think>\nfake\n</think>\n\n" if think else "") + ans
            self.reply(200, {"id": "x", "model": body["model"],
                             "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                             "usage": {"prompt_tokens": 10, "completion_tokens": 7}})

    class Server(ThreadingHTTPServer):
        request_queue_size = 256   # the stdlib default (5) resets connections under concurrency 64
        def handle_error(self, request, client_address):
            if not isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
                super().handle_error(request, client_address)   # a killed client is expected in the self-test

    srv = Server(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, state


def cmd_fake_vllm(args):
    """Stand-in for `vllm serve` used by the self-test to exercise the real launcher path."""
    a = args.rest
    srv, _ = fake_server(a[a.index("--served-model-name") + 1], fail_first=0, port=int(a[a.index("--port") + 1]))
    signal.signal(signal.SIGINT, lambda *_: (srv.shutdown(), sys.exit(0)))
    while True:
        time.sleep(1)


def cmd_selftest(args):
    import summarize_s1
    n_ok = 0

    def check(name, cond):
        nonlocal n_ok
        if not cond:
            sys.exit(f"SELFTEST FAIL: {name}")
        n_ok += 1
        print(f"ok   {name}")

    sample = load_sample(verify=False)
    # Requests
    b = build_request("Qwen3-8B", "hi", "nonthinking", 1, MAX_NEW_TOKENS)
    check("nonthinking request", b["chat_template_kwargs"] == {"enable_thinking": False} and b["temperature"] == 0.7
          and b["top_p"] == 0.8 and b["top_k"] == 20 and b["min_p"] == 0.0 and b["max_tokens"] == 16384 and b["seed"] == 1)
    b = build_request("Qwen3-32B", "hi", "thinking", 0, MAX_NEW_TOKENS)
    check("thinking request", b["chat_template_kwargs"] == {"enable_thinking": True} and b["temperature"] == 0.6
          and b["top_p"] == 0.95 and "presence_penalty" not in b)
    b = build_request("Qwen3-8B", "hi", "parity", 0, None)
    check("parity request has no max_tokens", "max_tokens" not in b and b["temperature"] == 0.2 and b["top_p"] == 1.0
          and "presence_penalty" not in b)
    # Amendment A1: presence_penalty 1.5 on the three AWQ models only; nothing else differs.
    modes = ("thinking", "nonthinking")
    check("A1: the three AWQ models send presence_penalty 1.5",
          AWQ_MODELS == ["Qwen3-8B-AWQ", "Qwen3-14B-AWQ", "Qwen3-32B-AWQ"]
          and all(build_request(m, "hi", st, 0, MAX_NEW_TOKENS)["presence_penalty"] == 1.5
                  for m in AWQ_MODELS for st in modes))
    check("A1: bf16 models send no presence_penalty", all(
        "presence_penalty" not in build_request(m, "hi", st, 0, MAX_NEW_TOKENS)
        for m in ("Qwen3-32B", "Qwen3-8B") for st in SETTINGS))

    def strip(body, extra=()):
        return {k: v for k, v in body.items() if k not in ("model", *extra)}
    check("A1: AWQ request otherwise identical to the bf16 request", all(
        strip(build_request(m, "hi", st, 1, MAX_NEW_TOKENS), ("presence_penalty",))
        == strip(build_request("Qwen3-32B", "hi", st, 1, MAX_NEW_TOKENS)) for m in AWQ_MODELS for st in modes))
    keys = [j["key"] for j in plan_jobs(RUN_SPECS["s1"], sample)]
    check("s1 plan: 4000 unique keys", len(keys) == 4000 == len(set(keys)))
    check("smoke plan: 2 per benchmark x 2 modes", len(plan_jobs(RUN_SPECS["smoke"], sample)) == 16)
    check("job keys stable", job_key("Qwen3-8B", "thinking", 0, "gpqa", 7) == "Qwen3-8B|thinking|0|gpqa:7")
    # Sample procedure (synthetic frame): deterministic, prefix property, duplicate handling
    recs = [{"index": i, "prompt": f"p{i % 50}"} for i in range(1, 61)]
    frame, dropped = sampling_frame(recs)
    check("duplicate prompts dropped (later copies)", len(frame) == 50 and dropped == list(range(51, 61)))
    p1 = [r["index"] for r in permutation("gpqa", frame)]
    check("permutation deterministic", p1 == [r["index"] for r in permutation("gpqa", frame)])
    check("permutation depends on dataset", p1 != [r["index"] for r in permutation("mmlupro", frame)])
    check("wilson", all(abs(a - b) < 1e-3 for a, b in zip(wilson(30, 60), (0.3773, 0.6227))))
    # Torn-line repair
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "g.jsonl"
        p.write_text('{"a": 1}\n{"a": 2')
        check("torn final line ignored on read", read_jsonl(p) == [{"a": 1}])
        side = repair_tail(p)
        check("torn final line moved aside", p.read_text() == '{"a": 1}\n' and side.read_text() == '{"a": 2')

    # Interruption + resume against a fake server, through the real CLI in a subprocess.
    model = "Qwen3-8B-AWQ"
    srv, state = fake_server(model, delay=0.15, fail_first=1)
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    with tempfile.TemporaryDirectory() as td:
        base_cmd = [sys.executable, str(Path(__file__)), "smoke", "--runs-dir", td, "--external-server", url,
                    "--model", model, "--skip-lcb", "--retry-wait", "0", "--concurrency", "2"]
        gen = Path(td) / "smoke" / "generations.jsonl"
        proc = subprocess.Popen(base_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t0 = time.time()
        while time.time() - t0 < 60 and (not gen.exists() or gen.read_text().count("\n") < 5):
            time.sleep(0.05)
        proc.kill()
        proc.wait()
        n_before = len(generation_state(gen)[0])
        check("interrupted mid-run with partial results", 0 < n_before < 16)
        with open(gen, "a") as f:
            f.write('{"key": "torn')   # simulate a write cut by the kill
        r = subprocess.run(base_cmd, capture_output=True, text=True)
        ok, attempts = generation_state(gen)
        all_ok = [x for x in read_jsonl(gen) if x["status"] == "ok"]
        check("resume completes every job", len(ok) == 16 and r.returncode == 0)
        check("no job generated twice", len(all_ok) == 16)
        check("injected HTTP 500 recorded as error, then retried",
              sum(x["status"] == "error" for x in read_jsonl(gen)) == 1)
        check("torn line moved aside", any(Path(td, "smoke").glob("generations.jsonl.torn-*")))
        bodies = state["bodies"]
        check("server received vendor sampling + enable_thinking switch",
              {(b["chat_template_kwargs"]["enable_thinking"], b["temperature"]) for b in bodies}
              == {(True, 0.6), (False, 0.7)} and all(b["max_tokens"] == 1024 for b in bodies))
        check("A1: server received presence_penalty 1.5 on every AWQ request",
              all(b.get("presence_penalty") == 1.5 for b in bodies))
        sc = latest_scores(Path(td) / "smoke")
        nonlcb = [s for s in sc.values() if s["dataset"] != "livecodebench"]
        qs = {q["key"]: q for q in sample}

        def expected(s):   # the fake server answers "A" / \\boxed{4049}
            gt = qs[qkey(s["dataset"], s["index"])]["answer"]
            return float(gt.strip("$") == "4049") if s["dataset"] == "livemathbench" else float(gt == "A")
        exp = all(s["score"] == expected(s) for s in nonlcb)
        check("scored with unchanged evaluators (fake answers)", len(nonlcb) == 12 and exp)
        check("score records carry required fields", all(
            all(k in s for k in ("dataset", "index", "model", "mode", "seed", "score", "completion_tokens"))
            for s in nonlcb))
        # A server that serves another model is refused before anything is sent.
        n_posts = len(state["bodies"])
        r = subprocess.run([*base_cmd[:4], str(Path(td) / "other"), *base_cmd[5:6], url, "--model", "Qwen3-32B",
                            "--skip-lcb"], capture_output=True, text=True)
        check("wrong served model refused, nothing sent",
              r.returncode != 0 and "server checks failed" in r.stderr and len(state["bodies"]) == n_posts)
        # Resume over a record sent without the A1 penalty is refused before anything is sent.
        old = Path(td) / "pre_a1"
        (old / "smoke").mkdir(parents=True)
        rec = dict(all_ok[0], request={k: v for k, v in all_ok[0]["request"].items() if k != "presence_penalty"})
        append_jsonl(old / "smoke" / "generations.jsonl", rec)
        r = subprocess.run([*base_cmd[:4], str(old), *base_cmd[5:]], capture_output=True, text=True)
        check("A1: resume over pre-amendment records refused, nothing sent",
              r.returncode != 0 and "different request" in r.stderr and len(state["bodies"]) == n_posts)
    srv.shutdown()

    # Launcher path: a stand-in `vllm` executable is started, health-checked and stopped.
    with tempfile.TemporaryDirectory() as td:
        fake_bin = Path(td) / "vllm"
        fake_bin.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{Path(__file__).resolve()}" _fake-vllm "$@"\n')
        fake_bin.chmod(0o755)
        with socket.socket() as s_:
            s_.bind(("127.0.0.1", 0))
            port = s_.getsockname()[1]
        r = subprocess.run([sys.executable, str(Path(__file__)), "smoke", "--runs-dir", td, "--vllm-bin", str(fake_bin),
                            "--port", str(port), "--model", model, "--skip-lcb", "--startup-timeout", "60"],
                           capture_output=True, text=True)
        logs = list(Path(td, "smoke", "servers").glob("*.json"))
        man = json.loads(logs[0].read_text()) if logs else {}
        check("launcher: server started, used, stopped",
              r.returncode == 0 and "server ready" in r.stdout and not port_in_use(port) and not vllm_running())
        check("launcher: manifest records pinned repo/revision/command",
              len(logs) == 1 and man["repo"] == MODELS[model]["repo"] and man["revision"] == MODELS[model]["revision"]
              and man["command"][1:3] == ["serve", MODELS[model]["repo"]] and "--max-model-len" in man["command"])
        check("launcher: seed probe ran", Path(td, "smoke", "seed_probe.json").exists())
    check("summarize_s1 selftest", summarize_s1.selftest() is None)
    print(f"\nSELFTEST PASS ({n_ok} checks)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, gen=False):
        p.add_argument("--runs-dir", default=str(RUNS_DIR))
        p.add_argument("--workers", type=int, default=8, help="scoring threads")
        if gen:
            p.add_argument("--vllm-bin", default=os.environ.get("S1_VLLM_BIN", "vllm"))
            p.add_argument("--port", type=int, default=8000)
            p.add_argument("--tp", type=int, default=int(os.environ.get("S1_TP", 1)))
            p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
            p.add_argument("--startup-timeout", type=int, default=3600)
            p.add_argument("--request-timeout", type=float, default=10800.0)
            p.add_argument("--concurrency", type=int, default=None)
            p.add_argument("--retry-passes", type=int, default=2)
            p.add_argument("--retry-wait", type=float, default=30.0)
            p.add_argument("--max-attempts", type=int, default=5)
            p.add_argument("--external-server", default=None,
                           help="use an already running server for ONE model instead of launching vllm")
        return p

    sub.add_parser("sample")
    p = sub.add_parser("fetch-data")
    p.add_argument("--bench-tarball", default=None, help="local copy of bench-release.tar.gz")
    p.add_argument("--skip-lcb", action="store_true")
    p.add_argument("--skip-bench", action="store_true")
    p = common(sub.add_parser("check"))
    p.add_argument("--hash-lcb", action="store_true", help="sha256 the 4.5 GB LiveCodeBench files")
    p.add_argument("--execute-lcb", action="store_true", help="re-run cached LiveCodeBench code (pod only)")
    common(sub.add_parser("preflight"), gen=True)
    p = common(sub.add_parser("smoke"), gen=True)
    p.add_argument("--model", default=None, choices=list(MODELS))
    p.add_argument("--skip-lcb", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--probe-external", action="store_true", help=argparse.SUPPRESS)
    common(sub.add_parser("parity"), gen=True)
    p = common(sub.add_parser("run"), gen=True)
    p.add_argument("--models", default=None, help=f"comma list, default {','.join(RUN_ORDER)}")
    p.add_argument("--no-score", dest="score", action="store_false")
    p.add_argument("--accept-parity-limitation", dest="accept", default=None, metavar="REASON")
    p.add_argument("--no-gates", action="store_true", help=argparse.SUPPRESS)
    p = common(sub.add_parser("score"))
    p.add_argument("--run", default="s1", choices=list(RUN_SPECS))
    p.add_argument("--skip-lcb", action="store_true", help=argparse.SUPPRESS)
    common(sub.add_parser("status"))
    sub.add_parser("selftest")
    p = sub.add_parser("_fake-vllm")
    p.add_argument("rest", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    # tmux kill / SIGTERM / hang-up: unwind normally so a launched vLLM server is stopped.
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda signum, frame: sys.exit(128 + signum))
    {"sample": cmd_sample, "fetch-data": cmd_fetch_data, "check": cmd_check, "preflight": cmd_preflight,
     "smoke": cmd_smoke, "parity": cmd_parity, "run": cmd_run, "score": cmd_score, "status": cmd_status,
     "selftest": cmd_selftest, "_fake-vllm": cmd_fake_vllm}[args.cmd](args)


if __name__ == "__main__":
    main()
