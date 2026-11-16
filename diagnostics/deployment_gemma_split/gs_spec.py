"""GEMMA-SPLIT constants: every protocol parameter of PREREG_GEMMA_SPLIT.md, frozen by the tag gemma-split-prereg-v1.

Self-contained by design (PREREG §1.3): no gs_*.py file imports, reads or hashes a file of deployment_e0 or
deployment_e1opt. The Qwen3 E1-OPT experiment appears below only as a written description, used to record how the
Gemma deployment differs from it. Values only the lab can supply (the DGX Spark hostname, the checkpoint revision,
the split locations, the endpoint, the running configuration) are NOT here: Stage 0 captures them into capture/
and the freeze record carries them. This file and the PREREG must stay byte-identical to the tag (gs_freeze).
"""
from pathlib import Path

GS_DIR = Path(__file__).resolve().parent
NAME = "GEMMA-SPLIT"
PREREG_FILE = "PREREG_GEMMA_SPLIT.md"
PREREG_TAG = "gemma-split-prereg-v1"
PROTOCOL_VERSION = 1
DECISION_RULE_VERSION = "GS-r1"
ADAPTER_CONTRACT_VERSION = 1
RECORD_VERSION = 1
PROFILE_ROLES = ("target", "dev")          # only "target" outputs may enter a gate

# --------------------------------------------------------------------------- #
# Roles (fixed by the PI; verified at Stage 0, never derived from a measurement; PREREG §3)
# --------------------------------------------------------------------------- #
ROLES = ("edge", "server")
ROLE_MACHINE = {"edge": "DGX Spark (client / edge)", "server": "main lab server (backend / server)"}
SERVER_IDENTITY = {"hostname": "asus-ESC4000-E11", "gpu_name_contains": "RTX A6000", "gpu_count": 3,
                   "gpu_memory_mib_min": 45000}               # 48 GB nominal per GPU
EDGE_IDENTITY = {"arch": "aarch64", "gpu_name_contains": "GB10"}   # public DGX Spark facts, identity markers only
MODEL_DIR_NAME = "gemma-4-E4B-it"          # directory name given by the PI; path and revision are captured

# --------------------------------------------------------------------------- #
# Workload (PREREG §7). Texts are frozen in workload/*.json; token ids are produced at Stage 1 with the captured
# tokenizer and pinned in the freeze record.
# --------------------------------------------------------------------------- #
WORKLOAD_SOURCE = "diagnostics/deployment_s1/sample_200.jsonl"     # prompt texts only (read once by gs_workload build)
WORKLOAD_SOURCE_SHA256 = "2c4df49360f997c3e35593030da40111c2826f068f03b1401284a394d9716955"
CORR_PER_DATASET = {"gpqa": 9, "livecodebench": 6, "livemathbench": 6, "mmlupro": 9}   # 30 natural prompts
CORR_SEED = "gemma-split:correctness:20261116"
CORR_GEN_TOKENS = 1024                     # greedy tokens per natural prompt (stop at EOS)
COVERAGE_ITEMS = 4                         # long synthetic items so correctness covers every timed position
COVERAGE_PROMPT_TOKENS = 2048
COVERAGE_GEN_TOKENS = 256
TIMING_SEED = "gemma-split:timing:20261116"
TIMING_GROUPS = {"short": 256, "long": 2048}   # prompt tokens, BOS included
TIMING_GEN_TOKENS = 256
TIMING_WARMUP = 2                          # per group and block; never analysed
TIMING_MEASURED = 20                       # per group and block
POOL_JOIN = "\n\n"

# --------------------------------------------------------------------------- #
# Stage 3: correctness (PREREG §9). The rule logic follows E0 PREREG §4 (written anew here, not imported).
# --------------------------------------------------------------------------- #
CORR_TOPK = 32                             # reference top-K stored per teacher-forced position
TF_D_MARGIN_PP = 0.2                       # D_split <= max(floors) + 0.2 percentage points
TF_KL_FACTOR = 2.0                         # KL_split <= 2 x max(floors)
TF_KL_EPS = 1e-6                           # KL threshold when every floor is exactly 0
GEN_MARGIN_NATS = 0.5                      # a divergence is explained iff the reference top1-top2 margin < 0.5 nats
GEN_MAX_UNEXPLAINED = 3                    # at most 3 of the 34 items may diverge at a margin >= 0.5 nats
DEGEN_WINDOW = 200                         # last 200 tokens periodic with period <= 50 => degenerate
DEGEN_MAX_PERIOD = 50
TIER_FULL = "T1"                           # teacher-forced log-probs + greedy tokens
TIER_TOKENS = "T2"                         # greedy tokens only (adapter lacks teacher_forced_logprobs, justified)

# --------------------------------------------------------------------------- #
# Stage 4 / 5: runtime (PREREG §10, §11)
# --------------------------------------------------------------------------- #
COMPONENT_REQUESTS = 10                    # per group: the first 10 measured timing prompts
COMPONENT_WARMUP = 2                       # per group: the warm-up timing prompts
# Stage-5 blocks. The baselines of the primary endpoint ALWAYS use the reference runtime (HF transformers), so the
# primary comparator does not depend on anything the deployment's authors declare. Runtime-matched baselines (the
# deployment's own unsplit mode) are added as secondary blocks only if `unsplit_same_runtime` was declared at Stage 0.
BLOCK_ORDER = ("split_A", "edge_only", "server_only", "split_B")
NATIVE_BLOCKS = ("edge_only_native", "server_only_native")
BLOCK_HOST = {"split_A": "edge", "edge_only": "edge", "server_only": "server", "edge_only_native": "edge",
              "server_only_native": "server", "split_B": "edge"}


def block_order(capabilities):
    return BLOCK_ORDER[:3] + (NATIVE_BLOCKS if "unsplit_same_runtime" in capabilities else ()) + BLOCK_ORDER[3:]


DRIFT_MAX_REL = 0.10                       # |median TPOT split_B - split_A| / split_A
CLOSURE_MIN_RATIO = 0.90                   # split TPOT >= 0.90 x (edge step + server step), else decomposition flagged
CLOCK_SKEW_TOLERANCE_S = 30                # cross-host timestamp comparisons (block order, serve windows)
BOOT_SEED = 20261116
BOOT_N = 10000
CI_LEVEL = 0.95
EFFICIENCY_CATEGORIES = ("SPLIT_FASTER_THAN_EDGE", "NOT_DISTINGUISHABLE_FROM_EDGE", "SPLIT_SLOWER_THAN_EDGE")

# --------------------------------------------------------------------------- #
# Stage 2: memory feasibility (PREREG §8)
# --------------------------------------------------------------------------- #
OOM_DEVICE_PATTERNS = ("cuda out of memory", "outofmemoryerror", "cudaerrormemoryallocation", "cublas_status_alloc_failed",
                       "cudnn_status_alloc_failed")
OOM_HOST_PATTERNS = ("memoryerror", "cannot allocate memory", "std::bad_alloc")   # memory limit only on a unified-memory host
FOREIGN_PROCESSES_ALLOWED = 0              # foreign compute processes on an experiment device at block start / end
MONITOR_PERIOD_S = 1.0                     # foreign compute processes are also sampled at this period DURING a memory probe
OCCUPIED_DEVICE_MAX_MIB = 1024             # device memory held by others before a probe loads anything (discrete GPU)
OCCUPIED_HOST_MAX_FRAC = 0.10              # unified-memory host: memory held by other processes before the probe, / MemTotal
REQUEST_STALL_S = 300                      # Stage 5: no token for this long aborts the block (a crash: repeated, capped)
SERVE_STOP_GRACE_S = 15                    # after a stop signal the server record is written even if serve() does not return

# --------------------------------------------------------------------------- #
# Repeats (PREREG §13.2). A gate lists the parts that may be repeated and why; `invalidate` accepts nothing else.
# Caps count earlier invalidations of the same part for the same cause. None = unlimited (nothing was measured).
# A cause absent from the table is not repeatable (it needs a corrected capture and a new freeze record).
# --------------------------------------------------------------------------- #
REPEAT_CAPS = {"stage2": {"blocked": None, "companion": None, "confirm_infeasible": 2, "invalid": 2},
               "stage3": {"exceptions": 2, "contention": 2, "invalid": 2},
               "stage4": {"invalid": 2},
               "stage5": {"drift": 1, "failed_requests": 2, "contention": 2, "invalid": 2}}
# Caps count across the whole chain of freeze records, and a supersede counts as a repeat of the causes that justified
# it (a new record never resets them). Stage 5 repeats the whole block sequence (no single block can be swapped for a
# preferred attempt); its caps count repeat events per cause. Stage 2 repeats both hosts together (the split probe is one
# two-host observation: no record of one attempt is ever combined with the other host's record of another); caps count
# per host part, and the host repeated only for that reason has cause `companion` (no cap). A stale file
# (older than its part's last invalidation) is never a repeat: the gate refuses to compute until the fresh file is synced.
REPEAT_SCOPE = {"stage2": "both", "stage3": "part", "stage4": "part", "stage5": "sequence"}
# Total repeats per part (Stages 2-4) or per sequence (Stage 5) over ALL causes together (causes with cap None, where
# nothing ran, excluded): mixing causes can never add up to more redraws than this.
REPEAT_TOTAL = {"stage2": 3, "stage3": 2, "stage4": 2, "stage5": 2}
# A new freeze record may follow an INVALID gate only for these causes (a capture fact to correct, or a possible
# adapter defect); operational causes (contention, failed requests, drift, missing or stale files) are repeated in-run.
SUPERSEDE_CAUSES = ("boundary", "workload_mismatch", "exceptions")
MAX_RECORD_REVISIONS = 3

# --------------------------------------------------------------------------- #
# Verdict vocabulary (PREREG §13)
# --------------------------------------------------------------------------- #
STAGE0_VERDICTS = ("CONTINUE", "STOP_INCOMPLETE", "STOP_ROLE_MISMATCH", "STOP_CHECKPOINT_MISMATCH",
                   "STOP_SPLIT_CAPTURE_INVALID", "STOP_NO_CONFIGURATION")
MEMORY_STATUSES = ("FITS", "INFEASIBLE", "BLOCKED_CONTENTION", "INVALID")
STAGE_VERDICTS = {"stage2": ("CONTINUE", "STOP_INFEASIBLE", "INFEASIBLE_UNCONFIRMED", "BLOCKED_CONTENTION", "INVALID", "INCOMPLETE"),
                  "stage3": ("PASS", "FAIL", "NOT_EVALUABLE", "INVALID", "INCOMPLETE"),
                  "stage4": ("COMPLETE", "INVALID", "INCOMPLETE"),
                  "stage5": ("COMPLETE", "UNSTABLE", "INVALID", "INCOMPLETE")}
OPEN_VERDICT = {"stage2": "CONTINUE", "stage3": "PASS", "stage4": "COMPLETE"}
OUTCOMES = ("INCOMPLETE", "INFEASIBLE", "NOT_EQUIVALENT", "NOT_EVALUABLE", "COMPLETE", "UNSTABLE")
STAGE_DIRS = {"stage1": "stage1_checks", "stage2": "stage2_memory", "stage3": "stage3_correctness", "stage4": "stage4_components",
              "stage5": "stage5_e2e", "final": "final"}
STAGE_ORDER = ("stage2", "stage3", "stage4", "stage5")

# --------------------------------------------------------------------------- #
# Freeze (PREREG §6). Decision files are byte-identical to the tag at every check; only the remaining (driver) files
# may differ, and only through a documented fix in a new freeze record (frozen/HARNESS_FIXES.md).
# --------------------------------------------------------------------------- #
DECISION_FILES = (PREREG_FILE, "gs_spec.py", "gs_rules.py", "gs_stats.py", "gs_gate.py", "gs_split.py", "gs_results.py", "gs_freeze.py",
                  "gs_adapter.py", "gs_common.py", "gs_checkpoint.py", "gs_workload.py", "gs_reference.py", "gs_correctness.py",
                  "gs_memory.py", "gs_profile.py", "workload/correctness_prompts.json", "workload/timing_spec.json")
ORCHESTRATION_FILES = ("gs_bench.py", "gs_inventory.py", "gs_deployment.py", "gs_archive.py")   # compute no measurement
DEV_MAX_MODEL_BYTES = 512 * 1024 ** 2      # --dev runs tiny test checkpoints only, never the captured model

# --------------------------------------------------------------------------- #
# The Qwen3 E1-OPT design, as a DESCRIPTION for the difference record (gs_split.differences). Nothing is read
# from or imported out of the E1-OPT directory; the tag and commit are citations.
# --------------------------------------------------------------------------- #
QWEN3_E1OPT_REFERENCE = {
    "experiment": "E1-OPT (Qwen3-32B)", "tag": "e1opt-prereg-v1", "commit": "d9ba549f006bd18f960b385953acbce8259b9dff",
    "topology": "u_shape",
    "edge": ["embed_tokens", "layers [0,k)", "layers [L-j,L)", "final_norm", "lm_head", "sampling", "tokenizer"],
    "server": ["layers [k,L-j)"], "wire": ["residual stream, bf16"],
    "never_on_wire": ["raw text", "token ids", "logits", "sampled tokens", "KV entries", "per-layer inputs"],
    "kv_cache": "each host caches its own layers", "server_gpus": 1, "dtype": "bfloat16", "quantization": "none",
}
