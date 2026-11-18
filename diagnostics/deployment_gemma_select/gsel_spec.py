"""GEMMA-SELECT: every fixed parameter of the experiment (PREREG_GEMMA_SELECT.md). A decision file: never edited after the tag.

Nothing here identifies a machine, a user or a path. Host-specific values live in a private site file (site.example.json).
"""
NAME = "GEMMA-SELECT"
TAG = "gemma-select-prereg-v1"
PROTOCOL_VERSION = 1
SCHEMA = "gemma-select/1"

# ---- model and split implementation (identity only; neither is redistributed) ----
MODEL_DIR_NAME = "gemma-4-E4B-it"
CHECKPOINT_FILES = {   # file -> (sha256, bytes); identical on both hosts when captured
    "chat_template.jinja": ("0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5", 18569),
    "config.json": ("33b10c02df3c2e8536cf323d29d53262aaa2f4d11dbe19bc729373fbe90295d4", 5145),
    "generation_config.json": ("d4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de", 208),
    "model.safetensors": ("cfbd3d2f1cd71bd471c37fe2bf8546d5028d41e5736f64e1ca6c6b8893125503", 15992595884),
    "processor_config.json": ("32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c", 1689),
    "tokenizer.json": ("cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f", 32169626),
    "tokenizer_config.json": ("9f4fec4b1dc6ecddf8f4a92e9caea5971c0e67d81309f3f9066a2bee8c362633", 3082),
}
N_LAYERS = 42
CUT = 24                                   # edge: layers [0, 24); server: layers [24, 42)
SPLIT_DTYPE = "float32"                    # as implemented by the split runtime (eager attention)
SOURCE_MANIFEST_SHA256 = "cc030037a1e5a1625860838d73bbd381a15e7bfa49c8aab154af681b5112ce7f"   # 54 source files
SOURCE_N_FILES = 54
ADAPTER_SHA256 = "151cad644147217a5cdba3f1430a4402076827340924b35591b5aaa7c029a04e"        # the private adapter file, pinned at the freeze

# ---- roles (model names only; hostnames and device ids are in the site file) ----
EDGE_ARCH = "aarch64"
EDGE_GPU_NAME = "GB10"
SERVER_GPU_NAME = "RTX A6000"
ROLES = ("edge", "server")

# ---- workload ----
S1_SAMPLE_REL = "diagnostics/deployment_s1/sample_200.jsonl"
S1_SAMPLE_SHA256 = "2c4df49360f997c3e35593030da40111c2826f068f03b1401284a394d9716955"
BENCHMARKS = ("mmlupro", "gpqa", "livemathbench")          # LiveCodeBench excluded: its scorer executes model-written code
WORKLOAD_FILE = "workload/workload.json"
WORKLOAD_SHA256 = "6c13ed3eab8b85b6ab22b0faac8fba415683b272e65ecfb77c6bf2b126254489"
ORDER_SEED = "gemma-select:order:v1"
SENTINEL_PER_BENCH = {"mmlupro": 9, "gpqa": 9, "livemathbench": 6}
N_CORRECTNESS_SHORT = 5
N_CORRECTNESS_LONG = 5
N_CONT = 64                                # continuation tokens per correctness prompt
MAX_NEW_TOKENS = 1024                      # the split runtime's own limit, applied to every configuration
QUERY_CUTOFF_S = 600.0                     # = the largest deadline: a query still running is late in every cell
WARMUP_NEW_TOKENS = 8

# ---- correctness gate (fixed before any GPU result) ----
MARGIN_NATS = 0.5
MIN_DECISIVE_AGREEMENT = 0.995
MAX_MEAN_KL_NATS = 0.05
MIN_GREEDY_OK = 9                          # of 10 prompts

# ---- deadline utility and the decision rule (E0's rule, unchanged) ----
CONFIGS = ("S", "Em", "Es")                # split; edge whole float32 eager; edge whole bfloat16 as stored
EDGE_CONFIGS = ("Em", "Es")
BLOCKS = ("S1", "Es", "Em", "S2")          # execution order; S2 repeats the sentinel queries only
BLOCK_CONFIG = {"S1": "S", "S2": "S", "Es": "Es", "Em": "Em"}
WHOLE_DTYPE = {"Em": ("float32", "eager"), "Es": ("bfloat16", None)}
DEADLINES_S = (5, 10, 30, 60, 120, 300, 600)
MARGIN_PT = 3.0
BOOT_N = 10000
BOOT_SEED = 20261119
CI_LEVEL = 0.95
MIN_CELL_UTILITY = 0.05                    # cells where every configuration is below this are not evaluated
STABILITY_BAND = (0.90, 1.10)              # median completion time, S2 / S1, on the sentinel queries

# ---- secondary: the screen's timing bound ----
BOUND_DISCOUNT = 0.90
PROFILE_LENGTHS = (64, 256, 1024)
PROFILE_DECODE_STEPS = 32
PROFILE_REPS = 3
BOUND_OPTIMISTIC_SHARE = 0.95

# ---- resource gates ----
LOAD_FACTOR = 1.25                         # split halves: required free memory = 1.25 x weight bytes + 1 GiB (the runtime's own rule)
WHOLE_LOAD_FACTOR = 2.0                    # whole model: 2 x weight bytes + 1 GiB (it is read into RAM, then moved to the device)
LOAD_HEADROOM_BYTES = 1 << 30
MAX_BLOCK_REPEATS = 2

OUTCOMES = ("SCREEN_CONFIRMED", "SCREEN_CONTRADICTED", "CORRECTNESS_FAIL", "SPLIT_INFEASIBLE", "NOT_EVALUABLE", "UNSTABLE",
            "INCOMPLETE")
