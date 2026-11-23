"""ROUTER-INTEG: every fixed parameter of the router-deployment integration (PROTOCOL_ROUTER_INTEG.md).

A decision file. After the freeze record (frozen/FREEZE.json) exists it is never edited; a changed rule is a new
protocol version with a new freeze record. Nothing here identifies a machine, a user or a path: host-specific values
live in a private site file (site.example.json).
"""
NAME = "ROUTER-INTEG"
PROTOCOL = "router-integ-v1"
SCHEMA = "router-integ/1"

# ---- the reused, frozen runtime (QWEN-SELECT, imported unchanged from a clean clone at this commit) ----
REUSED_TAG = "qwen-select-prereg-v1"
REUSED_COMMIT = "4484a872d1007f8cf40a579aad14440dcbd75ad0"
REUSED_DIRS = ("diagnostics/deployment_e0", "diagnostics/deployment_e1opt", "diagnostics/deployment_k16fit",
               "diagnostics/deployment_qwen_select", "evaluation")
TRANSFORMERS_REQUIRED = "4.51.3"           # the reused loader calls that release's Qwen3 classes

# ---- the model pair of the frozen router. Never remapped to another pool. ----
DS, QW = "DS", "QW"
MODELS = {
    DS: {"name": "DeepSeek-R1-0528-Qwen3-8B", "repo": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
         "revision": "6e8885a6ff5c1dc5201574c8fd700323f23c25fa", "n_layers": 36,
         "manifest": "frozen/manifest_DS.json", "tensors": "frozen/tensors_DS.json",
         "small_files": ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"),
         "eos_ids": (151645,),                        # config.json eos_token_id; the checkpoint has no generation_config.json
         "top_k": None},                              # no model sampling default exists (winner-label stability audit)
    QW: {"name": "Qwen3-8B", "repo": "Qwen/Qwen3-8B",
         "revision": "b968826d9c46dd6066d109eabc6255188de91218", "n_layers": 36,
         "manifest": "frozen/manifest_QW.json", "tensors": "frozen/tensors_QW.json",
         "small_files": ("config.json", "generation_config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"),
         "eos_ids": (151643, 151645),                 # generation_config.json eos_token_id
         "top_k": 20},                                # generation_config.json top_k, applied by the original server default
}
MODEL_ORDER = (DS, QW)
ROUTER_CLASS_TO_MODEL = {0: DS, 1: QW}     # the router study's label: 0 = DeepSeek wins, 1 = Qwen3-8B wins
ROUTER_THRESHOLD = 0.0                     # score > 0 -> Qwen3-8B. Fixed by the router study, never tuned, not tuned here
QW_MANIFEST_SHA256 = ("29bb36d2cfa872b5fe9c7d0f26b181459b9c0394146ccb4bf04aac9726633585",      # = qsel_spec.MANIFEST_SHA256["8B"]:
                      "22e1b0e9e7c700f42475b8dcd1fd2e268aeabc7b7be454c32dbbd3d13cb8c949")      # the manifest verified on both hosts on 20 November

# ---- the frozen router (router/, built and verified by tools/rint_router_build.py) ----
ROUTER_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
ROUTER_ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
ROUTER_ENCODER_FILES = {"model.safetensors": "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db",
                        "tokenizer.json": "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
                        "config.json": "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41",
                        "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
                        "tokenizer_config.json": "acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b",
                        "special_tokens_map.json": "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3"}
ROUTER_MAX_TOKENS = 256                    # truncation of the router study
ROUTER_THREADS = 4                         # the router study's CPU timing used four threads
ROUTER_FOLDS = 5
ROUTER_HEAD_PARAMS = 24705

# ---- serving protocol: the reconstructed protocol of the winner-label stability experiment, where feasible ----
TEMPERATURE = 0.2                          # recorded original request sampling
TOP_P = 1.0
SYSTEM_PROMPT = None                       # one user turn, no system prompt
CHAT_TEMPLATE_KWARGS = {}                  # the checkpoint's own chat template with its defaults: reasoning traces are produced
ADD_GENERATION_PROMPT = True
CONTEXT_TOKENS = 32768                     # the original total-context limit
MAX_NEW_TOKENS = 16384                     # DEVIATION (memory of one shared server GPU): min(this, CONTEXT_TOKENS - prompt tokens)
PREFILL_CHUNK = 2048                       # as in QWEN-SELECT's whole-model loop
POSITION_SCALE_EXPECTED = 1.0              # rotary cos/sin scale of both models under the original engine (rint_backend.align_position_scaling)
QUERY_CUTOFF_S = 600.0                     # DEVIATION (the original had no time limit): a query still running is stopped; = the largest deadline
WARMUP_NEW_TOKENS = 8
SEED_FORMAT = "router-integ-v1|{key}|{model}|{mode}|{phase}|{rep}"     # per-request sampling seed = first 8 hex digits of its sha256

# ---- execution configurations ----
EDGE, REMOTE, SPLIT = "EDGE", "REMOTE", "SPLIT"
CONFIGS = {"DS_EDGE": (DS, EDGE), "QW_EDGE": (QW, EDGE), "DS_REMOTE": (DS, REMOTE), "QW_REMOTE": (QW, REMOTE)}
REQUIRED_CONFIGS = ("DS_EDGE", "QW_EDGE", "DS_REMOTE", "QW_REMOTE")
SPLIT_STATUS = "NOT_RUN"                   # no correctness gate exists for a layer split of either 8B model; see the protocol, section 5
EDGE_ARCH, EDGE_GPU_NAME, SERVER_GPU_NAME = "aarch64", "GB10", "RTX A6000"

# ---- data policies and the fixed placement rule ----
POLICY_A, POLICY_B = "A_no_raw_query_transfer", "B_raw_query_transfer_allowed"
# What leaves the edge: REMOTE sends the query's token ids (the raw query); a layer split sends hidden states.
MODE_FEASIBLE = {POLICY_A: (EDGE,), POLICY_B: (EDGE, REMOTE)}          # SPLIT would be feasible under both, but is not run
PLACEMENT = {POLICY_A: EDGE, POLICY_B: REMOTE}                         # one fixed rule per data policy, the same for both models
PLACEMENT_FALLBACK = {POLICY_A: None, POLICY_B: EDGE}                  # live pipeline only: if the server does not answer
PLACEMENT_BASIS = ("Fixed before any execution of this pair. Policy A admits EDGE only. Under policy B the server is used "
                   "because, in the archived QWEN-SELECT records, the server GPU produced tokens faster than the edge "
                   "(Qwen3-14B: 57.5 ms per token through the wire against 140.1 ms on the edge). That evidence is for "
                   "another model size; the rule is not revised after this experiment's records exist.")

# ---- evaluation sample (frozen/workload.json) ----
FRAME_EXCLUDED_BENCHMARKS = {"humaneval": "its scorer executes model-written code and no validated sandbox exists here",
                             "livecodebench": "its scorer executes model-written code and no validated sandbox exists here"}
N_QUERIES = 120
SAMPLE_SEED = "router-integ-v1:sample"
ORDER_SEED = {"shared": "router-integ-v1:order:shared", "live": "router-integ-v1:order:live"}
N_PROFILE = 6
PROFILE_SEED = "router-integ-v1:profile"
CLASS4 = ("both_correct", "both_wrong", "ds_only", "qw_only")           # of the ORIGINAL released scores (labels A)

# ---- deadline utility ----
DEADLINES_S = (30, 60, 120, 300, 600)      # E0's grid; QWEN-SELECT used its first four
BOOT_N = 10000
BOOT_SEED = 20261123
CI_LEVEL = 0.95

# ---- replication ----
REPLICATES_PER_CONFIG = 1                  # single stochastic draw per (query, configuration). Each model is drawn twice per
#                                            query (once per placement, different seeds); the protocol states what that supports.

# ---- resources ----
EDGE_MARGIN_BYTES = 4 << 30
SERVER_MARGIN_BYTES = 2 << 30
ACTIVATION_ALLOWANCE = 1.25
HOST_RAM_MIN_BYTES = 8 << 30
MAX_CONSECUTIVE_ERRORS = 3                 # a lane stops after this many requests in a row ended in an error
MAX_LANE_ATTEMPTS = 3                      # a lane: the first attempt and at most two resumptions
SERVER_FIRST_WAIT_S = 3600
SERVER_IDLE_EXIT_S = 1800
STAGE_SLACK_S = 1800                       # loading and warm-up
# GPU-hours: one GPU held by one of our processes for one hour, computing or waiting. Planned values are arithmetic from
# the original output lengths of the sampled queries and archived per-token timings (protocol, section 12).
ALLOTMENT_H = {"edge": 36.0, "server": 30.0}
GPU_HOURS_CEILING = 66.0
STAGES = ("profile", "shared", "live_B", "live_A")
STAGE_LIMIT_H = {"edge": {"profile": 2.5, "shared": 22.0, "live_A": 11.5},      # per stage, over all its attempts
                 "server": {"profile": 2.5, "shared": 16.0, "live_B": 8.0}}
# A stage driven from the edge without an edge GPU (the remote lanes, live B) stops STAGE_SLACK_S before its server does.
# If the server fails during live B, the edge models are loaded for the fallback inside what is left of live B's time.
MAX_CONSECUTIVE_FALLBACKS = 3              # live B stops when this many requests in a row needed the fallback
LIVE_REDUCED_N = 60                        # the pre-declared reduction of a live pass: the first 60 queries of its order
PROFILE_STOP_FACTOR = 1.5                  # profiling: stop and report if a host needs more than this times the planning time per token
ARCHIVED_S_PER_TOKEN = {"edge_8B": 0.0966, "server_14B_through_wire": 0.0575}   # QWEN-SELECT summary; context <= 4,096, greedy
