"""E1-OPT constants. Everything the E0 pre-registration fixed is imported read-only from deployment_e0/e0_spec.py.

E1-OPT changes the RUNTIME of the server and edge segments only. The model, the cut, the memory classes, the
workload, the S1 scores, the utility, the deadlines, the margin and the bootstrap are E0's, by import.
"""
import sys
from pathlib import Path

E1_DIR = Path(__file__).resolve().parent
E0_DIR = E1_DIR.parent / "deployment_e0"
sys.path.insert(0, str(E0_DIR))
import e0_spec as E0  # noqa: E402  (read-only use of the frozen E0 constants)

NAME = "E1-OPT"
PROTOCOL_VERSION = 2                      # same 72-byte header and message set as E0; preallocated buffers
E0_VALIDATED_COMMIT = "81461b6c76f84ea6d40962da141631fac6fad601"
E0_STAGE2_SHA256 = "f0df5369c30e70013fa8b3c5dbb54b2df78b06ee5fedaeffbe9b9945839fdae9"
E0_STAGE1_SHA256 = "3984d73af2dee6a2aee334e2a180066b30ea648c1f5c0e2381ff31bfb9ffb84f"

# Runtime modes. "prefix" reproduces E0's computation bit for bit (prefix views of the KV cache, dynamic mask).
# "static" uses full-length KV tensors and a static causal mask so that a decode step has fixed shapes and
# can be captured in a CUDA graph. Both call the same transformers 4.51.3 Qwen3 modules with the same weights.
MODES = ("prefix", "static")

# Required computation-equivalence tests (CPU, before any GPU time; and again on each GPU before measurement)
EQ_PREFIX_BITWISE = True                  # static-cache prefix mode == E0 split == HF unsplit, bitwise
EQ_STATIC_TOP1 = True                     # static mode: greedy tokens identical to HF on the Q2 prompts
EQ_STATIC_MAX_ABS_BF16 = 2 ** -6          # static mode hidden states vs prefix mode, bf16, final norm output
# On GPU the E0 Q1/Q2 thresholds apply unchanged (E0 PREREG §4): D <= floors + 0.2 pp; KL <= 2 x floors.

# Decision rule (frozen from E0): delta >= 3.0 pt with CI lower bound > 0 in some evaluated cell.
R_MARGIN_PT = E0.R_MARGIN_PT
UB_SEGMENT_DISCOUNT = E0.UB_SEGMENT_DISCOUNT

# Required server speedup, from the archived E0 profiles with E0's gate code unchanged (reanalysis/threshold_audit.py,
# corrected 2026-11-11; the first version used ci_lo >= 0 and reported 0.705). f scales the measured Stage-2 server
# profile: server time = 0.90 * f * measured. Scope: archived RTX 3090 edge and competitor profiles, zero link.
F_SERVER_SCALE_FOR_DELTA3 = 0.829669          # largest f with delta >= 3.0 in some cell as e0_r.py computes it (1.205x)
F_SERVER_SCALE_FOR_DELTA3_EXACT = 0.840963    # same, comparing delta rounded to 1e-6 pt (delta = 3.00 exactly; 1.189x)
F_SERVER_SCALE_FOR_PASS = 0.688766            # largest f with delta >= 3.0 AND ci_lo > 0 (strict; the gate's predicate; 1.452x)
F_SERVER_SCALE_FOR_PASS_WITHDRAWN = 0.705394  # ci_lo >= 0 variant reported on 2026-11-11 first pass; NOT the gate; ci_lo == 0 there
SERVER_STEP_MS_MAX_FOR_DELTA3_K3_RAW = 55.5   # 66.89 ms x 0.829669 (raw measured intercept at k = 3, 58 layers)
SERVER_STEP_MS_MAX_FOR_PASS_K3_RAW = 46.1     # 66.89 ms x 0.688766
# Server bandwidth-roofline scan (reanalysis/bw_threshold.py): E0 gate-ub1 form, server at its roofline (no discount),
# archived RTX 3090 edge segments x 0.9, zero link. Roofline steps differ from the scaled-measured steps above because
# the roofline profile has its own slope and prefill terms. Valid for the archived edge profiles ONLY: a different edge
# device changes the split's edge segment and every competitor, so this is not a bound for every target edge.
SERVER_MIN_EFFECTIVE_BW_GBPS_DELTA3 = 1127    # roofline step 50.2 ms at k = 3
SERVER_MIN_EFFECTIVE_BW_GBPS_PASS = 1357      # roofline step 41.7 ms at k = 3 (strict pass; was 1,322 with ci_lo >= 0)
SERVER_ROOFLINE_STEP_MS_MAX_FOR_DELTA3_K3 = 50.2
SERVER_ROOFLINE_STEP_MS_MAX_FOR_PASS_K3 = 41.7

# Arithmetic rule for the E1-OPT decision (pre-specified before any E1 measurement; E0 code is not modified):
# e0_r.py compares delta_pt >= 3.0 and ci95[0] > 0 on floating-point values. With binary scores and 200 x 2 queries
# both are multiples of 0.25 pt, but the float difference of two means can land at 2.9999999999999973 for an exact
# 3.00 (observed in the reanalysis at f in [0.8297, 0.8410]). E1-OPT reports e0_r.py's flag unchanged AND decides on
# values rounded to DECISION_ROUND_PT before comparison. Any disagreement between the two is reported.
DECISION_ROUND_PT = 1e-6


def decide_cell(delta_pt, ci_lo, margin_pt=R_MARGIN_PT, rnd=DECISION_ROUND_PT):
    """E0's rule (delta >= margin and CI lower bound > 0) on values rounded to `rnd` points."""
    return bool(round(delta_pt / rnd) * rnd >= margin_pt and round(ci_lo / rnd) * rnd > 0)


# --------------------------------------------------------------------------- #
# Approved protocol decisions (2026-11-11; PREREG_E1OPT revision 3)            #
# --------------------------------------------------------------------------- #
DECISION_RULE_VERSION = "E1-OPT-r3"       # E0 §8 rule; delta rounded to DECISION_ROUND_PT before >= 3.0; CI low > 0 strict
TORCH_COMPILE = False                     # fixed: never used in E1-OPT
ONE_SERVER_GPU = True                     # the server segment runs on exactly one physical GPU; no TP, no PP, no sharding
FIXED_CUTS = {12: (3, 3), 16: (5, 5), 24: (9, 9)}   # E0 U-max(M); 24 GB stays governed by E0 §8.4(b)
U_MIN = tuple(E0.U_MIN)                   # (1, 1): the "SPLIT U-min" cells of E0's frozen cell grid (not a new cut)
PRIMARY_CLASSES = (12, 16)
PROFILE_ROLES = ("target", "screening")   # only "target" rows may enter a decision gate
ATTENTION_BRANCHES = ("flash_attn_with_kvcache", "sdpa_static_fallback")
GRAPH_BRANCHES = ("on", "off_capture_failed")
MIN_COMPUTE_CAPABILITY = 8.0              # bf16 + FlashAttention-2 need Ampere or newer
SERVER_MEMORY_MARGIN_GIB = 2.0            # CUDA context, allocator slack, wire buffers


class CutError(ValueError):
    pass


def validate_cut(mem_class, k, j, dev_mode=False):
    """Returns the profile role a cut may carry: "target" for E0's frozen cuts (U-max(M) fixed per class, or
    U-min for the E0 U-min cells), "screening" for any other cut and only when dev_mode is set. Otherwise raises."""
    if (k, j) == FIXED_CUTS.get(int(mem_class)) or (k, j) == U_MIN:
        return "target"
    if dev_mode:
        return "screening"
    raise CutError(f"cut ({k},{j}) is not the frozen cut for the {mem_class} GB class {FIXED_CUTS.get(int(mem_class))}"
                   " and dev_mode is off (PREREG_E1OPT §3)")


def decide_cells(cells, margin_pt=R_MARGIN_PT, rnd=DECISION_ROUND_PT):
    """E1 decision rows for E0 gate/R cells: raw delta, rounded delta, E0's own flag, E1's rounded flag."""
    rows = []
    for c in cells:
        if c.get("status") != "evaluated":
            continue
        d, lo = float(c["delta_pt"]), float(c["ci95"][0])
        rows.append({"M": c["M"], "privacy": c["privacy"], "state": c.get("state"), "rtt_ms": c.get("rtt_ms"),
                     "D_s": c["D_s"], "delta_pt_raw": d, "delta_pt_rounded": round(d / rnd) * rnd, "ci95": c["ci95"],
                     "split_best": c.get("split_best"), "comp_best": c.get("comp_best"),
                     "e0_raw_pass": bool(c["pass"]), "e1_pass": decide_cell(d, lo, margin_pt, rnd)})
    return rows
