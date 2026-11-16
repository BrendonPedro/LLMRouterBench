"""GEMMA-SPLIT adapter contract (PREREG §5): the only interface between the harness and the system under test.

The system under test is the existing Gemma split deployment, exactly as captured at Stage 0. Its code is not
known when this protocol is tagged, so the harness drives it through an ADAPTER written after Stage 0 against this
contract, CPU-tested, and frozen (hashed in the freeze record) before any correctness or timing run.

Permitted adapter scope (PREREG §5.2): call the deployment's own code from the hash-verified snapshot; expose the
methods below; force greedy decoding and the frozen token ids. NOT permitted: changing the checkpoint, dtype,
layer ownership, cut, kernels, KV policy, transport, serialisation or any computation; adding timers (the harness
times); caching across requests beyond what the deployment itself does.

An adapter module is a file with
  ADAPTER_INFO = {...}          a literal dict (read with ast, no import needed): see INFO_KEYS
  def make_adapter(freeze, role, options) -> SplitAdapter
Capabilities (ADAPTER_INFO["capabilities"], equal to the Stage-0 declaration capture/capabilities.json); REQUIRED ones
must be present:
  greedy_tokens            split-path greedy generation returning token ids                       (Stages 2, 3)
  stream_tokens            split-path generation calling on_token(i, token_id) when the edge has it (Stage 5)
  boundary_audit           names/dtypes/shapes/bytes of every message of the last request          (Stage 3)
  teacher_forced_logprobs  per-position full-vocabulary log-probs of the split path (tier T1)    (Stage 3)
  ignore_eos               generate exactly max_new_tokens                                         (Stage 5)
  segment_step             one prefill / decode step of THIS host's segment, in isolation          (Stage 4 isolated)
  step_trace               per-step in-path compute / round-trip decomposition                     (Stage 4 in-path, 5)
  unsplit_same_runtime     the whole model on one host with the deployment's own runtime          (Stage 5 secondary blocks)
  in_host_split            both segments in one process on one device (a diagnostic, never a floor) (Stage 3)
"""
import ast
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # python -I: the script dir is not on sys.path
import gs_spec as S  # noqa: E402

CAPABILITIES = ("greedy_tokens", "stream_tokens", "boundary_audit", "teacher_forced_logprobs", "ignore_eos", "segment_step",
                "step_trace", "unsplit_same_runtime", "in_host_split")
REQUIRED_CAPABILITIES = ("greedy_tokens", "stream_tokens", "boundary_audit")
INFO_KEYS = ("contract_version", "implementation", "configuration_id", "capabilities", "test_double", "notes",
             "missing_capability_justification")
STEP_TRACE_KEYS = ("phase", "edge_compute_ns", "server_compute_ns", "roundtrip_ns")


class AdapterError(RuntimeError):
    pass


class SplitAdapter:
    """Base class; every method a capability does not cover raises AdapterError. Token ids are lists of ints.
    Log-prob tensors are torch float32 [n, V] on any device. Methods are called on the host given by `role`."""

    def __init__(self, info, role):
        self.info, self.role = info, role

    def capabilities(self):
        return set(self.info["capabilities"])

    # ---- server host -------------------------------------------------------
    def serve(self, ready=None, stop=None):
        """Load the server half under the frozen configuration and serve until `stop` (threading.Event) is set or the
        process is interrupted. Call ready() once the server accepts connections."""
        raise AdapterError("serve not implemented")

    def server_devices(self):
        """[torch device strings] the server half occupies (for memory accounting)."""
        raise AdapterError("server_devices not implemented")

    # ---- edge host ---------------------------------------------------------
    def connect(self):
        """Load the edge half and connect to the running server at the frozen endpoint."""
        raise AdapterError("connect not implemented")

    def edge_devices(self):
        raise AdapterError("edge_devices not implemented")

    def close(self):
        pass

    def greedy(self, prompt_ids, max_new_tokens, eos_ids, ignore_eos=False):
        """-> {"tokens": [generated ids], "stopped_on_eos": bool}. Greedy (argmax), no sampling, no penalties. `tokens`
        should include the EOS id that stopped generation (as the reference runtime does); the rules strip one trailing
        EOS from both sequences before comparing, so either convention is scored the same."""
        raise AdapterError("greedy_tokens not implemented")

    def stream(self, prompt_ids, max_new_tokens, eos_ids, ignore_eos, on_token):
        """Like greedy; calls on_token(i, token_id) on the edge as soon as token i is available there."""
        raise AdapterError("stream_tokens not implemented")

    def teacher_forced_logprobs(self, ids, start):
        """Yield (pos0, logprobs[n, V]) chunks covering positions start .. len(ids)-2 of the split path: row p is the
        log-softmax of the next-token distribution after ids[:p+1]."""
        raise AdapterError("teacher_forced_logprobs not implemented")

    def boundary_audit(self):
        """[{"direction": "edge_to_server"|"server_to_edge", "tensors": [{"name","dtype","shape"}], "bytes": int}] of the
        last request."""
        raise AdapterError("boundary_audit not implemented")

    def set_trace(self, on):
        raise AdapterError("step_trace not implemented")

    def step_trace(self):
        """[{"phase": "prefill"|"decode", "edge_compute_ns", "server_compute_ns", "roundtrip_ns"}] of the last request."""
        raise AdapterError("step_trace not implemented")

    def segment_runner(self, prompt_ids):
        """Object with prefill() and decode_step() running THIS host's segment once each call (no network), on
        boundary inputs of the declared shapes and dtypes; attribute `inputs` = "recorded" | "synthetic"."""
        raise AdapterError("segment_step not implemented")

    def unsplit_runner(self):
        """Object with stream(prompt_ids, max_new_tokens, eos_ids, ignore_eos, on_token) running the whole model on
        THIS host with the deployment's own runtime (baseline only)."""
        raise AdapterError("unsplit_same_runtime not implemented")

    def in_host_runner(self):
        """Object with greedy(...) and teacher_forced_logprobs(...) running both segments in one process on this host."""
        raise AdapterError("in_host_split not implemented")


def safe_close(adapter):
    """Disconnect without letting an error on close discard what the caller has already recorded."""
    try:
        adapter.close()
    except Exception as e:
        print(f"adapter.close() raised {type(e).__name__}: {e}", file=sys.stderr)


def read_info(path):
    """ADAPTER_INFO of an adapter file without importing it (the analysis host may lack torch)."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ADAPTER_INFO" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AdapterError(f"{path}: no literal ADAPTER_INFO assignment")


def check_info(info, target=True):
    """Problems of an ADAPTER_INFO. Target runs refuse test doubles and missing required capabilities."""
    P = [f"ADAPTER_INFO lacks {k}" for k in INFO_KEYS if k not in info]
    if P:
        return P
    if info["contract_version"] != S.ADAPTER_CONTRACT_VERSION:
        P.append(f"contract_version {info['contract_version']} != {S.ADAPTER_CONTRACT_VERSION}")
    caps = set(info["capabilities"])
    unknown = caps - set(CAPABILITIES)
    if unknown:
        P.append(f"unknown capabilities {sorted(unknown)}")
    miss = [c for c in REQUIRED_CAPABILITIES if c not in caps]
    if miss:
        P.append(f"required capabilities missing {miss}")
    if "teacher_forced_logprobs" not in caps and not (info.get("missing_capability_justification") or {}).get("teacher_forced_logprobs"):
        P.append("teacher_forced_logprobs absent without a written justification (PREREG §5.3: mandatory when reachable)")
    if target and info["test_double"]:
        P.append("adapter is a test double: refused for a target run")
    return P


def tier(info):
    return S.TIER_FULL if "teacher_forced_logprobs" in info["capabilities"] else S.TIER_TOKENS


def load(path):
    spec = importlib.util.spec_from_file_location(f"gs_adapter_module_{abs(hash(str(path)))}", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(path).resolve().parent))
    spec.loader.exec_module(mod)
    if not hasattr(mod, "make_adapter"):
        raise AdapterError(f"{path}: no make_adapter(freeze, role, options)")
    if read_info(path) != getattr(mod, "ADAPTER_INFO", None):
        raise AdapterError(f"{path}: runtime ADAPTER_INFO differs from the literal one")
    return mod


def make(freeze, role, gs_dir, options=None):
    mod = load(Path(gs_dir) / freeze["adapter"]["path"])
    ad = mod.make_adapter(freeze, role, options or {})
    if not isinstance(ad, SplitAdapter):
        raise AdapterError("make_adapter must return a gs_adapter.SplitAdapter")
    return ad
