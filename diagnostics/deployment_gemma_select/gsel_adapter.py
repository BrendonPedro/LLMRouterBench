"""GEMMA-SELECT contract for the split implementation (PREREG §3.2).

The split runtime is a third party's unpublished code. It is not in this repository. A private adapter file, pinned
by SHA-256 in gsel_spec.ADAPTER_SHA256, calls that code unmodified from the hash-verified source directory and exposes
the methods below. The adapter may set the runtime's configuration from the site file, forward calls and read the
runtime's own counters. It may not change a computation, a dtype, the cut, the transport or the serialisation.
"""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gsel_common as C  # noqa: E402
import gsel_spec as S  # noqa: E402


class Cutoff(Exception):
    """Raised by the harness callback to end a generation that passed the cutoff."""


class SplitAdapter:
    def __init__(self, site, role):
        self.site, self.role = site, role

    # ---- server host ----
    def serve(self):
        """Load the server half on the site's device and serve on 127.0.0.1:<port> until SIGINT / SIGTERM.
        Returns {"served_from_utc", "served_to_utc", "memory": {...}} after shutdown."""
        raise NotImplementedError

    def server_profile(self, q, n_decode):
        """Server half alone, in this process, on inputs of the wire shapes: {"q", "prefill_s", "decode_s": [...]}."""
        raise NotImplementedError

    # ---- edge host ----
    def server_ready(self):
        """True if the experiment's own server answers ready through the forward. Loads nothing and never raises."""
        raise NotImplementedError

    def connect(self, require_server=True):
        """Load the edge half; with require_server, the experiment's own server must answer ready. Returns {"eos": [...]}."""
        raise NotImplementedError

    def generate(self, ids, max_new_tokens, on_token=None, stop_on_eos=True):
        """Greedy generation through the runtime's own generation function. on_token(i, id) is called when token i is
        on the edge and may raise Cutoff. Returns {"tokens", "cutoff", "impl"}; impl holds the runtime's own counters
        summed over the request: request_bytes, response_bytes, roundtrip_s, server_compute_s, edge_compute_s, round_trips."""
        raise NotImplementedError

    def teacher_forced(self, prompt_ids, forced):
        """Yield (i, logits float32 [V]) as gsel_whole.WholeRunner.teacher_forced does, through the split."""
        raise NotImplementedError

    def edge_profile(self, ids, n_decode):
        """Edge half alone (no network): {"q", "prefill_s", "decode_s": [...]}."""
        raise NotImplementedError

    def memory(self):
        raise NotImplementedError

    def close(self):
        pass


def load(site, role, dev=False):
    path = Path(site["adapter_path"])
    digest = C.sha256_file(path)
    if not dev and digest != S.ADAPTER_SHA256:
        raise C.Stop("the adapter file is not the pinned one")
    spec = importlib.util.spec_from_file_location("gsel_adapter_impl", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.make_adapter(site, role), {"sha256": digest, "info": getattr(mod, "ADAPTER_INFO", {})}
