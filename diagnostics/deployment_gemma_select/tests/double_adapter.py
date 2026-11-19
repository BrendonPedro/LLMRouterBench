"""Test double for the split adapter contract (CPU tests only; never a result). It runs a tiny whole model in this
process and pretends to be a split. site["double"] may set "fault": "noise" (wrong logits and tokens) or "delay_s"."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gsel_adapter as AD  # noqa: E402
import gsel_common as C  # noqa: E402
import gsel_whole as W  # noqa: E402

ADAPTER_INFO = {"name": "test double", "test_double": True}


class Double(AD.SplitAdapter):
    def __init__(self, site, role):
        super().__init__(site, role)
        self.opt = site.get("double") or {}

    def serve(self):
        return {"served_from_utc": C.utc(), "served_to_utc": C.utc(), "memory": self.memory()}

    def server_profile(self, q, n_decode):
        return {"q": q, "prefill_s": 1e-4 * q, "decode_s": [1e-4] * n_decode}

    def server_ready(self):
        return not self.opt.get("server_down")

    def connect(self, require_server=True):
        model, _ = W.load_model(self.site["model_dir"], "float32", "cpu", "eager")
        self.runner, self.eos = W.WholeRunner(model, "cpu"), W.eos_ids(self.site["model_dir"])
        return {"eos": self.eos}

    def generate(self, ids, max_new_tokens, on_token=None, stop_on_eos=True):
        delay, n = float(self.opt.get("delay_s", 0)), [0]

        def cb(i, t):
            n[0] += 1
            if delay:
                time.sleep(delay)
            if on_token:
                on_token(i, t)
        t0 = time.perf_counter()
        res = self.runner.generate(ids, max_new_tokens, self.eos if stop_on_eos else (), on_token=cb, stop=AD.Cutoff)
        dt = time.perf_counter() - t0
        if self.opt.get("fault") == "noise":
            res["tokens"] = [(t + 1) % 50 + 6 for t in res["tokens"]]
        res["impl"] = {"request_bytes": 1000 * (len(ids) + n[0]), "response_bytes": 100 * n[0], "roundtrip_s": 0.6 * dt,
                       "server_compute_s": 0.3 * dt, "edge_compute_s": 0.4 * dt, "round_trips": n[0] + 2}
        return res

    def teacher_forced(self, prompt_ids, forced):
        import torch
        for i, row in self.runner.teacher_forced(prompt_ids, forced):
            if self.opt.get("fault") == "noise":
                row = torch.roll(row, 7) * 3.0
            yield i, row

    def edge_profile(self, ids, n_decode):
        t0 = time.perf_counter()
        self.runner.generate(ids, 1, ())
        t1 = time.perf_counter()
        return {"q": len(ids), "prefill_s": t1 - t0, "decode_s": [(t1 - t0) / max(len(ids), 1)] * n_decode}

    def memory(self):
        return {"device": "cpu", "peak_allocated_bytes": 0}


def make_adapter(site, role):
    return Double(site, role)
