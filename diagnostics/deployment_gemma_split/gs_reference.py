"""GEMMA-SPLIT reference runtime: the UNSPLIT checkpoint with Hugging Face transformers (PREREG §9.1, §11.2).

Used for (a) the correctness references R_server and R_edge, (b) the memory probe of those references and (c) the
edge_only / server_only baselines of the primary endpoint (always this runtime, labelled "reference runtime").
The model class is the one named in config.json `architectures[0]` (transformers' auto mapping as fallback); dtype is
the deployment's frozen dtype; decoding is greedy argmax with a DynamicCache, one sequence, no sampling, no logits
processors. Nothing here writes a file and nothing here decides: the metrics on its outputs are in gs_rules.
"""
import json
from pathlib import Path


def norm_uuid(u):
    u = str(u or "").strip().lower()
    return u[4:] if u.startswith("gpu-") else u


def device_identity(device):
    import torch
    dev = torch.device(device)
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(dev)
        return {"device": str(dev), "name": p.name, "uuid": norm_uuid(getattr(p, "uuid", "")), "total_mib": p.total_memory // (1 << 20),
                "compute_cap": float(f"{p.major}.{p.minor}")}
    return {"device": str(dev), "name": "cpu", "uuid": "cpu", "total_mib": None, "compute_cap": None}


def check_device_uuid(device, expected_uuids):
    """The device a driver uses must be one of the frozen UUIDs for its role (PREREG §6.3)."""
    ident = device_identity(device)
    want = {norm_uuid(u) for u in expected_uuids or []}
    if ident["uuid"] != "cpu" and ident["uuid"] not in want:
        raise RuntimeError(f"device {device} has UUID {ident['uuid']}, not one of the frozen {sorted(want)}")
    return ident


def setup_torch():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


def load_model(model_dir, dtype_name, device, attn_implementation=None):
    torch = setup_torch()
    import transformers
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    arch = (cfg.get("architectures") or [None])[0]
    cls, source = (getattr(transformers, arch, None), "config.architectures[0]") if arch else (None, None)
    if cls is None:
        cls, source = transformers.AutoModelForCausalLM, "AutoModelForCausalLM (architectures[0] unavailable)"
    dtype = getattr(torch, str(dtype_name).replace("torch.", ""))
    kw = {("dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"): dtype}
    if attn_implementation:
        kw["attn_implementation"] = attn_implementation
    model = cls.from_pretrained(model_dir, **kw).to(device).eval()
    info = {"class": type(model).__name__, "class_source": source, "dtype": str(dtype),
            "attn_implementation": getattr(model.config, "_attn_implementation", None), "transformers": transformers.__version__,
            "torch": torch.__version__, "device": device_identity(device)}
    return model, info


def _last_logits(model, x, past):
    try:
        out = model(input_ids=x, past_key_values=past, use_cache=True, logits_to_keep=1)
    except TypeError:
        out = model(input_ids=x, past_key_values=past, use_cache=True)
    return out.logits[0, -1].float(), out.past_key_values


def greedy(model, ids, n_new, eos_ids, ignore_eos, device):
    """Greedy continuation with the reference top1 - top2 log-prob margin (nats) at every generated step."""
    import torch
    eos = set(int(e) for e in eos_ids)
    toks, margins, nan = [], [], False
    with torch.inference_mode():
        logits, past = _last_logits(model, torch.tensor([ids], device=device), None)
        for step in range(n_new):
            nan = nan or bool(torch.isnan(logits).any())
            lp = torch.log_softmax(logits, dim=-1)
            top = torch.topk(lp, 2)
            t = int(top.indices[0])
            toks.append(t)
            margins.append(float(top.values[0] - top.values[1]))
            if (t in eos and not ignore_eos) or step == n_new - 1:
                break
            logits, past = _last_logits(model, torch.tensor([[t]], device=device), past)
    return {"tokens": toks, "margins": margins, "stopped_on_eos": bool(toks and toks[-1] in eos and not ignore_eos), "nan": nan}


def teacher_forced(model, ids, start, device, rows=256):
    """Yield (pos0, logprobs float32 [n, V]) for positions start .. len(ids)-2 of ONE full forward pass."""
    import torch
    with torch.inference_mode():
        logits = model(input_ids=torch.tensor([ids], device=device), use_cache=False).logits[0]
        for a in range(start, len(ids) - 1, rows):
            b = min(a + rows, len(ids) - 1)
            yield a, torch.log_softmax(logits[a:b].float(), dim=-1)


class HFRunner:
    """The reference runtime as a baseline / in-host runner: stream() calls on_token(i, id) once the id is on the host."""

    def __init__(self, model, device):
        self.model, self.device = model, device

    def stream(self, prompt_ids, max_new_tokens, eos_ids, ignore_eos, on_token):
        import torch
        eos = set(int(e) for e in eos_ids)
        toks = []
        with torch.inference_mode():
            logits, past = _last_logits(self.model, torch.tensor([prompt_ids], device=self.device), None)
            for step in range(max_new_tokens):
                t = int(torch.argmax(logits))
                toks.append(t)
                on_token(step, t)
                if (t in eos and not ignore_eos) or step == max_new_tokens - 1:
                    break
                logits, past = _last_logits(self.model, torch.tensor([[t]], device=self.device), past)
        return {"tokens": toks, "stopped_on_eos": bool(toks and toks[-1] in eos and not ignore_eos)}

    def greedy(self, prompt_ids, max_new_tokens, eos_ids, ignore_eos=False):
        return self.stream(prompt_ids, max_new_tokens, eos_ids, ignore_eos, lambda i, t: None)

    def teacher_forced_logprobs(self, ids, start):
        return teacher_forced(self.model, ids, start, self.device)
