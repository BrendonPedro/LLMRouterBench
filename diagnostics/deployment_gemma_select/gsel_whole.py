"""GEMMA-SELECT edge-whole configurations: the unsplit checkpoint with Hugging Face transformers (PREREG §3).

Em = float32 with eager attention (the split runtime's arithmetic); Es = bfloat16 as stored, default attention.
Greedy argmax with the model's own KV cache, one sequence, no sampling. Also the tokenizer call every configuration
shares. Nothing here decides anything.
"""
import json
import time
from pathlib import Path

TIED_OK = ("lm_head.weight",)


def now():
    return time.perf_counter()


def eos_ids(model_dir):
    """The EOS set of generation_config.json (the split runtime reads the same field)."""
    v = json.loads((Path(model_dir) / "generation_config.json").read_text()).get("eos_token_id")
    return sorted(int(x) for x in (v if isinstance(v, list) else [v]))


def load_tokenizer(model_dir):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_dir, local_files_only=True)


def tokenize(tok, prompt):
    """Chat-formatted token ids of one user turn, generation prompt added, thinking off: the same for every configuration."""
    out = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True,
                                  return_dict=True, enable_thinking=False)
    ids = out["input_ids"]
    ids = ids[0] if ids and isinstance(ids[0], (list, tuple)) else ids
    return [int(x) for x in ids]


def load_model(model_dir, dtype_name, device, attn_implementation=None):
    """The class named in config.json; refuses a load that leaves any weight newly initialised."""
    import torch
    import transformers
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    arch = (cfg.get("architectures") or [None])[0]
    cls = getattr(transformers, arch, None) if arch else None
    if cls is None:
        raise RuntimeError(f"transformers has no class {arch!r}")
    kw = {"dtype": getattr(torch, dtype_name), "output_loading_info": True, "local_files_only": True}
    if attn_implementation:
        kw["attn_implementation"] = attn_implementation
    model, info = cls.from_pretrained(model_dir, **kw)
    missing = [k for k in info.get("missing_keys", []) if not k.endswith(TIED_OK)]
    if missing or info.get("mismatched_keys"):
        raise RuntimeError(f"checkpoint did not fill the model: {len(missing)} missing, {len(info.get('mismatched_keys', []))} mismatched")
    model = model.to(device).eval()
    return model, {"class": type(model).__name__, "dtype": dtype_name, "attn_implementation": getattr(model.config, "_attn_implementation", None),
                   "transformers": transformers.__version__, "torch": torch.__version__,
                   "unexpected_keys": len(info.get("unexpected_keys", []))}


def _last_logits(model, x, past):
    try:
        out = model(input_ids=x, past_key_values=past, use_cache=True, logits_to_keep=1)
    except TypeError:
        out = model(input_ids=x, past_key_values=past, use_cache=True)
    return out.logits[0, -1].float(), out.past_key_values


class WholeRunner:
    def __init__(self, model, device):
        self.model, self.device = model, device

    def generate(self, ids, max_new_tokens, eos, on_token=None, stop=Exception):
        """Greedy continuation. on_token(i, id) is called once token i is on the host; if it raises `stop` (the harness
        cutoff), generation ends there. Otherwise it ends on a token of `eos` or at the cap."""
        import torch
        eos, toks, cut = set(eos), [], False
        with torch.inference_mode():
            logits, past = _last_logits(self.model, torch.tensor([ids], device=self.device), None)
            for step in range(max_new_tokens):
                t = int(torch.argmax(logits))
                toks.append(t)
                try:
                    if on_token:
                        on_token(step, t)
                except stop:
                    cut = True
                    break
                if t in eos or step == max_new_tokens - 1:
                    break
                logits, past = _last_logits(self.model, torch.tensor([[t]], device=self.device), past)
        return {"tokens": toks, "cutoff": cut}

    def teacher_forced(self, prompt_ids, forced):
        """Yield (i, logits float32 [V]) for i = 0 .. len(forced)-1: the next-token logits after the prompt and the first
        i forced tokens, computed as generation computes them (prefill, then one token per step)."""
        import torch
        with torch.inference_mode():
            logits, past = _last_logits(self.model, torch.tensor([prompt_ids], device=self.device), None)
            for i in range(len(forced)):
                yield i, logits
                if i < len(forced) - 1:
                    logits, past = _last_logits(self.model, torch.tensor([[forced[i]]], device=self.device), past)


def cuda_memory(device):
    import torch
    if not str(device).startswith("cuda"):
        return {"device": "cpu"}
    free, total = torch.cuda.mem_get_info(device)
    p = torch.cuda.get_device_properties(device)
    return {"device": str(device), "name": p.name, "uuid": str(getattr(p, "uuid", "")), "free_bytes": free, "total_bytes": total,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
