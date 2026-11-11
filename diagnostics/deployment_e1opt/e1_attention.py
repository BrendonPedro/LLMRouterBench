"""Attention dispatch for E1-OPT static mode.

transformers 4.51.3's SDPA path materialises repeat_kv over the FULL key length every step (8x for Qwen3-32B's
64/8 GQA). With a static max_len cache that would copy ~4 GB per step on the server. The static GPU path
therefore uses FlashAttention-2's flash_attn_with_kvcache, which reads only cache_seqlens entries, handles GQA
natively and has fixed shapes (CUDA-graph capturable). Without flash_attn (CPU, or a GPU without a wheel) the
function falls back to the stock SDPA path with the static mask: same mathematics, slower, used by the tests.

The function is registered under the name "e1_static" and selected by setting cfg._attn_implementation.
Kernel choice is part of the frozen runtime (PREREG_E1OPT §5): it is fixed before any measurement and both
candidates must pass the Q tests on the GPU they run on.
"""
import torch
from transformers.modeling_utils import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward

try:
    from flash_attn import flash_attn_with_kvcache  # type: ignore
except Exception:  # pragma: no cover - optional on GPU hosts
    flash_attn_with_kvcache = None

_SEQLENS = {}   # module id -> int32 tensor [1] holding (position + n), updated in place by the Segment
FORCE_SDPA = False   # set by the drivers when the frozen branch for this device is "sdpa_static_fallback"
LAST_KERNEL = {"name": None}   # the kernel the last call actually used (read by the equivalence driver)


def kernel_available():
    return flash_attn_with_kvcache is not None


def set_branch(attention_branch):
    """Apply the frozen attention branch (e1_spec.ATTENTION_BRANCHES) to this process."""
    global FORCE_SDPA
    if attention_branch not in ("flash_attn_with_kvcache", "sdpa_static_fallback"):
        raise ValueError(attention_branch)
    FORCE_SDPA = attention_branch == "sdpa_static_fallback"


def set_seqlens(module, t):
    _SEQLENS[id(module)] = t


def e1_static_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    """query [b, h, n, d]; key/value: the FULL static cache [b, kvh, max_len, d] (already updated in place)."""
    seqlens = _SEQLENS.get(id(module))
    use_flash = (flash_attn_with_kvcache is not None and not FORCE_SDPA and query.is_cuda and seqlens is not None
                 and query.dtype in (torch.float16, torch.bfloat16))
    if not use_flash:
        LAST_KERNEL["name"] = "sdpa_static_fallback"
        return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling)
    LAST_KERNEL["name"] = "flash_attn_with_kvcache"
    q = query.transpose(1, 2)                       # [b, n, h, d]
    kc, vc = key.transpose(1, 2), value.transpose(1, 2)   # [b, max_len, kvh, d] strided views of the cache
    out = flash_attn_with_kvcache(q, kc, vc, cache_seqlens=seqlens, softmax_scale=scaling, causal=True)
    return out, None


AttentionInterface.register("e1_static", e1_static_attention)
