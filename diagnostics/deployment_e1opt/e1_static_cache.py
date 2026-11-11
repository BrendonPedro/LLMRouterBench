"""Fixed-shape KV cache for a layer subset, usable by transformers 4.51.3 Qwen3 decoder layers.

Two read-out modes:
  prefix: update() writes in place and returns views of the filled prefix [:, :, :seen+n]. This is exactly
          E0's SliceKVCache semantics, so the attention kernel sees the same tensors and the output is bitwise
          equal to E0 and to the unsplit HF model (verified in tests/test_e1_cpu.py).
  static: update() writes in place with index_copy_ at cache_position and returns the FULL max_len tensors.
          Shapes are then independent of the position, so a single-token step can be captured as a CUDA graph.
          Positions beyond the current one are masked by the static causal mask (e1_segment.static_causal_mask).
No Python-side bookkeeping happens inside update() in static mode: bounds are checked by the owner before the
step (CUDA-graph replay cannot run Python checks).
"""
import torch
from transformers.cache_utils import Cache


class CacheOverflow(RuntimeError):
    pass


class StaticKV(Cache):
    def __init__(self, layer_ids, max_len, n_kv, head_dim, dtype, device, mode="prefix"):
        super().__init__()
        assert mode in ("prefix", "static"), mode
        self.mode, self.max_len, self.seen = mode, int(max_len), 0
        shape = (1, n_kv, self.max_len, head_dim)
        self.k = {i: torch.zeros(shape, dtype=dtype, device=device) for i in layer_ids}
        self.v = {i: torch.zeros(shape, dtype=dtype, device=device) for i in layer_ids}
        self.layer_ids = list(layer_ids)
        self.device = torch.device(device)

    # -- transformers Cache interface --------------------------------------------------------------
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        pos = cache_kwargs["cache_position"]
        k, v = self.k[layer_idx], self.v[layer_idx]
        if self.mode == "prefix":
            n, s = key_states.shape[-2], self.seen
            if s + n > self.max_len:
                raise CacheOverflow(f"layer {layer_idx}: {s}+{n} > max_len {self.max_len}")
            k[:, :, s:s + n].copy_(key_states)
            v[:, :, s:s + n].copy_(value_states)
            return k[:, :, :s + n], v[:, :, :s + n]
        k.index_copy_(2, pos, key_states)
        v.index_copy_(2, pos, value_states)
        return k, v

    def get_seq_length(self, layer_idx=0):
        return self.seen

    def get_max_cache_shape(self):
        return self.max_len if self.mode == "static" else None

    # -- owner-side bookkeeping ----------------------------------------------------------------------
    def check(self, start, n):
        if start != self.seen:
            raise CacheOverflow(f"step start {start} != committed length {self.seen}")
        if start + n > self.max_len:
            raise CacheOverflow(f"{start}+{n} > max_len {self.max_len}")

    def commit(self, n):
        self.seen += int(n)

    def reset(self):
        self.seen = 0   # stale entries beyond `seen` are never read: prefix mode slices, static mode masks

    def nbytes(self):
        return sum(t.numel() * t.element_size() for d in (self.k, self.v) for t in d.values())
