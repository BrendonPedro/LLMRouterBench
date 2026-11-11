"""E1-OPT segment runtime: the same Qwen3 modules and weights as E0, with a static KV cache and an optional
CUDA-graph-captured single-token decode step.

Ownership is E0's (PREREG §2): the edge holds embed_tokens, layers [0, k), layers [L-j, L), the final norm, the
LM head and sampling; the server holds layers [k, L-j). Each side holds the KV cache of its own layers only.
The wire carries one bf16 residual-stream tensor per direction per forward step (e1_protocol).

Weights are loaded by E0's Slice (imported, not modified), so the parameters, the decoder-layer objects, the
rotary embedding and the SDPA attention implementation are identical to E0's. What changes is only how the
KV cache is addressed and how the decode step is launched.
"""
import sys
import time
from pathlib import Path

import torch

import e1_spec as S
sys.path.insert(0, str(S.E0_DIR))
import split_runtime as E0RT  # noqa: E402  (read-only: Slice, load_config, config_hash, weight loading)
from e1_static_cache import StaticKV, CacheOverflow  # noqa: E402


def static_causal_mask(cache_position, max_len, dtype, device):
    """[1, 1, n, max_len] additive mask: column c is masked for row i iff c > cache_position[i].
    Unfilled cache positions are always > every current position, so they are masked too."""
    cols = torch.arange(max_len, device=device)
    m = cols[None, :] > cache_position[:, None]
    out = torch.zeros((1, 1, cache_position.numel(), max_len), dtype=dtype, device=device)
    out.masked_fill_(m[None, None], torch.finfo(dtype).min)
    return out


class Segment:
    """A contiguous layer range of an E0 Slice with its own StaticKV, run in prefix or static mode."""

    def __init__(self, sl, layer_ids, max_len, mode="prefix"):
        self.sl, self.layer_ids, self.max_len, self.mode = sl, list(layer_ids), int(max_len), mode
        missing = set(self.layer_ids) - set(sl.layer_ids)
        if missing:
            raise ValueError(f"slice lacks layers {sorted(missing)}")
        cfg = sl.cfg
        self.cache = StaticKV(self.layer_ids, max_len, cfg.num_key_value_heads, cfg.head_dim, sl.dtype, sl.device, mode)
        self.graph = None

    def reset(self):
        self.cache.reset()

    @torch.inference_mode()
    def _run(self, h, cache_position):
        position_ids = cache_position.unsqueeze(0)
        if self.mode == "prefix":
            mask = self.sl.skel_model._update_causal_mask(None, h, cache_position, self.cache, False)
        else:
            mask = static_causal_mask(cache_position, self.max_len, h.dtype, h.device)
        pe = self.sl.rotary(h, position_ids)
        for i in self.layer_ids:
            h = self.sl.layers[i](h, attention_mask=mask, position_ids=position_ids, past_key_value=self.cache,
                                  output_attentions=False, use_cache=True, cache_position=cache_position,
                                  position_embeddings=pe)[0]
        return h

    @torch.inference_mode()
    def forward(self, h, start, commit=True):
        """One forward step over this segment for the n tokens in h ([1, n, hidden]) at positions start.."""
        n = h.shape[1]
        self.cache.check(start, n)
        if n == 1 and self.graph is not None:
            y = self.graph(h, start)
        else:
            cache_position = torch.arange(start, start + n, device=h.device)
            y = self._run(h, cache_position)
        if commit:
            self.cache.commit(n)
        return y

    def enable_graph(self, warmup=3):
        """Capture the single-token decode step as a CUDA graph (static mode, CUDA only). On CPU this installs
        the same static-buffer step without capture, so the control flow is exercised by the CPU tests."""
        if self.mode != "static":
            raise ValueError("CUDA-graph capture needs static mode")
        self.graph = GraphedStep(self, warmup)
        return self.graph


class GraphedStep:
    """Static input/output/position buffers around Segment._run for n = 1; CUDA graph when available."""

    def __init__(self, seg, warmup=3):
        self.seg = seg
        dev, H = seg.sl.device, seg.sl.cfg.hidden_size
        self.h_in = torch.zeros((1, 1, H), dtype=seg.sl.dtype, device=dev)
        self.pos = torch.zeros((1,), dtype=torch.long, device=dev)
        self.out = torch.zeros((1, 1, H), dtype=seg.sl.dtype, device=dev)
        self.cuda = dev.type == "cuda"
        self.g = None
        if self.cuda:
            saved = seg.cache.seen
            s = torch.cuda.Stream(dev)
            s.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(s):
                for _ in range(warmup):
                    self._step()
            torch.cuda.current_stream(dev).wait_stream(s)
            self.g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.g):
                self._step()
            seg.cache.seen = saved   # warm-up/capture wrote at position 0; the owner re-fills from a reset

    def _step(self):
        self.out.copy_(self.seg._run(self.h_in, self.pos))

    @torch.inference_mode()
    def __call__(self, h, start):
        self.h_in.copy_(h)
        self.pos.fill_(int(start))
        if self.g is not None:
            self.g.replay()
        else:
            self._step()
        return self.out.clone()


class USplitEdge:
    """Edge half: embed -> head segment -> middle (local or remote) -> tail segment -> norm -> lm_head -> argmax."""

    def __init__(self, sl, k, j, middle, max_len, mode="prefix"):
        L = sl.n_layers
        if not (k >= 1 and j >= 1 and k + j <= L):
            raise ValueError(f"illegal split (k={k}, j={j}): raw text must stay on the edge")
        self.sl, self.k, self.j, self.middle, self.max_len = sl, k, j, middle, max_len
        self.head = Segment(sl, range(k), max_len, mode)
        self.tail = Segment(sl, range(L - j, L), max_len, mode)
        self.steps, self.token_ns = [], []

    def reset(self):
        self.head.reset(); self.tail.reset(); self.middle.reset(); self.steps = []

    def enable_graph(self):
        self.head.enable_graph(); self.tail.enable_graph()

    @torch.inference_mode()
    def forward_hidden(self, ids, start):
        t0 = time.perf_counter_ns()
        x = self.sl.embed(ids)
        x = self.head.forward(x, start)
        E0RT._sync(self.sl.device); t1 = time.perf_counter_ns()
        y = self.middle.forward(x, start)
        t2 = time.perf_counter_ns()
        y = self.tail.forward(y, start)
        y = self.sl.norm(y)
        E0RT._sync(self.sl.device); t3 = time.perf_counter_ns()
        m = dict(getattr(self.middle, "last", {}))
        m.update(n=ids.shape[1], start=start, edge_head_ns=t1 - t0, edge_tail_ns=t3 - t2, middle_ns=t2 - t1,
                 total_ns=t3 - t0, lm_head_ns=0)
        self.steps.append(m)
        return y

    @torch.inference_mode()
    def generate(self, prompt_ids, max_new, chunk, eos_ids=(), force_len=False, margins=False):
        """Greedy decoding, argmax over bf16/fp32 logits, first index on ties (E0 semantics).
        margins=True also records self.margins[i] = top-1 minus top-2 log-probability at token i (E0 Q2)."""
        self.reset()
        dev = self.sl.device
        ids = torch.as_tensor(prompt_ids, dtype=torch.long, device=dev).view(1, -1)
        P = ids.shape[1]
        if P + max_new > self.max_len:
            raise CacheOverflow(f"prompt {P} + {max_new} > cache {self.max_len}")
        t_start = time.perf_counter_ns()
        for s in range(0, P, chunk):
            h = self.forward_hidden(ids[:, s:s + chunk], s)
        out, stamps, pos, self.margins = [], [], P, []
        while True:
            t_h = time.perf_counter_ns()
            logits = self.sl.lm_head(h[:, -1:, :])[0, -1].float()
            tok = int(torch.argmax(logits))
            if margins:
                top = torch.topk(torch.log_softmax(logits, -1), 2).values
                self.margins.append(float(top[0] - top[1]))
            now = time.perf_counter_ns()
            self.steps[-1]["lm_head_ns"] = now - t_h
            out.append(tok); stamps.append(now - t_start)
            if len(out) >= max_new or (not force_len and tok in eos_ids):
                break
            h = self.forward_hidden(torch.tensor([[tok]], device=dev), pos)
            pos += 1
        self.token_ns = stamps
        return out

    @torch.inference_mode()
    def teacher_forced_hidden(self, ids, chunk=None):
        """Final normed hidden states for the whole sequence. One pass when it fits one chunk (bitwise claim);
        otherwise chunked prefill with the KV cache, which is the computation the real runtime performs."""
        self.reset()
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.sl.device).view(1, -1)
        P = ids.shape[1]
        if chunk is None or P <= chunk:
            return self.forward_hidden(ids, 0)[0]
        outs = [self.forward_hidden(ids[:, s:s + chunk], s)[0] for s in range(0, P, chunk)]
        return torch.cat(outs, 0)


class LocalMiddle:
    """Server segment in-process (tests, in-host floors, and the server-side segment profiler)."""

    def __init__(self, sl, layer_ids, max_len, mode="prefix"):
        self.seg = Segment(sl, layer_ids, max_len, mode)
        self.last = {}

    def reset(self):
        self.seg.reset()

    def enable_graph(self):
        self.seg.enable_graph()

    def forward(self, h, start):
        E0RT._sync(h.device); t0 = time.perf_counter_ns()
        y = self.seg.forward(h, start)
        E0RT._sync(h.device); t1 = time.perf_counter_ns()
        self.last = {"server_compute_ns": t1 - t0, "ser_ns": 0, "rt_ns": t1 - t0, "server_ns": t1 - t0, "net_ns": 0,
                     "bytes_out": 0, "bytes_in": 0, "server_seen": self.seg.cache.seen}
        return y


def edge_slice(model_dir, k, j, device, dtype=torch.bfloat16, expected_config=None):
    return E0RT.edge_slice(model_dir, k, j, device, dtype, expected_config)


def server_slice(model_dir, k, j, device, dtype=torch.bfloat16, expected_config=None):
    cfg = E0RT.load_config(model_dir)
    return E0RT.Slice(model_dir, list(range(k, cfg.num_hidden_layers - j)), False, False, device, dtype, expected_config)


def ownership(edge_sl, k, j, server_layer_ids=None):
    """The frozen ownership (E0 PREREG §2) as a checkable dict; used by the equivalence driver and the freeze."""
    L = edge_sl.n_layers
    d = {"edge_layers": list(range(k)) + list(range(L - j, L)), "server_layers": list(range(k, L - j)),
         "edge_holds": ["embed_tokens", "head_layers", "tail_layers", "final_norm", "lm_head", "sampling", "tokenizer"],
         "server_holds": ["middle_layers"], "wire": "bf16 residual stream, one tensor per direction per step",
         "never_on_wire": ["raw text", "token ids", "logits", "sampled tokens", "KV entries"],
         "edge_slice_ok": sorted(edge_sl.layer_ids) == list(range(k)) + list(range(L - j, L))
                          and edge_sl.embed is not None and edge_sl.norm is not None and edge_sl.lm_head is not None}
    if server_layer_ids is not None:
        d["server_slice_ok"] = sorted(server_layer_ids) == list(range(k, L - j))
    return d
