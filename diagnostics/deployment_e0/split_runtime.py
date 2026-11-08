"""E0 split runtime: U-shaped layer split of a Hugging Face Qwen3 checkpoint over TCP.

Edge process: embedding, layers [0, k), layers [L-j, L), final norm, LM head, sampling.
Server process: layers [k, L-j) (a superset range may be loaded; HELLO selects the active range).
Per forward step the edge sends the bf16 residual stream after layer k-1 ([n, hidden] raw bytes)
and receives the residual stream after layer L-j-1. Token ids, text and logits never leave the edge.

The runtime reuses the transformers 4.51.3 Qwen3 modules unchanged (decoder layers, rotary
embedding, causal-mask construction). Each side holds a preallocated KV cache for its own layers
only. Built in from the start: pinned host buffers and non-blocking copies (the single optimisation
pass E0 §3.3 allows). No batching, speculation, compression or CUDA graphs.

  python split_runtime.py serve --model-dir DIR --layers 3-61 --port 29500 --device cuda:0
  python split_runtime.py serve --echo --port 29501          # identity "server" (F / edge profiles)
"""

import argparse
import hashlib
import json
import os
import socket
import struct
import sys
import time
from pathlib import Path

import torch
import transformers
from transformers.cache_utils import Cache
from transformers.models.qwen3 import modeling_qwen3 as mq

PROTOCOL_VERSION = 1
MAGIC = b"E0SP"
# magic, version, type, request id, start position, n tokens, hidden, flags, payload bytes,
# server t_recv_ns (parse + h2d), t_compute_ns, t_prep_ns (d2h), server_seen (cache length after)
HDR = struct.Struct("<4sHHQIIIIQQQQQ")
T_HELLO, T_HELLO_OK, T_FWD, T_FWD_OUT, T_RESET, T_RESET_OK, T_BYE, T_ERROR = range(1, 9)
F_USE_CACHE = 1
assert sys.byteorder == "little", "raw bf16 wire format assumes little-endian hosts"


class TransportError(RuntimeError):
    pass


class CacheOverflow(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Identity                                                                    #
# --------------------------------------------------------------------------- #
def config_hash(cfg):
    keys = ["num_hidden_layers", "hidden_size", "intermediate_size", "num_attention_heads",
            "num_key_value_heads", "head_dim", "vocab_size", "tie_word_embeddings", "rope_theta",
            "rms_norm_eps", "max_position_embeddings"]
    blob = json.dumps({k: getattr(cfg, k) for k in keys}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def software_identity():
    return {"torch": torch.__version__, "transformers": transformers.__version__,
            "cuda": torch.version.cuda, "protocol": PROTOCOL_VERSION}


def load_config(model_dir):
    cfg = transformers.AutoConfig.from_pretrained(model_dir)
    cfg._attn_implementation = "sdpa"
    return cfg


def check_config(cfg, expected):
    bad = {k: (getattr(cfg, k, None), v) for k, v in expected.items()
           if k != "torch_dtype" and getattr(cfg, k, None) != v}
    if str(getattr(cfg, "torch_dtype", "")).replace("torch.", "") != expected.get("torch_dtype", "bfloat16"):
        bad["torch_dtype"] = (cfg.torch_dtype, expected.get("torch_dtype"))
    if bad:
        raise ValueError(f"config mismatch: {bad}")


# --------------------------------------------------------------------------- #
# Weights                                                                     #
# --------------------------------------------------------------------------- #
def weight_index(model_dir):
    d = Path(model_dir)
    idx = d / "model.safetensors.index.json"
    if idx.exists():
        return {k: d / v for k, v in json.loads(idx.read_text())["weight_map"].items()}
    from safetensors import safe_open
    single = d / "model.safetensors"
    with safe_open(str(single), framework="pt") as f:
        return {k: single for k in f.keys()}


class _Reader:
    def __init__(self, index, device):
        from safetensors import safe_open
        self.index, self.handles = index, {}
        self.dev = str(device) if str(device).startswith("cuda") else "cpu"
        self._open = safe_open

    def get(self, name):
        path = self.index[name]
        h = self.handles.get(path)
        if h is None:
            h = self.handles[path] = self._open(str(path), framework="pt", device=self.dev)
        return h.get_tensor(name)

    def close(self):
        self.handles.clear()


class Slice:
    """The owned parts of a Qwen3ForCausalLM, materialised on `device`; everything else stays on meta."""

    def __init__(self, model_dir, layer_ids, with_embed, with_head, device="cpu", dtype=torch.bfloat16,
                 expected_config=None):
        self.cfg = load_config(model_dir)
        if expected_config:
            check_config(self.cfg, expected_config)
        self.device, self.dtype = torch.device(device), dtype
        with torch.device("meta"):
            skel = mq.Qwen3ForCausalLM(self.cfg)
        skel.to(dtype).eval()
        self.skel_model = skel.model          # meta weights; used only for _update_causal_mask
        self.n_layers = self.cfg.num_hidden_layers
        self.layer_ids = sorted(layer_ids)
        self.layers = {i: skel.model.layers[i] for i in self.layer_ids}
        self.embed = skel.model.embed_tokens if with_embed else None
        self.norm = skel.model.norm if with_head else None
        self.lm_head = skel.lm_head if with_head else None
        index = weight_index(model_dir)
        reader = _Reader(index, self.device)
        self.param_bytes = 0
        mods = [(f"model.layers.{i}.", self.layers[i]) for i in self.layer_ids]
        if self.embed is not None:
            mods.append(("model.embed_tokens.", self.embed))
        if self.norm is not None:
            mods.append(("model.norm.", self.norm))
        tied_head = False
        if self.lm_head is not None:
            if "lm_head.weight" in index:
                mods.append(("lm_head.", self.lm_head))
            elif self.cfg.tie_word_embeddings:
                tied_head = True
            else:
                raise KeyError("lm_head.weight missing from an untied checkpoint")
        with torch.no_grad():
            for prefix, mod in mods:
                mod.to_empty(device=self.device)
                for pname, p in mod.named_parameters():
                    t = reader.get(prefix + pname)
                    if tuple(t.shape) != tuple(p.shape):
                        raise ValueError(f"{prefix + pname}: shape {tuple(t.shape)} != {tuple(p.shape)}")
                    p.copy_(t.to(dtype))
                    self.param_bytes += p.numel() * p.element_size()
            if tied_head:
                if self.embed is None:
                    emb = torch.nn.Embedding(self.cfg.vocab_size, self.cfg.hidden_size, device=self.device,
                                             dtype=dtype)
                    emb.weight.copy_(reader.get("model.embed_tokens.weight").to(dtype))
                    w = emb.weight
                    self.param_bytes += w.numel() * w.element_size()
                else:
                    w = self.embed.weight
                self.lm_head.to_empty(device=self.device)
                self.lm_head.weight = w
        reader.close()
        self.rotary = mq.Qwen3RotaryEmbedding(config=self.cfg, device=self.device)

    @torch.inference_mode()
    def new_cache(self, max_len, layer_ids=None):
        return SliceKVCache(layer_ids if layer_ids is not None else self.layer_ids, max_len,
                            self.cfg.num_key_value_heads, self.cfg.head_dim, self.dtype, self.device)

    @torch.inference_mode()
    def run_layers(self, h, layer_ids, start, cache):
        """Mirror of Qwen3Model.forward's layer loop for a contiguous sub-range of layers."""
        n = h.shape[1]
        cache_position = torch.arange(start, start + n, device=h.device)
        position_ids = cache_position.unsqueeze(0)
        mask = self.skel_model._update_causal_mask(None, h, cache_position, cache, False)
        pe = self.rotary(h, position_ids)
        for i in layer_ids:
            h = self.layers[i](h, attention_mask=mask, position_ids=position_ids, past_key_value=cache,
                               output_attentions=False, use_cache=cache is not None,
                               cache_position=cache_position, position_embeddings=pe)[0]
        return h


class SliceKVCache(Cache):
    """Preallocated KV for the owned layers only; returns views of the filled prefix.

    Same tensor values and shapes as DynamicCache (concatenation semantics) without reallocation,
    so edge peak memory is fixed at allocation time. `seen` is the committed context length; it is
    advanced by the owner after all of its layers have processed a step.
    """

    def __init__(self, layer_ids, max_len, n_kv, head_dim, dtype, device):
        super().__init__()
        self.max_len, self.seen = max_len, 0
        shape = (1, n_kv, max_len, head_dim)
        self.k = {i: torch.zeros(shape, dtype=dtype, device=device) for i in layer_ids}
        self.v = {i: torch.zeros(shape, dtype=dtype, device=device) for i in layer_ids}
        self.fill = {i: 0 for i in layer_ids}

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        n, s = key_states.shape[-2], self.fill[layer_idx]
        if s != self.seen:
            raise CacheOverflow(f"layer {layer_idx} fill {s} != committed length {self.seen}")
        if s + n > self.max_len:
            raise CacheOverflow(f"layer {layer_idx}: {s}+{n} > max_len {self.max_len}")
        self.k[layer_idx][:, :, s:s + n].copy_(key_states)
        self.v[layer_idx][:, :, s:s + n].copy_(value_states)
        self.fill[layer_idx] = s + n
        return self.k[layer_idx][:, :, :s + n], self.v[layer_idx][:, :, :s + n]

    def get_seq_length(self, layer_idx=0):
        return self.seen

    def get_max_cache_shape(self):
        return None

    def commit(self, n):
        self.seen += n

    def reset(self):
        self.seen = 0
        for i in self.fill:
            self.fill[i] = 0

    def nbytes(self):
        return sum(t.numel() * t.element_size() for d in (self.k, self.v) for t in d.values())


# --------------------------------------------------------------------------- #
# Wire                                                                        #
# --------------------------------------------------------------------------- #
def _recv_exact(sock, buf):
    view, got = memoryview(buf), 0
    while got < len(view):
        r = sock.recv_into(view[got:])
        if r == 0:
            raise TransportError("peer closed the connection")
        got += r


def send_msg(sock, mtype, req=0, start=0, n=0, hidden=0, flags=0, payload=b"", t=(0, 0, 0), seen=0):
    payload = memoryview(payload).cast("B") if not isinstance(payload, (bytes, bytearray)) else payload
    sock.sendall(HDR.pack(MAGIC, PROTOCOL_VERSION, mtype, req, start, n, hidden, flags, len(payload),
                          t[0], t[1], t[2], seen))
    if len(payload):
        sock.sendall(payload)
    return HDR.size + len(payload)


def recv_header(sock):
    raw = bytearray(HDR.size)
    _recv_exact(sock, raw)
    f = HDR.unpack(raw)
    if f[0] != MAGIC or f[1] != PROTOCOL_VERSION:
        raise TransportError(f"bad header {f[:2]}")
    keys = ["magic", "version", "type", "req", "start", "n", "hidden", "flags", "payload", "t_recv_ns",
            "t_compute_ns", "t_prep_ns", "seen"]
    return dict(zip(keys, f))


class HostBuffers:
    """Reusable (pinned on CUDA) host staging buffers for raw bf16 activations."""

    def __init__(self, device, hidden):
        self.cuda = torch.device(device).type == "cuda"
        self.hidden, self.cap = hidden, 0
        self.out = self.inp = None

    def ensure(self, n):
        if n <= self.cap:
            return
        self.cap = max(n, 2 * self.cap)
        mk = lambda: torch.empty(self.cap * self.hidden, dtype=torch.bfloat16, pin_memory=self.cuda)
        self.out, self.inp = mk(), mk()

    @staticmethod
    def as_bytes(t, numel):
        return memoryview(t[:numel].view(torch.int16).numpy()).cast("B")

    def d2h(self, x):
        if x.dtype != torch.bfloat16:   # E0 §7: the cut tensor crosses as raw bf16; never cast
            raise TypeError(f"wire dtype is bfloat16 only; refusing to send {x.dtype}")
        numel = x.numel()
        self.ensure(numel // self.hidden)
        self.out[:numel].copy_(x.reshape(-1), non_blocking=self.cuda)
        if self.cuda:
            torch.cuda.current_stream(x.device).synchronize()
        return self.as_bytes(self.out, numel)

    def recv_into(self, sock, n):
        numel = n * self.hidden
        self.ensure(n)
        _recv_exact(sock, self.as_bytes(self.inp, numel))
        return self.inp[:numel]

    def h2d(self, host, n, device):
        y = torch.empty((1, n, self.hidden), dtype=torch.bfloat16, device=device)
        y.view(-1).copy_(host, non_blocking=self.cuda)
        return y


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


class AppShaper:
    """Application-level link emulation on the edge side of a strictly synchronous protocol:
    each round trip is delayed by rtt_ms plus serialisation of both payloads at bw_mbps.
    Used only where tc netem is unavailable (PREREG §6.4); recorded as shaping="app"."""

    def __init__(self, rtt_ms=None, bw_mbps=None):
        self.rtt_s = (rtt_ms or 0) / 1e3
        self.bps = bw_mbps * 1e6 if bw_mbps else None

    def _wait(self, seconds):
        if seconds <= 0:
            return
        end = time.perf_counter() + seconds
        if seconds > 0.002:
            time.sleep(seconds - 0.001)
        while time.perf_counter() < end:
            pass

    def before_send(self, nbytes):
        self._wait(self.rtt_s / 2 + (nbytes * 8 / self.bps if self.bps else 0))

    def after_recv(self, nbytes):
        self._wait(self.rtt_s / 2 + (nbytes * 8 / self.bps if self.bps else 0))


# --------------------------------------------------------------------------- #
# Middle segments as seen from the edge                                       #
# --------------------------------------------------------------------------- #
class LocalMiddle:
    """In-process middle (reference / unit tests): same layers, same cache semantics."""

    def __init__(self, sl, layer_ids, max_len):
        self.sl, self.ids = sl, layer_ids
        self.cache = sl.new_cache(max_len, layer_ids) if max_len else None
        self.last = {}

    def reset(self):
        if self.cache:
            self.cache.reset()

    def forward(self, h, start, use_cache):
        c = self.cache if use_cache else None
        y = self.sl.run_layers(h, self.ids, start, c)
        if c is not None:
            c.commit(h.shape[1])
        self.last = {"bytes_out": 0, "bytes_in": 0}
        return y

    def close(self):
        pass


class RemoteMiddle:
    """Middle layers on a SegmentServer over TCP (TCP_NODELAY), raw bf16 both ways."""

    def __init__(self, host, port, hello, device, hidden, shaper=None, timeout_s=600):
        self.sock = socket.create_connection((host, port), timeout=timeout_s)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.device, self.hidden = torch.device(device), hidden
        self.buf = HostBuffers(device, hidden)
        self.shaper = shaper or AppShaper()
        self.req = 0
        self.bytes_sent = self.bytes_recv = 0
        body = json.dumps(hello).encode()
        self.bytes_sent += send_msg(self.sock, T_HELLO, payload=body)
        h = recv_header(self.sock)
        payload = bytearray(h["payload"])
        _recv_exact(self.sock, payload)
        self.bytes_recv += HDR.size + h["payload"]
        if h["type"] == T_ERROR:
            raise TransportError(f"server refused HELLO: {payload.decode()}")
        if h["type"] != T_HELLO_OK:
            raise TransportError(f"unexpected reply {h['type']} to HELLO")
        self.server_identity = json.loads(payload)
        self.last = {}

    def reset(self):
        self.req += 1
        self.bytes_sent += send_msg(self.sock, T_RESET, req=self.req)
        h = recv_header(self.sock)
        self.bytes_recv += HDR.size
        if h["type"] != T_RESET_OK:
            raise TransportError(f"bad RESET reply {h['type']}")

    def forward(self, h, start, use_cache):
        n = h.shape[1]
        t0 = time.perf_counter_ns()
        payload = self.buf.d2h(h)
        t1 = time.perf_counter_ns()
        nbytes = HDR.size + len(payload)
        self.shaper.before_send(nbytes)
        t2 = time.perf_counter_ns()
        self.bytes_sent += send_msg(self.sock, T_FWD, req=self.req, start=start, n=n, hidden=self.hidden,
                                    flags=F_USE_CACHE if use_cache else 0, payload=payload)
        r = recv_header(self.sock)
        if r["type"] == T_ERROR:
            msg = bytearray(r["payload"])
            _recv_exact(self.sock, msg)
            raise TransportError(f"server error: {msg.decode()}")
        if r["type"] != T_FWD_OUT or r["n"] != n or r["start"] != start or r["payload"] != n * self.hidden * 2:
            raise TransportError(f"bad FWD_OUT {r}")
        host = self.buf.recv_into(self.sock, n)
        t3 = time.perf_counter_ns()
        self.bytes_recv += HDR.size + r["payload"]
        self.shaper.after_recv(HDR.size + r["payload"])
        t4 = time.perf_counter_ns()
        y = self.buf.h2d(host, n, self.device)
        _sync(self.device)
        t5 = time.perf_counter_ns()
        server_ns = r["t_recv_ns"] + r["t_compute_ns"] + r["t_prep_ns"]
        self.last = {"ser_ns": (t1 - t0) + (t5 - t4), "shape_ns": (t2 - t1) + (t4 - t3),
                     "rt_ns": t3 - t2, "server_ns": server_ns, "server_compute_ns": r["t_compute_ns"],
                     "net_ns": (t3 - t2) - server_ns, "bytes_out": nbytes, "bytes_in": HDR.size + r["payload"],
                     "server_seen": r["seen"]}
        return y

    def close(self):
        try:
            send_msg(self.sock, T_BYE)
        except OSError:
            pass
        self.sock.close()


# --------------------------------------------------------------------------- #
# Edge runner                                                                 #
# --------------------------------------------------------------------------- #
class EdgeRunner:
    """Edge half of the U-split: embedding + head layers, middle (local or remote), tail layers + LM head."""

    def __init__(self, sl, k, j, middle, max_len=None):
        assert sl.embed is not None and sl.lm_head is not None
        L = sl.n_layers
        if not (k >= 1 and j >= 1 and k + j <= L):
            raise ValueError(f"illegal split (k={k}, j={j}): raw text must stay on the edge (k, j >= 1)")
        self.sl, self.k, self.j, self.middle = sl, k, j, middle
        self.head_ids, self.tail_ids = list(range(k)), list(range(L - j, L))
        missing = set(self.head_ids + self.tail_ids) - set(sl.layer_ids)
        if missing:
            raise ValueError(f"edge slice lacks layers {sorted(missing)}")
        self.cache = sl.new_cache(max_len, self.head_ids + self.tail_ids) if max_len else None
        self.steps = []

    def reset(self):
        if self.cache:
            self.cache.reset()
        self.middle.reset()
        self.steps = []

    @torch.inference_mode()
    def forward_hidden(self, ids, start, use_cache, record=True):
        dev = self.sl.device
        c = self.cache if use_cache else None
        if use_cache and c is None:
            raise ValueError("runner built without a cache")
        t0 = time.perf_counter_ns()
        x = self.sl.embed(ids)
        x = self.sl.run_layers(x, self.head_ids, start, c)
        _sync(dev)
        t1 = time.perf_counter_ns()
        y = self.middle.forward(x, start, use_cache)
        t2 = time.perf_counter_ns()
        y = self.sl.run_layers(y, self.tail_ids, start, c)
        y = self.sl.norm(y)
        _sync(dev)
        t3 = time.perf_counter_ns()
        if c is not None:
            c.commit(ids.shape[1])
        if record:
            m = dict(self.middle.last)
            m.update(n=ids.shape[1], start=start, edge_head_ns=t1 - t0, edge_tail_ns=t3 - t2,
                     middle_ns=t2 - t1, total_ns=t3 - t0, lm_head_ns=0)
            self.steps.append(m)
        return y

    @torch.inference_mode()
    def last_logits(self, hidden):
        return self.sl.lm_head(hidden[:, -1:, :])[0, -1]

    @torch.inference_mode()
    def generate(self, prompt_ids, max_new, chunk, eos_ids=(), force_len=False, margins=False):
        """Greedy decoding (argmax over the bf16 logits, first index on ties).

        force_len: ignore EOS and emit exactly max_new tokens (latency workload). Afterwards
        self.token_ns[i] = ns from the call to token i, self.steps = per-forward components, and
        self.margins[i] = top-1 minus top-2 log-probability at token i (if margins=True).
        """
        self.reset()
        dev = self.sl.device
        ids = torch.as_tensor(prompt_ids, dtype=torch.long, device=dev).view(1, -1)
        P = ids.shape[1]
        if P + max_new > self.cache.max_len:
            raise CacheOverflow(f"prompt {P} + {max_new} > cache {self.cache.max_len}")
        t_start = time.perf_counter_ns()
        for s in range(0, P, chunk):
            h = self.forward_hidden(ids[:, s:s + chunk], s, True)
        out, stamps, self.margins = [], [], []
        pos = P
        while True:
            t_h = time.perf_counter_ns()
            logits = self.last_logits(h).float()
            tok = int(torch.argmax(logits))
            if margins:
                top = torch.topk(torch.log_softmax(logits, -1), 2).values
                self.margins.append(float(top[0] - top[1]))
            now = time.perf_counter_ns()
            self.steps[-1]["lm_head_ns"] = now - t_h
            out.append(tok)
            stamps.append(now - t_start)
            if len(out) >= max_new or (not force_len and tok in eos_ids):
                break
            h = self.forward_hidden(torch.tensor([[tok]], device=dev), pos, True)
            pos += 1
        self.token_ns = stamps
        return out

    def teacher_forced_hidden(self, ids):
        """Final normed hidden states for the whole sequence, one step, no cache (E0-Q1)."""
        self.middle.reset()
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.sl.device).view(1, -1)
        return self.forward_hidden(ids, 0, False)[0]


# --------------------------------------------------------------------------- #
# Server                                                                      #
# --------------------------------------------------------------------------- #
class SegmentServer:
    """Serves FWD requests for an active contiguous layer range; one client at a time, batch 1."""

    def __init__(self, sl, model_id, max_len, echo=False):
        self.sl, self.model_id, self.max_len, self.echo = sl, model_id, max_len, echo
        self.device = sl.device if sl else torch.device("cpu")
        self.hidden = sl.cfg.hidden_size if sl else None
        self.active, self.cache = [], None
        self.log = []

    def identity(self):
        d = {"model_id": self.model_id, "echo": self.echo, "software": software_identity(),
             "host": socket.gethostname(), "device": str(self.device),
             "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "cpu"}
        if self.sl:
            d.update(config_hash=config_hash(self.sl.cfg), loaded_layers=[self.sl.layer_ids[0],
                                                                          self.sl.layer_ids[-1] + 1])
        return d

    def _hello(self, req):
        if self.echo:
            self.hidden = req["hidden"]
            return {"active": [req["k"], req["n_layers"] - req["j"]]}
        if req["model_id"] != self.model_id:
            raise ValueError(f"model {req['model_id']} != served {self.model_id}")
        if req["config_hash"] != config_hash(self.sl.cfg):
            raise ValueError("config hash mismatch")
        if req["software"]["torch"] != torch.__version__ or req["software"]["transformers"] != \
                transformers.__version__ or req["software"]["cuda"] != torch.version.cuda:
            raise ValueError(f"software mismatch: edge {req['software']} vs server {software_identity()}")
        lo, hi = req["k"], self.sl.n_layers - req["j"]
        want = list(range(lo, hi))
        if not set(want) <= set(self.sl.layer_ids):
            raise ValueError(f"requested middle [{lo}, {hi}) not loaded ({self.sl.layer_ids[0]}..)")
        self.active = want
        self.cache = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self.cache = self.sl.new_cache(req.get("max_len", self.max_len), want)
        return {"active": [lo, hi], "kv_bytes": self.cache.nbytes()}

    def handle(self, conn):
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        buf = None
        while True:
            try:
                h = recv_header(conn)
            except TransportError:
                return
            try:
                if h["type"] == T_HELLO:
                    p = bytearray(h["payload"])
                    _recv_exact(conn, p)
                    info = self._hello(json.loads(p))
                    info.update(self.identity())
                    buf = HostBuffers(self.device, self.hidden)
                    send_msg(conn, T_HELLO_OK, payload=json.dumps(info).encode())
                elif h["type"] == T_RESET:
                    if self.cache:
                        self.cache.reset()
                    send_msg(conn, T_RESET_OK, req=h["req"])
                elif h["type"] == T_FWD:
                    n, use_cache = h["n"], bool(h["flags"] & F_USE_CACHE)
                    if h["hidden"] != self.hidden or h["payload"] != n * self.hidden * 2:
                        raise ValueError(f"bad FWD shape {h}")
                    t0 = time.perf_counter_ns()
                    host = buf.recv_into(conn, n)
                    if self.echo:   # identity: reply with the received bytes
                        payload = HostBuffers.as_bytes(host, n * self.hidden)
                        t1 = t2 = t3 = time.perf_counter_ns()
                        seen = h["start"] + n
                    else:
                        if use_cache and h["start"] != self.cache.seen:
                            raise ValueError(f"position {h['start']} != server cache length {self.cache.seen}")
                        if not use_cache and h["start"] != 0:
                            raise ValueError("cache-less FWD must start at 0")
                        x = buf.h2d(host, n, self.device)
                        _sync(self.device)
                        t1 = time.perf_counter_ns()
                        c = self.cache if use_cache else None
                        y = self.sl.run_layers(x, self.active, h["start"], c)
                        if c is not None:
                            c.commit(n)
                        _sync(self.device)
                        t2 = time.perf_counter_ns()
                        payload = buf.d2h(y)
                        t3 = time.perf_counter_ns()
                        seen = self.cache.seen if use_cache else 0
                    send_msg(conn, T_FWD_OUT, req=h["req"], start=h["start"], n=n, hidden=self.hidden,
                             payload=payload, t=(t1 - t0, t2 - t1, t3 - t2), seen=seen)
                elif h["type"] == T_BYE:
                    return
                else:
                    raise ValueError(f"unexpected message type {h['type']}")
            except (TransportError, OSError):
                return
            except Exception as e:  # report to the edge, then drop the connection
                try:
                    send_msg(conn, T_ERROR, payload=f"{type(e).__name__}: {e}".encode())
                finally:
                    return

    def serve_forever(self, host, port, ready_file=None, once=False):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(1)
        if ready_file:
            Path(ready_file).write_text(json.dumps({"port": srv.getsockname()[1], **self.identity()}))
        print(f"segment server ready on {host}:{srv.getsockname()[1]} {json.dumps(self.identity())}", flush=True)
        while True:
            conn, _ = srv.accept()
            with conn:
                self.handle(conn)
            if once:
                srv.close()
                return


def hello_for(sl, model_id, k, j, max_len):
    return {"model_id": model_id, "config_hash": config_hash(sl.cfg), "software": software_identity(),
            "k": k, "j": j, "n_layers": sl.n_layers, "hidden": sl.cfg.hidden_size, "max_len": max_len,
            "wire_dtype": "bfloat16"}


def edge_slice(model_dir, k, j, device, dtype=torch.bfloat16, expected_config=None):
    cfg = load_config(model_dir)
    L = cfg.num_hidden_layers
    return Slice(model_dir, list(range(k)) + list(range(L - j, L)), True, True, device, dtype, expected_config)


def parse_range(s):
    a, b = s.split("-")
    return list(range(int(a), int(b)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--model-dir")
    s.add_argument("--model-id", default="Qwen/Qwen3-32B@9216db5781bf21249d130ec9da846c4624c16137")
    s.add_argument("--layers", help="half-open loaded range a-b, e.g. 3-61")
    s.add_argument("--echo", action="store_true")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=29500)
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--max-len", type=int, default=17920)
    s.add_argument("--ready-file")
    s.add_argument("--once", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "serve":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)   # same mode as the Q clients (E0 §7)
        if a.echo:
            srv = SegmentServer(None, "echo", a.max_len, echo=True)
        else:
            from e0_spec import QWEN3_32B_CONFIG
            expected = QWEN3_32B_CONFIG if a.model_id.startswith("Qwen/Qwen3-32B@") else None
            sl = Slice(a.model_dir, parse_range(a.layers), False, False, a.device, torch.bfloat16, expected)
            srv = SegmentServer(sl, a.model_id, a.max_len)
        srv.serve_forever(a.host, a.port, a.ready_file, a.once)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
