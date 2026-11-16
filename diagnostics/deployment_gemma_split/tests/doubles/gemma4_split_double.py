"""TEST DOUBLE, never a target system: ADAPTER_INFO["test_double"] is True and gs_adapter.check_info refuses it for
target runs; every output it produces in tests is profile_role "dev".

A real two-part split of a TINY RANDOM Gemma 4 text model (transformers' Gemma4ForCausalLM; tested with transformers
5.19.0, torch 2.14.1 CPU) over local TCP, implementing the GEMMA-SPLIT adapter contract, so that the harness can be
exercised end to end on CPU. Edge: token embedding + per-layer input embeddings (PLE) + layers [0, cut). Server:
layers [cut, L) + final norm + LM head (+ softcapping) + greedy argmax. Wire, edge -> server: hidden_states,
per_layer_inputs of the server's layers, and the shared KV of every attention type whose KV-sharing source layer is on
the edge; server -> edge: next_token (generation) or logprobs (teacher-forced test path). No KV cache: each step
recomputes the whole sequence (a slow but legitimate split). This is the kind of Gemma-4 split structure the Stage-0
capture must record: it differs from the Qwen3 E1-OPT U-split in topology, wire content and server ownership.

Faults (env GS_DOUBLE_FAULTS or options["faults"], comma-separated) let the tests check that the harness catches them:
  edge:   drop_ple (zeros sent), omit_shared_kv (not sent), wire_bf16_silent (precision lost, float32 on the wire),
          wire_bf16 (bfloat16 on the wire, undeclared dtype), undeclared_extra (token ids sent)
  server: skip_layer (first server layer skipped), position_offset (positions shifted by one)
"""
import json
import os
import socket
import string
import struct
import sys
import time
from collections import UserDict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import gs_adapter as AD  # noqa: E402

ADAPTER_INFO = {
    "contract_version": 1,
    "implementation": "gemma4-tiny-split-double (test double)",
    "configuration_id": "double-cut3",
    "capabilities": ["greedy_tokens", "stream_tokens", "boundary_audit", "teacher_forced_logprobs", "ignore_eos", "segment_step",
                     "step_trace", "unsplit_same_runtime", "in_host_split"],
    "test_double": True,
    "notes": "CPU test double for the harness; never a target system",
    "missing_capability_justification": {},
}

LAYER_TYPES = ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "sliding_attention",
               "full_attention", "sliding_attention", "full_attention"]
TINY = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=8, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            global_head_dim=16, hidden_size_per_layer_input=8, num_kv_shared_layers=3, sliding_window=8, max_position_embeddings=4096,
            layer_types=LAYER_TYPES, final_logit_softcapping=30.0, initializer_range=0.5, pad_token_id=0, eos_token_id=1, bos_token_id=2)
SPECIALS = ["<pad>", "<eos>", "<bos>", "<unk>", "<start_of_turn>", "<end_of_turn>"]
CHAT_TEMPLATE = ("{{ bos_token }}{% for m in messages %}<start_of_turn>{{ m['role'] }}\n{{ m['content'] }}<end_of_turn>\n{% endfor %}"
                 "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}")
CUT = 3
LEN = struct.Struct("!I")


def make_tiny_checkpoint(out_dir, seed=0):
    """Character-level tokenizer with a chat template + a tiny random Gemma4ForCausalLM (float32), saved to out_dir."""
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig, PreTrainedTokenizerFast
    chars = sorted(set(string.printable) - set(" \t\n\r\x0b\x0c")) + [" ", "\n"]
    vocab = {s: i for i, s in enumerate(SPECIALS + chars)}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    tk.add_special_tokens(SPECIALS)
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<bos>", eos_token="<eos>", pad_token="<pad>", unk_token="<unk>")
    tok.chat_template = CHAT_TEMPLATE
    tok.save_pretrained(out_dir)
    torch.manual_seed(seed)
    cfg = Gemma4TextConfig(vocab_size=len(vocab), vocab_size_per_layer_input=len(vocab), **TINY)
    Gemma4ForCausalLM(cfg).eval().save_pretrained(out_dir)
    return out_dir


# --------------------------------------------------------------------------- #
# Wire                                                                        #
# --------------------------------------------------------------------------- #
def send_msg(sock, kind, tensors=(), meta=None):
    specs, blobs = [], []
    for name, t in tensors:
        b = t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
        specs.append({"name": name, "dtype": str(t.dtype).replace("torch.", ""), "shape": list(t.shape), "nbytes": len(b)})
        blobs.append(b)
    head = json.dumps({"kind": kind, "tensors": specs, "meta": meta or {}}).encode()
    payload = LEN.pack(len(head)) + head + b"".join(blobs)
    sock.sendall(LEN.pack(len(payload)) + payload)
    return {"tensors": [{k: s[k] for k in ("name", "dtype", "shape")} for s in specs], "bytes": len(payload) + LEN.size}


def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed")
        buf += chunk
    return bytes(buf)


def recv_msg(sock):
    (n,) = LEN.unpack(_recv_exact(sock, LEN.size))
    payload = _recv_exact(sock, n)
    (h,) = LEN.unpack(payload[:LEN.size])
    head = json.loads(payload[LEN.size:LEN.size + h])
    off, tensors = LEN.size + h, {}
    for s in head["tensors"]:
        b = bytearray(payload[off:off + s["nbytes"]])
        off += s["nbytes"]
        tensors[s["name"]] = torch.frombuffer(b, dtype=getattr(torch, s["dtype"])).reshape(s["shape"]).clone()
    spec = [{k: s[k] for k in ("name", "dtype", "shape")} for s in head["tensors"]]
    return head["kind"], tensors, head["meta"], spec, n + LEN.size


# --------------------------------------------------------------------------- #
# The two parts                                                               #
# --------------------------------------------------------------------------- #
class Halves:
    def __init__(self, model_dir, cut, device="cpu"):
        from transformers import AutoModelForCausalLM
        from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
        self.mask_fns = {"full_attention": create_causal_mask, "sliding_attention": create_sliding_window_causal_mask}
        self.m = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).to(device).eval()
        self.tm, self.cfg, self.cut, self.device = self.m.model, self.m.config, cut, device
        L, n = self.cfg.num_hidden_layers, self.cfg.num_kv_shared_layers
        first = L - n
        self.kv_source = {lt: max(j for j in range(first) if self.cfg.layer_types[j] == lt) for lt in set(self.cfg.layer_types[first:])}
        self.server_needs_kv = sorted(lt for lt, src in self.kv_source.items() if src < cut)

    def _ctx(self, like, pos):
        kw = {"config": self.cfg, "inputs_embeds": like, "attention_mask": None, "past_key_values": None, "position_ids": pos}
        masks = {lt: fn(**kw) for lt, fn in self.mask_fns.items()}
        pe = {lt: self.tm.rotary_emb(like, pos, lt) for lt in set(self.cfg.layer_types)}
        return masks, pe

    def _layer(self, i, h, ple_i, shared, masks, pe, pos):
        lt = self.cfg.layer_types[i]
        return self.tm.layers[i](h, ple_i, shared_kv_states=shared, position_embeddings=pe[lt], attention_mask=masks[lt],
                                 position_ids=pos, past_key_values=None)

    @torch.inference_mode()
    def edge_forward(self, ids, faults=()):
        x = torch.tensor([ids], device=self.device)
        emb = self.tm.embed_tokens(x)
        ple = self.tm.project_per_layer_inputs(emb, self.tm.get_per_layer_inputs(x, emb))
        pos = torch.arange(len(ids), device=self.device).unsqueeze(0)
        masks, pe = self._ctx(emb, pos)
        shared, h = UserDict(), emb
        for i in range(self.cut):
            h = self._layer(i, h, ple[:, :, i, :], shared, masks, pe, pos)
        server_ple = ple[:, :, self.cut:, :]
        if "drop_ple" in faults:
            server_ple = torch.zeros_like(server_ple)
        if "wire_bf16_silent" in faults:
            h = h.to(torch.bfloat16).to(torch.float32)
        if "wire_bf16" in faults:
            h = h.to(torch.bfloat16)
        out = [("hidden_states", h), ("per_layer_inputs", server_ple)]
        if "omit_shared_kv" not in faults:
            for lt in self.server_needs_kv:
                k, v = shared[lt]
                out += [(f"shared_kv_{lt}_k", k), (f"shared_kv_{lt}_v", v)]
        if "undeclared_extra" in faults:
            out.append(("input_ids", x))
        return out

    @torch.inference_mode()
    def server_forward(self, t, n, faults=()):
        h = t["hidden_states"].to(torch.float32)
        ple = t["per_layer_inputs"]
        pos = torch.arange(n, device=self.device).unsqueeze(0) + (1 if "position_offset" in faults else 0)
        masks, pe = self._ctx(h, pos)
        shared = UserDict()
        for lt in self.server_needs_kv:
            if f"shared_kv_{lt}_k" in t:
                shared[lt] = (t[f"shared_kv_{lt}_k"], t[f"shared_kv_{lt}_v"])
        for i in range(self.cut + (1 if "skip_layer" in faults else 0), self.cfg.num_hidden_layers):
            h = self._layer(i, h, ple[:, :, i - self.cut, :], shared, masks, pe, pos)
        logits = self.m.lm_head(self.tm.norm(h))
        cap = self.cfg.final_logit_softcapping
        return torch.tanh(logits / cap) * cap if cap else logits

    def local(self, ids, faults=()):
        msg = dict(self.edge_forward(ids, faults))
        return self.server_forward(msg, len(ids), faults)

    @torch.inference_mode()
    def unsplit_logits(self, ids):
        return self.m(input_ids=torch.tensor([ids], device=self.device), use_cache=False).logits


def _generate(next_token, prompt_ids, max_new_tokens, eos_ids, ignore_eos, on_token):
    eos = set(int(e) for e in eos_ids)
    seq, toks = list(prompt_ids), []
    for step in range(max_new_tokens):
        tok = next_token(seq, step)
        toks.append(tok)
        on_token(step, tok)
        seq.append(tok)
        if tok in eos and not ignore_eos:
            break
    return {"tokens": toks, "stopped_on_eos": bool(toks and toks[-1] in eos and not ignore_eos)}


class _Runner:
    def __init__(self, logits_fn):
        self.logits_fn = logits_fn

    def stream(self, prompt_ids, max_new_tokens, eos_ids, ignore_eos, on_token):
        return _generate(lambda seq, step: int(torch.argmax(self.logits_fn(seq)[0, -1])), prompt_ids, max_new_tokens, eos_ids,
                         ignore_eos, on_token)

    def greedy(self, prompt_ids, max_new_tokens, eos_ids, ignore_eos=False):
        return self.stream(prompt_ids, max_new_tokens, eos_ids, ignore_eos, lambda i, t: None)

    def teacher_forced_logprobs(self, ids, start):
        yield start, torch.log_softmax(self.logits_fn(ids)[0, start:len(ids) - 1].float(), dim=-1)


class _Segment:
    """Isolated segment of one host. Edge: token ids in. Server: the boundary tensors the edge part produced for the
    prompt (recorded once, untimed), extended by repeating the last position for each decode step."""
    SEQ_DIM = {"hidden_states": 1, "per_layer_inputs": 1}

    def __init__(self, halves, role, prompt_ids, faults):
        self.h, self.role, self.seq, self.faults = halves, role, list(prompt_ids), faults
        self.inputs = "token ids" if role == "edge" else "recorded prompt boundary, last position repeated per decode step"
        self.msg = None if role == "edge" else dict(halves.edge_forward(self.seq, faults))
        self.n = len(self.seq)

    def _extend(self):
        for k, t in self.msg.items():
            d = self.SEQ_DIM.get(k, 2)      # shared KV: [1, kv_heads, n, head_dim]
            self.msg[k] = torch.cat([t, t.narrow(d, t.shape[d] - 1, 1)], dim=d)
        self.n += 1

    def prefill(self):
        if self.role == "edge":
            self.h.edge_forward(self.seq, self.faults)
        else:
            self.h.server_forward(self.msg, self.n, self.faults)

    def decode_step(self):
        if self.role == "edge":
            self.seq.append(self.seq[-1])
            self.h.edge_forward(self.seq, self.faults)
        else:
            self._extend()
            self.h.server_forward(self.msg, self.n, self.faults)


class DoubleAdapter(AD.SplitAdapter):
    def __init__(self, info, role, freeze, options):
        super().__init__(info, role)
        cap = freeze["split"]["capture"]
        self.cut = max(b for a, b in cap["ownership"]["edge"]["layers"])
        self.endpoint = options.get("endpoint") or cap["transport"]["endpoint"]
        self.device = options.get("device") or cap["gpu_placement"][role]["devices"][0]
        self.model_dir = options.get("model_dir") or freeze["checkpoint"][role]["model_dir"]
        self.faults = set(filter(None, (options.get("faults") or os.environ.get("GS_DOUBLE_FAULTS", "")).split(",")))
        self._h, self._sock, self._audit, self._trace, self._trace_on = None, None, [], [], False

    def _need(self, c):
        if c not in self.info["capabilities"]:
            raise AD.AdapterError(f"capability {c} not declared")

    def halves(self):
        if self._h is None:
            self._h = Halves(self.model_dir, self.cut, self.device)
        return self._h

    def _addr(self):
        host, port = self.endpoint.rsplit(":", 1)
        return host, int(port)

    # ---- server
    def serve(self, ready=None, stop=None):
        h = self.halves()
        srv = socket.create_server(self._addr())
        srv.settimeout(0.2)
        if ready:
            ready()
        try:
            while not (stop is not None and stop.is_set()):
                try:
                    conn, _ = srv.accept()
                except socket.timeout:
                    continue
                conn.settimeout(None)
                with conn:
                    self._handle(conn, h)
        finally:
            srv.close()

    def _handle(self, conn, h):
        while True:
            try:
                kind, t, meta, _, _ = recv_msg(conn)
            except ConnectionError:
                return
            if kind == "BYE":
                return
            t0 = time.perf_counter_ns()
            logits = h.server_forward(t, meta["n"], self.faults)
            server_ns = time.perf_counter_ns() - t0
            if kind == "TF":
                out = [("logprobs", torch.log_softmax(logits[0, meta["start"]:meta["n"] - 1].float(), dim=-1))]
            else:
                out = [("next_token", torch.argmax(logits[0, -1]).view(1).to(torch.int64))]
            send_msg(conn, "OUT", out, {"server_compute_ns": server_ns})

    def server_devices(self):
        return [self.device]

    # ---- edge
    def connect(self):
        self.halves()
        self._sock = socket.create_connection(self._addr(), timeout=120)

    def close(self):
        if self._sock is not None:
            try:
                send_msg(self._sock, "BYE")
            except OSError:
                pass                      # the server half is already gone (e.g. it crashed); nothing to say goodbye to
            finally:
                self._sock.close()
                self._sock = None

    def edge_devices(self):
        return [self.device]

    def _remote(self, kind, ids, phase, meta=None):
        t0 = time.perf_counter_ns()
        msg = self.halves().edge_forward(ids, self.faults)
        edge_ns = time.perf_counter_ns() - t0
        t1 = time.perf_counter_ns()
        a = send_msg(self._sock, kind, msg, {"n": len(ids), **(meta or {})})
        _, t, rmeta, spec, nb = recv_msg(self._sock)
        rt = time.perf_counter_ns() - t1
        self._audit += [{"direction": "edge_to_server", **a}, {"direction": "server_to_edge", "tensors": spec, "bytes": nb}]
        if self._trace_on:
            self._trace.append({"phase": phase, "edge_compute_ns": edge_ns, "server_compute_ns": rmeta["server_compute_ns"], "roundtrip_ns": rt})
        return t

    def _gen(self, prompt_ids, max_new_tokens, eos_ids, ignore_eos, on_token):
        if ignore_eos:
            self._need("ignore_eos")
        self._audit, self._trace = [], []
        return _generate(lambda seq, step: int(self._remote("FWD", seq, "prefill" if step == 0 else "decode")["next_token"][0]),
                         prompt_ids, max_new_tokens, eos_ids, ignore_eos, on_token)

    def greedy(self, prompt_ids, max_new_tokens, eos_ids, ignore_eos=False):
        self._need("greedy_tokens")
        return self._gen(prompt_ids, max_new_tokens, eos_ids, ignore_eos, lambda i, t: None)

    def stream(self, prompt_ids, max_new_tokens, eos_ids, ignore_eos, on_token):
        self._need("stream_tokens")
        return self._gen(prompt_ids, max_new_tokens, eos_ids, ignore_eos, on_token)

    def teacher_forced_logprobs(self, ids, start):
        self._need("teacher_forced_logprobs")
        self._audit = []
        yield start, self._remote("TF", ids, "prefill", {"start": start})["logprobs"]

    def boundary_audit(self):
        self._need("boundary_audit")
        return list(self._audit)

    def set_trace(self, on):
        self._need("step_trace")
        self._trace_on = bool(on)

    def step_trace(self):
        self._need("step_trace")
        return list(self._trace)

    def segment_runner(self, prompt_ids):
        self._need("segment_step")
        return _Segment(self.halves(), self.role, prompt_ids, self.faults)

    def unsplit_runner(self):
        self._need("unsplit_same_runtime")
        return _Runner(self.halves().unsplit_logits)

    def in_host_runner(self):
        self._need("in_host_split")
        return _Runner(lambda ids: self.halves().local(ids, self.faults))


def make_adapter(freeze, role, options):
    return DoubleAdapter(ADAPTER_INFO, role, freeze, options)
