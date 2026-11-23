"""ROUTER-INTEG: the backend dispatch interface.

Every model computation is the reused runtime's: the whole Hugging Face Qwen3ForCausalLM loaded by
k16_runtime.hf_model (bfloat16 as published, the stock KV cache), driven by a loop that performs the steps of
qsel_runtime.whole_generate (chunked prefill, one token per step, the LM head on the last position only, one clock
started when the token ids are handed over). The only change to that loop is the token choice: the frozen loop takes
the argmax; this protocol follows the original serving protocol and samples (temperature, optional top-k), with a
per-request seed, on the CPU from the device's logits.

Backends (one interface: generate(model, ids, max_new, seed, cutoff_s) -> result):
  EdgeBackend    both models whole in this process, on this host's device
  RemoteBackend  the edge side of the REMOTE wire: sends the token ids, the model key and the seed to the server, stamps
                 the output tokens as they arrive. No model and no GPU on the edge.
  MockBackend    CPU fixture for tests only. Its results carry mock=True and are refused as measurements.
The result always names the model the backend actually ran (`model`), and every caller checks it against the model it
asked for. Nothing here decides anything.
"""
import hmac
import json
import select
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rint_common as C  # noqa: E402
import rint_spec as S  # noqa: E402


class BackendError(RuntimeError):
    pass


class TransportError(BackendError):
    """The wire or the server could not be reached, or the connection was lost. Nothing is known about the model."""


def sampling_of(model):
    """The frozen sampling settings of a model: the request's temperature and top_p, and the model's own top-k default."""
    return {"temperature": S.TEMPERATURE, "top_p": S.TOP_P, "top_k": S.MODELS[model]["top_k"]}


def pick(torch, logits, sampling, gen):
    """One token from the last-position logits. temperature 0 is the frozen loop's argmax. Otherwise: temperature, then
    top-k if the model has one, then one draw from the CPU generator. top_p is 1.0 in this protocol (no nucleus cut)."""
    t = float(sampling["temperature"])
    if t <= 0.0:
        return int(torch.argmax(logits))
    if float(sampling["top_p"]) != 1.0:
        raise ValueError("top_p other than 1.0 is not part of this protocol")
    l = logits.double() / t
    if sampling.get("top_k"):
        v, idx = torch.topk(l, int(sampling["top_k"]))
        return int(idx[int(torch.multinomial(torch.softmax(v, -1).cpu(), 1, generator=gen))])
    return int(torch.multinomial(torch.softmax(l, -1).cpu(), 1, generator=gen))


def sync_fn(torch):
    def sync(device):
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
    return sync


def generate(torch, hf, ids, max_new, eos, device, chunk, cutoff_s, sync, sampling, seed, on_token=None, should_stop=None):
    """qsel_runtime.whole_generate with a sampled token choice. Ends on a token of `eos`, at `max_new`, at the first
    token stamped after `cutoff_s`, or when should_stop() is true (the REMOTE server's cancel)."""
    eos, cutoff_ns = set(int(x) for x in eos), int(cutoff_s * 1e9)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    with torch.inference_mode():
        t0 = time.perf_counter_ns()
        x = torch.as_tensor(ids, dtype=torch.long, device=device).view(1, -1)
        past, fwd = None, []

        def forward(inp):
            nonlocal past
            a = time.perf_counter_ns()
            out = hf.model(input_ids=inp, past_key_values=past, use_cache=True)
            past = out.past_key_values
            sync(device)
            fwd.append(time.perf_counter_ns() - a)
            return out.last_hidden_state
        for s in range(0, x.shape[1], chunk):
            h = forward(x[:, s:s + chunk])
        n_prefill = len(fwd)
        toks, stamps, cut, reason = [], [], False, None
        while True:
            tok = pick(torch, hf.lm_head(h[:, -1:, :])[0, -1].float(), sampling, gen)
            now = time.perf_counter_ns() - t0
            toks.append(tok)
            stamps.append(now)
            if on_token is not None:
                on_token(len(toks) - 1, tok, now)
            if now > cutoff_ns:
                cut, reason = True, "cutoff"
            elif tok in eos:
                reason = "eos"
            elif len(toks) >= max_new:
                reason = "cap"
            elif should_stop is not None and should_stop():
                reason = "cancel"
            if reason:
                break
            h = forward(torch.tensor([[tok]], device=device))
    dec = sorted(fwd[n_prefill:])
    return {"tokens": toks, "stamps_ns": stamps, "cutoff": cut, "stop": reason, "n_prefill_steps": n_prefill,
            "prefill_ns": sum(fwd[:n_prefill]), "decode_forward_ns_sum": sum(dec), "decode_forward_ns_median": dec[len(dec) // 2] if dec else None}


def times(res):
    """Seconds: time to first token, completion time, time per output token ((last - first) / (n - 1))."""
    st = res["stamps_ns"]
    return {"ttft_s": st[0] / 1e9, "total_s": st[-1] / 1e9, "per_token_s": (st[-1] - st[0]) / 1e9 / (len(st) - 1) if len(st) > 1 else None}


# --------------------------------------------------------------------------- #
# EDGE                                                                         #
# --------------------------------------------------------------------------- #
class EdgeBackend:
    mode = S.EDGE

    def __init__(self, torch, models, device, identities, chunk=None):
        """models: {model key: loaded Qwen3ForCausalLM}; identities: {model key: rint_models.model_id(...)}."""
        self.torch, self.models, self.device, self.identities = torch, dict(models), device, dict(identities)
        self.chunk, self.sync = int(chunk or S.PREFILL_CHUNK), sync_fn(torch)

    def available(self):
        return True

    def generate(self, model, ids, max_new, seed, cutoff_s):
        if model not in self.models:
            raise BackendError(f"model {model} is not loaded on the edge")
        res = generate(self.torch, self.models[model], ids, max_new, S.MODELS[model]["eos_ids"], self.device, self.chunk, cutoff_s,
                       self.sync, sampling_of(model), seed)
        res.update(model=model, model_identity=self.identities[model], mode=self.mode, seed=int(seed))
        return res

    def close(self):
        pass


# --------------------------------------------------------------------------- #
# REMOTE wire: token ids go to the server, output tokens come back one by one  #
# (framing, token record and accept loop are qsel_runtime's, imported)         #
# --------------------------------------------------------------------------- #
class RemoteServer:
    """Serves one edge at a time, one request at a time, for the models this process holds on its device."""

    def __init__(self, torch, models, device, identity, token, chunk=None, log=None):
        self.qrt = C.reused()["qrt"]
        self.torch, self.models, self.device = torch, dict(models), device
        self.identity, self.token, self.log = dict(identity), token, log or (lambda ev: None)
        self.chunk, self.sync, self.n_requests = int(chunk or S.PREFILL_CHUNK), sync_fn(torch), 0

    def handle(self, conn):
        q = self.qrt
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        ftype, body = q.recv_frame(conn)
        try:
            hello = json.loads(body.decode()) if ftype == q.R_HELLO else None
        except ValueError:
            hello = None
        why = None
        if not isinstance(hello, dict):
            why = "first message is not a HELLO"
        elif not hmac.compare_digest(str(hello.get("token", "")), self.token):
            why = "token mismatch"
        elif hello.get("spec_hash") != self.identity["spec_hash"] or hello.get("protocol") != S.PROTOCOL:
            why = "protocol or spec hash mismatch"
        if why:
            self.log({"event": "refused", "why": why})
            q.send_frame(conn, q.R_ERROR, why.encode())
            return
        self.log({"event": "hello_ok"})
        q.send_frame(conn, q.R_HELLO_OK, json.dumps(self.identity).encode())
        while True:
            ftype, body = q.recv_frame(conn)
            if ftype == q.R_BYE:
                self.log({"event": "bye", "n_requests": self.n_requests})
                return
            if ftype == q.R_CANCEL:                 # a cancel that crossed the end of its request
                continue
            if ftype != q.R_GEN:
                q.send_frame(conn, q.R_ERROR, b"unexpected message")
                return
            t_req = time.perf_counter_ns()
            req = json.loads(body.decode())
            model, ids, max_new = req.get("model"), [int(x) for x in req["ids"]], int(req["max_new"])
            if model not in self.models:
                q.send_frame(conn, q.R_ERROR, f"model {model} is not served here".encode())
                return
            if not ids or max_new < 1 or len(ids) + max_new > S.CONTEXT_TOKENS:
                q.send_frame(conn, q.R_ERROR, b"request exceeds the context")
                return

            def on_token(i, tok, now_ns):
                conn.sendall(q.FRAME.pack(q.RMAGIC, q.R_TOKEN, q.TOKEN.size) + q.TOKEN.pack(tok, time.perf_counter_ns() - t_req))

            def cancelled():
                r, _, _ = select.select([conn], [], [], 0)
                if not r:
                    return False
                ft, _b = q.recv_frame(conn)
                return ft == q.R_CANCEL
            # the server's own guard: the edge decides lateness on its clock and cancels; this only bounds a lost cancel
            try:
                res = generate(self.torch, self.models[model], ids, max_new, req["eos"], self.device, self.chunk, float(req["cutoff_s"]) + 10.0,
                               self.sync, req["sampling"], int(req["seed"]), on_token=on_token, should_stop=cancelled)
            except (OSError, q.WireError):
                raise
            except Exception as e:                  # a model failure is reported to the edge, which records it
                self.log({"event": "request_failed", "error": type(e).__name__})
                q.send_frame(conn, q.R_ERROR, f"generation failed: {type(e).__name__}: {e}"[:400].encode())
                return
            self.n_requests += 1
            q.send_frame(conn, q.R_END, json.dumps({"model": model, "n": len(res["tokens"]), "stop": res["stop"], "seed": int(req["seed"]),
                                                    "n_prefill_steps": res["n_prefill_steps"], "prefill_ns": res["prefill_ns"],
                                                    "decode_forward_ns_sum": res["decode_forward_ns_sum"],
                                                    "decode_forward_ns_median": res["decode_forward_ns_median"],
                                                    "server_total_ns": time.perf_counter_ns() - t_req}).encode())


class RemoteBackend:
    """Edge side of the REMOTE wire. Connects lazily; a failed connection makes available() false."""
    mode = S.REMOTE

    def __init__(self, addr, token, spec_hash, timeout_s=600):
        self.qrt = C.reused()["qrt"]
        self.addr, self.token, self.spec_hash, self.timeout_s = addr, token, spec_hash, timeout_s
        self.sock, self.server_identity, self.bytes_sent, self.bytes_recv = None, None, 0, 0

    def connect(self):
        q = self.qrt
        host, port = self.addr.rsplit(":", 1)
        try:
            sock = socket.create_connection((host, int(port)), timeout=self.timeout_s)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.bytes_sent += q.send_frame(sock, q.R_HELLO, json.dumps({"token": self.token, "spec_hash": self.spec_hash, "protocol": S.PROTOCOL}).encode())
            ftype, body = q.recv_frame(sock)
        except (OSError, q.WireError) as e:
            raise TransportError(f"the server does not answer: {type(e).__name__}: {e}"[:300]) from e
        self.bytes_recv += q.FRAME.size + len(body)
        if ftype != q.R_HELLO_OK:
            sock.close()
            raise BackendError("HELLO refused: " + body.decode(errors="replace")[:120])
        self.sock, self.server_identity = sock, json.loads(body.decode())
        return self.server_identity

    def available(self):
        if self.sock is not None:
            return True
        try:
            self.connect()
            return True
        except (OSError, BackendError, ValueError, self.qrt.WireError):
            return False

    def _drop(self):
        try:
            self.sock.close()
        except Exception:
            pass
        self.sock = None

    def generate(self, model, ids, max_new, seed, cutoff_s):
        q = self.qrt
        if self.sock is None:
            self.connect()
        served = self.server_identity.get("models", {})
        if model not in served:
            raise BackendError(f"model {model} is not served by the server")
        cutoff_ns = int(cutoff_s * 1e9)
        b0, b1 = self.bytes_sent, self.bytes_recv
        try:
            t0 = time.perf_counter_ns()
            self.bytes_sent += q.send_frame(self.sock, q.R_GEN, json.dumps({
                "model": model, "ids": [int(x) for x in ids], "max_new": int(max_new), "eos": sorted(int(x) for x in S.MODELS[model]["eos_ids"]),
                "cutoff_s": float(cutoff_s), "sampling": sampling_of(model), "seed": int(seed)}).encode())
            toks, stamps, server_ns, cut, extra = [], [], [], False, 0
            while True:
                ftype, body = q.recv_frame(self.sock)
                now = time.perf_counter_ns() - t0
                self.bytes_recv += q.FRAME.size + len(body)
                if ftype == q.R_TOKEN:
                    tok, s_ns = q.TOKEN.unpack(body)
                    if cut:
                        extra += 1                  # produced before the server saw the cancel: not part of the answer
                        continue
                    toks.append(tok)
                    stamps.append(now)
                    server_ns.append(s_ns)
                    if now > cutoff_ns:
                        cut = True
                        self.bytes_sent += q.send_frame(self.sock, q.R_CANCEL)
                elif ftype == q.R_END:
                    end = json.loads(body.decode())
                    break
                else:
                    raise BackendError("server error: " + body.decode(errors="replace")[:400])
        except (OSError, q.WireError) as e:
            self._drop()                            # the connection state is unknown: the next request reconnects
            raise TransportError(f"connection lost during the request: {type(e).__name__}: {e}"[:300]) from e
        except BackendError:
            self._drop()
            raise
        if end.get("model") != model:
            self._drop()
            raise BackendError(f"the server ran {end.get('model')}, the request named {model}")
        return {"tokens": toks, "stamps_ns": stamps, "cutoff": cut, "stop": "cutoff" if cut else end["stop"], "model": end["model"],
                "model_identity": served[model], "mode": self.mode, "seed": int(seed), "server": end,
                "server_last_token_ns": server_ns[-1] if server_ns else None, "server_first_token_ns": server_ns[0] if server_ns else None,
                "tokens_after_cutoff": extra, "bytes_out": self.bytes_sent - b0, "bytes_in": self.bytes_recv - b1,
                "n_prefill_steps": end.get("n_prefill_steps"), "prefill_ns": end.get("prefill_ns"),
                "decode_forward_ns_sum": end.get("decode_forward_ns_sum"), "decode_forward_ns_median": end.get("decode_forward_ns_median")}

    def close(self):
        if self.sock is not None:
            try:
                self.qrt.send_frame(self.sock, self.qrt.R_BYE)
            except OSError:
                pass
            self._drop()


# --------------------------------------------------------------------------- #
# Position scaling of a loaded model                                           #
# --------------------------------------------------------------------------- #
def align_position_scaling(hf):
    """Give the rotary position embedding the scale the ORIGINAL serving engine applied.

    The DeepSeek checkpoint's config.json sets rope_scaling = {rope_type: yarn, factor: 4, attn_factor: 0.878...}.
    vLLM 0.8.4, the engine of the reconstructed original protocol, multiplies the rotary cos/sin by
    (0.1 ln(factor) + 1) x attn_factor = 1.0000. transformers 4.51.3 does not read the key `attn_factor` (it reads
    `attention_factor`, warns about the unknown key and uses 0.1 ln(factor) + 1 = 1.1386). The inverse frequencies of
    the two engines are identical; only this scale differs. Left alone, the Hugging Face path would therefore run
    DeepSeek, and only DeepSeek, with position terms 13.9% larger than under the engine its original labels came from.
    This function sets the embedding's scale to the original engine's value and returns what it did. A checkpoint
    without the key (Qwen3-8B) is not touched. No checkpoint file and no reused code is modified."""
    import math
    rs = getattr(hf.config, "rope_scaling", None) or {}
    emb = hf.model.rotary_emb
    rec = {"rope_type": rs.get("rope_type", "default"), "library_scale": float(emb.attention_scaling),
           "applied_scale": float(emb.attention_scaling), "aligned_to_original_engine": False}
    if rs.get("rope_type") == "yarn" and "attn_factor" in rs and rs.get("attention_factor") is None:
        target = (0.1 * math.log(float(rs["factor"])) + 1.0) * float(rs["attn_factor"]) if float(rs["factor"]) > 1 else float(rs["attn_factor"])
        emb.attention_scaling = target
        rec.update(applied_scale=float(target), aligned_to_original_engine=True, attn_factor=float(rs["attn_factor"]), factor=float(rs["factor"]))
    return rec


# --------------------------------------------------------------------------- #
# Test fixture                                                                 #
# --------------------------------------------------------------------------- #
class MockBackend:
    """CPU fixture for tests. Deterministic pseudo-tokens, no model. Every result carries mock=True."""

    def __init__(self, mode, token_table=None, s_per_token=0.001, fail_on=(), up=True):
        self.mode, self.table, self.spt, self.fail_on, self.up, self.calls = mode, token_table or {}, s_per_token, set(fail_on), up, []

    def available(self):
        return self.up

    def generate(self, model, ids, max_new, seed, cutoff_s):
        self.calls.append({"model": model, "n_ids": len(ids), "max_new": max_new, "seed": seed})
        if not self.up:
            raise BackendError("mock server is down")
        if (model, len(ids)) in self.fail_on or model in self.fail_on:
            raise BackendError("mock failure")
        toks = list(self.table.get(model, [10 + (seed % 7), 11, 12]))[:max_new]
        stamps, cut = [], False
        for i in range(len(toks)):
            stamps.append(int((i + 1) * self.spt * 1e9))
            if stamps[-1] > cutoff_s * 1e9:
                toks, cut = toks[:i + 1], True
                break
        eos = set(S.MODELS[model]["eos_ids"])
        stop = "cutoff" if cut else ("eos" if toks and toks[-1] in eos else "cap")
        return {"tokens": toks, "stamps_ns": stamps, "cutoff": cut, "stop": stop, "model": model, "mode": self.mode, "seed": int(seed),
                "model_identity": {"key": model, "name": S.MODELS[model]["name"], "mock": True}, "mock": True,
                "n_prefill_steps": 1, "prefill_ns": 0, "decode_forward_ns_sum": 0, "decode_forward_ns_median": None}

    def close(self):
        pass
