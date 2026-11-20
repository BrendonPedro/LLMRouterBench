"""QWEN-SELECT runtime glue. Every model computation is the reused runtime's, imported unchanged:

  K16-HWFIT  k16_runtime.hf_model           the whole Hugging Face Qwen3ForCausalLM, bfloat16, loaded straight onto the device
             k16_runtime.ServerSide / serve / connect / edge_split / ping / write_reference / evaluate_config
  E1         e1_segment.USplitEdge          the k = 16 split's edge half ("prefix" mode = E0's computation)
  E0         split_runtime.Slice            weight loading of a layer subset

This module adds only: generation loops that stop at EOS, at the output cap or at the harness cutoff and stamp every
token on one clock (the split loop performs USplitEdge.generate's steps; the whole-model loop performs the steps of
K16-HWFIT's unsplit baseline), and the REMOTE wire, on which the edge sends token ids and the server streams the
output tokens back. Nothing here decides anything.
"""
import hmac
import json
import select
import socket
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # python -I: the script dir is not on sys.path
import qsel_spec as S  # noqa: E402

K16_DIR = Path(__file__).resolve().parent.parent / "deployment_k16fit"


def k16():
    """(k16_spec, k16_runtime): the reused runtime, imported unchanged. Importing it needs torch and transformers."""
    if str(K16_DIR) not in sys.path:
        sys.path.insert(0, str(K16_DIR))
    import k16_spec
    import k16_runtime
    return k16_spec, k16_runtime


DEV_CONTEXT, DEV_CHUNK, DEV_N_CONT = 256, 48, 8


def params(dev=False, model="32B", context=None, chunk=None, n_cont=None):
    """Run sizes handed to the reused runtime. dev=True (CPU tests with tiny random models) shrinks them unless the
    test's site file sets them; a target run always uses the frozen values."""
    m = S.MODELS[model]
    if dev:
        return {"dev": True, "max_len": int(context or DEV_CONTEXT), "chunk": int(chunk or DEV_CHUNK), "greedy_tokens": int(n_cont or DEV_N_CONT),
                "model_id": "dev"}
    return {"dev": False, "max_len": S.CONTEXT_TOKENS, "chunk": S.PREFILL_CHUNK, "greedy_tokens": S.N_CONT, "model_id": f"{m['repo']}@{m['revision']}"}


def _result(toks, stamps, cut, reason, **extra):
    return dict({"tokens": toks, "stamps_ns": stamps, "cutoff": cut, "stop": reason}, **extra)


# --------------------------------------------------------------------------- #
# Whole model on one device (EDGE configurations, and the REMOTE server)       #
# --------------------------------------------------------------------------- #
def whole_generate(torch, hf, ids, max_new, eos, device, chunk, cutoff_s, sync, on_token=None, should_stop=None):
    """Greedy generation with the model's own KV cache: chunked prefill, then one token per step; the LM head sees the
    last position only. The clock starts when the token ids are handed over. Token i is stamped once its id is a host
    integer. Ends on a token of `eos`, at `max_new`, at the first token stamped after `cutoff_s`, or when
    should_stop() is true (the REMOTE server's cancel)."""
    eos, cutoff_ns = set(eos), int(cutoff_s * 1e9)
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
            tok = int(torch.argmax(hf.lm_head(h[:, -1:, :])[0, -1].float()))
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
    return _result(toks, stamps, cut, reason, n_prefill_steps=n_prefill, prefill_ns=sum(fwd[:n_prefill]), decode_forward_ns=fwd[n_prefill:])


# --------------------------------------------------------------------------- #
# Split (the edge half; the server half is k16_runtime.serve)                  #
# --------------------------------------------------------------------------- #
STEP_KEYS = ("edge_head_ns", "edge_tail_ns", "lm_head_ns", "middle_ns", "rt_ns", "server_ns", "server_compute_ns", "net_ns", "ser_ns",
             "bytes_out", "bytes_in")


def split_generate(torch, e, ids, max_new, eos, chunk, cutoff_s):
    """Greedy generation through e1_segment.USplitEdge `e`: the steps of USplitEdge.generate, with the clock started
    before the session reset (so the reset round trip is inside the completion time), an EOS set, and the harness
    cutoff. Returns the tokens, their stamps and the runtime's own per-step counters."""
    eos, cutoff_ns, dev = set(eos), int(cutoff_s * 1e9), e.sl.device
    with torch.inference_mode():
        t0 = time.perf_counter_ns()
        e.reset()
        x = torch.as_tensor(ids, dtype=torch.long, device=dev).view(1, -1)
        P = x.shape[1]
        if P + max_new > e.max_len:
            raise ValueError(f"prompt {P} + {max_new} new tokens exceed the context of {e.max_len}")
        for s in range(0, P, chunk):
            h = e.forward_hidden(x[:, s:s + chunk], s)
        n_prefill = len(e.steps)
        toks, stamps, pos, cut, reason = [], [], P, False, None
        while True:
            t_h = time.perf_counter_ns()
            tok = int(torch.argmax(e.sl.lm_head(h[:, -1:, :])[0, -1].float()))
            now = time.perf_counter_ns()
            e.steps[-1]["lm_head_ns"] = now - t_h
            toks.append(tok)
            stamps.append(now - t0)
            if now - t0 > cutoff_ns:
                cut, reason = True, "cutoff"
            elif tok in eos:
                reason = "eos"
            elif len(toks) >= max_new:
                reason = "cap"
            if reason:
                break
            h = e.forward_hidden(torch.tensor([[tok]], device=dev), pos)
            pos += 1
    steps = e.steps
    kv_ok = all(s.get("server_seen") == s["start"] + s["n"] for s in steps)
    col = lambda rows, k: [int(s.get(k, 0)) for s in rows]  # noqa: E731
    pre, dec = steps[:n_prefill], steps[n_prefill:]
    return _result(toks, stamps, cut, reason, n_prefill_steps=n_prefill, kv_positions_in_step=kv_ok,
                   prefill={k: sum(col(pre, k)) for k in STEP_KEYS}, decode_sum={k: sum(col(dec, k)) for k in STEP_KEYS},
                   decode_steps={k: col(dec, k) for k in ("edge_head_ns", "edge_tail_ns", "server_compute_ns", "net_ns", "rt_ns")})


# --------------------------------------------------------------------------- #
# REMOTE wire: token ids go to the server, output tokens come back one by one  #
# --------------------------------------------------------------------------- #
RMAGIC = b"QSR1"
FRAME = struct.Struct("<4sBI")             # magic, type, payload bytes
TOKEN = struct.Struct("<IQ")               # token id, server nanoseconds since the request arrived
R_HELLO, R_HELLO_OK, R_GEN, R_TOKEN, R_END, R_CANCEL, R_BYE, R_ERROR = range(1, 9)
MAX_CONTROL_BYTES = 1 << 20


class WireError(RuntimeError):
    pass


def _recv_exact(sock, n):
    buf = bytearray(n)
    view, got = memoryview(buf), 0
    while got < n:
        r = sock.recv_into(view[got:])
        if r == 0:
            raise WireError("peer closed the connection")
        got += r
    return bytes(buf)


def send_frame(sock, ftype, payload=b""):
    sock.sendall(FRAME.pack(RMAGIC, ftype, len(payload)) + payload)
    return FRAME.size + len(payload)


def recv_frame(sock):
    magic, ftype, n = FRAME.unpack(_recv_exact(sock, FRAME.size))
    if magic != RMAGIC or n > MAX_CONTROL_BYTES:
        raise WireError("bad frame")
    return ftype, _recv_exact(sock, n) if n else b""


class RemoteServer:
    """Serves one edge at a time, one request at a time, for a whole model on this host's device."""

    def __init__(self, torch, hf, device, sync, identity, token, chunk, log=None):
        self.torch, self.hf, self.device, self.sync = torch, hf, device, sync
        self.identity, self.token, self.chunk, self.log = dict(identity), token, chunk, log or (lambda ev: None)
        self.n_requests = 0

    def handle(self, conn):
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        ftype, body = recv_frame(conn)
        try:
            hello = json.loads(body.decode()) if ftype == R_HELLO else None
        except ValueError:
            hello = None
        why = None
        if not isinstance(hello, dict):
            why = "first message is not a HELLO"
        elif not hmac.compare_digest(str(hello.get("token", "")), self.token):
            why = "token mismatch"
        elif hello.get("spec_hash") != self.identity["spec_hash"] or hello.get("model") != self.identity["model"]:
            why = "spec hash or model mismatch"
        if why:
            self.log({"event": "refused", "why": why})
            send_frame(conn, R_ERROR, why.encode())
            return
        self.log({"event": "hello_ok"})
        send_frame(conn, R_HELLO_OK, json.dumps(self.identity).encode())
        while True:
            ftype, body = recv_frame(conn)
            if ftype == R_BYE:
                self.log({"event": "bye", "n_requests": self.n_requests})
                return
            if ftype == R_CANCEL:                  # a cancel that crossed the end of its request: nothing is left to cancel
                continue
            if ftype != R_GEN:
                send_frame(conn, R_ERROR, b"unexpected message")
                return
            t_req = time.perf_counter_ns()
            req = json.loads(body.decode())
            ids, max_new = [int(x) for x in req["ids"]], int(req["max_new"])
            if not ids or max_new < 1 or len(ids) + max_new > int(self.identity["context_tokens"]):
                send_frame(conn, R_ERROR, b"request exceeds the context")
                return

            def on_token(i, tok, now_ns):
                conn.sendall(FRAME.pack(RMAGIC, R_TOKEN, TOKEN.size) + TOKEN.pack(tok, time.perf_counter_ns() - t_req))

            def cancelled():
                r, _, _ = select.select([conn], [], [], 0)
                if not r:
                    return False
                ft, _b = recv_frame(conn)
                return ft == R_CANCEL
            # the server's own guard: the edge decides lateness on its clock and cancels; this only bounds a lost cancel
            res = whole_generate(self.torch, self.hf, ids, max_new, req["eos"], self.device, self.chunk, float(req["cutoff_s"]) + 10.0,
                                 self.sync, on_token=on_token, should_stop=cancelled)
            self.n_requests += 1
            dec = sorted(res["decode_forward_ns"])
            send_frame(conn, R_END, json.dumps({"n": len(res["tokens"]), "stop": res["stop"], "n_prefill_steps": res["n_prefill_steps"],
                                                "prefill_ns": res["prefill_ns"], "decode_forward_ns_sum": sum(dec),
                                                "decode_forward_ns_median": dec[len(dec) // 2] if dec else None,
                                                "server_total_ns": time.perf_counter_ns() - t_req}).encode())


def serve_remote(server, listen_sock, first_wait_s, idle_exit_s, stop=None, conn_timeout_s=1800):
    """Accept edge connections one at a time until `stop` is set, no edge came within first_wait_s, or the last
    connection closed idle_exit_s ago. Returns "stopped", "no_edge" or "idle"."""
    listen_sock.settimeout(1.0)
    waited, served = 0.0, False
    while True:
        if stop is not None and stop.is_set():
            return "stopped"
        try:
            conn, _peer = listen_sock.accept()
        except socket.timeout:
            waited += 1.0
            if waited >= (idle_exit_s if served else first_wait_s):
                return "idle" if served else "no_edge"
            continue
        with conn:
            conn.settimeout(conn_timeout_s)
            try:
                server.log({"event": "connection_open"})
                server.handle(conn)
            except (WireError, OSError, ValueError, KeyError) as e:
                server.log({"event": "connection_closed", "why": type(e).__name__})
        waited, served = 0.0, True


class RemoteClient:
    """Edge side of the REMOTE wire. No model and no GPU: it sends token ids and stamps the tokens as they arrive."""

    def __init__(self, addr, token, spec_hash, model, timeout_s=600):
        host, port = addr.rsplit(":", 1)
        self.sock = socket.create_connection((host, int(port)), timeout=timeout_s)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.bytes_sent = send_frame(self.sock, R_HELLO, json.dumps({"token": token, "spec_hash": spec_hash, "model": model}).encode())
        ftype, body = recv_frame(self.sock)
        self.bytes_recv = FRAME.size + len(body)
        if ftype != R_HELLO_OK:
            self.sock.close()
            raise WireError("HELLO refused: " + body.decode(errors="replace")[:120])
        self.server_identity = json.loads(body.decode())

    def generate(self, ids, max_new, eos, cutoff_s):
        cutoff_ns = int(cutoff_s * 1e9)
        b0, b1 = self.bytes_sent, self.bytes_recv
        t0 = time.perf_counter_ns()
        self.bytes_sent += send_frame(self.sock, R_GEN, json.dumps({"ids": [int(x) for x in ids], "max_new": int(max_new),
                                                                     "eos": sorted(int(x) for x in eos), "cutoff_s": float(cutoff_s)}).encode())
        toks, stamps, server_ns, cut, extra = [], [], [], False, 0
        while True:
            ftype, body = recv_frame(self.sock)
            now = time.perf_counter_ns() - t0
            self.bytes_recv += FRAME.size + len(body)
            if ftype == R_TOKEN:
                tok, s_ns = TOKEN.unpack(body)
                if cut:
                    extra += 1                      # tokens the server produced before it saw the cancel: not part of the answer
                    continue
                toks.append(tok)
                stamps.append(now)
                server_ns.append(s_ns)
                if now > cutoff_ns:
                    cut = True
                    self.bytes_sent += send_frame(self.sock, R_CANCEL)
            elif ftype == R_END:
                end = json.loads(body.decode())
                break
            else:
                raise WireError("server error: " + body.decode(errors="replace")[:120])
        reason = "cutoff" if cut else end["stop"]
        return _result(toks, stamps, cut, reason, server=end, server_token_ns=server_ns, tokens_after_cutoff=extra,
                       bytes_out=self.bytes_sent - b0, bytes_in=self.bytes_recv - b1)

    def close(self):
        try:
            send_frame(self.sock, R_BYE)
        except OSError:
            pass
        self.sock.close()


def remote_ready(addr, token, spec_hash, model, timeout_s=20):
    """The server identity if the experiment's REMOTE server answers through the forward, else None. Never raises."""
    try:
        c = RemoteClient(addr, token, spec_hash, model, timeout_s=timeout_s)
        ident = c.server_identity
        c.close()
        return ident
    except (OSError, WireError, ValueError):
        return None


def split_ready(R, addr, model_dir, spec_hash, p, timeout_s=20):
    """The server identity if the experiment's split server answers through the forward, else None. Never raises."""
    try:
        return R.ping(addr, model_dir, spec_hash, p, timeout_s=timeout_s)
    except Exception:
        return None


def summarize_times(res):
    """Seconds: time to first token, completion time, time per output token ((last - first) / (n - 1))."""
    st = res["stamps_ns"]
    return {"ttft_s": st[0] / 1e9, "total_s": st[-1] / 1e9, "per_token_s": (st[-1] - st[0]) / 1e9 / (len(st) - 1) if len(st) > 1 else None}
