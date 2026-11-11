"""E1-OPT wire: the E0 message set and 72-byte header, protocol version 2, with buffers preallocated once.

Identical content rules to E0 PREREG §2.1: exactly two model-content tensors cross per forward step (the bf16
residual stream after layer k-1, edge -> server; after layer L-j-1, server -> edge). Raw text, token ids,
logits, sampled tokens and KV entries never cross. Bytes per direction per step = n * hidden * 2 + 72.

Control metadata that crosses (and nothing else):
  header (72 B): magic, version, type, request counter, start position, n tokens, hidden, flags, payload bytes,
                 server t_recv / t_compute / t_prep (reply only), server cache length (reply only);
  HELLO payload: the HELLO_KEYS below (JSON). A HELLO with any other key is refused (fail closed).
  HELLO_OK payload: the server identity (JSON; no model content).
The server verifies the freeze hash, the config hash, the cut, the model id and the protocol version in HELLO
and refuses a mismatch with T_ERROR, then closes the connection. Malformed headers close the connection.
"""
import json
import socket
import struct
import time

import torch

import e1_spec as S

MAGIC = b"E1SP"
HDR = struct.Struct("<4sHHQIIIIQQQQQ")     # same layout as E0 (72 bytes)
T_HELLO, T_HELLO_OK, T_FWD, T_FWD_OUT, T_RESET, T_RESET_OK, T_BYE, T_ERROR = range(1, 9)
assert HDR.size == 72
F_USE_CACHE = 1

# The complete control vocabulary of a HELLO. Anything else is refused.
HELLO_KEYS = frozenset({"hidden", "wire_dtype", "config_hash", "freeze_sha256", "k", "j", "model_id",
                        "protocol_version", "n_layers", "max_len", "runtime"})
HELLO_REQUIRED = frozenset({"hidden", "wire_dtype", "config_hash", "freeze_sha256", "k", "j", "model_id",
                            "protocol_version"})


class TransportError(RuntimeError):
    pass


def _recv_exact(sock, view):
    got = 0
    while got < len(view):
        r = sock.recv_into(view[got:])
        if r == 0:
            raise TransportError("peer closed the connection")
        got += r


def send_msg(sock, mtype, req=0, start=0, n=0, hidden=0, flags=0, payload=b"", t=(0, 0, 0), seen=0):
    hdr = HDR.pack(MAGIC, S.PROTOCOL_VERSION, mtype, req, start, n, hidden, flags, len(payload), t[0], t[1], t[2], seen)
    sock.sendall(hdr)
    if len(payload):
        sock.sendall(payload)
    return HDR.size + len(payload)


def recv_header(sock):
    raw = bytearray(HDR.size)
    _recv_exact(sock, memoryview(raw))
    f = HDR.unpack(raw)
    if f[0] != MAGIC or f[1] != S.PROTOCOL_VERSION:
        raise TransportError(f"bad header {f[:2]}")
    keys = ["magic", "version", "type", "req", "start", "n", "hidden", "flags", "payload", "t_recv_ns",
            "t_compute_ns", "t_prep_ns", "seen"]
    return dict(zip(keys, f))


def hello_for(sl, model_id, k, j, max_len, freeze_sha256):
    import split_runtime as E0RT
    return {"model_id": model_id, "config_hash": E0RT.config_hash(sl.cfg), "freeze_sha256": freeze_sha256,
            "k": k, "j": j, "n_layers": sl.n_layers, "hidden": sl.cfg.hidden_size, "max_len": max_len,
            "wire_dtype": "bfloat16", "protocol_version": S.PROTOCOL_VERSION, "runtime": S.NAME}


def check_hello(hello, identity, hidden):
    """Returns None if the HELLO is acceptable for this server identity, else the refusal reason."""
    if not isinstance(hello, dict):
        return "HELLO is not a JSON object"
    extra = set(hello) - HELLO_KEYS
    if extra:
        return f"HELLO carries keys outside the control set: {sorted(extra)}"
    missing = HELLO_REQUIRED - set(hello)
    if missing:
        return f"HELLO lacks {sorted(missing)}"
    if hello["hidden"] != hidden or hello["wire_dtype"] != "bfloat16":
        return "HELLO hidden size or wire dtype mismatch"
    if hello["protocol_version"] != S.PROTOCOL_VERSION:
        return f"protocol version {hello['protocol_version']} != {S.PROTOCOL_VERSION}"
    for key in ("freeze_sha256", "config_hash", "model_id", "k", "j"):
        if key in identity and identity[key] is not None and hello[key] != identity[key]:
            return f"{key} mismatch"
    return None


class WireBuffers:
    """Host staging buffers sized ONCE for the largest message (one prefill chunk); pinned on CUDA."""

    def __init__(self, device, hidden, max_tokens):
        self.cuda = torch.device(device).type == "cuda"
        self.hidden, self.max_tokens = hidden, max_tokens
        mk = lambda: torch.empty(max_tokens * hidden, dtype=torch.bfloat16, pin_memory=self.cuda)
        self.out, self.inp = mk(), mk()
        self.dev_in = torch.empty((1, max_tokens, hidden), dtype=torch.bfloat16, device=device)

    @staticmethod
    def as_bytes(t, numel):
        return memoryview(t[:numel].view(torch.int16).numpy()).cast("B")

    def d2h(self, x):
        if x.dtype != torch.bfloat16:
            raise TypeError(f"wire dtype is bfloat16 only; refusing {x.dtype}")
        numel = x.numel()
        if numel > self.out.numel():
            raise TransportError(f"message of {numel // self.hidden} tokens exceeds the preallocated {self.max_tokens}")
        self.out[:numel].copy_(x.reshape(-1), non_blocking=self.cuda)
        if self.cuda:
            torch.cuda.current_stream(x.device).synchronize()
        return self.as_bytes(self.out, numel)

    def recv_into(self, sock, n):
        _recv_exact(sock, self.as_bytes(self.inp, n * self.hidden))
        return self.inp[:n * self.hidden]

    def h2d(self, host, n):
        y = self.dev_in[:, :n, :]
        y.reshape(-1).copy_(host, non_blocking=self.cuda)
        return y


class AppShaper:
    """E0's application-level link emulation (split_runtime.AppShaper semantics), used only where tc netem is
    unavailable; recorded as shaping="app"."""

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


class RemoteMiddle:
    """Edge-side client of a SegmentServer; strictly synchronous, one message one reply."""

    def __init__(self, host, port, hello, device, hidden, max_tokens, timeout_s=600, shaper=None):
        self.sock = socket.create_connection((host, port), timeout=timeout_s)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.device, self.hidden = torch.device(device), hidden
        self.buf = WireBuffers(device, hidden, max_tokens)
        self.shaper = shaper
        self.req, self.bytes_sent, self.bytes_recv, self.last = 0, 0, 0, {}
        self.bytes_sent += send_msg(self.sock, T_HELLO, payload=json.dumps(hello).encode())
        r = recv_header(self.sock)
        body = bytearray(r["payload"]); _recv_exact(self.sock, memoryview(body))
        self.bytes_recv += HDR.size + r["payload"]
        if r["type"] != T_HELLO_OK:
            raise TransportError(f"HELLO rejected: {body.decode(errors='replace')}")
        self.server_identity = json.loads(body.decode())

    def reset(self):
        self.req += 1
        self.bytes_sent += send_msg(self.sock, T_RESET, req=self.req)
        r = recv_header(self.sock)
        self.bytes_recv += HDR.size
        if r["type"] != T_RESET_OK:
            raise TransportError(f"bad RESET reply {r}")

    def forward(self, h, start):
        n = h.shape[1]
        t0 = time.perf_counter_ns()
        payload = self.buf.d2h(h)
        t1 = time.perf_counter_ns()
        nbytes = HDR.size + len(payload)
        if self.shaper:
            self.shaper.before_send(nbytes)
        t2 = time.perf_counter_ns()
        self.bytes_sent += send_msg(self.sock, T_FWD, req=self.req, start=start, n=n, hidden=self.hidden,
                                    flags=F_USE_CACHE, payload=payload)
        r = recv_header(self.sock)
        if r["type"] == T_ERROR:
            msg = bytearray(r["payload"]); _recv_exact(self.sock, memoryview(msg))
            raise TransportError(f"server error: {msg.decode(errors='replace')}")
        if r["type"] != T_FWD_OUT or r["n"] != n or r["start"] != start or r["payload"] != n * self.hidden * 2:
            raise TransportError(f"bad FWD_OUT {r}")
        host = self.buf.recv_into(self.sock, n)
        t3 = time.perf_counter_ns()
        self.bytes_recv += HDR.size + r["payload"]
        if self.shaper:
            self.shaper.after_recv(HDR.size + r["payload"])
        t4 = time.perf_counter_ns()
        y = self.buf.h2d(host, n)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t5 = time.perf_counter_ns()
        server_ns = r["t_recv_ns"] + r["t_compute_ns"] + r["t_prep_ns"]
        self.last = {"ser_ns": (t1 - t0) + (t5 - t4), "shape_ns": (t2 - t1) + (t4 - t3), "rt_ns": t3 - t2,
                     "server_ns": server_ns, "server_compute_ns": r["t_compute_ns"], "net_ns": (t3 - t2) - server_ns,
                     "bytes_out": nbytes, "bytes_in": HDR.size + r["payload"], "server_seen": r["seen"]}
        return y

    def close(self):
        try:
            send_msg(self.sock, T_BYE)
        except OSError:
            pass
        self.sock.close()


class SegmentServer:
    """Serves one client at a time, batch 1, for a Segment (e1_segment.Segment) or an echo stub (seg=None).

    `identity` is sent back in HELLO_OK and is the reference for check_hello(). It must carry
    freeze_sha256, config_hash (None for echo), k, j, model_id and protocol_version. `log` (optional callable)
    receives one dict per connection event (hello accepted / refused, errors, bye); never model content.
    """

    def __init__(self, seg, identity, hidden, max_tokens, log=None):
        self.seg, self.identity, self.hidden, self.max_tokens = seg, dict(identity), hidden, max_tokens
        self.identity.setdefault("protocol_version", S.PROTOCOL_VERSION)
        self.identity.setdefault("runtime", S.NAME)
        self.device = seg.sl.device if seg else torch.device("cpu")
        self.buf = WireBuffers(self.device, hidden, max_tokens)
        self.log = log or (lambda ev: None)
        self.n_fwd = 0

    def _refuse(self, conn, why):
        self.log({"event": "refused", "why": why})
        send_msg(conn, T_ERROR, payload=why.encode())

    def handle(self, conn):
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        r = recv_header(conn)
        if r["type"] != T_HELLO or r["payload"] > 1 << 16:
            self._refuse(conn, "first message is not a HELLO"); return
        body = bytearray(r["payload"]); _recv_exact(conn, memoryview(body))
        try:
            hello = json.loads(body.decode())
        except Exception:
            self._refuse(conn, "HELLO payload is not JSON"); return
        why = check_hello(hello, self.identity, self.hidden)
        if why:
            self._refuse(conn, why); return
        self.log({"event": "hello_ok", "peer_hello": hello})
        send_msg(conn, T_HELLO_OK, payload=json.dumps(self.identity).encode())
        while True:
            r = recv_header(conn)
            if r["type"] == T_BYE:
                self.log({"event": "bye", "n_fwd": self.n_fwd}); return
            if r["type"] == T_RESET:
                if self.seg is not None:
                    self.seg.reset()
                send_msg(conn, T_RESET_OK, req=r["req"]); continue
            if r["type"] != T_FWD:
                self._refuse(conn, "unexpected message type %d" % r["type"]); return
            n, start = r["n"], r["start"]
            if r["hidden"] != self.hidden or r["payload"] != n * self.hidden * 2 or n < 1 or n > self.max_tokens:
                self._refuse(conn, "bad FWD shape"); return
            t0 = time.perf_counter_ns()
            host = self.buf.recv_into(conn, n)
            x = self.buf.h2d(host, n)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            t1 = time.perf_counter_ns()
            try:
                y = x if self.seg is None else self.seg.forward(x, start)
            except Exception as e:  # position/cache errors are rejected, then the connection ends (E0 semantics)
                self._refuse(conn, f"{type(e).__name__}: {e}"); return
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            t2 = time.perf_counter_ns()
            payload = self.buf.d2h(y)
            t3 = time.perf_counter_ns()
            seen = self.seg.cache.seen if self.seg is not None else start + n
            self.n_fwd += 1
            send_msg(conn, T_FWD_OUT, req=r["req"], start=start, n=n, hidden=self.hidden, payload=payload,
                     t=(t1 - t0, t2 - t1, t3 - t2), seen=seen)

    def serve_forever(self, host, port, ready=None, once=False, ready_file=None):
        srv = socket.create_server((host, port))
        if ready_file:
            from pathlib import Path
            Path(ready_file).write_text(json.dumps({"port": srv.getsockname()[1], **self.identity}, default=str))
        if ready is not None:
            ready.set()
        while True:
            conn, _ = srv.accept()
            with conn:
                try:
                    self.handle(conn)
                except (TransportError, OSError) as e:
                    self.log({"event": "connection_closed", "why": f"{type(e).__name__}: {e}"})
            if once:
                srv.close(); return
