"""Joint driver and protocol tests (CPU, tiny model, loopback): handshake refusals, fail-closed on malformed messages,
wire content audit, a frozen cell through the real run-cell command in dev mode. python -I tests/test_e1_joint.py"""
import json
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1_testlib as T  # noqa: E402
import e1_spec as S  # noqa: E402
import e1_protocol as P  # noqa: E402
import e1_joint as J  # noqa: E402
sys.path.insert(0, str(S.E0_DIR))
import split_runtime as E0RT  # noqa: E402
from e1_segment import LocalMiddle, edge_slice, server_slice  # noqa: E402


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


class _CountingConn:
    """Socket proxy that counts every byte the server reads (the content audit)."""
    def __init__(self, sock, counter):
        self._s, self._c = sock, counter

    def recv_into(self, buf, *a, **kw):
        n = self._s.recv_into(buf, *a, **kw); self._c["n"] += n; return n

    def __getattr__(self, name):
        return getattr(self._s, name)


class CountingServer(P.SegmentServer):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.counter = {"n": 0}

    def handle(self, conn):
        return super().handle(_CountingConn(conn, self.counter))

    @property
    def received(self):
        return self.counter["n"]


class Handshake(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = T.tiny_model_dir()
        cls.k, cls.j = 1, 1
        cls.ssl = server_slice(cls.model, 1, 1, "cpu", torch.bfloat16, None)
        cls.sl = edge_slice(cls.model, 1, 1, "cpu", torch.bfloat16, None)
        cls.fz = "a" * 64
        cls.ident = {"freeze_sha256": cls.fz, "config_hash": E0RT.config_hash(cls.ssl.cfg), "k": 1, "j": 1, "model_id": "dev"}

    def _serve(self, events=None, cls_=P.SegmentServer):
        mid = LocalMiddle(self.ssl, self.ssl.layer_ids, 128, "prefix")
        srv = cls_(mid.seg, self.ident, 64, 64, log=(events.append if events is not None else None))
        port = free_port(); ready = threading.Event()
        t = threading.Thread(target=srv.serve_forever, args=("127.0.0.1", port, ready, True), daemon=True); t.start(); ready.wait(5)
        return srv, port, t

    def hello(self, **kw):
        h = P.hello_for(self.sl, "dev", 1, 1, 128, self.fz); h.update(kw); return h

    def test_hello_refusals(self):
        for bad, why in ((dict(freeze_sha256="b" * 64), "freeze_sha256"), (dict(k=2), "k mismatch"), (dict(model_id="other"), "model_id"),
                         (dict(protocol_version=1), "protocol version"), (dict(config_hash="x"), "config_hash"), (dict(extra_key=1), "outside the control set"),
                         (dict(wire_dtype="float16"), "wire dtype")):
            events = []
            srv, port, t = self._serve(events)
            with self.assertRaises(P.TransportError) as cm:
                P.RemoteMiddle("127.0.0.1", port, self.hello(**bad), "cpu", 64, 64)
            self.assertIn(why, str(cm.exception), bad)
            t.join(5)
            self.assertEqual(events[0]["event"], "refused")

    def test_malformed_messages_fail_closed(self):
        srv, port, t = self._serve()
        s = socket.create_connection(("127.0.0.1", port))
        s.sendall(struct.pack("<4sHHQIIIIQQQQQ", b"E0SP", 1, P.T_HELLO, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))   # E0 magic/version
        self.assertEqual(s.recv(72), b"")   # server closed the connection without replying
        s.close(); t.join(5)
        srv, port, t = self._serve()
        s = socket.create_connection(("127.0.0.1", port))
        P.send_msg(s, P.T_FWD, n=1, hidden=64, payload=b"\0" * 128)       # FWD before HELLO
        r = P.recv_header(s); self.assertEqual(r["type"], P.T_ERROR); s.close(); t.join(5)
        srv, port, t = self._serve()
        mid = P.RemoteMiddle("127.0.0.1", port, self.hello(), "cpu", 64, 64)
        P.send_msg(mid.sock, P.T_FWD, req=1, start=0, n=2, hidden=64, flags=1, payload=b"\0" * 128)   # payload size != n*hidden*2
        r = P.recv_header(mid.sock); self.assertEqual(r["type"], P.T_ERROR)
        mid.sock.close(); t.join(5)
        srv, port, t = self._serve()
        mid = P.RemoteMiddle("127.0.0.1", port, self.hello(), "cpu", 64, 64)
        mid.reset()
        with self.assertRaises(P.TransportError):   # position 5 != committed length 0 -> server refuses, connection ends
            mid.forward(torch.zeros((1, 1, 64), dtype=torch.bfloat16), 5)
        t.join(5)

    def test_wire_carries_only_hidden_states_and_headers(self):
        events = []
        srv, port, t = self._serve(events, CountingServer)
        hello = self.hello()
        mid = P.RemoteMiddle("127.0.0.1", port, hello, "cpu", 64, 64)
        from e1_segment import USplitEdge
        e1 = USplitEdge(self.sl, 1, 1, mid, 128, "prefix")
        ids = T.check_ids(21)
        toks = e1.generate(ids, 6, 16, force_len=True)
        mid.close(); t.join(5)
        n_in, N, chunk = len(ids), 6, 16
        c = -(-n_in // chunk)
        fwd_bytes = 64 * 2 * (n_in + N - 1) + 72 * (c + N - 1)
        hello_bytes = 72 + len(json.dumps(hello).encode())
        reset_bytes = 72; bye_bytes = 72
        self.assertEqual(srv.received, hello_bytes + reset_bytes + fwd_bytes + bye_bytes)   # nothing else crossed
        self.assertTrue(set(hello) <= P.HELLO_KEYS)
        self.assertEqual(len(toks), 6)
        self.assertEqual([e["event"] for e in events], ["hello_ok", "bye"])
        # the server-side log carries control metadata only: no ids, no text, no logits
        self.assertNotIn("ids", json.dumps(events)); self.assertNotIn("tokens", json.dumps(events))


class RunCellDev(unittest.TestCase):
    def test_frozen_cell_core01_in_dev_mode_and_refusals(self):
        tmp = Path(tempfile.mkdtemp(prefix="e1rc_"))
        model = T.tiny_model_dir()
        path, rec, roles, inv = T.write_freeze(tmp, local_is="edge")
        fz = rec["freeze_sha256"]
        server_host = roles["server"]["host"]
        ssl = server_slice(model, 1, 1, "cpu", torch.bfloat16, None); ssl.cfg._attn_implementation = "e1_static"
        mid = LocalMiddle(ssl, ssl.layer_ids, S.E0.L_F, "static")
        ident = {"runtime": S.NAME, "freeze_sha256": fz, "config_hash": E0RT.config_hash(ssl.cfg), "k": 1, "j": 1, "model_id": "dev",
                 "protocol_version": S.PROTOCOL_VERSION, "host": server_host, "gpu": {"name": "cpu", "uuid": "cpu", "index": 0},
                 "branch": rec["machines"][server_host]["branch"], "active": [1, 5], "mode": "static", "one_server_gpu": True}
        srv = P.SegmentServer(mid.seg, ident, 64, S.E0.PREFILL_CHUNK)
        port = free_port(); ready = threading.Event()
        t = threading.Thread(target=srv.serve_forever, args=("127.0.0.1", port, ready, True), daemon=True); t.start(); ready.wait(5)
        out = tmp / "stage4"
        wl = T.tiny_workload(tmp / "workload.json")
        cmd = [sys.executable, "-I", str(T.HERE / "e1_bench.py"), "run-cell", "--freeze", str(path), "--cell", "core-01", "--model-dir", model,
               "--server", f"127.0.0.1:{port}", "--device", "cpu", "--dev-mode", "--out-dir", str(out), "--with-depth", "0", "--dev-max-requests", "3",
               "--workload", wl]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        t.join(10)
        rows = [json.loads(l) for l in open(out / "core-01.jsonl") if l.strip()]
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(r_["ok"] for r_ in rows)); self.assertEqual(sum(1 for r_ in rows if r_["warmup"]), 2)
        r0 = rows[-1]
        self.assertEqual(len(r0["token_ns"]), S.E0.GEN_TOKENS); self.assertEqual(r0["path"], "SPLIT"); self.assertEqual((r0["k"], r0["j"]), (1, 1))
        phases = {s["phase"] for s in r0["steps"]}
        self.assertEqual(phases, {"prefill", "decode"})
        self.assertEqual(sum(1 for s in r0["steps"] if s["phase"] == "decode"), S.E0.GEN_TOKENS - 1)
        for key in ("server_compute_ns", "net_ns", "rt_ns", "edge_head_ns", "edge_tail_ns", "lm_head_ns", "bytes_out", "bytes_in"):
            self.assertIn(key, r0["steps"][0])
        self.assertIn("link_ns_total", r0); self.assertIn("server_compute_ns_total", r0); self.assertIn("edge_compute_ns_total", r0)
        self.assertEqual(r0["provenance"]["profile_role"], "screening"); self.assertEqual(r0["provenance"]["freeze_sha256"], fz)
        self.assertEqual(r0["provenance"]["cell"], "core-01"); self.assertEqual(r0["server_identity"]["freeze_sha256"], fz)
        self.assertEqual(r0["edge_device"]["name"], "cpu"); self.assertEqual(r0["server_identity"]["gpu"]["name"], "cpu")
        # E0's fit_e2e reads these rows unchanged
        import e0_r as R
        p, st = R.fit_e2e(R.request_rows(out / "core-01.jsonl"), label="core-01")
        self.assertEqual(st["n_steps"], S.E0.GEN_TOKENS - 1)
        # refusals: unknown cell, a non-SPLIT cell, and no freeze record
        for bad in (["--cell", "nope-99"], ["--cell", "core-03"]):
            r = subprocess.run([x if x not in ("core-01",) else bad[1] for x in cmd], capture_output=True, text=True, timeout=120)
            self.assertNotEqual(r.returncode, 0)
        r = subprocess.run([x if x != str(path) else str(tmp / "missing.json") for x in cmd], capture_output=True, text=True, timeout=120)
        self.assertNotEqual(r.returncode, 0)

    def test_resolve_cell_cut_uses_frozen_cuts_only(self):
        cells = {c["id"]: c for c in S.E0.cells()}
        self.assertEqual(J.resolve_cell_cut(cells["core-01"], None), (1, 1, 12))
        self.assertEqual(J.resolve_cell_cut(cells["core-02"], {"u_max": {"16": [5, 5]}}), (5, 5, 16))
        with self.assertRaises(ValueError):
            J.resolve_cell_cut(cells["core-02"], {"u_max": {"16": None}})
        with self.assertRaises(ValueError):
            J.resolve_cell_cut(cells["core-02"], {"u_max": {"16": [6, 6]}})
        with self.assertRaises(ValueError):
            J.resolve_cell_cut(cells["core-03"], None)   # REMOTE


if __name__ == "__main__":
    unittest.main(verbosity=2)
