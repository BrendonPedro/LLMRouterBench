"""E1-OPT CPU tests (no GPU, no Qwen3-32B). Decide nothing about SPLIT; they validate the implementation.

  python tests/test_e1_cpu.py            # tiny random Qwen3 (seconds) + Qwen3-0.6B real weights if cached
"""
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import e1_spec as S  # noqa: E402
sys.path.insert(0, str(S.E0_DIR))
import split_runtime as E0RT  # noqa: E402
import e1_attention  # noqa: E402,F401
from e1_segment import USplitEdge, LocalMiddle, edge_slice, server_slice  # noqa: E402
from e1_static_cache import CacheOverflow  # noqa: E402
import e1_protocol as P  # noqa: E402
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

torch.manual_seed(0)
REAL = Path.home() / ".cache/LLMRouterBench/e0_hf/qwen3-0.6b"


def tiny_model(tmp, dtype):
    cfg = Qwen3Config(vocab_size=211, hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, max_position_embeddings=512, tie_word_embeddings=False,
                      rope_theta=10000.0, rms_norm_eps=1e-6, torch_dtype=dtype)
    m = Qwen3ForCausalLM(cfg).to(dtype).eval()
    m.save_pretrained(tmp, safe_serialization=True)
    return m


class _Hidden:
    """Unsplit HF reference: final normed hidden states and greedy tokens."""
    def __init__(self, model):
        self.m = model

    @torch.inference_mode()
    def hidden(self, ids):
        return self.m.model(torch.as_tensor(ids).view(1, -1)).last_hidden_state[0]

    @torch.inference_mode()
    def greedy(self, ids, n):
        out = self.m.generate(torch.as_tensor(ids).view(1, -1), max_new_tokens=n, min_new_tokens=n, do_sample=False,
                              pad_token_id=0, eos_token_id=None)
        return out[0, len(ids):].tolist()


def build(model_dir, k, j, mode, dtype, max_len=256, remote=None):
    cfg = {"num_hidden_layers": 6, "hidden_size": 64, "torch_dtype": "bfloat16" if dtype == torch.bfloat16 else "float32"}
    sl = edge_slice(model_dir, k, j, "cpu", dtype, None)
    if mode == "static":
        sl.cfg._attn_implementation = "e1_static"
    if remote is None:
        ssl = server_slice(model_dir, k, j, "cpu", dtype, None)
        if mode == "static":
            ssl.cfg._attn_implementation = "e1_static"
        mid = LocalMiddle(ssl, ssl.layer_ids, max_len, mode)
    else:
        mid = remote
    return USplitEdge(sl, k, j, mid, max_len, mode)


class TinyModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="e1tiny_")
        cls.ref32 = _Hidden(tiny_model(cls.tmp, torch.float32))
        cls.ref16 = _Hidden(Qwen3ForCausalLM.from_pretrained(cls.tmp, torch_dtype=torch.bfloat16).eval())
        g = torch.Generator().manual_seed(1)
        cls.ids = torch.randint(1, 211, (37,), generator=g).tolist()

    def test_prefix_mode_bitwise_equals_hf_and_e0(self):
        for dtype, ref in ((torch.float32, self.ref32), (torch.bfloat16, self.ref16)):
            for k, j in ((1, 1), (2, 3), (3, 3)):
                e1 = build(self.tmp, k, j, "prefix", dtype)
                h = e1.teacher_forced_hidden(self.ids)
                self.assertTrue(torch.equal(h, ref.hidden(self.ids)), f"prefix {dtype} U({k},{j}) != HF")
                # E0's own runtime on the same weights (read-only import)
                sl0 = E0RT.edge_slice(self.tmp, k, j, "cpu", dtype, None)
                mid0 = E0RT.LocalMiddle(sl0 if k + j == 6 else E0RT.Slice(self.tmp, list(range(k, 6 - j)), False, False, "cpu", dtype, None), list(range(k, 6 - j)), 256)
                r0 = E0RT.EdgeRunner(sl0, k, j, mid0, max_len=256)
                self.assertTrue(torch.equal(h, r0.teacher_forced_hidden(self.ids)), f"prefix {dtype} U({k},{j}) != E0")

    def test_static_mode_matches_prefix_within_tolerance_and_top1(self):
        for dtype in (torch.float32, torch.bfloat16):
            e1s, e1p = build(self.tmp, 2, 2, "static", dtype), build(self.tmp, 2, 2, "prefix", dtype)
            hs, hp = e1s.teacher_forced_hidden(self.ids), e1p.teacher_forced_hidden(self.ids)
            tol = 1e-5 if dtype == torch.float32 else S.EQ_STATIC_MAX_ABS_BF16
            self.assertLessEqual((hs.float() - hp.float()).abs().max().item(), tol)
            with torch.inference_mode():
                ls, lp = e1s.sl.lm_head(hs).float(), e1p.sl.lm_head(hp).float()
            self.assertTrue(torch.equal(ls.argmax(-1), lp.argmax(-1)))

    def test_greedy_chunked_prefill_equals_hf_generate(self):
        for mode in ("prefix", "static"):
            e1 = build(self.tmp, 2, 2, mode, torch.float32)
            for chunk in (5, 16, 64):
                toks = e1.generate(self.ids, 12, chunk, force_len=True)
                self.assertEqual(toks, self.ref32.greedy(self.ids, 12), f"{mode} chunk {chunk}")
            self.assertEqual(len(e1.token_ns), 12)

    def test_graph_step_control_flow_on_cpu(self):
        e1 = build(self.tmp, 2, 2, "static", torch.float32)
        ref = e1.generate(self.ids, 8, 16, force_len=True)
        e1.enable_graph(); e1.middle.enable_graph()
        self.assertEqual(e1.generate(self.ids, 8, 16, force_len=True), ref)

    def test_overflow_is_refused_before_the_step(self):
        e1 = build(self.tmp, 1, 1, "static", torch.float32, max_len=40)
        with self.assertRaises(CacheOverflow):
            e1.generate(self.ids, 8, 16, force_len=True)
        e1 = build(self.tmp, 1, 1, "static", torch.float32, max_len=45)
        e1.generate(self.ids, 8, 16, force_len=True)
        with self.assertRaises(CacheOverflow):
            e1.head.forward(torch.zeros((1, 1, 64)), 3)   # position != committed length

    def test_wire_roundtrip_bytes_and_identity(self):
        ssl = server_slice(self.tmp, 2, 2, "cpu", torch.bfloat16, None)
        seg = LocalMiddle(ssl, ssl.layer_ids, 256, "prefix").seg
        ident = {"config_hash": E0RT.config_hash(ssl.cfg), "active": [2, 4], "runtime": S.NAME, "freeze_sha256": "f" * 64,
                 "k": 2, "j": 2, "model_id": "dev"}
        srv = P.SegmentServer(seg, ident, 64, 64)
        ready = threading.Event()
        port = 29711
        t = threading.Thread(target=srv.serve_forever, args=("127.0.0.1", port, ready, True), daemon=True); t.start()
        ready.wait(5)
        hello = P.hello_for(ssl, "dev", 2, 2, 256, "f" * 64)
        mid = P.RemoteMiddle("127.0.0.1", port, hello, "cpu", 64, 64)
        e1 = build(self.tmp, 2, 2, "prefix", torch.bfloat16, remote=mid)
        toks = e1.generate(self.ids, 6, 16, force_len=True)
        self.assertEqual(toks, build(self.tmp, 2, 2, "prefix", torch.bfloat16).generate(self.ids, 6, 16, force_len=True))
        n_in, N = len(self.ids), 6
        c = -(-n_in // 16)
        expect = 64 * 2 * (n_in + N - 1) + 72 * (c + N - 1)          # FWD payloads + headers (prefill chunks + decode)
        fwd_out = sum(s["bytes_out"] for s in e1.steps)
        self.assertEqual(fwd_out, expect)
        self.assertEqual(sum(s["bytes_in"] for s in e1.steps), expect)
        self.assertEqual(e1.steps[-1]["server_seen"], n_in + N - 1)
        self.assertEqual(mid.server_identity["runtime"], S.NAME)
        mid.close(); t.join(5)

    def test_wire_refuses_wrong_version_and_hash(self):
        ssl = server_slice(self.tmp, 2, 2, "cpu", torch.bfloat16, None)
        seg = LocalMiddle(ssl, ssl.layer_ids, 256, "prefix").seg
        srv = P.SegmentServer(seg, {"config_hash": "x", "freeze_sha256": "f" * 64, "k": 2, "j": 2, "model_id": "dev"}, 64, 64)
        ready = threading.Event(); port = 29712
        t = threading.Thread(target=srv.serve_forever, args=("127.0.0.1", port, ready, True), daemon=True); t.start(); ready.wait(5)
        with self.assertRaises(P.TransportError):
            P.RemoteMiddle("127.0.0.1", port, dict(P.hello_for(ssl, "dev", 2, 2, 256, "f" * 64), config_hash="y"), "cpu", 64, 64)
        t.join(5)
        old = S.PROTOCOL_VERSION
        try:
            S.PROTOCOL_VERSION = 1
            raw = P.HDR.pack(P.MAGIC, 1, P.T_HELLO, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        finally:
            S.PROTOCOL_VERSION = old
        import socket
        with self.assertRaises(P.TransportError):
            a, b = socket.socketpair(); a.sendall(raw); P.recv_header(b)


class E0ImmutabilityTest(unittest.TestCase):
    def test_e0_directory_unchanged_since_validated_commit(self):
        repo = S.E0_DIR.parents[1]
        r = subprocess.run(["git", "diff", "--quiet", S.E0_VALIDATED_COMMIT, "--", "diagnostics/deployment_e0"], cwd=repo)
        self.assertEqual(r.returncode, 0, "deployment_e0 differs from the validated commit")
        st = subprocess.run(["git", "status", "--porcelain", "--", "diagnostics/deployment_e0"], cwd=repo, capture_output=True, text=True).stdout
        self.assertEqual(st.strip(), "", f"untracked/modified files under deployment_e0:\n{st}")
        tag = subprocess.run(["git", "rev-parse", "e0-prereg-v1"], cwd=repo, capture_output=True, text=True).stdout.strip()
        self.assertEqual(tag, "1b277b37c4d59b0e7a9f10b2dd86fa6981e71bc9")


@unittest.skipUnless(REAL.exists(), "Qwen3-0.6B weights not cached")
class RealWeightsTest(unittest.TestCase):
    def test_qwen3_0_6b_prefix_bitwise_and_static_top1(self):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(REAL))
        ids = tok.apply_chat_template([{"role": "user", "content": "Name three prime numbers and add them."}],
                                      add_generation_prompt=True, enable_thinking=False)
        ref = _Hidden(Qwen3ForCausalLM.from_pretrained(str(REAL), torch_dtype=torch.bfloat16).eval())
        k, j = 2, 3
        sl = edge_slice(str(REAL), k, j, "cpu", torch.bfloat16, None)
        ssl = server_slice(str(REAL), k, j, "cpu", torch.bfloat16, None)
        e1 = USplitEdge(sl, k, j, LocalMiddle(ssl, ssl.layer_ids, 512, "prefix"), 512, "prefix")
        self.assertTrue(torch.equal(e1.teacher_forced_hidden(ids), ref.hidden(ids)))
        self.assertEqual(e1.generate(ids, 16, 16, force_len=True), ref.greedy(ids, 16))
        sl.cfg._attn_implementation = "e1_static"; ssl.cfg._attn_implementation = "e1_static"
        e1s = USplitEdge(sl, k, j, LocalMiddle(ssl, ssl.layer_ids, 512, "static"), 512, "static")
        self.assertEqual(e1s.generate(ids, 16, 16, force_len=True), ref.greedy(ids, 16))


if __name__ == "__main__":
    unittest.main(verbosity=2)
