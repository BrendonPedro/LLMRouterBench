"""Equivalence driver tests (CPU, tiny model): output schema has no timing fields; local in-host checks; segments-only
checks; joint Q1/Q2 over loopback in E0's file formats; E0 decision rules; CLI. python -I tests/test_e1_equivalence.py"""
import json
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
import e1_equivalence as EQ  # noqa: E402
import e1_protocol as P  # noqa: E402
import e1_freeze as FZ  # noqa: E402
sys.path.insert(0, str(S.E0_DIR))
import split_runtime as E0RT  # noqa: E402
import e0_instrument as I  # noqa: E402
from e1_segment import USplitEdge, LocalMiddle, edge_slice, server_slice  # noqa: E402

BRANCH_CPU = {"attention": "sdpa_static_fallback", "attention_reason": "cpu", "cuda_graphs": "on", "cuda_graphs_reason": "x", "torch_compile": False}


class Schema(unittest.TestCase):
    def test_no_timing_fields_detector(self):
        EQ.assert_no_timing_fields({"pass": True, "runtime_branch": {"attention": "x"}, "n_check_tokens": 37, "utc": "now", "stage": "1b",
                                    "wire": {"wire_bytes_out_equal_formula": True}, "nested": [{"ok": 1}]})
        for bad in ({"tpot_ms": 1}, {"a": {"server_compute_ns": 2}}, {"token_ns": [1]}, {"latency": 3}, {"tokens_per_s": 4}, {"elapsed": 5},
                    {"x": [{"t0_ns": 1}]}, {"bulk_mbps_median": 1}, {"ttft_s": 1}, {"wall_clock": 1}, {"cuda_event_ms": 1}, {"seconds": 1},
                    {"throughput": 1}, {"duration": 1}, {"time_s": 1}):
            with self.assertRaises(EQ.TimingFieldError, msg=str(bad)):
                EQ.assert_no_timing_fields(bad)


class LocalChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="e1eq_")
        cls.model = T.tiny_model_dir()
        cls.ids = T.check_ids()
        cls.path, cls.rec, cls.roles, cls.inv = T.write_freeze(cls.tmp, local_is="edge")

    def test_full_in_host_checks_pass_and_fields(self):
        res = EQ.local_checks(self.model, 2, 2, "cpu", None, BRANCH_CPU, self.ids, 256, 12, 16, self.rec["freeze_sha256"])
        EQ.assert_no_timing_fields(res)
        self.assertTrue(res["prefix_bitwise_vs_hf"]); self.assertTrue(res["prefix_bitwise_vs_e0"]); self.assertTrue(res["prefix_greedy_equals_hf"])
        self.assertTrue(res["static_within_tolerance"]); self.assertTrue(res["static_greedy_equals_hf"])
        self.assertEqual(res["branch_in_use"]["cuda_graphs"], "cpu_control_flow"); self.assertTrue(res["branch_matches_freeze"])
        self.assertTrue(res["ownership"]["edge_slice_ok"]); self.assertTrue(res["ownership"]["server_slice_ok"]); self.assertTrue(res["ownership"]["server_has_no_embed_or_head"])
        self.assertEqual(res["ownership"]["edge_layers"], [0, 1, 4, 5]); self.assertEqual(res["ownership"]["server_layers"], [2, 3])
        w = res["wire"]
        self.assertTrue(w["wire_bytes_out_equal_formula"] and w["wire_bytes_in_equal_formula"] and w["hello_keys_subset_of_control_set"])
        self.assertTrue(w["server_events_only_control"] and w["server_fwd_count_equals_steps"])
        self.assertTrue(res["pass"])

    def test_branch_mismatch_is_detected(self):
        wrong = dict(BRANCH_CPU, attention="flash_attn_with_kvcache")
        res = EQ.local_checks(self.model, 2, 2, "cpu", None, wrong, self.ids, 256, 8, 16, self.rec["freeze_sha256"])
        self.assertFalse(res["branch_matches_freeze"]); self.assertFalse(res["pass"])

    def test_segments_only(self):
        res = EQ.local_checks(self.model, 2, 2, "cpu", None, BRANCH_CPU, self.ids, 256, 8, 16, self.rec["freeze_sha256"], segments_only=True)
        EQ.assert_no_timing_fields(res)
        self.assertTrue(res["segments"]["head"]["prefix_bitwise_vs_e0_run_layers"]); self.assertTrue(res["segments"]["tail"]["within_tolerance"])
        self.assertTrue(res["pass"])

    def test_cli_local_dev_writes_correctness_only(self):
        ids_f = Path(self.tmp) / "ids.json"; ids_f.write_text(json.dumps({"ids": self.ids}))
        out = Path(self.tmp) / "local_edge.json"
        r = subprocess.run([sys.executable, "-I", str(T.HERE / "e1_bench.py"), "equivalence", "local", "--freeze", str(self.path), "--model-dir", self.model,
                            "--device", "cpu", "--dev-mode", "--role", "edge", "--class", "12", "--dev-k", "2", "--dev-j", "2", "--max-len", "256",
                            "--chunk", "16", "--check-ids", str(ids_f), "--n-greedy", "8", "--out", str(out)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        d = json.loads(out.read_text())
        EQ.assert_no_timing_fields(d)
        self.assertTrue(d["pass"]); self.assertEqual(d["profile_role"], "screening"); self.assertEqual(d["freeze_sha256"], self.rec["freeze_sha256"])
        self.assertNotIn('"steps"', json.dumps(d)); self.assertNotIn("token_ns", json.dumps(d))   # no per-step timer arrays were written
        # the frozen cut cannot be overridden without --dev-mode
        r = subprocess.run([sys.executable, "-I", str(T.HERE / "e1_bench.py"), "equivalence", "local", "--freeze", str(self.path), "--model-dir", self.model,
                            "--device", "cpu", "--role", "edge", "--class", "12", "--dev-k", "2", "--dev-j", "2", "--check-ids", str(ids_f), "--out", str(out)],
                           capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)


class JointQ(unittest.TestCase):
    """Q1/Q2 in E0's formats over the E1 wire; both ends on this CPU so the pipeline floor is exact."""
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="e1jq_"))
        cls.model = T.tiny_model_dir()
        cls.path, cls.rec, cls.roles, cls.inv = T.write_freeze(cls.tmp, local_is="edge")
        cls.fz = cls.rec["freeze_sha256"]
        hf = EQ.hf_model(cls.model, "cpu")
        # synthetic "continuations" and Q2 references in E0's row formats
        cont = cls.tmp / "cont.jsonl"
        for i, key in enumerate(("mmlupro:1", "gpqa:2", "livemathbench:3")):
            ids = T.check_ids(20 + i, seed=10 + i)
            I.append_jsonl(cont, {"key": key, "prompt_ids": ids[:12], "cont_ids": ids[12:], "eos": False, "max_new": 8})
        cls.cont = cont
        EQ.q1_reference(hf, cont, cls.tmp / "q1_ref", "cpu")
        cls.q2ref = cls.tmp / "q2_ref.jsonl"
        for i, key in enumerate(("mmlupro:1", "gpqa:2", "livemathbench:3")):
            ids = T.check_ids(15, seed=20 + i)
            toks = EQ.reference_greedy(hf, ids, 8, "cpu")
            I.append_jsonl(cls.q2ref, {"key": key, "prompt_ids": ids, "tokens": toks, "margins": [1.0] * 8, "nan": False})
        cls.k, cls.j = 2, 2

    def _server(self):
        ssl = server_slice(self.model, self.k, self.j, "cpu", torch.bfloat16, None); ssl.cfg._attn_implementation = "e1_static"
        mid = LocalMiddle(ssl, ssl.layer_ids, 256, "static")
        ident = {"freeze_sha256": self.fz, "config_hash": E0RT.config_hash(ssl.cfg), "k": self.k, "j": self.j, "model_id": "dev"}
        srv = P.SegmentServer(mid.seg, ident, 64, 64)
        import socket
        s0 = socket.socket(); s0.bind(("127.0.0.1", 0)); port = s0.getsockname()[1]; s0.close()
        ready = threading.Event()
        t = threading.Thread(target=srv.serve_forever, args=("127.0.0.1", port, ready, True), daemon=True); t.start(); ready.wait(5)
        return port, t

    def test_joint_q1_q2_and_decide(self):
        port, t = self._server()
        sl = edge_slice(self.model, self.k, self.j, "cpu", torch.bfloat16, None); sl.cfg._attn_implementation = "e1_static"
        mid = P.RemoteMiddle("127.0.0.1", port, P.hello_for(sl, "dev", self.k, self.j, 256, self.fz), "cpu", 64, 64)
        e1 = USplitEdge(sl, self.k, self.j, mid, 256, "static")
        n = EQ.q1_compare(lambda ids: e1.teacher_forced_hidden(ids, 16), e1.sl.lm_head, self.cont, self.tmp / "q1_ref", self.tmp / "q1_split")
        self.assertEqual(n, 3)
        EQ.q2_run(e1, I.read_jsonl(self.q2ref), self.tmp / "q2_split.jsonl", 8, 16)
        mid.close(); t.join(5)
        # pipeline floor: prefix-mode in-host split vs HF (exact on one device); hardware floor: HF vs HF (zero)
        e1p = EQ.build_split(self.model, self.k, self.j, "cpu", "prefix", 256, None, BRANCH_CPU)
        EQ.q1_compare(lambda ids: e1p.teacher_forced_hidden(ids, 16), e1p.sl.lm_head, self.cont, self.tmp / "q1_ref", self.tmp / "q1_pipe")
        hf = EQ.hf_model(self.model, "cpu")
        EQ.q1_compare(lambda ids: EQ.reference_hidden(hf, ids, "cpu"), hf.lm_head, self.cont, self.tmp / "q1_ref", self.tmp / "q1_hw")
        import e0_q as Q
        s_split, s_pipe, s_hw = Q.q1_summary(self.tmp / "q1_split"), Q.q1_summary(self.tmp / "q1_pipe"), Q.q1_summary(self.tmp / "q1_hw")
        self.assertEqual(s_hw["D_pp"], 0.0); self.assertEqual(s_pipe["D_pp"], 0.0)
        rule = Q.q1_rule(s_split, s_pipe, s_hw)
        self.assertEqual(rule["D_threshold_pp"], S.E0.Q1_D_MARGIN_PP)      # E0 thresholds, unchanged
        res = EQ.decide([], [str(self.tmp / "q1_split")] * 2, [str(self.tmp / "q1_pipe")] * 2, str(self.tmp / "q1_hw"), str(self.q2ref),
                        [str(self.tmp / "q2_split.jsonl")] * 2, 211, 0, str(self.tmp / "decision.json"))
        EQ.assert_no_timing_fields(res)
        self.assertIn("Q1", res); self.assertIn("Q2", res)
        self.assertFalse(res["Q2"]["U-max(12)"]["pass"])     # 3 prompts, not the frozen 30: Q2 cannot pass; the rule is E0's
        self.assertEqual(res["Q2"]["U-max(12)"]["n"], 3); self.assertEqual(res["Q2"]["U-max(12)"]["required"], S.E0.Q2_MIN_NEAR_TIE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
