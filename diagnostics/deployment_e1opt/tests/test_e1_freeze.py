"""Freeze writer/checker tests (CPU). python -I tests/test_e1_freeze.py"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1_testlib as T  # noqa: E402
import e1_freeze as FZ  # noqa: E402
import e1_spec as S  # noqa: E402


class FreezeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="e1fz_")
        self.path, self.rec, self.roles, self.inv = T.write_freeze(self.tmp)

    def test_record_schema_and_hash_set(self):
        r = self.rec
        for key in ("experiment", "git_commit", "e1_dir", "file_sha256", "prereg_sha256", "protocol_version", "model", "transformers_required",
                    "cuts", "one_server_gpu", "static_cache", "buffer_strategy", "ownership", "seeds", "decision_rule", "roles", "machines", "freeze_sha256",
                    "e0", "wire_header_bytes", "decision_rule_version"):
            self.assertIn(key, r)
        files = set(r["file_sha256"])
        for must in ("e1_spec.py", "e1_segment.py", "e1_static_cache.py", "e1_attention.py", "e1_protocol.py", "e1_memory.py", "e1_bench.py",
                     "e1_freeze.py", "e1_inventory.py", "e1_provenance.py", "e1_equivalence.py", "e1_profile.py", "e1_joint.py", "e1_gate.py",
                     "tests/test_e1_cpu.py", "tests/test_reanalysis.py", "PREREG_E1OPT.md", "reanalysis/threshold_audit.py"):
            self.assertIn(must, files, must)
        self.assertEqual(r["e0"]["validated_commit"], S.E0_VALIDATED_COMMIT)
        self.assertIn("e0_r.py", r["e0"]["file_sha256"])
        self.assertEqual(r["decision_rule"]["gate_code_sha256"], r["e0"]["file_sha256"]["e0_r.py"])
        self.assertEqual(r["model"]["revision"], S.E0.SPLIT_REVISION)
        self.assertEqual(r["cuts"], {"12": [3, 3], "16": [5, 5], "24": [9, 9]})
        self.assertTrue(r["one_server_gpu"])
        self.assertEqual(r["decision_rule"]["delta_round_pt"], 1e-6)
        for host, m in r["machines"].items():
            self.assertIn(m["role"], ("server", "edge"))
            self.assertIs(m["branch"]["torch_compile"], False)
            self.assertIn(m["branch"]["attention"], S.ATTENTION_BRANCHES)
            self.assertIn(m["branch"]["cuda_graphs"], S.GRAPH_BRANCHES)
            for k in ("torch", "transformers", "python"):
                self.assertIn(k, m["versions"])
        self.assertEqual(r["freeze_sha256"], FZ.freeze_sha256(r))

    def test_write_refuses_overwrite_and_load_detects_edit(self):
        with self.assertRaises(FileExistsError):
            FZ.write(self.rec, self.path)
        edited = json.loads(self.path.read_text()); edited["notes"] = "tampered"
        p2 = Path(self.tmp) / "edited.json"; p2.write_text(json.dumps(edited))
        with self.assertRaises(FZ.FreezeMismatch):
            FZ.load(p2)

    def test_check_passes_on_this_host_without_tag(self):
        rec = FZ.check(self.path, require_tag=False)
        self.assertEqual(rec["freeze_sha256"], self.rec["freeze_sha256"])

    def test_check_requires_tag_by_default(self):
        with self.assertRaises(FZ.FreezeMismatch) as cm:
            FZ.check(self.path)
        self.assertIn("git_tag", str(cm.exception))

    def test_check_detects_file_change_missing_coverage_and_version_mismatch(self):
        rec = json.loads(self.path.read_text())
        rec["file_sha256"]["e1_segment.py"] = "0" * 64
        rec["freeze_sha256"] = FZ.freeze_sha256(rec)
        p = Path(self.tmp) / "f1.json"; p.write_text(json.dumps(rec))
        with self.assertRaises(FZ.FreezeMismatch) as cm:
            FZ.check(p, require_tag=False)
        self.assertIn("file changed since freeze: e1_segment.py", str(cm.exception))
        rec = json.loads(self.path.read_text())
        del rec["file_sha256"]["e1_joint.py"]
        rec["freeze_sha256"] = FZ.freeze_sha256(rec)
        p = Path(self.tmp) / "f2.json"; p.write_text(json.dumps(rec))
        with self.assertRaises(FZ.FreezeMismatch) as cm:
            FZ.check(p, require_tag=False)
        self.assertIn("file not covered by the freeze: e1_joint.py", str(cm.exception))
        live = dict(FZ.versions()); live["torch"] = "0.0.0+fake"
        with self.assertRaises(FZ.FreezeMismatch) as cm:
            FZ.check(self.path, require_tag=False, live_versions=live)
        self.assertIn("torch 0.0.0+fake", str(cm.exception))

    def test_check_refuses_unknown_host_and_wrong_role(self):
        with self.assertRaises(FZ.FreezeMismatch) as cm:
            FZ.check(self.path, require_tag=False, host="not-in-record")
        self.assertIn("not in the freeze record", str(cm.exception))
        with self.assertRaises(FZ.FreezeMismatch) as cm:
            FZ.check(self.path, require_tag=False, role="server")   # this host is frozen as the edge
        self.assertIn("asked to act as server", str(cm.exception))

    def test_branch_rule_uses_availability_only(self):
        b = FZ.branch_for_device({"compute_cap": 9.0, "flash_attn": {"importable": True, "has_flash_attn_with_kvcache": True}})
        self.assertEqual((b["attention"], b["cuda_graphs"], b["torch_compile"]), ("flash_attn_with_kvcache", "on", False))
        b = FZ.branch_for_device({"compute_cap": 12.1, "flash_attn": {"importable": False, "error": "no wheel"}})
        self.assertEqual(b["attention"], "sdpa_static_fallback"); self.assertIn("not importable", b["attention_reason"])
        b = FZ.branch_for_device({"compute_cap": 7.5, "flash_attn": {"importable": True, "has_flash_attn_with_kvcache": True}})
        self.assertEqual(b["attention"], "sdpa_static_fallback"); self.assertIn("compute capability", b["attention_reason"])
        self.assertNotIn("speed", json.dumps(b).lower()); self.assertNotIn("latency", json.dumps(b).lower())

    def test_cli_check_exit_codes(self):
        py = sys.executable
        r = subprocess.run([py, "-I", str(T.HERE / "e1_bench.py"), "check-freeze", "--freeze", str(self.path), "--no-tag"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        r = subprocess.run([py, "-I", str(T.HERE / "e1_bench.py"), "check-freeze", "--freeze", str(self.path)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("git_tag", r.stderr)
        r = subprocess.run([py, "-I", str(T.HERE / "e1_bench.py"), "check-freeze", "--freeze", str(self.path), "--no-tag", "--host", "nope"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)

    def test_build_refuses_infeasible_roles(self):
        roles = dict(self.roles, feasible=False)
        with self.assertRaises(ValueError):
            FZ.build(self.inv, roles)


if __name__ == "__main__":
    unittest.main(verbosity=2)
