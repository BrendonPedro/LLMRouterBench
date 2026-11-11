"""Stage-3 profiler and gate tests (CPU, tiny model, dev mode): fit-split child/exit-status, edge-segment and
server-segment profiles readable by E0's fitters, summarize-fit, and the gate's provenance rejections.
python -I tests/test_e1_profile_gate.py"""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1_testlib as T  # noqa: E402
import e1_spec as S  # noqa: E402
import e1_provenance as PV  # noqa: E402

PY = [sys.executable, "-I", str(T.HERE / "e1_bench.py")]


def run(args, **kw):
    return subprocess.run(PY + args, capture_output=True, text=True, timeout=900, **kw)


class ProfileAndGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="e1pg_"))
        cls.model = T.tiny_model_dir()
        cls.path, cls.rec, cls.roles, cls.inv = T.write_freeze(cls.tmp, local_is="edge")
        cls.s3, cls.s2 = cls.tmp / "stage3", cls.tmp / "stage2"
        cls.wl = T.tiny_workload(cls.tmp / "workload.json")
        cls.common = ["--freeze", str(cls.path), "--device", "cpu", "--dev-mode", "--workload", cls.wl]

    def test_01_fit_split_dev_dry(self):
        r = run(["fit-split", *self.common, "--model-dir", self.model, "--class", "12", "--out-dir", str(self.s3), "--cpu-dry", "--dev-depth-tokens", "40"])
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        f = json.loads((self.s3 / "split_fit_c12_k3.json").read_text())
        self.assertTrue(f["pass"]); self.assertEqual(f["exit_status"], 0); self.assertEqual((f["k"], f["j"]), (3, 3))
        self.assertEqual(f["provenance"]["profile_role"], "screening"); self.assertEqual(f["provenance"]["stage"], "3")
        self.assertEqual(f["final_context"], 48); self.assertIn("peak_allocator_view_mib", f)   # 40 depth tokens + 8 dev decode steps
        r = run(["fit-split", *self.common, "--model-dir", self.model, "--class", "12", "--out-dir", str(self.s3), "--cpu-dry"])
        self.assertNotEqual(r.returncode, 0)   # refuses to overwrite an attempt
        r = run(["fit-split", "--freeze", str(self.path), "--device", "cpu", "--model-dir", self.model, "--class", "12", "--k", "2", "--j", "2",
                 "--out-dir", str(self.tmp / "x"), "--cpu-dry", "--workload", self.wl])
        self.assertNotEqual(r.returncode, 0); self.assertIn("not permitted", r.stderr + r.stdout)   # cut override without dev mode

    def test_02_edge_segment_profile(self):
        r = run(["profile-edge-segment", *self.common, "--model-dir", self.model, "--class", "12", "--out-dir", str(self.s3), "--with-depth", "0",
                 "--dev-max-requests", "3", "--max-len", "4096"])
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        import e0_r as R
        prof = R.load_seg_profiles(self.s3, R.EDGE_KEYS, "P-edge-U(")
        self.assertIn(3, prof)
        rows = R.request_rows(self.s3 / "P-edge-U(3,3)-cap12.jsonl")
        self.assertEqual(len(rows), 1); self.assertEqual(rows[0]["provenance"]["profile_role"], "screening")
        self.assertEqual(rows[0]["runtime"], "E1-OPT"); self.assertEqual(rows[0]["mode"], "static")

    def test_03_server_segment_profile(self):
        r = run(["profile-server-segment", *self.common, "--model-dir", self.model, "--class", "12", "--out-dir", str(self.s2),
                 "--dev-max-requests", "3", "--max-len", "4096"])
        # this host is frozen as the edge: acting as the server must be refused
        self.assertNotEqual(r.returncode, 0); self.assertIn("asked to act as server", r.stderr + r.stdout)
        r = run(["profile-server-segment", "--freeze", str(self.path), "--device", "cpu", "--host", self.roles["server"]["host"], "--workload", self.wl,
                 "--model-dir", self.model, "--class", "12", "--out-dir", str(self.s2), "--dev-max-requests", "3", "--max-len", "4096"])
        self.assertNotEqual(r.returncode, 0); self.assertIn("--host", r.stderr + r.stdout)   # identity override needs dev mode
        tmp2 = Path(tempfile.mkdtemp(prefix="e1srv_"))
        path2, rec2, roles2, inv2 = T.write_freeze(tmp2, local_is="server")
        r = run(["profile-server-segment", "--freeze", str(path2), "--device", "cpu", "--dev-mode", "--workload", self.wl, "--model-dir", self.model,
                 "--class", "12", "--out-dir", str(self.s2), "--dev-max-requests", "3", "--max-len", "4096"])
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        import e0_r as R
        prof = R.load_seg_profiles(self.s2, R.SERVER_KEYS, "P-server-U(")
        self.assertIn(3, prof)
        rows = R.request_rows(self.s2 / "P-server-U(3,3)-load0.jsonl")
        self.assertEqual(rows[0]["provenance"]["role"], "server"); self.assertEqual(rows[0]["provenance"]["stage"], "2")
        self.assertEqual({s["phase"] for s in rows[0]["steps"]}, {"prefill", "decode"})
        type(self).path2, type(self).rec2 = path2, rec2

    def test_04_summarize_and_gate_rejections(self):
        r = run(["summarize-fit", *self.common, "--dir", str(self.s3)])
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        f = json.loads((self.s3 / "f_results.json").read_text())
        self.assertEqual(f["u_max"]["12"], [3, 3]); self.assertIsNone(f["u_max"]["16"]); self.assertEqual(f["provenance"]["profile_role"], "screening")
        self.assertFalse(f["valid"])   # desk-eligible competitors were not fitted (missing) -> defects listed
        gate = ["gate-ub", "--stage3-dir", str(self.s3), "--stage2-dir", str(self.s2), "--freeze", str(self.path), "--roles", str(self.tmp / "roles.json")]
        r = run(gate + ["--out", str(self.tmp / "gate.json")])
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("PROVENANCE CHECK FAILED", r.stderr + r.stdout); self.assertIn("not target", r.stderr + r.stdout)
        self.assertFalse((self.tmp / "gate.json").exists())
        # screening rows from a different freeze are rejected even with --allow-screening? No: allow-screening accepts screening
        # rows but still requires the SAME freeze hash. Stage 2 rows were produced under the server fixture's freeze (another hash).
        r = run(gate + ["--out", str(self.tmp / "gate.json"), "--allow-screening"])
        self.assertIn("freeze hash differs", r.stderr + r.stdout)
        # stamp the stage-2 rows with the edge fixture's freeze hash (what a real paired run has) and re-run as a labelled screen
        rows = [json.loads(l) for l in open(self.s2 / "P-server-U(3,3)-load0.jsonl") if l.strip()]
        for row in rows:
            row["provenance"]["freeze_sha256"] = self.rec["freeze_sha256"]
            row["provenance"]["runtime_branch"] = self.rec["machines"][self.roles["server"]["host"]]["branch"]
            row["provenance"]["machine"] = self.roles["server"]["host"]; row["provenance"]["gpu"] = {"name": "FAKE-PEER-GPU", "uuid": "GPU-peer-0000", "index": 0}
        with open(self.s2 / "P-server-U(3,3)-load0.jsonl", "w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        f["valid"] = True; f["harness_defects"] = []   # the dev fixture has no competitors; let E0's gate run to exercise the path
        (self.s3 / "f_results.json").write_text(json.dumps(f))
        r = run(gate + ["--out", str(self.tmp / "gate.json"), "--allow-screening"])
        self.assertEqual(r.returncode, 0, r.stderr[-3000:] + r.stdout[-2000:])
        self.assertTrue((self.tmp / "gate.screening.json").exists()); self.assertFalse((self.tmp / "gate.json").exists())
        side = json.loads((self.tmp / "gate.screening.e1decision.json").read_text())
        self.assertEqual(side["profile_role"], "screening"); self.assertIn("SCREEN", side["e1_decision"]); self.assertIn("e0_raw_pass", json.dumps(side))
        # an archived E0 directory (no provenance) is rejected outright
        SP = Path("/tmp/claude-1000/-home-brendonpedro-dev-cloud-slm-router/cc43ec3d-c786-405c-b343-6aa3b7bb800e/scratchpad/arch1/e0_runs/stage1")
        if SP.exists():
            r = run(["gate-ub", "--stage3-dir", str(SP), "--stage2-dir", str(self.s2), "--freeze", str(self.path), "--roles", str(self.tmp / "roles.json"),
                     "--out", str(self.tmp / "g2.json")])
            self.assertNotEqual(r.returncode, 0); self.assertIn("no provenance", r.stderr + r.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
