"""Regression tests for the CPU reanalysis of the archived E0 profiles (2026-11-11 correction pass).

CPU only. No inference. Uses the two archived E0 result tarballs in the repository root (sha256-verified) and
E0's gate code unchanged. Decides nothing about SPLIT; it pins the numbers the E1-OPT documents quote.

  ~/research/LLMRouterBench/.venv-wls-score/bin/python -I tests/test_reanalysis.py
"""
import hashlib
import inspect
import json
import re
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import e1_spec as S1  # noqa: E402
sys.path.insert(0, str(S1.E0_DIR))
import e0_r as R  # noqa: E402
import e0_spec as S  # noqa: E402

REPO = HERE.parent.parent
TARS = {"stage1": REPO / "e0_stage1_VALID_UB1_CONTINUE.tgz", "stage2": REPO / "e0_stage2_VALID_UB2_STOP.tgz"}
SHAS = {"stage1": S1.E0_STAGE1_SHA256, "stage2": S1.E0_STAGE2_SHA256}

# Pinned results (threshold_audit.py, 40-step bisection; f_true passes, f_false = f_true + 2**-40 fails).
F_DELTA3 = 0.829669        # largest f with delta >= 3.0 in some no-raw-text cell
F_PASS_STRICT = 0.688766   # largest f with delta >= 3.0 AND ci_lo > 0 in some cell (the gate's predicate)
F_PASS_NONSTRICT = 0.705394  # the withdrawn 2026-11-11 first-pass figure (ci_lo >= 0); must NOT pass the gate
F_DELTA3_EXACT = 0.840963  # largest f with delta rounded to 1e-6 pt >= 3.0 (delta = 3.00 exactly; e0_r.py's float says 2.9999999999999973)


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Fixture:
    data = None

    @classmethod
    def load(cls):
        if cls.data is not None:
            return cls.data
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        dirs = {}
        for key, tar in TARS.items():
            assert tar.exists(), f"missing archive {tar}"
            assert _sha(tar) == SHAS[key], f"sha256 mismatch for {tar}"
            dest = root / key
            dest.mkdir()
            with tarfile.open(tar) as tf:
                tf.extractall(dest, filter="data")
            found = list(dest.rglob(key))
            assert len(found) == 1, found
            dirs[key] = found[0]
        s1dir, s2dir = dirs["stage1"], dirs["stage2"]
        data = R.S1Data()
        W = R.boot_weights(data.strata)
        fres = json.loads((s1dir / "f_results.json").read_text())
        blocked = set(fres.get("no_split_pass_classes", []))
        umax = {int(m): tuple(v) for m, v in fres["u_max"].items() if v and int(m) not in blocked}
        eligible = {int(m): v for m, v in fres["edge_eligible"].items()}
        edge_prof = R.edge_profiles_from_dir(s1dir)
        eseg = R.load_seg_profiles(s1dir, R.EDGE_KEYS, "P-edge-U(")
        sseg = R.load_seg_profiles(s2dir, R.SERVER_KEYS, "P-server-U(")
        disc = S.UB_SEGMENT_DISCOUNT
        edge_seg = {(M, k): R.scaled(eseg[k], disc) for M, (k, j) in umax.items()}
        ref = json.loads((s2dir.parent / "gates" / "gate_ub2.json").read_text())

        def gate(f):
            server_seg = {(M, k): R.scaled(sseg[k], disc * f) for M, (k, j) in umax.items()}
            return R.upper_bound_gate(data, edge_seg, server_seg, edge_prof, eligible, W, umax)

        cls.data = {"gate": gate, "ref": ref, "sseg": sseg}
        return cls.data


def nrt(g):
    return [c for c in g["cells"] if c["status"] == "evaluated" and c["privacy"] == "no-raw-text"]


class ReanalysisRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx = Fixture.load()

    def gate(self, f):
        return self.fx["gate"](f)

    def test_f1_reproduces_archived_gate_ub2_exactly(self):
        g = self.gate(1.0)
        ref = self.fx["ref"]
        ev, evr = [c for c in g["cells"] if c["status"] == "evaluated"], [c for c in ref["cells"] if c["status"] == "evaluated"]
        self.assertEqual(len(ev), len(evr))
        self.assertGreater(len(ev), 0)
        for a, b in zip(ev, evr):
            self.assertEqual((a["M"], a["privacy"], a["D_s"]), (b["M"], b["privacy"], b["D_s"]))
            self.assertEqual(a["delta_pt"], b["delta_pt"])          # bitwise, not approximate
            self.assertEqual(a["ci95"], b["ci95"])
            self.assertEqual(a["pass"], b["pass"])
            self.assertEqual(a["split_best"], b["split_best"])
            self.assertEqual(a["comp_best"], b["comp_best"])
        self.assertEqual(g["max_delta_bound_pt"], ref["max_delta_bound_pt"])
        self.assertTrue(g["decision"].startswith("STOP"))
        self.assertEqual(sum(c["pass"] for c in ev), 0)

    def test_gate_pass_predicate_is_strict_and_matches_preregistration(self):
        src = inspect.getsource(R.evaluate_grid)
        line = [l for l in src.splitlines() if '"pass"' in l and "ci95" in l]
        self.assertEqual(len(line), 1)
        self.assertRegex(line[0], r'delta_pt"\]\s*>=\s*S\.R_MARGIN_PT\s+and\s+st_\["ci95"\]\[0\]\s*>\s*0')
        self.assertNotIn(">= 0", line[0])
        self.assertEqual(S.R_MARGIN_PT, 3.0)
        self.assertEqual(S1.R_MARGIN_PT, 3.0)
        # A cell whose CI lower bound is exactly zero does not pass.
        g = self.gate(0.700)
        c = next(x for x in nrt(g) if x["M"] == 12 and x["D_s"] == 60)
        self.assertEqual(c["ci95"][0], 0.0)
        self.assertGreaterEqual(c["delta_pt"], 3.0)
        self.assertFalse(c["pass"])
        self.assertEqual(sum(x["pass"] for x in nrt(g)), 0)

    def test_per_cell_monotonicity_as_server_gets_faster(self):
        prev = None
        for f in np.round(np.arange(1.0, -0.001, -0.05), 3):
            cur = {(c["M"], c["D_s"]): (max(c["U_split"].values()), c["delta_pt"], c["ci95"][0], c["pass"]) for c in nrt(self.gate(float(f)))}
            if prev is not None:
                for key, (us, d, lo, p) in cur.items():
                    us0, d0, lo0, p0 = prev[key]
                    self.assertGreaterEqual(us, us0 - 1e-12, f"U_split fell at f={f} in {key}")
                    self.assertGreaterEqual(d, d0 - 1e-12, f"delta fell at f={f} in {key}")
                    self.assertGreaterEqual(lo, lo0 - 1e-12, f"ci_lo fell at f={f} in {key}")
                    self.assertFalse(p0 and not p, f"pass lost at f={f} in {key}")
            prev = cur

    def test_threshold_delta_ge_3(self):
        self.assertTrue(any(c["delta_pt"] >= 3.0 for c in nrt(self.gate(F_DELTA3))))
        self.assertFalse(any(c["delta_pt"] >= 3.0 for c in nrt(self.gate(F_DELTA3 + 1e-4))))
        self.assertAlmostEqual(1.0 / F_DELTA3, 1.205, places=3)

    def test_threshold_full_pass_strict(self):
        g_pass = self.gate(F_PASS_STRICT)
        self.assertTrue(any(c["pass"] for c in nrt(g_pass)))
        self.assertFalse(any(c["pass"] for c in nrt(self.gate(F_PASS_STRICT + 1e-4))))
        self.assertFalse(any(c["pass"] for c in nrt(self.gate(F_PASS_NONSTRICT))), "the withdrawn 0.705 figure must not pass the gate")
        self.assertAlmostEqual(1.0 / F_PASS_STRICT, 1.452, places=3)
        first = [c for c in nrt(g_pass) if c["pass"]]
        self.assertEqual([(c["M"], c["D_s"]) for c in first], [(12, 60)])
        a_ms = 1e3 * self.fx["sseg"][3].a
        self.assertAlmostEqual(a_ms, 66.89, delta=0.05)
        self.assertAlmostEqual(a_ms * F_PASS_STRICT, S1.SERVER_STEP_MS_MAX_FOR_PASS_K3_RAW, delta=0.1)
        self.assertAlmostEqual(a_ms * F_DELTA3, S1.SERVER_STEP_MS_MAX_FOR_DELTA3_K3_RAW, delta=0.1)

    def test_float_comparison_defect_is_known_and_rounded_rule_handles_it(self):
        # The frozen gate compares floats. At f = 0.83 the 12 GB / 60 s cell has U 0.485 vs 0.455: delta is exactly
        # 3.00 pt in arithmetic, but the float is below 3.0 and e0_r.py does not count it. E1-OPT pre-specifies
        # rounding to 1e-6 pt (e1_spec.decide_cell) and reports both flags. This test pins the defect and the rule.
        c = next(x for x in nrt(self.gate(0.83)) if x["M"] == 12 and x["D_s"] == 60)
        self.assertEqual(max(c["U_split"].values()), 0.485)
        self.assertEqual(max(c["U_comp"].values()), 0.455)
        self.assertLess(c["delta_pt"], 3.0)
        self.assertGreater(c["delta_pt"], 3.0 - 1e-9)
        self.assertGreaterEqual(round(c["delta_pt"], 6), 3.0)
        self.assertFalse(S1.decide_cell(c["delta_pt"], c["ci95"][0]))     # CI low is -2.5: no pass either way
        self.assertFalse(S1.decide_cell(5.25, 0.0))                         # exact zero CI bound does not pass
        self.assertFalse(S1.decide_cell(5.25, 1e-15))                       # float noise at zero does not pass
        self.assertTrue(S1.decide_cell(3.0 - 1e-12, 0.25 - 1e-12))          # exact 3.00 and 0.25 pass
        self.assertTrue(any(round(x["delta_pt"], 6) >= 3.0 for x in nrt(self.gate(F_DELTA3_EXACT))))
        self.assertFalse(any(round(x["delta_pt"], 6) >= 3.0 for x in nrt(self.gate(F_DELTA3_EXACT + 1e-4))))
        self.assertAlmostEqual(S1.F_SERVER_SCALE_FOR_DELTA3_EXACT, F_DELTA3_EXACT, places=5)
        # the strict pass threshold is the same under both arithmetic rules (CI low is exactly 0.0 just above it)
        g = self.gate(F_PASS_STRICT)
        self.assertEqual([S1.decide_cell(x["delta_pt"], x["ci95"][0]) for x in nrt(g)], [x["pass"] for x in nrt(g)])
        g = self.gate(F_PASS_STRICT + 1e-4)
        self.assertFalse(any(S1.decide_cell(x["delta_pt"], x["ci95"][0]) for x in nrt(g)))

    def test_spec_constants_match_threshold_json(self):
        thr = json.loads((HERE / "reanalysis/out/thresholds.json").read_text())["any_cell"]
        self.assertAlmostEqual(thr["f_for_delta_ge_3"], F_DELTA3, places=5)
        self.assertAlmostEqual(thr["f_for_full_pass"], F_PASS_STRICT, places=5)
        self.assertAlmostEqual(thr["f_for_pass_nonstrict_ci_ge_0"], F_PASS_NONSTRICT, places=5)
        self.assertAlmostEqual(S1.F_SERVER_SCALE_FOR_DELTA3, F_DELTA3, places=5)
        self.assertAlmostEqual(S1.F_SERVER_SCALE_FOR_PASS, F_PASS_STRICT, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
