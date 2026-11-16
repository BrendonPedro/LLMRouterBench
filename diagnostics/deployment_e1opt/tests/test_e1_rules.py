"""Arithmetic rule, fixed cuts, inventory/feasibility/roles, one-server-GPU, provenance rejection (CPU).
python -I tests/test_e1_rules.py"""
import json
import socket
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1_testlib as T  # noqa: E402
import e1_spec as S  # noqa: E402
import e1_inventory as INV  # noqa: E402
import e1_provenance as PV  # noqa: E402


class RoundingRule(unittest.TestCase):
    def test_delta_threshold_cases(self):
        self.assertTrue(S.decide_cell(2.9999999999999973, 0.25))    # exact 3.00 in float form -> pass
        self.assertFalse(S.decide_cell(2.9999994, 0.25))            # below 3.0 by more than the rounding -> fail
        self.assertTrue(S.decide_cell(3.0000004, 0.25))             # pass
        self.assertTrue(S.decide_cell(3.0, 0.25))

    def test_ci_cases(self):
        self.assertFalse(S.decide_cell(5.25, 0.0))                  # CI low = 0 -> fail (strict)
        self.assertTrue(S.decide_cell(5.25, 1e-6))                  # tiny positive at the rounding resolution -> pass
        self.assertTrue(S.decide_cell(5.25, 0.24999999999999467))
        self.assertFalse(S.decide_cell(5.25, 1e-9))                 # below the rounding resolution counts as zero
        self.assertFalse(S.decide_cell(5.25, -0.25))
        self.assertFalse(S.decide_cell(2.5, 1.0))                   # CI fine, delta below

    def test_decide_cells_rows(self):
        cells = [{"status": "evaluated", "M": 12, "privacy": "no-raw-text", "D_s": 60, "delta_pt": 2.9999999999999973, "ci95": [0.25, 8.0], "pass": False},
                 {"status": "evaluated", "M": 16, "privacy": "no-raw-text", "D_s": 60, "delta_pt": 5.25, "ci95": [0.0, 10.0], "pass": False},
                 {"status": "excluded: x", "M": 24}]
        rows = S.decide_cells(cells)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["delta_pt_raw"], 2.9999999999999973); self.assertEqual(rows[0]["delta_pt_rounded"], 3.0)
        self.assertTrue(rows[0]["e1_pass"]); self.assertFalse(rows[0]["e0_raw_pass"])
        self.assertFalse(rows[1]["e1_pass"]); self.assertFalse(rows[1]["e0_raw_pass"])


class FixedCuts(unittest.TestCase):
    def test_validate_cut(self):
        self.assertEqual(S.validate_cut(12, 3, 3), "target"); self.assertEqual(S.validate_cut(16, 5, 5), "target")
        self.assertEqual(S.validate_cut(12, 1, 1), "target")       # E0's U-min cells
        with self.assertRaises(S.CutError):
            S.validate_cut(12, 4, 4)
        with self.assertRaises(S.CutError):
            S.validate_cut(16, 3, 3)
        self.assertEqual(S.validate_cut(12, 4, 4, dev_mode=True), "screening")
        self.assertEqual(S.FIXED_CUTS[12], (3, 3)); self.assertEqual(S.FIXED_CUTS[16], (5, 5))
        self.assertIs(S.TORCH_COMPILE, False); self.assertIs(S.ONE_SERVER_GPU, True)


class Inventory(unittest.TestCase):
    def test_nvidia_smi_parser(self):
        txt = "0, NVIDIA H100 80GB HBM3, GPU-1234, 81559 MiB, 550.54.15, 9.0\n1, NVIDIA H100 80GB HBM3, GPU-5678, 81559 MiB, 550.54.15, 9.0\n"
        g = INV.parse_nvidia_smi_csv(txt)
        self.assertEqual([x["index"] for x in g], [0, 1]); self.assertEqual(g[0]["memory_mib"], 81559); self.assertEqual(g[1]["uuid"], "GPU-5678")
        self.assertEqual(g[0]["compute_cap"], 9.0)
        self.assertEqual(INV.parse_nvidia_smi_csv("NVIDIA-SMI has failed"), [])
        self.assertEqual(INV.parse_free_g("              total        used        free\nMem:            251          10         200\n"), 251)
        nics = INV.parse_nics("lo               UNKNOWN        00:00:00:00:00:00 <LOOPBACK,UP>\neth0             UP             aa:bb <BROADCAST>\n", lambda n: "Speed: 10000Mb/s")
        self.assertEqual(nics, [{"name": "eth0", "state": "UP", "speed_mbps": 10000}])

    def test_server_need_and_feasibility(self):
        need = INV.server_need_gib(3)
        self.assertGreater(need, 58); self.assertLess(need, 61)      # 52.7 GiB weights + 4.0 GiB KV + buffers + 2 GiB margin
        f = INV.feasibility(T.fake_inventory("a", [T.fake_gpu(0, memory_mib=24576)]))
        self.assertFalse(f["server"]["feasible"]); self.assertIn("one-server-GPU", f["server"]["reasons"][0])
        self.assertTrue(f["edge"]["feasible"])
        f = INV.feasibility(T.fake_inventory("b", [T.fake_gpu(0, memory_mib=81920)]))
        self.assertTrue(f["server"]["feasible"]); self.assertTrue(f["edge"]["feasible"])
        f = INV.feasibility(T.fake_inventory("c", [T.fake_gpu(0, memory_mib=81920)], competitor=False))
        self.assertTrue(f["server"]["feasible"]); self.assertFalse(f["edge"]["feasible"]); self.assertIn("competitor vLLM", f["edge"]["reasons"][0])
        f = INV.feasibility(T.fake_inventory("c2", [T.fake_gpu(0, memory_mib=81920)], vllm=False))   # v2 (§16 B): vLLM need not import
        self.assertTrue(f["edge"]["feasible"])                                                       # in the E1 runtime Python
        f = INV.feasibility(T.fake_inventory("d", [T.fake_gpu(0, memory_mib=131072, cc=7.5)]))
        self.assertFalse(f["server"]["feasible"])

    def test_one_server_gpu_lowest_index_and_no_aggregation(self):
        inv = T.fake_inventory("multi", [T.fake_gpu(0, uuid="G0", memory_mib=24576), T.fake_gpu(1, uuid="G1", memory_mib=81920), T.fake_gpu(2, uuid="G2", memory_mib=81920)])
        f = INV.feasibility(inv)
        self.assertTrue(f["server"]["feasible"]); self.assertEqual(f["server"]["gpu"]["uuid"], "G1")   # lowest index that holds the segment
        inv2 = T.fake_inventory("two24", [T.fake_gpu(0, uuid="A", memory_mib=24576), T.fake_gpu(1, uuid="B", memory_mib=24576)])
        self.assertFalse(INV.feasibility(inv2)["server"]["feasible"])   # 2 x 24 GB is not aggregated

    def test_role_assignment_rules(self):
        big = T.fake_inventory("big", [T.fake_gpu(0, uuid="BIG", memory_mib=81920)])
        small = T.fake_inventory("small", [T.fake_gpu(0, uuid="SMALL", memory_mib=24576)])
        r = INV.assign_roles(big, small)
        self.assertTrue(r["feasible"]); self.assertEqual(r["server"]["host"], "big"); self.assertEqual(r["edge"]["host"], "small")
        self.assertIn("6.3.2", r["rule_applied"])
        a = T.fake_inventory("alpha", [T.fake_gpu(0, uuid="A", memory_mib=81920)])
        b = T.fake_inventory("beta", [T.fake_gpu(0, uuid="B", memory_mib=131072)])
        r = INV.assign_roles(a, b)
        self.assertEqual(r["server"]["host"], "beta"); self.assertIn("capacity", r["rule_applied"])
        b2 = T.fake_inventory("beta", [T.fake_gpu(0, uuid="B", memory_mib=81920)])
        r = INV.assign_roles(a, b2)
        self.assertEqual(r["server"]["host"], "alpha"); self.assertIn("tie", r["rule_applied"])
        r = INV.assign_roles(a, b, lab_server="alpha", lab_edge="beta")
        self.assertEqual(r["server"]["host"], "alpha"); self.assertIn("6.3.1", r["rule_applied"])
        r = INV.assign_roles(big, small, lab_server="small")
        self.assertFalse(r["feasible"]); self.assertIn("infeasible", r["rule_applied"])
        r = INV.assign_roles(small, T.fake_inventory("small2", [T.fake_gpu(0, uuid="S2", memory_mib=24576)]))
        self.assertFalse(r["feasible"]); self.assertIn("INCOMPLETE", r["rule_applied"])
        r = INV.assign_roles(big, T.fake_inventory("nolink", [T.fake_gpu(0, uuid="N", memory_mib=24576)], route="ERROR no route"))
        self.assertFalse(r["feasible"]); self.assertIn("link", r["rule_applied"])
        self.assertNotIn("bandwidth", json.dumps(r).lower())


class AmendmentV2(unittest.TestCase):
    """PREREG_E1OPT §16 (v2): GB10 memory capacity fallback and the explicitly supplied competitor vLLM executable."""

    GB10_SMI = "0, NVIDIA GB10, GPU-5f2a9c1e-0b7d-4e33-9a1c-2d4e6f8a0b1c, [N/A], 580.95.05, 12.1\n"
    GB10_BYTES = 128524095488                                   # torch.cuda.get_device_properties(0).total_memory on the DGX Spark

    def test_numeric_nvidia_smi_is_kept(self):
        g = INV.parse_nvidia_smi_csv("0, NVIDIA RTX A6000, GPU-aaaa, 49140 MiB, 550.54.15, 8.6\n")
        self.assertEqual((g[0]["memory_mib"], g[0]["memory_source"]), (49140, "nvidia-smi"))
        td = {"devices": [{"index": 0, "uuid": "aaaa", "total_mib": 99999}]}     # torch never replaces a numeric nvidia-smi value
        INV.fill_memory_from_torch(g, td)
        self.assertEqual((g[0]["memory_mib"], g[0]["memory_source"]), (49140, "nvidia-smi"))

    def test_gb10_na_falls_back_to_torch_total_memory(self):
        g = INV.parse_nvidia_smi_csv(self.GB10_SMI)
        self.assertEqual((g[0]["memory_mib"], g[0]["memory_source"]), (None, None))
        total_mib = self.GB10_BYTES // (1 << 20)
        self.assertEqual(total_mib, 122570)                     # ~119.7 GiB physical capacity
        td = {"devices": [{"index": 0, "uuid": "5f2a9c1e-0b7d-4e33-9a1c-2d4e6f8a0b1c", "total_mib": total_mib}]}   # torch's uuid: no GPU- prefix
        INV.fill_memory_from_torch(g, td)
        self.assertEqual((g[0]["memory_mib"], g[0]["memory_source"]), (122570, "torch_cuda_properties"))
        inv = T.fake_inventory("spark", g, arch="aarch64")
        f = INV.feasibility(inv)
        self.assertTrue(f["server"]["feasible"]); self.assertTrue(f["edge"]["feasible"])
        before = T.fake_inventory("spark-v1", INV.parse_nvidia_smi_csv(self.GB10_SMI), arch="aarch64")   # the v1 defect
        self.assertFalse(INV.feasibility(before)["server"]["feasible"]); self.assertFalse(INV.feasibility(before)["edge"]["feasible"])

    def test_capacity_never_from_free_memory_or_a_wrong_device(self):
        src = Path(INV.__file__).read_text()
        self.assertNotIn("mem_get_info", src); self.assertNotIn("memory.free", src); self.assertNotIn("memory.used", src)
        self.assertIn("p.total_memory", src)
        g = [dict(x) for x in INV.parse_nvidia_smi_csv(self.GB10_SMI + "1, NVIDIA GB10, GPU-other, [N/A], 580.95.05, 12.1\n")]
        td = {"devices": [{"index": 0, "uuid": "unrelated-1", "total_mib": 1}, {"index": 1, "uuid": "unrelated-2", "total_mib": 2}]}
        INV.fill_memory_from_torch(g, td)                       # several GPUs and no UUID match: never matched by index
        self.assertEqual([(x["memory_mib"], x["memory_source"]) for x in g], [(None, "unavailable"), (None, "unavailable")])
        one = INV.parse_nvidia_smi_csv(self.GB10_SMI)
        INV.fill_memory_from_torch(one, {"devices": [{"index": 0, "uuid": "", "total_mib": 122570}]})   # one GPU each: by index
        self.assertEqual(one[0]["memory_source"], "torch_cuda_properties")
        none = INV.parse_nvidia_smi_csv(self.GB10_SMI)
        INV.fill_memory_from_torch(none, {"devices": []})       # torch sees nothing: unknown, not zero, not guessed
        self.assertEqual((none[0]["memory_mib"], none[0]["memory_source"]), (None, "unavailable"))

    def _script(self, d, name, text, mode=0o755):
        p = Path(d) / name
        p.write_text(text)
        p.chmod(mode)
        return str(p)

    def test_competitor_vllm_probe_fails_closed(self):
        import os
        from unittest import mock
        d = tempfile.mkdtemp(prefix="e1vllm_")
        site = Path(d) / "site"
        (site / "vllm").mkdir(parents=True)
        (site / "vllm" / "__init__.py").write_text("__version__ = '0.8.5.post1'\n")
        (site / "vllm-0.8.5.post1.dist-info").mkdir()
        (site / "vllm-0.8.5.post1.dist-info" / "METADATA").write_text("Metadata-Version: 2.1\nName: vllm\nVersion: 0.8.5.post1\n")
        cli = "import sys, vllm\nif sys.argv[1:] == ['--version']:\n    print(vllm.__version__)\n"   # as vllm's CLI answers --version
        good = self._script(d, "vllm", f"#!{sys.executable}\n{cli}")
        long_form = self._script(d, "vllm-pip", f"#!/bin/sh\n'''exec' \"{sys.executable}\" \"$0\" \"$@\"\n' '''\n{cli}")
        uv_form = self._script(d, "vllm-uv", f"#!/bin/sh\n'''exec' '{sys.executable}' \"$0\" \"$@\"\n' '''\n{cli}")
        other = self._script(d, "pip", f"#!{sys.executable}\nimport sys\nprint('pip 25.0')\n")   # another script of the same environment
        liar = self._script(d, "not-vllm", f"#!{sys.executable}\nraise SystemExit('I am not vllm')\n")
        with mock.patch.dict(os.environ, {"PYTHONPATH": str(site)}):
            for path in (good, long_form, uv_form):
                r = INV.probe_vllm(path)
                self.assertTrue(r["valid"], r); self.assertEqual(r["version"], "0.8.5.post1")
                self.assertEqual(r["resolved"], str(Path(path).resolve())); self.assertEqual(len(r["sha256"]), 64)
                self.assertTrue(r["vllm_file"].startswith(str(site))); self.assertTrue(r["python_executable"])
            for path in (other, liar):                           # a vLLM environment, but not the vLLM executable
                r = INV.probe_vllm(path)
                self.assertFalse(r["valid"], path); self.assertIn("not the vLLM executable", r["error"])
        self.assertFalse(INV.probe_vllm(good)["valid"])          # the same script, but its environment has no vllm: wrong executable
        self.assertIn("cannot import vllm", INV.probe_vllm(good)["error"])
        cases = {None: "no competitor", "relative/vllm": "absolute path", str(Path(d) / "missing"): "does not exist",
                 self._script(d, "noexec", f"#!{sys.executable}\n", 0o644): "not executable",
                 self._script(d, "shell", "#!/bin/sh\necho 0.8.5.post1\n"): "not a Python console script",
                 self._script(d, "versionliar", "#!/bin/sh\necho 0.8.5.post1\n"): "not a Python console script",
                 self._script(d, "badquote", "#!/usr/bin/env 'python3\n"): "not a Python console script"}   # no crash
        for path, why in cases.items():
            r = INV.probe_vllm(path)
            self.assertFalse(r["valid"], path); self.assertIn(why, r["error"], path)

    def test_collect_refuses_an_invalid_named_executable(self):
        from unittest import mock
        out = Path(tempfile.mkdtemp(prefix="e1inv_")) / "host.json"
        stub = lambda peer, vllm_bin: dict(T.fake_inventory("h", []), competitor_vllm=INV.probe_vllm(vllm_bin))   # noqa: E731  (control flow only)
        with mock.patch.object(INV, "collect", stub):
            for argv in (["--vllm-bin", "/nonexistent/bin/vllm"], [], ["--vllm-bin", "/x/vllm", "--no-competitor", "server only"]):
                with self.assertRaises(SystemExit) as cm:          # an invalid executable, or no explicit choice: nothing written
                    INV.main(["collect", *argv, "--out", str(out)])
                self.assertIn("REFUSED", str(cm.exception.code), argv); self.assertFalse(out.exists())
            INV.main(["collect", "--no-competitor", "the lab runs no competitor here", "--out", str(out)])
        rec = json.loads(out.read_text())["competitor_vllm"]
        self.assertEqual((rec["valid"], rec["no_competitor_reason"]), (False, "the lab runs no competitor here"))

    def test_frozen_runtime_checks_unchanged(self):
        import e1_freeze as FZ
        self.assertEqual(FZ.STRICT_VERSION_KEYS, ("torch", "torch_cuda", "transformers", "flash_attn", "python"))
        self.assertEqual(S.E0.TRANSFORMERS_VERSION, "4.51.3"); self.assertEqual(S.E0.VLLM_VERSION, "0.8.5.post1")
        self.assertEqual(S.FIXED_CUTS, {12: (3, 3), 16: (5, 5), 24: (9, 9)}); self.assertEqual(S.PRIMARY_CLASSES, (12, 16))
        self.assertIs(S.ONE_SERVER_GPU, True); self.assertEqual(S.DECISION_RULE_VERSION, "E1-OPT-r3")


class Provenance(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="e1pv_")
        self.path, self.rec, self.roles, self.inv = T.write_freeze(self.tmp, local_is="edge")
        self.me = socket.gethostname()
        self.gpu = {"name": "cpu", "uuid": "cpu", "index": 0}

    def good(self, **kw):
        p = PV.make(self.rec, "3", "edge", self.gpu, "target", (3, 3), 12, self.me)
        p.update(kw)
        return p

    def test_good_provenance_passes(self):
        self.assertEqual(PV.check(self.good(), self.rec, self.roles, "edge"), [])

    def test_each_rejection_rule(self):
        R = lambda p, **kw: PV.check(p, self.rec, self.roles, kw.get("expect", "edge"), kw.get("target", True))
        self.assertTrue(any("no provenance" in x for x in R(None)))
        self.assertTrue(any("freeze hash" in x for x in R(self.good(freeze_sha256="f" * 64))))
        self.assertTrue(any("experiment" in x for x in R(self.good(experiment="E0"))))
        self.assertTrue(any("not target" in x for x in R(self.good(profile_role="screening"))))
        self.assertEqual(R(self.good(profile_role="screening"), target=False), [])
        self.assertTrue(any("device identity" in x for x in R(self.good(gpu={"name": "x", "uuid": "", "index": 0}))))
        self.assertTrue(any("model" in x for x in R(self.good(model="Qwen/Qwen3-32B@other"))))
        self.assertTrue(any("runtime branch" in x for x in R(self.good(runtime_branch={"attention": "other"}))))
        self.assertTrue(any("not the inventoried edge" in x for x in R(self.good(machine="peer-fake-host"))))
        self.assertTrue(any("row role server != expected edge" in x for x in R(self.good(role="server"))))
        self.assertTrue(any("not frozen" in x for x in R(self.good(cut=[4, 4]))))
        self.assertTrue(any("decision rule" in x for x in R(self.good(decision_rule_version="E1-OPT-r1"))))
        self.assertTrue(any("missing" in x for x in R({"experiment": "E1-OPT"})))

    def test_validate_dir_and_stamp(self):
        d = Path(self.tmp) / "stage3"; d.mkdir()
        rows = [{"prompt_id": "short-0", "ok": True, "provenance": self.good()}, {"prompt_id": "short-1", "ok": True}]
        with open(d / "P-edge-U(3,3)-cap12.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        probs = PV.validate_dir(d, self.rec, self.roles, "edge", ("P-edge-*.jsonl",), ())
        self.assertEqual(len(probs), 1); self.assertIn(":2", probs[0]); self.assertIn("no provenance", probs[0])
        PV.stamp_jsonl(d / "P-edge-U(3,3)-cap12.jsonl", self.good())
        self.assertEqual(PV.validate_dir(d, self.rec, self.roles, "edge", ("P-edge-*.jsonl",), ()), [])
        (d / "f_results.json").write_text(json.dumps({"u_max": {}}))
        probs = PV.validate_dir(d, self.rec, self.roles, "edge", ("P-edge-*.jsonl",), ("f_results.json",))
        self.assertTrue(any("f_results.json" in x for x in probs))
        PV.stamp_json(d / "f_results.json", self.good())
        self.assertEqual(PV.validate_dir(d, self.rec, self.roles, "edge", ("P-edge-*.jsonl",), ("f_results.json",)), [])
        # archived E0 rows (no provenance) are rejected
        SP = Path("/tmp/claude-1000/-home-brendonpedro-dev-cloud-slm-router/cc43ec3d-c786-405c-b343-6aa3b7bb800e/scratchpad/arch1/e0_runs/stage1")
        if SP.exists():
            probs = PV.validate_dir(SP, self.rec, self.roles, "edge", ("P-edge-U(3,3)-cap12.jsonl",), ())
            self.assertTrue(probs and all("no provenance" in x for x in probs))


if __name__ == "__main__":
    unittest.main(verbosity=2)
