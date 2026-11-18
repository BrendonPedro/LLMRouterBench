"""GEMMA-SELECT CPU tests (no GPU, no network beyond loopback, no lab machine).

  <python with torch + transformers> -I -B tests/test_gsel.py
"""
import json
import re
import socket
import struct
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gsel_testlib as T  # noqa: E402
import gsel_common as C  # noqa: E402
import gsel_spec as S  # noqa: E402
import gsel_stats as ST  # noqa: E402
import gsel_workload as WL  # noqa: E402

GS = T.GS


def rows_of(n, split_time, edge_time, correct=1, bench="b"):
    return [{"key": f"q{i}", "bench": bench, "correct": {"S": correct, "Em": correct, "Es": correct},
             "time": {"S": split_time, "Em": edge_time, "Es": edge_time}} for i in range(n)]


class SpecAndWorkload(unittest.TestCase):
    def test_workload_is_pinned_and_reproducible(self):
        self.assertEqual(C.sha256_file(GS / S.WORKLOAD_FILE), S.WORKLOAD_SHA256)
        built = WL.build(WL.sample_rows())
        self.assertEqual(built, C.read_json(GS / S.WORKLOAD_FILE))
        self.assertEqual((len(built["utility"]), len(built["sentinel"]), len(built["correctness"])), (160, 24, 10))
        self.assertEqual(sorted(built["utility"]), sorted(built["queries"]))
        self.assertTrue(set(built["sentinel"]) <= set(built["utility"]) and set(built["correctness"]) <= set(built["utility"]))
        self.assertEqual({q["dataset"] for q in built["queries"].values()}, set(S.BENCHMARKS))

    def test_loaded_queries_carry_prompt_and_answer(self):
        w = WL.load()
        self.assertEqual(len(w["utility"]), 160)
        self.assertTrue(all(q["prompt"] and q["answer"] is not None for q in w["utility"]))

    def test_spec_is_consistent(self):
        self.assertEqual(S.BLOCKS[0], "S1")
        self.assertEqual(max(S.DEADLINES_S), S.QUERY_CUTOFF_S)
        self.assertEqual(sum(S.SENTINEL_PER_BENCH.values()), 24)
        self.assertTrue(re.fullmatch(r"[0-9a-f]{64}", S.SOURCE_MANIFEST_SHA256))


class Identity(unittest.TestCase):
    def test_source_manifest_rule_and_stop(self):
        d = T.tmpdir()
        (d / "a").mkdir()
        (d / "a" / "x.py").write_text("1")
        (d / "README.md").write_text("r")
        (d / ".gitignore").write_text("ignored: no source suffix")
        (d / "__pycache__").mkdir()
        (d / "__pycache__" / "y.py").write_text("skipped directory")
        n, h = C.source_manifest(d)
        self.assertEqual(n, 2)
        (d / "a" / "x.py").write_text("2")
        self.assertNotEqual(C.source_manifest(d)[1], h)
        with self.assertRaises(C.Stop):
            C.check_source(d)

    def test_checkpoint_check(self):
        d = T.tmpdir()
        (d / "w.bin").write_bytes(b"abc")
        files = {"w.bin": (C.sha256_bytes(b"abc"), 3)}
        self.assertTrue(C.check_checkpoint(d, True, files)["w.bin"]["sha256"])
        for bad in ({"w.bin": (C.sha256_bytes(b"abd"), 3)}, {"w.bin": (C.sha256_bytes(b"abc"), 4)}, {"missing": ("0" * 64, 1)}):
            with self.assertRaises(C.Stop):
                C.check_checkpoint(d, True, bad)

    def test_estimate_from_header_only(self):
        d = T.tmpdir()
        t = lambda *shape: {"dtype": "BF16", "shape": list(shape), "data_offsets": [0, 0]}  # noqa: E731
        hdr = {"model.language_model.embed_tokens.weight": t(10, 4), "model.language_model.embed_tokens_per_layer.weight": t(10, 6),
               "model.language_model.layers.0.w": t(5), "model.language_model.layers.1.w": t(5), "model.language_model.layers.2.w": t(7),
               "model.language_model.norm.weight": t(4), "model.vision_tower.w": t(100), "__metadata__": {"format": "pt"}}
        blob = json.dumps(hdr).encode()
        (d / "model.safetensors").write_bytes(struct.pack("<Q", len(blob)) + blob)
        e = C.estimate_bytes(d, cut=2)
        self.assertEqual(e["n_params"], {"edge": 40 + 60 + 10, "server": 7 + 4 + 40, "whole": 40 + 60 + 17 + 4 + 100})
        self.assertEqual(e["server_float32"], 4 * 51)
        self.assertEqual(C.need_bytes(1000), 1250 + (1 << 30))
        self.assertEqual(C.need_bytes(1000, S.WHOLE_LOAD_FACTOR), 2000 + (1 << 30))

    def test_site_file_rules(self):
        d = T.tmpdir()
        ok = T.site_file(d, "edge", d)
        self.assertEqual(C.load_site(ok, "edge")["role"], "edge")
        with self.assertRaises(C.Stop):
            C.load_site(ok, "server")
        with self.assertRaises(C.Stop):
            C.load_site(T.site_file(d, "edge", d, port=9), "edge")              # a port another service uses
        inside = d / "src" / "key"
        inside.write_text("k")
        with self.assertRaises(C.Stop):
            C.load_site(T.site_file(d, "edge", d, credential_file=str(inside)), "edge")
        site = C.load_site(ok, "edge")
        Path(site["credential_file"]).chmod(0o644)
        with self.assertRaises(C.Stop):
            C.check_credential(site)

    def test_role_check(self):
        site = {"expected_hostname": "h", "gpu_uuid": "GPU-aa"}
        rows = [{"uuid": "aa", "name": "NVIDIA RTX A6000", "total_mib": 49140, "used_mib": 1, "free_mib": 49139}]
        self.assertEqual(C.check_role("server", site, {"hostname": "h", "arch": "x86_64"}, rows)["role"], "server")
        with self.assertRaises(C.Stop):
            C.check_role("server", site, {"hostname": "other", "arch": "x86_64"}, rows)
        with self.assertRaises(C.Stop):
            C.check_role("edge", site, {"hostname": "h", "arch": "x86_64"}, rows)
        edge_rows = [{"uuid": "bb", "name": "NVIDIA GB10", "total_mib": None, "used_mib": None, "free_mib": None}]
        self.assertEqual(C.check_role("edge", site, {"hostname": "h", "arch": "aarch64"}, edge_rows)["role"], "edge")

    def test_target_command_refuses_outside_the_tag(self):
        head, tag = C.git("rev-parse", "HEAD"), C.git("rev-parse", "--verify", "--quiet", f"{S.TAG}^{{commit}}")
        if head is not None and head == tag and not (C.git("status", "--porcelain", "--", "diagnostics/deployment_gemma_select") or "").strip():
            self.assertEqual(C.freeze_check()["tag"], S.TAG)
        else:
            with self.assertRaises(C.Stop):
                C.freeze_check()
        self.assertEqual(C.freeze_check(dev=True), {"dev": True})


class Resources(unittest.TestCase):
    ROWS = "GPU-aa, NVIDIA RTX A6000, 49140, 37787, 11353\nGPU-bb, NVIDIA RTX A6000, 49140, 2148, 46992\nGPU-cc, NVIDIA GB10, [N/A], [N/A], [N/A]"
    MEM = "MemTotal: 125000000 kB\nMemAvailable: 81788928 kB\nSwapTotal: 16000000 kB\nSwapFree: 6353000 kB\n"

    def test_gate_passes_only_with_enough_free_memory(self):
        rows, mem = C.parse_gpu_rows(self.ROWS), C.meminfo(self.MEM)
        self.assertEqual(rows[2]["free_mib"], None)
        need = C.need_bytes(9_543_000_000)                                       # about 12.1 GiB
        ok = C.memory_gate("server", {"device": "cuda", "gpu_uuid": "GPU-bb"}, need, rows, mem)
        self.assertTrue(ok["pass"] and ok["gpu_free_mib"] == 46992)
        with self.assertRaises(C.Stop):
            C.memory_gate("server", {"device": "cuda", "gpu_uuid": "GPU-aa"}, need, rows, mem)   # 11,353 MiB free is not enough
        with self.assertRaises(C.Stop):
            C.memory_gate("server", {"device": "cuda", "gpu_uuid": "GPU-zz"}, need, rows, mem)   # GPU not visible
        with self.assertRaises(C.Stop):
            C.memory_gate("edge", {"device": "cuda", "gpu_uuid": "GPU-cc"}, 90 << 30, rows, mem)  # more than the RAM available
        self.assertTrue(C.memory_gate("edge", {"device": "cuda", "gpu_uuid": "GPU-cc"}, 30 << 30, rows, mem)["pass"])
        self.assertGreater(mem["swap_used"], 9 << 30)

    def test_other_processes_are_counted_never_named(self):
        apps = "GPU-bb, 111, 1660\nGPU-bb, 222, 33322\nGPU-aa, 333, 9220\nGPU-bb, 444, 500"
        agg = C.parse_apps(apps, "GPU-bb", own_pids=(444,))
        self.assertEqual(agg, {"other_processes": 2, "other_mib": 34982})
        self.assertEqual(set(agg), {"other_processes", "other_mib"})

    def test_port_probe(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
        self.assertFalse(C.port_free(port))
        s.close()
        self.assertTrue(C.port_free(port))

    def test_outputs_are_written_once(self):
        d = T.tmpdir()
        C.write_json(d / "x.json", {"a": 1})
        with self.assertRaises(FileExistsError):
            C.write_json(d / "x.json", {"a": 2})
        self.assertEqual(C.read_json(d / "x.json"), {"a": 1})


class Rules(unittest.TestCase):
    def test_utility_and_decision(self):
        rows = rows_of(100, split_time=4.0, edge_time=8.0)                       # same answers, split twice as fast
        cells = {c["deadline_s"]: c for c in ST.decide(rows, n_boot=200)}
        self.assertEqual(cells[5]["selected"], "SPLIT")                         # on time only with the split
        self.assertAlmostEqual(cells[5]["delta_pt"], 100.0)
        self.assertEqual(cells[10]["selected"], "EDGE")                         # both on time: no advantage
        self.assertAlmostEqual(cells[10]["delta_pt"], 0.0)
        slow = {c["deadline_s"]: c for c in ST.decide(rows_of(100, 9.0, 4.0), n_boot=200)}
        self.assertEqual(slow[5]["selected"], "EDGE")
        self.assertLess(slow[5]["delta_pt"], 0)

    def test_margin_and_interval_are_both_required(self):
        rows = rows_of(88, 20.0, 20.0) + rows_of(10, 4.0, 4.0) + rows_of(2, 4.0, 8.0)   # +2 points at 5 s: below the margin
        c5 = ST.decide(rows, deadlines=(5,), n_boot=300)[0]
        self.assertAlmostEqual(c5["delta_pt"], 2.0)
        self.assertEqual(c5["selected"], "EDGE")
        rows = rows_of(80, 20.0, 20.0) + rows_of(10, 4.0, 4.0) + rows_of(10, 4.0, 8.0)  # +10 points, interval above zero
        self.assertEqual(ST.decide(rows, deadlines=(5,), n_boot=300)[0]["selected"], "SPLIT")
        noisy = rows_of(50, 4.0, 8.0) + rows_of(50, 8.0, 4.0)                    # +0 on average
        self.assertEqual(ST.decide(noisy, deadlines=(5,), n_boot=300)[0]["selected"], "EDGE")

    def test_best_edge_is_reselected_and_cells_can_be_excluded(self):
        rows = [{"key": f"q{i}", "bench": "b", "correct": {"S": 1, "Em": 1, "Es": 1}, "time": {"S": 4, "Em": 20, "Es": 4.5}} for i in range(40)]
        c = ST.decide(rows, deadlines=(5,), n_boot=100)[0]
        self.assertEqual((c["best_edge"], c["selected"]), ("Es", "EDGE"))
        wrong = rows_of(40, 1.0, 1.0, correct=0)
        c = ST.decide(wrong, deadlines=(5,), n_boot=50)[0]
        self.assertFalse(c["evaluated"])
        self.assertIsNone(c["selected"])

    def test_bootstrap_is_deterministic_and_stratified(self):
        rows = rows_of(30, 4.0, 8.0, bench="a") + rows_of(30, 8.0, 4.0, bench="b")
        a, b = ST.bootstrap_delta(rows, 5, n_boot=400), ST.bootstrap_delta(rows, 5, n_boot=400)
        self.assertEqual(a, b)
        self.assertEqual(a, [0.0, 0.0])                                          # strata keep their sizes: the difference is fixed

    def test_correctness_gate(self):
        pos = [{"ref_top1": 1, "split_top1": 1, "ref_margin": 2.0, "kl": 1e-4, "finite": True} for _ in range(640)]
        greedy = [{"identical": True, "first_divergence": None, "ref_margin_there": None} for _ in range(10)]
        self.assertEqual(ST.correctness_gate(pos, greedy)["verdict"], "PASS")
        three = [dict(p, split_top1=2) if i < 3 else p for i, p in enumerate(pos)]
        self.assertEqual(ST.correctness_gate(three, greedy)["verdict"], "PASS")   # 3 of 640 = 99.53 %
        four = [dict(p, split_top1=2) if i < 4 else p for i, p in enumerate(pos)]
        self.assertEqual(ST.correctness_gate(four, greedy)["verdict"], "FAIL")
        tie = [dict(p, split_top1=2, ref_margin=0.1) if i < 50 else p for i, p in enumerate(pos)]
        self.assertEqual(ST.correctness_gate(tie, greedy)["verdict"], "PASS")     # near ties are not decisive
        self.assertEqual(ST.correctness_gate([dict(p, kl=0.2) for p in pos], greedy)["verdict"], "FAIL")
        self.assertEqual(ST.correctness_gate([dict(pos[0], finite=False)] + pos[1:], greedy)["verdict"], "FAIL")
        near = [{"identical": False, "first_divergence": 5, "ref_margin_there": 0.1}] + greedy[1:]
        self.assertEqual(ST.correctness_gate(pos, near)["verdict"], "PASS")
        two = [{"identical": False, "first_divergence": 5, "ref_margin_there": 3.0}] * 2 + greedy[2:]
        self.assertEqual(ST.correctness_gate(pos, two)["verdict"], "FAIL")
        self.assertEqual(ST.correctness_gate(pos, greedy[:9])["verdict"], "FAIL")  # a prompt is missing
        self.assertEqual(ST.first_divergence([1, 2, 3], [1, 2, 3]), None)
        self.assertEqual(ST.first_divergence([1, 2, 3], [1, 9, 3]), 1)
        self.assertEqual(ST.first_divergence([1, 2, 3], [1, 2]), 2)

    def test_bound_and_stability(self):
        prof = [{"q": q, "prefill_s": 0.01 + 0.001 * q, "decode_s": [0.02] * 4} for q in (64, 256, 1024)]
        m = ST.segment_model(prof)
        self.assertAlmostEqual(m["prefill_slope_s"], 0.001, places=6)
        self.assertAlmostEqual(m["decode_s"], 0.02)
        self.assertAlmostEqual(ST.bound_time(m, m, 100, 11), 0.9 * 2 * (0.01 + 0.1 + 10 * 0.02))
        rows = rows_of(40, split_time=4.0, edge_time=8.0)
        meta = {r["key"]: (100, 11) for r in rows}
        b = ST.bound_analysis(rows, meta, m, m, n_boot=100)
        self.assertTrue(b["optimistic"] and not b["would_stop"])
        s1 = [{"key": f"q{i}", "total_s": 10.0} for i in range(24)]
        self.assertTrue(ST.stability(s1, [dict(r, total_s=10.5) for r in s1])["stable"])
        self.assertFalse(ST.stability(s1, [dict(r, total_s=12.0) for r in s1])["stable"])
        self.assertFalse(ST.stability(s1, [])["stable"])

    def test_outcomes(self):
        edge = [{"evaluated": True, "selected": "EDGE"}]
        split = edge + [{"evaluated": True, "selected": "SPLIT"}]
        self.assertEqual(ST.outcome("PASS", edge, True, True), "SCREEN_CONFIRMED")
        self.assertEqual(ST.outcome("PASS", split, True, True), "SCREEN_CONTRADICTED")
        self.assertEqual(ST.outcome("FAIL", split, True, True), "CORRECTNESS_FAIL")
        self.assertEqual(ST.outcome("PASS", split, False, True), "UNSTABLE")
        self.assertEqual(ST.outcome("PASS", split, True, False), "INCOMPLETE")
        self.assertEqual(ST.outcome("PASS", [{"evaluated": False, "selected": None}], True, True), "NOT_EVALUABLE")
        self.assertEqual(ST.outcome("PASS", edge, True, True, split_ran=False), "SPLIT_INFEASIBLE")
        self.assertEqual(ST.outcome("PASS", edge, True, True, matched_edge_ran=False), "NOT_EVALUABLE")
        self.assertTrue({ST.outcome("PASS", edge, True, True), ST.outcome("PASS", split, True, True)} <= set(S.OUTCOMES))


class WholeModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import gsel_whole as W
        cls.W, cls.dir = W, T.make_tiny_checkpoint(T.tmpdir())
        model, cls.info = W.load_model(cls.dir, "float32", "cpu", "eager")
        cls.runner, cls.tok = W.WholeRunner(model, "cpu"), W.load_tokenizer(cls.dir)

    def test_tokenize_and_eos(self):
        ids = self.W.tokenize(self.tok, "hi")
        self.assertEqual(ids[0], 2)                                              # BOS first, generation prompt added
        self.assertTrue(all(isinstance(i, int) for i in ids) and len(ids) > 10)
        self.assertEqual(self.W.eos_ids(self.dir), [1])

    def test_greedy_matches_a_full_forward_and_teacher_forcing(self):
        import torch
        ids = self.W.tokenize(self.tok, "What is 2+2?")
        toks = self.runner.generate(ids, 12, eos=())["tokens"]
        self.assertEqual(len(toks), 12)
        seq = list(ids)
        for t in toks:                                                           # independent check without a cache
            with torch.inference_mode():
                nxt = int(self.runner.model(input_ids=torch.tensor([seq])).logits[0, -1].argmax())
            self.assertEqual(nxt, t)
            seq.append(t)
        rows = list(self.runner.teacher_forced(ids, toks))
        self.assertEqual([i for i, _ in rows], list(range(12)))
        self.assertEqual([int(r.argmax()) for _, r in rows], toks)

    def test_callback_cutoff_and_eos_stop(self):
        import gsel_adapter as AD
        ids = self.W.tokenize(self.tok, "abc")
        seen = []

        def cb(i, t):
            seen.append(t)
            if i == 2:
                raise AD.Cutoff()
        res = self.runner.generate(ids, 10, eos=(), on_token=cb, stop=AD.Cutoff)
        self.assertTrue(res["cutoff"] and len(res["tokens"]) == 3 == len(seen))
        first = self.runner.generate(ids, 10, eos=())["tokens"]
        stop_at = first[-1]
        self.assertEqual(self.runner.generate(ids, 10, eos=(stop_at,))["tokens"], first[:first.index(stop_at) + 1])

    def test_a_checkpoint_that_does_not_fill_the_model_is_refused(self):
        from safetensors.torch import load_file, save_file
        import shutil
        d = T.tmpdir()
        shutil.copytree(self.dir, d / "m")
        f = d / "m" / "model.safetensors"
        t = load_file(f)
        t.pop(sorted(k for k in t if "layers.0" in k)[0])
        save_file(t, f)
        with self.assertRaises(RuntimeError):
            self.W.load_model(d / "m", "float32", "cpu", "eager")


class Pipeline(unittest.TestCase):
    """Every stage in --dev mode on CPU with the test double: order, gates, write-once, summary."""

    @classmethod
    def setUpClass(cls):
        cls.root = T.tmpdir()
        cls.model = T.make_tiny_checkpoint(cls.root / "model")
        cls.listener = socket.socket()
        cls.listener.bind(("127.0.0.1", 0))
        cls.listener.listen(1)
        cls.edge = T.site_file(cls.root, "edge", cls.model, port=cls.listener.getsockname()[1], double={"delay_s": 0.002})
        free = socket.socket()
        free.bind(("127.0.0.1", 0))
        cls.server = T.site_file(cls.root, "server", cls.model, port=free.getsockname()[1])
        free.close()
        cls.erun, cls.srun = cls.root / "run_edge", cls.root / "run_server"

    @classmethod
    def tearDownClass(cls):
        cls.listener.close()

    def test_full_dev_pipeline(self):
        B = T.bench
        self.assertIn("GO", B("preflight", "--dev", "--role", "edge", "--site", self.edge).stdout)
        self.assertIn("GO", B("preflight", "--dev", "--role", "server", "--site", self.server).stdout)
        B("serve", "--dev", "--site", self.server)
        B("profile", "--dev", "--role", "server", "--site", self.server)
        r = B("timed", "--dev", "--site", self.edge, "--block", "S1", expect=3)        # no correctness PASS yet
        self.assertIn("correctness", r.stderr)
        B("reference", "--dev", "--site", self.edge)
        ref = C.read_json(self.erun / "reference.json")
        self.assertEqual(ref["screen_decision"]["decision"], "EDGE")
        self.assertTrue(all(i["self_consistent"] and i["finite"] for i in ref["items"]))
        B("reference", "--dev", "--site", self.edge, expect=1)                         # written once
        self.assertIn("PASS", B("correctness", "--dev", "--site", self.edge).stdout)
        B("profile", "--dev", "--role", "edge", "--site", self.edge)
        r = B("timed", "--dev", "--site", self.edge, "--block", "Es", expect=3)        # block order is fixed
        self.assertIn("S1 must complete", r.stderr)
        for blk in S.BLOCKS:
            B("timed", "--dev", "--site", self.edge, "--block", blk, "--limit", 3, "--max-new", 6)
        B("timed", "--dev", "--site", self.edge, "--block", "S1", "--limit", 3, expect=3)   # a complete block is final
        rows = C.read_jsonl(self.erun / "timed_S1.jsonl")
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(len(r["stamps_s"]) == r["n_out"] and r["total_s"] >= r["ttft_s"] > 0 and r["impl"] for r in rows))
        self.assertEqual([r["ids_sha256"] for r in rows], [r["ids_sha256"] for r in C.read_jsonl(self.erun / "timed_Em.jsonl")])
        self.assertEqual([r["tokens"] for r in rows], [r["tokens"] for r in C.read_jsonl(self.erun / "timed_Em.jsonl")])
        B("score", "--dev", "--edge-run", self.erun, "--scorer", "exact")
        B("summary", "--edge-run", self.erun, "--server-run", self.srun, "--out", self.root / "bad", expect=3)   # target mode needs the tag
        self.assertEqual(C.read_json(self.erun / "timed_S1.done.json")["dev"], True)
        B("summary", "--dev", "--edge-run", self.erun, "--server-run", self.srun, "--out", self.root / "out")
        s = C.read_json(self.root / "out" / "summary.json")
        self.assertTrue(s["dev"] and s["blocks_complete"] and s["correctness"]["verdict"] == "PASS" and s["environment_consistent"])
        self.assertIn(s["outcome"], S.OUTCOMES)
        self.assertEqual(len(s["cells"]), len(S.DEADLINES_S))
        self.assertIn("network", s)
        self.assertIn("bound", s)
        self.assertNotIn("host", json.dumps(s))                                        # the summary carries no host record
        self.assertNotIn("uuid", json.dumps(s))
        # tampering with a completed block is detected
        f = self.erun / "timed_Es.jsonl"
        f.write_text(f.read_text().replace('"n": 0', '"n": 0 '))
        B("summary", "--dev", "--edge-run", self.erun, "--server-run", self.srun, "--out", self.root / "out2", expect=3)

    def test_wrong_split_fails_correctness_and_blocks_timing(self):
        root = T.tmpdir()
        site = T.site_file(root, "edge", self.model, port=self.listener.getsockname()[1], double={"fault": "noise"})
        T.bench("reference", "--dev", "--site", site)
        self.assertIn("FAIL", T.bench("correctness", "--dev", "--site", site, expect=2).stdout)
        T.bench("timed", "--dev", "--site", site, "--block", "S1", expect=3)

    def test_crashed_block_is_kept_and_repeats_are_capped(self):
        root = T.tmpdir()
        site = T.site_file(root, "edge", self.model, port=self.listener.getsockname()[1])
        T.bench("reference", "--dev", "--site", site)
        T.bench("correctness", "--dev", "--site", site)
        run = root / "run_edge"
        (run / "timed_S1.jsonl").write_text('{"partial": true}\n')                     # a block that died
        T.bench("timed", "--dev", "--site", site, "--block", "S1", "--limit", 2, "--max-new", 4)
        self.assertTrue((run / "timed_S1.jsonl.crashed-1").exists())
        self.assertEqual(C.read_json(run / "timed_S1.done.json")["attempts"], 2)
        (run / "timed_Es.jsonl").write_text("x\n")
        for n in (1, 2):
            (run / f"timed_Es.jsonl.crashed-{n}").write_text("x\n")
        r = T.bench("timed", "--dev", "--site", site, "--block", "Es", "--limit", 2, expect=3)
        self.assertIn("INCOMPLETE", r.stderr)

    def test_failed_attempts_decide_the_infeasible_outcomes(self):
        import gsel_bench as B
        root = T.tmpdir()
        site = T.site_file(root, "edge", self.model, port=self.listener.getsockname()[1])
        run = root / "run_edge"
        self.assertEqual(B.summarise(run, None, dev=True)["outcome"], "INCOMPLETE")            # nothing ran
        bad = T.site_file(root, "edge", root / "no_such_model", port=self.listener.getsockname()[1])
        for _ in range(2):
            T.bench("reference", "--dev", "--site", bad, expect=1)                               # the whole model cannot load
        self.assertEqual(B.summarise(run, None, dev=True)["outcome"], "NOT_EVALUABLE")
        root2 = T.tmpdir()
        site2 = T.site_file(root2, "edge", self.model, port=self.listener.getsockname()[1])
        T.bench("reference", "--dev", "--site", site2)
        self.assertEqual(B.summarise(root2 / "run_edge", None, dev=True)["outcome"], "INCOMPLETE")
        broken = json.loads(Path(site2).read_text())
        broken["adapter_path"] = str(GS / "tests" / "gsel_testlib.py")                           # no adapter in that file
        p = root2 / "site_broken.json"
        p.write_text(json.dumps(broken))
        for _ in range(2):
            T.bench("correctness", "--dev", "--site", p, expect=1)
        s = B.summarise(root2 / "run_edge", None, dev=True)
        self.assertEqual((s["outcome"], s["failed_attempts"]["correctness"]), ("SPLIT_INFEASIBLE", 2))
        self.assertEqual(s["screen_decision"]["decision"], "EDGE")

    def test_dev_only_switches_are_refused_in_target_mode(self):
        r = T.bench("timed", "--site", self.edge, "--block", "S1", "--limit", 2, expect=3)
        self.assertIn("dev only", r.stderr)
        r = T.bench("reference", "--site", self.edge, expect=3)                        # target mode needs the frozen tag
        self.assertIn("STOP", r.stderr)


class Publication(unittest.TestCase):
    """Nothing that names a machine, a user, a path or a device may be committed here. Site-specific patterns (names,
    ports, identifiers) are read from a private file named by GSEL_PRIVATE_PATTERNS, one regular expression per line."""
    GENERIC = ["/" + "home/", "/" + "mnt/", r"\bGPU-[0-9a-f]{8}", r"\b(?!127\.0\.0\.1\b)\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", r"BEGIN [A-Z ]*PRIVATE KEY"]

    def patterns(self):
        import os
        extra = os.environ.get("GSEL_PRIVATE_PATTERNS")
        lines = [l.strip() for l in Path(extra).read_text().splitlines() if l.strip()] if extra else []
        return re.compile("|".join(self.GENERIC + lines), re.I)

    def test_no_private_detail_in_the_experiment_directory(self):
        bad, pat = [], self.patterns()
        for p in sorted(GS.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                for n, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if pat.search(line):
                        bad.append(f"{p.relative_to(GS)}:{n}")
        self.assertEqual(bad, [])

    def test_no_adapter_or_site_file_is_shipped(self):
        names = [p.name for p in GS.rglob("*") if p.is_file()]
        self.assertFalse([n for n in names if n.startswith("site") and n.endswith(".json") and n != "site.example.json"])
        self.assertFalse([n for n in names if "private" in n.lower()])

    def test_earlier_experiments_are_untouched(self):
        for tag, rel in (("gemma-split-prereg-v1", "diagnostics/deployment_gemma_split"), ("e1opt-prereg-v1", "diagnostics/deployment_e1opt")):
            if C.git("rev-parse", "--verify", "--quiet", f"{tag}^{{commit}}") is None:
                continue
            r = subprocess.run(["git", "--no-optional-locks", "-C", str(C.REPO), "diff", "--quiet", tag, "--", rel])
            self.assertEqual(r.returncode, 0, f"{rel} differs from {tag}")


if __name__ == "__main__":
    unittest.main(verbosity=1)
