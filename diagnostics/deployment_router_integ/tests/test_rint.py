"""ROUTER-INTEG tests. CPU only: tiny random models, fixture workloads, a loopback wire, mock backends.

  python -I -B tests/test_rint.py            (an environment with torch and transformers 4.51.3)

Nothing here is a measurement. Tests that need files this machine may not have (the router encoder, the private
workload, the router study's repository) are skipped and say so.
"""
import json
import os
import shutil
import socket
import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rint_testlib as T  # noqa: E402  (puts the experiment directory on sys.path)
import rint_backend as B  # noqa: E402
import rint_common as C  # noqa: E402
import rint_eval as EV  # noqa: E402
import rint_models as M  # noqa: E402
import rint_policy as P  # noqa: E402
import rint_record as REC  # noqa: E402
import rint_spec as S  # noqa: E402
import rint_workload as WL  # noqa: E402

PRIVATE = Path(os.environ.get("RINT_PRIVATE_WORKLOAD", Path.home() / "research" / "router_integration_private" / "workload_private.jsonl"))
STUDY_REPO = Path(os.environ.get("RINT_STUDY_REPO", Path.home() / "research" / "LLMRouterBench"))
ENC = T.encoder_dir()


# --------------------------------------------------------------------------- #
class TestSpecAndPolicy(unittest.TestCase):
    def test_pair_is_the_routers_pair_only(self):
        self.assertEqual(set(S.MODELS), {S.DS, S.QW})
        self.assertEqual(S.MODELS[S.DS]["repo"], "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
        self.assertEqual(S.MODELS[S.QW]["repo"], "Qwen/Qwen3-8B")
        self.assertEqual(S.ROUTER_CLASS_TO_MODEL, {0: S.DS, 1: S.QW})
        self.assertEqual({m for m, _ in S.CONFIGS.values()}, {S.DS, S.QW})
        for bad in ("Qwen3-14B", "Qwen3-32B", "14B", "32B"):
            with self.assertRaises(ValueError):
                P.place(bad, S.POLICY_B)

    def test_revisions_are_those_of_the_label_stability_experiment(self):
        src = (STUDY_REPO / "diagnostics" / "scripts" / "wls_common.py")
        if not src.exists():
            self.skipTest("router study repository not on this machine")
        text = src.read_text()
        for m in S.MODEL_ORDER:
            self.assertIn(S.MODELS[m]["revision"], text)
        self.assertIn("TEMPERATURE = 0.2", text)
        self.assertIn("TOP_P = 1.0", text)
        self.assertEqual((S.TEMPERATURE, S.TOP_P, S.MODELS[S.QW]["top_k"], S.MODELS[S.DS]["top_k"]), (0.2, 1.0, 20, None))

    def test_data_policy_feasibility(self):
        self.assertEqual(P.feasible(S.EDGE, S.POLICY_A), (True, None))
        self.assertFalse(P.feasible(S.REMOTE, S.POLICY_A)[0])
        self.assertTrue(P.feasible(S.REMOTE, S.POLICY_B)[0])
        self.assertTrue(P.feasible(S.EDGE, S.POLICY_B)[0])
        for pol in (S.POLICY_A, S.POLICY_B):
            self.assertFalse(P.feasible(S.SPLIT, pol)[0])                 # not run: never reported as feasible

    def test_placement_never_violates_its_policy(self):
        for m in S.MODEL_ORDER:
            for pol in (S.POLICY_A, S.POLICY_B):
                for up in (True, False):
                    pl = P.place(m, pol, server_available=up)
                    if pl["config"] is not None:
                        self.assertTrue(P.feasible(pl["mode"], pol)[0])
                        self.assertEqual(S.CONFIGS[pl["config"]], (m, pl["mode"]))
        self.assertEqual(P.place(S.DS, S.POLICY_A)["config"], "DS_EDGE")
        self.assertEqual(P.place(S.QW, S.POLICY_B)["config"], "QW_REMOTE")
        fb = P.place(S.QW, S.POLICY_B, server_available=False)
        self.assertEqual((fb["config"], fb["fallback"]), ("QW_EDGE", True))
        self.assertEqual(P.place(S.DS, S.POLICY_A, server_available=False)["config"], "DS_EDGE")   # policy A never needs the server

    def test_matrix_lists_every_cell(self):
        mx = P.matrix()
        self.assertEqual(len(mx), 6)
        self.assertTrue(all(r[S.POLICY_A].startswith("infeasible") for r in mx if r["mode"] != S.EDGE))

    def test_seeds_differ_by_model_mode_and_phase(self):
        seeds = {C.request_seed("gpqa:1", m, x, ph) for m in S.MODEL_ORDER for x in (S.EDGE, S.REMOTE) for ph in ("shared", "live_A", "live_B")}
        self.assertEqual(len(seeds), 12)
        self.assertEqual(C.request_seed("gpqa:1", S.DS, S.EDGE, "shared"), C.request_seed("gpqa:1", S.DS, S.EDGE, "shared"))

    def test_stage_limits_fit_the_allotments(self):
        self.assertLessEqual(sum(S.STAGE_LIMIT_H["edge"].values()), S.ALLOTMENT_H["edge"])
        self.assertLessEqual(sum(S.STAGE_LIMIT_H["server"].values()), S.ALLOTMENT_H["server"])
        self.assertEqual(sum(S.ALLOTMENT_H.values()), S.GPU_HOURS_CEILING)
        self.assertEqual(S.QUERY_CUTOFF_S, max(S.DEADLINES_S))


# --------------------------------------------------------------------------- #
class TestManifests(unittest.TestCase):
    def test_both_checkpoints_have_the_same_tensors(self):
        _, td = M.manifest(S.DS)
        _, tq = M.manifest(S.QW)                                           # also checks the pinned QWEN-SELECT hashes
        self.assertEqual(set(td), set(tq))
        self.assertTrue(all(td[k]["shape"] == tq[k]["shape"] and td[k]["dtype"] == tq[k]["dtype"] == "BF16" for k in td))
        self.assertEqual(M.arch(S.DS), M.arch(S.QW))
        self.assertEqual(M.arch(S.DS)["num_hidden_layers"], 36)

    def test_deepseek_has_no_generation_config(self):
        man, _ = M.manifest(S.DS)
        self.assertNotIn("generation_config.json", man["files"])
        self.assertNotIn("generation_config.json", M.needed_files(S.DS, man))
        self.assertIn("generation_config.json", M.needed_files(S.QW, M.manifest(S.QW)[0]))

    def test_memory_need(self):
        both, one = M.need("server"), M.need("server", (S.DS,))
        self.assertGreater(both["required_free"], one["required_free"])
        self.assertEqual(both["weights"], 2 * one["weights"])
        self.assertLess(both["required_free"], 48 << 30)                   # must be loadable on one 48 GiB server GPU
        self.assertGreater(M.need("edge")["required_free"], both["required_free"])
        self.assertLessEqual(both["context_tokens"], S.CONTEXT_TOKENS)

    def test_verify_detects_a_wrong_file(self):
        d = T.TMP / "fake_ckpt"
        d.mkdir(exist_ok=True)
        rec = M.verify(d, S.DS, deep=False)
        self.assertFalse(rec["ok"])
        self.assertTrue(any(p.startswith("missing") for p in rec["problems"]))


# --------------------------------------------------------------------------- #
class TestRouter(unittest.TestCase):
    def test_heads_are_the_verified_export(self):
        import rint_router as RR
        h = RR.Heads()
        rep = C.read_json(C.RI_DIR / "router" / "router_build_report.json")
        self.assertEqual(h.sha256, rep["heads_canonical_sha256"])
        self.assertEqual(rep["n_checked"], 11467)
        self.assertEqual(rep["decisions_differing"], 0)
        self.assertLess(rep["max_abs_diff_vs_stored_score"], 1e-6)
        self.assertEqual(rep["problems"], [])
        self.assertEqual([rep["folds"][str(f)]["epochs"] for f in range(5)], [rep["folds"][str(f)]["epochs_stored"] for f in range(5)])

    def test_choice_mapping_and_threshold(self):
        import rint_router as RR
        self.assertEqual(RR.choice_of(0.3), S.QW)
        self.assertEqual(RR.choice_of(-0.3), S.DS)
        self.assertEqual(RR.choice_of(0.0), S.DS)                           # score > 0 -> Qwen3-8B; zero is not greater

    def test_fold_map(self):
        import rint_router as RR
        fm = RR.FoldMap()
        self.assertEqual(len(fm.map), 11467)
        with self.assertRaises(KeyError):
            fm.fold("notabenchmark:1")
        k = next(iter(fm.map))
        with self.assertRaises(ValueError):
            fm.fold(k, "some other text")

    @unittest.skipUnless(ENC and PRIVATE.exists(), "router encoder or private workload not on this machine")
    def test_end_to_end_scores_equal_the_pinned_scores(self):
        import rint_router as RR
        wl, priv = WL.load(PRIVATE)
        r = RR.Router(ENC, threads=4)
        worst, disagree, lat = 0.0, 0, []
        for q in wl["queries"]:
            d = r.route(q["key"], priv[q["key"]]["prompt"])
            self.assertEqual(d["fold"], q["fold"])
            worst = max(worst, abs(d["score"] - q["router_pinned"]["score"]))
            disagree += d["choice"] != q["router_pinned"]["choice"]
            lat.append(d["router_s"])
        print(f"\n  router end to end on {len(lat)} workload queries: max |score - pinned| {worst:.2e}, choices differing {disagree}, "
              f"median {1000 * sorted(lat)[len(lat) // 2]:.1f} ms on this desktop CPU")
        self.assertLess(worst, 1e-4)
        self.assertEqual(disagree, 0)


# --------------------------------------------------------------------------- #
class TestWorkload(unittest.TestCase):
    def setUp(self):
        self.wl = WL.public()
        self.ev = [q for q in self.wl["queries"] if q["role"] == "eval"]

    def test_sample_shape(self):
        self.assertEqual(len(self.ev), S.N_QUERIES)
        self.assertEqual(self.wl["n_profile"], S.N_PROFILE)
        self.assertFalse({q["dataset"] for q in self.wl["queries"]} & set(S.FRAME_EXCLUDED_BENCHMARKS))
        self.assertEqual(len({q["key"] for q in self.wl["queries"]}), S.N_QUERIES + S.N_PROFILE)
        self.assertEqual(self.wl["prompt_token_count_mismatches_vs_original_records"], [])

    def test_natural_distribution_is_kept(self):
        frame, got = self.wl["class4_frame_counts"], self.wl["class4_counts"]
        tot = sum(frame.values())
        for c in S.CLASS4:
            self.assertLess(abs(got[c] - S.N_QUERIES * frame[c] / tot), 2.0, c)       # ties are not removed, winners not enriched
        self.assertGreater(got["both_correct"] + got["both_wrong"], got["ds_only"] + got["qw_only"])

    def test_every_query_is_held_out_for_the_model_that_scores_it(self):
        import rint_router as RR
        fm = RR.FoldMap()
        for q in self.wl["queries"]:
            self.assertEqual(fm.map[q["key"]], [q["fold"], q["query_sha1"]])
            self.assertEqual(q["router_pinned"]["fold"], q["fold"])
            self.assertEqual(q["router_pinned"]["choice"], RR.choice_of(q["router_pinned"]["score"]))

    def test_orders(self):
        keys = {q["key"] for q in self.ev}
        for name in ("shared", "live"):
            o = self.wl["orders"][name]
            self.assertEqual({k for k, _ in o}, keys)
            self.assertEqual(len(o), len(keys))
            self.assertEqual(sum(f == S.DS for _, f in o), len(o) // 2)       # which model runs first is balanced
        self.assertNotEqual([k for k, _ in self.wl["orders"]["shared"]], [k for k, _ in self.wl["orders"]["live"]])
        self.assertFalse(set(self.wl["profile_keys"]) & keys)

    def test_caps(self):
        for q in self.wl["queries"]:
            for m in S.MODEL_ORDER:
                self.assertEqual(q["max_new"][m], min(S.MAX_NEW_TOKENS, S.CONTEXT_TOKENS - q["n_prompt"][m]))

    @unittest.skipUnless(PRIVATE.exists(), "private workload not on this machine")
    def test_private_file_is_the_pinned_one(self):
        wl, priv = WL.load(PRIVATE)
        self.assertEqual(len(priv), S.N_QUERIES + S.N_PROFILE)

    @unittest.skipUnless((STUDY_REPO / "results" / "bench").exists(), "router study repository not on this machine")
    def test_the_draw_is_reproducible(self):
        rows, excluded, _ = WL.frame(STUDY_REPO)
        chosen, profile, _ = WL.draw(rows)
        self.assertEqual(sorted(r["key"] for r in chosen), sorted(q["key"] for q in self.ev))
        self.assertEqual([r["key"] for r in profile], self.wl["profile_keys"])
        self.assertEqual(len(rows), self.wl["n_frame"])


# --------------------------------------------------------------------------- #
class FakeHF:
    """A scripted stand-in for the model: the logits of step i are script[i]."""

    def __init__(self, torch, script, delay=0.0):
        self.torch, self.script, self.i, self.delay = torch, script, 0, delay

    def model(self, input_ids=None, past_key_values=None, use_cache=True):
        if self.delay:
            time.sleep(self.delay)
        class O:  # noqa: E701
            pass
        o = O()
        o.last_hidden_state = self.torch.zeros(1, input_ids.shape[1], 4)
        o.past_key_values = (past_key_values or 0) + 1
        return o

    def lm_head(self, h):
        row = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return self.torch.tensor(row, dtype=self.torch.float32).view(1, 1, -1)


class TestGenerationLoop(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.sync = lambda d: None

    def gen(self, script, seed=1, sampling=None, max_new=6, eos=(9,), cutoff=1e9, delay=0.0, **kw):
        hf = FakeHF(self.torch, script, delay)
        return B.generate(self.torch, hf, [1, 2, 3], max_new, eos, "cpu", 48, cutoff, self.sync,
                          sampling or {"temperature": 0.2, "top_p": 1.0, "top_k": None}, seed, **kw)

    def test_temperature_zero_is_argmax(self):
        res = self.gen([[0, 5, 1, 0, 0, 0, 0, 0, 0, 0]] * 3 + [[0] * 9 + [9]], sampling={"temperature": 0.0, "top_p": 1.0, "top_k": None})
        self.assertEqual(res["tokens"], [1, 1, 1, 9])
        self.assertEqual(res["stop"], "eos")

    def test_seed_determinism_and_variation(self):
        flat = [[0.0] * 10] * 40
        a = self.gen(flat, seed=7, max_new=40, eos=())
        b = self.gen(flat, seed=7, max_new=40, eos=())
        c = self.gen(flat, seed=8, max_new=40, eos=())
        self.assertEqual(a["tokens"], b["tokens"])
        self.assertNotEqual(a["tokens"], c["tokens"])
        self.assertGreater(len(set(a["tokens"])), 3)                           # it samples: a flat distribution spreads

    def test_top_k_restricts_the_support(self):
        row = [3.0, 2.9, 2.8, 2.7, 2.6, 2.5, 2.4, 2.3, 2.2, 2.1]
        res = self.gen([row] * 200, seed=3, max_new=200, eos=(), sampling={"temperature": 5.0, "top_p": 1.0, "top_k": 3})
        self.assertEqual(set(res["tokens"]), {0, 1, 2})
        free = self.gen([row] * 200, seed=3, max_new=200, eos=(), sampling={"temperature": 5.0, "top_p": 1.0, "top_k": None})
        self.assertGreater(len(set(free["tokens"])), 3)

    def test_low_temperature_concentrates(self):
        row = [0.0] * 9 + [1.0]
        res = self.gen([row] * 100, seed=5, max_new=100, eos=())               # T = 0.2: p(top) = e^5 / (e^5 + 9) = 0.94
        self.assertGreater(res["tokens"].count(9), 80)

    def test_top_p_other_than_one_is_refused(self):
        with self.assertRaises(ValueError):
            self.gen([[0.0] * 10], sampling={"temperature": 0.2, "top_p": 0.9, "top_k": None})

    def test_stop_reasons(self):
        peak = lambda t: [50.0 if i == t else 0.0 for i in range(10)]  # noqa: E731
        self.assertEqual(self.gen([peak(1), peak(9)])["stop"], "eos")
        cap = self.gen([peak(1)] * 10, max_new=4)
        self.assertEqual((cap["stop"], len(cap["tokens"]), cap["cutoff"]), ("cap", 4, False))
        cut = self.gen([peak(1)] * 50, max_new=50, cutoff=0.05, delay=0.02)
        self.assertEqual((cut["stop"], cut["cutoff"]), ("cutoff", True))
        self.assertLess(len(cut["tokens"]), 50)
        n = {"k": 0}

        def stop():
            n["k"] += 1
            return n["k"] >= 2
        self.assertEqual(self.gen([peak(1)] * 50, max_new=50, should_stop=stop)["stop"], "cancel")

    def test_stamps_increase_and_times(self):
        res = self.gen([[0.0] * 10] * 8, max_new=8, eos=(), delay=0.002)
        self.assertEqual(res["stamps_ns"], sorted(res["stamps_ns"]))
        t = B.times(res)
        self.assertGreater(t["total_s"], t["ttft_s"])
        self.assertGreater(t["per_token_s"], 0.0)


# --------------------------------------------------------------------------- #
class TestTinyModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch, cls.R = C.runtime()
        md = T.tiny_models()
        cls.hf = {m: cls.R.hf_model(md[m], "cpu") for m in S.MODEL_ORDER}       # the reused loader, unchanged
        cls.pos = {m: B.align_position_scaling(cls.hf[m]) for m in S.MODEL_ORDER}
        cls.toks = {m: WL.load_tokenizer(md[m]) for m in S.MODEL_ORDER}
        cls.ids = {m: WL.tokenize(cls.toks[m], "w1 w2 w3 w4 w5 w6") for m in S.MODEL_ORDER}
        for m, e in T.EOS.items():
            S.MODELS[m]["eos_ids"] = tuple(e)

    def test_the_reused_loader_takes_the_deepseek_position_scaling(self):
        self.assertEqual(self.hf[S.DS].config.rope_scaling["rope_type"], "yarn")
        self.assertIsNone(self.hf[S.QW].config.rope_scaling)
        self.assertEqual(self.hf[S.DS].config.num_hidden_layers, 36)

    def test_position_scale_is_aligned_to_the_original_engine(self):
        md = T.tiny_models()
        fresh = {m: self.R.hf_model(md[m], "cpu") for m in S.MODEL_ORDER}
        self.assertAlmostEqual(fresh[S.DS].model.rotary_emb.attention_scaling, 1.138629436111989, places=9)   # what the library would use
        rec = {m: B.align_position_scaling(fresh[m]) for m in S.MODEL_ORDER}
        self.assertTrue(rec[S.DS]["aligned_to_original_engine"])
        self.assertAlmostEqual(rec[S.DS]["applied_scale"], S.POSITION_SCALE_EXPECTED, places=9)                # (0.1 ln 4 + 1) x attn_factor
        self.assertAlmostEqual(fresh[S.DS].model.rotary_emb.attention_scaling, 1.0, places=9)
        self.assertFalse(rec[S.QW]["aligned_to_original_engine"])
        self.assertEqual(rec[S.QW]["applied_scale"], 1.0)
        cos, _ = fresh[S.DS].model.rotary_emb(self.torch.zeros(1, 1, 4), self.torch.tensor([[0, 7]]))
        self.assertAlmostEqual(float(cos[0, 0, 0]), 1.0, places=6)                                             # cos(0) is 1, not 1.1386
        # the same formula as vLLM 0.8.4's YaRNScalingRotaryEmbedding for the inverse frequencies
        import math
        dim, base, factor, omax = 16, 1000000.0, 4.0, 32768
        pos = base ** (self.torch.arange(0, dim, 2, dtype=self.torch.float) / dim)
        fd = lambda n: (dim * math.log(omax / (n * 2 * math.pi))) / (2 * math.log(base))  # noqa: E731
        low, high = max(math.floor(fd(32)), 0), min(math.ceil(fd(1)), dim - 1)
        ramp = self.torch.clamp((self.torch.arange(dim // 2, dtype=self.torch.float) - low) / (high - low), 0, 1)
        inv = (1.0 / (factor * pos)) * ramp + (1.0 / pos) * (1 - ramp)
        self.assertLess(float((inv - fresh[S.DS].model.rotary_emb.inv_freq).abs().max()), 1e-9)

    def test_templates_differ_and_ids_are_per_model(self):
        self.assertNotEqual(self.ids[S.DS], self.ids[S.QW])
        self.assertEqual(self.ids[S.DS][0], 1)                                  # the DeepSeek template starts with its BOS token

    def test_with_temperature_zero_the_loop_is_the_frozen_loop(self):
        qrt = C.reused()["qrt"]
        for m in S.MODEL_ORDER:
            mine = B.generate(self.torch, self.hf[m], self.ids[m], 10, (), "cpu", 48, 1e9, self.R.E0RT._sync,
                              {"temperature": 0.0, "top_p": 1.0, "top_k": None}, 1)
            frozen = qrt.whole_generate(self.torch, self.hf[m], self.ids[m], 10, (), "cpu", 48, 1e9, self.R.E0RT._sync)
            self.assertEqual(mine["tokens"], frozen["tokens"])
            self.assertEqual(mine["n_prefill_steps"], frozen["n_prefill_steps"])

    def test_edge_backend_runs_the_model_it_is_asked_for(self):
        ids = {m: M.model_id(m, "dev") for m in S.MODEL_ORDER}
        be = B.EdgeBackend(self.torch, self.hf, "cpu", ids, chunk=48)
        out = {m: be.generate(m, self.ids[S.DS], 10, 11, 1e9) for m in S.MODEL_ORDER}   # the same ids and seed through both models
        for m in S.MODEL_ORDER:
            self.assertEqual(out[m]["model"], m)
            self.assertEqual(out[m]["model_identity"]["repo"], S.MODELS[m]["repo"])
            self.assertEqual(out[m]["mode"], S.EDGE)
        self.assertNotEqual(out[S.DS]["tokens"], out[S.QW]["tokens"])           # different weights answered
        again = be.generate(S.DS, self.ids[S.DS], 10, 11, 1e9)
        self.assertEqual(again["tokens"], out[S.DS]["tokens"])                  # same seed, same model: reproducible
        with self.assertRaises(B.BackendError):
            be.generate("14B", self.ids[S.DS], 10, 11, 1e9)

    def _serve(self, models, token="tok", spec="h"):
        qrt = C.reused()["qrt"]
        ident = {"protocol": S.PROTOCOL, "spec_hash": spec, "models": {m: M.model_id(m, "dev") for m in models}, "context_tokens": 256, "dev": True}
        srv = B.RemoteServer(self.torch, {m: self.hf[m] for m in models}, "cpu", ident, token, chunk=48)
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        stop = threading.Event()
        th = threading.Thread(target=qrt.serve_remote, args=(srv, sock, 60, 60, stop), daemon=True)
        th.start()
        return f"127.0.0.1:{sock.getsockname()[1]}", stop, srv

    def test_remote_wire_passes_the_model_and_the_seed(self):
        addr, stop, srv = self._serve(S.MODEL_ORDER)
        try:
            rb = B.RemoteBackend(addr, "tok", "h")
            self.assertTrue(rb.available())
            ids = {m: M.model_id(m, "dev") for m in S.MODEL_ORDER}
            edge = B.EdgeBackend(self.torch, self.hf, "cpu", ids, chunk=48)
            for m in S.MODEL_ORDER:
                r = rb.generate(m, self.ids[m], 10, 21, 1e9)
                e = edge.generate(m, self.ids[m], 10, 21, 1e9)
                self.assertEqual(r["model"], m)
                self.assertEqual(r["server"]["model"], m)
                self.assertEqual(r["tokens"], e["tokens"])                      # same weights, same seed, same machine
                self.assertEqual(r["mode"], S.REMOTE)
                self.assertGreater(r["bytes_out"], 0)
                self.assertEqual(r["server"]["seed"], 21)
            self.assertEqual(srv.n_requests, 2)
            self.assertNotEqual(rb.generate(S.DS, self.ids[S.DS], 10, 21, 1e9)["tokens"], rb.generate(S.DS, self.ids[S.DS], 10, 22, 1e9)["tokens"])
            rb.close()
        finally:
            stop.set()

    def test_remote_refuses_wrong_token_revision_and_unserved_model(self):
        addr, stop, _ = self._serve((S.DS,))
        try:
            self.assertFalse(B.RemoteBackend(addr, "wrong", "h").available())
            self.assertFalse(B.RemoteBackend(addr, "tok", "other-revision").available())
            rb = B.RemoteBackend(addr, "tok", "h")
            self.assertTrue(rb.available())
            with self.assertRaises(B.BackendError):
                rb.generate(S.QW, self.ids[S.QW], 5, 1, 1e9)                    # the server holds DeepSeek only
            self.assertEqual(rb.generate(S.DS, self.ids[S.DS], 5, 1, 1e9)["model"], S.DS)
            rb.close()
        finally:
            stop.set()

    def test_remote_cutoff_cancels(self):
        addr, stop, _ = self._serve((S.DS,))
        try:
            rb = B.RemoteBackend(addr, "tok", "h")
            r = rb.generate(S.DS, self.ids[S.DS], 200, 1, 0.0)                  # cutoff of zero seconds: the first token is already late
            self.assertEqual((r["stop"], r["cutoff"]), ("cutoff", True))
            self.assertLess(len(r["tokens"]), 200)
            ok = rb.generate(S.DS, self.ids[S.DS], 5, 1, 1e9)                   # the connection is still usable
            self.assertEqual(len(ok["tokens"]), 5)
            rb.close()
        finally:
            stop.set()

    def test_unreachable_server_is_unavailable(self):
        self.assertFalse(B.RemoteBackend(f"127.0.0.1:{T.free_port()}", "tok", "h", timeout_s=2).available())


# --------------------------------------------------------------------------- #
class TestRecorder(unittest.TestCase):
    def setUp(self):
        self.q = {"key": "gpqa:1", "dataset": "gpqa", "fold": 2, "n_prompt": {S.DS: 5, S.QW: 7}, "ids_sha256": {S.DS: "a", S.QW: "b"}}
        self.hdr = {"freeze": {"spec_hash": "h"}, "run_id": "r1", "dev": True, "role": "edge", "host": {"gpu_names": []}, "stage": "shared:edge"}

    def test_ok_row_has_every_required_field(self):
        be = B.MockBackend(S.EDGE, token_table={S.QW: [5, 6, S.MODELS[S.QW]["eos_ids"][-1]]})
        row = REC.base_row(self.hdr, "shared", "edge", self.q, S.QW, "QW_EDGE", 1, 42, 100)
        REC.ok_row(row, be.generate(S.QW, [1] * 7, 100, 42, 600.0), "text")
        for k in ("id", "key", "router_fold", "model_name", "model_repo", "model_revision", "config", "mode", "hardware", "quantisation",
                  "settings", "seed", "text", "ttft_s", "total_s", "n_out", "ended_within_s", "stop", "error", "data_policy_feasible",
                  "run_id", "phase", "attempt", "backend_model"):
            self.assertIn(k, row)
        self.assertEqual(row["stop"], "eos")
        self.assertEqual(row["data_policy_feasible"], {S.POLICY_A: True, S.POLICY_B: True})
        self.assertEqual(row["settings"]["top_k"], 20)
        self.assertTrue(row["mock"])
        self.assertEqual(be.calls[0]["model"], S.QW)                            # the selected identifier reached the backend

    def test_remote_rows_are_infeasible_under_policy_a(self):
        row = REC.base_row(self.hdr, "shared", "remote", self.q, S.DS, "DS_REMOTE", 1, 1, 10)
        self.assertEqual(row["data_policy_feasible"], {S.POLICY_A: False, S.POLICY_B: True})
        self.assertIsNone(row["settings"]["top_k"])

    def test_a_backend_that_ran_another_model_is_rejected(self):
        be = B.MockBackend(S.EDGE)
        row = REC.base_row(self.hdr, "shared", "edge", self.q, S.QW, "QW_EDGE", 1, 1, 10)
        with self.assertRaises(RuntimeError):
            REC.ok_row(row, be.generate(S.DS, [1], 10, 1, 600.0), "x")

    def test_a_failed_request_is_kept_as_a_row_and_an_event(self):
        import rint_run as RUNM
        rd = T.TMP / "rec_fail"
        rec = REC.Recorder(rd, "shared_edge", self.hdr)
        be = B.MockBackend(S.EDGE, fail_on={S.DS})

        class Tok:
            def decode(self, toks, skip_special_tokens=True):
                return " ".join(map(str, toks))
        priv = {"ids": {S.DS: [1] * 5, S.QW: [1] * 7}}
        r1 = RUNM.run_request(rec, self.hdr, be, "shared", "edge", self.q, priv, S.DS, 1, {S.DS: Tok(), S.QW: Tok()})
        r2 = RUNM.run_request(rec, self.hdr, be, "shared", "edge", self.q, priv, S.QW, 1, {S.DS: Tok(), S.QW: Tok()})
        self.assertEqual((r1["status"], r1["stop"], r1["text"], r1["total_s"]), ("error", "error", None, None))
        self.assertEqual(r1["error"]["type"], "BackendError")
        self.assertFalse(r1["transport_error"])                                 # a model failure settles the request: it is not repeated
        self.assertTrue(RUNM.counts(r1))
        self.assertEqual(r2["status"], "ok")
        rows, _ = rec.existing()
        self.assertEqual([r["status"] for r in rows], ["error", "ok"])           # nothing dropped
        events, _ = C.read_jsonl(rd / "events.jsonl")
        self.assertEqual([e["event"] for e in events], ["request_error"])


# --------------------------------------------------------------------------- #
def synth(tmp, spec, live=None, router_s=0.05, drop=None):
    """A synthetic run directory. spec: {key: {config: (correct, total_s or None)}}. Returns (run dir, scores, workload)."""
    rd = Path(tmp)
    rd.mkdir(parents=True, exist_ok=True)
    qs, scores = [], []
    for i, (key, cells) in enumerate(spec.items()):
        ds = key.split(":")[0]
        choice = cells.pop("router", S.QW)
        origin = cells.pop("orig", (1, 1))
        qs.append({"key": key, "role": "eval", "dataset": ds, "fold": 0, "labels_A_original": {"class4": WL.class4(*origin), "score": {S.DS: origin[0], S.QW: origin[1]}},
                   "router_pinned": {"fold": 0, "score": 0.3 if choice == S.QW else -0.3, "choice": choice},
                   "task_lookup_pinned": {"predicted_benchmark": ds, "D1_true_benchmark_choice": S.DS, "D2_predicted_benchmark_choice": S.DS}})
        for cfg, (corr, tot) in cells.items():
            if drop and (key, cfg) in drop:
                continue
            lane = "shared_edge" if cfg.endswith("EDGE") else "shared_remote"
            rid = REC.row_id("shared", lane.split("_")[1], key, cfg, 1)
            row = {"id": rid, "phase": "shared", "key": key, "dataset": ds, "config": cfg, "mode": S.CONFIGS[cfg][1], "model": S.CONFIGS[cfg][0],
                   "status": "ok" if tot is not None else "error", "stop": "eos" if tot is not None else "error", "total_s": tot, "n_out": 10 if tot else 0,
                   "dev": True}
            C.append_jsonl(rd / f"rows_{lane}.jsonl", row)
            scores.append({"id": rid, "score": float(corr) if tot is not None else 0.0})
        C.append_jsonl(rd / "router_timing.jsonl", {"key": key, "router_s": router_s, "choice": choice})
    for ph, rows in (live or {}).items():
        for key, (cfg, corr, tot, e2e, fb) in rows.items():
            rid = REC.row_id(ph, ph, key, cfg, 1)
            m = S.CONFIGS[cfg][0]
            C.append_jsonl(rd / f"rows_{ph}.jsonl", {"id": rid, "phase": ph, "key": key, "dataset": key.split(":")[0], "config": cfg, "mode": S.CONFIGS[cfg][1],
                                                     "model": m, "backend_model": m, "status": "ok", "stop": "eos", "total_s": tot, "n_out": 10, "dev": True,
                                                     "router": {"choice": m}, "router_s": router_s, "dispatch_overhead_s": 0.001, "end_to_end_s": e2e,
                                                     "fallback": fb, "ids_match_pinned": True})
            scores.append({"id": rid, "score": float(corr)})
    sp = rd / "scores.jsonl"
    for s in scores:
        C.append_jsonl(sp, s)
    wp = rd / "workload.json"
    wp.write_text(json.dumps({"queries": qs}))
    return rd, sp, wp


class TestEvaluator(unittest.TestCase):
    def spec(self):
        return {
            # key: DS_EDGE, QW_EDGE, DS_REMOTE, QW_REMOTE as (correct, seconds); router's choice; original scores
            "gpqa:1": {"DS_EDGE": (1, 100.0), "QW_EDGE": (0, 50.0), "DS_REMOTE": (1, 40.0), "QW_REMOTE": (0, 20.0), "router": S.DS, "orig": (1, 0)},
            "gpqa:2": {"DS_EDGE": (0, 100.0), "QW_EDGE": (1, 59.97), "DS_REMOTE": (0, 40.0), "QW_REMOTE": (1, 29.0), "router": S.QW, "orig": (0, 1)},
            "medqa:1": {"DS_EDGE": (1, 500.0), "QW_EDGE": (1, 200.0), "DS_REMOTE": (1, 250.0), "QW_REMOTE": (1, 90.0), "router": S.QW, "orig": (1, 1)},
            "medqa:2": {"DS_EDGE": (0, 600.5), "QW_EDGE": (0, None), "DS_REMOTE": (1, 300.0), "QW_REMOTE": (0, 100.0), "router": S.QW, "orig": (0, 0)},
        }

    def run_eval(self, name, **kw):
        rd, sp, wp = synth(T.TMP / name, self.spec(), **kw)
        return EV.evaluate(EV.load(rd, sp, wp, dev=True), n_boot=200)

    def test_refuses_dev_rows_as_results(self):
        rd, sp, wp = synth(T.TMP / "ev_refuse", self.spec())
        with self.assertRaises(SystemExit):
            EV.load(rd, sp, wp, dev=False)

    def test_policy_tables_by_hand(self):
        o = self.run_eval("ev_hand")
        self.assertEqual((o["status"], o["n_evaluated"]), ("COMPLETE", 4))
        a = o["data_policies"][S.POLICY_A]["policies"]                          # placement EDGE
        self.assertEqual(a["always_DS"]["correct_within_cutoff_pct"], 50.0)     # gpqa:1, medqa:1
        self.assertEqual(a["always_QW"]["correct_within_cutoff_pct"], 50.0)     # gpqa:2, medqa:1
        self.assertEqual(a["router"]["correct_within_cutoff_pct"], 75.0)        # DS, QW, QW, QW(error)
        self.assertEqual(a["hindsight_model"]["correct_within_cutoff_pct"], 75.0)
        self.assertEqual(a["always_QW"]["failures"], 1)                         # the failed request is counted, and counted as wrong
        self.assertEqual(a["router"]["failures"], 1)
        self.assertEqual(a["always_DS"]["correct_by_deadline_pct"]["120"], 25.0)
        self.assertEqual(a["always_DS"]["correct_by_deadline_pct"]["600"], 50.0)
        # the router's latency is added to the router policy only: gpqa:2 takes 59.97 s + 0.05 s and misses 60 s
        self.assertEqual(a["always_QW"]["correct_by_deadline_pct"]["60"], 25.0)
        self.assertEqual(a["router"]["correct_by_deadline_pct"]["60"], 0.0)
        self.assertEqual(a["router"]["correct_by_deadline_pct"]["120"], 50.0)
        self.assertTrue(a["router"]["end_to_end_latency_s"]["includes_router_latency"])
        self.assertFalse(a["always_QW"]["end_to_end_latency_s"]["includes_router_latency"])
        b = o["data_policies"][S.POLICY_B]["policies"]                          # placement REMOTE
        self.assertEqual(b["always_DS"]["correct_within_cutoff_pct"], 75.0)
        self.assertEqual(b["always_QW"]["correct_within_cutoff_pct"], 50.0)
        self.assertEqual(b["router"]["correct_within_cutoff_pct"], 75.0)
        self.assertEqual(b["hindsight_model"]["correct_within_cutoff_pct"], 100.0)
        self.assertEqual(b["router"]["configuration_counts"], {"DS_REMOTE": 1, "QW_REMOTE": 3})
        for pol in (S.POLICY_A, S.POLICY_B):
            for p in EV.POLICIES:
                self.assertEqual(o["data_policies"][pol]["policies"][p]["data_policy_violations"], 0)
        self.assertNotIn("REMOTE", " ".join(a["hindsight_config"]["configuration_counts"]))   # policy A never uses the server
        d = o["data_policies"][S.POLICY_A]["paired_differences"]["router minus always_QW"]["acc"]
        self.assertEqual((d["delta_pt"], d["discordant"]), (25.0, [1, 0]))
        self.assertLessEqual(d["ci95"][0], d["delta_pt"])

    def test_labels_and_semantic_metrics(self):
        o = self.run_eval("ev_labels")
        lab = o["labels"]
        self.assertEqual(lab["A_original_counts"], {S.DS: 1, S.QW: 1, "tie": 2})
        self.assertEqual(lab["B_edge_counts"], {S.DS: 1, S.QW: 1, "tie": 2})
        self.assertEqual(lab["B_remote_counts"], {S.DS: 2, S.QW: 1, "tie": 1})  # medqa:2 has a winner on the server only
        self.assertEqual(lab["agreement_B_edge_B_remote"], 0.75)
        sem = o["semantic_router"]
        self.assertEqual((sem["A_original"]["n_with_winner"], sem["A_original"]["n_ties"]), (2, 2))
        self.assertEqual(sem["A_original"]["macro_f1"], 1.0)
        self.assertEqual(sem["A_original"]["auc"], 1.0)
        # on the server: winners DS, QW, DS; the router chose DS, QW, QW -> F1(DS) = 2/3, F1(QW) = 2/3
        self.assertAlmostEqual(sem["B_remote"]["macro_f1"], 2 / 3, places=9)
        self.assertEqual(sem["B_remote"]["n_with_winner"], 3)

    def test_macro_f1_and_auc(self):
        self.assertEqual(EV.macro_f1([S.DS, S.QW], [S.DS, S.QW]), 1.0)
        self.assertAlmostEqual(EV.macro_f1([S.DS, S.DS, S.QW, S.QW], [S.DS, S.QW, S.QW, S.QW]), (2 / 3 + 0.8) / 2)
        self.assertEqual(EV.macro_f1([S.DS, S.QW], [S.QW, S.QW]), (0.0 + 2 / 3) / 2)
        self.assertEqual(EV.auc([S.DS, S.QW], [0.1, 0.9]), 1.0)
        self.assertEqual(EV.auc([S.DS, S.QW], [0.9, 0.1]), 0.0)
        self.assertEqual(EV.auc([S.DS, S.QW], [0.5, 0.5]), 0.5)
        self.assertIsNone(EV.auc([S.QW, S.QW], [0.5, 0.6]))

    def test_a_query_without_every_configuration_is_reported_not_hidden(self):
        rd, sp, wp = synth(T.TMP / "ev_missing", self.spec(), drop={("gpqa:2", "QW_REMOTE")})
        o = EV.evaluate(EV.load(rd, sp, wp, dev=True), n_boot=100)
        self.assertEqual((o["status"], o["n_evaluated"], o["queries_without_all_configurations"]), ("PARTIAL_PREFIX", 3, ["gpqa:2"]))

    def test_live_comparison(self):
        live = {"live_A": {"gpqa:1": ("DS_EDGE", 1, 110.0, 110.1, False), "gpqa:2": ("QW_EDGE", 0, 70.0, 70.1, False)},
                "live_B": {"gpqa:1": ("DS_REMOTE", 1, 44.0, 44.1, False), "gpqa:2": ("QW_EDGE", 1, 61.0, 75.0, True)}}
        o = self.run_eval("ev_live", live=live)
        la, lb = o["live"]["live_A"], o["live"]["live_B"]
        self.assertEqual((la["n"], la["selection_agreement_with_pinned_router"], la["executed_the_planned_configuration"]), (2, 2, 2))
        self.assertEqual(la["correctness_record_vs_live"]["record_only"], 1)    # gpqa:2 was correct in the record, wrong live: reported
        self.assertEqual(la["generation_time_live_minus_record_s"]["max"], 10.030000000000001)
        self.assertEqual(la["data_policy_violations"], 0)
        self.assertEqual((lb["fallbacks"], lb["executed_the_planned_configuration"]), (1, 1))
        # both sides of the deadline comparison use the same two queries; the fallback query is compared with the record of
        # the configuration policy C planned for it (QW_REMOTE: correct, 29 s), not with the configuration it fell back to
        self.assertEqual(lb["n_compared_with_record"], 2)
        self.assertEqual(lb["correct_by_deadline_pct_record_same_queries"]["60"], 100.0)
        self.assertEqual(lb["correct_by_deadline_pct_live"]["60"], 50.0)        # the fallback answer arrived after 75 s
        self.assertEqual(lb["generation_time_live_minus_record_s"]["n"], 1)     # times are compared only for the same configuration
        self.assertEqual(o["live"]["live_A"]["status"], "PARTIAL")
        md = EV.markdown(o)
        self.assertIn("retrospective reference, not deployable", md)
        self.assertIn("non-deployable diagnostic", md)

    def test_a_transport_failure_is_not_an_outcome(self):
        rd, sp, wp = synth(T.TMP / "ev_transport", self.spec())
        # the wire was lost during gpqa:2 on the server; the request was repeated later and succeeded
        C.append_jsonl(rd / "rows_shared_remote.jsonl", {"id": "x-lost", "phase": "shared", "key": "gpqa:2", "dataset": "gpqa", "config": "QW_REMOTE",
                                                         "mode": S.REMOTE, "model": S.QW, "status": "error", "stop": "error", "total_s": None,
                                                         "n_out": 0, "dev": True, "transport_error": True})
        o = EV.evaluate(EV.load(rd, sp, wp, dev=True), n_boot=50)
        self.assertEqual(o["transport_failures_recorded_and_repeated"], 1)
        self.assertEqual(o["configurations"]["QW_REMOTE"]["failures"], 0)       # the lost request does not count against the model
        self.assertEqual(o["n_evaluated"], 4)
        # ... and if it was never repeated, the query is not evaluated at all, and that is reported
        rd2, sp2, wp2 = synth(T.TMP / "ev_transport2", self.spec(), drop={("gpqa:2", "QW_REMOTE")})
        C.append_jsonl(rd2 / "rows_shared_remote.jsonl", {"id": "x-lost", "phase": "shared", "key": "gpqa:2", "dataset": "gpqa", "config": "QW_REMOTE",
                                                          "mode": S.REMOTE, "model": S.QW, "status": "error", "stop": "error", "total_s": None,
                                                          "n_out": 0, "dev": True, "transport_error": True})
        o2 = EV.evaluate(EV.load(rd2, sp2, wp2, dev=True), n_boot=50)
        self.assertEqual((o2["n_evaluated"], o2["queries_without_all_configurations"]), (3, ["gpqa:2"]))

    def test_one_winner_class_has_no_macro_f1(self):
        spec = self.spec()
        spec["gpqa:2"]["QW_EDGE"] = (0, 59.97)                                  # on the edge only DeepSeek ever wins
        rd, sp, wp = synth(T.TMP / "ev_oneclass", spec)
        o = EV.evaluate(EV.load(rd, sp, wp, dev=True), n_boot=50)
        e = o["semantic_router"]["B_edge"]
        self.assertEqual((e["both_winner_classes_present"], e["macro_f1"], e["macro_f1_ci95"], e["auc"]), (False, None, None, None))
        self.assertEqual(e["n_with_winner"], 1)

    def test_a_torn_line_does_not_make_a_file_unreadable(self):
        p = T.TMP / "torn.jsonl"
        C.append_jsonl(p, {"a": 1})
        with open(p, "a") as f:
            f.write('{"a": 2, "text": "cut sho')                                # a process died in the middle of a row
        C.append_jsonl(p, {"a": 3})                                             # the resumed process writes the next row
        rows, torn = C.read_jsonl(p)
        self.assertEqual(([r["a"] for r in rows], torn), ([1, 3], 1))

    def test_bootstrap_is_seeded_and_stratified(self):
        w1 = EV.boot_weights(["a", "a", "b", "b", "b"], 50, 1)
        self.assertEqual(w1, EV.boot_weights(["a", "a", "b", "b", "b"], 50, 1))
        self.assertTrue(all(sum(w[:2]) == 2 and sum(w[2:]) == 3 for w in w1))


# --------------------------------------------------------------------------- #
class TestFreeze(unittest.TestCase):
    def test_target_commands_need_the_freeze_record_and_unchanged_files(self):
        import importlib
        d = T.TMP / "copy"
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(C.RI_DIR, d, ignore=shutil.ignore_patterns("__pycache__", "tests"))
        code = ("import sys, json; sys.path.insert(0, %r); import rint_common as C\n"
                "try:\n    print(json.dumps(C.freeze_check({'qsel_repo': %r})))\nexcept SystemExit as e:\n    print('STOP', e.code)\n") % (str(d), os.environ["RINT_QSEL_REPO"])
        import subprocess
        run = lambda: subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True).stdout.strip()  # noqa: E731
        (d / C.FREEZE_FILE).unlink(missing_ok=True)
        self.assertEqual(run(), "STOP 3")                                       # no freeze record: nothing runs
        sys.path.insert(0, str(C.RI_DIR / "tools"))
        fz = importlib.import_module("rint_freeze")
        fz.write(d, "test")
        out = json.loads(run())
        self.assertEqual(out["reused"]["commit"], S.REUSED_COMMIT)
        with open(d / "rint_spec.py", "a") as f:
            f.write("\n# edited after the freeze\n")
        self.assertEqual(run(), "STOP 3")                                       # a frozen file changed: nothing runs

    def test_the_reused_runtime_is_at_its_tag_and_unmodified(self):
        self.assertEqual(C.reused_check(C.qsel_repo())["commit"], S.REUSED_COMMIT)

    def test_launcher_refuses_without_dev_when_not_frozen_or_limited(self):
        rd = T.TMP / "run_refuse"
        s = T.site("edge", rd, T.free_port())
        r = T.run("shared", "--site", s, "--lane", "edge", "--limit", "1")
        self.assertEqual(r.returncode, 3)
        self.assertIn("--limit is for --dev only", r.stderr)


# --------------------------------------------------------------------------- #
@unittest.skipUnless(ENC, "router encoder not on this machine")
class TestPipelineDev(unittest.TestCase):
    """The complete command sequence on a CPU with tiny random models: server, both lanes, router timing, both live
    passes, a server that is down, resumption. Every row is a dev row."""

    @classmethod
    def setUpClass(cls):
        cls.port = T.free_port()
        cls.rd_edge, cls.rd_srv = T.TMP / "run_edge", T.TMP / "run_server"
        cls.site_edge = T.site("edge", cls.rd_edge, cls.port)
        cls.site_srv = T.site("server", cls.rd_srv, cls.port)
        cls.wl = json.loads(Path(T.dev_workload()[0]).read_text())
        cls.n = cls.wl["n_eval"]
        cls.srv = T.start_server(cls.site_srv)
        cls.ready = T.wait_ready(cls.srv)

    @classmethod
    def tearDownClass(cls):
        if cls.srv.poll() is None:
            T.stop_server(cls.srv)

    def ok(self, r, code=0):
        self.assertEqual(r.returncode, code, r.stdout[-1500:] + r.stderr[-3000:])
        return r

    def test_1_preflight(self):
        r = self.ok(T.run("preflight", "--dev", "--site", self.site_edge, "--role", "edge"))
        out = json.loads(r.stdout)
        self.assertEqual(out["tokenisation"]["n_mismatches"], 0)
        self.assertEqual(out["router"]["choices_differing"], 0)

    def test_2_profile_and_shared_lanes(self):
        self.ok(T.run("shared", "--dev", "--site", self.site_edge, "--lane", "edge", "--phase", "profile"))
        self.ok(T.run("shared", "--dev", "--site", self.site_edge, "--lane", "remote", "--phase", "profile"))
        # an interrupted lane: two queries, then a resumption that must not repeat them
        self.ok(T.run("shared", "--dev", "--site", self.site_edge, "--lane", "edge", "--limit", "2"))
        (self.rd_edge / "lane_shared_edge.done.json").unlink()                  # the limited run "completed" its two queries: undo for the resumption
        self.ok(T.run("shared", "--dev", "--site", self.site_edge, "--lane", "edge"))
        self.ok(T.run("shared", "--dev", "--site", self.site_edge, "--lane", "remote"))
        for lane, mode in (("edge", S.EDGE), ("remote", S.REMOTE)):
            rows = [r for r in T.rows_of(self.rd_edge, f"shared_{lane}") if r["phase"] == "shared"]
            self.assertEqual(len(rows), 2 * self.n)                             # every (query, model) once: nothing repeated, nothing dropped
            self.assertEqual(len({(r["key"], r["config"]) for r in rows}), 2 * self.n)
            self.assertTrue(all(r["mode"] == mode and r["dev"] and r["status"] == "ok" for r in rows))
            self.assertTrue(all(r["backend_model"] == r["model"] == S.CONFIGS[r["config"]][0] for r in rows))
            self.assertTrue(all(r["data_policy_feasible"][S.POLICY_A] == (mode == S.EDGE) for r in rows))
            self.assertTrue(all(r["seed"] == C.request_seed(r["key"], r["model"], mode, "shared") for r in rows))
            first = {k: f for k, f in self.wl["orders"]["shared"]}
            self.assertTrue(all(r["order_first_model"] == first[r["key"]] for r in rows))
            warm = [r for r in T.rows_of(self.rd_edge, f"shared_{lane}") if r["phase"] == "warmup"]
            self.assertGreaterEqual(len(warm), 2)
            self.assertTrue(all(r["n_out"] <= 3 for r in warm))
        self.assertEqual({r["attempt"] for r in T.rows_of(self.rd_edge, "shared_edge") if r["phase"] == "shared"}, {1, 2})
        r = T.run("shared", "--dev", "--site", self.site_edge, "--lane", "edge")
        self.assertEqual(r.returncode, 3)                                       # a completed stage is final
        self.assertEqual(len([r for r in T.rows_of(self.rd_edge, "profile_edge") if r["phase"] == "profile"]), 2 * self.wl["n_profile"])

    def test_3_router_timing(self):
        r = self.ok(T.run("router-timing", "--dev", "--site", self.site_edge))
        out = json.loads(r.stdout)
        self.assertEqual((out["n"], out["choices_agreeing"]), (self.n, self.n))
        self.assertLess(out["max_abs_score_diff"], 1e-4)

    def test_4_live_a_never_touches_the_server(self):
        self.ok(T.run("live", "--dev", "--site", self.site_edge, "--policy", "A"))
        rows = [r for r in T.rows_of(self.rd_edge, "live_A") if r["phase"] == "live_A"]
        self.assertEqual(len(rows), self.n)
        by = {q["key"]: q for q in self.wl["queries"]}
        for r in rows:
            self.assertEqual(r["mode"], S.EDGE)
            self.assertEqual(r["model"], r["router"]["choice"])                 # the router's decision is the model that ran
            self.assertEqual(r["backend_model"], r["model"])
            self.assertEqual(r["router"]["choice"], by[r["key"]]["router_pinned"]["choice"])
            self.assertTrue(r["ids_match_pinned"])
            self.assertTrue(r["data_policy_feasible"][S.POLICY_A])
            self.assertGreater(r["end_to_end_s"], r["total_s"])                 # router and dispatch are inside the end-to-end time
            self.assertAlmostEqual(r["end_to_end_s"], r["router_s"] + r["dispatch_overhead_s"] + r["backend_ready_s"] + r["total_s"], places=6)
        self.assertEqual({r["model"] for r in rows}, {by[r["key"]]["router_pinned"]["choice"] for r in rows})

    def test_5_live_b_uses_the_server(self):
        self.ok(T.run("live", "--dev", "--site", self.site_edge, "--policy", "B"))
        rows = [r for r in T.rows_of(self.rd_edge, "live_B") if r["phase"] == "live_B"]
        self.assertEqual(len(rows), self.n)
        self.assertTrue(all(r["mode"] == S.REMOTE and not r["fallback"] and r["backend_model"] == r["router"]["choice"] for r in rows))
        self.assertTrue(all(r["bytes_out"] > 0 and r["server"]["model"] == r["model"] for r in rows))
        self.assertTrue(all(r["config"] == r["planned_config"] and r["backend_ready_s"] < 0.05 for r in rows))

    def test_6_wire_lost_during_live_b_falls_back_to_the_edge(self):
        rd = T.TMP / "run_edge_fallback"
        s = T.site("edge", rd, self.port, extra_dev={"break_wire_after": 2}, name="site_edge_fallback.json")
        r = T.run("live", "--dev", "--site", s, "--policy", "B")
        self.assertEqual(r.returncode, 3, r.stderr[-2000:])                    # three fallbacks in a row: the pass stops, rows are kept
        self.assertIn("needed the fallback", r.stderr)
        rows = [x for x in T.rows_of(rd, "live_B") if x["phase"] == "live_B"]
        self.assertEqual([x["mode"] for x in rows], [S.REMOTE] * 2 + [S.EDGE] * 3)
        self.assertEqual([x["fallback"] for x in rows], [False] * 2 + [True] * 3)
        self.assertTrue(all(x["status"] == "ok" and x["data_policy_feasible"][S.POLICY_B] for x in rows))
        first = rows[2]                                                         # the first fallback loaded the edge models: counted
        self.assertGreater(first["backend_ready_s"], 0.05)
        self.assertAlmostEqual(first["end_to_end_s"], first["router_s"] + first["dispatch_overhead_s"] + first["lost_s"]
                               + first["backend_ready_s"] + first["total_s"], places=6)
        self.assertEqual(first["planned_config"], P.config_of(first["model"], S.REMOTE))
        led = [json.loads(l) for l in (rd / "gpu_ledger.jsonl").read_text().splitlines()]
        self.assertEqual([e["event"] for e in led], ["open", "close"])          # the edge GPU entry was opened once and closed
        ev, _ = C.read_jsonl(rd / "events.jsonl")
        self.assertEqual([e["event"] for e in ev].count("fallback"), 3)

    def test_6a_a_server_that_is_down_is_a_stop_not_an_attempt(self):
        rd = T.TMP / "run_edge_noserver"
        s = T.site("edge", rd, T.free_port(), name="site_edge_noserver.json")   # a port nobody listens on
        for cmd in (("shared", "--lane", "remote"), ("live", "--policy", "B")):
            for _ in range(4):                                                  # more often than a lane has attempts
                r = T.run(cmd[0], "--dev", "--site", s, *cmd[1:])
                self.assertEqual(r.returncode, 3)
                self.assertIn("not an attempt", r.stderr)
        self.assertEqual(list(rd.glob("*.attempt*.json")), [])
        self.assertEqual(list(rd.glob("rows_*.jsonl")), [])

    def test_6c_a_lost_server_stops_the_remote_lane_and_the_request_is_repeated(self):
        rd, port = T.TMP / "run_edge_lost", T.free_port()
        s_edge = T.site("edge", rd, port, name="site_edge_lost.json")
        srv = T.start_server(T.site("server", T.TMP / "run_server_lost1", port, name="site_server_lost1.json"))
        T.wait_ready(srv)
        self.ok(T.run("shared", "--dev", "--site", s_edge, "--lane", "remote", "--limit", "1"))
        (rd / "lane_shared_remote.done.json").unlink()
        # a row as a lost wire leaves it (written by the recorder in the real run; the lane stops at once)
        q = next(x for x in self.wl["queries"] if x["key"] == self.wl["orders"]["shared"][1][0])
        hdr = {"freeze": {"spec_hash": "h"}, "run_id": "lost", "dev": True, "role": "edge", "host": {"gpu_names": []}, "stage": "shared:remote"}
        row = REC.error_row(REC.base_row(hdr, "shared", "remote", q, S.DS, "DS_REMOTE", 1, 1, 12), B.TransportError("connection lost"), transport=True)
        C.append_jsonl(rd / "rows_shared_remote.jsonl", row)
        try:
            self.ok(T.run("shared", "--dev", "--site", s_edge, "--lane", "remote"))
        finally:
            T.stop_server(srv)
        rows = [x for x in T.rows_of(rd, "shared_remote") if x["phase"] == "shared"]
        again = [x for x in rows if x["key"] == q["key"] and x["config"] == "DS_REMOTE"]
        self.assertEqual([x["status"] for x in again], ["error", "ok"])         # the lost request was kept and repeated
        self.assertEqual(C.read_json(rd / "lane_shared_remote.done.json")["n_transport_failures_repeated"], 1)
        self.assertEqual(len({(x["key"], x["config"]) for x in rows if x["status"] == "ok"}), 2 * self.n)

    def test_6d_stage_limit_covers_all_attempts(self):
        rd = T.TMP / "run_edge_limit"
        s = T.site("edge", rd, T.free_port(), extra_dev={"limit_s": 300}, name="site_edge_limit.json")
        r = T.run("shared", "--dev", "--site", s, "--lane", "edge", "--phase", "profile")
        self.assertEqual(r.returncode, 0, r.stderr[-1500:])
        led = (rd / "gpu_ledger.jsonl")
        rows = [json.loads(l) for l in led.read_text().splitlines()]
        opened = next(e for e in rows if e["event"] == "open")
        closed = dict(opened, event="close", t=opened["t"] + 299.5)             # as if the first attempt had used the stage's 300 s
        led.write_text("".join(json.dumps(e) + "\n" for e in [opened, closed]))
        (rd / "lane_profile_edge.done.json").unlink()
        r = T.run("shared", "--dev", "--site", s, "--lane", "edge", "--phase", "profile")
        self.assertEqual(r.returncode, 3)
        self.assertIn("is used up over its attempts", r.stderr)

    def test_6e_reduced_live_pass_is_marked(self):
        r = T.run("live", "--site", self.site_edge, "--policy", "A", "--reduced")
        self.assertEqual(r.returncode, 3)                                       # outside --dev it still needs the freeze record: nothing bypasses it
        self.assertNotIn("--limit is for --dev only", r.stderr)

    def test_6b_one_model_at_a_time_on_the_server(self):
        rd = T.TMP / "run_edge_onemodel"
        port = T.free_port()
        s_edge = T.site("edge", rd, port, name="site_edge_onemodel.json")
        for m in S.MODEL_ORDER:
            srv = T.start_server(T.site("server", T.TMP / f"run_server_one_{m}", port, name=f"site_server_one_{m}.json"), models=m)
            try:
                T.wait_ready(srv)
                self.ok(T.run("shared", "--dev", "--site", s_edge, "--lane", "remote", "--models", m))
            finally:
                T.stop_server(srv)
            self.assertEqual((rd / "lane_shared_remote.done.json").exists(), m == S.QW)      # complete only after the second model
        rows = [r for r in T.rows_of(rd, "shared_remote") if r["phase"] == "shared"]
        self.assertEqual(len({(r["key"], r["config"]) for r in rows}), 2 * self.n)
        self.assertTrue(C.read_json(rd / "lane_shared_remote.done.json")["one_model_at_a_time"])
        self.assertEqual(T.run("shared", "--dev", "--site", s_edge, "--lane", "remote", "--models", S.DS).returncode, 3)

    def test_7_evaluator_runs_on_the_dev_run(self):
        scores = T.TMP / "dev_scores.jsonl"
        if scores.exists():
            scores.unlink()
        for name in ("shared_edge", "shared_remote", "live_A", "live_B"):
            for r in T.rows_of(self.rd_edge, name):
                if r["phase"] != "warmup":
                    C.append_jsonl(scores, {"id": r["id"], "score": float(len(r["tokens"]) % 2)})      # fixture scores: not evaluator output
        o = EV.evaluate(EV.load(self.rd_edge, scores, T.dev_workload()[0], dev=True), n_boot=100)
        self.assertEqual((o["status"], o["n_evaluated"]), ("COMPLETE", self.n))
        self.assertTrue(o["router_latency_available"])
        self.assertEqual(o["live"]["live_A"]["status"], "COMPLETE")
        self.assertEqual(o["live"]["live_A"]["selection_agreement_with_pinned_router"], self.n)
        self.assertEqual(o["live"]["live_B"]["the_backend_ran_the_selected_model"], self.n)
        with self.assertRaises(SystemExit):
            EV.load(self.rd_edge, scores, T.dev_workload()[0], dev=False)       # dev rows are never results


# --------------------------------------------------------------------------- #
class TestNoPrivateContent(unittest.TestCase):
    def test_public_files_name_no_host_and_hold_no_prompt(self):
        import re
        # host names, logins and paths are private: their patterns live in a private file, never in this repository
        pats = [r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"]
        for extra in (os.environ.get("RINT_PRIVATE_PATTERNS"), Path.home() / "research" / "router_integration_private" / "private_patterns.txt"):
            if extra and Path(extra).exists():
                pats += [l.strip() for l in Path(extra).read_text().splitlines() if l.strip() and not l.startswith("#")]
        hits = []
        for p in C.RI_DIR.rglob("*"):
            if not p.is_file() or "__pycache__" in p.parts or p.suffix in (".npz", ".pyc"):
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
            for pat in pats:
                for m in re.finditer(pat, text):
                    if m.group(0) != "127.0.0.1":
                        hits.append((str(p.relative_to(C.RI_DIR)), m.group(0)))
        self.assertEqual(hits, [])
        wl = WL.public()
        self.assertTrue(all("prompt" not in q and "ground_truth" not in q and "ids" not in q for q in wl["queries"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
