"""GEMMA-SPLIT end-to-end CPU pipeline with the Gemma 4 test double (dev mode; never a decision).

Needs torch and a transformers with Gemma 4 modules (tested: torch 2.14.1 CPU, transformers 5.19.0); skipped with a
message otherwise. Runs every stage through gs_bench.py (the server half as a real subprocess, stopped by SIGTERM or
Ctrl-C), checks that the reference runtime and the split agree, that Gemma-4-specific split faults are caught (missing
per-layer inputs, missing shared KV, skipped layer, shifted positions, reduced-precision wire, undeclared wire content),
that a defect shared by the split and the deployment's own in-host pipeline still FAILs, that a correctness FAIL closes
the runtime stages, and that the archive verifies.   python -I -B tests/test_gs_pipeline.py
"""
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gs_testlib as T  # noqa: E402

try:
    import torch  # noqa: F401
    import transformers
    from transformers import Gemma4ForCausalLM  # noqa: F401
    sys.path.insert(0, str(T.GS / "tests" / "doubles"))
    import gemma4_split_double as D  # noqa: E402
    HAVE, WHY = True, ""
    transformers.utils.logging.set_verbosity_error()
    transformers.utils.logging.disable_progress_bar()
except Exception as e:  # pragma: no cover
    HAVE, WHY = False, f"{type(e).__name__}: {e}"

import gs_adapter as AD  # noqa: E402
import gs_bench as B  # noqa: E402
import gs_checkpoint as CK  # noqa: E402
import gs_common as C  # noqa: E402
import gs_correctness as CO  # noqa: E402
import gs_deployment as DP  # noqa: E402
import gs_freeze as F  # noqa: E402
import gs_gate as G  # noqa: E402
import gs_results as R  # noqa: E402
import gs_rules as RU  # noqa: E402
import gs_spec as S  # noqa: E402
import gs_split as SP  # noqa: E402
import gs_workload as WL  # noqa: E402

DOUBLE = "tests/doubles/gemma4_split_double.py"
DEV_ITEMS = ([(f"nat-{i:02d}", "natural", "natural", "correctness", 12 + 2 * i, 12, False) for i in range(6)]
             + [(f"cov-{i:02d}", "coverage", "coverage", "correctness", 24, 8, True) for i in range(2)]
             + [(f"short-{k}{i:02d}", "timing", "short", "warmup" if k == "w" else "measured", 8, 6, True) for k, n in (("w", 1), ("m", 3)) for i in range(n)]
             + [(f"long-{k}{i:02d}", "timing", "long", "warmup" if k == "w" else "measured", 16, 6, True) for k, n in (("w", 1), ("m", 3)) for i in range(n)])
VARIANT = '''"""{doc}"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[{up}] / "tests" / "doubles"))
from gemma4_split_double import DoubleAdapter  # noqa: E402

ADAPTER_INFO = {{"contract_version": 1, "implementation": "gemma4-tiny-split-double (test double, {name})", "configuration_id": "double-cut3",
                "capabilities": {caps!r}, "test_double": True, "notes": "{name} variant", "missing_capability_justification": {just!r}}}


def make_adapter(freeze, role, options):
    return DoubleAdapter(ADAPTER_INFO, role, freeze, options)
'''
T2_CAPS = ["greedy_tokens", "stream_tokens", "boundary_audit", "ignore_eos", "segment_step", "unsplit_same_runtime"]
T2_JUST = {"teacher_forced_logprobs": "test of tier T2"}
MIN_CAPS = ["greedy_tokens", "stream_tokens", "boundary_audit", "teacher_forced_logprobs", "ignore_eos", "step_trace"]


def variant(gs, rel, name, caps, just=None):
    up = len(Path(rel).parts) - 1
    (gs / rel).write_text(VARIANT.format(doc=f"{name} variant of the test double", up=up, name=name, caps=caps, just=just or {}))
    return rel


def write_dev_capture(gs, model, endpoint, caps=None, just=None, tokens=None):
    host = socket.gethostname()
    cap = gs / "capture"
    for r in S.ROLES:
        C.write_json(cap / r / "inventory.json", T.fake_inventory(r, hostname=host))
        C.write_json(cap / r / "checkpoint.json", CK.capture(model, r))
        C.write_json(cap / r / "deployment.json", DP.capture(r, [gs / "tests" / "doubles"]))
    C.write_json(cap / "roles.json", T.roles_for(host, host))          # single CPU host plays both roles (dev only)
    C.write_json(cap / "split_capture.json", T.double_split_capture(endpoint=endpoint))
    C.write_json(cap / "capabilities.json", T.capabilities(caps or D.ADAPTER_INFO["capabilities"], just))
    cands = {"candidates": [T.candidates("double-cut3", edge_host=host, server_host=host, running_at_capture=True)]}
    C.write_json(cap / "candidates.json", cands)
    C.write_json(cap / "selected_config.json", SP.select(cands["candidates"], C.read_json(cap / "roles.json")))
    v = G.stage0(cap)
    C.write_json(cap / "stage0_verdict.json", v)
    C.write_json(cap / "workload_tokens.json", tokens or T.synthetic_tokens(gs, dev_items=DEV_ITEMS))
    return v


def run_cli(*args):
    try:
        B.main([str(a) for a in args])
        return 0
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else (0 if e.code is None else 1)


def new_tree(prefix, caps=None, just=None, adapter=DOUBLE, tokens_fn=None):
    """A copied GEMMA-SPLIT tree with a tiny Gemma 4 checkpoint, a dev capture and a dev freeze record."""
    base = T.tmpdir(prefix)
    model = base / "models" / S.MODEL_DIR_NAME
    model.parent.mkdir(parents=True)
    D.make_tiny_checkpoint(model)
    gs = T.copy_gs(base / "repo")
    if adapter != DOUBLE:
        variant(gs, adapter, Path(adapter).stem, caps, just)
    endpoint = f"127.0.0.1:{T.free_port()}"
    v = write_dev_capture(gs, model, endpoint, caps, just, tokens=tokens_fn(model, gs) if tokens_fn else None)
    fp = gs / "frozen" / "freeze_record.json"
    F.write(F.build(gs, adapter, dev=True, notes="pipeline test"), fp)
    return base, model, gs, endpoint, v, fp


class _Tree:
    def dev(self, *args):
        return ["--freeze", self.freeze_path, "--gs-dir", self.gs, "--dev", *args]

    def overrides(self):
        return ["--endpoint", self.endpoint, "--device", "cpu", "--model-dir", self.model]

    def start_server(self, stage, block=None):
        ready = self.base / f"ready-{time.time_ns()}"
        cmd = [sys.executable, "-I", "-B", str(self.gs / "gs_bench.py"), "serve", *map(str, self.dev("--stage", stage)), *map(str, self.overrides()),
               "--ready-file", str(ready)] + (["--block", block] if block else [])
        p = subprocess.Popen(cmd, env=dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_VERBOSITY="error"), stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
        t0 = time.time()
        while not ready.exists():
            if p.poll() is not None or time.time() - t0 > 180:
                _, err = p.communicate(timeout=10) if p.poll() is not None else ("", "timeout")
                self.fail(f"server did not start: {err[-2000:]}")
            time.sleep(0.1)
        return p

    def stop_server(self, p, sig=signal.SIGTERM):
        p.send_signal(sig)
        out, err = p.communicate(timeout=120)
        self.assertEqual(p.returncode, 0, err[-2000:])
        self.assertIn("record written" if "--stage 2" not in " ".join(p.args) else "server split memory", out + err)

    def run_block(self, block):
        if block.startswith("split"):
            p = self.start_server(5, block)
            self.assertEqual(run_cli("e2e", *self.dev("--block", block), *self.overrides()), 0)
            self.stop_server(p, signal.SIGINT if block == "split_B" else signal.SIGTERM)   # Ctrl-C must also write the record
        else:
            self.assertEqual(run_cli("e2e", *self.dev("--block", block), *self.overrides()), 0)


@unittest.skipUnless(HAVE, f"torch / transformers with Gemma 4 unavailable: {WHY}")
class Pipeline(_Tree, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base, cls.model, cls.gs, cls.endpoint, cls.stage0, cls.freeze_path = new_tree("gs_pipe_")
        cls.freeze = F.load(cls.freeze_path)
        cls.rd = R.run_dir(cls.gs, cls.freeze)

    def test_1_reference_runtime_matches_hf_generate(self):
        import gs_reference as REF
        model, info = REF.load_model(self.model, "float32", "cpu")
        self.assertEqual((info["class"], info["class_source"]), ("Gemma4ForCausalLM", "config.architectures[0]"))
        ids = T.synthetic_tokens(self.gs, dev_items=DEV_ITEMS)["items"][0]["ids"]
        g = REF.greedy(model, ids, 12, [1], True, "cpu")
        gen = model.generate(torch.tensor([ids]), max_new_tokens=12, do_sample=False, min_new_tokens=12)
        self.assertEqual(g["tokens"], gen[0, len(ids):].tolist())
        tids, _, nan = RU.topk_rows(REF.teacher_forced(model, ids + g["tokens"], len(ids) - 1, "cpu"), 4)
        self.assertFalse(nan)
        self.assertEqual([r[0] for r in tids], g["tokens"])
        h = D.Halves(self.model, 3)
        self.assertTrue(torch.equal(h.local(ids), h.unsplit_logits(ids)))           # the double's split is exact on CPU
        self.assertEqual(h.server_needs_kv, ["full_attention"])                      # a Gemma-4 cut can need shared KV on the wire
        with self.assertRaises(ValueError):                                          # skipped teacher-forced positions are refused
            RU.compare_rows(iter([(5, torch.zeros(2, 4))]), [[0, 1]] * 4, [[-1.0, -2.0]] * 4, 4)

    def test_2_tokenize_frozen_workload(self):
        w = WL.tokenize(self.model, self.gs / "workload")
        self.assertEqual(WL.validate_tokens(w, self.gs / "workload"), [])
        totals = {i["id"]: i["n"] + i["gen_tokens"] for i in w["items"]}
        self.assertEqual(totals[w["probe_item"]], max(totals.values()))      # character-level test tokenizer: a natural prompt
        self.assertEqual(w["eos_token_ids"], [1])
        long_ = next(i for i in w["items"] if i["id"] == "long-m00")
        self.assertEqual((long_["n"], long_["ids"][0]), (2048, w["bos_token_id"]))

    def test_3_full_pipeline(self):
        self.assertEqual(self.stage0["verdict"], "CONTINUE", self.stage0["reasons"])
        self.assertTrue(self.stage0["derived"]["differences_from_qwen3_e1opt"])
        # refusals before anything runs
        self.assertEqual(run_cli("memory", "--freeze", self.freeze_path, "--gs-dir", self.gs, "--what", "reference", "--role", "edge"), 1)
        self.assertEqual(run_cli("components", *self.dev("--role", "edge"), *self.overrides()), 3)          # gate closed
        sparse = self.base / "big_model"
        sparse.mkdir()
        with open(sparse / "weights.bin", "wb") as f:
            f.truncate(S.DEV_MAX_MODEL_BYTES + 1)                                                         # sparse: no disk used
        self.assertEqual(run_cli("memory", *self.dev("--what", "reference", "--role", "edge"), "--model-dir", sparse, "--device", "cpu"), 1)
        # Stage 2
        p = self.start_server(2)
        self.assertEqual(run_cli("memory", *self.dev("--what", "split", "--role", "edge"), *self.overrides()), 0)
        self.stop_server(p)
        for role in S.ROLES:
            self.assertEqual(run_cli("memory", *self.dev("--what", "reference", "--role", role), *self.overrides()), 0)
        self.assertEqual(run_cli("gate", *self.dev("--stage", 2)), 0)
        g2 = G.read_gate(self.rd, "stage2")
        self.assertEqual((g2["verdict"], g2["split_per_host"], g2["mode"]), ("CONTINUE", {"edge": "FITS", "server": "FITS"}, "dev"))
        mem = C.read_json(R.stage_dir(self.rd, "stage2") / "edge" / "split_memory.json")
        self.assertEqual((mem["probe_item"], mem["completed_tokens"], mem["ooms"]), ("nat-05", 12, 0))     # 22 + 12: the longest dev item
        # Stage 3
        for role in ("server", "edge"):
            self.assertEqual(run_cli("reference", *self.dev("--role", role), *self.overrides()), 0)
        p = self.start_server(3)
        self.assertEqual(run_cli("correctness", *self.dev("--what", "split"), *self.overrides()), 0)
        self.stop_server(p)
        self.assertEqual(run_cli("correctness", *self.dev("--what", "pipeline"), *self.overrides()), 0)
        self.assertEqual(run_cli("gate", *self.dev("--stage", 3)), 0)
        dec = C.read_json(R.stage_dir(self.rd, "stage3") / "decision.json")
        self.assertEqual((dec["verdict"], dec["tier"]), ("PASS", "T1"))
        tf = dec["rules"]["teacher_forced"]
        self.assertEqual((tf["D_split_pp"], tf["KL_split"], tf["floor"]), (0.0, 0.0, "hardware"))
        self.assertIn("pipeline", dec["diagnostics"])                                 # reported, never a floor
        self.assertEqual(dec["boundary_audit"]["problems"], [])
        self.assertGreater(dec["boundary_audit"]["bytes_by_direction"]["edge_to_server"], 0)
        # Stage 4 (isolated segments)
        for role in S.ROLES:
            self.assertEqual(run_cli("components", *self.dev("--role", role), *self.overrides()), 0)
        self.assertEqual(run_cli("gate", *self.dev("--stage", 4)), 0)
        comp = C.read_json(R.stage_dir(self.rd, "stage4") / "summary.json")
        self.assertEqual(comp["mode"], "isolated")
        self.assertGreater(comp["decode_step_ns"]["server"]["long"], 0)
        # Stage 5, frozen block order (the double declares unsplit_same_runtime: native blocks are secondary)
        order = S.block_order(self.freeze["adapter"]["info"]["capabilities"])
        self.assertEqual(len(order), 6)
        self.assertEqual(run_cli("e2e", *self.dev("--block", "edge_only"), *self.overrides()), 1)              # out of order
        for block in order:
            self.run_block(block)
        rc = run_cli("gate", *self.dev("--stage", 5))
        summ = C.read_json(R.stage_dir(self.rd, "stage5") / "summary.json")
        g5 = G.read_gate(self.rd, "stage5")
        self.assertIn(summ["verdict"], ("COMPLETE", "UNSTABLE"))           # CPU timing noise may trip the drift rule
        self.assertEqual(rc, 0 if g5["verdict"] == "COMPLETE" else 3)
        self.assertIn(summ["primary"]["category"], S.EFFICIENCY_CATEGORIES)
        self.assertEqual(summ["primary"]["n_pairs"], 6)
        self.assertTrue(summ["blocks"]["edge_only"]["runtime"].startswith("reference runtime"))
        self.assertIn("tpot_split_over_edge_only_native", summ["secondary"])
        self.assertTrue(summ["decomposition"]["per_group"]["short"]["available"])
        self.assertEqual(summ["token_agreement"]["split_A~edge_only_native"]["identical"], 6)   # same tokens, split and unsplit
        rows = C.read_jsonl(R.stage_dir(self.rd, "stage5") / "split_A" / "requests.jsonl")
        self.assertTrue(all(r["n_tokens"] == 6 and len(r["token_ns"]) == 6 for r in rows))
        # Stage 6
        for role in S.ROLES:
            self.assertEqual(run_cli("verify-checkpoint", *self.dev("--role", role), *self.overrides()), 0)
        run_cli("outcome", *self.dev())
        res = G.outcome(self.rd, self.freeze)
        self.assertEqual(res["problems"], [])                             # every stored gate recomputes identically
        if g5["verdict"] == "COMPLETE":
            self.assertEqual(C.read_json(R.stage_dir(self.rd, "final") / "outcome.json")["outcome"], "COMPLETE")
        else:
            self.assertIn("PENDING", res["note"])
        out = self.base / "archive" / f"gemma_split_{self.freeze['run_id']}_DEV.tgz"
        self.assertEqual(run_cli("archive", "--run-id", self.freeze["run_id"], "--out", out, "--gs-dir", self.gs), 0)
        self.assertEqual(run_cli("verify-archive", "--archive", out), 0)
        ev = R.read_events(self.rd)
        self.assertTrue(any(e["kind"] == "END" and e["cmd"] == "e2e" and e["rc"] == "0" and e["part"] == "split_B" for e in ev))
        self.assertTrue(any(e["kind"] == "END" and e["cmd"] == "components" and e["rc"] == "3" for e in ev))   # refusals are logged too
        self.assertFalse(R.crashed(self.rd, "stage5", "split_A"))


@unittest.skipUnless(HAVE, f"torch / transformers with Gemma 4 unavailable: {WHY}")
class MinimalCapabilities(_Tree, unittest.TestCase):
    """The likely lab case: no isolated segments, no unsplit mode, no in-host split. Stage 3 uses the hardware floor,
    Stage 4 the in-path trace (server half serving), Stage 5 the reference-runtime baselines only."""

    @classmethod
    def setUpClass(cls):
        cls.base, cls.model, cls.gs, cls.endpoint, cls.stage0, cls.freeze_path = new_tree(
            "gs_min_", caps=MIN_CAPS, adapter="adapter/gs_adapter_minimal_double.py")
        cls.freeze = F.load(cls.freeze_path)
        cls.rd = R.run_dir(cls.gs, cls.freeze)

    def test_minimal_pipeline(self):
        p = self.start_server(2)
        self.assertEqual(run_cli("memory", *self.dev("--what", "split", "--role", "edge"), *self.overrides()), 0)
        self.stop_server(p, signal.SIGINT)
        for role in S.ROLES:
            self.assertEqual(run_cli("memory", *self.dev("--what", "reference", "--role", role), *self.overrides()), 0)
        self.assertEqual(run_cli("gate", *self.dev("--stage", 2)), 0)
        for role in ("server", "edge"):
            self.assertEqual(run_cli("reference", *self.dev("--role", role), *self.overrides()), 0)
        p = self.start_server(3)
        self.assertEqual(run_cli("correctness", *self.dev("--what", "split"), *self.overrides()), 0)
        self.stop_server(p)
        self.assertEqual(run_cli("gate", *self.dev("--stage", 3)), 0)
        self.assertNotIn("pipeline", C.read_json(R.stage_dir(self.rd, "stage3") / "decision.json")["diagnostics"])
        p = self.start_server(4)
        self.assertEqual(run_cli("components", *self.dev("--role", "edge"), *self.overrides()), 0)
        self.stop_server(p)
        self.assertEqual(run_cli("gate", *self.dev("--stage", 4)), 0)
        comp = C.read_json(R.stage_dir(self.rd, "stage4") / "summary.json")
        self.assertEqual(comp["mode"], "in_path")
        self.assertGreater(comp["in_path"]["long"]["server_compute"]["n"], 0)
        order = S.block_order(MIN_CAPS)
        self.assertEqual(order, S.BLOCK_ORDER)
        for block in order:
            self.run_block(block)
        run_cli("gate", *self.dev("--stage", 5))
        summ = C.read_json(R.stage_dir(self.rd, "stage5") / "summary.json")
        self.assertIn(summ["verdict"], ("COMPLETE", "UNSTABLE"))
        self.assertTrue(summ["blocks"]["server_only"]["runtime"].startswith("reference runtime"))
        self.assertEqual(summ["decomposition"]["source"], "in_path")
        self.assertEqual(summ["token_agreement"]["split_A~server_only"]["identical"], 6)   # the reference runtime generates the split's tokens


@unittest.skipUnless(HAVE, f"torch / transformers with Gemma 4 unavailable: {WHY}")
class StageOneTools(_Tree, unittest.TestCase):
    """verify-workload (the frozen token ids re-derived with a host's tokenizer) and adapter-smoke (timer-free contract check)."""

    @classmethod
    def setUpClass(cls):
        cls.base, cls.model, cls.gs, cls.endpoint, cls.stage0, cls.freeze_path = new_tree(
            "gs_s1_", tokens_fn=lambda model, gs: WL.tokenize(model, gs / "workload"))
        cls.freeze = F.load(cls.freeze_path)
        cls.rd = R.run_dir(cls.gs, cls.freeze)

    def test_verify_workload_and_smoke(self):
        for role in S.ROLES:
            self.assertEqual(run_cli("verify-workload", *self.dev("--role", role), *self.overrides()), 0)
            self.assertTrue(C.read_json(R.stage_dir(self.rd, "stage1") / f"{role}_workload.json")["match"])
        server = AD.make(F.load(self.freeze_path), "server", self.gs, {"endpoint": self.endpoint, "device": "cpu", "model_dir": str(self.model)})
        ready, stop = threading.Event(), threading.Event()
        th = threading.Thread(target=server.serve, kwargs={"ready": ready.set, "stop": stop}, daemon=True)
        th.start()
        self.assertTrue(ready.wait(120))
        try:
            self.assertEqual(run_cli("adapter-smoke", "--adapter", DOUBLE, "--role", "edge", "--gs-dir", self.gs, "--endpoint", self.endpoint,
                                     "--device", "cpu", "--model-dir", self.model, "--dev"), 0)
            self.assertEqual(run_cli("adapter-smoke", "--adapter", DOUBLE, "--role", "edge", "--gs-dir", self.gs, "--endpoint", self.endpoint), 1)
            # ^ without --dev: a test double and an uncommitted declaration are refused before anything runs
            T.write_fake_adapter(self.gs, test_double=False)
            try:
                self.assertEqual(run_cli("adapter-smoke", "--adapter", "adapter/gs_adapter_fake.py", "--role", "edge", "--gs-dir", self.gs,
                                         "--endpoint", self.endpoint, "--device", "cpu", "--model-dir", self.model, "--dev"), 1)
                # ^ --dev never runs a deployment's (non-test-double) adapter
            finally:
                (self.gs / "adapter" / "gs_adapter_fake.py").unlink()
        finally:
            stop.set()
            th.join(timeout=30)
        self.assertFalse((self.rd / "stage2_memory").exists())          # the smoke test writes nothing


@unittest.skipUnless(HAVE, f"torch / transformers with Gemma 4 unavailable: {WHY}")
class Faults(unittest.TestCase):
    """Stage 3 with injected split faults: each must be caught (FAIL or INVALID), never PASS."""

    @classmethod
    def setUpClass(cls):
        cls.templates, cls.models = {}, {}
        for tier, adapter, caps, just in (("T1", DOUBLE, None, None), ("T2", "tests/doubles/gemma4_double_t2.py", T2_CAPS, T2_JUST)):
            base, model, gs, _, _, fp = new_tree(f"gs_faults_{tier}_", caps=caps, just=just, adapter=adapter)
            fr = F.load(fp)
            rd = R.run_dir(gs, fr)
            G.write_gate(rd, fr, "stage2", {"verdict": "CONTINUE", "final": False, "repeatable": [], "reference_feasible": True})   # shortcut
            CO.run_reference_server(fr, gs, "cpu", str(model), "dev")
            CO.run_reference_edge(fr, gs, "cpu", str(model), "dev")
            if tier == "T1":
                CO.run_pipeline(fr, gs, AD.make(fr, "server", gs, {"device": "cpu", "model_dir": str(model)}), "dev")
            cls.templates[tier], cls.models[tier] = base, model

    def run_split(self, tier="T1", faults="", pipeline_faults=None):
        root = T.tmpdir("gs_fault_case_")
        shutil.rmtree(root)
        shutil.copytree(self.templates[tier], root)
        gs = root / "repo" / "diagnostics" / "deployment_gemma_split"
        fr = F.load(gs / "frozen" / "freeze_record.json")
        rd = R.run_dir(gs, fr)
        model = str(root / "models" / S.MODEL_DIR_NAME)
        if pipeline_faults is not None:
            shutil.rmtree(R.stage_dir(rd, "stage3") / "pipeline")
            CO.run_pipeline(fr, gs, AD.make(fr, "server", gs, {"device": "cpu", "model_dir": model, "faults": pipeline_faults}), "dev")
        opts = {"endpoint": f"127.0.0.1:{T.free_port()}", "device": "cpu", "model_dir": model, "faults": faults}
        server = AD.make(fr, "server", gs, opts)
        ready, stop = threading.Event(), threading.Event()
        th = threading.Thread(target=lambda: _quiet(server.serve, ready=ready.set, stop=stop), daemon=True)
        started = C.utc_ms()
        th.start()
        self.assertTrue(ready.wait(120))
        try:
            CO.run_split(fr, gs, AD.make(fr, "edge", gs, opts), "dev")
        finally:
            stop.set()
            th.join(timeout=30)
        C.write_json(R.stage_dir(rd, "stage3") / "split" / "server_serve.json",          # what `gs_bench.py serve --stage 3` writes
                     {"started_utc": started, "ended_utc": C.utc_ms(), "provenance": R.provenance(fr, "stage3", "server", ["cpu"], "dev")})
        return RU.decide_stage3(rd, fr, dev=True), rd, fr, gs

    def test_correct_split_passes_both_tiers(self):
        for tier in ("T1", "T2"):
            dec = self.run_split(tier)[0]
            self.assertEqual((dec["verdict"], dec["tier"]), ("PASS", tier), dec["problems"])

    def test_numerical_faults_fail(self):
        for fault in ("drop_ple", "skip_layer", "position_offset", "wire_bf16_silent"):
            dec = self.run_split("T1", fault)[0]
            self.assertEqual(dec["verdict"], "FAIL", fault)
            self.assertFalse(dec["rules"]["teacher_forced"]["pass_KL"], fault)

    def test_defect_shared_with_in_host_pipeline_still_fails(self):
        """The deployment's in-host pipeline carries the same defect: it must not become a floor that absorbs it."""
        dec = self.run_split("T1", "skip_layer", pipeline_faults="skip_layer")[0]
        self.assertEqual(dec["verdict"], "FAIL")
        self.assertGreater(dec["diagnostics"]["pipeline"]["teacher_forced"]["D_pp"], dec["rules"]["teacher_forced"]["hardware"]["D_pp"])

    def test_t2_tier_catches_skipped_layer(self):
        dec = self.run_split("T2", "skip_layer")[0]
        self.assertEqual(dec["tier"], "T2")
        self.assertNotIn("teacher_forced", dec["rules"])
        self.assertEqual(dec["verdict"], "FAIL", dec["rules"]["generation"]["per_item"])

    def test_invalid_cases(self):
        for fault, needle, cause in (("omit_shared_kv", "exceptions", "exceptions"),       # server cannot run its KV-shared layers
                                     ("undeclared_extra", "input_ids", "boundary"),         # token ids on the wire, not in the capture
                                     ("wire_bf16", "bfloat16", "boundary")):               # bfloat16 on the wire, float32 declared
            dec = self.run_split("T1", fault)[0]
            self.assertEqual(dec["verdict"], "INVALID", fault)
            self.assertIn(needle, " ".join(p["reason"] for p in dec["problems"]), fault)
            self.assertIn(cause, [p["cause"] for p in dec["problems"]], fault)

    def test_pipeline_errors_and_contention_never_fail(self):
        dec = self.run_split("T1", pipeline_faults="omit_shared_kv")[0]       # the in-host diagnostic breaks, the split is correct
        self.assertEqual(dec["verdict"], "PASS", dec["problems"])
        self.assertTrue(dec["diagnostics"]["pipeline_errors"])
        dec, rd, fr, gs = self.run_split("T1", "omit_shared_kv")              # split exceptions ...
        p = R.stage_dir(rd, "stage3") / "split" / "server_serve.json"
        C.write_json(p, dict(C.read_json(p), foreign_during=[{"pid": 4242, "gpu_uuid": "GPU-other-user"}]), overwrite=True)
        causes = [x["cause"] for x in RU.decide_stage3(rd, fr, dev=True)["problems"]]   # ... while another user's job ran
        self.assertIn("contention", causes)
        self.assertNotIn("exceptions", causes)                                 # never counted toward FAIL
        dec, rd, fr, gs = self.run_split("T1", "omit_shared_kv")              # a job that appears only when the server is stopped
        smeta = C.read_json(R.stage_dir(rd, "stage3") / "split" / "meta.json")
        late = C.utc_ms(C.parse_utc(smeta["ended_utc"]) + 3600)
        p = R.stage_dir(rd, "stage3") / "split" / "server_serve.json"
        C.write_json(p, dict(C.read_json(p), contention_after={"foreign": [{"pid": 4242}]},
                             foreign_during=[{"pid": 4242, "first_seen_utc": late}]), overwrite=True)
        causes = [x["cause"] for x in RU.decide_stage3(rd, fr, dev=True)["problems"]]
        self.assertIn("exceptions", causes)                                    # it cannot turn the exceptions into contention
        self.assertNotIn("contention", causes)
        dec, rd, fr, gs = self.run_split("T1", "skip_layer")                  # a rule failure ...
        p = R.stage_dir(rd, "stage3") / "split" / "server_serve.json"
        C.write_json(p, dict(C.read_json(p), foreign_during=[{"pid": 4242}]), overwrite=True)
        self.assertEqual(RU.decide_stage3(rd, fr, dev=True)["verdict"], "FAIL")   # ... is never erased by contention

    def test_exception_repeats_end_in_fail_and_fail_closes_runtime(self):
        dec, rd, fr, gs = self.run_split("T1", "omit_shared_kv")
        T.log_runs(rd, fr, "stage3")                          # the drivers ran directly: the events the CLI would log
        g = G.gate_stage3(rd, fr, dev=True)
        self.assertEqual((g["verdict"], g["repeatable"][0]["cause"]), ("INVALID", "exceptions"))
        for attempt in range(2):
            G.invalidate(rd, fr, "stage3", ["split"], f"attempt {attempt}")
            opts = {"endpoint": f"127.0.0.1:{T.free_port()}", "device": "cpu", "model_dir": str(Path(gs).parents[2] / "models" / S.MODEL_DIR_NAME),
                    "faults": "omit_shared_kv"}
            server = AD.make(fr, "server", gs, opts)
            ready, stop = threading.Event(), threading.Event()
            th = threading.Thread(target=lambda: _quiet(server.serve, ready=ready.set, stop=stop), daemon=True)
            started = C.utc_ms()
            th.start()
            ready.wait(120)
            CO.run_split(fr, gs, AD.make(fr, "edge", gs, opts), "dev")
            stop.set()
            th.join(timeout=30)
            C.write_json(R.stage_dir(rd, "stage3") / "split" / "server_serve.json",
                         {"started_utc": started, "ended_utc": C.utc_ms(), "provenance": R.provenance(fr, "stage3", "server", ["cpu"], "dev")})
            T.log_runs(rd, fr, "stage3")
            g = G.gate_stage3(rd, fr, dev=True)
        self.assertEqual((g["verdict"], g["final"]), ("FAIL", True))       # not runnable as captured, after the allowed attempts
        fp = gs / "frozen" / "freeze_record.json"
        args = ["--freeze", fp, "--gs-dir", gs, "--dev", "--device", "cpu", "--endpoint", "127.0.0.1:1",
                "--model-dir", Path(gs).parents[2] / "models" / S.MODEL_DIR_NAME]
        self.assertEqual(run_cli("components", *args, "--role", "edge"), 3)
        self.assertEqual(run_cli("gate", "--freeze", fp, "--gs-dir", gs, "--dev", "--stage", 4), 3)
        self.assertEqual(run_cli("invalidate", "--freeze", fp, "--gs-dir", gs, "--dev", "--stage", 3, "--part", "split", "--note", "x"), 3)

    def test_timing_field_and_reference_infeasible(self):
        dec, rd, fr, gs = self.run_split("T1")
        p = R.stage_dir(rd, "stage3") / "split" / "compare.jsonl"
        rows = C.read_jsonl(p)
        rows[0]["latency_ms"] = 1.0
        C.write_jsonl(p, rows, overwrite=True)
        dec2 = RU.decide_stage3(rd, fr, dev=True)
        self.assertEqual(dec2["verdict"], "INVALID")
        self.assertIn("timing field", " ".join(p["reason"] for p in dec2["problems"]))
        self.assertEqual(RU.decide_stage3(rd, fr, reference_feasible=False, dev=True)["verdict"], "NOT_EVALUABLE")
        p2 = R.stage_dir(rd, "stage3") / "split" / "server_serve.json"
        C.write_json(p2, dict(C.read_json(p2), started_utc="2000-01-01T00:00:00.000Z", ended_utc="2000-01-01T00:01:00.000Z"), overwrite=True)
        rows[0].pop("latency_ms")
        C.write_jsonl(p, rows, overwrite=True)
        dec3 = RU.decide_stage3(rd, fr, dev=True)
        self.assertIn("outside the server's serve window", " ".join(p["reason"] for p in dec3["problems"]))   # some other process served


def _quiet(fn, **kw):
    try:
        fn(**kw)
    except Exception:
        pass        # the omit_shared_kv fault crashes the server half by design; the edge records the failure


if __name__ == "__main__":
    unittest.main(verbosity=2)
