"""GEMMA-SPLIT protocol-logic tests (CPU, standard library only; no torch).  python -I -B tests/test_gs_protocol.py"""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gs_testlib as T  # noqa: E402
import gs_adapter as AD  # noqa: E402
import gs_archive as AR  # noqa: E402
import gs_checkpoint as CK  # noqa: E402
import gs_common as C  # noqa: E402
import gs_deployment as DP  # noqa: E402
import gs_freeze as F  # noqa: E402
import gs_gate as G  # noqa: E402
import gs_inventory as INV  # noqa: E402
import gs_memory as M  # noqa: E402
import gs_results as R  # noqa: E402
import gs_rules as RU  # noqa: E402
import gs_spec as S  # noqa: E402
import gs_split as SP  # noqa: E402
import gs_stats as ST  # noqa: E402
import gs_workload as WL  # noqa: E402

REPO = T.GS.parents[1]
E1OPT_COMMIT = "d9ba549f006bd18f960b385953acbce8259b9dff"
E1OPT_DIR = "diagnostics/deployment_e1opt"     # the only reference in GEMMA-SPLIT code: a read-only immutability guard
GUARD_LITERALS = {("test_gs_protocol.py", E1OPT_DIR)}
TP = {"split_A": 60.0, "split_B": 60.0, "edge_only": 40.0, "server_only": 20.0}


class Independence(unittest.TestCase):
    def test_no_code_reference_to_qwen3_experiments(self):
        """No import of an E0 / E1-OPT module and no string literal naming their directories in any executable file
        (docstrings, which cite them, are excluded). The literals of this test are assembled, so it does not flag itself."""
        import ast
        dirs = ("deployment_" + "e0", "deployment_" + "e1opt")
        mods = ("e0_", "e1_", "split_runtime")
        for p in list(T.GS.glob("gs_*.py")) + list(T.GS.glob("tests/**/*.py")):
            tree = ast.parse(p.read_text(), filename=str(p))
            docs = {id(n.body[0].value) for n in ast.walk(tree) if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body
                    and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
            for n in ast.walk(tree):
                if isinstance(n, ast.Import):
                    self.assertFalse([a.name for a in n.names if a.name.startswith(mods)], f"{p.name} imports an E0/E1 module")
                elif isinstance(n, ast.ImportFrom):
                    self.assertFalse((n.module or "").startswith(mods), f"{p.name} imports from an E0/E1 module")
                elif isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs and (p.name, n.value) not in GUARD_LITERALS:
                    self.assertFalse(any(d in n.value for d in dirs), f"{p.name}: string literal names an E0/E1-OPT directory")

    def test_e1opt_frozen_reference_untouched(self):
        if C.git(["rev-parse", "--git-dir"], REPO) is None:
            self.skipTest("not a git checkout")
        tag = C.git(["rev-parse", "--verify", "--quiet", "e1opt-prereg-v1^{commit}"], REPO)
        if tag is None:
            self.skipTest("tag e1opt-prereg-v1 not present in this clone (git fetch --tags)")
        self.assertEqual(tag, E1OPT_COMMIT)
        r = subprocess.run(["git", "diff", "--quiet", "e1opt-prereg-v1", "--", E1OPT_DIR], cwd=REPO)
        self.assertEqual(r.returncode, 0, "the E1-OPT directory differs from its frozen tag")
        self.assertEqual(C.git(["status", "--porcelain", "--", E1OPT_DIR], REPO), "")

    def test_spec_values(self):
        self.assertEqual(sum(S.CORR_PER_DATASET.values()), 30)
        self.assertEqual(S.block_order([]), ("split_A", "edge_only", "server_only", "split_B"))
        self.assertEqual(S.block_order(["unsplit_same_runtime"])[-1], "split_B")
        self.assertEqual((S.TF_D_MARGIN_PP, S.TF_KL_FACTOR, S.GEN_MARGIN_NATS, S.GEN_MAX_UNEXPLAINED), (0.2, 2.0, 0.5, 3))
        self.assertNotEqual(S.PREREG_TAG, S.QWEN3_E1OPT_REFERENCE["tag"])
        for f in ("gs_rules.py", "gs_gate.py", "gs_freeze.py", "gs_spec.py", S.PREREG_FILE):
            self.assertIn(f, S.DECISION_FILES)
        for f in ("gs_reference.py", "gs_correctness.py", "gs_memory.py", "gs_profile.py"):
            self.assertIn(f, S.DECISION_FILES)                # measurement drivers are pinned as well
        for f in S.ORCHESTRATION_FILES:
            self.assertNotIn(f, S.DECISION_FILES)            # orchestration: documented fixes allowed in a new record
        self.assertTrue(set(F.protocol_params()) >= {"BOOT_SEED", "DRIFT_MAX_REL", "TF_KL_EPS", "REPEAT_CAPS", "DECISION_FILES"})


class CommonAndInventory(unittest.TestCase):
    def test_redaction(self):
        env = C.redact_env({"HF_TOKEN": "hf_abcdefghijklmnopqrstuv", "CUDA_VISIBLE_DEVICES": "1", "MY_API_KEY": "x", "URL": "https://u:p@h/x"})
        self.assertEqual(env["HF_TOKEN"], C.REDACTED)
        self.assertEqual(env["MY_API_KEY"], C.REDACTED)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")
        self.assertNotIn("u:p", env["URL"])
        s = C.redact_text("python serve.py --token abc123 --api-key=zzz HF_TOKEN=hf_0123456789abcdefghij git+https://ghp_aaaaaaaaaaaaaaaaaaaaaa@github.com/x")
        for leak in ("abc123", "zzz", "hf_0123456789", "ghp_aaaa"):
            self.assertNotIn(leak, s)
        self.assertEqual(INV.filtered_env({"HF_TOKEN": "s", "HOME": "/h", "CUDA_HOME": "/c", "GEMMA_PORT": "1"}),
                         {"CUDA_HOME": "/c", "GEMMA_PORT": "1", "HF_TOKEN": C.REDACTED})

    def test_write_once_and_timestamps(self):
        d = T.tmpdir()
        C.write_json(d / "a.json", {"x": 1})
        with self.assertRaises(FileExistsError):
            C.write_json(d / "a.json", {"x": 2})
        self.assertEqual(C.read_json(d / "a.json"), {"x": 1})
        self.assertEqual(C.parse_utc(C.utc_ms(1700000000.25)), 1700000000.25)

    def test_smi_parsers_unified_memory(self):
        s = INV.parse_smi_gpus(T.SMI_SERVER + "\n")
        self.assertEqual([g["memory_total_mib"] for g in s], [49140] * 3)
        e = INV.parse_smi_gpus(T.SMI_EDGE + ", [N/A]")
        self.assertIsNone(e[0]["memory_total_mib"])
        self.assertIsNone(e[0]["power_limit_w"])
        self.assertEqual(e[0]["memory_raw"][0], "[N/A]")
        self.assertEqual(INV.parse_smi_gpus("0, NVIDIA RTX A6000, GPU-a, 49140 MiB, 4 MiB, 580, 8.6, 0:1:0, 300.00 W")[0]["power_limit_w"], 300.0)
        apps = INV.parse_smi_apps("1234, python, [N/A], GPU-gb10-0000\n77, x, 512 MiB, GPU-a")
        self.assertEqual((apps[0]["used_memory_mib"], apps[1]["used_memory_mib"]), (None, 512))
        self.assertEqual(INV.parse_ethtool_speed("\tSpeed: 10000Mb/s\n"), 10000)
        self.assertEqual(INV.parse_meminfo("MemTotal:  10 kB\nMemAvailable: 4 kB")["MemAvailable"], 4096)

    def test_live_collect_smoke(self):
        """The real collector on this machine (a CPU-only box): every tool failure is recorded, nothing raises."""
        inv = INV.collect("edge", "localhost")
        for k in ("hostname", "arch", "gpus", "torch", "packages", "env", "nics", "route_to_peer", "versions", "pip_freeze", "memory", "clock"):
            self.assertIn(k, inv)
        self.assertEqual(inv["peer_resolved"], "127.0.0.1")
        self.assertEqual(inv["route_to_peer"]["rc"], 0)
        self.assertIn(inv["hostname"], INV.summary_text(inv))
        self.assertTrue(set(F.STRICT_VERSION_KEYS) <= set(inv["versions"]))
        v = INV.verify_role(inv, "edge")
        if not inv["gpus"]:
            self.assertFalse(v["ok"])                    # no GB10 here: this box is not the DGX Spark
        self.assertFalse(any(C.SECRET_NAME.search(k) and val != C.REDACTED for k, val in inv["env"].items()))

    def test_role_verification(self):
        ok = INV.roles_record(T.fake_inventory("edge"), T.fake_inventory("server"))
        self.assertTrue(ok["verified"], ok["problems"])
        self.assertFalse(INV.roles_record(T.fake_inventory("server"), T.fake_inventory("edge"))["verified"])
        two = INV.roles_record(T.fake_inventory("edge"), T.fake_inventory("server", gpus=[T.gpu(i, "NVIDIA RTX A6000", f"u{i}", 49140) for i in range(2)]))
        self.assertTrue(any("gpu_count" in p for p in two["problems"]))
        other = INV.roles_record(T.fake_inventory("edge"), T.fake_inventory("server", hostname="some-other-box"))
        self.assertTrue(any("hostname" in p for p in other["problems"]))
        x86 = INV.roles_record(T.fake_inventory("edge", arch="x86_64"), T.fake_inventory("server"))
        self.assertTrue(any("arch" in p for p in x86["problems"]))
        self.assertFalse(INV.roles_record(T.fake_inventory("edge", hostname=T.SERVER_HOST), T.fake_inventory("server"))["verified"])
        noroute = INV.roles_record(T.fake_inventory("edge", route_ok=False), T.fake_inventory("server"))
        self.assertTrue(any("route" in p for p in noroute["problems"]))


class Checkpoint(unittest.TestCase):
    def test_manifest_summary_features_compare(self):
        d = T.tmpdir()
        md = T.fake_model_dir(d)
        os.symlink(md / "tokenizer.json", md / "tokenizer_link.json")
        (md / ".cache").mkdir()
        (md / ".cache" / "junk").write_text("x")
        cap = CK.capture(md, "server")
        paths = [e["path"] for e in cap["manifest"]]
        self.assertIn("tokenizer_link.json", paths)
        self.assertNotIn(".cache/junk", paths)
        self.assertTrue(cap["dir_name_matches"])
        f = cap["features"]
        self.assertTrue(f["per_layer_embeddings"] and f["kv_sharing"] and f["mixed_attention"] and f["logit_softcapping"])
        self.assertIsNone(f["tied_embeddings"])           # absent from config.json: never assumed
        self.assertEqual(CK.kv_sharing_map(cap["summary"]), {5: 2, 6: 4, 7: 2})
        self.assertIsNone(cap["hf_revision"]["revision"])
        nested = CK.summarize_config({"architectures": ["Gemma4ForConditionalGeneration"], "model_type": "gemma4", "torch_dtype": "bfloat16",
                                      "text_config": {"num_hidden_layers": 42, "model_type": "gemma4_text"}, "vision_config": {}})
        self.assertEqual((nested["num_hidden_layers"], nested["dtype"], nested["has_vision"]), (42, "bfloat16", True))
        e = dict(cap, manifest=[x for x in cap["manifest"] if x["path"] != "tokenizer.json"])
        self.assertEqual(CK.compare(cap, cap)["status"], "identical")
        self.assertEqual(CK.compare(e, cap)["status"], "edge_subset")
        bad = dict(cap, manifest=[dict(x, sha256="0" * 64) if x["path"] == "config.json" else x for x in cap["manifest"]])
        self.assertEqual(CK.compare(bad, cap)["status"], "mismatch")
        self.assertEqual(CK.quick_check(cap), [])
        self.assertTrue(CK.deep_check(cap))
        w = md / "model.safetensors"
        st = w.stat()
        w.write_bytes(b"\1" * 64)                                   # same size, new content ...
        os.utime(w, ns=(st.st_atime_ns, st.st_mtime_ns + 10 ** 9))   # ... written later (mtime ticks are coarse)
        self.assertEqual(CK.quick_check(cap), ["changed model.safetensors"])
        self.assertFalse(CK.deep_check(cap))
        os.utime(w, ns=(st.st_atime_ns, st.st_mtime_ns))             # stat-identical but different bytes: only the deep check sees it
        self.assertEqual(CK.quick_check(cap), [])
        self.assertFalse(CK.deep_check(cap))
        (md / "extra.bin").write_bytes(b"x")
        self.assertIn("not in manifest extra.bin", CK.quick_check(cap))

    def test_hf_snapshot_revision(self):
        d = T.tmpdir()
        snap = d / "models--google--gemma-4-E4B-it" / "snapshots" / "0123abc"
        snap.mkdir(parents=True)
        self.assertEqual(CK.hf_revision(snap), {"source": "hf cache snapshot path", "repo": "google/gemma-4-E4B-it", "revision": "0123abc"})


class Deployment(unittest.TestCase):
    def test_git_and_plain_capture_snapshot(self):
        d = T.tmpdir()
        src = d / "brian"
        (src / "pkg").mkdir(parents=True)
        (src / "pkg" / "edge.py").write_text("x = 1\n")
        (src / "run.sh").write_text("python edge.py --token SECRET123\n")
        (src / "weights.bin").write_bytes(b"0" * 10)
        plain = DP.source_capture(src)
        self.assertIsNone(plain["git"])
        self.assertEqual(sorted(e["path"] for e in plain["manifest"]), ["pkg/edge.py", "run.sh"])
        T.git(src, "init", "-q")
        T.git(src, "config", "user.email", "t@e.invalid")
        T.git(src, "config", "user.name", "t")
        T.git(src, "add", "pkg/edge.py", "run.sh")
        T.git(src, "commit", "-q", "-m", "x")
        (src / "pkg" / "new.py").write_text("y = 2\n")
        g = DP.source_capture(src)
        self.assertTrue(g["git"]["dirty"])
        self.assertIn("pkg/new.py", [e["path"] for e in g["manifest"]])
        self.assertEqual(sorted(e["path"] for e in DP.source_capture(src / "pkg")["manifest"]), ["edge.py", "new.py"])
        self.assertNotIn("SECRET123", DP.launch_capture(src / "run.sh")["text_redacted"])
        rec = DP.snapshot(g, d / "snap")
        self.assertEqual(DP.verify_snapshot(g, d / "snap"), [])
        with self.assertRaises(FileExistsError):
            DP.snapshot(g, d / "snap")
        (src / "pkg" / "edge.py").write_text("x = 2\n")
        with self.assertRaises(RuntimeError):
            DP.snapshot(g, d / "snap2")
        self.assertEqual(rec["manifest_sha256"], g["manifest_sha256"])
        (d / "snap" / "pkg" / "edge.py").write_text("tampered\n")
        self.assertEqual(DP.verify_snapshot(g, d / "snap"), ["pkg/edge.py"])
        (d / "snap" / "pkg" / "planted.py").write_text("x = 3\n")
        self.assertIn("extra: pkg/planted.py", DP.verify_snapshot(g, d / "snap"))
        rows = DP.parse_ps("  10   1 brian 55 python serve.py --api-key K\n  11 1 root 3 bash\n")
        self.assertEqual(len(rows), 1)
        self.assertNotIn(" K", rows[0]["args"])


class Split(unittest.TestCase):
    def setUp(self):
        self.summ = CK.summarize_config(T.TINY_CONFIG)

    def test_valid_double_capture_and_differences(self):
        cap = T.double_split_capture()
        self.assertEqual(SP.validate(cap, self.summ)[0], [])
        d = SP.derive(cap, self.summ)
        self.assertEqual(d["topology"], "prefix_edge_suffix_server")
        self.assertTrue(d["per_layer_inputs_cross"] and d["kv_crosses"] and d["generated_tokens_on_server"] and d["server_holds_embedding_table"])
        diff = " | ".join(d["differences_from_qwen3_e1opt"])
        for must in ("topology prefix_edge_suffix_server", "LM head on the server", "sampling on the server", "per-layer input",
                     "KV entries", "generated tokens", "token-embedding table", "keeps no KV cache", "float32"):
            self.assertIn(must, diff)

    def test_u_shape_like_qwen3(self):
        cap = T.double_split_capture()
        cap["ownership"]["edge"].update(layers=[[0, 2], [6, 8]], final_norm=True, lm_head=True, sampling=True)
        cap["ownership"]["server"].update(layers=[[2, 6]], final_norm=False, lm_head=False, sampling=False)
        cap["boundary"] = [b for b in cap["boundary"] if b["direction"] == "edge_to_server" and "shared_kv" not in b["name"]]
        cap["boundary"] += [{"direction": "server_to_edge", "name": "hidden_states", "dtype": "bfloat16", "shape": "[1,n,h]", "when": "every_step"}]
        cap["model"].update(dtype_edge="bfloat16", dtype_server="bfloat16")
        summ = CK.summarize_config(dict(T.TINY_CONFIG, num_kv_shared_layers=0))
        self.assertEqual(SP.validate(cap, summ)[0], [])
        d = SP.derive(cap, summ)
        self.assertEqual(d["topology"], "u_shape")
        self.assertFalse(d["generated_tokens_on_server"])
        self.assertNotIn("LM head on the server (E1-OPT: edge)", d["differences_from_qwen3_e1opt"])

    def test_invalid_captures(self):
        def probs(mut, summ=None):
            cap = T.double_split_capture()
            mut(cap)
            return " | ".join(SP.validate(cap, summ or self.summ)[0])
        self.assertIn("owned by no host", probs(lambda c: c["ownership"]["server"].update(layers=[[4, 8]])))
        self.assertIn("owned by both", probs(lambda c: c["ownership"]["server"].update(layers=[[2, 8]])))
        self.assertIn("per-layer input embeddings", probs(lambda c: c.update(boundary=[b for b in c["boundary"] if b["name"] != "per_layer_inputs"])))
        self.assertIn("KV-sharing", probs(lambda c: (c.update(boundary=[b for b in c["boundary"] if "shared_kv" not in b["name"]]),
                                                     c["kv_cache"].update(shared_kv_handling=""))))
        self.assertIn("unknowns remain", probs(lambda c: c.update(unknowns=["endpoint?"])))
        self.assertIn("config.json says 8", probs(lambda c: c["model"].update(num_hidden_layers=9)))
        self.assertIn("greedy", probs(lambda c: c["generation"].update(greedy_supported=False)))
        self.assertIn("must be true or false", probs(lambda c: c["generation"].update(streaming_supported="yes")))
        self.assertIn("must be true or false", probs(lambda c: c["ownership"]["edge"].update(lm_head="FILL: true | false")))
        self.assertIn("without a recorded reason", probs(lambda c: c["ownership"]["server"].update(tokenizer=True)))
        self.assertIn("uuids", probs(lambda c: c["gpu_placement"]["server"].update(uuids=[])))
        self.assertIn("missing top-level", probs(lambda c: c.pop("boundary")))

    def test_templates_never_validate(self):
        tpl = C.read_json(T.GS / "capture" / "split_capture.template.json")
        P = " | ".join(SP.validate(tpl, self.summ)[0])
        self.assertIn("placeholders", P)
        self.assertIn("unknowns remain", P)
        sel = SP.select(C.read_json(T.GS / "capture" / "candidates.template.json")["candidates"], T.roles_for())
        self.assertEqual(sel["verdict"], "STOP_NO_CONFIGURATION")
        self.assertIn("placeholders", " ".join(sel["trace"][0]["excluded_because"]))
        self.assertTrue(G.capability_problems(C.read_json(T.GS / "capture" / "capabilities.template.json"), "cfg"))

    def test_boundary_audit(self):
        cap = T.double_split_capture()
        obs = [{"direction": "edge_to_server", "tensors": [{"name": "hidden_states", "dtype": "float32", "shape": [1, 4, 32]},
                                                          {"name": "input_ids", "dtype": "int64", "shape": [1, 4]},
                                                          {"name": "shared_kv_full_attention_k", "dtype": "float32", "shape": [1, 4, 16]}], "bytes": 10},
               {"direction": "server_to_edge", "tensors": [{"name": "next_token", "dtype": "int32", "shape": [1]}], "bytes": 5}]
        a = SP.audit_boundary(cap, obs)
        P = " | ".join(a["problems"])
        self.assertIn("undeclared boundary tensor edge_to_server:input_ids", P)
        self.assertIn("int32", P)
        self.assertIn("shared_kv_full_attention_k has 3 dimensions", P)                      # rank differs from [1, kv, n, d]
        self.assertIn("declared edge_to_server:per_layer_inputs (every_step) never crossed", P)
        self.assertIn("edge_to_server:per_layer_inputs", a["declared_never_observed"])

    def test_selection_rule(self):
        roles, c = T.roles_for(), T.candidates
        self.assertEqual(SP.select([c("b"), c("a")], roles)["selected"], "a")                                       # R4
        r = SP.select([c("b", implementation_default=True), c("a")], roles)
        self.assertEqual((r["selected"], r["rule_applied"][0][:2]), ("b", "R3"))
        self.assertEqual(SP.select([c("b", implementation_default=True), c("c", running_at_capture=True)], roles)["selected"], "c")   # R2 > R3
        r = SP.select([c("c", running_at_capture=True), c("z", designated_by_implementer=True, designation_ref="lab_answers.md#q2")], roles)
        self.assertEqual(r["selected"], "z")                                                                       # R1 first
        self.assertEqual(SP.select([c("z", designated_by_implementer=True), c("y")], roles)["selected"], "y")     # undocumented designation
        r = SP.select([dict(c("llama"), model_dir_name="Llama-3.1-8B"), c("w", edge_host="other-box")], roles)
        self.assertEqual(r["verdict"], "STOP_NO_CONFIGURATION")
        self.assertEqual(r["performance_inputs"], "none (PREREG §4.5)")


class Workload(unittest.TestCase):
    def test_build_reproduces_committed_files(self):
        sample = REPO / S.WORKLOAD_SOURCE
        if not sample.exists():
            self.skipTest("S1 sample not present")
        d = T.tmpdir()
        WL.build(sample, d)
        for f in (WL.CORR_FILE, WL.TIMING_FILE):
            self.assertEqual(C.sha256_file(d / f), C.sha256_file(T.GS / "workload" / f), f)

    def test_spec_structure(self):
        corr, timing = WL.load_spec(T.GS / "workload")
        self.assertEqual(len(corr["items"]), 30)
        self.assertEqual({k: sum(1 for i in corr["items"] if i["dataset"] == k) for k in S.CORR_PER_DATASET}, S.CORR_PER_DATASET)
        self.assertFalse({i["key"] for i in corr["items"]} & {p["key"] for p in timing["pool"]})
        self.assertEqual(len(timing["pool"]), 170)
        for g, P in S.TIMING_GROUPS.items():
            self.assertEqual(sum(1 for i in timing["items"] if i["group"] == g and i["kind"] == "measured"), S.TIMING_MEASURED)
            self.assertTrue(all(i["prompt_tokens"] == P for i in timing["items"] if i["group"] == g))
        self.assertEqual(sum(1 for i in timing["items"] if i["set"] == "coverage"), S.COVERAGE_ITEMS)
        self.assertEqual(len({i["start"] for i in timing["items"]}), len(timing["items"]))
        for it in corr["items"]:
            self.assertEqual(C.sha256_bytes(it["prompt"].encode()), it["prompt_sha256"])

    def test_validate_tokens(self):
        w = T.synthetic_tokens(T.GS)
        self.assertEqual(WL.validate_tokens(w, T.GS / "workload"), [])
        self.assertEqual((w["max_total_tokens"], w["probe_item"]), (2048 + 256, "cov-00"))
        bad = json.loads(json.dumps(w))
        it = next(i for i in bad["items"] if i["id"] == "long-m00")
        it["ids"].pop()
        it["n"] -= 1
        self.assertTrue(any("long-m00" in p for p in WL.validate_tokens(WL.finalize(bad), T.GS / "workload")))
        self.assertTrue(WL.validate_tokens(dict(w, ids_sha256="0"), T.GS / "workload", dev=True))


class AdapterAndResults(unittest.TestCase):
    def test_adapter_info_checks(self):
        info = AD.read_info(T.GS / "tests" / "doubles" / "gemma4_split_double.py")
        self.assertTrue(info["test_double"])
        self.assertIn("test double", " ".join(AD.check_info(info, target=True)))
        self.assertEqual(AD.check_info(info, target=False), [])
        no_tf = dict(info, capabilities=[c for c in info["capabilities"] if c != "teacher_forced_logprobs"], test_double=False)
        self.assertTrue(any("justification" in p for p in AD.check_info(no_tf)))
        no_tf["missing_capability_justification"] = {"teacher_forced_logprobs": "opaque engine exposes no logits"}
        self.assertEqual(AD.check_info(no_tf), [])
        self.assertEqual(AD.tier(no_tf), S.TIER_TOKENS)
        self.assertTrue(any("required" in p for p in AD.check_info(dict(no_tf, capabilities=["greedy_tokens"]))))

    def test_provenance_and_timing_guard(self):
        fr = T.fake_freeze()
        p = R.provenance(fr, "stage3", "edge", ["cuda:0"], "target", host=T.EDGE_HOST)
        self.assertEqual(R.check_provenance(p, fr, "stage3", "edge"), [])
        self.assertTrue(R.check_provenance(dict(p, host="elsewhere"), fr, "stage3", "edge"))
        self.assertTrue(R.check_provenance(dict(p, freeze_sha256="0"), fr, "stage3"))
        self.assertTrue(R.check_provenance(dict(p, profile_role="dev"), fr, "stage3"))
        self.assertEqual(R.check_provenance(dict(p, profile_role="dev"), fr, "stage3", require_target=False), [])
        self.assertTrue(R.check_provenance(None, fr, "stage3"))
        for k in ("ttft_ns", "latency_ms", "duration", "tokens_per_s", "elapsed", "wall_clock", "tpot", "step_time", "throughput"):
            with self.assertRaises(R.TimingFieldError, msg=k):
                R.assert_no_timing_fields({"rows": [{"ok": 1, k: 2}]})
        R.assert_no_timing_fields({"agree": [1], "kl": [0.0], "margins": [], "utc": "x", "started_utc": "x", "tokens": [], "n_positions": 3,
                                   "stopped_on_eos": 1, "status": "FITS", "max_allocated_bytes": 1, "process_peak_rss_bytes": 2, "num_ooms": 0})
        self.assertEqual(R.run_dir("/x", fr), Path("/x/runs/ffffffffffff"))
        self.assertEqual(R.expected("stage4", ["step_trace"]), ["edge/trace.jsonl", "edge/meta.json", "edge/server_serve.json"])
        self.assertIn("server_only_native/requests.jsonl", R.expected("stage5", ["unsplit_same_runtime"]))
        self.assertIn("split/server_serve.json", R.expected("stage3", []))


class Stats(unittest.TestCase):
    def test_quantile_bootstrap_categories(self):
        self.assertEqual(ST.quantile([1, 2, 3, 4], 0.25), 1.75)
        self.assertEqual(ST.median([3, 1, 2]), 2.0)
        pairs = {"short": [(2.0 + 0.01 * i, 1.0 + 0.01 * i) for i in range(20)], "long": [(4.0 + 0.01 * i, 2.0) for i in range(20)]}
        a = ST.paired_bootstrap_ratio(pairs, n_boot=500)
        self.assertEqual(a, ST.paired_bootstrap_ratio(pairs, n_boot=500))
        self.assertLessEqual(a["ci"][0], a["point"])
        self.assertGreaterEqual(a["ci"][1], a["point"])
        self.assertEqual(ST.efficiency_category(a["ci"]), "SPLIT_SLOWER_THAN_EDGE")
        self.assertEqual(ST.efficiency_category([0.5, 0.9]), "SPLIT_FASTER_THAN_EDGE")
        self.assertEqual(ST.efficiency_category([0.9, 1.0]), "NOT_DISTINGUISHABLE_FROM_EDGE")
        self.assertIsNone(ST.efficiency_category([None, None]))


class CorrectnessRules(unittest.TestCase):
    def test_divergence_degenerate_eos(self):
        self.assertIsNone(RU.first_divergence([1, 2, 3], [1, 2, 3]))
        self.assertEqual(RU.first_divergence([1, 2, 3], [1, 9, 3]), 1)
        self.assertEqual(RU.first_divergence([1, 2], [1, 2, 3]), 2)
        self.assertEqual(RU.strip_eos([4, 5, 1], {1}), [4, 5])
        self.assertEqual(RU.strip_eos([4, 5], {1}), [4, 5])
        self.assertEqual(RU.is_degenerate([5] * 250, 100)[1], "period 1")
        self.assertEqual(RU.is_degenerate([1, 200], 100)[1], "invalid token id")
        self.assertFalse(RU.is_degenerate(list(range(250)), 1000)[0])

    def test_tf_rule_hardware_floor_only(self):
        hw = {"D_pp": 0.15, "KL": 0.006}
        r = RU.tf_rule({"D_pp": 0.30, "KL": 0.010}, hw)
        self.assertTrue(r["pass"])
        self.assertEqual(r["floor"], "hardware")
        self.assertAlmostEqual(r["D_threshold_pp"], 0.35)
        self.assertFalse(RU.tf_rule({"D_pp": 0.36, "KL": 0.0}, hw)["pass"])
        self.assertFalse(RU.tf_rule({"D_pp": 0.0, "KL": 0.013}, hw)["pass"])
        self.assertFalse(RU.tf_rule({"D_pp": 0.0, "KL": 2e-6}, {"D_pp": 0.0, "KL": 0.0})["pass"])           # eps rule
        self.assertFalse(RU.tf_rule({"D_pp": 0.0, "KL": float("inf")}, {"D_pp": 0.0, "KL": 1.0})["pass"])
        # a defect shared by the split and the deployment's own in-host pipeline (D 6 pp, KL 0.08) is not absorbed:
        self.assertFalse(RU.tf_rule({"D_pp": 6.0, "KL": 0.08}, {"D_pp": 0.1, "KL": 0.001})["pass"])
        self.assertEqual(RU.tf_summary([{"agree": [1, 1, 0, 1], "kl": [0.0, 0.0, 0.4, 0.0]}])["D_pp"], 25.0)

    def test_gen_rule(self):
        ref = [{"id": f"i{k}", "tokens": [1, 2, 3, 4], "margins": [2.0, 0.49, 0.5, 3.0]} for k in range(5)]
        same = [{"id": f"i{k}", "tokens": [1, 2, 3, 4]} for k in range(5)]
        self.assertTrue(RU.gen_rule(ref, same, 100, 5, [99])["pass"])
        near = [dict(x, tokens=[1, 9, 9, 9]) for x in same]                       # diverge where margin 0.49 < 0.5: explained
        self.assertTrue(RU.gen_rule(ref, near, 100, 5, [99])["pass"])
        at05 = [dict(x, tokens=[1, 2, 9, 9]) for x in same]                       # margin exactly 0.5: unexplained
        g = RU.gen_rule(ref, at05, 100, 5, [99])
        self.assertEqual((g["unexplained"], g["pass"]), (5, False))
        self.assertTrue(RU.gen_rule(ref, at05[:3] + same[3:], 100, 5, [99])["pass"])
        self.assertFalse(RU.gen_rule(ref, same[:4], 100, 5, [99])["pass"])        # an item missing
        trunc = [dict(same[0], tokens=[1, 2], eos_truncated=True)] + same[1:]
        self.assertTrue(RU.gen_rule(ref, trunc, 100, 5, [99])["pass"])
        nan = [dict(same[0], nan=True)] + same[1:]
        self.assertEqual(RU.gen_rule(ref, nan, 100, 5, [99])["degenerate_only_split"], ["i0"])
        # EOS convention: the reference keeps the stopping EOS (id 4 here), an implementation may strip it
        stripped = [dict(x, tokens=[1, 2, 3]) for x in same]
        self.assertEqual(RU.gen_rule(ref, stripped, 100, 5, [4])["unexplained"], 0)
        self.assertEqual(RU.gen_rule(ref, stripped, 100, 5, [99])["unexplained"], 5)   # without the EOS rule: a false FAIL
        audit = RU.aggregate_audit([{"direction": "edge_to_server", "tensors": [{"name": "h", "dtype": "f", "shape": [1, n]}], "bytes": 4}
                                    for n in range(3)])
        self.assertEqual((len(audit), audit[0]["count"], audit[0]["bytes"]), (1, 3, 12))


class MemoryRules(unittest.TestCase):
    def test_classify_and_status(self):
        self.assertEqual(RU.classify("RuntimeError: CUDA out of memory. Tried to allocate"), "oom_device")
        self.assertEqual(RU.classify("torch.OutOfMemoryError: x"), "oom_device")
        self.assertEqual(RU.classify("OSError: [Errno 12] Cannot allocate memory"), "oom_host")
        self.assertEqual(RU.classify("OSError: [Errno 28] No space left on device"), "other")    # /dev/shm full: not infeasibility
        self.assertEqual(RU.classify("KeyError: full_attention"), "other")
        self.assertEqual(RU.status_of(None, [], []), "FITS")
        self.assertEqual(RU.status_of(None, [], [], ooms=2), "FITS")                               # recovered inside the deployment
        self.assertEqual(RU.status_of("CUDA out of memory", [], []), "INFEASIBLE")
        self.assertEqual(RU.status_of("Cannot allocate memory", [], []), "INVALID")                # host memory on a discrete-GPU host
        self.assertEqual(RU.status_of("Cannot allocate memory", [], [], unified=True), "INFEASIBLE")   # ... on the DGX Spark
        self.assertEqual(RU.status_of("ConnectionResetError", [], [], ooms=1), "INFEASIBLE")
        self.assertEqual(RU.status_of("CUDA out of memory", [], [{"pid": 1}]), "INVALID")
        self.assertEqual(RU.status_of(None, [{"pid": 1}], []), "BLOCKED_CONTENTION")
        self.assertEqual(RU.combine_split({"status": "BLOCKED_CONTENTION"}, {"status": "INFEASIBLE"})[0], "INFEASIBLE")
        self.assertEqual(RU.combine_split({"status": "INVALID"}, {"status": "FITS", "ooms": 1})[0], "INFEASIBLE")   # server OOM seen by the edge
        self.assertEqual(RU.combine_split({"status": "INVALID"}, {"status": "FITS", "ooms": 0})[0], "INVALID")
        self.assertEqual(RU.combine_split({"status": "FITS"}, {"status": "FITS"})[0], "FITS")
        snap = {"apps": [{"pid": 999999, "gpu_uuid": "GPU-a6000-0000"}, {"pid": 1, "gpu_uuid": "GPU-other"}]}
        self.assertEqual(len(M.foreign_processes(snap, ["GPU-a6000-0000"], mine={os.getpid()})), 1)
        self.assertEqual(M.oom_events({"torch": {"cuda:0": {"num_ooms": 1}}}, {"torch": {"cuda:0": {"num_ooms": 3}}}), 2)


def write_stage5(rd, fr, tp, drift=1.0, fail=False, foreign=False, blocks=None, skew=0.0, log=True):
    """Synthetic Stage-5 raw files as gs_profile.run_block and gs_bench serve write them (and their exit-status events)."""
    sd = R.stage_dir(rd, "stage5")
    t = 1_800_000_000.0
    for b in blocks or S.block_order(fr["adapter"]["info"]["capabilities"]):
        role = S.BLOCK_HOST[b]
        host = T.SERVER_HOST if role == "server" else T.EDGE_HOST
        rows = []
        for g in S.TIMING_GROUPS:
            for i in range(20):
                v = tp[b] * (drift if b == "split_B" else 1.0) * (1 + 0.001 * i)
                rows.append({"group": g, "prompt_id": f"{g}-m{i:02d}", "kind": "measured", "n_tokens": 256, "ttft_ns": 10 * v, "tpot_ns": v,
                             "total_ns": 300 * v, "tokens_sha256": "same", **({"error": "boom"} if fail and b == "edge_only" and i == 3 else {})})
        C.write_jsonl(sd / b / "requests.jsonl", rows)
        cont = {"utc": "x", "foreign": [{"pid": 1}] if foreign and b == "split_B" else [], "smi_available": True}
        mode = "dev" if fr.get("dev") else "target"
        C.write_json(sd / b / "meta.json", {"runtime": "reference runtime" if b in ("edge_only", "server_only") else "deployment",
                                            "started_utc": C.utc_ms(t), "ended_utc": C.utc_ms(t + 100), "contention_before": cont,
                                            "contention_after": cont, "provenance": R.provenance(fr, "stage5", role, ["cpu"], mode, host=host)})
        if b.startswith("split"):
            C.write_json(sd / b / "server_meta.json", {"started_utc": C.utc_ms(t - 10 + skew), "ended_utc": C.utc_ms(t + 110 + skew),
                                                       "contention_before": cont, "contention_after": cont, "failure": None,
                                                       "provenance": R.provenance(fr, "stage5", "server", ["cpu"], mode, host=T.SERVER_HOST)})
        t += 200
    if log:
        T.log_runs(rd, fr, "stage5")


def open_gates(rd, fr, upto=("stage2", "stage3", "stage4")):
    verdict = {"stage2": "CONTINUE", "stage3": "PASS", "stage4": "COMPLETE"}
    for st in upto:
        G.write_gate(rd, fr, st, {"verdict": verdict[st], "final": False, "repeatable": [], "reasons": []})


class Runtime(unittest.TestCase):
    comps = {"mode": "isolated", "decode_step_ns": {"edge": {"short": 30.0, "long": 30.0}, "server": {"short": 15.0, "long": 15.0}}}

    def test_e2e_summary(self):
        fr = T.fake_freeze()
        rd = T.tmpdir()
        write_stage5(rd, fr, TP)
        s = RU.summarize_e2e(rd, fr, self.comps)
        self.assertEqual(s["verdict"], "COMPLETE", s["problems"])
        self.assertEqual(s["primary"]["category"], "SPLIT_SLOWER_THAN_EDGE")
        self.assertAlmostEqual(s["primary"]["point"], 1.5, places=6)
        dec = s["decomposition"]["per_group"]["short"]
        self.assertAlmostEqual(dec["transport_and_overhead_ns"], 60.0 * 1.0095 - 45.0, delta=0.5)
        self.assertTrue(dec["closure_ok"])
        self.assertEqual(s["token_agreement"]["split_A~split_B"], {"n": 40, "identical": 40})
        rd = T.tmpdir()
        write_stage5(rd, fr, dict(TP, split_A=30.0, split_B=30.0))
        s = RU.summarize_e2e(rd, fr, self.comps)
        self.assertEqual(s["primary"]["category"], "SPLIT_FASTER_THAN_EDGE")
        self.assertFalse(s["decomposition"]["per_group"]["long"]["closure_ok"])
        rd = T.tmpdir()
        write_stage5(rd, fr, TP, drift=1.2)
        self.assertEqual(RU.summarize_e2e(rd, fr, self.comps)["verdict"], "UNSTABLE")
        for kw, cause in ((dict(fail=True), "failed_requests"), (dict(foreign=True), "contention"), (dict(skew=200.0), "invalid")):
            rd = T.tmpdir()
            write_stage5(rd, fr, TP, **kw)
            s = RU.summarize_e2e(rd, fr, self.comps)
            self.assertEqual(s["verdict"], "INVALID", kw)
            self.assertIn(cause, [p["cause"] for p in s["problems"]], kw)
        rd = T.tmpdir()
        write_stage5(rd, fr, TP, skew=20.0)                                  # within the 30 s clock tolerance
        self.assertEqual(RU.summarize_e2e(rd, fr, self.comps)["verdict"], "COMPLETE")
        rd = T.tmpdir()
        write_stage5(rd, fr, TP)
        m = C.read_json(R.stage_dir(rd, "stage5") / "edge_only" / "meta.json")
        C.write_json(R.stage_dir(rd, "stage5") / "edge_only" / "meta.json", dict(m, runtime="deployment runtime, unsplit"), overwrite=True)
        s = RU.summarize_e2e(rd, fr, self.comps)
        self.assertIn("must run the reference runtime", " ".join(p["reason"] for p in s["problems"]))
        self.assertIsNotNone(s["primary"]["point"])                                       # an INVALID attempt is still summarised
        rd = T.tmpdir()
        write_stage5(rd, fr, TP, blocks=["split_A", "server_only", "edge_only", "split_B"])   # executed out of order
        s = RU.summarize_e2e(rd, fr, self.comps)
        self.assertIn("frozen order", " ".join(p["reason"] for p in s["problems"]))

    def test_native_blocks_are_secondary(self):
        fr = T.fake_freeze(caps=tuple(T.FAKE_CAPS) + ("unsplit_same_runtime",))
        rd = T.tmpdir()
        write_stage5(rd, fr, dict(TP, edge_only_native=50.0, server_only_native=25.0))
        s = RU.summarize_e2e(rd, fr, self.comps)
        self.assertEqual(s["verdict"], "COMPLETE", s["problems"])
        self.assertAlmostEqual(s["primary"]["point"], 1.5, places=6)                       # primary stays the reference-runtime edge_only
        self.assertIn("tpot_split_over_edge_only_native", s["secondary"])


class Gates(unittest.TestCase):
    def test_require_mode_integrity(self):
        fr, dv = T.fake_freeze(), T.fake_freeze(dev=True)
        rd = T.tmpdir()
        with self.assertRaises(G.GateClosed) as cm:
            G.require(rd, fr, "stage3")
        self.assertIn("copy it here", str(cm.exception))
        G.write_gate(rd, fr, "stage2", {"verdict": "CONTINUE", "final": False, "repeatable": []})
        G.require(rd, fr, "stage3")
        g = C.read_json(G.gate_path(rd, "stage2"))
        C.write_json(G.gate_path(rd, "stage2"), dict(g, verdict="CONTINUE ", reasons=["edited"]), overwrite=True)
        with self.assertRaises(G.GateClosed):
            G.require(rd, fr, "stage3")                                         # hand-edited gate
        rd2 = T.tmpdir()
        G.write_gate(rd2, dict(dv, freeze_sha256=fr["freeze_sha256"], run_id=fr["run_id"]), "stage2", {"verdict": "CONTINUE", "final": False,
                                                                                                       "repeatable": []})
        with self.assertRaises(G.GateClosed) as cm:
            G.require(rd2, fr, "stage3")                                        # a dev gate never opens a target stage
        self.assertIn("dev gate", str(cm.exception))
        rd3 = T.tmpdir()
        G.write_gate(rd3, fr, "stage3", {"verdict": "FAIL", "final": True, "repeatable": []})
        with self.assertRaises(G.GateClosed):
            G.require(rd3, fr, "stage4")

    def test_outcome_recomputes_stored_gates(self):
        fr = T.fake_freeze()
        rd = T.tmpdir()
        G.write_gate(rd, fr, "stage2", {"verdict": "STOP_INFEASIBLE", "final": True, "repeatable": []})   # a gate with no raw files behind it
        o = G.outcome(rd, fr)
        self.assertIsNone(o["outcome"])
        self.assertIn("does not match a recomputation", " ".join(o["problems"]))
        self.assertEqual(G.OUTCOME_OF[("stage3", "FAIL")], "NOT_EQUIVALENT")

    def test_stage2_confirmation_and_contention(self):
        fr = T.fake_freeze(dev=True)
        state = {}

        def fresh():
            state["rd"] = T.tmpdir()
            return state["rd"]

        def records(edge_split, server_split, edge_ref="FITS", server_ref="FITS", server_ooms=0):
            sd = R.stage_dir(state["rd"], "stage2")
            for role, what, st in (("edge", "split", edge_split), ("server", "split", server_split), ("edge", "reference", edge_ref),
                                   ("server", "reference", server_ref)):
                C.write_json(sd / role / f"{what}_memory.json", T.memory_record(fr, role, what, st, ooms=server_ooms if role == "server" else 0))
            T.log_runs(state["rd"], fr, "stage2")
        rd = fresh()
        records("FITS", "INFEASIBLE")
        b = G.compute_stage2(rd, fr, dev=True)
        self.assertEqual((b["verdict"], b["final"]), ("INFEASIBLE_UNCONFIRMED", False))
        G.write_gate(rd, fr, "stage2", b)
        G.invalidate(rd, fr, "stage2", [r["part"] for r in b["repeatable"]], "confirm")
        records("FITS", "INFEASIBLE")
        b2 = G.gate_stage2(rd, fr, dev=True)
        self.assertEqual((b2["verdict"], b2["final"]), ("STOP_INFEASIBLE", True))
        with self.assertRaises(ValueError):
            G.invalidate(rd, fr, "stage2", ["edge", "server"], "try again")      # final: never repeatable
        rd = fresh()
        records("INVALID", "FITS", server_ooms=2)                               # the edge's failure was the server running out of memory
        self.assertEqual(G.compute_stage2(rd, fr, dev=True)["verdict"], "INFEASIBLE_UNCONFIRMED")
        rd = fresh()
        records("BLOCKED_CONTENTION", "FITS")
        b = G.compute_stage2(rd, fr, dev=True)
        self.assertEqual((b["verdict"], b["repeatable"][0]["cap"]), ("BLOCKED_CONTENTION", None))
        rd = fresh()
        records("FITS", "FITS", server_ref="INFEASIBLE")
        G.write_gate(rd, fr, "stage2", G.compute_stage2(rd, fr, dev=True))
        G.invalidate(rd, fr, "stage2", ["edge", "server"], "confirm reference")
        records("FITS", "FITS", server_ref="INFEASIBLE")
        b = G.gate_stage2(rd, fr, dev=True)
        self.assertEqual((b["verdict"], b["reference_feasible"]), ("CONTINUE", False))   # Stage 3 will be NOT_EVALUABLE

    def test_stale_attempt_is_refused(self):
        fr = T.fake_freeze(dev=True)
        rd = T.tmpdir()
        sd = R.stage_dir(rd, "stage2")
        old = {}
        for role in S.ROLES:
            for what in ("split", "reference"):
                rec = T.memory_record(fr, role, what, "INVALID" if (role, what) == ("edge", "split") else "FITS")
                rec["provenance"]["utc"] = "2020-01-01T00:00:00Z"
                old[(role, what)] = rec
                C.write_json(sd / role / f"{what}_memory.json", rec)
        T.log_runs(rd, fr, "stage2")
        g = G.gate_stage2(rd, fr, dev=True)
        self.assertEqual(g["verdict"], "INVALID")
        G.invalidate(rd, fr, "stage2", ["edge", "server"], "repeat")
        for (role, what), rec in old.items():                                  # an rsync brings the old attempt back
            C.write_json(sd / role / f"{what}_memory.json", dict(rec, status="FITS"))
        T.log_runs(rd, fr, "stage2")
        with self.assertRaises(G.GateNotReady) as cm:                         # never a repeat (no cap, no redraw): sync the fresh file
            G.compute_stage2(rd, fr, dev=True)
        self.assertIn("stale", str(cm.exception))
        self.assertEqual(len(G.invalidation_log(rd, "stage2")), 2)            # nothing else was moved or logged

    def test_retire_on_a_host(self):
        fr = T.fake_freeze()
        desk, host = T.tmpdir(), T.tmpdir()
        for rd in (desk, host):
            open_gates(rd, fr)
            write_stage5(rd, fr, TP, foreign=True)
        g = G.gate_stage5(desk, fr)
        G.invalidate(desk, fr, "stage5", [r["part"] for r in g["repeatable"]], "contention")
        log = R.stage_dir(desk, "stage5") / "_invalidated" / "LOG.jsonl"
        (R.stage_dir(host, "stage5") / "_invalidated").mkdir(parents=True)
        (R.stage_dir(host, "stage5") / "_invalidated" / "LOG.jsonl").write_bytes(log.read_bytes())
        moved = G.retire(host, "stage5")
        self.assertEqual(len(moved), 4)
        self.assertFalse((R.stage_dir(host, "stage5") / "split_A").exists())
        self.assertEqual(G.retire(host, "stage5"), [])

    def test_stage5_repeats_cover_the_sequence_and_are_capped(self):
        fr = T.fake_freeze()
        rd = T.tmpdir()
        open_gates(rd, fr)
        write_stage5(rd, fr, TP, drift=1.3)
        g = G.gate_stage5(rd, fr)
        self.assertEqual((g["verdict"], g["final"], len(g["repeatable"])), ("UNSTABLE", False, 4))
        with self.assertRaises(ValueError):
            G.invalidate(rd, fr, "stage5", ["split_B"], "only the block I dislike")     # never a single block
        G.invalidate(rd, fr, "stage5", [r["part"] for r in g["repeatable"]], "repeat once for drift")
        self.assertFalse(G.gate_path(rd, "stage5").exists())
        write_stage5(rd, fr, TP, drift=1.3)
        g = G.gate_stage5(rd, fr)
        self.assertEqual((g["verdict"], g["final"], g["repeatable"]), ("UNSTABLE", True, []))
        with self.assertRaises(ValueError) as cm:
            G.invalidate(rd, fr, "stage5", ["split_A"], "please")
        self.assertIn("final", str(cm.exception))

    def test_failed_request_cap_counts_events_not_text(self):
        fr = T.fake_freeze()
        rd = T.tmpdir()
        open_gates(rd, fr)
        write_stage5(rd, fr, TP, fail=True)
        for attempt in range(3):
            g = G.gate_stage5(rd, fr)
            if attempt < 2:
                self.assertEqual(g["verdict"], "INVALID")
                self.assertEqual({r["part"] for r in g["repeatable"]}, set(S.BLOCK_ORDER))        # the whole sequence
                self.assertEqual({r["cause"] for r in g["repeatable"]}, {"failed_requests"})
                G.invalidate(rd, fr, "stage5", [r["part"] for r in g["repeatable"]], ["reworded note", "another wording", ""][attempt])
                write_stage5(rd, fr, TP, fail=True)
        self.assertEqual((g["verdict"], g["final"], g.get("unstable_cause")), ("UNSTABLE", True, "request_failures"))
        rep = G.repeats(rd, fr)
        self.assertEqual([e["causes"] for e in rep], [["failed_requests"], ["failed_requests"]])
        self.assertTrue(all(e["would_have_shown"] in S.EFFICIENCY_CATEGORIES for e in rep))   # invalidated attempts stay visible

    def test_caps_count_across_the_record_chain(self):
        root = T.tmpdir()
        gs = root / "diagnostics" / "deployment_gemma_split"
        old = T.fake_freeze()
        old.update(revision=1)
        new = dict(T.fake_freeze(), freeze_sha256="9" * 64, run_id="9" * 12, revision=2,
                   supersedes={"freeze_sha256": old["freeze_sha256"], "run_id": old["run_id"],
                               "counts_as_repeat": {"stage": "stage5", "entries": [{"part": "split_A", "cause": "failed_requests"}]}})
        C.write_json(gs / "frozen" / "freeze_record.json", dict(old, experiment=S.NAME))
        rd_old, rd_new = R.run_dir(gs, old), R.run_dir(gs, new)
        for rd, fr in ((rd_old, old), (rd_new, new)):
            open_gates(rd, fr)
        write_stage5(rd_old, old, TP, fail=True)
        g = G.gate_stage5(rd_old, old)                                         # one failed-request repeat in the OLD run ...
        G.invalidate(rd_old, old, "stage5", [r["part"] for r in g["repeatable"]], "x")
        self.assertEqual(G.chain_run_dirs(rd_new, new), [rd_old])
        self.assertEqual([e["moved_to"] for e in G.supersede_events(rd_new, new, "stage5")], ["supersede:" + old["run_id"]])
        write_stage5(rd_new, new, TP, fail=True)
        g = G.gate_stage5(rd_new, new)                                         # ... plus the supersede: the cap (2) is reached
        self.assertEqual((g["verdict"], g["final"]), ("UNSTABLE", True))

    def test_crash_accounting(self):
        """Crashes, deleted outputs and re-runs are read from the exit-status logs; nothing is repeated without a gate."""
        fr = T.fake_freeze()

        def fresh():
            rd = T.tmpdir()
            open_gates(rd, fr)
            return rd
        rd = fresh()                                                           # split_A crashed while measuring: own files incomplete
        write_stage5(rd, fr, TP, blocks=["split_A"], log=False)
        (R.stage_dir(rd, "stage5") / "split_A" / "meta.json").unlink()
        T.log_run(rd, "stage5", "split_A", "serve")
        T.log_run(rd, "stage5", "split_A", "e2e", rc=None)
        g = G.gate_stage5(rd, fr)                                              # no other block has run: the sequence is repeated
        self.assertEqual((g["verdict"], g["final"], sorted(r["part"] for r in g["repeatable"])), ("INVALID", False, sorted(S.BLOCK_ORDER)))
        G.invalidate(rd, fr, "stage5", list(S.BLOCK_ORDER), "edge crashed")
        self.assertEqual(len(G.consumed(rd, "stage5", "split_A")), 2)          # the crashed run and its server half are accounted for
        write_stage5(rd, fr, TP)
        self.assertEqual(G.gate_stage5(rd, fr)["verdict"], "COMPLETE")
        rd = fresh()                                                           # outputs deleted and the block re-run
        write_stage5(rd, fr, TP)
        T.log_run(rd, "stage5", "edge_only", "e2e", out="0" * 16)
        g = G.gate_stage5(rd, fr)
        self.assertEqual((g["verdict"], g["final"]), ("INCOMPLETE", True))
        self.assertIn("deleted", " ".join(g["reasons"]))
        self.assertEqual((g["would_be"]["verdict"], g["would_be"]["category"]), ("COMPLETE", "SPLIT_SLOWER_THAN_EDGE"))  # still reported
        self.assertIn("(present)", " ".join(g["reasons"]))
        self.assertIn("(not present)", " ".join(g["reasons"]))                 # which attempt's outputs survived
        with self.assertRaises(ValueError):
            G.invalidate(rd, fr, "stage5", list(S.BLOCK_ORDER), "try again")
        rd = fresh()                                                           # a file of a completed run is missing: copy it
        write_stage5(rd, fr, TP)
        (R.stage_dir(rd, "stage5") / "split_A" / "server_meta.json").unlink()
        T.log_run(rd, "stage5", "split_A", "serve", rc="1")                   # a failed relaunch does not turn it into a crash
        with self.assertRaises(G.GateNotReady) as cm:
            G.gate_stage5(rd, fr)
        self.assertIn("copy them", str(cm.exception))
        rd = fresh()                                                           # a killed run, then a re-run without a repeat in between
        T.log_run(rd, "stage5", "edge_only", "e2e", rc=None)
        write_stage5(rd, fr, TP)
        g = G.gate_stage5(rd, fr)
        self.assertEqual((g["verdict"], g["final"]), ("INVALID", False))
        self.assertIn("preceded the completed one", " ".join(g["reasons"]))
        rd = fresh()                                                           # a refusal before measuring is not an attempt
        T.log_run(rd, "stage5", "edge_only", "e2e", rc="3", run=False)
        write_stage5(rd, fr, TP)
        self.assertEqual(G.gate_stage5(rd, fr)["verdict"], "COMPLETE")
        rd = fresh()                                                           # logs not copied yet
        write_stage5(rd, fr, TP, log=False)
        with self.assertRaises(G.GateNotReady) as cm:
            G.gate_stage5(rd, fr)
        self.assertIn("exit-status log", str(cm.exception))
        self.assertFalse(hasattr(G.invalidate, "incomplete") or "incomplete" in G.invalidate.__code__.co_varnames)

    def test_stage0_verdicts(self):
        root = T.tmpdir()
        gs = T.copy_gs(root)
        self.assertEqual(G.stage0(gs / "capture")["verdict"], "STOP_INCOMPLETE")
        src = root / "src"
        src.mkdir()
        (src / "a.py").write_text("x=1\n")
        cap = T.write_target_capture(gs, root / "models", src)
        v = C.read_json(cap / "stage0_verdict.json")
        self.assertEqual(v["verdict"], "CONTINUE", v["reasons"])
        self.assertIn("topology prefix_edge_suffix_server (E1-OPT: u_shape, edge = first k and last k layers)", v["derived"]["differences_from_qwen3_e1opt"])

        def verdict_with(name, value):
            orig = C.read_json(cap / name)
            C.write_json(cap / name, value(orig), overwrite=True)
            out = G.stage0(cap)
            C.write_json(cap / name, orig, overwrite=True)
            return out
        self.assertEqual(verdict_with("roles.json", lambda r: dict(r, verified=False, problems=["x"]))["verdict"], "STOP_ROLE_MISMATCH")
        unsynced = verdict_with("edge/inventory.json", lambda i: dict(i, clock={"timedatectl": "NTPSynchronized=no", "chrony": ""}))
        self.assertEqual(unsynced["verdict"], "STOP_INCOMPLETE")
        self.assertIn("clock synchronisation", " ".join(unsynced["reasons"]))
        self.assertEqual(verdict_with("edge/checkpoint.json", lambda c: dict(c, manifest=[dict(e, sha256="0" * 64) for e in c["manifest"]]))["verdict"],
                         "STOP_CHECKPOINT_MISMATCH")
        self.assertEqual(verdict_with("split_capture.json", lambda s: dict(s, unknowns=["endpoint"]))["verdict"], "STOP_INCOMPLETE")
        self.assertEqual(verdict_with("split_capture.json", lambda s: dict(s, boundary=s["boundary"][:1]))["verdict"], "STOP_SPLIT_CAPTURE_INVALID")
        self.assertEqual(verdict_with("split_capture.json", lambda s: dict(s, configuration_id="other"))["verdict"], "STOP_SPLIT_CAPTURE_INVALID")
        alien = verdict_with("split_capture.json", lambda s: json.loads(json.dumps(s).replace('"GPU-a6000-0000"', '"GPU-not-here"')))
        self.assertIn("not GPUs of the verified server host", " ".join(alien["reasons"]))
        v = verdict_with("capabilities.json", lambda c: dict(c, capabilities=["greedy_tokens", "stream_tokens", "boundary_audit"]))
        self.assertEqual(v["verdict"], "STOP_SPLIT_CAPTURE_INVALID")
        self.assertIn("justification", " ".join(v["reasons"]))

    def test_block_order(self):
        import gs_bench as B
        rd = T.tmpdir()
        caps = T.FAKE_CAPS
        with self.assertRaises(B.UsageError):
            B.check_local_order(rd, "edge_only", "edge", caps)                 # split_A has not run on the edge
        with self.assertRaises(B.UsageError):
            B.check_local_order(rd, "server_only", "server", caps)             # nor its server half
        C.write_jsonl(R.stage_dir(rd, "stage5") / "split_A" / "requests.jsonl", [])
        B.check_local_order(rd, "edge_only", "edge", caps)
        C.write_json(R.stage_dir(rd, "stage5") / "split_A" / "server_meta.json", {})
        B.check_local_order(rd, "server_only", "server", caps)
        with self.assertRaises(B.UsageError):
            B.check_local_order(rd, "split_B", "edge", caps)                   # edge_only missing on the edge
        with self.assertRaises(B.UsageError):
            B.check_local_order(rd, "edge_only_native", "edge", caps)          # not declared: not in this run's order


class Archive(unittest.TestCase):
    def test_reproducible_all_runs_and_verified(self):
        root = T.tmpdir()
        gs = T.copy_gs(root)
        C.write_json(gs / "runs" / "abc" / "final" / "outcome.json", {"outcome": "COMPLETE"})
        C.write_json(gs / "runs" / "old" / "stage2_memory" / "gate.json", {"verdict": "INVALID"})
        a = AR.create(gs, "abc", root / "a.tgz")
        b = AR.create(gs, "abc", root / "b.tgz")
        self.assertEqual(a["sha256"], b["sha256"])
        self.assertIn("runs/old/stage2_memory/gate.json", AR.members(gs, "abc"))           # superseded runs are archived too
        self.assertEqual(AR.verify(root / "a.tgz"), [])
        with self.assertRaises(FileExistsError):
            AR.create(gs, "abc", root / "a.tgz")
        raw = bytearray((root / "b.tgz").read_bytes())
        raw[len(raw) // 2] ^= 0xFF
        (root / "b.tgz").write_bytes(bytes(raw))
        try:
            P = AR.verify(root / "b.tgz")
        except Exception as e:      # a corrupted gzip stream may not even open
            P = [str(e)]
        self.assertTrue(P)
        with self.assertRaises(FileNotFoundError):
            AR.create(gs, "nope", root / "c.tgz")


class Freeze(unittest.TestCase):
    """Target-mode freeze writer / checker in a temporary git repository (tag, commit A0, commit A, record, commit B)."""

    def setUp(self):
        self.root = T.tmpdir()
        self.gs = T.git_repo_with_gs(self.root)
        src = self.root / "brian"
        src.mkdir()
        (src / "edge.py").write_text("x=1\n")
        T.write_target_capture(self.gs, self.root / "models", src, adapter=False)
        self.commit("commit A0: Stage-0 capture incl. the capability declaration")
        T.write_fake_adapter(self.gs)
        self.commit("commit A: adapter")
        self.rec = F.build(self.gs, "adapter/gs_adapter_fake.py")
        self.path = self.gs / "frozen" / "freeze_record.json"
        F.write(self.rec, self.path)
        self.commit("commit B: freeze record")

    def commit(self, msg):
        T.git(self.root, "add", "-A")
        T.git(self.root, "commit", "-q", "--allow-empty", "-m", msg)

    def check(self, path=None, **kw):
        kw.setdefault("role", "server")
        kw.setdefault("host", T.SERVER_HOST)
        kw.setdefault("live_versions", dict(T.VERSIONS))
        kw.setdefault("live_env", {})
        return F.check(path or self.path, gs_dir=self.gs, **kw)

    def problems(self, path=None, **kw):
        try:
            self.check(path, **kw)
            return ""
        except F.FreezeMismatch as e:
            return str(e)

    def build_fails(self, needle, **kw):
        with self.assertRaises(F.FreezeMismatch) as cm:
            F.build(self.gs, "adapter/gs_adapter_fake.py", **kw)
        self.assertIn(needle, str(cm.exception))

    def test_record_contents_and_pass(self):
        r = self.rec
        self.assertEqual((r["run_id"], r["revision"], r["supersedes"]), (r["freeze_sha256"][:12], 1, None))
        for k in ("PREREG_GEMMA_SPLIT.md", "RUNBOOK.md", "README.md", "gs_spec.py", "gs_rules.py", "adapter/gs_adapter_fake.py",
                  "capture/split_capture.json", "capture/capabilities.json", "capture/workload_tokens.json", "workload/correctness_prompts.json",
                  "tests/doubles/gemma4_split_double.py", "capture/edge/inventory.json"):
            self.assertIn(k, r["file_sha256"])
        self.assertFalse([f for f in r["file_sha256"] if f.startswith(("frozen/", "runs/"))])
        self.assertEqual(r["harness_changes_since_tag"], [])
        self.assertEqual(r["split"]["derived"]["topology"], "prefix_edge_suffix_server")
        self.assertEqual(r["machines"]["edge"]["host"], T.EDGE_HOST)
        self.assertEqual(r["adapter"]["tier"], "T1")
        self.check()
        self.check(role="edge", host=T.EDGE_HOST)

    def test_refusals(self):
        self.assertIn("not the frozen server host", self.problems(host="intruder"))
        self.assertIn("not the frozen edge host", self.problems(role="edge"))
        self.assertIn("transformers", self.problems(live_versions=dict(T.VERSIONS, transformers="4.57.0")))
        self.assertIn("vllm", self.problems(live_versions=dict(T.VERSIONS, vllm="0.11.0")))
        self.assertIn("environment CUDA_VISIBLE_DEVICES", self.problems(live_env={"CUDA_VISIBLE_DEVICES": "2"}))
        self.assertIn("environment LD_PRELOAD", self.problems(live_env={"LD_PRELOAD": "/tmp/libevil.so"}))
        self.assertIn("dev mode cannot use it", self.problems(dev=True))
        with self.assertRaises(FileExistsError):
            F.write(self.rec, self.path)
        with self.assertRaises(F.FreezeMismatch):
            F.write(self.rec, self.gs / "frozen" / "record_b.json")                 # records have fixed names
        p2 = self.gs / "frozen" / "tampered.json"
        p2.write_text(json.dumps(dict(self.rec, notes="edited")))
        with self.assertRaises(F.FreezeMismatch):
            F.load(p2)
        self.assertIn("non-standard name", self.problems())                        # a record-shaped file under another name
        p2.unlink()

    def test_stray_bytecode_symlink_uncommitted(self):
        (self.gs / "helpers.py").write_text("x=1\n")                      # not gs_*: still importable, so still covered
        self.assertIn("file not covered by the freeze: helpers.py", self.problems())
        (self.gs / "helpers.py").unlink()
        (self.gs / "__pycache__").mkdir()
        (self.gs / "__pycache__" / "gs_rules.cpython-312.pyc").write_bytes(b"\0" * 16)
        self.assertIn("bytecode in the experiment directory", self.problems())
        (self.gs / "__pycache__" / "gs_rules.cpython-312.pyc").unlink()
        (self.gs / "__pycache__").rmdir()
        os.symlink(self.root / "brian", self.gs / "adapter" / "brian")
        self.assertIn("symlink in the experiment directory: adapter/brian", self.problems())
        (self.gs / "adapter" / "brian").unlink()
        self.check()
        text = self.path.read_text()
        self.path.write_text(text.replace('"notes": ""', '"notes": ""  '))
        self.assertIn("not committed unchanged", self.problems())
        self.path.write_text(text)

    def test_decision_files_pinned_even_if_documented(self):
        for name in ("PREREG_GEMMA_SPLIT.md", "gs_spec.py", "gs_rules.py", "gs_gate.py", "gs_reference.py", "gs_profile.py"):
            p = self.gs / name
            orig = p.read_text()
            p.write_text(orig + "\n# changed\n")
            (self.gs / "frozen" / "HARNESS_FIXES.md").write_text(f"{name} | abc123 | test_x | tweak\n")
            self.assertIn(f"{name} differs from the tag (a decision file", self.problems())
            p.write_text(orig)
            (self.gs / "frozen" / "HARNESS_FIXES.md").unlink()

    def test_executing_harness_must_be_the_frozen_one(self):
        """A record built from an edited copy cannot drive the (unedited) executing harness: the --gs-dir bypass."""
        p = self.gs / "gs_bench.py"
        p.write_text(p.read_text() + "\n# edited copy\n")
        rec = F.build(self.gs, "adapter/gs_adapter_fake.py", supersede_reason="test")   # the first run produced nothing
        p2 = self.gs / "frozen" / "freeze_record.r2.json"
        F.write(rec, p2)
        self.assertIn("executing harness file", self.problems(p2))

    def test_post_freeze_commits_and_moved_tag(self):
        C.write_json(self.gs / "runs" / self.rec["run_id"] / "stage2_memory" / "x.json", {"a": 1})
        self.commit("results")
        self.check()
        (self.gs / "capture" / "lab_answers.md").write_text("late edit\n")
        self.commit("late capture edit")
        msg = self.problems()
        self.assertIn("file not covered by the freeze: capture/lab_answers.md", msg)
        self.assertIn("outside", msg)
        T.git(self.root, "tag", "-f", S.PREREG_TAG, "HEAD")
        self.assertIn("the tag moved", self.problems())

    def test_capabilities_must_be_declared_first_and_match(self):
        C.write_json(self.gs / "capture" / "capabilities.json", T.capabilities(T.FAKE_CAPS + ["ignore_eos"]), overwrite=True)
        self.build_fails("differ from the Stage-0 declaration", supersede_reason="r")
        self.commit("capabilities changed after the adapter existed")
        self.build_fails("changed after it was first committed", supersede_reason="r")
        root = T.tmpdir()
        gs = T.git_repo_with_gs(root)
        src = root / "s"
        src.mkdir()
        (src / "a.py").write_text("1\n")
        T.write_target_capture(gs, root / "m", src)                             # declaration and adapter in one commit
        T.git(root, "add", "-A")
        T.git(root, "commit", "-q", "-m", "capture + adapter together")
        with self.assertRaises(F.FreezeMismatch) as cm:
            F.build(gs, "adapter/gs_adapter_fake.py")
        self.assertIn("first committed together with or after adapter files", str(cm.exception))

    def test_inventory_environment_matches_the_deployment(self):
        sc = C.read_json(self.gs / "capture" / "split_capture.json")
        sc["gpu_placement"]["server"]["CUDA_VISIBLE_DEVICES"] = "1"
        C.write_json(self.gs / "capture" / "split_capture.json", sc, overwrite=True)
        self.build_fails("collect the inventory in the deployment's environment", supersede_reason="r")

    def test_supersede_rule(self):
        rd = R.run_dir(self.gs, self.rec)
        self.build_fails("--supersede-reason")                                  # always state why
        T.stage2_raw(rd, self.rec, status="INVALID")
        self.build_fails("no gate", supersede_reason="r")                       # gate what was measured first
        g = G.gate_stage2(rd, self.rec)
        self.assertEqual((g["verdict"], g["supersede_eligible"]), ("INVALID", False))
        self.build_fails("only when the previous run produced nothing", supersede_reason="non-memory failure again")   # operational
        forged = dict(g, supersede_eligible=True)
        forged["gate_sha256"] = G.gate_hash(forged)                             # a hand-made gate with a valid hash ...
        C.write_json(G.gate_path(rd, "stage2"), forged, overwrite=True)
        self.build_fails("does not match a recomputation", supersede_reason="pretend")   # ... does not follow from the raw files
        G.gate_path(rd, "stage2").unlink()                                      # (test shortcuts: nothing was committed)
        T.stage2_raw(rd, self.rec)
        self.assertEqual(G.gate_stage2(rd, self.rec)["verdict"], "CONTINUE")
        T.write_fake_adapter(self.gs, test_double=False)
        (self.gs / "adapter" / "gs_adapter_fake.py").write_text((self.gs / "adapter" / "gs_adapter_fake.py").read_text() + "# faster path\n")
        self.build_fails("only orchestration files", supersede_reason="adapter tweak after Stage 2")        # not an orchestration fix
        T.write_fake_adapter(self.gs)
        G.gate_path(rd, "stage2").unlink()
        T.stage2_raw(rd, self.rec, match=("server",))                          # the edge's tokenizer does not reproduce the ids
        g = G.gate_stage2(rd, self.rec)
        self.assertEqual((g["verdict"], g["supersede_eligible"]), ("INVALID", True))
        (self.gs / "tests" / "doubles" / "gemma4_split_double.py").write_text(
            (self.gs / "tests" / "doubles" / "gemma4_split_double.py").read_text() + "\n# fix H1\n")
        (self.gs / "frozen" / "HARNESS_FIXES.md").write_text("tests/doubles/gemma4_split_double.py | def456 | test_h1 | tokenizer path\n")
        self.commit("documented fix after an eligible INVALID")
        rec2 = F.build(self.gs, "adapter/gs_adapter_fake.py", supersede_reason="workload mismatch on the edge: capture corrected")
        self.assertEqual((rec2["revision"], rec2["supersedes"]["run_id"], rec2["supersedes"]["latest_gate"]), (2, self.rec["run_id"], ["stage2", "INVALID"]))
        self.assertEqual(rec2["supersedes"]["counts_as_repeat"], {"stage": "stage2", "entries": [{"part": "edge", "cause": "workload_mismatch"}]})
        p2 = self.gs / "frozen" / "freeze_record.r2.json"
        F.write(rec2, p2)
        self.commit("record r2")
        self.check(p2)
        self.assertIn("superseded by freeze_record.r2.json", self.problems())  # the old record can no longer drive anything
        hist = F.history(self.gs, rec2)
        self.assertEqual((len(hist), hist[0]["superseded_because"]), (1, "workload mismatch on the edge: capture corrected"))
        rd2 = R.run_dir(self.gs, rec2)
        G.write_gate(rd2, rec2, "stage2", {"verdict": "STOP_INFEASIBLE", "final": True, "repeatable": []})
        self.commit("r2 final")
        self.build_fails("final verdict", supersede_reason="try once more")    # the experiment is over

    def test_adapter_fix_is_documented_and_archived(self):
        import gs_archive as A
        rd = R.run_dir(self.gs, self.rec)
        T.stage2_raw(rd, self.rec, match=("server",))
        self.assertTrue(G.gate_stage2(rd, self.rec)["supersede_eligible"])
        a = self.gs / "adapter" / "gs_adapter_fake.py"
        a.write_text(a.read_text() + "# adapter fix\n")
        self.commit("adapter changed after an eligible INVALID")
        self.build_fails("adapter files changed since freeze_record.json without an entry", supersede_reason="adapter fix")
        (self.gs / "frozen" / "HARNESS_FIXES.md").write_text("adapter/gs_adapter_fake.py | abc123 | test_fix | wrong tensor order\n")
        self.commit("documented")
        rec2 = F.build(self.gs, "adapter/gs_adapter_fake.py", supersede_reason="adapter fix")
        self.assertIn("adapter/gs_adapter_fake.py", rec2["supersedes"]["changed_files"])
        F.write(rec2, self.gs / "frozen" / "freeze_record.r2.json")
        self.commit("record r2")
        hist = F.history(self.gs, rec2)
        self.assertEqual((hist[0]["adapter_sha256"], hist[0]["changed_in_successor"]), (self.rec["adapter"]["sha256"], rec2["supersedes"]["changed_files"]))
        old = A.adapter_history(self.gs)
        self.assertEqual(sorted(old), sorted([f"adapter_history/{self.rec['run_id']}/gs_adapter_fake.py",
                                              f"adapter_history/{rec2['run_id']}/gs_adapter_fake.py"]))   # both versions archived, hashes match

    def test_crash_basis_needs_committed_evidence(self):
        """Basis (c), a stage that could not run: only with committed crash evidence and no completed command (PREREG §6.4)."""
        rd = R.run_dir(self.gs, self.rec)
        T.stage2_raw(rd, self.rec, hosts=False)                                 # events logged under this machine's name only
        self.assertEqual(G.gate_stage2(rd, self.rec)["verdict"], "CONTINUE")
        self.commit("stage 2 gated")
        self.build_fails("must be copied here and committed unchanged", supersede_reason="serve crashes at start-up")
        for host in (T.EDGE_HOST, T.SERVER_HOST):
            R.exit_log(rd, "log copied", host=host)
        self.commit("both hosts' logs, no crash")
        self.build_fails("no stage3 command crashed while measuring", supersede_reason="serve crashes at start-up")
        T.log_run(rd, "stage3", "split", "correctness", rc=None, host=T.EDGE_HOST)
        self.build_fails("committed unchanged", supersede_reason="serve crashes at start-up")       # the crash must be committed
        self.commit("crash evidence")
        rec2 = F.build(self.gs, "adapter/gs_adapter_fake.py", supersede_reason="serve crashes at start-up")
        self.assertEqual(rec2["supersedes"]["counts_as_repeat"], {"stage": "stage3", "entries": [{"part": "split", "cause": "invalid"}]})
        T.log_run(rd, "stage3", "reference_server", "reference", host=T.SERVER_HOST)
        self.commit("a stage-3 command had completed")
        self.build_fails("completed stage3 commands", supersede_reason="serve crashes at start-up")   # outputs exist (or were deleted)

    def test_no_output_basis_checks_the_logs(self):
        rd = R.run_dir(self.gs, self.rec)
        T.log_run(rd, "stage2", "edge", "memory:split", host=T.EDGE_HOST)      # completed, but its output is gone
        self.build_fails("outputs were deleted", supersede_reason="environment changed before any output")

    def test_append_only_logs(self):
        rd = R.run_dir(self.gs, self.rec)
        log = rd / f"{T.EDGE_HOST}.exit_status.log"
        T.log_run(rd, "stage2", "edge", "memory:split", host=T.EDGE_HOST)
        self.commit("log v1")
        T.log_run(rd, "stage2", "edge", "memory:reference", host=T.EDGE_HOST)
        self.commit("log v2 (appended)")
        self.check()
        log.write_text("\n".join(log.read_text().splitlines()[3:]) + "\n")     # the first run's lines removed
        self.assertIn("append-only log rewritten in the working tree", self.problems())
        self.commit("rewrite the log")
        self.assertIn("append-only log rewritten in git history", self.problems())

    def test_adapter_must_not_name_the_live_tree(self):
        a = self.gs / "adapter" / "gs_adapter_fake.py"
        a.write_text(a.read_text() + f"\nLIVE = {str((self.root / 'brian').resolve())!r}\n")
        self.build_fails("names the live deployment tree", supersede_reason="r")

    def test_revision_cap(self):
        for n in (2, 3):
            rec = F.build(self.gs, "adapter/gs_adapter_fake.py", supersede_reason=f"environment changed before any output ({n})")
            F.write(rec, self.gs / "frozen" / F.expected_record_name(n))
            self.commit(f"record r{n}")
        self.build_fails(f"at most {S.MAX_RECORD_REVISIONS} freeze records", supersede_reason="again")

    def test_evidence_cannot_be_removed(self):
        rd = R.run_dir(self.gs, self.rec)
        G.write_gate(rd, self.rec, "stage2", {"verdict": "STOP_INFEASIBLE", "final": True, "repeatable": []})
        self.commit("final gate committed")
        T.git(self.root, "rm", "-q", "-r", str(rd.relative_to(self.root)))
        self.commit("delete the run")
        self.build_fails("deleted from git history", supersede_reason="fresh start")
        root = T.tmpdir()
        gs = T.git_repo_with_gs(root)
        src = root / "s"
        src.mkdir()
        (src / "a.py").write_text("1\n")
        T.write_target_capture(gs, root / "m", src, adapter=False)
        T.git(root, "add", "-A")
        T.git(root, "commit", "-q", "-m", "A0")
        T.write_fake_adapter(gs)
        T.git(root, "add", "-A")
        T.git(root, "commit", "-q", "-m", "A")
        rec = F.build(gs, "adapter/gs_adapter_fake.py")
        F.write(rec, gs / "frozen" / "freeze_record.json")
        T.git(root, "add", "-A")
        T.git(root, "commit", "-q", "-m", "B")
        T.git(root, "mv", str((gs / "frozen" / "freeze_record.json").relative_to(root)), str((gs / "frozen" / "old_record.json").relative_to(root)))
        T.git(root, "commit", "-q", "-m", "hide the record")
        with self.assertRaises(F.FreezeMismatch) as cm:
            F.build(gs, "adapter/gs_adapter_fake.py", supersede_reason="fresh start")
        self.assertTrue("renamed in git history" in str(cm.exception) or "non-standard name" in str(cm.exception))

    def test_dev_records(self):
        root = T.tmpdir()
        gs = T.git_repo_with_gs(root)
        src = root / "s"
        src.mkdir()
        (src / "a.py").write_text("1\n")
        T.write_target_capture(gs, root / "m", src, test_double=True)
        with self.assertRaises(F.FreezeMismatch) as cm:
            F.build(gs, "adapter/gs_adapter_fake.py")
        self.assertIn("test double", str(cm.exception))
        dev = F.build(gs, "adapter/gs_adapter_fake.py", dev=True)
        p = gs / "frozen" / "freeze_record.json"
        F.write(dev, p)
        with self.assertRaises(F.FreezeMismatch) as cm:
            F.check(p, gs_dir=gs)
        self.assertIn("dev record", str(cm.exception))                     # a dev record never drives a target run
        F.check(p, gs_dir=gs, dev=True)
        root2 = T.tmpdir()
        gs2 = T.git_repo_with_gs(root2)
        T.write_target_capture(gs2, root2 / "m", src, test_double=False)
        with self.assertRaises(F.FreezeMismatch) as cm:
            F.build(gs2, "adapter/gs_adapter_fake.py", dev=True)
        self.assertIn("test-double adapters only", str(cm.exception))      # the deployment's adapter never runs in dev mode


class LiveTree(unittest.TestCase):
    def test_guard_refuses_the_live_tree(self):
        import importlib.util
        import subprocess
        import gs_bench as B
        root = T.tmpdir()
        live, model, snap = root / "brian", root / "brian" / "models" / "m", root / "snap"
        model.mkdir(parents=True)
        snap.mkdir()
        (live / "gs_livemod_probe.py").write_text("X = 1\n")
        (model / "config.json").write_text("{}")
        (snap / "ok.txt").write_text("ok")
        guard = B.LiveTreeGuard([live], [model, snap]).install()
        try:
            sys.path.insert(0, str(live))
            with self.assertRaises(ImportError):
                import gs_livemod_probe  # noqa: F401
            sys.path.remove(str(live))
            spec = importlib.util.spec_from_file_location("gs_x_probe", str(live / "gs_livemod_probe.py"))
            for fn in (lambda: open(live / "gs_livemod_probe.py").read(),
                       lambda: spec.loader.exec_module(importlib.util.module_from_spec(spec)),
                       lambda: subprocess.run(["cat", str(live / "gs_livemod_probe.py")], capture_output=True),
                       lambda: os.system(f"cat {live}/gs_livemod_probe.py"),
                       lambda: os.chdir(live)):
                with self.assertRaises(PermissionError):
                    fn()
            self.assertEqual(open(model / "config.json").read(), "{}")            # the hash-verified checkpoint stays readable
            self.assertEqual(open(snap / "ok.txt").read(), "ok")
            self.assertEqual(subprocess.run(["cat", str(snap / "ok.txt")], capture_output=True, text=True).stdout, "ok")
        finally:
            B.LiveTreeGuard.ACTIVE.clear()
            if guard in sys.meta_path:
                sys.meta_path.remove(guard)
            if str(live) in sys.path:
                sys.path.remove(str(live))
        self.assertEqual(open(live / "gs_livemod_probe.py").read(), "X = 1\n")    # disabled again
        gs = root / "gs"                                                       # the guard a target command builds from the capture
        launch = root / "start_split.sh"
        launch.write_text("#!/bin/sh\n")
        C.write_json(gs / "capture" / "edge" / "deployment.json", {"sources": [{"root": str(live)}], "launch_files": [{"path": str(launch)}]})
        C.write_json(gs / "capture" / "edge" / "checkpoint.json", {"model_dir": str(model)})
        guard = B.live_guard(gs, ["edge"], [{"sources": [{"snapshot_root": str(snap)}]}])
        try:
            with self.assertRaises(PermissionError):
                open(launch).read()                                            # a launch script outside the roots is guarded too
            self.assertEqual(open(model / "config.json").read(), "{}")
            allowed = [str(x) for x in guard.allowed]
            self.assertIn(str(Path(sys.prefix).resolve()), allowed)           # the interpreter's environment (a venv inside the root)
        finally:
            B.LiveTreeGuard.ACTIVE.clear()
            if guard in sys.meta_path:
                sys.meta_path.remove(guard)


if __name__ == "__main__":
    unittest.main(verbosity=2)
