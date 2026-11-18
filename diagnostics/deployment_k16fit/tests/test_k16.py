"""K16-HWFIT CPU tests. No GPU, no network beyond loopback, no real checkpoint.

  <python with torch + transformers 4.51.3> -I tests/test_k16.py            # all tests
  ... -I tests/test_k16.py --quick                                         # skip the two-process end-to-end runs

The model is a random 64-layer Qwen3 (hidden 64) with a sharpened LM head, so the k = 16 cut is the real layer
allocation (edge 0-15 and 48-63, server 16-47) and the check set has decisive positions. Mocks cover inventories
only; every number compared here is computed by the code under test.
"""
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import k16_spec as S  # noqa: E402
import k16_memory as KM  # noqa: E402
import k16_checkpoint as KC  # noqa: E402
import k16_preflight as PF  # noqa: E402
import k16_criteria as CR  # noqa: E402
from k16_common import read_json, read_jsonl, gate_path, EXIT_STOP, EXIT_GATE  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import k16_runtime as R  # noqa: E402

PY = [sys.executable, "-I"]
STRICT = os.environ.get("K16_TEST_STRICT", "1") != "0"     # bitwise assertions; the host launchers turn them off (other CPUs, other BLAS)
TMP = Path(tempfile.mkdtemp(prefix="k16test_"))
_TINY = {}


def tiny_dir():
    if "d" not in _TINY:
        from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.manual_seed(0)
        cfg = Qwen3Config(torch_dtype=torch.bfloat16, vocab_size=211, hidden_size=64, intermediate_size=128, num_hidden_layers=64,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=32768,
                          tie_word_embeddings=False, rope_theta=10000.0, rms_norm_eps=1e-6)
        m = Qwen3ForCausalLM(cfg).to(torch.bfloat16).eval()
        m.lm_head.weight.data.mul_(40.0)            # peaked next-token distributions: the check set has decisive positions
        d = TMP / "tiny64"
        m.save_pretrained(d, safe_serialization=True)
        _TINY["d"] = str(d)
    return _TINY["d"]


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close()
    return p


class InProcess:
    """Reference + server thread + edge slice for the tiny model, all on CPU, over loopback TCP."""

    def __init__(self, sabotage=False):
        self.p, self.h = S.params(True), S.spec_hash(True)
        d = tiny_dir()
        R.setup_torch()
        self.prompts = S.check_prompts(True, 211)
        self.ref = TMP / f"ref_{int(sabotage)}"
        hf = R.hf_model(d, "cpu")
        self.ref_rows = R.write_reference(hf, self.prompts, self.p, "cpu", self.ref)
        self.side = R.ServerSide(d, "cpu", self.p, None, with_static=True)
        if sabotage:
            self.side.segs["prefix"].layer_ids = list(reversed(self.side.segs["prefix"].layer_ids))
        self.sl = R.edge_slice(d, S.K, S.J, "cpu", torch.bfloat16, None)
        self.sock = socket.socket(); self.sock.bind(("127.0.0.1", 0)); self.sock.listen(1)
        self.events, self.stop = [], threading.Event()
        self.th = threading.Thread(target=R.serve, args=(self.side, self.sock, self.h, self.events.append), kwargs={"stop": self.stop}, daemon=True)
        self.th.start()
        self.addr = f"127.0.0.1:{self.sock.getsockname()[1]}"

    def close(self):
        self.stop.set(); self.th.join(10); self.sock.close()


# --------------------------------------------------------------------------- #
def test_partition_is_k16():
    assert (S.K, S.J, S.N_LAYERS) == (16, 16, 64)
    assert S.EDGE_LAYERS == list(range(0, 16)) + list(range(48, 64)) and S.SERVER_LAYERS == list(range(16, 48))
    assert not set(S.EDGE_LAYERS) & set(S.SERVER_LAYERS) and sorted(S.EDGE_LAYERS + S.SERVER_LAYERS) == list(range(64))
    assert S.PRIMARY_MODE == "prefix" and S.DEFAULT_PORT != 29500
    assert S.spec_hash() != S.spec_hash(dev=True) and len(S.spec_hash()) == 64
    ps = S.check_prompts()
    assert [p["n_in"] for p in ps] == [256] * 5 + [2048] * 5 and max(p["n_in"] for p in ps) + S.LATENCY["gen_tokens"] <= S.MAX_LEN


def test_cut_and_criterion_are_unchanged():
    """The correction of 2026-11-18 (loading, paths, transport) must not move the cut or the correctness rule."""
    assert hashlib.sha256(S.canonical(S.CRITERIA).encode()).hexdigest() == "4d16bc0a726da897898d84d820e448263f244279dc5db71cdab1f24f247df030"
    assert (S.K, S.J) == (16, 16) and S.CRITERIA["id"] == "K16-CORRECT-v1" and S.CRITERIA["fixed_utc"] == "2026-11-18"
    assert S.LISTEN_HOST == "127.0.0.1" and S.MODEL_ID == "Qwen/Qwen3-32B@9216db5781bf21249d130ec9da846c4624c16137"


def test_reused_runtime_is_unmodified():
    repo = HERE.parents[1]
    base = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{S.BASE_TAG}^{{commit}}"], cwd=repo, capture_output=True, text=True)
    if base.returncode != 0:
        return "SKIP: base tag not present in this clone (the spec hash still pins every reused file)"
    r = subprocess.run(["git", "diff", "--quiet", base.stdout.strip(), "--", "diagnostics/deployment_e0", "diagnostics/deployment_e1opt"], cwd=repo)
    assert r.returncode == 0, "deployment_e0 or deployment_e1opt differs from the base tag"


def test_memory_from_actual_tensors():
    t, man = KM.tensor_table(), KC.manifest()
    w = KM.frozen_weights()
    assert len(t) == 707 and w["all"] == 65524246528 == S.E0.QWEN3_32B_TOTAL_BYTES == man["total_tensor_bytes"]
    import e0_memory as M0
    pc = M0.param_counts(S.E0.QWEN3_32B_CONFIG)
    assert w["server"] == 32 * pc["layer"] * 2 and w["edge"] == (32 * pc["layer"] + pc["embed"] + pc["lm_head"] + pc["norm"]) * 2
    srv, edge, ref = KM.expected("server"), KM.expected("edge_twohost"), KM.expected("edge_reference")
    a6000 = 48 * 2**30
    assert srv["required_free"] < a6000 and 30 * 2**30 < srv["total"] < 35 * 2**30, srv
    assert 35 * 2**30 < edge["total"] < 39 * 2**30 and ref["required_free"] < 119 * 2**30, (edge, ref)
    # the E1-OPT blocker: at E0's frozen cut k = 3 the server segment alone exceeds one 48 GiB card
    w3 = {"server": sum(t[n]["nbytes"] for n in KM.role_tensors("server", t, 3, 3)), "edge": 0, "largest": w["largest"]}
    w3["all"] = w["all"]
    assert KM.expected("server", weights=w3, n_server_layers=58, max_len=S.E0.L_F, static=False)["total"] > a6000
    assert tiny_dir() and KM.owner("model.layers.15.mlp.up_proj.weight") == "edge" and KM.owner("model.layers.16.self_attn.q_proj.weight") == "server"
    assert KM.owner("model.layers.47.mlp.up_proj.weight") == "server" and KM.owner("model.layers.48.mlp.up_proj.weight") == "edge"
    assert KM.owner("lm_head.weight") == KM.owner("model.embed_tokens.weight") == KM.owner("model.norm.weight") == "edge"
    lw = KM.local_weights(tiny_dir())
    sl = R.server_slice(tiny_dir(), S.K, S.J, "cpu", torch.bfloat16, None)
    assert lw["server"] == sl.param_bytes and lw["n_tensors"] == 64 * 11 + 3, lw


def test_checkpoint_manifest_and_verify():
    man = KC.manifest()
    assert man["revision"] == S.E0.SPLIT_REVISION and man["files"]["tokenizer.json"]["sha256"] == S.E0.TOKENIZER_JSON_SHA256
    assert len(KC.needed_files("edge")) == 17 + 4 and "tokenizer.json" not in KC.needed_files("server")
    srv_shards = [f for f in KC.needed_files("server") if f.endswith(".safetensors")]
    t = KM.tensor_table()
    assert {t[n]["shard"] for n in KM.role_tensors("server", t)} == set(srv_shards) and 8 <= len(srv_shards) <= 10
    # verify() against a synthetic manifest of the tiny checkpoint: pass, then corrupt / truncate / remove
    d = TMP / "ckpt"; shutil.copytree(tiny_dir(), d)
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    hl, hdr = KM.read_safetensors_header(d / "model.safetensors")
    table = {n: {"shard": "model.safetensors", "dtype": v["dtype"], "shape": v["shape"], "offsets": v["data_offsets"], "nbytes": 0} for n, v in hdr.items()}
    files = {f: {"size": (d / f).stat().st_size, "sha256": KC.sha256_file(d / f)} for f in ("config.json", "model.safetensors.index.json", "model.safetensors")}
    files["model.safetensors"]["header_len"] = hl
    m = {"repo": "tiny", "revision": "x", "files": files}
    ok = KC.verify(d, "server", cache_file=TMP / "cache.json", man=m, table=table)
    assert ok["ok"] and ok["bytes_verified"] > 0 and (TMP / "cache.json").exists(), ok["problems"]
    b = bytearray((d / "model.safetensors").read_bytes()); b[-1] ^= 1; (d / "model.safetensors").write_bytes(bytes(b))
    bad = KC.verify(d, "server", man=m, table=table)
    assert not bad["ok"] and any("sha256" in x for x in bad["problems"]), bad["problems"]
    (d / "model.safetensors").write_bytes(bytes(b[:-2]))
    assert any("size" in x for x in KC.verify(d, "server", man=m, table=table)["problems"])
    (d / "model.safetensors").unlink()
    assert any("missing" in x for x in KC.verify(d, "server", man=m, table=table)["problems"])
    assert KC.locate(str(TMP / "nowhere"))[0] is None
    # the Asus checkpoint location is known and is tried before any search
    assert S.KNOWN_MODEL_DIRS == ["/mnt/mydisk/brian/splitllm-cloud/models/Qwen3-32B/9216db5781bf21249d130ec9da846c4624c16137"]
    saved, env = list(S.KNOWN_MODEL_DIRS), os.environ.pop("K16_MODEL_DIR", None)
    try:
        S.KNOWN_MODEL_DIRS[:] = [str(TMP / "nowhere"), tiny_dir()]
        assert KC.locate()[0] == tiny_dir()
    finally:
        S.KNOWN_MODEL_DIRS[:] = saved
        if env is not None:
            os.environ["K16_MODEL_DIR"] = env


def test_reference_loads_straight_onto_the_device():
    """k16_runtime.hf_model builds the unsplit model without a host copy. It must be the same model, bit for bit,
    as transformers' own from_pretrained, in bfloat16, with nothing left on the meta device."""
    from transformers import AutoModelForCausalLM
    d = tiny_dir()
    a = R.hf_model(d, "cpu")
    b = AutoModelForCausalLM.from_pretrained(d, torch_dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    sa, sb = a.state_dict(), b.state_dict()
    assert type(a) is type(b) and set(sa) == set(sb) and all(torch.equal(sa[k], sb[k]) and sa[k].dtype == torch.bfloat16 for k in sa)
    assert not any(t.is_meta for t in list(a.parameters()) + list(a.buffers())) and not any(p.requires_grad for p in a.parameters())
    assert a.config._attn_implementation == b.config._attn_implementation == "sdpa"
    ids = torch.randint(1, 211, (1, 40), generator=torch.Generator().manual_seed(3))
    with torch.inference_mode():
        assert torch.equal(a.model(input_ids=ids).last_hidden_state, b.model(input_ids=ids).last_hidden_state)
        assert torch.equal(a(input_ids=ids).logits, b(input_ids=ids).logits)
    # a checkpoint that lacks a tensor, or carries another dtype, is refused
    import safetensors.torch as st
    bad = TMP / "bad_ckpt"; shutil.copytree(d, bad)
    sd = st.load_file(str(bad / "model.safetensors"))
    st.save_file({k: v for k, v in sd.items() if k != "model.layers.20.mlp.up_proj.weight"}, str(bad / "model.safetensors"))
    try:
        R.hf_model(str(bad), "cpu"); raise AssertionError("a checkpoint with a missing tensor was accepted")
    except KeyError:
        pass
    st.save_file({k: (v.float() if k == "lm_head.weight" else v) for k, v in sd.items()}, str(bad / "model.safetensors"))
    try:
        R.hf_model(str(bad), "cpu"); raise AssertionError("a non-bf16 checkpoint was accepted")
    except ValueError:
        pass
    # the memory gates count the loading transients: host staging + device copy share one pool on the GB10
    w = KM.frozen_weights()
    assert KM.expected("edge_reference")["load_transient"] == 2 * w["largest"]["edge"] == KM.expected("edge_twohost")["load_transient"]
    assert KM.expected("server")["load_transient"] == w["largest"]["server"] and KM.HOST_RAM_MIN_MIB >= 4096
    lm = R.load_memory("cpu")
    assert lm["host_peak_rss_mib"] > 0 and lm["device_load_transient_mib"] is None


ASUS = ("0, GPU-aaaa, NVIDIA RTX A6000, 49140, 41000, 8140\n1, GPU-bbbb, NVIDIA RTX A6000, 49140, 3, 49137\n"
        "2, GPU-cccc, NVIDIA RTX A6000, 49140, 20000, 29140\n")
ASUS_APPS = "GPU-aaaa, 4242, 40990\nGPU-cccc, 5151, 19990\n"
DGX = "0, GPU-dddd, NVIDIA GB10, [N/A], [N/A], [N/A]\n"


def test_preflight_gpu_selection_and_memory_gate():
    gpus, apps = PF.parse_gpus(ASUS), PF.parse_apps(ASUS_APPS)
    assert len(gpus) == 3 and gpus[1]["free_mib"] == 49137 and apps[0] == {"gpu_uuid": "GPU-aaaa", "pid": 4242, "used_mib": 40990}
    need = KM.required_free_mib("server")
    owners = {4242: {"user": "brian"}, 5151: {"user": "other"}}
    auto = PF.select_gpu(gpus, apps, need, "A6000", owners=owners)
    assert auto["ok"] and auto["gpu"]["index"] == 1 and not auto["gpu"]["occupied"]            # the free card, never an occupied one
    req0 = PF.select_gpu(gpus, apps, need, "A6000", requested="0", owners=owners)
    assert not req0["ok"] and req0["verdict"] == "STOP_MEMORY_INSUFFICIENT" and "brian" in " ".join(req0["reasons"])
    req2 = PF.select_gpu(gpus, apps, 20000, "A6000", requested="GPU-cccc")
    assert not req2["ok"] and req2["verdict"] == "STOP_GPU_OCCUPIED"                              # memory would fit, but it is in use
    assert PF.select_gpu(gpus, apps, 20000, "A6000", requested="GPU-cccc", allow_shared=True)["ok"]
    assert not PF.select_gpu(gpus, apps, 30000, "A6000", requested="GPU-cccc", allow_shared=True)["ok"]   # sharing never overrides memory
    assert PF.select_gpu(gpus, apps, need, "GB10")["verdict"] == "STOP_ROLE_MISMATCH"
    assert PF.select_gpu(gpus, apps, need, "A6000", requested="7")["verdict"] == "STOP_GPU_NOT_FOUND"
    busy = PF.parse_gpus(ASUS.replace("49140, 3, 49137", "49140, 30000, 19140"))
    assert PF.select_gpu(busy, apps, need, "A6000")["verdict"] == "STOP_MEMORY_INSUFFICIENT"
    # GB10: nvidia-smi reports no memory figures; MemAvailable decides
    g = PF.parse_gpus(DGX)
    assert g[0]["total_mib"] is None
    need_e = KM.required_free_mib("edge_reference")
    assert PF.select_gpu(g, [], need_e, "GB10", unified_available_mib=110000)["ok"]
    low = PF.select_gpu(g, [], need_e, "GB10", unified_available_mib=60000)
    assert not low["ok"] and low["verdict"] == "STOP_MEMORY_INSUFFICIENT"
    occ = PF.select_gpu(g, PF.parse_apps("GPU-dddd, 999, [N/A]\n"), need_e, "GB10", unified_available_mib=110000)
    assert not occ["ok"] and occ["verdict"] == "STOP_GPU_OCCUPIED"
    mi = PF.meminfo("MemTotal:       125000000 kB\nMemFree: 1 kB\nMemAvailable:   104857600 kB\n")
    assert mi == {"total_mib": 122070, "available_mib": 102400}
    inv = {"gpus": g, "apps": [], "owners": {}, "meminfo": {"available_mib": 50000}}
    mg = PF.memory_gate("edge_twohost", inv=inv)
    assert mg["ok"] and mg["need_mib"] == KM.required_free_mib("edge_twohost")
    assert not PF.memory_gate("edge_reference", inv=inv)["ok"]


def _teacher(n, margin=2.0, wrong=(), kl=0.0, finite=True):
    agree = np.ones(n, dtype=bool); agree[list(wrong)] = False
    return {"agree": agree, "kl": np.full(n, kl), "ref_margin": np.full(n, margin), "finite": finite}


def test_criteria_rule():
    ids = [f"p{i}" for i in range(10)]
    ref = {i: {"tokens": list(range(1, 65)), "margins": [2.0] * 64} for i in ids}
    same = {i: {"tokens": list(range(1, 65)), "finite": True} for i in ids}
    ok = CR.evaluate(ids, {i: _teacher(400) for i in ids}, ref, same, 1000)
    assert ok["pass"] and ok["teacher_forced"]["n_decisive"] == 4000 and ok["greedy"]["failures_allowed"] == 1
    # C2: 21 decisive disagreements of 4,000 = 0.525 % > 0.5 %; 20 = 0.5 % passes
    t = {i: _teacher(400) for i in ids}; t["p0"] = _teacher(400, wrong=range(21))
    assert not CR.evaluate(ids, t, ref, same, 1000)["checks"]["C2_decisive_agreement"]
    t["p0"] = _teacher(400, wrong=range(20))
    assert CR.evaluate(ids, t, ref, same, 1000)["pass"]
    # disagreements at near ties never count against C2
    t = {i: _teacher(400) for i in ids}; t["p1"] = _teacher(400, margin=0.1, wrong=range(300))
    r = CR.evaluate(ids, t, ref, same, 1000)
    assert r["checks"]["C2_decisive_agreement"] and abs(r["teacher_forced"]["all_position_disagreement_pp"] - 7.5) < 1e-9
    # C3
    assert not CR.evaluate(ids, {i: _teacher(400, kl=0.051) for i in ids}, ref, same, 1000)["checks"]["C3_mean_kl"]
    assert CR.evaluate(ids, {i: _teacher(400, kl=0.049) for i in ids}, ref, same, 1000)["pass"]
    # C1
    assert not CR.evaluate(ids, {i: _teacher(400, finite=(i != "p3")) for i in ids}, ref, same, 1000)["pass"]
    assert not CR.evaluate(ids, {i: _teacher(400, kl=float("nan")) for i in ids}, ref, same, 1000)["pass"]
    # C4: one prompt may diverge at a decisive position (10 %), two may not; a near-tie divergence is not a failure
    div = lambda at: {"tokens": list(range(1, at + 1)) + [999] + list(range(at + 2, 65)), "finite": True}
    g = dict(same); g["p0"] = div(5)
    assert CR.evaluate(ids, {i: _teacher(400) for i in ids}, ref, g, 1000)["pass"]
    g["p1"] = div(9)
    assert not CR.evaluate(ids, {i: _teacher(400) for i in ids}, ref, g, 1000)["checks"]["C4_greedy"]
    ref2 = {i: {"tokens": list(range(1, 65)), "margins": [2.0] * 5 + [0.2] + [2.0] * 58} for i in ids}
    g = {i: div(5) for i in ids}
    assert CR.evaluate(ids, {i: _teacher(400) for i in ids}, ref2, g, 1000)["checks"]["C4_greedy"]
    # fail closed
    assert not CR.evaluate(ids, {i: _teacher(400) for i in ids[:9]}, ref, same, 1000)["pass"]
    assert not CR.evaluate(ids, {i: _teacher(400, margin=0.1) for i in ids}, ref, same, 1000)["pass"]      # no decisive position at all
    assert not CR.evaluate([], {}, {}, {}, 1000)["pass"]
    short = dict(same); short["p0"] = {"tokens": list(range(1, 60)), "finite": True}
    assert not CR.evaluate(ids, {i: _teacher(400) for i in ids}, ref, short, 1000)["pass"]


def test_split_equals_unsplit_over_the_wire():
    x = InProcess()
    try:
        p = x.p
        assert all(r["finite"] and r["share_decisive"] > 0.2 for r in x.ref_rows), x.ref_rows
        # ownership: each side holds its own layers and nothing else
        assert x.side.sl.layer_ids == S.SERVER_LAYERS and x.side.sl.embed is None and x.side.sl.lm_head is None and x.side.sl.norm is None
        assert x.sl.layer_ids == sorted(S.EDGE_LAYERS) and x.sl.embed is not None and x.sl.lm_head is not None
        mid = R.connect(x.addr, x.sl, "prefix", x.h, p)
        e, info = R.edge_split(x.sl, mid, "prefix", p)
        assert e.head.layer_ids == list(range(16)) and e.tail.layer_ids == list(range(48, 64))
        assert sorted(e.head.cache.k) == list(range(16)) and sorted(e.tail.cache.k) == list(range(48, 64))          # edge KV: its own layers
        assert sorted(x.side.segs["prefix"].cache.k) == list(range(16, 48))                                        # server KV: its own layers
        res = R.evaluate_config(e, x.prompts, x.ref, p, TMP / "cmp_prefix", "test prefix")
        assert res["pass"] and res["kv_positions_in_step_all"] and res["teacher_forced"]["n_decisive"] > 50, res["checks"]
        if STRICT:      # same operations on the same shapes as the unsplit pass: bit-identical hidden states and tokens
            assert res["all_bitwise_equal_reference_hidden"] and res["greedy"]["identical"] == len(x.prompts), res["descriptive"]
        assert max(d["prefill_chunks"] for d in res["descriptive"].values()) == 1
        # wire: one residual tensor per direction per step, 72-byte header, nothing else
        ids = x.prompts[0]["ids"]
        b0, b1 = mid.bytes_sent, mid.bytes_recv
        toks = e.generate(ids, 8, p["chunk"], force_len=True)
        H, per_dir = x.sl.cfg.hidden_size, None
        per_dir = H * 2 * (len(ids) + 7) + 72 * (1 + 7)
        assert sum(s["bytes_out"] for s in e.steps) == per_dir == sum(s["bytes_in"] for s in e.steps) and len(toks) == 8
        assert mid.bytes_sent - b0 == per_dir + 72 and mid.bytes_recv - b1 == per_dir + 72       # + the RESET exchange
        assert R.kv_invariant(e.steps) and e.steps[-1]["server_seen"] == len(ids) + 7 == e.head.cache.seen == e.tail.cache.seen
        mid.close()
        # static mode: same criterion, same reference
        mid = R.connect(x.addr, x.sl, "static", x.h, p)
        es, info_s = R.edge_split(x.sl, mid, "static", p)
        rs = R.evaluate_config(es, x.prompts, x.ref, p, TMP / "cmp_static", "test static")
        assert rs["pass"] and info_s["cuda_graphs"] == "cpu_control_flow" and x.sl.cfg._attn_implementation == "e1_static"
        mid.close(); R.use_mode(x.sl, "prefix")
        # the server refuses another spec hash, another cut and an unknown mode
        for bad in ({"freeze_sha256": "0" * 64}, {"k": 3}, {"model_id": "other"}):
            hello = dict(R.P.hello_for(x.sl, p["model_id"], S.K, S.J, p["max_len"], x.h), runtime=f"{S.NAME}/prefix", **bad)
            try:
                R.P.RemoteMiddle("127.0.0.1", int(x.addr.split(":")[1]), hello, "cpu", H, p["chunk"])
                raise AssertionError(f"server accepted {bad}")
            except R.P.TransportError as ex:
                assert "mismatch" in str(ex), ex
        x.side.segs.pop("static"); x.close()
        x.stop.clear(); x.sock = socket.socket(); x.sock.bind(("127.0.0.1", 0)); x.sock.listen(1)
        x.th = threading.Thread(target=R.serve, args=(x.side, x.sock, x.h, x.events.append), kwargs={"stop": x.stop}, daemon=True); x.th.start()
        try:
            R.connect(f"127.0.0.1:{x.sock.getsockname()[1]}", x.sl, "static", x.h, p)
            raise AssertionError("server accepted a mode it does not have")
        except R.P.TransportError as ex:
            assert "not available" in str(ex), ex
        assert any(ev.get("event") == "refused" for ev in x.events)
    finally:
        x.close()


def test_chunked_prefill_keeps_both_kv_caches_in_step():
    x = InProcess()
    try:
        p = x.p
        mid = R.connect(x.addr, x.sl, "prefix", x.h, p)
        e, _ = R.edge_split(x.sl, mid, "prefix", p)
        ids = x.prompts[-1]["ids"] + [5] * 8                       # 48 tokens
        h1 = e.teacher_forced_hidden(ids, p["chunk"])              # one pass, no KV read
        h3 = e.teacher_forced_hidden(ids, 16)                      # three chunks: each side reads its own KV across the borders
        steps = list(e.steps)
        assert [(s["start"], s["n"]) for s in steps] == [(0, 16), (16, 16), (32, 16)] and R.kv_invariant(steps)
        assert e.head.cache.seen == e.tail.cache.seen == x.side.segs["prefix"].cache.seen == 48
        assert h3.shape == h1.shape and bool(torch.isfinite(h3.float()).all())
        with torch.inference_mode():
            same = (x.sl.lm_head(h1).float().argmax(-1) == x.sl.lm_head(h3).float().argmax(-1)).float().mean()
        assert same >= 0.9, same                                   # the same function; rounding may differ between chunkings
        t1 = e.generate(ids[:40], 8, 16, force_len=True)           # chunked prefill (16 + 16 + 8), then 7 cached decode steps
        assert len(t1) == 8 and R.kv_invariant(e.steps) and e.steps[-1]["server_seen"] == 47 and [s["n"] for s in e.steps[:3]] == [16, 16, 8]
        mid.close()
    finally:
        x.close()


def test_positive_control_rounding_noise_passes():
    """A genuinely small numerical difference must PASS the rule without being bit-identical: the tensor the server
    returns gets +1 unit in the last place on ONE element per token, at every step. Deterministic, no random numbers.

    The control runs on 40 prompts instead of the 5 dev prompts. With 5 prompts (about 150 decisive positions) the
    rule's tolerances round down to zero events: 0.5 % of 150 positions is less than one position and 10 % of 5
    prompts is less than one prompt. This fixture's bf16 logits are quantised in steps of up to 0.25, so a single
    position at a 0.5-nat margin can flip under any non-zero noise, and whether one does depends on the random draw
    of the tiny model, which differs between torch builds (the earlier control, 1 and 4 units on half of all
    elements, failed C2 on torch 2.14 / aarch64 and on other seeds here). On 40 prompts the unchanged thresholds of
    K16-CORRECT-v1 allow a few events, which is what they mean on the real check set (about 12,000 positions)."""
    import random
    x = InProcess()
    try:
        rng = random.Random("k16-positive-control")
        prompts = [{"id": f"pc-{i}", "group": "short", "n_in": 40, "ids": [rng.randrange(1, 211) for _ in range(40)]} for i in range(40)]
        ref = TMP / "ref_positive_control"
        R.write_reference(R.hf_model(tiny_dir(), "cpu"), prompts, x.p, "cpu", ref)
        bumped = []

        class OneUlp(R.LocalMiddle):
            def forward(self, h, start):
                i = super().forward(h, start).contiguous().view(torch.int16).clone()
                i[..., 0] += 1                             # element 0 of every token: the adjacent bf16 value
                bumped.append(i.shape[1])
                return i.view(torch.bfloat16)
        mid = OneUlp(x.side.sl, x.side.sl.layer_ids, x.p["max_len"], "prefix")
        e, _ = R.edge_split(x.sl, mid, "prefix", x.p)
        res = R.evaluate_config(e, prompts, ref, x.p, TMP / "cmp_positive_control", "test one-ulp control")
        tf, g = res["teacher_forced"], res["greedy"]
        note = (f"decisive flips {tf['decisive_disagreements']} of {tf['n_decisive']} (rule allows {int(S.CRITERIA['max_decisive_disagreement'] * tf['n_decisive'])}), "
                f"mean KL {tf['mean_kl_nats']:.2e}, greedy failures {g['n_prompts'] - g['identical_or_near_tie']} of {g['n_prompts']} (rule allows {g['failures_allowed']})")
        assert sum(bumped) > 0 and not res["all_bitwise_equal_reference_hidden"], "the control did not perturb anything"
        assert res["pass"] and all(res["checks"].values()), (res["checks"], note)
        return note
    finally:
        x.close()


def test_negative_control_wrong_layer_order_fails():
    x = InProcess(sabotage=True)
    try:
        mid = R.connect(x.addr, x.sl, "prefix", x.h, x.p)
        e, _ = R.edge_split(x.sl, mid, "prefix", x.p)
        res = R.evaluate_config(e, x.prompts, x.ref, x.p, TMP / "cmp_bad", "test sabotage")
        mid.close()
        assert not res["pass"] and not res["checks"]["C2_decisive_agreement"], res["checks"]
        assert res["teacher_forced"]["decisive_disagreement_share"] > 0.2 and not res["all_bitwise_equal_reference_hidden"]
    finally:
        x.close()


def _server(run, port, extra=()):
    return subprocess.Popen(PY + [str(HERE / "k16_server.py"), "all", "--dev", "--model-dir", tiny_dir(), "--run-dir", str(run), "--port", str(port),
                                  "--wait-s", "180", *extra], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _wait_ready(run, proc, t=120):
    end = time.time() + t
    while time.time() < end:
        if (run / "server_ready.json").exists():
            return
        assert proc.poll() is None, proc.stdout.read()
        time.sleep(0.3)
    raise AssertionError("server not ready")


def _edge(run, port, extra=()):
    return subprocess.run(PY + [str(HERE / "k16_edge.py"), "all", "--dev", "--device", "cpu", "--model-dir", tiny_dir(), "--run-dir", str(run),
                                "--server", f"127.0.0.1:{port}", *extra], capture_output=True, text=True)


def test_end_to_end_two_processes():
    port, srun, erun = free_port(), TMP / "e2e_server", TMP / "e2e_edge"
    srv = _server(srun, port)
    try:
        _wait_ready(srun, srv)
        r = _edge(erun, port)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
        assert srv.wait(60) == 0                                                    # the edge asked it to shut down
    finally:
        if srv.poll() is None:
            srv.terminate()
    s = read_json(erun / "summary.json")
    assert all(s["gates"][g] == "PASS" for g in ("reference", "inhost_prefix", "twohost_prefix", "generation", "latency_prefix", "twohost_static", "latency_static")), s["gates"]
    order = [x["stage"] for x in read_json(erun / "stages.json")]
    assert order == ["preflight", "reference", "inhost", "twohost"]
    rows = read_jsonl(erun / "latency" / "prefix.jsonl")
    assert len(rows) == 1 + 5 and all(x["ok"] and len(x["token_ns"]) == 8 and x["spec_hash"] == s["spec_hash"] for x in rows)
    assert set(s["latency"]) == {"prefix", "static"} and s["latency"]["prefix"]["groups"]["short"]["decode_step_ms"]["server_compute_median"] > 0
    ready, done = read_json(srun / "server_ready.json"), read_json(srun / "server_done.json")
    assert ready["ownership"]["equals_spec"] and done["exit"] == "shutdown" and ready["spec_hash"] == s["spec_hash"]
    ev = read_jsonl(srun / "server_events.jsonl")
    assert sum(e["event"] == "hello_ok" for e in ev) == 4 and not any("tokens" in json.dumps(e) for e in ev)     # ping, prefix, static, final
    assert ready["listen"].startswith("127.0.0.1:") and read_json(erun / "preflight.json")["server_identity"]["freeze_sha256"] == s["spec_hash"]
    assert read_json(erun / "reference" / "reference.json")["load_memory"]["host_peak_rss_mib"] > 0
    assert (erun.parent / (erun.name + ".tgz")).exists()
    # a stage never runs behind a closed gate: twohost in a run directory without the in-host gate
    bare = TMP / "bare"; (bare / "gates").mkdir(parents=True)
    shutil.copy(erun / "preflight.json", bare / "preflight.json"); shutil.copy(gate_path(erun, "reference"), gate_path(bare, "reference"))
    r = subprocess.run(PY + [str(HERE / "k16_edge.py"), "twohost", "--dev", "--device", "cpu", "--run-dir", str(bare), "--server", f"127.0.0.1:{port}"],
                       capture_output=True, text=True)
    assert r.returncode == EXIT_GATE and "REFUSED" in r.stdout and not (bare / "latency").exists(), r.stdout[-800:]


def test_end_to_end_failed_gate_stops_before_latency():
    port, srun, erun = free_port(), TMP / "bad_server", TMP / "bad_edge"
    srv = _server(srun, port, ["--dev-sabotage"])
    try:
        _wait_ready(srun, srv)
        r = _edge(erun, port)
        assert r.returncode == EXIT_GATE, r.stdout[-2000:]
        assert srv.wait(60) == 0                                                    # released even though the gate failed
    finally:
        if srv.poll() is None:
            srv.terminate()
    s = read_json(erun / "summary.json")
    assert s["gates"]["inhost_prefix"] == "PASS" and s["gates"]["twohost_prefix"] == "FAIL", s["gates"]
    assert s["gates"]["latency_prefix"] == "not run" and s["gates"]["generation"] == "not run" and s["latency"] == {}
    assert not (erun / "latency").exists() and not (erun / "generation").exists()


def test_preflight_stop_starts_nothing():
    run = TMP / "stop_edge"
    r = subprocess.run(PY + [str(HERE / "k16_edge.py"), "all", "--dev", "--device", "cpu", "--model-dir", str(TMP / "no_such_checkpoint"),
                             "--run-dir", str(run), "--server", f"127.0.0.1:{free_port()}"], capture_output=True, text=True)
    assert r.returncode == EXIT_STOP, r.stdout[-1500:]
    pf = read_json(run / "preflight.json")
    assert pf["verdict"] == "STOP" and {b["check"] for b in pf["blockers"]} >= {"checkpoint directory found"}
    assert any("answers HELLO" in b["check"] for b in pf["blockers"])                          # no server was listening either
    assert [x["stage"] for x in read_json(run / "stages.json")] == ["preflight"] and not (run / "reference").exists()
    # a stage called directly without a CONTINUE preflight refuses too
    r = subprocess.run(PY + [str(HERE / "k16_edge.py"), "reference", "--dev", "--run-dir", str(run)], capture_output=True, text=True)
    assert r.returncode == EXIT_STOP and not (run / "reference").exists()


FAKE_SSH = r'''#!%s
# Test stand-in for `ssh -N ... -L 127.0.0.1:LP:127.0.0.1:RP target`: records its arguments and relays TCP like the forward.
import os, socket, sys, threading
open(os.environ["FAKE_SSH_LOG"], "a").write(" ".join(sys.argv[1:]) + "\n")
if os.environ.get("FAKE_SSH_FAIL"):
    sys.stderr.write("Permission denied (publickey).\n"); sys.exit(255)
spec = sys.argv[sys.argv.index("-L") + 1].split(":")
lp, rp = int(spec[1]), int(spec[3])
srv = socket.create_server(("127.0.0.1", lp))
def pump(a, b):
    try:
        while True:
            d = a.recv(1 << 16)
            if not d:
                break
            b.sendall(d)
    except OSError:
        pass
    for s in (a, b):
        try:
            s.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
while True:
    c, _ = srv.accept()
    try:
        r = socket.create_connection(("127.0.0.1", rp))
    except OSError:
        c.close(); continue
    threading.Thread(target=pump, args=(c, r), daemon=True).start()
    threading.Thread(target=pump, args=(r, c), daemon=True).start()
'''


def test_launchers_loopback_server_and_ssh_tunnel():
    """The shipped launch path: ASUS launcher (loopback listener) and DGX launcher (its own SSH tunnel, closed at exit).
    `ssh` is a stand-in that relays TCP on this machine; the options the launcher passes to it are checked."""
    if os.environ.get("K16_IN_LAUNCHER") == "1":
        return "SKIP: running inside a launcher's stage 0"
    fake = TMP / "bin"; fake.mkdir()
    (fake / "ssh").write_text(FAKE_SSH % sys.executable); (fake / "ssh").chmod(0o755)
    out, sshlog = TMP / "launch_out", TMP / "ssh_args.log"
    port, lport = free_port(), free_port()
    env = dict(os.environ, K16_DEV="1", K16_DEV_SKIP_SELFTEST="1", K16_PY=sys.executable, K16_OUT=str(out), FAKE_SSH_LOG=str(sshlog),
               PATH=f"{fake}:{os.environ['PATH']}")
    for k in ("K16_SERVER", "K16_SSH", "K16_PORT", "K16_LOCAL_PORT", "K16_MODEL_DIR"):
        env.pop(k, None)
    # no ssh target and no server address: refuse before anything starts
    r = subprocess.run(["bash", str(HERE / "launch_dgx_edge.sh"), "--dev"], env=env, capture_output=True, text=True)
    assert r.returncode == EXIT_STOP and "K16_SSH" in r.stderr, r.stderr
    # a tunnel that cannot be opened: refuse before anything starts
    r = subprocess.run(["bash", str(HERE / "launch_dgx_edge.sh"), "--dev"], env=dict(env, K16_SSH="me@asus", FAKE_SSH_FAIL="1", K16_LOCAL_PORT=str(lport)),
                       capture_output=True, text=True)
    assert r.returncode == EXIT_STOP and "did not come up" in r.stderr and not list(out.glob("*_edge")), r.stderr
    # a public bind is refused by the server driver
    r = subprocess.run(PY + [str(HERE / "k16_server.py"), "all", "--dev", "--model-dir", tiny_dir(), "--host-bind", "0.0.0.0", "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == EXIT_STOP and "REFUSED" in r.stdout and not list(out.glob("*_server")), r.stdout
    srv = subprocess.Popen(["bash", str(HERE / "launch_asus_server.sh"), "--dev", "--model-dir", tiny_dir(), "--port", str(port), "--wait-s", "240"],
                           env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        end = time.time() + 120
        while time.time() < end and not list(out.glob("*_server/server_ready.json")):
            assert srv.poll() is None, srv.stdout.read()
            time.sleep(0.3)
        ready = read_json(next(out.glob("*_server/server_ready.json")))
        assert ready["listen"] == f"127.0.0.1:{port}"
        r = subprocess.run(["bash", str(HERE / "launch_dgx_edge.sh"), "--dev", "--device", "cpu", "--model-dir", tiny_dir()],
                           env=dict(env, K16_SSH="me@asus", K16_PORT=str(port), K16_LOCAL_PORT=str(lport)), capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-1500:]
        assert srv.wait(60) == 0
    finally:
        if srv.poll() is None:
            srv.terminate()
    args = sshlog.read_text().splitlines()[-1]
    assert f"-L 127.0.0.1:{lport}:127.0.0.1:{port} me@asus" in args and "-N" in args.split(), args
    assert all(o in args for o in ("BatchMode=yes", "ExitOnForwardFailure=yes", "ServerAliveInterval=30", "Compression=no")), args
    erun = next(out.glob("*_edge"))
    s = read_json(erun / "summary.json")
    assert s["gates"]["twohost_prefix"] == s["gates"]["latency_prefix"] == "PASS", s["gates"]
    rows = read_jsonl(erun / "latency" / "prefix.jsonl")
    assert rows and all(x["transport"].startswith("ssh tunnel") and x["ok"] for x in rows)
    assert f"127.0.0.1:{lport}" in r.stdout and "SSH tunnel up" in r.stdout
    time.sleep(0.5)
    try:                                           # the launcher closed its own tunnel: nothing listens on the local end any more
        socket.create_connection(("127.0.0.1", lport), timeout=2).close()
        raise AssertionError("the tunnel is still open after the launcher exited")
    except OSError:
        pass
    assert list(out.glob("*_edge_tunnel.log")) and list(out.glob("*_edge_console.log")) and list(out.glob("*_server_console.log"))


SLOW = {"test_launchers_loopback_server_and_ssh_tunnel", "test_end_to_end_two_processes", "test_end_to_end_failed_gate_stops_before_latency", "test_preflight_stop_starts_nothing"}


def main():
    quick = "--quick" in sys.argv
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f) and not (quick and n in SLOW)]
    failed = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            note = fn()
            print(f"PASS {name} ({time.time() - t0:.1f} s){' ' + note if note else ''}", flush=True)
        except Exception as e:
            failed += 1
            import traceback
            print(f"FAIL {name}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed" + (" (quick: end-to-end runs skipped)" if quick else ""))
    shutil.rmtree(TMP, ignore_errors=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
