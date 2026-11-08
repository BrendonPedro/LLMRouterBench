"""E0 local self-test (CPU only; no GPU, no network except none).

  python e0_selftest.py [--tokenizer-dir DIR] [--real-model-dir DIR] [--quick] [--report FILE]

Groups
  spec      frozen constants: held-out draw, cell table, L_F vs S1 lengths, budgets, U-max prediction
  s1        S1 evidence: pinned hashes, 4,000 complete rows, CONTINUE recomputed from per-query data
  template  chat-template parity: the pinned tokenizer reproduces S1 prompt_tokens (needs --tokenizer-dir)
  runtime   split runtime on a tiny random Qwen3 (bf16 and fp32): bitwise equivalence to HF, TCP wire
            format, KV/position checks, failure paths, app shaper
  real      the same on a real checkpoint (Qwen3-0.6B, needs --real-model-dir), incl. the Q pipeline CLIs
  q         Q metrics and rules on constructed cases
  f         U-max search, sampler accounting
  latency   workload, plans, netem parsing, SPLIT cell runner over TCP, streaming client vs a fake vLLM
  r         profiles, bootstrap, grid evaluation, upper-bound monotonicity, additive-model recovery and
            validation, decision table
  docs      every `e0_*.py <subcommand>` in RUNBOOK.md exists
"""

import argparse
import contextlib
import http.server
import io
import json
import math
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e0_spec as S  # noqa: E402

RESULTS = []


def check(group, name):
    def deco(fn):
        fn._check = (group, name)
        return fn
    return deco


class Skip(Exception):
    pass


# --------------------------------------------------------------------------- #
# Tiny model fixtures                                                         #
# --------------------------------------------------------------------------- #
_TINY = {}


def tiny_dir(dtype="bfloat16", vocab=512, layers=8):
    key = (dtype, vocab, layers)
    if key in _TINY:
        return _TINY[key]
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(1234)
    cfg = Qwen3Config(vocab_size=vocab, hidden_size=64, intermediate_size=160, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=40960,
                      tie_word_embeddings=False, use_sliding_window=False, sliding_window=None, torch_dtype=dtype)
    m = Qwen3ForCausalLM(cfg).to(getattr(torch, dtype)).eval()
    d = Path(tempfile.mkdtemp(prefix="e0tiny_"))
    m.save_pretrained(d, safe_serialization=True)
    _TINY[key] = d
    return d


def hf_model(d, dtype):
    import torch
    from transformers import Qwen3ForCausalLM
    return Qwen3ForCausalLM.from_pretrained(d, torch_dtype=getattr(torch, dtype), attn_implementation="sdpa").eval()


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def start_server(srv):
    port = free_port()
    threading.Thread(target=srv.serve_forever, args=("127.0.0.1", port), daemon=True).start()
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            time.sleep(0.05)
    return port


def make_split(d, k, j, dtype="bfloat16", remote=True, max_len=40960, server_layers=None):
    import torch
    import split_runtime as R
    dt = getattr(torch, dtype)
    edge = R.edge_slice(d, k, j, "cpu", dt)
    L = edge.n_layers
    if remote:
        lay = server_layers or list(range(1, L - 1))
        srv = R.SegmentServer(R.Slice(d, lay, False, False, "cpu", dt), "tiny", max_len)
        port = start_server(srv)
        mid = R.RemoteMiddle("127.0.0.1", port, R.hello_for(edge, "tiny", k, j, max_len), "cpu",
                             edge.cfg.hidden_size)
    else:
        mid = R.LocalMiddle(R.Slice(d, list(range(k, L - j)), False, False, "cpu", dt), list(range(k, L - j)),
                            max_len)
    return R.EdgeRunner(edge, k, j, mid, max_len=max_len), mid


# --------------------------------------------------------------------------- #
# spec                                                                        #
# --------------------------------------------------------------------------- #
@check("spec", "held-out cells equal their seeded draw; 22 decision cells")
def t_spec_cells():
    assert S.draw_heldout() == S.HELDOUT, (S.draw_heldout(), S.HELDOUT)
    cs = S.cells()
    assert sum(c["role"] == "decision" for c in cs) == S.N_DECISION_CELLS
    assert len({c["id"] for c in cs}) == len(cs)
    blocks = {b: sum(c["block"] == b for c in cs) for b in ("core", "rtt", "cut", "bw", "load", "heldout")}
    assert blocks == {"core": 6, "rtt": 6, "cut": 1, "bw": 1, "load": 4, "heldout": 4}, blocks


@check("spec", "L_F covers every S1 prompt + 16,384 tokens; budgets; U-max prediction = desk check")
def t_spec_lf():
    import e0_memory as M
    rows = [json.loads(x) for x in open(S.S1_Q3_SCORES)]
    need = max(r["prompt_tokens"] for r in rows) + S.MAX_NEW_TOKENS_S1
    assert S.L_F >= need and S.L_F - need < 512 and S.L_F % 512 == 0, (S.L_F, need)
    assert [S.budget_mib(m) for m in S.MEMORY_CLASSES] == [11776, 15872, 24064]
    for m in S.MEMORY_CLASSES:
        assert M.predict_umax(m, S.QWEN3_32B_CONFIG) == S.U_MAX_PREDICTED[m]
    assert M.param_counts(S.QWEN3_32B_CONFIG)["total"] * 2 == S.QWEN3_32B_TOTAL_BYTES


@check("spec", "E0 model revisions equal the S1 run's")
def t_spec_rev():
    s1 = S.load_s1()
    assert s1.MODELS[S.SPLIT_MODEL]["revision"] == S.SPLIT_REVISION
    assert s1.MODELS[S.FLOOR_MODEL]["revision"] == S.FLOOR_REVISION
    assert s1.VLLM_VERSION == S.VLLM_VERSION
    assert set(S.EDGE_MODELS) == set(s1.MODELS) - {S.SPLIT_MODEL}


# --------------------------------------------------------------------------- #
# s1                                                                          #
# --------------------------------------------------------------------------- #
@check("s1", "pinned hashes of the S1 per-query file and sample")
def t_s1_hash():
    assert S.sha256_file(S.S1_Q3_SCORES) == S.S1_Q3_SHA256
    assert S.sha256_file(S.load_s1().SAMPLE_JSONL) == S.S1_SAMPLE_SHA256


@check("s1", "the pre-declared 400-query extension sample is pinned, contains the 200, and re-draws identically")
def t_s1_ext400():
    import hashlib
    ext = json.loads((HERE / "frozen" / "extension_400_keys.json").read_text())
    keys = ext["keys"]
    assert hashlib.sha256(json.dumps(sorted(keys)).encode()).hexdigest() == S.EXT400_KEYS_SHA256 == ext["keys_sha256"]
    assert len(keys) == len(set(keys)) == 400 and ext["sample_sha256"] == S.EXT400_SAMPLE_SHA256
    assert ext["per_benchmark"] == {"mmlupro": 120, "gpqa": 120, "livemathbench": 80, "livecodebench": 80}
    s1 = S.load_s1()
    first = {json.loads(x)["key"] for x in open(s1.SAMPLE_JSONL)}
    assert first <= set(keys)
    try:
        rows, _ = s1.draw_sample(2)
    except Exception as e:   # benchmark cache absent (pods clone without LFS): the pinned hashes above still hold
        raise Skip(f"re-draw needs the benchmark cache: {type(e).__name__}") from None
    assert hashlib.sha256(s1.sample_bytes(rows)).hexdigest() == S.EXT400_SAMPLE_SHA256
    assert sorted(r["key"] for r in rows) == sorted(keys)


@check("s1", "4,000 complete rows; CONTINUE recomputed; gaps equal S1_SUMMARY")
def t_s1_continue():
    import e0_r as Rr
    d = Rr.S1Data()
    assert len(d.score) == 10 and all(v.shape == (200, 2) for v in d.score.values())
    el = S.EDGE_ELIGIBLE_DESK
    gaps = {}
    for b_i, b in enumerate(Rr.BENCH_ORDER):
        msk = d.strata == b_i
        for md in S.MODES:
            acc = lambda m: 100 * d.score[(m, md)][msk].mean()
            for M in (12, 16):
                best = max(el[M], key=lambda m: (acc(m), -S.EDGE_MODELS.index(m)))
                gaps[(b, md, M)] = acc(S.SPLIT_MODEL) - acc(best)
    rule16 = all(g < 3.0 for (b, md, M), g in gaps.items() if M == 16)
    assert not rule16, "S1 verdict would not be CONTINUE"
    assert abs(gaps[("mmlupro", "thinking", 16)] - 5.8) < 0.05 and abs(gaps[("gpqa", "nonthinking", 16)] - 7.5) < 0.05
    summ = json.loads((S.S1_RESULT_DIR / "s1_summary.json").read_text())
    assert summ["decision"] == "CONTINUE", summ.get("decision")


# --------------------------------------------------------------------------- #
# template                                                                    #
# --------------------------------------------------------------------------- #
@check("template", "pinned tokenizer + chat template reproduce S1 prompt_tokens for 200 x 2 modes")
def t_template(args):
    if not args.tokenizer_dir:
        raise Skip("no --tokenizer-dir")
    from transformers import AutoTokenizer
    from e0_latency import templated_ids
    assert S.sha256_file(Path(args.tokenizer_dir) / "tokenizer.json") == S.TOKENIZER_JSON_SHA256
    tok = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    want = {}
    for r in (json.loads(x) for x in open(S.S1_Q3_SCORES)):
        want[(f"{r['dataset']}:{r['index']}", r["mode"])] = r["prompt_tokens"]
    s1 = S.load_s1()
    bad = []
    for q in (json.loads(x) for x in open(s1.SAMPLE_JSONL)):
        for md, th in (("thinking", True), ("nonthinking", False)):
            n = len(templated_ids(tok, q["prompt"], th))
            if n != want[(q["key"], md)]:
                bad.append((q["key"], md, n, want[(q["key"], md)]))
    assert not bad, bad[:5]


@check("template", "frozen workload rebuilds to the pinned ids hash; frozen Q2 keys reproduce")
def t_workload(args):
    import e0_q
    import e0_latency as L
    w = L.load_workload()
    assert w["ids_sha256"] == S.WORKLOAD_IDS_SHA256
    assert [(p["group"], p["n_in"]) for p in w["prompts"]].count(("depth", S.L_F - S.GEN_TOKENS)) == 2
    assert json.loads(e0_q.Q2_KEYS_JSON.read_text())["keys"] == e0_q.q2_keys()
    if not args.tokenizer_dir:
        raise Skip("workload rebuild needs --tokenizer-dir (hash and lengths checked)")
    assert L.build_workload(args.tokenizer_dir)["ids_sha256"] == S.WORKLOAD_IDS_SHA256


# --------------------------------------------------------------------------- #
# runtime (tiny random Qwen3)                                                 #
# --------------------------------------------------------------------------- #
def _tf_equal(dtype, k, j, remote):
    import torch
    d = tiny_dir(dtype)
    ref = hf_model(d, dtype)
    ids = torch.randint(0, 512, (1, 57), generator=torch.Generator().manual_seed(k * 10 + j))
    with torch.inference_mode():
        href = ref.model(ids).last_hidden_state[0]
    runner, mid = make_split(d, k, j, dtype, remote, max_len=256)
    h = runner.teacher_forced_hidden(ids)
    if remote:
        mid.close()
    return torch.equal(h, href)


@check("runtime", "teacher-forced split == HF forward, bitwise (bf16 local+TCP, fp32 local; U-min, (2,3), (3,3)); wire refuses non-bf16")
def t_rt_tf():
    for dtype in ("bfloat16", "float32"):
        for kj in ((1, 1), (2, 3), (3, 3)):
            for remote in ((False, True) if dtype == "bfloat16" else (False,)):
                assert _tf_equal(dtype, *kj, remote), (dtype, kj, remote)
    try:                                   # the wire refuses any dtype change (E0 §7)
        _tf_equal("float32", 1, 1, True)
        raise AssertionError("fp32 activations were sent over the bf16 wire")
    except TypeError as e:
        assert "bfloat16 only" in str(e)


@check("runtime", "greedy split decoding == HF generate (chunked prefill 7 / 16 / unchunked, TCP)")
def t_rt_greedy():
    import torch
    d = tiny_dir("bfloat16")
    ref = hf_model(d, "bfloat16")
    ids = torch.randint(0, 512, (1, 41), generator=torch.Generator().manual_seed(7))
    with torch.inference_mode():
        g = ref.generate(ids, max_new_tokens=40, min_new_tokens=40, do_sample=False)[0, 41:].tolist()
    runner, mid = make_split(d, 2, 2, max_len=256)
    for chunk in (7, 16, 4096):
        out = runner.generate(ids[0].tolist(), 40, chunk, force_len=True)
        assert out == g, chunk
        assert len(runner.token_ns) == 40 and all(b >= a for a, b in zip(runner.token_ns, runner.token_ns[1:]))
    mid.close()


@check("runtime", "wire carries exactly n x hidden x 2 bytes + header each way; server cache tracks positions")
def t_rt_wire():
    import torch
    import split_runtime as R
    d = tiny_dir("bfloat16")
    runner, mid = make_split(d, 1, 1, max_len=256)
    runner.generate(list(range(30)), 5, 8, force_len=True)
    pre = [s for s in runner.steps if s["n"] > 1 or s["start"] < 30]
    for s in runner.steps:
        assert s["bytes_out"] == s["bytes_in"] == R.HDR.size + s["n"] * 64 * 2
        assert s["server_seen"] == s["start"] + s["n"]
    assert [s["n"] for s in pre] == [8, 8, 8, 6]
    assert R.HDR.size == S.WIRE_HEADER_BYTES
    mid.close()


@check("runtime", "failure paths: wrong position, illegal split, version mismatch, cache overflow, peer drop")
def t_rt_fail():
    import torch
    import split_runtime as R
    d = tiny_dir("bfloat16")
    runner, mid = make_split(d, 2, 2, max_len=64)
    runner.reset()
    x = torch.zeros((1, 3, 64), dtype=torch.bfloat16)
    try:
        mid.forward(x, 5, True)       # server cache length is 0
        raise AssertionError("position mismatch not detected")
    except R.TransportError as e:
        assert "position" in str(e)
    edge = R.edge_slice(d, 2, 2, "cpu")
    for kj in ((0, 2), (2, 0)):
        try:
            R.EdgeRunner(edge, *kj, R.LocalMiddle(edge, [], None))
            raise AssertionError("illegal split accepted")
        except ValueError:
            pass
    srv = R.SegmentServer(R.Slice(d, list(range(1, 7)), False, False, "cpu"), "tiny", 64)
    port = start_server(srv)
    hello = R.hello_for(edge, "tiny", 2, 2, 64)
    hello["software"] = dict(hello["software"], transformers="0.0.0")
    try:
        R.RemoteMiddle("127.0.0.1", port, hello, "cpu", 64)
        raise AssertionError("software mismatch accepted")
    except R.TransportError as e:
        assert "software mismatch" in str(e)
    runner2, mid2 = make_split(d, 2, 2, max_len=64)
    try:
        runner2.generate(list(range(60)), 10, 16)
        raise AssertionError("overflow not detected")
    except R.CacheOverflow:
        pass
    runner2.reset()
    mid2.sock.close()
    try:
        runner2.generate(list(range(10)), 3, 16)
        raise AssertionError("closed socket not detected")
    except (R.TransportError, OSError):
        pass


@check("runtime", "echo stub returns the payload unchanged; app shaper adds RTT + serialisation per round trip")
def t_rt_echo_shaper():
    import torch
    import split_runtime as R
    d = tiny_dir("bfloat16")
    edge = R.edge_slice(d, 1, 1, "cpu")
    port = start_server(R.SegmentServer(None, "echo", 64, echo=True))
    x = torch.randn((1, 5, 64)).to(torch.bfloat16)
    base = R.RemoteMiddle("127.0.0.1", port, R.hello_for(edge, "echo", 1, 1, 64), "cpu", 64)
    base.reset()
    assert torch.equal(base.forward(x, 0, True), x)
    port2 = start_server(R.SegmentServer(None, "echo", 64, echo=True))
    shaped = R.RemoteMiddle("127.0.0.1", port2, R.hello_for(edge, "echo", 1, 1, 64), "cpu", 64,
                            R.AppShaper(rtt_ms=20, bw_mbps=1))
    shaped.reset()
    t0 = time.perf_counter()
    shaped.forward(x, 0, True)
    dt = time.perf_counter() - t0
    expect = 0.020 + 2 * (R.HDR.size + 5 * 64 * 2) * 8 / 1e6
    assert expect <= dt < expect + 0.015, (dt, expect)
    assert abs(shaped.last["shape_ns"] / 1e9 - expect) < 0.003


@check("runtime", "Slice materialises only owned weights (param bytes exact; rest on meta)")
def t_rt_bytes():
    import torch
    import e0_memory as M
    import split_runtime as R
    d = tiny_dir("bfloat16")
    cfg = json.loads((d / "config.json").read_text())
    pc = M.param_counts(cfg)
    edge = R.edge_slice(d, 2, 3, "cpu")
    assert edge.param_bytes == 2 * (pc["embed"] + pc["lm_head"] + pc["norm"] + 5 * pc["layer"])
    assert all(p.device.type == "meta" for p in edge.skel_model.layers[3].parameters())
    c = edge.new_cache(100)
    assert c.nbytes() == 5 * 100 * M.kv_bytes_per_token_layer(cfg)


# --------------------------------------------------------------------------- #
# real checkpoint                                                             #
# --------------------------------------------------------------------------- #
@check("real", "Qwen3-0.6B real weights: TCP split U(2,3) bitwise == HF; greedy == HF generate (chat prompt)")
def t_real_rt(args):
    if not args.real_model_dir:
        raise Skip("no --real-model-dir")
    import torch
    from transformers import AutoTokenizer
    import split_runtime as R
    from e0_latency import templated_ids
    d = args.real_model_dir
    tok = AutoTokenizer.from_pretrained(d)
    s1 = S.load_s1()
    q = json.loads(open(s1.SAMPLE_JSONL).readline())
    ids = templated_ids(tok, q["prompt"])
    ref = hf_model(d, "bfloat16")
    t = torch.tensor([ids])
    with torch.inference_mode():
        href = ref.model(t).last_hidden_state[0]
        g = ref.generate(t, max_new_tokens=24, do_sample=False)[0, len(ids):].tolist()
    L = ref.config.num_hidden_layers
    runner, mid = make_split(d, 2, 3, server_layers=list(range(1, L - 1)), max_len=2048)
    assert torch.equal(runner.teacher_forced_hidden(t), href)
    out = runner.generate(ids, 24, S.PREFILL_CHUNK, eos_ids=S.EOS_IDS)
    assert out == g[:len(out)] and (len(out) == 24 or out[-1] in S.EOS_IDS), (out, g)
    mid.close()


@check("real", "Q pipeline CLIs end to end on Qwen3-0.6B (cont-gen, ref, cmp hf/split, q2-ref/run, rules)")
def t_real_q(args):
    if not args.real_model_dir:
        raise Skip("no --real-model-dir")
    import split_runtime as R
    import torch
    import e0_q as Q
    d = args.real_model_dir
    tmp = Path(tempfile.mkdtemp(prefix="e0q_"))
    py = [sys.executable, str(HERE / "e0_q.py")]
    run = lambda *a: subprocess.run(py + list(a), check=True, capture_output=True, text=True)
    run("cont-gen", "--model-dir", d, "--device", "cpu", "--out", str(tmp / "cont.jsonl"), "--batch", "3",
        "--max-new", "20", "--limit", "3")
    run("ref", "--model-dir", d, "--device", "cpu", "--cont", str(tmp / "cont.jsonl"), "--out-dir", str(tmp / "ref"))
    run("cmp", "--config", "hf", "--model-dir", d, "--device", "cpu", "--cont", str(tmp / "cont.jsonl"),
        "--ref-dir", str(tmp / "ref"), "--out-dir", str(tmp / "hw"))
    cfg = R.load_config(d)
    L = cfg.num_hidden_layers
    srv = R.SegmentServer(R.Slice(d, list(range(1, L - 1)), False, False, "cpu"), "real", 4096)
    port = start_server(srv)
    for k in (1, 3):
        run("cmp", "--config", "split", "--model-dir", d, "--model-id", "real", "--device", "cpu", "--cont",
            str(tmp / "cont.jsonl"), "--ref-dir", str(tmp / "ref"), "--out-dir", str(tmp / f"split{k}"),
            "--server", f"127.0.0.1:{port}", "--k", str(k), "--j", str(k), "--expect-32b", "0")
    for name in ("hw", "split1", "split3"):
        s = Q.q1_summary(tmp / name)
        assert s["n_queries"] == 3 and s["D_pp"] == 0.0 and s["KL"] < 1e-9, (name, s)
    run("q2-ref", "--model-dir", d, "--device", "cpu", "--out", str(tmp / "q2ref.jsonl"), "--max-new", "16",
        "--limit", "2")
    run("q2-run", "--model-dir", d, "--model-id", "real", "--device", "cpu", "--ref", str(tmp / "q2ref.jsonl"),
        "--out", str(tmp / "q2s.jsonl"), "--server", f"127.0.0.1:{port}", "--k", "1", "--j", "1", "--max-new", "16",
        "--expect-32b", "0")
    rr = Q.q2_rule(Q.I.read_jsonl(tmp / "q2ref.jsonl"), Q.I.read_jsonl(tmp / "q2s.jsonl"), len(
        __import__("transformers").AutoTokenizer.from_pretrained(d)))
    assert rr["near_tie_or_none"] == 2 and all(p["first_div"] is None for p in rr["per_prompt"]), rr
    shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# q                                                                           #
# --------------------------------------------------------------------------- #
@check("q", "coarse KL: 0 for identical, > 0 otherwise, <= full-vocabulary KL")
def t_q_kl():
    import numpy as np
    import e0_q as Q
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = rng.dirichlet(np.ones(200) * 0.3)
        q = rng.dirichlet(np.ones(200) * 0.3)
        top = np.argsort(-p)[:32]
        lp, lq = np.log(p[top])[None], np.log(q[top])[None]
        full = float((p * (np.log(p) - np.log(q))).sum())
        ck = float(Q.coarse_kl(lp, lq)[0])
        assert 0 <= ck <= full + 1e-9, (ck, full)
        assert Q.coarse_kl(lp, lp)[0] < 1e-12


@check("q", "Q1 rule thresholds incl. zero floors; Q2 near-tie count, degeneration, first divergence")
def t_q_rules():
    import e0_q as Q
    f = {"D_pp": 0.5, "KL": 1e-3}
    z = {"D_pp": 0.0, "KL": 0.0}
    assert Q.q1_rule({"D_pp": 0.7, "KL": 2e-3}, z, f)["pass"]
    assert not Q.q1_rule({"D_pp": 0.71, "KL": 1e-4}, z, f)["pass"]
    assert not Q.q1_rule({"D_pp": 0.1, "KL": 2.1e-3}, f, z)["pass"]
    assert not Q.q1_rule({"D_pp": 0.0, "KL": 2e-6}, z, z)["pass"] and Q.q1_rule({"D_pp": 0.0, "KL": 0.0}, z, z)["pass"]
    assert Q.first_divergence([1, 2, 3], [1, 2, 3]) is None and Q.first_divergence([1, 2, 3], [1, 5, 3]) == 1
    assert Q.first_divergence([1, 2], [1, 2, 9]) == 2
    assert Q.is_degenerate([7, 8] * 150, 1000)[0] and not Q.is_degenerate(list(range(300)), 1000)[0]
    assert Q.is_degenerate([1, 2, 5000], 1000)[0]
    ref = [{"key": f"k{i}", "tokens": list(range(10)), "margins": [0.1] * 5 + [2.0] * 5} for i in range(30)]
    xs = [{"key": f"k{i}", "tokens": list(range(10))} for i in range(27)] + \
         [{"key": f"k{i}", "tokens": list(range(7)) + [99, 98, 97]} for i in range(27, 30)]
    r = Q.q2_rule(ref, xs, 1000)
    assert r["near_tie_or_none"] == 27 and r["pass"], r["near_tie_or_none"]
    xs[0] = {"key": "k0", "tokens": list(range(6)) + [0, 0, 0, 0]}
    assert not Q.q2_rule(ref, xs, 1000)["pass"]
    xs[0] = {"key": "k0", "tokens": [3] * 250}
    ref[0] = dict(ref[0], tokens=list(range(250)), margins=[0.1] * 250)
    r = Q.q2_rule(ref, xs, 1000)
    assert r["degenerate_only_split"] == ["k0"] and not r["pass"]


# --------------------------------------------------------------------------- #
# f                                                                           #
# --------------------------------------------------------------------------- #
@check("f", "U-max search finds the largest passing k from any start; None if (1,1) fails")
def t_f_search():
    import e0_f as F
    for true in (None, 1, 3, 5, 9, 10):
        for k0 in (1, 3, 5, 9, 12):
            calls = []

            def attempt(k):
                calls.append(k)
                return true is not None and k <= true
            assert F.umax_search(k0, attempt) == true, (true, k0, calls)
            assert len(calls) == len(set(calls))


@check("f", "edge-fit keeps the S5 profile only from the passing (last) attempt; others set aside")
def t_f_settle():
    import e0_f as F
    tmp = Path(tempfile.mkdtemp(prefix="e0settle_"))
    for passed, n in ((True, 2), (False, 2), (True, 1)):
        for f in tmp.iterdir():
            f.unlink()
        for i in range(n):
            (tmp / f"attempt{i}-P-edge-Qwen3-8B-AWQ-cap12.jsonl").write_text(f"{i}\n")
        F.settle_profiles(tmp, "Qwen3-8B-AWQ", 12, n, passed)
        names = sorted(f.name for f in tmp.iterdir())
        canon = tmp / "P-edge-Qwen3-8B-AWQ-cap12.jsonl"
        assert canon.exists() == passed, names
        if passed:
            assert canon.read_text() == f"{n - 1}\n"
        assert sum(x.endswith(".rejected") for x in names) == n - passed, names
        assert not [x for x in names if x.startswith("attempt") and x.endswith(".jsonl")], names
    shutil.rmtree(tmp, ignore_errors=True)


@check("f", "failure classes: memory vs harness/engine; summarize marks invalid Stage 1 and blocked classes")
def t_f_classify():
    import e0_f as F
    for t in ("OutOfMemoryError: CUDA out of memory. Tried to allocate 560.00 MiB",
              "RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`",
              "ValueError: No available memory for the cache blocks. Try increasing `gpu_memory_utilization`",
              "ValueError: To serve at least one request ... (2.1 GiB KV cache is needed, which is larger than the "
              "available KV cache memory (1.3 GiB)"):
        assert F.failure_kind(t) == "memory", t
    for t in ("AttributeError: 'NoneType' object has no attribute 'cfg'", "CacheOverflow: 17921 > 17920",
              "TransportError: peer closed the connection", "server did not start: ImportError: libcuda", None):
        assert F.failure_kind(t) == "other", t
    tmp = Path(tempfile.mkdtemp(prefix="e0sum_"))
    for m, k in ((12, 3), (16, 5), (24, 9)):
        (tmp / f"umax_c{m}.json").write_text(json.dumps({"u_max": [k, k]}))
    (tmp / "split_fit_c12_k1.json").write_text(json.dumps({"pass": True}))
    (tmp / "edge_fit_Qwen3-8B-AWQ_c12.json").write_text(json.dumps({"pass": True}))
    (tmp / "edge_fit_Qwen3-14B-AWQ_c16.json").write_text(json.dumps({"pass": False, "fail_kind": "engine"}))
    (tmp / "edge_fit_Qwen3-32B-AWQ_c12.json").write_text(json.dumps({"pass": False, "fail_kind": "engine"}))
    with contextlib.redirect_stdout(io.StringIO()):
        F.summarize(argparse.Namespace(dir=str(tmp)))
    r = json.loads((tmp / "f_results.json").read_text())
    # 14B-AWQ is desk-eligible at 16 GB -> class blocked; 32B-AWQ is not desk-eligible at 12 GB -> no block
    assert r["valid"] and r["no_split_pass_classes"] == [16] and r["edge_eligible"]["12"] == ["Qwen3-8B-AWQ"], r
    (tmp / "split_fit_c16_k6.json").write_text(json.dumps({"pass": False, "error_kind": "harness"}))
    with contextlib.redirect_stdout(io.StringIO()):
        F.summarize(argparse.Namespace(dir=str(tmp)))
    r = json.loads((tmp / "f_results.json").read_text())
    assert not r["valid"] and r["harness_defects"] == ["split_fit_c16_k6.json"]
    shutil.rmtree(tmp, ignore_errors=True)


@check("f", "sampler: device-view peak minus baseline, trapezoid energy")
def t_f_sampler():
    import e0_instrument as I

    class Fake:
        def __init__(self):
            self.t = 0

        def used_bytes(self):
            self.t += 1
            return (1000 + (500 if self.t == 5 else 0)) * I.MIB

        def power_w(self):
            return 100.0

        def sm_clock_mhz(self):
            return 1500
    s = I.Sampler(Fake(), 0.01, baseline_bytes=1000 * I.MIB)
    s.start()
    time.sleep(0.2)
    summ = s.stop()
    assert summ["peak_used_mib"] == 500 and abs(summ["mean_power_w"] - 100) < 1e-9
    span = s.samples[-1][0] - s.samples[0][0]
    assert abs(summ["energy_j"] - 100 * span) < 1e-6


@check("f", "split-fit CLI, CPU dry run: load, depth prefill (17,664 tok, chunked), 256 decode, JSON record")
def t_f_dry():
    d = tiny_dir("bfloat16", vocab=151936, layers=6)
    out = Path(tempfile.mkdtemp(prefix="e0f_")) / "fit.json"
    r = subprocess.run([sys.executable, str(HERE / "e0_f.py"), "split-fit", "--model-dir", str(d), "--class", "12",
                        "--k", "2", "--device", "cpu", "--out", str(out), "--expect-32b", "0", "--cpu-dry"],
                       capture_output=True, text=True)
    rec = json.loads(out.read_text())
    assert r.returncode == 0 and rec["pass"] and rec["dry"], (r.stderr[-800:], rec.get("error"))
    assert rec["final_context"] == S.L_F and rec["kv_mib"] > 0 and rec["weights_mib"] > 0


# --------------------------------------------------------------------------- #
# latency                                                                     #
# --------------------------------------------------------------------------- #
@check("latency", "request plans: 2 warm-ups + 30 (36 with depth); seeded order; all cells scheduled once")
def t_lat_plan():
    import e0_latency as L
    w = L.load_workload()
    a = L.requests_for("core-02", "core", w)
    b = L.requests_for("rtt-01", "rtt", w)
    assert len(a) == 2 + 12 * 3 and len(b) == 2 + 10 * 3
    assert all(r["warmup"] for r in a[:2]) and not any(r["warmup"] for r in a[2:])
    assert a == L.requests_for("core-02", "core", w)
    assert sorted(c["id"] for c in L.execution_order()) == sorted(c["id"] for c in S.cells())


@check("latency", "netem commands and state parsing (both directions, delay = RTT/2, rate)")
def t_lat_netem():
    import e0_latency as L
    cmds = L.netem_commands("eth0", 20, 100)
    assert "sudo tc qdisc add dev eth0 root netem delay 10ms rate 100mbit" in cmds
    assert "sudo tc qdisc add dev ifb0 root netem delay 10ms rate 100mbit" in cmds
    show = ("qdisc netem 8001: root refcnt 2 limit 1000 delay 10ms rate 100Mbit\n Sent 1 bytes\n"
            "qdisc netem 8002: root refcnt 2 limit 1000 delay 10ms rate 100Mbit\n Sent 1 bytes")
    assert L.netem_matches(show, 20, 100) and not L.netem_matches(show, 50, 100)
    assert not L.netem_matches(show.replace("rate 100Mbit", ""), 20, 100)
    assert L.netem_matches("qdisc fq_codel 0: root\nqdisc noqueue 0: root", None, 1000)
    assert L.netem_commands("eth0", None, 1000, "apply") == L.netem_commands("eth0", None, None, "clear")


@check("latency", "SPLIT cell runner over TCP writes complete per-token/per-step records (tiny model, workload)")
def t_lat_split():
    import split_runtime as R
    import e0_latency as L
    d = tiny_dir("bfloat16", vocab=151936, layers=6)
    tmp = Path(tempfile.mkdtemp(prefix="e0lat_"))
    port = start_server(R.SegmentServer(R.Slice(d, list(range(1, 5)), False, False, "cpu"), "tiny", S.L_F))
    args = argparse.Namespace(device="cpu", cap_class=None, expect_32b=False, model_dir=str(d), model_id="tiny",
                              server=f"127.0.0.1:{port}", shaping="app", k=1, j=1, workload=str(L.WORKLOAD_JSON),
                              with_depth=True, dev="lo")
    cell = {"id": "heldout-x", "block": "heldout", "rtt_ms": 2, "bw_mbps": 1000, "load": 0}
    out = tmp / "c.jsonl"
    assert L.run_split(args, cell, out) == 0
    rows = [r for r in L.I.read_jsonl(out) if "prompt_id" in r]
    assert len(rows) == 2 + 12 * 3 and all(r["ok"] for r in rows)
    r = next(r for r in rows if r["group"] == "depth")
    assert len(r["token_ns"]) == S.GEN_TOKENS and r["n_in"] == S.L_F - S.GEN_TOKENS
    assert len(r["steps"]) == math.ceil(r["n_in"] / S.PREFILL_CHUNK) + S.GEN_TOKENS - 1
    assert r["steps"][-1]["server_seen"] == S.L_F - 1
    assert all(s["shape_ns"] >= 2e6 for s in r["steps"])
    assert L.run_split(args, cell, out) == 0 and len([x for x in L.I.read_jsonl(out) if "prompt_id" in x]) == 38
    shutil.rmtree(tmp, ignore_errors=True)


@check("latency", "linkcheck over the split protocol against a local echo stub")
def t_lat_link():
    import split_runtime as R
    import e0_latency as L
    port = start_server(R.SegmentServer(None, "echo", 64, echo=True))
    with contextlib.redirect_stdout(io.StringIO()):
        res = L.linkcheck(argparse.Namespace(server=f"127.0.0.1:{port}", n=200, n_big=3, out=None))
    assert res["rt_ms_p50"] > 0 and res["bulk_mbps_median"] > 0 and res["joint_allowed"], res


class _FakeVLLM(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        n_in, n = len(body["prompt"]), body["max_tokens"]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i in range(1, n + 1):
            step = 2 if i % 50 == 0 and i < n else 1      # occasional coalesced chunk
            if step == 2:
                continue
            ev = {"choices": [{"text": "x", "finish_reason": "length" if i == n else None}],
                  "usage": {"prompt_tokens": n_in, "completion_tokens": i, "total_tokens": n_in + i}}
            self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")


@check("latency", "streaming client counts tokens from continuous usage (coalesced chunks), app RTT adjust")
def t_lat_openai():
    import e0_latency as L
    port = free_port()
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), _FakeVLLM)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    tmp = Path(tempfile.mkdtemp(prefix="e0oai_"))
    args = argparse.Namespace(workload=str(L.WORKLOAD_JSON), served_model="m", base_url=f"http://127.0.0.1:{port}/v1",
                              edge_class=None, shaping="app", with_depth=False, nvml=False, dev="lo")
    cell = {"id": "rtt-x", "block": "rtt", "rtt_ms": 20, "bw_mbps": 1000, "load": 0}
    assert L.run_openai(args, cell, tmp / "o.jsonl", "REMOTE") == 0
    rows = [r for r in L.I.read_jsonl(tmp / "o.jsonl") if "prompt_id" in r]
    assert len(rows) == 32 and all(r["ok"] for r in rows), rows[0]
    assert all(len(r["token_ns"]) == S.GEN_TOKENS and r["app_adjust_ns"] == 20_000_000 for r in rows)
    httpd.shutdown()
    st = tmp / "load.json"
    L.I.write_json(st, {"concurrency": 8, "uptime_s": 120})
    assert L.check_load(st, 8)[0] and not L.check_load(st, 32)[0] and L.check_load(st, 0)[0]
    shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# r                                                                           #
# --------------------------------------------------------------------------- #
@check("r", "Profile.T closed form == explicit sum; addition; extrapolation")
def t_r_profile():
    import numpy as np
    import e0_r as Rr
    p = Rr.Profile([256, 2048], [0.2, 1.0], 0.03, 1e-6)
    for n_in, N in ((100, 1), (300, 57), (1000, 4000)):
        explicit = p.pre(np.array([n_in]))[0] + sum(0.03 + 1e-6 * (n_in + i) for i in range(1, N))
        assert abs(p.T(n_in, N) - explicit) < 1e-9
    q = p + Rr.Profile([64], [0.01], 0.01, 0.0)
    assert abs(q.T(300, 10) - p.T(300, 10) - 0.01 - 9 * 0.01) < 1e-12
    assert abs(p.pre(np.array([0.0]))[0] - (0.2 - 256 * 0.8 / 1792)) < 1e-12


@check("r", "stratified bootstrap keeps per-benchmark counts; identical actions give delta 0 and fail")
def t_r_boot():
    import numpy as np
    import e0_r as Rr
    d = Rr.S1Data()
    W = Rr.boot_weights(d.strata, 500)
    for s in range(4):
        assert np.all(W[:, d.strata == s].sum(1) == (d.strata == s).sum())
    prof = Rr.Profile([256, 2048], [0.1, 0.5], 0.02, 0.0)
    cells = Rr.evaluate_grid(d, {(12, "good", None): prof}, {}, {("good", None): prof}, {12: []}, W,
                             [(12, "open", "good", None, 600)])
    c = cells[0]
    assert c["status"] == "evaluated" and abs(c["delta_pt"]) < 1e-9 and not c["pass"]
    cells = Rr.evaluate_grid(d, {(12, "good", None): prof}, {}, {}, {12: []}, W, [(12, "no-raw-text", "good", None, 600)])
    assert cells[0]["status"].startswith("excluded: no admissible EDGE")


@check("r", "grid: SPLIT strictly better on-time passes; competitor faster at short D wins; open fails")
def t_r_grid():
    import e0_r as Rr
    d = Rr.S1Data()
    W = Rr.boot_weights(d.strata, 2000)
    fast = Rr.Profile([256, 2048], [0.05, 0.2], 0.01, 0.0)
    slow = Rr.Profile([256, 2048], [0.5, 2.0], 0.20, 0.0)
    grid = [(12, "no-raw-text", "good", None, 600), (12, "no-raw-text", "good", None, 30), (12, "open", "good", None, 600)]
    cells = Rr.evaluate_grid(d, {(12, "good", None): fast}, {("Qwen3-8B-AWQ", 12): fast}, {("good", None): fast},
                             {12: ["Qwen3-8B-AWQ"]}, W, grid)
    assert cells[0]["pass"] and cells[0]["delta_pt"] > 10, cells[0]["delta_pt"]
    assert not cells[2]["pass"]
    cells = Rr.evaluate_grid(d, {(12, "good", None): slow}, {("Qwen3-8B-AWQ", 12): fast}, {}, {12: ["Qwen3-8B-AWQ"]},
                             W, grid[1:2])
    assert cells[0]["delta_pt"] < 0 and not cells[0]["pass"]


@check("r", "upper-bound gate dominates the true delta in every cell and replicate (monotonicity)")
def t_r_ub():
    import numpy as np
    import e0_r as Rr
    d = Rr.S1Data()
    W = Rr.boot_weights(d.strata, 1000)
    true = Rr.Profile([256, 2048], [0.3, 1.2], 0.06, 2e-7)
    lb = Rr.Profile([256, 2048], [0.2, 0.9], 0.04, 1e-7)
    edge = {("Qwen3-8B-AWQ", 12): Rr.Profile([256, 2048], [0.05, 0.3], 0.011, 1e-7)}
    grid = [(12, "no-raw-text", "good", None, D) for D in S.R_DEADLINES_S]
    a = Rr.evaluate_grid(d, {(12, "good", None): true}, edge, {}, {12: ["Qwen3-8B-AWQ"]}, W, grid)
    b = Rr.evaluate_grid(d, {(12, "good", None): lb}, edge, {}, {12: ["Qwen3-8B-AWQ"]}, W, grid)
    for x, y in zip(a, b):
        assert y["delta_pt"] >= x["delta_pt"] - 1e-9 and y["ci95"][0] >= x["ci95"][0] - 1e-9
    g = Rr.upper_bound_gate(d, {(12, 3): lb}, {(12, 3): Rr.Profile([64], [0.0], 0.0, 0.0)}, edge,
                            {12: ["Qwen3-8B-AWQ"]}, W, {12: (3, 3)})
    assert isinstance(g["R_possible"], bool) and len(g["cells"]) == 30


@check("r", "upper-bound STOP rule: never STOP while a cell could pass or trigger §8.5; INCOMPLETE; open bound")
def t_r_ub_stop():
    import e0_r as Rr
    d = Rr.S1Data()
    W = Rr.boot_weights(d.strata, 2000)
    zero = Rr.Profile([64.0], [0.0], 0.0, 0.0, "instant")
    seg = {(16, 5): zero}
    e16 = {("Qwen3-8B-AWQ", 16): zero, ("Qwen3-14B-AWQ", 16): zero}
    # 16 GB with every action instantaneous: S1 pooled quality gap = +3.25 pt with CI low < 0. The old rule
    # (STOP unless a bound cell passes) would STOP here although a real cell could trigger the extension.
    g = Rr.upper_bound_gate(d, seg, seg, e16, {16: ["Qwen3-8B-AWQ", "Qwen3-14B-AWQ"]}, W, {16: (5, 5)})
    ev = [c for c in g["cells"] if c["status"] == "evaluated"]
    assert not any(c["pass"] for c in ev) and g["max_delta_bound_pt"] >= S.R_MARGIN_PT, g["max_delta_bound_pt"]
    assert min(c["ci95"][0] for c in ev) <= 0 and g["decision"].startswith("CONTINUE") and g["R_possible"]
    assert "NOT evidence" in g["decision"]
    # 24 GB with 32B-AWQ instantaneous: max bound delta +2.5 pt < 3.0 => STOP is safe
    seg24 = {(24, 9): zero}
    e24 = {(m, 24): zero for m in S.EDGE_ELIGIBLE_DESK[24]}
    g = Rr.upper_bound_gate(d, seg24, seg24, e24, {24: S.EDGE_ELIGIBLE_DESK[24]}, W, {24: (9, 9)})
    assert g["max_delta_bound_pt"] < S.R_MARGIN_PT and g["decision"].startswith("STOP") and not g["R_possible"]
    # a class with a U-max but no segment profile: INCOMPLETE, never STOP
    g = Rr.upper_bound_gate(d, {}, seg24, e24, {24: S.EDGE_ELIGIBLE_DESK[24]}, W, {24: (9, 9)})
    assert g["decision"].startswith("INCOMPLETE") and g["missing_profiles"] == [24] and g["R_possible"]
    # no EDGE competitor fits: no-raw-text excluded, but the open cell is bounded with REMOTE at U = 0
    g = Rr.upper_bound_gate(d, {(12, 3): zero}, {(12, 3): zero}, {}, {12: []}, W, {12: (3, 3)})
    op = [c for c in g["cells"] if c["M"] == 12 and c["privacy"] == "open" and c["status"] == "evaluated"]
    assert len(op) == 5 and g["decision"].startswith("CONTINUE"), [c["status"] for c in g["cells"]][:12]
    # the measured-segment discount only shortens times
    p = Rr.Profile([256, 2048], [0.2, 1.0], 0.03, 1e-6)
    q = Rr.scaled(p, S.UB_SEGMENT_DISCOUNT)
    assert 0 < S.UB_SEGMENT_DISCOUNT < 1 and abs(q.T(500, 300) - S.UB_SEGMENT_DISCOUNT * p.T(500, 300)) < 1e-12


def _synthetic_logs(tmp, coef, perturb_heldout=0.0, seed=0):
    """E0-L logs generated from a known additive model (+ noise) for every SPLIT/REMOTE cell."""
    rng = random.Random(seed)
    H = S.QWEN3_32B_CONFIG["hidden_size"]
    umax = {12: (3, 3), 16: (5, 5), 24: (9, 9)}
    for c in S.cells():
        if c["path"].startswith("EDGE") or c["role"] != "decision":
            continue
        rows = []
        is_split = c["path"].startswith("SPLIT")
        if is_split:
            kj = S.U_MIN if c["path"] == "SPLIT U-min" else umax[int(re.search(r"\((\d+)\)", c["path"]).group(1))]
            Le, Ls = sum(kj), 64 - sum(kj)
        r_load = {0: 1.0, 8: 1.3, 32: 1.9}[c["load"]]
        ho = 1 + (perturb_heldout if c["block"] == "heldout" else 0.0)
        bw = c["bw_mbps"] * 1e6
        for p in (("short", 256), ("long", 2048)) + ((("depth", S.L_F - 256),) if c["block"] == "core" else ()):
            for rep in range(3):
                n_in = p[1]
                steps, t, stamps = [], 0.0, []
                if is_split:
                    for s0 in range(0, n_in, S.PREFILL_CHUNK):
                        n = min(S.PREFILL_CHUNK, n_in - s0)
                        e = coef["pe0"] + coef["pe1"] * Le * n
                        sv = r_load * (coef["ps0"] + coef["ps1"] * Ls * n)
                        x = coef["px0"] + coef["px1"] * n
                        by = 2 * (S.WIRE_HEADER_BYTES + n * H * 2)
                        net = coef["l0"] + (c["rtt_ms"] or 0) / 1e3 + by * 8 / bw
                        steps.append({"n": n, "start": s0, "edge_head_ns": e * 0.5e9, "edge_tail_ns": e * 0.5e9,
                                      "lm_head_ns": 0, "server_compute_ns": sv * 1e9, "server_ns": (sv + x / 2) * 1e9,
                                      "ser_ns": x / 2 * 1e9, "net_ns": net * 1e9, "shape_ns": 0,
                                      "bytes_out": by / 2, "bytes_in": by / 2})
                        t += e + sv + x + net
                    stamps.append(t)
                    for i in range(1, S.GEN_TOKENS):
                        ctx = n_in + i
                        e = coef["e0"] + coef["e1"] * Le + coef["e2"] * Le * ctx
                        sv = r_load * (coef["s0"] + coef["s1"] * Ls + coef["s2"] * Ls * ctx)
                        x = coef["x"]
                        by = 2 * (S.WIRE_HEADER_BYTES + H * 2)
                        net = coef["l0"] + (c["rtt_ms"] or 0) / 1e3 + by * 8 / bw
                        tot = (e + sv + x + net) * ho * (1 + rng.gauss(0, 0.01))
                        steps.append({"n": 1, "start": ctx - 1, "edge_head_ns": e * 0.5e9, "edge_tail_ns": e * 0.4e9,
                                      "lm_head_ns": e * 0.1e9, "server_compute_ns": sv * 1e9,
                                      "server_ns": (sv + x / 2) * 1e9, "ser_ns": x / 2 * 1e9, "net_ns": net * 1e9,
                                      "shape_ns": 0, "bytes_out": by / 2, "bytes_in": by / 2})
                        t += tot
                        stamps.append(t)
                else:
                    t = coef["r_pre"] + (c["rtt_ms"] or 0) / 1e3
                    stamps.append(t)
                    for i in range(1, S.GEN_TOKENS):
                        t += (coef["ra"] + coef["rb"] * (n_in + i)) * r_load * ho * (1 + rng.gauss(0, 0.01))
                        stamps.append(t)
                row = {"cell": c["id"], "block": c["block"], "path": "SPLIT" if is_split else "REMOTE",
                       "rtt_ms": c["rtt_ms"], "bw_mbps": c["bw_mbps"], "load": c["load"], "n_in": n_in,
                       "prompt_id": f"{p[0]}-{rep}", "group": p[0], "rep": rep, "warmup": False, "ok": True,
                       "token_ns": [int(s * 1e9) for s in stamps]}
                if is_split:
                    row.update(k=kj[0], j=kj[1], steps=steps)
                rows.append(row)
        with open(tmp / f"{c['id']}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    (tmp / "f_results.json").write_text(json.dumps({"u_max": {str(k): list(v) for k, v in umax.items()}}))


COEF = {"pe0": 0.002, "pe1": 2.5e-7, "ps0": 0.003, "ps1": 6e-8, "px0": 0.001, "px1": 2e-6,
        "e0": 0.004, "e1": 0.0015, "e2": 2e-9, "s0": 0.002, "s1": 0.0003, "s2": 1.2e-9, "x": 0.0008, "l0": 0.0004,
        "r_pre": 0.15, "ra": 0.025, "rb": 1.5e-7}


@check("r", "additive model recovers synthetic coefficients; held-out validation passes, fails at +20%")
def t_r_additive():
    import numpy as np
    import e0_r as Rr
    tmp = Path(tempfile.mkdtemp(prefix="e0fit_"))
    _synthetic_logs(tmp, COEF)
    b = Rr.fit_all(tmp, tmp / "f_results.json")
    am = b["additive"]
    assert abs(am["e"][1] - COEF["e1"]) / COEF["e1"] < 0.02 and abs(am["s"][1] - COEF["s1"]) / COEF["s1"] < 0.02
    assert abs(am["r"][32] - 1.9) < 0.05 and abs(am["r"][8] - 1.3) < 0.05, am["r"]
    assert b["validation"]["valid"], b["validation"]
    sp, src = Rr.split_profiles_from_bundle(b)
    assert src.startswith("additive") and (12, "contended32", 20) in sp
    ref = sp[(16, "good", None)]
    meas = Rr.Profile.from_json(b["measured"]["core-02"]["profile"])
    assert abs(ref.T(500, 2000) / meas.T(500, 2000) - 1) < 0.03
    tmp2 = Path(tempfile.mkdtemp(prefix="e0fit2_"))
    _synthetic_logs(tmp2, COEF, perturb_heldout=0.2)
    b2 = Rr.fit_all(tmp2, tmp2 / "f_results.json")
    assert not b2["validation"]["valid"]
    sp2, src2 = Rr.split_profiles_from_bundle(b2)
    assert src2.startswith("measured cells only") and set(sp2) == {(16, "good", None), (16, "good", 20),
                                                                   (16, "contended32", None)}
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(tmp2, ignore_errors=True)


@check("r", "R CLIs on synthetic stage dirs: gate-ub1, gate-ub2, fit, r, decide")
def t_r_cli():
    tmp = Path(tempfile.mkdtemp(prefix="e0cli_"))
    st1, st2, lat = tmp / "stage1", tmp / "stage2", tmp / "lat"
    for d in (st1, st2, lat):
        d.mkdir()
    _synthetic_logs(lat, COEF)
    fres = {"u_max": {"12": [3, 3], "16": [5, 5], "24": [9, 9]},
            "edge_eligible": {"12": ["Qwen3-8B-AWQ"], "16": ["Qwen3-8B-AWQ", "Qwen3-14B-AWQ"],
                              "24": ["Qwen3-8B-AWQ", "Qwen3-14B-AWQ"]}}
    (st1 / "f_results.json").write_text(json.dumps(fres))
    # EDGE competitor profiles (end-to-end rows) and edge/server segment profiles (component rows)
    core = [json.loads(x) for x in open(lat / "core-03.jsonl")]
    for m, cap, scale in (("Qwen3-8B-AWQ", 12, 0.35), ("Qwen3-8B-AWQ", 16, 0.35), ("Qwen3-14B-AWQ", 16, 0.6),
                          ("Qwen3-8B-AWQ", 24, 0.35), ("Qwen3-14B-AWQ", 24, 0.6)):
        with open(st1 / f"P-edge-{m}-cap{cap}.jsonl", "w") as f:
            for r in core:
                f.write(json.dumps(dict(r, token_ns=[int(t * scale) for t in r["token_ns"]])) + "\n")
    split_rows = [json.loads(x) for x in open(lat / "heldout-01.jsonl")] + \
        [json.loads(x) for x in open(lat / "core-02.jsonl")] + [json.loads(x) for x in open(lat / "heldout-02.jsonl")]
    for k in (3, 5, 9):
        rows = [r for r in split_rows if r["k"] == k] or [dict(r, k=k, j=k) for r in split_rows[:6]]
        with open(st1 / f"P-edge-U({k},{k})-cap12.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        with open(st2 / f"P-server-U({k},{k})-load0.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    py = [sys.executable, str(HERE / "e0_r.py")]
    run = lambda *a: subprocess.run(py + list(a), check=True, capture_output=True, text=True).stdout
    for g in ("gate-ub1", "gate-ub2"):
        extra = ["--stage2-dir", str(st2)] if g == "gate-ub2" else []
        run(g, "--stage1-dir", str(st1), "--out", str(tmp / f"{g}.json"), *extra)
        gj = json.loads((tmp / f"{g}.json").read_text())
        assert gj["decision"].startswith(("CONTINUE", "STOP")) and len(gj["cells"]) == 30
        assert gj["segment_discount"] == S.UB_SEGMENT_DISCOUNT and gj["missing_profiles"] == []
        assert all(c["status"] == "evaluated" for c in gj["cells"]), [c["status"] for c in gj["cells"]][:3]
    run("fit", "--lat-dir", str(lat), "--f-results", str(st1 / "f_results.json"), "--out", str(tmp / "bundle.json"))
    run("r", "--bundle", str(tmp / "bundle.json"), "--stage1-dir", str(st1), "--out-dir", str(tmp))
    rc = json.loads((tmp / "R_cells.json").read_text())
    assert rc["source"].startswith("additive") and rc["summary"]["n_cells"] == 120 and rc["sample_n"] == 200
    assert rc["summary"]["n_evaluated"] >= 100, rc["summary"]
    (tmp / "q.json").write_text(json.dumps({"pass": True}))
    run("decide", "--f-results", str(st1 / "f_results.json"), "--lat-dir", str(lat), "--q", str(tmp / "q.json"),
        "--r", str(tmp / "R_cells.json"), "--out-dir", str(tmp))
    dj = json.loads((tmp / "DECISION.json").read_text())
    assert dj["E0"] in ("PASS", "FAIL", "PENDING") and dj["F"] is True and "consequence" in dj
    shutil.rmtree(tmp, ignore_errors=True)


@check("r", "decision table: Q/F/R combinations map to the E0 §8 consequences")
def t_r_decide():
    import e0_r as Rr
    tmp = Path(tempfile.mkdtemp(prefix="e0dec_"))
    (tmp / "lat").mkdir()
    f = tmp / "f.json"
    f.write_text(json.dumps({"u_max": {"12": [3, 3], "16": None, "24": None}}))

    def run(Q, R, only12=True, open_=False, err=False, best=None, n=200):
        (tmp / "q.json").write_text(json.dumps({"pass": Q}))
        (tmp / "r.json").write_text(json.dumps({"sample_n": n, "summary": {
            "R_pass": R, "only_12GB": only12, "open_pass": open_, "pass_robust_to_multiplicity": R,
            "pass_classes": [12] if (R and only12) else ([12, 24] if R else []), "best": best}}))
        lat = tmp / "lat" / "x.jsonl"
        lat.write_text(json.dumps({"prompt_id": "a", "ok": not err, "error": "OOM" if err else None}) + "\n")
        a = argparse.Namespace(f_results=str(f), lat_dir=[str(tmp / "lat")], q=str(tmp / "q.json"), r=str(tmp / "r.json"),
                               out_dir=str(tmp))
        with contextlib.redirect_stdout(io.StringIO()):
            Rr.decide(a)
        return json.loads((tmp / "DECISION.json").read_text())
    x = run(True, True)
    assert x["E0"] == "PASS" and "no-raw-text quality tier" in x["consequence"] and "12 GB" in x["consequence"]
    assert "EMULATED" in x["memory_class_scope"]
    straddle = {"delta_pt": 3.4, "ci95": [-0.3, 7.1]}
    e = run(True, False, best=straddle)
    assert e["E0"] == "PENDING" and e["extension_triggered"] and "400-query extension" in e["consequence"]
    assert run(True, False, best=straddle, n=400)["E0"] == "FAIL"          # the extension is used at most once
    assert run(True, False, best={"delta_pt": 2.9, "ci95": [-0.3, 6.0]})["E0"] == "FAIL"
    assert run(False, False, best=straddle)["E0"] == "FAIL"                 # Q or F failures are never pending
    assert run(True, False, best=straddle, err=True)["E0"] == "FAIL"
    (tmp / "r.json").write_text(json.dumps({"sample_n": 200, "summary": {
        "R_pass": True, "only_12GB": False, "open_pass": False, "pass_robust_to_multiplicity": True,
        "pass_classes": [24], "best": None}}))
    a = argparse.Namespace(f_results=str(f), lat_dir=[str(tmp / "lat")], q=str(tmp / "q.json"), r=str(tmp / "r.json"),
                           out_dir=str(tmp), a3b_added=False)
    (tmp / "lat" / "x.jsonl").write_text(json.dumps({"prompt_id": "a", "ok": True}) + "\n")
    (tmp / "q.json").write_text(json.dumps({"pass": True}))
    with contextlib.redirect_stdout(io.StringIO()):
        Rr.decide(a)
    z = json.loads((tmp / "DECISION.json").read_text())
    assert z["E0"] == "PENDING" and "30B-A3B" in z["consequence"], z          # §8.4 (a)
    assert "EMULATED" not in z["memory_class_scope"]
    assert run(False, True)["E0"] == "FAIL" and "SPLIT removed" in run(False, True)["consequence"]
    assert "lack of a regime" in run(True, False)["consequence"]
    y = run(True, True, err=True)
    assert y["E0"] == "FAIL" and not y["F"]
    assert "unexpected" in run(True, True, open_=True)["consequence"]
    shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# docs                                                                        #
# --------------------------------------------------------------------------- #
@check("docs", "every `e0_*.py <subcommand>` named in RUNBOOK.md exists")
def t_docs():
    rb = HERE / "RUNBOOK.md"
    if not rb.exists():
        raise Skip("RUNBOOK.md not written yet")
    text = rb.read_text()
    subs = {}
    for f in ("e0_q.py", "e0_f.py", "e0_latency.py", "e0_r.py", "split_runtime.py", "e0_selftest.py"):
        src = (HERE / f).read_text()
        names = set(re.findall(r'add_parser\("([a-z0-9-]+)"', src))
        for group in re.findall(r'for name in \(([^)]*)\)', src):
            names |= {x.strip().strip('"') for x in group.split(",") if x.strip()}
        subs[f] = names
    bad = []
    for f, sub in re.findall(r"(e0_[a-z]+\.py|split_runtime\.py) ([a-z][a-z0-9-]+)", text):
        if sub not in subs[f] and not sub.startswith("-"):
            bad.append((f, sub))
    assert not bad, bad


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-dir")
    ap.add_argument("--real-model-dir")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--report")
    args = ap.parse_args()
    import torch
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    tests = [v for v in globals().values() if callable(v) and hasattr(v, "_check")]
    t_all = time.time()
    for fn in tests:
        group, name = fn._check
        if args.only and group not in args.only:
            continue
        t0 = time.time()
        try:
            if fn.__code__.co_argcount:
                fn(args)
            else:
                fn()
            status, detail = "PASS", ""
        except Skip as e:
            status, detail = "SKIP", str(e)
        except Exception as e:
            status, detail = "FAIL", f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}"
        RESULTS.append({"group": group, "check": name, "status": status, "seconds": round(time.time() - t0, 1),
                        "detail": detail})
        print(f"[{status}] {group}: {name} ({time.time() - t0:.1f}s){(' -- ' + detail) if detail else ''}", flush=True)
    n = {s: sum(r["status"] == s for r in RESULTS) for s in ("PASS", "FAIL", "SKIP")}
    print(f"\n{n['PASS']} PASS, {n['FAIL']} FAIL, {n['SKIP']} SKIP in {time.time() - t_all:.0f}s")
    if args.report:
        import e0_instrument as I
        I.write_json(args.report, {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "counts": n,
                                   "results": RESULTS, "env": I.environment(torch)})
    sys.exit(1 if n["FAIL"] else 0)


if __name__ == "__main__":
    main()
