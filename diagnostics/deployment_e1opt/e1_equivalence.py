"""E1-OPT Stage 1b: computation-equivalence driver (PREREG_E1OPT §5). A correctness test, NOT a benchmark.

No latency, throughput, CUDA-event duration, tokens/s, wall clock or profiler output is produced, stored or
printed. `assert_no_timing_fields` is applied to every result object before it is written, and
tests/test_e1_equivalence.py checks the schema.

Subcommands (all require --freeze; the freeze checker runs first):
  reference      on a machine that holds the whole model: HF unsplit final hidden states and greedy tokens for the
                 check sequence(s) -> ref_hidden.pt (bf16) and, optionally, E0-format Q1 top-K npz and Q2 rows
  local          on one GPU, in-host (both segments on this device): prefix mode bitwise vs HF/E0, static mode
                 (frozen kernel + graph branch) vs prefix within tolerance and identical argmax, greedy tokens
                 vs HF, wire and ownership invariants, branch-in-use vs freeze. --segments-only for a machine
                 that cannot hold the whole model (the capped edge): segment-level checks only
  joint          edge <-> server over the link: Q1 teacher-forced comparison (E0 npz format) and Q2 greedy runs
                 (E0 row format) with the E1 runtime, for e0_q.decide / `decide` below
  decide         E0's Q1/Q2 rules (thresholds unchanged, e0_q.q1_rule / q2_rule) over the files above plus the
                 local-check JSONs -> stage1b_decision.json
"""
import json
import re
import socket
import sys
import threading
from pathlib import Path

import torch

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))   # python -I: the script dir is not on sys.path
import e1_spec as S
sys.path.insert(0, str(S.E0_DIR))
import split_runtime as E0RT  # noqa: E402
import e0_q as Q  # noqa: E402
import e1_attention  # noqa: E402
import e1_freeze  # noqa: E402
import e1_protocol as P  # noqa: E402
from e1_segment import USplitEdge, LocalMiddle, Segment, edge_slice, server_slice, ownership  # noqa: E402

PROHIBITED = re.compile(r"(_ns$|_ms$|_us$|(^|_)t[0-9]_|(^|_)time(_|$)|latenc|throughput|tok(en)?s?_per_s|per_sec|duration|"
                        r"elapsed|cuda_?event|tflop|speed|bandwidth|tpot|ttft|wall|_rate$|second|perf_counter|mbps|gbps)", re.I)
ALLOWED_KEYS = {"utc", "frozen_at_utc", "stage", "runtime_branch", "runtime"}


class TimingFieldError(ValueError):
    pass


class GraphCaptureError(RuntimeError):
    pass


def assert_no_timing_fields(obj, path="$"):
    """Raise TimingFieldError if any key anywhere in obj looks like a performance quantity."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            ks = str(k)
            if ks not in ALLOWED_KEYS and PROHIBITED.search(ks):
                raise TimingFieldError(f"{path}.{ks}")
            assert_no_timing_fields(v, f"{path}.{ks}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            assert_no_timing_fields(v, f"{path}[{i}]")


def device_identity(device):
    dev = torch.device(device)
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(dev)
        return {"name": p.name, "uuid": str(getattr(p, "uuid", "")), "index": dev.index if dev.index is not None else torch.cuda.current_device(),
                "compute_cap": float(f"{p.major}.{p.minor}"), "memory_mib": p.total_memory // (1 << 20)}
    return {"name": "cpu", "uuid": "cpu", "index": None, "compute_cap": None, "memory_mib": None}


def _setup_torch():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_split(model_dir, k, j, device, mode, max_len, expected_cfg, branch, in_host=True, remote=None):
    """USplitEdge with the frozen branch applied. in_host: server segment in this process (LocalMiddle)."""
    e1_attention.set_branch(branch["attention"])
    sl = edge_slice(model_dir, k, j, device, torch.bfloat16, expected_cfg)
    if mode == "static":
        sl.cfg._attn_implementation = "e1_static"
    if remote is not None:
        mid = remote
    elif in_host:
        ssl = server_slice(model_dir, k, j, device, torch.bfloat16, expected_cfg)
        if mode == "static":
            ssl.cfg._attn_implementation = "e1_static"
        mid = LocalMiddle(ssl, ssl.layer_ids, max_len, mode)
    else:
        raise ValueError("need a remote middle or in_host")
    e1 = USplitEdge(sl, k, j, mid, max_len, mode)
    if mode == "static" and branch["cuda_graphs"] == "on":
        try:
            e1.enable_graph()
            if hasattr(mid, "enable_graph"):
                mid.enable_graph()
        except Exception as e:  # a capture failure is a compatibility probe (PREREG §4.2): report, do not time
            raise GraphCaptureError(f"{type(e).__name__}: {e}") from e
    return e1


def branch_in_use(e1):
    dev = e1.sl.device
    graphs = "off"
    if e1.head.graph is not None:
        graphs = "on" if e1.head.graph.g is not None else ("cpu_control_flow" if dev.type != "cuda" else "off_capture_failed")
    attention = "sdpa_static_fallback"
    if e1_attention.kernel_available() and not e1_attention.FORCE_SDPA and dev.type == "cuda":
        attention = "flash_attn_with_kvcache"
    return {"attention": attention, "attention_last_kernel": e1_attention.LAST_KERNEL["name"], "cuda_graphs": graphs,
            "torch_compile": False}


def branch_matches(in_use, frozen, device):
    att_ok = in_use["attention"] == frozen["attention"]
    if torch.device(device).type == "cuda":
        g_ok = in_use["cuda_graphs"] == frozen["cuda_graphs"]
    else:
        g_ok = (in_use["cuda_graphs"] == "cpu_control_flow") == (frozen["cuda_graphs"] == "on")
    return bool(att_ok and g_ok and frozen.get("torch_compile") is False)


def hf_model(model_dir, device):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()


@torch.inference_mode()
def reference_hidden(hf, ids, device):
    return hf.model(torch.as_tensor(ids, dtype=torch.long, device=device).view(1, -1)).last_hidden_state[0]


@torch.inference_mode()
def reference_greedy(hf, ids, n, device):
    t = torch.as_tensor(ids, dtype=torch.long, device=device).view(1, -1)
    g = hf.generate(t, attention_mask=torch.ones_like(t), max_new_tokens=n, min_new_tokens=n, do_sample=False, pad_token_id=0, eos_token_id=None)
    return g[0, t.shape[1]:].tolist()


def wire_and_content_check(model_dir, k, j, device, mode, max_len, expected_cfg, branch, ids, n_new, chunk, freeze_sha):
    """Run a short greedy generation through the real wire against an in-process echo stub and audit the wire:
    byte totals equal the hidden-state formula, the HELLO carries only control keys, the server saw only
    HELLO/RESET/FWD/BYE. Nothing but the residual stream can have crossed."""
    sl = edge_slice(model_dir, k, j, device, torch.bfloat16, expected_cfg)
    H = sl.cfg.hidden_size
    events = []
    ident = {"freeze_sha256": freeze_sha, "config_hash": None, "k": k, "j": j, "model_id": None, "runtime": S.NAME}
    srv = P.SegmentServer(None, ident, H, chunk, log=events.append)
    ready = threading.Event()
    port_holder = {}
    import socket as _s
    s0 = _s.socket(); s0.bind(("127.0.0.1", 0)); port = s0.getsockname()[1]; s0.close()
    t = threading.Thread(target=srv.serve_forever, args=("127.0.0.1", port, ready, True), daemon=True); t.start(); ready.wait(10)
    hello = P.hello_for(sl, f"{S.E0.SPLIT_REPO}@{S.E0.SPLIT_REVISION}", k, j, max_len, freeze_sha)
    mid = P.RemoteMiddle("127.0.0.1", port, hello, device, H, chunk)
    e1 = USplitEdge(sl, k, j, mid, max_len, "prefix")
    toks = e1.generate(ids, n_new, chunk, force_len=True)
    mid.close(); t.join(10)
    n_in, N = len(ids), n_new
    c = -(-n_in // chunk)
    expect = H * 2 * (n_in + N - 1) + P.HDR.size * (c + N - 1)
    out_bytes = sum(s["bytes_out"] for s in e1.steps)
    in_bytes = sum(s["bytes_in"] for s in e1.steps)
    hello_keys_ok = set(hello) <= P.HELLO_KEYS
    ev_types = sorted({e["event"] for e in events})
    return {"wire_bytes_out_equal_formula": out_bytes == expect, "wire_bytes_in_equal_formula": in_bytes == expect,
            "wire_formula": "hidden*2*(n_in+N-1) + 72*(prefill_chunks+N-1) per direction", "hello_keys_subset_of_control_set": hello_keys_ok,
            "server_events": ev_types, "server_events_only_control": set(ev_types) <= {"hello_ok", "bye"},
            "server_fwd_count_equals_steps": srv.n_fwd == len(e1.steps), "n_tokens_generated": len(toks),
            "payload_dtype": "bfloat16", "tokens_or_text_on_wire": False}


def local_checks(model_dir, k, j, device, expected_cfg, branch, ids, max_len, n_greedy, chunk, freeze_sha,
                 ref_hidden=None, ref_tokens=None, segments_only=False):
    _setup_torch()
    out = {"mode_reference": "prefix", "mode_measured": "static", "k": k, "j": j, "n_check_tokens": len(ids), "segments_only": segments_only}
    dev = torch.device(device)
    if segments_only:
        # The capped edge cannot hold the whole model: compare its own head/tail segments prefix vs static, and
        # prefix vs E0's run_layers on the same slice, with synthetic residual inputs (bitwise / tolerance).
        e1_attention.set_branch(branch["attention"])
        sl_p = edge_slice(model_dir, k, j, device, torch.bfloat16, expected_cfg)
        sl_s = edge_slice(model_dir, k, j, device, torch.bfloat16, expected_cfg); sl_s.cfg._attn_implementation = "e1_static"
        g = torch.Generator().manual_seed(0)
        x = torch.randn((1, min(len(ids), chunk), sl_p.cfg.hidden_size), generator=g).to(torch.bfloat16).to(device)
        res = {}
        for name, lids in (("head", list(range(k))), ("tail", list(range(sl_p.n_layers - j, sl_p.n_layers)))):
            seg_p, seg_s = Segment(sl_p, lids, max_len, "prefix"), Segment(sl_s, lids, max_len, "static")
            if branch["cuda_graphs"] == "on":
                try:
                    seg_s.enable_graph()
                except Exception as e:
                    raise GraphCaptureError(f"{type(e).__name__}: {e}") from e
            yp = seg_p.forward(x, 0); ys = seg_s.forward(x, 0)
            c0 = sl_p.new_cache(max_len, lids); y0 = sl_p.run_layers(x, lids, 0, c0)
            # one decode step through the (graphed) static path at the next position
            x1 = torch.randn((1, 1, sl_p.cfg.hidden_size), generator=g).to(torch.bfloat16).to(device)
            yp1 = seg_p.forward(x1, x.shape[1]); ys1 = seg_s.forward(x1, x.shape[1])
            res[name] = {"prefix_bitwise_vs_e0_run_layers": bool(torch.equal(yp, y0)),
                         "static_max_abs_vs_prefix": float((ys.float() - yp.float()).abs().max()),
                         "static_decode_step_max_abs_vs_prefix": float((ys1.float() - yp1.float()).abs().max()),
                         "within_tolerance": bool((ys.float() - yp.float()).abs().max() <= S.EQ_STATIC_MAX_ABS_BF16 * 4
                                                  and (ys1.float() - yp1.float()).abs().max() <= S.EQ_STATIC_MAX_ABS_BF16 * 4)}
        out["segments"] = res
        out["ownership"] = ownership(sl_p, k, j)
        out["branch_in_use"] = {"attention": "flash_attn_with_kvcache" if e1_attention.kernel_available() and not e1_attention.FORCE_SDPA and dev.type == "cuda" else "sdpa_static_fallback",
                                "cuda_graphs": "on" if (branch["cuda_graphs"] == "on" and dev.type == "cuda") else ("cpu_control_flow" if dev.type != "cuda" else "off"), "torch_compile": False}
        out["branch_matches_freeze"] = branch_matches(out["branch_in_use"], branch, device)
        out["wire"] = wire_and_content_check(model_dir, k, j, device, "prefix", max_len, expected_cfg, branch, ids, n_greedy, chunk, freeze_sha)
        out["pass"] = bool(all(r["prefix_bitwise_vs_e0_run_layers"] and r["within_tolerance"] for r in res.values())
                           and out["ownership"]["edge_slice_ok"] and out["branch_matches_freeze"]
                           and all(v for kk, v in out["wire"].items() if isinstance(v, bool) and kk != "tokens_or_text_on_wire"))
        return out
    # Full in-host checks (this device holds the whole model)
    if ref_hidden is None or ref_tokens is None:
        hf = hf_model(model_dir, device)
        ref_hidden = reference_hidden(hf, ids, device)
        ref_tokens = reference_greedy(hf, ids, n_greedy, device)
        del hf
    ref_hidden = ref_hidden.to(device)
    # E0 runtime (read-only import) on the same device
    sl0 = E0RT.edge_slice(model_dir, k, j, device, torch.bfloat16, expected_cfg)
    ssl0 = E0RT.Slice(model_dir, list(range(k, sl0.n_layers - j)), False, False, device, torch.bfloat16, expected_cfg)
    r0 = E0RT.EdgeRunner(sl0, k, j, E0RT.LocalMiddle(ssl0, list(range(k, sl0.n_layers - j)), max_len), max_len=max_len)
    h0 = r0.teacher_forced_hidden(ids)
    del r0, sl0, ssl0
    e1p = build_split(model_dir, k, j, device, "prefix", max_len, expected_cfg, branch)
    hp = e1p.teacher_forced_hidden(ids, chunk)
    out["prefix_bitwise_vs_hf"] = bool(torch.equal(hp, ref_hidden))
    out["prefix_bitwise_vs_e0"] = bool(torch.equal(hp, h0))
    out["prefix_greedy_equals_hf"] = e1p.generate(ids, n_greedy, chunk, force_len=True) == ref_tokens
    del e1p
    e1s = build_split(model_dir, k, j, device, "static", max_len, expected_cfg, branch)
    hs = e1s.teacher_forced_hidden(ids, chunk)
    with torch.inference_mode():
        ap, as_ = e1s.sl.lm_head(hp).float().argmax(-1), e1s.sl.lm_head(hs).float().argmax(-1)
    out["static_max_abs_vs_prefix"] = float((hs.float() - hp.float()).abs().max())
    out["static_within_tolerance"] = out["static_max_abs_vs_prefix"] <= S.EQ_STATIC_MAX_ABS_BF16
    out["static_argmax_equal_prefix_share"] = float((ap == as_).float().mean())
    out["static_greedy_equals_hf"] = e1s.generate(ids, n_greedy, chunk, force_len=True) == ref_tokens
    out["branch_in_use"] = branch_in_use(e1s)
    out["branch_matches_freeze"] = branch_matches(out["branch_in_use"], branch, device)
    out["ownership"] = ownership(e1s.sl, k, j, e1s.middle.seg.layer_ids)
    out["ownership"]["server_has_no_embed_or_head"] = e1s.middle.seg.sl.embed is None and e1s.middle.seg.sl.lm_head is None
    out["wire"] = wire_and_content_check(model_dir, k, j, device, "prefix", max_len, expected_cfg, branch, ids, n_greedy, chunk, freeze_sha)
    out["pass"] = bool(out["prefix_bitwise_vs_hf"] and out["prefix_bitwise_vs_e0"] and out["prefix_greedy_equals_hf"]
                       and out["static_within_tolerance"] and out["static_greedy_equals_hf"] and out["branch_matches_freeze"]
                       and out["ownership"]["edge_slice_ok"] and out["ownership"]["server_slice_ok"] and out["ownership"]["server_has_no_embed_or_head"]
                       and all(v for kk, v in out["wire"].items() if isinstance(v, bool) and kk != "tokens_or_text_on_wire"))
    return out


# --------------------------------------------------------------------------- #
# Q1 / Q2 in E0's file formats over the E1 runtime                             #
# --------------------------------------------------------------------------- #
def q1_compare(hidden_fn, lm_head, cont_file, ref_dir, out_dir, limit=None):
    import numpy as np
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    n = 0
    for key, ids, pos in Q._sequences(cont_file, limit):
        f = out / f"{key.replace(':', '_')}.npz"
        if f.exists():
            n += 1; continue
        r = np.load(Path(ref_dir) / f"{key.replace(':', '_')}.npz")
        assert list(r["positions"]) == pos, key
        h = hidden_fn(ids)
        ag, kl, xm = Q.compare_to_ref(torch, lm_head, h, pos, r["ids"], r["lps"])
        np.savez_compressed(f, agree=ag, kl=kl, x_margin=xm, ref_margin=r["lps"][:, 0] - r["lps"][:, 1])
        n += 1
    return n


def q1_reference(hf, cont_file, out_dir, device, limit=None):
    import numpy as np
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for key, ids, pos in Q._sequences(cont_file, limit):
        f = out / f"{key.replace(':', '_')}.npz"
        if f.exists():
            continue
        h = reference_hidden(hf, ids, device)
        rid, rlp = Q.ref_topk(torch, hf.lm_head, h, pos)
        np.savez_compressed(f, ids=rid, lps=rlp, positions=np.array(pos, dtype=np.int32))


def q2_run(e1, ref_rows, out_file, max_new, chunk):
    import math
    import e0_instrument as I
    done = {r["key"] for r in I.read_jsonl(out_file)}
    for r in ref_rows:
        if r["key"] in done:
            continue
        toks = e1.generate(r["prompt_ids"], max_new, chunk, eos_ids=S.E0.EOS_IDS, margins=True)
        I.append_jsonl(out_file, {"key": r["key"], "tokens": toks, "margins": e1.margins,
                                  "nan": any(math.isnan(x) for x in e1.margins), "k": e1.k, "j": e1.j})


def decide(local_jsons, split_dirs, pipeline_dirs, hw_dir, q2_ref, q2_runs, vocab_len, fix_attempt, out):
    import e0_instrument as I
    res = {"experiment": S.NAME, "stage": "1b", "local": {}, "Q1": {}, "Q2": {}}
    for f in local_jsons:
        d = json.loads(Path(f).read_text()); res["local"][Path(f).name] = {"pass": d.get("pass"), "machine": d.get("machine"), "role": d.get("role")}
    local_ok = all(v["pass"] for v in res["local"].values()) if res["local"] else None
    hw = Q.q1_summary(hw_dir) if hw_dir else None
    res["floors"] = {"hardware": hw, "pipeline": {}}
    ref2 = I.read_jsonl(q2_ref) if q2_ref else []
    q_ok = True
    for cfg, d, pd, f2 in zip(("U-max(12)", "U-max(16)"), split_dirs, pipeline_dirs, q2_runs):
        pipe = Q.q1_summary(pd); sp = Q.q1_summary(d)
        res["floors"]["pipeline"][cfg] = pipe
        if sp and pipe and hw:
            res["Q1"][cfg] = dict(Q.q1_rule(sp, pipe, hw), summary=sp)
        else:
            res["Q1"][cfg] = {"pass": False, "missing": True}
        res["Q2"][cfg] = Q.q2_rule(ref2, I.read_jsonl(f2), vocab_len) if ref2 else {"pass": False, "missing": True}
        q_ok = q_ok and res["Q1"][cfg]["pass"] and res["Q2"][cfg]["pass"]
    res["Q"] = bool(q_ok)
    res["pass"] = bool(q_ok and (local_ok is not False))
    res["fix_attempt"] = fix_attempt
    res["consequence"] = ("Stage 1b PASS: performance measurement may begin" if res["pass"] else
                          ("Stage 1b FAIL: one implementation fix allowed, then a NEW freeze record and a full re-run (PREREG §4.4)"
                           if fix_attempt == 0 else "Stage 1b FAIL after the allowed fix: STOP (implementation result)"))
    for cfg in res["Q2"]:
        res["Q2"][cfg].pop("per_prompt", None)
    assert_no_timing_fields(res)
    Path(out).write_text(json.dumps(res, indent=1, default=str) + "\n")
    return res


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def _freeze_and_branch(a, role):
    if a.host and not a.dev_mode:
        sys.exit("--host overrides the machine identity and is permitted only with --dev-mode (a stage runs as the machine it is on)")
    rec = e1_freeze.check(a.freeze, role=role, require_tag=not a.dev_mode, host=a.host)
    m = e1_freeze.machine_entry(rec, a.host)
    return rec, m["branch"]


def _check_ids(a, model_dir):
    if a.check_ids:
        return json.loads(Path(a.check_ids).read_text())["ids"]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    s1 = S.E0.load_s1()
    row = json.loads(open(s1.SAMPLE_JSONL).readline())
    return tok.apply_chat_template([{"role": "user", "content": row["prompt"]}], tokenize=True, add_generation_prompt=True, enable_thinking=True)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, role):
        p.add_argument("--freeze", required=True); p.add_argument("--model-dir", required=True)
        p.add_argument("--device", default="cuda:0"); p.add_argument("--host", default=None)
        p.add_argument("--dev-mode", action="store_true", help="tests only: no tag check, tiny model, output marked screening")
        p.add_argument("--max-len", type=int, default=S.E0.L_F); p.add_argument("--chunk", type=int, default=S.E0.PREFILL_CHUNK)
        p.add_argument("--check-ids", default=None, help="JSON {ids: [...]} check sequence (default: first S1 prompt)")
        p.add_argument("--n-greedy", type=int, default=32)
        p.set_defaults(role=role)
    r = sub.add_parser("reference"); common(r, "server")
    r.add_argument("--out-dir", required=True); r.add_argument("--cont", default=None); r.add_argument("--limit", type=int, default=None)
    l = sub.add_parser("local"); common(l, None)
    l.add_argument("--role", dest="role_arg", choices=["server", "edge"], required=True)
    l.add_argument("--class", dest="mem_class", type=int, choices=S.E0.MEMORY_CLASSES, required=True)
    l.add_argument("--out", required=True); l.add_argument("--reference-dir", default=None); l.add_argument("--segments-only", action="store_true")
    l.add_argument("--dev-k", type=int, default=None); l.add_argument("--dev-j", type=int, default=None)
    jn = sub.add_parser("joint"); common(jn, "edge")
    jn.add_argument("--class", dest="mem_class", type=int, choices=S.E0.MEMORY_CLASSES, required=True)
    jn.add_argument("--server", required=True, help="host:port of e1_joint serve"); jn.add_argument("--out-dir", required=True)
    jn.add_argument("--cont", default=None); jn.add_argument("--ref-dir", default=None); jn.add_argument("--q2-ref", default=None)
    jn.add_argument("--limit", type=int, default=None); jn.add_argument("--max-new", type=int, default=S.E0.Q2_NEW_TOKENS)
    d = sub.add_parser("decide")
    d.add_argument("--local", nargs="*", default=[]); d.add_argument("--split-dirs", nargs=2, required=True)
    d.add_argument("--pipeline-dirs", nargs=2, required=True); d.add_argument("--hw-dir", required=True)
    d.add_argument("--q2-ref", required=True); d.add_argument("--q2-runs", nargs=2, required=True)
    d.add_argument("--vocab-len", type=int, default=151669); d.add_argument("--fix-attempt", type=int, default=0, choices=[0, 1])
    d.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "decide":
        res = decide(a.local, a.split_dirs, a.pipeline_dirs, a.hw_dir, a.q2_ref, a.q2_runs, a.vocab_len, a.fix_attempt, a.out)
        print(json.dumps({k: res[k] for k in ("Q", "pass", "consequence")}, indent=1)); return
    expected = None if a.dev_mode else S.E0.QWEN3_32B_CONFIG
    if a.cmd == "reference":
        rec, branch = _freeze_and_branch(a, "server")
        _setup_torch()
        ids = _check_ids(a, a.model_dir)
        hf = hf_model(a.model_dir, a.device)
        out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
        torch.save({"ids": ids, "hidden": reference_hidden(hf, ids, a.device).cpu(), "greedy": reference_greedy(hf, ids, a.n_greedy, a.device),
                    "freeze_sha256": rec["freeze_sha256"], "device": device_identity(a.device)}, out / "ref_hidden.pt")
        if a.cont:
            q1_reference(hf, a.cont, out / "q1_ref", a.device, a.limit)
        meta = {"experiment": S.NAME, "stage": "1b-reference", "freeze_sha256": rec["freeze_sha256"], "device": device_identity(a.device),
                "n_check_tokens": len(ids), "machine": socket.gethostname()}
        assert_no_timing_fields(meta); (out / "reference.json").write_text(json.dumps(meta, indent=1) + "\n")
        print(json.dumps(meta, indent=1)); return
    if a.cmd == "local":
        role = a.role_arg
        rec, branch = _freeze_and_branch(a, role)
        k, j = S.FIXED_CUTS[a.mem_class]
        if a.dev_mode and a.dev_k is not None:
            k, j = a.dev_k, a.dev_j
        elif a.dev_k is not None:
            sys.exit("--dev-k/--dev-j need --dev-mode (PREREG §3: the cut is frozen)")
        ids = _check_ids(a, a.model_dir)
        ref_h = ref_t = None
        if a.reference_dir:
            ref = torch.load(Path(a.reference_dir) / "ref_hidden.pt")
            if ref["freeze_sha256"] != rec["freeze_sha256"] or ref["ids"] != ids:
                sys.exit("reference was produced under another freeze or check sequence")
            ref_h, ref_t = ref["hidden"], ref["greedy"][: a.n_greedy]
        try:
            res = local_checks(a.model_dir, k, j, a.device, expected, branch, ids, a.max_len, a.n_greedy, a.chunk, rec["freeze_sha256"],
                               ref_h, ref_t, a.segments_only)
        except GraphCaptureError as e:
            res = {"pass": False, "graph_capture_failed": str(e), "consequence": "compatibility probe failed: write a NEW freeze record with cuda_graphs=off_capture_failed and rerun (PREREG §4.2)"}
        res.update(experiment=S.NAME, stage="1b", role=role, machine=a.host or socket.gethostname(), device=device_identity(a.device),
                   freeze_sha256=rec["freeze_sha256"], frozen_branch=branch, memory_class=a.mem_class, profile_role="screening" if a.dev_mode else "target")
        assert_no_timing_fields(res)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1, default=str) + "\n")
        print(json.dumps({k: res.get(k) for k in ("pass", "branch_in_use", "branch_matches_freeze", "graph_capture_failed")}, indent=1))
        sys.exit(0 if res["pass"] else 4)
    if a.cmd == "joint":
        rec, branch = _freeze_and_branch(a, "edge")
        _setup_torch()
        k, j = S.FIXED_CUTS[a.mem_class]
        e1_attention.set_branch(branch["attention"])
        sl = edge_slice(a.model_dir, k, j, a.device, torch.bfloat16, expected); sl.cfg._attn_implementation = "e1_static"
        host, port = a.server.split(":")
        model_id = f"{S.E0.SPLIT_REPO}@{S.E0.SPLIT_REVISION}" if not a.dev_mode else "dev"
        mid = P.RemoteMiddle(host, int(port), P.hello_for(sl, model_id, k, j, a.max_len, rec["freeze_sha256"]), a.device, sl.cfg.hidden_size, a.chunk)
        e1 = USplitEdge(sl, k, j, mid, a.max_len, "static")
        if branch["cuda_graphs"] == "on":
            e1.enable_graph()
        out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
        meta = {"experiment": S.NAME, "stage": "1b-joint", "freeze_sha256": rec["freeze_sha256"], "edge": device_identity(a.device),
                "server_identity": {k_: v for k_, v in mid.server_identity.items() if k_ in ("host", "gpu", "freeze_sha256", "config_hash", "k", "j", "branch")},
                "branch_in_use": branch_in_use(e1), "branch_matches_freeze": branch_matches(branch_in_use(e1), branch, a.device), "k": k, "j": j}
        if a.cont and a.ref_dir:
            meta["q1_n"] = q1_compare(lambda ids: e1.teacher_forced_hidden(ids, a.chunk), e1.sl.lm_head, a.cont, a.ref_dir, out / "q1_split", a.limit)
        if a.q2_ref:
            import e0_instrument as I
            q2_run(e1, I.read_jsonl(a.q2_ref)[: a.limit or None], out / "q2_split.jsonl", a.max_new, a.chunk)
            meta["q2_n"] = len(I.read_jsonl(out / "q2_split.jsonl"))
        mid.close()
        assert_no_timing_fields(meta)
        (out / "joint.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
        print(json.dumps(meta, indent=1, default=str))


if __name__ == "__main__":
    main()
