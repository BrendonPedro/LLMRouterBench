"""K16-HWFIT runtime glue. Every computation is the existing runtime's, imported unchanged:

  E0  split_runtime.Slice               weight loading for a layer subset
  E1  e1_segment.Segment / USplitEdge   per-segment KV caches; "prefix" mode is E0's computation
  E1  e1_protocol.SegmentServer / RemoteMiddle / HELLO checks   the wire

This module adds only: the k = 16 server object (one slice, one Segment per mode), an accept loop that routes a
connection to the requested mode and stops on a shutdown request, the edge-side connection with identity
checks, the unsplit reference, the comparison against it, and the latency loop.
"""
import json
import math
import random
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))   # python -I: the script dir is not on sys.path
import k16_spec as S  # noqa: E402

sys.path.insert(0, str(S.E0_DIR))
import split_runtime as E0RT  # noqa: E402
import e0_q as Q  # noqa: E402
import e1_attention  # noqa: E402,F401  (registers the "e1_static" attention function used by static mode)
import e1_protocol as P  # noqa: E402
from e1_segment import Segment, USplitEdge, LocalMiddle, edge_slice, server_slice, ownership  # noqa: E402,F401
import k16_criteria  # noqa: E402
from k16_common import append_jsonl, utc  # noqa: E402

ATTN = {"prefix": "sdpa", "static": "e1_static"}     # cfg._attn_implementation per mode (E0's and E1's values)
MIB = 1 << 20


def setup_torch():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def use_mode(sl, mode):
    """Select the attention path of a slice. prefix needs "sdpa" (E0, identical to the unsplit model's path)."""
    sl.cfg._attn_implementation = ATTN[mode]


def device_identity(device):
    dev = torch.device(device)
    if dev.type != "cuda":
        return {"name": "cpu", "uuid": "cpu", "compute_cap": None, "total_mib": None}
    p = torch.cuda.get_device_properties(dev)
    return {"name": p.name, "uuid": str(getattr(p, "uuid", "")), "compute_cap": float(f"{p.major}.{p.minor}"),
            "total_mib": p.total_memory // MIB}


def mem_stats(device):
    dev = torch.device(device)
    if dev.type != "cuda":
        return {"device": "cpu"}
    try:
        free, total = (x / MIB for x in torch.cuda.mem_get_info(dev))
    except Exception:
        free = total = None
    return {"allocated_mib": torch.cuda.memory_allocated(dev) / MIB, "peak_allocated_mib": torch.cuda.max_memory_allocated(dev) / MIB,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(dev) / MIB, "device_free_mib": free, "device_total_mib": total}


# --------------------------------------------------------------------------- #
# Server half: layers [K, L-J)                                                 #
# --------------------------------------------------------------------------- #
class ServerSide:
    """One slice holding the middle layers and one Segment (own KV cache) per mode."""

    def __init__(self, model_dir, device, p, expected_cfg, with_static=True):
        self.sl = server_slice(model_dir, S.K, S.J, device, torch.bfloat16, expected_cfg)
        self.p = p
        use_mode(self.sl, "prefix")
        self.segs = {"prefix": Segment(self.sl, self.sl.layer_ids, p["max_len"], "prefix")}
        self.static = {"available": False, "requested": bool(with_static)}
        if with_static:
            try:
                use_mode(self.sl, "static")
                seg = Segment(self.sl, self.sl.layer_ids, p["max_len"], "static")
                try:
                    seg.enable_graph()
                    graphs = "on" if self.sl.device.type == "cuda" else "cpu_control_flow"
                except Exception as e:      # a capture failure leaves the static path without a graph
                    seg.graph, graphs = None, "off_capture_failed"
                    self.static["graph_error"] = f"{type(e).__name__}: {str(e)[:300]}"
                seg.reset()
                self.segs["static"] = seg
                self.static.update(available=True, cuda_graphs=graphs, attention="sdpa_static_fallback")
            except Exception as e:
                self.static["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            finally:
                use_mode(self.sl, "prefix")

    def identity(self, spec_hash, extra=None):
        d = {"runtime": S.NAME, "freeze_sha256": spec_hash, "config_hash": E0RT.config_hash(self.sl.cfg), "k": S.K, "j": S.J,
             "model_id": self.p["model_id"], "protocol_version": S.E1.PROTOCOL_VERSION, "active": [S.K, self.sl.n_layers - S.J],
             "loaded_layers": [self.sl.layer_ids[0], self.sl.layer_ids[-1] + 1], "host": socket.gethostname(),
             "device": device_identity(self.sl.device), "modes": sorted(self.segs), "static": self.static,
             "param_bytes": self.sl.param_bytes, "kv_bytes": {m: s.cache.nbytes() for m, s in self.segs.items()}}
        d.update(extra or {})
        return d


def _peek_hello(conn, wait_s=30.0):
    """The HELLO of a new connection, read WITHOUT consuming it: e1_protocol.SegmentServer.handle reads it again.
    Returns (HELLO dict, message bytes), or (None, 0) if the first message is not a well-formed HELLO (handle()
    then refuses it)."""
    conn.settimeout(wait_s)
    end, need = time.monotonic() + wait_s, P.HDR.size
    while time.monotonic() < end:
        try:
            buf = conn.recv(need, socket.MSG_PEEK)
        except OSError:
            return None, 0
        if not buf:
            return None, 0
        if len(buf) >= need:
            f = P.HDR.unpack(buf[:P.HDR.size])
            if f[0] != P.MAGIC or f[2] != P.T_HELLO or f[8] > 1 << 16:
                return None, 0
            if need == P.HDR.size and f[8] > 0:
                need = P.HDR.size + f[8]
                continue
            try:
                h = json.loads(bytes(buf[P.HDR.size:need]).decode())
                return (h, need) if isinstance(h, dict) else (None, 0)
            except ValueError:
                return None, 0
        time.sleep(0.002)
    return None, 0


def _consume(conn, n):
    buf = bytearray(n)
    P._recv_exact(conn, memoryview(buf))


def serve(side, listen_sock, spec_hash, log, extra_identity=None, accept_timeout_s=None, conn_timeout_s=1800, stop=None):
    """Accept edge connections one at a time until a shutdown request, an accept timeout or `stop` is set.
    The edge names the mode in HELLO `runtime` ("K16-HWFIT/<mode>[/final]"); "/final" asks the server to exit after
    that connection. Returns "shutdown", "accept_timeout" or "stopped"."""
    state = {"shutdown": False}

    def cb(ev):
        if ev.get("event") == "hello_ok" and str(ev.get("peer_hello", {}).get("runtime", "")).endswith("/final"):
            state["shutdown"] = True
        log(dict(ev, utc=utc()))
    hidden = side.sl.cfg.hidden_size
    servers = {m: P.SegmentServer(seg, side.identity(spec_hash, dict(extra_identity or {}, mode=m)), hidden, side.p["chunk"], log=cb)
               for m, seg in side.segs.items()}
    listen_sock.settimeout(1.0)
    waited = 0.0
    while not state["shutdown"]:
        if stop is not None and stop.is_set():
            return "stopped"
        try:
            conn, peer = listen_sock.accept()
        except socket.timeout:
            waited += 1.0
            if accept_timeout_s is not None and waited >= accept_timeout_s:
                return "accept_timeout"
            continue
        waited = 0.0
        with conn:
            hello, hello_bytes = _peek_hello(conn)
            parts = str((hello or {}).get("runtime", "")).split("/")
            mode = parts[1] if len(parts) > 1 and parts[0] == S.NAME else "prefix"
            try:
                if hello is not None and mode not in servers:
                    _consume(conn, hello_bytes)
                    why = f"mode {mode!r} is not available on this server: {side.static}"
                    log({"event": "refused", "why": why, "utc": utc()})
                    P.send_msg(conn, P.T_ERROR, payload=why.encode())
                    continue
                srv = servers.get(mode, servers["prefix"])
                use_mode(side.sl, mode if mode in servers else "prefix")
                srv.seg.reset()
                conn.settimeout(conn_timeout_s)
                n0 = srv.n_fwd
                log({"event": "connection_open", "peer": peer[0], "mode": mode, "utc": utc()})
                srv.handle(conn)
                log({"event": "connection_done", "mode": mode, "n_fwd": srv.n_fwd - n0, "memory": mem_stats(side.sl.device), "utc": utc()})
            except (P.TransportError, OSError) as e:
                log({"event": "connection_closed", "why": f"{type(e).__name__}: {e}", "utc": utc()})
            finally:
                use_mode(side.sl, "prefix")
    return "shutdown"


# --------------------------------------------------------------------------- #
# Edge half                                                                    #
# --------------------------------------------------------------------------- #
def connect(server, sl, mode, spec_hash, p, final=False, timeout_s=600):
    """RemoteMiddle to the k = 16 server in `mode`. The server refuses another spec hash, config, cut or model id
    (e1_protocol.check_hello); the edge refuses a server whose identity is not this experiment's."""
    host, port = server.rsplit(":", 1)
    hello = P.hello_for(sl, p["model_id"], S.K, S.J, p["max_len"], spec_hash)
    hello["runtime"] = f"{S.NAME}/{mode}" + ("/final" if final else "")
    mid = P.RemoteMiddle(host, int(port), hello, sl.device, sl.cfg.hidden_size, p["chunk"], timeout_s=timeout_s)
    si = mid.server_identity
    want = {"freeze_sha256": spec_hash, "k": S.K, "j": S.J, "mode": mode, "active": [S.K, sl.n_layers - S.J],
            "config_hash": E0RT.config_hash(sl.cfg), "model_id": p["model_id"]}
    bad = {k: si.get(k) for k, v in want.items() if si.get(k) != v}
    if bad:
        mid.close()
        raise P.TransportError(f"server identity does not match this experiment: {bad}")
    return mid


def request_shutdown(server, sl, spec_hash, p):
    mid = connect(server, sl, "prefix", spec_hash, p, final=True, timeout_s=30)
    mid.close()


def edge_split(sl, mid, mode, p):
    """USplitEdge at the k = 16 cut in `mode`. Static mode tries CUDA-graph capture and records what it got."""
    use_mode(sl, mode)
    e = USplitEdge(sl, S.K, S.J, mid, p["max_len"], mode)
    info = {"mode": mode, "attention": ATTN[mode], "cuda_graphs": "not used"}
    if mode == "static":
        try:
            e.enable_graph()
            info["cuda_graphs"] = "on" if sl.device.type == "cuda" else "cpu_control_flow"
        except Exception as ex:
            e.head.graph = e.tail.graph = None
            info.update(cuda_graphs="off_capture_failed", graph_error=f"{type(ex).__name__}: {str(ex)[:300]}")
    return e, info


def kv_invariant(steps):
    """After every remote step the server's committed KV length equals the edge's position (both boundaries in step)."""
    return bool(steps) and all(s.get("server_seen") == s["start"] + s["n"] for s in steps)


# --------------------------------------------------------------------------- #
# Unsplit reference (Hugging Face model, one device)                           #
# --------------------------------------------------------------------------- #
def hf_model(model_dir, device):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()


@torch.inference_mode()
def hf_greedy(hf, ids, n, device, eos_ids=()):
    """Greedy tokens and top-1 minus top-2 log-probability margins from the unsplit model, with its own KV cache.
    The LM head sees the last position only, as on the split path."""
    t = torch.as_tensor(ids, dtype=torch.long, device=device).view(1, -1)
    out = hf.model(input_ids=t, use_cache=True)
    toks, margins = [], []
    while True:
        logits = hf.lm_head(out.last_hidden_state[:, -1:, :])[0, -1].float()
        tok = int(torch.argmax(logits))
        top = torch.topk(torch.log_softmax(logits, -1), 2).values
        toks.append(tok)
        margins.append(float(top[0] - top[1]))
        if len(toks) >= n or tok in eos_ids:
            return toks, margins
        out = hf.model(input_ids=torch.tensor([[tok]], device=device), past_key_values=out.past_key_values, use_cache=True)


@torch.inference_mode()
def hf_hidden(hf, ids, device):
    t = torch.as_tensor(ids, dtype=torch.long, device=device).view(1, -1)
    return hf.model(input_ids=t, use_cache=False).last_hidden_state[0]


def write_reference(hf, prompts, p, device, out_dir):
    """Per check prompt: greedy continuation + margins, final hidden states of prompt + continuation (one pass),
    and the reference top-K log-probabilities at every position (E0's e0_q.ref_topk)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for pr in prompts:
        toks, margins = hf_greedy(hf, pr["ids"], p["greedy_tokens"], device)
        full = list(pr["ids"]) + toks
        h = hf_hidden(hf, full, device)
        rid, rlp = Q.ref_topk(torch, hf.lm_head, h, list(range(len(full))), S.CRITERIA["topk"])
        finite = bool(torch.isfinite(h.float()).all()) and all(math.isfinite(m) for m in margins)
        np.savez(out_dir / f"{pr['id']}.npz", prompt_ids=np.array(pr["ids"], dtype=np.int64), tokens=np.array(toks, dtype=np.int64),
                 margins=np.array(margins, dtype=np.float64), topk_ids=rid, topk_lps=rlp,
                 hidden_bf16_as_i16=h.contiguous().view(torch.int16).cpu().numpy())
        degen, why = Q.is_degenerate(toks, hf.config.vocab_size)
        rows.append({"id": pr["id"], "n_in": len(pr["ids"]), "n_new": len(toks), "finite": finite, "degenerate": degen, "why": why,
                     "share_decisive": float(((rlp[:, 0] - rlp[:, 1]) >= S.CRITERIA["decisive_margin_nats"]).mean())})
        print(f"[reference] {pr['id']}: {len(pr['ids'])} + {len(toks)} tokens, finite={finite}", flush=True)
    return rows


def load_reference(ref_dir, pid, device):
    z = np.load(Path(ref_dir) / f"{pid}.npz")
    hid = torch.from_numpy(z["hidden_bf16_as_i16"]).view(torch.bfloat16).to(device)
    return {"prompt_ids": z["prompt_ids"].tolist(), "tokens": z["tokens"].tolist(), "margins": z["margins"].tolist(),
            "topk_ids": z["topk_ids"], "topk_lps": z["topk_lps"], "hidden": hid}


# --------------------------------------------------------------------------- #
# Comparison of a split configuration with the reference                       #
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def evaluate_config(e, prompts, ref_dir, p, out_dir, label):
    """Run split `e` on the check set and apply K16-CORRECT-v1. Writes per-sequence arrays to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sl = e.sl
    teacher, gx, gref, desc = {}, {}, {}, {}
    remote = hasattr(e.middle, "server_identity")
    for pr in prompts:
        r = load_reference(ref_dir, pr["id"], sl.device)
        if r["prompt_ids"] != list(pr["ids"]):
            raise ValueError(f"reference for {pr['id']} was computed on other token ids")
        full = list(pr["ids"]) + r["tokens"]
        h = e.teacher_forced_hidden(full, p["chunk"])
        kv_tf = kv_invariant(e.steps) if remote else None
        agree, kl, xm = Q.compare_to_ref(torch, sl.lm_head, h, list(range(len(full))), r["topk_ids"], r["topk_lps"])
        ref_margin = (r["topk_lps"][:, 0] - r["topk_lps"][:, 1]).astype(np.float64)
        finite = bool(torch.isfinite(h.float()).all())
        diff = (h.float() - r["hidden"].float()).abs()
        toks = e.generate(pr["ids"], p["greedy_tokens"], p["chunk"], force_len=True, margins=True)
        desc[pr["id"]] = {"bitwise_equal_reference_hidden": bool(torch.equal(h, r["hidden"])),
                          "max_abs_vs_reference_hidden": float(diff.max()) if finite else None,
                          "prefill_chunks": -(-len(full) // p["chunk"]),
                          "kv_positions_in_step": None if not remote else bool(kv_tf and kv_invariant(e.steps))}
        teacher[pr["id"]] = {"agree": agree, "kl": kl, "ref_margin": ref_margin, "finite": finite}
        gx[pr["id"]] = {"tokens": toks, "finite": not any(math.isnan(m) for m in e.margins)}
        gref[pr["id"]] = {"tokens": r["tokens"], "margins": r["margins"]}
        np.savez(out_dir / f"{pr['id']}.npz", agree=agree, kl=kl, x_margin=xm, ref_margin=ref_margin, tokens=np.array(toks, dtype=np.int64))
        print(f"[{label}] {pr['id']}: disagreements {int((~agree).sum())}/{agree.size}, mean KL {kl.mean():.3e}, "
              f"greedy identical {toks == r['tokens']}", flush=True)
    res = k16_criteria.evaluate([pr["id"] for pr in prompts], teacher, gref, gx, sl.cfg.vocab_size)
    res.update(label=label, descriptive=desc, all_bitwise_equal_reference_hidden=all(d["bitwise_equal_reference_hidden"] for d in desc.values()))
    if remote:
        res["kv_positions_in_step_all"] = all(d["kv_positions_in_step"] for d in desc.values())
    return res


# --------------------------------------------------------------------------- #
# Generation with text, and latency                                            #
# --------------------------------------------------------------------------- #
def chat_ids(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True, enable_thinking=False)


def chat_prompts(model_dir, p, dev_prompts=None):
    """(tokenizer or None, [{id, text, ids}]). dev: token-id prompts without a tokenizer."""
    if p["dev"]:
        return None, [{"id": f"chat-{i}", "text": None, "ids": list(pr["ids"])} for i, pr in enumerate(dev_prompts[:2])]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    return tok, [{"id": c["id"], "text": c["text"], "ids": chat_ids(tok, c["text"])} for c in S.CHAT_PROMPTS]


def latency_requests(prompts, p):
    first = next(pr for pr in prompts if pr["group"] == "short")
    body = [{"prompt": pr, "rep": r, "warmup": False} for pr in prompts for r in range(p["reps"])]
    random.Random(S.LATENCY["order_seed"]).shuffle(body)
    return [{"prompt": first, "rep": -1 - i, "warmup": True} for i in range(p["warmups"])] + body


def run_latency(e, mid, prompts, p, out_file, base):
    """Forced-length greedy requests; one JSONL row per request with per-token stamps and per-step components."""
    rows = []
    for req in latency_requests(prompts, p):
        pr = req["prompt"]
        b0, b1 = mid.bytes_sent, mid.bytes_recv
        row = dict(base, prompt_id=pr["id"], group=pr["group"], n_in=len(pr["ids"]), rep=req["rep"], warmup=req["warmup"],
                   gen_tokens=p["gen_tokens"], utc=utc())
        try:
            toks = e.generate(pr["ids"], p["gen_tokens"], p["chunk"], force_len=True)
            steps = e.steps
            for s in steps:
                s["phase"] = "prefill" if (s["n"] > 1 or s["start"] < len(pr["ids"])) else "decode"
            row.update(ok=len(toks) == p["gen_tokens"] and kv_invariant(steps), token_ns=e.token_ns, steps=steps,
                       bytes_out=mid.bytes_sent - b0, bytes_in=mid.bytes_recv - b1)
        except Exception as ex:
            row.update(ok=False, error=f"{type(ex).__name__}: {ex}")
        append_jsonl(out_file, row)
        rows.append(row)
        if not row["ok"]:
            break                      # fail closed: a dropped or malformed exchange ends the measurement
        print(f"[latency] {pr['id']} rep {req['rep']}: {(row['token_ns'][-1] - row['token_ns'][0]) / 1e6 / max(1, len(row['token_ns']) - 1):.1f} ms/token",
              flush=True)
    return rows


def _med(v):
    v = sorted(v)
    return None if not v else (v[len(v) // 2] if len(v) % 2 else 0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2]))


def _pct(v, q):
    v = sorted(v)
    return None if not v else v[min(len(v) - 1, int(math.ceil(q * len(v))) - 1)]


def summarize_latency(rows):
    """Medians per prompt group over measured (non-warm-up) requests. Times in ms."""
    out = {"n_requests": len(rows), "n_ok": sum(bool(r.get("ok")) for r in rows), "all_ok": bool(rows) and all(r.get("ok") for r in rows), "groups": {}}
    for g in sorted({r["group"] for r in rows}):
        rs = [r for r in rows if r["group"] == g and not r["warmup"] and r.get("ok")]
        if not rs:
            continue
        dec = [s for r in rs for s in r["steps"] if s["phase"] == "decode"]
        ms = lambda key: [s.get(key, 0) / 1e6 for s in dec]
        edge = [(s["edge_head_ns"] + s["edge_tail_ns"] + s["lm_head_ns"]) / 1e6 for s in dec]
        tpot = [(r["token_ns"][-1] - r["token_ns"][0]) / 1e6 / (len(r["token_ns"]) - 1) for r in rs if len(r["token_ns"]) > 1]
        out["groups"][g] = {
            "n_requests": len(rs), "n_in": rs[0]["n_in"], "gen_tokens": rs[0]["gen_tokens"], "n_decode_steps": len(dec),
            "ttft_ms_median": _med([r["token_ns"][0] / 1e6 for r in rs]), "tpot_ms_median": _med(tpot),
            "tokens_per_s_median": (1000.0 / _med(tpot)) if tpot and _med(tpot) else None,
            "decode_step_ms": {"total_median": _med(ms("total_ns")), "total_p95": _pct(ms("total_ns"), 0.95), "edge_compute_median": _med(edge),
                               "server_compute_median": _med(ms("server_compute_ns")), "network_median": _med(ms("net_ns")),
                               "network_p95": _pct(ms("net_ns"), 0.95), "serialise_median": _med(ms("ser_ns")),
                               "round_trip_median": _med(ms("rt_ns"))},
            "bytes_per_decode_step_each_way": _med([s.get("bytes_out", 0) for s in dec])}
    return out
