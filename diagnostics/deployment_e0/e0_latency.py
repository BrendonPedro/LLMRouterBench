"""E0-L latency harness: frozen workload, cell runner (SPLIT / REMOTE / EDGE / profiles), load generator, netem.

Subcommands
  build-workload  tokenise the frozen fixed-length prompts from the S1 sample (CPU; output committed)
  plan            print the frozen cell table and the randomised within-block execution order
  run-split       SPLIT requests: edge runner -> segment server (or the echo stub for edge profiles)
  run-openai      REMOTE / EDGE requests against a vLLM OpenAI server (streaming, forced length)
  run-cell        run one frozen E0-L cell by id (dispatches to run-split / run-openai)
  profile-segment server-segment compute profile with a synthetic-activation client (server host only)
  run-hf-local    informational REMOTE-HF: unsplit HF model, batch 1, on the server host
  load-gen        constant-concurrency background load against the server's vLLM (state kappa)
  netem           print / apply / clear / show the tc netem + ifb commands on the edge
  linkcheck       per-token RTT and bulk bandwidth over the split protocol against an echo stub (go/no-go)

Every request is one JSON line: identity, cell factors, n_in, per-token timestamps (ns from request
start), per-forward split components, bytes on the wire, NVML power/memory summary, ok/error.
Generation is forced to exactly GEN_TOKENS tokens (EOS ignored), batch 1, greedy.
"""

import argparse
import http.client
import http.server
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e0_instrument as I  # noqa: E402
import e0_spec as S  # noqa: E402

WORKLOAD_JSON = S.E0_DIR / "frozen" / "workload.json"


# --------------------------------------------------------------------------- #
# Workload                                                                    #
# --------------------------------------------------------------------------- #
def templated_ids(tok, prompt, thinking=True):
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                   add_generation_prompt=True, enable_thinking=thinking)


def build_workload(tokenizer_dir):
    """Fixed-length token sequences: concatenated thinking-template S1 prompts, truncated to n_in.

    Group g's prompt i starts at sample query OFFSET[g] + i*STRIDE[g] (sample file order) and continues
    through the following queries until n_in tokens are collected. Content is irrelevant to latency
    (generation length is forced); the S1 prompts are used so token statistics are realistic.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    s1 = S.load_s1()
    sample = [json.loads(x) for x in open(s1.SAMPLE_JSONL)]
    offset, stride = {"short": 0, "long": 50, "depth": 100}, {"short": 1, "long": 1, "depth": 20}
    prompts = []
    cache = {}
    for g, (n_in, count) in S.WORKLOAD.items():
        for i in range(count):
            ids, q = [], offset[g] + i * stride[g]
            while len(ids) < n_in:
                key = sample[q % len(sample)]["key"]
                if key not in cache:
                    cache[key] = templated_ids(tok, sample[q % len(sample)]["prompt"])
                ids.extend(cache[key])
                q += 1
            prompts.append({"id": f"{g}-{i}", "group": g, "n_in": n_in, "ids": ids[:n_in]})
    return {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tokenizer": f"{S.SPLIT_REPO}@{S.SPLIT_REVISION}", "gen_tokens": S.GEN_TOKENS,
            "groups": {g: list(v) for g, v in S.WORKLOAD.items()}, "prompts": prompts,
            "ids_sha256": __import__("hashlib").sha256(S.canonical_json([p["ids"] for p in prompts])
                                                       .encode()).hexdigest()}


def load_workload(path=WORKLOAD_JSON):
    w = json.loads(Path(path).read_text())
    for p in w["prompts"]:
        assert len(p["ids"]) == p["n_in"], p["id"]
    return w


def requests_for(cell_id, block, workload, with_depth=None):
    """Warm-ups first, then every (prompt, rep) once in a cell-seeded random order."""
    with_depth = block in S.DEPTH_BLOCKS if with_depth is None else with_depth
    ps = [p for p in workload["prompts"] if with_depth or p["group"] != "depth"]
    first_short = next(p for p in ps if p["group"] == "short")
    reqs = [{"prompt": first_short, "rep": -1 - i, "warmup": True} for i in range(S.WARMUPS_PER_CELL)]
    body = [{"prompt": p, "rep": r, "warmup": False} for p in ps for r in range(S.REPS)]
    random.Random(f"{S.ORDER_SEED}:{cell_id}").shuffle(body)
    return reqs + body


def execution_order():
    """Block order fixed (core, load, rtt, heldout, bw, cut); cell order shuffled within each block."""
    cs = S.cells()
    order = []
    for b in ["core", "load", "rtt", "heldout", "bw", "cut", "info"]:
        block = [c for c in cs if c["block"] == b]
        random.Random(f"{S.ORDER_SEED}:block:{b}").shuffle(block)
        order += block
    return order


# --------------------------------------------------------------------------- #
# Link shaping                                                                #
# --------------------------------------------------------------------------- #
def netem_commands(dev, rtt_ms, bw_mbps, action="apply"):
    half = f"{rtt_ms / 2:g}ms" if rtt_ms else None
    rate = f"rate {bw_mbps}mbit" if bw_mbps and bw_mbps < 1000 else ""
    clear = [f"sudo tc qdisc del dev {dev} root 2>/dev/null || true",
             f"sudo tc qdisc del dev {dev} handle ffff: ingress 2>/dev/null || true",
             "sudo tc qdisc del dev ifb0 root 2>/dev/null || true"]
    if action == "clear" or (not half and not rate):
        return clear
    spec = " ".join(x for x in [f"delay {half}" if half else "", rate] if x)
    return clear + [
        "sudo modprobe ifb numifbs=1", "sudo ip link add ifb0 type ifb 2>/dev/null || true", "sudo ip link set ifb0 up",
        f"sudo tc qdisc add dev {dev} root netem {spec}",
        f"sudo tc qdisc add dev {dev} handle ffff: ingress",
        f"sudo tc filter add dev {dev} parent ffff: matchall action mirred egress redirect dev ifb0",
        f"sudo tc qdisc add dev ifb0 root netem {spec}"]


def netem_show(dev):
    out = []
    for d in (dev, "ifb0"):
        r = subprocess.run(f"tc -s qdisc show dev {d}", shell=True, capture_output=True, text=True)
        out.append(r.stdout.strip() or r.stderr.strip())
    return "\n".join(out)


def netem_matches(show, rtt_ms, bw_mbps):
    """True iff both directions carry the expected netem delay (and rate when < 1 Gbps)."""
    blocks = show.split("\n")
    want_delay = f"delay {rtt_ms / 2:g}ms" if rtt_ms else None
    nets = [b for b in blocks if "netem" in b]
    if not rtt_ms and not (bw_mbps and bw_mbps < 1000):
        return not nets
    if len(nets) < 2:
        return False
    ok = all((want_delay is None or re.search(rf"delay {rtt_ms / 2:g}(\.0)?ms", b)) for b in nets)
    if bw_mbps and bw_mbps < 1000:
        ok = ok and all(re.search(rf"rate {bw_mbps}Mbit", b, re.I) for b in nets)
    return bool(ok)


# --------------------------------------------------------------------------- #
# Request runners                                                             #
# --------------------------------------------------------------------------- #
def base_record(args, cell, req, path, extra=None):
    rec = {"cell": cell.get("id"), "block": cell.get("block"), "path": path, "rtt_ms": cell.get("rtt_ms"),
           "bw_mbps": cell.get("bw_mbps"), "load": cell.get("load"), "shaping": args.shaping,
           "prompt_id": req["prompt"]["id"], "group": req["prompt"]["group"], "n_in": req["prompt"]["n_in"],
           "rep": req["rep"], "warmup": req["warmup"], "gen_tokens": S.GEN_TOKENS,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    rec.update(extra or {})
    return rec


def done_keys(out):
    return {(r["cell"], r["prompt_id"], r["rep"]) for r in I.read_jsonl(out) if r.get("ok")}


def run_split(args, cell, out):
    import torch
    import split_runtime as R
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    nv = I.NVML() if args.device.startswith("cuda") else None
    baseline = I.idle_baseline(nv) if nv and nv.available else 0
    cap = None
    if args.cap_class and nv and nv.available:
        cap = I.apply_cap(torch, args.device, S.budget_mib(args.cap_class), nv, baseline)
    expected = S.QWEN3_32B_CONFIG if args.expect_32b else None
    sl = R.edge_slice(args.model_dir, args.k, args.j, args.device, torch.bfloat16, expected)
    host, port = args.server.split(":")
    shaper = R.AppShaper(cell.get("rtt_ms"), cell.get("bw_mbps")) if args.shaping == "app" else None
    mid = R.RemoteMiddle(host, int(port), R.hello_for(sl, args.model_id, args.k, args.j, S.L_F), args.device,
                         sl.cfg.hidden_size, shaper)
    runner = R.EdgeRunner(sl, args.k, args.j, mid, max_len=S.L_F)
    wl = load_workload(args.workload)
    done = done_keys(out)
    sampler = None
    if nv and nv.available:
        sampler = I.Sampler(nv, S.POWER_SAMPLE_S, baseline)
        sampler.start()
    if args.shaping == "netem":
        I.append_jsonl(out, {"cell": cell["id"], "netem_before": netem_show(args.dev),
                             "netem_ok": netem_matches(netem_show(args.dev), cell.get("rtt_ms"), cell.get("bw_mbps"))})
    ident = {"k": args.k, "j": args.j, "cap": cap, "server_identity": mid.server_identity,
             "edge_software": R.software_identity(), "model_id": args.model_id}
    n_err = 0
    for req in requests_for(cell["id"], cell["block"], wl, args.with_depth):
        if (cell["id"], req["prompt"]["id"], req["rep"]) in done:
            continue
        rec = base_record(args, cell, req, "SPLIT", ident)
        b0, b1 = mid.bytes_sent, mid.bytes_recv
        t0 = time.perf_counter()
        try:
            if nv and nv.available:
                torch.cuda.reset_peak_memory_stats(args.device)
            toks = runner.generate(req["prompt"]["ids"], S.GEN_TOKENS, S.PREFILL_CHUNK, force_len=True)
            rec.update(ok=len(toks) == S.GEN_TOKENS, token_ns=runner.token_ns, steps=runner.steps,
                       bytes_out=mid.bytes_sent - b0, bytes_in=mid.bytes_recv - b1)
            if nv and nv.available:
                rec["alloc"] = I.allocator_stats(torch, args.device)
        except Exception as e:   # OOM and transport errors are F failures; recorded, never retried
            n_err += 1
            rec.update(ok=False, error=f"{type(e).__name__}: {e}")
        t1 = time.perf_counter()
        if sampler:
            rec["nvml"] = sampler.summary(t0, t1)
        I.append_jsonl(out, rec)
        if not rec["ok"] and isinstance(rec.get("error"), str) and "TransportError" in rec["error"]:
            break
    if sampler:
        sampler.stop()
    if args.shaping == "netem":
        I.append_jsonl(out, {"cell": cell["id"], "netem_after": netem_show(args.dev)})
    mid.close()
    return n_err


def _stream_completion(base_url, body, timeout=900):
    u = urllib.parse.urlparse(base_url)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    data = json.dumps(body).encode()
    t0 = time.perf_counter_ns()
    conn.request("POST", u.path.rstrip("/") + "/completions", body=data,
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:500]!r}")
    stamps, last_tokens, usage, finish = [], 0, None, None
    for raw in resp:
        line = raw.decode().strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        now = time.perf_counter_ns() - t0
        ev = json.loads(payload)
        u_ = ev.get("usage") or {}
        ct = u_.get("completion_tokens")
        if ev.get("choices"):
            finish = ev["choices"][0].get("finish_reason") or finish
        if ct is not None and ct > last_tokens:
            stamps.extend([now] * (ct - last_tokens))
            last_tokens = ct
        if u_:
            usage = u_
    conn.close()
    return stamps, usage, finish, len(data)


def run_openai(args, cell, out, path):
    wl = load_workload(args.workload)
    done = done_keys(out)
    nv = I.NVML() if args.nvml else None
    sampler = None
    if nv and nv.available:
        sampler = I.Sampler(nv, S.POWER_SAMPLE_S, 0)
        sampler.start()
    if args.shaping == "netem":
        I.append_jsonl(out, {"cell": cell["id"], "netem_before": netem_show(args.dev),
                             "netem_ok": netem_matches(netem_show(args.dev), cell.get("rtt_ms"), cell.get("bw_mbps"))})
    n_err = 0
    for req in requests_for(cell["id"], cell["block"], wl, args.with_depth):
        if (cell["id"], req["prompt"]["id"], req["rep"]) in done:
            continue
        body = {"model": args.served_model, "prompt": req["prompt"]["ids"], "max_tokens": S.GEN_TOKENS,
                "min_tokens": S.GEN_TOKENS, "ignore_eos": True, "temperature": 0.0, "stream": True,
                "stream_options": {"include_usage": True, "continuous_usage_stats": True}}
        rec = base_record(args, cell, req, path, {"served_model": args.served_model, "base_url": args.base_url,
                                                   "edge_class": args.edge_class})
        t0 = time.perf_counter()
        try:
            stamps, usage, finish, nbytes = _stream_completion(args.base_url, body)
            adj = 0
            if args.shaping == "app" and cell.get("rtt_ms"):
                adj = int(cell["rtt_ms"] * 1e6)      # one round trip: request + response propagation
                stamps = [s + adj for s in stamps]
            rec.update(ok=len(stamps) == S.GEN_TOKENS and (usage or {}).get("prompt_tokens") == req["prompt"]["n_in"],
                       token_ns=stamps, usage=usage, finish_reason=finish, request_bytes=nbytes, app_adjust_ns=adj)
        except Exception as e:
            n_err += 1
            rec.update(ok=False, error=f"{type(e).__name__}: {e}")
        if sampler:
            rec["nvml"] = sampler.summary(t0, time.perf_counter())
        I.append_jsonl(out, rec)
    if sampler:
        sampler.stop()
    if args.shaping == "netem":
        I.append_jsonl(out, {"cell": cell["id"], "netem_after": netem_show(args.dev)})
    return n_err


def profile_segment(args, out):
    """Synthetic-activation client: measures the server segment's own compute (t_compute_ns) per step."""
    import torch
    import split_runtime as R
    host, port = args.server.split(":")
    args.config_hash = R.config_hash(R.load_config(args.model_dir))
    cfg_hidden = S.QWEN3_32B_CONFIG["hidden_size"]

    class _Stub:   # just enough of a Slice for hello_for()
        class cfg:
            num_hidden_layers = S.QWEN3_32B_CONFIG["num_hidden_layers"]
            hidden_size = cfg_hidden
    hello = {"model_id": args.model_id, "config_hash": args.config_hash, "software": R.software_identity(),
             "k": args.k, "j": args.j, "n_layers": _Stub.cfg.num_hidden_layers, "hidden": cfg_hidden,
             "max_len": S.L_F, "wire_dtype": "bfloat16"}
    mid = R.RemoteMiddle(host, int(port), hello, "cpu", cfg_hidden)
    wl = load_workload(args.workload)
    g = torch.Generator().manual_seed(0)
    cell = {"id": f"P-server-U({args.k},{args.j})-load{args.load}", "block": "profile", "load": args.load}
    for req in requests_for(cell["id"], "core", wl, True):
        mid.reset()
        P = req["prompt"]["n_in"]
        steps = []
        for s in range(0, P, S.PREFILL_CHUNK):
            n = min(S.PREFILL_CHUNK, P - s)
            mid.forward(torch.randn((1, n, cfg_hidden), generator=g).to(torch.bfloat16), s, True)
            steps.append(dict(mid.last, n=n, start=s))
        for i in range(S.GEN_TOKENS - 1):
            mid.forward(torch.randn((1, 1, cfg_hidden), generator=g).to(torch.bfloat16), P + i, True)
            steps.append(dict(mid.last, n=1, start=P + i))
        rec = base_record(args, cell, req, "PROFILE-SERVER", {"k": args.k, "j": args.j,
                                                              "server_identity": mid.server_identity})
        rec.update(ok=True, steps=steps)
        I.append_jsonl(out, rec)
    mid.close()


def run_hf_local(args, out):
    """Informational engine-gap reference: unsplit HF Qwen3-32B, batch 1, greedy, forced length."""
    import torch
    import split_runtime as R
    sl = R.Slice(args.model_dir, list(range(64)), True, True, args.device, torch.bfloat16, S.QWEN3_32B_CONFIG)
    runner = R.EdgeRunner(sl, 32, 32, _NoMiddle(), max_len=S.L_F)
    wl = load_workload(args.workload)
    cell = {"id": "info-01", "block": "info"}
    for req in requests_for(cell["id"], "core", wl, True):
        rec = base_record(args, cell, req, "REMOTE-HF")
        toks = runner.generate(req["prompt"]["ids"], S.GEN_TOKENS, S.PREFILL_CHUNK, force_len=True)
        rec.update(ok=len(toks) == S.GEN_TOKENS, token_ns=runner.token_ns)
        I.append_jsonl(out, rec)


class _NoMiddle:
    last = {}

    def reset(self):
        pass

    def forward(self, h, start, use_cache):
        return h


def linkcheck(a):
    """HARDWARE-PLAN §3 go/no-go #3 over the split protocol itself (no ICMP / iperf3 needed):
    per-token round trip (n=1, 10 KB each way) and bulk throughput (n=2048 chunks) against an echo stub."""
    import statistics
    import torch
    import split_runtime as R
    H = S.QWEN3_32B_CONFIG["hidden_size"]
    host, port = a.server.split(":")
    hello = {"model_id": "echo", "k": 1, "j": 1, "n_layers": S.QWEN3_32B_CONFIG["num_hidden_layers"], "hidden": H}
    mid = R.RemoteMiddle(host, int(port), hello, "cpu", H)
    mid.reset()
    x1, xb = torch.zeros((1, 1, H), dtype=torch.bfloat16), torch.zeros((1, S.PREFILL_CHUNK, H), dtype=torch.bfloat16)
    rt = []
    for _ in range(a.n):
        mid.forward(x1, 0, False)
        rt.append(mid.last["rt_ns"] / 1e6)
    big = []
    for _ in range(a.n_big):
        mid.forward(xb, 0, False)
        big.append(mid.last["rt_ns"] / 1e9)
    mid.close()
    bits = 2 * (R.HDR.size + S.PREFILL_CHUNK * H * 2) * 8
    q = sorted(rt)
    res = {"server": a.server, "rt_ms_p50": statistics.median(rt), "rt_ms_p95": q[int(0.95 * (len(q) - 1))],
           "bulk_mbps_median": bits / statistics.median(big) / 1e6, "n": a.n, "n_big": a.n_big}
    res["go_rtt_lt_1ms"] = res["rt_ms_p50"] < 1.0
    res["go_bw_ge_1gbps"] = res["bulk_mbps_median"] >= 1000
    # G-link (PREREG §6.3, §9): the native link must realise the "native, 1 Gbps" cells, else Stage 3 is blocked
    res["joint_allowed"] = res["rt_ms_p50"] <= S.LINK_MAX_RTT_P50_MS and res["bulk_mbps_median"] >= S.LINK_MIN_BULK_MBPS
    if a.out:
        I.write_json(a.out, res)
    print(json.dumps(res, indent=1))
    return res


# --------------------------------------------------------------------------- #
# Load generator (server state kappa)                                         #
# --------------------------------------------------------------------------- #
def load_gen(args):
    """Keep exactly `concurrency` requests in flight; writes a status file the cell runner checks."""
    s1 = S.load_s1()
    sample = [json.loads(x) for x in open(s1.SAMPLE_JSONL)]
    stop = threading.Event()
    stats = {"done": 0, "errors": 0, "tokens": 0, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "t0": time.time(), "concurrency": args.concurrency}
    lock = threading.Lock()

    def worker(i):
        rng = random.Random(f"loadgen:{i}")
        u = urllib.parse.urlparse(args.base_url)
        while not stop.is_set():
            q = rng.choice(sample)
            body = {"model": args.served_model, "messages": [{"role": "user", "content": q["prompt"]}],
                    "max_tokens": 512, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": True}}
            try:
                c = http.client.HTTPConnection(u.hostname, u.port, timeout=900)
                c.request("POST", u.path.rstrip("/") + "/chat/completions", body=json.dumps(body),
                          headers={"Content-Type": "application/json"})
                r = json.loads(c.getresponse().read())
                c.close()
                with lock:
                    stats["done"] += 1
                    stats["tokens"] += r["usage"]["completion_tokens"]
            except Exception:
                with lock:
                    stats["errors"] += 1
                time.sleep(1)

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(args.concurrency)]
    for t in threads:
        t.start()

    class Status(http.server.BaseHTTPRequestHandler):   # GET /status: what run-cell on the edge checks
        def log_message(self, *a):
            pass

        def do_GET(self):
            with lock:
                st = dict(stats, uptime_s=time.time() - stats["t0"],
                          tok_per_s=stats["tokens"] / max(1e-9, time.time() - stats["t0"]), served_utc=time.time())
            b = json.dumps(st).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b)
    if args.status_port:
        srv = http.server.ThreadingHTTPServer(("0.0.0.0", args.status_port), Status)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        end = time.time() + args.duration_s if args.duration_s else None
        while end is None or time.time() < end:
            with lock:
                st = dict(stats, uptime_s=time.time() - stats["t0"],
                          tok_per_s=stats["tokens"] / max(1e-9, time.time() - stats["t0"]))
            I.write_json(args.status_file, st)
            time.sleep(5)
    except KeyboardInterrupt:
        pass
    stop.set()


def check_load(status, load, min_uptime_s=60):
    """status = a local status file or http://host:port/status of the server-side load generator."""
    if load == 0:
        return True, "no load"
    try:
        if str(status).startswith("http"):
            import urllib.request
            with urllib.request.urlopen(status, timeout=10) as r:
                st = json.loads(r.read())
            fresh = True
        else:
            st = json.loads(Path(status).read_text())
            fresh = time.time() - os.path.getmtime(status) < 30
    except Exception as e:
        return False, f"no load status: {e}"
    ok = st.get("concurrency") == load and st.get("uptime_s", 0) >= min_uptime_s and st.get("errors", 0) == 0 \
        and fresh
    return ok, st


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def cell_by_id(cid):
    for c in S.cells():
        if c["id"] == cid:
            return c
    raise SystemExit(f"unknown cell {cid}")


def resolve_split(path, f_results):
    if path == "SPLIT U-min":
        return S.U_MIN
    m = int(re.search(r"U-max\((\d+)\)", path).group(1))
    if f_results:
        r = json.loads(Path(f_results).read_text())
        kj = r["u_max"].get(str(m))
        if kj is None:
            raise SystemExit(f"class {m} failed F: its SPLIT cells are dropped (E0 §8 F)")
        return tuple(kj)
    raise SystemExit("--f-results (measured U-max) is required for U-max cells")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-workload")
    b.add_argument("--tokenizer-dir", required=True)
    b.add_argument("--out", default=str(WORKLOAD_JSON))
    sub.add_parser("plan")

    def common(p):
        p.add_argument("--out-dir", required=True)
        p.add_argument("--workload", default=str(WORKLOAD_JSON))
        p.add_argument("--shaping", choices=["none", "netem", "app"], default="none")
        p.add_argument("--dev", default="eth0", help="edge egress interface for netem")
        p.add_argument("--with-depth", type=lambda s: s == "1", default=None)
    for name in ("run-split", "run-cell"):
        p = sub.add_parser(name)
        common(p)
        p.add_argument("--model-dir")
        p.add_argument("--model-id", default=f"{S.SPLIT_REPO}@{S.SPLIT_REVISION}")
        p.add_argument("--server", help="host:port of the segment server (or echo stub)")
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--k", type=int)
        p.add_argument("--j", type=int)
        p.add_argument("--cap-class", type=int, choices=S.MEMORY_CLASSES)
        p.add_argument("--expect-32b", type=lambda s: s != "0", default=True)
        p.add_argument("--label", default=None, help="cell id for run-split profiles")
        p.add_argument("--base-url")
        p.add_argument("--served-model", default="Qwen3-32B")
        p.add_argument("--edge-class", type=int)
        p.add_argument("--nvml", action="store_true")
        p.add_argument("--f-results")
        p.add_argument("--load-status", default=None, help="http://SERVER:8099/status (load cells)")
    o = sub.add_parser("run-openai")
    common(o)
    o.add_argument("--base-url", required=True)
    o.add_argument("--served-model", required=True)
    o.add_argument("--path", choices=["REMOTE", "EDGE"], required=True)
    o.add_argument("--edge-class", type=int)
    o.add_argument("--label", required=True)
    o.add_argument("--nvml", action="store_true")
    ps = sub.add_parser("profile-segment")
    ps.add_argument("--out-dir", required=True)
    ps.add_argument("--workload", default=str(WORKLOAD_JSON))
    ps.add_argument("--server", required=True)
    ps.add_argument("--k", type=int, required=True)
    ps.add_argument("--j", type=int, required=True)
    ps.add_argument("--load", type=int, default=0)
    ps.add_argument("--model-id", default=f"{S.SPLIT_REPO}@{S.SPLIT_REVISION}")
    ps.add_argument("--model-dir", required=True, help="config.json source (no weights are loaded)")
    ps.add_argument("--shaping", default="none")
    hl = sub.add_parser("run-hf-local")
    hl.add_argument("--out-dir", required=True)
    hl.add_argument("--workload", default=str(WORKLOAD_JSON))
    hl.add_argument("--model-dir", required=True)
    hl.add_argument("--device", default="cuda:0")
    hl.add_argument("--shaping", default="none")
    lg = sub.add_parser("load-gen")
    lg.add_argument("--base-url", required=True)
    lg.add_argument("--served-model", default="Qwen3-32B")
    lg.add_argument("--concurrency", type=int, required=True)
    lg.add_argument("--duration-s", type=float, default=0)
    lg.add_argument("--status-file", required=True)
    lg.add_argument("--status-port", type=int, default=8099)
    lc = sub.add_parser("linkcheck")
    lc.add_argument("--server", required=True, help="host:port of an echo stub (split_runtime.py serve --echo)")
    lc.add_argument("--n", type=int, default=1000)
    lc.add_argument("--n-big", type=int, default=20)
    lc.add_argument("--out")
    n = sub.add_parser("netem")
    n.add_argument("--dev", required=True)
    n.add_argument("--rtt-ms", type=float, default=None)
    n.add_argument("--bw-mbps", type=int, default=None)
    n.add_argument("--action", choices=["print", "apply", "clear", "show"], default="print")
    a = ap.parse_args(argv)

    if a.cmd == "build-workload":
        w = build_workload(a.tokenizer_dir)
        I.write_json(a.out, w)
        print(f"wrote {a.out}: {len(w['prompts'])} prompts, ids sha256 {w['ids_sha256']}")
    elif a.cmd == "plan":
        for c in execution_order():
            print(json.dumps(c))
    elif a.cmd == "run-split":
        cell = {"id": a.label or f"P-edge-U({a.k},{a.j})-cap{a.cap_class}", "block": "profile", "rtt_ms": None,
                "bw_mbps": None, "load": 0}
        out = Path(a.out_dir) / f"{cell['id']}.jsonl"
        sys.exit(1 if run_split(a, cell, out) else 0)
    elif a.cmd == "run-openai":
        cell = {"id": a.label, "block": "profile", "rtt_ms": None, "bw_mbps": None, "load": 0}
        out = Path(a.out_dir) / f"{a.label}.jsonl"
        sys.exit(1 if run_openai(a, cell, out, a.path) else 0)
    elif a.cmd == "run-cell":
        cell = cell_by_id(a.label)
        ok, st = check_load(a.load_status, cell["load"])
        if not ok:
            sys.exit(f"cell {cell['id']} needs load {cell['load']}: {st}")
        if (cell["rtt_ms"] or cell["bw_mbps"] < 1000) and a.shaping == "none":
            sys.exit(f"cell {cell['id']} needs shaping (netem or app)")
        if a.shaping == "netem" and not netem_matches(netem_show(a.dev), cell["rtt_ms"], cell["bw_mbps"]):
            sys.exit(f"netem state does not match cell {cell['id']}:\n{netem_show(a.dev)}")
        out = Path(a.out_dir) / f"{cell['id']}.jsonl"
        if cell["path"].startswith("SPLIT"):
            a.k, a.j = resolve_split(cell["path"], a.f_results)
            m = re.search(r"U-max\((\d+)\)", cell["path"])
            a.cap_class = int(m.group(1)) if m else min(S.MEMORY_CLASSES)   # every SPLIT cell runs under its cap
            sys.exit(1 if run_split(a, cell, out) else 0)
        if cell["path"] == "REMOTE":
            sys.exit(1 if run_openai(a, cell, out, "REMOTE") else 0)
        sys.exit(f"cell {cell['id']} ({cell['path']}) is measured in a single-machine stage (RUNBOOK)")
    elif a.cmd == "profile-segment":
        profile_segment(a, Path(a.out_dir) / f"P-server-U({a.k},{a.j})-load{a.load}.jsonl")
    elif a.cmd == "run-hf-local":
        run_hf_local(a, Path(a.out_dir) / "info-01.jsonl")
    elif a.cmd == "load-gen":
        load_gen(a)
    elif a.cmd == "linkcheck":
        linkcheck(a)
    elif a.cmd == "netem":
        if a.action == "show":
            print(netem_show(a.dev))
            return
        cmds = netem_commands(a.dev, a.rtt_ms, a.bw_mbps, "clear" if a.action == "clear" else "apply")
        for c in cmds:
            print(c)
            if a.action in ("apply", "clear"):
                subprocess.run(c, shell=True, check=not c.endswith("|| true"))


if __name__ == "__main__":
    main()
