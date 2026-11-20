"""QWEN-SELECT drivers (PREREG section 5; exact order in RUNBOOK.md).

  select-gpu    --site TEMPLATE --out F          server, CPU only, once: choose the GPU by the fixed rule and pin it in F
  preflight     --role edge|server --site F     CPU only: identity, checkpoint hashes, tokenisation, memory gates, ports
  serve         --site F                         server: the split's server half, until idle, Ctrl-C or its budget
  serve-remote  --site F                         server: the whole REMOTE model, until idle, Ctrl-C or its budget
  reference     --site F                         edge: whole 32B model on the correctness set (no timers); records the screen decision
  correctness   --site F                         edge: split against that reference (no timers) -> PASS / FAIL
  timed         --site F --block B              edge: end-to-end timing of one block, in this order:
                                                 S1a E32a E14a E8a (first 80 queries), S1b S2 E32b E14b E8b (last 80; S2 = sentinel), R14
  ledger        --role edge|server --site F     GPU-hours used and left on this host (reads the ledger; no GPU)
  score         --edge-run DIR --workload-private FILE          analysis host: unchanged benchmark evaluators
  summary       --edge-run DIR --server-run DIR --out DIR       analysis host

Every target command first checks the frozen tag, the site file, the host role, the pinned checkpoints, the free
memory and the GPU-hour allotment. --dev is for CPU tests with tiny random models only; dev outputs are refused by
`summary` as results.
"""
import argparse
import json
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qsel_common as C  # noqa: E402
import qsel_models as M  # noqa: E402
import qsel_runtime as RT  # noqa: E402
import qsel_spec as S  # noqa: E402
import qsel_stats as ST  # noqa: E402
import qsel_workload as WL  # noqa: E402

STATE = {"run_dir": None, "hdr": None, "ledger": None}


class LimitReached(BaseException):
    """The wall-clock limit of a command. Deliberately not an Exception: TimeoutError is an OSError, which the reused
    server loops catch and survive, and a limit must end the command."""


def arm(seconds):
    """Wall-clock limit of a command (PREREG section 12). Reaching it ends the command as an attempt that died."""
    def stop(*_):
        raise LimitReached(f"wall-clock limit of {int(seconds)} s reached")
    signal.signal(signal.SIGALRM, stop)
    signal.alarm(max(1, int(seconds)))


def versions():
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for p in ("torch", "transformers", "safetensors", "tokenizers", "numpy"):
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = None
    return out


def dev_of(a, site):
    """Overrides of the CPU tests (site["dev"], and one environment variable); honoured only with --dev."""
    if not a.dev:
        return {}
    d = dict(site.get("dev") or {})
    if os.environ.get("QSEL_DEV_ALLOTMENT_H"):
        d["allotment_h"] = float(os.environ["QSEL_DEV_ALLOTMENT_H"])
    return d


def sizes(a, site):
    d = dev_of(a, site)
    return {"cutoff_s": float(d.get("cutoff_s", S.QUERY_CUTOFF_S)), "max_new": int(d.get("max_new", S.MAX_NEW_TOKENS)),
            "context": int(d.get("context", RT.DEV_CONTEXT if a.dev else S.CONTEXT_TOKENS)),
            "chunk": int(d.get("chunk", RT.DEV_CHUNK if a.dev else S.PREFILL_CHUNK)),
            "n_cont": int(d.get("n_cont", RT.DEV_N_CONT if a.dev else S.N_CONT)), "limit": d.get("limit")}


def runtime_params(a, site, model="32B"):
    sz = sizes(a, site)
    return RT.params(a.dev, model, context=sz["context"], chunk=sz["chunk"], n_cont=sz["n_cont"])


def workload(a, site):
    d = dev_of(a, site)
    return WL.load(site["workload_private"], dev_workload=C.read_json(d["workload_public"]) if a.dev else None)


def checkpoints(a, site, role):
    """Every checkpoint this role loads, verified against its frozen manifest (hashes cached per file state)."""
    if a.dev:
        return None
    out = {}
    for m in (("32B", "14B", "8B") if role == "edge" else ("32B", "14B")):
        use = "edge" if role == "edge" else ("split_server" if m == S.SPLIT_MODEL else "server_whole")
        rec = M.verify(site["model_dirs"][m], m, use, cache_file=site["hash_cache"])
        if not rec["ok"]:
            raise C.Stop(f"checkpoint {m} is not the pinned one: {rec['problems'][:3]}")
        if role == "edge" and tuple(M.eos_ids(site["model_dirs"][m])) != tuple(sorted(S.EOS_IDS)):
            raise C.Stop(f"checkpoint {m}: EOS set differs from the pinned one")
        out[m] = {"identity_sha256": rec["identity_sha256"], "bytes_verified": rec["bytes_verified"], "revision": rec["revision"]}
    return out


def prepare(a, role, cuda=True):
    """Checks that precede every command. Returns (site, run_dir, header). Nothing is loaded before this returns."""
    fz = C.freeze_check(a.dev)
    site = C.load_site(a.site, role)
    if versions()["transformers"] != S.TRANSFORMERS_REQUIRED:
        raise C.Stop(f"this interpreter does not have transformers {S.TRANSFORMERS_REQUIRED}, which the reused runtime needs")
    if not cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""                                # this command must not touch a GPU
    elif site.get("gpu_uuid") and str(site["device"]).startswith("cuda"):
        os.environ["CUDA_VISIBLE_DEVICES"] = site["gpu_uuid"]                  # no other GPU is visible to this process
    host = C.check_role(role, site, dev=a.dev)
    sel = None
    if role == "server" and (not a.dev or (Path(site["run_dir"]) / C.GPU_SELECTION).exists()):
        sel = C.check_selection(site, site["run_dir"])
    hdr = {"schema": S.SCHEMA, "experiment": S.NAME, "freeze": fz, "dev": bool(a.dev), "role": role, "site_sha256": site["_sha256"],
           "host": host, "utc_start": C.utc(), "checkpoints": checkpoints(a, site, role), "gpu_selection": sel, "versions": versions(),
           "others_on_gpu_start": C.gpu_others(site.get("gpu_uuid"), (os.getpid(),)) if cuda else None}
    STATE.update(run_dir=Path(site["run_dir"]), hdr=hdr)
    return site, Path(site["run_dir"]), hdr


def need_of(a, load, reference_seq=None):
    return 1 << 20 if a.dev else M.needs(reference_seq)[load]["required_free"]


def gate(a, role, site, hdr, load, reference_seq=None):
    """Free memory before anything is loaded (recorded in the header)."""
    need = need_of(a, load, reference_seq)
    hdr["memory_gate"] = {"load": load, **(C.memory_gate(role, site, need) if not a.dev else {"need_bytes": need, "pass": True})}
    return need


def gpu_gate(a, site, need):
    """After torch is imported and before any weight is read: one visible GPU, the site's, and the in-process memory check."""
    import torch
    dev = site["device"]
    if not str(dev).startswith("cuda"):
        return {"device": "cpu"}
    if torch.cuda.device_count() != 1:
        raise C.Stop("exactly one GPU must be visible to this process")
    p = torch.cuda.get_device_properties(0)
    if site.get("gpu_uuid") and C.norm_uuid(getattr(p, "uuid", "")) != C.norm_uuid(site["gpu_uuid"]):
        raise C.Stop("the visible GPU is not the site's GPU")
    free, total = torch.cuda.mem_get_info(0)
    return dict(C.device_gate(site["role"], need, free), device=dev, total_bytes=total, is_integrated=bool(getattr(p, "is_integrated", False)))


def budget(a, site, rd, role, stage, natural_s):
    """Open a ledger entry for a GPU stage and arm the wall-clock limit with what the allotment leaves."""
    led = C.Ledger(rd, role, allotment_h=dev_of(a, site).get("allotment_h"))
    used = led.used_s() / 3600
    eid, limit = led.open(stage, natural_s)
    STATE["ledger"] = led
    arm(limit)
    return {"ledger_id": eid, "wall_clock_limit_s": limit, "gpu_hours_used_before": used, "allotment_h": led.allot_s / 3600}


def record_failure(cmd, exc):
    """A command that died after its checks passed leaves a record (type and first words only). Stops leave none."""
    rd = STATE["run_dir"]
    if rd is None or STATE["hdr"] is None:
        return
    n = 1
    while (rd / f"failure_{cmd}.{n}.json").exists():
        n += 1
    C.write_json(rd / f"failure_{cmd}.{n}.json", dict(STATE["hdr"], utc_end=C.utc(), command=cmd, error_type=type(exc).__name__, error=str(exc)[:300]))


def failures(rd, cmd):
    return len(list(Path(rd).glob(f"failure_{cmd}.*.json")))


def attempts(rd, stem):
    """Attempts of a stage so far: one marker per attempt, written once its checks have passed and before anything is
    loaded, so that an attempt that is killed without a trace still counts."""
    return len(list(Path(rd).glob(f"{stem}.attempt*.json")))


def mark_attempt(rd, stem, hdr, **body):
    n = attempts(rd, stem) + 1
    C.write_json(Path(rd) / f"{stem}.attempt{n}.json", dict(hdr, attempt=n, **body))
    return n


def finish(hdr, site, **body):
    return dict(hdr, utc_end=C.utc(), others_on_gpu_end=C.gpu_others(site.get("gpu_uuid"), (os.getpid(),)) if hdr.get("others_on_gpu_start") is not None else None, **body)


def numbered(rd, stem):
    n = 1
    while (rd / f"{stem}.{n}.json").exists():
        n += 1
    return n


def torch_setup():
    import torch
    _, R = RT.k16()
    R.setup_torch()
    return torch, R


def expected_cfg(a):
    K, _ = RT.k16()
    return None if a.dev else K.E0.QWEN3_32B_CONFIG


def mem_stats(R, device):
    return {k: v for k, v in R.mem_stats(device).items()}


# --------------------------------------------------------------------------- #
def cmd_select_gpu(a):
    """Server, CPU only: the GPU is chosen by the fixed rule and pinned in a new site file. Until a server stage has
    written an output, a selection may be replaced by a new one (the earlier record and site file are kept, renamed).
    After that it is fixed for the run."""
    fz = C.freeze_check(a.dev)
    site = C.load_site(a.site, "server", selecting=True)
    rd, out = Path(site["run_dir"]), Path(a.out)
    host = C.host_facts()
    if not a.dev and host["hostname"] != site["expected_hostname"]:
        raise C.Stop("this host is not the one the site file names")
    old = C.read_json(rd / C.GPU_SELECTION) if (rd / C.GPU_SELECTION).exists() else None
    if old is not None and C.server_stage_began(rd):
        raise C.Stop("a server stage has run on the selected GPU: the selection is fixed for this run")
    old_site = Path(old["site_path"]) if old is not None else None
    if out.exists() and out.resolve() != old_site:
        raise C.Stop("the output site file exists: outputs are written once")
    if old_site is not None and old_site.exists() and C.sha256_file(old_site) != old["site_sha256"]:
        raise C.Stop("the earlier pinned site file was changed by hand")
    need = max(need_of(a, "split_server"), need_of(a, "server_whole_14B"))     # the GPU hosts either load, never both at once
    mem = C.meminfo()
    fake = (site.get("dev") or {}) if a.dev else {}                             # CPU tests only: recorded probe text in place of the probe
    rows = C.parse_gpu_rows(fake["gpu_rows"]) if "gpu_rows" in fake else C.gpu_rows()
    chosen, considered = C.select_gpu(rows, fake.get("gpu_apps", "") if "gpu_rows" in fake else C.apps_text(), need, site["excluded_gpu_uuids"])
    for c in considered:
        print(f"GPU index {c['index']}: free {c['free_mib']} MiB, other compute processes {c['other_processes']}: "
              + ("eligible" if c["eligible"] else c["reason"]) + ("  <- SELECTED" if chosen is not None and c["uuid"] == chosen["uuid"] else ""))
    if chosen is None or (not a.dev and (mem.get("MemAvailable") or 0) < S.HOST_RAM_MIN_BYTES):
        raise C.Stop(f"BLOCKED, no eligible GPU with {need >> 20} MiB free now (nothing was selected or loaded; try later, "
                     "never ask anyone to free memory)")
    superseded = 0
    if old is not None:                                                         # replaced before any server stage: kept under another name
        superseded = C.set_aside(rd, C.GPU_SELECTION, label="superseded")
        if old_site.exists():
            old_site.rename(old_site.with_name(old_site.name + f".superseded-{superseded}"))
    pinned = {k: v for k, v in site.items() if not k.startswith("_")}
    pinned["gpu_uuid"] = "GPU-" + chosen["uuid"]
    out.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps(pinned, indent=1) + "\n")
    C.write_json(rd / C.GPU_SELECTION, {"schema": S.SCHEMA, "experiment": S.NAME, "freeze": fz, "dev": bool(a.dev), "host": host,
                                        "utc_start": C.utc(), "template_sha256": site["_sha256"], "site_sha256": C.sha256_file(out),
                                        "site_path": str(out.resolve()), "earlier_selections_set_aside": superseded,
                                        "need_bytes": need, "ram_available_bytes": mem.get("MemAvailable"), "considered": considered,
                                        "selected": chosen, "rule": "fewest other compute processes, then most free memory, then lowest index"})
    print(f"selected GPU index {chosen['index']} ({chosen['free_mib']} MiB free, required {need >> 20} MiB); pinned in {out}")


def cmd_preflight(a):
    site, rd, hdr = prepare(a, a.role, cuda=False)
    token_ok = bool(C.check_token(site, create=a.role == "server"))
    gates, blocked, tok_check, remote_fit = {}, [], None, None
    if a.role == "edge":
        wl = workload(a, site)
        tok = WL.load_tokenizer(site["model_dirs"]["32B"])
        n = [len(WL.token_ids(tok, q)) for q in wl["utility"]]                  # refuses any query whose token ids are not the pinned ones
        tok_check = {"queries": len(n), "max_prompt_tokens": max(n), "all_equal_pinned_ids": True}
        ref_seq = max(q["n_prompt"] for q in wl["correctness"]) + sizes(a, site)["n_cont"]
        loads = [("split_edge", None), ("edge_reference", ref_seq), ("edge_whole_32B", None), ("edge_whole_14B", None), ("edge_whole_8B", None)]
    else:
        loads = [("split_server", None), ("server_whole_14B", None)]
        if not a.dev:
            row = next((r for r in C.gpu_rows() if r["uuid"] == C.norm_uuid(site["gpu_uuid"])), None)
            need32 = need_of(a, "server_whole_32B")
            remote_fit = {"model": "32B", "need_bytes": need32, "gpu_total_mib": row and row["total_mib"],
                          "fits_one_server_gpu": bool(row and row["total_mib"] and row["total_mib"] * (1 << 20) >= need32)}
    for load, seq in loads:
        need = need_of(a, load, seq)
        try:
            gates[load] = C.memory_gate(a.role, site, need) if not a.dev else {"need_bytes": need, "pass": True}
        except C.Stop:
            gates[load] = {"pass": False, "need_bytes": need}
            blocked.append(load)
    ports = {name: not C.port_free(site[name]) for name in ("split_port", "remote_port")}     # True = something listens
    port_ok = not any(ports.values()) if a.role == "server" else all(ports.values())
    led = C.Ledger(rd, a.role, allotment_h=dev_of(a, site).get("allotment_h")).report()
    rec = finish(hdr, site, gates=gates, ports_listening=ports, ports_ok=port_ok, token_file_ok=token_ok, tokenisation=tok_check,
                 remote_32B=remote_fit, ledger={k: led[k] for k in ("allotment_h", "used_h", "remaining_h")},
                 verdict="GO" if not blocked and port_ok else "BLOCKED")
    n = 1
    while (rd / ("preflight.json" if n == 1 else f"preflight.{n}.json")).exists():
        n += 1
    C.write_json(rd / ("preflight.json" if n == 1 else f"preflight.{n}.json"), rec)
    print(f"preflight {a.role}: {rec['verdict']}  required MiB { {k: v['need_bytes'] >> 20 for k, v in gates.items()} }  "
          f"blocked now {blocked}  ports {'ok' if port_ok else ('must be free' if a.role == 'server' else 'forwards are not up')}")
    print(f"GPU-hours on this host: used {led['used_h']:.2f} of {led['allotment_h']:.2f}")
    if a.role == "edge":
        m = C.meminfo()
        print(f"RAM now: available {(m.get('MemAvailable') or 0) >> 20} MiB. Tokenisation: {tok_check['queries']} queries equal the pinned ids, "
              f"longest prompt {tok_check['max_prompt_tokens']} tokens.")
    elif remote_fit:
        print(f"REMOTE with the 32B model: needs {remote_fit['need_bytes'] >> 20} MiB on one GPU of {remote_fit['gpu_total_mib']} MiB -> "
              f"{'fits (unexpected: report this line)' if remote_fit['fits_one_server_gpu'] else 'does not fit; excluded as pre-registered'}")
    sys.exit(0 if rec["verdict"] == "GO" else 3)


def _server_loop(a, site, rd, hdr, stage, started_stem, record_stem, load, build):
    """Shared body of the two server commands: gates, ledger, load, listen on loopback, serve, record, release."""
    port = site["split_port"] if stage == "serve" else site["remote_port"]
    if not C.port_free(port):
        raise C.Stop("the site's port is already in use")
    need = gate(a, "server", site, hdr, load)
    n = numbered(rd, started_stem)
    bud = budget(a, site, rd, "server", stage, None)
    torch, R = torch_setup()
    gpu = gpu_gate(a, site, need)
    events = rd / f"{record_stem.replace('record', 'events')}.{n}.jsonl"
    count = {"connection_done": 0, "connection_open": 0, "hello_ok": 0, "refused": 0, "connection_closed": 0, "bye": 0}

    def log(ev):
        if ev.get("event") in count:
            count[ev["event"]] += 1
        C.append_jsonl(events, {k: v for k, v in dict(ev, utc=C.utc()).items() if k in ("event", "utc", "why", "mode", "n_fwd", "n_requests")})
    stop, asked = threading.Event(), []

    def on_signal(*_):                                                          # once: finish the open connection, then release. Twice: now
        asked.append(1)
        stop.set()
        if len(asked) > 1:
            raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    serve_fn, ident, close = build(torch, R, log, stop)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", int(port)))                                       # loopback only: the edge comes through its SSH forward
    sock.listen(1)
    C.write_json(rd / f"{started_stem}.{n}.json", dict(hdr, gpu_gate=gpu, budget=bud, identity=ident))   # from here on the selected GPU is fixed
    print(json.dumps({"ready": True, "stage": stage, "port": int(port), "wall_clock_limit_h": round(bud["wall_clock_limit_s"] / 3600, 2)}), flush=True)
    d = dev_of(a, site)
    why, t0 = "error", C.utc()
    try:
        why = serve_fn(sock, float(d.get("first_wait_s", S.SERVER_FIRST_WAIT_S)), float(d.get("idle_exit_s", S.SERVER_IDLE_EXIT_S)))
    except LimitReached:
        why = "budget_or_wall_clock_limit"
    finally:
        signal.alarm(0)
        sock.close()
        mem = mem_stats(R, site["device"])
        close()
        STATE["ledger"].close(why)
        C.write_json(rd / f"{record_stem}.{n}.json", finish(hdr, site, gpu_gate=gpu, budget=bud, identity=ident, ended=why, events=count,
                                                            served={"from_utc": t0, "to_utc": C.utc()}, memory=mem))
        print(f"server record written: {record_stem}.{n}.json ({why}); the GPU is released")


def cmd_serve(a):
    site, rd, hdr = prepare(a, "server")
    p = runtime_params(a, site)

    def build(torch, R, log, stop):
        K, _ = RT.k16()
        side = R.ServerSide(site["model_dirs"]["32B"], site["device"], p, expected_cfg(a), with_static=False)
        if side.sl.layer_ids != K.SERVER_LAYERS or side.sl.embed is not None or side.sl.lm_head is not None:
            raise RuntimeError("the loaded slice is not exactly the server layers of the cut")
        if a.dev and os.environ.get("QSEL_DEV_SABOTAGE") == "1":                # CPU tests only: a wrong layer order must fail the gate
            side.segs[S.SPLIT_MODE].layer_ids = list(reversed(side.segs[S.SPLIT_MODE].layer_ids))
        ident = {"layers": [side.sl.layer_ids[0], side.sl.layer_ids[-1]], "param_bytes": side.sl.param_bytes, "mode": S.SPLIT_MODE,
                 "load_memory": R.load_memory(site["device"])}

        def serve_fn(sock, first_wait_s, idle_exit_s):
            last = {"t": time.time(), "served": False}

            def log2(ev):
                if ev.get("event") in ("connection_open", "connection_done", "connection_closed"):
                    last.update(t=time.time(), served=True)
                log(ev)
            tick = float(dev_of(a, site).get("accept_tick_s", 30))
            while True:                                                         # k16_runtime.serve, re-entered so that our idle rule applies
                r = R.serve(side, sock, hdr["freeze"]["spec_hash"], log2, extra_identity={"experiment": S.NAME}, accept_timeout_s=tick, stop=stop)
                if r != "accept_timeout":
                    return r
                if time.time() - last["t"] >= (idle_exit_s if last["served"] else first_wait_s):
                    return "idle" if last["served"] else "no_edge"
        return serve_fn, ident, lambda: None
    _server_loop(a, site, rd, hdr, "serve", "server_started", "server_record", "split_server", build)


def cmd_serve_remote(a):
    site, rd, hdr = prepare(a, "server")
    model = S.CONFIGS["R14"]["model"]
    token = C.check_token(site)
    sz = sizes(a, site)

    def build(torch, R, log, stop):
        hf = R.hf_model(site["model_dirs"][model], site["device"])
        ident = {"spec_hash": hdr["freeze"]["spec_hash"], "model": model, "context_tokens": sz["context"], "experiment": S.NAME,
                 "load_memory": R.load_memory(site["device"])}
        srv = RT.RemoteServer(torch, hf, site["device"], R.E0RT._sync, ident, token, sz["chunk"], log=log)
        return (lambda sock, first_wait_s, idle_exit_s: RT.serve_remote(srv, sock, first_wait_s, idle_exit_s, stop=stop)), ident, (lambda: None)
    _server_loop(a, site, rd, hdr, "serve-remote", "remote_started", "remote_record", f"server_whole_{model}", build)


# --------------------------------------------------------------------------- #
def correctness_prompts(a, site, tok, wl):
    return [{"id": f"c{i:02d}", "key": q["key"], "ids": WL.token_ids(tok, q)} for i, q in enumerate(wl["correctness"])]


def cmd_reference(a):
    site, rd, hdr = prepare(a, "edge")
    if (rd / "reference.json").exists():
        raise C.Stop("reference.json is complete: outputs are written once")
    if attempts(rd, "reference") >= S.MAX_STAGE_ATTEMPTS:
        raise C.Stop("reference already had two attempts: NOT_EVALUABLE (PREREG section 13)")
    died = C.set_aside(rd, "reference")                                        # an earlier attempt died: its files are kept, renamed
    wl, sz = workload(a, site), sizes(a, site)
    need = gate(a, "edge", site, hdr, "edge_reference", max(q["n_prompt"] for q in wl["correctness"]) + sz["n_cont"])
    bud = budget(a, site, rd, "edge", "reference", S.LIMITS_S["reference"])
    torch, R = torch_setup()
    gpu = gpu_gate(a, site, need)
    mark_attempt(rd, "reference", hdr, budget=bud)                             # from here on this is an attempt
    tok = WL.load_tokenizer(site["model_dirs"]["32B"])
    prompts = correctness_prompts(a, site, tok, wl)
    hf = R.hf_model(site["model_dirs"]["32B"], site["device"])
    load_mem = R.load_memory(site["device"])
    p = runtime_params(a, site)
    rows = R.write_reference(hf, prompts, p, site["device"], rd / "reference")
    ok = len(rows) == len(prompts) and all(r["finite"] and not r["degenerate"] for r in rows)
    if not ok:
        raise RuntimeError("the reference is not finite or is degenerate")
    files = {f.name: C.sha256_file(f) for f in sorted((rd / "reference").glob("*.npz"))}
    STATE["ledger"].close("done")
    C.write_json(rd / "reference.json", finish(hdr, site, gpu_gate=gpu, budget=bud, load_memory=load_mem, memory=mem_stats(R, site["device"]),
                 items=[dict(r, key=pr["key"], ids_sha256=WL.ids_sha(pr["ids"])) for r, pr in zip(rows, prompts)], files=files,
                 died_attempts_set_aside=died, n_cont=sz["n_cont"],
                 screen_decision={"stage": 1, "decision": "EDGE",
                                  "basis": "the split's model is the whole 32B model, which loaded on the edge: quality lead 0.0 points "
                                           "< 3.0 by identity (PREREG section 6)"}))
    print(f"reference.json written ({len(rows)} prompts); screen decision recorded: EDGE at Stage 1")


def require_reference(rd, hdr, a):
    if not (rd / "reference.json").exists():
        raise C.Stop("reference must complete first")
    ref = C.read_json(rd / "reference.json")
    if ref["dev"] != bool(a.dev) or ref["freeze"] != hdr["freeze"]:
        raise C.Stop("reference.json belongs to another freeze or mode")
    for name, h in ref["files"].items():
        if C.sha256_file(rd / "reference" / name) != h:
            raise C.Stop("reference arrays changed on disk")
    return ref


def split_edge(a, site, hdr, torch, R, p):
    """The edge half of the split, connected to the experiment's server: (USplitEdge, connection, record)."""
    K, _ = RT.k16()
    sl = R.edge_slice(site["model_dirs"]["32B"], K.K, K.J, site["device"], torch.bfloat16, expected_cfg(a))
    load_mem = R.load_memory(site["device"])
    own = R.ownership(sl, K.K, K.J)
    if not own["edge_slice_ok"] or sl.layer_ids != sorted(K.EDGE_LAYERS):
        raise RuntimeError("the loaded slice is not exactly the edge layers of the cut")
    mid = R.connect(C.local_addr(site, "split_port"), sl, S.SPLIT_MODE, hdr["freeze"]["spec_hash"], p)
    e, info = R.edge_split(sl, mid, S.SPLIT_MODE, p)
    si = mid.server_identity
    return e, mid, {"mode": info, "load_memory": load_mem, "edge_layers": [K.EDGE_LAYERS[0], K.EDGE_LAYERS[-1]],
                    "server": {"layers": si.get("loaded_layers"), "device_name": (si.get("device") or {}).get("name"), "param_bytes": si.get("param_bytes")}}


def server_must_answer(ok, what):
    """A stage that needs a server starts only if the experiment's own server answers through the forward. Otherwise it
    is a stop of the operator's setup, before anything is loaded, and not an attempt."""
    if not ok:
        raise C.Stop(f"the experiment's {what} server does not answer through the forward (nothing was loaded; check the server "
                     "terminal and the SSH forward, then run the same command again)")


def cmd_correctness(a):
    site, rd, hdr = prepare(a, "edge")
    if (rd / "correctness.json").exists():
        raise C.Stop("correctness.json is complete: outputs are written once")
    if attempts(rd, "correctness") >= S.MAX_STAGE_ATTEMPTS:
        raise C.Stop("correctness already had two attempts that did not finish: SPLIT_INFEASIBLE (PREREG section 13)")
    ref = require_reference(rd, hdr, a)
    wl, sz = workload(a, site), sizes(a, site)
    p = runtime_params(a, site)
    need = gate(a, "edge", site, hdr, "split_edge")
    _, R0 = RT.k16()
    server_must_answer(RT.split_ready(R0, C.local_addr(site, "split_port"), site["model_dirs"]["32B"], hdr["freeze"]["spec_hash"], p), "split")
    died = C.set_aside(rd, "correctness_arrays")
    bud = budget(a, site, rd, "edge", "correctness", S.LIMITS_S["correctness"])
    torch, R = torch_setup()
    gpu = gpu_gate(a, site, need)
    mark_attempt(rd, "correctness", hdr, budget=bud)                           # from here on this is an attempt
    tok = WL.load_tokenizer(site["model_dirs"]["32B"])
    prompts = correctness_prompts(a, site, tok, wl)
    if [WL.ids_sha(pr["ids"]) for pr in prompts] != [it["ids_sha256"] for it in ref["items"]]:
        raise C.Stop("token ids differ from the reference run")
    e, mid, info = split_edge(a, site, hdr, torch, R, p)
    res = R.evaluate_config(e, prompts, rd / "reference", p, rd / "correctness_arrays", "split")
    mid.close()
    verdict = bool(res["pass"] and res.get("kv_positions_in_step_all"))
    STATE["ledger"].close("done")
    C.write_json(rd / "correctness.json", finish(hdr, site, gpu_gate=gpu, budget=bud, split=info, result=res, verdict="PASS" if verdict else "FAIL",
                                                 memory=mem_stats(R, site["device"]), died_attempts_set_aside=died))
    tf, g = res.get("teacher_forced", {}), res.get("greedy", {})
    print(f"correctness: {'PASS' if verdict else 'FAIL'}  decisive disagreements {tf.get('decisive_disagreements')}/{tf.get('n_decisive')}  "
          f"mean KL {tf.get('mean_kl_nats')}  greedy ok {g.get('identical_or_near_tie')}/{g.get('n_prompts')}")
    sys.exit(0 if verdict else 2)


def block_queries(wl, block, limit=None):
    """(queries of a block in order, index of its first query in the utility order). The halves are the first and the
    second half of the seeded utility order; S2 is the sentinel set; R14 is everything."""
    util = wl["utility"][:int(limit)] if limit else wl["utility"]
    half, kind = len(util) // 2, S.BLOCK_QUERIES[block]
    if kind == "first":
        return util[:half], 0
    if kind == "second":
        return util[half:], half
    if kind == "sentinel":
        keys = {q["key"] for q in util}
        return [q for q in wl["sentinel"] if q["key"] in keys], 0
    return util, 0


def block_parts(rd, block):
    """(part files in order, valid rows so far, torn lines). A part is one attempt's rows; nothing is ever rewritten."""
    parts = sorted(rd.glob(f"timed_{block}.part*.jsonl"), key=lambda p: int(p.name.split(".part")[1].split(".")[0]))
    rows, torn = [], 0
    for p in parts:
        r, t = C.read_jsonl(p, tolerate_torn_tail=True)
        rows += r
        torn += t
    return parts, rows, torn


def cmd_timed(a):
    cfg = S.BLOCK_CONFIG[a.block]
    kind, model = S.CONFIGS[cfg]["execution"], S.CONFIGS[cfg]["model"]
    site, rd, hdr = prepare(a, "edge", cuda=kind != "REMOTE")
    corr = C.read_json(rd / "correctness.json") if (rd / "correctness.json").exists() else None
    if corr is None or corr["verdict"] != "PASS" or corr["dev"] != bool(a.dev) or corr["freeze"] != hdr["freeze"]:
        raise C.Stop("no timing without a correctness PASS under this freeze (PREREG section 7)")
    for prev in S.BLOCKS[:S.BLOCKS.index(a.block)]:
        if not (rd / f"timed_{prev}.done.json").exists():
            raise C.Stop(f"block {prev} must complete before {a.block}")
    done = rd / f"timed_{a.block}.done.json"
    if done.exists():
        raise C.Stop(f"block {a.block} is complete: outputs are written once")
    parts, have, torn = block_parts(rd, a.block)
    n_attempts = attempts(rd, f"timed_{a.block}")
    if n_attempts >= S.MAX_BLOCK_ATTEMPTS:
        raise C.Stop(f"block {a.block} already had {n_attempts} attempts: INCOMPLETE (PREREG section 12)")
    wl, sz = workload(a, site), sizes(a, site)
    queries, first = block_queries(wl, a.block, sz["limit"])
    if not queries:
        raise C.Stop(f"block {a.block} has no queries")
    if [r["key"] for r in have] != [q["key"] for q in queries[:len(have)]] or [r["n"] for r in have] != list(range(first, first + len(have))):
        raise C.Stop(f"block {a.block}: earlier rows are not a prefix of the block's queries")
    if a.block == S.SECOND_HALF_GATE_BLOCK and not n_attempts:                   # PREREG section 12: no second half that cannot be finished
        led = C.Ledger(rd, "edge", allotment_h=dev_of(a, site).get("allotment_h"))
        spent = sum(led.charged(e) for e in led.entries() if "open" in e and e["stage"] in [f"timed-{b}" for b in S.FIRST_HALF_BLOCKS])
        if led.remaining_s() < S.SECOND_HALF_BUDGET_FACTOR * spent:
            raise C.Stop(f"BUDGET: the first halves used {spent / 3600:.2f} GPU-hours on this host and {led.remaining_s() / 3600:.2f} are left: the second "
                         "halves are not started. Nothing was loaded. The run is INCOMPLETE; the first halves are reported as a balanced partial result")
    p = runtime_params(a, site, model)
    spec_hash, attempt = hdr["freeze"]["spec_hash"], n_attempts + 1
    out = rd / f"timed_{a.block}.part{attempt}.jsonl"
    natural = (len(queries) - len(have)) * sz["cutoff_s"] + S.TIMED_SLACK_S
    if kind == "SPLIT":
        need = gate(a, "edge", site, hdr, M.LOAD_OF["S"])
        _, R0 = RT.k16()
        server_must_answer(RT.split_ready(R0, C.local_addr(site, "split_port"), site["model_dirs"]["32B"], spec_hash, p), "split")
    elif kind == "REMOTE":
        token = C.check_token(site)
        server_must_answer(RT.remote_ready(C.local_addr(site, "remote_port"), token, spec_hash, model), "REMOTE")
        need = None
    else:
        need = gate(a, "edge", site, hdr, M.LOAD_OF[cfg])
    eos = sorted(S.EOS_IDS) if not a.dev else M.eos_ids(site["model_dirs"]["32B"])
    tok = WL.load_tokenizer(site["model_dirs"][model])
    marker = lambda bud: mark_attempt(rd, f"timed_{a.block}", hdr, block=a.block, config=cfg, start_index=len(have),  # noqa: E731
                                      budget=bud, torn_lines_before=torn)      # from here on this is an attempt
    if kind == "REMOTE":
        arm(natural)                                                            # no GPU of this host is used: no ledger entry
        bud, gpu = {"wall_clock_limit_s": natural, "ledger_id": None}, {"device": "none (client only)"}
        marker(bud)
        client = RT.RemoteClient(C.local_addr(site, "remote_port"), token, spec_hash, model)
        info = {"runner": "REMOTE client", "server": {k: client.server_identity.get(k) for k in ("model", "context_tokens")}}
        gen = lambda ids, n: client.generate(ids, n, eos, sz["cutoff_s"])  # noqa: E731
        mem, closer = (lambda: None), client.close
    else:
        bud = budget(a, site, rd, "edge", f"timed-{a.block}", natural)
        torch, R = torch_setup()
        gpu = gpu_gate(a, site, need)                                           # a BLOCKED here is a stop, not an attempt
        marker(bud)
        if kind == "SPLIT":
            e, mid, info = split_edge(a, site, hdr, torch, R, p)
            gen = lambda ids, n: RT.split_generate(torch, e, ids, n, eos, p["chunk"], sz["cutoff_s"])  # noqa: E731
            closer = mid.close
        else:
            hf = R.hf_model(site["model_dirs"][model], site["device"])
            info = {"runner": "k16_runtime.hf_model", "model": model, "load_memory": R.load_memory(site["device"])}
            gen = lambda ids, n: RT.whole_generate(torch, hf, ids, n, eos, site["device"], p["chunk"], sz["cutoff_s"], R.E0RT._sync)  # noqa: E731
            closer = lambda: None  # noqa: E731
        mem = lambda: mem_stats(R, site["device"])  # noqa: E731
    cap = lambda n_prompt: max(1, min(sz["max_new"], sz["context"] - n_prompt))  # noqa: E731
    w_ids = WL.token_ids(tok, wl["sentinel"][0])
    gen(w_ids, min(S.WARMUP_NEW_TOKENS, cap(len(w_ids))))                       # warm-up, not recorded
    for i, q in enumerate(queries):
        if i < len(have):
            continue
        n = first + i                                                           # position in the utility order (in the sentinel order for S2)
        if a.dev and os.environ.get("QSEL_DEV_DIE_AT") == str(i):              # CPU tests only: an attempt that dies mid-block
            raise RuntimeError("dev: simulated death of the attempt")
        ids = WL.token_ids(tok, q)
        res = gen(ids, cap(len(ids)))
        if kind == "SPLIT" and not res["kv_positions_in_step"]:
            raise RuntimeError("edge and server KV positions went out of step")
        toks, t = res.pop("tokens"), RT.summarize_times(res)
        row = {"n": n, "key": q["key"], "dataset": q["dataset"], "block": a.block, "config": cfg, "attempt": attempt, "n_prompt": len(ids),
               "ids_sha256": q["ids_sha256"], "max_new": cap(len(ids)), "tokens": toks, "n_out": len(toks), "utc": C.utc(),
               "text": tok.decode(toks, skip_special_tokens=True), **t, **res}
        row["stamps_s"] = [round(x / 1e9, 6) for x in row.pop("stamps_ns")]
        C.append_jsonl(out, row)
        print(f"[{a.block}] {i + 1}/{len(queries)} {q['key']} out={len(toks)} stop={res['stop']} total={t['total_s']:.2f}s", flush=True)
    signal.alarm(0)
    memory = mem()
    closer()
    if STATE["ledger"] is not None:
        STATE["ledger"].close("done")
    parts, rows, torn = block_parts(rd, a.block)
    C.write_json(done, finish(hdr, site, block=a.block, config=cfg, runner=info, gpu_gate=gpu, budget=bud, n_rows=len(rows), first_index=first, attempts=attempt,
                              parts={p.name: C.sha256_file(p) for p in parts}, torn_lines=torn, max_new_tokens=sz["max_new"],
                              context_tokens=sz["context"], cutoff_s=sz["cutoff_s"], memory=memory))
    print(f"block {a.block} complete ({len(rows)} rows, {attempt} attempt(s))")


def cmd_ledger(a):
    site = C.load_site(a.site, a.role)
    rep = C.Ledger(site["run_dir"], a.role, allotment_h=dev_of(a, site).get("allotment_h")).report()
    print(json.dumps(rep, indent=1))
    print(f"{a.role}: used {rep['used_h']:.2f} h of {rep['allotment_h']:.2f} h, {rep['remaining_h']:.2f} h left. Planning estimates per stage "
          f"(advisory, from archived token counts; never a gate): {S.EXPECTED_H[a.role]}")


# --------------------------------------------------------------------------- #
# Analysis host                                                                #
# --------------------------------------------------------------------------- #
def evaluators():
    sys.path.insert(0, str(C.REPO))
    from evaluation.GPQA.gpqa import GPQAEvaluator
    from evaluation.LiveMathBench.livemathbench import LiveMathBenchEvaluator
    from evaluation.MMLUPro.mmlupro import MMLUProEvaluator
    return {"mmlupro": MMLUProEvaluator(), "gpqa": GPQAEvaluator(), "livemathbench": LiveMathBenchEvaluator()}


def _block(rd, block, dev):
    done = C.read_json(rd / f"timed_{block}.done.json")
    if done["dev"] != dev:
        raise C.Stop(f"block {block} is a {'dev' if done['dev'] else 'target'} output")
    parts, rows, torn = block_parts(rd, block)
    if {p.name: C.sha256_file(p) for p in parts} != done["parts"]:
        raise C.Stop(f"block {block}: rows changed after the block completed")
    if len(rows) != done["n_rows"] or [r["n"] for r in rows] != list(range(done["first_index"], done["first_index"] + len(rows))):
        raise C.Stop(f"block {block}: rows missing or out of order")
    return done, rows


def cmd_score(a):
    C.freeze_check(a.dev)
    rd = Path(a.edge_run)
    complete = all((rd / f"timed_{b}.done.json").exists() for b in S.PRIMARY_BLOCKS)
    if not complete and not a.partial:
        raise C.Stop("the primary blocks are not complete: no score is computed while the run can still continue (PREREG section 5). "
                     "For a run that has ended incomplete, --partial scores what exists and records that it did")
    ev = None if a.scorer else evaluators()
    pub = C.read_json(a.workload_public) if a.dev else None
    wl = WL.load(a.workload_private, dev_workload=pub)
    answers = {q["key"]: q["answer"] for q in wl["utility"]}
    for block in S.BLOCKS:
        if not (rd / f"timed_{block}.done.json").exists():
            continue
        done, rows = _block(rd, block, bool(a.dev))
        out = rd / f"scores_{block}.json"
        if out.exists():                                                        # a repeated `score` continues with the blocks not yet scored
            if C.read_json(out)["parts"] != done["parts"]:
                raise C.Stop(f"scores_{block}.json belongs to other rows")
            print(f"scores_{block}.json: already written")
            continue
        scores, errors = {}, []
        for r in rows:
            try:
                if a.scorer:                                                    # dev only: exact text match, for pipeline tests
                    ok = r["text"].strip() == str(answers[r["key"]]).strip()
                else:
                    ok = ev[r["dataset"]].evaluate({"answer": answers[r["key"]]}, r["text"]).get("is_correct", False)
            except Exception as e:                                              # an output the evaluator cannot read is a wrong answer
                ok = False
                errors.append({"key": r["key"], "error_type": type(e).__name__})
            scores[r["key"]] = 1 if ok is True or ok == 1 else 0
        C.write_json(out, {"schema": S.SCHEMA, "block": block, "parts": done["parts"], "scored_utc": C.utc(),
                           "primary_blocks_complete_when_scored": complete, "evaluator_errors": errors,
                           "scorer": a.scorer or "unchanged benchmark evaluators", "scores": scores})
        print(f"scores_{block}.json: {sum(scores.values())}/{len(scores)} correct" + (f", {len(errors)} outputs the evaluator could not read" if errors else ""))


def _latency(rows, dones):
    tp = [r["per_token_s"] for r in rows if r["per_token_s"] is not None]
    peaks = [(d.get("memory") or {}).get("peak_allocated_mib") for d in dones]
    return {"queries": len(rows), "blocks": [d["block"] for d in dones], "ttft_s": ST.median([r["ttft_s"] for r in rows]), "per_token_s": ST.median(tp),
            "total_s": ST.median([r["total_s"] for r in rows]), "p90_total_s": ST.quantile([r["total_s"] for r in rows], 0.9),
            "cutoffs": sum(bool(r["cutoff"]) for r in rows), "cap_stops": sum(r["stop"] == "cap" for r in rows),
            "eos_stops": sum(r["stop"] == "eos" for r in rows), "output_tokens": sum(r["n_out"] for r in rows),
            "block_time_h": sum(r["total_s"] for r in rows) / 3600, "attempts": sum(d["attempts"] for d in dones),
            "peak_allocated_mib": max([p for p in peaks if p is not None], default=None),
            "required_free_bytes": (dones[0].get("memory_gate") or {}).get("need_bytes")}


def _decision_rows(per, score, cfgs, keys):
    t = lambda r: float("inf") if r["cutoff"] else r["total_s"]  # noqa: E731
    return [{"key": k, "bench": per["S"][k]["dataset"], "correct": {c: score[c][k] for c in cfgs}, "time": {c: t(per[c][k]) for c in cfgs}} for k in keys]


def _ledger_hours(run_dir, role):
    if not run_dir or not (Path(run_dir) / C.LEDGER).exists():
        return None
    return C.Ledger(run_dir, role).report()["used_h"]


def summarise(edge_run, server_run, dev=False, n_boot=None):
    rd = Path(edge_run)
    fit = [C.read_json(f).get("remote_32B") for f in (sorted(Path(server_run).glob("preflight*.json")) if server_run else [])]
    base = {"schema": S.SCHEMA, "experiment": S.NAME, "dev": dev, "tables": {}, "primary_blocks_complete": False,
            "remote_32B_memory_fit": next((x for x in reversed(fit) if x), None),
            "attempts": {"reference": attempts(rd, "reference"), "correctness": attempts(rd, "correctness"),
                         "serve_failed": failures(server_run, "serve") if server_run else 0},
            "gpu_hours": {"edge": _ledger_hours(rd, "edge"), "server": _ledger_hours(server_run, "server"), "ceiling": S.GPU_HOURS_CEILING}}
    if not (rd / "reference.json").exists():                                   # the split's model never ran whole on the edge
        return dict(base, outcome=ST.outcome(None, [], False, False, reference_ran=base["attempts"]["reference"] < S.MAX_STAGE_ATTEMPTS))
    ref = C.read_json(rd / "reference.json")
    if ref["dev"] != dev:
        raise C.Stop("dev outputs cannot be summarised as a result, nor target outputs as dev")
    base.update(freeze=ref["freeze"], screen_decision=ref["screen_decision"])
    if not (rd / "correctness.json").exists():                                 # the split never completed its first stage
        ran = base["attempts"]["correctness"] + base["attempts"]["serve_failed"] < S.MAX_STAGE_ATTEMPTS
        return dict(base, outcome=ST.outcome(None, [], False, False, split_ran=ran))
    corr = C.read_json(rd / "correctness.json")
    res = corr["result"]
    passed = bool(res["pass"] and res.get("kv_positions_in_step_all"))
    if passed != (corr["verdict"] == "PASS") or res.get("criteria_id") != S.CRITERIA_ID:
        raise C.Stop("stored correctness verdict does not match its record")
    out = dict(base, correctness={"verdict": corr["verdict"], "criteria_id": res.get("criteria_id"), "checks": res.get("checks"),
                                  "teacher_forced": res.get("teacher_forced"),
                                  "greedy": {k: v for k, v in (res.get("greedy") or {}).items() if k != "per_prompt"}})
    stages = [ref, corr]
    blocks = {}
    for b in S.BLOCKS:
        if (rd / f"timed_{b}.done.json").exists() and (rd / f"scores_{b}.json").exists():
            done, rows = _block(rd, b, dev)
            sc = C.read_json(rd / f"scores_{b}.json")
            if sc["parts"] != done["parts"]:
                raise C.Stop(f"block {b}: scores belong to other rows")
            blocks[b] = (done, rows, sc["scores"])
            stages.append(done)
            out["scored_before_primary_blocks_complete"] = out.get("scored_before_primary_blocks_complete", False) or not sc["primary_blocks_complete_when_scored"]
    if len({json.dumps(x["freeze"], sort_keys=True) for x in stages}) != 1:
        raise C.Stop("outputs come from more than one freeze")
    out["environment_consistent"] = len({json.dumps(x["versions"], sort_keys=True) for x in stages}) == 1
    server_sites = {C.read_json(f)["site_sha256"] for pat in ("server_record.*.json", "remote_record.*.json")
                    for f in (sorted(Path(server_run).glob(pat)) if server_run else [])}
    out["site_consistent"] = len({x["site_sha256"] for x in stages}) == 1 and len(server_sites) <= 1
    consistent = out["environment_consistent"] and out["site_consistent"]
    complete = all(b in blocks for b in S.PRIMARY_BLOCKS) and consistent
    have_remote = complete and "R14" in blocks
    tables, stab, quality, latency = {}, {"stable": False, "ratio": None, "n": 0}, {}, {}
    per, score, halves = {}, {}, {}
    for c, bs in S.CONFIG_BLOCKS.items():                                       # whatever was measured and scored is reported, complete or not
        got = [b for b in bs if b in blocks]
        if not got:
            continue
        rs = [r for b in got for r in blocks[b][1]]
        per[c], score[c] = {r["key"]: r for r in rs}, {k: v for b in got for k, v in blocks[b][2].items()}
        quality[c] = {d: 100.0 * sum(score[c][r["key"]] for r in rs if r["dataset"] == d) / max(1, sum(r["dataset"] == d for r in rs)) for d in S.BENCHMARKS}
        quality[c]["all"] = 100.0 * sum(score[c].values()) / max(1, len(rs))
        latency[c] = _latency(rs, [blocks[b][0] for b in got])
        if len(bs) == 2 and len(got) == 2:                                      # the same configuration half a day apart, on other queries
            m = [ST.median([r["per_token_s"] for r in blocks[b][1] if r["per_token_s"] is not None] or [0.0]) for b in bs]
            halves[c] = {"per_token_s": m, "second_over_first": (m[1] / m[0]) if m[0] else None}
    out["per_token_time_by_half"] = halves

    def same_queries(cfgs, keys):
        for c in cfgs:
            if [k for k in keys if k in per[c]] != keys or len(per[c]) != len(keys) or any(per[c][k]["ids_sha256"] != per["S"][k]["ids_sha256"] for k in keys):
                raise C.Stop(f"configuration {c} did not run the same queries with the same token ids as S")
    if not complete and consistent and all(b in blocks for b in S.FIRST_HALF_BLOCKS):
        # a run that ended after its first halves: the rule on that balanced half, as description only, never the outcome
        cfgs = [c for c in S.CONFIGS if c != "R14"]
        half = {c: {r["key"]: r for r in blocks[S.CONFIG_BLOCKS[c][0]][1]} for c in cfgs}
        keys = [r["key"] for r in blocks[S.FIRST_HALF_BLOCKS[0]][1]]
        if any(list(half[c]) != keys for c in cfgs):
            raise C.Stop("the first halves did not run the same queries")
        out["first_half_only"] = {"queries": len(keys), "descriptive_not_the_outcome": True,
                                  "tables": ST.tables(_decision_rows(half, score, cfgs, keys), False, n_boot=n_boot)}
    if complete:
        cfgs = [c for c in S.CONFIGS if c != "R14" or have_remote]
        keys = [r["key"] for b in S.CONFIG_BLOCKS["S"] for r in blocks[b][1]]
        same_queries(cfgs, keys)
        rows = _decision_rows(per, score, cfgs, keys)
        tables = ST.tables(rows, have_remote, n_boot=n_boot)
        s1 = [per["S"][k] for k in keys]
        stab = ST.stability(s1, blocks["S2"][1])
        # communication, from the split runtime's own counters: never assumed
        wire = {r["key"]: (r["prefill"]["rt_ns"] + r["decode_sum"]["rt_ns"] - r["prefill"]["server_compute_ns"] - r["decode_sum"]["server_compute_ns"]) / 1e9
                for r in s1}
        out["network"] = {"n": len(s1), "bytes_out_median": ST.median([r["prefill"]["bytes_out"] + r["decode_sum"]["bytes_out"] for r in s1]),
                          "round_trip_minus_server_compute_s_median": ST.median(list(wire.values())),
                          "share_of_completion_time_median": ST.median([wire[r["key"]] / r["total_s"] for r in s1]),
                          "prefill_round_trip_s_median": ST.median([r["prefill"]["rt_ns"] / 1e9 for r in s1]),
                          "decode_step_net_ms_median": ST.median([x / 1e6 for r in s1 for x in r["decode_steps"]["net_ns"]] or [0.0]),
                          "decode_step_server_compute_ms_median": ST.median([x / 1e6 for r in s1 for x in r["decode_steps"]["server_compute_ns"]] or [0.0])}
        out["zero_network_counterfactual"] = ST.zero_network(rows, wire, n_boot=n_boot)
        # descriptive labels
        edge_layer_ms = ST.median([(a_ + b_) / 1e6 / (2 * S.CUT) for r in s1 for a_, b_ in zip(r["decode_steps"]["edge_head_ns"], r["decode_steps"]["edge_tail_ns"])] or [0.0])
        e32_layer_ms = ST.median([x / 1e6 / S.MODELS["32B"]["n_layers"] for r in per["E32"].values() for x in r["decode_forward_ns"]] or [0.0])
        ratio = (e32_layer_ms / edge_layer_ms) if edge_layer_ms else None
        out["runtime_parity"] = {"split_edge_ms_per_layer": edge_layer_ms, "E32_ms_per_layer": e32_layer_ms, "ratio": ratio,
                                 "within_band": ratio is not None and ratio <= S.RUNTIME_PARITY_BAND}
        same = lambda x, y: sum(per[x][k]["tokens"] == per[y][k]["tokens"] for k in keys) / len(keys)  # noqa: E731
        out["token_agreement"] = {"S_vs_E32_identical_share": same("S", "E32"), "R14_vs_E14_identical_share": same("R14", "E14") if have_remote else None}
        out["same_model_split_advantage"] = any(c["selected"] == "S" for c in tables["same-model"])
        out["remote_evaluated"] = have_remote
    primary = tables.get(S.PRIMARY_POLICY, [])
    out.update(tables=tables, stability=stab, quality_pct=quality, latency=latency, primary_blocks_complete=complete,
               blocks_present=sorted(blocks), outcome=ST.outcome(passed, primary, stab["stable"], complete))
    out["selection_claimed"] = out["outcome"] in ("SCREEN_CONFIRMED", "SCREEN_CONTRADICTED")
    return out


def cmd_summary(a):
    C.freeze_check(a.dev)
    s = summarise(a.edge_run, a.server_run, dev=a.dev)
    out = Path(a.out)
    C.write_json(out / "summary.json", s)
    L = [f"# {S.NAME} summary ({'DEV, not a result' if s['dev'] else (s.get('freeze') or {}).get('tag')})", "",
         f"Outcome: **{s['outcome']}**. Screen decision: {(s.get('screen_decision') or {}).get('decision')} at Stage 1.",
         f"Correctness: {(s.get('correctness') or {}).get('verdict')}. Stability ratio: {(s.get('stability') or {}).get('ratio')}. "
         f"Same-model split advantage: {s.get('same_model_split_advantage')}. REMOTE evaluated: {s.get('remote_evaluated')}.",
         f"GPU-hours: edge {s['gpu_hours']['edge']}, server {s['gpu_hours']['server']} (ceiling {s['gpu_hours']['ceiling']})."]
    for name, cells in s["tables"].items():
        cfgs = list(cells[0]["utility"]) if cells else []
        sel = "Selected" if s.get("selection_claimed") else "The rule would give (not claimed)"
        L += ["", f"## {name}", "", "| Deadline (s) | " + " | ".join(f"U {c}" for c in cfgs) + f" | Best competitor | Delta (pt) | 95% CI | {sel} |",
              "|---" * (len(cfgs) + 5) + "|"]
        for c in cells:
            ci = "n/a" if c["ci95"][0] is None else f"[{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}]"
            L.append(f"| {c['deadline_s']} | " + " | ".join(f"{c['utility'][k]:.3f}" for k in cfgs)
                     + f" | {c['best_competitor']} | {c['delta_pt']:+.2f} | {ci} | {c['selected']} |")
    (out / "SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"outcome {s['outcome']} -> {out / 'summary.json'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    cmds = (("select-gpu", cmd_select_gpu), ("preflight", cmd_preflight), ("serve", cmd_serve), ("serve-remote", cmd_serve_remote),
            ("reference", cmd_reference), ("correctness", cmd_correctness), ("timed", cmd_timed), ("ledger", cmd_ledger), ("score", cmd_score),
            ("summary", cmd_summary))
    for name, fn in cmds:
        p = sp.add_parser(name)
        p.add_argument("--dev", action="store_true")
        if name in ("preflight", "ledger"):
            p.add_argument("--role", required=True, choices=S.ROLES)
        if name not in ("score", "summary"):
            p.add_argument("--site", required=True)
        if name == "select-gpu":
            p.add_argument("--out", required=True)
        if name == "timed":
            p.add_argument("--block", required=True, choices=S.BLOCKS)
        if name in ("score", "summary"):
            p.add_argument("--edge-run", required=True)
        if name == "score":
            p.add_argument("--workload-private", required=True)
            p.add_argument("--workload-public", help="dev only")
            p.add_argument("--scorer", choices=["exact"], help="dev only")
            p.add_argument("--partial", action="store_true", help="score a run that has ended before its primary blocks were complete")
        if name == "summary":
            p.add_argument("--server-run")
            p.add_argument("--out", required=True)
        p.set_defaults(fn=fn)
    a = ap.parse_args(argv)
    if (getattr(a, "scorer", None) or getattr(a, "workload_public", None)) and not a.dev:
        raise C.Stop("--scorer and --workload-public are dev only")
    try:
        if a.cmd in S.LIMITS_S and a.cmd in ("select-gpu", "preflight"):
            arm(S.LIMITS_S[a.cmd])
        a.fn(a)
    except (SystemExit, KeyboardInterrupt):
        if STATE["ledger"] is not None:
            STATE["ledger"].close("stopped")
        raise
    except LimitReached as e:                                                   # an attempt that died at its limit: recorded, resumable
        if STATE["ledger"] is not None:
            STATE["ledger"].close("died: wall-clock limit")
        if a.cmd not in ("select-gpu", "preflight", "ledger", "score", "summary"):
            record_failure(a.cmd, e)
        print(f"LIMIT: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        if STATE["ledger"] is not None:
            STATE["ledger"].close(f"died: {type(e).__name__}")
        if a.cmd not in ("select-gpu", "preflight", "ledger", "score", "summary"):
            record_failure(a.cmd, e)
        raise


if __name__ == "__main__":
    main()
