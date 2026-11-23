"""ROUTER-INTEG launcher: every command that runs on a laboratory host, and the live pipeline.

  select-gpu     server, CPU only   choose and pin the server GPU by QWEN-SELECT's fixed rule
  preflight      both, CPU only     freeze, role, checkpoints, tokenisation against the pinned ids, router, memory, port
  serve-remote   server, GPU        hold the models on the pinned GPU and serve the REMOTE wire on loopback
  shared         edge               one lane of the shared measurement set (or of the profiling run):
                                      --lane edge    both models whole on the edge GPU
                                      --lane remote  the same requests through the wire (no GPU on the edge)
  router-timing  edge, CPU only     the frozen router on every evaluation query, from raw text, timed, against the pinned decisions
  live           edge               the end-to-end pipeline under one data policy:
                                      query -> router -> model decision -> placement -> feasibility -> backend -> output
  ledger         both               the GPU-hour account of this host

Every target command first checks the freeze record, the reused runtime's tag, the site file and the host role.
--dev runs the same code on a CPU with tiny random models and fixture workloads; dev outputs are refused as results.
Nothing here scores an answer or evaluates a policy (rint_score.py, rint_eval.py, on the analysis machine).
"""
import argparse
import json
import os
import signal
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rint_backend as B  # noqa: E402
import rint_common as C  # noqa: E402
import rint_models as M  # noqa: E402
import rint_policy as P  # noqa: E402
import rint_record as REC  # noqa: E402
import rint_spec as S  # noqa: E402
import rint_workload as WL  # noqa: E402

STATE = {"ledger": None, "clock": None, "clock_deadline": None, "position_scaling": None}
ROUTER_THREADS = S.ROUTER_THREADS
DEV_KEYS = {"max_new": "MAX_NEW_TOKENS", "context": "CONTEXT_TOKENS", "cutoff_s": "QUERY_CUTOFF_S", "chunk": "PREFILL_CHUNK",
            "first_wait_s": "SERVER_FIRST_WAIT_S", "idle_exit_s": "SERVER_IDLE_EXIT_S", "warmup_new": "WARMUP_NEW_TOKENS"}


class LimitReached(BaseException):
    """A wall-clock limit. Not an Exception: the request loops record and survive ordinary errors, a limit must end
    the command."""


def arm(seconds):
    def stop(*_):
        raise LimitReached(f"wall-clock limit of {int(seconds)} s reached")
    signal.signal(signal.SIGALRM, stop)
    signal.alarm(max(1, int(seconds)))


def dev_overrides(site):
    """CPU tests only: tiny models have a tiny vocabulary and short limits. Applied identically by server and edge."""
    d = site.get("dev") or {}
    for m, e in (d.get("eos_ids") or {}).items():
        S.MODELS[m]["eos_ids"] = tuple(int(x) for x in e)
    for k, name in DEV_KEYS.items():
        if k in d:
            setattr(S, name, d[k])


def prepare(a, role, stage):
    site = C.load_site(a.site, role)
    if a.dev:
        dev_overrides(site)
    freeze = C.freeze_check(site, a.dev)
    facts = C.check_role(role, site, a.dev)
    rd = Path(site["run_dir"])
    rd.mkdir(parents=True, exist_ok=True)
    hdr = REC.header(freeze, site, role, facts, f"{stage}.{int(time.time())}.{os.getpid()}", stage, a.dev)
    return site, rd, hdr


def workload(a, site):
    return WL.load(site["workload_private"], dev_workload=(site.get("dev") or {}).get("workload_public") if a.dev else None)


def model_list(a):
    ms = [m.strip() for m in (a.models or ",".join(S.MODEL_ORDER)).split(",") if m.strip()]
    if not ms or any(m not in S.MODELS for m in ms):
        raise C.Stop(f"--models takes a subset of {S.MODEL_ORDER}")
    return [m for m in S.MODEL_ORDER if m in ms]


def checkpoints(a, site, models):
    """{model: exact identifier}; every pinned file verified by SHA-256 (cached per file size and mtime)."""
    out = {}
    for m in models:
        if a.dev:
            out[m] = dict(M.model_id(m, "dev"), dev=True)
            continue
        rec = M.verify(site["model_dirs"][m], m, cache_file=site["hash_cache"])
        if not rec["ok"]:
            raise C.Stop(f"checkpoint of {m} is not the pinned one: {rec['problems'][:3]}")
        out[m] = M.model_id(m, rec["identity_sha256"])
    return out


def gate(a, role, site, models):
    """Free memory before anything is loaded (QWEN-SELECT's gate). Short of memory: BLOCKED, nothing is loaded."""
    if a.dev:
        return {"dev": True}
    need = M.need(role, models)
    rec = C.reused(site)["qc"].memory_gate(role, site, need["required_free"])
    return dict(rec, need=need)


def load(a, role, site, models, need_rec):
    torch, R = C.runtime(site)
    if not a.dev:
        import transformers
        if transformers.__version__ != S.TRANSFORMERS_REQUIRED:
            raise C.Stop(f"transformers {S.TRANSFORMERS_REQUIRED} is required by the reused loader")
        if str(site["device"]).startswith("cuda"):
            free = torch.cuda.mem_get_info(torch.device(site["device"]))[0]
            C.reused(site)["qc"].device_gate(role, need_rec["need"]["required_free"], free)
    loaded = {m: R.hf_model(site["model_dirs"][m], site["device"]) for m in models}
    STATE["position_scaling"] = {m: B.align_position_scaling(loaded[m]) for m in models}
    if not a.dev and any(abs(v["applied_scale"] - S.POSITION_SCALE_EXPECTED) > 1e-6 for v in STATE["position_scaling"].values()):
        raise C.Stop(f"position scaling is not the original engine's: {STATE['position_scaling']}")
    return torch, R, loaded


def stage_used_s(led, name):
    return sum(led.charged(e) for e in led.entries() if "open" in e and e["stage"] == name)


def open_ledger(a, site, rd, role, stage, hdr, natural=None):
    """Open the GPU-hour entry of a stage and arm its wall-clock limit. The limit of a stage covers all its attempts:
    what earlier attempts of the same stage were charged is subtracted. `natural`: an explicit limit in seconds (the
    live B fallback, which may only use what is left of live B's time). Raises Stop when nothing is left."""
    qc = C.reused(site)["qc"]
    dv = (site.get("dev") or {}) if a.dev else {}
    allot = dv.get("allotment_h", S.ALLOTMENT_H[role])
    led = qc.Ledger(rd, role, allotment_h=allot)
    name = stage.replace(":", "-")
    if natural is None:
        base = dv.get("limit_s") or 3600.0 * S.STAGE_LIMIT_H[role].get(stage.split(":")[-1], 0.0)
        if base <= 0:
            raise C.Stop(f"stage {stage} has no GPU time on the {role} under this protocol")
        natural = base - stage_used_s(led, name)
    if natural <= 60:
        raise C.Stop(f"LIMIT: the wall-clock limit of stage {stage} is used up over its attempts. Nothing was loaded")
    eid, limit = led.open(name, natural)
    STATE["ledger"] = led
    if STATE.get("clock_deadline"):
        limit = min(limit, max(1.0, STATE["clock_deadline"] - time.time()))
    arm(limit)
    hdr.setdefault("budget", {}).update({"ledger_id": eid, "wall_clock_limit_s": limit, "allotment_h": allot})
    return led


def open_clock(a, site, rd, stage, hdr):
    """Wall-clock account of a stage that the edge drives without an edge GPU (a remote lane, live B). It is the reused
    ledger with the stage's limit as its allotment, in a directory of its own, so that the limit covers all attempts of
    the stage and a killed attempt is charged to its last heartbeat."""
    qc = C.reused(site)["qc"]
    dv = (site.get("dev") or {}) if a.dev else {}
    limit_s = dv.get("limit_s") or (3600.0 * S.STAGE_LIMIT_H["server"][stage] - S.STAGE_SLACK_S)
    led = qc.Ledger(rd / f"clock_{stage}", "edge", allotment_h=limit_s / 3600.0)
    if led.remaining_s() <= 660:
        raise C.Stop(f"LIMIT: the wall-clock limit of stage {stage} is used up over its attempts")
    eid, limit = led.open(stage, limit_s)
    STATE["clock"], STATE["clock_deadline"] = led, time.time() + limit
    arm(limit)
    hdr.setdefault("budget", {}).update({"clock_id": eid, "stage_wall_clock_limit_s": limit})
    return led


def close_gpu_ledger(note=""):
    if STATE["ledger"] is not None:
        STATE["ledger"].close(note)
        STATE["ledger"] = None


def close_ledger(note=""):
    signal.alarm(0)
    close_gpu_ledger(note)
    if STATE.get("clock") is not None:
        STATE["clock"].close(note)
        STATE["clock"], STATE["clock_deadline"] = None, None


def numbered(rd, stem):
    n = 1
    while (rd / f"{stem}.{n}.json").exists():
        n += 1
    return rd / f"{stem}.{n}.json"


def others(site, a):
    if a.dev or not site.get("gpu_uuid"):
        return {"other_processes": None, "other_mib": None}
    return C.reused(site)["qc"].gpu_others(site["gpu_uuid"], own_pids=(os.getpid(),))


def tokenizers(site, models):
    return {m: WL.load_tokenizer(site["model_dirs"][m]) for m in models}


def decode(tok, tokens):
    return tok.decode([int(t) for t in tokens], skip_special_tokens=True)


# --------------------------------------------------------------------------- #
# select-gpu, preflight                                                        #
# --------------------------------------------------------------------------- #
def cmd_select_gpu(a):
    site = C.read_json(a.site)
    if site.get("role") != "server" or site.get("gpu_uuid"):
        raise C.Stop("select-gpu takes the server template, whose gpu_uuid is empty")
    C.freeze_check(site, a.dev)
    if not a.dev and socket.gethostname() != site.get("expected_hostname"):
        raise C.Stop("this host is not the one the site file names")
    qc = C.reused(site)["qc"]
    need = M.need("server", model_list(a))
    rows, apps = qc.gpu_rows(), qc.apps_text()
    chosen, considered = qc.select_gpu(rows, apps, need["required_free"], site.get("excluded_gpu_uuids") or [])
    rd = Path(site["run_dir"])
    rd.mkdir(parents=True, exist_ok=True)
    if list(rd.glob("remote_started.*.json")):
        raise C.Stop("a server stage has already started on this host: the selection is fixed")
    rec = {"schema": S.SCHEMA, "utc": C.utc(), "need": need, "chosen": chosen and {k: chosen[k] for k in ("uuid", "index", "free_mib", "total_mib", "other_processes")},
           "considered": [{k: c.get(k) for k in ("index", "free_mib", "total_mib", "other_processes", "other_mib", "eligible", "reason")} for c in considered]}
    C.write_json(numbered(rd, "gpu_selection"), rec)
    if chosen is None:
        raise C.Stop("BLOCKED: no eligible GPU has the memory for both models now (nothing was loaded; try later, never ask anyone to free memory)")
    pinned = dict(site, gpu_uuid="GPU-" + chosen["uuid"])
    C.write_json(a.out, pinned)
    print(json.dumps({"selected_index": chosen["index"], "free_mib": chosen["free_mib"], "pinned_site": str(a.out),
                      "export": f"CUDA_VISIBLE_DEVICES=GPU-{chosen['uuid']}"}, indent=1))


def cmd_preflight(a):
    role = a.role
    site, rd, hdr = prepare(a, role, "preflight")
    models = model_list(a)
    ids = checkpoints(a, site, models)
    rec = {"header": hdr, "checkpoints": ids, "problems": []}
    qc = C.reused(site)["qc"]
    if not a.dev:
        need = M.need(role, models)
        mem = qc.meminfo()
        rec["memory"] = {"need": need, "ram_available_bytes": mem.get("MemAvailable"), "swap_used_bytes": mem.get("swap_used")}
        if role == "edge":
            rec["memory"]["ok_now"] = (mem.get("MemAvailable") or 0) >= need["required_free"]
        else:
            row = next((r for r in qc.gpu_rows() if r["uuid"] == qc.norm_uuid(site["gpu_uuid"])), None)
            rec["memory"].update(gpu_free_mib=row and row["free_mib"], gpu_total_mib=row and row["total_mib"])
            rec["memory"]["ok_now"] = bool(row and row["free_mib"] is not None and row["free_mib"] * (1 << 20) >= need["required_free"])
            rec["gpu_pinned_in_environment"] = os.environ.get("CUDA_VISIBLE_DEVICES") == site["gpu_uuid"]
        rec["others_on_gpu"] = others(site, a)
        rec["ledger"] = {k: v for k, v in qc.Ledger(rd, role, allotment_h=S.ALLOTMENT_H[role]).report().items() if k != "entries"}
    if role == "server":
        C.token_of(site, create=True)
        rec["port_free"] = qc.port_free(site["remote_port"])
        if not rec["port_free"]:
            rec["problems"].append("the wire port is in use")
    else:
        wl, priv = workload(a, site)
        toks = tokenizers(site, models)
        bad = [q["key"] for q in wl["queries"] for m in models
               if WL.ids_sha(WL.tokenize(toks[m], priv[q["key"]]["prompt"])) != q["ids_sha256"][m]]
        rec["tokenisation"] = {"queries": len(wl["queries"]), "models": models, "mismatches": bad[:10], "n_mismatches": len(bad)}
        if bad:
            rec["problems"].append("token ids recomputed on this host differ from the pinned ids")
        import rint_router as RR
        dv = site.get("dev") or {}
        r = RR.Router(site["router_encoder_dir"], threads=ROUTER_THREADS, check=not a.dev,
                      heads=RR.Heads(dv.get("router_heads"), check=False) if a.dev and dv.get("router_heads") else None,
                      fold_map=RR.FoldMap(dv.get("fold_map")) if a.dev and dv.get("fold_map") else None)
        diffs, disagree = [], 0
        for q in wl["queries"]:
            if q["role"] != "profile":
                continue
            d = r.route(q["key"], priv[q["key"]]["prompt"])
            diffs.append(abs(d["score"] - q["router_pinned"]["score"]))
            disagree += d["choice"] != q["router_pinned"]["choice"]
        rec["router"] = {"identity": r.identity, "profile_queries": len(diffs), "max_abs_score_diff_vs_pinned": max(diffs) if diffs else None,
                         "choices_differing": disagree}
        if disagree:
            rec["problems"].append("router choices on this host differ from the pinned choices")
        rec["token_file_present"] = Path(site["token_file"]).is_file()
    rec["ok"] = not rec["problems"] and (a.dev or rec["memory"]["ok_now"])
    C.write_json(numbered(rd, "preflight"), rec)
    print(json.dumps({k: rec.get(k) for k in ("ok", "problems", "memory", "tokenisation", "router", "port_free")}, indent=1, default=str))
    sys.exit(0 if rec["ok"] else 3)


# --------------------------------------------------------------------------- #
# serve-remote (server)                                                        #
# --------------------------------------------------------------------------- #
def cmd_serve_remote(a):
    site, rd, hdr = prepare(a, "server", f"serve:{a.stage}")
    models = model_list(a)
    if not a.dev and os.environ.get("CUDA_VISIBLE_DEVICES") != site["gpu_uuid"]:
        raise C.Stop("the server process must see exactly the pinned GPU (CUDA_VISIBLE_DEVICES=<the site's gpu_uuid>)")
    ids = checkpoints(a, site, models)
    token = C.token_of(site, create=a.dev)
    qc, qrt = C.reused(site)["qc"], C.reused(site)["qrt"]
    if not qc.port_free(site["remote_port"]):
        raise C.Stop("the wire port is in use")
    need_rec = gate(a, "server", site, models)
    open_ledger(a, site, rd, "server", f"serve:{a.stage}", hdr)
    started = numbered(rd, "remote_started")
    note = "died"
    try:
        hdr.update(models=models, memory_gate=need_rec, others_on_gpu_start=others(site, a))
        C.write_json(started, hdr)
        torch, R, loaded = load(a, "server", site, models, need_rec)
        log_path = rd / f"remote_log.{started.name.split('.')[1]}.jsonl"
        identity = {"protocol": S.PROTOCOL, "spec_hash": hdr["freeze"]["spec_hash"], "models": ids, "context_tokens": S.CONTEXT_TOKENS,
                    "versions": hdr["versions"], "dev": bool(a.dev), "position_scaling": STATE["position_scaling"]}
        hdr["position_scaling"] = STATE["position_scaling"]
        srv = B.RemoteServer(torch, loaded, site["device"], identity, token,
                             log=lambda ev: C.append_jsonl(log_path, dict(ev, utc=C.utc())))
        sock = socket.socket()
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", int(site["remote_port"])))           # loopback only: the edge comes through its SSH forward
        sock.listen(1)
        print(json.dumps({"ready": True, "models": models, "port": int(site["remote_port"])}), flush=True)
        why = qrt.serve_remote(srv, sock, S.SERVER_FIRST_WAIT_S, S.SERVER_IDLE_EXIT_S)
        sock.close()
        note = why
        C.write_json(numbered(rd, "remote_record"), dict(hdr, ended=why, n_requests=srv.n_requests, utc_end=C.utc(),
                                                         memory=R.mem_stats(site["device"]), others_on_gpu_end=others(site, a)))
    except KeyboardInterrupt:
        note = "interrupted"
    except LimitReached as e:
        note = str(e)
        C.append_jsonl(rd / "events.jsonl", REC.event(hdr, "wall_clock_limit", detail=str(e)))
    finally:
        close_ledger(note)
    print(json.dumps({"ended": note}), flush=True)


# --------------------------------------------------------------------------- #
# shared measurement lanes and the profiling run (edge)                        #
# --------------------------------------------------------------------------- #
def lane_attempt(rd, stem):
    n = len(list(rd.glob(f"{stem}.attempt*.json"))) + 1
    if n > S.MAX_LANE_ATTEMPTS:
        raise C.Stop(f"{stem} has used its {S.MAX_LANE_ATTEMPTS} attempts: the stage is INCOMPLETE")
    return n


def connect_server(a, site, hdr, models):
    """The edge side of the wire, connected and checked. A server that does not answer is a stop, not an attempt."""
    backend = B.RemoteBackend(f"127.0.0.1:{int(site['remote_port'])}", C.token_of(site), hdr["freeze"]["spec_hash"])
    try:
        ident = backend.connect()
    except B.BackendError as e:
        raise C.Stop(f"the server does not answer through the forward (not an attempt): {e}")
    if ident.get("spec_hash") != hdr["freeze"]["spec_hash"] or bool(ident.get("dev")) != bool(a.dev):
        raise C.Stop("the server is on another revision of the protocol")
    if sorted(ident["models"]) != sorted(models):
        raise C.Stop(f"the server serves {sorted(ident['models'])}, this command asks for {sorted(models)}: "
                     "start both with the same --models (the one-model fallback of the protocol, section 13) or with none")
    return backend, ident


def counts(row):
    """A row that settles its request. A transport failure does not: the request is repeated on resumption."""
    return not (row["status"] == "error" and row.get("transport_error"))


def run_request(rec, hdr, backend, phase, lane, q, priv, model, attempt, toks, cutoff_s=None, max_new=None, extra=None):
    """One request, one row, whatever happens. Returns the row."""
    config = P.config_of(model, backend.mode)
    max_new = WL.max_new(q["n_prompt"][model]) if max_new is None else max_new
    seed = C.request_seed(q["key"], model, backend.mode, phase)
    row = REC.base_row(hdr, phase, lane, q, model, config, attempt, seed, max_new, cutoff_s=cutoff_s)
    try:
        res = backend.generate(model, priv["ids"][model], max_new, seed, S.QUERY_CUTOFF_S if cutoff_s is None else cutoff_s)
        REC.ok_row(row, res, decode(toks[model], res["tokens"]), extra)
    except LimitReached:
        raise
    except Exception as e:                          # recorded, never dropped, never turned into an answer
        REC.error_row(row, e, extra, transport=isinstance(e, B.TransportError))
    rec.write(row)
    return row


def cmd_shared(a):
    phase, lane = a.phase, a.lane
    stage = f"{phase}:{lane}"
    site, rd, hdr = prepare(a, "edge", stage)
    models = model_list(a)
    wl, priv = workload(a, site)
    by_key = {q["key"]: q for q in wl["queries"]}
    if phase == "profile":
        plan = [[k, S.MODEL_ORDER[i % 2]] for i, k in enumerate(wl["profile_keys"])]
    else:
        plan = wl["orders"]["shared"]
    if a.limit:
        plan = plan[:a.limit]
    stem = f"lane_{phase}_{lane}"
    # the server's one-model-at-a-time fallback (protocol, section 13) runs the lane once per model: its attempts and
    # its part records are counted per model subset; the lane is complete only when every model has its rows
    part = "" if models == list(S.MODEL_ORDER) else "_" + "".join(models)
    if (rd / f"{stem}.done.json").exists() or (part and (rd / f"{stem}{part}.part.json").exists()):
        raise C.Stop(f"{stem}{part} is complete: a completed stage is final")
    rec = REC.Recorder(rd, f"{phase}_{lane}", hdr)
    rows, torn = rec.existing()
    done = {(r["key"], r["config"]) for r in rows if r["phase"] == phase and counts(r)}
    toks = tokenizers(site, models)
    backend = ids = need_rec = None
    if lane == "edge":
        ids = checkpoints(a, site, models)
        need_rec = gate(a, "edge", site, models)
    else:
        backend, ident = connect_server(a, site, hdr, models)
        hdr["server_identity"] = ident
    attempt = lane_attempt(rd, stem + part)         # an attempt begins here: every check has passed
    if lane == "edge":
        open_ledger(a, site, rd, "edge", phase, hdr)
    else:
        open_clock(a, site, rd, phase, hdr)
    note, n_err, consecutive, n_new = "died", 0, 0, 0
    try:
        hdr.update(lane=lane, phase=phase, attempt=attempt, models=models, memory_gate=need_rec, torn_lines_before=torn,
                   rows_before=len(rows), others_on_gpu_start=others(site, a) if lane == "edge" else None)
        C.write_json(rd / f"{stem}{part}.attempt{attempt}.json", hdr)
        if lane == "edge":
            torch, R, loaded = load(a, "edge", site, models, need_rec)
            backend = B.EdgeBackend(torch, loaded, site["device"], ids)
            hdr["load_memory"] = R.mem_stats(site["device"])
            hdr["position_scaling"] = STATE["position_scaling"]

        def after(row):
            """Stop rules after a row. A transport failure stops the lane at once: the request is repeated on resumption."""
            nonlocal n_err, consecutive
            if row["status"] != "error":
                consecutive = 0
                return
            n_err, consecutive = n_err + 1, consecutive + 1
            if row.get("transport_error"):
                rec.event("lane_stopped", why="the server or the wire was lost", lane=lane, phase=phase, row_id=row["id"])
                raise C.Stop("the server or the wire was lost: the lane stops; rows are kept and the request is repeated on resumption")
            if consecutive >= S.MAX_CONSECUTIVE_ERRORS:
                rec.event("lane_stopped", why=f"{consecutive} requests in a row ended in an error", lane=lane, phase=phase)
                raise C.Stop(f"{consecutive} requests in a row ended in an error: the lane stops; rows are kept")
        wq = by_key[wl["warmup_key"]]
        for m in models:                            # warm-up: recorded, never analysed
            after(run_request(rec, hdr, backend, "warmup", lane, wq, priv[wq["key"]], m, attempt, toks, max_new=S.WARMUP_NEW_TOKENS,
                              extra={"warmup_for": phase}))
        for key, first in plan:
            q = by_key[key]
            for m in [first] + [x for x in S.MODEL_ORDER if x != first]:
                if m not in models or (key, P.config_of(m, backend.mode)) in done:
                    continue
                row = run_request(rec, hdr, backend, phase, lane, q, priv[key], m, attempt, toks, extra={"order_first_model": first})
                n_new += 1
                print(json.dumps({"key": key, "config": row["config"], "status": row["status"], "stop": row["stop"], "n_out": row["n_out"],
                                  "total_s": row["total_s"], "per_token_s": row["per_token_s"]}), flush=True)
                after(row)
        all_rows, _ = rec.existing()
        mine = [r for r in all_rows if r["phase"] == phase and counts(r)]
        have = {(r["key"], r["config"]) for r in mine}
        want = {(k, P.config_of(m, backend.mode)) for k, _ in plan for m in models}
        want_all = {(k, P.config_of(m, backend.mode)) for k, _ in plan for m in S.MODEL_ORDER}
        closing = dict(hdr, utc_end=C.utc(), n_rows=len(mine), n_errors=sum(r["status"] == "error" for r in mine),
                       n_transport_failures_repeated=sum(not counts(r) for r in all_rows if r["phase"] == phase),
                       models=models, others_on_gpu_end=others(site, a) if lane == "edge" else None,
                       memory=None if lane != "edge" else hdr.get("load_memory"))
        if want <= have:
            if part:
                C.write_json(rd / f"{stem}{part}.part.json", closing)
            if want_all <= have:
                C.write_json(rd / f"{stem}.done.json", dict(closing, models=list(S.MODEL_ORDER),
                                                             one_model_at_a_time=bool(part) or bool(list(rd.glob(f"{stem}_*.part.json")))))
            note = "complete"
        else:
            note = "incomplete"
    except LimitReached as e:
        note = str(e)
        rec.event("wall_clock_limit", detail=str(e), lane=lane, phase=phase)
    finally:
        if backend is not None:
            backend.close()
        close_ledger(note)
    print(json.dumps({"lane": lane, "phase": phase, "ended": note, "new_rows": n_new, "errors": n_err}), flush=True)
    sys.exit(0 if note == "complete" else 4)


# --------------------------------------------------------------------------- #
# router timing (edge, CPU)                                                    #
# --------------------------------------------------------------------------- #
def make_router(a, site):
    import rint_router as RR
    dv = (site.get("dev") or {}) if a.dev else {}
    return RR.Router(site["router_encoder_dir"], threads=ROUTER_THREADS, check=not a.dev,
                     heads=RR.Heads(dv["router_heads"], check=False) if dv.get("router_heads") else None,
                     fold_map=RR.FoldMap(dv["fold_map"]) if dv.get("fold_map") else None)


def cmd_router_timing(a):
    site, rd, hdr = prepare(a, "edge", "router_timing")
    out = rd / "router_timing.jsonl"
    if out.exists():
        raise C.Stop("router_timing.jsonl exists: a completed stage is final")
    wl, priv = workload(a, site)
    by_key = {q["key"]: q for q in wl["queries"]}
    r = make_router(a, site)
    for k in wl["profile_keys"]:                    # warm-up, untimed
        r.route(k, priv[k]["prompt"])
    rows = []
    for key, _ in wl["orders"]["live"]:
        q = by_key[key]
        d = r.route(key, priv[key]["prompt"])
        d.update(pinned_score=q["router_pinned"]["score"], pinned_choice=q["router_pinned"]["choice"],
                 abs_score_diff=abs(d["score"] - q["router_pinned"]["score"]), agrees_with_pinned=d["choice"] == q["router_pinned"]["choice"],
                 run_id=hdr["run_id"], dev=hdr["dev"])
        C.append_jsonl(out, d)
        rows.append(d)
    lat = sorted(x["router_s"] for x in rows)
    summ = dict(hdr, router=r.identity, n=len(rows), mean_s=sum(lat) / len(lat), median_s=lat[len(lat) // 2], p95_s=lat[int(0.95 * (len(lat) - 1))],
                max_abs_score_diff=max(x["abs_score_diff"] for x in rows), choices_agreeing=sum(x["agrees_with_pinned"] for x in rows),
                warmup_queries=len(wl["profile_keys"]), utc_end=C.utc())
    C.write_json(rd / "router_timing.done.json", summ)
    print(json.dumps({k: summ[k] for k in ("n", "mean_s", "median_s", "p95_s", "max_abs_score_diff", "choices_agreeing")}, indent=1))


# --------------------------------------------------------------------------- #
# live pipeline (edge)                                                         #
# --------------------------------------------------------------------------- #
def cmd_live(a):
    policy = {"A": S.POLICY_A, "B": S.POLICY_B}[a.policy]
    phase = f"live_{a.policy}"
    site, rd, hdr = prepare(a, "edge", phase)
    models = list(S.MODEL_ORDER)
    dv = (site.get("dev") or {}) if a.dev else {}
    wl, priv = workload(a, site)
    by_key = {q["key"]: q for q in wl["queries"]}
    plan = wl["orders"]["live"]
    if a.reduced:                                   # the pre-declared reduction (protocol, section 12)
        plan = plan[:S.LIVE_REDUCED_N]
    if a.limit:
        plan = plan[:a.limit]
    stem = f"lane_{phase}"
    if (rd / f"{stem}.done.json").exists():
        raise C.Stop(f"{stem} is complete: a completed stage is final")
    rec = REC.Recorder(rd, phase, hdr)
    rows, torn = rec.existing()
    done = {r["key"] for r in rows if r["phase"] == phase}
    toks = tokenizers(site, models)
    router = make_router(a, site)
    mode = S.PLACEMENT[policy]
    ids_edge = checkpoints(a, site, models)
    edge_state = {"backend": None, "load_s": None, "failed": None}
    remote = None
    if mode == S.REMOTE:                            # live B starts only with the server up and holding both models
        remote, ident = connect_server(a, site, hdr, models)
        hdr["server_identity"] = ident
    else:
        need0 = gate(a, "edge", site, models)       # BLOCKED here is a stop before the attempt
    attempt = lane_attempt(rd, stem)

    def edge_backend():
        """The edge models, loaded when first needed: at once under policy A, at the first fallback under policy B.
        Under B a blocked or failed load makes the fallback unavailable for the rest of the pass; it never ends the
        pass by itself, and it is never charged twice."""
        if edge_state["backend"] is not None:
            return edge_state["backend"]
        if edge_state["failed"]:
            raise B.BackendError(f"the edge models are not available in this pass: {edge_state['failed']}")
        t = time.perf_counter()
        try:
            if mode == S.EDGE:
                need_rec = need0
                open_ledger(a, site, rd, "edge", phase, hdr)
            else:
                need_rec = gate(a, "edge", site, models)
                open_ledger(a, site, rd, "edge", "live_B_fallback", hdr, natural=max(0.0, STATE["clock_deadline"] - time.time()))
            torch, R, loaded = load(a, "edge", site, models, need_rec)
        except LimitReached:
            raise
        except BaseException as e:                  # BLOCKED, budget, or a failed load
            edge_state["failed"] = f"{type(e).__name__}: {e}"[:200]
            if mode == S.EDGE:
                raise
            close_gpu_ledger("fallback load failed")
            if isinstance(e, (KeyboardInterrupt, GeneratorExit)):
                raise
            raise B.BackendError(f"the edge fallback is not available: {edge_state['failed']}") from None
        edge_state.update(backend=B.EdgeBackend(torch, loaded, site["device"], ids_edge), load_s=time.perf_counter() - t)
        hdr["position_scaling"] = STATE["position_scaling"]
        return edge_state["backend"]

    note, n_err, consecutive, n_fallback, consecutive_fb, n_live = "died", 0, 0, 0, 0, 0
    try:
        if mode == S.REMOTE:
            open_clock(a, site, rd, phase, hdr)
        hdr.update(phase=phase, policy=policy, placement_mode=mode, attempt=attempt, torn_lines_before=torn, rows_before=len(rows),
                   router=router.identity, reduced=bool(a.reduced), n_planned=len(plan))
        C.write_json(rd / f"{stem}.attempt{attempt}.json", hdr)
        warm = edge_backend() if mode == S.EDGE else remote
        wq = by_key[wl["warmup_key"]]
        router.route(wq["key"], priv[wq["key"]]["prompt"])                  # warm-up of the router, untimed
        for m in models:
            w = run_request(rec, hdr, warm, "warmup", phase, wq, priv[wq["key"]], m, attempt, toks, max_new=S.WARMUP_NEW_TOKENS,
                            extra={"warmup_for": phase})
            if w["status"] == "error":
                raise C.Stop(f"the warm-up request of {m} failed: {w['error']}")
        for key, _ in plan:
            if key in done:
                continue
            if dv.get("break_wire_after") is not None and n_live == int(dv["break_wire_after"]) and remote is not None:
                remote._drop()                                              # dev tests only: the wire is lost from here on
                remote.addr = "127.0.0.1:1"
            n_live += 1
            q, text = by_key[key], priv[key]["prompt"]
            t0 = time.perf_counter()
            dec = router.route(key, text)                                   # query -> router inference -> model decision
            model = dec["choice"]
            t1 = time.perf_counter()
            up = remote.available() if mode == S.REMOTE else True
            pl = P.place(model, policy, server_available=up)                # placement and feasibility
            ids = WL.tokenize(toks[model], text)                            # the request is built from the raw query text
            t2 = time.perf_counter()
            max_new = WL.max_new(len(ids))
            extra = {"policy": policy, "router": dec, "router_pinned": q["router_pinned"],
                     "router_agrees_with_pinned": dec["choice"] == q["router_pinned"]["choice"], "placement": pl,
                     "planned_config": P.config_of(model, mode), "ids_match_pinned": WL.ids_sha(ids) == q["ids_sha256"][model],
                     "router_s": t1 - t0, "dispatch_overhead_s": t2 - t1, "fallback": bool(pl["fallback"]),
                     "first_attempt_error": None if not pl["fallback"] else {"type": "TransportError", "message": "the server did not answer before dispatch"},
                     "lost_s": 0.0, "backend_ready_s": 0.0, "end_to_end_s": None}
            if pl["fallback"]:
                rec.event("fallback", key=key, model=model, from_mode=mode, to_mode=pl["mode"], error="the server did not answer before dispatch", lost_s=0.0)
            tries = [pl["mode"]] if pl["mode"] else []
            if pl["mode"] == S.REMOTE and S.PLACEMENT_FALLBACK[policy] and P.feasible(S.PLACEMENT_FALLBACK[policy], policy)[0]:
                tries.append(S.PLACEMENT_FALLBACK[policy])
            res, err = None, RuntimeError(pl["infeasible"] or "no feasible execution mode")
            row = REC.base_row(hdr, phase, phase, q, model, P.config_of(model, mode), attempt, 0, max_new)
            for i, md in enumerate(tries):
                assert P.feasible(md, policy)[0]                            # a mode the data policy prohibits is never dispatched
                seed = C.request_seed(key, model, md, phase)
                row = REC.base_row(hdr, phase, phase, q, model, P.config_of(model, md), attempt, seed, max_new)
                ta = time.perf_counter()
                try:
                    backend = remote if md == S.REMOTE else edge_backend()  # under B this may load the edge models: counted
                    tb = time.perf_counter()
                    res = backend.generate(model, ids, max_new, seed, S.QUERY_CUTOFF_S)
                    extra["backend_ready_s"] = tb - ta
                    break
                except LimitReached:
                    raise
                except Exception as e:
                    err = e
                    if i + 1 < len(tries):                                  # the server failed: the edge takes over
                        extra["lost_s"] += time.perf_counter() - ta
                        extra.update(fallback=True, first_attempt_error={"type": type(e).__name__, "message": str(e)[:300]})
                        rec.event("fallback", key=key, model=model, from_mode=md, to_mode=tries[i + 1],
                                  error=f"{type(e).__name__}: {e}"[:300], lost_s=extra["lost_s"])
            if res is None:
                REC.error_row(row, err, extra, transport=isinstance(err, B.TransportError))
            else:
                # raw query in -> last token on the edge: router, dispatch, time lost in a failed attempt, backend readiness, generation
                extra["end_to_end_s"] = (t2 - t0) + extra["lost_s"] + extra["backend_ready_s"] + res["stamps_ns"][-1] / 1e9
                extra["end_to_end_ended_within_s"] = {str(d): bool(extra["end_to_end_s"] <= d) for d in S.DEADLINES_S}
                REC.ok_row(row, res, decode(toks[model], res["tokens"]), extra)
            rec.write(row)
            print(json.dumps({"key": key, "choice": model, "config": row["config"], "status": row["status"], "stop": row["stop"],
                              "end_to_end_s": row.get("end_to_end_s"), "fallback": row.get("fallback")}), flush=True)
            n_fallback += bool(row.get("fallback"))
            consecutive_fb = consecutive_fb + 1 if row.get("fallback") else 0
            consecutive = consecutive + 1 if row["status"] == "error" else 0
            n_err += row["status"] == "error"
            if consecutive >= S.MAX_CONSECUTIVE_ERRORS:
                rec.event("lane_stopped", why=f"{consecutive} requests in a row ended in an error", phase=phase)
                raise C.Stop(f"{consecutive} requests in a row ended in an error: the pipeline stops; rows are kept")
            if consecutive_fb >= S.MAX_CONSECUTIVE_FALLBACKS:
                rec.event("lane_stopped", why=f"{consecutive_fb} requests in a row needed the fallback: the server is down", phase=phase)
                raise C.Stop(f"{consecutive_fb} requests in a row needed the fallback: the pipeline stops; resume it when the server is back")
        all_rows, _ = rec.existing()
        mine = [r for r in all_rows if r["phase"] == phase]
        if {k for k, _ in plan} <= {r["key"] for r in mine}:
            C.write_json(rd / f"{stem}.done.json", dict(hdr, utc_end=C.utc(), n_rows=len(mine),
                                                         n_errors=sum(r["status"] == "error" for r in mine),
                                                         n_fallbacks=sum(bool(r.get("fallback")) for r in mine),
                                                         edge_models_load_s=edge_state["load_s"]))
            note = "complete"
        else:
            note = "incomplete"
    except LimitReached as e:
        note = str(e)
        rec.event("wall_clock_limit", detail=str(e), phase=phase)
    finally:
        if remote is not None:
            remote.close()
        close_ledger(note)
    print(json.dumps({"phase": phase, "ended": note, "errors": n_err, "fallbacks": n_fallback}), flush=True)
    sys.exit(0 if note == "complete" else 4)


def cmd_ledger(a):
    site = C.load_site(a.site, a.role)
    led = C.reused(site)["qc"].Ledger(site["run_dir"], a.role, allotment_h=S.ALLOTMENT_H[a.role])
    print(json.dumps(led.report(), indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, **kw):
        p = sp.add_parser(name)
        p.add_argument("--site", required=True)
        p.add_argument("--dev", action="store_true", help="CPU test with tiny random models; outputs are refused as results")
        p.set_defaults(fn=fn)
        return p
    p = add("select-gpu", cmd_select_gpu)
    p.add_argument("--out", required=True)
    p.add_argument("--models", help="only for the one-model-at-a-time fallback of the protocol, section 13")
    p = add("preflight", cmd_preflight)
    p.add_argument("--role", required=True, choices=["edge", "server"])
    p.add_argument("--models")
    p = add("serve-remote", cmd_serve_remote)
    p.add_argument("--stage", required=True, choices=["profile", "shared", "live_B"])
    p.add_argument("--models")
    p = add("shared", cmd_shared)
    p.add_argument("--lane", required=True, choices=["edge", "remote"])
    p.add_argument("--phase", default="shared", choices=["profile", "shared"])
    p.add_argument("--models")
    p.add_argument("--limit", type=int, help="dev only")
    add("router-timing", cmd_router_timing)
    p = add("live", cmd_live)
    p.add_argument("--policy", required=True, choices=["A", "B"])
    p.add_argument("--reduced", action="store_true", help="the pre-declared reduction: the first 60 queries of the live order")
    p.add_argument("--limit", type=int, help="dev only")
    p = add("ledger", cmd_ledger)
    p.add_argument("--role", required=True, choices=["edge", "server"])
    a = ap.parse_args(argv)
    if getattr(a, "limit", None) and not a.dev:
        raise C.Stop("--limit is for --dev only: a target run uses the frozen sample")
    a.fn(a)


if __name__ == "__main__":
    main()
