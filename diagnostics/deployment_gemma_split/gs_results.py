"""GEMMA-SPLIT output layout, provenance and result validation (PREREG §14).

Layout (deterministic: every path is a function of the freeze record's run_id, the stage and the role only):
  runs/<run_id>/stage2_memory/{edge,server}/{split_memory,reference_memory}.json      gate.json
  runs/<run_id>/stage3_correctness/reference_server/{reference.jsonl,meta.json}
                                   reference_edge/{compare.jsonl,meta.json}
                                   [pipeline/{compare.jsonl,meta.json}]               (in_host_split only)
                                   split/{compare.jsonl,boundary_audit.json,meta.json,server_serve.json}  decision.json  gate.json
  runs/<run_id>/stage4_components/{edge,server}/{components.jsonl,meta.json} | edge/{trace.jsonl,meta.json,server_serve.json}
                                   summary.json gate.json
  runs/<run_id>/stage5_e2e/<block>/requests.jsonl, <block>/meta.json  (block in gs_spec.block_order)  summary.json gate.json
                           split_{A,B}/server_meta.json       (the server half's record for the split blocks)
  runs/<run_id>/stage1_checks/{edge,server}_workload.json     (target runs: token ids re-derived on each host)
  runs/<run_id>/final/outcome.json, checkpoint_verify_{edge,server}.json ; runs/<run_id>/<host>.exit_status.log
  runs/<run_id>/<stage>/_invalidated/NNN-.../ and LOG.jsonl    (repeated parts, never deleted)
run_id = the first 12 hex digits of freeze_sha256. Each host writes only under its role's directories; files from
the two hosts merge without conflict on the analysis host.
"""
import re
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # python -I: the script dir is not on sys.path
import gs_common as C  # noqa: E402
import gs_spec as S  # noqa: E402

PROVENANCE_KEYS = ("experiment", "protocol_version", "decision_rule_version", "stage", "role", "host", "devices", "freeze_sha256",
                   "run_id", "adapter_sha256", "configuration_id", "checkpoint_manifest_sha256", "profile_role", "utc")


def run_dir(gs_dir, freeze):
    return Path(gs_dir) / "runs" / freeze["run_id"]


def stage_dir(rd, stage):
    return Path(rd) / S.STAGE_DIRS[stage]


def parts(stage, info_caps):
    """{part: {"role": host role that runs the part, "own": files it writes, "companion": files the OTHER host writes}}."""
    caps = set(info_caps)
    if stage == "stage2":
        return {r: {"role": r, "own": [f"{r}/split_memory.json", f"{r}/reference_memory.json"], "companion": []} for r in S.ROLES}
    if stage == "stage3":
        out = {"reference_server": {"role": "server", "own": ["reference_server/reference.jsonl", "reference_server/meta.json"], "companion": []},
               "reference_edge": {"role": "edge", "own": ["reference_edge/compare.jsonl", "reference_edge/meta.json"], "companion": []},
               "split": {"role": "edge", "own": ["split/compare.jsonl", "split/boundary_audit.json", "split/meta.json"],
                         "companion": ["split/server_serve.json"]}}
        if "in_host_split" in caps:
            out["pipeline"] = {"role": "server", "own": ["pipeline/compare.jsonl", "pipeline/meta.json"], "companion": []}
        return out
    if stage == "stage4":
        if "segment_step" in caps:
            return {r: {"role": r, "own": [f"{r}/components.jsonl", f"{r}/meta.json"], "companion": []} for r in S.ROLES}
        if "step_trace" in caps:
            return {"edge": {"role": "edge", "own": ["edge/trace.jsonl", "edge/meta.json"], "companion": ["edge/server_serve.json"]}}
        return {}
    if stage == "stage5":
        return {b: {"role": S.BLOCK_HOST[b], "own": [f"{b}/requests.jsonl", f"{b}/meta.json"],
                    "companion": [f"{b}/server_meta.json"] if b.startswith("split") else []} for b in S.block_order(caps)}
    raise ValueError(stage)


def expected(stage, info_caps):
    if stage == "stage1":
        return [f"{r}_workload.json" for r in S.ROLES]
    return [f for p in parts(stage, info_caps).values() for f in p["own"] + p["companion"]]


def provenance(freeze, stage, role, devices, profile_role, host=None, extra=None):
    if profile_role not in S.PROFILE_ROLES:
        raise ValueError(f"profile_role must be one of {S.PROFILE_ROLES}")
    p = {"experiment": S.NAME, "protocol_version": S.PROTOCOL_VERSION, "decision_rule_version": S.DECISION_RULE_VERSION,
         "stage": stage, "role": role, "host": host or socket.gethostname(), "devices": devices, "freeze_sha256": freeze["freeze_sha256"],
         "run_id": freeze["run_id"], "adapter_sha256": freeze["adapter"]["sha256"], "configuration_id": freeze["split"]["configuration_id"],
         "checkpoint_manifest_sha256": freeze["checkpoint"][role]["manifest_sha256"] if role in S.ROLES else None,
         "profile_role": profile_role, "utc": C.utc()}
    if extra:
        p.update(extra)
    return p


def check_provenance(p, freeze, stage, role=None, require_target=True, what=""):
    pre = f"{what}: " if what else ""
    if not isinstance(p, dict):
        return [pre + "no provenance"]
    probs = [pre + f"provenance lacks {k}" for k in PROVENANCE_KEYS if k not in p]
    if probs:
        return probs
    for k, want in (("experiment", S.NAME), ("protocol_version", S.PROTOCOL_VERSION), ("decision_rule_version", S.DECISION_RULE_VERSION),
                    ("freeze_sha256", freeze["freeze_sha256"]), ("run_id", freeze["run_id"]), ("adapter_sha256", freeze["adapter"]["sha256"]),
                    ("configuration_id", freeze["split"]["configuration_id"]), ("stage", stage)):
        if p[k] != want:
            probs.append(pre + f"{k} {p[k]!r} != {want!r}")
    if role is not None and p["role"] != role:
        probs.append(pre + f"role {p['role']} != {role}")
    if p["role"] in S.ROLES:
        frozen_host = freeze["roles"][p["role"]]["host"]
        if p["host"] != frozen_host:
            probs.append(pre + f"host {p['host']} is not the frozen {p['role']} ({frozen_host})")
        if p["checkpoint_manifest_sha256"] != freeze["checkpoint"][p["role"]]["manifest_sha256"]:
            probs.append(pre + "checkpoint manifest differs from the freeze record")
    if require_target and p["profile_role"] != "target":
        probs.append(pre + f"profile_role {p['profile_role']} is not target")
    return probs


# --------------------------------------------------------------------------- #
# No timing in correctness or memory outputs (PREREG §9.6)                    #
# --------------------------------------------------------------------------- #
PROHIBITED = re.compile(r"(_ns$|_ms$|_us$|_s$|(^|_)time(_|$)|latenc|throughput|tok(en)?s?_per_s|per_sec|duration|elapsed|"
                        r"cuda_?event|speed|bandwidth|tpot|ttft|wall|_rate$|second|perf_counter|mbps|gbps|clock)", re.I)
ALLOWED_KEYS = {"utc", "frozen_at_utc", "captured_utc", "started_utc", "ended_utc", "stage", "runtime", "etimes_s"}   # timestamps, not durations


class TimingFieldError(ValueError):
    pass


def assert_no_timing_fields(obj, path="$"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            ks = str(k)
            if ks not in ALLOWED_KEYS and PROHIBITED.search(ks):
                raise TimingFieldError(f"{path}.{ks}")
            assert_no_timing_fields(v, f"{path}.{ks}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            assert_no_timing_fields(v, f"{path}[{i}]")


RUN_CTX = {}          # set by gs_bench.logged for the duration of a command: {rd, token, stage, part, cmd}


def file_result(obj):
    """The result a written record carries: a Stage-2 status, or a Stage-3 split run's rule preview."""
    if isinstance(obj, dict):
        if isinstance(obj.get("status"), str):
            return obj["status"]
        if isinstance((obj.get("rules_preview") or {}).get("verdict"), str):
            return obj["rules_preview"]["verdict"]
    return None


def _logged_write(path, text, obj):
    """An output written by a logged run (PREREG §13.2): WROTE (file, digest, result) BEFORE the write, SAVED after it.
    Fails closed: without its WROTE event nothing is written; if SAVED cannot be logged the file is removed again. So a
    file present was written by the run that logged WROTE; a WROTE without SAVED is a write that never completed."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists: GEMMA-SPLIT outputs are written once (write a new file instead)")
    if not RUN_CTX:
        return C.write_text(path, text)
    sd = stage_dir(RUN_CTX["rd"], RUN_CTX["stage"]) if RUN_CTX.get("stage") in S.STAGE_DIRS else Path(RUN_CTX["rd"])
    try:
        rel = path.resolve().relative_to(Path(sd).resolve()).as_posix()
    except ValueError:
        rel = path.name
    ctx, digest = dict(RUN_CTX), C.sha256_bytes(text.encode("utf-8"))[:16]
    log_event(ctx["rd"], "WROTE", ctx["token"], ctx["stage"], ctx["part"], ctx["cmd"], out=digest, res=file_result(obj), file=rel)
    C.write_text(path, text)
    try:
        log_event(ctx["rd"], "SAVED", ctx["token"], ctx["stage"], ctx["part"], ctx["cmd"], out=digest, file=rel)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def write(path, obj, timing_allowed):
    if not timing_allowed:
        assert_no_timing_fields(obj)
    obj = C.redact_obj(obj)
    return _logged_write(path, C.json_text(obj), obj)


def write_rows(path, rows, timing_allowed):
    if not timing_allowed:
        for r in rows:
            assert_no_timing_fields(r)
    return _logged_write(path, C.jsonl_text([C.redact_obj(r) for r in rows]), None)


def exit_log(rd, line, host=None):
    p = Path(rd) / f"{host or socket.gethostname()}.exit_status.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(f"{C.utc()} {C.redact_text(line)}\n")


EVENT_RE = re.compile(r"^(\S+) (START|RUN|WROTE|SAVED|END) run=(\S+) stage=(\S+) part=(\S+) rc=(\S+) cmd=(\S+)(?: out=(\S+))?(?: res=(\S+))?"
                      r"(?: file=(\S+))?")


def log_event(rd, kind, token, stage, part, cmd, rc="-", host=None, out=None, res=None, file=None):
    """START, RUN (measurement begins: after connect and the device check) and END of a command in
    runs/<run_id>/<host>.exit_status.log; gs_gate reads these events for attempt accounting (PREREG §13.2). A completed
    writer's END carries `out`, the digest of the files it wrote (which attempt's outputs are present), and `res`, the
    result it recorded where there is one (a Stage-2 probe's status, a Stage-3 split run's rule preview), so that a
    deleted attempt's result still counts (PREREG §13.2)."""
    exit_log(rd, f"{kind} run={token} stage={stage or '-'} part={part or '-'} rc={rc} cmd={cmd}" + (f" out={out}" if out else "") +
             (f" res={res}" if res else "") + (f" file={file}" if file else ""), host=host)


def read_events(rd):
    """[{utc, kind, token, stage, part, rc, cmd, host}] from every host's exit-status log of a run, in log order."""
    out = []
    for p in sorted(Path(rd).glob("*.exit_status.log")):
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            m = EVENT_RE.match(line)
            if m:
                out.append(dict(zip(("utc", "kind", "token", "stage", "part", "rc", "cmd", "out", "res", "file"), m.groups()),
                                host=p.name.split(".exit_status")[0]))
    return out


WRITER_CMDS = ("serve", "memory:split", "memory:reference", "reference", "correctness", "components", "e2e")


def writer(stage, part, f):
    """The command (as gs_bench logs it) that writes file f of a part: the unit of attempt accounting."""
    name = f.rsplit("/", 1)[-1]
    if name in ("server_serve.json", "server_meta.json") or (stage, part, name) == ("stage2", "server", "split_memory.json"):
        return "serve"
    if stage == "stage2":
        return "memory:" + name.split("_")[0]
    if stage == "stage3":
        return "reference" if part.startswith("reference_") else "correctness"
    return {"stage4": "components", "stage5": "e2e"}[stage]


def outputs_digest(rd, stage, part, cmd, caps):
    """Digest of the files a writer command wrote for a part (logged in its END line), or None."""
    spec = parts(stage, caps).get(part) if stage in S.STAGE_ORDER else None
    if not spec:
        return None
    sd = stage_dir(rd, stage)
    files = sorted(f for f in spec["own"] + spec["companion"] if writer(stage, part, f) == cmd and (sd / f).exists())
    return C.sha256_obj([[f, C.sha256_file(sd / f)] for f in files])[:16] if files else None


def outputs_result(rd, stage, part, cmd, caps):
    """The result a writer recorded, for its END line: a Stage-2 record's status, a Stage-3 split run's rule preview."""
    spec = parts(stage, caps).get(part) if stage in S.STAGE_ORDER else None
    sd = stage_dir(rd, stage)
    for f in (spec or {}).get("own", []) + (spec or {}).get("companion", []):
        if writer(stage, part, f) == cmd and f.endswith(".json") and (sd / f).exists() and file_result(C.read_json(sd / f)):
            return file_result(C.read_json(sd / f))
    return None


def wrote_events(events, stage, part, cmd, kind="WROTE"):
    """{run token: {file: (digest, result)}} from the WROTE (intent) or SAVED (written) events of a writer."""
    out = {}
    for e in events:
        if e["kind"] == kind and (e["stage"], e["part"], e["cmd"]) == (stage, part, cmd) and e.get("file"):
            out.setdefault(f"{e['host']}/{e['token']}", {})[e["file"]] = (e.get("out"), e.get("res"))
    return out


def file_digest(path):
    return C.sha256_file(path)[:16]


def _tokens(events, kind, stage, part, cmd, rc=None):
    return {f"{e['host']}/{e['token']}" for e in events
            if e["kind"] == kind and (e["stage"], e["part"], e["cmd"]) == (stage, part, cmd) and (rc is None or e["rc"] == rc)}


def completed_tokens(events, stage, part, cmd):
    """Runs of a command for (stage, part) that ended with rc=0: they finished and wrote their outputs."""
    return _tokens(events, "END", stage, part, cmd, "0")


def failed_tokens(events, stage, part, cmd):
    """Runs that began measuring (RUN) and did not end with rc=0: crashed, killed or failed mid-way. A refusal before
    the measurement began (preflight, gate, order, connection) is not an attempt."""
    return _tokens(events, "RUN", stage, part, cmd) - completed_tokens(events, stage, part, cmd)


def token_ms(tok):
    """Start time (ms, the host's clock) encoded in a token `<host>/<pid>-<ms>`."""
    try:
        return int(tok.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return 0


def crash_evidence(rd, stage, part, cmds, exclude=()):
    """(crashed, completed) among the runs of `cmds` for (stage, part) that no earlier repeat accounted for (`exclude`)."""
    ev, ex = read_events(rd), set(exclude)
    failed = set().union(*[failed_tokens(ev, stage, part, c) for c in cmds]) - ex
    done = set().union(*[completed_tokens(ev, stage, part, c) for c in cmds]) - ex
    return bool(failed), bool(done)


def stage_runs(events, stage):
    """{(part, cmd): (completed tokens, failed tokens)} for every writer command logged for a stage."""
    keys = {(e["part"], e["cmd"]) for e in events if e["stage"] == stage and e["cmd"] in WRITER_CMDS}
    return {k: (completed_tokens(events, stage, *k), failed_tokens(events, stage, *k)) for k in sorted(keys)}


def crashed(rd, stage, part):
    return crash_evidence(rd, stage, part, WRITER_CMDS)[0]
