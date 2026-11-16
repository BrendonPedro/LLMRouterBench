"""GEMMA-SPLIT gates: the STOP / CONTINUE rules, repeats and the final outcome (PREREG §13). A DECISION file.

  Stage 0  CONTINUE | STOP_INCOMPLETE | STOP_ROLE_MISMATCH | STOP_CHECKPOINT_MISMATCH | STOP_SPLIT_CAPTURE_INVALID | STOP_NO_CONFIGURATION
  Stage 2  CONTINUE | STOP_INFEASIBLE (final) | INFEASIBLE_UNCONFIRMED (one confirming repeat) | BLOCKED_CONTENTION | INVALID | INCOMPLETE (final)
  Stage 3  PASS | FAIL (final) | NOT_EVALUABLE (final) | INVALID | INCOMPLETE (final)
  Stage 4  COMPLETE | INVALID | INCOMPLETE (final)          Stage 5  COMPLETE (final) | UNSTABLE | INVALID | INCOMPLETE (final)
A gate is written once, carries the run mode (dev / target) and its own hash, and lists the parts that may be repeated
and why (`repeatable`). `invalidate` moves only such parts (and the stage's derived files) to _invalidated/, logs the
gate's causes and the run tokens it accounts for, and refuses a final verdict; nothing is ever repeated without a gate.
Caps (gs_spec.REPEAT_CAPS) count earlier invalidations and supersedes over the whole chain of freeze records: per part
and cause in Stages 2-4, per repeat event and cause in Stage 5, whose repeats always cover the whole block sequence.
When a cap is reached the gate turns final. Attempt accounting (exit-status logs): a writer that crashed while measuring
makes its part repeatable; two completed runs since the last repeat mean outputs were deleted (final INCOMPLETE). A file
written before its part's last invalidation is stale: the gate waits for the fresh file (GateNotReady), as it does while
outputs or logs are missing. `outcome` recomputes every stored gate from the raw files before it reports anything.
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # python -I: the script dir is not on sys.path
import gs_common as C  # noqa: E402
import gs_results as R  # noqa: E402
import gs_rules as RU  # noqa: E402
import gs_spec as S  # noqa: E402

PREV = {"stage3": "stage2", "stage4": "stage3", "stage5": "stage4"}
DERIVED = ("gate.json", "summary.json", "decision.json")


class GateClosed(RuntimeError):
    pass


class GateNotReady(RuntimeError):
    pass


def mode_of(freeze):
    return "dev" if freeze.get("dev") else "target"


def gate_path(rd, stage):
    return R.stage_dir(rd, stage) / "gate.json"


def read_gate(rd, stage):
    p = gate_path(rd, stage)
    return C.read_json(p) if p.exists() else None


def gate_hash(g):
    return C.sha256_obj({k: v for k, v in g.items() if k != "gate_sha256"})


def write_gate(rd, freeze, stage, body):
    g = dict(body, stage=stage, freeze_sha256=freeze["freeze_sha256"], run_id=freeze["run_id"], mode=mode_of(freeze), utc=C.utc())
    g["gate_sha256"] = gate_hash(g)
    C.write_json(gate_path(rd, stage), g)
    return g


def check_gate(g, freeze, what):
    P = []
    if g.get("gate_sha256") != gate_hash(g):
        P.append(f"{what}: gate_sha256 does not match its content (edited)")
    if g.get("freeze_sha256") != freeze["freeze_sha256"]:
        P.append(f"{what}: written under another freeze record")
    if g.get("mode") != mode_of(freeze):
        P.append(f"{what}: a {g.get('mode')} gate cannot open a {mode_of(freeze)} run")
    return P


def require(rd, freeze, stage):
    prev = PREV.get(stage)
    if prev is None:
        return None
    g = read_gate(rd, prev)
    if g is None:
        raise GateClosed(f"{stage} needs runs/{freeze['run_id']}/{S.STAGE_DIRS[prev]}/gate.json: compute it on the analysis host and copy it here")
    P = check_gate(g, freeze, f"{prev} gate")
    if P:
        raise GateClosed("; ".join(P))
    if g["verdict"] != S.OPEN_VERDICT[prev]:
        raise GateClosed(f"{prev} verdict {g['verdict']}: {stage} may not run (PREREG §13)")
    return g


def invalidation_log(rd, stage):
    p = R.stage_dir(rd, stage) / "_invalidated" / "LOG.jsonl"
    return C.read_jsonl(p) if p.exists() else []


def record_files(gs_dir):
    """Every record-shaped JSON in frozen/ (whatever its name): {freeze_sha256: record}."""
    out = {}
    for p in sorted((Path(gs_dir) / "frozen").glob("*.json")):
        try:
            rec = C.read_json(p)
        except Exception:
            continue
        if isinstance(rec, dict) and rec.get("experiment") == S.NAME and rec.get("freeze_sha256"):
            out[rec["freeze_sha256"]] = rec
    return out


def chain_run_dirs(rd, freeze):
    """Run directories of the records this one supersedes, newest first (followed through `supersedes`)."""
    gs_dir = Path(rd).resolve().parents[1]
    recs, out, cur, seen = record_files(gs_dir), [], freeze, set()
    while (cur.get("supersedes") or {}).get("freeze_sha256") and cur["supersedes"]["freeze_sha256"] not in seen:
        seen.add(cur["supersedes"]["freeze_sha256"])
        out.append(gs_dir / "runs" / cur["supersedes"]["run_id"])
        cur = recs.get(cur["supersedes"]["freeze_sha256"], {})
    return out


def supersede_events(rd, freeze, stage):
    """A new record counts as one repeat of the causes that justified it (its `supersedes.counts_as_repeat`, PREREG §6.4)."""
    gs_dir = Path(rd).resolve().parents[1]
    recs, out, cur, seen = record_files(gs_dir), [], freeze, set()
    while (cur.get("supersedes") or {}).get("freeze_sha256") and cur["supersedes"]["freeze_sha256"] not in seen:
        sup = cur["supersedes"]
        seen.add(sup["freeze_sha256"])
        rep = sup.get("counts_as_repeat") or {}
        if rep.get("stage") == stage:
            out += [{"part": e["part"], "cause": e["cause"], "causes": [e["cause"]], "moved_to": f"supersede:{sup['run_id']}",
                     "utc": cur.get("frozen_at_utc", "")} for e in rep.get("entries", [])]
        cur = recs.get(sup["freeze_sha256"], {})
    return out


def chain_log(rd, freeze, stage):
    return [e for d in [Path(rd)] + chain_run_dirs(rd, freeze) for e in invalidation_log(d, stage)] + supersede_events(rd, freeze, stage)


def _causes(e):
    return e.get("causes") or [e.get("cause")]


def apply_caps(rd, freeze, stage, problems):
    """Split problems into repeatable (cap not reached), exhausted (cap reached) and fixed (not repeatable).
    Stage 5 counts repeat events (one per invalidation of the sequence) per cause; Stages 2-4 count per part and cause."""
    log, caps = chain_log(rd, freeze, stage), S.REPEAT_CAPS[stage]
    repeatable, exhausted, fixed, seen = [], [], [], set()
    for p in problems:
        if p["cause"] not in caps:
            fixed.append(p)
            continue
        cap = caps[p["cause"]]
        if S.REPEAT_SCOPE[stage] == "sequence":
            prior = len({e["moved_to"] for e in log if p["cause"] in _causes(e)})
        else:
            prior = len({e["moved_to"] for e in log if e["part"] == p["part"] and p["cause"] in _causes(e)})
        if cap is None or prior < cap:
            if (p["part"], p["cause"]) not in seen:
                repeatable.append({"part": p["part"], "cause": p["cause"], "prior": prior, "cap": cap})
                seen.add((p["part"], p["cause"]))
        else:
            exhausted.append(dict(p, prior=prior, cap=cap))
    return repeatable, exhausted, fixed


def stale_problems(rd, freeze, stage):
    """A part's JSON outputs must be written after its last invalidation (an rsync must not bring the old attempt back)."""
    log = invalidation_log(rd, stage)
    sd = R.stage_dir(rd, stage)
    P = []
    for part, spec in R.parts(stage, freeze["adapter"]["info"]["capabilities"]).items():
        last = max([C.parse_utc(e["utc"]) for e in log if e["part"] == part] or [None], key=lambda x: x or 0)
        if last is None:
            continue
        for f in spec["own"] + spec["companion"]:
            if f.endswith(".json") and (sd / f).exists():
                utc = (C.read_json(sd / f).get("provenance") or {}).get("utc")
                if utc and C.parse_utc(utc) < last - S.CLOCK_SKEW_TOLERANCE_S:
                    P.append(RU.problem(part, "stale", f"{f} was written before {part} was invalidated (an old attempt): sync the fresh file"))
    return P


def consumed(rd, stage, part):
    """Run tokens that earlier repeats of the part accounted for (recorded by invalidate)."""
    return {t for e in invalidation_log(rd, stage) if e["part"] == part for t in e.get("consumed", [])}


def part_tokens(events, stage, part):
    return sorted({f"{e['host']}/{e['token']}" for e in events if (e["stage"], e["part"]) == (stage, part) and e["cmd"] in R.WRITER_CMDS})


def attempts(rd, freeze, stage):
    """Attempt accounting from the exit-status logs (PREREG §13.2), per part and writer command, over the runs that no
    earlier repeat accounted for. Returns (crashed, problems, waiting): crashed = a writer crashed or was killed while
    measuring and nothing completed since (the part is repeated); problems = two completed runs (outputs were deleted and
    the command re-run) or a crashed run before the completed one without a repeat in between; waiting = files or logs
    that are not here yet."""
    ev, sd, caps = R.read_events(rd), R.stage_dir(rd, stage), freeze["adapter"]["info"]["capabilities"]
    crashed, probs, waiting = [], [], []
    for part, spec in R.parts(stage, caps).items():
        used, files = consumed(rd, stage, part), spec["own"] + spec["companion"]
        for cmd in sorted({R.writer(stage, part, f) for f in files}):
            missing = [f for f in files if R.writer(stage, part, f) == cmd and not (sd / f).exists()]
            done = sorted(R.completed_tokens(ev, stage, part, cmd) - used, key=R.token_ms)
            failed = R.failed_tokens(ev, stage, part, cmd) - used
            if len(done) > 1:
                outs = {f"{e['host']}/{e['token']}": e.get("out") for e in ev if e["kind"] == "END" and e["rc"] == "0"
                        and (e["stage"], e["part"], e["cmd"]) == (stage, part, cmd)}
                cur = R.outputs_digest(rd, stage, part, cmd, caps)
                which = ", ".join(f"{t} out={outs.get(t)} ({'present' if outs.get(t) and outs.get(t) == cur else 'not present'})" for t in done)
                probs.append(RU.problem(part, "discarded_attempt", f"{part}: {len(done)} completed `{cmd}` runs since its last repeat [{which}]: "
                                                                    "the outputs of an attempt were deleted and the command re-run"))
            elif done and missing:
                waiting.append(f"{missing} (`{cmd}` completed: copy them; lost outputs end the run INCOMPLETE)")
            elif not done and failed:
                crashed.append(RU.problem(part, "invalid", f"{part}: `{cmd}` crashed or was killed while measuring; its outputs are incomplete"))
            elif not done:
                waiting.append(f"{missing or part} (`{cmd}` has not completed yet, or copy every host's exit-status log)")
            elif failed and min(map(R.token_ms, failed)) < R.token_ms(done[0]):
                probs.append(RU.problem(part, "invalid", f"{part}: a crashed `{cmd}` run preceded the completed one without a repeat in between"))
    return crashed, probs, waiting


def _discarded_body(discarded, would_be=None):
    wb = [f"the outputs present would give {would_be['verdict']}" + (f" ({would_be['category']})" if would_be.get("category") else "")
          ] if would_be else []
    return {"verdict": "INCOMPLETE", "final": True, "repeatable": [], "problems": discarded, "would_be": would_be,
            "reasons": [p["reason"] for p in discarded] + ["outputs of a completed attempt were deleted and the command re-run: the run's "
                                                            "integrity cannot be established; the verdict is final (PREREG §13.2)"] + wb}


def _finish(body, discarded):
    """A discarded attempt turns any verdict into a final INCOMPLETE; the verdict of the outputs present is reported with it."""
    if not discarded:
        return body
    return _discarded_body(discarded, {"verdict": body["verdict"], "category": body.get("category"), "reasons": body.get("reasons", [])})


def accounting(rd, freeze, stage):
    """Shared by every gate: (body, problems, discarded). body is a finished gate body (INVALID for a crash, or a final
    INCOMPLETE when outputs were discarded and the rest is incomplete); otherwise the problems to merge and the discarded
    attempts (_finish) are returned. Raises GateNotReady while files or logs are missing or stale."""
    stale = stale_problems(rd, freeze, stage)
    if stale:
        raise GateNotReady("stale files (older than their part's last invalidation): " + "; ".join(p["reason"] for p in stale))
    crashed, probs, waiting = attempts(rd, freeze, stage)
    discarded = [p for p in probs if p["cause"] == "discarded_attempt"]
    probs = [p for p in probs if p["cause"] != "discarded_attempt"]
    if discarded and (crashed or waiting):
        return _discarded_body(discarded), [], discarded
    if crashed:
        return _invalid_body(rd, freeze, stage, crashed + probs), [], []
    if waiting:
        raise GateNotReady(f"{stage} not complete: " + "; ".join(waiting))
    return None, probs, discarded


def _invalid_body(rd, freeze, stage, problems, exhausted_verdict="INCOMPLETE", extra=None):
    """Body for a stage with problems: INVALID with repeatable parts, or final when a cap is exhausted."""
    repeatable, exhausted, fixed = apply_caps(rd, freeze, stage, problems)
    if S.REPEAT_SCOPE[stage] == "sequence" and repeatable:      # Stage 5: the whole block sequence is repeated
        causes = sorted({r["cause"] for r in repeatable})
        repeatable = [{"part": b, "cause": causes[0], "causes": causes} for b in R.parts(stage, freeze["adapter"]["info"]["capabilities"])]
    body = dict(extra or {}, problems=problems, reasons=[p["reason"] for p in problems],
                supersede_eligible=any(p["cause"] in S.SUPERSEDE_CAUSES for p in problems))
    if exhausted:
        body.update(verdict=exhausted_verdict(exhausted) if callable(exhausted_verdict) else exhausted_verdict, final=True, repeatable=[],
                    exhausted=exhausted)
        body["reasons"].append(f"repeat cap reached for {[(e['part'], e['cause']) for e in exhausted]}: the verdict is final")
    else:
        body.update(verdict="INVALID", final=False, repeatable=repeatable, needs_new_record=[p["reason"] for p in fixed])
        if fixed:
            body["reasons"].append("not repeatable: correct the capture or fix the harness, then write a new freeze record (PREREG §6.4)")
    return body


def invalidate(rd, freeze, stage, parts, note):
    """Move the parts the stage's gate lists as repeatable aside and log the gate's causes and the run tokens accounted for.
    There is no other way to repeat anything (a crash is handled by the gate: attempts())."""
    sd = R.stage_dir(rd, stage)
    g = read_gate(rd, stage)
    if g is None:
        raise ValueError(f"{stage} has no gate: compute it first (gs_bench.py gate --stage N)")
    P = check_gate(g, freeze, f"{stage} gate")
    if P:
        raise ValueError("; ".join(P))
    if g.get("final"):
        raise ValueError(f"{stage} verdict {g['verdict']} is final: it cannot be invalidated (PREREG §13.2)")
    rep = {r["part"]: r for r in g.get("repeatable", [])}
    bad = [p for p in parts if p not in rep]
    if bad or not parts:
        raise ValueError(f"not repeatable according to the {stage} gate: {bad or parts}; repeatable: {sorted(rep)}")
    if S.REPEAT_SCOPE[stage] == "sequence" and set(parts) != set(rep):
        raise ValueError(f"{stage} repeats the whole block sequence: invalidate all of {sorted(rep)} (--all)")
    ev = R.read_events(rd)
    entries = [{"part": p, "cause": rep[p]["cause"], "causes": rep[p].get("causes", [rep[p]["cause"]]), "consumed": part_tokens(ev, stage, p),
                "reason": "; ".join(g.get("reasons", []))[:2000]} for p in parts]
    n = len(invalidation_log(rd, stage)) + 1
    dest = sd / "_invalidated" / f"{n:03d}-{C.utc().replace(':', '')}"
    dest.mkdir(parents=True, exist_ok=False)
    for e in entries:
        if (sd / e["part"]).exists():
            shutil.move(str(sd / e["part"]), str(dest / e["part"]))
    for d in DERIVED:
        if (sd / d).exists():
            shutil.move(str(sd / d), str(dest / d))
    with open(sd / "_invalidated" / "LOG.jsonl", "a", encoding="utf-8") as f:
        for e in entries:
            f.write(C.canonical(dict(e, utc=C.utc(), gate_verdict=g["verdict"], gate_sha256=g["gate_sha256"], note=note,
                                     moved_to=dest.name)) + "\n")
    return dest


def retire(rd, stage):
    """On a lab host: apply the analysis host's invalidation log (copied here) to the local copies of the same parts, so
    that write-once files do not block the repeat and an rsync cannot bring an old attempt back. Writes no log entry."""
    sd = R.stage_dir(rd, stage)
    moved = []
    for e in invalidation_log(rd, stage):
        src, dest = sd / e["part"], sd / "_invalidated" / e["moved_to"] / e["part"]
        if src.exists() and not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            moved.append(f"{e['part']} -> {e['moved_to']}")
        for d in DERIVED:
            if (sd / d).exists() and (sd / "_invalidated" / e["moved_to"]).exists():
                shutil.move(str(sd / d), str(sd / "_invalidated" / e["moved_to"] / d))
    return moved


# --------------------------------------------------------------------------- #
# Stage 0 (capture/, before any freeze)                                       #
# --------------------------------------------------------------------------- #
STAGE0_FILES = ("roles.json", "edge/inventory.json", "server/inventory.json", "edge/checkpoint.json", "server/checkpoint.json",
                "edge/deployment.json", "server/deployment.json", "split_capture.json", "capabilities.json", "candidates.json",
                "selected_config.json")


def capability_problems(capd, configuration_id):
    import gs_adapter as AD
    import gs_split as SP
    P = [f"capabilities.json: template placeholders left {SP.placeholders(capd)}"] if SP.placeholders(capd) else []
    info = {"contract_version": S.ADAPTER_CONTRACT_VERSION, "implementation": "declared at Stage 0", "configuration_id": configuration_id,
            "capabilities": capd.get("capabilities", []), "test_double": False, "notes": "",
            "missing_capability_justification": capd.get("missing_capability_justification", {})}
    P += [f"capabilities.json: {p}" for p in AD.check_info(info, target=True)]
    if not capd.get("basis"):
        P.append("capabilities.json: basis missing (code reading and lab_answers.md anchors)")
    return P


def clock_synchronised(inv):
    import re
    clock = inv.get("clock") or {}
    return "NTPSynchronized=yes" in (clock.get("timedatectl") or "") or bool(re.search(r"Leap status\s*:\s*Normal", clock.get("chrony") or ""))


def stage0(capture_dir):
    import gs_checkpoint as CK
    import gs_split as SP
    from gs_reference import norm_uuid
    cap = Path(capture_dir)
    v = {"schema": "gemma-split-stage0/1", "utc": C.utc(), "verdict": None, "reasons": [], "warnings": []}
    missing = [f for f in STAGE0_FILES if not (cap / f).exists()]
    if missing:
        v.update(verdict="STOP_INCOMPLETE", reasons=[f"missing capture files {missing}"])
        return v
    rj = lambda p: C.read_json(cap / p)  # noqa: E731
    roles = rj("roles.json")
    if not roles.get("verified"):
        v.update(verdict="STOP_ROLE_MISMATCH", reasons=roles.get("problems", ["roles not verified"]))
        return v
    for r in S.ROLES:
        if not clock_synchronised(rj(f"{r}/inventory.json")):
            v["reasons"].append(f"{r}: clock synchronisation not established (timedatectl NTPSynchronized=yes or chrony "
                                "'Leap status: Normal'); cross-host timestamps bind and order the runs (PREREG §12)")
    ck = {r: rj(f"{r}/checkpoint.json") for r in S.ROLES}
    for r in S.ROLES:
        if ck[r]["host"] != roles[r]["host"]:
            v["reasons"].append(f"{r} checkpoint captured on {ck[r]['host']}, not on the {r} host {roles[r]['host']}")
        if not ck[r].get("summary"):
            v["reasons"].append(f"{r} checkpoint has no config.json")
    if v["reasons"]:
        v["verdict"] = "STOP_INCOMPLETE"
        return v
    cmp_ = CK.compare(ck["edge"], ck["server"])
    v["checkpoint_comparison"] = cmp_
    split_cap = rj("split_capture.json")
    if not ck["server"]["dir_name_matches"] and not split_cap.get("model", {}).get("checkpoint_note"):
        v["reasons"].append(f"server checkpoint directory is {ck['server']['dir_name']!r}, not {S.MODEL_DIR_NAME!r}, and "
                            "split_capture.model.checkpoint_note does not explain the identity")
    if cmp_["status"] not in ("identical", "edge_subset"):
        v["reasons"].append(f"edge and server checkpoints differ ({cmp_['status']}): {cmp_['differing_paths'][:5]}")
    if cmp_["status"] != "identical":
        er = cap / "edge" / "checkpoint_reference.json"
        if not er.exists() or C.read_json(er)["manifest_sha256"] != ck["server"]["manifest_sha256"]:
            v["reasons"].append("no full checkpoint copy identical to the server's on the edge (edge/checkpoint_reference.json): "
                                "R_edge and the edge-only baseline need one")
    if v["reasons"]:
        v["verdict"] = "STOP_CHECKPOINT_MISMATCH"
        return v
    P, W = SP.validate(split_cap, ck["server"]["summary"])
    v["warnings"] += W
    for r in S.ROLES:
        known = {norm_uuid(d.get("uuid")) for d in roles[r].get("devices", [])}
        bad = [u for u in (split_cap.get("gpu_placement", {}).get(r) or {}).get("uuids", []) if u != "cpu" and norm_uuid(u) not in known]
        if bad:
            P.append(f"gpu_placement.{r}.uuids {bad} are not GPUs of the verified {r} host")
    P += capability_problems(rj("capabilities.json"), split_cap.get("configuration_id"))
    if P:
        only_unknowns = all(("unknowns remain" in p or "placeholders" in p) for p in P)
        v.update(verdict="STOP_INCOMPLETE" if only_unknowns else "STOP_SPLIT_CAPTURE_INVALID", reasons=P)
        return v
    sel = rj("selected_config.json")
    if sel.get("verdict") != "CONTINUE":
        v.update(verdict="STOP_NO_CONFIGURATION", reasons=[f"selection verdict {sel.get('verdict')}"])
        return v
    if sel.get("selected") != split_cap["configuration_id"]:
        v.update(verdict="STOP_SPLIT_CAPTURE_INVALID", reasons=[f"selected {sel.get('selected')} but split_capture describes "
                                                                f"{split_cap['configuration_id']}"])
        return v
    for r in S.ROLES:
        dep = rj(f"{r}/deployment.json")
        if dep["host"] != roles[r]["host"]:
            v.update(verdict="STOP_INCOMPLETE", reasons=[f"{r} deployment captured on {dep['host']}"])
            return v
        for s in dep["sources"]:
            if (s.get("git") or {}).get("dirty"):
                v["warnings"].append(f"{r}: source {s['root']} has uncommitted changes (frozen as found by the snapshot)")
    v["derived"] = SP.derive(split_cap, ck["server"]["summary"])
    v["capabilities"] = sorted(rj("capabilities.json")["capabilities"])
    v["verdict"] = "CONTINUE"
    return v


# --------------------------------------------------------------------------- #
# Stages 2-5: pure computations (no writes), then gate_* = require + compute + write                       #
# --------------------------------------------------------------------------- #
def _missing(rd, stage, freeze):
    sd = R.stage_dir(rd, stage)
    return [f for f in R.expected(stage, freeze["adapter"]["info"]["capabilities"]) if not (sd / f).exists()]


def compute_stage2(rd, freeze, dev=False):
    missing = [] if dev else [f"stage1:{f}" for f in _missing(rd, "stage1", freeze)]
    if missing:
        raise GateNotReady(f"stage1 checks missing: {missing}")
    body, P, discarded = accounting(rd, freeze, "stage2")
    if body:
        return body
    return _finish(_stage2_body(rd, freeze, dev, P), discarded)


def _stage2_body(rd, freeze, dev, P):
    sd = R.stage_dir(rd, "stage2")
    rec = {(r, w): C.read_json(sd / r / f"{w}_memory.json") for r in S.ROLES for w in ("split", "reference")}
    for (r, w), v in rec.items():
        P += [RU.problem(r, "invalid", p) for p in R.check_provenance(v.get("provenance"), freeze, "stage2", r, require_target=not dev,
                                                                        what=f"{r}/{w}_memory.json")]
        if not RU.devices_ok(v.get("devices"), freeze, r):
            P.append(RU.problem(r, "invalid", f"{r}/{w}_memory.json ran on {v.get('devices')}, not the frozen {r} placement"))
        try:
            R.assert_no_timing_fields({k: x for k, x in v.items() if k != "provenance"})
        except R.TimingFieldError as e:
            P.append(RU.problem(r, "invalid", f"{r}/{w}_memory.json: timing field {e}"))
    if not dev:
        for r in S.ROLES:
            chk = C.read_json(R.stage_dir(rd, "stage1") / f"{r}_workload.json")
            P += [RU.problem(r, "invalid", p) for p in R.check_provenance(chk.get("provenance"), freeze, "stage1", r, what=f"{r}_workload.json")]
            if not chk.get("match"):
                P.append(RU.problem(r, "workload_mismatch", f"{r}: the frozen token ids are not reproduced by this host's tokenizer"))
    if P:
        return _invalid_body(rd, freeze, "stage2", P)
    split, per = RU.combine_split(rec[("edge", "split")], rec[("server", "split")])
    refs = {r: rec[(r, "reference")]["status"] for r in S.ROLES}
    extra = {"split": split, "split_per_host": per, "reference_per_host": refs}
    statuses = list(per.values()) + list(refs.values())
    if "INFEASIBLE" in statuses:
        probs = [RU.problem(r, "confirm_infeasible", f"allocation failure ({'split' if split == 'INFEASIBLE' else 'reference'}): "
                                                     "confirm once with fresh processes") for r in S.ROLES]
        repeatable, exhausted, _ = apply_caps(rd, freeze, "stage2", probs)
        if not exhausted:
            return dict(extra, verdict="INFEASIBLE_UNCONFIRMED", final=False, repeatable=repeatable, reasons=[p["reason"] for p in probs])
        if split == "INFEASIBLE":
            return dict(extra, verdict="STOP_INFEASIBLE", final=True, repeatable=[], reasons=[
                "the frozen configuration does not fit on " + ", ".join(h for h, s in per.items() if s == "INFEASIBLE") +
                " (confirmed): recorded as infeasible; the configuration is not changed (PREREG §8)"])
        return dict(extra, verdict="CONTINUE", final=False, repeatable=[], reference_feasible=False,
                    reasons=["an unsplit reference does not fit (confirmed): Stage 3 will be NOT_EVALUABLE"])
    if "BLOCKED_CONTENTION" in statuses:
        probs = [RU.problem(r, "blocked", "foreign compute process at the start: nothing ran; retry when the GPU is quiet") for r in S.ROLES]
        return dict(extra, verdict="BLOCKED_CONTENTION", final=False, repeatable=apply_caps(rd, freeze, "stage2", probs)[0], reasons=[probs[0]["reason"]])
    if "INVALID" in statuses:
        return _invalid_body(rd, freeze, "stage2", [RU.problem(r, "invalid", f"non-memory failure or contention ({per} / {refs})") for r in S.ROLES],
                             extra=extra)
    return dict(extra, verdict="CONTINUE", final=False, repeatable=[], reference_feasible=True, reasons=[])


def compute_stage3(rd, freeze, dev=False, st2=None):
    st2 = st2 if st2 is not None else read_gate(rd, "stage2") or {}
    if st2.get("reference_feasible", True) is False:         # nothing to run: the reference does not fit (Stage 2)
        dec = RU.decide_stage3(rd, freeze, reference_feasible=False, dev=dev)
        return {"verdict": "NOT_EVALUABLE", "final": True, "repeatable": [], "tier": dec["tier"],
                "reasons": dec["reasons"] + ["performance is not interpreted for this configuration (PREREG §13)"]}, dec
    body, att, discarded = accounting(rd, freeze, "stage3")
    if body:
        return dict(body, tier=freeze["adapter"]["tier"]), {"stage": "stage3", "tier": freeze["adapter"]["tier"], "verdict": body["verdict"],
                                                            "problems": body["problems"], "rules": {}, "diagnostics": {}}
    dec = RU.decide_stage3(rd, freeze, reference_feasible=True, dev=dev)
    if att:
        dec.update(verdict="INVALID", problems=dec["problems"] + att)
    if dec["verdict"] == "INVALID":
        body = _invalid_body(rd, freeze, "stage3", dec["problems"], extra={"tier": dec["tier"]},
                             exhausted_verdict=lambda ex: "FAIL" if any(e["cause"] == "exceptions" for e in ex) else "INCOMPLETE")
        if body["verdict"] == "FAIL":
            body["reasons"].append("the deployment did not complete the correctness runs within the allowed attempts: not runnable as captured")
    else:
        body = {"verdict": dec["verdict"], "final": dec["verdict"] != "PASS", "repeatable": [], "tier": dec["tier"],
                "reasons": dec.get("reasons", []) + ([] if dec["verdict"] == "PASS" else
                                                     ["performance is not interpreted for this configuration (PREREG §13)"])}
    return _finish(body, discarded), dec


def compute_stage4(rd, freeze, dev=False):
    body, P, discarded = accounting(rd, freeze, "stage4")
    if body:
        return body, None
    sd = R.stage_dir(rd, "stage4")
    for role in S.ROLES:
        mp = sd / role / "meta.json"
        if not mp.exists():
            continue
        meta = C.read_json(mp)
        P += [RU.problem(role, "invalid", p) for p in R.check_provenance(meta["provenance"], freeze, "stage4", role, require_target=not dev,
                                                                           what=f"{role}/meta.json")]
        if not RU.devices_ok(meta["provenance"].get("devices"), freeze, role):
            P.append(RU.problem(role, "invalid", f"{role} components ran on {meta['provenance'].get('devices')}, not the frozen placement"))
        if meta.get("failure"):
            P.append(RU.problem(role, "invalid", f"{role}: {meta['failure'][:200]}"))
        for k in ("contention_before", "contention_after"):
            if len(meta[k]["foreign"]) > S.FOREIGN_PROCESSES_ALLOWED:
                P.append(RU.problem(role, "invalid", f"{role}: foreign compute process ({k})"))
    serve_p = sd / "edge" / "server_serve.json"
    if serve_p.exists():
        serve = C.read_json(serve_p)
        P += [RU.problem("edge", "invalid", p) for p in R.check_provenance(serve.get("provenance"), freeze, "stage4", "server",
                                                                             require_target=not dev, what="edge/server_serve.json")]
        if not RU.devices_ok((serve.get("provenance") or {}).get("devices"), freeze, "server"):
            P.append(RU.problem("edge", "invalid", "the server half served on devices outside the frozen server placement"))
        rows = C.read_jsonl(sd / "edge" / "trace.jsonl")
        if not RU.within_window([r.get("utc") for r in rows], serve["started_utc"], serve["ended_utc"]):
            P.append(RU.problem("edge", "invalid", "in-path trace falls outside the server's serve window for this run"))
    if P:
        return _finish(_invalid_body(rd, freeze, "stage4", P), discarded), None
    summ = RU.summarize_components(rd, freeze)
    return _finish({"verdict": "COMPLETE", "final": False, "repeatable": [], "mode": summ["mode"],
                    "reasons": [] if summ["mode"] != "not_available" else ["no component profile: neither segment_step nor step_trace (reported)"]},
                   discarded), summ


def compute_stage5(rd, freeze, dev=False, components=None):
    body, att, discarded = accounting(rd, freeze, "stage5")
    if body:
        return dict(body, category=None), {"verdict": body["verdict"], "problems": body["problems"], "primary": None}
    components = components if components is not None else RU.summarize_components(rd, freeze)
    summ = RU.summarize_e2e(rd, freeze, components, dev=dev)
    if att:
        summ.update(verdict="INVALID", problems=summ["problems"] + att)
    if summ["verdict"] == "INVALID":
        body = _invalid_body(rd, freeze, "stage5", summ["problems"],
                             exhausted_verdict=lambda ex: "UNSTABLE" if any(e["cause"] == "failed_requests" for e in ex) else "INCOMPLETE")
        if body["verdict"] == "UNSTABLE":
            body["unstable_cause"] = "request_failures"
    elif summ["verdict"] == "UNSTABLE":
        probs = [RU.problem(b, "drift", f"split_A vs split_B drift {summ['drift']['rel']} > {S.DRIFT_MAX_REL}") for b in summ["block_order"]]
        repeatable, exhausted, _ = apply_caps(rd, freeze, "stage5", probs)
        body = {"verdict": "UNSTABLE", "final": bool(exhausted), "repeatable": [] if exhausted else repeatable, "unstable_cause": "drift",
                "reasons": [probs[0]["reason"] + ("; already repeated once: final" if exhausted else "; repeat all blocks once")]}
    else:
        body = {"verdict": "COMPLETE", "final": True, "repeatable": [], "reasons": []}
    body["category"] = (summ.get("primary") or {}).get("category") if body["verdict"] == "COMPLETE" else None
    return _finish(body, discarded), summ


def gate_stage2(rd, freeze, dev=False):
    return write_gate(rd, freeze, "stage2", compute_stage2(rd, freeze, dev))


def gate_stage3(rd, freeze, dev=False):
    require(rd, freeze, "stage3")
    body, dec = compute_stage3(rd, freeze, dev)
    C.write_json(R.stage_dir(rd, "stage3") / "decision.json", dec)
    return write_gate(rd, freeze, "stage3", body)


def gate_stage4(rd, freeze, dev=False):
    require(rd, freeze, "stage4")
    body, summ = compute_stage4(rd, freeze, dev)
    if summ is not None:
        C.write_json(R.stage_dir(rd, "stage4") / "summary.json", summ)
    return write_gate(rd, freeze, "stage4", body)


def gate_stage5(rd, freeze, dev=False):
    require(rd, freeze, "stage5")
    body, summ = compute_stage5(rd, freeze, dev)
    C.write_json(R.stage_dir(rd, "stage5") / "summary.json", summ)
    return write_gate(rd, freeze, "stage5", body)


# --------------------------------------------------------------------------- #
# Outcome (PREREG §13.3)                                                      #
# --------------------------------------------------------------------------- #
def recompute(rd, freeze, dev=False):
    """Every stored gate recomputed from the raw files (nothing written). None where outputs are missing."""
    out = {}
    try:
        out["stage2"] = compute_stage2(rd, freeze, dev)
    except GateNotReady:
        out["stage2"] = None
    comps = None
    for st, fn in (("stage3", compute_stage3), ("stage4", compute_stage4), ("stage5", compute_stage5)):
        if read_gate(rd, st) is None:
            break
        try:
            if st == "stage3":
                out[st] = fn(rd, freeze, dev, out.get("stage2") or {})[0]
            elif st == "stage4":
                out[st], comps = fn(rd, freeze, dev)
            else:
                out[st] = fn(rd, freeze, dev, comps)[0]
        except GateNotReady:
            out[st] = None
    return out


def repeats(rd, freeze):
    """Every repeat over the record chain, with what an invalidated Stage-5 attempt would have shown (transparency)."""
    out = []
    for d in [Path(rd)] + chain_run_dirs(rd, freeze):
        for stage in S.STAGE_ORDER:
            events = {}
            for e in invalidation_log(d, stage):
                ev = events.setdefault(e["moved_to"], {"run_id": d.name, "stage": stage, "moved_to": e["moved_to"], "utc": e["utc"],
                                                       "parts": [], "causes": set(), "gate_verdict": e.get("gate_verdict")})
                ev["parts"].append(e["part"])
                ev["causes"] |= set(_causes(e))
            for ev in events.values():
                summ = R.stage_dir(d, stage) / "_invalidated" / ev["moved_to"] / "summary.json"
                ev["causes"] = sorted(ev["causes"])
                dec = R.stage_dir(d, stage) / "_invalidated" / ev["moved_to"] / "decision.json"
                ev["would_have_shown"] = ((C.read_json(summ).get("primary") or {}).get("category") if stage == "stage5" and summ.exists() else
                                          (C.read_json(dec).get("would_be") or {}).get("verdict") if stage == "stage3" and dec.exists() else None)
                out.append(ev)
    return out


OUTCOME_OF = {("stage2", "STOP_INFEASIBLE"): "INFEASIBLE", ("stage3", "FAIL"): "NOT_EQUIVALENT", ("stage3", "NOT_EVALUABLE"): "NOT_EVALUABLE",
              ("stage5", "COMPLETE"): "COMPLETE", ("stage5", "UNSTABLE"): "UNSTABLE"}


def outcome(rd, freeze, history=None):
    dev = bool(freeze.get("dev"))
    stored = {s: read_gate(rd, s) for s in S.STAGE_ORDER}
    res = {"freeze_sha256": freeze["freeze_sha256"], "run_id": freeze["run_id"], "mode": mode_of(freeze),
           "gates": {s: (g or {}).get("verdict") for s, g in stored.items()}, "outcome": None, "efficiency_category": None,
           "problems": [], "note": "", "history": history or [], "repeats": repeats(rd, freeze)}
    for s, g in stored.items():
        if g:
            res["problems"] += check_gate(g, freeze, f"{s} gate")
    again = recompute(rd, freeze, dev)
    for s, g in stored.items():
        if g and (again.get(s) is None or again[s]["verdict"] != g["verdict"] or bool(again[s].get("final")) != bool(g.get("final"))):
            res["problems"].append(f"{s}: the stored gate does not match a recomputation from its raw files")
    verify = {}
    for role in S.ROLES:
        p = R.stage_dir(rd, "final") / f"checkpoint_verify_{role}.json"
        verify[role] = C.read_json(p).get("identical") if p.exists() else None
        if verify[role] is False:
            res["problems"].append(f"{role}: checkpoint differs from its Stage-0 manifest: every stage of this run is INVALID")
    res["checkpoint_verified"] = verify
    if res["problems"]:
        res["note"] = "no outcome: " + "; ".join(res["problems"])
        return res
    final = next(((s, g) for s, g in stored.items() if g and g.get("final")), None)
    if final is None:
        pending = [s for s, g in stored.items() if g and g["verdict"] not in (S.OPEN_VERDICT.get(s),)]
        res["note"] = f"PENDING: {pending or 'stages not complete'}"
        return res
    if None in verify.values():
        res["note"] = "PENDING: run gs_bench.py verify-checkpoint on both hosts first (PREREG §15)"
        return res
    s, g = final
    res["outcome"] = OUTCOME_OF.get((s, g["verdict"]), "INCOMPLETE")
    res["final_stage"] = s
    res["efficiency_category"] = g.get("category") if res["outcome"] == "COMPLETE" else None
    res["note"] = "; ".join(g.get("reasons", []))
    return res
