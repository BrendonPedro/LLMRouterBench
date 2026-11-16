"""GEMMA-SPLIT command hub (PREREG §2). Machines: EDGE = DGX Spark, SERVER = asus-ESC4000-E11, DESK = analysis host.

Stage 0  (facts; no model, no GPU kernel, no timer)
  inventory --role R --peer P --out-dir capture/R          EDGE, SERVER   gs_inventory
  checkpoint --role R --model-dir D --out capture/R/checkpoint.json        gs_checkpoint
  deployment --role R --source-root D [--launch-file F] --out capture/R/deployment.json   gs_deployment
  roles --capture-dir capture                               DESK           gs_inventory
  split-validate --capture capture/split_capture.json --checkpoint capture/server/checkpoint.json
  select --candidates capture/candidates.json --roles capture/roles.json --out capture/selected_config.json
  stage0 --capture-dir capture                              DESK           gs_gate.stage0 -> capture/stage0_verdict.json
Stage 1  (adapter + freeze; still no timer)
  tokenize --tokenizer-dir D --out capture/workload_tokens.json            gs_workload
  snapshot --role R --dest DIR                              EDGE, SERVER   -> capture/R/snapshot.json
  adapter-check --adapter adapter/<file>.py                 DESK           static contract check
  adapter-smoke --adapter adapter/<file>.py --role R        EDGE, SERVER   one short request, types only; no timer, no output kept
  freeze --adapter adapter/<file>.py --out frozen/freeze_record.json [--supersede-reason TEXT] | check-freeze --freeze F --role R
  verify-workload --freeze F --role R                       EDGE, SERVER   frozen token ids re-derived with this host's tokenizer
Stages 2-5 (every command: --freeze F first runs check-freeze, then the gate of the previous stage; preflight checks
           the frozen GPU placement, CUDA_VISIBLE_DEVICES, the checkpoint and the source snapshot before loading anything)
  serve --stage {2,3,4,5} [--block split_A|split_B]         SERVER         the deployment's server half (stop: Ctrl-C or SIGTERM)
  memory --what split --role edge | --what reference --role R               Stage 2 (no timer)
  reference --role server|edge ; correctness --what split|pipeline          Stage 3 (no timer)
  components --role R                                       Stage 4
  e2e --block B                                             Stage 5 (blocks: gs_spec.block_order; split_*/edge_*: EDGE, server_*: SERVER)
  gate --stage {2,3,4,5} ; invalidate --stage N (--part P ... | --all) --note TEXT ; outcome     DESK
  retire --stage N                                          EDGE, SERVER   apply DESK's invalidations to local copies
Stage 6  verify-checkpoint --role R (EDGE, SERVER) ; archive --out F.tgz ; verify-archive --archive F.tgz
--dev (tests and dry runs only): a dev freeze record, a test-double adapter, a model below gs_spec.DEV_MAX_MODEL_BYTES,
--device/--endpoint/--model-dir/--gs-dir overrides; outputs are labelled dev and no target gate accepts them.
"""
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # python -I: the script dir is not on sys.path
import gs_adapter as AD  # noqa: E402
import gs_common as C  # noqa: E402
import gs_freeze as F  # noqa: E402
import gs_gate as G  # noqa: E402
import gs_results as R  # noqa: E402
import gs_rules as RU  # noqa: E402
import gs_spec as S  # noqa: E402

DELEGATED = {"inventory": ("gs_inventory", "collect"), "roles": ("gs_inventory", "roles"), "checkpoint": ("gs_checkpoint", "capture"),
             "deployment": ("gs_deployment", "capture"), "split-validate": ("gs_split", "validate"), "select": ("gs_split", "select"),
             "tokenize": ("gs_workload", "tokenize"), "freeze": ("gs_freeze", "freeze"), "check-freeze": ("gs_freeze", "check-freeze"),
             "archive": ("gs_archive", "create"), "verify-archive": ("gs_archive", "verify")}
OPEN = ("CONTINUE", "PASS", "COMPLETE")
LOCAL_EVIDENCE = {"edge": {"split_A": "requests.jsonl", "edge_only": "requests.jsonl", "edge_only_native": "requests.jsonl",
                           "split_B": "requests.jsonl"},
                  "server": {"split_A": "server_meta.json", "server_only": "requests.jsonl", "server_only_native": "requests.jsonl",
                             "split_B": "server_meta.json"}}


class UsageError(SystemExit):
    pass


def _common(p, freeze=True):
    if freeze:
        p.add_argument("--freeze", required=True)
    p.add_argument("--gs-dir", default=str(S.GS_DIR), help="dev only")
    p.add_argument("--dev", action="store_true")
    p.add_argument("--device", default=None, help="dev only")
    p.add_argument("--endpoint", default=None, help="dev only")
    p.add_argument("--model-dir", default=None, help="dev only")


def _dir_bytes(d):
    return sum(p.stat().st_size for p in Path(d).rglob("*") if p.is_file())


def ctx(a, role):
    gs_dir = Path(a.gs_dir).resolve()
    if not a.dev:
        if a.device or a.endpoint or a.model_dir:
            raise UsageError("--device / --endpoint / --model-dir are dev-only overrides (PREREG §6.3)")
        if gs_dir != S.GS_DIR:
            raise UsageError(f"--gs-dir must be the directory of this gs_bench.py ({S.GS_DIR}) in a target run")
    try:
        freeze = F.check(a.freeze, gs_dir=gs_dir, role=None if a.dev else role, dev=a.dev)
    except F.FreezeMismatch as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)
    if a.dev:
        sizes = [freeze["checkpoint"][r]["total_bytes"] for r in S.ROLES] + ([_dir_bytes(a.model_dir)] if a.model_dir else [])
        if max(sizes) > S.DEV_MAX_MODEL_BYTES:
            raise UsageError("dev mode runs tiny test checkpoints only (PREREG §5.4): this model is too large")
    return gs_dir, freeze, R.run_dir(gs_dir, freeze), ("dev" if a.dev else "target")


def opts(a):
    return {k: v for k, v in (("endpoint", a.endpoint), ("device", a.device), ("model_dir", a.model_dir)) if v} if a.dev else {}


def placement(freeze, role):
    gp = freeze["split"]["capture"]["gpu_placement"][role]
    return gp["devices"], gp.get("uuids", [])


def preflight(freeze, gs_dir, role, a, loads_deployment=True):
    """Before anything is loaded (target runs): CUDA_VISIBLE_DEVICES as frozen, every frozen device carries its frozen UUID,
    the deployment's checkpoint matches its Stage-0 manifest, the source snapshot matches the capture. Exit 2 otherwise."""
    if a.dev:
        return
    import gs_checkpoint as CK
    import gs_deployment as DP
    import gs_reference as REF
    P = []
    gp = freeze["split"]["capture"]["gpu_placement"][role]
    want, live = gp.get("CUDA_VISIBLE_DEVICES") or None, os.environ.get("CUDA_VISIBLE_DEVICES") or None
    if want != live:
        P.append(f"CUDA_VISIBLE_DEVICES={live!r}, frozen {want!r}")
    else:
        for dev, uuid in zip(gp["devices"], gp.get("uuids", [])):
            try:
                REF.check_device_uuid(dev, [uuid])
            except Exception as e:
                P.append(str(e))
    if loads_deployment:
        cap = C.read_json(Path(gs_dir) / "capture" / role / "checkpoint.json")
        if CK.quick_check(cap) and not CK.deep_check(cap):
            P.append(f"{cap['model_dir']} differs from its Stage-0 manifest")
        snap = freeze["deployment"][role].get("snapshot") or {}
        dep = C.read_json(Path(gs_dir) / "capture" / role / "deployment.json")
        for src, rec in zip(dep["sources"], snap.get("sources", [])):
            bad = DP.verify_snapshot(src, rec["snapshot_root"])
            if bad:
                P.append(f"source snapshot {rec['snapshot_root']} differs from the capture: {bad[:5]}")
    if P:
        print("PREFLIGHT FAILED (nothing loaded, nothing written):\n  - " + "\n  - ".join(P), file=sys.stderr)
        sys.exit(2)


def check_devices(freeze, role, used):
    frozen, _ = placement(freeze, role)
    if not set(used or []) <= set(frozen):
        raise UsageError(f"the adapter uses {used}, not the frozen {role} placement {frozen}")


class LiveTreeGuard:
    """Refuses the live deployment tree in a target run: the adapter runs the deployment's code from the hash-verified
    snapshot only (PREREG §4.3). An import hook refuses modules found there; an audit hook (PEP 578) refuses opening,
    executing or spawning anything there (exec(open(...)), spec_from_file_location, runpy, subprocess, os.system / exec /
    spawn, chdir). The checkpoint directories (hash-verified) and the snapshot stay allowed even inside the tree. Not
    covered: a child process that reads the live tree by itself (the deployment's scripts come from the snapshot)."""
    ACTIVE = []             # audit hooks cannot be removed: the hook checks the guards listed here (tests clear it)
    EVENTS = ("open", "subprocess.Popen", "os.system", "os.exec", "os.spawn", "os.posix_spawn", "os.chdir")
    _hooked = False
    _busy = threading.local()

    def __init__(self, roots, allowed=()):
        self.roots = sorted({str(Path(r).resolve()) for r in roots} | {os.path.abspath(r) for r in roots})
        self.allowed = [Path(a).resolve() for a in allowed]

    def _live(self, f):
        try:
            a = os.path.abspath(os.fsdecode(f))
        except Exception:
            return False
        if not any(a == r or a.startswith(r.rstrip("/") + "/") for r in self.roots):
            return False
        rp = Path(a).resolve()
        return not any(rp == x or x in rp.parents for x in self.allowed)

    def _text_hits(self, text):
        for r in self.roots:
            i = text.find(r)
            while i >= 0:
                j = i
                while j < len(text) and text[j] not in " \t\n'\";,=":
                    j += 1
                if self._live(text[i:j]):
                    return text[i:j]
                i = text.find(r, j)
        return None

    def check_event(self, event, args):
        if event == "open":
            hit = args[0] if isinstance(args[0], (str, bytes, os.PathLike)) and self._live(args[0]) else None
        elif event == "os.chdir":
            hit = args[0] if self._live(args[0]) else None
        else:
            parts = [x for a in args for x in (a if isinstance(a, (list, tuple)) else [a]) if isinstance(x, (str, bytes, os.PathLike))]
            hit = next((h for h in (self._text_hits(os.fsdecode(x)) for x in parts) if h), None)
        if hit:
            raise PermissionError(f"{event}: {hit} lies in the live deployment tree: use the snapshot (PREREG §4.3)")

    @classmethod
    def _audit(cls, event, args):
        if not cls.ACTIVE or event not in cls.EVENTS or getattr(cls._busy, "on", False):
            return
        cls._busy.on = True
        try:
            for g in cls.ACTIVE:
                g.check_event(event, args)
        finally:
            cls._busy.on = False

    def install(self):
        if not LiveTreeGuard._hooked:
            sys.addaudithook(LiveTreeGuard._audit)
            LiveTreeGuard._hooked = True
        if self._live(os.getcwd()):
            raise UsageError(f"the working directory {os.getcwd()} lies in the live deployment tree")
        sys.meta_path.insert(0, self)
        LiveTreeGuard.ACTIVE.append(self)
        return self

    def find_spec(self, name, path=None, target=None):
        import importlib.machinery
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is not None and spec.origin and self._live(spec.origin):
            raise ImportError(f"{name} would be imported from the live deployment tree ({spec.origin}): import it from the snapshot")
        return None

    def loaded_live(self):
        return sorted(n for n, m in list(sys.modules.items()) if getattr(m, "__file__", None) and self._live(m.__file__))


def live_guard(gs_dir, roles, snapshots=()):
    """The guard for a target command: the captured source roots and launch files are live; the checkpoints, the snapshots
    and the running interpreter's environment (a venv may sit inside the source root) are allowed."""
    import site
    cap = Path(gs_dir) / "capture"
    deps = [C.read_json(cap / r / "deployment.json") for r in roles]
    roots = [src["root"] for d in deps for src in d["sources"]] + [lf["path"] for d in deps for lf in d.get("launch_files", [])
                                                                   if not lf.get("missing")]
    allowed = [C.read_json(q)["model_dir"] for r in roles for q in (cap / r / "checkpoint.json", cap / r / "checkpoint_reference.json") if q.exists()]
    allowed += [x["snapshot_root"] for snap in snapshots if snap for x in snap.get("sources", [])]
    allowed += [sys.prefix, sys.exec_prefix] + list(site.getsitepackages()) + [site.getusersitepackages()]   # the interpreter's own environment
    guard = LiveTreeGuard(roots, allowed)
    live = guard.loaded_live()
    if live:
        raise UsageError(f"modules already imported from the live deployment tree: {live[:5]}")
    return guard.install()


def make_adapter(freeze, role, gs_dir, a):
    """AD.make behind the live-tree guard (target runs)."""
    if not a.dev:
        live_guard(gs_dir, [role], [freeze["deployment"][role].get("snapshot")])
    return AD.make(freeze, role, gs_dir, opts(a))


def mark_run(a):
    """RUN event (once): the measurement begins, after connect and the device check. Attempt accounting counts only runs
    that got here; a refusal or connection failure before it is not an attempt (PREREG §13.2)."""
    lc = getattr(a, "log_ctx", None)
    if lc and not lc.get("run"):
        R.log_event(lc["rd"], "RUN", lc["token"], lc["stage"], lc["part"], lc["cmd"])
        lc["run"] = True


def end_forced(a):
    """END rc=0 for a server half whose record was written after a forced stop (os._exit skips the normal END)."""
    lc = getattr(a, "log_ctx", None)
    if lc:
        R.log_event(lc["rd"], "END", lc["token"], lc["stage"], lc["part"], lc["cmd"], rc="0")


def ref_device(freeze, role, a):
    return a.device if (a.dev and a.device) else placement(freeze, role)[0][0]


def ref_model_dir(freeze, role, a, gs_dir):
    if a.dev and a.model_dir:
        return a.model_dir
    import gs_checkpoint as CK
    use_ref = role == "edge" and freeze["checkpoint"].get("edge_reference")
    cap = C.read_json(Path(gs_dir) / "capture" / role / ("checkpoint_reference.json" if use_ref else "checkpoint.json"))
    if not a.dev and CK.quick_check(cap) and not CK.deep_check(cap):
        raise SystemExit(f"{cap['model_dir']}: checkpoint differs from its Stage-0 manifest: the stage is INVALID (PREREG §6.2)")
    return cap["model_dir"]


def _gate_or_exit(rd, freeze, stage):
    try:
        G.require(rd, freeze, stage)
    except G.GateClosed as e:
        print(f"GATE CLOSED: {e}", file=sys.stderr)
        sys.exit(3)


def check_local_order(rd, block, role, caps):
    """Every earlier block this host took part in must exist here (the gate re-checks the order of all blocks)."""
    order = S.block_order(caps)
    if block not in order:
        raise UsageError(f"block {block} is not in this run's order {order}")
    ev = LOCAL_EVIDENCE[role]
    missing = [b for b in order[:order.index(block)] if b in ev and not (R.stage_dir(rd, "stage5") / b / ev[b]).exists()]
    if missing:
        raise UsageError(f"blocks run in the frozen order {order}: on this host {missing} have not run")


def _probe_item(freeze, gs_dir):
    w = C.read_json(Path(gs_dir) / freeze["workload"]["tokens_file"])
    return next(it for it in w["items"] if it["id"] == freeze["workload"]["probe_item"])


def _unified(freeze, role):
    return any(d.get("memory_total_mib") is None and d.get("uuid") not in (None, "cpu") for d in freeze["machines"][role]["devices"])


# --------------------------------------------------------------------------- #
def cmd_stage0(a):
    v = G.stage0(a.capture_dir)
    C.write_json(Path(a.capture_dir) / "stage0_verdict.json", v)
    print(v["verdict"], *v["reasons"], sep="\n  - ")
    for d in (v.get("derived") or {}).get("differences_from_qwen3_e1opt", []):
        print("  difference from Qwen3 E1-OPT:", d)
    sys.exit(0 if v["verdict"] == "CONTINUE" else 3)


def cmd_snapshot(a):
    import gs_deployment as DP
    cap = Path(a.gs_dir) / "capture" / a.role
    dep = C.read_json(cap / "deployment.json")
    recs = [DP.snapshot(src, Path(a.dest) / f"source{i}") for i, src in enumerate(dep["sources"])]
    C.write_json(cap / "snapshot.json", {"role": a.role, "host": dep["host"], "sources": recs})
    print(C.canonical(recs))


def cmd_adapter_check(a):
    info = AD.read_info(Path(a.gs_dir) / a.adapter)
    P = AD.check_info(info, target=not a.dev)
    print(C.canonical(info))
    print("adapter OK, tier", AD.tier(info) if not P else "", *P, sep="\n  - " if P else " ")
    sys.exit(0 if not P else 2)


def pseudo_freeze(gs_dir, adapter_rel):
    """What an adapter can read before a freeze record exists (adapter-smoke): the capture, never a measurement."""
    cap = Path(gs_dir) / "capture"
    return {"split": {"capture": C.read_json(cap / "split_capture.json")},
            "checkpoint": {r: {"model_dir": C.read_json(cap / r / "checkpoint.json")["model_dir"]} for r in S.ROLES},
            "deployment": {r: {"snapshot": C.read_json(cap / r / "snapshot.json") if (cap / r / "snapshot.json").exists() else None}
                           for r in S.ROLES},
            "adapter": {"path": adapter_rel, "info": AD.read_info(Path(gs_dir) / adapter_rel)}, "dev": False}


def cmd_adapter_smoke(a):
    """Contract smoke test for the deployment's adapter BEFORE the freeze (PREREG §5.4): one short greedy request through
    the split, types and boundary audit checked; no timer, no reference, nothing written, token ids not printed."""
    gs_dir = Path(a.gs_dir)
    fr = pseudo_freeze(gs_dir, a.adapter)
    if a.dev:
        if not fr["adapter"]["info"].get("test_double"):
            raise UsageError("adapter-smoke --dev runs test doubles only: the deployment's adapter is smoke-tested without --dev (PREREG §5.4)")
        dirs = [fr["checkpoint"][r]["model_dir"] for r in S.ROLES] + ([a.model_dir] if a.model_dir else [])
        if max(_dir_bytes(d) for d in dirs) > S.DEV_MAX_MODEL_BYTES:
            raise UsageError("dev mode runs tiny test checkpoints only (PREREG §5.4): this model is too large")
    else:
        info, capd = fr["adapter"]["info"], C.read_json(gs_dir / "capture" / "capabilities.json")
        P = AD.check_info(info, target=True) + F.capabilities_declared_first(gs_dir)
        if set(info["capabilities"]) != set(capd["capabilities"]):
            P.append("ADAPTER_INFO capabilities differ from the committed declaration capture/capabilities.json")
        if P:
            raise UsageError("adapter-smoke refused:\n  - " + "\n  - ".join(P))
        live_guard(gs_dir, S.ROLES, [C.read_json(gs_dir / "capture" / r / "snapshot.json") for r in S.ROLES
                                     if (gs_dir / "capture" / r / "snapshot.json").exists()])
    ad = AD.load(gs_dir / a.adapter).make_adapter(fr, a.role, {k: v for k, v in (("endpoint", a.endpoint), ("device", a.device),
                                                                                ("model_dir", a.model_dir)) if v})
    if a.role == "server":
        stop = threading.Event()
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        ad.serve(ready=lambda: print("READY (smoke)", flush=True), stop=stop)
        return
    P = []
    ad.connect()
    try:
        out = ad.greedy(list(range(2, 18)) if a.synthetic_ids else _smoke_ids(gs_dir), 4, [], False)
        if not (isinstance(out.get("tokens"), list) and all(isinstance(t, int) for t in out["tokens"]) and len(out["tokens"]) == 4):
            P.append(f"greedy() must return 4 int token ids, got {type(out.get('tokens'))}")
        if not isinstance(out.get("stopped_on_eos"), bool):
            P.append("greedy() must return stopped_on_eos as bool")
        audit = ad.boundary_audit()
        if not audit or any(m.get("direction") not in ("edge_to_server", "server_to_edge") or not m.get("tensors") for m in audit):
            P.append("boundary_audit() returned no messages or malformed messages")
    finally:
        AD.safe_close(ad)
    print("contract OK" if not P else "contract problems:\n  - " + "\n  - ".join(P))
    sys.exit(0 if not P else 2)


def _smoke_ids(gs_dir):
    w = Path(gs_dir) / "capture" / "workload_tokens.json"
    if not w.exists():
        raise UsageError("adapter-smoke needs capture/workload_tokens.json (or --synthetic-ids)")
    return C.read_json(w)["items"][0]["ids"][:16]


def cmd_verify_workload(a):
    gs_dir, freeze, rd, pr = ctx(a, a.role)
    import gs_workload as WL
    tok_dir = ref_model_dir(freeze, a.role, a, gs_dir)
    w = WL.tokenize(tok_dir, gs_dir / "workload")
    out = {"role": a.role, "tokenizer_dir": tok_dir, "ids_sha256": w["ids_sha256"], "frozen_ids_sha256": freeze["workload"]["ids_sha256"],
           "match": w["ids_sha256"] == freeze["workload"]["ids_sha256"], "provenance": R.provenance(freeze, "stage1", a.role, [], pr)}
    R.write(R.stage_dir(rd, "stage1") / f"{a.role}_workload.json", out, timing_allowed=False)
    print(a.role, "token ids reproduced" if out["match"] else "TOKEN IDS DIFFER: the record is wrong for this host (INVALID)")
    sys.exit(0 if out["match"] else 2)


def cmd_serve(a):
    gs_dir, freeze, rd, pr = ctx(a, "server")
    stage = f"stage{a.stage}"
    _gate_or_exit(rd, freeze, stage)
    caps = freeze["adapter"]["info"]["capabilities"]
    if a.stage == 5:
        if a.block not in ("split_A", "split_B"):
            raise UsageError("serve --stage 5 needs --block split_A or split_B")
        check_local_order(rd, a.block, "server", caps)
    if a.stage == 4 and "segment_step" in caps:
        raise UsageError("Stage 4 serves only for the in-path trace (step_trace without segment_step)")
    import gs_memory as M
    import gs_profile as PR
    preflight(freeze, gs_dir, "server", a)
    devices, uuids = placement(freeze, "server")
    if a.dev and a.device:
        devices = [a.device]
    ad = make_adapter(freeze, "server", gs_dir, a)
    stop, done, lock = threading.Event(), threading.Event(), threading.Lock()
    stamp = C.utc_ms if a.stage == 5 else C.utc          # Stages 2-4: second resolution, as coarse as the exit log
    state = {"started_utc": stamp(), "failure": None, "signals": 0, "written": False}
    mon = M.ServerMonitor(devices) if a.stage == 2 else None
    before = M.snapshot(devices) if a.stage == 2 else PR.contention(uuids)

    def finalize(forced=False):
        with lock:
            if state["written"]:
                return
            state["written"] = True
            ended = stamp()
            try:
                used = ad.server_devices()
            except Exception:
                used = devices
            prov = R.provenance(freeze, stage, "server", used, pr)
            note = "serve() did not return within the grace period after the stop signal" if forced else None
            if a.stage == 2:
                after = M.snapshot(devices)
                fb = M.foreign_processes(before["smi"], uuids)
                fa = M.foreign_processes(after["smi"], uuids)
                ooms = M.oom_events(before, after, mon.peak if mon else None)
                rec = {"what": "split", "role": "server", "devices": list(used), "before": before, "after": after, "foreign_before": fb,
                       "foreign_after": fa, "failure": state["failure"], "ooms": ooms, "server_peaks": mon.peak if mon else {},
                       "status": RU.status_of(state["failure"], fb, fa, _unified(freeze, "server"), ooms),
                       "started_utc": state["started_utc"], "ended_utc": ended, "stop_note": note, "provenance": prov}
                R.write(R.stage_dir(rd, "stage2") / "server" / "split_memory.json", rec, timing_allowed=False)
                print("server split memory:", rec["status"], flush=True)
            else:
                path = {3: R.stage_dir(rd, "stage3") / "split" / "server_serve.json", 4: R.stage_dir(rd, "stage4") / "edge" / "server_serve.json",
                        5: R.stage_dir(rd, "stage5") / (a.block or "") / "server_meta.json"}[a.stage]
                rec = {"stage": stage, "block": a.block, "fresh_start": True, "started_utc": state["started_utc"], "ended_utc": ended,
                       "contention_before": before, "contention_after": PR.contention(uuids), "failure": state["failure"], "stop_note": note,
                       "provenance": prov}
                R.write(path, rec, timing_allowed=a.stage == 5)
                print(f"server record written: {path}", flush=True)
            R.exit_log(rd, f"serve stage{a.stage} {a.block or ''} record written (forced={forced})")

    def on_signal(signum, frame):
        state["signals"] += 1
        stop.set()
        if state["signals"] == 1:
            threading.Timer(S.SERVE_STOP_GRACE_S, lambda: (None if done.is_set() else (finalize(forced=True), end_forced(a), os._exit(0)))).start()
        else:
            finalize(forced=True)
            end_forced(a)
            os._exit(0)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    def ready():
        print("READY", flush=True)
        if a.ready_file:
            C.write_text(a.ready_file, C.utc() + "\n", overwrite=True)

    R.exit_log(rd, f"serve stage{a.stage} {a.block or ''} start")
    blocked = a.stage == 2 and bool(M.foreign_processes(before["smi"], uuids))
    if not blocked:
        mark_run(a)
    try:
        if blocked:
            state["failure"] = "not run: foreign compute process on an experiment device"
        elif mon is not None:
            with mon:
                ad.serve(ready=ready, stop=stop)
        else:
            ad.serve(ready=ready, stop=stop)
    except Exception as e:      # an allocation failure while loading or serving is a fact to record
        import traceback
        state["failure"] = f"{type(e).__name__}: {e}\n" + traceback.format_exc()[-4000:]
    finally:
        done.set()
        finalize()


def cmd_memory(a):
    if a.what == "split" and a.role != "edge":
        raise UsageError("the server's split memory is recorded by `serve --stage 2`")
    gs_dir, freeze, rd, pr = ctx(a, a.role)
    import gs_memory as M
    it = _probe_item(freeze, gs_dir)
    eos = freeze["workload"]["eos_token_ids"]
    devices, uuids = placement(freeze, a.role)
    preflight(freeze, gs_dir, a.role, a, loads_deployment=a.what == "split")
    if a.what == "split":
        ad = make_adapter(freeze, "edge", gs_dir, a)
        ign = it["ignore_eos"] and "ignore_eos" in ad.capabilities()

        def body():
            ad.connect()
            try:
                check_devices(freeze, "edge", ad.edge_devices())
                g = ad.greedy(it["ids"], it["gen_tokens"], eos, ign)
            finally:
                AD.safe_close(ad)
            return {"probe_item": it["id"], "probe_total_tokens": it["n"] + it["gen_tokens"], "completed_tokens": len(g["tokens"]),
                    "stopped_on_eos": bool(g["stopped_on_eos"])}
    else:
        import gs_correctness as CO
        import gs_reference as REF
        device, model_dir = ref_device(freeze, a.role, a), ref_model_dir(freeze, a.role, a, gs_dir)
        devices = [device]

        def body():
            model, info = REF.load_model(model_dir, CO.reference_dtype(freeze), device, freeze["split"]["capture"]["model"].get("attn_implementation"))
            g = REF.greedy(model, it["ids"], it["gen_tokens"], eos, it["ignore_eos"], device)
            for _ in REF.teacher_forced(model, it["ids"] + g["tokens"], it["n"] - 1, device):
                pass
            return {"probe_item": it["id"], "probe_total_tokens": it["n"] + it["gen_tokens"], "completed_tokens": len(g["tokens"]), "model": info}
    mark_run(a)
    rec = M.run_probe(a.what, a.role, devices, uuids, body, unified=_unified(freeze, a.role))
    rec["provenance"] = R.provenance(freeze, "stage2", a.role, devices, pr)
    R.write(R.stage_dir(rd, "stage2") / a.role / f"{a.what}_memory.json", rec, timing_allowed=False)
    R.exit_log(rd, f"memory {a.what} {a.role} -> {rec['status']}")
    print(a.what, a.role, rec["status"], (rec.get("failure") or "")[:300])


def cmd_reference(a):
    gs_dir, freeze, rd, pr = ctx(a, a.role)
    _gate_or_exit(rd, freeze, "stage3")
    preflight(freeze, gs_dir, a.role, a, loads_deployment=False)
    import gs_correctness as CO
    device, model_dir = ref_device(freeze, a.role, a), ref_model_dir(freeze, a.role, a, gs_dir)
    mark_run(a)
    (CO.run_reference_server if a.role == "server" else CO.run_reference_edge)(freeze, gs_dir, device, model_dir, pr)
    R.exit_log(rd, f"reference {a.role} done")


def cmd_correctness(a):
    role = "edge" if a.what == "split" else "server"
    gs_dir, freeze, rd, pr = ctx(a, role)
    _gate_or_exit(rd, freeze, "stage3")
    preflight(freeze, gs_dir, role, a)
    import gs_correctness as CO
    ad = make_adapter(freeze, role, gs_dir, a)
    (CO.run_split if a.what == "split" else CO.run_pipeline)(freeze, gs_dir, ad, pr, check=lambda used: (check_devices(freeze, role, used),
                                                                                                         mark_run(a)))
    R.exit_log(rd, f"correctness {a.what} done")


def cmd_components(a):
    gs_dir, freeze, rd, pr = ctx(a, a.role)
    _gate_or_exit(rd, freeze, "stage4")
    preflight(freeze, gs_dir, a.role, a)
    import gs_profile as PR
    ad = make_adapter(freeze, a.role, gs_dir, a)
    PR.run_components(freeze, gs_dir, ad, a.role, placement(freeze, a.role)[1], pr,
                      check=lambda used: (check_devices(freeze, a.role, used), mark_run(a)))
    R.exit_log(rd, f"components {a.role} done")


def cmd_e2e(a):
    role = S.BLOCK_HOST[a.block]
    gs_dir, freeze, rd, pr = ctx(a, role)
    _gate_or_exit(rd, freeze, "stage5")
    caps = freeze["adapter"]["info"]["capabilities"]
    check_local_order(rd, a.block, role, caps)
    import gs_profile as PR
    reference_block = a.block in ("edge_only", "server_only")
    preflight(freeze, gs_dir, role, a, loads_deployment=not reference_block)
    uuids = placement(freeze, role)[1]
    close = None
    if reference_block:      # the primary comparator: always the reference runtime (PREREG §11.2)
        import gs_correctness as CO
        import gs_reference as REF
        device = ref_device(freeze, role, a)
        model, _ = REF.load_model(ref_model_dir(freeze, role, a, gs_dir), CO.reference_dtype(freeze), device,
                                  freeze["split"]["capture"]["model"].get("attn_implementation"))
        runner, devices, label = REF.HFRunner(model, device), [device], "reference runtime (HF transformers), unsplit, one host"
    else:
        ad = make_adapter(freeze, role, gs_dir, a)
        if a.block.startswith("split"):
            ad.connect()
            runner, devices, label, close = ad, ad.edge_devices(), "deployment split (edge <-> server over the lab link)", lambda: AD.safe_close(ad)
        else:
            runner = ad.unsplit_runner()
            devices, label = (ad.edge_devices() if role == "edge" else ad.server_devices()), "deployment runtime, unsplit, one host"
        check_devices(freeze, role, devices)
    mark_run(a)
    try:
        PR.run_block(freeze, gs_dir, a.block, runner, role, devices, uuids, label, "ignore_eos" in caps, pr)
    finally:
        if close:
            close()
    R.exit_log(rd, f"e2e {a.block} done")


def cmd_gate(a):
    gs_dir = Path(a.gs_dir).resolve()
    if not a.dev and gs_dir != S.GS_DIR:
        raise UsageError(f"--gs-dir must be {S.GS_DIR} in a target run")
    try:
        freeze = F.check(a.freeze, gs_dir=gs_dir, dev=a.dev)
    except F.FreezeMismatch as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)
    rd = R.run_dir(gs_dir, freeze)
    fn = {2: G.gate_stage2, 3: G.gate_stage3, 4: G.gate_stage4, 5: G.gate_stage5}[a.stage]
    try:
        g = fn(rd, freeze, dev=a.dev)
    except G.GateClosed as e:
        print(f"GATE CLOSED: {e}", file=sys.stderr)
        sys.exit(3)
    except G.GateNotReady as e:
        print(f"NOT READY (no gate written): {e}", file=sys.stderr)
        sys.exit(4)
    print(f"stage{a.stage}:", g["verdict"] + (" (final)" if g.get("final") else ""), *(g.get("reasons") or []), sep="\n  - ")
    if g.get("repeatable"):
        print("  repeatable parts:", [(r["part"], r["cause"]) for r in g["repeatable"]])
    sys.exit(0 if g["verdict"] in OPEN else 3)


def cmd_invalidate(a):
    gs_dir = Path(a.gs_dir).resolve()
    if not a.dev and gs_dir != S.GS_DIR:
        raise UsageError(f"--gs-dir must be {S.GS_DIR} in a target run")
    try:
        freeze = F.check(a.freeze, gs_dir=gs_dir, dev=a.dev)
    except F.FreezeMismatch as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)
    rd = R.run_dir(gs_dir, freeze)
    stage = f"stage{a.stage}"
    parts = list(a.part or [])
    if a.all:
        g = G.read_gate(rd, stage) or {}
        parts = [r["part"] for r in g.get("repeatable", [])]
    try:
        dest = G.invalidate(rd, freeze, stage, parts, a.note)
    except ValueError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        sys.exit(3)
    print("moved to", dest)


def cmd_retire(a):
    """On a lab host, after copying the analysis host's runs/<id>/<stage>/_invalidated/LOG.jsonl here: move this host's
    copies of the invalidated parts aside (no new log entry), so the repeat is not blocked and no old file comes back."""
    gs_dir, freeze, rd, pr = ctx(a, None)
    print("retired:", G.retire(rd, f"stage{a.stage}") or "nothing to move")


def cmd_verify_checkpoint(a):
    """Stage 6: full re-hash of every checkpoint directory this host used, against its Stage-0 manifest (PREREG §15)."""
    gs_dir, freeze, rd, pr = ctx(a, a.role)
    import gs_checkpoint as CK
    names = ["checkpoint.json"] + (["checkpoint_reference.json"] if a.role == "edge" and freeze["checkpoint"].get("edge_reference") else [])
    checks = []
    for n in names:
        cap = C.read_json(gs_dir / "capture" / a.role / n)
        md = a.model_dir if (a.dev and a.model_dir) else cap["model_dir"]
        checks.append({"capture": n, "model_dir": md, "manifest_sha256": cap["manifest_sha256"], "identical": CK.deep_check(cap, md)})
    out = {"role": a.role, "checks": checks, "identical": all(c["identical"] for c in checks),
           "provenance": R.provenance(freeze, "final", a.role, [], pr)}
    R.write(R.stage_dir(rd, "final") / f"checkpoint_verify_{a.role}.json", out, timing_allowed=False)
    print(a.role, "checkpoint identical to the Stage-0 manifest" if out["identical"] else "CHECKPOINT CHANGED: every stage is INVALID")
    sys.exit(0 if out["identical"] else 2)


def cmd_outcome(a):
    gs_dir = Path(a.gs_dir).resolve()
    if not a.dev and gs_dir != S.GS_DIR:
        raise UsageError(f"--gs-dir must be {S.GS_DIR} in a target run")
    freeze = F.check(a.freeze, gs_dir=gs_dir, dev=a.dev)
    rd = R.run_dir(gs_dir, freeze)
    res = G.outcome(rd, freeze, history=F.history(gs_dir, freeze))
    if res["outcome"]:
        C.write_json(R.stage_dir(rd, "final") / "outcome.json", res)
    print(C.canonical({k: v for k, v in res.items() if k != "history"}))


def stage_part(a):
    """(stage, part) a command writes, for the exit-status log (gs_results.crashed)."""
    c = a.cmd
    if c == "memory":
        return "stage2", a.role
    if c == "serve":
        return f"stage{a.stage}", {2: "server", 3: "split", 4: "edge", 5: a.block}[a.stage]
    if c == "reference":
        return "stage3", f"reference_{a.role}"
    if c == "correctness":
        return "stage3", a.what
    if c in ("components", "verify-workload", "verify-checkpoint"):
        return {"components": "stage4", "verify-workload": "stage1", "verify-checkpoint": "final"}[c], a.role
    if c == "e2e":
        return "stage5", a.block
    return (f"stage{a.stage}" if getattr(a, "stage", None) else None), None


def logged(fn, a):
    """Run a command; log START and END (exit status, crashes included) to runs/<run_id>/<host>.exit_status.log."""
    rd, token, caps = None, f"{os.getpid()}-{int(time.time() * 1000)}", []
    stage, part = stage_part(a)
    label = a.cmd + (f":{a.what}" if a.cmd == "memory" else "")     # the writer of the file (gs_results.writer)
    try:
        fr = F.load(a.freeze)
        rd, caps = R.run_dir(Path(a.gs_dir).resolve(), fr), fr["adapter"]["info"]["capabilities"]
        R.log_event(rd, "START", token, stage, part, label)
        a.log_ctx = {"rd": rd, "token": token, "stage": stage, "part": part, "cmd": label}
    except Exception:
        rd = None
    code = "crash"
    try:
        fn(a)
        code = "0"
    except SystemExit as e:
        code = str(e.code if isinstance(e.code, int) else (0 if e.code is None else 1))
        raise
    except BaseException as e:
        if rd is not None:
            R.exit_log(rd, f"CRASH token={token} {type(e).__name__}: {str(e)[:500]}")
        raise
    finally:
        if rd is not None:
            out = None
            if code == "0" and label in R.WRITER_CMDS:
                try:
                    out = R.outputs_digest(rd, stage, part, label, caps)
                except Exception:
                    out = None
            R.log_event(rd, "END", token, stage, part, label, rc=code, out=out)


def main(argv=None):
    import argparse
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in DELEGATED:
        mod, sub = DELEGATED[argv[0]]
        return __import__(mod).main([sub] + argv[1:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("stage0")
    p.add_argument("--capture-dir", required=True)
    p.set_defaults(fn=cmd_stage0)
    p = sp.add_parser("snapshot")
    p.add_argument("--role", required=True, choices=S.ROLES)
    p.add_argument("--dest", required=True)
    p.add_argument("--gs-dir", default=str(S.GS_DIR))
    p.set_defaults(fn=cmd_snapshot)
    p = sp.add_parser("adapter-check")
    p.add_argument("--adapter", required=True)
    p.add_argument("--gs-dir", default=str(S.GS_DIR))
    p.add_argument("--dev", action="store_true")
    p.set_defaults(fn=cmd_adapter_check)
    p = sp.add_parser("adapter-smoke")
    p.add_argument("--adapter", required=True)
    p.add_argument("--role", required=True, choices=S.ROLES)
    p.add_argument("--gs-dir", default=str(S.GS_DIR))
    p.add_argument("--endpoint", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--synthetic-ids", action="store_true", help="use ids 2..17 instead of the first natural prompt")
    p.add_argument("--dev", action="store_true", help="tests: a test double, no git checks")
    p.set_defaults(fn=cmd_adapter_smoke)
    p = sp.add_parser("verify-workload")
    _common(p)
    p.add_argument("--role", required=True, choices=S.ROLES)
    p.set_defaults(fn=cmd_verify_workload)
    p = sp.add_parser("serve")
    _common(p)
    p.add_argument("--stage", type=int, required=True, choices=(2, 3, 4, 5))
    p.add_argument("--block", default=None)
    p.add_argument("--ready-file", default=None)
    p.set_defaults(fn=cmd_serve)
    p = sp.add_parser("memory")
    _common(p)
    p.add_argument("--what", required=True, choices=("split", "reference"))
    p.add_argument("--role", required=True, choices=S.ROLES)
    p.set_defaults(fn=cmd_memory)
    p = sp.add_parser("reference")
    _common(p)
    p.add_argument("--role", required=True, choices=S.ROLES)
    p.set_defaults(fn=cmd_reference)
    p = sp.add_parser("correctness")
    _common(p)
    p.add_argument("--what", required=True, choices=("split", "pipeline"))
    p.set_defaults(fn=cmd_correctness)
    p = sp.add_parser("components")
    _common(p)
    p.add_argument("--role", required=True, choices=S.ROLES)
    p.set_defaults(fn=cmd_components)
    p = sp.add_parser("e2e")
    _common(p)
    p.add_argument("--block", required=True, choices=S.BLOCK_ORDER + S.NATIVE_BLOCKS)
    p.set_defaults(fn=cmd_e2e)
    p = sp.add_parser("gate")
    _common(p)
    p.add_argument("--stage", type=int, required=True, choices=(2, 3, 4, 5))
    p.set_defaults(fn=cmd_gate)
    p = sp.add_parser("invalidate")
    _common(p)
    p.add_argument("--stage", type=int, required=True, choices=(2, 3, 4, 5))
    p.add_argument("--part", action="append", help="a part the stage's gate lists as repeatable (repeat the flag for several)")
    p.add_argument("--all", action="store_true", help="every part the gate lists as repeatable")
    p.add_argument("--note", required=True, help="operator note; the cause is taken from the gate")
    p.set_defaults(fn=cmd_invalidate)
    p = sp.add_parser("retire")
    _common(p)
    p.add_argument("--stage", type=int, required=True, choices=(2, 3, 4, 5))
    p.set_defaults(fn=cmd_retire)
    p = sp.add_parser("verify-checkpoint")
    _common(p)
    p.add_argument("--role", required=True, choices=S.ROLES)
    p.set_defaults(fn=cmd_verify_checkpoint)
    p = sp.add_parser("outcome")
    _common(p)
    p.set_defaults(fn=cmd_outcome)
    a = ap.parse_args(argv)
    if getattr(a, "freeze", None):
        logged(a.fn, a)
    else:
        a.fn(a)


if __name__ == "__main__":
    main()
