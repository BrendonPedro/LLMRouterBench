"""GEMMA-SPLIT freeze record: writer and checker (PREREG §6). A DECISION file.

Order: tag gemma-split-prereg-v1 (protocol + harness) -> Stage 0 capture -> commit A (capture/ and adapter/ only)
-> `freeze` writes frozen/freeze_record.json from commit A -> commit B (frozen/ only) -> every later command runs
`check` first and refuses (exit 2, nothing written) on any mismatch.

A record is never edited, renamed, deleted or overwritten, and records have fixed names; every record-shaped JSON in
frozen/ counts as a record and git history is checked for removed evidence. A NEW record (frozen/freeze_record.rN.json,
at most MAX_RECORD_REVISIONS) is accepted only if no run of the chain is final, every stage with outputs is gated, and the
previous run produced nothing, ended INVALID for a cause in SUPERSEDE_CAUSES, or a command of the next stage crashed
while measuring before that stage produced output (committed crash evidence) with only orchestration files changed
(supersede_problems). A new record counts as a repeat of its causes; repeat caps count over the chain. Exit-status and
invalidation logs are append-only in git history. The superseded record can
no longer drive a command; its run stays in runs/ and in the archive. Decision files (gs_spec.DECISION_FILES, measurement
drivers included) equal the tag in every record; orchestration files, tests and the adapter may differ from the tag only
with an exact entry in frozen/HARNESS_FIXES.md. Dev records (tests, dry runs) are a separate kind: they need a
test-double adapter and can never drive a target run.

freeze_sha256 = SHA-256 of the canonical record without freeze_sha256 and run_id; run_id = its first 12 hex digits.
"""
import os
import re
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # python -I: the script dir is not on sys.path
import gs_adapter as AD  # noqa: E402
import gs_common as C  # noqa: E402
import gs_spec as S  # noqa: E402

EXCLUDED_PREFIXES = ("runs/", "results/", "frozen/")    # outputs and records: not inputs of the experiment
EXCLUDED_PARTS = ("__pycache__",)
POST_TAG_DIRS = ("capture/", "adapter/")                # written after the tag (Stage 0, Stage 1)
POST_FREEZE_DIRS = ("frozen/", "runs/", "results/")     # may change after the freeze commit
FIXES_FILE = "frozen/HARNESS_FIXES.md"                  # lines "<path> | <fix commit> | <regression test> | <reason>"
STRICT_VERSION_KEYS = ("python", "torch", "torch_cuda", "transformers", "accelerate", "safetensors", "tokenizers", "flash_attn", "vllm",
                       "driver")
EXEC_ENV_KEYS = ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "PYTORCH_CUDA_ALLOC_CONF", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "NVIDIA_TF32_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG", "TOKENIZERS_PARALLELISM", "LD_PRELOAD", "LD_LIBRARY_PATH",
                 "PYTHONPATH", "PYTHONHOME")
EXEC_ENV_PREFIXES = ("TORCH_", "NCCL_", "TRANSFORMERS_", "VLLM_", "HF_HUB_")
RECORD_RE = re.compile(r"^freeze_record(?:\.r(\d+))?\.json$")


class FreezeMismatch(RuntimeError):
    pass


def repo_of(gs_dir):
    return Path(gs_dir).resolve().parents[1]


def rel_dir(gs_dir):
    return Path(gs_dir).resolve().relative_to(repo_of(gs_dir)).as_posix()


def hash_set(gs_dir):
    """Every file of the experiment directory except outputs, records and bytecode: any stray file is 'not covered'."""
    gs_dir = Path(gs_dir)
    out = {}
    for p in sorted(gs_dir.rglob("*")):
        rel = p.relative_to(gs_dir).as_posix()
        if p.is_file() and not rel.startswith(EXCLUDED_PREFIXES) and not any(x in p.parts for x in EXCLUDED_PARTS) and p.suffix != ".pyc":
            out[rel] = C.sha256_file(p)
    return out


def tag_commit(gs_dir, tag=S.PREREG_TAG):
    return C.git(["rev-parse", "--verify", "--quiet", f"{tag}^{{commit}}"], repo_of(gs_dir))


def tag_file_sha256(gs_dir, tag, relpath):
    import subprocess
    try:
        r = subprocess.run(["git", "show", f"{tag}:{rel_dir(gs_dir)}/{relpath}"], cwd=repo_of(gs_dir), capture_output=True, timeout=60)
    except Exception:
        return None
    return C.sha256_bytes(r.stdout) if r.returncode == 0 else None


def tag_files(gs_dir, tag):
    out = C.git(["ls-tree", "-r", "--name-only", tag, "--", rel_dir(gs_dir)], repo_of(gs_dir)) or ""
    pre = rel_dir(gs_dir) + "/"
    return [l[len(pre):] for l in out.splitlines() if l.startswith(pre)]


def changes_since_tag(gs_dir, tag=S.PREREG_TAG):
    """Harness / protocol files whose working-tree content differs from the tag (post-tag dirs excluded), plus harness
    files added after the tag outside the post-tag dirs."""
    gs_dir = Path(gs_dir)
    in_tag = set(tag_files(gs_dir, tag))
    changed = []
    for f in sorted(in_tag | set(hash_set(gs_dir))):
        if f.startswith(POST_TAG_DIRS + POST_FREEZE_DIRS):
            continue
        p = gs_dir / f
        if f not in in_tag or not p.exists() or C.sha256_file(p) != tag_file_sha256(gs_dir, tag, f):
            changed.append(f)
    return changed


def documented_fixes(gs_dir):
    p = Path(gs_dir) / FIXES_FILE
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        f = [x.strip() for x in line.split("|")]
        if len(f) >= 4 and f[0] and not f[0].startswith("#") and all(f[1:4]):
            out[f[0]] = {"commit": f[1], "test": f[2], "reason": f[3]}
    return out


def tree_problems(gs_dir):
    """Bytecode (it runs instead of the source even under -B) and symlinks (rglob does not follow them, so their targets
    would escape the hash set) are refused anywhere in the experiment directory."""
    P = []
    for dirpath, dirnames, filenames in os.walk(gs_dir, followlinks=False):
        for n in dirnames + filenames:
            q = Path(dirpath) / n
            rel = q.relative_to(gs_dir).as_posix()
            if q.is_symlink():
                P.append(f"symlink in the experiment directory: {rel}")
            elif n == "__pycache__" or n.endswith((".pyc", ".pyo")):
                P.append(f"bytecode in the experiment directory: {rel} (delete it; run python with -B)")
    return P


def record_shaped(gs_dir):
    """Every JSON in frozen/ that looks like a freeze record, whatever its name: [(path, record)]."""
    out = []
    for q in sorted((Path(gs_dir) / "frozen").glob("*.json")):
        try:
            rec = C.read_json(q)
        except Exception:
            continue
        if isinstance(rec, dict) and rec.get("experiment") == S.NAME and "freeze_sha256" in rec:
            out.append((q, rec))
    return out


def expected_record_name(revision):
    return "freeze_record.json" if revision == 1 else f"freeze_record.r{revision}.json"


def evidence_problems(gs_dir):
    """Records and run outputs are append-only evidence: a record under another name, a tracked record or output missing
    from the working tree, or a deletion / rename in git history (other than a move into _invalidated/) is refused."""
    gs_dir = Path(gs_dir)
    P = [f"record-shaped file with a non-standard name: frozen/{q.name}" for q, r in record_shaped(gs_dir)
         if not RECORD_RE.match(q.name) or q.name != expected_record_name(r.get("revision", 1))]
    repo, rel = repo_of(gs_dir), rel_dir(gs_dir)
    if C.git(["rev-parse", "--git-dir"], repo) is None:
        return P
    tracked = (C.git(["ls-files", "--", f"{rel}/frozen", f"{rel}/runs"], repo) or "").splitlines()
    P += [f"tracked evidence missing from the working tree: {t}" for t in tracked if t and not (repo / t).exists()]
    hist = C.git(["log", "--all", "-M", "--name-status", "--format=", "--", f"{rel}/frozen", f"{rel}/runs"], repo) or ""
    for line in hist.splitlines():
        f = line.split("\t")
        if not f or not f[0]:
            continue
        if f[0] == "D":
            P.append(f"deleted from git history: {f[1]}")
        elif f[0].startswith("R") and len(f) == 3 and ("/_invalidated/" not in f[2] or f[1].startswith(f"{rel}/frozen/")):
            P.append(f"renamed in git history: {f[1]} -> {f[2]}")
    for t in tracked:          # exit-status and invalidation logs are append-only: each version extends the previous one
        if not t.endswith((".exit_status.log", "/_invalidated/LOG.jsonl")):
            continue
        prev = ""
        for h in (C.git(["log", "--first-parent", "--reverse", "--format=%H", "HEAD", "--", t], repo) or "").split():
            cur = C.git(["show", f"{h}:{t}"], repo)
            if cur is not None and not cur.startswith(prev):
                P.append(f"append-only log rewritten in git history: {t}")
                break
            prev = cur if cur is not None else prev
        else:
            if (repo / t).exists() and not (repo / t).read_text(encoding="utf-8", errors="replace").strip().startswith(prev):
                P.append(f"append-only log rewritten in the working tree: {t}")
    return sorted(set(P))


def had_outputs(gs_dir, run_id):
    """A run produced outputs if its directory has any stage output now or ever had committed files."""
    gs_dir = Path(gs_dir)
    rd = gs_dir / "runs" / run_id
    now = any(q.is_file() for st in S.STAGE_ORDER for q in (rd / S.STAGE_DIRS[st]).rglob("*")) if rd.exists() else False
    hist = C.git(["log", "--all", "--format=%H", "-n", "1", "--", f"{rel_dir(gs_dir)}/runs/{run_id}"], repo_of(gs_dir))
    return now or bool(hist)


def capabilities_declared_first(gs_dir):
    """capture/capabilities.json must have been committed before any adapter file existed, and unchanged since (PREREG §5.3)."""
    repo, rel = repo_of(gs_dir), rel_dir(gs_dir)
    first = (C.git(["log", "--diff-filter=A", "--format=%H", "--", f"{rel}/capture/capabilities.json"], repo) or "").splitlines()
    if not first:
        return ["capture/capabilities.json is not committed: commit the declaration before writing the adapter"]
    c0 = first[-1]
    P = []
    early = [l for l in (C.git(["ls-tree", "-r", "--name-only", c0, "--", f"{rel}/adapter"], repo) or "").splitlines()
             if not l.endswith("/README.md")]
    if early:
        P.append(f"capture/capabilities.json was first committed together with or after adapter files {early[:3]}")
    if C.run(["git", "diff", "--quiet", c0, "--", f"{rel}/capture/capabilities.json"], cwd=str(repo))["rc"] != 0:
        P.append("capture/capabilities.json changed after it was first committed")
    return P


def protocol_params():
    keys = ("CORR_PER_DATASET", "CORR_SEED", "CORR_GEN_TOKENS", "COVERAGE_ITEMS", "COVERAGE_PROMPT_TOKENS", "COVERAGE_GEN_TOKENS",
            "TIMING_SEED", "TIMING_GROUPS", "TIMING_GEN_TOKENS", "TIMING_WARMUP", "TIMING_MEASURED", "CORR_TOPK", "TF_D_MARGIN_PP",
            "TF_KL_FACTOR", "TF_KL_EPS", "GEN_MARGIN_NATS", "GEN_MAX_UNEXPLAINED", "DEGEN_WINDOW", "DEGEN_MAX_PERIOD",
            "COMPONENT_REQUESTS", "COMPONENT_WARMUP", "BLOCK_ORDER", "NATIVE_BLOCKS", "DRIFT_MAX_REL", "CLOSURE_MIN_RATIO",
            "CLOCK_SKEW_TOLERANCE_S", "BOOT_SEED", "BOOT_N", "CI_LEVEL", "OOM_DEVICE_PATTERNS", "OOM_HOST_PATTERNS",
            "FOREIGN_PROCESSES_ALLOWED", "REPEAT_CAPS", "DECISION_FILES")
    return {k: getattr(S, k) for k in keys}


def freeze_sha256(rec):
    return C.sha256_obj({k: v for k, v in rec.items() if k not in ("freeze_sha256", "run_id")})


def records(gs_dir, dev=None):
    """Freeze records (every record-shaped JSON in frozen/, whatever its name), oldest first by revision; dev=None: all."""
    out = [(int(r.get("revision", 1)), q, r) for q, r in record_shaped(gs_dir) if dev is None or bool(r.get("dev")) == bool(dev)]
    return [(q, r) for _, q, r in sorted(out, key=lambda x: (x[0], x[1].name))]


def run_status(gs_dir, rec):
    """Gates and outputs of a record's run (for the supersede rule)."""
    import gs_gate as G
    import gs_results as R
    rd = R.run_dir(gs_dir, rec)
    st = {"has_outputs": had_outputs(gs_dir, rec["run_id"]), "ungated": [], "gates": {}, "final": False, "latest": None,
          "latest_gate": None, "next_stage_has_outputs": False}
    for stage in S.STAGE_ORDER:
        sd = R.stage_dir(rd, stage)
        has = sd.exists() and any(x.name not in ("_invalidated",) for x in sd.iterdir())
        g = G.read_gate(rd, stage)
        if has and g is None:
            st["ungated"].append(stage)
        if g:
            st["gates"][stage] = g["verdict"]
            st["final"] |= bool(g.get("final"))
            st["latest"] = [stage, g["verdict"]]
            st["latest_gate"] = g
    if st["latest"]:
        i = S.STAGE_ORDER.index(st["latest"][0])
        nxt = S.STAGE_ORDER[i + 1] if i + 1 < len(S.STAGE_ORDER) else None
        st["next_stage_has_outputs"] = bool(nxt) and R.stage_dir(rd, nxt).exists() and any(R.stage_dir(rd, nxt).iterdir())
    return st


ORCHESTRATION_PATTERNS = tuple(S.ORCHESTRATION_FILES) + ("tests/",)


def committed_unchanged(gs_dir, path):
    repo = repo_of(gs_dir)
    rel = Path(path).resolve().relative_to(repo).as_posix()
    return C.git(["ls-files", "--error-unmatch", rel], repo) is not None and \
        C.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=str(repo))["rc"] == 0


def crash_basis(gs_dir, rec, stage):
    """Basis (c) evidence: (problems, counts_as_repeat). Both frozen hosts' exit-status logs, committed unchanged, must show
    a writer command of `stage` that crashed while measuring, and none that completed (PREREG §6.4)."""
    import gs_results as R
    rd = R.run_dir(gs_dir, rec)
    P = []
    for role in S.ROLES:
        log = rd / f"{rec['roles'][role]['host']}.exit_status.log"
        if not log.exists() or not committed_unchanged(gs_dir, log):
            P.append(f"{log.name}: the {role} host's exit-status log must be copied here and committed unchanged first")
    runs = R.stage_runs(R.read_events(rd), stage)
    done = [f"{part}:{cmd}" for (part, cmd), (ok, _) in runs.items() if ok]
    failed = sorted({part for (part, cmd), (ok, bad) in runs.items() if bad and not ok})
    if done:
        P.append(f"the exit-status logs show completed {stage} commands {done[:5]}: their outputs exist (or were deleted); no new record")
    if not failed:
        P.append(f"no {stage} command crashed while measuring in the committed exit-status logs: nothing justifies a new record")
    return P, {"stage": stage, "entries": [{"part": part, "cause": "invalid"} for part in failed]}


def supersede_problems(gs_dir, prev, new_hashes):
    """The rule for a new record (PREREG §6.4). prev: [(path, record)] of this kind, oldest first.
    Returns (problems, basis, counts_as_repeat): a new record counts as a repeat of the causes that justified it."""
    P = []
    if len(prev) + 1 > S.MAX_RECORD_REVISIONS:
        P.append(f"at most {S.MAX_RECORD_REVISIONS} freeze records per experiment: the run cannot be restarted again")
    for q, r in prev:
        st = run_status(gs_dir, r)
        if st["final"]:
            P.append(f"{q.name}: its run has a final verdict {st['gates']}: the experiment is over, no new record")
    p_path, p_rec = prev[-1]
    st = run_status(gs_dir, p_rec)
    if st["ungated"]:
        P.append(f"{p_path.name}: stages {st['ungated']} have outputs but no gate: gate them before superseding the record")
    if st["gates"]:            # a gate's hash is not a signature: the basis must follow from the raw files (as the outcome does)
        import gs_gate as G
        import gs_results as R
        rd = R.run_dir(gs_dir, p_rec)
        again = G.recompute(rd, p_rec, dev=bool(p_rec.get("dev")))
        key = lambda g: (g.get("verdict"), bool(g.get("final")), bool(g.get("supersede_eligible")))  # noqa: E731
        for stage in st["gates"]:
            g = G.read_gate(rd, stage)
            P += [f"{p_path.name}: {x}" for x in G.check_gate(g, p_rec, f"{stage} gate")]
            if again.get(stage) is None or key(again[stage]) != key(g):
                P.append(f"{p_path.name}: the stored {stage} gate {key(g)} does not match a recomputation from its raw files "
                         f"{key(again[stage]) if again.get(stage) else None}")
    g = st["latest_gate"]
    if not st["has_outputs"]:
        import gs_results as R
        ev = R.read_events(R.run_dir(gs_dir, p_rec))
        done = sorted({f"{e['stage']}/{e['part']}:{e['cmd']}" for e in ev if e["cmd"] in R.WRITER_CMDS and e["kind"] == "END" and e["rc"] == "0"})
        if done:
            P.append(f"{p_path.name}: no outputs, but the exit-status logs show completed commands {done[:5]}: outputs were deleted; "
                     "no new record")
        return P, "the previous run produced no output", None
    if g and g["verdict"] == "INVALID" and g.get("supersede_eligible"):
        entries = sorted({(p["part"], p["cause"]) for p in g.get("problems", [])})
        return P, f"{st['latest'][0]} INVALID with a cause that needs a new record", \
            {"stage": st["latest"][0], "entries": [{"part": a, "cause": b} for a, b in entries]}
    i = S.STAGE_ORDER.index(st["latest"][0]) if st["latest"] else -1
    if g and g["verdict"] == S.OPEN_VERDICT.get(st["latest"][0]) and not st["next_stage_has_outputs"] and i + 1 < len(S.STAGE_ORDER):
        nxt = S.STAGE_ORDER[i + 1]
        changed = sorted(f for f in set(new_hashes) | set(p_rec["file_sha256"]) if new_hashes.get(f) != p_rec["file_sha256"].get(f))
        bad = [f for f in changed if not f.startswith(ORCHESTRATION_PATTERNS)]
        if bad:
            P.append(f"{p_path.name}: the run stopped before {nxt} produced output, so only orchestration files "
                     f"({', '.join(S.ORCHESTRATION_FILES)}) or tests may change; changed: {bad[:5]}")
        P_c, counts = crash_basis(gs_dir, p_rec, nxt)
        return P + [f"{p_path.name}: {p}" for p in P_c], f"{nxt} could not run (orchestration defect); only orchestration files changed", counts
    P.append(f"{p_path.name}: a new record is allowed only when the previous run produced nothing, ended INVALID with a cause in "
             f"{S.SUPERSEDE_CAUSES}, or crashed before a stage produced output (latest gate {st['latest']})")
    return P, None, None


def build(gs_dir, adapter_rel, dev=False, notes="", supersede_reason=None):
    """Build the record from capture/ (all Stage-0 files plus workload_tokens.json and <role>/snapshot.json).
    Refuses if Stage 0 did not CONTINUE, any input fails validation, or the supersede rule is not met."""
    import gs_checkpoint as CK
    import gs_split as SP
    import gs_workload as WL
    gs_dir = Path(gs_dir)
    cap = gs_dir / "capture"
    rj = lambda p: C.read_json(cap / p)  # noqa: E731
    problems = []
    st0 = rj("stage0_verdict.json")
    if st0.get("verdict") != "CONTINUE":
        problems.append(f"stage0 verdict {st0.get('verdict')}: nothing to freeze")
    roles, sel, split_cap, capd = rj("roles.json"), rj("selected_config.json"), rj("split_capture.json"), rj("capabilities.json")
    inv = {r: rj(f"{r}/inventory.json") for r in S.ROLES}
    ck = {r: rj(f"{r}/checkpoint.json") for r in S.ROLES}
    dep = {r: rj(f"{r}/deployment.json") for r in S.ROLES}
    tokens = rj("workload_tokens.json")
    snaps = {r: rj(f"{r}/snapshot.json") for r in S.ROLES if (cap / r / "snapshot.json").exists()}
    if not roles.get("verified"):
        problems.append("roles not verified")
    if sel.get("verdict") != "CONTINUE" or sel.get("selected") != split_cap.get("configuration_id"):
        problems.append(f"selected configuration {sel.get('selected')} != split capture {split_cap.get('configuration_id')}")
    P, _ = SP.validate(split_cap, ck["server"]["summary"])
    problems += [f"split capture: {p}" for p in P]
    problems += [f"workload: {p}" for p in WL.validate_tokens(tokens, gs_dir / "workload", dev=dev)]
    adapter_path = gs_dir / adapter_rel
    info = AD.read_info(adapter_path)
    problems += [f"adapter: {p}" for p in AD.check_info(info, target=not dev)]
    if dev and not info.get("test_double"):
        problems.append("a dev record accepts test-double adapters only: the deployment's adapter never runs in dev mode (PREREG §5.4)")
    if info.get("configuration_id") != split_cap.get("configuration_id"):
        problems.append(f"adapter configuration {info.get('configuration_id')} != split capture {split_cap.get('configuration_id')}")
    if set(info.get("capabilities", [])) != set(capd.get("capabilities", [])) or \
            info.get("missing_capability_justification", {}) != capd.get("missing_capability_justification", {}):
        problems.append("adapter capabilities differ from the Stage-0 declaration capture/capabilities.json (PREREG §5.3)")
    if not dev:
        if not adapter_rel.startswith("adapter/"):
            problems.append("a target adapter must live under adapter/")
        for r in S.ROLES:
            if r not in snaps:
                problems.append(f"no source snapshot recorded for the {r} host (gs_bench.py snapshot -> capture/{r}/snapshot.json)")
        if not tag_commit(gs_dir):
            problems.append(f"tag {S.PREREG_TAG} does not exist")
    prev = records(gs_dir, dev=dev)
    supersedes = None
    problems += tree_problems(gs_dir)
    if not dev:
        problems += evidence_problems(gs_dir) + capabilities_declared_first(gs_dir)
    for r in S.ROLES:
        want = (split_cap.get("gpu_placement", {}).get(r) or {}).get("CUDA_VISIBLE_DEVICES") or None
        have = exec_env(inv[r].get("env", {})).get("CUDA_VISIBLE_DEVICES") or None
        if want != have:
            problems.append(f"{r}: the inventory was taken with CUDA_VISIBLE_DEVICES={have!r} but the deployment uses {want!r}: "
                            "collect the inventory in the deployment's environment")
    if not dev:
        roots = sorted({s["root"] for r in S.ROLES for s in dep[r]["sources"]})
        for q in sorted((gs_dir / "adapter").rglob("*")):
            if q.is_file() and q.name != "README.md":
                text = q.read_text(encoding="utf-8", errors="replace")
                problems += [f"adapter/{q.relative_to(gs_dir / 'adapter').as_posix()} names the live deployment tree {root}: use the "
                             "snapshot (PREREG §5.2)" for root in roots if root in text]
    counts, changed = None, []
    if prev:
        new_hashes = hash_set(gs_dir)
        P_sup, basis, counts = supersede_problems(gs_dir, prev, new_hashes)
        problems += P_sup
        old_hashes = prev[-1][1].get("file_sha256", {})
        changed = sorted(f for f in set(new_hashes) | set(old_hashes) if new_hashes.get(f) != old_hashes.get(f))
        fixes = documented_fixes(gs_dir)
        undocumented = [f for f in changed if f.startswith("adapter/") and f not in fixes]
        if undocumented:
            problems.append(f"adapter files changed since {prev[-1][0].name} without an entry in {FIXES_FILE}: {undocumented} (an adapter "
                            "fix needs a fix commit, a regression test and a reason, PREREG §6.4)")
        if not supersede_reason:
            problems.append(f"{prev[-1][0].name} exists: a new record must state why it supersedes it (--supersede-reason)")
        st = run_status(gs_dir, prev[-1][1])
        supersedes = {"record": prev[-1][0].name, "freeze_sha256": prev[-1][1].get("freeze_sha256"), "run_id": prev[-1][1].get("run_id"),
                      "latest_gate": st["latest"], "basis": basis, "counts_as_repeat": counts, "changed_files": changed,
                      "reason": supersede_reason}
    if problems:
        raise FreezeMismatch("cannot freeze:\n  - " + "\n  - ".join(problems))
    edge_ref = rj("edge/checkpoint_reference.json") if (cap / "edge" / "checkpoint_reference.json").exists() else None
    rec = {
        "experiment": S.NAME, "record_version": S.RECORD_VERSION, "protocol_version": S.PROTOCOL_VERSION,
        "decision_rule_version": S.DECISION_RULE_VERSION, "adapter_contract_version": S.ADAPTER_CONTRACT_VERSION,
        "frozen_at_utc": C.utc(), "dev": bool(dev), "revision": len(prev) + 1, "supersedes": supersedes,
        "git": {"commit": C.git(["rev-parse", "HEAD"], repo_of(gs_dir)), "tag": S.PREREG_TAG, "tag_commit": tag_commit(gs_dir),
                "gs_dir": rel_dir(gs_dir)},
        "file_sha256": hash_set(gs_dir),
        "harness_changes_since_tag": changes_since_tag(gs_dir) if tag_commit(gs_dir) else None,
        "documented_fixes": documented_fixes(gs_dir),
        "roles": roles,
        "machines": {r: {"host": inv[r]["hostname"], "devices": roles[r]["devices"], "versions": inv[r]["versions"],
                         "exec_env": exec_env(inv[r].get("env", {})), "inventory_sha256": C.sha256_file(cap / r / "inventory.json")}
                     for r in S.ROLES},
        "checkpoint": {**{r: {"model_dir": ck[r]["model_dir"], "manifest_sha256": ck[r]["manifest_sha256"], "n_files": ck[r]["n_files"],
                              "total_bytes": ck[r]["total_bytes"], "hf_revision": ck[r]["hf_revision"]} for r in S.ROLES},
                       "edge_reference": {"model_dir": edge_ref["model_dir"], "manifest_sha256": edge_ref["manifest_sha256"]} if edge_ref else None,
                       "comparison": CK.compare(ck["edge"], ck["server"]), "summary": ck["server"]["summary"],
                       "features": ck["server"]["features"], "kv_sharing_map": ck["server"]["kv_sharing_map"],
                       "generation_config": ck["server"]["generation_config"]},
        "deployment": {r: {"sources": [{"root": s["root"], "git_commit": (s.get("git") or {}).get("commit"), "dirty": (s.get("git") or {}).get("dirty"),
                                        "manifest_sha256": s["manifest_sha256"]} for s in dep[r]["sources"]], "snapshot": snaps.get(r)}
                       for r in S.ROLES},
        "split": {"configuration_id": split_cap["configuration_id"], "capture_sha256": C.sha256_file(cap / "split_capture.json"),
                  "capture": split_cap, "derived": SP.derive(split_cap, ck["server"]["summary"]), "selection": sel},
        "adapter": {"path": adapter_rel, "sha256": C.sha256_file(adapter_path), "info": info, "tier": AD.tier(info),
                    "declared_capabilities": capd},
        "workload": {"tokens_file": "capture/workload_tokens.json", "sha256": C.sha256_file(cap / "workload_tokens.json"),
                     "ids_sha256": tokens["ids_sha256"], "max_total_tokens": tokens["max_total_tokens"], "probe_item": tokens["probe_item"],
                     "n_items": len(tokens["items"]), "bos_token_id": tokens["bos_token_id"], "eos_token_ids": tokens["eos_token_ids"]},
        "protocol": protocol_params(), "notes": notes,
    }
    rec["freeze_sha256"] = freeze_sha256(rec)
    rec["run_id"] = rec["freeze_sha256"][:12]
    return rec


def exec_env(env):
    return {k: v for k, v in sorted(env.items()) if k in EXEC_ENV_KEYS or k.startswith(EXEC_ENV_PREFIXES)}


def write(rec, path):
    """Write a record once, under the name its revision requires (frozen/freeze_record.json, frozen/freeze_record.rN.json)."""
    path = Path(path)
    want = expected_record_name(rec["revision"])
    if path.name != want or path.parent.name != "frozen":
        raise FreezeMismatch(f"revision {rec['revision']} must be written as frozen/{want}, not {path}")
    return C.write_json(path, rec)      # refuses to overwrite


def load(path):
    rec = C.read_json(path)
    if rec.get("freeze_sha256") != freeze_sha256(rec) or rec.get("run_id") != rec.get("freeze_sha256", "")[:12]:
        raise FreezeMismatch(f"{path}: freeze_sha256 / run_id do not match the record content (edited or corrupted)")
    return rec


def check(path, gs_dir=S.GS_DIR, role=None, host=None, live_versions=None, live_env=None, dev=False):
    """Verify a record against this checkout and this machine. Raises FreezeMismatch listing every problem."""
    rec = load(path)
    gs_dir = Path(gs_dir)
    P = []
    for k, want in (("experiment", S.NAME), ("protocol_version", S.PROTOCOL_VERSION), ("decision_rule_version", S.DECISION_RULE_VERSION),
                    ("adapter_contract_version", S.ADAPTER_CONTRACT_VERSION), ("record_version", S.RECORD_VERSION)):
        if rec.get(k) != want:
            P.append(f"{k} {rec.get(k)!r} != {want!r}")
    if bool(rec.get("dev")) != bool(dev):
        P.append("this is a dev record: it cannot drive a target run" if rec.get("dev") else
                 "this is a target record: dev mode cannot use it (dev runs need their own dev record and a test double)")
    newest = records(gs_dir, dev=bool(rec.get("dev")))
    if newest and Path(newest[-1][0]).resolve() != Path(path).resolve():
        P.append(f"{Path(path).name} is superseded by {newest[-1][0].name}: only the newest record may drive a command")
    if Path(path).name != expected_record_name(rec.get("revision", 1)) or Path(path).resolve().parent != (gs_dir / "frozen").resolve():
        P.append(f"a revision-{rec.get('revision')} record must be frozen/{expected_record_name(rec.get('revision', 1))}")
    P += tree_problems(gs_dir)
    if not dev:
        P += evidence_problems(gs_dir)
    live = hash_set(gs_dir)
    for f, h in rec["file_sha256"].items():
        if live.get(f) != h:
            P.append(f"file changed since freeze: {f}" if f in live else f"file missing since freeze: {f}")
    P += [f"file not covered by the freeze: {f}" for f in live if f not in rec["file_sha256"]]
    for mod in sorted(S.GS_DIR.glob("gs_*.py")):            # the harness that is EXECUTING must be the frozen one
        if rec["file_sha256"].get(mod.name) != C.sha256_file(mod):
            P.append(f"executing harness file {mod} differs from the frozen {mod.name} (run gs_bench.py from the frozen checkout)")
    P += [f"adapter: {p}" for p in AD.check_info(rec["adapter"]["info"], target=not dev)]
    if dev and not rec["adapter"]["info"].get("test_double"):
        P.append("dev mode runs test-double adapters only")
    if not dev:
        repo = repo_of(gs_dir)
        tc = tag_commit(gs_dir)
        if not tc:
            P.append(f"tag {S.PREREG_TAG} does not exist")
        else:
            if tc != rec["git"]["tag_commit"]:
                P.append(f"tag {S.PREREG_TAG} -> {tc} != recorded {rec['git']['tag_commit']} (the tag moved)")
            for f in S.DECISION_FILES:
                if not (gs_dir / f).exists() or C.sha256_file(gs_dir / f) != tag_file_sha256(gs_dir, S.PREREG_TAG, f):
                    P.append(f"{f} differs from the tag (a decision file: never allowed; a protocol change needs a new pre-registration)")
            fixes = documented_fixes(gs_dir)
            for f in changes_since_tag(gs_dir):
                if f not in S.DECISION_FILES and f not in fixes:
                    P.append(f"{f} differs from the tag and has no entry in {FIXES_FILE} (path | commit | test | reason)")
        head = C.git(["rev-parse", "HEAD"], repo)
        if head != rec["git"]["commit"]:
            diff = C.git(["diff", "--name-only", rec["git"]["commit"], "HEAD", "--", rec["git"]["gs_dir"]], repo)
            if diff is None:
                P.append(f"cannot diff the frozen commit {rec['git']['commit'][:12]} against HEAD")
            else:
                pre = rec["git"]["gs_dir"] + "/"
                bad = [l for l in diff.splitlines() if l and not l[len(pre):].startswith(POST_FREEZE_DIRS)]
                if bad:
                    P.append(f"HEAD {head[:12]} changed files outside {POST_FREEZE_DIRS} since the frozen commit: {bad[:5]}")
        relp = Path(path).resolve().relative_to(repo).as_posix() if Path(path).resolve().is_relative_to(repo) else None
        if relp is None or C.git(["ls-files", "--error-unmatch", relp], repo) is None or \
                C.run(["git", "diff", "--quiet", "HEAD", "--", relp], cwd=str(repo))["rc"] != 0:
            P.append(f"{Path(path).name} is not committed unchanged in this checkout (commit B touches only frozen/)")
    if role is not None:
        host = host or socket.gethostname()
        m = rec["machines"][role]
        if m["host"] != host:
            other = [r for r, x in rec["machines"].items() if x["host"] == host]
            P.append(f"this machine ({host}) is not the frozen {role} host ({m['host']})" + (f"; it is frozen as {other}" if other else ""))
        else:
            if live_versions is None or live_env is None:
                import gs_inventory as INV
                live_versions = INV.live_versions() if live_versions is None else live_versions
                live_env = INV.filtered_env() if live_env is None else live_env
            for k in STRICT_VERSION_KEYS:
                if m["versions"].get(k) != live_versions.get(k):
                    P.append(f"{host}: {k} {live_versions.get(k)!r} != frozen {m['versions'].get(k)!r}")
            le = exec_env(live_env)
            for k in sorted(set(le) | set(m.get("exec_env", {}))):
                if le.get(k) != m.get("exec_env", {}).get(k):
                    P.append(f"{host}: environment {k}={le.get(k)!r} != frozen {m.get('exec_env', {}).get(k)!r}")
    if P:
        raise FreezeMismatch("freeze check FAILED:\n  - " + "\n  - ".join(P))
    return rec


def history(gs_dir, rec):
    """The records this one superseded, oldest first, with the latest gate of each run (reported with the outcome)."""
    out = []
    for p, r in records(gs_dir, dev=bool(rec.get("dev"))):
        if r.get("freeze_sha256") == rec["freeze_sha256"]:
            break
        dec = Path(gs_dir) / "runs" / r.get("run_id", "-") / S.STAGE_DIRS["stage3"] / "decision.json"
        succ = next((x for _, x in records(gs_dir, dev=bool(rec.get("dev"))) if (x.get("supersedes") or {}).get("freeze_sha256") ==
                     r.get("freeze_sha256")), {})
        out.append({"record": p.name, "run_id": r.get("run_id"), "status": run_status(gs_dir, r), "git_commit": (r.get("git") or {}).get("commit"),
                    "adapter_sha256": (r.get("adapter") or {}).get("sha256"),
                    "changed_in_successor": (succ.get("supersedes") or {}).get("changed_files"),
                    "stage3_would_be": (C.read_json(dec).get("would_be") or {}).get("verdict") if dec.exists() else None,
                    "superseded_because":
                    next((x["supersedes"]["reason"] for _, x in records(gs_dir, dev=bool(rec.get("dev")))
                          if (x.get("supersedes") or {}).get("freeze_sha256") == r.get("freeze_sha256")), None)})
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("freeze")
    w.add_argument("--adapter", required=True, help="path relative to the GEMMA-SPLIT directory, e.g. adapter/gs_adapter_brian.py")
    w.add_argument("--out", required=True)
    w.add_argument("--notes", default="")
    w.add_argument("--supersede-reason", default=None, help="required for every record after the first (PREREG §6.4)")
    w.add_argument("--dev", action="store_true", help="tests / dry runs only: test-double adapters, no tag required")
    c = sub.add_parser("check-freeze")
    c.add_argument("--freeze", required=True)
    c.add_argument("--role", choices=S.ROLES, default=None)
    c.add_argument("--dev", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "freeze":
        try:
            rec = build(S.GS_DIR, a.adapter, dev=a.dev, notes=a.notes, supersede_reason=a.supersede_reason)
        except FreezeMismatch as e:
            print(str(e), file=sys.stderr)
            sys.exit(2)
        write(rec, a.out)
        print(C.canonical({"out": a.out, "freeze_sha256": rec["freeze_sha256"], "run_id": rec["run_id"], "git_commit": rec["git"]["commit"],
                           "revision": rec["revision"], "files": len(rec["file_sha256"]), "tier": rec["adapter"]["tier"]}))
    else:
        try:
            rec = check(a.freeze, role=a.role, dev=a.dev)
        except FreezeMismatch as e:
            print(str(e), file=sys.stderr)
            sys.exit(2)
        print("freeze OK", rec["freeze_sha256"], "run_id", rec["run_id"])


if __name__ == "__main__":
    main()
