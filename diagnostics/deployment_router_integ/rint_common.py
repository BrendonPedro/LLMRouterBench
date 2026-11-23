"""ROUTER-INTEG: shared helpers. Hashing, the freeze check, the site file, and access to the reused frozen runtime.

The reused runtime is QWEN-SELECT's, imported unchanged from a clean clone at its tag (rint_spec.REUSED_COMMIT). The
clone's location is the environment variable RINT_QSEL_REPO or the site file's `qsel_repo`. Nothing in that clone is
written to.
"""
import hashlib
import importlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

RI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RI_DIR))            # python -I: the script dir is not on sys.path
import rint_spec as S  # noqa: E402

FROZEN_FILES = ("PROTOCOL_ROUTER_INTEG.md", "rint_spec.py", "rint_common.py", "rint_models.py", "rint_router.py", "rint_policy.py",
                "rint_backend.py", "rint_record.py", "rint_workload.py", "rint_run.py", "rint_score.py", "rint_eval.py",
                "frozen/workload.json", "frozen/manifest_DS.json", "frozen/tensors_DS.json", "frozen/manifest_QW.json",
                "frozen/tensors_QW.json", "router/router_heads.npz", "router/fold_map.json", "router/task_lookup.json",
                "router/router_build_report.json", "tools/rint_router_build.py")
FREEZE_FILE = "frozen/FREEZE.json"
SITE_KEYS = ("role", "expected_hostname", "qsel_repo", "model_dirs", "device", "gpu_uuid", "remote_port", "token_file", "run_dir",
             "workload_private", "hash_cache", "router_encoder_dir", "ports_in_use_by_others", "excluded_gpu_uuids")


class Stop(SystemExit):
    """A pre-declared stop: message on stderr, exit code 3, nothing loaded."""

    def __init__(self, msg):
        print(f"STOP: {msg}", file=sys.stderr)
        super().__init__(3)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=False)


def sha256_obj(obj):
    return sha256_bytes(canonical(obj).encode("utf-8"))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj, replace=False):
    """Outputs are written once: an existing file is never replaced unless the caller says so."""
    path = Path(path)
    if path.exists() and not replace:
        raise FileExistsError(f"{path} exists: outputs are written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torn = False
    if path.exists() and path.stat().st_size:
        with open(path, "rb") as f:                 # a line cut short by a dying process must not swallow the next one
            f.seek(-1, 2)
            torn = f.read(1) != b"\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(("\n" if torn else "") + json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path):
    """(rows, torn): a line cut short by a dying process is skipped and counted, wherever it stands. append_jsonl ends
    such a fragment with a newline before the next row, so after a resumption it is no longer the last line."""
    rows, torn = [], 0
    p = Path(path)
    for l in (p.read_text(encoding="utf-8").splitlines() if p.exists() else []):
        if not l.strip():
            continue
        try:
            rows.append(json.loads(l))
        except ValueError:
            torn += 1
    return rows, torn


def hash_rank(seed, key):
    """Deterministic pseudo-random rank of `key` under `seed`: no random-number library, the same on every machine."""
    return hashlib.sha256(f"{seed}|{key}".encode()).hexdigest()


def request_seed(key, model, mode, phase, rep=0):
    return int(hashlib.sha256(S.SEED_FORMAT.format(key=key, model=model, mode=mode, phase=phase, rep=rep).encode()).hexdigest()[:8], 16)


# --------------------------------------------------------------------------- #
# Freeze                                                                       #
# --------------------------------------------------------------------------- #
def file_hashes():
    return {rel: sha256_file(RI_DIR / rel) for rel in FROZEN_FILES}


def spec_hash():
    """Identity of this protocol's code and frozen inputs. It crosses in the HELLO of the REMOTE wire and is written
    into every record."""
    spec = {k: getattr(S, k) for k in dir(S) if k.isupper()}
    return sha256_obj({"spec": spec, "files": file_hashes()})


def qsel_repo(site=None):
    p = os.environ.get("RINT_QSEL_REPO") or (site or {}).get("qsel_repo")
    if not p:
        raise Stop("the reused runtime's clone is not named (RINT_QSEL_REPO or the site file's qsel_repo)")
    return Path(p).expanduser().resolve()


def _git(repo, *args):
    try:
        r = subprocess.run(["git", "--no-optional-locks", "-C", str(repo), *args], capture_output=True, text=True, timeout=60)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def reused_check(repo):
    """The reused clone must be at the frozen commit with unmodified runtime directories."""
    head = _git(repo, "rev-parse", "HEAD")
    dirty = _git(repo, "status", "--porcelain", "--", *S.REUSED_DIRS)
    dirty = None if dirty is None else "\n".join(l for l in dirty.splitlines() if "__pycache__" not in l)
    if head != S.REUSED_COMMIT:
        raise Stop(f"the reused runtime is not at {S.REUSED_TAG} ({S.REUSED_COMMIT[:12]})")
    if dirty is None or dirty.strip():
        raise Stop("the reused runtime directories differ from their tag")
    return {"tag": S.REUSED_TAG, "commit": head}


def freeze_check(site=None, dev=False):
    """A target command runs only if frozen/FREEZE.json exists and every frozen file still has its recorded hash, and
    the reused runtime is at its tag. dev=True (CPU tests with tiny random models): no check; dev outputs are refused
    as results."""
    if dev:
        return {"dev": True, "protocol": S.PROTOCOL, "spec_hash": spec_hash()}
    fz = RI_DIR / FREEZE_FILE
    if not fz.exists():
        raise Stop("the protocol is not frozen (frozen/FREEZE.json is absent): no target command runs")
    rec = read_json(fz)
    now = file_hashes()
    changed = sorted(k for k in set(now) | set(rec["files"]) if now.get(k) != rec["files"].get(k))
    if changed or rec.get("protocol") != S.PROTOCOL:
        raise Stop(f"files differ from the freeze record: {changed[:6]}")
    return {"protocol": S.PROTOCOL, "freeze_sha256": sha256_file(fz), "spec_hash": spec_hash(), "reused": reused_check(qsel_repo(site))}


# --------------------------------------------------------------------------- #
# The reused runtime                                                           #
# --------------------------------------------------------------------------- #
_REUSED = {}


def reused(site=None):
    """Namespace of the reused frozen modules: qc (qsel_common), qm (qsel_models), qrt (qsel_runtime). Standard library
    only at import time; the model runtime (k16_runtime) is imported by runtime() and needs torch and transformers."""
    if not _REUSED:
        d = qsel_repo(site) / "diagnostics" / "deployment_qwen_select"
        if not (d / "qsel_runtime.py").is_file():
            raise Stop(f"{d} does not hold the reused runtime")
        sys.path.insert(0, str(d))
        for short, name in (("qc", "qsel_common"), ("qm", "qsel_models"), ("qrt", "qsel_runtime")):
            _REUSED[short] = importlib.import_module(name)
        _REUSED["dir"] = d
    return _REUSED


def runtime(site=None):
    """(torch, k16_runtime): the reused whole-model loader k16_runtime.hf_model and its torch setup, unchanged."""
    import torch
    _, R = reused(site)["qrt"].k16()
    R.setup_torch()
    return torch, R


# --------------------------------------------------------------------------- #
# Site file and host                                                           #
# --------------------------------------------------------------------------- #
def load_site(path, role):
    site = read_json(path)
    missing = [k for k in SITE_KEYS if k not in site]
    if missing:
        raise Stop(f"site file lacks {missing}")
    if site["role"] != role:
        raise Stop(f"site file is for role {site['role']}, command is for {role}")
    if not isinstance(site["model_dirs"], dict) or any(m not in site["model_dirs"] for m in S.MODEL_ORDER):
        raise Stop(f"model_dirs must name a directory for each of {S.MODEL_ORDER}")
    if int(site["remote_port"]) in [int(x) for x in site["ports_in_use_by_others"]]:
        raise Stop("the site's port is one the site lists as used by others")
    site["_sha256"] = sha256_file(path)
    return site


def host_facts():
    return {"hostname": socket.gethostname(), "arch": platform.machine(), "python": platform.python_version()}


def check_role(role, site, dev=False):
    facts = host_facts()
    if dev:
        return dict(facts, role=role, dev=True, gpu_names=[])
    qc = reused(site)["qc"]
    rows = qc.gpu_rows()
    if facts["hostname"] != site["expected_hostname"]:
        raise Stop("this host is not the one the site file names")
    names = [r["name"] for r in rows]
    if role == "edge" and (facts["arch"] != S.EDGE_ARCH or not any(S.EDGE_GPU_NAME in n for n in names)):
        raise Stop(f"edge role needs {S.EDGE_ARCH} with a {S.EDGE_GPU_NAME} GPU")
    if role == "server":
        row = next((r for r in rows if r["uuid"] == qc.norm_uuid(site["gpu_uuid"])), None)
        if row is None or S.SERVER_GPU_NAME not in row["name"]:
            raise Stop(f"server role needs the site's GPU to be an {S.SERVER_GPU_NAME}")
        if qc.norm_uuid(site["gpu_uuid"]) in {qc.norm_uuid(u) for u in site["excluded_gpu_uuids"]}:
            raise Stop("the site's GPU is one the site file excludes")
    return dict(facts, role=role, gpu_names=sorted(set(names)))


def versions():
    out = {}
    for m in ("torch", "transformers", "numpy", "safetensors", "tokenizers"):
        try:
            out[m] = importlib.import_module(m).__version__
        except Exception:
            out[m] = None
    return out


def token_of(site, create=False):
    """The shared token of the REMOTE wire: a file of the experiment's own, mode 0600. The server creates it."""
    p = Path(site["token_file"])
    if not p.is_file():
        if not create:
            raise Stop("the wire token file is absent (the server preflight creates it; the edge needs a copy)")
        p.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(hashlib.sha256(os.urandom(32)).hexdigest())
    if p.stat().st_mode & 0o077:
        raise Stop("the wire token file must be mode 0600")
    return p.read_text().strip()
