"""GEMMA-SELECT shared helpers: write-once outputs, hashes, identity checks, resource gates. Standard library only."""
import hashlib
import json
import os
import platform
import re
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # python -I: the script dir is not on sys.path
import gsel_spec as S  # noqa: E402

GS_DIR = Path(__file__).resolve().parent
REPO = GS_DIR.parent.parent
SOURCE_SUFFIXES = (".py", ".sh", ".bash", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini", ".txt", ".md", ".jinja", ".proto",
                   ".service", ".env.example")
SOURCE_NAMES = ("Dockerfile", "Makefile", "Procfile", "docker-compose.yml", "docker-compose.yaml")
SKIP_DIRS = (".git", "__pycache__", ".venv", "venv", "env", ".env", "node_modules", ".cache", ".mypy_cache", ".pytest_cache",
             "wandb", ".ipynb_checkpoints", "site-packages")
SITE_KEYS = ("role", "expected_hostname", "source_root", "model_dir", "device", "gpu_uuid", "port", "credential_file", "client_id",
             "run_dir", "adapter_path", "ports_in_use_by_others")


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


def write_json(path, obj):
    """Outputs are written once: an existing file is never replaced."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists: outputs are written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


# --------------------------------------------------------------------------- #
# Identity of the split source and of the checkpoint                           #
# --------------------------------------------------------------------------- #
def source_manifest(root):
    """(n_files, sha256) over the source-like files of `root`: [[path, sha256, size], ...] sorted by path, canonical JSON.
    The rule (suffixes, names, skipped directories) is the one the pinned hash was produced with."""
    root = Path(root)
    rel = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for fn in filenames:
            if Path(fn).suffix in SOURCE_SUFFIXES or fn in SOURCE_NAMES:
                rel.append((Path(dirpath) / fn).relative_to(root).as_posix())
    entries = [[r, sha256_file(root / r), (root / r).stat().st_size] for r in sorted(rel)]
    return len(entries), sha256_obj(entries)


def check_source(root):
    n, h = source_manifest(root)
    if (n, h) != (S.SOURCE_N_FILES, S.SOURCE_MANIFEST_SHA256):
        raise Stop(f"split source at the site's source_root is not the pinned one ({n} files, manifest {h[:16]})")
    return {"n_files": n, "manifest_sha256": h}


def check_checkpoint(model_dir, deep=True, files=None):
    """Every pinned file present with its size (and sha256 when deep). Returns the per-file record."""
    files = files or S.CHECKPOINT_FILES
    out = {}
    for name, (want, size) in files.items():
        p = Path(model_dir) / name
        if not p.is_file():
            raise Stop(f"checkpoint file {name} is missing")
        if p.stat().st_size != size:
            raise Stop(f"checkpoint file {name} has {p.stat().st_size} bytes, pinned {size}")
        got = sha256_file(p) if deep else None
        if deep and got != want:
            raise Stop(f"checkpoint file {name} has sha256 {got[:16]}, pinned {want[:16]}")
        out[name] = {"bytes": size, "sha256": got, "deep": deep}
    return out


def safetensors_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if not 0 < n < (100 << 20):
            raise ValueError("not a safetensors header")
        return {k: v for k, v in json.loads(f.read(n)).items() if k != "__metadata__"}


def _numel(shape):
    n = 1
    for x in shape:
        n *= int(x)
    return n


LAYER_RE = re.compile(r"(?:^|\.)language_model\.layers\.(\d+)\.")


def estimate_bytes(model_dir, cut=None, weights="model.safetensors"):
    """Bytes each configuration must hold, from the checkpoint header alone (no weight is read). Text tensors are those
    under `language_model.`; the server half = layers >= cut, the final norm and the tied embedding matrix as its head;
    the edge half = the remaining text tensors. Whole-model figures include every tensor of the file."""
    cut = S.CUT if cut is None else cut
    hdr = safetensors_header(Path(model_dir) / weights)
    edge = server = whole = 0
    for name, t in hdr.items():
        n = _numel(t["shape"])
        whole += n
        if "language_model." not in name:
            continue
        m = LAYER_RE.search(name)
        if m:
            if int(m.group(1)) >= cut:
                server += n
            else:
                edge += n
        elif re.search(r"language_model\.norm\.", name):
            server += n
        else:
            edge += n
            if name.endswith("language_model.embed_tokens.weight"):
                server += n
    if not edge or not server:
        raise Stop("cannot estimate the halves: the checkpoint has no language_model layers on one side of the cut")
    return {"edge_float32": 4 * edge, "server_float32": 4 * server, "whole_float32": 4 * whole, "whole_bfloat16": 2 * whole,
            "n_params": {"edge": edge, "server": server, "whole": whole}}


def need_bytes(weight_bytes, factor=None):
    return int((S.LOAD_FACTOR if factor is None else factor) * weight_bytes) + S.LOAD_HEADROOM_BYTES


# --------------------------------------------------------------------------- #
# Resources (read-only probes; other users' processes are counted, never named) #
# --------------------------------------------------------------------------- #
def run(cmd, timeout=30):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout
    except Exception as e:   # tool missing, timeout
        return None, f"{type(e).__name__}"


def norm_uuid(u):
    u = str(u or "").strip().lower()
    return u[4:] if u.startswith("gpu-") else u


def parse_gpu_rows(text):
    rows = []
    for line in (text or "").splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) < 5:
            continue
        num = lambda x: int(x) if x.isdigit() else None  # noqa: E731  (unified memory reports [N/A])
        rows.append({"uuid": norm_uuid(f[0]), "name": f[1], "total_mib": num(f[2]), "used_mib": num(f[3]), "free_mib": num(f[4])})
    return rows


def gpu_rows():
    rc, out = run(["nvidia-smi", "--query-gpu=uuid,name,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"])
    return parse_gpu_rows(out) if rc == 0 else []


def parse_apps(text, uuid, own_pids=()):
    """Aggregate of OTHER compute processes on one GPU: count and MiB only."""
    n = mib = 0
    for line in (text or "").splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) < 3 or norm_uuid(f[0]) != norm_uuid(uuid) or (f[1].isdigit() and int(f[1]) in own_pids):
            continue
        n += 1
        mib += int(f[2]) if f[2].isdigit() else 0
    return {"other_processes": n, "other_mib": mib}


def gpu_others(uuid, own_pids=()):
    rc, out = run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"])
    return parse_apps(out, uuid, own_pids) if rc == 0 else {"other_processes": None, "other_mib": None}


def meminfo(text=None):
    text = Path("/proc/meminfo").read_text() if text is None else text
    m = {}
    for row in text.splitlines():
        k, v, *_ = row.split()
        if k.rstrip(":") in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
            m[k.rstrip(":")] = int(v) * 1024
    m["swap_used"] = m.get("SwapTotal", 0) - m.get("SwapFree", 0)
    return m


def memory_gate(role, site, need, rows=None, mem=None):
    """Free memory against `need` bytes before anything is loaded. Returns the record; raises Stop when short.
    Server: free memory of the pinned GPU and host RAM. Edge (unified memory): available RAM."""
    mem = meminfo() if mem is None else mem
    rec = {"need_bytes": need, "ram_available_bytes": mem.get("MemAvailable"), "swap_used_bytes": mem.get("swap_used")}
    short = []
    if (mem.get("MemAvailable") or 0) < need:
        short.append(f"RAM available {mem.get('MemAvailable', 0) >> 20} MiB < required {need >> 20} MiB")
    if site.get("device", "").startswith("cuda") and role == "server":
        rows = gpu_rows() if rows is None else rows
        row = next((r for r in rows if r["uuid"] == norm_uuid(site["gpu_uuid"])), None)
        if row is None:
            raise Stop("the site's GPU is not visible to nvidia-smi")
        rec.update(gpu_free_mib=row["free_mib"], gpu_total_mib=row["total_mib"])
        if row["free_mib"] is None or row["free_mib"] * (1 << 20) < need:
            short.append(f"GPU free {row['free_mib']} MiB < required {need >> 20} MiB")
    rec["pass"] = not short
    if short:
        raise Stop("BLOCKED, not enough free memory (nothing was loaded; try later, never ask anyone to free memory): " + "; ".join(short))
    return rec


def port_free(port):
    s = socket.socket()
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        s.bind(("127.0.0.1", int(port)))
        return True
    except OSError:
        return False
    finally:
        s.close()


# --------------------------------------------------------------------------- #
# Site file, host role, freeze                                                 #
# --------------------------------------------------------------------------- #
def load_site(path, role):
    site = read_json(path)
    missing = [k for k in SITE_KEYS if k not in site]
    if missing:
        raise Stop(f"site file lacks {missing}")
    if site["role"] != role:
        raise Stop(f"site file is for role {site['role']}, command is for {role}")
    if int(site["port"]) in [int(p) for p in site["ports_in_use_by_others"]]:
        raise Stop("the site's port is one the site lists as used by another service")
    cred = Path(site["credential_file"]).resolve()
    if Path(site["source_root"]).resolve() in cred.parents:
        raise Stop("the credential file must lie outside the split source directory")
    site["_sha256"] = sha256_file(path)
    return site


def check_credential(site):
    p = Path(site["credential_file"])
    if not p.is_file():
        raise Stop("credential file missing (generate the experiment's own pair; see RUNBOOK)")
    if p.stat().st_mode & 0o077:
        raise Stop("credential file must have mode 0600")
    return {"present": True, "mode_ok": True}


def host_facts():
    return {"hostname": socket.gethostname(), "arch": platform.machine(), "python": platform.python_version()}


def check_role(role, site, facts=None, rows=None, dev=False):
    facts = host_facts() if facts is None else facts
    if dev:
        return dict(facts, role=role, dev=True)
    rows = gpu_rows() if rows is None else rows
    if facts["hostname"] != site["expected_hostname"]:
        raise Stop("this host is not the one the site file names")
    names = [r["name"] for r in rows]
    if role == "edge" and (facts["arch"] != S.EDGE_ARCH or not any(S.EDGE_GPU_NAME in n for n in names)):
        raise Stop(f"edge role needs {S.EDGE_ARCH} with a {S.EDGE_GPU_NAME} GPU")
    if role == "server":
        row = next((r for r in rows if r["uuid"] == norm_uuid(site["gpu_uuid"])), None)
        if row is None or S.SERVER_GPU_NAME not in row["name"]:
            raise Stop(f"server role needs the site's GPU to be an {S.SERVER_GPU_NAME}")
    return dict(facts, role=role, gpu_names=sorted(set(names)))


def git(*args):
    rc, out = run(["git", "--no-optional-locks", "-C", str(REPO), *args])
    return out.strip() if rc == 0 else None


def freeze_check(dev=False):
    """A target command runs only from a checkout whose HEAD is the tag and whose experiment directory is unmodified."""
    if dev:
        return {"dev": True}
    head, tag = git("rev-parse", "HEAD"), git("rev-parse", "--verify", "--quiet", f"{S.TAG}^{{commit}}")
    rel = str(GS_DIR.relative_to(REPO))
    dirty = git("status", "--porcelain", "--", rel)
    dirty = None if dirty is None else "\n".join(l for l in dirty.splitlines() if "__pycache__" not in l)
    if not tag or head != tag:
        raise Stop(f"HEAD is not the frozen tag {S.TAG}")
    if dirty is None or dirty.strip():
        raise Stop("the experiment directory differs from the tag")
    return {"tag": S.TAG, "commit": head}
