"""QWEN-SELECT shared helpers: write-once outputs, hashes, the freeze check, site files, resource gates, the GPU-hour
ledger. Standard library only."""
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # python -I: the script dir is not on sys.path
import qsel_spec as S  # noqa: E402

QS_DIR = Path(__file__).resolve().parent
REPO = QS_DIR.parent.parent
SITE_KEYS = ("role", "expected_hostname", "model_dirs", "device", "gpu_uuid", "split_port", "remote_port", "token_file", "run_dir",
             "workload_private", "hash_cache", "ports_in_use_by_others", "excluded_gpu_uuids")
GPU_SELECTION = "gpu_selection.json"
LEDGER = "gpu_ledger.jsonl"
SERVER_STAGE_OUTPUTS = ("server_started.*.json", "server_record.*.json", "remote_started.*.json", "remote_record.*.json")
HASHED_SOURCES = ("qsel_spec.py", "qsel_common.py", "qsel_models.py", "qsel_runtime.py", "qsel_stats.py", "qsel_workload.py", "qsel_bench.py",
                  "frozen/manifest_14B.json", "frozen/tensors_14B.json", "frozen/manifest_8B.json", "frozen/tensors_8B.json",
                  "workload/workload.json", "../deployment_k16fit/k16_spec.py", "../deployment_k16fit/k16_runtime.py",
                  "../deployment_k16fit/k16_criteria.py", "../deployment_k16fit/frozen/checkpoint_manifest.json",
                  "../deployment_k16fit/frozen/tensor_table.json", "../deployment_e1opt/e1_segment.py", "../deployment_e1opt/e1_protocol.py",
                  "../deployment_e1opt/e1_static_cache.py", "../deployment_e1opt/e1_attention.py", "../deployment_e1opt/e1_spec.py",
                  "../deployment_e0/split_runtime.py", "../deployment_e0/e0_q.py", "../deployment_e0/e0_spec.py")


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


def read_jsonl(path, tolerate_torn_tail=False):
    """Rows of a JSONL file. tolerate_torn_tail: a last line cut short by a dying process is dropped and counted."""
    rows, torn = [], 0
    lines = [l for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    for i, l in enumerate(lines):
        try:
            rows.append(json.loads(l))
        except ValueError:
            if tolerate_torn_tail and i == len(lines) - 1:
                torn = 1
            else:
                raise
    return (rows, torn) if tolerate_torn_tail else rows


def spec_hash():
    """Identity of the experiment's code and frozen inputs. It crosses in the HELLO of both wires: a server refuses an
    edge on another revision, and the edge refuses such a server."""
    spec = {k: getattr(S, k) for k in dir(S) if k.isupper()}
    return sha256_obj({"spec": spec, "sources": {rel: sha256_file(QS_DIR / rel) for rel in HASHED_SOURCES}})


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
        rows.append({"uuid": norm_uuid(f[0]), "name": f[1], "total_mib": num(f[2]), "used_mib": num(f[3]), "free_mib": num(f[4]),
                     "index": num(f[5]) if len(f) > 5 else None})
    return rows


def gpu_rows():
    rc, out = run(["nvidia-smi", "--query-gpu=uuid,name,memory.total,memory.used,memory.free,index", "--format=csv,noheader,nounits"])
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


def apps_text():
    rc, out = run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"])
    return out if rc == 0 else None


def gpu_others(uuid, own_pids=()):
    out = apps_text()
    return parse_apps(out, uuid, own_pids) if (out is not None and uuid) else {"other_processes": None, "other_mib": None}


def select_gpu(rows, apps, need, excluded=()):
    """The server GPU, chosen once before the first server stage (PREREG section 11). Eligible: a GPU of the server
    model that the site file does not exclude and that has `need` bytes free now. Among the eligible: fewest compute
    processes of others, then most free memory, then lowest index. Returns (chosen row or None, every row with its reason)."""
    excluded = {norm_uuid(u) for u in excluded}
    considered = []
    for r in rows:
        o = parse_apps(apps, r["uuid"]) if apps is not None else {"other_processes": None, "other_mib": None}
        if S.SERVER_GPU_NAME not in r["name"]:
            why = "not the server GPU model"
        elif r["uuid"] in excluded:
            why = "excluded by the site file"
        elif o["other_processes"] is None:
            why = "other processes could not be counted"
        elif r["free_mib"] is None or r["free_mib"] * (1 << 20) < need:
            why = f"free {r['free_mib']} MiB < required {need >> 20} MiB"
        else:
            why = None
        considered.append(dict(r, **o, eligible=why is None, reason=why))
    ok = sorted((c for c in considered if c["eligible"]),
                key=lambda c: (c["other_processes"], -c["free_mib"], c["index"] if c["index"] is not None else 1 << 30, c["uuid"]))
    return (ok[0] if ok else None), considered


def meminfo(text=None):
    text = Path("/proc/meminfo").read_text() if text is None else text
    m = {}
    for row in text.splitlines():
        k, v, *_ = row.split()
        if k.rstrip(":") in ("MemTotal", "MemAvailable", "MemFree", "SwapTotal", "SwapFree"):
            m[k.rstrip(":")] = int(v) * 1024
    m["swap_used"] = m.get("SwapTotal", 0) - m.get("SwapFree", 0)
    return m


def memory_gate(role, site, need, rows=None, mem=None):
    """Free memory against `need` bytes before anything is loaded. Returns the record; raises Stop when short.
    Edge (unified memory): available RAM, because the device's own figure excludes reclaimable page cache there (records
    of 18 and 19 November). Server: free memory of the pinned GPU, and a fixed amount of host RAM for staging."""
    mem = meminfo() if mem is None else mem
    rec = {"need_bytes": need, "ram_available_bytes": mem.get("MemAvailable"), "swap_used_bytes": mem.get("swap_used")}
    short = []
    if role == "edge":
        if (mem.get("MemAvailable") or 0) < need:
            short.append(f"RAM available {(mem.get('MemAvailable') or 0) >> 20} MiB < required {need >> 20} MiB")
    else:
        if (mem.get("MemAvailable") or 0) < S.HOST_RAM_MIN_BYTES:
            short.append(f"host RAM available {(mem.get('MemAvailable') or 0) >> 20} MiB < {S.HOST_RAM_MIN_BYTES >> 20} MiB")
        if str(site.get("device", "")).startswith("cuda"):
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


def device_gate(role, need, device_free, mem=None):
    """The check inside the process, after the device is initialised and before any weight is read. Server (discrete
    GPU): the device's own free memory. Edge (unified memory): available RAM, read again now; the device's own figure is
    recorded and not applied."""
    mem = meminfo() if mem is None else mem
    avail = mem.get("MemAvailable") or 0
    rec = {"need_bytes": need, "device_free_bytes": device_free, "ram_available_bytes": avail, "ram_free_bytes": mem.get("MemFree"),
           "device_figure_applied": role == "server", "ram_figure_applied": role == "edge"}
    short = []
    if role == "server" and device_free < need:
        short.append(f"device reports {device_free >> 20} MiB free < required {need >> 20} MiB")
    if role == "edge" and avail < need:
        short.append(f"RAM available {avail >> 20} MiB < required {need >> 20} MiB")
    if short:
        raise Stop("BLOCKED, " + "; ".join(short) + " (nothing was loaded; try later, never ask anyone to free memory)")
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
def load_site(path, role, selecting=False):
    """`selecting`: the template handed to select-gpu, whose gpu_uuid is still empty."""
    site = read_json(path)
    missing = [k for k in SITE_KEYS if k not in site]
    if missing:
        raise Stop(f"site file lacks {missing}")
    if site["role"] != role:
        raise Stop(f"site file is for role {site['role']}, command is for {role}")
    if not isinstance(site["excluded_gpu_uuids"], list):
        raise Stop("excluded_gpu_uuids must be a list (it may be empty)")
    if selecting and site["gpu_uuid"]:
        raise Stop("select-gpu takes a template whose gpu_uuid is empty")
    if not selecting and site["gpu_uuid"] and norm_uuid(site["gpu_uuid"]) in {norm_uuid(u) for u in site["excluded_gpu_uuids"]}:
        raise Stop("the site's GPU is one the site file excludes")
    ports = [int(site["split_port"]), int(site["remote_port"])]
    if ports[0] == ports[1] or any(p in [int(x) for x in site["ports_in_use_by_others"]] for p in ports):
        raise Stop("the site's ports must differ from each other and from the ports the site lists as used by others")
    want = ("32B", "14B", "8B") if role == "edge" else ("32B", "14B")
    if not isinstance(site["model_dirs"], dict) or any(m not in site["model_dirs"] for m in want):
        raise Stop(f"model_dirs must name a directory for each of {want}")
    site["_sha256"] = sha256_file(path)
    return site


def local_addr(site, name):
    """Loopback address of one of the experiment's two ports: the server binds it, the edge's SSH forward listens on it."""
    return f"127.0.0.1:{int(site[name])}"


def check_token(site, create=False):
    """The shared token of the REMOTE wire: a file of the experiment's own, mode 0600. The server creates it."""
    p = Path(site["token_file"])
    if not p.is_file():
        if not create:
            raise Stop("token file missing (the server preflight creates it; copy it to the edge as the runbook says)")
        p.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(os.urandom(32).hex() + "\n")
    if p.stat().st_mode & 0o077:
        raise Stop("token file must have mode 0600")
    tok = p.read_text().strip()
    if len(tok) != 64:
        raise Stop("token file is not a 64-character token")
    return tok


def check_selection(site, run_dir):
    """Server: the pinned GPU must be the one select-gpu chose for this run, and stays that one."""
    f = Path(run_dir) / GPU_SELECTION
    if not f.is_file():
        raise Stop("no GPU has been selected for this run (run select-gpu first)")
    sel = read_json(f)
    if norm_uuid(sel["selected"]["uuid"]) != norm_uuid(site["gpu_uuid"]):
        raise Stop("the site's GPU is not the one selected for this run")
    return {"selected_utc": sel["utc_start"], "selection_sha256": sha256_file(f)}


def server_stage_began(run_dir):
    """True once a server stage that loads a model has written anything. From then on the selected GPU is fixed."""
    return any(list(Path(run_dir).glob(p)) for p in SERVER_STAGE_OUTPUTS)


def set_aside(run_dir, pattern, label="crashed"):
    """Outputs of an attempt that died (or a selection that was replaced) are kept under another name, never deleted.
    Returns the number given to them, 0 if there were none."""
    left = sorted(p for p in Path(run_dir).glob(pattern) if f".{label}-" not in p.name)
    if not left:
        return 0
    n = 1
    while list(Path(run_dir).glob(f"*.{label}-{n}")):
        n += 1
    for p in left:
        p.rename(p.with_name(p.name + f".{label}-{n}"))
    return n


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
    if role == "server" and site["gpu_uuid"]:
        row = next((r for r in rows if r["uuid"] == norm_uuid(site["gpu_uuid"])), None)
        if row is None or S.SERVER_GPU_NAME not in row["name"]:
            raise Stop(f"server role needs the site's GPU to be an {S.SERVER_GPU_NAME}")
    return dict(facts, role=role, gpu_names=sorted(set(names)))


def git(*args):
    rc, out = run(["git", "--no-optional-locks", "-C", str(REPO), *args])
    return out.strip() if rc == 0 else None


def freeze_check(dev=False):
    """A target command runs only from a checkout whose HEAD is the tag and whose experiment directory and reused
    runtime directories are unmodified."""
    if dev:
        return {"dev": True, "spec_hash": spec_hash()}
    head, tag = git("rev-parse", "HEAD"), git("rev-parse", "--verify", "--quiet", f"{S.TAG}^{{commit}}")
    rels = [str(QS_DIR.relative_to(REPO))] + list(S.REUSED_DIRS)
    dirty = git("status", "--porcelain", "--", *rels)
    dirty = None if dirty is None else "\n".join(l for l in dirty.splitlines() if "__pycache__" not in l)
    if not tag or head != tag:
        raise Stop(f"HEAD is not the frozen tag {S.TAG}")
    if dirty is None or dirty.strip():
        raise Stop("the experiment directory or the reused runtime differs from the tag")
    return {"tag": S.TAG, "commit": head, "spec_hash": spec_hash()}


# --------------------------------------------------------------------------- #
# GPU-hour ledger (PREREG section 12)                                          #
# --------------------------------------------------------------------------- #
class Ledger:
    """Append-only account of the time our processes hold a GPU on this host. Every GPU stage opens an entry before it
    loads anything and closes it when it ends. An entry that was never closed (a process that was killed) is charged to
    its last heartbeat plus a grace period. A stage starts only inside the host's allotment and is armed with what is
    left, so the sum over both hosts cannot exceed the ceiling."""

    def __init__(self, run_dir, role, allotment_h=None):
        self.dir, self.role = Path(run_dir), role
        self.path = self.dir / LEDGER
        self.allot_s = 3600.0 * (S.ALLOTMENT_H[role] if allotment_h is None else allotment_h)
        self._hb = None

    def entries(self):
        rows, self.bad_lines = [], 0
        for line in (self.path.read_text(encoding="utf-8").splitlines() if self.path.exists() else []):
            try:
                rows.append(json.loads(line))
            except ValueError:                      # a torn line is counted and never stops a later stage
                self.bad_lines += bool(line.strip())
        out = {}
        for r in rows:
            e = out.setdefault(r["id"], {"id": r["id"], "stage": r["stage"]})
            e[r["event"]] = r
        return list(out.values())

    def charged(self, e, now=None):
        now = time.time() if now is None else now
        t0 = e["open"]["t"]
        if "close" in e:
            return max(0.0, e["close"]["t"] - t0)
        hb = self.dir / f"heartbeat_{e['id']}"
        last = hb.stat().st_mtime if hb.exists() else t0
        return max(0.0, min(now, last + S.HEARTBEAT_GRACE_S) - t0)

    def used_s(self, now=None):
        return sum(self.charged(e, now) for e in self.entries() if "open" in e)

    def remaining_s(self, now=None):
        return self.allot_s - self.used_s(now)

    def open(self, stage, natural_limit_s):
        """Returns (entry id, wall-clock limit). Raises Stop when the allotment is used up."""
        rem = self.remaining_s()
        if rem <= S.HEARTBEAT_GRACE_S:
            raise Stop(f"BUDGET: the GPU-hour allotment of this host is used up ({self.used_s() / 3600:.2f} of {self.allot_s / 3600:.2f} h). "
                       "Nothing was loaded. The run is INCOMPLETE; the budget is not raised under this tag")
        limit = min(float(natural_limit_s), rem) if natural_limit_s else rem
        eid = f"{stage}.{int(time.time())}.{os.getpid()}"
        append_jsonl(self.path, {"id": eid, "stage": stage, "event": "open", "t": time.time(), "utc": utc(), "limit_s": limit,
                                 "used_before_s": self.allot_s - rem, "allotment_s": self.allot_s})
        self._id = eid
        (self.dir / f"heartbeat_{eid}").touch()
        stop = threading.Event()

        def beat():
            while not stop.wait(S.HEARTBEAT_S):
                try:
                    os.utime(self.dir / f"heartbeat_{eid}")
                except OSError:
                    pass
        threading.Thread(target=beat, daemon=True).start()
        self._hb = stop
        return eid, limit

    def close(self, note=""):
        if self._hb is None:
            return
        self._hb.set()
        self._hb = None
        append_jsonl(self.path, {"id": self._id, "stage": self._id.split(".")[0], "event": "close", "t": time.time(), "utc": utc(), "note": note})

    def report(self, now=None):
        es = [e for e in self.entries() if "open" in e]
        return {"role": self.role, "allotment_h": self.allot_s / 3600, "used_h": self.used_s(now) / 3600, "remaining_h": self.remaining_s(now) / 3600,
                "unreadable_lines": self.bad_lines,
                "entries": [{"id": e["id"], "stage": e["stage"], "opened_utc": e["open"]["utc"], "closed": "close" in e,
                             "charged_h": self.charged(e, now) / 3600} for e in es]}
