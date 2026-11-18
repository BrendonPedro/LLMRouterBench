"""K16-HWFIT shared helpers: JSON I/O, gates, environment and revision capture. Standard library only."""
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

MIB = 1 << 20
EXIT_STOP = 3          # a preflight or memory gate refused to start
EXIT_GATE = 4          # a correctness gate failed, or a stage was asked to run behind a closed gate


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n")
    os.replace(tmp, path)
    return path


def read_json(path):
    return json.loads(Path(path).read_text())


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def read_jsonl(path):
    p = Path(path)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def run(cmd, timeout=60, cwd=None):
    """stdout of a read-only command, or 'ERROR ...'. Never raises."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd, shell=isinstance(cmd, str))
        return r.stdout.strip() if r.returncode == 0 else f"ERROR rc={r.returncode} {(r.stderr or r.stdout).strip()[:300]}"
    except Exception as e:
        return f"ERROR {type(e).__name__}: {e}"


# --------------------------------------------------------------------------- #
# Gates: a stage runs only behind the gates it names                           #
# --------------------------------------------------------------------------- #
class GateClosed(RuntimeError):
    pass


def gate_path(run_dir, name):
    return Path(run_dir) / "gates" / f"{name}.json"


def gate_write(run_dir, name, passed, spec_hash, detail=None):
    rec = {"gate": name, "pass": bool(passed), "spec_hash": spec_hash, "utc": utc(), "host": socket.gethostname(),
           "detail": detail or {}}
    write_json(gate_path(run_dir, name), rec)
    print(f"[gate] {name}: {'PASS' if passed else 'FAIL'}", flush=True)
    return rec


def gate_status(run_dir, name, spec_hash):
    """(open, reason). A gate is open only if its file exists, says pass, and was written under this spec hash."""
    p = gate_path(run_dir, name)
    if not p.exists():
        return False, f"gate {name} has not run"
    g = read_json(p)
    if g.get("spec_hash") != spec_hash:
        return False, f"gate {name} was written under another spec hash"
    if g.get("pass") is not True:
        return False, f"gate {name} did not pass"
    return True, "open"


def gate_require(run_dir, names, spec_hash):
    for n in names:
        ok, why = gate_status(run_dir, n, spec_hash)
        if not ok:
            raise GateClosed(why)


# --------------------------------------------------------------------------- #
# Provenance                                                                   #
# --------------------------------------------------------------------------- #
def versions():
    v = {"python": platform.python_version(), "executable": sys.executable, "platform": platform.platform(),
         "machine": platform.machine()}
    for mod in ("torch", "transformers", "safetensors", "numpy", "tokenizers", "accelerate", "flash_attn"):
        try:
            v[mod] = getattr(__import__(mod), "__version__", "present")
        except Exception as e:
            v[mod] = None
            v[mod + "_error"] = f"{type(e).__name__}: {str(e)[:120]}"
    try:
        import torch
        v["torch_cuda"] = torch.version.cuda
        v["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        v["torch_cuda"], v["cuda_available"] = None, False
    return v


def git_info(repo, base_tag, paths=("diagnostics/deployment_k16fit", "diagnostics/deployment_e0", "diagnostics/deployment_e1opt")):
    """Exact code revision. `dirty` lists tracked files of the experiment and of the reused runtime that differ from
    HEAD. `runtime_unchanged_vs_base` is True iff deployment_e0 and deployment_e1opt equal the base tag."""
    g = lambda *a: run(["git", *a], cwd=repo)
    head = g("rev-parse", "HEAD")
    dirty = g("status", "--porcelain", "--untracked-files=all", "--", *paths)
    base = g("rev-parse", "--verify", "--quiet", f"{base_tag}^{{commit}}")
    unchanged = None
    if not base.startswith("ERROR") and base:
        r = subprocess.run(["git", "diff", "--quiet", base, "--", *paths[1:]], cwd=repo)
        unchanged = r.returncode == 0
    return {"commit": None if head.startswith("ERROR") else head, "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": [] if not dirty or dirty.startswith("ERROR") else dirty.splitlines(),
            "dirty_error": dirty if dirty.startswith("ERROR") else None,
            "base_tag": base_tag, "base_tag_commit": None if base.startswith("ERROR") or not base else base,
            "runtime_unchanged_vs_base": unchanged}


def host_info():
    return {"hostname": socket.gethostname(), "fqdn": socket.getfqdn(), "user": os.environ.get("USER"), "utc": utc(),
            "uname": " ".join(platform.uname()), "addresses": run("hostname -I").split()}


class Tee:
    """Mirror prints into a log file (the launchers also tee the whole console)."""

    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.f, self.out = open(path, "a"), sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()
