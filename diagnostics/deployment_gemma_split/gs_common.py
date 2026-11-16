"""GEMMA-SPLIT shared helpers (standard library only): hashing, canonical JSON, write-once files, subprocess
capture that never raises, and redaction of secrets in environment variables, command lines and URLs.

Every capture and result file goes through `write_json` / `write_jsonl`, which refuse to overwrite an existing
file unless the caller says so explicitly (PREREG §14: outputs are written once; a repeat is a new file).
"""
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def utc_ms(t=None):
    """ISO-8601 UTC with milliseconds: a provenance timestamp (when something happened), never a measurement."""
    t = time.time() if t is None else t
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + f".{int((t % 1) * 1000):03d}Z"


def parse_utc(s):
    """Seconds since the epoch for a utc() / utc_ms() string."""
    import calendar
    base, _, frac = s.rstrip("Z").partition(".")
    return calendar.timegm(time.strptime(base, "%Y-%m-%dT%H:%M:%S")) + (int(frac) / 10 ** len(frac) if frac else 0.0)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=False)


def sha256_obj(obj):
    return sha256_bytes(canonical(obj).encode("utf-8"))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_write(path, text, overwrite):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists: GEMMA-SPLIT outputs are written once (write a new file instead)")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return path


def write_text(path, text, overwrite=False):
    return _atomic_write(path, text, overwrite)


def write_json(path, obj, overwrite=False):
    return _atomic_write(path, json.dumps(obj, indent=1, sort_keys=True, default=str, ensure_ascii=False) + "\n", overwrite)


def write_jsonl(path, rows, overwrite=False):
    return _atomic_write(path, "".join(json.dumps(r, sort_keys=True, default=str, ensure_ascii=False) + "\n" for r in rows),
                         overwrite)


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run(cmd, timeout=60, cwd=None):
    """Run a command and return {"cmd", "rc", "out", "err"}; never raises (a missing tool is a recorded fact)."""
    shell = isinstance(cmd, str)
    try:
        r = subprocess.run(cmd, shell=shell, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        return {"cmd": cmd if shell else " ".join(cmd), "rc": r.returncode, "out": (r.stdout or "").strip(),
                "err": (r.stderr or "").strip()}
    except Exception as e:  # FileNotFoundError, TimeoutExpired, PermissionError
        return {"cmd": cmd if shell else " ".join(cmd), "rc": None, "out": "", "err": f"{type(e).__name__}: {e}"}


def git(args, cwd):
    r = run(["git", *args], timeout=60, cwd=str(cwd))
    return r["out"] if r["rc"] == 0 else None


# --------------------------------------------------------------------------- #
# Redaction (PREREG §4.1: secrets are never captured)                          #
# --------------------------------------------------------------------------- #
SECRET_NAME = re.compile(r"(TOKEN|SECRET|PASSW|API_?KEY|ACCESS_?KEY|PRIVATE_?KEY|CREDENTIAL|AUTH|COOKIE|SESSION)", re.I)
REDACTED = "<redacted>"
_TEXT_RULES = (
    (re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^/\s:@]+:[^/\s@]+@"), r"\1" + REDACTED + "@"),       # user:pass@ in URLs
    (re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^/\s:@]{16,}@"), r"\1" + REDACTED + "@"),             # token@ in URLs
    (re.compile(r"(?i)(--?(?:api[-_]?key|access[-_]?key|token|password|passwd|secret|auth)(?:[= ]+))(\S+)"), r"\1" + REDACTED),
    (re.compile(r"\bhf_[A-Za-z0-9]{16,}\b"), REDACTED),
    (re.compile(r"\b(?:sk|ghp|gho|ghs|github_pat|xox[abp])[-_][A-Za-z0-9_-]{16,}\b"), REDACTED),
    (re.compile(r"(?i)\b([A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|PASSWD|API_?KEY|ACCESS_?KEY|CREDENTIAL)[A-Z0-9_]*)=(\S+)"),
     r"\1=" + REDACTED),
    (re.compile(r"(?i)(authorization:\s*(?:bearer|basic)\s+)\S+"), r"\1" + REDACTED),
)


def redact_text(s):
    if not s:
        return s
    for pat, rep in _TEXT_RULES:
        s = pat.sub(rep, s)
    return s


def redact_env(env):
    """Copy of `env` with every value whose NAME looks secret replaced; other values pass through redact_text."""
    out = {}
    for k, v in env.items():
        out[k] = REDACTED if SECRET_NAME.search(k) else redact_text(v)
    return out
