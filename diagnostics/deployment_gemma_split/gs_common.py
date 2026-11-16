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


def json_text(obj):
    return json.dumps(obj, indent=1, sort_keys=True, default=str, ensure_ascii=False) + "\n"


def jsonl_text(rows):
    return "".join(json.dumps(r, sort_keys=True, default=str, ensure_ascii=False) + "\n" for r in rows)


def write_json(path, obj, overwrite=False):
    return _atomic_write(path, json_text(obj), overwrite)


def write_jsonl(path, rows, overwrite=False):
    return _atomic_write(path, jsonl_text(rows), overwrite)


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
SECRET_NAME = re.compile(r"(TOKEN(?!IZER)|SECRET|PASSW|PASSPHRASE|(?:^|_)PASS(?:$|_)|(?:^|_)PWD(?:$|_)|API_?KEY|ACCESS_?KEY|PRIVATE_?KEY|"
                         r"(?:^|_)KEY(?:$|_)|CREDENTIAL|AUTH|COOKIE|SESSION|BEARER|(?:^|_)JWT(?:$|_))", re.I)
REDACTED = "<redacted>"
_TEXT_RULES = (
    (re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^/\s:@]+:[^/\s@]+@"), r"\1" + REDACTED + "@"),       # user:pass@ in URLs
    (re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^/\s:@]{16,}@"), r"\1" + REDACTED + "@"),             # token@ in URLs
    (re.compile(r"(?i)(--?(?:api[-_]?key|access[-_]?key|token|password|passwd|secret|auth)(?:[= ]+))(\S+)"), r"\1" + REDACTED),
    (re.compile(r"\bhf_[A-Za-z0-9]{16,}\b"), REDACTED),
    (re.compile(r"\b(?:sk|ghp|gho|ghs|github_pat|xox[abp])[-_][A-Za-z0-9_-]{16,}\b"), REDACTED),
    (re.compile(r"(?i)\b([A-Z0-9_]*(?:TOKEN(?!IZER)|SECRET|PASSWORD|PASSWD|PASSPHRASE|API_?KEY|ACCESS_?KEY|PRIVATE_?KEY|CREDENTIAL|_KEY)"
                r"[A-Z0-9_]*)=(\S+)"), r"\1=" + REDACTED),
    (re.compile(r"(?i)(authorization:\s*(?:bearer|basic)\s+)\S+"), r"\1" + REDACTED),
    (re.compile(r"(?i)\b(bearer|basic)\s+(?!<redacted>)[A-Za-z0-9._~+/=-]{16,}"), r"\1 " + REDACTED),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)", re.S), REDACTED),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), REDACTED),
    (re.compile(r"(?i)(authorization:\s*(?:token|apikey)\s+)\S+"), r"\1" + REDACTED),
    (re.compile(r"(?i)(\bx-api-key\s*[:=]\s*)\S+"), r"\1" + REDACTED),
    (re.compile(r"(\bsshpass\s+-p\s*)\S+"), r"\1" + REDACTED),
    (re.compile(r"(?i)(--?auth[-_]?token(?:[= ]+))(\S+)"), r"\1" + REDACTED),
)

# Credentials must never enter a GEMMA-SPLIT artifact (PREREG §4.1): capture/, adapter/, runs/, freeze records, git, archive.
# These high-precision patterns are checked before a freeze and before an archive is written; a finding names the rule
# and the line, never the value.
_SECRET_FINDINGS = (
    ("credentials in a URL", re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^/\s:@<>\"']+:[^/\s@<>\"']+@")),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    ("API token", re.compile(r"\b(?:sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,}|gho_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
                             r"xox[abp]-[A-Za-z0-9-]{10,})")),
    ("AWS access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("authorization header", re.compile(r"(?i)authorization[\"']?\s*[:=]\s*[\"']?(?:bearer|basic|token|apikey)\s+(?!<redacted>)"
                                        r"[A-Za-z0-9._~+/=-]{8,}")),
    ("API key header", re.compile(r"(?i)\bx-api-key[\"']?\s*[:=]\s*[\"']?(?!<redacted>)[A-Za-z0-9._~+/=-]{8,}")),
    ("credential option", re.compile(r"(?i)(?:^|\s)--?(?:api[-_]?key|access[-_]?key|auth[-_]?token|token|password|passwd|secret)(?:=|\s+)"
                                     r"(?!<redacted>|\$)[\"']?[A-Za-z0-9._~+/=-]{8,}")),
    ("sshpass password", re.compile(r"\bsshpass\s+-p\s*(?!<redacted>|\$)\S+")),
    ("secret assignment", re.compile(r"(?i)(?:^|[^A-Za-z0-9])[A-Za-z0-9_]*(?:token|secret|password|passwd|passphrase|api_?key|access_?key|"
                                     r"private_?key|credentials?)[\"']?\s*[=:]\s*[\"']?(?=[^\s\"']*\d)[A-Za-z0-9_\-./+=]{12,}")),
)


def secret_findings(text):
    """[(rule, line number)] for credential-like content in text (redacted values and ${VAR} references never match)."""
    out = []
    for n, line in enumerate((text or "").splitlines(), 1):
        for name, pat in _SECRET_FINDINGS:
            if pat.search(line):
                out.append((name, n))
    return out


TEXT_SUFFIXES = (".json", ".jsonl", ".md", ".txt", ".log", ".py", ".sh", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".env", ".csv")


def scan_files(paths, root):
    """['<relative path>:<line> (<rule>)'] over the text files among `paths` (relative to root)."""
    out = []
    for p in paths:
        p = Path(p)
        if p.is_file() and (p.suffix in TEXT_SUFFIXES or p.name.startswith(".env")):
            out += [f"{p.relative_to(root).as_posix()}:{n} ({rule})" for rule, n in secret_findings(p.read_text(encoding="utf-8", errors="replace"))]
    return out


def redact_text(s):
    if not s:
        return s
    for pat, rep in _TEXT_RULES:
        s = pat.sub(rep, s)
    return s


def redact_obj(obj):
    """Copy of a JSON-like object with redact_text applied to every string value (keys unchanged): harness outputs may
    carry exception messages, which must never carry a credential into runs/ (PREREG §4.1)."""
    if isinstance(obj, str):
        return redact_text(obj)
    if isinstance(obj, dict):
        return {k: redact_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact_obj(v) for v in obj]
    return obj


def redact_env(env):
    """Copy of `env` with every value whose NAME looks secret replaced; other values pass through redact_text."""
    out = {}
    for k, v in env.items():
        out[k] = REDACTED if SECRET_NAME.search(k) else redact_text(v)
    return out
