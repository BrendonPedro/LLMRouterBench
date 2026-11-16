"""GEMMA-SPLIT Stage 0d: capture of the existing split deployment's implementation (PREREG §4.3). Read-only.

Records, per host: the implementation source (git commit, branch, dirty state and a SHA-256 manifest of every
tracked or relevant file; or, without git, a manifest of source-like files), the launch scripts / config files the
operator names, the processes that look related, the listening TCP sockets and the containers. Command lines and
URLs are redacted. Nothing is modified: Brian's files are never edited; the experiment later runs a hash-verified
SNAPSHOT copy (`snapshot`, Stage 1) so a later change to the live deployment cannot leak into a measurement.

  python -I gs_deployment.py capture --role edge|server --source-root <dir> [--source-root <dir> ...] \
      [--launch-file <file> ...] --out capture/<role>/deployment.json
  python -I gs_deployment.py snapshot --capture capture/<role>/deployment.json --dest <run-local dir>
"""
import os
import re
import shutil
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # python -I: the script dir is not on sys.path
import gs_common as C  # noqa: E402
import gs_spec as S  # noqa: E402

SOURCE_SUFFIXES = (".py", ".sh", ".bash", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini", ".txt", ".md", ".jinja", ".proto",
                   ".service", ".env.example")
SOURCE_NAMES = ("Dockerfile", "Makefile", "Procfile", "docker-compose.yml", "docker-compose.yaml")
SKIP_DIRS = (".git", "__pycache__", ".venv", "venv", "env", ".env", "node_modules", ".cache", ".mypy_cache", ".pytest_cache",
             "wandb", ".ipynb_checkpoints", "site-packages")
MAX_SOURCE_FILE_BYTES = 5 * 1024 * 1024
MAX_SOURCE_FILES = 20000
MAX_LAUNCH_TEXT = 200_000
PROCESS_PATTERN = re.compile(r"(python|torchrun|vllm|llama|uvicorn|gunicorn|fastapi|grpc|ray|docker|split|gemma|server|client)", re.I)


def _manifest(root, files):
    out = []
    for rel in sorted(files):
        p = root / rel
        if not p.is_file():
            out.append({"path": rel, "missing": True})
            continue
        st = p.stat()
        if st.st_size > MAX_SOURCE_FILE_BYTES:
            out.append({"path": rel, "size": st.st_size, "skipped": "larger than MAX_SOURCE_FILE_BYTES"})
            continue
        out.append({"path": rel, "size": st.st_size, "sha256": C.sha256_file(p)})
    return out


def source_capture(source_root):
    root = Path(source_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"source root {root} does not exist")
    top = C.git(["rev-parse", "--show-toplevel"], root)
    rec = {"root": str(root), "git": None}
    if top:
        top = Path(top).resolve()
        prefix = os.path.relpath(root, top)            # "." or the source root's path inside the repository
        scope = [] if prefix == "." else ["--", prefix]
        tracked = (C.git(["ls-files", "-z", "--full-name", *scope], top) or "").split("\0")
        status = C.git(["--no-optional-locks", "status", "--porcelain=v1", "--untracked-files=all", *scope], top) or ""   # read-only
        untracked = [l[3:] for l in status.splitlines() if l.startswith("??")]
        rec["git"] = {"toplevel": str(top), "subdir": prefix, "commit": C.git(["rev-parse", "HEAD"], top),
                      "branch": C.git(["rev-parse", "--abbrev-ref", "HEAD"], top), "dirty": bool(status.strip()),
                      "status_porcelain": status.splitlines()[:500], "remotes": C.redact_text(C.git(["remote", "-v"], top) or ""),
                      "last_commit_utc": C.git(["log", "-1", "--format=%cI"], top)}
        rel_files = [f for f in tracked if f] + [f for f in untracked if Path(f).suffix in SOURCE_SUFFIXES or Path(f).name in SOURCE_NAMES]
        if prefix != ".":   # paths are repository-relative: keep the source root's files, relative to it
            rel_files = [os.path.relpath(f, prefix) for f in rel_files]
        rec["manifest_rule"] = "git tracked files + untracked source-like files (within the source root)"
    else:
        rel_files = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            for fn in filenames:
                if Path(fn).suffix in SOURCE_SUFFIXES or fn in SOURCE_NAMES:
                    rel_files.append((Path(dirpath) / fn).relative_to(root).as_posix())
                    if len(rel_files) > MAX_SOURCE_FILES:
                        raise RuntimeError(f"{root}: more than {MAX_SOURCE_FILES} source files; name the implementation directory, not a parent")
        rec["manifest_rule"] = f"no git: files with suffixes {SOURCE_SUFFIXES} or names {SOURCE_NAMES}, skipping {SKIP_DIRS}"
    rec["manifest"] = _manifest(root, set(rel_files))
    rec["manifest_sha256"] = C.sha256_obj([[e["path"], e.get("sha256"), e.get("size")] for e in rec["manifest"]])
    return rec


def launch_capture(path):
    p = Path(path).resolve()
    if not p.is_file():
        return {"path": str(p), "missing": True}
    raw = p.read_bytes()
    return {"path": str(p), "sha256": C.sha256_bytes(raw), "size": len(raw),
            "text_redacted": C.redact_text(raw[:MAX_LAUNCH_TEXT].decode("utf-8", errors="replace")),
            "truncated": len(raw) > MAX_LAUNCH_TEXT}


def parse_ps(text):
    rows = []
    for line in (text or "").splitlines():
        f = line.split(None, 4)
        if len(f) < 5 or not f[0].isdigit():
            continue
        args = f[4]
        if PROCESS_PATTERN.search(args):
            rows.append({"pid": int(f[0]), "ppid": int(f[1]), "user": f[2], "etimes_s": int(f[3]) if f[3].isdigit() else None,
                         "args": C.redact_text(args)})
    return rows


def capture(role, source_roots, launch_files=(), note=""):
    if role not in S.ROLES:
        raise ValueError(f"role must be one of {S.ROLES}")
    ps = C.run(["ps", "-eo", "pid=,ppid=,user=,etimes=,args="])
    return {"schema": "gemma-split-deployment/1", "experiment": S.NAME, "role": role, "host": socket.gethostname(), "utc": C.utc(),
            "sources": [source_capture(r) for r in source_roots], "launch_files": [launch_capture(f) for f in launch_files],
            "processes": parse_ps(ps["out"]), "listening_tcp": C.run("ss -ltnp 2>/dev/null")["out"],
            "containers": C.redact_text(C.run(["docker", "ps", "--no-trunc", "--format", "{{json .}}"], timeout=30)["out"]),
            "python_executable": sys.executable, "note": note}


def snapshot(source_rec, dest):
    """Copy the manifest files of one captured source root to `dest` and verify every hash (Stage 1). Returns the
    snapshot record; raises if any file differs from the capture (the live deployment changed after Stage 0)."""
    src, dest = Path(source_rec["root"]), Path(dest)
    if dest.resolve() == src.resolve() or src.resolve() in dest.resolve().parents:
        raise ValueError(f"{dest} lies inside Brian's source tree {src}: the snapshot is written outside it (nothing of his is modified)")
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"{dest} is not empty: a snapshot is written once")
    problems = []
    for e in source_rec["manifest"]:
        if e.get("missing") or e.get("skipped"):
            continue
        s, d = src / e["path"], dest / e["path"]
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
        if C.sha256_file(d) != e["sha256"]:
            problems.append(e["path"])
    if problems:
        raise RuntimeError(f"snapshot differs from the Stage-0 capture (deployment changed?): {problems[:10]}")
    rec = {"source_root": str(src), "snapshot_root": str(dest.resolve()), "manifest_sha256": source_rec["manifest_sha256"],
           "files": sum(1 for e in source_rec["manifest"] if e.get("sha256")), "utc": C.utc()}
    C.write_json(dest / "GEMMA_SPLIT_SNAPSHOT.json", rec)
    return rec


def verify_snapshot(source_rec, snapshot_root):
    """Files of the snapshot that differ from the capture, are missing, or were added (a planted module is a difference)."""
    root = Path(snapshot_root)
    want = {e["path"] for e in source_rec["manifest"] if e.get("sha256")}
    bad = [e["path"] for e in source_rec["manifest"] if e.get("sha256") and
           (not (root / e["path"]).is_file() or C.sha256_file(root / e["path"]) != e["sha256"])]
    extra = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if (p.is_file() or p.is_symlink())
                   and p.relative_to(root).as_posix() not in want and p.name != "GEMMA_SPLIT_SNAPSHOT.json")
    return bad + [f"extra: {x}" for x in extra]


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--role", required=True, choices=S.ROLES)
    c.add_argument("--source-root", action="append", required=True)
    c.add_argument("--launch-file", action="append", default=[])
    c.add_argument("--note", default="")
    c.add_argument("--out", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("--capture", required=True)
    s.add_argument("--dest", required=True, help="run-local directory outside Brian's tree, e.g. ~/gemma_split_snapshot/<role>")
    a = ap.parse_args(argv)
    if a.cmd == "capture":
        rec = capture(a.role, a.source_root, a.launch_file, a.note)
        C.write_json(a.out, rec)
        for s_ in rec["sources"]:
            print(s_["root"], "git", (s_.get("git") or {}).get("commit"), "dirty", (s_.get("git") or {}).get("dirty"),
                  "files", len(s_["manifest"]), "manifest", s_["manifest_sha256"][:16])
        print("related processes:", len(rec["processes"]))
    else:
        cap = C.read_json(a.capture)
        for i, src in enumerate(cap["sources"]):
            dest = Path(a.dest) / f"source{i}"
            print(snapshot(src, dest))


if __name__ == "__main__":
    main()
