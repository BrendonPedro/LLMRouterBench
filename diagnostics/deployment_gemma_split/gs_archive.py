"""GEMMA-SPLIT final archive and hash procedure (PREREG §15).

`create` writes a byte-reproducible .tar.gz of the protocol, the harness, the tests, the workload, capture/, adapter/,
frozen/ (every freeze record) and runs/ (every run, superseded ones and _invalidated/ parts included) (no model weights;
the checkpoint is identified by its manifest in the freeze record; the
deployment source snapshot is included only with --include-snapshot, i.e. with the implementer's consent). Members
are sorted, owner/group/mtime are zeroed, gzip carries mtime 0, and a SHA256SUMS member lists every file. The
archive's own SHA-256 is written next to it (<archive>.sha256). `verify` recomputes both levels.

  python -I gs_archive.py create --run-id <run_id> --out <dir>/gemma_split_<run_id>_<OUTCOME>.tgz [--include-snapshot DIR ...]
  python -I gs_archive.py verify --archive <file>.tgz
"""
import gzip
import io
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # python -I: the script dir is not on sys.path
import gs_common as C  # noqa: E402
import gs_spec as S  # noqa: E402

TOP_FILES = ("README.md", S.PREREG_FILE, "RUNBOOK.md")
TREES = ("tests", "workload", "capture", "adapter", "frozen")
SKIP_PARTS = ("__pycache__",)
SKIP_SUFFIXES = (".pyc", ".pyo")


def members(gs_dir, run_id, snapshots=()):
    gs_dir = Path(gs_dir)
    out = {}
    for f in TOP_FILES:
        if (gs_dir / f).is_file():
            out[f] = gs_dir / f
    for p in sorted(gs_dir.glob("gs_*.py")):
        out[p.name] = p
    trees = list(TREES) + ["runs"]          # every run, superseded ones included (PREREG §6.4, §15)
    for t in trees:
        root = gs_dir / t
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file() and not any(s in p.parts for s in SKIP_PARTS) and p.suffix not in SKIP_SUFFIXES:
                out[p.relative_to(gs_dir).as_posix()] = p
    for i, snap in enumerate(snapshots):
        for p in sorted(Path(snap).rglob("*")):
            if p.is_file():
                out[f"snapshot/{i}/{p.relative_to(snap).as_posix()}"] = p
    out.update(adapter_history(gs_dir))
    if not any(k.startswith(f"runs/{run_id}/") for k in out):
        raise FileNotFoundError(f"no files under runs/{run_id}")
    return dict(sorted(out.items()))


def adapter_history(gs_dir):
    """The adapter of every freeze record as committed with it (a superseded record's adapter is no longer in adapter/):
    {"adapter_history/<run_id>/<file>": bytes}, checked against the record's adapter hash. Empty without git."""
    import subprocess
    gs_dir = Path(gs_dir).resolve()
    top = C.git(["rev-parse", "--show-toplevel"], gs_dir)
    out = {}
    if not top:
        return out
    for q in sorted((gs_dir / "frozen").glob("*.json")):
        try:
            rec = C.read_json(q)
        except Exception:
            continue
        commit, path = (rec.get("git") or {}).get("commit"), (rec.get("adapter") or {}).get("path")
        if not (isinstance(rec, dict) and rec.get("experiment") == S.NAME and commit and path):
            continue
        rel = (gs_dir / path).relative_to(Path(top).resolve()).as_posix()
        r = subprocess.run(["git", "show", f"{commit}:{rel}"], cwd=top, capture_output=True)
        if r.returncode == 0:
            ok = C.sha256_bytes(r.stdout) == rec["adapter"].get("sha256")
            out[f"adapter_history/{rec['run_id']}/{Path(path).name}" + ("" if ok else ".HASH_MISMATCH")] = r.stdout
    return out


def _info(name, size):
    ti = tarfile.TarInfo(name)
    ti.size, ti.mtime, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = size, 0, 0o644, 0, 0, "", ""
    return ti


def create(gs_dir, run_id, out, snapshots=()):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists: an archive is written once")
    mem = members(gs_dir, run_id, snapshots)
    blob = lambda p: p if isinstance(p, bytes) else p.read_bytes()  # noqa: E731
    lab = ("capture/", "adapter/", "adapter_history/", "frozen/", "runs/", "snapshot/")     # where lab data can enter (the harness is fixed)
    found = [f"{name}:{n} ({rule})" for name, p in mem.items() if name.startswith(lab) and (name.endswith(C.TEXT_SUFFIXES) or "/." in name
             or name.startswith("snapshot/")) for rule, n in C.secret_findings(blob(p).decode("utf-8", errors="replace"))]
    if found:
        raise ValueError(f"credential-like content in archive members (PREREG §4.1, §15): {found[:10]}: remove it; nothing is archived")
    sums = "".join(f"{C.sha256_bytes(blob(p))}  {name}\n" for name, p in mem.items()).encode()
    prefix = f"gemma_split_{run_id}/"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tf:
        tf.addfile(_info(prefix + "SHA256SUMS", len(sums)), io.BytesIO(sums))
        for name, p in mem.items():
            data = blob(p)
            tf.addfile(_info(prefix + name, len(data)), io.BytesIO(data))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        with gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0, compresslevel=9) as gz:
            gz.write(buf.getvalue())
    digest = C.sha256_file(out)
    C.write_text(out.with_name(out.name + ".sha256"), f"{digest}  {out.name}\n")
    return {"archive": str(out), "sha256": digest, "members": len(mem)}


def verify(archive):
    archive = Path(archive)
    problems = []
    side = archive.with_name(archive.name + ".sha256")
    if side.exists():
        want = side.read_text().split()[0]
        if C.sha256_file(archive) != want:
            problems.append("archive sha256 differs from its .sha256 file")
    else:
        problems.append("no .sha256 file next to the archive")
    with tarfile.open(archive, "r:gz") as tf:
        names = tf.getnames()
        sums_name = next((n for n in names if n.endswith("/SHA256SUMS")), None)
        if sums_name is None:
            return problems + ["no SHA256SUMS member"]
        prefix = sums_name[:-len("SHA256SUMS")]
        listed = {}
        for line in tf.extractfile(sums_name).read().decode().splitlines():
            h, name = line.split("  ", 1)
            listed[name] = h
        for n in names:
            if n == sums_name:
                continue
            rel = n[len(prefix):]
            if rel not in listed:
                problems.append(f"member not listed in SHA256SUMS: {rel}")
            elif C.sha256_bytes(tf.extractfile(n).read()) != listed[rel]:
                problems.append(f"member hash differs: {rel}")
        missing = set(listed) - {n[len(prefix):] for n in names}
        problems += [f"listed but absent: {m}" for m in sorted(missing)]
    return problems


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--run-id", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--include-snapshot", action="append", default=[])
    c.add_argument("--gs-dir", default=str(S.GS_DIR))
    v = sub.add_parser("verify")
    v.add_argument("--archive", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "create":
        print(C.canonical(create(a.gs_dir, a.run_id, a.out, a.include_snapshot)))
    else:
        P = verify(a.archive)
        print("archive OK" if not P else "archive problems:\n  - " + "\n  - ".join(P))
        sys.exit(0 if not P else 2)


if __name__ == "__main__":
    main()
