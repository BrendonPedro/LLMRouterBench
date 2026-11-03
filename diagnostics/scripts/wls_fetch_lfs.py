"""Fetch the 18 Git LFS objects the pipeline reads, OUTSIDE the working tree.

17 are read by the KorBench/MBPP evaluators; data/FinQA/test.json is read only by the FinQA
eligibility rule (non-empty `answer`) in the sample freezer.

git-lfs is not installed here and data/** are LFS pointer stubs. Overwriting the stubs would
show modified tracked files, so the objects go to a separate directory
(default ~/.cache/LLMRouterBench/wls_lfs_data, override with WLS_LFS_DIR or --out).

Procedure (mission §7a): ask the GitHub LFS batch API of `origin` for the oids named by the
committed pointers, download each object, and keep it only if its sha256 and size equal the
pointer's. Objects already present and valid are not downloaded again.

Equivalent once git-lfs is installed (restores in place instead):
    git lfs pull --include="data/KORBench/**,data/MBPP/test.json,data/FinQA/test.json"

Usage: python diagnostics/scripts/wls_fetch_lfs.py [--out DIR] [--verify-only]
"""

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import wls_common as C


def lfs_batch_url():
    url = subprocess.run(["git", "remote", "get-url", "origin"], cwd=C.REPO,
                         capture_output=True, text=True, check=True).stdout.strip()
    if not url.startswith("https://github.com/"):
        sys.exit(f"origin is not an https GitHub URL: {url}")
    return url.removesuffix("/").removesuffix(".git") + ".git/info/lfs/objects/batch"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    out = (args.out or C.lfs_dir()).resolve()
    if C.REPO in out.parents or out == C.REPO:
        sys.exit(f"refusing to write inside the working tree: {out}")

    status = C.verify_lfs_copy(out)
    missing = [rel for rel, ok in status.items() if not ok]
    if args.verify_only or not missing:
        for rel, ok in status.items():
            print(("OK   " if ok else "MISS ") + rel)
        print(f"{sum(status.values())}/{len(status)} verified in {out}")
        sys.exit(0 if not missing else 1)

    ptrs = {rel: C.lfs_pointer(rel) for rel in missing}
    body = {"operation": "download", "transfers": ["basic"],
            "objects": [{"oid": o, "size": s} for o, s in sorted(set(ptrs.values()))]}
    req = urllib.request.Request(
        lfs_batch_url(), data=json.dumps(body).encode(),
        headers={"Accept": "application/vnd.git-lfs+json",
                 "Content-Type": "application/vnd.git-lfs+json"})
    objs = {o["oid"]: o for o in json.load(urllib.request.urlopen(req, timeout=60))["objects"]}
    for rel, (oid, size) in ptrs.items():
        a = objs[oid]["actions"]["download"]
        data = urllib.request.urlopen(
            urllib.request.Request(a["href"], headers=a.get("header", {})), timeout=120).read()
        if hashlib.sha256(data).hexdigest() != oid or len(data) != size:
            sys.exit(f"BAD  {rel}: sha256/size does not match the committed pointer")
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(dst)
        print(f"GOT  {rel} ({size} bytes)")

    status = C.verify_lfs_copy(out)
    print(f"{sum(status.values())}/{len(status)} verified in {out}")
    sys.exit(0 if all(status.values()) else 1)


if __name__ == "__main__":
    main()
