"""ROUTER-INTEG: write or check the freeze record (frozen/FREEZE.json).

The freeze record lists the SHA-256 of every decision file, code file and frozen input of the protocol
(rint_common.FROZEN_FILES). Every target command refuses to run without it, or if any listed file has changed.
No Git commit is involved: the record is the freeze. A changed rule is a new protocol version with a new record;
an existing record is never replaced.

  python -I -B tools/rint_freeze.py write --note "..."     # once, after review
  python -I -B tools/rint_freeze.py check
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rint_common as C  # noqa: E402
import rint_spec as S  # noqa: E402


def write(root, note):
    root = Path(root)
    p = root / C.FREEZE_FILE
    if p.exists():
        raise SystemExit(f"STOP: {p} exists: a freeze record is never replaced")
    rec = {"protocol": S.PROTOCOL, "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "note": note,
           "reused_runtime": {"tag": S.REUSED_TAG, "commit": S.REUSED_COMMIT},
           "files": {rel: C.sha256_file(root / rel) for rel in C.FROZEN_FILES}}
    p.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    w = sp.add_parser("write")
    w.add_argument("--note", required=True)
    sp.add_parser("check")
    a = ap.parse_args(argv)
    if a.cmd == "write":
        rec = write(C.RI_DIR, a.note)
        print(json.dumps({"protocol": rec["protocol"], "files": len(rec["files"]), "freeze_sha256": C.sha256_file(C.RI_DIR / C.FREEZE_FILE)}, indent=1))
    else:
        rec = C.read_json(C.RI_DIR / C.FREEZE_FILE)
        now = C.file_hashes()
        changed = sorted(k for k in set(now) | set(rec["files"]) if now.get(k) != rec["files"].get(k))
        print(json.dumps({"protocol": rec["protocol"], "frozen_utc": rec["frozen_utc"], "changed": changed,
                          "freeze_sha256": C.sha256_file(C.RI_DIR / C.FREEZE_FILE), "spec_hash": C.spec_hash()}, indent=1))
        sys.exit(0 if not changed else 3)


if __name__ == "__main__":
    main()
