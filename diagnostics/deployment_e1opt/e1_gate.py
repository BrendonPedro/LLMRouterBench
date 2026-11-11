"""E1-OPT decision gates: E0's gate code unchanged, behind a provenance check and the approved arithmetic rule.

  gate-ub   G-UB-E1 (PREREG §8, stage 3): e0_r.py gate-ub2 on {Stage-3 target edge profiles + f_results} x
            {Stage-2 target server profiles}; refuses unless every input row carries matching target provenance
  gate-r    G-R (stage 4): e0_r.py fit + r on the Stage-4 cells, same provenance rule, then the rounded decision

Both write a sidecar `<out>.e1decision.json` with, per evaluated cell: raw delta_pt, delta rounded to 1e-6 pt,
E0's own float flag (e0_raw_pass) and the E1 flag (delta_rounded >= 3.0 and ci_low > 0). The E1 flag decides.
--allow-screening produces a file named *.screening.json only, for labelled screens; never a decision.
"""
import json
import sys
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))   # python -I: the script dir is not on sys.path
import e1_spec as S
sys.path.insert(0, str(S.E0_DIR))
import e1_freeze  # noqa: E402
import e1_provenance as PV  # noqa: E402


def _verify_freeze_files(freeze_path):
    """The gate runs on the analysis host: verify the record's integrity and the hash set, not this machine's role."""
    rec = e1_freeze.load(freeze_path)
    live = e1_freeze.hash_set()
    bad = [f for f, h in rec["file_sha256"].items() if live.get(f) != h] + [f for f in live if f not in rec["file_sha256"]]
    if bad:
        raise e1_freeze.FreezeMismatch(f"E1 files differ from the freeze record: {bad[:8]}")
    return rec


def _gate_inputs(a, stage_dirs):
    rec = _verify_freeze_files(a.freeze)
    roles = json.loads(Path(a.roles).read_text())
    if not roles.get("feasible"):
        sys.exit("roles record says the pair is not feasible")
    probs = []
    for d, role, globs, jsons in stage_dirs:
        probs += PV.validate_dir(d, rec, roles, role, globs, jsons, require_target=not a.allow_screening)
    if probs:
        msg = "PROVENANCE CHECK FAILED (%d problems):\n  - " % len(probs) + "\n  - ".join(probs[:40])
        if not a.allow_screening:
            sys.exit(msg + "\nNo decision file was written.")
        print(msg + "\n(--allow-screening: continuing as a labelled screen)")
    return rec, roles


def _sidecar(gate_json, out, rec, screening):
    g = json.loads(Path(gate_json).read_text())
    rows = S.decide_cells(g["cells"])
    side = {"experiment": S.NAME, "decision_rule_version": S.DECISION_RULE_VERSION, "freeze_sha256": rec["freeze_sha256"],
            "profile_role": "screening" if screening else "target",
            "rule": f"E1 pass iff round(delta_pt, 6) >= {S.R_MARGIN_PT} and ci95_low > 0 (strict); e0_raw_pass is e0_r.py's own float flag, reported for transparency",
            "cells": rows, "n_e1_pass": sum(r["e1_pass"] for r in rows), "n_e0_raw_pass": sum(r["e0_raw_pass"] for r in rows),
            "any_delta_ge3_rounded": any(r["delta_pt_rounded"] >= S.R_MARGIN_PT for r in rows),
            "disagreements": [r for r in rows if r["e1_pass"] != r["e0_raw_pass"]],
            "e0_decision_text": g.get("decision")}
    side["e1_decision"] = ("SCREEN (not a decision)" if screening else
                           ("CONTINUE: some bound cell reaches delta >= 3.0 (rounded)" if side["any_delta_ge3_rounded"] else
                            "STOP: no bound cell reaches delta >= 3.0 (rounded)") if "gate" in g else
                           ("PASS: some cell has delta >= 3.0 (rounded) and CI low > 0" if side["n_e1_pass"] else "FAIL: no cell passes"))
    Path(out).write_text(json.dumps(side, indent=1) + "\n")
    print(json.dumps({k: side[k] for k in ("e1_decision", "n_e1_pass", "n_e0_raw_pass", "any_delta_ge3_rounded")}, indent=1))
    if side["disagreements"]:
        print("NOTE: E0 float flag and E1 rounded flag disagree in %d cell(s); both are in the sidecar" % len(side["disagreements"]))
    return side


def gate_ub(a):
    import e0_r as R
    rec, roles = _gate_inputs(a, [(a.stage3_dir, "edge", ("P-edge-*.jsonl",), ("f_results.json",)),
                                  (a.stage2_dir, "server", ("P-server-*.jsonl",), ())])
    out = a.out
    if a.allow_screening and not out.endswith(".screening.json"):
        out = out[:-5] + ".screening.json" if out.endswith(".json") else out + ".screening.json"
    R.main(["gate-ub2", "--stage1-dir", a.stage3_dir, "--stage2-dir", a.stage2_dir, "--out", out])
    _sidecar(out, out.replace(".json", ".e1decision.json"), rec, a.allow_screening)


def gate_r(a):
    import e0_r as R
    rec, roles = _gate_inputs(a, [(a.stage3_dir, "edge", ("P-edge-*.jsonl",), ("f_results.json",)),
                                  (a.lat_dir, "edge", ("*.jsonl",), ())])
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    tag = ".screening" if a.allow_screening else ""
    bundle = out / f"profiles{tag}.json"
    R.main(["fit", "--lat-dir", a.lat_dir, "--f-results", str(Path(a.stage3_dir) / "f_results.json"), "--out", str(bundle)])
    R.main(["r", "--bundle", str(bundle), "--stage1-dir", a.stage3_dir, "--out-dir", str(out)])
    if tag:
        (out / "R_cells.json").rename(out / "R_cells.screening.json")
    _sidecar(out / f"R_cells{tag}.json", out / f"R_cells{tag}.e1decision.json", rec, a.allow_screening)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate-ub")
    g.add_argument("--stage3-dir", required=True); g.add_argument("--stage2-dir", required=True); g.add_argument("--freeze", required=True)
    g.add_argument("--roles", required=True); g.add_argument("--out", required=True); g.add_argument("--allow-screening", action="store_true")
    r = sub.add_parser("gate-r")
    r.add_argument("--lat-dir", required=True); r.add_argument("--stage3-dir", required=True); r.add_argument("--freeze", required=True)
    r.add_argument("--roles", required=True); r.add_argument("--out-dir", required=True); r.add_argument("--allow-screening", action="store_true")
    a = ap.parse_args(argv)
    {"gate-ub": gate_ub, "gate-r": gate_r}[a.cmd](a)


if __name__ == "__main__":
    main()
