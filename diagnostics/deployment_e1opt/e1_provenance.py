"""Provenance of every E1-OPT measurement and the rejection rules of the decision gates (PREREG_E1OPT §8, §10).

Every profile row and every summary JSON produced by an E1 driver carries a `provenance` object. The gates
(`e1_bench.py gate-ub`, `gate-r`) refuse to run unless every input row carries one that matches the freeze
record and the role record, and is marked profile_role = "target". Screening rows (rented hardware, dev cuts,
archived E0 profiles, which carry no provenance at all) can never enter a decision.
"""
import json
import os
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))   # python -I: the script dir is not on sys.path
import e1_spec as S

REQUIRED = ("experiment", "stage", "machine", "gpu", "freeze_sha256", "profile_role", "model", "cut", "memory_class",
            "runtime_branch", "decision_rule_version", "protocol_version", "role")


class ProvenanceError(ValueError):
    pass


def make(freeze, stage, role, gpu, profile_role, cut, mem_class, host=None, extra=None):
    import socket
    if profile_role not in S.PROFILE_ROLES:
        raise ProvenanceError(f"profile_role must be one of {S.PROFILE_ROLES}")
    host = host or socket.gethostname()
    m = freeze["machines"].get(host, {})
    p = {"experiment": S.NAME, "stage": stage, "role": role, "machine": host,
         "gpu": {"name": gpu.get("name"), "uuid": gpu.get("uuid"), "index": gpu.get("index")},
         "freeze_sha256": freeze["freeze_sha256"], "profile_role": profile_role,
         "model": f"{freeze['model']['repo']}@{freeze['model']['revision']}", "cut": list(cut) if cut else None,
         "memory_class": mem_class, "runtime_branch": m.get("branch", {}), "decision_rule_version": freeze["decision_rule_version"],
         "protocol_version": freeze["protocol_version"]}
    if extra:
        p.update(extra)
    return p


def check(p, freeze, roles, expect_role=None, require_target=True, what=""):
    """Problems (list of str) of one provenance object against the freeze and role records."""
    pre = f"{what}: " if what else ""
    if not isinstance(p, dict):
        return [pre + "no provenance (archived E0 or foreign row)"]
    probs = [pre + f"missing {k}" for k in REQUIRED if k not in p]
    if probs:
        return probs
    if p["experiment"] != S.NAME:
        probs.append(pre + f"experiment {p['experiment']} != {S.NAME}")
    if p["freeze_sha256"] != freeze["freeze_sha256"]:
        probs.append(pre + "freeze hash differs from the gate's freeze record")
    if require_target and p["profile_role"] != "target":
        probs.append(pre + f"profile_role {p['profile_role']} is not target")
    g = p.get("gpu") or {}
    if not g.get("uuid") or not g.get("name"):
        probs.append(pre + "device identity missing (gpu name/uuid)")
    want_model = f"{freeze['model']['repo']}@{freeze['model']['revision']}"
    if p["model"] != want_model:
        probs.append(pre + f"model {p['model']} != {want_model}")
    if p["decision_rule_version"] != freeze["decision_rule_version"]:
        probs.append(pre + "decision rule version differs")
    if p["protocol_version"] != freeze["protocol_version"]:
        probs.append(pre + "protocol version differs")
    role = p["role"]
    if expect_role and role != expect_role:
        probs.append(pre + f"row role {role} != expected {expect_role}")
    r = (roles or {}).get(role) or {}
    if r.get("host") and p["machine"] != r["host"]:
        probs.append(pre + f"machine {p['machine']} is not the inventoried {role} ({r['host']})")
    if r.get("gpu_uuid") and g.get("uuid") != r["gpu_uuid"]:
        probs.append(pre + f"GPU {g.get('uuid')} is not the inventoried {role} GPU ({r['gpu_uuid']})")
    fm = freeze["machines"].get(p["machine"], {})
    if fm and fm.get("role") != role:
        probs.append(pre + f"freeze record says {p['machine']} is {fm.get('role')}, row says {role}")
    if fm and p["runtime_branch"] != fm.get("branch"):
        probs.append(pre + "runtime branch differs from the freeze record for this machine")
    cut, mc = p.get("cut"), p.get("memory_class")
    if cut and mc is not None and tuple(cut) != S.FIXED_CUTS.get(int(mc)) and tuple(cut) != S.U_MIN:
        probs.append(pre + f"cut {cut} is not frozen for class {mc}")
    return probs


def validate_dir(d, freeze, roles, expect_role, globs=("*.jsonl",), json_files=(), require_target=True):
    """Every row of every matching jsonl (request rows: those with prompt_id) and every named json needs provenance."""
    d = Path(d)
    probs = []
    n_rows = 0
    for g in globs:
        for f in sorted(d.glob(g)):
            if f.name.endswith(".rejected"):
                continue
            with open(f) as fh:
                for i, line in enumerate(fh):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "prompt_id" not in r:
                        continue
                    n_rows += 1
                    probs += check(r.get("provenance"), freeze, roles, expect_role, require_target, f"{f.name}:{i + 1}")
    for name in json_files:
        f = d / name
        if not f.exists():
            probs.append(f"{name} missing")
            continue
        obj = json.loads(f.read_text())
        probs += check(obj.get("provenance"), freeze, roles, expect_role, require_target, name)
    if n_rows == 0 and globs:
        probs.append(f"{d}: no request rows found")
    return probs


def stamp_jsonl(path, prov):
    """Add `provenance` to every request row of a jsonl written by an E0 driver (atomic rewrite)."""
    path = Path(path)
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "prompt_id" in r:
                r["provenance"] = prov
            rows.append(r)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    os.replace(tmp, path)
    return len(rows)


def stamp_json(path, prov):
    path = Path(path)
    obj = json.loads(path.read_text())
    obj["provenance"] = prov
    tmp = str(path) + ".tmp"
    Path(tmp).write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)
