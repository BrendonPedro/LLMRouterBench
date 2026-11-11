"""E1-OPT runtime freeze record: writer and checker (PREREG_E1OPT §4.3).

The freeze record is an INPUT to every later stage. It is written once (Stage 1a) from the Stage-0 inventory and
the role record, before any GPU executes E1 code. Later stages verify it (`check`) and refuse to run on any
mismatch. Nothing here ever regenerates or edits an existing record: a new record is a new file and a new
commit (PREREG §4.4).

freeze_sha256 = sha256 of the canonical JSON of the record without the "freeze_sha256" field. It is exchanged
in HELLO, stamped into every profile row (e1_provenance) and compared by the gates.
"""
import hashlib
import json
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))   # python -I: the script dir is not on sys.path
import e1_spec as S

E1_DIR = S.E1_DIR
REPO = E1_DIR.parents[1]
SOURCE_GLOBS = ("e1_*.py", "tests/*.py", "reanalysis/*.py")
DOC_FILES = ("PREREG_E1OPT.md",)
E0_DEPENDENCIES = ("e0_spec.py", "e0_r.py", "e0_latency.py", "e0_instrument.py", "e0_memory.py", "e0_f.py", "e0_q.py",
                   "split_runtime.py", "frozen/workload.json", "frozen/q2_keys.json")
STRICT_VERSION_KEYS = ("torch", "torch_cuda", "transformers", "flash_attn", "python")
RECORD_VERSION = 3


class FreezeMismatch(RuntimeError):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def hash_set(e1_dir=E1_DIR):
    """Every file whose content can affect E1 execution or its decision, relative to the E1 directory."""
    files = set()
    for g in SOURCE_GLOBS:
        files |= {p.relative_to(e1_dir).as_posix() for p in Path(e1_dir).glob(g) if p.is_file()}
    files |= {d for d in DOC_FILES if (Path(e1_dir) / d).exists()}
    return {f: sha256_file(Path(e1_dir) / f) for f in sorted(files)}


def e0_hashes(e0_dir=S.E0_DIR):
    return {f: sha256_file(Path(e0_dir) / f) for f in E0_DEPENDENCIES if (Path(e0_dir) / f).exists()}


def git(args, cwd=REPO):
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # pragma: no cover
        return f"ERROR {e}"


def git_head(repo=REPO):
    return git(["rev-parse", "HEAD"], repo)


def git_tag_commit(tag, repo=REPO):
    out = git(["rev-parse", "--verify", "--quiet", f"{tag}^{{commit}}"], repo)
    return out or None


def versions():
    v = {"python": platform.python_version(), "os": platform.platform(), "machine": platform.machine()}
    try:
        import torch
        v["torch"] = torch.__version__
        v["torch_cuda"] = torch.version.cuda
        v["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as e:
        v["torch"] = None; v["torch_cuda"] = None; v["cuda_available"] = False; v["torch_error"] = str(e)
    for mod, key in (("transformers", "transformers"), ("flash_attn", "flash_attn"), ("vllm", "vllm"), ("numpy", "numpy"),
                     ("pynvml", "pynvml")):
        try:
            m = __import__(mod)
            v[key] = getattr(m, "__version__", "present")
        except Exception:
            v[key] = None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=20).stdout.strip().splitlines()
        v["nvidia_driver"] = out[0].strip() if out else None
    except Exception:
        v["nvidia_driver"] = None
    return v


def branch_for_device(dev):
    """PREREG §4.2 compatibility rule from inventory facts only (never from measured speed)."""
    cc = dev.get("compute_cap")
    fa = dev.get("flash_attn") or {}
    reasons = []
    if cc is not None and float(cc) < S.MIN_COMPUTE_CAPABILITY:
        reasons.append(f"compute capability {cc} < {S.MIN_COMPUTE_CAPABILITY}: bf16/FlashAttention-2 unsupported")
    if not fa.get("importable"):
        reasons.append("flash_attn not importable on this machine: " + str(fa.get("error", "no wheel")))
    elif not fa.get("has_flash_attn_with_kvcache", fa.get("has_kvcache_fn", True)):
        reasons.append("flash_attn present but flash_attn_with_kvcache missing")
    attention = "sdpa_static_fallback" if reasons else "flash_attn_with_kvcache"
    return {"attention": attention, "attention_reason": "; ".join(reasons) or "flash_attn importable for this compute capability",
            "cuda_graphs": "on", "cuda_graphs_reason": "planned; a capture failure at Stage 1b is a compatibility probe that "
            "produces a NEW freeze record with off_capture_failed and the error text",
            "torch_compile": S.TORCH_COMPILE}


def build(inventory, roles, e1_dir=E1_DIR, notes=None, now=None):
    """inventory: {hostname: inventory dict}; roles: the Stage-0 role record (e1_inventory.assign_roles)."""
    if not roles.get("feasible"):
        raise ValueError("roles record says the pair is not feasible; nothing to freeze")
    machines = {}
    for role in ("server", "edge"):
        r = roles[role]
        inv = inventory[r["host"]]
        gpu = next((g for g in inv.get("gpus", []) if g.get("index") == r["gpu_index"]), None)
        if gpu is None:
            raise ValueError(f"{role}: GPU index {r['gpu_index']} not in the inventory of {r['host']}")
        dev = dict(gpu, flash_attn=inv.get("flash_attn", {}))
        machines[r["host"]] = {"role": role, "gpu": {"index": gpu["index"], "name": gpu.get("name"), "uuid": gpu.get("uuid"),
                                                    "memory_mib": gpu.get("memory_mib"), "compute_cap": gpu.get("compute_cap")},
                               "n_gpus_on_machine": len(inv.get("gpus", [])), "versions": inv.get("versions", {}),
                               "branch": branch_for_device(dev)}
    rec = {
        "experiment": S.NAME, "record_version": RECORD_VERSION, "decision_rule_version": S.DECISION_RULE_VERSION,
        "frozen_at_utc": now or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_head(), "git_tag": None, "e1_dir": Path(e1_dir).resolve().relative_to(REPO).as_posix(),
        "file_sha256": hash_set(e1_dir), "prereg_sha256": sha256_file(Path(e1_dir) / "PREREG_E1OPT.md"),
        "e0": {"validated_commit": S.E0_VALIDATED_COMMIT, "tag": "e0-prereg-v1", "file_sha256": e0_hashes(),
               "s1_q3_scores_sha256": S.E0.S1_Q3_SHA256, "s1_sample_sha256": S.E0.S1_SAMPLE_SHA256,
               "workload_ids_sha256": S.E0.WORKLOAD_IDS_SHA256, "stage1_archive_sha256": S.E0_STAGE1_SHA256,
               "stage2_archive_sha256": S.E0_STAGE2_SHA256},
        "protocol_version": S.PROTOCOL_VERSION, "wire_header_bytes": S.E0.WIRE_HEADER_BYTES,
        "model": {"repo": S.E0.SPLIT_REPO, "revision": S.E0.SPLIT_REVISION, "dtype": "bfloat16", "wire_dtype": S.E0.WIRE_DTYPE,
                  "config": S.E0.QWEN3_32B_CONFIG},
        "transformers_required": S.E0.TRANSFORMERS_VERSION, "vllm_competitor_pin": S.E0.VLLM_VERSION,
        "cuts": {str(m): list(v) for m, v in S.FIXED_CUTS.items()}, "u_min": list(S.U_MIN), "primary_classes": list(S.PRIMARY_CLASSES),
        "one_server_gpu": S.ONE_SERVER_GPU, "server_parallelism": "none (no TP, no PP, no sharding)",
        "static_cache": {"max_len_tokens": S.E0.L_F, "prefill_chunk": S.E0.PREFILL_CHUNK, "kv_dtype": S.E0.KV_DTYPE,
                         "layout": "[1, n_kv, max_len, head_dim] per owned layer, written in place at cache_position",
                         "mode_measured": "static", "mode_reference": "prefix"},
        "buffer_strategy": {"wire": "host staging buffers allocated once for one prefill chunk, pinned on CUDA",
                            "graph": "static [1,1,hidden] in/out and [1] position buffers per segment",
                            "device_extra_bytes": __import__("e1_memory").extra_bytes_e1()},
        "ownership": {"edge": ["embed_tokens", "layers [0,k)", "layers [L-j,L)", "final_norm", "lm_head", "sampling", "tokenizer"],
                      "server": ["layers [k,L-j)"], "wire_content": "bf16 residual stream only",
                      "never_on_wire": ["raw text", "token ids", "logits", "sampled tokens", "KV entries"], "no_raw_text_iff": "k >= 1 and j >= 1"},
        "seeds": {"bootstrap": S.E0.R_BOOT_SEED, "n_boot": S.E0.R_BOOT, "order": S.E0.ORDER_SEED, "q2": S.E0.Q2_SEED,
                  "heldout": S.E0.HELDOUT_SEED, "synthetic_activations_torch_seed": 0},
        "decision_rule": {"margin_pt": S.R_MARGIN_PT, "delta_round_pt": S.DECISION_ROUND_PT, "ci_rule": "CI lower bound > 0 (strict)",
                          "e0_raw_flag_reported": True, "gate_code": "deployment_e0/e0_r.py", "gate_code_commit": S.E0_VALIDATED_COMMIT,
                          "gate_code_sha256": e0_hashes().get("e0_r.py"), "segment_discount": S.UB_SEGMENT_DISCOUNT},
        "roles": roles, "machines": machines, "notes": notes or "",
    }
    rec["freeze_sha256"] = freeze_sha256(rec)
    return rec


def freeze_sha256(rec):
    return hashlib.sha256(canonical({k: v for k, v in rec.items() if k != "freeze_sha256"}).encode()).hexdigest()


def write(rec, path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists: a freeze record is never overwritten; write a new file (PREREG §4.4)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    return path


def load(path):
    rec = json.loads(Path(path).read_text())
    if rec.get("freeze_sha256") != freeze_sha256(rec):
        raise FreezeMismatch(f"{path}: freeze_sha256 does not match the record content (edited or corrupted)")
    return rec


def check(path, e1_dir=E1_DIR, host=None, role=None, live_versions=None, require_tag=True, repo=REPO):
    """Verify a freeze record against this checkout and this machine. Raises FreezeMismatch listing everything
    that differs; returns the record otherwise. Never writes."""
    rec = load(path)
    problems = []
    if rec.get("experiment") != S.NAME:
        problems.append(f"experiment {rec.get('experiment')} != {S.NAME}")
    if rec.get("decision_rule_version") != S.DECISION_RULE_VERSION:
        problems.append(f"decision rule {rec.get('decision_rule_version')} != {S.DECISION_RULE_VERSION}")
    if rec.get("protocol_version") != S.PROTOCOL_VERSION:
        problems.append("protocol version differs")
    live = hash_set(e1_dir)
    for f, h in rec["file_sha256"].items():
        if live.get(f) != h:
            problems.append(f"file changed since freeze: {f}" if f in live else f"file missing since freeze: {f}")
    for f in live:
        if f not in rec["file_sha256"]:
            problems.append(f"file not covered by the freeze: {f}")
    if sha256_file(Path(e1_dir) / "PREREG_E1OPT.md") != rec["prereg_sha256"]:
        problems.append("PREREG_E1OPT.md changed since freeze")
    for f, h in rec["e0"]["file_sha256"].items():
        if e0_hashes().get(f) != h:
            problems.append(f"E0 dependency changed: {f}")
    head = git_head(repo)
    if head != rec["git_commit"]:
        # a later commit is acceptable only if it changed nothing outside frozen/ in the E1 directory
        diff = git(["diff", "--name-only", rec["git_commit"], head, "--", rec["e1_dir"]], repo)
        touched = [l for l in diff.splitlines() if l and not l.startswith(rec["e1_dir"] + "/frozen/")]
        if diff.startswith("ERROR") or touched:
            problems.append(f"git HEAD {head[:12]} differs from the frozen commit {rec['git_commit'][:12]} outside frozen/: {touched[:5]}")
    if require_tag:
        tag = rec.get("git_tag")
        if not tag:
            problems.append("record carries no git_tag (PREREG: the experiment is frozen by repository tag)")
        else:
            tc = git_tag_commit(tag, repo)
            if tc != rec["git_commit"]:
                problems.append(f"tag {tag} -> {tc} != frozen commit {rec['git_commit']}")
    host = host or socket.gethostname()
    m = rec["machines"].get(host)
    if m is None:
        problems.append(f"this machine ({host}) is not in the freeze record ({sorted(rec['machines'])})")
    else:
        if role and m["role"] != role:
            problems.append(f"this machine is frozen as {m['role']}, asked to act as {role}")
        lv = live_versions if live_versions is not None else versions()
        for k in STRICT_VERSION_KEYS:
            if m["versions"].get(k) != lv.get(k):
                problems.append(f"{host}: {k} {lv.get(k)} != frozen {m['versions'].get(k)}")
        if m["branch"].get("torch_compile") is not False:
            problems.append("torch_compile is not False in the record")
        if m["branch"].get("attention") not in S.ATTENTION_BRANCHES or m["branch"].get("cuda_graphs") not in S.GRAPH_BRANCHES:
            problems.append(f"unknown runtime branch {m['branch']}")
    if problems:
        raise FreezeMismatch("freeze check FAILED:\n  - " + "\n  - ".join(problems))
    return rec


def machine_entry(rec, host=None):
    host = host or socket.gethostname()
    return rec["machines"][host]


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("freeze")
    w.add_argument("--inventory", required=True, help="merged inventory JSON ({hostname: inventory})")
    w.add_argument("--roles", required=True); w.add_argument("--out", required=True)
    w.add_argument("--tag", default=None, help="git tag the record will be frozen under (e1opt-prereg-v1)")
    w.add_argument("--notes", default="")
    c = sub.add_parser("check-freeze")
    c.add_argument("--freeze", required=True); c.add_argument("--role", default=None); c.add_argument("--host", default=None)
    c.add_argument("--no-tag", action="store_true", help="dry runs before the tag exists (never for a stage)")
    a = ap.parse_args(argv)
    if a.cmd == "freeze":
        inv = json.loads(Path(a.inventory).read_text()); roles = json.loads(Path(a.roles).read_text())
        rec = build(inv, roles, notes=a.notes)
        if a.tag:
            rec["git_tag"] = a.tag; rec["freeze_sha256"] = freeze_sha256(rec)
        write(rec, a.out)
        print(json.dumps({"out": a.out, "freeze_sha256": rec["freeze_sha256"], "git_commit": rec["git_commit"],
                          "files": len(rec["file_sha256"])}, indent=1))
    else:
        try:
            rec = check(a.freeze, host=a.host, role=a.role, require_tag=not a.no_tag)
        except FreezeMismatch as e:
            print(str(e), file=sys.stderr); sys.exit(2)
        print("freeze OK", rec["freeze_sha256"])


if __name__ == "__main__":
    main()
