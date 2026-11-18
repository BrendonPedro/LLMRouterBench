"""K16-HWFIT preflight: nothing is loaded and nothing is measured. Read-only towards everything it inspects.

Checks, in this order, and stops (exit 3) at the end if any blocking check failed:
  software     Python, torch with CUDA, transformers == 4.51.3 with its Qwen3 internals, safetensors, numpy,
               tokenizers (edge), and that the reused runtime imports
  revision     git commit, dirty files, reused E0/E1 runtime unchanged against the base tag, spec hash
  checkpoint   located, sizes, safetensors headers and SHA-256 against frozen/checkpoint_manifest.json
  GPU          the role's GPU type is present; the requested GPU (or, by default, a free one) is not occupied
               by another compute process; a bf16 matmul and an SDPA call run on it
  memory       currently free device memory (unified memory: MemAvailable) >= expected peak + margin
  network      server: the port is free on the loopback interface (the only one it may listen on);
               edge: the server answers a HELLO with this experiment's identity (through the SSH tunnel)
It never kills, signals or reconfigures any process. An occupied GPU is a STOP unless --allow-shared is given,
and then only if the memory still suffices.

  python -I k16_preflight.py --role server|edge --out preflight.json [--model-dir DIR] [--gpu INDEX|UUID]
                             [--port N | --server HOST:PORT] [--allow-shared]
"""
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # python -I: the script dir is not on sys.path
import k16_spec as S  # noqa: E402
import k16_memory as KM  # noqa: E402
import k16_checkpoint as KC  # noqa: E402
from k16_common import run, write_json, utc, versions, git_info, host_info, EXIT_STOP  # noqa: E402

GPU_QUERY = "index,uuid,name,memory.total,memory.used,memory.free"
APP_QUERY = "gpu_uuid,pid,used_memory"
PROBE = ("import json, torch\n"
         "p = torch.cuda.get_device_properties(0)\n"
         "try:\n    free = torch.cuda.mem_get_info(0)[0]\nexcept Exception:\n    free = -2**20\n"
         "x = torch.randn(64, 64, dtype=torch.bfloat16, device='cuda:0'); m = bool(torch.isfinite((x @ x).float()).all())\n"
         "q = torch.randn(1, 4, 8, 16, dtype=torch.bfloat16, device='cuda:0')\n"
         "a = bool(torch.isfinite(torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True).float()).all())\n"
         "print('K16PROBE=' + json.dumps({'name': p.name, 'uuid': str(getattr(p, 'uuid', '')), 'compute_cap': f'{p.major}.{p.minor}',\n"
         "      'total_mib': p.total_memory // 2**20, 'free_mib': free // 2**20, 'n_visible': torch.cuda.device_count(),\n"
         "      'bf16_matmul_finite': m, 'sdpa_bf16_finite': a}))\n")


def _num(x):
    x = x.strip()
    return int(float(x)) if x.replace(".", "", 1).isdigit() else None      # "[N/A]" (GB10 unified memory) -> None


def parse_gpus(text):
    """Rows of nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits."""
    out = []
    for line in (text or "").splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) < 6 or not f[0].isdigit():
            continue
        out.append({"index": int(f[0]), "uuid": f[1], "name": f[2], "total_mib": _num(f[3]), "used_mib": _num(f[4]), "free_mib": _num(f[5])})
    return out


def parse_apps(text):
    """Rows of nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits."""
    out = []
    for line in (text or "").splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) < 2 or not f[1].isdigit():
            continue
        out.append({"gpu_uuid": f[0], "pid": int(f[1]), "used_mib": _num(f[2]) if len(f) > 2 else None})
    return out


def meminfo(text=None):
    text = text if text is not None else Path("/proc/meminfo").read_text()
    kb = {l.split(":")[0]: int(l.split()[1]) for l in text.splitlines() if ":" in l and l.split()[1].isdigit()}
    return {"total_mib": kb.get("MemTotal", 0) // 1024, "available_mib": kb.get("MemAvailable", 0) // 1024}


def proc_owner(pid):
    out = run(["ps", "-o", "user=,etimes=,args=", "-p", str(pid)])
    f = out.split(None, 2) if not out.startswith("ERROR") else []
    return {"user": f[0] if f else None, "running_s": int(f[1]) if len(f) > 1 and f[1].isdigit() else None,
            "command": f[2][:160] if len(f) > 2 else None}


def select_gpu(gpus, apps, need_mib, marker, requested=None, allow_shared=False, unified_available_mib=None, owners=None):
    """Pick the GPU for a role, or explain why none may be used. Pure function of the inventory.
    marker: substring the GPU name must contain (the role's hardware). requested: index or UUID, else automatic.
    A GPU with another compute process is 'occupied'. Free memory of a unified-memory GPU is MemAvailable."""
    owners = owners or {}
    res = {"ok": False, "verdict": None, "gpu": None, "need_mib": need_mib, "marker": marker, "requested": requested,
           "allow_shared": allow_shared, "considered": [], "reasons": []}
    cands = [g for g in gpus if marker.lower() in g["name"].lower()]
    if requested is not None:
        req = [g for g in gpus if str(g["index"]) == str(requested) or g["uuid"] == str(requested)]
        if not req:
            res.update(verdict="STOP_GPU_NOT_FOUND", reasons=[f"requested GPU {requested} is not on this machine"])
            return res
        if req[0] not in cands:
            res.update(verdict="STOP_ROLE_MISMATCH", reasons=[f"requested GPU {requested} is a {req[0]['name']}, not a {marker} device"])
            return res
        cands = req
    if not cands:
        res.update(verdict="STOP_ROLE_MISMATCH", reasons=[f"no GPU whose name contains {marker!r} on this machine: {[g['name'] for g in gpus]}"])
        return res
    for g in cands:
        others = [dict(a, **owners.get(a["pid"], {})) for a in apps if a["gpu_uuid"] == g["uuid"]]
        unified = g["total_mib"] is None
        free = unified_available_mib if unified else g["free_mib"]
        c = dict(g, unified_memory=unified, free_now_mib=free, other_processes=others, occupied=bool(others),
                 memory_ok=free is not None and free >= need_mib)
        c["eligible"] = c["memory_ok"] and (not c["occupied"] or allow_shared)
        if c["occupied"]:
            res["reasons"].append(f"GPU {g['index']} ({g['name']}) is occupied by " + "; ".join(
                f"pid {o['pid']} user {o.get('user')} {o.get('used_mib')} MiB" for o in others))
        if not c["memory_ok"]:
            res["reasons"].append(f"GPU {g['index']} ({g['name']}): free {free} MiB < required {need_mib} MiB")
        res["considered"].append(c)
    ok = sorted([c for c in res["considered"] if c["eligible"]], key=lambda c: (c["occupied"], -(c["free_now_mib"] or 0), c["index"]))
    if ok:
        res.update(ok=True, verdict="OK", gpu=ok[0])
    elif any(c["memory_ok"] for c in res["considered"]):
        res["verdict"] = "STOP_GPU_OCCUPIED"
    else:
        res["verdict"] = "STOP_MEMORY_INSUFFICIENT"
    return res


def inventory():
    g = run(["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"])
    a = run(["nvidia-smi", f"--query-compute-apps={APP_QUERY}", "--format=csv,noheader,nounits"])
    gpus, apps = parse_gpus(g), parse_apps("" if a.startswith("ERROR") else a)
    return {"gpus": gpus, "apps": apps, "owners": {x["pid"]: proc_owner(x["pid"]) for x in apps}, "meminfo": meminfo(),
            "nvidia_smi_error": g if g.startswith("ERROR") else None, "apps_error": a if a.startswith("ERROR") else None}


def memory_gate(stage, gpu_uuid=None, marker=None, allow_shared=False, inv=None):
    """Re-read free memory now and compare with the stage's requirement. Called right before every load."""
    inv = inv or inventory()
    need = KM.required_free_mib(stage)
    marker = marker or S.ROLES["server" if stage == "server" else "edge"]["gpu_name_marker"]
    sel = select_gpu(inv["gpus"], inv["apps"], need, marker, requested=gpu_uuid, allow_shared=allow_shared,
                     unified_available_mib=inv["meminfo"]["available_mib"], owners=inv["owners"])
    return {"stage": stage, "ok": sel["ok"], "verdict": sel["verdict"], "need_mib": need, "free_now_mib": (sel["gpu"] or {}).get("free_now_mib"),
            "reasons": sel["reasons"], "utc": utc()}


def device_probe(gpu, n_gpus):
    """A separate process touches the chosen GPU (CUDA_VISIBLE_DEVICES = its UUID when the machine has several)."""
    env = dict(os.environ)
    if n_gpus > 1:
        env["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
    try:
        r = subprocess.run([sys.executable, "-I", "-c", PROBE], capture_output=True, text=True, timeout=300, env=env)
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    line = next((l for l in r.stdout.splitlines() if l.startswith("K16PROBE=")), None)
    if line is None:
        return {"ok": False, "error": (r.stderr or r.stdout).strip()[-600:]}
    p = json.loads(line[len("K16PROBE="):])
    same = (not p["uuid"]) or (not gpu["uuid"]) or p["uuid"].lower().replace("gpu-", "") == gpu["uuid"].lower().replace("gpu-", "")
    p.update(ok=bool(p["bf16_matmul_finite"] and p["sdpa_bf16_finite"] and same), is_selected_gpu=same)
    return p


def software_checks(role, dev=False):
    v, checks = versions(), []
    add = lambda name, ok, detail="": checks.append({"check": name, "ok": bool(ok), "detail": str(detail), "blocking": True})
    add("python >= 3.9", sys.version_info >= (3, 9), v["python"])
    add("torch importable", v.get("torch"), v.get("torch") or v.get("torch_error"))
    add(f"transformers == {S.TRANSFORMERS_REQUIRED}", v.get("transformers") == S.TRANSFORMERS_REQUIRED, v.get("transformers") or v.get("transformers_error"))
    add("safetensors importable", v.get("safetensors"), v.get("safetensors") or v.get("safetensors_error"))
    add("numpy importable", v.get("numpy"), v.get("numpy") or v.get("numpy_error"))
    if role == "edge":
        add("tokenizers importable", v.get("tokenizers"), v.get("tokenizers") or v.get("tokenizers_error"))
    if not dev:
        add("torch sees CUDA", v.get("cuda_available"), f"torch {v.get('torch')} cuda {v.get('torch_cuda')}")
    try:
        from transformers.models.qwen3 import modeling_qwen3 as mq
        ok = hasattr(mq.Qwen3Model, "_update_causal_mask") and hasattr(mq, "Qwen3RotaryEmbedding")
        add("transformers Qwen3 internals used by the runtime", ok, "Qwen3Model._update_causal_mask, Qwen3RotaryEmbedding")
        import k16_runtime  # noqa: F401
        add("reused runtime imports (split_runtime, e1_segment, e1_protocol, e1_attention)", True)
    except Exception as e:
        add("reused runtime imports (split_runtime, e1_segment, e1_protocol, e1_attention)", False, f"{type(e).__name__}: {e}")
    return v, checks


def collect(role, out_dir, model_dir=None, gpu=None, server=None, port=None, allow_shared=False, dev=False):
    rec = {"experiment": S.NAME, "status": S.STATUS, "role": role, "role_spec": S.ROLES[role], "utc": utc(), "dev": dev, "host": host_info(),
           "spec_hash": S.spec_hash(dev), "checks": []}
    add = lambda name, ok, detail="", blocking=True: rec["checks"].append({"check": name, "ok": bool(ok), "detail": str(detail), "blocking": blocking})
    rec["versions"], sw = software_checks(role, dev)
    rec["checks"] += sw
    rec["git"] = git_info(S.K16_DIR.parents[1], S.BASE_TAG)
    add("code is a git commit", rec["git"]["commit"], rec["git"]["commit"], blocking=not dev)
    add("no uncommitted change in the experiment or the reused runtime", not rec["git"]["dirty"], rec["git"]["dirty"][:8], blocking=not dev)
    add("reused E0/E1 runtime equals the base tag", rec["git"]["runtime_unchanged_vs_base"] is not False,
        f"{S.BASE_TAG}: {rec['git']['runtime_unchanged_vs_base']} (None = tag not present here; the spec hash still pins every file)")
    # checkpoint
    d, tried = KC.locate(model_dir)
    add("checkpoint directory found", d, d or f"looked in {tried}; set K16_MODEL_DIR")
    if d:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        cache = Path(out_dir).parent / "checkpoint_sha256_cache.json"
        print(f"[preflight] verifying checkpoint files for the {role} in {d} (SHA-256; cached after the first run) ...", flush=True)
        rec["checkpoint"] = KC.verify(d, role, cache_file=cache, dev=dev)
        add("checkpoint files match the frozen manifest (size, safetensors headers, SHA-256)", rec["checkpoint"]["ok"], rec["checkpoint"]["problems"][:6])
    # memory expectation and GPU
    stage = "server" if role == "server" else "edge_reference"       # the edge's largest stage
    rec["expected_memory"] = {st: {k: (v if k == "stage" else round(v / KM.MIB, 1)) for k, v in KM.expected(st).items()} for st in KM.STAGES
                              if (st == "server") == (role == "server")}
    if dev:
        rec["gpu"] = {"ok": True, "verdict": "DEV_CPU", "gpu": None}
    else:
        inv = inventory()
        rec["inventory"] = inv
        add("nvidia-smi lists GPUs", inv["gpus"], inv["nvidia_smi_error"] or f"{len(inv['gpus'])} GPU(s)")
        need = KM.required_free_mib(stage)
        sel = select_gpu(inv["gpus"], inv["apps"], need, S.ROLES[role]["gpu_name_marker"], requested=gpu, allow_shared=allow_shared,
                         unified_available_mib=inv["meminfo"]["available_mib"], owners=inv["owners"])
        rec["gpu"] = sel
        add(f"a {S.ROLES[role]['gpu_name_marker']} GPU is usable: not occupied, free memory >= {need} MiB", sel["ok"], sel["verdict"] + " " + "; ".join(sel["reasons"]))
        if sel["ok"]:
            if sel["gpu"]["occupied"]:
                add("GPU is shared with another process (allowed by --allow-shared; timing rows will be flagged)", True, sel["reasons"], blocking=False)
            rec["probe"] = device_probe(sel["gpu"], len(inv["gpus"]))
            add("bf16 matmul and SDPA run on the selected GPU", rec["probe"].get("ok"), {k: v for k, v in rec["probe"].items() if k != "ok"})
    # disk and network
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(out_dir).free / 1e9
    add("output directory has >= 2 GB free", free_gb >= 2, f"{free_gb:.1f} GB free in {out_dir}")
    if role == "server":
        try:
            s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind((S.LISTEN_HOST, port)); s.close()
            add(f"port {port} is free on {S.LISTEN_HOST}", True)
        except OSError as e:
            add(f"port {port} is free on {S.LISTEN_HOST}", False, e)
        if not dev:
            ram = meminfo()["available_mib"]
            add(f"host RAM available >= {KM.HOST_RAM_MIN_MIB} MiB (staging buffers while loading)", ram >= KM.HOST_RAM_MIN_MIB, f"{ram} MiB available")
    elif server:
        name = f"server {server} answers HELLO as this experiment (through the tunnel)"
        try:
            import k16_runtime as R
            si = R.ping(server, d, rec["spec_hash"], S.params(dev))
            rec["server_identity"] = si
            add(name, True, f"{si.get('host')} {(si.get('device') or {}).get('name')}")
        except Exception as e:
            add(name, False, f"{type(e).__name__}: {e}. Start the ASUS command first, wait for its READY line, and check the SSH tunnel.")
    rec["blockers"] = [c for c in rec["checks"] if c["blocking"] and not c["ok"]]
    rec["verdict"] = "CONTINUE" if not rec["blockers"] else "STOP"
    return rec


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role", choices=["server", "edge"], required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model-dir", default=None); ap.add_argument("--gpu", default=None)
    ap.add_argument("--server", default=None); ap.add_argument("--port", type=int, default=S.DEFAULT_PORT)
    ap.add_argument("--allow-shared", action="store_true"); ap.add_argument("--dev", action="store_true")
    a = ap.parse_args(argv)
    rec = collect(a.role, Path(a.out).parent, a.model_dir, a.gpu, a.server, a.port, a.allow_shared, a.dev)
    write_json(a.out, rec)
    for c in rec["checks"]:
        print(f"[preflight] {'ok  ' if c['ok'] else ('FAIL' if c['blocking'] else 'note')} {c['check']}" + (f"  ({c['detail']})" if c["detail"] and (not c["ok"] or not c["blocking"]) else ""))
    print(f"[preflight] {a.role}: {rec['verdict']}  spec {rec['spec_hash'][:12]}  -> {a.out}", flush=True)
    sys.exit(0 if rec["verdict"] == "CONTINUE" else EXIT_STOP)


if __name__ == "__main__":
    main()
