"""GEMMA-SPLIT Stage 0a/0b: machine inventory (facts only) and role verification (PREREG §3, §4.1).

`collect` records facts: no model is loaded, no GPU kernel runs, nothing is timed, nothing on the machine is
modified. torch is imported only if this interpreter has it (device enumeration). Package versions come from
installed metadata, so heavy packages are not imported. Environment variables are filtered to the families
that can change execution and redacted by name; process command lines and URLs are redacted (gs_common).

Roles are FIXED by the PI (DGX Spark = edge/client, asus-ESC4000-E11 = server/backend). `verify_role` checks
the identity markers in gs_spec against the inventory; a mismatch stops Stage 0 for review. It never swaps roles.

  python -I gs_inventory.py collect --role edge|server --peer <other host or IP> --out-dir capture/<role>
  python -I gs_inventory.py roles --capture-dir capture
"""
import os
import platform
import re
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # python -I: the script dir is not on sys.path
import gs_common as C  # noqa: E402
import gs_spec as S  # noqa: E402

ENV_PREFIXES = ("CUDA", "NVIDIA", "NCCL", "TORCH", "PYTORCH", "HF_", "HUGGINGFACE", "TRANSFORMERS", "TOKENIZERS",
                "SAFETENSORS", "ACCELERATE", "VLLM", "TRITON", "XLA", "OMP_", "MKL_", "OPENBLAS", "GLOO", "UCX", "NVSHMEM",
                "MASTER_", "WORLD_SIZE", "RANK", "LOCAL_RANK", "LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHON", "VIRTUAL_ENV",
                "CONDA", "PATH", "TMPDIR", "XDG_CACHE_HOME")
ENV_CONTAINS = ("GEMMA", "SPLIT", "MODEL", "LLM", "EDGE", "SERVER", "PORT", "HOST")
PACKAGES = ("torch", "transformers", "accelerate", "safetensors", "tokenizers", "huggingface-hub", "numpy", "flash-attn",
            "flash_attn", "vllm", "triton", "xformers", "bitsandbytes", "nvidia-ml-py", "pynvml", "sentencepiece",
            "protobuf", "fastapi", "uvicorn", "grpcio", "pyzmq", "requests", "ray")
SMI_GPU_FIELDS = "index,name,uuid,memory.total,memory.used,driver_version,compute_cap,pci.bus_id,power.limit"
SMI_APP_FIELDS = "pid,process_name,used_memory,gpu_uuid"
NA = ("[N/A]", "N/A", "[Not Supported]", "Not Supported", "")


def _num(s):
    m = re.match(r"\s*([0-9]+(?:\.[0-9]+)?)", s or "")
    return float(m.group(1)) if m else None


def parse_smi_gpus(text):
    """Rows of `nvidia-smi --query-gpu=<SMI_GPU_FIELDS> --format=csv,noheader`. Unified-memory devices (GB10)
    report memory as [N/A] / Not Supported: kept as None with the raw string, never guessed."""
    gpus = []
    for line in (text or "").splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 8 or not p[0].isdigit():
            continue
        tot, used = _num(p[3]), _num(p[4])
        gpus.append({"index": int(p[0]), "name": p[1], "uuid": p[2],
                     "memory_total_mib": int(tot) if tot is not None and p[3] not in NA else None,
                     "memory_used_mib": int(used) if used is not None and p[4] not in NA else None,
                     "memory_raw": [p[3], p[4]], "driver": p[5], "compute_cap": _num(p[6]) if p[6] not in NA else None,
                     "pci_bus_id": p[7], "power_limit_w": _num(p[8]) if len(p) > 8 and p[8] not in NA else None})
    return gpus


def parse_smi_apps(text):
    apps = []
    for line in (text or "").splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 4 or not p[0].isdigit():
            continue
        used = _num(p[2])
        apps.append({"pid": int(p[0]), "process_name": p[1], "used_memory_mib": int(used) if used is not None and p[2] not in NA else None,
                     "gpu_uuid": p[3]})
    return apps


def parse_meminfo(text):
    out = {}
    for line in (text or "").splitlines():
        m = re.match(r"(\w+):\s+(\d+)\s*kB", line)
        if m:
            out[m.group(1)] = int(m.group(2)) * 1024
    return {k: out.get(k) for k in ("MemTotal", "MemAvailable", "MemFree", "SwapTotal", "SwapFree")}


def parse_ip_links(text):
    nics = []
    for line in (text or "").splitlines():
        f = line.split()
        if not f or f[0] == "lo":
            continue
        nics.append({"name": f[0].split("@")[0], "state": f[1] if len(f) > 1 else None})
    return nics


def parse_ethtool_speed(text):
    m = re.search(r"Speed:\s*(\d+)\s*Mb/s", text or "")
    return int(m.group(1)) if m else None


def filtered_env(env=None):
    env = dict(os.environ if env is None else env)
    keep = {k: v for k, v in env.items() if k.upper().startswith(ENV_PREFIXES) or any(s in k.upper() for s in ENV_CONTAINS)}
    return dict(sorted(C.redact_env(keep).items()))


def package_versions():
    try:
        from importlib import metadata
    except ImportError:  # pragma: no cover
        return {}
    out = {}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except Exception:
            out[p] = None
    return out


def torch_facts():
    try:
        import torch
    except Exception as e:
        return {"importable": False, "error": f"{type(e).__name__}: {e}"}
    d = {"importable": True, "version": torch.__version__, "cuda": torch.version.cuda,
         "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
         "cuda_available": bool(torch.cuda.is_available()), "devices": []}
    if d["cuda_available"]:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            d["devices"].append({"index": i, "name": p.name, "uuid": str(getattr(p, "uuid", "")),
                                 "total_mib": p.total_memory // (1 << 20), "compute_cap": float(f"{p.major}.{p.minor}"),
                                 "is_integrated": bool(getattr(p, "is_integrated", False))})
    return d


def versions_from(inv):
    """The strict version keys the freeze checker compares (gs_freeze.STRICT_VERSION_KEYS)."""
    t = inv.get("torch") or {}
    pk = inv.get("packages") or {}
    return {"python": inv.get("python", {}).get("version"), "torch": t.get("version"), "torch_cuda": t.get("cuda"),
            "transformers": pk.get("transformers"), "accelerate": pk.get("accelerate"), "safetensors": pk.get("safetensors"),
            "tokenizers": pk.get("tokenizers"), "flash_attn": pk.get("flash-attn") or pk.get("flash_attn"), "vllm": pk.get("vllm"),
            "driver": (inv.get("gpus") or [{}])[0].get("driver") if inv.get("gpus") else None}


def live_versions():
    inv = {"python": {"version": platform.python_version()}, "torch": torch_facts(), "packages": package_versions(),
           "gpus": parse_smi_gpus(C.run(["nvidia-smi", f"--query-gpu={SMI_GPU_FIELDS}", "--format=csv,noheader"])["out"])}
    return versions_from(inv)


def collect(role, peer=None):
    if role not in S.ROLES:
        raise ValueError(f"role must be one of {S.ROLES}")
    smi = C.run(["nvidia-smi", f"--query-gpu={SMI_GPU_FIELDS}", "--format=csv,noheader"])
    apps = C.run(["nvidia-smi", f"--query-compute-apps={SMI_APP_FIELDS}", "--format=csv,noheader"])
    links = C.run(["ip", "-br", "link"])
    nics = parse_ip_links(links["out"])
    for n in nics:
        eth = C.run(["ethtool", n["name"]], timeout=15)
        n["speed_mbps"] = parse_ethtool_speed(eth["out"])
        mtu = C.run(["cat", f"/sys/class/net/{n['name']}/mtu"], timeout=5)
        n["mtu"] = int(mtu["out"]) if mtu["out"].isdigit() else None
    meminfo = C.run(["cat", "/proc/meminfo"], timeout=5)
    try:   # name resolution only (no traffic to the peer); `ip route get` needs an address
        peer_ip = socket.gethostbyname(peer) if peer else None
        route = C.run(["ip", "route", "get", peer_ip]) if peer_ip else None
    except OSError as e:
        peer_ip, route = None, {"cmd": f"resolve {peer}", "rc": None, "out": "", "err": f"{type(e).__name__}: {e}"}
    inv = {
        "schema": "gemma-split-inventory/1", "experiment": S.NAME, "declared_role": role, "role_machine": S.ROLE_MACHINE[role],
        "hostname": socket.gethostname(), "fqdn": socket.getfqdn(), "utc": C.utc(),
        "uname": C.run(["uname", "-a"])["out"], "arch": platform.machine(),
        "os_release": C.run(["cat", "/etc/os-release"], timeout=5)["out"], "dgx_release": C.run(["cat", "/etc/dgx-release"], timeout=5)["out"],
        "cpu": C.run("lscpu 2>/dev/null | grep -E 'Model name|^CPU\\(s\\)|Architecture|Vendor ID'")["out"],
        "memory": parse_meminfo(meminfo["out"]),
        "gpus": parse_smi_gpus(smi["out"]), "nvidia_smi_query": smi, "nvidia_smi_full": C.run(["nvidia-smi", "-q"], timeout=60),
        "nvidia_smi_topo": C.run(["nvidia-smi", "topo", "-m"])["out"], "compute_apps": parse_smi_apps(apps["out"]),
        "cuda_toolkit": C.run("nvcc --version 2>/dev/null | tail -2")["out"],
        "cuda_version_json": C.run(["cat", "/usr/local/cuda/version.json"], timeout=5)["out"],
        "python": {"version": platform.python_version(), "executable": sys.executable, "prefix": sys.prefix,
                   "implementation": platform.python_implementation()},
        "pip_freeze": C.redact_text(C.run([sys.executable, "-m", "pip", "freeze", "--all"], timeout=180)["out"]),
        "packages": package_versions(), "torch": torch_facts(), "env": filtered_env(),
        "nics": nics, "ip_addr": C.run(["ip", "-br", "addr"])["out"], "peer": peer, "peer_resolved": peer_ip, "route_to_peer": route,
        "listening_tcp": C.run("ss -ltnp 2>/dev/null")["out"],
        "clock": {"timedatectl": C.run(["timedatectl", "show", "-p", "NTPSynchronized", "-p", "Timezone"], timeout=10)["out"],
                  "chrony": C.run(["chronyc", "tracking"], timeout=10)["out"]},
    }
    inv["versions"] = versions_from(inv)
    return inv


def summary_text(inv):
    lines = [f"{inv['hostname']} (declared role {inv['declared_role']}) arch {inv['arch']} utc {inv['utc']}", inv.get("uname", "")]
    for g in inv.get("gpus", []):
        lines.append(f"GPU {g['index']}: {g['name']} total {g['memory_total_mib']} MiB (raw {g['memory_raw'][0]!r}) cc {g['compute_cap']} "
                     f"driver {g['driver']} uuid {g['uuid']}")
    t = inv.get("torch", {})
    lines.append(f"python {inv['python']['version']} at {inv['python']['executable']}; torch {t.get('version')} cuda {t.get('cuda')} "
                 f"available {t.get('cuda_available')}; transformers {inv['packages'].get('transformers')}")
    lines.append(f"memory {inv['memory']}")
    lines.append(f"NICs {inv['nics']}; route to peer: {(inv.get('route_to_peer') or {}).get('out')}")
    lines.append(f"foreign compute apps now: {inv.get('compute_apps')}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Role verification (pure functions)                                           #
# --------------------------------------------------------------------------- #
def _short(h):
    return (h or "").split(".")[0].lower()


def verify_role(inv, role):
    checks = []

    def ck(name, expected, observed, ok):
        checks.append({"check": name, "expected": expected, "observed": observed, "ok": bool(ok)})

    ck("declared_role", role, inv.get("declared_role"), inv.get("declared_role") == role)
    gpus = inv.get("gpus") or []
    if role == "server":
        want = S.SERVER_IDENTITY
        ck("hostname", want["hostname"], inv.get("hostname"), _short(inv.get("hostname")) == want["hostname"].lower())
        named = [g for g in gpus if want["gpu_name_contains"] in (g.get("name") or "")]
        ck("gpu_count", want["gpu_count"], len(gpus), len(gpus) == want["gpu_count"])
        ck("gpu_model", f"all GPUs contain {want['gpu_name_contains']!r}", [g.get("name") for g in gpus],
           len(named) == len(gpus) and len(gpus) > 0)
        ck("gpu_memory", f">= {want['gpu_memory_mib_min']} MiB each", [g.get("memory_total_mib") for g in gpus],
           all((g.get("memory_total_mib") or 0) >= want["gpu_memory_mib_min"] for g in gpus) and len(gpus) > 0)
    else:
        want = S.EDGE_IDENTITY
        ck("arch", want["arch"], inv.get("arch"), inv.get("arch") == want["arch"])
        ck("gpu_model", f"a GPU containing {want['gpu_name_contains']!r}", [g.get("name") for g in gpus],
           any(want["gpu_name_contains"] in (g.get("name") or "") for g in gpus))
        ck("not_the_server", f"hostname != {S.SERVER_IDENTITY['hostname']}", inv.get("hostname"),
           _short(inv.get("hostname")) != S.SERVER_IDENTITY["hostname"].lower())
    tf = inv.get("torch") or {}
    ck("torch_cuda", "torch importable with CUDA available", {"importable": tf.get("importable"), "cuda": tf.get("cuda_available")},
       tf.get("importable") and tf.get("cuda_available"))
    return {"role": role, "host": inv.get("hostname"), "checks": checks, "ok": all(c["ok"] for c in checks)}


def _route_ok(inv):
    r = inv.get("route_to_peer") or {}
    return bool(r.get("rc") == 0 and r.get("out"))


def roles_record(inv_edge, inv_server):
    """Role record for the freeze (PREREG §3): verified identities, devices and the routes in both directions."""
    ve, vs = verify_role(inv_edge, "edge"), verify_role(inv_server, "server")
    problems = [f"{v['role']}: {c['check']} expected {c['expected']!r}, observed {c['observed']!r}"
                for v in (ve, vs) for c in v["checks"] if not c["ok"]]
    if _short(inv_edge.get("hostname")) == _short(inv_server.get("hostname")):
        problems.append("edge and server inventories come from the same host")
    for name, inv in (("edge", inv_edge), ("server", inv_server)):
        if not _route_ok(inv):
            problems.append(f"{name}: no route to the peer recorded ({(inv.get('route_to_peer') or {}).get('err') or 'missing --peer'})")

    def devs(inv):
        return [{"index": g["index"], "name": g["name"], "uuid": g["uuid"], "memory_total_mib": g.get("memory_total_mib")}
                for g in inv.get("gpus") or []]

    return {"schema": "gemma-split-roles/1", "utc": C.utc(), "assignment": "fixed by the PI (PREREG §3); verified, never derived",
            "edge": {"host": inv_edge.get("hostname"), "machine": S.ROLE_MACHINE["edge"], "devices": devs(inv_edge), "verification": ve},
            "server": {"host": inv_server.get("hostname"), "machine": S.ROLE_MACHINE["server"], "devices": devs(inv_server), "verification": vs},
            "verified": not problems, "problems": problems}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--role", required=True, choices=S.ROLES)
    c.add_argument("--peer", required=True, help="IP or hostname of the other machine (route lookup only, no traffic)")
    c.add_argument("--out-dir", required=True)
    r = sub.add_parser("roles")
    r.add_argument("--capture-dir", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "collect":
        inv = collect(a.role, a.peer)
        out = Path(a.out_dir)
        C.write_json(out / "inventory.json", inv)
        C.write_text(out / "inventory.txt", summary_text(inv))
        v = verify_role(inv, a.role)
        print(summary_text(inv))
        print("role verification:", "OK" if v["ok"] else "MISMATCH", [c["check"] for c in v["checks"] if not c["ok"]])
    else:
        cap = Path(a.capture_dir)
        rec = roles_record(C.read_json(cap / "edge" / "inventory.json"), C.read_json(cap / "server" / "inventory.json"))
        C.write_json(cap / "roles.json", rec)
        print("roles verified" if rec["verified"] else "roles NOT verified:\n  - " + "\n  - ".join(rec["problems"]))
        sys.exit(0 if rec["verified"] else 2)


if __name__ == "__main__":
    main()
