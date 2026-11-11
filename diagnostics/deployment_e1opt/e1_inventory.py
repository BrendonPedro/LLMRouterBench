"""E1-OPT Stage 0: machine inventory and deterministic role feasibility (PREREG_E1OPT §6).

`collect` gathers facts only: no benchmark, no model inference, no CUDA timing. Device enumeration through
torch is allowed. `feasibility` and `assign_roles` are pure functions of inventory dicts and use memory fit,
architecture/software support, the one-server-GPU rule and connectivity. No datasheet bandwidth and no measured
speed enters any role decision.

  python e1_inventory.py collect --peer <other host or ip> --out env/<hostname>.json
  python e1_inventory.py roles --inventory env/A.json env/B.json --out env/roles.json [--lab-server HOST --lab-edge HOST]
"""
import json
import platform
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))   # python -I: the script dir is not on sys.path
import e1_spec as S
import e1_memory as M


def _run(cmd, timeout=60):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or "").strip() or (r.stderr or "").strip()
    except Exception as e:
        return f"ERROR {e}"


def parse_nvidia_smi_csv(text):
    """Rows of `nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version,compute_cap --format=csv,noheader`."""
    gpus = []
    for line in (text or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6 or not parts[0].isdigit():
            continue
        mem = re.match(r"(\d+)", parts[3])
        cc = None
        try:
            cc = float(parts[5])
        except ValueError:
            pass
        gpus.append({"index": int(parts[0]), "name": parts[1], "uuid": parts[2], "memory_mib": int(mem.group(1)) if mem else None,
                     "driver": parts[4], "compute_cap": cc})
    return gpus


def parse_free_g(text):
    for line in (text or "").splitlines():
        if line.startswith("Mem:"):
            f = line.split()
            return int(f[1]) if len(f) > 1 and f[1].isdigit() else None
    return None


def parse_nics(ip_br_link, ethtool_fn=None):
    nics = []
    for line in (ip_br_link or "").splitlines():
        f = line.split()
        if not f or f[0] == "lo":
            continue
        name = f[0].split("@")[0]
        speed = None
        if ethtool_fn:
            m = re.search(r"Speed:\s*(\d+)Mb/s", ethtool_fn(name) or "")
            speed = int(m.group(1)) if m else None
        nics.append({"name": name, "state": f[1] if len(f) > 1 else None, "speed_mbps": speed})
    return nics


def python_module_info(mod, attr=None):
    code = (f"import {mod}; v=getattr({mod},'__version__','present'); "
            + (f"ok=hasattr({mod},'{attr}'); " if attr else "ok=True; ")
            + "print(v); print(ok)")
    out = _run(f"{sys.executable} -c \"{code}\" 2>&1", timeout=120)
    lines = out.splitlines()
    if len(lines) >= 2 and lines[-1] in ("True", "False"):
        return {"importable": True, "version": lines[-2], ("has_" + attr if attr else "ok"): lines[-1] == "True"}
    return {"importable": False, "error": out[-500:]}


def torch_devices():
    try:
        import torch
        d = {"version": torch.__version__, "cuda": torch.version.cuda, "cuda_available": bool(torch.cuda.is_available()),
             "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0, "devices": []}
        for i in range(d["device_count"]):
            p = torch.cuda.get_device_properties(i)
            d["devices"].append({"index": i, "name": p.name, "total_mib": p.total_memory // (1 << 20),
                                 "compute_cap": float(f"{p.major}.{p.minor}"), "uuid": str(getattr(p, "uuid", ""))})
        return d
    except Exception as e:
        return {"version": None, "error": str(e), "cuda_available": False, "device_count": 0, "devices": []}


def collect(peer=None):
    host = socket.gethostname()
    smi = _run("nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version,compute_cap --format=csv,noheader")
    gpus = parse_nvidia_smi_csv(smi)
    td = torch_devices()
    for g in gpus:   # fill compute capability from torch when nvidia-smi lacks the column
        if g.get("compute_cap") is None:
            t = next((x for x in td["devices"] if x["index"] == g["index"]), None)
            if t:
                g["compute_cap"] = t["compute_cap"]
    inv = {
        "hostname": host, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "uname": _run("uname -a"),
        "arch": platform.machine(), "os": _run("head -3 /etc/os-release"), "cpu": _run("lscpu | grep -E 'Model name|^CPU\\(s\\)|Architecture'"),
        "ram_gib": parse_free_g(_run("free -g")), "nvidia_smi_raw": smi, "nvidia_smi_topo": _run("nvidia-smi topo -m"),
        "gpus": gpus, "gpu_count": len(gpus), "cuda_toolkit": _run("nvcc --version 2>/dev/null | tail -2"),
        "python": platform.python_version(), "torch": td,
        "vllm": python_module_info("vllm"), "flash_attn": python_module_info("flash_attn", "flash_attn_with_kvcache"),
        "pynvml": python_module_info("pynvml"),
        "nics": parse_nics(_run("ip -br link"), lambda n: _run(f"ethtool {n} 2>/dev/null")),
        "route_to_peer": _run(f"ip route get {peer}") if peer else None, "peer": peer,
        "tc_present": not _run("tc -V").startswith("ERROR") and "command not found" not in _run("tc -V"),
        "tc_version": _run("tc -V"), "sudo_noninteractive": _run("sudo -n true 2>/dev/null && echo yes || echo no") == "yes",
        "capsh": _run("capsh --print 2>/dev/null | grep -i cap_net_admin | head -1"),
        "cap_net_admin": "check manually: sudo -n or capsh output above; tc netem needs CAP_NET_ADMIN",
        "unified_memory": "unknown",
    }
    if inv["arch"] == "aarch64" and any("GB10" in (g.get("name") or "") for g in gpus):
        inv["unified_memory"] = "likely (GB10, aarch64)"
    try:
        import e1_freeze
        inv["versions"] = e1_freeze.versions()
    except Exception as e:
        inv["versions"] = {"error": str(e)}
    return inv


def summary_text(inv):
    lines = [f"host {inv['hostname']} arch {inv['arch']} ram {inv.get('ram_gib')} GiB", inv.get("uname", "")]
    for g in inv.get("gpus", []):
        lines.append(f"GPU {g['index']}: {g['name']} {g.get('memory_mib')} MiB cc {g.get('compute_cap')} driver {g.get('driver')} uuid {g.get('uuid')}")
    t = inv.get("torch", {})
    lines.append(f"torch {t.get('version')} cuda {t.get('cuda')} available {t.get('cuda_available')} devices {t.get('device_count')}")
    lines.append(f"vllm {inv.get('vllm')} ; flash_attn {inv.get('flash_attn')}")
    lines.append(f"NICs {inv.get('nics')} ; route to peer: {inv.get('route_to_peer')}")
    lines.append(f"tc present {inv.get('tc_present')} ; sudo -n {inv.get('sudo_noninteractive')} ; capsh {inv.get('capsh')!r}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Feasibility and roles (pure functions; PREREG §6.2, §6.3)                   #
# --------------------------------------------------------------------------- #
def server_need_gib(k):
    cfg = S.E0.QWEN3_32B_CONFIG
    p = M.M.param_counts(cfg)
    L = cfg["num_hidden_layers"]
    w = (L - 2 * k) * p["layer"] * 2
    kv = (L - 2 * k) * M.M.kv_bytes_per_token_layer(cfg) * S.E0.L_F
    return (w + kv + M.extra_bytes_e1()["total_device_bytes"]) / 2 ** 30 + S.SERVER_MEMORY_MARGIN_GIB


def _software_ok(inv):
    t = inv.get("torch", {})
    reasons = []
    if not t.get("cuda_available"):
        reasons.append("torch reports no CUDA")
    return reasons


def _gpus_with(inv, min_mib):
    return [g for g in inv.get("gpus", []) if (g.get("memory_mib") or 0) >= min_mib
            and (g.get("compute_cap") is None or g["compute_cap"] >= S.MIN_COMPUTE_CAPABILITY)]


def feasibility(inv, classes=S.PRIMARY_CLASSES):
    """Server: one physical GPU holds the server segment at the smallest-k frozen cut (largest segment); edge: one GPU
    can emulate the largest primary class budget and the competitor engine is importable. Lowest-index GPU that
    qualifies is selected (approved rule 2); no benchmark."""
    out = {"host": inv["hostname"]}
    k_min = min(S.FIXED_CUTS[m][0] for m in classes)
    need_mib = int(server_need_gib(k_min) * 1024)
    sw = _software_ok(inv)
    cands = _gpus_with(inv, need_mib)
    out["server"] = {"feasible": bool(cands) and not sw, "need_mib": need_mib, "gpu": cands[0] if cands else None,
                     "reasons": sw + ([] if cands else [f"no single GPU with >= {need_mib} MiB (one-server-GPU rule)"])}
    edge_need_mib = max(classes) * 1024
    ecands = _gpus_with(inv, edge_need_mib)
    ereasons = list(sw)
    if not ecands:
        ereasons.append(f"no GPU with >= {edge_need_mib} MiB to emulate the {max(classes)} GB budget")
    if not inv.get("vllm", {}).get("importable"):
        ereasons.append("vLLM not importable (competitors)")
    out["edge"] = {"feasible": not ereasons, "need_mib": edge_need_mib, "gpu": ecands[0] if ecands else None, "reasons": ereasons}
    return out


def assign_roles(inv_a, inv_b, lab_server=None, lab_edge=None, classes=S.PRIMARY_CLASSES):
    """Deterministic role assignment (PREREG §6.3):
      1. an explicit lab assignment is used if it is feasible (else INCOMPLETE with the reason);
      2. if exactly one assignment is feasible, it is used;
      3. if both are feasible, the machine whose qualifying server GPU has the LARGER memory capacity is the server
         (capacity, not bandwidth, not speed); tie: the smaller hostname in byte order is the server;
      4. if none is feasible, the pair is INCOMPLETE.
    The chosen server GPU is the lowest-index GPU on that machine that holds the segment."""
    fa, fb = feasibility(inv_a, classes), feasibility(inv_b, classes)
    by = {inv_a["hostname"]: (inv_a, fa), inv_b["hostname"]: (inv_b, fb)}
    link_ok = bool(inv_a.get("route_to_peer")) and bool(inv_b.get("route_to_peer")) and \
        not str(inv_a.get("route_to_peer")).startswith("ERROR") and not str(inv_b.get("route_to_peer")).startswith("ERROR")
    options = []
    for s_host, e_host in ((inv_a["hostname"], inv_b["hostname"]), (inv_b["hostname"], inv_a["hostname"])):
        fs, fe = by[s_host][1]["server"], by[e_host][1]["edge"]
        options.append({"server": s_host, "edge": e_host, "feasible": fs["feasible"] and fe["feasible"],
                        "server_gpu": fs["gpu"], "edge_gpu": fe["gpu"], "reasons": fs["reasons"] + fe["reasons"]})
    rec = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "classes": list(classes), "feasibility": {inv_a["hostname"]: fa, inv_b["hostname"]: fb},
           "options": options, "link_route_both_ways": link_ok, "feasible": False, "rule_applied": None, "server": None, "edge": None}
    feas = [o for o in options if o["feasible"]]
    chosen = None
    if lab_server or lab_edge:
        want = [o for o in options if (not lab_server or o["server"] == lab_server) and (not lab_edge or o["edge"] == lab_edge)]
        if want and want[0]["feasible"]:
            chosen, rec["rule_applied"] = want[0], "6.3.1 lab assignment"
        else:
            rec["rule_applied"] = "6.3.1 lab assignment requested but infeasible"
            rec["reason"] = want[0]["reasons"] if want else "lab assignment names an unknown host"
    elif len(feas) == 1:
        chosen, rec["rule_applied"] = feas[0], "6.3.2 exactly one feasible assignment"
    elif len(feas) == 2:
        feas.sort(key=lambda o: o["server"].encode())                                   # tie-break: smaller hostname (bytes)
        feas.sort(key=lambda o: o["server_gpu"]["memory_mib"] or 0, reverse=True)         # primary: larger single-GPU capacity (stable)
        tie = feas[0]["server_gpu"]["memory_mib"] == feas[1]["server_gpu"]["memory_mib"]
        chosen = feas[0]
        rec["rule_applied"] = "6.3.3 both feasible: " + ("tie on capacity -> smaller hostname is the server" if tie
                                                          else "larger single-GPU memory capacity is the server")
    else:
        rec["rule_applied"], rec["reason"] = "6.3.4 no feasible assignment -> INCOMPLETE", [o["reasons"] for o in options]
    if chosen and not link_ok:
        rec["rule_applied"] += "; link route missing -> INCOMPLETE"
        chosen = None
    if chosen:
        rec.update(feasible=True,
                   server={"host": chosen["server"], "gpu_index": chosen["server_gpu"]["index"], "gpu_uuid": chosen["server_gpu"].get("uuid"),
                           "gpu_name": chosen["server_gpu"].get("name"), "selection": "lowest CUDA index holding the segment"},
                   edge={"host": chosen["edge"], "gpu_index": chosen["edge_gpu"]["index"], "gpu_uuid": chosen["edge_gpu"].get("uuid"),
                         "gpu_name": chosen["edge_gpu"].get("name"), "selection": "lowest CUDA index able to emulate the largest primary budget"})
    return rec


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect"); c.add_argument("--peer", default=None); c.add_argument("--out", required=True)
    r = sub.add_parser("roles"); r.add_argument("--inventory", nargs=2, required=True); r.add_argument("--out", required=True)
    r.add_argument("--lab-server", default=None); r.add_argument("--lab-edge", default=None)
    r.add_argument("--merged-out", default=None, help="also write {hostname: inventory} for e1_freeze")
    a = ap.parse_args(argv)
    if a.cmd == "collect":
        inv = collect(a.peer)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(inv, indent=1, default=str) + "\n")
        Path(a.out).with_suffix(".txt").write_text(summary_text(inv))
        print(summary_text(inv))
    else:
        ia, ib = (json.loads(Path(p).read_text()) for p in a.inventory)
        rec = assign_roles(ia, ib, a.lab_server, a.lab_edge)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rec, indent=1, default=str) + "\n")
        if a.merged_out:
            Path(a.merged_out).write_text(json.dumps({ia["hostname"]: ia, ib["hostname"]: ib}, indent=1, default=str) + "\n")
        print(json.dumps({k: rec[k] for k in ("feasible", "rule_applied", "server", "edge")}, indent=1))


if __name__ == "__main__":
    main()
