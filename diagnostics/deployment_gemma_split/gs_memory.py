"""GEMMA-SPLIT Stage 2 driver: memory feasibility probes on each host (PREREG §8). No timers. A driver file: the
verdict logic (status_of, combine_split, classify) lives in gs_rules, a decision file.

Feasibility is a fact, not a threshold: the frozen configuration FITS on a host iff its part loads and the workload's
longest item (`probe_item`) completes there. Peak memory is recorded from every source the host supports (torch
allocator peaks and OOM-event counters per device, cudaMemGetInfo, process peak RSS, the largest resident processes,
/proc/meminfo, nvidia-smi); unified-memory devices that report "[N/A]" keep None. A first allocation failure gives
INFEASIBLE_UNCONFIRMED at the gate and is repeated once with fresh processes before it becomes STOP_INFEASIBLE.
"""
import os
import sys
import threading
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # python -I: the script dir is not on sys.path
import gs_common as C  # noqa: E402
import gs_inventory as INV  # noqa: E402
import gs_rules as RU  # noqa: E402
import gs_spec as S  # noqa: E402


def _status_kb(field):
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(field + ":"):
                return int(line.split()[1]) * 1024
    except Exception:
        return None
    return None


def own_pids():
    pids = {os.getpid()}
    try:
        for d in Path("/proc").iterdir():
            if d.name.isdigit():
                try:
                    ppid = int((d / "stat").read_text().split(")")[1].split()[1])
                except Exception:
                    continue
                if ppid in pids:
                    pids.add(int(d.name))
    except Exception:
        pass
    return pids


def smi_snapshot():
    g = C.run(["nvidia-smi", f"--query-gpu={INV.SMI_GPU_FIELDS}", "--format=csv,noheader"], timeout=30)
    a = C.run(["nvidia-smi", f"--query-compute-apps={INV.SMI_APP_FIELDS}", "--format=csv,noheader"], timeout=30)
    return {"gpus": INV.parse_smi_gpus(g["out"]), "apps": INV.parse_smi_apps(a["out"]), "available": g["rc"] == 0}


def foreign_processes(snap, uuids, mine=None):
    from gs_reference import norm_uuid
    mine = mine or own_pids()
    want = {norm_uuid(u) for u in uuids}
    return [a for a in snap.get("apps", []) if norm_uuid(a["gpu_uuid"]) in want and a["pid"] not in mine]


def top_rss(n=15):
    """Largest resident processes: on a unified-memory host (DGX Spark) CPU processes compete with the GPU for memory."""
    out = []
    for line in C.run(["ps", "-eo", "pid=,rss=,comm=", "--sort=-rss"], timeout=15)["out"].splitlines()[:n]:
        f = line.split(None, 2)
        if len(f) == 3 and f[0].isdigit() and f[1].isdigit():
            out.append({"pid": int(f[0]), "rss_kib": int(f[1]), "comm": f[2]})
    return out


def snapshot(devices):
    snap = {"utc": C.utc(), "meminfo": INV.parse_meminfo(C.run(["cat", "/proc/meminfo"], timeout=5)["out"]),
            "process_rss_bytes": _status_kb("VmRSS"), "process_peak_rss_bytes": _status_kb("VmHWM"), "smi": smi_snapshot(),
            "top_rss": top_rss(), "torch": {}}
    try:
        import torch
        for d in devices:
            if str(d).startswith("cuda") and torch.cuda.is_available():
                dev = torch.device(d)
                free, total = torch.cuda.mem_get_info(dev)
                snap["torch"][str(d)] = {"max_allocated_bytes": torch.cuda.max_memory_allocated(dev),
                                         "max_reserved_bytes": torch.cuda.max_memory_reserved(dev),
                                         "num_ooms": int(torch.cuda.memory_stats(dev).get("num_ooms", 0)),
                                         "mem_get_info_free_bytes": free, "mem_get_info_total_bytes": total}
    except Exception as e:
        snap["torch_error"] = f"{type(e).__name__}: {e}"
    return snap


def oom_events(before, after, peaks=None):
    """Allocator OOM events in this process between two snapshots (also seen by a monitor); 0 without CUDA."""
    n = 0
    for d, v in (after.get("torch") or {}).items():
        n = max(n, v.get("num_ooms", 0) - (before.get("torch") or {}).get(d, {}).get("num_ooms", 0))
    for d, v in (peaks or {}).items():
        if isinstance(v, dict):
            n = max(n, v.get("num_ooms", 0) - (before.get("torch") or {}).get(d, {}).get("num_ooms", 0))
    return n


def run_probe(what, role, devices, uuids, body, unified=False):
    """Run body() (loads + executes the probe) between two snapshots; never raises; returns the record."""
    mine = own_pids()
    before = snapshot(devices)
    fb = foreign_processes(before["smi"], uuids, mine)
    rec = {"what": what, "role": role, "devices": [str(d) for d in devices], "before": before, "foreign_before": fb, "unified_memory": unified}
    failure, extra = None, {}
    if len(fb) > S.FOREIGN_PROCESSES_ALLOWED:
        failure = "not run: foreign compute process on an experiment device"
    else:
        try:
            extra = body() or {}
        except Exception as e:   # an OOM may surface as RuntimeError, torch.OutOfMemoryError or MemoryError
            failure = f"{type(e).__name__}: {e}\n" + traceback.format_exc()[-4000:]
    after = snapshot(devices)
    fa = foreign_processes(after["smi"], uuids, own_pids() | mine)
    ooms = oom_events(before, after)
    rec.update(after=after, foreign_after=fa, failure=failure, failure_class=None if failure is None else RU.classify(failure), ooms=ooms,
               status=RU.status_of(failure, fb, fa, unified, ooms), **extra)
    return rec


class ServerMonitor:
    """Polls allocator peaks and OOM counters in the server process while the adapter serves (Stage 2 only)."""

    def __init__(self, devices, period=1.0):
        self.devices, self.period, self.peak = devices, period, {}
        self.stop = threading.Event()
        self.t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self.stop.wait(self.period):
            s = snapshot(self.devices)
            for d, v in s["torch"].items():
                cur = self.peak.setdefault(d, {})
                for k in ("max_allocated_bytes", "max_reserved_bytes", "num_ooms"):
                    cur[k] = max(cur.get(k, 0), v.get(k, 0))
            rss = s.get("process_peak_rss_bytes")
            if rss:
                self.peak["process_peak_rss_bytes"] = max(self.peak.get("process_peak_rss_bytes", 0), rss)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.t.join(timeout=10)
