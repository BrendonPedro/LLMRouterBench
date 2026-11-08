"""E0 instrumentation: device-memory cap, NVML memory/power sampling, allocator statistics, env capture.

Memory is measured two ways and both must respect the class budget (PREREG §5):
  device view  : NVML device `used` minus the idle baseline taken before this process touches CUDA
                 (the GPU must be exclusive; container PID namespaces make per-process NVML unreliable);
  allocator view: non-torch overhead (CUDA context, cuBLAS) + torch.cuda max_memory_reserved.
The cap itself is torch.cuda.set_per_process_memory_fraction, sized so that non-torch overhead plus the
allocator limit equals the budget: an over-budget allocation raises OutOfMemoryError instead of
silently succeeding.
"""

import json
import os
import platform
import socket
import subprocess
import threading
import time

MIB = 1 << 20


class NVML:
    """Thin wrapper over pynvml (nvidia-ml-py). `available` is False on hosts without an NVIDIA driver."""

    def __init__(self, index=0):
        self.available = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nv, self.h = pynvml, pynvml.nvmlDeviceGetHandleByIndex(index)
            self.available = True
        except Exception as e:   # no driver / no package
            self.error = f"{type(e).__name__}: {e}"

    def used_bytes(self):
        return self.nv.nvmlDeviceGetMemoryInfo(self.h).used

    def total_bytes(self):
        return self.nv.nvmlDeviceGetMemoryInfo(self.h).total

    def power_w(self):
        return self.nv.nvmlDeviceGetPowerUsage(self.h) / 1000.0

    def sm_clock_mhz(self):
        return self.nv.nvmlDeviceGetClockInfo(self.h, self.nv.NVML_CLOCK_SM)

    def name(self):
        n = self.nv.nvmlDeviceGetName(self.h)
        return n.decode() if isinstance(n, bytes) else n

    def processes(self):
        try:
            return [(p.pid, p.usedGpuMemory) for p in self.nv.nvmlDeviceGetComputeRunningProcesses(self.h)]
        except Exception:
            return []


class Sampler(threading.Thread):
    """Background sampler of (t, used_bytes, power_w, sm_mhz); summary() integrates energy."""

    def __init__(self, nvml, interval_s, baseline_bytes=0):
        super().__init__(daemon=True)
        self.nvml, self.interval, self.baseline = nvml, interval_s, baseline_bytes
        self.samples, self._halt = [], threading.Event()

    def run(self):
        while not self._halt.is_set():
            t = time.perf_counter()
            try:
                self.samples.append((t, self.nvml.used_bytes(), self.nvml.power_w(), self.nvml.sm_clock_mhz()))
            except Exception:
                pass
            self._halt.wait(self.interval)

    def stop(self):
        self._halt.set()
        self.join(timeout=5)
        return self.summary()

    def window(self, t0, t1):
        return [s for s in self.samples if t0 <= s[0] <= t1]

    def summary(self, t0=None, t1=None):
        s = self.samples if t0 is None else self.window(t0, t1)
        if not s:
            return {"n": 0}
        energy = sum((b[0] - a[0]) * (a[2] + b[2]) / 2 for a, b in zip(s, s[1:]))
        return {"n": len(s), "peak_used_mib": (max(x[1] for x in s) - self.baseline) / MIB,
                "mean_power_w": sum(x[2] for x in s) / len(s), "energy_j": energy,
                "sm_mhz_min": min(x[3] for x in s), "sm_mhz_max": max(x[3] for x in s)}


def idle_baseline(nvml, n=10, interval_s=0.05):
    """Median device `used` before CUDA initialisation in this process (must be called first)."""
    vals = []
    for _ in range(n):
        vals.append(nvml.used_bytes())
        time.sleep(interval_s)
    return sorted(vals)[len(vals) // 2]


def apply_cap(torch, device, budget_mib, nvml, baseline_bytes):
    """Initialise CUDA + cuBLAS, measure the non-torch overhead, and cap the allocator at the remainder."""
    dev = torch.device(device)
    torch.cuda.init()
    a = torch.ones((256, 256), device=dev, dtype=torch.bfloat16)
    (a @ a).sum().item()
    del a
    torch.cuda.synchronize(dev)
    nontorch = nvml.used_bytes() - baseline_bytes - torch.cuda.memory_reserved(dev)
    total = torch.cuda.get_device_properties(dev).total_memory
    allowed = budget_mib * MIB - nontorch
    if allowed <= 0:
        raise RuntimeError(f"non-torch overhead {nontorch / MIB:.0f} MiB exceeds budget {budget_mib} MiB")
    frac = allowed / total
    torch.cuda.set_per_process_memory_fraction(frac, dev)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)
    return {"budget_mib": budget_mib, "nontorch_mib": nontorch / MIB, "allocator_limit_mib": allowed / MIB,
            "fraction": frac, "device_total_mib": total / MIB, "baseline_mib": baseline_bytes / MIB}


def allocator_stats(torch, device):
    st = torch.cuda.memory_stats(device)
    return {"allocated_peak_mib": st.get("allocated_bytes.all.peak", 0) / MIB,
            "reserved_peak_mib": st.get("reserved_bytes.all.peak", 0) / MIB,
            "allocated_now_mib": st.get("allocated_bytes.all.current", 0) / MIB,
            "num_alloc_retries": st.get("num_alloc_retries", 0), "num_ooms": st.get("num_ooms", 0)}


def _run(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:
        return f"ERROR {e}"


def environment(torch=None, extra=None):
    """Everything E0 §7 requires to be recorded per host."""
    env = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": socket.gethostname(),
           "platform": platform.platform(), "python": platform.python_version(),
           "nvidia_smi": _run("nvidia-smi --query-gpu=index,name,memory.total,power.limit,clocks.max.sm,"
                              "clocks.sm,persistence_mode,compute_mode,driver_version --format=csv"),
           "nvidia_topo": _run("nvidia-smi topo -m"),
           "mig": _run("nvidia-smi -q | grep -iE 'MIG Mode' -A2 | head -6"),
           "cpu": _run("lscpu | grep -E 'Model name|^CPU\\(s\\)'"), "mem": _run("free -g | head -2"),
           "pip_freeze": _run("python -m pip freeze 2>/dev/null").splitlines(),
           "git": _run("git -C " + os.path.dirname(os.path.abspath(__file__)) + " rev-parse HEAD"),
           "git_dirty": bool(_run("git -C " + os.path.dirname(os.path.abspath(__file__)) +
                                  " status --porcelain -- .")),
           "tc": _run("tc -V"), "net_admin": _run("sudo -n true 2>/dev/null && echo sudo-ok || echo no-sudo")}
    if torch is not None:
        env["torch"] = {"version": torch.__version__, "cuda": torch.version.cuda,
                        "cuda_available": torch.cuda.is_available(),
                        "bf16": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None,
                        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                        "deterministic": torch.are_deterministic_algorithms_enabled()}
    if extra:
        env.update(extra)
    return env


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


def append_jsonl(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass   # torn tail after a crash; the request is re-run
    return out
