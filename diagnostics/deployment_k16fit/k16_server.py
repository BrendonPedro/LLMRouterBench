"""K16-HWFIT server driver (ASUS main server): layers 16-47 of Qwen3-32B on ONE RTX A6000.

  all     preflight (separate process) -> serve. This is what launch_asus_server.sh runs.
  serve   memory gate -> load layers 16-47 -> listen -> serve the edge until it requests shutdown

The server listens on its loopback interface only (the wire has no authentication of its own); the edge reaches
it through an SSH tunnel that launch_dgx_edge.sh opens from the DGX Spark. A non-loopback bind is refused.
The server holds only its own layers and their KV caches. It receives the bf16 residual stream after layer 15
and returns the one after layer 47. It exits when the edge's last connection asks for shutdown, when no edge
connects for --wait-s seconds, or on Ctrl-C; in every case it frees the GPU and writes server_done.json.
It never touches another process.
"""
import ipaddress
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # python -I: the script dir is not on sys.path
import k16_spec as S  # noqa: E402
from k16_common import (append_jsonl, read_json, write_json, utc, run, EXIT_STOP)  # noqa: E402

EXIT_NO_EDGE = 5


def _stop(run_dir, why, detail=None):
    write_json(Path(run_dir) / "server_done.json", {"experiment": S.NAME, "exit": "STOP", "why": why, "detail": detail, "utc": utc()})
    print(f"[server] STOP: {why}", flush=True)
    sys.exit(EXIT_STOP)


def serve(a):
    run_dir = Path(a.run_dir)
    pf = read_json(run_dir / "preflight.json")
    spec_hash = S.spec_hash(a.dev)
    if pf.get("verdict") != "CONTINUE" or pf.get("role") != "server" or pf.get("spec_hash") != spec_hash or pf.get("dev") != a.dev:
        _stop(run_dir, "preflight.json is not a CONTINUE for this server and this spec")
    gpu = pf["gpu"].get("gpu")
    if not a.dev and len(pf["inventory"]["gpus"]) > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]          # exactly one GPU is visible to this process
    import torch
    import k16_memory as KM
    import k16_preflight as PF
    import k16_runtime as R
    p = S.params(a.dev)
    device = a.device if a.dev else "cuda:0"
    if not a.dev:
        g = PF.memory_gate("server", gpu_uuid=gpu["uuid"], allow_shared=a.allow_shared)
        write_json(run_dir / "gates" / "memory_server.json", g)
        if not g["ok"]:
            _stop(run_dir, f"memory gate: {g['verdict']}", g)
        ident = R.device_identity(device)
        if S.ROLES["server"]["gpu_name_marker"].lower() not in ident["name"].lower() or torch.cuda.device_count() != 1:
            _stop(run_dir, f"visible device is not one {S.ROLES['server']['gpu_name_marker']}: {ident}, {torch.cuda.device_count()} visible")
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    R.setup_torch()
    model_dir = pf["checkpoint"]["model_dir"]
    print(f"[server] loading layers {S.SERVER_LAYERS[0]}-{S.SERVER_LAYERS[-1]} from {model_dir} on {device} ...", flush=True)
    t0 = time.time()
    side = R.ServerSide(model_dir, device, p, None if a.dev else S.E0.QWEN3_32B_CONFIG, with_static=not a.no_static)
    load_mem = R.load_memory(device)
    if a.dev_sabotage:
        if not a.dev:
            sys.exit("--dev-sabotage needs --dev")
        side.segs["prefix"].layer_ids = list(reversed(side.segs["prefix"].layer_ids))    # tests: a wrong layer order must fail the gate
    own = {"server_layers_loaded": side.sl.layer_ids, "equals_spec": side.sl.layer_ids == S.SERVER_LAYERS,
           "holds_embedding_or_head": side.sl.embed is not None or side.sl.lm_head is not None or side.sl.norm is not None}
    if not own["equals_spec"] or own["holds_embedding_or_head"]:
        _stop(run_dir, "loaded slice is not exactly layers 16-47", own)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((a.host_bind, a.port))
    sock.listen(1)
    port = sock.getsockname()[1]
    extra = {"server_run_dir": str(run_dir), "git_commit": pf["git"]["commit"], "checkpoint_identity_sha256": pf["checkpoint"]["identity_sha256"],
             "torch": torch.__version__, "torch_cuda": torch.version.cuda, "dev": a.dev, "gpu_shared_at_start": bool((gpu or {}).get("occupied"))}
    ready = {"experiment": S.NAME, "status": S.STATUS, "utc": utc(), "spec_hash": spec_hash, "port": port, "addresses": pf["host"]["addresses"],
             "identity": side.identity(spec_hash, extra), "ownership": own, "load_wall_s_not_a_result": round(time.time() - t0, 1),
             "listen": f"{a.host_bind}:{port}", "transport": S.spec(a.dev)["transport"], "load_memory": load_mem, "memory_after_load": R.mem_stats(device), "expected_memory_mib": {k: (v if k == "stage" else round(v / KM.MIB, 1)) for k, v in KM.expected("server").items()}}
    write_json(run_dir / "server_ready.json", ready)
    addr = (pf["host"]["addresses"] or [socket.gethostname()])[0]
    print(f"[server] READY on {a.host_bind}:{port} (loopback only; reach it through an SSH tunnel); static mode: {side.static}", flush=True)
    print(f"[server] load: {json.dumps(load_mem)}", flush=True)
    print("[server] now run the DGX command. Its K16_SSH must be an ssh target of this machine that works from the DGX without a prompt, e.g.\n"
          f"    K16_SSH={os.environ.get('USER', 'USER')}@{addr}" + (f" K16_PORT={port}" if port != S.DEFAULT_PORT else ""), flush=True)
    log = lambda ev: append_jsonl(run_dir / "server_events.jsonl", ev)
    reason = "interrupted"
    try:
        reason = R.serve(side, sock, spec_hash, log, extra_identity=extra, accept_timeout_s=a.wait_s)
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        done = {"experiment": S.NAME, "exit": reason, "utc": utc(), "memory_peak": R.mem_stats(device), "spec_hash": spec_hash,
                "nvidia_smi_at_exit": None if a.dev else run(["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total", "--format=csv,noheader"])}
        write_json(run_dir / "server_done.json", done)
        print(f"[server] exit: {reason}; peak {json.dumps(done['memory_peak'])}", flush=True)
    sys.exit({"shutdown": 0, "accept_timeout": EXIT_NO_EDGE}.get(reason, 130))


def run_all(a):
    host = socket.gethostname()
    run_dir = Path(a.run_dir) if a.run_dir else Path(a.out).expanduser() / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{host}_server"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[server] {S.NAME} ({S.STATUS})\n[server] run directory: {run_dir}", flush=True)
    py, here = [sys.executable, "-I"], Path(__file__).resolve().parent
    flags = (["--dev"] if a.dev else []) + (["--allow-shared"] if a.allow_shared else [])
    pre = py + [str(here / "k16_preflight.py"), "--role", "server", "--out", str(run_dir / "preflight.json"), "--port", str(a.port)] + flags
    pre += (["--model-dir", a.model_dir] if a.model_dir else []) + (["--gpu", a.gpu] if a.gpu else [])
    rc = subprocess.run(pre).returncode
    if rc != 0:
        print(f"[server] preflight stopped (exit {rc}); nothing was loaded. See {run_dir / 'preflight.json'}", flush=True)
        sys.exit(rc)
    srv = py + [str(here / "k16_server.py"), "serve", "--run-dir", str(run_dir), "--port", str(a.port), "--wait-s", str(a.wait_s),
                "--host-bind", a.host_bind, "--device", a.device] + flags + (["--no-static"] if a.no_static else []) + (["--dev-sabotage"] if a.dev_sabotage else [])
    sys.exit(subprocess.run(srv).returncode)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("all", "serve"):
        s = sub.add_parser(name)
        s.add_argument("--run-dir", default=None); s.add_argument("--out", default=S.default_out())
        s.add_argument("--model-dir", default=None); s.add_argument("--gpu", default=os.environ.get("K16_GPU") or None)
        s.add_argument("--port", type=int, default=int(os.environ.get("K16_PORT", S.DEFAULT_PORT))); s.add_argument("--host-bind", default=S.LISTEN_HOST)
        s.add_argument("--wait-s", type=float, default=float(os.environ.get("K16_SERVER_WAIT_S", 14400)))
        s.add_argument("--allow-shared", action="store_true", default=os.environ.get("K16_ALLOW_SHARED") == "1")
        s.add_argument("--no-static", action="store_true", default=os.environ.get("K16_NO_STATIC") == "1")
        s.add_argument("--dev", action="store_true"); s.add_argument("--device", default="cpu"); s.add_argument("--dev-sabotage", action="store_true")
    a = ap.parse_args(argv)
    try:
        loopback = ipaddress.ip_address(a.host_bind).is_loopback
    except ValueError:
        loopback = a.host_bind == "localhost"
    if not loopback:
        print(f"[server] REFUSED: --host-bind {a.host_bind} is not a loopback address. The wire is unauthenticated; the server listens on "
              "loopback only and the edge connects through an SSH tunnel.", flush=True)
        sys.exit(EXIT_STOP)
    {"all": run_all, "serve": serve}[a.cmd](a)


if __name__ == "__main__":
    main()
