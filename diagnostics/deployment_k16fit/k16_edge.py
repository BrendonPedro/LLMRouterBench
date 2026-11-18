"""K16-HWFIT edge driver (DGX Spark): embeddings, layers 0-15 and 48-63, final norm, LM head, sampling.

  all        the whole edge sequence, one stage per process, stopping at the first failed gate:
             preflight -> reference -> inhost -> twohost -> summary.   launch_dgx_edge.sh runs this.
  reference  Stage 2a  unsplit Hugging Face model on the edge GPU: the reference of every comparison
  inhost     Stage 2b  the k = 16 split with both segments on the edge GPU, over the real wire on loopback,
                       reference mode, against the reference                      -> gate inhost_prefix
  twohost    Stage 2c  the real two-host split (reference mode) against the reference -> gate twohost_prefix
             Stage 3   two-host generation of chat prompts, decoded text (needs twohost_prefix)
             Stage 4   two-host latency (needs twohost_prefix)
             Stage 5   optional: static mode against the same reference -> gate twohost_static; its latency
                       only if that gate passes. A failure here does not change the result of stages 2-4.
  summary    gates, key numbers, SUMMARY.md, archive with SHA-256

Every stage re-reads the free memory first and refuses to load if it is short. No stage runs behind a closed
gate. Latency rows exist only for a mode whose two-host correctness gate passed.
"""
import hashlib
import json
import os
import socket
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # python -I: the script dir is not on sys.path
import k16_spec as S  # noqa: E402
from k16_common import (append_jsonl, read_json, read_jsonl, write_json, utc, gate_write, gate_require, gate_status, gate_path,
                        GateClosed, EXIT_STOP, EXIT_GATE)  # noqa: E402

STAGE_MEMORY = {"reference": "edge_reference", "inhost": "edge_inhost", "twohost": "edge_twohost"}


def _begin(a, stage):
    """Common start of a GPU stage: preflight record, visible device, memory gate. Returns (run_dir, pf, p, spec_hash, device)."""
    run_dir = Path(a.run_dir)
    pf = read_json(run_dir / "preflight.json")
    spec_hash = S.spec_hash(a.dev)
    if pf.get("verdict") != "CONTINUE" or pf.get("role") != "edge" or pf.get("spec_hash") != spec_hash or pf.get("dev") != a.dev:
        print(f"[{stage}] STOP: preflight.json is not a CONTINUE for this edge and this spec", flush=True)
        sys.exit(EXIT_STOP)
    if a.dev:
        return run_dir, pf, S.params(True), spec_hash, a.device
    gpu = pf["gpu"]["gpu"]
    if len(pf["inventory"]["gpus"]) > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
    import k16_preflight as PF
    g = PF.memory_gate(STAGE_MEMORY[stage], gpu_uuid=gpu["uuid"], allow_shared=a.allow_shared)
    write_json(run_dir / "gates" / f"memory_{stage}.json", g)
    print(f"[{stage}] memory gate: need {g['need_mib']} MiB, free now {g['free_now_mib']} MiB -> {g['verdict']}", flush=True)
    if not g["ok"]:
        print(f"[{stage}] STOP: {'; '.join(g['reasons'])}", flush=True)
        sys.exit(EXIT_STOP)
    return run_dir, pf, S.params(False), spec_hash, "cuda:0"


def _require(run_dir, names, spec_hash, stage):
    try:
        gate_require(run_dir, names, spec_hash)
    except GateClosed as e:
        print(f"[{stage}] REFUSED: {e}", flush=True)
        sys.exit(EXIT_GATE)


def _prompts(pf, dev):
    vocab = read_json(Path(pf["checkpoint"]["model_dir"]) / "config.json")["vocab_size"] if dev else None
    return S.check_prompts(dev, vocab)


def _meta(pf, spec_hash, device, R, extra=None):
    d = {"experiment": S.NAME, "status": S.STATUS, "utc": utc(), "spec_hash": spec_hash, "git_commit": pf["git"]["commit"],
         "host": socket.gethostname(), "device": R.device_identity(device), "versions": pf["versions"],
         "checkpoint_identity_sha256": pf["checkpoint"]["identity_sha256"], "model_dir": pf["checkpoint"]["model_dir"],
         "cut": {"k": S.K, "j": S.J, "edge_layers": S.EDGE_LAYERS, "server_layers": S.SERVER_LAYERS}, "criteria": S.CRITERIA}
    d.update(extra or {})
    return d


# --------------------------------------------------------------------------- #
# Stage 2a: unsplit reference                                                  #
# --------------------------------------------------------------------------- #
def reference(a):
    run_dir, pf, p, spec_hash, device = _begin(a, "reference")
    import torch
    import k16_runtime as R
    R.setup_torch()
    prompts = _prompts(pf, a.dev)
    model_dir = pf["checkpoint"]["model_dir"]
    print(f"[reference] loading the unsplit model from {model_dir} on {device} ...", flush=True)
    hf = R.hf_model(model_dir, device)
    load_mem = R.load_memory(device)
    print(f"[reference] loaded directly onto {device}: {json.dumps(load_mem)}", flush=True)
    out = run_dir / "reference"
    rows = R.write_reference(hf, prompts, p, device, out)
    tok, chats = R.chat_prompts(model_dir, p, prompts)
    (out / "chat.jsonl").unlink(missing_ok=True)
    for c in chats:
        toks, margins = R.hf_greedy(hf, c["ids"], p["chat_max_new"], device, eos_ids=() if a.dev else S.E0.EOS_IDS)
        append_jsonl(out / "chat.jsonl", {"id": c["id"], "text": c["text"], "prompt_ids": c["ids"], "tokens": toks, "margins": margins,
                                          "output_text": tok.decode(toks, skip_special_tokens=True) if tok else None})
    ok = len(rows) == len(prompts) and all(r["finite"] and not r["degenerate"] for r in rows)
    write_json(out / "reference.json", _meta(pf, spec_hash, device, R, {"stage": "2a-reference", "sequences": rows, "n_chat": len(chats),
                                                                        "load_memory": load_mem, "memory": R.mem_stats(device), "reference_is": S.CRITERIA["reference"]}))
    gate_write(run_dir, "reference", ok, spec_hash, {"n_sequences": len(rows), "all_finite": all(r["finite"] for r in rows)})
    del hf
    sys.exit(0 if ok else EXIT_GATE)


# --------------------------------------------------------------------------- #
# Stage 2b: in-host split over the real wire                                   #
# --------------------------------------------------------------------------- #
def inhost(a):
    run_dir, pf, p, spec_hash, device = _begin(a, "inhost")
    _require(run_dir, ["reference"], spec_hash, "inhost")
    import torch
    import k16_runtime as R
    R.setup_torch()
    model_dir, expected = pf["checkpoint"]["model_dir"], (None if a.dev else S.E0.QWEN3_32B_CONFIG)
    print(f"[inhost] loading both segments on {device} ...", flush=True)
    side = R.ServerSide(model_dir, device, p, expected, with_static=False)
    sl = R.edge_slice(model_dir, S.K, S.J, device, torch.bfloat16, expected)
    load_mem = R.load_memory(device)
    sock = socket.socket(); sock.bind(("127.0.0.1", 0)); sock.listen(1)
    events, stop = [], threading.Event()
    th = threading.Thread(target=R.serve, args=(side, sock, spec_hash, events.append), kwargs={"stop": stop}, daemon=True)
    th.start()
    mid = R.connect(f"127.0.0.1:{sock.getsockname()[1]}", sl, "prefix", spec_hash, p)
    e, info = R.edge_split(sl, mid, "prefix", p)
    res = R.evaluate_config(e, _prompts(pf, a.dev), run_dir / "reference", p, run_dir / "correctness" / "inhost_prefix", "inhost prefix")
    mid.close()
    stop.set(); th.join(10); sock.close()
    res.update(_meta(pf, spec_hash, device, R, {"stage": "2b-inhost", "mode": info, "load_memory": load_mem, "memory": R.mem_stats(device),
                                                 "ownership": R.ownership(sl, S.K, S.J, side.sl.layer_ids)}))
    write_json(run_dir / "correctness" / "inhost_prefix.json", res)
    gate_write(run_dir, "inhost_prefix", res["pass"], spec_hash, {"checks": res.get("checks"), "bitwise": res["all_bitwise_equal_reference_hidden"]})
    sys.exit(0 if res["pass"] else EXIT_GATE)


# --------------------------------------------------------------------------- #
# Stages 2c, 3, 4, 5: the two-host split                                       #
# --------------------------------------------------------------------------- #
def _latency(R, e, mid, prompts, p, run_dir, mode, base):
    f = run_dir / "latency" / f"{mode}.jsonl"
    f.unlink(missing_ok=True)
    rows = R.run_latency(e, mid, prompts, p, f, dict(base, mode=mode))
    summ = R.summarize_latency(rows)
    summ.update(mode=mode, label="exploratory; valid only together with the passed correctness gate twohost_" + mode, **base)
    write_json(run_dir / "latency" / f"{mode}_summary.json", summ)
    return summ


def twohost(a):
    run_dir, pf, p, spec_hash, device = _begin(a, "twohost")
    _require(run_dir, ["reference", "inhost_prefix"], spec_hash, "twohost")
    import torch
    import k16_runtime as R
    R.setup_torch()
    model_dir, expected = pf["checkpoint"]["model_dir"], (None if a.dev else S.E0.QWEN3_32B_CONFIG)
    prompts = _prompts(pf, a.dev)
    print(f"[twohost] loading the edge slice (embeddings, layers 0-15 and 48-63, norm, LM head) on {device} ...", flush=True)
    sl = R.edge_slice(model_dir, S.K, S.J, device, torch.bfloat16, expected)
    load_mem = R.load_memory(device)
    own = R.ownership(sl, S.K, S.J)
    own["edge_layers_loaded_equal_spec"] = sl.layer_ids == sorted(S.EDGE_LAYERS)
    exit_code, static_done = 0, False
    try:
        # ---- Stage 2c: correctness in reference mode
        mid = R.connect(a.server, sl, "prefix", spec_hash, p)
        si = mid.server_identity
        base = {"server_host": si.get("host"), "server_device": si.get("device"), "edge_host": socket.gethostname(),
                "edge_device": R.device_identity(device), "spec_hash": spec_hash, "git_commit": pf["git"]["commit"],
                "transport": os.environ.get("K16_TRANSPORT", "direct TCP to " + a.server.rsplit(":", 1)[0]),
                "gpu_shared": bool(si.get("gpu_shared_at_start")) or bool((pf["gpu"].get("gpu") or {}).get("occupied"))}
        same_commit = si.get("git_commit") == pf["git"]["commit"]
        e, info = R.edge_split(sl, mid, "prefix", p)
        res = R.evaluate_config(e, prompts, run_dir / "reference", p, run_dir / "correctness" / "twohost_prefix", "twohost prefix")
        res.update(_meta(pf, spec_hash, device, R, {"stage": "2c-twohost", "mode": info, "server_identity": si, "ownership": own,
                                                     "same_git_commit_on_both_hosts": same_commit, "transport": base["transport"],
                                                     "load_memory": load_mem, "memory": R.mem_stats(device)}))
        ok = bool(res["pass"] and own["edge_slice_ok"] and own["edge_layers_loaded_equal_spec"] and res.get("kv_positions_in_step_all"))
        write_json(run_dir / "correctness" / "twohost_prefix.json", res)
        gate_write(run_dir, "twohost_prefix", ok, spec_hash, {"checks": res.get("checks"), "kv_positions_in_step": res.get("kv_positions_in_step_all"),
                                                             "server": si.get("host"), "server_gpu": (si.get("device") or {}).get("name")})
        if not ok:
            mid.close()
            exit_code = EXIT_GATE
            return
        # ---- Stage 3: generation with text
        _require(run_dir, ["twohost_prefix"], spec_hash, "generation")
        R.use_mode(sl, "prefix")
        tok, chats = R.chat_prompts(model_dir, p, prompts)
        ref = {r["id"]: r for r in read_jsonl(run_dir / "reference" / "chat.jsonl")}
        gen = run_dir / "generation" / "chat.jsonl"
        gen.unlink(missing_ok=True)
        g_ok = True
        for c in chats:
            toks = e.generate(c["ids"], p["chat_max_new"], p["chunk"], eos_ids=() if a.dev else S.E0.EOS_IDS, margins=True)
            r = ref[c["id"]]
            d = next((i for i, (x, y) in enumerate(zip(r["tokens"], toks)) if x != y), None if len(toks) == len(r["tokens"]) else min(len(toks), len(r["tokens"])))
            row = {"id": c["id"], "prompt": c["text"], "n_prompt_tokens": len(c["ids"]), "tokens": toks, "n_new": len(toks),
                   "output_text": tok.decode(toks, skip_special_tokens=True) if tok else None, "kv_positions_in_step": R.kv_invariant(e.steps),
                   "identical_to_unsplit_reference": d is None, "first_divergence": d,
                   "reference_margin_there_nats": None if d is None or d >= len(r["margins"]) else r["margins"][d],
                   "reference_output_text": r.get("output_text"), "mode": "prefix", "utc": utc()}
            g_ok = g_ok and row["kv_positions_in_step"] and len(toks) > 0
            append_jsonl(gen, row)
            print(f"[generation] {c['id']}: {len(toks)} tokens, identical to unsplit: {d is None}\n    {row['output_text']!r}", flush=True)
        gate_write(run_dir, "generation", g_ok, spec_hash, {"n": len(chats)})
        # ---- Stage 4: latency, reference mode
        _require(run_dir, ["twohost_prefix"], spec_hash, "latency")
        R.use_mode(sl, "prefix")
        if device != "cpu":
            torch.cuda.reset_peak_memory_stats(device)
        summ = _latency(R, e, mid, prompts, p, run_dir, "prefix", base)
        write_json(run_dir / "latency" / "prefix_edge_memory.json", R.mem_stats(device))
        gate_write(run_dir, "latency_prefix", summ["all_ok"], spec_hash, {"n_requests": summ["n_requests"]})
        mid.close()
        if not summ["all_ok"]:
            exit_code = EXIT_GATE
            return
        # ---- Stage 5 (optional): static mode under the same criterion; timed only if it passes
        if a.no_static or not si.get("static", {}).get("available"):
            write_json(run_dir / "correctness" / "twohost_static.json", {"ran": False, "why": "disabled" if a.no_static else f"server has no static mode: {si.get('static')}"})
            return
        try:
            mid = R.connect(a.server, sl, "static", spec_hash, p)
            es, info_s = R.edge_split(sl, mid, "static", p)
            res_s = R.evaluate_config(es, prompts, run_dir / "reference", p, run_dir / "correctness" / "twohost_static", "twohost static")
            res_s.update(_meta(pf, spec_hash, device, R, {"stage": "5-twohost-static", "mode": info_s, "server_static": si.get("static"),
                                                           "note": "static mode always runs SDPA with an explicit mask (PREREG_E1OPT §17); flash_attn is never used"}))
            ok_s = bool(res_s["pass"] and res_s.get("kv_positions_in_step_all"))
            write_json(run_dir / "correctness" / "twohost_static.json", res_s)
            gate_write(run_dir, "twohost_static", ok_s, spec_hash, {"checks": res_s.get("checks")})
            if ok_s:
                _require(run_dir, ["twohost_static"], spec_hash, "latency-static")
                R.use_mode(sl, "static")
                summ_s = _latency(R, es, mid, prompts, p, run_dir, "static", base)
                gate_write(run_dir, "latency_static", summ_s["all_ok"], spec_hash, {"n_requests": summ_s["n_requests"]})
            mid.close()
            static_done = True
        except Exception as ex:        # the optional stage never changes the outcome of stages 2-4
            write_json(run_dir / "correctness" / "twohost_static_error.json", {"error": f"{type(ex).__name__}: {ex}", "utc": utc()})
            print(f"[static] optional stage failed: {type(ex).__name__}: {ex}", flush=True)
        finally:
            R.use_mode(sl, "prefix")
    finally:
        if not a.keep_server:
            try:
                R.request_shutdown(a.server, sl, spec_hash, p)
                print("[twohost] asked the server to shut down and free its GPU", flush=True)
            except Exception as ex:
                print(f"[twohost] could not ask the server to shut down: {type(ex).__name__}: {ex}", flush=True)
        write_json(run_dir / "twohost_done.json", {"exit_code": exit_code, "static_stage_completed": static_done, "utc": utc()})
        if exit_code:
            sys.exit(exit_code)


# --------------------------------------------------------------------------- #
# Summary and archive                                                          #
# --------------------------------------------------------------------------- #
def summary(a):
    run_dir = Path(a.run_dir)
    spec_hash = S.spec_hash(a.dev)
    names = ["reference", "inhost_prefix", "twohost_prefix", "generation", "latency_prefix", "twohost_static", "latency_static"]
    gates = {n: (read_json(gate_path(run_dir, n)) if gate_path(run_dir, n).exists() else None) for n in names}
    st = lambda n: "not run" if gates[n] is None else ("PASS" if gates[n]["pass"] and gates[n]["spec_hash"] == spec_hash else "FAIL")
    out = {"experiment": S.NAME, "status": S.STATUS, "utc": utc(), "spec_hash": spec_hash, "run_dir": str(run_dir), "dev": a.dev,
           "gates": {n: st(n) for n in names}, "criteria": S.CRITERIA}
    pf = read_json(run_dir / "preflight.json") if (run_dir / "preflight.json").exists() else {}
    out["preflight"] = {"verdict": pf.get("verdict"), "blockers": pf.get("blockers"), "git": pf.get("git"),
                        "checkpoint_identity_sha256": (pf.get("checkpoint") or {}).get("identity_sha256")}
    for n in ("inhost_prefix", "twohost_prefix", "twohost_static"):
        f = run_dir / "correctness" / f"{n}.json"
        if f.exists():
            r = read_json(f)
            out[n] = {k: r.get(k) for k in ("pass", "checks", "teacher_forced", "all_bitwise_equal_reference_hidden", "kv_positions_in_step_all", "ran", "why")}
            if r.get("greedy"):
                out[n]["greedy"] = {k: v for k, v in r["greedy"].items() if k != "per_prompt"}
    out["latency"] = {}
    for mode in ("prefix", "static"):
        f = run_dir / "latency" / f"{mode}_summary.json"
        if f.exists() and st(f"twohost_{mode}") == "PASS":       # a number is reported only behind its passed correctness gate
            out["latency"][mode] = {k: v for k, v in read_json(f).items() if k in ("groups", "all_ok", "n_requests", "gpu_shared", "label")}
    out["primary_outcome"] = ("two-host k=16 split ran, passed K16-CORRECT-v1 in reference mode and was timed" if st("latency_prefix") == "PASS"
                              else "two-host k=16 split passed K16-CORRECT-v1 in reference mode; latency incomplete" if st("twohost_prefix") == "PASS"
                              else "no validated two-host result: see gates")
    write_json(run_dir / "summary.json", out)
    lines = [f"# {S.NAME} edge run summary", "", f"{S.STATUS}.", "", f"Run directory: `{run_dir}`  ", f"Spec hash: `{spec_hash}`  ",
             f"Git commit: `{(pf.get('git') or {}).get('commit')}`", "", f"Outcome: **{out['primary_outcome']}**", "", "| gate | result |", "|---|---|"]
    lines += [f"| {n} | {out['gates'][n]} |" for n in names]
    for n in ("inhost_prefix", "twohost_prefix", "twohost_static"):
        t = (out.get(n) or {}).get("teacher_forced")
        if t:
            lines += ["", f"{n}: decisive disagreements {t['decisive_disagreements']} of {t['n_decisive']} "
                          f"(limit {S.CRITERIA['max_decisive_disagreement']:.1%}), mean KL {t['mean_kl_nats']:.3e} nats (limit {S.CRITERIA['max_mean_kl_nats']}), "
                          f"greedy identical or near-tie {out[n]['greedy']['identical_or_near_tie']} of {out[n]['greedy']['n_prompts']}, "
                          f"bitwise equal hidden states: {out[n]['all_bitwise_equal_reference_hidden']}."]
    for mode, s in out["latency"].items():
        for g, v in s["groups"].items():
            d = v["decode_step_ms"]
            lines += ["", f"Latency, {mode} mode, {g} prompts ({v['n_in']} in, {v['gen_tokens']} out, {v['n_requests']} requests): TTFT {v['ttft_ms_median']:.0f} ms, "
                          f"{v['tpot_ms_median']:.1f} ms per token ({v['tokens_per_s_median']:.2f} tokens/s). Decode step: edge {d['edge_compute_median']:.1f} ms, "
                          f"server {d['server_compute_median']:.1f} ms, network {d['network_median']:.2f} ms. Exploratory."]
    (run_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    tar = run_dir.parent / (run_dir.name + ".tgz")
    with tarfile.open(tar, "w:gz") as t:
        t.add(run_dir, arcname=run_dir.name, filter=lambda ti: None if ti.name.endswith(".npz") and "/reference/" in ti.name else ti)
    h = hashlib.sha256(tar.read_bytes()).hexdigest()
    Path(str(tar) + ".sha256").write_text(f"{h}  {tar.name}\n")
    print("\n".join(lines))
    print(f"\n[summary] archive {tar} sha256 {h} (reference hidden-state arrays stay in the run directory)", flush=True)


# --------------------------------------------------------------------------- #
# Orchestrator                                                                 #
# --------------------------------------------------------------------------- #
def run_all(a):
    host = socket.gethostname()
    run_dir = Path(a.run_dir) if a.run_dir else Path(a.out).expanduser() / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{host}_edge"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[edge] {S.NAME} ({S.STATUS})\n[edge] run directory: {run_dir}\n[edge] server: {a.server}", flush=True)
    py, here = [sys.executable, "-I"], Path(__file__).resolve().parent
    flags = (["--dev", "--device", a.device] if a.dev else []) + (["--allow-shared"] if a.allow_shared else [])
    pre = py + [str(here / "k16_preflight.py"), "--role", "edge", "--out", str(run_dir / "preflight.json"), "--server", a.server]
    pre += (["--dev"] if a.dev else []) + (["--allow-shared"] if a.allow_shared else [])
    pre += (["--model-dir", a.model_dir] if a.model_dir else []) + (["--gpu", a.gpu] if a.gpu else [])
    stages = [("preflight", pre, None),
              ("reference", py + [str(here / "k16_edge.py"), "reference", "--run-dir", str(run_dir)] + flags, "reference"),
              ("inhost", py + [str(here / "k16_edge.py"), "inhost", "--run-dir", str(run_dir)] + flags, "inhost_prefix"),
              ("twohost", py + [str(here / "k16_edge.py"), "twohost", "--run-dir", str(run_dir), "--server", a.server] + flags
               + (["--no-static"] if a.no_static else []) + (["--keep-server"] if a.keep_server else []), None)]
    spec_hash, rc, log = S.spec_hash(a.dev), 0, []
    for name, cmd, gate in stages:
        if gate and gate_status(run_dir, gate, spec_hash)[0]:
            print(f"[edge] stage {name}: gate {gate} already passed in this run directory; not repeated", flush=True)
            continue
        t0 = time.time()
        print(f"\n[edge] ===== stage {name} =====", flush=True)
        rc = subprocess.run(cmd).returncode
        log.append({"stage": name, "exit_code": rc, "wall_s_not_a_result": round(time.time() - t0, 1), "utc": utc()})
        write_json(run_dir / "stages.json", log)
        if rc != 0:
            print(f"[edge] stage {name} stopped the run (exit {rc}: {'preflight/memory STOP' if rc == EXIT_STOP else 'gate FAIL' if rc == EXIT_GATE else 'error'}). "
                  "Later stages were not started.", flush=True)
            break
    subprocess.run(py + [str(here / "k16_edge.py"), "summary", "--run-dir", str(run_dir)] + (["--dev"] if a.dev else []))
    sys.exit(rc)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("all", "reference", "inhost", "twohost", "summary"):
        s = sub.add_parser(name)
        s.add_argument("--run-dir", default=None, required=name != "all"); s.add_argument("--out", default=S.default_out())
        s.add_argument("--server", default=os.environ.get("K16_SERVER", f"127.0.0.1:{S.DEFAULT_PORT}"),
                       help="local end of the SSH tunnel to the server's loopback port (launch_dgx_edge.sh opens it)")
        s.add_argument("--model-dir", default=None); s.add_argument("--gpu", default=os.environ.get("K16_GPU") or None)
        s.add_argument("--allow-shared", action="store_true", default=os.environ.get("K16_ALLOW_SHARED") == "1")
        s.add_argument("--no-static", action="store_true", default=os.environ.get("K16_NO_STATIC") == "1")
        s.add_argument("--keep-server", action="store_true", default=os.environ.get("K16_KEEP_SERVER") == "1")
        s.add_argument("--dev", action="store_true"); s.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    if ":" not in a.server:
        a.server = f"{a.server}:{S.DEFAULT_PORT}"
    {"all": run_all, "reference": reference, "inhost": inhost, "twohost": twohost, "summary": summary}[a.cmd](a)


if __name__ == "__main__":
    main()
