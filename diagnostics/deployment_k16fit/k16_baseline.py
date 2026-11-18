"""K16-HWFIT matched unsplit baseline: Qwen3-32B whole on the DGX Spark, timed exactly like the split.

The correctness reference of the split run is NOT a latency measurement (64 tokens, no warm-up, no per-step
stamps). This driver measures the unsplit model with everything the split's latency stage fixed:
  model     the reference runtime, k16_runtime.hf_model (Hugging Face Qwen3ForCausalLM, bf16, SDPA, its own KV cache)
  requests  k16_runtime.latency_requests: the same ten prompts, warm-ups, repetitions and seeded order
  length    the same number of forced greedy tokens (256), EOS ignored
  clocks    the same stamps as e1_segment.USplitEdge.generate: start before the prefill, device synchronised after
            every forward, token i stamped after its argmax; TTFT = stamp of token 0, per-token time =
            (last - first) / (n - 1)
  summary   the same function, k16_runtime.summarize_latency
It changes no file the spec hash covers, so its rows carry the spec hash of the split run they are matched to.

  all      preflight (edge role, no server) -> run -> compare.   launch_dgx_baseline.sh runs this.
  run      memory gate (unsplit model) -> load -> timed requests -> baseline/unsplit.jsonl, gate baseline_unsplit
  compare  split run directory + this run -> comparison.json, COMPARISON.md. Refused unless the split run's two-host
           correctness and latency gates passed under the same spec hash, on the same edge host and checkpoint.

Exploratory, like the split run. A shared GPU is recorded on both sides and stated in the comparison.
"""
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # python -I: the script dir is not on sys.path
import k16_spec as S  # noqa: E402
from k16_common import (append_jsonl, read_json, read_jsonl, write_json, utc, gate_write, gate_status, EXIT_STOP, EXIT_GATE)  # noqa: E402


def generate_timed(R, torch, hf, prompt_ids, max_new, chunk, device):
    """Forced-length greedy generation on the unsplit model with USplitEdge.generate's stamps.
    Returns (tokens, token_ns, steps). No server and no network exist here, so those step fields are zero."""
    sync = R.E0RT._sync
    with torch.inference_mode():
        ids = torch.as_tensor(prompt_ids, dtype=torch.long, device=device).view(1, -1)
        P, steps, past = ids.shape[1], [], None

        def forward(x, start):
            nonlocal past
            t0 = time.perf_counter_ns()
            out = hf.model(input_ids=x, past_key_values=past, use_cache=True)
            past = out.past_key_values
            sync(device)
            t1 = time.perf_counter_ns()
            steps.append({"n": x.shape[1], "start": start, "total_ns": t1 - t0, "edge_head_ns": t1 - t0, "edge_tail_ns": 0, "lm_head_ns": 0,
                          "server_compute_ns": 0, "net_ns": 0, "ser_ns": 0, "rt_ns": 0, "bytes_out": 0, "bytes_in": 0})
            return out.last_hidden_state
        t_start = time.perf_counter_ns()
        for s in range(0, P, chunk):
            h = forward(ids[:, s:s + chunk], s)
        out, stamps, pos = [], [], P
        while True:
            t_h = time.perf_counter_ns()
            logits = hf.lm_head(h[:, -1:, :])[0, -1].float()
            tok = int(torch.argmax(logits))
            now = time.perf_counter_ns()
            steps[-1]["lm_head_ns"] = now - t_h
            out.append(tok)
            stamps.append(now - t_start)
            if len(out) >= max_new:
                return out, stamps, steps
            h = forward(torch.tensor([[tok]], device=device), pos)
            pos += 1


def run(a):
    run_dir = Path(a.run_dir)
    pf = read_json(run_dir / "preflight.json")
    spec_hash = S.spec_hash(a.dev)
    if pf.get("verdict") != "CONTINUE" or pf.get("role") != "edge" or pf.get("spec_hash") != spec_hash or pf.get("dev") != a.dev:
        print("[baseline] STOP: preflight.json is not a CONTINUE for this edge and this spec", flush=True)
        sys.exit(EXIT_STOP)
    p, device, gpu = S.params(a.dev), a.device, None
    if not a.dev:
        gpu, device = pf["gpu"]["gpu"], "cuda:0"
        if len(pf["inventory"]["gpus"]) > 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
        import k16_preflight as PF
        g = PF.memory_gate("edge_reference", gpu_uuid=gpu["uuid"], allow_shared=a.allow_shared)
        write_json(run_dir / "gates" / "memory_baseline.json", g)
        print(f"[baseline] memory gate: need {g['need_mib']} MiB, free now {g['free_now_mib']} MiB -> {g['verdict']}", flush=True)
        if not g["ok"]:
            print(f"[baseline] STOP: {'; '.join(g['reasons'])}", flush=True)
            sys.exit(EXIT_STOP)
    import torch
    import numpy as np
    import k16_runtime as R
    R.setup_torch()
    model_dir = pf["checkpoint"]["model_dir"]
    vocab = read_json(Path(model_dir) / "config.json")["vocab_size"] if a.dev else None
    prompts = S.check_prompts(a.dev, vocab)
    print(f"[baseline] loading the unsplit model from {model_dir} directly onto {device} ...", flush=True)
    hf = R.hf_model(model_dir, device)
    load_mem = R.load_memory(device)
    base = {"experiment": S.NAME, "configuration": "unsplit", "runtime": "k16_runtime.hf_model (Hugging Face Qwen3ForCausalLM, bfloat16, SDPA, own KV cache)",
            "edge_host": socket.gethostname(), "edge_device": R.device_identity(device), "spec_hash": spec_hash, "git_commit": pf["git"]["commit"],
            "checkpoint_identity_sha256": pf["checkpoint"]["identity_sha256"], "gpu_shared": bool((gpu or {}).get("occupied")),
            "baseline_source_sha256": S.sha256_file(Path(__file__).resolve()), "mode": "unsplit"}
    out = run_dir / "baseline" / "unsplit.jsonl"
    out.unlink(missing_ok=True)
    rows, ref_dir = [], (Path(a.split_run) / "reference" if a.split_run else None)
    for req in R.latency_requests(prompts, p):
        pr = req["prompt"]
        row = dict(base, prompt_id=pr["id"], group=pr["group"], n_in=len(pr["ids"]), rep=req["rep"], warmup=req["warmup"], gen_tokens=p["gen_tokens"], utc=utc())
        try:
            toks, stamps, steps = generate_timed(R, torch, hf, pr["ids"], p["gen_tokens"], p["chunk"], device)
            for s in steps:
                s["phase"] = "prefill" if (s["n"] > 1 or s["start"] < len(pr["ids"])) else "decode"
            row.update(ok=len(toks) == p["gen_tokens"], token_ns=stamps, steps=steps, bytes_out=0, bytes_in=0)
            if ref_dir is not None and (ref_dir / f"{pr['id']}.npz").exists():      # the same model must repeat its own reference tokens
                ref = np.load(ref_dir / f"{pr['id']}.npz")["tokens"].tolist()
                k = min(len(ref), len(toks))
                row["first_tokens_equal_split_run_reference"] = toks[:k] == ref[:k]
        except Exception as ex:
            row.update(ok=False, error=f"{type(ex).__name__}: {ex}")
        append_jsonl(out, row)
        rows.append(row)
        if not row["ok"]:
            break
        print(f"[baseline] {pr['id']} rep {req['rep']}: {(stamps[-1] - stamps[0]) / 1e6 / max(1, len(stamps) - 1):.1f} ms/token", flush=True)
    summ = R.summarize_latency(rows)
    same = [r.get("first_tokens_equal_split_run_reference") for r in rows if "first_tokens_equal_split_run_reference" in r]
    summ.update(base, label="exploratory; unsplit baseline matched to the split run's latency stage", load_memory=load_mem, memory=R.mem_stats(device),
                requests_checked_against_split_reference=len(same), first_tokens_equal_split_run_reference=all(same) if same else None,
                params={k: p[k] for k in ("gen_tokens", "warmups", "reps", "chunk")}, order_seed=S.LATENCY["order_seed"])
    write_json(run_dir / "baseline" / "unsplit_summary.json", summ)
    gate_write(run_dir, "baseline_unsplit", summ["all_ok"], spec_hash, {"n_requests": summ["n_requests"]})
    sys.exit(0 if summ["all_ok"] else EXIT_GATE)


def _key(rows):
    return sorted((r["prompt_id"], r["n_in"], r["gen_tokens"], r["rep"], bool(r["warmup"])) for r in rows)


def compare(a):
    """Split run versus this unsplit run, same summary code on both sets of raw rows. Refuses an unmatched pair."""
    import k16_runtime as R
    run_dir, split = Path(a.run_dir), Path(a.split_run)
    spec_hash = S.spec_hash(a.dev)
    problems = []
    for g in ("twohost_prefix", "latency_prefix"):
        ok, why = gate_status(split, g, spec_hash)
        if not ok:
            problems.append(f"split run: {why}")
    ok, why = gate_status(run_dir, "baseline_unsplit", spec_hash)
    if not ok:
        problems.append(f"baseline: {why}")
    if problems:
        print("[compare] REFUSED: " + "; ".join(problems), flush=True)
        sys.exit(EXIT_GATE)
    base_rows = read_jsonl(run_dir / "baseline" / "unsplit.jsonl")
    modes = {"prefix": read_jsonl(split / "latency" / "prefix.jsonl")}
    if gate_status(split, "twohost_static", spec_hash)[0] and gate_status(split, "latency_static", spec_hash)[0]:
        modes["static"] = read_jsonl(split / "latency" / "static.jsonl")
    pf_b, pf_s = read_json(run_dir / "preflight.json"), read_json(split / "preflight.json")
    b0 = base_rows[0]
    matched = {"same_spec_hash": all(r["spec_hash"] == spec_hash for rows in list(modes.values()) + [base_rows] for r in rows),
               "same_requests": all(_key(rows) == _key(base_rows) for rows in modes.values()),
               "same_edge_host": all(r["edge_host"] == b0["edge_host"] for rows in modes.values() for r in rows),
               "same_edge_device": all(r["edge_device"]["name"] == b0["edge_device"]["name"] for rows in modes.values() for r in rows),
               "same_checkpoint": pf_b["checkpoint"]["identity_sha256"] == pf_s["checkpoint"]["identity_sha256"]}
    if not all(matched.values()):
        print(f"[compare] REFUSED: the two runs are not matched: {matched}", flush=True)
        sys.exit(EXIT_GATE)
    sb = R.summarize_latency(base_rows)
    out = {"experiment": S.NAME, "status": S.STATUS, "utc": utc(), "spec_hash": spec_hash, "split_run": str(split), "baseline_run": str(run_dir),
           "matched": matched, "definitions": "TTFT = stamp of token 0 from the call; per-token = (last stamp - first stamp) / (n - 1); medians over measured requests",
           "caveats": ["the two runs were not interleaved in time", "the unsplit runtime keeps a growing Hugging Face KV cache; the split runtime preallocates its caches",
                       f"GPU shared during the unsplit run: {b0['gpu_shared']}",
                       f"GPU shared during the split run: {any(r.get('gpu_shared') for rows in modes.values() for r in rows)}",
                       "exploratory; one run of each"], "groups": {}}
    lines = [f"# {S.NAME}: split versus unsplit on the same edge host (exploratory)", "", f"Split run: `{split}`  ", f"Unsplit run: `{run_dir}`  ",
             f"Spec hash: `{spec_hash}`", "", "| prompts | configuration | TTFT ms | ms per token | tokens/s | per-token vs unsplit |", "|---|---|---|---|---|---|"]
    for g, u in sb["groups"].items():
        row = {"unsplit": {k: u[k] for k in ("ttft_ms_median", "tpot_ms_median", "tokens_per_s_median", "n_requests", "n_in", "gen_tokens")}}
        lines.append(f"| {g} ({u['n_in']} in, {u['gen_tokens']} out) | unsplit on the edge | {u['ttft_ms_median']:.0f} | {u['tpot_ms_median']:.1f} | {u['tokens_per_s_median']:.2f} | 1.00 |")
        for mode, rows in modes.items():
            s = R.summarize_latency(rows)["groups"][g]
            ratio = s["tpot_ms_median"] / u["tpot_ms_median"]
            row[f"split_{mode}"] = {"ttft_ms_median": s["ttft_ms_median"], "tpot_ms_median": s["tpot_ms_median"], "tokens_per_s_median": s["tokens_per_s_median"],
                                    "per_token_ratio_split_over_unsplit": ratio, "ttft_ratio_split_over_unsplit": s["ttft_ms_median"] / u["ttft_ms_median"],
                                    "split_is_faster_per_token": bool(ratio < 1.0), "decode_step_ms": s["decode_step_ms"]}
            lines.append(f"| | split, {mode} mode | {s['ttft_ms_median']:.0f} | {s['tpot_ms_median']:.1f} | {s['tokens_per_s_median']:.2f} | {ratio:.2f} |")
        out["groups"][g] = row
    lines += ["", "A ratio below 1 means the split produced tokens faster than the whole model on the DGX Spark alone.", ""] + [f"- {c}" for c in out["caveats"]]
    write_json(run_dir / "comparison.json", out)
    (run_dir / "COMPARISON.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def latest_split_run(out_root, spec_hash):
    """Newest *_edge run directory under out_root whose two-host correctness and latency gates passed on this spec hash."""
    for d in sorted(Path(out_root).expanduser().glob("*_edge"), reverse=True):
        if d.is_dir() and gate_status(d, "twohost_prefix", spec_hash)[0] and gate_status(d, "latency_prefix", spec_hash)[0]:
            return str(d)
    return None


def run_all(a):
    host, spec_hash = socket.gethostname(), S.spec_hash(a.dev)
    run_dir = Path(a.run_dir) if a.run_dir else Path(a.out).expanduser() / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{host}_baseline"
    run_dir.mkdir(parents=True, exist_ok=True)
    split = a.split_run or latest_split_run(a.out, spec_hash)
    print(f"[baseline] {S.NAME} matched unsplit baseline ({S.STATUS})\n[baseline] run directory: {run_dir}\n[baseline] split run to match: {split}", flush=True)
    if split is None:
        print("[baseline] STOP: no split run with passed two-host gates on this spec hash was found; set K16_SPLIT_RUN", flush=True)
        sys.exit(EXIT_STOP)
    for g in ("twohost_prefix", "latency_prefix"):          # nothing is loaded for a split run that cannot be compared
        ok, why = gate_status(split, g, spec_hash)
        if not ok:
            print(f"[baseline] REFUSED: split run {split}: {why}", flush=True)
            sys.exit(EXIT_GATE)
    py, here = [sys.executable, "-I"], Path(__file__).resolve().parent
    dev = (["--dev"] if a.dev else [])
    pre = py + [str(here / "k16_preflight.py"), "--role", "edge", "--out", str(run_dir / "preflight.json")] + dev
    pre += (["--allow-shared"] if a.allow_shared else []) + (["--model-dir", a.model_dir] if a.model_dir else []) + (["--gpu", a.gpu] if a.gpu else [])
    common = ["--run-dir", str(run_dir), "--split-run", split] + dev + (["--device", a.device] if a.dev else []) + (["--allow-shared"] if a.allow_shared else [])
    rc, log = 0, []
    for name, cmd in (("preflight", pre), ("run", py + [str(here / "k16_baseline.py"), "run"] + common), ("compare", py + [str(here / "k16_baseline.py"), "compare"] + common)):
        t0 = time.time()
        print(f"\n[baseline] ===== stage {name} =====", flush=True)
        rc = subprocess.run(cmd).returncode
        log.append({"stage": name, "exit_code": rc, "wall_s_not_a_result": round(time.time() - t0, 1), "utc": utc()})
        write_json(run_dir / "stages.json", log)
        if rc != 0:
            print(f"[baseline] stage {name} stopped the run (exit {rc}); later stages were not started.", flush=True)
            break
    sys.exit(rc)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("all", "run", "compare"):
        s = sub.add_parser(name)
        s.add_argument("--run-dir", default=None, required=name != "all"); s.add_argument("--out", default=S.default_out())
        s.add_argument("--split-run", default=os.environ.get("K16_SPLIT_RUN") or None, help="the split run directory to match (default: newest passed *_edge run)")
        s.add_argument("--model-dir", default=None); s.add_argument("--gpu", default=os.environ.get("K16_GPU") or None)
        s.add_argument("--allow-shared", action="store_true", default=os.environ.get("K16_ALLOW_SHARED") == "1")
        s.add_argument("--dev", action="store_true"); s.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    if a.cmd == "compare" and not a.split_run:
        sys.exit("compare needs --split-run")
    {"all": run_all, "run": run, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    main()
