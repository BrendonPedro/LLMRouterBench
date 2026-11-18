"""GEMMA-SELECT drivers (PREREG §5; exact order in RUNBOOK.md).

  preflight    --role edge|server --site F    CPU only: identity, hashes, load estimate, free-memory gate
  serve        --site F                        server: the split's server half until Ctrl-C
  profile      --role edge|server --site F    one segment alone (secondary bound)
  reference    --site F                        edge: whole float32 model on the correctness set (no timers)
  correctness  --site F                        edge: split against that reference (no timers) -> PASS / FAIL
  timed        --site F --block S1|Es|Em|S2   edge: end-to-end timing on the utility set
  score        --edge-run DIR                  analysis host: unchanged benchmark evaluators
  summary      --edge-run DIR --server-run DIR --out DIR

Every target command first checks the frozen tag, the site file, the host role, the pinned source and checkpoint and
the free memory. --dev is for CPU tests with a test double only; dev outputs are refused by `summary`.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gsel_adapter as AD  # noqa: E402
import gsel_common as C  # noqa: E402
import gsel_spec as S  # noqa: E402
import gsel_stats as ST  # noqa: E402
import gsel_workload as WL  # noqa: E402

NEED = {"server": ("server_float32", S.LOAD_FACTOR), "split_edge": ("edge_float32", S.LOAD_FACTOR),
        "Em": ("whole_float32", S.WHOLE_LOAD_FACTOR), "Es": ("whole_bfloat16", S.WHOLE_LOAD_FACTOR)}
STATE = {"run_dir": None, "hdr": None}


def versions():
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for p in ("torch", "transformers", "safetensors", "tokenizers", "numpy"):
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = None
    return out


def prepare(a, role, loads=(), deep=False):
    """Checks that precede every command. Returns (site, run_dir, header). Nothing is loaded before this returns."""
    fz = C.freeze_check(a.dev)
    site = C.load_site(a.site, role)
    if site.get("gpu_uuid") and str(site["device"]).startswith("cuda"):
        os.environ["CUDA_VISIBLE_DEVICES"] = site["gpu_uuid"]          # no other GPU is visible to this process
    host = C.check_role(role, site, dev=a.dev)
    src = ck = est = gate = None
    if not a.dev:
        src = C.check_source(site["source_root"])
        ck = C.check_checkpoint(site["model_dir"], deep=deep)
        est = C.estimate_bytes(site["model_dir"])
        if loads:
            gate = C.memory_gate(role, site, max(C.need_bytes(est[NEED[load][0]], NEED[load][1]) for load in loads))
    hdr = {"schema": S.SCHEMA, "experiment": S.NAME, "freeze": fz, "dev": bool(a.dev), "role": role, "site_sha256": site["_sha256"],
           "host": host, "utc_start": C.utc(), "source": src, "checkpoint": ck, "estimate": est, "memory_gate": gate,
           "versions": versions(), "others_on_gpu_start": C.gpu_others(site.get("gpu_uuid"), (os.getpid(),))}
    STATE.update(run_dir=Path(site["run_dir"]), hdr=hdr)
    return site, Path(site["run_dir"]), hdr


def record_failure(cmd, exc):
    """A command that died after its checks passed leaves a record (type and first words only). Stops leave none."""
    rd = STATE["run_dir"]
    if rd is None:
        return
    n = 1
    while (rd / f"failure_{cmd}.{n}.json").exists():
        n += 1
    C.write_json(rd / f"failure_{cmd}.{n}.json", dict(STATE["hdr"], utc_end=C.utc(), command=cmd, error_type=type(exc).__name__,
                                                       error=str(exc)[:300]))


def gpu_gate(site, need):
    """After torch is imported and before any weight is loaded: the device itself must report the memory as free."""
    import torch
    dev = site["device"]
    if not str(dev).startswith("cuda"):
        return {"device": "cpu"}
    if torch.cuda.device_count() != 1:
        raise C.Stop("exactly one GPU must be visible to this process")
    uuid = C.norm_uuid(getattr(torch.cuda.get_device_properties(0), "uuid", ""))
    if site.get("gpu_uuid") and uuid != C.norm_uuid(site["gpu_uuid"]):
        raise C.Stop("the visible GPU is not the site's GPU")
    free, total = torch.cuda.mem_get_info(0)
    if free < need:
        raise C.Stop(f"BLOCKED, device reports {free >> 20} MiB free < required {need >> 20} MiB (nothing was loaded)")
    return {"device": dev, "free_bytes": free, "total_bytes": total, "need_bytes": need}


def need_for(hdr, load):
    return 0 if hdr["estimate"] is None else C.need_bytes(hdr["estimate"][NEED[load][0]], NEED[load][1])


def finish(hdr, site, **body):
    return dict(hdr, utc_end=C.utc(), others_on_gpu_end=C.gpu_others(site.get("gpu_uuid"), (os.getpid(),)), **body)


# --------------------------------------------------------------------------- #
def cmd_preflight(a):
    site, rd, hdr = prepare(a, a.role, deep=True)
    C.check_credential(site)
    gates, blocked = {}, []
    for load in (("server",) if a.role == "server" else ("split_edge", "Em", "Es")):
        try:
            gates[load] = C.memory_gate(a.role, site, need_for(hdr, load)) if not a.dev else {"pass": True}
        except C.Stop:
            gates[load] = {"pass": False, "need_bytes": need_for(hdr, load)}
            blocked.append(load)
    listening = not C.port_free(site["port"])
    port_ok = (not listening) if a.role == "server" else listening
    rec = finish(hdr, site, gates=gates, port={"listening": listening, "ok": port_ok},
                 verdict="GO" if not blocked and port_ok else "BLOCKED")
    name = "preflight.json"
    n = 1
    while (rd / name).exists():
        n += 1
        name = f"preflight.{n}.json"
    C.write_json(rd / name, rec)
    need = {k: v.get("need_bytes", 0) >> 20 for k, v in gates.items()}
    print(f"preflight {a.role}: {rec['verdict']}  required MiB {need}  port {'listening' if listening else 'free'}"
          + ("" if port_ok else "  (server: the port must be free; edge: the SSH forward must be up)"))
    sys.exit(0 if rec["verdict"] == "GO" else 3)


def cmd_serve(a):
    site, rd, hdr = prepare(a, "server", loads=("server",))
    if not C.port_free(site["port"]):
        raise C.Stop("the site's port is already in use")
    adapter, info = AD.load(site, "server", a.dev)
    gpu = gpu_gate(site, need_for(hdr, "server"))
    served = adapter.serve()
    n = 1
    while (rd / f"server_record.{n}.json").exists():
        n += 1
    C.write_json(rd / f"server_record.{n}.json", finish(hdr, site, adapter=info, gpu_gate=gpu, served=served))
    print(f"server record written: server_record.{n}.json")


def cmd_profile(a):
    load = "server" if a.role == "server" else "split_edge"
    site, rd, hdr = prepare(a, a.role, loads=(load,))
    adapter, info = AD.load(site, a.role, a.dev)
    gpu = gpu_gate(site, need_for(hdr, load))
    lengths = S.PROFILE_LENGTHS if not a.dev else (4, 8, 16)
    steps = S.PROFILE_DECODE_STEPS if not a.dev else 3
    rows = []
    if a.role == "server":
        for q in lengths:
            rows += [adapter.server_profile(q, steps) for _ in range(S.PROFILE_REPS)]
    else:
        import gsel_whole as W
        tok = W.load_tokenizer(site["model_dir"])
        adapter.connect(require_server=False)
        pool = []
        for q in WL.load(a.dev)["utility"]:
            pool += W.tokenize(tok, q["prompt"])
            if len(pool) >= max(lengths) + steps:
                break
        for q in lengths:
            rows += [adapter.edge_profile(pool[:q], steps) for _ in range(S.PROFILE_REPS)]
    C.write_json(rd / f"profile_{a.role}.json", finish(hdr, site, adapter=info, gpu_gate=gpu, profile=rows,
                                                       model=ST.segment_model(rows), memory=adapter.memory()))
    adapter.close()
    print(f"profile_{a.role}.json written")


def _ids_sha(ids):
    return C.sha256_bytes(json.dumps(ids).encode())


def cmd_reference(a):
    site, rd, hdr = prepare(a, "edge", loads=("Em",))
    import torch
    import gsel_whole as W
    gpu = gpu_gate(site, need_for(hdr, "Em"))
    tok = W.load_tokenizer(site["model_dir"])
    model, minfo = W.load_model(site["model_dir"], *([S.WHOLE_DTYPE["Em"][0], site["device"], S.WHOLE_DTYPE["Em"][1]]))
    runner, items = W.WholeRunner(model, site["device"]), []
    n_cont = S.N_CONT if not a.dev else 6
    for i, q in enumerate(WL.load(a.dev)["correctness"]):
        ids = W.tokenize(tok, q["prompt"])
        toks = runner.generate(ids, n_cont, eos=())["tokens"]                       # EOS ignored: a fixed-length continuation
        rows = torch.stack([r.cpu() for _, r in runner.teacher_forced(ids, toks)])     # [n_cont, V] float32
        lp = torch.log_softmax(rows.double(), dim=-1)
        top = torch.topk(lp, 2, dim=-1)
        f = rd / f"reference_logits_{i}.pt"
        if f.exists():
            raise FileExistsError(f"{f} exists: outputs are written once")
        rd.mkdir(parents=True, exist_ok=True)
        torch.save(rows, f)
        items.append({"key": q["key"], "n_prompt": len(ids), "ids_sha256": _ids_sha(ids), "tokens": toks,
                      "margins": [float(x) for x in (top.values[:, 0] - top.values[:, 1])],
                      "self_consistent": [int(x) for x in top.indices[:, 0]] == toks, "finite": bool(torch.isfinite(rows).all()),
                      "logits_file": f.name, "logits_sha256": C.sha256_file(f)})
    C.write_json(rd / "reference.json", finish(hdr, site, gpu_gate=gpu, model=minfo, items=items, eos=W.eos_ids(site["model_dir"]),
                 memory=W.cuda_memory(site["device"]),
                 screen_decision={"stage": 1, "decision": "EDGE",
                                  "basis": "the split's model is the whole float32 model, which loaded on the edge: quality lead "
                                           "0.0 points < 3.0 by identity (PREREG §6)"}))
    print(f"reference.json written ({len(items)} prompts); screen decision recorded: EDGE at Stage 1")


def cmd_correctness(a):
    site, rd, hdr = prepare(a, "edge", loads=("split_edge",))
    ref = C.read_json(rd / "reference.json")
    if ref["dev"] != bool(a.dev) or ref["freeze"] != hdr["freeze"]:
        raise C.Stop("reference.json belongs to another freeze or mode")
    import torch
    import gsel_whole as W
    adapter, info = AD.load(site, "edge", a.dev)
    gpu = gpu_gate(site, need_for(hdr, "split_edge"))
    tok = W.load_tokenizer(site["model_dir"])
    conn = adapter.connect()
    if sorted(conn["eos"]) != ref["eos"]:
        raise C.Stop("the split runtime's EOS set differs from generation_config.json")
    queries, positions, greedy = {q["key"]: q for q in WL.load(a.dev)["correctness"]}, [], []
    for item in ref["items"]:
        ids = W.tokenize(tok, queries[item["key"]]["prompt"])
        if _ids_sha(ids) != item["ids_sha256"]:
            raise C.Stop("token ids differ from the reference run")
        f = rd / item["logits_file"]
        if C.sha256_file(f) != item["logits_sha256"]:
            raise C.Stop("reference logits changed on disk")
        ref_lp = torch.log_softmax(torch.load(f).double(), dim=-1)
        for i, row in adapter.teacher_forced(ids, item["tokens"]):
            row = torch.as_tensor(row).detach().cpu().double().reshape(-1)
            finite = bool(torch.isfinite(row).all())
            lp = torch.log_softmax(row, dim=-1)
            kl = float((ref_lp[i].exp() * (ref_lp[i] - lp)).sum()) if finite else None
            positions.append({"key": item["key"], "i": i, "ref_top1": item["tokens"][i], "split_top1": int(row.argmax()),
                              "ref_margin": item["margins"][i], "kl": kl, "finite": finite and kl is not None and kl == kl})
        got = adapter.generate(ids, len(item["tokens"]), stop_on_eos=False)["tokens"]
        d = ST.first_divergence(item["tokens"], got)
        greedy.append({"key": item["key"], "identical": d is None, "first_divergence": d, "tokens": got,
                       "ref_margin_there": item["margins"][d] if d is not None and d < len(item["margins"]) else None})
    gate = ST.correctness_gate(positions, greedy)
    C.write_json(rd / "correctness.json", finish(hdr, site, adapter=info, gpu_gate=gpu, connect=conn, positions=positions,
                                                 greedy=greedy, gate=gate, memory=adapter.memory()))
    adapter.close()
    print(f"correctness: {gate['verdict']}  decisive agreement {gate['decisive_agreement']}  mean KL {gate['mean_kl_nats']}  "
          f"greedy ok {gate['greedy_ok']}/{gate['n_prompts']}")
    sys.exit(0 if gate["verdict"] == "PASS" else 2)


def cmd_timed(a):
    cfg = S.BLOCK_CONFIG[a.block]
    load = "split_edge" if cfg == "S" else cfg
    site, rd, hdr = prepare(a, "edge", loads=(load,))
    corr = C.read_json(rd / "correctness.json") if (rd / "correctness.json").exists() else None
    if corr is None or corr["gate"]["verdict"] != "PASS" or corr["dev"] != bool(a.dev) or corr["freeze"] != hdr["freeze"]:
        raise C.Stop("no timing without a correctness PASS under this freeze (PREREG §7)")
    for prev in S.BLOCKS[:S.BLOCKS.index(a.block)]:
        if not (rd / f"timed_{prev}.done.json").exists():
            raise C.Stop(f"block {prev} must complete before {a.block}")
    out, done = rd / f"timed_{a.block}.jsonl", rd / f"timed_{a.block}.done.json"
    if done.exists():
        raise C.Stop(f"block {a.block} is complete: outputs are written once")
    attempts = len(list(rd.glob(f"timed_{a.block}.jsonl.crashed-*")))
    if out.exists():                                                        # an earlier attempt of this block died
        if attempts >= S.MAX_BLOCK_REPEATS:
            raise C.Stop(f"block {a.block} already repeated {attempts} times: INCOMPLETE")
        attempts += 1
        out.rename(rd / f"timed_{a.block}.jsonl.crashed-{attempts}")
    import gsel_whole as W
    tok = W.load_tokenizer(site["model_dir"])
    eos = W.eos_ids(site["model_dir"])
    wl = WL.load(a.dev)
    queries = wl["sentinel"] if a.block == "S2" else wl["utility"]
    max_new, cutoff = S.MAX_NEW_TOKENS, S.QUERY_CUTOFF_S
    if a.dev:
        queries, max_new = queries[:a.limit or 4], a.max_new or 8
    gpu = gpu_gate(site, need_for(hdr, load))
    if cfg == "S":
        adapter, info = AD.load(site, "edge", a.dev)
        conn = adapter.connect()
        if sorted(conn["eos"]) != eos:
            raise C.Stop("the split runtime's EOS set differs from generation_config.json")
        gen = lambda ids, n, cb: adapter.generate(ids, n, on_token=cb, stop_on_eos=True)  # noqa: E731
        mem = adapter.memory
    else:
        model, info = W.load_model(site["model_dir"], S.WHOLE_DTYPE[cfg][0], site["device"], S.WHOLE_DTYPE[cfg][1])
        runner = W.WholeRunner(model, site["device"])
        gen = lambda ids, n, cb: runner.generate(ids, n, eos, on_token=cb, stop=AD.Cutoff)  # noqa: E731
        mem = lambda: W.cuda_memory(site["device"])  # noqa: E731
    gen(W.tokenize(tok, wl["sentinel"][0]["prompt"]), S.WARMUP_NEW_TOKENS, None)   # warm-up, not recorded
    rd.mkdir(parents=True, exist_ok=True)
    for n, q in enumerate(queries):
        ids = W.tokenize(tok, q["prompt"])
        stamps = []
        t0 = time.perf_counter()

        def on_token(i, t):
            stamps.append(time.perf_counter() - t0)
            if stamps[-1] > cutoff:
                raise AD.Cutoff()
        res = gen(ids, max_new, on_token)
        toks = res["tokens"]
        C.append_jsonl(out, {"n": n, "key": q["key"], "dataset": q["dataset"], "block": a.block, "config": cfg, "n_prompt": len(ids),
                             "ids_sha256": _ids_sha(ids), "tokens": toks, "n_out": len(toks), "eos": bool(toks and toks[-1] in eos),
                             "cutoff": bool(res["cutoff"]), "ttft_s": stamps[0], "total_s": stamps[-1], "stamps_s": stamps,
                             "impl": res.get("impl"), "text": tok.decode(toks, skip_special_tokens=True)})
        print(f"[{a.block}] {n + 1}/{len(queries)} {q['key']} out={len(toks)} total={stamps[-1]:.2f}s", flush=True)
    C.write_json(done, finish(hdr, site, block=a.block, config=cfg, runner=info, gpu_gate=gpu, n_rows=len(queries), attempts=attempts + 1,
                              rows_sha256=C.sha256_file(out), max_new_tokens=max_new, cutoff_s=cutoff, memory=mem()))
    print(f"block {a.block} complete")


# --------------------------------------------------------------------------- #
# Analysis host                                                                #
# --------------------------------------------------------------------------- #
def evaluators():
    sys.path.insert(0, str(C.REPO))
    from evaluation.GPQA.gpqa import GPQAEvaluator
    from evaluation.LiveMathBench.livemathbench import LiveMathBenchEvaluator
    from evaluation.MMLUPro.mmlupro import MMLUProEvaluator
    return {"mmlupro": MMLUProEvaluator(), "gpqa": GPQAEvaluator(), "livemathbench": LiveMathBenchEvaluator()}


def cmd_score(a):
    C.freeze_check(a.dev)
    rd, ev = Path(a.edge_run), None if a.scorer else evaluators()
    answers = {q["key"]: q["answer"] for q in WL.load(a.dev)["utility"]}
    for block in S.BLOCKS:
        f = rd / f"timed_{block}.jsonl"
        if not f.exists() or not (rd / f"timed_{block}.done.json").exists():
            continue
        scores = {}
        for r in C.read_jsonl(f):
            if a.scorer:                                                    # dev only: exact text match, for pipeline tests
                ok = r["text"].strip() == str(answers[r["key"]]).strip()
            else:
                ok = ev[r["dataset"]].evaluate({"answer": answers[r["key"]]}, r["text"]).get("is_correct", False)
            scores[r["key"]] = 1 if ok is True or ok == 1 else 0
        C.write_json(rd / f"scores_{block}.json", {"schema": S.SCHEMA, "block": block, "rows_sha256": C.sha256_file(f),
                                                   "scorer": a.scorer or "unchanged benchmark evaluators", "scores": scores})
        print(f"scores_{block}.json: {sum(scores.values())}/{len(scores)} correct")


def _block(rd, block, dev):
    done = C.read_json(rd / f"timed_{block}.done.json")
    f = rd / f"timed_{block}.jsonl"
    sc = C.read_json(rd / f"scores_{block}.json")
    if done["dev"] != dev:
        raise C.Stop(f"block {block} is a {'dev' if done['dev'] else 'target'} output")
    if C.sha256_file(f) != done["rows_sha256"] or sc["rows_sha256"] != done["rows_sha256"]:
        raise C.Stop(f"block {block}: rows changed after the block completed")
    rows = C.read_jsonl(f)
    if len(rows) != done["n_rows"] or [r["n"] for r in rows] != list(range(len(rows))):
        raise C.Stop(f"block {block}: rows missing or out of order")
    return done, rows, sc["scores"]


def summarise(edge_run, server_run, dev=False, n_boot=None):
    rd = Path(edge_run)
    fails = lambda d, name: len(list(Path(d).glob(f"failure_{name}.*.json"))) if d else 0  # noqa: E731
    base = {"schema": S.SCHEMA, "experiment": S.NAME, "dev": dev, "cells": [], "blocks_complete": False,
            "failed_attempts": {"reference": fails(rd, "reference"), "correctness": fails(rd, "correctness"), "serve": fails(server_run, "serve")}}
    if not (rd / "reference.json").exists():                                # the matched edge competitor never ran
        return dict(base, outcome="NOT_EVALUABLE" if base["failed_attempts"]["reference"] >= 2 else "INCOMPLETE")
    ref = C.read_json(rd / "reference.json")
    if ref["dev"] != dev:
        raise C.Stop("dev outputs cannot be summarised as a result, nor target outputs as dev")
    if not (rd / "correctness.json").exists():                              # the split never completed its first stage
        infeasible = base["failed_attempts"]["correctness"] + base["failed_attempts"]["serve"] >= 2
        return dict(base, freeze=ref["freeze"], screen_decision=ref["screen_decision"],
                    outcome="SPLIT_INFEASIBLE" if infeasible else "INCOMPLETE")
    corr = C.read_json(rd / "correctness.json")
    gate = ST.correctness_gate(corr["positions"], corr["greedy"])           # recomputed from the raw record
    if gate["verdict"] != corr["gate"]["verdict"]:
        raise C.Stop("stored correctness verdict does not match its recomputation")
    freezes = {json.dumps(x["freeze"], sort_keys=True) for x in (ref, corr)}
    mem = lambda m: {k: (m or {}).get(k) for k in ("peak_allocated_bytes", "total_bytes")}  # noqa: E731  (no device identifiers)
    out = dict(base, freeze=ref["freeze"], screen_decision=ref["screen_decision"], correctness=dict(gate),
               memory={"reference_Em": mem(ref["memory"]), "split_edge": mem(corr["memory"])}, estimate=ref["estimate"])
    envs = {json.dumps(x["versions"], sort_keys=True) for x in (ref, corr)}
    blocks = {}
    for b in S.BLOCKS:
        if (rd / f"timed_{b}.done.json").exists() and (rd / f"scores_{b}.json").exists():
            blocks[b] = _block(rd, b, dev)
            freezes.add(json.dumps(blocks[b][0]["freeze"], sort_keys=True))
            envs.add(json.dumps(blocks[b][0]["versions"], sort_keys=True))
    if len(freezes) != 1:
        raise C.Stop("outputs come from more than one freeze")
    out["environment_consistent"] = len(envs) == 1                          # one interpreter and library set for every edge stage
    complete = all(b in blocks for b in S.BLOCKS) and out["environment_consistent"]
    cells, stab, quality, latency = [], {"stable": False, "ratio": None, "n": 0}, {}, {}
    if complete:
        per = {b: {r["key"]: r for r in blocks[b][1]} for b in ("S1", "Es", "Em")}
        keys = [r["key"] for r in blocks["S1"][1]]
        for c, b in (("Es", "Es"), ("Em", "Em")):
            if [r["key"] for r in blocks[b][1]] != keys or any(per[b][k]["ids_sha256"] != per["S1"][k]["ids_sha256"] for k in keys):
                raise C.Stop(f"block {b} did not run the same queries with the same token ids as S1")
        t = lambda r: float("inf") if r["cutoff"] else r["total_s"]  # noqa: E731
        rows = [{"key": k, "bench": per["S1"][k]["dataset"],
                 "correct": {"S": blocks["S1"][2][k], "Em": blocks["Em"][2][k], "Es": blocks["Es"][2][k]},
                 "time": {"S": t(per["S1"][k]), "Em": t(per["Em"][k]), "Es": t(per["Es"][k])}} for k in keys]
        cells = ST.decide(rows, n_boot=n_boot)
        stab = ST.stability(blocks["S1"][1], blocks["S2"][1])
        for c, b in (("S", "S1"), ("Em", "Em"), ("Es", "Es")):
            rs = blocks[b][1]
            quality[c] = {d: 100.0 * sum(blocks[b][2][r["key"]] for r in rs if r["dataset"] == d) / max(1, sum(r["dataset"] == d for r in rs))
                          for d in S.BENCHMARKS}
            quality[c]["all"] = 100.0 * sum(blocks[b][2].values()) / len(rs)
            tp = [(r["total_s"] - r["ttft_s"]) / (r["n_out"] - 1) for r in rs if r["n_out"] > 1]
            latency[c] = {"ttft_s": ST.median([r["ttft_s"] for r in rs]), "per_token_s": ST.median(tp),
                          "total_s": ST.median([r["total_s"] for r in rs]), "p90_total_s": ST.quantile([r["total_s"] for r in rs], 0.9),
                          "cutoffs": sum(r["cutoff"] for r in rs), "output_tokens": sum(r["n_out"] for r in rs),
                          "peak_allocated_bytes": (blocks[b][0]["memory"] or {}).get("peak_allocated_bytes")}
        impl = [r["impl"] for r in blocks["S1"][1] if r.get("impl")]
        if impl:                                                            # the link is measured, never assumed
            wire = [i["roundtrip_s"] - i["server_compute_s"] for i in impl]
            out["network"] = {"n": len(impl), "request_bytes_median": ST.median([i["request_bytes"] for i in impl]),
                              "round_trip_minus_server_compute_s_median": ST.median(wire),
                              "share_of_completion_time_median": ST.median([w / r["total_s"] for w, r in zip(wire, blocks["S1"][1]) if r.get("impl")]),
                              "effective_mbit_s_median": ST.median([8e-6 * i["request_bytes"] / w for i, w in zip(impl, wire) if w > 0])}
        pe, ps = rd / "profile_edge.json", Path(server_run) / "profile_server.json" if server_run else None
        if pe.exists() and ps is not None and ps.exists():
            meta = {r["key"]: (r["n_prompt"], r["n_out"]) for r in blocks["S1"][1]}
            out["bound"] = ST.bound_analysis(rows, meta, C.read_json(pe)["model"], C.read_json(ps)["model"], n_boot=n_boot)
    out.update(cells=cells, stability=stab, quality_pct=quality, latency=latency, blocks_complete=complete,
               outcome=ST.outcome(gate["verdict"], cells, stab["stable"], complete))
    return out


def cmd_summary(a):
    C.freeze_check(a.dev)
    s = summarise(a.edge_run, a.server_run, dev=a.dev)
    out = Path(a.out)
    C.write_json(out / "summary.json", s)
    L = [f"# {S.NAME} summary ({'DEV, not a result' if s['dev'] else (s.get('freeze') or {}).get('tag')})", "",
         f"Outcome: **{s['outcome']}**. Screen decision: {(s.get('screen_decision') or {}).get('decision')} at Stage 1.",
         f"Correctness: {(s.get('correctness') or {}).get('verdict')}. Stability ratio: {(s.get('stability') or {}).get('ratio')}.", "",
         "| Deadline (s) | U split | U edge-matched | U edge-stored | Delta (pt) | 95% CI | Selected |", "|---|---|---|---|---|---|---|"]
    for c in s["cells"]:
        u = c["utility"]
        L.append(f"| {c['deadline_s']} | {u['S']:.3f} | {u['Em']:.3f} | {u['Es']:.3f} | {c['delta_pt']:+.2f} | {c['ci95']} | {c['selected']} |")
    (out / "SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"outcome {s['outcome']} -> {out / 'summary.json'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("preflight", cmd_preflight), ("serve", cmd_serve), ("profile", cmd_profile), ("reference", cmd_reference),
                     ("correctness", cmd_correctness), ("timed", cmd_timed), ("score", cmd_score), ("summary", cmd_summary)):
        p = sp.add_parser(name)
        p.add_argument("--dev", action="store_true")
        if name in ("preflight", "profile"):
            p.add_argument("--role", required=True, choices=S.ROLES)
        if name not in ("score", "summary"):
            p.add_argument("--site", required=True)
        if name == "timed":
            p.add_argument("--block", required=True, choices=S.BLOCKS)
            p.add_argument("--limit", type=int)
            p.add_argument("--max-new", type=int)
        if name in ("score", "summary"):
            p.add_argument("--edge-run", required=True)
        if name == "score":
            p.add_argument("--scorer", choices=["exact"], help="dev only")
        if name == "summary":
            p.add_argument("--server-run")
            p.add_argument("--out", required=True)
        p.set_defaults(fn=fn)
    a = ap.parse_args(argv)
    if getattr(a, "scorer", None) and not a.dev:
        raise C.Stop("--scorer is dev only")
    if (getattr(a, "limit", None) or getattr(a, "max_new", None)) and not a.dev:
        raise C.Stop("--limit and --max-new are dev only")
    try:
        a.fn(a)
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception as e:
        record_failure(a.cmd, e)
        raise


if __name__ == "__main__":
    main()
