"""Thin pre-flight generation client for the winner-label stability reconstruction.

Sends each pre-flight query, once, to the ONE model currently served by wls_serve_vllm.sh.

Request (wls_common.build_request): the exact stored prompt as one user message, no system
prompt, temperature 0.2, top_p 1.0 -- nothing else (no max_tokens: vLLM then allows
32768 - prompt_tokens, the mission's length policy; no seed; no top_k; no extra_body).
Plain HTTP is used so that no SDK adds fields, and the repository generator and its response
cache (generators/, common/cache) are not imported at all.

Every attempt is appended to preflight_generations.jsonl. A failed request is stored with
status "error" and raw_output null; it is never turned into a model answer. A query that
already has an "ok" record for this model is never re-requested.

Usage:
  python diagnostics/scripts/wls_preflight_generate.py --model Qwen3-8B [--base-url URL]
  python diagnostics/scripts/wls_preflight_generate.py --model Qwen3-8B --dry-run   # no network
"""

import argparse
import datetime
import json
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import wls_common as C

PHASE = "preflight"


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def http(method, url, body=None, timeout=60):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode("utf-8")


def load_queries(sample_csv):
    """(dataset, index, query_sha1, prompt_sha256, prompt) with the prompt taken from the stored
    original records and checked against the frozen sample's hashes."""
    sample = C.load_sample(sample_csv)
    C.assert_sample_eligible(sample)
    stored = {}
    out = []
    for r in sample.itertuples(index=False):
        for m in C.MODELS:
            if (r.dataset, m) not in stored:
                stored[(r.dataset, m)] = C.load_stored(r.dataset, m)
        p_ds = stored[(r.dataset, C.DS)][r.index]["prompt"]
        p_qw = stored[(r.dataset, C.QWEN)][r.index]["prompt"]
        if p_ds != p_qw or C.sha256_text(p_ds) != r.prompt_sha256 or C.sha1_16(p_ds) != r.query_sha1:
            sys.exit(f"prompt reconstruction failed for {r.dataset}/{r.index}")
        out.append((r.dataset, int(r.index), r.query_sha1, r.prompt_sha256, p_ds))
    return out


def parse_metrics(text):
    """vllm:cache_config_info labels and prefix-cache counters from /metrics."""
    info = None
    m = re.search(r'^vllm:cache_config_info\{([^}]*)\}', text, re.M)
    if m:
        info = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
    counters = {}
    for name in ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total"):
        vals = re.findall(rf'^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)$', text, re.M)
        counters[name] = sum(float(v) for v in vals) if vals else None
    return info, counters


def server_checks(base, model):
    """Identity and configuration of the running server. Any failure aborts before sending."""
    rec, errors = {}, []
    st, _ = http("GET", f"{base}/health")
    rec["health_status"] = st
    _, body = http("GET", f"{base}/v1/models")
    models = json.loads(body)["data"]
    rec["v1_models"] = [{k: m.get(k) for k in ("id", "root", "max_model_len", "owned_by")} for m in models]
    if [m["id"] for m in models] != [model]:
        errors.append(f"server serves {[m['id'] for m in models]}, expected exactly [{model!r}]")
    if models and models[0].get("max_model_len") != C.MAX_MODEL_LEN:
        errors.append(f"max_model_len {models[0].get('max_model_len')} != {C.MAX_MODEL_LEN}")
    _, body = http("GET", f"{base}/version")
    rec["vllm_version"] = json.loads(body).get("version")
    if rec["vllm_version"] != C.VLLM_VERSION:
        errors.append(f"vllm version {rec['vllm_version']} != {C.VLLM_VERSION}")
    _, body = http("GET", f"{base}/metrics")
    info, counters = parse_metrics(body)
    rec["cache_config_info"], rec["prefix_cache_counters_before"] = info, counters
    if info is None or info.get("enable_prefix_caching") != "False":
        errors.append(f"prefix caching not verifiably disabled: {info and info.get('enable_prefix_caching')}")
    return rec, errors


def check_manifest(path, model):
    if not path.exists():
        return None, [f"server manifest {path} missing (written by wls_serve_vllm.sh)"]
    m = json.loads(path.read_text())
    want = {"served_model_name": model, "hf_repo": C.HF_REPO[model], "revision": C.HF_REVISION[model],
            "engine": "vllm", "vllm_version": C.VLLM_VERSION, "dtype": "bfloat16",
            "max_model_len": C.MAX_MODEL_LEN, "generation_config": C.GENERATION_CONFIG,
            "prefix_caching": False, "reasoning_parser": None}
    errors = [f"manifest {k}={m.get(k)!r}, expected {v!r}" for k, v in want.items() if m.get(k) != v]
    return m, errors


def request_record(body):
    """The request as sent, with the prompt replaced by its hash."""
    rec = {k: v for k, v in body.items() if k != "messages"}
    rec["messages"] = [{"role": x["role"], "content_sha256": C.sha256_text(x["content"])}
                       for x in body["messages"]]
    return rec


def generate_one(base, model, q, timeout, run_id, attempt):
    dataset, index, qsha1, psha256, prompt = q
    body = C.build_request(model, prompt)
    rec = {"phase": PHASE, "run_id": run_id, "attempt": attempt, "dataset": dataset, "index": index,
           "query_sha1": qsha1, "prompt_sha256": psha256, "model": model,
           "request": request_record(body), "status": "error", "raw_output": None}
    rec["request_sent_utc"] = utcnow()
    t0 = time.time()
    try:
        st, text = http("POST", f"{base}/v1/chat/completions", body, timeout=timeout)
        rec["http_status"] = st
        resp = json.loads(text)
        choices = resp.get("choices") or []
        if len(choices) != 1:
            raise ValueError(f"{len(choices)} choices")
        ch = choices[0]
        msg = ch.get("message") or {}
        usage = resp.get("usage") or {}
        rec.update({
            "response_id": resp.get("id"), "response_model": resp.get("model"),
            "system_fingerprint": resp.get("system_fingerprint"),
            "finish_reason": ch.get("finish_reason"), "stop_reason": ch.get("stop_reason"),
            "usage": usage,
            "reasoning_field_present": any(msg.get(k) for k in ("reasoning_content", "reasoning")),
        })
        if resp.get("model") != model:
            raise ValueError(f"response model {resp.get('model')!r} != {model!r}")
        if not isinstance(msg.get("content"), str):
            raise ValueError("message.content missing")
        if ch.get("finish_reason") not in ("stop", "length"):
            raise ValueError(f"finish_reason {ch.get('finish_reason')!r}")
        if not isinstance(usage.get("prompt_tokens"), int) or not isinstance(usage.get("completion_tokens"), int):
            raise ValueError("usage token counts missing")
        rec["raw_output"] = msg["content"]
        rec["status"] = "ok"
    except urllib.error.HTTPError as e:
        rec["http_status"] = e.code
        rec["error"] = f"HTTPError {e.code}: {e.read().decode('utf-8', 'replace')[:2000]}"
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["raw_output"] = None
    rec["response_received_utc"] = utcnow()
    rec["latency_s"] = round(time.time() - t0, 3)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=C.MODELS)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--sample", type=Path, default=C.PREFLIGHT_SAMPLE_CSV)
    ap.add_argument("--out", type=Path, default=C.PREFLIGHT_GENERATIONS)
    ap.add_argument("--server-manifest", type=Path, default=None)
    ap.add_argument("--concurrency", type=int, default=32)   # original collector: 32
    ap.add_argument("--timeout", type=float, default=7200.0)  # client-side wait only
    ap.add_argument("--dry-run", action="store_true", help="build requests; send nothing")
    args = ap.parse_args()

    if args.out.resolve().parent != C.PREFLIGHT_DIR.resolve():
        sys.exit(f"pre-flight outputs must stay in {C.PREFLIGHT_DIR}")
    queries = load_queries(args.sample)

    if args.dry_run:
        for q in queries:
            print(json.dumps({"dataset": q[0], "index": q[1], "model": args.model,
                              "request": request_record(C.build_request(args.model, q[4]))}))
        print(f"DRY RUN: {len(queries)} requests built for {args.model}; nothing sent", file=sys.stderr)
        return

    manifest_path = args.server_manifest or (C.PREFLIGHT_DIR / f"server_{args.model}.json")
    manifest, errors = check_manifest(manifest_path, args.model)
    base = args.base_url.rstrip("/")
    server, server_errors = server_checks(base, args.model)
    errors += server_errors
    if errors:
        sys.exit("server checks failed; nothing sent:\n  " + "\n  ".join(errors))

    done, attempts = set(), {}
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["model"] != args.model:
                continue
            k = (r["dataset"], r["index"])
            attempts[k] = attempts.get(k, 0) + 1
            if r["status"] == "ok":
                done.add(k)
    todo = [q for q in queries if (q[0], q[1]) not in done]
    run_id = f"{PHASE}-{datetime.datetime.now(datetime.timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    print(f"{args.model}: {len(queries)} queries, {len(done)} already ok, sending {len(todo)}; run {run_id}")

    run = {"run_id": run_id, "phase": PHASE, "model": args.model, "started_utc": utcnow(),
           "base_url": base, "client": "diagnostics/scripts/wls_preflight_generate.py",
           "repo": C.git_head(), "server_manifest": manifest, "server": server,
           "repo_response_cache": "not used (client does not import generators/ or common/cache)",
           "request_seed": None, "max_tokens": None, "concurrency": args.concurrency,
           "client_timeout_s": args.timeout, "n_requests": len(todo)}
    n_ok = n_err = 0
    with args.out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(args.concurrency) as pool:
        futs = [pool.submit(generate_one, base, args.model, q, args.timeout, run_id,
                            attempts.get((q[0], q[1]), 0) + 1) for q in todo]
        for fut in as_completed(futs):
            rec = fut.result()
            rec["served_model_name"] = server["v1_models"][0]["id"]
            rec["vllm_version"] = server["vllm_version"]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n_ok += rec["status"] == "ok"
            n_err += rec["status"] != "ok"
            print(f"  {rec['status']:5} {rec['dataset']}/{rec['index']} "
                  f"{rec.get('finish_reason')} {rec.get('usage', {}).get('completion_tokens')}", flush=True)

    _, body = http("GET", f"{base}/metrics")
    run["prefix_cache_counters_after"] = parse_metrics(body)[1]
    run.update(finished_utc=utcnow(), n_ok=n_ok, n_error=n_err)
    with (C.PREFLIGHT_DIR / "preflight_runs.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(run) + "\n")
    print(f"done: {n_ok} ok, {n_err} error (errors are recorded, never scored)")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
