"""GEMMA-SPLIT Stage 0 evidence kit. Read-only helpers for COMPLETING Stage 0 of the frozen protocol
(tag gemma-split-prereg-v1, diagnostics/deployment_gemma_split). It is not part of that protocol and changes none
of its files: it lives in its own directory, imports the protocol's redaction code read-only and calls the frozen
`gs_bench.py` only on a scratch COPY of the capture directory.

Stage 0 stopped at STOP_INCOMPLETE. The verdict rules need facts that only the lab hosts and Brian hold:
`split_capture.json` (from code reading), `capabilities.json`, `candidates.json`, dated answers in `lab_answers.md`.

  collect  on a lab host, read-only: where each split-capture field is decided in the deployment's source (file:line,
           secrets redacted), the deployment's processes, listening ports, GPU placement, git state and, unless
           --no-api, the API's own schema (one GET of /openapi.json: no inference). Loads no model, starts and stops
           nothing, never reads another user's process environment, writes only into --out.
  drafts   split_capture.json / candidates.json / capabilities.json pre-filled with the facts the PI stated on
           2026-11-18, every other field still FILL, plus the lab_answers.md sections that are owed (PI statement,
           development-run log of PREREG §5.4, open questions). Drafts never validate by themselves.
  status   what Stage 0 still lacks: runs the frozen split-validate / select / stage0 on a scratch copy of the capture
           (drafts stand in for missing files only) and prints the verdict with its reasons. Exit 0 only on CONTINUE.

Standard library only. Run as `python3 -I -B gs0_kit.py ...` (-B: no bytecode may appear in the frozen directory).
"""
import sys

sys.dont_write_bytecode = True
import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import urllib.request  # noqa: E402
from pathlib import Path  # noqa: E402

KIT = "gemma-stage0-kit/1"
PROTOCOL_TAG = "gemma-split-prereg-v1"

# Facts stated by the PI on 2026-11-18 (to be cited as a lab_answers.md section, then confirmed from the captures).
PI_FACTS = {
    "model_dir_name": "gemma-4-E4B-it", "server_host": "asus-ESC4000-E11",
    "edge_source_root": "/home/brian/splitllm-edge/gemma-split",
    "server_source_root": "/mnt/mydisk/brian/splitllm-cloud/source/kv-privacy-research/experiments/distributed_cached_split_gemma",
    "edge_api": "127.0.0.1:18888", "server_backend": "127.0.0.1:18080", "edge_layers": [[0, 24]], "server_layers": [[24, 42]],
    "num_hidden_layers": 42, "link": "Brian's existing SSH tunnel from the DGX Spark to the server's loopback port",
}
CONFIG_ID = "brian-gemma4-e4b-edge0-24-server24-42"

# split_capture.json field -> patterns that show where the deployment's code decides it (evidence for the operator)
FIELD_PATTERNS = {
    "model.dtype / quantization": r"bfloat16|float16|torch_dtype|\.half\(|quantiz|load_in_\d|BitsAndBytes|gguf",
    "model.attn_implementation": r"attn_implementation|flash_attn|sdpa|eager",
    "ownership.layers / cut": r"split_(layer|point|index)|cut_layer|layer_(start|end|split)|num_(edge|server|client)_layers|layers\[[^\]]*:|range\(\s*\d+\s*,\s*\d+\s*\)",
    "ownership.parts (embedding, PLE, norm, head)": r"embed_tokens|per_layer|lm_head|final_norm|\.norm\(|get_input_embeddings|tokenizer",
    "boundary tensors": r"hidden_states|inputs_embeds|input_ids|past_key_values|logits|tobytes|frombuffer|safetensors|pickle|msgpack|torch\.save|np\.save|base64",
    "kv_cache": r"past_key_values|use_cache|DynamicCache|StaticCache|kv_cache|cache_position|prefix_cach|session",
    "transport / endpoints": r"uvicorn|fastapi|flask|aiohttp|grpc|zmq|websocket|socket\.|requests\.(post|get)|httpx|:\s*\d{4,5}\b|port\s*=|host\s*=|18080|18888",
    "generation (greedy, sampling, EOS, streaming)": r"do_sample|temperature|top_[pk]|argmax|multinomial|greedy|eos_token|ignore_eos|max_new_tokens|stream|yield |EventSource|text/event-stream",
    "capabilities (log-probs, single-host, in-process)": r"log_softmax|logprobs|output_scores|teacher|local_only|single_(host|process)|unsplit|full_model",
    "gpu placement": r"CUDA_VISIBLE_DEVICES|cuda:\d|device_map|\.to\(['\"]?cuda|torch\.device",
    "launch / environment": r"os\.environ|getenv|argparse|add_argument|ExecStart|python3? .*\.py|nohup|tmux|docker run",
    "authentication (NAMES only)": r"Authorization|Bearer|api[_-]?key|token(?!izer)|password|ssh |IdentityFile|-L \d+",
}
TEXT_SUFFIXES = {".py", ".sh", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".md", ".txt", ".service", ".env.example", ""}
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "models", "checkpoints", "outputs", "logs", "wandb", ".cache"}


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def frozen_common(gs_dir):
    """The frozen protocol's gs_common (redaction and credential scan), imported read-only."""
    gs = str(Path(gs_dir).resolve())
    if not (Path(gs) / "gs_common.py").exists():
        sys.exit(f"--gs-dir {gs_dir} is not the frozen deployment_gemma_split directory")
    if gs not in sys.path:
        sys.path.insert(0, gs)
    import gs_common
    return gs_common


def sh(cmd, timeout=30):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout if r.returncode == 0 else f"ERROR rc={r.returncode} {r.stderr.strip()[:200]}"
    except Exception as e:
        return f"ERROR {type(e).__name__}: {e}"


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n")


# --------------------------------------------------------------------------- #
# collect                                                                      #
# --------------------------------------------------------------------------- #
def scan_source(root, redact, max_hits=40, max_bytes=1 << 20):
    root = Path(root)
    pats = {k: re.compile(v, re.I) for k, v in FIELD_PATTERNS.items()}
    hits, files, total = {k: [] for k in pats}, 0, 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            try:
                size = p.stat().st_size
            except OSError:
                continue
            files += 1
            total += size
            if p.suffix.lower() not in TEXT_SUFFIXES or size > max_bytes or p.is_symlink():
                continue
            try:
                text = p.read_text(errors="replace")
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if len(line) > 400:
                    continue
                for k, pat in pats.items():
                    if len(hits[k]) < max_hits and pat.search(line):
                        hits[k].append({"file": str(p.relative_to(root)), "line": i, "text": redact(line.strip())[:300]})
    return {"n_files": files, "total_bytes": total, "evidence": hits}


def collect(a):
    C = frozen_common(a.gs_dir)
    root = Path(a.source_root)
    ports = [x for x in a.ports.split(",") if x]
    rec = {"schema": KIT, "kind": "stage0-evidence", "role": a.role, "utc": utc(), "host": socket.gethostname(), "user": os.environ.get("USER"),
           "protocol_tag": PROTOCOL_TAG, "source_root": str(root), "source_root_readable": root.is_dir() and os.access(root, os.R_OK),
           "note": "Read-only evidence for the operator who fills capture/split_capture.json and capabilities.json. Not a capture file."}
    if rec["source_root_readable"]:
        g = lambda *x: sh(["git", "--no-optional-locks", "-C", str(root), *x]).strip()      # never rewrites his index  # noqa: E731
        rec["git"] = {"head": g("rev-parse", "HEAD"), "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
                      "dirty_files": len([x for x in g("status", "--porcelain").splitlines() if x and not x.startswith("ERROR")])}
        rec["source"] = scan_source(root, C.redact_text)
    else:
        rec["source"] = {"error": "source root is not readable by this account: ask Brian for read access or for the files (Q1, Q13)"}
    keys = [str(root), "gemma", "splitllm"] + ports
    ps = sh(["ps", "-eo", "pid,user,lstart,args"])
    rec["processes"] = [C.redact_text(l.strip())[:400] for l in ps.splitlines() if any(k in l for k in keys) and "gs0_kit" not in l]
    ss = sh(["ss", "-ltn"])
    rec["listening"] = [l.strip() for l in ss.splitlines() if any(re.search(rf":{p}\b", l) for p in ports)]
    rec["gpus"] = [l.strip() for l in sh(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total", "--format=csv,noheader"]).splitlines()]
    rec["gpu_processes"] = [l.strip() for l in sh(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"]).splitlines()]
    rec["clock"] = sh(["timedatectl", "show", "-p", "NTPSynchronized"]).strip()
    if a.api_url and not a.no_api:
        url = a.api_url.rstrip("/") + "/openapi.json"
        try:
            with urllib.request.urlopen(url, timeout=5) as r:                    # metadata only: no generation request is ever sent
                spec = json.loads(r.read(4 << 20))
            rec["api_schema"] = {"url": url, "title": (spec.get("info") or {}).get("title"), "paths": {
                p: {m: {"summary": (d or {}).get("summary"), "request_fields": sorted(_request_fields(spec, d))} for m, d in ops.items() if isinstance(d, dict)}
                for p, ops in (spec.get("paths") or {}).items()}}
        except Exception as e:
            rec["api_schema"] = {"url": url, "error": f"{type(e).__name__}: {e}"}
    rec = C.redact_obj(rec)
    out = Path(a.out)
    write_json(out / f"evidence_{a.role}.json", rec)
    lines = [f"# Stage 0 evidence, {a.role} ({rec['host']}, {rec['utc']})", "", f"Source root `{root}`: readable {rec['source_root_readable']}; git {rec.get('git')}", ""]
    for k, hs in (rec["source"].get("evidence") or {}).items():
        lines += [f"## {k} ({len(hs)} lines)"] + [f"- `{h['file']}:{h['line']}` {h['text']}" for h in hs] + [""]
    lines += ["## Processes"] + [f"- {x}" for x in rec["processes"]] + ["", "## Listening ports"] + [f"- {x}" for x in rec["listening"]]
    lines += ["", "## GPUs"] + [f"- {x}" for x in rec["gpus"]] + ["", "## GPU processes"] + [f"- {x}" for x in rec["gpu_processes"]]
    lines += ["", "## API schema", "```", json.dumps(rec.get("api_schema"), indent=1), "```"]
    (out / f"evidence_{a.role}.md").write_text("\n".join(lines) + "\n")
    found = C.scan_files([out / f"evidence_{a.role}.json", out / f"evidence_{a.role}.md"], out)
    if found:                                                                    # fail closed: never leave a credential on disk
        for f in (f"evidence_{a.role}.json", f"evidence_{a.role}.md"):
            (out / f).unlink()
        sys.exit(f"credential-like content survived redaction ({found}); the evidence files were deleted. Report the pattern, not the value.")
    n = sum(len(v) for v in (rec["source"].get("evidence") or {}).values())
    print(f"[collect] {a.role}: {n} source lines, {len(rec['processes'])} processes, {len(rec['listening'])} listening ports -> {out}/evidence_{a.role}.md")


def _request_fields(spec, op):
    """Names of the JSON body fields of an OpenAPI operation (names only)."""
    try:
        sch = op["requestBody"]["content"]["application/json"]["schema"]
        if "$ref" in sch:
            for part in sch["$ref"].lstrip("#/").split("/"):
                spec = spec[part]
            sch = spec
        return list((sch.get("properties") or {}).keys())
    except Exception:
        return []


# --------------------------------------------------------------------------- #
# drafts                                                                       #
# --------------------------------------------------------------------------- #
def drafts(a):
    gs = Path(a.gs_dir)
    cap = json.loads((gs / "capture" / "split_capture.template.json").read_text())
    F = PI_FACTS
    cap["configuration_id"] = CONFIG_ID
    cap["sources"] = [{"kind": "lab_answer", "ref": "lab_answers.md#pi-statement-2026-11-18", "sha256": None},
                      {"kind": "FILL: code_reading | process | file", "ref": "FILL: the file, PID or capture that confirms each field below", "sha256": None}]
    cap["implementation"]["name"] = "distributed_cached_split_gemma"
    cap["implementation"]["notes"] = f"edge source root {F['edge_source_root']}; server source root {F['server_source_root']}; edge API on {F['edge_api']} (PI statement, to be confirmed)"
    cap["model"]["num_hidden_layers"] = F["num_hidden_layers"]
    cap["ownership"]["edge"]["layers"], cap["ownership"]["server"]["layers"] = F["edge_layers"], F["server_layers"]
    cap["transport"]["endpoint"] = F["server_backend"]
    cap["launch"]["working_dirs"] = {"edge": F["edge_source_root"], "server": F["server_source_root"]}
    cap["authentication"]["mechanism"] = "SSH tunnel from the DGX Spark to the server's loopback port; FILL: whether the API itself needs a credential (names only)"
    C = frozen_common(a.gs_dir)
    import gs_split as SP
    cap["unknowns"] = sorted(p for p in SP.placeholders(cap) if not p.startswith("$.unknowns"))     # every field still to be recorded
    cands = {"note": "Draft. One configuration is known. Add every other Gemma configuration found (Q2). No measurement may appear here (PREREG §4.5).",
             "candidates": [{"configuration_id": CONFIG_ID, "model_dir_name": F["model_dir_name"],
                             "edge_host": a.edge_host or "FILL: hostname from capture/edge/inventory.json", "server_host": F["server_host"],
                             "designated_by_implementer": False, "designation_ref": None, "running_at_capture": False, "implementation_default": False,
                             "source": "PI statement 2026-11-18 (lab_answers.md#pi-statement-2026-11-18); set running_at_capture / designated_by_implementer only with the "
                                       "process evidence or Brian's dated written answer"}]}
    caps = json.loads((gs / "capture" / "capabilities.template.json").read_text())
    out = Path(a.out)
    write_json(out / "split_capture.json", cap)
    write_json(out / "candidates.json", cands)
    write_json(out / "capabilities.json", caps)
    (out / "lab_answers.additions.md").write_text(LAB_ANSWERS.format(**F))
    bad = C.scan_files(sorted(out.glob("*")), out)
    print(f"[drafts] wrote split_capture.json ({len(cap['unknowns'])} fields still FILL), candidates.json, capabilities.json, lab_answers.additions.md -> {out}"
          + (f"\n[drafts] WARNING credential-like content: {bad}" if bad else ""))


LAB_ANSWERS = """<!-- Draft sections for capture/lab_answers.md. Review, correct, date and sign each one. Never write a credential value. -->

## PI statement (2026-11-18, Brendon) {{#pi-statement-2026-11-18}}
Stated by the PI from his own use of the deployment. Each item must be confirmed from the captures or by Brian.
- Model: `{model_dir_name}`. Edge = DGX Spark, server = `{server_host}`.
- Edge code: `{edge_source_root}`. Server code: `{server_source_root}`.
- Edge API listens on `{edge_api}`. Server backend listens on `{server_backend}`, reached through {link}.
- Split: edge layers 0-23, server layers 24-41 (42 layers).

## Development runs before the freeze (PREREG §5.4)
- 2026-11-17, PI: six sequential 32-token requests to the live edge API, to check that the deployment answers. All six
  succeeded. The response file is on the DGX Spark (`/home/brendon/gemma_api_diagnostic_20261117.json`). Latency figures
  were returned by the API and were seen by the PI. They are not recorded here, were compared with no reference, and
  are not an input of the configuration selection (PREREG §4.5): one configuration is known and Brian is asked for the
  current one, not the best one.
- FILL: any other request sent to the deployment before the freeze (date, who, purpose), or "none".

## Open questions (unanswered until Brian or the lab answers in writing, with a date)
Q1 code identity and uncommitted changes · Q2 current configuration, in writing (rule R1), and every other Gemma
configuration · Q3 start commands, environment, CUDA_VISIBLE_DEVICES · Q4 tuning history · Q5 checkpoint path on the
DGX Spark and its origin · Q6 dtype and attention implementation · Q7 owner of the embedding, PLE, final norm, LM head
and sampling · Q8 tensors on the wire and shared-KV handling · Q9 KV cache per side, prefix caching · Q10 transport and
serialisation · Q11 capabilities (a)-(g) and the stopping EOS · Q12 experiment GPUs and quiet windows · Q13 copies of
checkpoint and code · Q14 snapshot in the archive · Q15 Python environment · Q16 clock synchronisation · Q17
authentication mechanism and NAMES only · Q18 the experiment's OWN instance: permission, port
(`transport.experiment_endpoint`, not {server_backend} and not {edge_api}), account, cache directories.

Note on Q18. The frozen protocol never sends a measured request to Brian's running services. It runs its own instance
of his code from a hash-verified snapshot on another port, in windows the lab grants. The live endpoints above are
recorded as facts (`transport.endpoint`); they are not the experiment's endpoint.
"""


# --------------------------------------------------------------------------- #
# status                                                                       #
# --------------------------------------------------------------------------- #
STAGE0_FILES = ("roles.json", "edge/inventory.json", "server/inventory.json", "edge/checkpoint.json", "server/checkpoint.json",
                "edge/deployment.json", "server/deployment.json", "split_capture.json", "capabilities.json", "candidates.json", "selected_config.json")
OPERATOR_FILES = ("split_capture.json", "capabilities.json", "candidates.json")


def status(a):
    C = frozen_common(a.gs_dir)
    import gs_split as SP
    gs = Path(a.gs_dir).resolve()
    cap = Path(a.capture_dir) if a.capture_dir else gs / "capture"
    work = Path(a.work)
    prev = work / "preview_capture"
    if prev.exists():
        shutil.rmtree(prev)
    shutil.copytree(cap, prev, ignore=shutil.ignore_patterns("*.template.json", "stage0_verdict.json")) if cap.is_dir() else prev.mkdir(parents=True)
    rows, standins = [], []
    for f in STAGE0_FILES + ("lab_answers.md",):
        real = (cap / f).exists()
        if not real and a.drafts and f in OPERATOR_FILES and (Path(a.drafts) / f).exists():
            shutil.copy(Path(a.drafts) / f, prev / f)
            standins.append(f)
        n = len(SP.placeholders(json.loads((prev / f).read_text()))) if f.endswith(".json") and (prev / f).exists() else None
        rows.append((f, "present" if real else ("DRAFT stands in" if f in standins else "MISSING"), n))
    py = [sys.executable, "-I", "-B", str(gs / "gs_bench.py")]
    notes = []
    if (prev / "candidates.json").exists() and (prev / "roles.json").exists() and not (cap / "selected_config.json").exists():
        r = subprocess.run(py + ["select", "--candidates", str(prev / "candidates.json"), "--roles", str(prev / "roles.json"), "--out", str(prev / "selected_config.json")],
                           capture_output=True, text=True)
        notes.append(f"select (preview): exit {r.returncode} {(r.stdout + r.stderr).strip().splitlines()[-1][:200] if (r.stdout + r.stderr).strip() else ''}")
    r = subprocess.run(py + ["stage0", "--capture-dir", str(prev)], capture_output=True, text=True)
    vf = prev / "stage0_verdict.json"
    v = json.loads(vf.read_text()) if vf.exists() else {"verdict": "ERROR", "reasons": [(r.stdout + r.stderr).strip()[-600:]]}
    real_continue = v["verdict"] == "CONTINUE" and not standins
    res = {"schema": KIT, "kind": "stage0-status", "utc": utc(), "capture_dir": str(cap), "files": [{"file": f, "state": s, "placeholders_left": n} for f, s, n in rows],
           "drafts_standing_in": standins, "preview_verdict": v["verdict"], "reasons": v.get("reasons", []), "warnings": v.get("warnings", []), "notes": notes,
           "stage0_would_continue": real_continue,
           "meaning": "preview on a scratch copy; the real verdict is written only by the frozen `gs_bench.py stage0 --capture-dir capture`"}
    write_json(work / "stage0_status.json", C.redact_obj(res))
    print(f"Stage 0 status of {cap} (preview copy in {prev}; nothing was written into the capture directory)")
    for f, s, n in rows:
        print(f"  {s:17s} {f}" + (f"   ({n} FILL placeholders)" if n else ""))
    print(f"frozen stage0 verdict on the preview: {v['verdict']}")
    for x in v.get("reasons", [])[:40]:
        print(f"  - {C.redact_text(str(x))[:300]}")
    if standins:
        print(f"  (drafts stand in for {standins}: a draft is not evidence; Stage 0 cannot CONTINUE until the real files exist)")
    sys.exit(0 if real_continue else 3)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--role", required=True, choices=["edge", "server"]); c.add_argument("--source-root", default=None)
    c.add_argument("--gs-dir", required=True); c.add_argument("--out", required=True)
    c.add_argument("--ports", default="18888,18080"); c.add_argument("--api-url", default=None); c.add_argument("--no-api", action="store_true")
    d = sub.add_parser("drafts")
    d.add_argument("--gs-dir", required=True); d.add_argument("--out", required=True); d.add_argument("--edge-host", default=None)
    s = sub.add_parser("status")
    s.add_argument("--gs-dir", required=True); s.add_argument("--capture-dir", default=None); s.add_argument("--drafts", default=None); s.add_argument("--work", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "collect":
        if a.source_root is None:
            a.source_root = PI_FACTS["edge_source_root" if a.role == "edge" else "server_source_root"]
        if a.api_url is None and a.role == "edge":
            a.api_url = "http://" + PI_FACTS["edge_api"]
    {"collect": collect, "drafts": drafts, "status": status}[a.cmd](a)


if __name__ == "__main__":
    main()
