"""CPU tests of the Stage 0 evidence kit (standard library only). Fakes stand in for Brian's tree and for lab facts;
the frozen protocol's own validators decide every verdict here.

  python3 -I -B tests/test_gs0_kit.py
"""
import sys

sys.dont_write_bytecode = True
import hashlib  # noqa: E402
import http.server  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

KITDIR = Path(__file__).resolve().parents[1]
REPO = KITDIR.parents[1]
GS = KITDIR.parent / "deployment_gemma_split"
sys.path.insert(0, str(GS / "tests"))
import gs_testlib as T  # noqa: E402  (the frozen protocol's fixtures; puts the frozen directory on sys.path)
import gs_split as SP  # noqa: E402

KIT = [sys.executable, "-I", "-B", str(KITDIR / "gs0_kit.py")]
TMP = T.tmpdir("gs0kit_")
SECRETS = ("sk-live-abcdef1234567890abcdef", "abc123secretvalue9999zz")


def run(*args):
    return subprocess.run(KIT + [str(x) for x in args], capture_output=True, text=True)


def tree_digest(d):
    return hashlib.sha256("\n".join(f"{p.relative_to(d)}:{hashlib.sha256(p.read_bytes()).hexdigest()}" for p in sorted(Path(d).rglob("*")) if p.is_file()).encode()).hexdigest()


def test_frozen_protocol_untouched():
    r = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", "diagnostics/deployment_gemma_split"], cwd=REPO, capture_output=True, text=True)
    assert r.stdout.strip() == "", r.stdout                                        # no edit, no stray file, no bytecode
    tag = subprocess.run(["git", "rev-parse", "--verify", "--quiet", "gemma-split-prereg-v1^{commit}"], cwd=REPO, capture_output=True, text=True)
    if tag.returncode != 0:
        return "SKIP diff: tag not present in this clone"
    assert subprocess.run(["git", "diff", "--quiet", tag.stdout.strip(), "--", "diagnostics/deployment_gemma_split"], cwd=REPO).returncode == 0


def fake_source():
    src = TMP / "brian" / "gemma-split"
    (src / "models").mkdir(parents=True)
    (src / "server.py").write_text(
        'import torch\nmodel = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)\nSPLIT_LAYER = 24\nsplit_layer = SPLIT_LAYER\n'
        f'API_KEY = "{SECRETS[0]}"\nheaders = {{"Authorization": "Bearer {SECRETS[1]}"}}\n'
        'out = model(inputs_embeds=h, past_key_values=kv, use_cache=True)\nnext_id = logits.argmax(-1)  # greedy, do_sample=False\n'
        'uvicorn.run(app, host="127.0.0.1", port=18080)\n')
    (src / "run_edge.sh").write_text("#!/bin/sh\nCUDA_VISIBLE_DEVICES=0 python3 edge.py --port 18888\n")
    (src / "models" / "big.py").write_text("torch_dtype = 'float16'\n")        # skipped directory
    return src


def test_collect_is_read_only_and_redacts():
    src, out = fake_source(), TMP / "ev"
    before = tree_digest(src)
    spec = {"info": {"title": "gemma split edge"}, "paths": {"/generate": {"post": {"summary": "generate", "requestBody": {"content": {"application/json": {
        "schema": {"$ref": "#/components/schemas/Req"}}}}}}}, "components": {"schemas": {"Req": {"properties": {"prompt": {}, "max_new_tokens": {}, "temperature": {}, "stream": {}}}}}}

    class H(http.server.BaseHTTPRequestHandler):
        seen = []

        def do_GET(self):
            H.seen.append(self.path)
            body = json.dumps(spec).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)

        def do_POST(self):
            H.seen.append("POST " + self.path)
            self.send_response(500); self.end_headers()

        def log_message(self, *a):
            pass
    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        r = run("collect", "--role", "edge", "--source-root", src, "--gs-dir", GS, "--out", out, "--api-url", f"http://127.0.0.1:{srv.server_port}")
    finally:
        srv.shutdown()
    assert r.returncode == 0, r.stdout + r.stderr
    ev, md = json.loads((out / "evidence_edge.json").read_text()), (out / "evidence_edge.md").read_text()
    e = ev["source"]["evidence"]
    assert any("bfloat16" in h["text"] for h in e["model.dtype / quantization"]) and any("18080" in h["text"] for h in e["transport / endpoints"])
    assert any(h["file"] == "server.py" and "SPLIT_LAYER" in h["text"] for h in e["ownership.layers / cut"])
    assert not any(h["file"].startswith("models/") for hs in e.values() for h in hs)
    for s in SECRETS:                                                           # no credential value in any output
        assert s not in json.dumps(ev) and s not in md and s not in r.stdout
    assert any(h["file"] == "server.py" for h in e["authentication (NAMES only)"])         # the NAME is still evidence
    assert H.seen == ["/openapi.json"]                                           # one metadata GET; no generation request
    assert ev["api_schema"]["paths"]["/generate"]["post"]["request_fields"] == ["max_new_tokens", "prompt", "stream", "temperature"]
    assert tree_digest(src) == before and not list(src.rglob("__pycache__"))
    r = run("collect", "--role", "server", "--source-root", TMP / "no_such_dir", "--gs-dir", GS, "--out", out, "--no-api")
    assert r.returncode == 0 and "not readable" in json.loads((out / "evidence_server.json").read_text())["source"]["error"]


def test_drafts_record_stated_facts_and_never_validate():
    out = TMP / "drafts"
    r = run("drafts", "--gs-dir", GS, "--out", out, "--edge-host", "spark-6799")
    assert r.returncode == 0 and "WARNING" not in r.stdout, r.stdout + r.stderr
    cap, cands, caps = (json.loads((out / f).read_text()) for f in ("split_capture.json", "candidates.json", "capabilities.json"))
    assert cap["ownership"]["edge"]["layers"] == [[0, 24]] and cap["ownership"]["server"]["layers"] == [[24, 42]] and cap["model"]["num_hidden_layers"] == 42
    assert cap["transport"]["endpoint"] == "127.0.0.1:18080" and str(cap["transport"]["experiment_endpoint"]).startswith("FILL")
    assert len(SP.placeholders(cap)) > 20 and len(cap["unknowns"]) > 20 and SP.placeholders(caps)
    c = cands["candidates"][0]
    assert c["model_dir_name"] == "gemma-4-E4B-it" and c["server_host"] == "asus-ESC4000-E11" and c["edge_host"] == "spark-6799"
    assert c["designated_by_implementer"] is False and c["designation_ref"] is None            # no dated answer from Brian exists yet
    assert not any(k in json.dumps(cands).lower() for k in ("latency", "ttft", "tokens/s", "ms"))  # no measurement near the selection
    la = (out / "lab_answers.additions.md").read_text()
    assert "PREREG §5.4" in la and "2026-11-17" in la and "3.262" not in la and "0.297" not in la and "Q18" in la
    problems, _ = SP.validate(cap, {"num_hidden_layers": 42})
    assert problems, "a draft must not validate"


def test_status_previews_the_frozen_verdict_without_touching_the_capture():
    gs = T.copy_gs(TMP / "repo")
    cap = T.write_target_capture(gs, TMP / "models", fake_source() if not (TMP / "brian").exists() else TMP / "brian" / "gemma-split", adapter=False)
    (cap / "stage0_verdict.json").unlink()
    before = tree_digest(cap)
    r = run("status", "--gs-dir", GS, "--capture-dir", cap, "--work", TMP / "work1")
    st = json.loads((TMP / "work1" / "stage0_status.json").read_text())
    assert r.returncode == 0 and st["preview_verdict"] == "CONTINUE" and st["stage0_would_continue"] and tree_digest(cap) == before, r.stdout + r.stderr
    # the operator files are missing: drafts stand in, the frozen rules say what is still owed, and it can never CONTINUE
    for f in ("split_capture.json", "capabilities.json", "candidates.json", "selected_config.json"):
        (cap / f).unlink()
    before = tree_digest(cap)
    run("drafts", "--gs-dir", GS, "--out", TMP / "drafts2", "--edge-host", T.EDGE_HOST)
    r = run("status", "--gs-dir", GS, "--capture-dir", cap, "--drafts", TMP / "drafts2", "--work", TMP / "work2")
    st = json.loads((TMP / "work2" / "stage0_status.json").read_text())
    assert r.returncode == 3 and st["preview_verdict"].startswith("STOP_") and not st["stage0_would_continue"], r.stdout + r.stderr
    assert set(st["drafts_standing_in"]) == {"split_capture.json", "capabilities.json", "candidates.json"} and st["reasons"]
    assert any("placeholder" in x or "FILL" in x or "unknown" in x for x in st["reasons"]), st["reasons"]
    assert tree_digest(cap) == before and not (cap / "stage0_verdict.json").exists()
    # without drafts the frozen verdict names the missing files
    r = run("status", "--gs-dir", GS, "--capture-dir", cap, "--work", TMP / "work3")
    st = json.loads((TMP / "work3" / "stage0_status.json").read_text())
    assert r.returncode == 3 and st["preview_verdict"] == "STOP_INCOMPLETE" and "missing capture files" in st["reasons"][0]


def main():
    failed = 0
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        t0 = time.time()
        try:
            note = fn()
            print(f"PASS {name} ({time.time() - t0:.1f} s){' ' + note if note else ''}", flush=True)
        except Exception as e:
            failed += 1
            import traceback
            print(f"FAIL {name}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    shutil.rmtree(TMP, ignore_errors=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
