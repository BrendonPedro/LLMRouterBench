"""QWEN-SELECT CPU tests. No GPU, no network beyond loopback, no real checkpoint, no real prompt.

  <python with torch + transformers 4.51.3> -I -B tests/test_qsel.py            # all tests
  ... -I -B tests/test_qsel.py --quick                                         # skip the multi-process runs
  ... -I -B tests/test_qsel.py -k name                                         # tests whose name contains `name`

Models are random Qwen3 checkpoints with the real layer counts (64, 40, 36) and a tiny vocabulary. Mocks cover
inventories only; every number compared here is computed by the code under test. Nothing here is a result.
"""
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qsel_testlib as T  # noqa: E402
from qsel_testlib import HERE, TMP  # noqa: E402

import qsel_common as C  # noqa: E402
import qsel_models as M  # noqa: E402
import qsel_runtime as RT  # noqa: E402
import qsel_spec as S  # noqa: E402
import qsel_stats as ST  # noqa: E402
import qsel_workload as WL  # noqa: E402

REPO = HERE.parents[1]
GIB = 1 << 30


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError(f"{exc.__name__} not raised")


# --------------------------------------------------------------------------- #
# The frozen design                                                            #
# --------------------------------------------------------------------------- #
def test_spec_is_complete_and_consistent():
    text = (HERE / "qsel_spec.py").read_text()
    assert "PENDING" not in text and "0000000000" not in text
    assert set(S.CONFIGS) == {"S", "E32", "E14", "E8", "R14"} and S.CONFIGS["S"] == {"execution": "SPLIT", "model": "32B"}
    assert S.POLICIES == {"no-raw-text": ("E32", "E14", "E8"), "open": ("E32", "E14", "E8", "R14")} and S.PRIMARY_POLICY == "no-raw-text"
    assert not any(S.CONFIGS[c]["execution"] == "REMOTE" for c in S.POLICIES["no-raw-text"])       # REMOTE is excluded by the data rule
    assert S.BLOCKS == ("S1a", "E32a", "E14a", "E8a", "S1b", "S2", "E32b", "E14b", "E8b", "R14") and S.PRIMARY_BLOCKS == S.BLOCKS[:9]
    assert S.FIRST_HALF_BLOCKS == ("S1a", "E32a", "E14a", "E8a") and set(S.BLOCK_CONFIG) == set(S.BLOCKS) == set(S.BLOCK_QUERIES)
    assert all(S.BLOCK_CONFIG[b] == c for c, bs in S.CONFIG_BLOCKS.items() for b in bs) and S.BLOCK_QUERIES["S2"] == "sentinel"
    assert sorted(b for bs in S.CONFIG_BLOCKS.values() for b in bs) == sorted(b for b in S.BLOCKS if b != "S2")
    assert set(S.EXPECTED_H["edge"]) == {"reference", "correctness"} | set(S.PRIMARY_BLOCKS)
    assert S.DEADLINES_S == (30, 60, 120, 300) and S.QUERY_CUTOFF_S == max(S.DEADLINES_S) and S.MARGIN_PT == 3.0
    assert abs(sum(S.ALLOTMENT_H.values()) - S.GPU_HOURS_CEILING) < 1e-9 and S.GPU_HOURS_CEILING == 48.0
    assert sum(S.EXPECTED_H["edge"].values()) < S.ALLOTMENT_H["edge"] and sum(S.EXPECTED_H["server"].values()) < S.ALLOTMENT_H["server"]
    assert S.MAX_NEW_TOKENS < S.CONTEXT_TOKENS and sorted(S.TIE_ORDER) == sorted(S.POLICIES["open"])
    assert S.THINKING is False and S.DECODING == "greedy" and len(C.spec_hash()) == 64


def test_cut_and_criterion_are_k16s():
    """The cut, the runtime mode and the correctness rule are K16-HWFIT's, unchanged."""
    K, _ = RT.k16()
    assert (K.K, K.J, K.N_LAYERS) == (S.CUT, S.CUT, S.MODELS["32B"]["n_layers"]) and K.PRIMARY_MODE == S.SPLIT_MODE
    assert K.CRITERIA["id"] == S.CRITERIA_ID
    import hashlib
    assert hashlib.sha256(K.canonical(K.CRITERIA).encode()).hexdigest() == S.CRITERIA_SHA256
    assert K.MAX_LEN == S.CONTEXT_TOKENS and K.PREFILL_CHUNK == S.PREFILL_CHUNK and K.TRANSFORMERS_REQUIRED == S.TRANSFORMERS_REQUIRED
    assert f"{K.E0.SPLIT_REPO}@{K.E0.SPLIT_REVISION}" == f"{S.MODELS['32B']['repo']}@{S.MODELS['32B']['revision']}"
    assert tuple(sorted(K.E0.EOS_IDS)) == tuple(sorted(S.EOS_IDS))
    assert RT.params()["max_len"] == K.MAX_LEN and RT.params()["model_id"] == K.MODEL_ID


def test_reused_code_is_unmodified():
    """Nothing outside the experiment's own directory differs from the base commit."""
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True)  # noqa: E731
    if git("rev-parse", "--verify", "--quiet", f"{S.BASE_COMMIT}^{{commit}}").returncode != 0:
        print("    (base commit not present in this clone: skipped)")
        return
    r = git("diff", "--name-only", S.BASE_COMMIT, "--", ".")
    assert r.returncode == 0
    changed = [f for f in r.stdout.split() if not f.startswith("diagnostics/deployment_qwen_select/")]
    assert not changed, changed
    assert git("diff", "--quiet", S.BASE_COMMIT, "--", *S.REUSED_DIRS).returncode == 0


def test_public_files_carry_no_site_detail():
    """No host name, address, account, device id, private path or prompt text in anything that is published."""
    pats = [r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", r"GPU-[0-9a-f]{8}", r"/home/[a-z]", r"/mnt/", r"[a-z]+@[a-z0-9.-]+\.[a-z]{2,}"]
    allow = ("127.0.0.1", "noreply@anthropic.com")
    bad = []
    for f in sorted(HERE.rglob("*")):
        if not f.is_file() or "__pycache__" in f.parts or f.suffix in (".pyc",) or f.name == "test_qsel.py":
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        for a in allow:
            text = text.replace(a, "")
        bad += [(f.name, p, m.group(0)) for p in pats for m in [re.search(p, text)] if m]
    assert not bad, bad
    w = C.read_json(HERE / S.WORKLOAD_FILE)
    assert all(set(q) == {"dataset", "prompt_sha256", "answer_sha256", "n_prompt", "ids_sha256"} for q in w["queries"].values())


# --------------------------------------------------------------------------- #
# Workload                                                                     #
# --------------------------------------------------------------------------- #
def test_workload_is_the_pinned_heldout_set():
    w = WL.public()
    assert (len(w["utility"]), len(w["sentinel"]), len(w["correctness"])) == (160, 24, 10)
    assert sorted(w["utility"]) == sorted(w["queries"]) and set(w["sentinel"]) <= set(w["utility"]) and set(w["correctness"]) <= set(w["utility"])
    per = {b: sum(q["dataset"] == b for q in w["queries"].values()) for b in S.BENCHMARKS}
    assert per == S.HELDOUT_PER_BENCH and {b: sum(w["queries"][k]["dataset"] == b for k in w["sentinel"]) for b in S.BENCHMARKS} == S.SENTINEL_PER_BENCH
    first = {json.loads(l)["key"] for l in (REPO / S.S1_SAMPLE_REL).read_text(encoding="utf-8").splitlines() if l.strip()}
    assert C.sha256_file(REPO / S.S1_SAMPLE_REL) == S.S1_SAMPLE_SHA256 and not (set(w["queries"]) & first)       # held out from every earlier run
    ext = C.read_json(REPO / S.EXTENSION_KEYS_REL)
    assert ext["keys_sha256"] == S.EXTENSION_KEYS_SHA256 and set(w["queries"]) <= set(ext["keys"]) and first <= set(ext["keys"])
    n = [w["queries"][k]["n_prompt"] for k in w["correctness"]]
    allp = sorted(q["n_prompt"] for q in w["queries"].values())
    assert n[:5] == allp[:5] and n[5:] == allp[-5:] and max(allp) + S.N_CONT <= S.CONTEXT_TOKENS
    assert all(WL.max_new(x) >= 1024 for x in allp) and WL.max_new(100) == S.MAX_NEW_TOKENS and WL.max_new(4000) == 96
    assert max(allp) > S.PREFILL_CHUNK                                          # the correctness set exercises a two-chunk prefill
    priv = os.environ.get("QSEL_PRIVATE_WORKLOAD")
    if priv:                                                                    # maintainer only: the private file and the real tokenizer
        wl = WL.load(priv)
        assert len(wl["utility"]) == 160 and C.sha256_file(priv) == S.WORKLOAD_PRIVATE_SHA256
        if os.environ.get("QSEL_TOKENIZER_DIR"):
            tok = WL.load_tokenizer(os.environ["QSEL_TOKENIZER_DIR"])
            assert all(len(WL.token_ids(tok, q)) == q["n_prompt"] for q in wl["utility"])
    else:
        print("    (private workload not given: its pin and the tokenisation pin were not exercised)")


def test_workload_load_refuses_changed_content():
    pub, priv = T.dev_workload()
    w = C.read_json(pub)
    wl = WL.load(priv, dev_workload=w)
    assert [q["key"] for q in wl["utility"]] == w["utility"] and all(q["prompt"] and q["answer"] for q in wl["utility"])
    rows = C.read_jsonl(priv)
    rows[0]["prompt"] += " w1"
    bad = TMP / "bad_private.jsonl"
    bad.write_text("".join(json.dumps(r) + "\n" for r in rows))
    raises(SystemExit, WL.load, str(bad), dev_workload=w)
    bad.write_text("".join(json.dumps(r) + "\n" for r in rows[1:]))
    raises(SystemExit, WL.load, str(bad), dev_workload=w)
    raises(SystemExit, WL.load, priv)                                           # a dev file is not the pinned private file
    tok = WL.load_tokenizer(T.tiny_models()["32B"])
    q = dict(wl["utility"][0])
    assert WL.token_ids(tok, q) == WL.tokenize(tok, q["prompt"])
    q["ids_sha256"] = "0" * 64
    raises(SystemExit, WL.token_ids, tok, q)


# --------------------------------------------------------------------------- #
# Checkpoints and memory                                                       #
# --------------------------------------------------------------------------- #
def test_manifests_and_memory_needs():
    K, _ = RT.k16()
    for name, m in S.MODELS.items():
        man, table = M.manifest(name)
        assert (man["repo"], man["revision"]) == (m["repo"], m["revision"])
        assert man["total_tensor_bytes"] == sum(t["nbytes"] for t in table.values()) and man["n_tensors"] == len(table)
        assert all(f in man["files"] for f in S.CHECKPOINT_SMALL_FILES)
        assert man["files"]["tokenizer.json"]["sha256"] == S.TOKENIZER_JSON_SHA256
        cfg = M.config_of(name, table)
        assert cfg["num_hidden_layers"] == m["n_layers"] and max(int(n.split(".")[2]) for n in table if n.startswith("model.layers.")) == m["n_layers"] - 1
        assert all(t["dtype"] == "BF16" for t in table.values())
        assert len(M.needed_files(man, table, "edge")) == len(man["files"]) and "tokenizer.json" not in M.needed_files(man, table, "server_whole")
    man, table = M.manifest("32B")
    assert M.config_of("32B", table) == {k: K.E0.QWEN3_32B_CONFIG[k] for k in M.config_of("32B", table)}
    srv = M.needed_files(man, table, "split_server")
    assert set(srv) < set(M.needed_files(man, table, "server_whole")) and "model-00001-of-00017.safetensors" not in srv
    n = M.needs()
    import k16_memory as KM
    fw = KM.frozen_weights()
    assert n["split_edge"]["weights"] == fw["edge"] and n["split_server"]["weights"] == fw["server"] and n["edge_whole_32B"]["weights"] == fw["all"]
    a6000 = 48 * GIB
    assert n["split_server"]["required_free"] < a6000 and n["server_whole_14B"]["required_free"] < a6000
    assert n["server_whole_32B"]["weights"] > a6000                             # REMOTE with the 32B model cannot fit one server GPU
    assert n["edge_whole_8B"]["required_free"] < n["edge_whole_14B"]["required_free"] < n["edge_whole_32B"]["required_free"] < 76 * GIB
    # the split halves and the reference are K16-HWFIT's own gates, which held on both machines on 18 November
    assert n["split_edge"]["required_free"] == KM.expected("edge_twohost", static=False)["required_free"]
    assert n["split_server"]["required_free"] == KM.expected("server", static=False)["required_free"]
    assert n["edge_reference"]["required_free"] == KM.expected("edge_reference")["required_free"]
    import e0_memory as M0
    cfg = M.config_of("32B", table)
    assert all(M.step_transient_bytes(cfg, a_, b_) == M0.transient_bytes(K.E0.QWEN3_32B_CONFIG, a_, b_) for a_, b_ in ((1, 4096), (2048, 2048), (2048, 4096), (2887, 2887)))
    ref = M.needs(reference_seq=2823 + 64)["edge_reference"]["required_free"]
    assert ref == KM.expected("edge_reference", longest_seq=2887)["required_free"] and n["edge_reference"]["required_free"] < ref < 76 * GIB
    assert set(M.LOAD_OF.values()) <= set(n) and "| split_edge |" in M.memory_markdown()
    bad = dict(S.MANIFEST_SHA256)
    try:
        S.MANIFEST_SHA256 = dict(bad, **{"8B": ("0" * 64, bad["8B"][1])})
        raises(ValueError, M.manifest, "8B")
    finally:
        S.MANIFEST_SHA256 = bad


def test_checkpoint_verify():
    """verify() against a manifest built for a tiny checkpoint: accepts it, refuses a changed byte, a missing file,
    another header."""
    import shutil
    src = Path(T.tiny_models()["8B"])
    d = TMP / "verify8"
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(src, d)
    (d / "model.safetensors.index.json").write_text("{}")
    hl, hdr = M.read_safetensors_header(d / "model.safetensors")
    table = {n: {"shard": "model.safetensors", "dtype": t["dtype"], "shape": t["shape"], "offsets": t["data_offsets"], "nbytes": 0} for n, t in hdr.items()}
    files = {f: {"size": (d / f).stat().st_size, "sha256": C.sha256_file(d / f)} for f in S.CHECKPOINT_SMALL_FILES}
    files["model.safetensors"] = {"size": (d / "model.safetensors").stat().st_size, "sha256": C.sha256_file(d / "model.safetensors"), "header_len": hl}
    man = {"repo": "tiny", "revision": "r", "files": files}
    cache = TMP / "verify_cache.json"
    rec = M.verify(d, "8B", cache_file=cache, man=man, table=table)
    assert rec["ok"] and rec["bytes_verified"] == sum(f["size"] for f in files.values()) and cache.exists()
    assert M.verify(d, "8B", "server_whole", man=man, table=table)["ok"]
    with open(d / "model.safetensors", "r+b") as f:                             # one weight byte changed, size unchanged
        f.seek(-1, 2)
        b = f.read(1)
        f.seek(-1, 2)
        f.write(bytes([b[0] ^ 1]))
    rec = M.verify(d, "8B", cache_file=cache, man=man, table=table)
    assert not rec["ok"] and any("sha256" in p for p in rec["problems"])
    (d / "tokenizer.json").unlink()
    rec = M.verify(d, "8B", man=man, table=table)
    assert any("missing: tokenizer.json" in p for p in rec["problems"])
    assert not any("tokenizer" in p for p in M.verify(d, "8B", "server_whole", man=man, table=table)["problems"])
    t2 = dict(table)
    t2.pop(next(iter(t2)))
    assert any("header" in p for p in M.verify(src, "8B", man=man, table=t2, deep=False)["problems"])
    assert M.eos_ids(src) == T.EOS and M.model_config(src)["num_hidden_layers"] == 36


# --------------------------------------------------------------------------- #
# Site files, GPU selection, gates, ledger                                     #
# --------------------------------------------------------------------------- #
INV = ("GPU-aaaaaaaa-0000-0000-0000-000000000001, NVIDIA RTX A6000, 49140, 34065, 14606, 0\n"
       "GPU-bbbbbbbb-0000-0000-0000-000000000002, NVIDIA RTX A6000, 49140, 1000, 47800, 1\n"
       "GPU-cccccccc-0000-0000-0000-000000000003, NVIDIA RTX A6000, 49140, 6786, 41884, 2\n"
       "GPU-dddddddd-0000-0000-0000-000000000004, NVIDIA GB10, [N/A], [N/A], [N/A], 3\n")
APPS = ("GPU-aaaaaaaa-0000-0000-0000-000000000001, 11, 34000\nGPU-cccccccc-0000-0000-0000-000000000003, 12, 3000\n"
        "GPU-cccccccc-0000-0000-0000-000000000003, 13, 3700\n")


def test_gpu_selection_rule():
    rows = C.parse_gpu_rows(INV)
    assert [r["index"] for r in rows] == [0, 1, 2, 3] and rows[3]["free_mib"] is None
    need = 34 * GIB
    chosen, cons = C.select_gpu(rows, APPS, need, excluded=["GPU-bbbbbbbb-0000-0000-0000-000000000002"])
    assert chosen["index"] == 2 and [c["reason"] for c in cons][1] == "excluded by the site file"       # never the excluded GPU, however free
    assert "free 14606 MiB" in cons[0]["reason"] and cons[3]["reason"] == "not the server GPU model"
    assert C.select_gpu(rows, APPS, need)[0]["index"] == 1                       # fewest other processes first
    assert C.select_gpu(rows, APPS, 60 * GIB)[0] is None
    assert C.select_gpu(rows, None, need)[0] is None                            # other processes could not be counted: nothing is eligible
    assert C.parse_apps(APPS, "GPU-cccccccc-0000-0000-0000-000000000003") == {"other_processes": 2, "other_mib": 6700}
    assert C.parse_apps(APPS, "cccccccc-0000-0000-0000-000000000003", own_pids=(12,)) == {"other_processes": 1, "other_mib": 3700}


def test_memory_gates():
    mem = lambda avail, free=0: {"MemAvailable": avail, "MemFree": free, "swap_used": 0}  # noqa: E731
    edge = {"role": "edge", "device": "cuda", "gpu_uuid": "GPU-dddddddd-0000-0000-0000-000000000004"}
    assert C.memory_gate("edge", edge, 70 * GIB, mem=mem(75 * GIB))["pass"]     # unified memory: available RAM decides
    raises(SystemExit, C.memory_gate, "edge", edge, 70 * GIB, mem=mem(60 * GIB))
    rows = C.parse_gpu_rows(INV)
    srv = {"role": "server", "device": "cuda", "gpu_uuid": "GPU-cccccccc-0000-0000-0000-000000000003"}
    assert C.memory_gate("server", srv, 34 * GIB, rows=rows, mem=mem(100 * GIB))["gpu_free_mib"] == 41884
    raises(SystemExit, C.memory_gate, "server", srv, 42 * GIB, rows=rows, mem=mem(100 * GIB))
    raises(SystemExit, C.memory_gate, "server", srv, 34 * GIB, rows=rows, mem=mem(1 * GIB))       # host RAM for staging
    raises(SystemExit, C.memory_gate, "server", dict(srv, gpu_uuid="GPU-ffffffff"), 1, rows=rows, mem=mem(100 * GIB))
    # in-process: the edge's device figure excludes page cache (860 MiB reported with 74,911 MiB available on 18 November), so it is not applied
    g = C.device_gate("edge", 70 * GIB, 860 << 20, mem=mem(74911 << 20, 900 << 20))
    assert g["ram_figure_applied"] and not g["device_figure_applied"]
    raises(SystemExit, C.device_gate, "edge", 76 * GIB, 100 * GIB, mem=mem(74911 << 20))
    assert C.device_gate("server", 34 * GIB, 40 * GIB, mem=mem(1))["device_figure_applied"]
    raises(SystemExit, C.device_gate, "server", 34 * GIB, 30 * GIB, mem=mem(500 * GIB))


def test_site_file_rules():
    base = json.loads(Path(T.site("server", TMP / "site_rules", (T.free_port(), T.free_port()), name="site_rules.json")).read_text())

    def load(role="server", selecting=False, **change):
        p = TMP / "site_rules_x.json"
        p.write_text(json.dumps(dict(base, **change)))
        return C.load_site(p, role, selecting=selecting)
    assert load()["_sha256"] and load(selecting=True)["gpu_uuid"] is None
    raises(SystemExit, load, role="edge")
    raises(SystemExit, load, selecting=True, gpu_uuid="GPU-x")                  # select-gpu takes an unpinned template
    raises(SystemExit, load, gpu_uuid="GPU-AAAA", excluded_gpu_uuids=["gpu-aaaa"])
    raises(SystemExit, load, remote_port=base["split_port"])
    raises(SystemExit, load, ports_in_use_by_others=[base["remote_port"]])
    raises(SystemExit, load, model_dirs={"32B": "x"})
    d = dict(base)
    d.pop("token_file")
    p = TMP / "site_rules_y.json"
    p.write_text(json.dumps(d))
    raises(SystemExit, C.load_site, p, "server")
    tf = TMP / "tok_rules" / "token"
    s = dict(base, token_file=str(tf))
    raises(SystemExit, C.check_token, s)
    tok = C.check_token(s, create=True)
    assert len(tok) == 64 and (tf.stat().st_mode & 0o777) == 0o600 and C.check_token(s) == tok
    os.chmod(tf, 0o644)
    raises(SystemExit, C.check_token, s)
    assert C.local_addr(base, "split_port") == f"127.0.0.1:{base['split_port']}"


def test_ledger_accounts_and_enforces_the_allotment():
    rd = TMP / "ledger"
    rd.mkdir(exist_ok=True)
    led = C.Ledger(rd, "edge", allotment_h=1.0)
    eid, limit = led.open("reference", 600)
    assert limit == 600 and (rd / f"heartbeat_{eid}").exists()
    time.sleep(0.05)
    led.close("done")
    e = C.Ledger(rd, "edge", allotment_h=1.0).entries()
    assert len(e) == 1 and "close" in e[0] and 0 < led.charged(e[0]) < 5
    # an attempt that was killed: charged to its last heartbeat plus the grace period, never less
    now = time.time()
    C.append_jsonl(rd / C.LEDGER, {"id": "timed-S1.1.1", "stage": "timed-S1", "event": "open", "t": now - 2000, "utc": "x", "limit_s": 9000})
    (rd / "heartbeat_timed-S1.1.1").touch()
    os.utime(rd / "heartbeat_timed-S1.1.1", (now - 1000, now - 1000))
    led2 = C.Ledger(rd, "edge", allotment_h=1.0)
    killed = next(x for x in led2.entries() if x["id"] == "timed-S1.1.1")
    assert abs(led2.charged(killed, now) - (1000 + S.HEARTBEAT_GRACE_S)) < 1
    os.utime(rd / "heartbeat_timed-S1.1.1", (now - 10, now - 10))             # still running: charged up to now
    assert abs(led2.charged(killed, now) - 2000) < 1
    # the next stage is armed with what is left, and refused when nothing is left
    eid2, limit2 = led2.open("timed-E32", 99999)
    assert limit2 < 3600 - 2000 + 5
    led2.close("x")
    led3 = C.Ledger(rd, "edge", allotment_h=0.5)
    raises(SystemExit, led3.open, "timed-E14", 100)
    rep = led2.report()
    assert rep["allotment_h"] == 1.0 and len(rep["entries"]) == 3 and abs(rep["used_h"] + rep["remaining_h"] - 1.0) < 1e-6
    assert C.Ledger(TMP / "ledger_none", "server").report()["used_h"] == 0 and C.Ledger(TMP, "server").allot_s == S.ALLOTMENT_H["server"] * 3600


def test_write_once_and_torn_rows():
    p = TMP / "once.json"
    C.write_json(p, {"a": 1})
    raises(FileExistsError, C.write_json, p, {"a": 2})
    j = TMP / "torn.jsonl"
    C.append_jsonl(j, {"n": 0})
    with open(j, "a") as f:
        f.write('{"n": 1, "tok')
    rows, torn = C.read_jsonl(j, tolerate_torn_tail=True)
    assert rows == [{"n": 0}] and torn == 1
    raises(ValueError, C.read_jsonl, j)
    d = TMP / "aside"
    d.mkdir(exist_ok=True)
    (d / "reference").mkdir(exist_ok=True)
    assert C.set_aside(d, "reference") == 1 and (d / "reference.crashed-1").exists() and C.set_aside(d, "reference") == 0


# --------------------------------------------------------------------------- #
# Decision rule                                                                #
# --------------------------------------------------------------------------- #
def _rows(n, correct, time_):
    """n queries per benchmark; correct[c] and time_[c] are functions of the query index."""
    return [{"key": f"{b}:{i}", "bench": b, "correct": {c: int(correct[c](i)) for c in correct}, "time": {c: float(time_[c](i)) for c in time_}}
            for b in S.BENCHMARKS for i in range(n)]


def test_utility_and_rule():
    always, never = (lambda i: 1), (lambda i: 0)
    cfgs = ("S", "E32", "E14", "E8", "R14")
    # the split answers every query in 100 s; the whole 32B model needs 200 s; the small models are fast and right on half
    rows = _rows(20, dict({c: always for c in ("S", "E32")}, **{c: (lambda i: i % 2) for c in ("E14", "E8", "R14")}),
                 {"S": lambda i: 100, "E32": lambda i: 200, "E14": lambda i: 20, "E8": lambda i: 10, "R14": lambda i: 5})
    assert ST.utility(rows, "S", 120) == 1.0 and ST.utility(rows, "E32", 120) == 0.0 and ST.utility(rows, "E8", 30) == 0.5
    d, us, best = ST.delta(rows, 30, S.POLICIES["no-raw-text"])
    assert d == -50.0 and best == "E8"                                         # equal utility: the smaller model
    assert ST.delta(rows, 30, S.POLICIES["open"])[2] == "E8"                    # and the edge before the server
    cells = ST.decide(rows, S.POLICIES["no-raw-text"], "no-raw-text", n_boot=300)
    assert [c["selected"] for c in cells] == ["E8", "E8", "S", "E32"] and cells[2]["delta_pt"] == 50.0 and cells[2]["ci95"][0] > 0
    assert cells[3]["delta_pt"] == 0.0 and cells[3]["best_competitor"] == "E32"
    t = ST.tables(rows, True, n_boot=200)
    assert set(t) == {"no-raw-text", "same-model", "open"} and set(ST.tables(rows, False, n_boot=50)) == {"no-raw-text", "same-model"}
    assert [c["selected"] for c in t["same-model"]] == [None, None, "S", "E32"]  # below the utility floor: not evaluated
    assert ST.decide(rows, S.POLICIES["no-raw-text"], "x", n_boot=200) == ST.decide(rows, S.POLICIES["no-raw-text"], "x", n_boot=200)
    # the margin: exactly 3.0 points passes even when floating point gives 2.9999999999999973 (the rounding rule)
    x, y = next((a / 400, b / 400) for a in range(400) for b in range(a) if a - b == 12 and 100 * (a / 400 - b / 400) < 3.0)
    assert ST.passes(100 * (x - y), 0.5) and 100 * (x - y) < 3.0 and not ST.passes(2.99, 0.5)
    assert not ST.passes(5.0, 0.0) and not ST.passes(5.0, None) and ST.passes(3.0, 1e-9)
    # an advantage the interval does not separate from zero selects the competitor
    weak = _rows(20, {"S": lambda i: i < 12, "E32": lambda i: i >= 9, "E14": never, "E8": never}, {c: (lambda i: 1) for c in cfgs[:4]})
    c = ST.decide(weak, S.POLICIES["no-raw-text"], "w", deadlines=(30,), n_boot=400)[0]
    assert c["delta_pt"] > 3.0 and c["ci95"][0] <= 0 and c["selected"] == "E32"
    floor = _rows(40, {c: (lambda i: i == 0) for c in cfgs}, {c: (lambda i: 1) for c in cfgs})
    assert all(not c["evaluated"] and c["selected"] is None for c in ST.decide(floor, S.POLICIES["open"], "f", n_boot=10))
    late = _rows(4, {c: always for c in cfgs}, dict({c: (lambda i: 1) for c in cfgs}, S=lambda i: float("inf")))
    assert ST.utility(late, "S", 300) == 0.0


def test_stability_zero_network_and_outcome():
    s1 = [{"key": k, "total_s": t} for k, t in (("a", 10.0), ("b", 20.0), ("c", 30.0), ("d", 99.0))]
    assert ST.stability(s1, [{"key": "a", "total_s": 10.5}, {"key": "b", "total_s": 21.0}, {"key": "c", "total_s": 31.5}])["stable"]
    r = ST.stability(s1, [{"key": "a", "total_s": 12.0}, {"key": "b", "total_s": 24.0}, {"key": "c", "total_s": 36.0}])
    assert abs(r["ratio"] - 1.2) < 1e-9 and not r["stable"] and not ST.stability(s1, [])["stable"]
    always = lambda i: 1  # noqa: E731
    rows = _rows(20, {c: always for c in ("S", "E32", "E14", "E8")}, {"S": lambda i: 130, "E32": lambda i: 200, "E14": lambda i: 200, "E8": lambda i: 200})
    z = ST.zero_network(rows, {r["key"]: 20.0 for r in rows}, n_boot=200)       # without the link the split would make the 120 s deadline
    assert z["would_select_split"] and [c["selected"] for c in z["cells"]][2] == "S"
    assert not any(c["selected"] == "S" for c in ST.decide(rows, S.POLICIES["no-raw-text"], "p", n_boot=200)[:3])
    cell = lambda sel, ev=True: [{"selected": sel, "evaluated": ev}]  # noqa: E731
    assert ST.outcome(True, cell("E8"), True, True) == "SCREEN_CONFIRMED" and ST.outcome(True, cell("S"), True, True) == "SCREEN_CONTRADICTED"
    assert ST.outcome(False, cell("S"), True, True) == "CORRECTNESS_FAIL" and ST.outcome(None, [], False, False) == "INCOMPLETE"
    assert ST.outcome(True, cell("S"), False, True) == "UNSTABLE" and ST.outcome(True, cell("S"), True, False) == "INCOMPLETE"
    assert ST.outcome(True, cell(None, False), True, True) == "NOT_EVALUABLE" and ST.outcome(None, [], False, False, reference_ran=False) == "NOT_EVALUABLE"
    assert ST.outcome(None, [], False, False, split_ran=False) == "SPLIT_INFEASIBLE"
    assert set(S.OUTCOMES) >= {ST.outcome(True, cell("S"), True, True), ST.outcome(True, cell("E8"), True, True)}


# --------------------------------------------------------------------------- #
# Generation loops and the two wires (in-process, CPU)                         #
# --------------------------------------------------------------------------- #
class Split:
    """Server thread + edge slice for the tiny 64-layer model, over loopback TCP, through the reused runtime."""

    def __init__(self):
        import torch
        self.K, self.R = RT.k16()
        self.R.setup_torch()
        self.p, self.h, d = RT.params(True), C.spec_hash(), T.tiny_models()["32B"]
        self.side = self.R.ServerSide(d, "cpu", self.p, None, with_static=False)
        self.sl = self.R.edge_slice(d, self.K.K, self.K.J, "cpu", torch.bfloat16, None)
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(1)
        self.stop = threading.Event()
        self.th = threading.Thread(target=self.R.serve, args=(self.side, self.sock, self.h, lambda ev: None), kwargs={"stop": self.stop}, daemon=True)
        self.th.start()
        self.addr = f"127.0.0.1:{self.sock.getsockname()[1]}"
        self.mid = self.R.connect(self.addr, self.sl, S.SPLIT_MODE, self.h, self.p)
        self.e, self.info = self.R.edge_split(self.sl, self.mid, S.SPLIT_MODE, self.p)

    def close(self):
        self.mid.close()
        self.stop.set()
        self.th.join(10)
        self.sock.close()


def test_whole_and_split_generation_loops():
    import torch
    _, R = RT.k16()
    R.setup_torch()
    md = T.tiny_models()
    hf = R.hf_model(md["32B"], "cpu")
    sync = R.E0RT._sync
    ids = [5 + (i * 7) % 150 for i in range(70)]                                # longer than one dev chunk (48): two prefill steps
    ref, _m = R.hf_greedy(hf, ids, 10, "cpu")                                   # K16's own greedy loop, one prefill pass
    w = RT.whole_generate(torch, hf, ids, 10, (), "cpu", 48, 1e9, sync)
    assert w["tokens"] == ref and w["stop"] == "cap" and not w["cutoff"] and w["n_prefill_steps"] == 2 and len(w["decode_forward_ns"]) == 9
    assert w["stamps_ns"] == sorted(w["stamps_ns"]) and len(w["stamps_ns"]) == 10
    assert RT.whole_generate(torch, hf, ids, 10, (), "cpu", 4096, 1e9, sync)["tokens"] == ref      # chunking does not change the tokens
    e = RT.whole_generate(torch, hf, ids, 10, (ref[3],), "cpu", 48, 1e9, sync)
    assert e["tokens"] == ref[:ref.index(ref[3]) + 1] and e["stop"] == "eos"
    c = RT.whole_generate(torch, hf, ids, 10, (), "cpu", 48, 0.0, sync)          # the first token is already past a zero cutoff
    assert c["tokens"] == ref[:1] and c["cutoff"] and c["stop"] == "cutoff"
    t = RT.summarize_times(w)
    assert t["ttft_s"] == w["stamps_ns"][0] / 1e9 and abs(t["per_token_s"] - (w["stamps_ns"][-1] - w["stamps_ns"][0]) / 9e9) < 1e-12
    assert RT.summarize_times(c)["per_token_s"] is None
    sp = Split()
    try:
        own = sp.e.generate(ids, 10, sp.p["chunk"], force_len=True)             # the reused runtime's own loop
        s = RT.split_generate(torch, sp.e, ids, 10, (), sp.p["chunk"], 1e9)
        assert s["tokens"] == own == ref                                        # split = unsplit on one CPU, through the wire
        assert s["kv_positions_in_step"] and s["n_prefill_steps"] == 2 and len(s["decode_steps"]["net_ns"]) == 9
        assert s["prefill"]["bytes_out"] == 70 * 64 * 2 + 2 * 72 and s["decode_sum"]["bytes_out"] == 9 * (64 * 2 + 72)
        assert s["decode_sum"]["server_compute_ns"] > 0 and s["decode_sum"]["rt_ns"] >= s["decode_sum"]["server_compute_ns"]
        assert RT.split_generate(torch, sp.e, ids, 10, (ref[2],), sp.p["chunk"], 1e9)["stop"] == "eos"
        cut = RT.split_generate(torch, sp.e, ids, 10, (), sp.p["chunk"], 0.0)
        assert cut["cutoff"] and cut["tokens"] == ref[:1]
        raises(ValueError, RT.split_generate, torch, sp.e, ids, 256, (), sp.p["chunk"], 1e9)       # beyond the context
        sp.mid.close()                                                          # the server takes one connection at a time
        assert RT.split_ready(R, sp.addr, md["32B"], sp.h, sp.p) is not None
        assert RT.split_ready(R, sp.addr, md["32B"], "0" * 64, sp.p) is None     # another revision is refused
    finally:
        sp.close()
    assert RT.split_ready(R, f"127.0.0.1:{T.free_port()}", md["32B"], C.spec_hash(), RT.params(True), timeout_s=2) is None


def test_remote_wire():
    import torch
    _, R = RT.k16()
    R.setup_torch()
    hf = R.hf_model(T.tiny_models()["14B"], "cpu")
    ids = [5 + (i * 11) % 150 for i in range(60)]
    local = RT.whole_generate(torch, hf, ids, 12, (), "cpu", 48, 1e9, R.E0RT._sync)
    events = []
    ident = {"spec_hash": C.spec_hash(), "model": "14B", "context_tokens": 256}
    srv = RT.RemoteServer(torch, hf, "cpu", R.E0RT._sync, ident, "t" * 64, 48, log=events.append)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    stop, out = threading.Event(), {}
    th = threading.Thread(target=lambda: out.update(why=RT.serve_remote(srv, sock, 60, 60, stop=stop)), daemon=True)
    th.start()
    addr = f"127.0.0.1:{sock.getsockname()[1]}"
    try:
        assert RT.remote_ready(addr, "t" * 64, C.spec_hash(), "14B") == ident
        assert RT.remote_ready(addr, "x" * 64, C.spec_hash(), "14B") is None    # wrong token
        assert RT.remote_ready(addr, "t" * 64, "0" * 64, "14B") is None         # another revision
        assert RT.remote_ready(addr, "t" * 64, C.spec_hash(), "8B") is None     # another model
        c = RT.RemoteClient(addr, "t" * 64, C.spec_hash(), "14B")
        r = c.generate(ids, 12, (), 1e9)
        assert r["tokens"] == local["tokens"] and r["stop"] == "cap" and len(r["stamps_ns"]) == 12 and r["stamps_ns"] == sorted(r["stamps_ns"])
        assert r["server"]["n"] == 12 and r["server"]["n_prefill_steps"] == 2 and r["bytes_in"] == 12 * (9 + 12) + 9 + len(json.dumps(r["server"]))
        assert r["server_token_ns"] == sorted(r["server_token_ns"]) and r["server"]["server_total_ns"] >= r["server_token_ns"][-1]
        e = c.generate(ids, 12, (local["tokens"][4],), 1e9)
        assert e["stop"] == "eos" and e["tokens"] == local["tokens"][:local["tokens"].index(local["tokens"][4]) + 1]
        k = c.generate(ids, 12, (), 0.0)                                        # the edge's clock decides: cancel after the first token
        assert k["cutoff"] and k["stop"] == "cutoff" and k["tokens"] == local["tokens"][:1] and k["server"]["n"] < 12
        again = c.generate(ids, 12, (), 1e9)                                    # the connection is still in step after a cancel
        assert again["tokens"] == local["tokens"]
        raises(RT.WireError, c.generate, ids, 500, (), 1e9)                     # beyond the server's context
        c.close()
    finally:
        stop.set()
        th.join(10)
        sock.close()
    assert out["why"] == "stopped" and sum(e["event"] == "refused" for e in events) == 3 and not any("ids" in e or "tokens" in e for e in events)
    assert RT.remote_ready(addr, "t" * 64, C.spec_hash(), "14B", timeout_s=2) is None


def test_server_idle_rules():
    srv = RT.RemoteServer(None, None, "cpu", None, {"spec_hash": "s", "model": "14B", "context_tokens": 8}, "t" * 64, 4)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    t0 = time.time()
    assert RT.serve_remote(srv, sock, 2, 99) == "no_edge" and 1.5 < time.time() - t0 < 6      # nobody came
    out = {}
    th = threading.Thread(target=lambda: out.update(why=RT.serve_remote(srv, sock, 30, 2)), daemon=True)
    th.start()
    assert RT.remote_ready(f"127.0.0.1:{sock.getsockname()[1]}", "t" * 64, "s", "14B") is not None
    th.join(15)
    sock.close()
    assert out.get("why") == "idle"                                             # served once, then released when idle


# --------------------------------------------------------------------------- #
# Multi-process dry runs of the real command sequence (tiny models, CPU)       #
# --------------------------------------------------------------------------- #
def _ok(r, code=0):
    assert r.returncode == code, f"exit {r.returncode}\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    return r


def _sites(tag, extra_dev=None, edge_extra=None):
    ports = (T.free_port(), T.free_port())
    srun, erun = TMP / f"{tag}_server", TMP / f"{tag}_edge"
    tpl = T.site("server", srun, ports, dict({"gpu_rows": INV, "gpu_apps": APPS}, **(extra_dev or {})), template=True)
    pinned = str(TMP / f"site_server_{tag}_pinned.json")
    edge = T.site("edge", erun, ports, dict(extra_dev or {}, **(edge_extra or {})))
    return srun, erun, tpl, pinned, edge


def _timed(edge, *blocks, env=None):
    for b in blocks:
        _ok(T.bench("timed", "--dev", "--site", edge, "--block", b, env=env))


def _config_rows(erun, cfg):
    return [r for b in S.CONFIG_BLOCKS[cfg] for f in sorted(erun.glob(f"timed_{b}.part*.jsonl")) for r in C.read_jsonl(f)]


def test_end_to_end_command_sequence():
    """The runbook's sequence, every stage in its own process: both halves of every block, scoring and the summary."""
    srun, erun, tpl, pinned, edge = _sites("e2e")
    _ok(T.bench("select-gpu", "--dev", "--site", tpl, "--out", pinned))
    sel = C.read_json(srun / C.GPU_SELECTION)
    assert sel["selected"]["index"] == 1 and json.loads(Path(pinned).read_text())["gpu_uuid"] == "GPU-bbbbbbbb-0000-0000-0000-000000000002"
    assert _ok(T.bench("preflight", "--dev", "--role", "server", "--site", pinned)).stdout.startswith("preflight server: GO")
    r = T.bench("preflight", "--dev", "--role", "edge", "--site", edge)
    assert r.returncode == 3 and "forwards are not up" in r.stdout              # no forward: not GO
    es = json.loads(Path(edge).read_text())
    held = []
    for name in ("split_port", "remote_port"):                                  # stand-ins for the two SSH forwards
        sk = socket.socket()
        sk.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sk.bind(("127.0.0.1", es[name]))
        sk.listen(1)
        held.append(sk)
    out = _ok(T.bench("preflight", "--dev", "--role", "edge", "--site", edge)).stdout
    assert out.startswith("preflight edge: GO") and "equal the pinned ids" in out
    for sk in held:
        sk.close()
    r = T.bench("timed", "--dev", "--site", edge, "--block", "S1a")
    assert r.returncode == 3 and "no timing without a correctness PASS" in r.stderr
    _ok(T.bench("reference", "--dev", "--site", edge))
    ref = C.read_json(erun / "reference.json")
    assert ref["screen_decision"]["decision"] == "EDGE" and len(ref["items"]) == 10 and len(ref["files"]) == 10
    assert (erun / "reference.attempt1.json").exists() and T.bench("reference", "--dev", "--site", edge).returncode == 3      # never run again
    r = T.bench("correctness", "--dev", "--site", edge)
    assert r.returncode == 3 and "does not answer" in r.stderr and not list(erun.glob("failure_*")) and not list(erun.glob("correctness.attempt*"))
    srv = T.start_server("serve", pinned)
    try:
        ready = T.wait_ready(srv)
        assert ready["stage"] == "serve"
        r = T.bench("select-gpu", "--dev", "--site", tpl, "--out", pinned + ".2")
        assert r.returncode == 3 and "fixed for this run" in r.stderr           # a server stage has run: no re-selection
        assert "correctness: PASS" in _ok(T.bench("correctness", "--dev", "--site", edge)).stdout
        r = T.bench("timed", "--dev", "--site", edge, "--block", "E32a")
        assert r.returncode == 3 and "block S1a must complete before E32a" in r.stderr
        _timed(edge, "S1a")
    finally:
        log = T.stop_server(srv)
    assert "server record written" in log and srv.returncode == 0
    rec = C.read_json(srun / "server_record.1.json")
    assert rec["ended"] == "stopped" and rec["events"]["hello_ok"] >= 4 and rec["events"]["refused"] == 0
    _timed(edge, "E32a", "E14a", "E8a")
    r = T.bench("timed", "--dev", "--site", edge, "--block", "S1b")
    assert r.returncode == 3 and "split server does not answer" in r.stderr     # the server was released: it must be started again
    r = T.bench("timed", "--dev", "--site", edge, "--block", "E32b")
    assert r.returncode == 3 and "block S1b must complete before E32b" in r.stderr
    srv = T.start_server("serve", pinned)
    try:
        T.wait_ready(srv)
        _timed(edge, "S1b", "S2")
    finally:
        T.stop_server(srv)
    assert (srun / "server_record.2.json").exists()
    _timed(edge, "E32b", "E14b", "E8b")
    srv = T.start_server("serve-remote", pinned)
    try:
        assert T.wait_ready(srv)["stage"] == "serve-remote"
        _ok(T.bench("preflight", "--dev", "--role", "edge", "--site", edge), 3)  # not GO: only one of the two forwards answers
        _timed(edge, "R14")
    finally:
        T.stop_server(srv)
    assert C.read_json(srun / "remote_record.1.json")["events"]["hello_ok"] >= 2
    assert T.bench("timed", "--dev", "--site", edge, "--block", "E8b").returncode == 3           # outputs are written once
    # rows
    pub, priv = T.dev_workload()
    w = C.read_json(pub)
    n, half = len(w["utility"]), len(w["utility"]) // 2
    s1, e32, r14 = _config_rows(erun, "S"), _config_rows(erun, "E32"), _config_rows(erun, "R14")
    assert [r["key"] for r in s1] == w["utility"] == [r["key"] for r in r14] and [r["n"] for r in s1] == list(range(n))
    assert [r["key"] for r in C.read_jsonl(erun / "timed_S1a.part1.jsonl")] == w["utility"][:half]
    assert [r["key"] for r in C.read_jsonl(erun / "timed_E8b.part1.jsonl")] == w["utility"][half:]
    assert [r["key"] for r in C.read_jsonl(erun / "timed_S2.part1.jsonl")] == w["sentinel"]
    assert all(r["tokens"] == q["tokens"] and r["ids_sha256"] == q["ids_sha256"] for r, q in zip(s1, e32))    # the split repeats the whole model
    assert all(r["stop"] == "cap" and r["n_out"] == 12 and len(r["stamps_s"]) == 12 and abs(r["total_s"] - r["stamps_s"][-1]) < 1e-6 for r in s1 + e32 + r14)
    assert all(r["kv_positions_in_step"] and r["decode_sum"]["net_ns"] != 0 and len(r["decode_steps"]["rt_ns"]) == 11 for r in s1)
    assert any(r["n_prefill_steps"] == 2 for r in s1) and all(len(r["decode_forward_ns"]) == 11 for r in e32)
    assert all(r["server"]["n"] == 12 and r["bytes_out"] > 0 for r in r14)
    done = C.read_json(erun / "timed_S1b.done.json")
    assert done["attempts"] == 1 and done["n_rows"] == n - half and done["first_index"] == half and done["memory_gate"]["load"] == "split_edge"
    assert done["cutoff_s"] == 300.0 and C.read_json(erun / "timed_R14.done.json")["budget"]["ledger_id"] is None       # the REMOTE client holds no edge GPU
    # ledger: every GPU stage of each host opened and closed an entry
    le = C.Ledger(erun, "edge").report()
    assert [e["stage"] for e in le["entries"]] == ["reference", "correctness"] + [f"timed-{b}" for b in S.PRIMARY_BLOCKS]
    assert all(e["closed"] for e in le["entries"]) and 0 < le["used_h"] < 1
    ls = C.Ledger(srun, "server").report()
    assert [e["stage"] for e in ls["entries"]] == ["serve", "serve", "serve-remote"] and all(e["closed"] for e in ls["entries"])
    assert "used" in _ok(T.bench("ledger", "--dev", "--role", "edge", "--site", edge)).stdout
    # scoring and summary
    r = T.bench("summary", "--dev", "--edge-run", str(erun), "--server-run", str(srun), "--out", str(TMP / "e2e_sum0"))
    assert C.read_json(TMP / "e2e_sum0" / "summary.json")["outcome"] == "INCOMPLETE"             # no scores yet
    assert "scored_before_primary_blocks_complete" not in C.read_json(TMP / "e2e_sum0" / "summary.json")
    _ok(T.bench("score", "--dev", "--edge-run", str(erun), "--workload-private", priv, "--workload-public", pub, "--scorer", "exact"))
    assert sum(C.read_json(erun / "scores_S1a.json")["scores"].values()) == half                # the answers are the 32B stand-in's text
    assert sum(C.read_json(erun / "scores_E8b.json")["scores"].values()) == 0
    _ok(T.bench("summary", "--dev", "--edge-run", str(erun), "--server-run", str(srun), "--out", str(TMP / "e2e_sum")))
    s = C.read_json(TMP / "e2e_sum" / "summary.json")
    # CPU timings of a busy desktop need not pass the stability band; the rule itself is what is checked here
    assert s["outcome"] == ("SCREEN_CONFIRMED" if s["stability"]["stable"] else "UNSTABLE") and 0.3 < s["stability"]["ratio"] < 3
    assert s["dev"] and s["primary_blocks_complete"] and s["remote_evaluated"] and "first_half_only" not in s
    import qsel_bench as B
    band = S.STABILITY_BAND
    try:
        S.STABILITY_BAND = (0.0, 100.0)
        assert B.summarise(erun, srun, dev=True, n_boot=200)["outcome"] == "SCREEN_CONFIRMED"
    finally:
        S.STABILITY_BAND = band
    assert s["environment_consistent"] and s["site_consistent"] and s["correctness"]["verdict"] == "PASS"
    assert set(s["tables"]) == {"no-raw-text", "same-model", "open"} and len(s["tables"]["open"]) == 4
    c = s["tables"]["no-raw-text"][0]
    assert c["utility"]["S"] == 1.0 and c["utility"]["E32"] == 1.0 and c["delta_pt"] == 0.0 and c["selected"] == "E32"       # equal: Edge keeps it
    assert "R14" in s["tables"]["open"][0]["utility"] and s["same_model_split_advantage"] is False
    assert s["token_agreement"]["S_vs_E32_identical_share"] == 1.0 and s["token_agreement"]["R14_vs_E14_identical_share"] == 1.0
    assert s["quality_pct"]["S"]["all"] == 100.0 and s["quality_pct"]["E14"]["all"] == 0.0
    assert s["latency"]["S"]["attempts"] == 2 and s["latency"]["S"]["queries"] == n and s["latency"]["S"]["blocks"] == ["S1a", "S1b"]
    assert set(s["per_token_time_by_half"]) == {"S", "E32", "E14", "E8"} and all(v["second_over_first"] > 0 for v in s["per_token_time_by_half"].values())
    assert s["network"]["n"] == n and s["network"]["bytes_out_median"] > 0 and s["stability"]["n"] == len(w["sentinel"])
    assert s["gpu_hours"]["edge"] > 0 and s["gpu_hours"]["server"] > 0 and s["gpu_hours"]["ceiling"] == 48.0
    assert s["scored_before_primary_blocks_complete"] is False and s["remote_32B_memory_fit"] is None       # dev preflights record no real GPU
    assert s["zero_network_counterfactual"]["would_select_split"] is False and s["runtime_parity"]["ratio"] > 0
    text = json.dumps(s)
    assert socket.gethostname() not in text and "bbbbbbbb" not in text and str(TMP) not in text       # the summary carries no host detail
    assert s["outcome"] in (TMP / "e2e_sum" / "SUMMARY.md").read_text()
    # a target command refuses dev outputs, and refuses to run outside the frozen tag
    r = T.bench("summary", "--edge-run", str(erun), "--out", str(TMP / "e2e_sum_target"))
    assert r.returncode == 3 and ("HEAD is not the frozen tag" in r.stderr or "dev outputs cannot be summarised" in r.stderr or "differs from the tag" in r.stderr)


def test_resume_after_a_died_attempt_and_attempt_limit():
    srun, erun, tpl, pinned, edge = _sites("resume", extra_dev={"limit": 12})  # halves of six queries
    _ok(T.bench("select-gpu", "--dev", "--site", tpl, "--out", pinned))
    _ok(T.bench("preflight", "--dev", "--role", "server", "--site", pinned))
    _ok(T.bench("reference", "--dev", "--site", edge))
    srv = T.start_server("serve", pinned)
    try:
        T.wait_ready(srv)
        _ok(T.bench("correctness", "--dev", "--site", edge))
        r = T.bench("timed", "--dev", "--site", edge, "--block", "S1a", env={"QSEL_DEV_DIE_AT": "2"})
        assert r.returncode == 1 and "simulated death" in r.stderr
        assert len(C.read_jsonl(erun / "timed_S1a.part1.jsonl")) == 2 and not (erun / "timed_S1a.done.json").exists()
        assert C.read_json(erun / "failure_timed.1.json")["error_type"] == "RuntimeError"
        with open(erun / "timed_S1a.part1.jsonl", "a") as f:                    # a line cut short by the dying process
            f.write('{"n": 2, "key": "x", "tok')
        r = T.bench("timed", "--dev", "--site", edge, "--block", "S1a", env={"QSEL_DEV_DIE_AT": "4"})
        assert r.returncode == 1 and [x["n"] for x in C.read_jsonl(erun / "timed_S1a.part2.jsonl")] == [2, 3]
        _timed(edge, "S1a")                                                     # third attempt: finishes
        done = C.read_json(erun / "timed_S1a.done.json")
        assert done["attempts"] == 3 and done["n_rows"] == 6 and done["torn_lines"] == 1 and len(done["parts"]) == 3
        assert [x["attempt"] for x in C.read_jsonl(erun / "timed_S1a.part3.jsonl")] == [3, 3]
        assert C.read_json(erun / "timed_S1a.attempt3.json")["start_index"] == 4
        le = C.Ledger(erun, "edge").report()
        assert [e["stage"] for e in le["entries"]].count("timed-S1a") == 3 and all(e["closed"] for e in le["entries"])
    finally:
        T.stop_server(srv)
    for _ in range(3):                                                          # three attempts that all die: the block is INCOMPLETE
        assert T.bench("timed", "--dev", "--site", edge, "--block", "E32a", env={"QSEL_DEV_DIE_AT": "0"}).returncode == 1
    assert not list(erun.glob("timed_E32a.part*")) and len(list(erun.glob("timed_E32a.attempt*.json"))) == 3     # attempts count without a row
    r = T.bench("timed", "--dev", "--site", edge, "--block", "E32a")
    assert r.returncode == 3 and "INCOMPLETE" in r.stderr
    pub, priv = T.dev_workload()
    r = T.bench("score", "--dev", "--edge-run", str(erun), "--workload-private", priv, "--workload-public", pub, "--scorer", "exact")
    assert r.returncode == 3 and "primary blocks are not complete" in r.stderr and not (erun / "scores_S1a.json").exists()
    _ok(T.bench("score", "--dev", "--partial", "--edge-run", str(erun), "--workload-private", priv, "--workload-public", pub, "--scorer", "exact"))
    _ok(T.bench("summary", "--dev", "--edge-run", str(erun), "--server-run", str(srun), "--out", str(TMP / "resume_sum")))
    s = C.read_json(TMP / "resume_sum" / "summary.json")
    assert s["outcome"] == "INCOMPLETE" and s["blocks_present"] == ["S1a"] and s["tables"] == {} and s["scored_before_primary_blocks_complete"]
    assert s["latency"]["S"]["queries"] == 6 and s["latency"]["S"]["attempts"] == 3 and "first_half_only" not in s


def test_failed_correctness_ends_the_experiment():
    srun, erun, tpl, pinned, edge = _sites("fail")
    _ok(T.bench("select-gpu", "--dev", "--site", tpl, "--out", pinned))
    _ok(T.bench("reference", "--dev", "--site", edge))
    srv = T.start_server("serve", pinned, env={"QSEL_DEV_SABOTAGE": "1"})      # the server runs its layers in the wrong order
    try:
        T.wait_ready(srv)
        r = T.bench("correctness", "--dev", "--site", edge)
        assert r.returncode == 2 and "correctness: FAIL" in r.stdout
        r = T.bench("timed", "--dev", "--site", edge, "--block", "S1a")
        assert r.returncode == 3 and "no timing without a correctness PASS" in r.stderr
        assert T.bench("correctness", "--dev", "--site", edge).returncode == 3   # FAIL is final: no second run
    finally:
        T.stop_server(srv)
    corr = C.read_json(erun / "correctness.json")
    assert corr["verdict"] == "FAIL" and not corr["result"]["checks"]["C3_mean_kl"]
    _ok(T.bench("summary", "--dev", "--edge-run", str(erun), "--server-run", str(srun), "--out", str(TMP / "fail_sum")))
    s = C.read_json(TMP / "fail_sum" / "summary.json")
    assert s["outcome"] == "CORRECTNESS_FAIL" and s["tables"] == {} and not list(erun.glob("timed_*"))


def test_budget_stop_and_infeasible_outcomes():
    srun, erun, tpl, pinned, edge = _sites("budget", extra_dev={"allotment_h": 0.0})
    r = T.bench("reference", "--dev", "--site", edge)
    assert r.returncode == 3 and "BUDGET" in r.stderr and not (erun / "reference.json").exists() and not list(erun.glob("failure_*"))
    _ok(T.bench("select-gpu", "--dev", "--site", tpl, "--out", pinned))
    r = T.bench("serve", "--dev", "--site", pinned)
    assert r.returncode == 3 and "BUDGET" in r.stderr and not list(srun.glob("server_started*"))
    # the split's model cannot run whole on the edge: two died attempts of `reference` are NOT_EVALUABLE
    srun, erun, tpl, pinned, edge = _sites("noref")
    bad = json.loads(Path(edge).read_text())
    bad["model_dirs"] = dict(bad["model_dirs"], **{"32B": str(TMP / "no_such_model")})
    Path(edge).write_text(json.dumps(bad))
    for _ in range(2):
        assert T.bench("reference", "--dev", "--site", edge).returncode == 1
    r = T.bench("reference", "--dev", "--site", edge)
    assert r.returncode == 3 and "NOT_EVALUABLE" in r.stderr
    _ok(T.bench("summary", "--dev", "--edge-run", str(erun), "--out", str(TMP / "noref_sum")))
    assert C.read_json(TMP / "noref_sum" / "summary.json")["outcome"] == "NOT_EVALUABLE"
    # select-gpu: nothing eligible is BLOCKED, and nothing is written
    srun, erun, tpl, pinned, edge = _sites("blocked", extra_dev={"gpu_rows": INV.replace("47800", "100").replace("41884", "100"), "gpu_apps": APPS})
    site = json.loads(Path(tpl).read_text())
    site["excluded_gpu_uuids"] = []
    Path(tpl).write_text(json.dumps(site))
    r = T.bench("select-gpu", "--dev", "--site", tpl, "--out", pinned)
    assert r.returncode == 0                                                    # dev needs 1 MiB: every A6000 is eligible
    assert C.read_json(srun / C.GPU_SELECTION)["selected"]["index"] == 1 and T.bench("select-gpu", "--dev", "--site", tpl, "--out", pinned).returncode == 0
    assert (srun / (C.GPU_SELECTION + ".superseded-1")).exists()               # re-selection before any server stage keeps the earlier record


def test_wall_clock_limit_ends_a_server():
    """The limit is not an OSError: neither server loop can swallow it, idle or with an edge connected."""
    import signal
    import qsel_bench as B
    assert not issubclass(B.LimitReached, Exception)
    srv = RT.RemoteServer(None, None, "cpu", None, {"spec_hash": "s", "model": "14B", "context_tokens": 8}, "t" * 64, 4)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    try:
        t0 = time.time()
        B.arm(1)
        raises(B.LimitReached, RT.serve_remote, srv, sock, 60, 60)              # while waiting for an edge
        assert time.time() - t0 < 6
        c = socket.create_connection(sock.getsockname())
        RT.send_frame(c, RT.R_HELLO, json.dumps({"token": "t" * 64, "spec_hash": "s", "model": "14B"}).encode())
        B.arm(1)
        raises(B.LimitReached, RT.serve_remote, srv, sock, 60, 60)              # while an edge is connected and silent
        c.close()
        _, R = RT.k16()
        R.setup_torch()
        side = R.ServerSide(T.tiny_models()["32B"], "cpu", RT.params(True), None, with_static=False)
        B.arm(1)
        raises(B.LimitReached, R.serve, side, sock, "h", lambda ev: None, accept_timeout_s=60)      # the reused split server loop
        c = socket.create_connection(sock.getsockname())
        B.arm(1)
        raises(B.LimitReached, R.serve, side, sock, "h", lambda ev: None, accept_timeout_s=60)      # and with a silent connection
        c.close()
    finally:
        signal.alarm(0)
        sock.close()


def test_ledger_survives_a_torn_line():
    rd = TMP / "ledger_torn"
    rd.mkdir(exist_ok=True)
    led = C.Ledger(rd, "edge", allotment_h=1.0)
    led.open("reference", 60)
    led.close("done")
    with open(rd / C.LEDGER, "a") as f:
        f.write('{"id": "timed-S1.9.9", "stage": "timed-S1", "event": "op')         # cut short, no newline
    led2 = C.Ledger(rd, "edge", allotment_h=1.0)
    eid, _ = led2.open("timed-E32", 60)                                         # the next entry starts on its own line
    led2.close("done")
    rep = C.Ledger(rd, "edge", allotment_h=1.0).report()
    assert rep["unreadable_lines"] == 1 and [e["stage"] for e in rep["entries"]] == ["reference", "timed-E32"] and all(e["closed"] for e in rep["entries"])


def test_real_evaluators():
    """The unchanged benchmark evaluators, where they can be imported (the analysis machine)."""
    import warnings
    import qsel_bench as B
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ev = B.evaluators()
    except ImportError as e:
        print(f"    (benchmark evaluators not importable under this interpreter: {e.name}; skipped)")
        return
    assert set(ev) == set(S.BENCHMARKS)
    for b in ("mmlupro", "gpqa"):
        assert ev[b].evaluate({"answer": "B"}, "Some reasoning.\n\nAnswer: B")["is_correct"] is True
        assert ev[b].evaluate({"answer": "B"}, "Answer: C")["is_correct"] is False and ev[b].evaluate({"answer": "B"}, "")["is_correct"] is False
    assert ev["livemathbench"].evaluate({"answer": "42"}, "So the result is \\boxed{42}")["is_correct"] is True
    assert ev["livemathbench"].evaluate({"answer": "42"}, "x" * 4000)["is_correct"] is False


def test_split_server_releases_when_no_edge_comes():
    srun, erun, tpl, pinned, edge = _sites("idle", extra_dev={"first_wait_s": 2, "idle_exit_s": 2, "accept_tick_s": 1})
    _ok(T.bench("select-gpu", "--dev", "--site", tpl, "--out", pinned))
    srv = T.start_server("serve", pinned)
    T.wait_ready(srv)
    assert srv.wait(60) == 0                                                    # by itself: nobody connected
    rec = C.read_json(srun / "server_record.1.json")
    assert rec["ended"] == "no_edge" and rec["events"]["hello_ok"] == 0
    rep = C.Ledger(srun, "server").report()
    assert [e["stage"] for e in rep["entries"]] == ["serve"] and rep["entries"][0]["closed"]


def test_cut_off_queries_end_to_end():
    """Every query is cut off at its first token (a zero cutoff): rows, the REMOTE cancel, scoring and the summary.
    On the way: the report of a run that ended after its first halves, and the budget gate before the second halves."""
    import argparse
    import shutil
    import qsel_bench as B
    srun, erun, tpl, pinned, edge = _sites("cut", extra_dev={"cutoff_s": 0.0, "limit": 4})       # halves of two queries
    _ok(T.bench("select-gpu", "--dev", "--site", tpl, "--out", pinned))
    _ok(T.bench("preflight", "--dev", "--role", "server", "--site", pinned))
    _ok(T.bench("reference", "--dev", "--site", edge))
    pub, priv = T.dev_workload()
    srv = T.start_server("serve", pinned)
    try:
        T.wait_ready(srv)
        _ok(T.bench("correctness", "--dev", "--site", edge))
        _timed(edge, "S1a", "E32a", "E14a", "E8a")
        # the first halves used more than is left: the second halves are not started, and that is not an attempt
        used = C.Ledger(erun, "edge").report()["used_h"]
        r = T.bench("timed", "--dev", "--site", edge, "--block", "S1b", env={"QSEL_DEV_ALLOTMENT_H": str(used * 1.2)})
        assert r.returncode == 3 and "BUDGET: the first halves used" in r.stderr and not list(erun.glob("timed_S1b.*"))
        _ok(T.bench("score", "--dev", "--partial", "--edge-run", str(erun), "--workload-private", priv, "--workload-public", pub, "--scorer", "exact"))
        _ok(T.bench("summary", "--dev", "--edge-run", str(erun), "--server-run", str(srun), "--out", str(TMP / "cut_partial")))
        s = C.read_json(TMP / "cut_partial" / "summary.json")                    # an incomplete run still reports what it measured
        assert s["outcome"] == "INCOMPLETE" and not s["selection_claimed"] and s["tables"] == {} and s["scored_before_primary_blocks_complete"]
        assert set(s["latency"]) == {"S", "E32", "E14", "E8"} and all(v["cutoffs"] == 2 and v["output_tokens"] == 2 for v in s["latency"].values())
        assert set(s["quality_pct"]) == {"S", "E32", "E14", "E8"} and s["per_token_time_by_half"] == {}
        fh = s["first_half_only"]                                               # the balanced half, as description only
        assert fh["queries"] == 2 and fh["descriptive_not_the_outcome"] and set(fh["tables"]) == {"no-raw-text", "same-model"}
        _timed(edge, "S1b", "S2")
    finally:
        T.stop_server(srv)
    _timed(edge, "E32b", "E14b", "E8b")
    srv = T.start_server("serve-remote", pinned)
    try:
        T.wait_ready(srv)
        _timed(edge, "R14")
    finally:
        T.stop_server(srv)
    rows = {c: _config_rows(erun, c) for c in ("S", "E32", "R14")}
    assert all(len(v) == 4 and all(r["cutoff"] and r["stop"] == "cutoff" and r["n_out"] == 1 for r in v) for v in rows.values())
    assert all(r["server"]["n"] < r["max_new"] for r in rows["R14"])            # the server stopped at the edge's cancel
    r = T.bench("score", "--dev", "--edge-run", str(erun), "--workload-private", priv, "--workload-public", pub, "--scorer", "exact")
    assert _ok(r).stdout.count("already written") == 4 and "scores_S2.json" in r.stdout          # a repeated score continues
    _ok(T.bench("summary", "--dev", "--edge-run", str(erun), "--server-run", str(srun), "--out", str(TMP / "cut_sum")))
    s = C.read_json(TMP / "cut_sum" / "summary.json")
    assert s["primary_blocks_complete"] and s["outcome"] in ("NOT_EVALUABLE", "UNSTABLE") and not s["selection_claimed"] and "first_half_only" not in s
    assert all(u == 0.0 for c in s["tables"]["open"] for u in c["utility"].values()) and all(c["selected"] is None for c in s["tables"]["no-raw-text"])
    assert s["latency"]["R14"]["cutoffs"] == 4 and "not claimed" in (TMP / "cut_sum" / "SUMMARY.md").read_text()
    # an output the evaluator cannot read is a wrong answer, recorded; it never ends the scoring
    copy = TMP / "cut_edge_copy"
    shutil.copytree(erun, copy)
    for f in copy.glob("scores_*.json"):
        f.unlink()

    class Boom:
        def evaluate(self, item, text):
            raise ValueError("unreadable")
    real = B.evaluators
    try:
        B.evaluators = lambda: {b: Boom() for b in S.BENCHMARKS}
        B.cmd_score(argparse.Namespace(dev=True, edge_run=str(copy), workload_private=priv, workload_public=pub, scorer=None, partial=False))
    finally:
        B.evaluators = real
    sc = C.read_json(copy / "scores_E8b.json")
    assert sum(sc["scores"].values()) == 0 and len(sc["evaluator_errors"]) == 2 and sc["evaluator_errors"][0]["error_type"] == "ValueError"


def test_target_commands_refuse_outside_the_freeze():
    srun, erun, tpl, pinned, edge = _sites("target")
    for f in (tpl, edge):                                                       # as on a machine that is not the laboratory's
        d = json.loads(Path(f).read_text())
        d["expected_hostname"] = "not-this-host"
        Path(f).write_text(json.dumps(d))
    for args in (("select-gpu", "--site", tpl, "--out", pinned), ("preflight", "--role", "edge", "--site", edge), ("reference", "--site", edge),
                 ("timed", "--site", edge, "--block", "S1a"), ("serve", "--site", tpl), ("serve-remote", "--site", tpl)):
        r = T.bench(*args)                                                      # before the tag: not the frozen tag. At the tag: not the named host
        assert r.returncode == 3 and r.stderr.startswith("STOP:"), (args, r.stderr[-300:])
    assert not erun.exists() and not srun.exists() and not Path(pinned).exists()
    r = T.bench("score", "--edge-run", str(erun), "--workload-private", "x", "--scorer", "exact")
    assert r.returncode == 3 and "dev only" in r.stderr


# --------------------------------------------------------------------------- #
QUICK_SKIP = ("test_end_to_end_command_sequence", "test_resume_after_a_died_attempt_and_attempt_limit", "test_failed_correctness_ends_the_experiment",
              "test_budget_stop_and_infeasible_outcomes", "test_split_server_releases_when_no_edge_comes", "test_cut_off_queries_end_to_end")


def main():
    quick = "--quick" in sys.argv
    only = sys.argv[sys.argv.index("-k") + 1] if "-k" in sys.argv else None
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        if (quick and name in QUICK_SKIP) or (only and only not in name):
            continue
        t0 = time.time()
        try:
            fn()
            print(f"ok    {name}  ({time.time() - t0:.1f} s)", flush=True)
        except Exception:
            import traceback
            failed += 1
            print(f"FAIL  {name}", flush=True)
            traceback.print_exc()
    print("OK" if not failed else f"FAILED: {failed}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
