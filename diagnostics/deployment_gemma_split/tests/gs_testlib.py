"""Shared fixtures for the GEMMA-SPLIT CPU tests (standard library only). Fakes stand in for lab facts so that the
protocol logic can be exercised; no scientific number is faked, and nothing outside temporary directories is written.
"""
import json
import os
import shutil
import socket
import time
import subprocess
import sys
import tempfile
from pathlib import Path

GS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GS))
import gs_common as C  # noqa: E402
import gs_spec as S  # noqa: E402

SERVER_HOST = S.SERVER_IDENTITY["hostname"]
EDGE_HOST = "spark-test-0001"
SERVER_CVD = "0"                           # the target fixture's server pins its frozen GPU (PREREG §8)
VERSIONS = {"python": "3.12.3", "torch": "2.9.1", "torch_cuda": "13.0", "transformers": "5.19.0", "driver": "580.95"}
SMI_SERVER = "\n".join(f"{i}, NVIDIA RTX A6000, GPU-a6000-000{i}, 49140 MiB, 4 MiB, 580.95.05, 8.6, 00000000:0{i + 1}:00.0" for i in range(3))
SMI_EDGE = "0, NVIDIA GB10, GPU-gb10-0000, [N/A], [N/A], 580.95.05, 12.1, 0000000F:01:00.0"


def tmpdir(prefix="gs_"):
    return Path(tempfile.mkdtemp(prefix=prefix))


def _ignore(src, names):
    rel = Path(src).relative_to(GS).as_posix()
    skip = {n for n in names if n in ("__pycache__", "runs") or n.endswith(".pyc")}
    if rel in ("capture", "capture/edge", "capture/server"):
        skip |= {n for n in names if (Path(src) / n).is_file() and not (n.endswith(".template.json") or n in ("README.md", "LAB_QUESTIONS.md"))}
        skip |= {n for n in names if n in ("edge", "server")}
    if rel in ("adapter", "frozen"):
        skip |= {n for n in names if n != "README.md"}
    return skip


def copy_gs(root):
    """Copy the GEMMA-SPLIT directory (templates only in capture/, adapter/, frozen/) to <root>/diagnostics/... so that
    gs_freeze.repo_of(<copy>) == root."""
    dst = Path(root) / "diagnostics" / "deployment_gemma_split"
    shutil.copytree(GS, dst, ignore=_ignore)
    return dst


def git(root, *args):
    r = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {args}: {r.stderr}")
    return r.stdout.strip()


def git_repo_with_gs(root):
    gs = copy_gs(root)
    git(root, "init", "-q")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "config", "user.name", "gs-test")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "protocol")
    git(root, "tag", S.PREREG_TAG)
    return gs


def gpu(index, name, uuid, mem):
    return {"index": index, "name": name, "uuid": uuid, "memory_total_mib": mem, "memory_used_mib": None, "memory_raw": [str(mem), ""],
            "driver": "580.95.05", "compute_cap": 8.6 if mem else 12.1, "pci_bus_id": f"0:{index}"}


def fake_inventory(role, hostname=None, gpus=None, arch=None, route_ok=True, versions=None):
    hostname = hostname or (SERVER_HOST if role == "server" else EDGE_HOST)
    if gpus is None:
        gpus = [gpu(i, "NVIDIA RTX A6000", f"GPU-a6000-000{i}", 49140) for i in range(3)] if role == "server" else \
            [gpu(0, "NVIDIA GB10", "GPU-gb10-0000", None)]
    return {"schema": "gemma-split-inventory/1", "experiment": S.NAME, "declared_role": role, "hostname": hostname, "fqdn": hostname,
            "utc": "2026-11-16T00:00:00Z", "arch": arch or ("x86_64" if role == "server" else "aarch64"), "gpus": gpus,
            "torch": {"importable": True, "version": VERSIONS["torch"], "cuda": VERSIONS["torch_cuda"], "cuda_available": True, "devices": []},
            "packages": {"transformers": VERSIONS["transformers"]}, "python": {"version": VERSIONS["python"]},
            "route_to_peer": {"rc": 0, "out": "10.0.0.2 dev enp1s0 src 10.0.0.1"} if route_ok else {"rc": 2, "out": "", "err": "Network is unreachable"},
            "memory": {"MemTotal": 1 << 37}, "env": {}, "nics": [], "versions": dict(versions or VERSIONS),
            "clock": {"timedatectl": "NTPSynchronized=yes\nTimezone=Etc/UTC", "chrony": ""}}


def roles_for(edge_host=EDGE_HOST, server_host=SERVER_HOST, verified=True):
    dev = lambda u: [{"index": 0, "name": "x", "uuid": u, "memory_total_mib": None}]  # noqa: E731
    return {"schema": "gemma-split-roles/1", "edge": {"host": edge_host, "devices": dev("GPU-gb10-0000")},
            "server": {"host": server_host, "devices": dev("GPU-a6000-0000")}, "verified": verified, "problems": []}


TINY_CONFIG = {"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4_text", "num_hidden_layers": 8, "hidden_size": 32,
               "hidden_size_per_layer_input": 8, "vocab_size_per_layer_input": 106, "num_kv_shared_layers": 3, "sliding_window": 8,
               "layer_types": ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "sliding_attention",
                               "full_attention", "sliding_attention", "full_attention"],
               "final_logit_softcapping": 30.0, "vocab_size": 106, "dtype": "float32", "eos_token_id": 1, "bos_token_id": 2}


def fake_model_dir(root, name=S.MODEL_DIR_NAME, config=None):
    d = Path(root) / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(config or TINY_CONFIG))
    (d / "model.safetensors").write_bytes(b"\0" * 64)
    (d / "tokenizer.json").write_text("{}")
    (d / "generation_config.json").write_text(json.dumps({"eos_token_id": [1], "do_sample": True, "top_k": 64}))
    return d


def double_split_capture(endpoint="127.0.0.1:1", cut=3, L=8, device="cpu", uuid="cpu", config_id="double-cut3"):
    """The split capture that describes tests/doubles/gemma4_split_double.py (prefix edge, suffix server)."""
    return {
        "schema": "gemma-split-capture/1", "configuration_id": config_id, "captured_utc": "2026-11-16T00:00:00Z", "captured_by": "test",
        "sources": [{"kind": "code_reading", "ref": "tests/doubles/gemma4_split_double.py", "sha256": None}],
        "implementation": {"name": "gemma4-tiny-split-double", "framework": "pytorch+transformers", "entry_points": {"edge": "x", "server": "y"}},
        "model": {"num_hidden_layers": L, "dtype_edge": "float32", "dtype_server": "float32", "quantization": "none", "attn_implementation": None,
                  "checkpoint_note": "test double"},
        "ownership": {"edge": {"layers": [[0, cut]], "embed_tokens": True, "per_layer_embeddings": True, "final_norm": False, "lm_head": False,
                               "sampling": False, "tokenizer": True},
                      "server": {"layers": [[cut, L]], "embed_tokens": False, "per_layer_embeddings": False, "final_norm": True, "lm_head": True,
                                 "sampling": True, "tokenizer": False}, "duplicated_parts_reason": {}},
        "boundary": [
            {"direction": "edge_to_server", "name": "hidden_states", "dtype": "float32", "shape": "[1, n, hidden_size]", "when": "every_step"},
            {"direction": "edge_to_server", "name": "per_layer_inputs", "dtype": "float32", "shape": "[1, n, L-cut, ple_dim]", "when": "every_step"},
            {"direction": "edge_to_server", "name": "shared_kv_full_attention_k", "dtype": "float32", "shape": "[1, kv, n, d]", "when": "every_step"},
            {"direction": "edge_to_server", "name": "shared_kv_full_attention_v", "dtype": "float32", "shape": "[1, kv, n, d]", "when": "every_step"},
            {"direction": "server_to_edge", "name": "next_token", "dtype": "int64", "shape": "[1]", "when": "every_step"}],
        "kv_cache": {"kv_cache_edge": "none (full recompute per step)", "kv_cache_server": "none (full recompute per step)",
                     "shared_kv_handling": "edge sends the full-attention source layer's K/V every step", "prefix_caching": "none"},
        "transport": {"protocol": "tcp", "serialization": "length-prefixed JSON header + raw tensor bytes", "endpoint": endpoint,
                      "experiment_endpoint": "%s:%d" % (endpoint.rsplit(":", 1)[0], int(endpoint.rsplit(":", 1)[1]) + 1), "compression": "none"},
        "generation": {"sampling_location": "server", "default_sampling": {}, "greedy_supported": True, "streaming_supported": True,
                       "ignore_eos_supported": True},
        "gpu_placement": {"edge": {"devices": [device], "uuids": [uuid]}, "server": {"devices": [device], "uuids": [uuid]}},
        "launch": {"server_command": "gs_bench.py serve", "edge_command": "gs_bench.py ...", "working_dirs": {}, "env_overrides": {}},
        "authentication": {"mechanism": "none (local TCP, test double)", "credential_env_vars": [], "config_fields": [], "supplied_out_of_band": False},
        "tuning_history": "none (test double)",
        "unknowns": []}


def candidates(config_id="double-cut3", edge_host=EDGE_HOST, server_host=SERVER_HOST, **flags):
    c = {"configuration_id": config_id, "model_dir_name": S.MODEL_DIR_NAME, "edge_host": edge_host, "server_host": server_host,
         "designated_by_implementer": False, "designation_ref": None, "running_at_capture": False, "implementation_default": False}
    c.update(flags)
    return c


def synthetic_tokens(gs_dir, bos=2, eos=(1,), dev_items=None):
    """A tokenized workload with synthetic ids. dev_items=None: the full frozen construction (sizes from the spec);
    otherwise a list of (id, set, group, kind, n, gen, ignore_eos) tuples for a small dev workload."""
    import gs_workload as WL
    items = []
    if dev_items is None:
        corr, timing = WL.load_spec(Path(gs_dir) / "workload")
        for i, it in enumerate(corr["items"]):
            n = 40 + i
            items.append({"id": it["id"], "set": "natural", "group": "natural", "kind": "correctness", "n": n, "gen_tokens": corr["gen_tokens"],
                          "ignore_eos": corr["ignore_eos"], "ids": [bos] + [5 + (i + j) % 90 for j in range(n - 1)]})
        for it in timing["items"]:
            P = it["prompt_tokens"]
            items.append({"id": it["id"], "set": it["set"], "group": it["group"], "kind": it["kind"], "n": P, "gen_tokens": it["gen_tokens"],
                          "ignore_eos": it["ignore_eos"], "ids": [bos] + [6 + (it["start"] + j) % 90 for j in range(P - 1)]})
        spec = {"correctness": C.sha256_file(Path(gs_dir) / "workload" / WL.CORR_FILE), "timing": C.sha256_file(Path(gs_dir) / "workload" / WL.TIMING_FILE)}
    else:
        for k, (iid, set_, group, kind, n, gen, ign) in enumerate(dev_items):
            items.append({"id": iid, "set": set_, "group": group, "kind": kind, "n": n, "gen_tokens": gen, "ignore_eos": ign,
                          "ids": [bos] + [7 + (3 * k + 5 * j) % 90 for j in range(n - 1)]})
        spec = {"correctness": "dev", "timing": "dev"}
    return WL.finalize({"schema": "gemma-split-workload-tokens/1", "utc": "test", "tokenizer_dir": "test", "tokenizer_sha256": {}, "bos_token_id": bos,
                        "eos_token_ids": list(eos), "eos_source": "test", "chat_template_sha256": "test", "spec_sha256": spec, "items": items})


FAKE_CAPS = ["greedy_tokens", "stream_tokens", "boundary_audit", "teacher_forced_logprobs"]


def capabilities(caps, justification=None):
    return {"capabilities": list(caps), "missing_capability_justification": justification or {}, "basis": "test fixture",
            "declared_utc": "2026-11-16T00:00:00Z"}


FAKE_ADAPTER = '''"""Fake target adapter for freeze tests: never executed."""
ADAPTER_INFO = {"contract_version": 1, "implementation": "fake", "configuration_id": "%s",
                "capabilities": ["greedy_tokens", "stream_tokens", "boundary_audit", "teacher_forced_logprobs"],
                "test_double": %s, "notes": "", "missing_capability_justification": {}}


def make_adapter(freeze, role, options):
    raise RuntimeError("fake adapter")
'''


def write_target_capture(gs, model_root, source_root, config_id="cfg-1", test_double=False, adapter=True):
    """A complete, valid Stage-0/1 capture with fake lab facts (protocol tests of stage0 and freeze). adapter=False
    leaves adapter/ empty, so that the capability declaration can be committed first (PREREG §5.3)."""
    import gs_checkpoint as CK
    import gs_deployment as DP
    import gs_split as SP
    import gs_gate as G
    cap = Path(gs) / "capture"
    C.write_json(cap / "edge" / "inventory.json", fake_inventory("edge"))
    C.write_json(cap / "server" / "inventory.json", dict(fake_inventory("server"), env={"CUDA_VISIBLE_DEVICES": SERVER_CVD}))
    import gs_inventory as INV
    C.write_json(cap / "roles.json", INV.roles_record(fake_inventory("edge"), fake_inventory("server")))
    md = fake_model_dir(model_root)
    for r, host in (("edge", EDGE_HOST), ("server", SERVER_HOST)):
        ck = CK.capture(md, r)
        ck["host"] = host
        C.write_json(cap / r / "checkpoint.json", ck)
        dep = DP.capture(r, [source_root])
        dep["host"] = host
        C.write_json(cap / r / "deployment.json", dep)
        C.write_json(cap / r / "snapshot.json", {"role": r, "host": host, "sources": [{"snapshot_root": "/nonexistent", "manifest_sha256": "x"}]})
    sc = double_split_capture(endpoint="10.0.0.1:5000", device="cuda:0", uuid="GPU-a6000-0000", config_id=config_id)
    sc["gpu_placement"]["edge"]["uuids"] = ["GPU-gb10-0000"]
    sc["gpu_placement"]["server"]["CUDA_VISIBLE_DEVICES"] = SERVER_CVD
    C.write_json(cap / "split_capture.json", sc)
    C.write_json(cap / "capabilities.json", capabilities(FAKE_CAPS))
    cands = {"candidates": [candidates(config_id, designated_by_implementer=True, designation_ref="lab_answers.md#q2")]}
    C.write_json(cap / "candidates.json", cands)
    C.write_json(cap / "selected_config.json", SP.select(cands["candidates"], C.read_json(cap / "roles.json")))
    C.write_json(cap / "stage0_verdict.json", G.stage0(cap))
    C.write_json(cap / "workload_tokens.json", synthetic_tokens(gs))
    if adapter:
        write_fake_adapter(gs, config_id, test_double)
    return cap


def write_fake_adapter(gs, config_id="cfg-1", test_double=False):
    (Path(gs) / "adapter" / "gs_adapter_fake.py").write_text(FAKE_ADAPTER % (config_id, "True" if test_double else "False"))


def fake_freeze(caps=tuple(FAKE_CAPS), tier="T1", edge_host=EDGE_HOST, server_host=SERVER_HOST, capture=None, dev=False):
    """A minimal in-memory freeze record for gate/result unit tests (no git)."""
    return {"freeze_sha256": ("d" if dev else "f") * 64, "run_id": ("d" if dev else "f") * 12, "dev": dev,
            "adapter": {"sha256": "a" * 64, "info": {"capabilities": list(caps)}, "tier": tier, "path": "adapter/x.py"},
            "split": {"configuration_id": "cfg-1", "capture": capture or double_split_capture()},
            "checkpoint": {"edge": {"manifest_sha256": "e" * 64}, "server": {"manifest_sha256": "s" * 64}, "summary": {"vocab_size": 106}},
            "roles": roles_for(edge_host, server_host), "workload": {"tokens_file": "capture/workload_tokens.json", "eos_token_ids": [1]}}


def memory_record(freeze, role, what, status, ooms=0, foreign_after=(), failure=None, devices=("cpu",)):
    """A Stage-2 record as gs_memory / gs_bench serve write it (fields the gate reads)."""
    import gs_results as R
    host = SERVER_HOST if role == "server" else EDGE_HOST
    return {"what": what, "role": role, "devices": list(devices), "status": status, "ooms": ooms, "foreign_after": list(foreign_after),
            "failure": failure, "provenance": R.provenance(freeze, "stage2", role, list(devices), "dev" if freeze.get("dev") else "target",
                                                           host=host)}


def stage1_check(rd, freeze, role, match, host, log_host="same"):
    """A Stage-1 workload check as `verify-workload` writes it, with its logged run (WROTE / SAVED: written once).
    log_host: the host whose exit-status log records the run (None: this machine's)."""
    import gs_results as R
    f = R.stage_dir(rd, "stage1") / f"{role}_workload.json"
    if not f.exists():
        C.write_json(f, {"match": match, "provenance": R.provenance(freeze, "stage1", role, [], "target", host=host)})
        log_run(rd, "stage1", role, "verify-workload", rc="0" if match else "2", host=host if log_host == "same" else log_host,
                wrote=wrote_of(rd, "stage1", [f.name]))


def stage2_raw(rd, freeze, status="FITS", match=("edge", "server"), hosts=True):
    """Real target-mode Stage-2 raw files for a record built from write_target_capture (Stage-1 checks, memory records on
    the frozen devices, exit-status events), so that its gate recomputes. Overwrites (test shortcut)."""
    import gs_results as R
    for role in S.ROLES:
        host = SERVER_HOST if role == "server" else EDGE_HOST
        dev = freeze["split"]["capture"]["gpu_placement"][role]["devices"]
        stage1_check(rd, freeze, role, role in match, host, log_host="same" if hosts else None)
        for what in ("split", "reference"):
            C.write_json(R.stage_dir(rd, "stage2") / role / f"{what}_memory.json", memory_record(freeze, role, what, status, devices=dev),
                         overwrite=True)
    log_runs(rd, freeze, "stage2", hosts={"edge": EDGE_HOST, "server": SERVER_HOST} if hosts else None)


_TOKENS = [0]


def token():
    """A unique, increasing run token `<pid>-<ms>` (as gs_bench.logged makes them)."""
    _TOKENS[0] += 1
    return f"{os.getpid()}-{int(time.time() * 1000) + _TOKENS[0]}"


def wrote_of(rd, stage, files):
    """{file: (digest, result)} for the present files (relative to the stage dir), as WROTE events record them."""
    import gs_results as R
    sd = R.stage_dir(rd, stage)
    return {f: (R.file_digest(sd / f), R.file_result(C.read_json(sd / f)) if f.endswith(".json") else None) for f in files if (sd / f).exists()}


def log_run(rd, stage, part, cmd, rc="0", run=True, host=None, out=None, res=None, wrote=None, saved=True):
    """The exit-status events of one command run: START, RUN (measuring began), WROTE and (unless saved=False: the write
    never completed) SAVED per file (`wrote`: {file: (digest, result)}), END unless rc is None (a crash, or an END that
    could not be written)."""
    import gs_results as R
    tok = token()
    R.log_event(rd, "START", tok, stage, part, cmd, host=host)
    if run:
        R.log_event(rd, "RUN", tok, stage, part, cmd, host=host)
    for f, (d, r) in (wrote or {}).items():
        R.log_event(rd, "WROTE", tok, stage, part, cmd, host=host, out=d, res=r, file=f)
        if saved:
            R.log_event(rd, "SAVED", tok, stage, part, cmd, host=host, out=d, file=f)
    if rc is not None:
        R.log_event(rd, "END", tok, stage, part, cmd, rc=rc, host=host, out=out, res=res)
    return tok


def log_runs(rd, freeze, stage, hosts=None, rc="0"):
    """One run (START, RUN, WROTE per file, END unless rc is None) for every writer whose present files no logged run since
    the part's last repeat wrote (what the CLI logs; for tests that write outputs directly). hosts: {role: hostname}."""
    import gs_gate as G
    import gs_results as R
    caps = freeze["adapter"]["info"]["capabilities"]
    for part, spec in R.parts(stage, caps).items():
        used = G.consumed(rd, stage, part)
        for cmd in sorted({R.writer(stage, part, f) for f in spec["own"] + spec["companion"]}):
            present = wrote_of(rd, stage, [f for f in spec["own"] + spec["companion"] if R.writer(stage, part, f) == cmd])
            if not present:
                continue
            wrote = [w for t, w in R.wrote_events(R.read_events(rd), stage, part, cmd).items() if t not in used]
            if any(all(w.get(f, (None,))[0] == d for f, (d, _) in present.items()) for w in wrote):
                continue                        # already accounted for
            log_run(rd, stage, part, cmd, rc=rc, host=(hosts or {}).get("server" if cmd == "serve" else spec["role"]),
                    out=R.outputs_digest(rd, stage, part, cmd, caps) if rc == "0" else None,
                    res=R.outputs_result(rd, stage, part, cmd, caps) if rc == "0" else None, wrote=present)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
