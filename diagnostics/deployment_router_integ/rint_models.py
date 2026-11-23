"""ROUTER-INTEG checkpoints: frozen manifests, verification, download, and required memory.

Two public checkpoints, pinned by revision and by a per-file manifest (size, SHA-256, safetensors header):
  QW  Qwen/Qwen3-8B                           the manifest QWEN-SELECT froze and verified on both hosts (copied, hash-checked)
  DS  deepseek-ai/DeepSeek-R1-0528-Qwen3-8B    built here from the Hugging Face tree API, the small files and the
                                               safetensors headers by byte range; no weight was downloaded to build it
The hashing, header and memory formulas are the reused module's (qsel_models), imported unchanged. This module adds
only what differs for this pair: the DeepSeek checkpoint has no generation_config.json and two shards, and both models
are held by one process at the same time.

  python -I -B rint_models.py build                          # maintainer, before the freeze
  python -I -B rint_models.py fetch  --model DS --dest DIR   # any host: resumable download of the pinned files, verified
  python -I -B rint_models.py verify --model DS --dir DIR
  python -I -B rint_models.py memory
"""
import hashlib
import json
import os
import struct
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rint_common as C  # noqa: E402
import rint_spec as S  # noqa: E402

HF = "https://huggingface.co"
UA = {"User-Agent": "checkpoint-fetch/1"}


def manifest(model):
    m = S.MODELS[model]
    man, table = C.read_json(C.RI_DIR / m["manifest"]), C.read_json(C.RI_DIR / m["tensors"])
    if (man["repo"], man["revision"]) != (m["repo"], m["revision"]):
        raise ValueError(f"manifest of {model} is for {man['repo']}@{man['revision']}")
    if model == S.QW and (C.sha256_file(C.RI_DIR / m["manifest"]), C.sha256_file(C.RI_DIR / m["tensors"])) != tuple(S.QW_MANIFEST_SHA256):
        raise ValueError("the Qwen3-8B manifest is not the one QWEN-SELECT froze")
    return man, table


def needed_files(model, man):
    return sorted(set(S.MODELS[model]["small_files"]) | {f for f in man["files"] if f.endswith(".safetensors")})


def verify(model_dir, model, cache_file=None, deep=True):
    """Record of the check: every pinned file present with the manifest's size, safetensors header and SHA-256."""
    qm = C.reused()["qm"]
    man, table = manifest(model)
    d = Path(model_dir)
    cache = C.read_json(cache_file) if cache_file and Path(cache_file).exists() else {}
    rec = {"model": model, "repo": man["repo"], "revision": man["revision"], "deep": bool(deep), "files": {}, "problems": []}
    wanted = needed_files(model, man)
    for f in wanted:
        want, p = man["files"][f], d / f
        if not p.is_file():
            rec["problems"].append(f"missing: {f}")
            continue
        size = p.stat().st_size
        rec["files"][f] = {"size": size}
        if size != want["size"]:
            rec["problems"].append(f"size of {f}: {size} != {want['size']}")
            continue
        if f.endswith(".safetensors"):
            try:
                hl, hdr = qm.read_safetensors_header(p)
            except Exception as e:
                rec["problems"].append(f"header of {f}: {type(e).__name__}")
                continue
            frozen = {n: t for n, t in table.items() if t["shard"] == f}
            if not (hl == want["header_len"] and set(hdr) == set(frozen) and all(
                    hdr[n]["dtype"] == t["dtype"] and hdr[n]["shape"] == t["shape"] and hdr[n]["data_offsets"] == t["offsets"]
                    for n, t in frozen.items())):
                rec["problems"].append(f"safetensors header of {f} differs from the frozen tensor table")
                continue
        if deep:
            st = os.stat(p)
            key = f"{os.path.realpath(p)}|{st.st_size}|{st.st_mtime_ns}"
            if key not in cache:
                cache[key] = qm.sha256_file(p)
            rec["files"][f]["sha256"] = cache[key]
            if cache[key] != want["sha256"]:
                rec["problems"].append(f"sha256 of {f} differs from the manifest")
    if deep and cache_file:
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(cache_file) + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(cache, indent=0))
        os.replace(tmp, cache_file)
    rec["identity_sha256"] = hashlib.sha256(json.dumps({f: man["files"][f]["sha256"] for f in wanted}, sort_keys=True).encode()).hexdigest()
    rec["bytes_verified"] = sum(v["size"] for v in rec["files"].values() if v.get("sha256"))
    rec["ok"] = not rec["problems"] and len(rec["files"]) == len(wanted)
    return rec


def model_id(model, identity=None):
    """The exact model identifier written into every record."""
    m = S.MODELS[model]
    return {"key": model, "name": m["name"], "repo": m["repo"], "revision": m["revision"], "checkpoint_identity_sha256": identity,
            "precision": "bfloat16 as published", "quantisation": "none"}


# --------------------------------------------------------------------------- #
# Required memory                                                              #
# --------------------------------------------------------------------------- #
def arch(model):
    """Architecture numbers from the frozen tensor table (no checkpoint needed)."""
    _, table = manifest(model)
    emb = table["model.embed_tokens.weight"]["shape"]
    q = table["model.layers.0.self_attn.q_proj.weight"]["shape"]
    kq = table["model.layers.0.self_attn.k_proj.weight"]["shape"]
    up = table["model.layers.0.mlp.up_proj.weight"]["shape"]
    head_dim = table["model.layers.0.self_attn.q_norm.weight"]["shape"][0]
    n_layers = 1 + max(int(n.split(".")[2]) for n in table if n.startswith("model.layers."))
    return {"num_hidden_layers": n_layers, "hidden_size": emb[1], "vocab_size": emb[0], "intermediate_size": up[0], "head_dim": head_dim,
            "num_attention_heads": q[0] // head_dim, "num_key_value_heads": kq[0] // head_dim}


def max_context():
    """The longest context a request can reach: the output cap plus the longest prompt of the frozen workload."""
    wl = C.read_json(C.RI_DIR / "frozen" / "workload.json")
    return min(S.CONTEXT_TOKENS, S.MAX_NEW_TOKENS + int(wl["max_prompt_tokens"]))


def need(role, models=S.MODEL_ORDER, ctx=None):
    """Bytes that must be free to hold `models` in one process and generate with one of them at a time.
    QWEN-SELECT's formula (qsel_models.whole_need), restated for several models: all weights; one growing KV cache at
    the longest reachable context, counted twice because a growing cache is copied when it grows; the activations of
    one step (a prefill over the longest prompt, or one decode step at the longest context, whichever is larger); the loading transient of the largest tensor (twice on the edge, whose host and device share one pool);
    the fixed allowances; the margin."""
    qm = C.reused()["qm"]
    ctx = int(ctx or max_context())
    n_pre = min(S.PREFILL_CHUNK, int(C.read_json(C.RI_DIR / "frozen" / "workload.json")["max_prompt_tokens"]))
    unified = role == "edge"
    w = kv = act = largest = 0
    for m in models:
        _, table = manifest(m)
        cfg = arch(m)
        wm, lm = qm.weight_bytes(table)
        w, largest = w + wm, max(largest, lm)
        kv = max(kv, 2 * cfg["num_hidden_layers"] * qm.kv_bytes_per_token_layer(cfg) * ctx)
        act = max(act, int(S.ACTIVATION_ALLOWANCE * max(
            qm.step_transient_bytes(cfg, n_pre, n_pre) + cfg["num_attention_heads"] * n_pre * n_pre * 2,   # prefill: only over the prompt
            qm.step_transient_bytes(cfg, 1, ctx) + cfg["vocab_size"] * 12)))                              # a decode step and its logits
    c = {"weights": w, "kv": kv, "activations": act, "load_transient": (2 if unified else 1) * largest,
         "fixed": qm.CUDA_CONTEXT_BYTES + qm.ALLOCATOR_SLACK_BYTES, "margin": S.EDGE_MARGIN_BYTES if unified else S.SERVER_MARGIN_BYTES,
         "context_tokens": ctx, "models": list(models)}
    c["required_free"] = sum(v for k, v in c.items() if k not in ("context_tokens", "models"))
    return c


# --------------------------------------------------------------------------- #
# Building the manifests (maintainer) and fetching a checkpoint (any host)     #
# --------------------------------------------------------------------------- #
def _get(url, headers=None, timeout=120):
    req = urllib.request.Request(url, headers=dict(UA, **(headers or {})))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def build_ds():
    """Manifest and tensor table of the pinned DeepSeek revision from Hugging Face metadata. No weight data is
    downloaded: sizes and SHA-256 of the shards are the LFS object ids, headers are read by byte range."""
    qm = C.reused()["qm"]
    m = S.MODELS[S.DS]
    repo, rev = m["repo"], m["revision"]
    tree = json.loads(_get(f"{HF}/api/models/{repo}/tree/{rev}?recursive=true"))
    files, table = {}, {}
    for e in sorted(tree, key=lambda e: e["path"]):
        f = e["path"]
        if f.endswith(".safetensors"):
            url = f"{HF}/{repo}/resolve/{rev}/{f}"
            n = struct.unpack("<Q", _get(url, {"Range": "bytes=0-7"}))[0]
            hdr = json.loads(_get(url, {"Range": f"bytes=8-{7 + n}"}))
            hdr.pop("__metadata__", None)
            files[f] = {"size": e["lfs"]["size"], "sha256": e["lfs"]["oid"], "source": "Hugging Face LFS object id (sha256)", "header_len": n}
            for name, t in hdr.items():
                nb = qm.DTYPE_BYTES[t["dtype"]]
                for dim in t["shape"]:
                    nb *= dim
                table[name] = {"shard": f, "dtype": t["dtype"], "shape": t["shape"], "offsets": t["data_offsets"], "nbytes": nb}
        elif f in m["small_files"]:
            body = _get(f"{HF}/{repo}/resolve/{rev}/{f}", timeout=300)
            if e.get("lfs") and hashlib.sha256(body).hexdigest() != e["lfs"]["oid"]:
                raise ValueError(f"{f}: content does not match its LFS object id")
            if not e.get("lfs") and hashlib.sha1(b"blob %d\0" % len(body) + body).hexdigest() != e["oid"]:
                raise ValueError(f"{f}: content does not match its git blob id")
            files[f] = {"size": len(body), "sha256": hashlib.sha256(body).hexdigest(), "source": "content fetched at the pinned revision; object id verified"}
    missing = [f for f in m["small_files"] if f not in files]
    if missing or not table:
        raise ValueError(f"{repo}@{rev}: missing {missing or 'weight shards'}")
    man = {"repo": repo, "revision": rev, "built_utc": time.strftime("%Y-%m-%d", time.gmtime()),
           "built_from": "huggingface.co tree API, small files and safetensors headers (byte ranges); no weight data was downloaded",
           "total_tensor_bytes": sum(t["nbytes"] for t in table.values()), "n_tensors": len(table), "files": files}
    for rel, obj in ((m["manifest"], man), (m["tensors"], table)):
        p = C.RI_DIR / rel
        if p.exists():
            raise FileExistsError(f"{p} exists: a frozen manifest is never replaced")
        p.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return man


def copy_qw():
    """The Qwen3-8B manifest and tensor table, byte for byte from the reused clone, checked against the pinned hashes."""
    src = C.reused()["dir"] / "frozen"
    m = S.MODELS[S.QW]
    for name, rel, want in (("manifest_8B.json", m["manifest"], S.QW_MANIFEST_SHA256[0]), ("tensors_8B.json", m["tensors"], S.QW_MANIFEST_SHA256[1])):
        body = (src / name).read_bytes()
        if hashlib.sha256(body).hexdigest() != want:
            raise ValueError(f"{name} in the reused clone is not the pinned file")
        p = C.RI_DIR / rel
        if p.exists():
            raise FileExistsError(f"{p} exists: a frozen manifest is never replaced")
        p.write_bytes(body)


def fetch(model, dest, retries=5):
    man, _ = manifest(model)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for f in needed_files(model, man):
        want, p, part = man["files"][f], dest / f, dest / (f + ".part")
        if p.is_file() and p.stat().st_size == want["size"]:
            continue
        url = f"{HF}/{man['repo']}/resolve/{man['revision']}/{f}"
        for attempt in range(retries):
            have = part.stat().st_size if part.exists() else 0
            if have > want["size"]:
                part.unlink()
                have = 0
            try:
                if have < want["size"]:
                    req = urllib.request.Request(url, headers=dict(UA, Range=f"bytes={have}-"))
                    with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab") as out:
                        if have and r.status != 206:
                            raise OSError("server ignored the byte range")
                        for b in iter(lambda: r.read(1 << 22), b""):
                            out.write(b)
                if part.stat().st_size == want["size"]:
                    os.replace(part, p)
                    break
            except OSError as e:
                print(f"  {f}: {type(e).__name__}, retry {attempt + 1}/{retries}", flush=True)
                time.sleep(5 * (attempt + 1))
        else:
            raise SystemExit(f"STOP: could not download {f}")
        print(f"  {f}: {want['size']:,} bytes", flush=True)
    rec = verify(dest, model)
    print(json.dumps({k: rec[k] for k in ("model", "revision", "ok", "problems", "bytes_verified")}, indent=1))
    return rec


def memory_markdown():
    g = lambda b: f"{b / 2**30:,.2f}"  # noqa: E731
    rows = ["| load | context tokens | weights GiB | KV GiB | activations GiB | loading GiB | required free GiB |", "|---|---|---|---|---|---|---|"]
    for label, role, models in (("edge: both models", "edge", S.MODEL_ORDER), ("server: both models", "server", S.MODEL_ORDER),
                                ("server: one model", "server", (S.DS,))):
        c = need(role, models)
        rows.append(f"| {label} | {c['context_tokens']:,} | {g(c['weights'])} | {g(c['kv'])} | {g(c['activations'])} | {g(c['load_transient'])} | {g(c['required_free'])} |")
    return "\n".join(rows) + "\n"


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("build")
    for name in ("fetch", "verify"):
        p = sp.add_parser(name)
        p.add_argument("--model", required=True, choices=list(S.MODEL_ORDER))
        p.add_argument("--dest" if name == "fetch" else "--dir", required=True)
        if name == "verify":
            p.add_argument("--cache")
    sp.add_parser("memory")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        copy_qw()
        man = build_ds()
        print("DS", man["n_tensors"], "tensors,", man["total_tensor_bytes"], "bytes")
    elif a.cmd == "fetch":
        sys.exit(0 if fetch(a.model, a.dest)["ok"] else 3)
    elif a.cmd == "verify":
        rec = verify(a.dir, a.model, a.cache)
        print(json.dumps({k: rec[k] for k in ("model", "revision", "ok", "problems", "bytes_verified", "identity_sha256")}, indent=1))
        sys.exit(0 if rec["ok"] else 3)
    else:
        print(memory_markdown())


if __name__ == "__main__":
    main()
