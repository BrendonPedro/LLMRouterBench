"""QWEN-SELECT checkpoints: frozen manifests, verification, download, and required memory from the tensors.

Three public checkpoints are pinned by revision and by a per-file manifest (size, SHA-256, safetensors header):
Qwen3-32B (the manifest K16-HWFIT froze and verified on both machines), Qwen3-14B and Qwen3-8B (built here from the
Hugging Face tree API, the small files and the safetensors headers by byte range; no weight was downloaded to build
them). Standard library only, so it runs on any host before an environment is chosen.

  python -I -B qsel_models.py build  --model 14B            # maintainer, before the tag: writes frozen/manifest_14B.json
  python -I -B qsel_models.py fetch  --model 14B --dest DIR  # any host: resumable download of the pinned files, verified
  python -I -B qsel_models.py verify --model 14B --dir DIR   # size, safetensors header and SHA-256 of every pinned file
  python -I -B qsel_models.py memory                         # the required-memory table
"""
import hashlib
import json
import os
import struct
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # python -I: the script dir is not on sys.path
import qsel_spec as S  # noqa: E402

QS_DIR = Path(__file__).resolve().parent
HF = "https://huggingface.co"
MIB = 1 << 20
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
CUDA_CONTEXT_BYTES = 400 * MIB             # E0's memory model (deployment_e0/e0_memory.py), checked there on an RTX 3090
ALLOCATOR_SLACK_BYTES = 256 * MIB


def sha256_file(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def manifest(model, check=True):
    """(manifest, tensor table) of a pinned model. Both files are checked against qsel_spec.MANIFEST_SHA256."""
    m = S.MODELS[model]
    pm, pt = (QS_DIR / m["manifest"]).resolve(), (QS_DIR / m["tensors"]).resolve()
    if check and (sha256_file(pm), sha256_file(pt)) != tuple(S.MANIFEST_SHA256[model]):
        raise ValueError(f"manifest or tensor table of {model} is not the pinned file")
    man = _read(pm)
    if (man["repo"], man["revision"]) != (m["repo"], m["revision"]):
        raise ValueError(f"manifest of {model} is for {man['repo']}@{man['revision']}")
    return man, _read(pt)


def needed_files(man, table=None, use="edge"):
    """Files a use loads. edge: the small files (tokenizer, EOS set) and every shard. server_whole: the configuration,
    the index and every shard. split_server: the configuration, the index and the shards holding its layers."""
    shards = {f for f in man["files"] if f.endswith(".safetensors")}
    if use == "edge":
        return sorted(set(S.CHECKPOINT_SMALL_FILES) | shards)
    if use == "split_server":
        L, k = S.MODELS[S.SPLIT_MODEL]["n_layers"], S.CUT
        shards = {t["shard"] for n, t in table.items() if n.startswith("model.layers.") and k <= int(n.split(".")[2]) < L - k}
    elif use != "server_whole":
        raise ValueError(use)
    return sorted({"config.json", "model.safetensors.index.json"} | shards)


def read_safetensors_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if not 0 < n < 100 * MIB:
            raise ValueError(f"{path}: implausible safetensors header length {n}")
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return n, h


def verify(model_dir, model, use="edge", cache_file=None, workers=4, deep=True, man=None, table=None):
    """Record of the check. `ok` is True iff every needed file exists with the manifest's size, safetensors header and
    (deep) SHA-256. Hashes are cached per (path, size, mtime) so a repeated preflight does not read the weights again."""
    if man is None:
        man, table = manifest(model)
    d = Path(model_dir)
    cache = _read(cache_file) if cache_file and Path(cache_file).exists() else {}
    rec = {"model": model, "use": use, "repo": man["repo"], "revision": man["revision"], "deep": bool(deep), "files": {}, "problems": []}
    todo, wanted = [], needed_files(man, table, use)
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
                hl, hdr = read_safetensors_header(p)
            except Exception as e:
                rec["problems"].append(f"header of {f}: {type(e).__name__}")
                continue
            frozen = {n: t for n, t in table.items() if t["shard"] == f}
            same = hl == want["header_len"] and set(hdr) == set(frozen) and all(
                hdr[n]["dtype"] == t["dtype"] and hdr[n]["shape"] == t["shape"] and hdr[n]["data_offsets"] == t["offsets"]
                for n, t in frozen.items())
            if not same:
                rec["problems"].append(f"safetensors header of {f} differs from the frozen tensor table")
                continue
        todo.append(f)

    def digest(f):
        st = os.stat(d / f)
        key = f"{os.path.realpath(d / f)}|{st.st_size}|{st.st_mtime_ns}"
        if key not in cache:
            cache[key] = sha256_file(d / f)
        return cache[key]
    if deep:
        with ThreadPoolExecutor(max_workers=workers) as ex:        # hashlib releases the GIL
            for f, h in zip(todo, ex.map(digest, todo)):
                rec["files"][f]["sha256"] = h
                if h != man["files"][f]["sha256"]:
                    rec["problems"].append(f"sha256 of {f} differs from the manifest")
        if cache_file:
            Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
            tmp = Path(str(cache_file) + f".tmp{os.getpid()}")
            tmp.write_text(json.dumps(cache, indent=0))
            os.replace(tmp, cache_file)
    rec["identity_sha256"] = hashlib.sha256(json.dumps({f: man["files"][f]["sha256"] for f in wanted}, sort_keys=True).encode()).hexdigest()
    rec["bytes_verified"] = sum(v["size"] for v in rec["files"].values() if v.get("sha256"))
    rec["ok"] = not rec["problems"] and len(rec["files"]) == len(wanted)
    return rec


def model_config(model_dir):
    c = _read(Path(model_dir) / "config.json")
    return {k: c[k] for k in ("num_hidden_layers", "hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads",
                              "head_dim", "vocab_size")}


def eos_ids(model_dir):
    v = _read(Path(model_dir) / "generation_config.json").get("eos_token_id")
    return sorted(int(x) for x in (v if isinstance(v, list) else [v]))


# --------------------------------------------------------------------------- #
# Required memory, from the tensors that will be loaded                        #
# --------------------------------------------------------------------------- #
def weight_bytes(table):
    """(all bytes, largest tensor bytes, config-free facts) of a tensor table."""
    sizes = [t["nbytes"] for t in table.values()]
    return sum(sizes), max(sizes)


def kv_bytes_per_token_layer(cfg):
    return 2 * cfg["num_key_value_heads"] * cfg["head_dim"] * 2          # K and V, bfloat16


def step_transient_bytes(cfg, n_chunk, ctx):
    """Peak per-layer transient of a forward step of n_chunk tokens at total context ctx, in bfloat16 bytes: E0's
    e0_memory.transient_bytes, restated here so that this module needs no other."""
    H, I = cfg["hidden_size"], cfg["intermediate_size"]
    nh, nkv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    n = n_chunk
    mask = n * (ctx + 1) if (n > 1 and ctx > n) else 0
    attn = 2 * n * H + 4 * n * nh * hd + 2 * n * nkv * hd + 2 * nh * ctx * hd + mask
    mlp = 2 * n * H + 3 * n * I
    return max(attn, mlp) * 2


def whole_need(cfg, table, unified, reference_seq=None):
    """Bytes that must be free to load and run a whole model.
    Generation (timed blocks): a growing KV cache up to CONTEXT_TOKENS, chunked prefill. unified: host and device share
    one pool (the edge), so a host staging buffer and a device copy of the largest tensor both count while loading.
    reference_seq: the reference stage instead. It is K16-HWFIT's formula for its unsplit reference
    (k16_memory.expected("edge_reference")), whose gate held on the edge machine on 18 November: one pass over the whole
    sequence, the attention scores once more in case the kernel materialises them, and one LM-head chunk."""
    w, largest = weight_bytes(table)
    ctx, chunk = S.CONTEXT_TOKENS, S.PREFILL_CHUNK
    if reference_seq:
        L = int(reference_seq)
        kv = cfg["num_hidden_layers"] * kv_bytes_per_token_layer(cfg) * L
        act = int((step_transient_bytes(cfg, L, L) + cfg["num_attention_heads"] * L * L * 2) * S.ACTIVATION_ALLOWANCE)
        act += min(1024, L) * cfg["vocab_size"] * 10 + L * cfg["hidden_size"] * 2
    else:
        kv = 2 * cfg["num_hidden_layers"] * kv_bytes_per_token_layer(cfg) * ctx   # a growing cache is copied when it grows
        act = int(S.ACTIVATION_ALLOWANCE * (step_transient_bytes(cfg, chunk, ctx) + cfg["num_attention_heads"] * chunk * ctx * 2))
    load = (2 if unified else 1) * largest
    c = {"weights": w, "kv": kv, "activations": act, "load_transient": load, "fixed": CUDA_CONTEXT_BYTES + ALLOCATOR_SLACK_BYTES,
         "margin": S.EDGE_MARGIN_BYTES if unified else S.SERVER_MARGIN_BYTES}
    c["required_free"] = sum(c.values())
    return c


def split_need(role, table, cfg):
    """Bytes that must be free for one half of the k = 16 split of the 32B model: K16-HWFIT's formula
    (k16_memory.expected with static=False: one KV cache per owned layer, the reference mode's)."""
    L, k = cfg["num_hidden_layers"], S.CUT

    def owner(name):
        p = name.split(".")
        if len(p) > 2 and p[0] == "model" and p[1] == "layers":
            return "edge" if (int(p[2]) < k or int(p[2]) >= L - k) else "server"
        return "edge"
    mine = [t["nbytes"] for n, t in table.items() if owner(n) == role]
    n_layers = 2 * k if role == "edge" else L - 2 * k
    ctx, chunk = S.CONTEXT_TOKENS, S.PREFILL_CHUNK
    act = int(S.ACTIVATION_ALLOWANCE * max(step_transient_bytes(cfg, chunk, chunk), step_transient_bytes(cfg, 1, ctx)))
    if role == "edge":
        act += min(1024, chunk + S.N_CONT) * cfg["vocab_size"] * 10                  # teacher-forced LM-head chunk of the correctness stage
    c = {"weights": sum(mine), "kv": n_layers * kv_bytes_per_token_layer(cfg) * ctx, "activations": act,
         "load_transient": (2 if role == "edge" else 1) * max(mine), "wire": chunk * cfg["hidden_size"] * 2,
         "fixed": CUDA_CONTEXT_BYTES + ALLOCATOR_SLACK_BYTES, "margin": S.EDGE_MARGIN_BYTES if role == "edge" else S.SERVER_MARGIN_BYTES}
    c["required_free"] = sum(c.values())
    return c


def config_of(model, table):
    """The architecture numbers the memory model needs, read from the frozen tensor table (no checkpoint needed)."""
    L = S.MODELS[model]["n_layers"]
    emb = table["model.embed_tokens.weight"]["shape"]
    q = table["model.layers.0.self_attn.q_proj.weight"]["shape"]
    kq = table["model.layers.0.self_attn.k_proj.weight"]["shape"]
    up = table["model.layers.0.mlp.up_proj.weight"]["shape"]
    head_dim = table["model.layers.0.self_attn.q_norm.weight"]["shape"][0]
    return {"num_hidden_layers": L, "hidden_size": emb[1], "vocab_size": emb[0], "intermediate_size": up[0], "head_dim": head_dim,
            "num_attention_heads": q[0] // head_dim, "num_key_value_heads": kq[0] // head_dim}


def needs(reference_seq=None, check=True):
    """Required free bytes of every load of the experiment, from the frozen tensor tables."""
    out = {}
    for model in S.MODELS:
        _, table = manifest(model, check)
        cfg = config_of(model, table)
        out[f"edge_whole_{model}"] = whole_need(cfg, table, unified=True)
        out[f"server_whole_{model}"] = whole_need(cfg, table, unified=False)
        if model == S.SPLIT_MODEL:
            out["edge_reference"] = whole_need(cfg, table, unified=True, reference_seq=reference_seq or (S.PREFILL_CHUNK + S.N_CONT))
            out["split_edge"] = split_need("edge", table, cfg)
            out["split_server"] = split_need("server", table, cfg)
    return out


LOAD_OF = {"S": "split_edge", "E32": "edge_whole_32B", "E14": "edge_whole_14B", "E8": "edge_whole_8B", "reference": "edge_reference",
           "serve": "split_server", "serve-remote": "server_whole_14B"}


# --------------------------------------------------------------------------- #
# Building a manifest (maintainer) and fetching a checkpoint (any host)        #
# --------------------------------------------------------------------------- #
def _get(url, headers=None, timeout=60):
    req = urllib.request.Request(url, headers=dict({"User-Agent": "qsel-models/1"}, **(headers or {})))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def build(model):
    """Manifest and tensor table of a pinned revision from Hugging Face metadata. No weight data is downloaded: sizes
    and SHA-256 of the shards are the LFS object ids, headers are read by byte range, small files are fetched whole."""
    m = S.MODELS[model]
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
                nb = DTYPE_BYTES[t["dtype"]]
                for dim in t["shape"]:
                    nb *= dim
                table[name] = {"shard": f, "dtype": t["dtype"], "shape": t["shape"], "offsets": t["data_offsets"], "nbytes": nb}
        elif f in S.CHECKPOINT_SMALL_FILES:
            body = _get(f"{HF}/{repo}/resolve/{rev}/{f}", timeout=300)
            if e.get("lfs") and hashlib.sha256(body).hexdigest() != e["lfs"]["oid"]:
                raise ValueError(f"{f}: content does not match its LFS object id")
            if not e.get("lfs") and hashlib.sha1(b"blob %d\0" % len(body) + body).hexdigest() != e["oid"]:
                raise ValueError(f"{f}: content does not match its git blob id")
            files[f] = {"size": len(body), "sha256": hashlib.sha256(body).hexdigest(), "source": "content fetched at the pinned revision; object id verified"}
    missing = [f for f in S.CHECKPOINT_SMALL_FILES if f not in files]
    if missing or not table:
        raise ValueError(f"{repo}@{rev}: missing {missing or 'weight shards'}")
    man = {"repo": repo, "revision": rev, "built_utc": time.strftime("%Y-%m-%d", time.gmtime()),
           "built_from": "huggingface.co tree API, small files and safetensors headers (byte ranges); no weight data was downloaded",
           "total_tensor_bytes": sum(t["nbytes"] for t in table.values()), "n_tensors": len(table), "files": files}
    for rel, obj in ((m["manifest"], man), (m["tensors"], table)):
        p = QS_DIR / rel
        if p.exists():
            raise FileExistsError(f"{p} exists: a frozen manifest is never replaced")
        p.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return man


def fetch(model, dest, retries=5):
    """Download every pinned file of `model` into `dest` (resumable), then verify. Nothing is written elsewhere."""
    man, table = manifest(model)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for f in needed_files(man, table):
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
                    req = urllib.request.Request(url, headers={"User-Agent": "qsel-models/1", "Range": f"bytes={have}-"})
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


def memory_markdown(check=True):
    n = needs(check=check)
    g = lambda b: f"{b / 2**30:,.2f}"  # noqa: E731
    rows = ["| load | weights GiB | KV GiB | activations GiB | loading GiB | required free GiB |", "|---|---|---|---|---|---|"]
    for k in ("split_edge", "split_server", "edge_reference", "edge_whole_32B", "edge_whole_14B", "edge_whole_8B", "server_whole_14B", "server_whole_32B"):
        c = n[k]
        rows.append(f"| {k} | {g(c['weights'])} | {g(c['kv'])} | {g(c['activations'])} | {g(c['load_transient'])} | {g(c['required_free'])} |")
    return "\n".join(rows) + "\n"


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "fetch", "verify"):
        p = sp.add_parser(name)
        p.add_argument("--model", required=True, choices=sorted(S.MODELS))
        if name == "fetch":
            p.add_argument("--dest", required=True)
        if name == "verify":
            p.add_argument("--dir", required=True)
            p.add_argument("--cache")
            p.add_argument("--use", default="edge", choices=["edge", "server_whole", "split_server"])
    sp.add_parser("memory")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        man = build(a.model)
        print(a.model, man["n_tensors"], "tensors,", man["total_tensor_bytes"], "bytes")
    elif a.cmd == "fetch":
        sys.exit(0 if fetch(a.model, a.dest)["ok"] else 3)
    elif a.cmd == "verify":
        rec = verify(a.dir, a.model, a.use, a.cache)
        print(json.dumps({k: rec[k] for k in ("model", "revision", "ok", "problems", "bytes_verified", "identity_sha256")}, indent=1))
        sys.exit(0 if rec["ok"] else 3)
    else:
        print(memory_markdown())


if __name__ == "__main__":
    main()
