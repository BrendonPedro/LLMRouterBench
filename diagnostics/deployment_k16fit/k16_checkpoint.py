"""K16-HWFIT checkpoint identity: locate the local Qwen3-32B checkpoint and verify it against the frozen manifest.

frozen/checkpoint_manifest.json pins every file of Qwen/Qwen3-32B at the E0 revision by size and SHA-256 (the
Hugging Face LFS object ids). Both hosts verify the files they load against it, so "same checkpoint on both
hosts" is checked on each host without moving data between them. Per file: size, safetensors header equal to
frozen/tensor_table.json (names, dtypes, shapes, offsets), then SHA-256 of the content. Hashes are cached per
(path, size, mtime) so a rerun does not read 65 GB again. Nothing is downloaded and nothing is written into the
checkpoint directory.

  python -I k16_checkpoint.py --role server|edge [--model-dir DIR] [--out FILE]
"""
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # python -I: the script dir is not on sys.path
import k16_spec as S  # noqa: E402
import k16_memory as KM  # noqa: E402
from k16_common import read_json, write_json, utc  # noqa: E402

SMALL_FILES = {"server": ("config.json", "model.safetensors.index.json"),
               "edge": ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json")}


def manifest():
    return read_json(S.K16_DIR / "frozen" / "checkpoint_manifest.json")


def needed_files(role, table=None):
    """Files a role loads. The edge also computes the unsplit reference, so it needs every shard."""
    table = table or KM.tensor_table()
    names = table if role == "edge" else KM.role_tensors("server", table)
    return sorted(set(SMALL_FILES[role]) | {table[n]["shard"] for n in names})


def locate(explicit=None):
    """The checkpoint directory: --model-dir / K16_MODEL_DIR, else a known lab location (k16_spec.KNOWN_MODEL_DIRS),
    else the Hugging Face cache snapshot of the pinned revision, else a bounded search. Returns (path or None, [places
    looked]). Whatever is found is then verified against the manifest; nothing is written into it."""
    tried = []
    for cand in (explicit, os.environ.get("K16_MODEL_DIR")):
        if cand:
            tried.append(cand)
            return (cand if (Path(cand) / "config.json").exists() else None), tried
    for cand in S.KNOWN_MODEL_DIRS:
        tried.append(cand)
        if (Path(cand) / "config.json").exists():
            return cand, tried
    hubs = [os.environ.get("HF_HUB_CACHE"), os.environ.get("HUGGINGFACE_HUB_CACHE"),
            os.path.join(os.environ["HF_HOME"], "hub") if os.environ.get("HF_HOME") else None,
            os.path.expanduser("~/.cache/huggingface/hub")]
    for hub in [h for h in hubs if h]:
        snap = Path(hub) / ("models--" + S.E0.SPLIT_REPO.replace("/", "--")) / "snapshots" / S.E0.SPLIT_REVISION
        tried.append(str(snap))
        if (snap / "config.json").exists():
            return str(snap), tried
    # last resort: a directory named like the model whose config.json is byte-identical to the pinned revision's
    want = manifest()["files"]["config.json"]["sha256"]
    roots = [r for r in (os.path.expanduser("~"), "/data", "/models", "/raid", "/workspace", "/srv", "/opt", "/mnt") if os.path.isdir(r)]
    name = S.E0.SPLIT_REPO.split("/")[1]
    tried.append(f"find {' '.join(roots)} -maxdepth 7 -path '*{name}*' -name config.json (40 s per root)")
    out = []
    for root in roots:                      # bounded: a slow or huge mount costs at most 40 s and keeps what it found
        try:
            r = subprocess.run(["find", root, "-maxdepth", "7", "-path", f"*{name}*", "-name", "config.json"], capture_output=True, text=True, timeout=40)
            out += r.stdout.splitlines()
        except subprocess.TimeoutExpired as e:
            part = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            out += [x for x in part.splitlines() if x.endswith("config.json")]
        except Exception:
            continue
    for f in sorted(out):
        try:
            if sha256_file(f) == want and (Path(f).parent / "model.safetensors.index.json").exists():
                return str(Path(f).parent), tried
        except OSError:
            continue
    return None, tried


def sha256_file(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def _hash_cached(path, cache):
    st = os.stat(path)
    key = f"{os.path.realpath(path)}|{st.st_size}|{st.st_mtime_ns}"
    if key not in cache:
        cache[key] = sha256_file(path)
    return cache[key]


def verify(model_dir, role, cache_file=None, dev=False, workers=4, man=None, table=None):
    """Record of the check. `ok` is True iff every needed file exists with the manifest's size, header and SHA-256.
    dev=True (tiny test checkpoints): hashes whatever is there and compares with nothing."""
    d = Path(model_dir)
    cache = read_json(cache_file) if cache_file and Path(cache_file).exists() else {}
    rec = {"role": role, "model_dir": str(d), "realpath": os.path.realpath(d), "utc": utc(), "dev": dev, "files": {}, "problems": []}
    if dev:
        files = sorted(p.name for p in d.iterdir() if p.is_file())
        for f in files:
            rec["files"][f] = {"size": (d / f).stat().st_size, "sha256": _hash_cached(d / f, cache)}
    else:
        man = man or manifest()
        table = table or KM.tensor_table()
        files = needed_files(role, table)
        todo = []
        for f in files:
            want, p = man["files"][f], d / f
            if not p.exists():
                rec["problems"].append(f"missing: {f}")
                continue
            size = p.stat().st_size
            rec["files"][f] = {"size": size, "expected_sha256": want["sha256"]}
            if size != want["size"]:
                rec["problems"].append(f"size of {f}: {size} != {want['size']}")
                continue
            if f.endswith(".safetensors"):
                try:
                    hl, hdr = KM.read_safetensors_header(p)
                except Exception as e:
                    rec["problems"].append(f"header of {f}: {type(e).__name__}: {e}")
                    continue
                frozen = {n: t for n, t in table.items() if t["shard"] == f}
                same = hl == want["header_len"] and set(hdr) == set(frozen) and all(
                    hdr[n]["dtype"] == t["dtype"] and hdr[n]["shape"] == t["shape"] and hdr[n]["data_offsets"] == t["offsets"]
                    for n, t in frozen.items())
                rec["files"][f]["header_matches_tensor_table"] = bool(same)
                if not same:
                    rec["problems"].append(f"safetensors header of {f} differs from frozen/tensor_table.json")
                    continue
            todo.append(f)
        with ThreadPoolExecutor(max_workers=workers) as ex:        # hashlib releases the GIL
            for f, h in zip(todo, ex.map(lambda f: _hash_cached(d / f, cache), todo)):
                rec["files"][f]["sha256"] = h
                if h != man["files"][f]["sha256"]:
                    rec["problems"].append(f"sha256 of {f}: {h} != {man['files'][f]['sha256']}")
        rec["manifest_sha256"] = S.sha256_file(S.K16_DIR / "frozen" / "checkpoint_manifest.json")
        rec["repo"], rec["revision"] = man["repo"], man["revision"]
    if cache_file:
        write_json(cache_file, cache)
    rec["identity_sha256"] = hashlib.sha256(S.canonical({f: v.get("sha256") for f, v in rec["files"].items()}).encode()).hexdigest()
    rec["bytes_verified"] = sum(v["size"] for v in rec["files"].values() if v.get("sha256"))
    rec["ok"] = not rec["problems"] and bool(rec["files"])
    return rec


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role", choices=["server", "edge"], required=True)
    ap.add_argument("--model-dir", default=None); ap.add_argument("--out", default=None); ap.add_argument("--cache", default=None)
    a = ap.parse_args(argv)
    d, tried = locate(a.model_dir)
    if d is None:
        print(f"checkpoint not found; looked in {tried}. Set K16_MODEL_DIR.", file=sys.stderr); sys.exit(3)
    rec = verify(d, a.role, a.cache)
    if a.out:
        write_json(a.out, rec)
    print(json.dumps({k: rec[k] for k in ("model_dir", "ok", "problems", "identity_sha256", "bytes_verified")}, indent=1))
    sys.exit(0 if rec["ok"] else 3)


if __name__ == "__main__":
    main()
