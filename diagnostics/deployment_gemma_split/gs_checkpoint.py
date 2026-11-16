"""GEMMA-SPLIT Stage 0c: model / checkpoint identity capture (PREREG §4.2). Reads files; never loads weights.

The identity of the model is the SHA-256 manifest of the checkpoint directory each host actually uses (every
file: path, size, sha256), not a name or a revision string. A Hugging Face revision is recorded when the
directory reveals one (cache snapshot path or download metadata), and is never assumed. config.json is summarised
and the architecture features that matter for a split are derived from it: per-layer input embeddings, KV sharing
between layers, mixed attention types, logit softcapping, tied embeddings, MoE, multimodal towers, quantisation.

  python -I gs_checkpoint.py capture --role edge|server --model-dir <dir> --out capture/<role>/checkpoint.json
  python -I gs_checkpoint.py compare --edge capture/edge/checkpoint.json --server capture/server/checkpoint.json
"""
import os
import re
import socket
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # python -I: the script dir is not on sys.path
import gs_common as C  # noqa: E402
import gs_spec as S  # noqa: E402

EXCLUDED_DIRS = (".git", ".cache")          # VCS objects and download caches are not model content
TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json",
                   "chat_template.jinja", "chat_template.json", "generation_config.json", "processor_config.json",
                   "preprocessor_config.json", "config.json")


def manifest(model_dir):
    root = Path(model_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"model directory {root} does not exist")
    entries = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDED_DIRS)
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            st = p.stat()
            entries.append({"path": p.relative_to(root).as_posix(), "size": st.st_size, "sha256": C.sha256_file(p),
                            "mtime_ns": st.st_mtime_ns, "symlink": os.readlink(p) if p.is_symlink() else None})
    entries.sort(key=lambda e: e["path"])
    return entries


def manifest_sha256(entries):
    return C.sha256_obj([[e["path"], e["size"], e["sha256"]] for e in entries])


def summarize_config(cfg):
    text = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg

    def g(k):
        return text.get(k, cfg.get(k))

    lt = g("layer_types")
    return {
        "architectures": cfg.get("architectures"), "model_type": cfg.get("model_type"), "text_model_type": text.get("model_type"),
        "dtype": cfg.get("dtype") or cfg.get("torch_dtype") or text.get("dtype") or text.get("torch_dtype"),
        "num_hidden_layers": g("num_hidden_layers"), "hidden_size": g("hidden_size"), "intermediate_size": g("intermediate_size"),
        "num_attention_heads": g("num_attention_heads"), "num_key_value_heads": g("num_key_value_heads"), "head_dim": g("head_dim"),
        "vocab_size": g("vocab_size"), "max_position_embeddings": g("max_position_embeddings"), "sliding_window": g("sliding_window"),
        "layer_types": lt, "layer_type_counts": dict(Counter(lt)) if lt else None,
        "num_kv_shared_layers": g("num_kv_shared_layers"), "hidden_size_per_layer_input": g("hidden_size_per_layer_input"),
        "vocab_size_per_layer_input": g("vocab_size_per_layer_input"), "final_logit_softcapping": g("final_logit_softcapping"),
        "tie_word_embeddings": cfg.get("tie_word_embeddings", text.get("tie_word_embeddings")),   # None = absent (class default)
        "enable_moe_block": g("enable_moe_block"), "num_experts": g("num_experts"),
        "quantization_config": cfg.get("quantization_config") or text.get("quantization_config"),
        "has_vision": cfg.get("vision_config") is not None, "has_audio": cfg.get("audio_config") is not None,
        "eos_token_id": cfg.get("eos_token_id", text.get("eos_token_id")), "bos_token_id": cfg.get("bos_token_id", text.get("bos_token_id")),
    }


def features(summary):
    lt = summary.get("layer_types") or []
    return {"per_layer_embeddings": bool(summary.get("hidden_size_per_layer_input")),
            "kv_sharing": bool(summary.get("num_kv_shared_layers")), "mixed_attention": len(set(lt)) > 1,
            "sliding_window": summary.get("sliding_window") is not None and "sliding_attention" in lt,
            "logit_softcapping": summary.get("final_logit_softcapping") is not None,
            "tied_embeddings": summary.get("tie_word_embeddings"), "moe": bool(summary.get("enable_moe_block")),
            "multimodal": bool(summary.get("has_vision") or summary.get("has_audio")),
            "quantized": summary.get("quantization_config") is not None}


def kv_sharing_map(summary):
    """{consumer layer: source layer} as transformers' Gemma 4 modules define KV sharing (the last num_kv_shared_layers
    layers reuse the KV of the last earlier non-shared layer of the same attention type). Derived for validation of the
    split capture only; the capture must confirm or correct it for the implementation under test (PREREG §4.4)."""
    L, n, lt = summary.get("num_hidden_layers"), summary.get("num_kv_shared_layers") or 0, summary.get("layer_types")
    if not L or n <= 0 or not lt or len(lt) != L:
        return {}
    first = L - n
    out = {}
    for i in range(first, L):
        prev = [j for j in range(first) if lt[j] == lt[i]]
        out[i] = prev[-1] if prev else None
    return out


def hf_revision(model_dir):
    real = Path(os.path.realpath(model_dir))
    parts = real.parts
    if "snapshots" in parts:
        i = parts.index("snapshots")
        if i + 1 < len(parts) and i >= 1 and parts[i - 1].startswith("models--"):
            return {"source": "hf cache snapshot path", "repo": parts[i - 1][len("models--"):].replace("--", "/"), "revision": parts[i + 1]}
    meta = Path(model_dir) / ".cache" / "huggingface" / "download"
    commits = set()
    if meta.is_dir():
        for f in meta.rglob("*.metadata"):
            first = f.read_text(errors="replace").splitlines()[:1]
            if first and re.fullmatch(r"[0-9a-f]{40}", first[0].strip()):
                commits.add(first[0].strip())
    if commits:
        return {"source": "huggingface download metadata", "repo": None, "revision": sorted(commits)[0] if len(commits) == 1 else sorted(commits)}
    return {"source": "not determinable from the directory", "repo": None, "revision": None}


def capture(model_dir, role):
    if role not in S.ROLES:
        raise ValueError(f"role must be one of {S.ROLES}")
    root = Path(model_dir).resolve()
    entries = manifest(root)
    cfg_path = root / "config.json"
    cfg = C.read_json(cfg_path) if cfg_path.exists() else None
    summary = summarize_config(cfg) if cfg else None
    gen = root / "generation_config.json"
    return {"schema": "gemma-split-checkpoint/1", "experiment": S.NAME, "role": role, "host": socket.gethostname(), "utc": C.utc(),
            "model_dir": str(root), "dir_name": root.name, "dir_name_matches": root.name == S.MODEL_DIR_NAME,
            "excluded_dirs": list(EXCLUDED_DIRS), "manifest": entries, "manifest_sha256": manifest_sha256(entries),
            "n_files": len(entries), "total_bytes": sum(e["size"] for e in entries), "hf_revision": hf_revision(root),
            "config": cfg, "summary": summary, "features": features(summary) if summary else None,
            "kv_sharing_map": {str(k): v for k, v in kv_sharing_map(summary).items()} if summary else {},
            "generation_config": C.read_json(gen) if gen.exists() else None,
            "tokenizer_sha256": {e["path"]: e["sha256"] for e in entries if e["path"] in TOKENIZER_FILES}}


def compare(edge, server):
    me = {e["path"]: e["sha256"] for e in edge["manifest"]}
    ms = {e["path"]: e["sha256"] for e in server["manifest"]}
    if me == ms:
        status = "identical"
    elif me and all(ms.get(p) == h for p, h in me.items()):
        status = "edge_subset"
    elif ms and all(me.get(p) == h for p, h in ms.items()):
        status = "server_subset"
    else:
        status = "mismatch"
    diff = sorted(p for p in set(me) | set(ms) if me.get(p) != ms.get(p))
    return {"status": status, "edge_manifest_sha256": edge["manifest_sha256"], "server_manifest_sha256": server["manifest_sha256"],
            "differing_paths": diff[:50], "n_differing": len(diff)}


def quick_check(cap, model_dir=None):
    """Cheap per-stage check (size and mtime of every manifest file). A difference triggers a full re-hash
    (`deep_check`); a hash difference makes the stage INVALID (PREREG §6.2)."""
    root = Path(model_dir or cap["model_dir"])
    problems = []
    for e in cap["manifest"]:
        p = root / e["path"]
        if not p.exists():
            problems.append(f"missing {e['path']}")
            continue
        st = p.stat()
        if st.st_size != e["size"] or st.st_mtime_ns != e["mtime_ns"]:
            problems.append(f"changed {e['path']}")
    present = {x["path"] for x in manifest_paths(root)}
    problems += [f"not in manifest {p}" for p in sorted(present - {e["path"] for e in cap["manifest"]})]
    return problems


def manifest_paths(root):
    out = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDED_DIRS)
        out += [{"path": (Path(dirpath) / fn).relative_to(root).as_posix()} for fn in filenames]
    return out


def deep_check(cap, model_dir=None):
    return manifest_sha256(manifest(model_dir or cap["model_dir"])) == cap["manifest_sha256"]


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--role", required=True, choices=S.ROLES)
    c.add_argument("--model-dir", required=True)
    c.add_argument("--out", required=True)
    m = sub.add_parser("compare")
    m.add_argument("--edge", required=True)
    m.add_argument("--server", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "capture":
        cap = capture(a.model_dir, a.role)
        C.write_json(a.out, cap)
        print({k: cap[k] for k in ("model_dir", "dir_name_matches", "n_files", "total_bytes", "manifest_sha256", "hf_revision")})
        print("features:", cap["features"])
    else:
        print(C.canonical(compare(C.read_json(a.edge), C.read_json(a.server))))


if __name__ == "__main__":
    main()
