"""GEMMA-SPLIT Stage 0e/0f: split-configuration capture, validation, the difference record against the Qwen3 E1-OPT
design, the boundary audit and the deterministic configuration-selection rule (PREREG §4.4, §4.5).

The split of the existing Gemma deployment is RECORDED, never assumed: capture/split_capture.json is filled from
Brian's code, launch files and written answers (template: capture/split_capture.template.json) and validated here
against the captured config.json. Nothing in this module changes a split; a structure that differs from the
Qwen3 first-k/last-k design is listed in `differences`, not corrected.

  python -I gs_split.py validate --capture capture/split_capture.json --checkpoint capture/server/checkpoint.json
  python -I gs_split.py select --candidates capture/candidates.json --roles capture/roles.json --out capture/selected_config.json
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # python -I: the script dir is not on sys.path
import gs_common as C  # noqa: E402
import gs_checkpoint as CK  # noqa: E402
import gs_spec as S  # noqa: E402

SCHEMA = "gemma-split-capture/1"
HOSTS = ("edge", "server")
OWNED_PARTS = ("embed_tokens", "per_layer_embeddings", "final_norm", "lm_head", "sampling", "tokenizer")
DIRECTIONS = ("edge_to_server", "server_to_edge")
WHEN = ("prefill", "decode", "every_step", "once")
REQUIRED_TOP = ("schema", "configuration_id", "captured_utc", "captured_by", "sources", "implementation", "model", "ownership",
                "boundary", "kv_cache", "transport", "generation", "gpu_placement", "launch", "authentication", "tuning_history", "unknowns")
ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
HOST_PORT = re.compile(r"^(?:[a-z][a-z0-9+.-]*://)?[^\s/:@]+:\d{1,5}(?:/\S*)?$")
# Candidate fields that look like measurements are refused: the selection is performance-blind (PREREG §4.5).
MEASUREMENT_KEY = re.compile(r"(latenc|throughput|tpot|ttft|tok(en)?s?_per|per_s(ec)?$|speed|_ms$|_ns$|_s$|time|memory|mem_|vram|_gib$|"
                             r"_gb$|_mib$|_bytes$|bench|score|faster|slower)", re.I)
TOKEN_NAMES = ("input_ids", "token_ids", "token_id", "tokens", "prompt", "text")
PLE_NAMES = ("per_layer", "ple")
KV_NAMES = ("kv", "key", "value", "past")


def _ranges(spec):
    out = []
    for r in spec or []:
        if not (isinstance(r, (list, tuple)) and len(r) == 2 and all(isinstance(x, int) for x in r)):
            raise ValueError(f"layer range {r!r} is not [start, end)")
        out.append((r[0], r[1]))
    return out


def layer_sets(cap, L):
    sets = {}
    for h in HOSTS:
        s = set()
        for a, b in _ranges(cap["ownership"][h].get("layers")):
            s |= set(range(a, b))
        sets[h] = s
    return sets


def _tensor_names(cap, direction):
    return [str(t.get("name", "")).lower() for t in cap.get("boundary", []) if t.get("direction") == direction]


def _any(names, keys):
    return any(k in n for n in names for k in keys)


def placeholders(obj, path="$"):
    """Paths of template placeholders ("FILL...") still present anywhere in a capture."""
    if isinstance(obj, dict):
        return [p for k, v in obj.items() for p in placeholders(v, f"{path}.{k}")]
    if isinstance(obj, list):
        return [p for i, v in enumerate(obj) for p in placeholders(v, f"{path}[{i}]")]
    return [path] if isinstance(obj, str) and obj.strip().upper().startswith("FILL") else []


def _port(endpoint):
    """The port of host:port or of a URL; a URL without one uses its scheme's default (http/ws 80, https/wss 443)."""
    e = str(endpoint or "")
    m = re.search(r":(\d{1,5})(?:/|$)", re.sub(r"^[a-z][a-z0-9+.-]*://", "", e, flags=re.I))
    if m:
        return int(m.group(1))
    return {"http": 80, "ws": 80, "https": 443, "wss": 443}.get(e.split("://", 1)[0].lower()) if "://" in e else None


def authentication_problems(auth):
    """The split capture records HOW the deployment authenticates, never a credential (PREREG §4.1)."""
    if not isinstance(auth, dict):
        return ["authentication must be an object {mechanism, credential_env_vars, config_fields, supplied_out_of_band}"]
    P = []
    if not isinstance(auth.get("mechanism"), str) or not auth["mechanism"].strip():
        P.append("authentication.mechanism must name the mechanism (e.g. 'none', 'shared token in an environment variable', 'mTLS')")
    for k in ("credential_env_vars", "config_fields"):
        v = auth.get(k)
        if not isinstance(v, list) or not all(isinstance(x, str) and ENV_NAME.match(x) for x in v):
            P.append(f"authentication.{k} must be a list of NAMES (letters, digits, underscore), never values")
    if not isinstance(auth.get("supplied_out_of_band"), bool):
        P.append("authentication.supplied_out_of_band must be true or false")
    return P


def validate(cap, ck_summary):
    """(problems, warnings). Problems block Stage 0 (STOP_SPLIT_CAPTURE_INVALID or STOP_INCOMPLETE for unknowns)."""
    P, W = [], []
    missing = [k for k in REQUIRED_TOP if k not in cap]
    if missing:
        return [f"missing top-level fields {missing}"], W
    left = placeholders({k: v for k, v in cap.items() if k != "unknowns"})
    if left:
        P.append(f"template placeholders left: {left[:8]}")
    if cap["unknowns"]:
        P.append(f"unknowns remain: {cap['unknowns']}")
    if cap["schema"] != SCHEMA:
        P.append(f"schema {cap['schema']!r} != {SCHEMA!r}")
    if not isinstance(cap["configuration_id"], str) or not cap["configuration_id"].strip():
        P.append("configuration_id must be a non-empty string")
    P += authentication_problems(cap.get("authentication"))
    ep = (cap.get("transport") or {}).get("experiment_endpoint")
    if not isinstance(ep, str) or not HOST_PORT.match(ep):
        P.append(f"transport.experiment_endpoint must be host:port (or a URL with a port) for the experiment's own server instance, got {ep!r}")
    elif _port(ep) == _port((cap.get("transport") or {}).get("endpoint")):
        P.append("transport.experiment_endpoint uses the live deployment's port: the experiment's own instance listens on its own "
                 "port, so that the edge can never reach Brian's running server, whatever the host is called (PREREG §4.3)")
    if not isinstance(cap.get("tuning_history"), str) or not cap["tuning_history"].strip():
        P.append("tuning_history must record whether and how the configuration was chosen by measurement (Q4; 'none' if not)")
    srv = (cap.get("gpu_placement") or {}).get("server") or {}
    if any(str(d).startswith("cuda") for d in srv.get("devices", [])) and not srv.get("CUDA_VISIBLE_DEVICES"):
        P.append("gpu_placement.server.CUDA_VISIBLE_DEVICES must pin the frozen GPU(s) on the multi-GPU server, so that the "
                 "deployment can load on no other device (PREREG §8)")
    found = C.secret_findings(json.dumps(cap, indent=1))
    if found:
        P.append(f"credential-like content in the split capture {sorted({r for r, _ in found})}: record only mechanism and names (PREREG §4.1)")
    L = (ck_summary or {}).get("num_hidden_layers")
    if not L:
        return P + ["captured config.json has no num_hidden_layers: cannot validate ownership"], W
    if cap["model"].get("num_hidden_layers") != L:
        P.append(f"split capture says {cap['model'].get('num_hidden_layers')} layers, config.json says {L}")
    try:
        sets = layer_sets(cap, L)
    except (ValueError, KeyError) as e:
        return P + [f"ownership: {e}"], W
    for h in HOSTS:
        for a, b in _ranges(cap["ownership"][h].get("layers")):
            if not (0 <= a < b <= L):
                P.append(f"{h}: layer range [{a},{b}) outside [0,{L}) or empty")
    both = sets["edge"] & sets["server"]
    if both:
        P.append(f"layers owned by both hosts: {sorted(both)[:10]}")
    unowned = set(range(L)) - sets["edge"] - sets["server"]
    if unowned:
        P.append(f"layers owned by no host: {sorted(unowned)[:10]}")
    if not sets["server"]:
        P.append("the server owns no layer: not a two-host split")
    if not sets["edge"]:
        P.append("the edge owns no layer: not a two-host split")
    feats = CK.features(ck_summary)
    for h in HOSTS:
        for part in OWNED_PARTS:
            v = cap["ownership"][h].get(part)
            if not isinstance(v, bool) and not (part == "per_layer_embeddings" and not feats["per_layer_embeddings"] and v is None):
                P.append(f"ownership.{h}.{part} must be true or false, got {v!r}")
    for part in OWNED_PARTS:
        owners = [h for h in HOSTS if cap["ownership"][h].get(part) is True]
        if part == "per_layer_embeddings" and not feats["per_layer_embeddings"]:
            continue
        if not owners:
            P.append(f"{part}: no owner recorded")
        elif len(owners) == 2 and not cap["ownership"].get("duplicated_parts_reason", {}).get(part):
            P.append(f"{part}: owned by both hosts without a recorded reason (ownership.duplicated_parts_reason)")
    for t in cap["boundary"]:
        for k in ("direction", "name", "dtype", "shape", "when"):
            if k not in t:
                P.append(f"boundary entry {t!r} lacks {k}")
        if t.get("direction") not in DIRECTIONS:
            P.append(f"boundary direction {t.get('direction')!r} not in {DIRECTIONS}")
        if t.get("when") not in WHEN:
            P.append(f"boundary 'when' {t.get('when')!r} not in {WHEN}")
    for d in DIRECTIONS:
        if not _tensor_names(cap, d):
            P.append(f"no boundary tensor declared {d}")
    e2s = _tensor_names(cap, "edge_to_server")
    if feats["per_layer_embeddings"] and sets["server"]:
        server_has_ple = bool(cap["ownership"]["server"].get("per_layer_embeddings"))
        if not _any(e2s, PLE_NAMES) and not (server_has_ple and _any(e2s, TOKEN_NAMES)):
            P.append("config has per-layer input embeddings and the server owns layers, but the boundary carries neither "
                     "per-layer inputs nor token ids for a server-side PLE table: record how the server obtains them")
    kv_map = CK.kv_sharing_map(ck_summary)
    cross = [(c, s) for c, s in kv_map.items() if s is not None and ((c in sets["edge"]) != (s in sets["edge"]))]
    if cross and not _any(e2s + _tensor_names(cap, "server_to_edge"), KV_NAMES) and not cap["kv_cache"].get("shared_kv_handling"):
        P.append(f"KV-sharing consumers and sources sit on different hosts {cross[:5]} but no KV tensor is declared on the "
                 "boundary and kv_cache.shared_kv_handling is empty")
    if any(s is None for s in kv_map.values()):
        W.append("config's KV-sharing layout has a consumer without a source layer (as derived from transformers' Gemma 4 rule)")
    last_owner = "edge" if (L - 1) in sets["edge"] else "server"
    for part in ("final_norm", "lm_head"):
        if not cap["ownership"][last_owner].get(part):
            W.append(f"{part} is not on the host that owns the last layer ({last_owner}): the boundary must carry the post-layer hidden state")
    if 0 in sets["server"] and cap["ownership"]["edge"].get("embed_tokens"):
        W.append("layer 0 is on the server but embed_tokens on the edge: embeddings cross the boundary")
    for h in HOSTS:
        if not cap["model"].get(f"dtype_{h}"):
            P.append(f"model.dtype_{h} missing")
        gp = cap["gpu_placement"].get(h) or {}
        if not gp.get("devices"):
            P.append(f"gpu_placement.{h}.devices missing")
        if not gp.get("uuids") or len(gp.get("uuids")) != len(gp.get("devices") or []):
            P.append(f"gpu_placement.{h}.uuids missing or not one per device (GPU UUIDs from capture/{h}/inventory.json)")
    if not cap["transport"].get("protocol"):
        P.append("transport.protocol missing")
    if not cap["transport"].get("endpoint"):
        P.append("transport.endpoint missing (host:port or URL the edge uses)")
    g = cap["generation"]
    for k in ("sampling_location", "greedy_supported", "streaming_supported", "ignore_eos_supported"):
        if k not in g:
            P.append(f"generation.{k} missing")
        elif k != "sampling_location" and not isinstance(g[k], bool):
            P.append(f"generation.{k} must be true or false, got {g[k]!r}")
    if g.get("greedy_supported") is False:
        P.append("generation.greedy_supported is false: correctness (PREREG §9) needs greedy decoding")
    for h in HOSTS:
        if "kv_cache_" + h not in cap["kv_cache"]:
            P.append(f"kv_cache.kv_cache_{h} missing (per-layer cache | none | other: describe)")
    return P, W


def topology(cap, L):
    sets = layer_sets(cap, L)
    e, s = sets["edge"], sets["server"]
    if not e or not s:
        return "single_host"
    if e == set(range(min(e), max(e) + 1)) and min(e) == 0 and s == set(range(max(e) + 1, L)):
        return "prefix_edge_suffix_server"
    k = next(i for i in range(L) if i not in e)
    j = L - 1 - max(i for i in range(L) if i in s)
    if min(e) == 0 and e == set(range(0, k)) | set(range(L - j, L)) and s == set(range(k, L - j)) and j > 0:
        return "u_shape"
    if s == set(range(0, min(e))) and e == set(range(min(e), L)):
        return "prefix_server_suffix_edge"
    return "other"


def derive(cap, ck_summary):
    L = ck_summary["num_hidden_layers"]
    sets = layer_sets(cap, L)
    e2s, s2e = _tensor_names(cap, "edge_to_server"), _tensor_names(cap, "server_to_edge")
    own = cap["ownership"]
    feats = CK.features(ck_summary)
    tied = feats.get("tied_embeddings")
    d = {
        "topology": topology(cap, L), "num_hidden_layers": L,
        "edge_layers": sorted(sets["edge"]), "server_layers": sorted(sets["server"]),
        "n_edge_layers": len(sets["edge"]), "n_server_layers": len(sets["server"]),
        "token_ids_leave_edge": _any(e2s, TOKEN_NAMES), "per_layer_inputs_cross": _any(e2s, PLE_NAMES),
        "kv_crosses": _any(e2s + s2e, KV_NAMES), "logits_cross": _any(s2e, ("logit", "logprob")),
        "generated_tokens_on_server": bool(own["server"].get("sampling")) or _any(s2e, ("token",)),
        "lm_head_on_server": bool(own["server"].get("lm_head")),
        "server_holds_embedding_table": bool(own["server"].get("embed_tokens")) or (bool(own["server"].get("lm_head")) and tied is not False),
        "server_holds_embedding_table_basis": "embed_tokens on server" if own["server"].get("embed_tokens") else
        ("lm_head on server and tie_word_embeddings is " + ("true" if tied else "absent from config.json (class default)")
         if own["server"].get("lm_head") and tied is not False else "no"),
        "server_gpus": len((cap["gpu_placement"].get("server") or {}).get("devices") or []),
        "dtypes": {h: cap["model"].get(f"dtype_{h}") for h in HOSTS}, "quantization": cap["model"].get("quantization"),
        "kv_cache": {h: cap["kv_cache"].get("kv_cache_" + h) for h in HOSTS},
    }
    d["differences_from_qwen3_e1opt"] = differences(d, cap)
    return d


def differences(d, cap):
    """Plain-language list of every structural difference from the Qwen3 E1-OPT design (PREREG §17). Descriptive."""
    q = S.QWEN3_E1OPT_REFERENCE
    out = []
    if d["topology"] != q["topology"]:
        out.append(f"topology {d['topology']} (E1-OPT: u_shape, edge = first k and last k layers)")
    for part, label in (("lm_head", "LM head"), ("final_norm", "final norm"), ("sampling", "sampling"), ("tokenizer", "tokenizer"),
                        ("embed_tokens", "token embedding")):
        if cap["ownership"]["server"].get(part):
            out.append(f"{label} on the server (E1-OPT: edge)")
    if d["token_ids_leave_edge"]:
        out.append("token ids or text cross to the server (E1-OPT: never)")
    if d["per_layer_inputs_cross"]:
        out.append("per-layer input embeddings cross the wire (E1-OPT: residual stream only)")
    if d["kv_crosses"]:
        out.append("KV entries cross the wire (E1-OPT: never)")
    if d["logits_cross"]:
        out.append("logits / log-probs cross the wire (E1-OPT: never)")
    if d["generated_tokens_on_server"]:
        out.append("generated tokens are visible to the server (E1-OPT: never)")
    if d["server_holds_embedding_table"]:
        out.append(f"the server holds the token-embedding table ({d['server_holds_embedding_table_basis']})")
    if d["server_gpus"] != q["server_gpus"]:
        out.append(f"server uses {d['server_gpus']} GPU(s) (E1-OPT: exactly 1)")
    for h, dt in d["dtypes"].items():
        if dt and dt.replace("torch.", "") not in ("bfloat16", "bf16"):
            out.append(f"{h} dtype {dt} (E1-OPT: bfloat16)")
    if d["quantization"] not in (None, "none", "None"):
        out.append(f"quantization {d['quantization']} (E1-OPT: none)")
    extra = [t for t in cap["boundary"] if not any(k in str(t.get("name", "")).lower() for k in ("hidden", "residual"))]
    if extra:
        out.append("wire content beyond the residual stream: " + ", ".join(f"{t['direction']}:{t['name']}" for t in extra))
    for h, kv in d["kv_cache"].items():
        if kv and "none" in str(kv).lower():
            out.append(f"{h} keeps no KV cache: every step recomputes the prefix (E1-OPT: each host caches its own layers)")
    return out


def declared_rank(shape):
    """Number of dimensions of a declared shape such as "[1, n, hidden_size]"; None if it is not written as a list."""
    s = str(shape).strip()
    return len([x for x in s[1:-1].split(",") if x.strip()]) if s.startswith("[") and s.endswith("]") else None


def audit_boundary(cap, observed):
    """Compare the declared boundary with the messages of the deployment's generation path (adapter.boundary_audit()).
    observed: [{"direction", "tensors": [{"name", "dtype", "shape"}], "bytes"}]. Problems: undeclared content, a dtype or
    rank that differs from the declaration, and a declared per-step tensor that never appears (PREREG §9.3)."""
    declared = {(t["direction"], str(t["name"]).lower()): t for t in cap["boundary"]}
    seen, problems = set(), []
    for msg in observed:
        for t in msg.get("tensors", []):
            key = (msg["direction"], str(t["name"]).lower())
            seen.add(key)
            dec = declared.get(key)
            if dec is None:
                problems.append(f"undeclared boundary tensor {key[0]}:{t['name']} dtype {t.get('dtype')} shape {t.get('shape')}")
                continue
            if str(dec["dtype"]).replace("torch.", "") != str(t.get("dtype")).replace("torch.", ""):
                problems.append(f"{key[0]}:{t['name']} dtype {t.get('dtype')} != declared {dec['dtype']}")
            rk = declared_rank(dec.get("shape"))
            if rk is not None and t.get("shape") is not None and len(t["shape"]) != rk:
                problems.append(f"{key[0]}:{t['name']} has {len(t['shape'])} dimensions, declared {dec['shape']}")
    never = sorted(f"{k[0]}:{k[1]}" for k in declared if k not in seen)
    problems += [f"declared {k[0]}:{k[1]} ({d['when']}) never crossed the boundary" for k, d in declared.items()
                 if k not in seen and d.get("when") in ("every_step", "prefill", "decode")]
    return {"problems": sorted(set(problems)), "declared_never_observed": never, "n_messages": len(observed),
            "bytes_by_direction": {d: sum(m.get("bytes", 0) for m in observed if m["direction"] == d) for d in DIRECTIONS}}


# --------------------------------------------------------------------------- #
# Configuration selection (PREREG §4.5): deterministic, performance-blind      #
# --------------------------------------------------------------------------- #
SELECTION_RULES = (
    ("R1", "designated_by_implementer", "the implementer's dated written designation of the current Gemma configuration"),
    ("R2", "running_at_capture", "the configuration of the deployment processes running at Stage-0 capture"),
    ("R3", "implementation_default", "the implementation's default configuration (launch defaults, no overrides)"),
)


def select(candidates, roles):
    """candidates: [{configuration_id, model_dir_name, edge_host, server_host, designated_by_implementer,
    designation_ref, running_at_capture, implementation_default}]. Each rule narrows the eligible set only if at
    least one candidate satisfies it; the last tie-break is the smallest configuration_id in byte order."""
    eh, sh = roles["edge"]["host"], roles["server"]["host"]
    rec = {"schema": "gemma-split-selection/1", "utc": C.utc(), "n_candidates": len(candidates), "trace": [], "selected": None,
           "rule_applied": None, "verdict": None, "performance_inputs": "none (PREREG §4.5)"}
    elig = []
    for c in candidates:
        why = [f"template placeholders left {placeholders(c)}"] if placeholders(c) else []
        if c.get("model_dir_name") != S.MODEL_DIR_NAME:
            why.append(f"model {c.get('model_dir_name')!r} != {S.MODEL_DIR_NAME!r}")
        if c.get("edge_host") != eh or c.get("server_host") != sh:
            why.append(f"hosts ({c.get('edge_host')}, {c.get('server_host')}) != roles ({eh}, {sh})")
        if c.get("designated_by_implementer") and not c.get("designation_ref"):
            why.append("designation without a dated written reference (lab_answers.md anchor)")
        perf = sorted(k for k in c if MEASUREMENT_KEY.search(str(k)))
        if perf:
            why.append(f"measurement-like fields {perf}: a candidate carries Stage-0 facts only (PREREG §4.5)")
        rec["trace"].append({"configuration_id": c.get("configuration_id"), "eligible": not why, "excluded_because": why})
        if not why:
            elig.append(c)
    if not elig:
        rec["verdict"] = "STOP_NO_CONFIGURATION"
        return rec
    rule_applied = []
    for rid, key, desc in SELECTION_RULES:
        hit = [c for c in elig if c.get(key)]
        if hit:
            if len(hit) < len(elig):
                rule_applied.append(f"{rid}: {desc}")
            elig = hit
        if len(elig) == 1:
            break
    if len(elig) > 1:
        elig = sorted(elig, key=lambda c: str(c["configuration_id"]).encode())[:1]
        rule_applied.append("R4: smallest configuration_id in byte order")
    rec.update(selected=elig[0]["configuration_id"], rule_applied=rule_applied or ["single eligible candidate"], verdict="CONTINUE")
    return rec


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--capture", required=True)
    v.add_argument("--checkpoint", required=True, help="capture/server/checkpoint.json (its config summary)")
    s = sub.add_parser("select")
    s.add_argument("--candidates", required=True)
    s.add_argument("--roles", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "validate":
        cap, ck = C.read_json(a.capture), C.read_json(a.checkpoint)
        P, W = validate(cap, ck["summary"])
        for w in W:
            print("warning:", w)
        if P:
            print("INVALID split capture:\n  - " + "\n  - ".join(P))
            sys.exit(2)
        d = derive(cap, ck["summary"])
        print(C.canonical({k: d[k] for k in ("topology", "n_edge_layers", "n_server_layers", "token_ids_leave_edge", "kv_crosses")}))
        print("differences from Qwen3 E1-OPT:\n  - " + "\n  - ".join(d["differences_from_qwen3_e1opt"] or ["none"]))
    else:
        rec = select(C.read_json(a.candidates)["candidates"], C.read_json(a.roles))
        C.write_json(a.out, rec)
        print(rec["verdict"], rec["selected"], rec["rule_applied"])
        sys.exit(0 if rec["verdict"] == "CONTINUE" else 2)


if __name__ == "__main__":
    main()
