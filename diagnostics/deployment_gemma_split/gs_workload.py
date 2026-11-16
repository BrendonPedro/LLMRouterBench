"""GEMMA-SPLIT workload (PREREG §7): frozen prompt texts now, token ids at Stage 1.

`build` was run ONCE during preparation to write workload/correctness_prompts.json and workload/timing_spec.json
from the repository's S1 sample (prompt texts only: key, dataset, prompt, prompt_sha256; no score and no model
output is copied). The two JSON files are committed, hashed by the freeze record and are the workload; the sample
is not read again by any later stage. tests/test_gs_protocol.py checks that `build` reproduces them.

`tokenize` (Stage 1, CPU only, on a host that has the captured tokenizer) turns the texts into token ids:
  natural prompts  -> the checkpoint's chat template, one user turn, add_generation_prompt=True
  timing/coverage  -> [BOS] + tokens of pool texts joined from item.start onward, truncated to exactly P tokens

  python -I gs_workload.py build --sample <repo>/diagnostics/deployment_s1/sample_200.jsonl --out-dir workload   (done once)
  python -I gs_workload.py tokenize --tokenizer-dir <checkpoint dir> --out capture/workload_tokens.json
"""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # python -I: the script dir is not on sys.path
import gs_common as C  # noqa: E402
import gs_spec as S  # noqa: E402

WORKLOAD_DIR = S.GS_DIR / "workload"
CORR_FILE, TIMING_FILE = "correctness_prompts.json", "timing_spec.json"


def _order_key(seed, key):
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def build(sample_path, out_dir=WORKLOAD_DIR, overwrite=False):
    raw = Path(sample_path).read_bytes()
    src_sha = C.sha256_bytes(raw)
    if src_sha != S.WORKLOAD_SOURCE_SHA256:
        raise ValueError(f"sample sha256 {src_sha} != pinned {S.WORKLOAD_SOURCE_SHA256}")
    rows = [__import__("json").loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
    for r in rows:
        if C.sha256_bytes(r["prompt"].encode("utf-8")) != r["prompt_sha256"]:
            raise ValueError(f"{r['key']}: prompt hash mismatch")
    source = {"file": S.WORKLOAD_SOURCE, "sha256": src_sha, "fields_copied": ["key", "dataset", "prompt", "prompt_sha256"],
              "note": "prompt texts only; no score, answer or model output is copied"}
    chosen = []
    for ds in sorted(S.CORR_PER_DATASET):
        cand = sorted((r for r in rows if r["dataset"] == ds), key=lambda r: _order_key(S.CORR_SEED, r["key"]))
        if len(cand) < S.CORR_PER_DATASET[ds]:
            raise ValueError(f"{ds}: only {len(cand)} prompts")
        chosen += cand[:S.CORR_PER_DATASET[ds]]
    corr = {"schema": "gemma-split-workload-correctness/1", "source": source, "seed": S.CORR_SEED, "per_dataset": S.CORR_PER_DATASET,
            "gen_tokens": S.CORR_GEN_TOKENS, "ignore_eos": False,
            "items": [{"id": f"nat-{i:02d}", "key": r["key"], "dataset": r["dataset"], "prompt": r["prompt"],
                       "prompt_sha256": r["prompt_sha256"]} for i, r in enumerate(chosen)]}
    used = {r["key"] for r in chosen}
    pool = sorted((r for r in rows if r["key"] not in used), key=lambda r: _order_key(S.TIMING_SEED, r["key"]))
    items, q = [], 0
    for g in S.TIMING_GROUPS:
        for kind, n in (("warmup", S.TIMING_WARMUP), ("measured", S.TIMING_MEASURED)):
            for i in range(n):
                items.append({"id": f"{g}-{kind[0]}{i:02d}", "set": "timing", "group": g, "kind": kind, "prompt_tokens": S.TIMING_GROUPS[g],
                              "gen_tokens": S.TIMING_GEN_TOKENS, "ignore_eos": True, "start": q % len(pool)})
                q += 1
    for i in range(S.COVERAGE_ITEMS):
        items.append({"id": f"cov-{i:02d}", "set": "coverage", "group": "coverage", "kind": "correctness",
                      "prompt_tokens": S.COVERAGE_PROMPT_TOKENS, "gen_tokens": S.COVERAGE_GEN_TOKENS, "ignore_eos": True,
                      "start": q % len(pool)})
        q += 1
    timing = {"schema": "gemma-split-workload-timing/1", "source": source, "seed": S.TIMING_SEED, "join": S.POOL_JOIN,
              "construction": "ids = [BOS] + tokenize(join(pool[start], pool[start+1], ...))[: prompt_tokens - 1]",
              "pool": [{"key": r["key"], "prompt": r["prompt"], "prompt_sha256": r["prompt_sha256"]} for r in pool], "items": items}
    C.write_json(Path(out_dir) / CORR_FILE, corr, overwrite=overwrite)
    C.write_json(Path(out_dir) / TIMING_FILE, timing, overwrite=overwrite)
    return corr, timing


def load_spec(workload_dir=WORKLOAD_DIR):
    return C.read_json(Path(workload_dir) / CORR_FILE), C.read_json(Path(workload_dir) / TIMING_FILE)


def pool_text(pool, start, n_parts, join=S.POOL_JOIN):
    return join.join(pool[(start + i) % len(pool)]["prompt"] for i in range(n_parts))


def eos_ids_of(tokenizer_dir, tok):
    gen = Path(tokenizer_dir) / "generation_config.json"
    if gen.exists():
        e = C.read_json(gen).get("eos_token_id")
        if e is not None:
            return (e if isinstance(e, list) else [e]), "generation_config.json"
    cfg = Path(tokenizer_dir) / "config.json"
    if cfg.exists():
        c = C.read_json(cfg)
        e = c.get("eos_token_id", (c.get("text_config") or {}).get("eos_token_id"))
        if e is not None:
            return (e if isinstance(e, list) else [e]), "config.json"
    return [tok.eos_token_id], "tokenizer"


def _chat_ids(tok, text):
    out = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True)
    if hasattr(out, "keys"):          # BatchEncoding / dict (newer transformers)
        out = out["input_ids"]
    if out and isinstance(out[0], list):
        out = out[0]
    rendered = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False)
    return [int(x) for x in out], rendered


def tokenize(tokenizer_dir, workload_dir=WORKLOAD_DIR):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    corr, timing = load_spec(workload_dir)
    bos = tok.bos_token_id
    if bos is None:
        raise ValueError("tokenizer has no BOS token id: the timing construction needs one (PREREG §7.2)")
    eos, eos_src = eos_ids_of(tokenizer_dir, tok)
    items = []
    for it in corr["items"]:
        ids, rendered = _chat_ids(tok, it["prompt"])
        items.append({"id": it["id"], "set": "natural", "group": "natural", "kind": "correctness", "n": len(ids),
                      "gen_tokens": corr["gen_tokens"], "ignore_eos": corr["ignore_eos"], "ids": ids,
                      "rendered_sha256": C.sha256_bytes(rendered.encode("utf-8"))})
    pool = timing["pool"]
    for it in timing["items"]:
        P, parts = it["prompt_tokens"], 1
        while True:
            body = tok(pool_text(pool, it["start"], parts, timing["join"]), add_special_tokens=False)["input_ids"]
            if len(body) >= P - 1:
                break
            parts += 1
            if parts > len(pool):
                raise ValueError(f"{it['id']}: the whole pool is shorter than {P} tokens")
        ids = [int(bos)] + [int(x) for x in body[:P - 1]]
        items.append({"id": it["id"], "set": it["set"], "group": it["group"], "kind": it["kind"], "n": len(ids), "gen_tokens": it["gen_tokens"],
                      "ignore_eos": it["ignore_eos"], "ids": ids, "pool_parts": parts})
    tok_files = {p.name: C.sha256_file(p) for p in sorted(Path(tokenizer_dir).iterdir())
                 if p.is_file() and (p.name.startswith("tokenizer") or p.name.startswith("special_tokens") or "chat_template" in p.name
                                     or p.name == "generation_config.json")}
    return finalize({"schema": "gemma-split-workload-tokens/1", "utc": C.utc(), "tokenizer_dir": str(Path(tokenizer_dir).resolve()),
                     "tokenizer_sha256": tok_files, "tokenizer_class": type(tok).__name__, "bos_token_id": int(bos),
                     "eos_token_ids": [int(e) for e in eos], "eos_source": eos_src,
                     "chat_template_sha256": C.sha256_bytes((tok.chat_template or "").encode("utf-8")),
                     "spec_sha256": {"correctness": C.sha256_file(Path(workload_dir) / CORR_FILE),
                                     "timing": C.sha256_file(Path(workload_dir) / TIMING_FILE)},
                     "items": items})


def finalize(w):
    w["ids_sha256"] = C.sha256_obj([[it["id"], it["ids"], it["gen_tokens"], it["ignore_eos"]] for it in w["items"]])
    totals = [(it["n"] + it["gen_tokens"], it["id"]) for it in w["items"]]
    w["max_total_tokens"] = max(t for t, _ in totals)
    w["probe_item"] = min((i for t, i in totals if t == w["max_total_tokens"]))
    return w


def validate_tokens(w, workload_dir=WORKLOAD_DIR, dev=False):
    """Problems of a tokenized workload. Target mode: exactly the frozen construction (PREREG §7). Dev mode: structure only."""
    P = []
    if w.get("ids_sha256") != C.sha256_obj([[it["id"], it["ids"], it["gen_tokens"], it["ignore_eos"]] for it in w.get("items", [])]):
        P.append("ids_sha256 does not match the items")
    for it in w.get("items", []):
        if it["n"] != len(it["ids"]):
            P.append(f"{it['id']}: n != len(ids)")
    if dev:
        return P
    corr, timing = load_spec(workload_dir)
    if w.get("spec_sha256") != {"correctness": C.sha256_file(Path(workload_dir) / CORR_FILE), "timing": C.sha256_file(Path(workload_dir) / TIMING_FILE)}:
        P.append("tokens were produced from a different workload spec")
    by = {it["id"]: it for it in w["items"]}
    want = [it["id"] for it in corr["items"]] + [it["id"] for it in timing["items"]]
    if sorted(by) != sorted(want):
        P.append(f"item ids differ from the spec ({len(by)} vs {len(want)})")
        return P
    for it in timing["items"]:
        x = by[it["id"]]
        if x["n"] != it["prompt_tokens"] or x["gen_tokens"] != it["gen_tokens"] or x["ignore_eos"] != it["ignore_eos"]:
            P.append(f"{it['id']}: size/gen/eos differ from the spec")
        if x["ids"][0] != w["bos_token_id"]:
            P.append(f"{it['id']}: does not start with BOS")
    for it in corr["items"]:
        x = by[it["id"]]
        if x["gen_tokens"] != corr["gen_tokens"] or x["ignore_eos"] != corr["ignore_eos"]:
            P.append(f"{it['id']}: gen/eos differ from the spec")
    return P


def items(w, set_=None, group=None, kind=None):
    return [it for it in w["items"] if (set_ is None or it["set"] in ((set_,) if isinstance(set_, str) else set_))
            and (group is None or it["group"] == group) and (kind is None or it["kind"] == kind)]


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--sample", required=True)
    b.add_argument("--out-dir", default=str(WORKLOAD_DIR))
    t = sub.add_parser("tokenize")
    t.add_argument("--tokenizer-dir", required=True)
    t.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        corr, timing = build(a.sample, a.out_dir)
        print(len(corr["items"]), "natural prompts;", len(timing["items"]), "timing/coverage items;", len(timing["pool"]), "pool texts")
    else:
        w = tokenize(a.tokenizer_dir)
        P = validate_tokens(w)
        if P:
            print("tokenized workload does not match the frozen construction:\n  - " + "\n  - ".join(P))
            sys.exit(2)
        C.write_json(a.out, w)
        print(len(w["items"]), "items; max total tokens", w["max_total_tokens"], "probe item", w["probe_item"], "ids", w["ids_sha256"][:16])


if __name__ == "__main__":
    main()
