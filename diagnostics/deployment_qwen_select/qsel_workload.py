"""QWEN-SELECT workload: held-out queries of the E0 sample's pre-declared extension (PREREG section 4).

The 160 queries are ranks n..2n-1 of the S1 permutation for MMLU-Pro (60), GPQA (60) and LiveMathBench (40): the
part of the 400-query extension that E0 pinned on 2026-11-08 and never ran. No model of S1, E0, K16-HWFIT or
GEMMA-SELECT has answered them for us.

Two files. workload/workload.json is public: keys, order, sets, and per query the hashes of its prompt, gold answer and
token ids, and its prompt length in tokens. It holds no prompt text. The private file (prompts and gold answers, one
JSON object per line) is rebuilt from the public LLMRouterBench release by `build` and pinned by SHA-256. It is not
committed, because GPQA asks that its questions are not published in plain text.

  python -I -B qsel_workload.py build --tokenizer DIR --private-out FILE    # maintainer, before the tag
  python -I -B qsel_workload.py show
"""
import importlib.util
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qsel_common as C  # noqa: E402
import qsel_spec as S  # noqa: E402


def tokenize(tok, prompt):
    """Chat-formatted token ids of one user turn, generation prompt added, thinking off: the same for every configuration."""
    out = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True,
                                  enable_thinking=S.THINKING)
    ids = out["input_ids"] if isinstance(out, dict) or hasattr(out, "keys") else out
    ids = ids[0] if ids and isinstance(ids[0], (list, tuple)) else ids
    return [int(x) for x in ids]


def ids_sha(ids):
    return C.sha256_bytes(json.dumps([int(x) for x in ids]).encode())


def load_tokenizer(model_dir):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)


def heldout_rows(repo=None):
    """The held-out rows, drawn by S1's own code (run_s1.draw_sample(scale=2)) from the cached public release."""
    repo = Path(repo or C.REPO)
    spec = importlib.util.spec_from_file_location("run_s1", repo / "diagnostics" / "deployment_s1" / "run_s1.py")
    s1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s1)
    rows, _ = s1.draw_sample(scale=2)
    ext = C.read_json(repo / S.EXTENSION_KEYS_REL)
    if ext["keys_sha256"] != S.EXTENSION_KEYS_SHA256 or sorted(r["key"] for r in rows) != sorted(ext["keys"]):
        raise C.Stop("the drawn sample is not the pinned 400-query extension")
    first = {json.loads(l)["key"] for l in (repo / S.S1_SAMPLE_REL).read_text(encoding="utf-8").splitlines() if l.strip()}
    n = {d: k for d, _, k in s1.BENCHMARKS}
    held = [r for r in rows if r["dataset"] in S.BENCHMARKS and r["rank"] >= n[r["dataset"]]]
    if {r["key"] for r in held} & first or {d: sum(r["dataset"] == d for r in held) for d in S.BENCHMARKS} != S.HELDOUT_PER_BENCH:
        raise C.Stop("held-out rows overlap the first sample or have the wrong counts")
    return held


def private_bytes(rows):
    return "".join(json.dumps({"key": r["key"], "dataset": r["dataset"], "prompt": r["prompt"], "answer": r["answer"]},
                              ensure_ascii=False, sort_keys=True) + "\n" for r in sorted(rows, key=lambda r: r["key"])).encode("utf-8")


def build(rows, tok):
    keys = sorted(r["key"] for r in rows)
    order = list(keys)
    random.Random(S.ORDER_SEED).shuffle(order)
    sentinel = set()
    for b, n in S.SENTINEL_PER_BENCH.items():
        sentinel |= {r["key"] for r in sorted((r for r in rows if r["dataset"] == b), key=lambda r: (r["rank"], r["key"]))[:n]}
    q = {}
    for r in rows:
        ids = tokenize(tok, r["prompt"])
        q[r["key"]] = {"dataset": r["dataset"], "prompt_sha256": C.sha256_bytes(r["prompt"].encode("utf-8")),
                       "answer_sha256": C.sha256_bytes(str(r["answer"]).encode("utf-8")), "n_prompt": len(ids), "ids_sha256": ids_sha(ids)}
    by_len = sorted(keys, key=lambda k: (q[k]["n_prompt"], k))
    correctness = by_len[:S.N_CORRECTNESS_SHORT] + by_len[-S.N_CORRECTNESS_LONG:]
    return {"schema": S.SCHEMA, "experiment": S.NAME, "extension_keys_sha256": S.EXTENSION_KEYS_SHA256, "benchmarks": list(S.BENCHMARKS),
            "heldout": "ranks n..2n-1 of the S1 permutation (run_s1.draw_sample(scale=2)); disjoint from sample_200.jsonl",
            "tokenizer_json_sha256": S.TOKENIZER_JSON_SHA256, "thinking": S.THINKING,
            "utility": order, "sentinel": [k for k in order if k in sentinel], "correctness": correctness, "queries": q}


def public():
    path = C.QS_DIR / S.WORKLOAD_FILE
    if C.sha256_file(path) != S.WORKLOAD_SHA256:
        raise C.Stop("workload.json is not the pinned file")
    return C.read_json(path)


def max_new(n_prompt):
    """Output cap of one query: every configuration has the same context window."""
    return max(1, min(S.MAX_NEW_TOKENS, S.CONTEXT_TOKENS - int(n_prompt)))


def load(private_path, dev_workload=None):
    """{"utility": [query], "sentinel": [query], "correctness": [query]} with prompt and answer; every hash verified.
    dev_workload: a (public dict, private path) pair of the CPU tests in place of the pinned files."""
    if dev_workload is not None:
        w = dev_workload
    else:
        w = public()
        if C.sha256_file(private_path) != S.WORKLOAD_PRIVATE_SHA256:
            raise C.Stop("the private workload file is not the pinned one")
    rows = {r["key"]: r for r in C.read_jsonl(private_path)}
    if sorted(rows) != sorted(w["queries"]):
        raise C.Stop("the private workload file does not hold exactly the workload's queries")
    for k, q in w["queries"].items():
        if C.sha256_bytes(rows[k]["prompt"].encode("utf-8")) != q["prompt_sha256"] or C.sha256_bytes(str(rows[k]["answer"]).encode("utf-8")) != q["answer_sha256"]:
            raise C.Stop(f"prompt or answer of {k} differs from the workload")
    one = lambda k: dict(rows[k], n_prompt=w["queries"][k]["n_prompt"], ids_sha256=w["queries"][k]["ids_sha256"])  # noqa: E731
    return {name: [one(k) for k in w[name]] for name in ("utility", "sentinel", "correctness")}


def token_ids(tok, q):
    """Token ids of a query, refused unless they are the ids the workload pins (same ids on every host and configuration)."""
    ids = tokenize(tok, q["prompt"])
    if len(ids) != q["n_prompt"] or ids_sha(ids) != q["ids_sha256"]:
        raise C.Stop(f"tokenization of {q['key']} differs from the pinned token ids")
    return ids


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--tokenizer", required=True)
    b.add_argument("--private-out", required=True)
    sp.add_parser("show")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        rows = heldout_rows()
        if C.sha256_file(Path(a.tokenizer) / "tokenizer.json") != S.TOKENIZER_JSON_SHA256:
            raise C.Stop("tokenizer.json is not the pinned one")
        w = build(rows, load_tokenizer(a.tokenizer))
        C.write_json(C.QS_DIR / S.WORKLOAD_FILE, w)
        out = Path(a.private_out)
        if out.exists():
            raise C.Stop("the private output exists: outputs are written once")
        out.write_bytes(private_bytes(rows))
        print("workload.json sha256", C.sha256_file(C.QS_DIR / S.WORKLOAD_FILE))
        print("private file sha256 ", C.sha256_file(out))
    w = C.read_json(C.QS_DIR / S.WORKLOAD_FILE)
    n = [q["n_prompt"] for q in w["queries"].values()]
    print({k: len(w[k]) for k in ("utility", "sentinel", "correctness")}, "prompt tokens min/median/max", min(n), sorted(n)[len(n) // 2], max(n))
    print("correctness prompt tokens", [w["queries"][k]["n_prompt"] for k in w["correctness"]])


if __name__ == "__main__":
    main()
