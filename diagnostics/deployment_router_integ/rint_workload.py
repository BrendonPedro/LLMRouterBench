"""ROUTER-INTEG workload: the frozen evaluation sample, the profiling queries and the pinned token ids.

Sampling frame: the router study's pool of 11,467 queries. Every one of them is held out for exactly one of the five
fold models (frozen fold assignment), and is scored only by that model. Removed from the frame, by rules that do not
look at any router output or any new model output:
  - benchmarks whose scorer executes model-written code without a validated sandbox (rint_spec.FRAME_EXCLUDED_BENCHMARKS);
  - FinQA rows whose reference answer is empty (the winner-label stability experiment's data-validity rule);
  - rows whose pool text does not match the stored original prompt of both models.
Sample: N_QUERIES queries, allocated proportionally (largest remainder) over benchmark x original outcome class
(both correct, both wrong, only DeepSeek correct, only Qwen3-8B correct), so that ties and both winner classes keep
their natural shares. Inside a stratum, and for every order, queries are ranked by a seeded SHA-256 (no RNG library).

  build   (desktop, maintainer, before the freeze; needs the router study's repository and both tokenizers)
          writes frozen/workload.json (public: keys, order, hashes, counts, pinned router decisions; no prompt text)
          and the private workload file (prompts, reference answers, token ids).
"""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rint_common as C  # noqa: E402
import rint_spec as S  # noqa: E402

WORKLOAD = C.RI_DIR / "frozen" / "workload.json"
SPLIT_OF = {"aime": "hybrid", "mmlupro": "test_1000"}
# The field the unchanged evaluator reads from its item. Verified by re-scoring every stored original output of both
# models: the stored score is reproduced for all rows of every benchmark listed (tools/rint_scorer_check.py).
GT_FIELD = {"aime": "Answer", "bbh": "target", "emorynlp": "answer", "finqa": "answer", "gpqa": "answer", "math500": "answer",
            "mathbench": "answer", "medqa": "answer", "meld": "answer", "mmlupro": "answer", "kandk": "solution_text_format"}
LFS_BENCHMARKS = ("korbench", "mbpp")      # their evaluators read the benchmark's own data files (verified LFS copy)


def tokenize(tok, prompt):
    """Token ids of one user turn under the checkpoint's own chat template with its defaults, generation prompt added.
    The template text is tokenised without adding special tokens, as the original server did for chat requests."""
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=S.ADD_GENERATION_PROMPT,
                                   **S.CHAT_TEMPLATE_KWARGS)
    return [int(x) for x in tok(text, add_special_tokens=False)["input_ids"]]


def ids_sha(ids):
    return C.sha256_bytes(json.dumps([int(x) for x in ids]).encode())


def load_tokenizer(model_dir):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)


def max_new(n_prompt):
    return max(0, min(S.MAX_NEW_TOKENS, S.CONTEXT_TOKENS - int(n_prompt)))


def class4(ds_score, qw_score):
    return {(1, 1): "both_correct", (0, 0): "both_wrong", (1, 0): "ds_only", (0, 1): "qw_only"}[(int(ds_score), int(qw_score))]


def stored_records(repo, dataset, model_name):
    import glob
    fs = glob.glob(str(Path(repo) / "results" / "bench" / dataset / SPLIT_OF.get(dataset, "test") / model_name / "*.json"))
    if len(fs) != 1:
        raise RuntimeError(f"expected one stored file for {dataset}/{model_name}, found {len(fs)}")
    return {int(r["index"]): r for r in C.read_json(fs[0])["records"]}, C.sha256_file(fs[0])


def frame(repo):
    """Rows of the sampling frame, with what excluded the others."""
    import csv
    repo = Path(repo)
    with open(repo / "diagnostics" / "stratified_cv" / "fold_assignments.csv", encoding="utf-8") as f:
        folds = list(csv.DictReader(f))
    with open(repo / "diagnostics" / "capability_profile" / "heldout_predictions.csv", encoding="utf-8") as f:
        preds = {(r["dataset"], int(r["index"])): r for r in csv.DictReader(f)}
    wls = json.loads((repo / "diagnostics" / "winner_label_stability" / "sampling_manifest.json").read_text())
    finqa_empty = {int(i) for i in wls["finqa_excluded_empty_answer"]}
    stored, files, rows, excluded = {}, {}, [], Counter()
    for f in folds:
        ds, idx = f["dataset"], int(f["index"])
        if ds in S.FRAME_EXCLUDED_BENCHMARKS:
            excluded[f"benchmark:{ds}"] += 1
            continue
        if ds == "finqa" and idx in finqa_empty:
            excluded["finqa_empty_reference"] += 1
            continue
        for m in S.MODEL_ORDER:
            if (ds, m) not in stored:
                stored[(ds, m)], files[f"{ds}|{S.MODELS[m]['name']}"] = stored_records(repo, ds, S.MODELS[m]["name"])
        a, b = stored[(ds, S.DS)].get(idx), stored[(ds, S.QW)].get(idx)
        if a is None or b is None or a["prompt"] != b["prompt"] or hashlib.sha1(a["prompt"].encode("utf-8")).hexdigest()[:16] != f["query_sha1"]:
            excluded["pool_text_not_matching_stored_prompt"] += 1
            continue
        if a["ground_truth"] != b["ground_truth"]:
            excluded["reference_differs_between_models"] += 1
            continue
        p = preds[(ds, idx)]
        if int(p["fold"]) != int(f["fold"]):
            raise RuntimeError(f"fold mismatch at {ds}:{idx}")
        rows.append({"key": f"{ds}:{idx}", "dataset": ds, "index": idx, "fold": int(f["fold"]), "query_sha1": f["query_sha1"],
                     "prompt": a["prompt"], "ground_truth": a["ground_truth"],
                     "orig": {S.DS: {"score": int(float(a["score"])), "completion_tokens": int(float(a["completion_tokens"])),
                                     "prompt_tokens": int(float(a["prompt_tokens"]))},
                              S.QW: {"score": int(float(b["score"])), "completion_tokens": int(float(b["completion_tokens"])),
                                     "prompt_tokens": int(float(b["prompt_tokens"]))}},
                     "class4": class4(float(a["score"]), float(b["score"])),
                     "router_score": float(p["B_student_score"]), "router_choice": S.ROUTER_CLASS_TO_MODEL[int(float(p["B_student_score"]) > S.ROUTER_THRESHOLD)]})
    return rows, dict(excluded), files


def allocate(counts, n):
    """Largest-remainder allocation of n over strata with `counts` members; ties by stratum name."""
    total = sum(counts.values())
    quota = {k: n * v / total for k, v in counts.items()}
    base = {k: int(q) for k, q in quota.items()}
    rest = n - sum(base.values())
    for k in sorted(counts, key=lambda k: (-(quota[k] - base[k]), k))[:rest]:
        base[k] += 1
    return base


def draw(rows):
    strata = defaultdict(list)
    for r in rows:
        strata[(r["dataset"], r["class4"])].append(r)
    alloc = allocate({k: len(v) for k, v in strata.items()}, S.N_QUERIES)
    chosen = []
    for k in sorted(strata):
        ranked = sorted(strata[k], key=lambda r: C.hash_rank(S.SAMPLE_SEED, r["key"]))
        chosen += ranked[:alloc[k]]
    keys = {r["key"] for r in chosen}
    rest = sorted((r for r in rows if r["key"] not in keys), key=lambda r: C.hash_rank(S.PROFILE_SEED, r["key"]))
    return chosen, rest[:S.N_PROFILE], {f"{k[0]}|{k[1]}": [len(strata[k]), alloc[k]] for k in sorted(strata)}


def orders(chosen):
    """The execution order of each pass, and which model runs first for each query (exactly half each way)."""
    out = {}
    for name, seed in S.ORDER_SEED.items():
        ranked = sorted(chosen, key=lambda r: C.hash_rank(seed, r["key"]))
        by_first = sorted(ranked, key=lambda r: C.hash_rank(seed + ":first", r["key"]))
        first = {r["key"]: (S.DS if i % 2 == 0 else S.QW) for i, r in enumerate(by_first)}
        out[name] = [[r["key"], first[r["key"]]] for r in ranked]
    return out


def build(repo, tok_dirs, private_out):
    rows, excluded, files = frame(repo)
    chosen, profile, alloc = draw(rows)
    toks = {m: load_tokenizer(tok_dirs[m]) for m in S.MODEL_ORDER}
    lookup = C.read_json(C.RI_DIR / "router" / "task_lookup.json")
    wl, body = assemble(chosen, profile, toks, lookup, rows, excluded, files, alloc)
    Path(private_out).write_bytes(body)
    WORKLOAD.write_text(json.dumps(wl, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return wl


def assemble(chosen, profile, toks, lookup, rows, excluded=None, files=None, alloc=None):
    """(public workload, private bytes) for the chosen evaluation rows and profiling rows. Also used by the CPU tests
    with fixture rows."""
    pub, priv, tok_mismatch = [], [], []
    for role, group in (("eval", chosen), ("profile", profile)):
        for r in group:
            ids = {m: tokenize(toks[m], r["prompt"]) for m in S.MODEL_ORDER}
            for m in S.MODEL_ORDER:
                if len(ids[m]) != r["orig"][m]["prompt_tokens"]:
                    tok_mismatch.append([r["key"], m, len(ids[m]), r["orig"][m]["prompt_tokens"]])
            lk = lookup[r["key"]]
            pub.append({"key": r["key"], "role": role, "dataset": r["dataset"], "index": r["index"], "fold": r["fold"],
                        "query_sha1": r["query_sha1"], "prompt_sha256": C.sha256_bytes(r["prompt"].encode("utf-8")),
                        "n_prompt": {m: len(ids[m]) for m in S.MODEL_ORDER}, "ids_sha256": {m: ids_sha(ids[m]) for m in S.MODEL_ORDER},
                        "max_new": {m: max_new(len(ids[m])) for m in S.MODEL_ORDER},
                        "labels_A_original": {"class4": r["class4"], "score": {m: r["orig"][m]["score"] for m in S.MODEL_ORDER},
                                              "completion_tokens": {m: r["orig"][m]["completion_tokens"] for m in S.MODEL_ORDER}},
                        "router_pinned": {"fold": r["fold"], "score": r["router_score"], "choice": r["router_choice"]},
                        "task_lookup_pinned": {"predicted_benchmark": lk[1], "D1_true_benchmark_choice": S.ROUTER_CLASS_TO_MODEL[lk[2]],
                                               "D2_predicted_benchmark_choice": S.ROUTER_CLASS_TO_MODEL[lk[3]]}})
            priv.append({"key": r["key"], "role": role, "dataset": r["dataset"], "index": r["index"], "prompt": r["prompt"],
                         "ground_truth": r["ground_truth"], "ids": ids})
    body = "".join(json.dumps(p, sort_keys=True, ensure_ascii=False) + "\n" for p in priv).encode("utf-8")
    ev = [p for p in pub if p["role"] == "eval"]
    wl = {"schema": S.SCHEMA, "protocol": S.PROTOCOL, "n_frame": len(rows), "frame_excluded": excluded, "n_eval": len(ev),
          "n_profile": len(pub) - len(ev), "strata_frame_and_allocated": alloc,
          "class4_counts": dict(Counter(p["labels_A_original"]["class4"] for p in ev)),
          "class4_frame_counts": dict(Counter(r["class4"] for r in rows)),
          "benchmark_counts": dict(Counter(p["dataset"] for p in ev)), "fold_counts": dict(Counter(str(p["fold"]) for p in ev)),
          "max_prompt_tokens": max(max(p["n_prompt"].values()) for p in pub),
          "prompt_token_count_mismatches_vs_original_records": tok_mismatch,
          "orders": orders(chosen), "profile_keys": [p["key"] for p in pub if p["role"] == "profile"],
          "warmup_key": [p["key"] for p in pub if p["role"] == "profile"][0],
          "stored_original_files_sha256": files, "private_sha256": C.sha256_bytes(body), "queries": pub}
    return wl, body


def public():
    return C.read_json(WORKLOAD)


def load(private_path, dev_workload=None):
    """(public workload, {key: private row}). The private file must be the pinned one."""
    wl = C.read_json(dev_workload) if dev_workload else public()
    body = Path(private_path).read_bytes()
    if not dev_workload and C.sha256_bytes(body) != wl["private_sha256"]:
        raise C.Stop("the private workload file is not the pinned one")
    priv = {}
    for line in body.decode("utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            priv[r["key"]] = r
    for q in wl["queries"]:
        p = priv.get(q["key"])
        if p is None or any(ids_sha(p["ids"][m]) != q["ids_sha256"][m] for m in S.MODEL_ORDER):
            raise C.Stop(f"private workload does not match the public one at {q['key']}")
    return wl, priv


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--repo", required=True)
    b.add_argument("--tok-ds", required=True)
    b.add_argument("--tok-qw", required=True)
    b.add_argument("--private-out", required=True)
    a = ap.parse_args(argv)
    if WORKLOAD.exists():
        raise SystemExit(f"{WORKLOAD} exists: a frozen workload is never replaced")
    wl = build(a.repo, {S.DS: a.tok_ds, S.QW: a.tok_qw}, a.private_out)
    print(json.dumps({k: wl[k] for k in ("n_frame", "frame_excluded", "n_eval", "n_profile", "class4_counts", "class4_frame_counts",
                                         "benchmark_counts", "fold_counts", "max_prompt_tokens", "private_sha256")}, indent=1))
    print("prompt-token mismatches against the original records:", len(wl["prompt_token_count_mismatches_vs_original_records"]))


if __name__ == "__main__":
    main()
