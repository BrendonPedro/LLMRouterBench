"""GEMMA-SELECT workload: the fixed query sets, built once from the S1 sample and pinned by hash (PREREG §4).

  python -I -B gsel_workload.py build     # writes workload/workload.json (refuses to replace it)
  python -I -B gsel_workload.py show
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gsel_common as C  # noqa: E402
import gsel_spec as S  # noqa: E402


def sample_rows(repo=None):
    p = Path(repo or C.REPO) / S.S1_SAMPLE_REL
    if C.sha256_file(p) != S.S1_SAMPLE_SHA256:
        raise C.Stop("the S1 sample is not the pinned file")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def build(rows):
    rows = [r for r in rows if r["dataset"] in S.BENCHMARKS]
    keys = sorted(r["key"] for r in rows)
    order = list(keys)
    random.Random(S.ORDER_SEED).shuffle(order)
    sentinel = set()
    for b, n in S.SENTINEL_PER_BENCH.items():
        sentinel |= {r["key"] for r in sorted((r for r in rows if r["dataset"] == b), key=lambda r: (r["rank"], r["key"]))[:n]}
    by_len = sorted(rows, key=lambda r: (len(r["prompt"]), r["key"]))
    correctness = [r["key"] for r in by_len[:S.N_CORRECTNESS_SHORT]] + [r["key"] for r in by_len[-S.N_CORRECTNESS_LONG:]]
    return {"schema": S.SCHEMA, "experiment": S.NAME, "sample_sha256": S.S1_SAMPLE_SHA256, "benchmarks": list(S.BENCHMARKS),
            "utility": order, "sentinel": [k for k in order if k in sentinel], "correctness": correctness,
            "queries": {r["key"]: {"dataset": r["dataset"], "prompt_sha256": r["prompt_sha256"]} for r in rows}}


def load(dev=False, repo=None):
    """{"utility": [query], "sentinel": [query], "correctness": [query]} with prompt and answer; hashes verified."""
    path = C.GS_DIR / S.WORKLOAD_FILE
    if not dev and C.sha256_file(path) != S.WORKLOAD_SHA256:
        raise C.Stop("workload.json is not the pinned file")
    w = C.read_json(path)
    rows = {r["key"]: r for r in sample_rows(repo)}
    for k, q in w["queries"].items():
        if C.sha256_bytes(rows[k]["prompt"].encode("utf-8")) != q["prompt_sha256"]:
            raise C.Stop(f"prompt of {k} differs from the workload")
    q = lambda k: {"key": k, "dataset": rows[k]["dataset"], "prompt": rows[k]["prompt"], "answer": rows[k]["answer"]}  # noqa: E731
    return {name: [q(k) for k in w[name]] for name in ("utility", "sentinel", "correctness")}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    w = build(sample_rows())
    if cmd == "build":
        C.write_json(C.GS_DIR / S.WORKLOAD_FILE, w)
        print("sha256", C.sha256_file(C.GS_DIR / S.WORKLOAD_FILE))
    print({k: len(w[k]) for k in ("utility", "sentinel", "correctness")})
