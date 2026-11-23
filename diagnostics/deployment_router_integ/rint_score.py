"""ROUTER-INTEG scorer (analysis machine, CPU only). Never runs on a laboratory host.

Scores every measurement row with the UNCHANGED benchmark evaluator of the router study's repository:
    EvaluatorFactory().get_evaluator(<benchmark>).evaluate(item, raw_output),  score = float(is_correct)
The item is what the evaluator reads: the stored reference answer under the evaluator's own field name, or, for
KorBench and MBPP, the benchmark's own data row (verified LFS copy). MBPP runs only inside the Bubblewrap sandbox of
the winner-label stability experiment. Nothing of an evaluator is edited.

  score      --edge-run DIR --private FILE --repo DIR --out FILE
             one score row per measurement row (by id). A row with status "error" is not scored: it gets score 0 and
             scored = false. An answer cut off at the cutoff or the cap is scored as it stands. Rows are never edited.
  self-test  --private FILE --repo DIR
             no model output is involved: the STORED original outputs of the workload queries go through the same
             scoring path and must reproduce the stored scores exactly.

Run with the router study's own environment (its evaluators' dependencies).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rint_common as C  # noqa: E402
import rint_record as REC  # noqa: E402
import rint_spec as S  # noqa: E402
import rint_workload as WL  # noqa: E402

ROW_FILES = ("rows_profile_edge.jsonl", "rows_profile_remote.jsonl", "rows_shared_edge.jsonl", "rows_shared_remote.jsonl",
             "rows_live_A.jsonl", "rows_live_B.jsonl")


def kandk_item(ground_truth):
    """The Knights-and-Knaves evaluator reads `solution_text_format` and drops the first three characters of each
    line (the '(1)' numbering). The stored reference is the list it obtains; three filler characters restore the form."""
    return {"solution_text_format": "\n".join("-- " + str(a) for a in ground_truth)}


class Scorer:
    def __init__(self, repo):
        self.repo = Path(repo).expanduser().resolve()
        sys.path.insert(0, str(self.repo))
        sys.path.insert(0, str(self.repo / "diagnostics" / "scripts"))
        import wls_common as W
        self.W = W
        bad = [k for k, ok in W.verify_lfs_copy().items() if not ok and ("KORBench" in k or "MBPP" in k)]
        if bad:
            raise SystemExit(f"STOP: the verified benchmark data copy is incomplete: {bad[:3]}")
        W.point_evaluators_at_lfs()
        from evaluation import EvaluatorFactory
        self.factory, self.evs, self.data = EvaluatorFactory(), {}, {}

    def ev(self, dataset):
        if dataset not in self.evs:
            self.evs[dataset] = self.factory.get_evaluator(dataset)
        return self.evs[dataset]

    def item(self, dataset, index, prompt, ground_truth):
        if dataset in WL.LFS_BENCHMARKS:
            if dataset not in self.data:
                self.data[dataset] = self.ev(dataset).load_data("test")
            it = self.data[dataset][int(index) - 1]
            if it["prompt"] != prompt:
                raise RuntimeError(f"benchmark data row does not match the prompt of {dataset}:{index}")
            return it
        if dataset == "kandk":
            return kandk_item(ground_truth)
        if dataset not in WL.GT_FIELD:
            raise RuntimeError(f"no validated scorer for {dataset}")
        return {WL.GT_FIELD[dataset]: ground_truth}

    def score(self, jobs):
        """jobs: [{"id", "dataset", "index", "prompt", "ground_truth", "text"}] -> {id: {"score", "prediction", "eval_error", "scored_in"}}."""
        out, mbpp = {}, []
        for j in jobs:
            if j["dataset"] == "mbpp":
                mbpp.append((j["id"], int(j["index"]), j["prompt"], j["text"]))
                continue
            try:
                r = self.ev(j["dataset"]).evaluate(self.item(j["dataset"], j["index"], j["prompt"], j["ground_truth"]), j["text"])
                out[j["id"]] = {"score": float(r["is_correct"]), "prediction": str(r.get("prediction"))[:500], "eval_error": None, "scored_in": "host"}
            except Exception as e:                  # reported, never converted into a score
                out[j["id"]] = {"score": None, "prediction": None, "eval_error": f"{type(e).__name__}: {e}"[:300], "scored_in": "host"}
        if mbpp:
            import wls_preflight_score as PS
            res, ok = PS.score_mbpp_in_sandbox(mbpp)
            for jid, _i, _p, _t in mbpp:
                r = res.get(jid)
                if r is None or not ok:
                    out[jid] = {"score": None, "prediction": None, "eval_error": "no sandbox result", "scored_in": "bwrap_sandbox"}
                elif "error" in r:
                    out[jid] = {"score": None, "prediction": None, "eval_error": r["error"][:300], "scored_in": "bwrap_sandbox"}
                else:
                    out[jid] = {"score": float(r["score"]), "prediction": str(r.get("prediction"))[:500], "eval_error": None,
                                "scored_in": "bwrap_sandbox", "sandbox_result": r.get("result")}
        return out


def private_rows(path):
    return {json.loads(l)["key"]: json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()}


def cmd_score(a):
    priv = private_rows(a.private)
    rows = []
    for name in ROW_FILES:
        rs, torn = C.read_jsonl(Path(a.edge_run) / name)
        rows += rs
    if not rows:
        raise SystemExit("STOP: no measurement rows")
    if any(r.get("dev") or r.get("mock") for r in rows) and not a.dev:
        raise SystemExit("STOP: dev or mock rows are refused as results")
    if Path(a.out).exists():
        raise SystemExit(f"STOP: {a.out} exists: scores are written once")
    sc = Scorer(a.repo)
    jobs = [{"id": r["id"], "dataset": r["dataset"], "index": priv[r["key"]]["index"], "prompt": priv[r["key"]]["prompt"],
             "ground_truth": priv[r["key"]]["ground_truth"], "text": r["text"]}
            for r in rows if r["status"] == "ok" and r["phase"] != "warmup"]
    res = sc.score(jobs)
    n_err = 0
    for r in rows:
        if r["phase"] == "warmup":
            continue
        base = {"id": r["id"], "key": r["key"], "config": r["config"], "phase": r["phase"], "status": r["status"], "stop": r["stop"]}
        if r["status"] != "ok":
            base.update(score=0.0, scored=False, why_not_scored="the request failed", prediction=None, eval_error=None)
        else:
            s = res[r["id"]]
            if s["eval_error"]:
                n_err += 1
            base.update(score=s["score"], scored=s["eval_error"] is None, prediction=s["prediction"], eval_error=s["eval_error"],
                        scored_in=s["scored_in"], why_not_scored=s["eval_error"])
        C.append_jsonl(a.out, base)
    print(json.dumps({"rows": len(rows), "scored": len(jobs), "evaluator_errors": n_err, "out": str(a.out)}))
    sys.exit(0 if n_err == 0 else 5)


def cmd_self_test(a):
    """The stored original outputs of the workload queries through the same scoring path: must reproduce the stored scores."""
    priv = private_rows(a.private)
    sc = Scorer(a.repo)
    jobs, want = [], {}
    stored = {}
    for key, p in priv.items():
        for m in S.MODEL_ORDER:
            name = S.MODELS[m]["name"]
            if (p["dataset"], m) not in stored:
                stored[(p["dataset"], m)], _ = WL.stored_records(sc.repo, p["dataset"], name)
            r = stored[(p["dataset"], m)][int(p["index"])]
            jid = f"{key}|{m}"
            jobs.append({"id": jid, "dataset": p["dataset"], "index": p["index"], "prompt": p["prompt"], "ground_truth": p["ground_truth"],
                         "text": r["raw_output"]})
            want[jid] = float(r["score"])
    res = sc.score(jobs)
    bad = [(k, want[k], res[k]["score"], res[k]["eval_error"]) for k in want if res[k]["score"] != want[k]]
    per = {}
    for j in jobs:
        d = per.setdefault(j["dataset"], [0, 0])
        d[0] += 1
        d[1] += res[j["id"]]["score"] == want[j["id"]]
    print(json.dumps({"stored_outputs_rescored": len(jobs), "reproduced": len(jobs) - len(bad), "by_benchmark": per, "mismatches": bad[:10]}, indent=1))
    sys.exit(0 if not bad else 5)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("score")
    p.add_argument("--edge-run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dev", action="store_true")
    p.set_defaults(fn=cmd_score)
    q = sp.add_parser("self-test")
    q.set_defaults(fn=cmd_self_test)
    for x in (p, q):
        x.add_argument("--private", required=True)
        x.add_argument("--repo", required=True)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
