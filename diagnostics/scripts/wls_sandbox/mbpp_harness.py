"""Score MBPP outputs with the UNCHANGED MBPPEvaluator.evaluate. Runs ONLY inside wls_mbpp_sandbox.sh.

Input  /sbx/in/inputs.jsonl: {"key", "index", "prompt", "raw_output"} per line.
Output stdout lines "RESULT\t{json}" and a final "DONE".

As in the 2026-11-03 reproduction, the only additions are the DATA_DIR redirect and a
pass-through wrapper that records check_correctness' own result string without altering it.
An evaluator exception is reported as an error, never as a score.
"""
import json
import sys
import time

sys.path.insert(0, "/sbx/code")
import evaluation.MBPP.mbpp as mb  # noqa: E402  (importing the package sets the spawn start method)

mb.DATA_DIR = "/sbx/data/MBPP"
_orig = mb.check_correctness
_last = {}


def _recording_check_correctness(*a, **k):
    r = _orig(*a, **k)
    _last["result"] = r["result"]
    return r


mb.check_correctness = _recording_check_correctness


def main():
    import multiprocessing
    print("START_METHOD\t" + multiprocessing.get_start_method(), flush=True)
    ev = mb.MBPPEvaluator()
    data = ev.load_data("test")
    assert len(data) == 974
    for line in open("/sbx/in/inputs.jsonl", encoding="utf-8"):
        x = json.loads(line)
        item = data[x["index"] - 1]
        rec = {"key": x["key"], "index": x["index"]}
        if item["prompt"] != x["prompt"]:
            rec["error"] = "prompt does not match load_data('test')[index-1]"
        else:
            _last.clear()
            t0 = time.time()
            try:
                out = ev.evaluate(item, x["raw_output"])
                rec.update(score=float(out["is_correct"]), prediction=out["prediction"],
                           result=_last.get("result"))
            except Exception as e:  # reported, never converted into a score
                rec["error"] = f"{type(e).__name__}: {e}"
            rec["seconds"] = round(time.time() - t0, 3)
        print("RESULT\t" + json.dumps(rec), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
