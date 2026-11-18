"""K16-HWFIT correctness rule K16-CORRECT-v1 (k16_spec.CRITERIA), as one pure function.

Inputs are per-sequence arrays against the unsplit reference (E0's e0_q.compare_to_ref) and greedy token rows.
The rule fails closed: a missing sequence, an empty position set or a non-finite value is a FAIL, never a skip.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # python -I: the script dir is not on sys.path
import k16_spec as S  # noqa: E402

sys.path.insert(0, str(S.E0_DIR))
import e0_q as Q  # noqa: E402  (first_divergence, is_degenerate: E0's Q2 helpers, unchanged)


def evaluate(expected_ids, teacher, greedy_ref, greedy_x, vocab_len, criteria=None):
    """expected_ids: the check prompt ids, in order.
    teacher[id] = {"agree": bool[n], "kl": float[n], "ref_margin": float[n], "finite": bool}
    greedy_ref[id] = {"tokens": [...], "margins": [...]};  greedy_x[id] = {"tokens": [...], "finite": bool}"""
    c = criteria or S.CRITERIA
    out = {"criteria_id": c["id"], "thresholds": {k: c[k] for k in ("decisive_margin_nats", "max_decisive_disagreement",
                                                                    "max_mean_kl_nats", "max_greedy_fail_share")},
           "n_sequences_expected": len(expected_ids)}
    missing = [i for i in expected_ids if i not in teacher or i not in greedy_ref or i not in greedy_x]
    out["missing"] = missing
    if missing or not expected_ids:
        out.update({"pass": False, "why": "incomplete check set"})
        return out
    agree = np.concatenate([np.asarray(teacher[i]["agree"], dtype=bool) for i in expected_ids])
    kl = np.concatenate([np.asarray(teacher[i]["kl"], dtype=np.float64) for i in expected_ids])
    margin = np.concatenate([np.asarray(teacher[i]["ref_margin"], dtype=np.float64) for i in expected_ids])
    finite = bool(all(teacher[i]["finite"] for i in expected_ids) and all(greedy_x[i].get("finite", True) for i in expected_ids)
                  and np.isfinite(kl).all() and np.isfinite(margin).all())
    decisive = margin >= c["decisive_margin_nats"]
    n_dec = int(decisive.sum())
    dis_dec = int((~agree & decisive).sum())
    share = dis_dec / n_dec if n_dec else float("nan")
    wrong = margin[~agree]
    out["teacher_forced"] = {
        "n_positions": int(agree.size), "n_decisive": n_dec, "decisive_disagreements": dis_dec, "decisive_disagreement_share": share,
        "all_position_disagreement_pp": float(100.0 * (1.0 - agree.mean())) if agree.size else float("nan"),
        "largest_reference_margin_at_a_disagreement_nats": float(wrong.max()) if wrong.size else None,
        "mean_kl_nats": float(kl.mean()) if kl.size else float("nan"), "p99_kl_nats": float(np.quantile(kl, 0.99)) if kl.size else float("nan"),
        "max_kl_nats": float(kl.max()) if kl.size else float("nan")}
    per, n_ok, degenerate = [], 0, []
    for i in expected_ids:
        r, x = greedy_ref[i], greedy_x[i]
        d = Q.first_divergence(r["tokens"], x["tokens"])
        m = None if d is None or d >= len(r["margins"]) else r["margins"][d]
        ok = d is None or (m is not None and m < c["decisive_margin_nats"])
        dx, why = Q.is_degenerate(x["tokens"], vocab_len)
        dr, _ = Q.is_degenerate(r["tokens"], vocab_len)
        if dx and not dr:
            degenerate.append(i)
        n_ok += bool(ok)
        per.append({"id": i, "n_tokens": len(x["tokens"]), "first_divergence": d, "reference_margin_there_nats": m,
                    "identical_or_near_tie": bool(ok), "degenerate": bool(dx), "why": why})
    n = len(expected_ids)
    allowed = math.floor(c["max_greedy_fail_share"] * n + 1e-9)
    out["greedy"] = {"n_prompts": n, "identical_or_near_tie": n_ok, "identical": sum(p["first_divergence"] is None for p in per),
                     "failures_allowed": allowed, "degenerate_only_in_configuration": degenerate, "per_prompt": per}
    out["checks"] = {
        "C1_finite": finite,
        "C2_decisive_agreement": bool(n_dec > 0 and share <= c["max_decisive_disagreement"]),
        "C3_mean_kl": bool(kl.size > 0 and np.isfinite(kl).all() and kl.mean() <= c["max_mean_kl_nats"]),
        "C4_greedy": bool(n - n_ok <= allowed and not degenerate and all(len(greedy_x[i]["tokens"]) == len(greedy_ref[i]["tokens"]) for i in expected_ids)),
    }
    out["pass"] = bool(all(out["checks"].values()))
    return out
