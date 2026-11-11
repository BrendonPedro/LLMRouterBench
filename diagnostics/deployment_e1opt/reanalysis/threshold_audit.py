"""Audit of the server-speedup threshold search (2026-11-11 correction pass).

Re-uses the frozen E0 gate code (e0_r.py, commit 81461b6) unchanged. Reports, for the cell that the
earlier bisection selected, unrounded Delta and CI at a list of scaling factors f, checks whether the
displayed "+0.00" CI lower bound is exactly zero, and re-runs the three threshold searches with the gate's
own pass predicate (Delta >= 3.0 and CI_low > 0, strict). Also checks per-cell monotonicity on a fine grid.
Usage: python -I threshold_audit.py <stage1 dir> <stage2 dir> <out dir>
"""
import json, sys
from pathlib import Path
import numpy as np

E0 = Path.home() / "research/LLMRouterBench/diagnostics/deployment_e0"
sys.path.insert(0, str(E0))
import e0_r as R
import e0_spec as S

S1DIR, S2DIR, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)

data = R.S1Data()
W = R.boot_weights(data.strata)
fres = json.loads((S1DIR / "f_results.json").read_text())
blocked = set(fres.get("no_split_pass_classes", []))
umax = {int(m): tuple(v) for m, v in fres["u_max"].items() if v and int(m) not in blocked}
eligible = {int(m): v for m, v in fres["edge_eligible"].items()}
edge_prof = R.edge_profiles_from_dir(S1DIR)
eseg_k = R.load_seg_profiles(S1DIR, R.EDGE_KEYS, "P-edge-U(")
sseg_k = R.load_seg_profiles(S2DIR, R.SERVER_KEYS, "P-server-U(")
disc = S.UB_SEGMENT_DISCOUNT
edge_seg = {(M, k): R.scaled(eseg_k[k], disc) for M, (k, j) in umax.items() if k in eseg_k}


def gate(f):
    server_seg = {(M, k): R.scaled(sseg_k[k], disc * f) for M, (k, j) in umax.items()}
    return R.upper_bound_gate(data, edge_seg, server_seg, edge_prof, eligible, W, umax)


def nrt(g):
    return [c for c in g["cells"] if c["status"] == "evaluated" and c["privacy"] == "no-raw-text"]


def cell(g, M, D):
    return next(c for c in nrt(g) if c["M"] == M and c["D_s"] == D)


report = {}

# ---- G. f = 1 reproduces every evaluated G-UB2 cell exactly --------------------------------------
ref = json.loads((S2DIR.parent / "gates" / "gate_ub2.json").read_text())
g1 = gate(1.0)
ev1 = [c for c in g1["cells"] if c["status"] == "evaluated"]
evr = [c for c in ref["cells"] if c["status"] == "evaluated"]
assert len(ev1) == len(evr) and len(ev1) > 0
rep = []
for a, b in zip(ev1, evr):
    assert (a["M"], a["privacy"], a["D_s"]) == (b["M"], b["privacy"], b["D_s"])
    rep.append({"cell": f"{a['M']}GB/{a['privacy']}/D{a['D_s']}", "delta_equal": a["delta_pt"] == b["delta_pt"],
                "ci_equal": a["ci95"] == b["ci95"], "pass_equal": a["pass"] == b["pass"],
                "split_best_equal": a["split_best"] == b["split_best"], "comp_best_equal": a["comp_best"] == b["comp_best"]})
report["f1_reproduction"] = {"n_evaluated": len(ev1), "all_exact": all(all(v for k, v in r.items() if k != "cell") for r in rep),
                            "decision": g1["decision"], "ref_decision": ref["decision"], "cells": rep}
print("G. f=1 reproduces gate_ub2 exactly (bitwise ==):", report["f1_reproduction"]["all_exact"])

# ---- C. pass predicate in the gate code ----------------------------------------------------------
import inspect
src = inspect.getsource(R.evaluate_grid)
line = [l.strip() for l in src.splitlines() if '"pass"' in l and "ci95" in l][0]
report["pass_predicate_source"] = line
print("C. gate pass predicate:", line)

# ---- A, B, D. the fixed cell at the requested f values --------------------------------------------
FS = [0.710, 0.705, 0.700, 0.695, 0.690, 0.689, 0.680, 0.675, 0.650, 0.600]
rows = []
for f in FS:
    g = gate(f)
    c = cell(g, 12, 60)
    lo, hi = c["ci95"]
    rows.append({"f": f, "server_step_ms_k3_discounted": 1e3 * sseg_k[3].a * disc * f,
                 "split_best": c["split_best"], "comp_best": c["comp_best"],
                 "U_split": max(c["U_split"].values()), "U_comp": max(c["U_comp"].values()),
                 "delta_pt": c["delta_pt"], "ci_lo": lo, "ci_hi": hi, "ci_lo_repr": repr(lo),
                 "ci_lo_sign": "exactly zero" if lo == 0.0 else ("positive" if lo > 0 else "negative"),
                 "pass_gate": c["pass"], "n_pass_all_cells": sum(x["pass"] for x in nrt(g)),
                 "n_ge3_all_cells": sum(x["delta_pt"] >= 3.0 for x in nrt(g))})
    r = rows[-1]
    print(f"f={f:.3f} U_split={r['U_split']:.6f} U_comp={r['U_comp']:.6f} delta={r['delta_pt']:+.6f} "
          f"ci=[{lo!r}, {hi!r}] sign={r['ci_lo_sign']} pass={c['pass']} n_pass={r['n_pass_all_cells']}")
report["fixed_cell_12GB_D60"] = rows

# ---- E. monotonicity on a fine grid, per cell -------------------------------------------------------
fine = np.round(np.arange(1.0, -0.0001, -0.005), 4).tolist()
per_cell = {}
viol = []
prev = None
for f in fine:
    g = gate(f)
    cur = {f"{c['M']}GB/D{c['D_s']}": (c["delta_pt"], c["ci95"][0], c["ci95"][1], c["pass"], max(c["U_split"].values()))
           for c in nrt(g)}
    if prev is not None:
        for key, (d, lo, hi, p, us) in cur.items():
            d0, lo0, hi0, p0, us0 = prev[key]
            if us < us0 - 1e-12: viol.append(("U_split decreased", key, f, us0, us))
            if d < d0 - 1e-12: viol.append(("delta decreased", key, f, d0, d))
            if lo < lo0 - 1e-12: viol.append(("ci_lo decreased", key, f, lo0, lo))
            if p0 and not p: viol.append(("pass lost", key, f))
    for key, v in cur.items():
        per_cell.setdefault(key, []).append({"f": f, "delta_pt": v[0], "ci_lo": v[1], "ci_hi": v[2], "pass": v[3], "U_split": v[4]})
    prev = cur
report["monotonicity"] = {"grid_step": 0.005, "n_points": len(fine), "violations": viol, "monotone": not viol}
print("E. per-cell monotonicity violations on 0.005 grid:", viol if viol else "none")
(OUT / "fine_grid.json").write_text(json.dumps(per_cell, indent=1))

# ---- F. threshold search with the gate's own predicate, at float precision --------------------------
def stat(f, key, M=None, D=None):
    ev = [c for c in nrt(gate(f)) if (M is None or c["M"] == M) and (D is None or c["D_s"] == D)]
    if key == "delta_ge3": return any(c["delta_pt"] >= 3.0 for c in ev)
    if key == "ci_lo_gt0": return any(c["ci95"][0] > 0 for c in ev)
    if key == "pass_strict": return any(c["pass"] for c in ev)                                # gate predicate
    if key == "pass_nonstrict": return any(c["delta_pt"] >= 3.0 and c["ci95"][0] >= 0 for c in ev)  # earlier bisection
    raise KeyError(key)


def largest_f(pred, lo=0.0, hi=1.0, it=40):
    """Largest f with pred(f) True, assuming pred is True on [0, f*] and False on (f*, 1] (monotone)."""
    if pred(hi): return hi, hi
    if not pred(lo): return None, None
    for _ in range(it):
        mid = (lo + hi) / 2
        if pred(mid): lo = mid
        else: hi = mid
    return lo, hi   # pred(lo) True, pred(hi) False, hi - lo = 2**-40


thr = {}
for label, (M, D) in {"any_cell": (None, None), "12GB_D60": (12, 60), "16GB_D60": (16, 60),
                      "12GB_D600": (12, 600), "16GB_D600": (16, 600)}.items():
    thr[label] = {}
    for key in ("delta_ge3", "ci_lo_gt0", "pass_strict", "pass_nonstrict"):
        lo, hi = largest_f(lambda f: stat(f, key, M, D))
        thr[label][key] = {"f_true": lo, "f_false": hi,
                           "speedup": None if lo is None else 1.0 / lo,
                           "server_step_ms_k3_raw": None if lo is None else 1e3 * sseg_k[3].a * lo,
                           "server_step_ms_k3_discounted": None if lo is None else 1e3 * sseg_k[3].a * disc * lo}
    print(label, {k: (None if v["f_true"] is None else round(v["f_true"], 6)) for k, v in thr[label].items()})
report["thresholds_strict"] = thr

# ---- F2. floating-point comparison in the frozen gate code -------------------------------------------
# e0_r.py compares delta_pt >= 3.0 on floats. With binary scores and 200 x 2 queries, delta is a multiple of
# 0.25 pt, but (0.485 - 0.455) * 100 evaluates to 2.9999999999999973. The code therefore counts an exact 3.00 as
# a fail. Report both the code-as-frozen threshold (above) and the threshold with delta rounded to 1e-6 pt.
def any_ge3_rounded(f):
    return any(round(c["delta_pt"], 6) >= 3.0 for c in nrt(gate(f)))
lo_r, hi_r = largest_f(any_ge3_rounded)
c83 = cell(gate(0.83), 12, 60)
report["float_comparison"] = {
    "f_delta_ge3_rounded_1e-6": lo_r, "speedup_rounded": 1.0 / lo_r,
    "server_step_ms_k3_raw_rounded": 1e3 * sseg_k[3].a * lo_r,
    "example_f_0.83": {"U_split": max(c83["U_split"].values()), "U_comp": max(c83["U_comp"].values()),
                       "delta_repr": repr(c83["delta_pt"]), "code_counts_ge3": c83["delta_pt"] >= 3.0,
                       "rounded_counts_ge3": round(c83["delta_pt"], 6) >= 3.0},
    "ci_lo_repr_near_zero": {str(f): repr(cell(gate(f), 12, 60)["ci95"][0]) for f in (0.71, 0.70, 0.689, 0.688766)},
}
print("F2. delta >= 3 with rounding to 1e-6 pt: f =", round(lo_r, 6), "speedup", round(1 / lo_r, 4),
      "; at f=0.83 delta repr", repr(c83["delta_pt"]), "code ge3:", c83["delta_pt"] >= 3.0)

# Which cell passes first (strict), and what its values are at the threshold
fp = thr["any_cell"]["pass_strict"]["f_true"]
g = gate(fp)
first = [c for c in nrt(g) if c["pass"]]
report["first_pass_cells_at_threshold"] = [{"cell": f"{c['M']}GB/D{c['D_s']}", "delta_pt": c["delta_pt"], "ci95": c["ci95"],
                                           "split_best": c["split_best"], "comp_best": c["comp_best"]} for c in first]
print("first strict-pass cell(s) at f=%.6f:" % fp, report["first_pass_cells_at_threshold"])
(OUT / "threshold_audit.json").write_text(json.dumps(report, indent=1, default=str))
