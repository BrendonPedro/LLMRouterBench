"""CPU-only reanalysis of the archived E0 Stage-1/Stage-2 profiles.

Reuses the frozen gate code (e0_r.py, commit 81461b6) unchanged. The only change is a scaling factor f
applied to the Stage-2 measured server-segment profile before the gate's own 0.90 discount:
    server time = 0.90 * f * measured     (f = 1 reproduces gate_ub2.json exactly)
No inference. No protocol change. E0 files are not modified.
2026-11-11 correction: the full-pass bisection uses the gate's own per-cell `pass` flag (delta >= 3.0 AND
ci_lo > 0, strict). The first version used min(delta - 3, ci_lo) >= 0, which accepts ci_lo == 0 exactly; the
bootstrap distribution is quantised to 0.25 pt, so ci_lo sits at exactly 0.0 for f in (0.6888, 0.7054] in the
12 GB / 60 s cell and that version reported 0.705 instead of 0.689. See threshold_audit.py.
"""
import json, sys, copy
from pathlib import Path
import numpy as np

E0 = Path.home() / "research/LLMRouterBench/diagnostics/deployment_e0"
sys.path.insert(0, str(E0))
import e0_r as R
import e0_spec as S

S1DIR = Path(sys.argv[1]); S2DIR = Path(sys.argv[2]); OUT = Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)

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

def gate(f, extra_step_s=0.0):
    """Bound gate with server profile scaled by f; extra_step_s adds a constant per forward step (what-if link)."""
    server_seg = {}
    for M, (k, j) in umax.items():
        p = R.scaled(sseg_k[k], disc * f)
        if extra_step_s:
            # one synchronous round trip per forward step: per decode step + per prefill chunk (ceil(n/2048))
            grid = p.pre_n
            p = R.Profile(grid, p.pre_s + extra_step_s * np.ceil(grid / 2048.0), p.a + extra_step_s, p.b, p.label)
        server_seg[(M, k)] = p
    return R.upper_bound_gate(data, edge_seg, server_seg, edge_prof, eligible, W, umax)

# 0. Reproduce gate_ub2 exactly at f = 1
g1 = gate(1.0)
ref = json.loads((S2DIR.parent / "gates" / "gate_ub2.json").read_text())
ok = abs(g1["max_delta_bound_pt"] - ref["max_delta_bound_pt"]) < 1e-9
same = all(abs(a["delta_pt"] - b["delta_pt"]) < 1e-9 and abs(a["ci95"][0] - b["ci95"][0]) < 1e-9
           for a, b in zip([c for c in g1["cells"] if c["status"] == "evaluated"],
                           [c for c in ref["cells"] if c["status"] == "evaluated"]))
print("f=1 reproduces archived gate_ub2:", ok and same, g1["decision"])

# 1. Profile constants (seconds) for the record
prof = {}
for k, p in sorted(sseg_k.items()):
    prof[f"server_k{k}"] = {"a_s": p.a, "b_s_per_ctx_token": p.b, "pre_n": p.pre_n.tolist(), "pre_s": p.pre_s.tolist()}
for k, p in sorted(eseg_k.items()):
    prof[f"edge_seg_k{k}"] = {"a_s": p.a, "b_s_per_ctx_token": p.b, "pre_n": p.pre_n.tolist(), "pre_s": p.pre_s.tolist()}
for (m, M), p in sorted(edge_prof.items()):
    prof[f"edge_competitor_{m}_cap{M}"] = {"a_s": p.a, "b_s_per_ctx_token": p.b, "pre_n": p.pre_n.tolist(), "pre_s": p.pre_s.tolist()}
for M, (k, j) in umax.items():
    rl = R.server_lb_profile("H200", k, j)
    prof[f"server_roofline_H200_k{k}"] = {"a_s": rl.a, "b_s_per_ctx_token": rl.b, "pre_n": rl.pre_n.tolist(), "pre_s": rl.pre_s.tolist()}
(OUT / "profiles.json").write_text(json.dumps(prof, indent=1))

# 2. Curve over f
fs = [1.0, 0.9, 0.83, 0.8, 0.75, 0.705, 0.7, 0.69, 0.689, 0.688, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0]
rows = []
for f in fs:
    g = gate(f)
    ev = [c for c in g["cells"] if c["status"] == "evaluated" and c["privacy"] == "no-raw-text"]
    best = max(ev, key=lambda c: c["delta_pt"])
    row = {"f": f, "server_step_ms_k3": 1e3 * sseg_k[3].a * f, "server_step_ms_k5": 1e3 * sseg_k[5].a * f,
           "best_delta_pt": best["delta_pt"], "best_ci_lo": best["ci95"][0], "best_cell": (best["M"], best["D_s"], best["split_best"], best["comp_best"]),
           "n_pass": sum(c["pass"] for c in ev), "n_ge3": sum(c["delta_pt"] >= 3.0 for c in ev),
           "cells": {f"{c['M']}GB/D{c['D_s']}": [round(c["delta_pt"], 3), round(c["ci95"][0], 3), c["split_best"], round(max(c["U_split"].values()), 4), round(max(c["U_comp"].values()), 4)] for c in ev}}
    rows.append(row)
    print(f"f={f:5.2f} k3 step={row['server_step_ms_k3']:6.1f} ms  best={best['delta_pt']:+6.2f} lo={best['ci95'][0]:+6.2f} pass={row['n_pass']:2d} ge3={row['n_ge3']:2d} cell={row['best_cell']}")
(OUT / "curve.json").write_text(json.dumps(rows, indent=1))

# 3. Thresholds by bisection on the best no-raw-text cell (and on the 12/16 GB, D = 60 s cells specifically)
def best_stat(f, M=None, D=None, key="delta_pt"):
    g = gate(f)
    ev = [c for c in g["cells"] if c["status"] == "evaluated" and c["privacy"] == "no-raw-text"
          and (M is None or c["M"] == M) and (D is None or c["D_s"] == D)]
    if key == "delta_pt":
        return max(c["delta_pt"] for c in ev)
    if key == "ci_lo":
        return max(c["ci95"][0] for c in ev)
    if key == "pass":            # the gate's own predicate: delta >= 3.0 and ci_lo > 0 (strict), per cell
        return 1.0 if any(c["pass"] for c in ev) else -1.0
    if key == "pass_nonstrict":  # the 2026-11-11 first-pass predicate (ci_lo >= 0); kept only as a diagnostic
        return max((min(c["delta_pt"] - 3.0, c["ci95"][0]) for c in ev))

def bisect(target, M=None, D=None, key="delta_pt", lo=0.0, hi=1.0, it=40):
    """Largest f (least speedup) such that stat(f) >= target. Every stat is monotone non-increasing in f
    (threshold_audit.py checks this on a 0.005 grid). 40 halvings: the interval is below 1e-12."""
    if best_stat(hi, M, D, key) >= target: return hi
    if best_stat(lo, M, D, key) < target: return None
    for _ in range(it):
        mid = (lo + hi) / 2
        if best_stat(mid, M, D, key) >= target: lo = mid
        else: hi = mid
    return lo

thr = {}
for label, (M, D) in {"any_cell": (None, None), "12GB_D60": (12, 60), "16GB_D60": (16, 60), "12GB_D600": (12, 600), "16GB_D600": (16, 600)}.items():
    thr[label] = {
        "f_for_delta_ge_0": bisect(0.0, M, D, "delta_pt"),
        "f_for_delta_ge_3": bisect(3.0, M, D, "delta_pt"),
        "f_for_ci_lo_gt_0": bisect(1e-9, M, D, "ci_lo"),
        "f_for_full_pass": bisect(0.0, M, D, "pass"),                      # gate predicate (strict ci_lo > 0)
        "f_for_pass_nonstrict_ci_ge_0": bisect(0.0, M, D, "pass_nonstrict"),  # diagnostic only; NOT the gate
    }
    for name in ("f_for_delta_ge_3", "f_for_full_pass"):
        fv = thr[label][name]
        if fv is not None:
            thr[label][name + "_speedup"] = 1.0 / fv
            thr[label][name + "_server_step_ms_k3_raw"] = 1e3 * sseg_k[3].a * fv
            thr[label][name + "_server_step_ms_k3_discounted"] = 1e3 * sseg_k[3].a * disc * fv
    print(label, {k: (None if v is None else round(v, 6)) for k, v in thr[label].items()})
(OUT / "thresholds.json").write_text(json.dumps(thr, indent=1))

# 4. What-if: link round trip added per forward step (NOT part of the gate; the gate is zero-network by design)
link = {}
for rtt_ms in (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0):
    link[rtt_ms] = {}
    for f in (1.0, 0.5, 0.25, 0.1, 0.0):
        g = gate(f, rtt_ms / 1e3)
        ev = [c for c in g["cells"] if c["status"] == "evaluated" and c["privacy"] == "no-raw-text"]
        best = max(ev, key=lambda c: c["delta_pt"])
        link[rtt_ms][f] = {"best_delta_pt": round(best["delta_pt"], 3), "ci_lo": round(best["ci95"][0], 3), "cell": (best["M"], best["D_s"], best["split_best"]), "n_pass": sum(c["pass"] for c in ev)}
    print("rtt", rtt_ms, {f: v["best_delta_pt"] for f, v in link[rtt_ms].items()})
(OUT / "link_whatif.json").write_text(json.dumps({str(k): v for k, v in link.items()}, indent=1))
