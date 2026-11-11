"""Server bandwidth-roofline scan with the archived RTX 3090 edge and competitor profiles (E0 Stage 1).
Scope: this bound holds for the archived edge profiles only. A different edge device changes both the split's edge
segment and the competitors, so it is not a statement about every possible target edge.
"""
import json, sys
from pathlib import Path
import numpy as np
E0 = Path.home() / "research/LLMRouterBench/diagnostics/deployment_e0"
sys.path.insert(0, str(E0))
import e0_r as R, e0_spec as S
S1DIR, OUT = Path(sys.argv[1]), Path(sys.argv[2])
data = R.S1Data(); W = R.boot_weights(data.strata)
fres = json.loads((S1DIR / "f_results.json").read_text())
umax = {int(m): tuple(v) for m, v in fres["u_max"].items() if v and int(m) not in set(fres.get("no_split_pass_classes", []))}
eligible = {int(m): v for m, v in fres["edge_eligible"].items()}
edge_prof = R.edge_profiles_from_dir(S1DIR)
eseg = R.load_seg_profiles(S1DIR, R.EDGE_KEYS, "P-edge-U(")
d = S.UB_SEGMENT_DISCOUNT
edge_seg = {(M, k): R.scaled(eseg[k], d) for M, (k, j) in umax.items()}
def run(bw, eff=1.0, flops=989e12):
    S.GPU_PEAKS["scan"] = {"hbm_Bps": bw * eff, "bf16_flops": flops}
    srv = {(M, k): R.server_lb_profile("scan", k, j) for M, (k, j) in umax.items()}
    g = R.upper_bound_gate(data, edge_seg, srv, edge_prof, eligible, W, umax)
    ev = [c for c in g["cells"] if c["status"] == "evaluated" and c["privacy"] == "no-raw-text"]
    best = max(ev, key=lambda c: c["delta_pt"])
    fp = any(c["pass"] for c in ev)   # the gate's predicate (delta >= 3.0 and ci_lo > 0, strict); corrected 2026-11-11
    return best["delta_pt"], best["ci95"][0], fp, srv[(12, 3)].a
rows = []
for name, bw in [("GB10 / DGX Spark (public spec)", 273e9), ("RTX 4090", 1008e9), ("RTX 3090", 936e9), ("RTX 6000 Ada", 960e9), ("RTX PRO 6000 Blackwell", 1792e9), ("A100 80GB SXM", 2039e9), ("H100 PCIe", 2000e9), ("H100 SXM", 3350e9), ("H200", 4800e9)]:
    for eff in (1.0, 0.7, 0.5):
        dlt, lo, fp, a = run(bw, eff)
        rows.append({"gpu": name, "bw_GBps": bw / 1e9, "eff": eff, "server_step_ms_k3": 1e3 * a, "best_delta": round(dlt, 2), "ci_lo": round(lo, 2), "full_pass": fp})
        print(f"{name:32s} {bw/1e9:6.0f} GB/s eff={eff:.1f}: step(k=3)={1e3*a:6.1f} ms best={dlt:+6.2f} lo={lo:+6.2f} pass={fp}")
# bisection on bandwidth at eff=1 for delta>=3 and full pass
def bis(target_fn, lo=100e9, hi=5000e9):
    for _ in range(30):
        mid = (lo + hi) / 2
        if target_fn(mid): hi = mid
        else: lo = mid
    return hi
b3 = bis(lambda bw: run(bw)[0] >= 3.0); bp = bis(lambda bw: run(bw)[2])
print(f"min effective bandwidth for delta>=3 in some cell: {b3/1e9:.0f} GB/s (step k=3 {1e3*run(b3)[3]:.1f} ms); for a full pass: {bp/1e9:.0f} GB/s (step {1e3*run(bp)[3]:.1f} ms)")
json.dump({"rows": rows, "min_eff_bw_delta3_GBps": b3 / 1e9, "min_eff_bw_fullpass_GBps": bp / 1e9}, open(OUT / "bw_threshold.json", "w"), indent=1)
