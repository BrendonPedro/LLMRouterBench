import json, sys
from pathlib import Path
import numpy as np
E0 = Path.home() / "research/LLMRouterBench/diagnostics/deployment_e0"
sys.path.insert(0, str(E0))
import e0_r as R, e0_spec as S
S1DIR, S2DIR, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
data = R.S1Data(); W = R.boot_weights(data.strata)
fres = json.loads((S1DIR / "f_results.json").read_text())
umax = {int(m): tuple(v) for m, v in fres["u_max"].items() if v and int(m) not in set(fres.get("no_split_pass_classes", []))}
eligible = {int(m): v for m, v in fres["edge_eligible"].items()}
edge_prof = R.edge_profiles_from_dir(S1DIR)
eseg = R.load_seg_profiles(S1DIR, R.EDGE_KEYS, "P-edge-U("); sseg = R.load_seg_profiles(S2DIR, R.SERVER_KEYS, "P-server-U(")
d = S.UB_SEGMENT_DISCOUNT
prof = json.load(open(OUT / "profiles.json"))
print("== decode-step constants (ms): a (intercept) and b*1000 tokens of context")
for k, v in prof.items():
    print(f"{k:40s} a={1e3*v['a_s']:7.2f} ms  b*1000ctx={1e3*v['b_s_per_ctx_token']*1000:6.2f} ms  pre(256)={1e3*np.interp(256, v['pre_n'], v['pre_s']):8.1f} ms pre(1024)={1e3*np.interp(1024, v['pre_n'], v['pre_s']):8.1f} ms")
# roofline-equivalent f at k=3 and k=5
for k in (3, 5):
    rl = R.server_lb_profile("H200", k, k)
    print(f"k={k}: measured a={1e3*sseg[k].a:.2f} ms, H200 roofline a={1e3*rl.a:.2f} ms -> roofline/measured = {rl.a/sseg[k].a:.3f}; measured/roofline = {sseg[k].a/rl.a:.2f}x")
# per-query view of the best cell: 12 GB, D=60, SPLIT nonthinking vs EDGE 8B-AWQ thinking
M, k = 12, umax[12][0]
split_q = (S.SPLIT_MODEL, "nonthinking")
print("\n== 12 GB, D=60 s, SPLIT|nonthinking, per-query (200 q x 2 seeds)")
for f in (1.0, 0.83, 0.705, 0.688766, 0.5, 0.0):
    sp = R.scaled(eseg[k], d) + R.scaled(sseg[k], d * f)
    v, T = R.ontime_correct(data, split_q, sp, 60)
    sc = data.score[split_q]
    ontime = (T <= 60)
    print(f"f={f:5.3f}: U={v.mean():.4f}  on-time share={ontime.mean():.3f}  correct&on-time={((sc>0)&ontime).mean():.3f}  correct total={(sc>0).mean():.3f}  T median={np.median(T):6.1f}s p75={np.percentile(T,75):6.1f}s p90={np.percentile(T,90):6.1f}s")
sp1 = R.scaled(eseg[k], d) + R.scaled(sseg[k], d)
v1, T1 = R.ontime_correct(data, split_q, sp1, 60)
N = data.N[split_q]; n_in = data.n_in[split_q]; sc = data.score[split_q]
print("N tokens (32B nonthinking): median", np.median(N), "p75", np.percentile(N, 75), "p90", np.percentile(N, 90), "max", N.max())
print("n_in median", np.median(n_in), "max", n_in.max())
# how many correct answers sit in the 60-s window between f=1 and the thresholds (0.8297: delta >= 3; 0.6888: strict pass)
for f2 in (0.829669, 0.705394, 0.688766):
    sp2 = R.scaled(eseg[k], d) + R.scaled(sseg[k], d * f2)
    v2, T2 = R.ontime_correct(data, split_q, sp2, 60)
    flip = ((T1 > 60) & (T2 <= 60))
    print(f"queries x seeds crossing the deadline when f 1 -> {f2}:", int(flip.sum()), "of which correct:", int(((sc > 0) & flip).sum()), f"U {v1.mean():.4f} -> {v2.mean():.4f}")
# edge competitor numbers
for key in [("Qwen3-8B-AWQ", "thinking"), ("Qwen3-8B-AWQ", "nonthinking")]:
    ep = edge_prof[(key[0], 12)]
    for D in (30, 60, 120, 300, 600):
        v, T = R.ontime_correct(data, key, ep, D)
        print(f"EDGE {key} D={D}: U={v.mean():.4f} ontime={(T<=D).mean():.3f} correct={(data.score[key]>0).mean():.3f}", end=" | ")
    print("N median", np.median(data.N[key]), "p90", np.percentile(data.N[key], 90))
# per-token time at median ctx for SPLIT at f=1
ctx = 800
print(f"\nSPLIT per-token at ctx={ctx}: edge seg {1e3*d*(eseg[k].a+eseg[k].b*ctx):.1f} ms + server {1e3*d*(sseg[k].a+sseg[k].b*ctx):.1f} ms = {1e3*(sp1.a+sp1.b*ctx):.1f} ms (bound incl. 0.9 discounts); competitor 8B-AWQ cap12: {1e3*(edge_prof[('Qwen3-8B-AWQ',12)].a+edge_prof[('Qwen3-8B-AWQ',12)].b*ctx):.1f} ms")

# GB10-bandwidth server bound (public spec 273 GB/s; prefill compute kept at H200 peak to stay optimistic)
S.GPU_PEAKS["GB10-bw-only"] = {"hbm_Bps": 273e9, "bf16_flops": 989e12}
server_gb10 = {(M_, k_): R.server_lb_profile("GB10-bw-only", k_, j_) for M_, (k_, j_) in umax.items()}
edge_seg = {(M_, k_): R.scaled(eseg[k_], d) for M_, (k_, j_) in umax.items()}
g = R.upper_bound_gate(data, edge_seg, server_gb10, edge_prof, eligible, W, umax)
ev = [c for c in g["cells"] if c["status"] == "evaluated" and c["privacy"] == "no-raw-text"]
best = max(ev, key=lambda c: c["delta_pt"])
print("\n== Server = 273 GB/s bandwidth roofline (GB10 public spec), prefill at H200 FLOPs, zero link, edge measured x0.9:")
print(g["decision"]); print("best", best["M"], best["D_s"], best["split_best"], round(best["delta_pt"], 2), best["ci95"])
for c in ev: print(f"  {c['M']}GB D={c['D_s']:3d}: delta={c['delta_pt']:+6.2f} lo={c['ci95'][0]:+6.2f}")
for k_ in (3, 5): print(f"GB10 roofline decode step a at k={k_}: {1e3*server_gb10[(12 if k_==3 else 16, k_)].a:.1f} ms")
json.dump({"gb10_bw_bound": {"decision": g["decision"], "best": {kk: best[kk] for kk in ("M", "D_s", "split_best", "delta_pt", "ci95")}, "cells": [{kk: c[kk] for kk in ("M", "D_s", "delta_pt", "ci95", "split_best", "comp_best")} for c in ev]}}, open(OUT / "gb10_bound.json", "w"), indent=1)
