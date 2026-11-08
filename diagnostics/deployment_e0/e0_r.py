"""E0-R: on-time utility per cell, paired bootstrap, upper-bound gates, profile fitting, E0 decision.

Subcommands
  prior      informational pre-hardware R prior: S1 per-query (score, tokens) + spec-sheet roofline
             profiles (NOT a measurement, decides nothing)
  gate-ub1   after Stage 1: SPLIT time lower-bounded by the measured edge-segment compute (x 0.90) + the
             server's datasheet roofline + zero link; STOP only if no bound cell reaches delta >= 3.0 pt
  gate-ub2   after Stage 2: same with the measured server-segment compute (x 0.90) in place of the roofline
  fit        fit per-cell profiles + the additive SPLIT model from E0-L logs; held-out validation
  r          evaluate every R cell from measured profiles (E0 §8 R) -> R_cells.{json,md}
  decide     combine S1, F, Q, R into DECISION.{json,md} (E0 §8 consequence table)

Utility (PROBLEM-FORMULATION §3.4): U(a, D) = mean over the 200 S1 queries and both seeds of
score(q, s) * 1[T_a(q, s) <= D], with T from the profile and the per-query (n_in, N) of that action.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e0_instrument as I  # noqa: E402
import e0_memory as MEM  # noqa: E402
import e0_spec as S  # noqa: E402

BENCH_ORDER = ("mmlupro", "gpqa", "livemathbench", "livecodebench")


# --------------------------------------------------------------------------- #
# S1 per-query data                                                           #
# --------------------------------------------------------------------------- #
class S1Data:
    def __init__(self, path=S.S1_Q3_SCORES):
        s1 = S.load_s1()
        self.keys = [json.loads(x)["key"] for x in open(s1.SAMPLE_JSONL)]
        idx = {k: i for i, k in enumerate(self.keys)}
        self.strata = np.array([BENCH_ORDER.index(k.split(":")[0]) for k in self.keys])
        nq = len(self.keys)
        self.score, self.N, self.n_in = {}, {}, {}
        for r in I.read_jsonl(path):
            a = (r["model"], r["mode"])
            if a not in self.score:
                self.score[a] = np.full((nq, 2), np.nan)
                self.N[a] = np.full((nq, 2), np.nan)
                self.n_in[a] = np.full(nq, np.nan)
            q = idx[f"{r['dataset']}:{r['index']}"]
            self.score[a][q, r["seed"]] = r["score"]
            self.N[a][q, r["seed"]] = r["completion_tokens"]
            self.n_in[a][q] = r["prompt_tokens"]
        for a in self.score:
            if np.isnan(self.score[a]).any() or np.isnan(self.N[a]).any():
                raise ValueError(f"S1 data incomplete for {a}")


# --------------------------------------------------------------------------- #
# Profiles (seconds)                                                          #
# --------------------------------------------------------------------------- #
class Profile:
    """T(n_in, N) = pre(n_in) + sum_{i=1}^{N-1} (a + b * (n_in + i)).

    pre(n) is the time to the first token (prefill + first token), piecewise-linear through measured
    points (linear extrapolation outside). Decode step i attends to n_in + i tokens.
    """

    def __init__(self, pre_n, pre_s, a, b, label=""):
        o = np.argsort(pre_n)
        self.pre_n, self.pre_s = np.asarray(pre_n, float)[o], np.asarray(pre_s, float)[o]
        self.a, self.b, self.label = float(a), float(b), label

    def pre(self, n):
        shape = np.shape(n)
        n = np.atleast_1d(np.asarray(n, float))
        x, y = self.pre_n, self.pre_s
        if len(x) == 1:
            return np.full(shape, y[0]) if shape else float(y[0])
        out = np.interp(n, x, y)
        lo, hi = n < x[0], n > x[-1]
        out[lo] = y[0] + (n[lo] - x[0]) * (y[1] - y[0]) / (x[1] - x[0])
        out[hi] = y[-1] + (n[hi] - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2])
        out = np.maximum(out, 0.0)
        return out.reshape(shape) if shape else float(out[0])

    def T(self, n_in, N):
        n_in, N = np.asarray(n_in, float), np.asarray(N, float)
        m = np.maximum(N - 1, 0)
        return self.pre(n_in) + m * (self.a + self.b * n_in) + self.b * m * (m + 1) / 2

    def __add__(self, o):
        grid = np.union1d(self.pre_n, o.pre_n)
        return Profile(grid, self.pre(grid) + o.pre(grid), self.a + o.a, self.b + o.b, f"{self.label}+{o.label}")

    def to_json(self):
        return {"pre_n": self.pre_n.tolist(), "pre_s": self.pre_s.tolist(), "a": self.a, "b": self.b,
                "label": self.label}

    @staticmethod
    def from_json(d):
        return Profile(d["pre_n"], d["pre_s"], d["a"], d["b"], d.get("label", ""))


def roofline_decode(bytes_static, kv_bytes_per_token, bw, eff):
    """(a, b) of a memory-bound decode step: weights read once per token, KV grows with context."""
    return bytes_static / (bw * eff), kv_bytes_per_token / (bw * eff)


def roofline_profile(bytes_static, kv_per_tok, flops_per_tok, bw, flops, eff_bw, eff_flops, label,
                     const_s=0.0, first_token_step=True):
    """first_token_step adds one decode step to TTFT (a full model emits token 1 after its prefill pass).
    Lower bounds on a split segment set it False: the segment's prefill pass already yields token 1."""
    a, b = roofline_decode(bytes_static, kv_per_tok, bw, eff_bw)
    ns = np.array([64.0, 256.0, 2048.0, 17664.0])
    pre = np.maximum(ns * flops_per_tok / (flops * eff_flops), bytes_static / (bw * eff_bw)) + const_s
    if first_token_step:
        pre = pre + a
    return Profile(ns, pre, a + const_s, b, label)


# --------------------------------------------------------------------------- #
# Utility and bootstrap                                                       #
# --------------------------------------------------------------------------- #
def boot_weights(strata, n_boot=S.R_BOOT, seed=S.R_BOOT_SEED):
    """Stratified (by benchmark) resampling of queries with replacement -> counts [n_boot, n_q]."""
    rng = np.random.Generator(np.random.PCG64(seed))
    nq = len(strata)
    W = np.zeros((n_boot, nq), dtype=np.float64)
    for s in np.unique(strata):
        members = np.flatnonzero(strata == s)
        draws = rng.integers(0, len(members), size=(n_boot, len(members)))
        for b in range(n_boot):
            np.add.at(W[b], members[draws[b]], 1.0)
    return W


def ontime_correct(data, action_q, profile, D):
    """Per-query mean over seeds of score * 1[T <= D] for the quality source action_q = (model, mode)."""
    sc, N, n_in = data.score[action_q], data.N[action_q], data.n_in[action_q]
    T = profile.T(n_in[:, None], N)
    return (sc * (T <= D)).mean(axis=1), T


def cell_stats(v_split, v_comp, W):
    """Delta = max_split U - max_comp U; bootstrap re-selects both maxima inside each replicate."""
    nq = W.shape[1]
    Us = {k: v.mean() for k, v in v_split.items()}
    Uc = {k: v.mean() for k, v in v_comp.items()}
    s_best = max(Us, key=Us.get)
    c_best = max(Uc, key=Uc.get) if Uc else None
    Ms = W @ np.stack(list(v_split.values()), 1) / nq
    bs = Ms.max(1)
    if Uc:
        Mc = W @ np.stack(list(v_comp.values()), 1) / nq
        bc = Mc.max(1)
    else:
        bc = np.zeros_like(bs)
    d_star = (bs - bc) * 100
    delta = (Us[s_best] - (Uc[c_best] if c_best else 0.0)) * 100
    lo, hi = np.percentile(d_star, [2.5, 97.5])
    return {"split_best": s_best, "comp_best": c_best, "U_split": Us, "U_comp": Uc, "delta_pt": float(delta),
            "ci95": [float(lo), float(hi)], "d_star": d_star}


def evaluate_grid(data, split_prof, edge_prof, remote_prof, eligible, W, grid=None, label=""):
    """split_prof[(M, state, rtt)] -> Profile (missing key = cell not evaluable);
    edge_prof[(model, M)] -> Profile; remote_prof[(state, rtt)] -> Profile; eligible[M] -> [models]."""
    grid = grid or [(M, p, st, r, D) for M in S.R_MEMORY for p in S.R_PRIVACY for st in S.R_STATE
                    for r in S.R_RTT for D in S.R_DEADLINES_S]
    out = []
    for M, priv, st, rtt, D in grid:
        cell = {"M": M, "privacy": priv, "state": st, "rtt_ms": rtt, "D_s": D, "source": label}
        sp = split_prof.get((M, st, rtt))
        if sp is None:
            cell.update(status="not evaluable (no profile: F failed or not measured)")
            out.append(cell)
            continue
        v_split = {f"SPLIT|{md}": ontime_correct(data, (S.SPLIT_MODEL, md), sp, D)[0] for md in S.MODES}
        v_comp = {}
        for m in eligible.get(M, []):
            ep = edge_prof.get((m, M))
            if ep is None:
                continue
            for md in S.MODES:
                v_comp[f"EDGE:{m}|{md}"] = ontime_correct(data, (m, md), ep, D)[0]
        if priv == "open":
            rp = remote_prof.get((st, rtt))
            if rp is None:
                cell.update(status="not evaluable (no REMOTE profile)")
                out.append(cell)
                continue
            for md in S.MODES:
                v_comp[f"REMOTE:{S.SPLIT_MODEL}|{md}"] = ontime_correct(data, (S.SPLIT_MODEL, md), rp, D)[0]
        if not any(k.startswith("EDGE:") for k in v_comp) and priv == "no-raw-text":
            cell.update(status="excluded: no admissible EDGE competitor fits (capacity statement, not a regime)")
            out.append(cell)
            continue
        allU = [v.mean() for v in list(v_split.values()) + list(v_comp.values())]
        if max(allU) < S.R_MIN_UTILITY:
            cell.update(status=f"excluded: every action U < {S.R_MIN_UTILITY}")
            out.append(cell)
            continue
        st_ = cell_stats(v_split, v_comp, W)
        d_star = st_.pop("d_star")
        st_["pass"] = bool(st_["delta_pt"] >= S.R_MARGIN_PT and st_["ci95"][0] > 0)
        cell.update(st_, status="evaluated", _d_star=d_star)
        out.append(cell)
    ev = [c for c in out if c["status"] == "evaluated"]
    n = len(ev)
    for c in ev:   # multiplicity label (reported, not a decision input; PREREG §8.3)
        q = 100 * (0.05 / max(n, 1)) / 2
        c["bonferroni_lo"] = float(np.percentile(c.pop("_d_star"), q))
        c["bonferroni_pass"] = bool(c["pass"] and c["bonferroni_lo"] > 0)
    return out


def summarize_cells(cells):
    ev = [c for c in cells if c["status"] == "evaluated"]
    passing = [c for c in ev if c["pass"]]
    return {"n_cells": len(cells), "n_evaluated": len(ev), "n_pass": len(passing),
            "R_pass": bool(passing), "pass_robust_to_multiplicity": any(c["bonferroni_pass"] for c in passing),
            "best": max(ev, key=lambda c: c["delta_pt"]) if ev else None,
            "pass_classes": sorted({c["M"] for c in passing}),
            "only_12GB": bool(passing) and all(c["M"] == 12 for c in passing),
            "open_pass": any(c["privacy"] == "open" for c in passing)}


def cells_markdown(cells, title):
    rows = [f"# {title}", "", "| M | privacy | state | RTT | D s | SPLIT best (U) | best competitor (U) | "
            "Δ pt | 95% CI | pass | Bonferroni lo |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        rtt = "native" if c["rtt_ms"] is None else c["rtt_ms"]
        if c["status"] != "evaluated":
            rows.append(f"| {c['M']} | {c['privacy']} | {c['state']} | {rtt} | {c['D_s']} | {c['status']} |"
                        " | | | | |")
            continue
        us, uc = c["U_split"][c["split_best"]], (c["U_comp"][c["comp_best"]] if c["comp_best"] else 0.0)
        rows.append(f"| {c['M']} | {c['privacy']} | {c['state']} | {rtt} | {c['D_s']} | {c['split_best']} "
                    f"({us:.3f}) | {c['comp_best']} ({uc:.3f}) | {c['delta_pt']:+.1f} | "
                    f"[{c['ci95'][0]:+.1f}, {c['ci95'][1]:+.1f}] | {'PASS' if c['pass'] else '-'} | "
                    f"{c['bonferroni_lo']:+.1f} |")
    return "\n".join(rows) + "\n"


def _json_cells(cells):
    return [{k: v for k, v in c.items() if not k.startswith("_")} for c in cells]


# --------------------------------------------------------------------------- #
# Profile fitting from E0-L logs                                              #
# --------------------------------------------------------------------------- #
def request_rows(path):
    return [r for r in I.read_jsonl(path) if "prompt_id" in r and r.get("ok") and not r.get("warmup")]


def decode_steps(r):
    """(ctx, seconds) for every decode step of a request from its per-token timestamps."""
    t = np.asarray(r["token_ns"], float) / 1e9
    i = np.arange(1, len(t))
    return r["n_in"] + i, np.diff(t)


def fit_e2e(rows, slope=None, label=""):
    """End-to-end profile of one measured cell: TTFT by n_in, decode a + b*ctx (b borrowed if given)."""
    by_n = {}
    for r in rows:
        by_n.setdefault(r["n_in"], []).append(r["token_ns"][0] / 1e9)
    ctx, dt = zip(*(decode_steps(r) for r in rows))
    ctx, dt = np.concatenate(ctx), np.concatenate(dt)
    if slope is None:
        b, a = np.polyfit(ctx, dt, 1)
    else:
        b, a = slope, float(np.mean(dt - slope * ctx))
    ns = sorted(by_n)
    return Profile(ns, [float(np.mean(by_n[n])) for n in ns], a, b, label), {
        "tpot_p50_s": float(np.median(dt)), "tpot_p95_s": float(np.percentile(dt, 95)), "n_steps": int(dt.size)}


def fit_component(rows, keys, label):
    """Profile of a sum of per-step split components (ns) recorded in r['steps'] (edge or server LB)."""
    by_n, ctx, dt = {}, [], []
    for r in rows:
        st = r["steps"]
        pre = [s for s in st if s["n"] > 1 or s["start"] < r["n_in"]]
        dec = [s for s in st if s["n"] == 1 and s["start"] >= r["n_in"]]
        by_n.setdefault(r["n_in"], []).append(sum(sum(s.get(k, 0) for k in keys) for s in pre) / 1e9)
        for s in dec:
            ctx.append(s["start"] + 1)
            dt.append(sum(s.get(k, 0) for k in keys) / 1e9)
    b, a = np.polyfit(np.asarray(ctx, float), np.asarray(dt, float), 1)
    ns = sorted(by_n)
    return Profile(ns, [float(np.mean(by_n[n])) for n in ns], a, b, label)


EDGE_KEYS = ("edge_head_ns", "edge_tail_ns", "lm_head_ns")
SERVER_KEYS = ("server_compute_ns",)


class AdditiveModel:
    """Additive per-step SPLIT model fitted on component timers of the non-held-out SPLIT cells (PREREG §7.3)."""

    def fit(self, rows):
        E, Sv, X, Nn = [], [], [], []
        PE, PS, PX = [], [], []
        load_steps = {}
        for r in rows:
            Le, Ls = r["k"] + r["j"], S.QWEN3_32B_CONFIG["num_hidden_layers"] - r["k"] - r["j"]
            rho = r["rtt_ms"] or 0.0
            bw = (r["bw_mbps"] or 1000) * 1e6
            for s in r["steps"]:
                e = (s["edge_head_ns"] + s["edge_tail_ns"] + s.get("lm_head_ns", 0)) / 1e9
                sv = s["server_compute_ns"] / 1e9
                x = (s["ser_ns"] + s["server_ns"] - s["server_compute_ns"]) / 1e9
                nn = (s["net_ns"] + s["shape_ns"]) / 1e9 - (s["bytes_out"] + s["bytes_in"]) * 8 / bw
                n = s["n"]
                if n == 1 and s["start"] >= r["n_in"]:
                    ctx = s["start"] + 1
                    E.append((1, Le, Le * ctx, e))
                    X.append(x)
                    Nn.append((1, rho, nn))
                    if r["load"] == 0:
                        Sv.append((1, Ls, Ls * ctx, sv))
                    else:
                        load_steps.setdefault(r["load"], []).append((Ls, ctx, sv))
                elif r["load"] == 0:
                    PE.append((1, Le * n, e))
                    PS.append((1, Ls * n, sv))
                    PX.append((1, n, x))
        lst = lambda rows_, k: np.linalg.lstsq(np.array([q[:k] for q in rows_], float),
                                               np.array([q[k] for q in rows_], float), rcond=None)[0]
        self.e = lst(E, 3)
        self.s = lst(Sv, 3)
        self.x = float(np.mean(X))
        self.n = lst(Nn, 2)
        self.pe, self.ps, self.px = lst(PE, 2), lst(PS, 2), lst(PX, 2)
        self.r = {0: 1.0}
        for ld, st in load_steps.items():
            pred = np.array([self.s[0] + self.s[1] * Ls + self.s[2] * Ls * c for Ls, c, _ in st])
            self.r[ld] = float(np.mean([q[2] for q in st]) / np.mean(pred))
        return self

    def step(self, Le, ctx, rtt, bw_mbps, load):
        Ls = S.QWEN3_32B_CONFIG["num_hidden_layers"] - Le
        r = self.r[load]
        bits = 2 * (S.QWEN3_32B_CONFIG["hidden_size"] * 2 + S.WIRE_HEADER_BYTES) * 8
        return (self.e[0] + self.e[1] * Le + self.e[2] * Le * ctx + r * (self.s[0] + self.s[1] * Ls + self.s[2] * Ls * ctx)
                + self.x + self.n[0] + self.n[1] * (rtt or 0) + bits / (bw_mbps * 1e6))

    def profile(self, Le, rtt, bw_mbps, load, label=""):
        Ls = S.QWEN3_32B_CONFIG["num_hidden_layers"] - Le
        r = self.r[load]
        a = self.step(Le, 0, rtt, bw_mbps, load)
        b = self.e[2] * Le + r * self.s[2] * Ls
        H = S.QWEN3_32B_CONFIG["hidden_size"]
        ns = np.array([64.0, 256.0, 1024.0, 2048.0])
        pre = []
        for n in ns:
            t, left = 0.0, n
            while left > 0:
                c = min(S.PREFILL_CHUNK, left)
                t += (self.pe[0] + self.pe[1] * Le * c + r * (self.ps[0] + self.ps[1] * Ls * c) + self.px[0]
                      + self.px[1] * c + self.n[0] + self.n[1] * (rtt or 0)
                      + 2 * (c * H * 2 + S.WIRE_HEADER_BYTES) * 8 / (bw_mbps * 1e6))
                left -= c
            pre.append(t)
        return Profile(ns, pre, a, b, label)

    def to_json(self):
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in vars(self).items()}


def validate(model, remote_model, heldout_rows):
    """Held-out TPOT p50 within +-10% for every held-out cell (E0 §6)."""
    res = []
    for cid, rows in heldout_rows.items():
        r0 = rows[0]
        ctx, dt = zip(*(decode_steps(r) for r in rows))
        ctx, dt = np.concatenate(ctx), np.concatenate(dt)
        meas = float(np.median(dt))
        if r0["path"] == "SPLIT":
            pred = float(np.median([model.step(r0["k"] + r0["j"], c, r0["rtt_ms"], r0["bw_mbps"] or 1000,
                                               r0["load"]) for c in ctx]))
        else:
            a, b = remote_model[r0["load"]]
            pred = float(np.median(a + b * ctx))
        res.append({"cell": cid, "path": r0["path"], "tpot_p50_meas_s": meas, "tpot_p50_pred_s": pred,
                    "rel_err": pred / meas - 1, "ok": abs(pred / meas - 1) <= S.MODEL_VALIDATION_TOL})
    return {"cells": res, "valid": bool(res) and all(x["ok"] for x in res)}


def fit_all(lat_dir, f_results):
    """Fit everything R needs from the E0-L logs; returns a JSON-able profile bundle."""
    lat = Path(lat_dir)
    cells = {c["id"]: c for c in S.cells()}
    rows = {cid: request_rows(lat / f"{cid}.jsonl") for cid in cells if (lat / f"{cid}.jsonl").exists()}
    umax = {int(m): tuple(v) for m, v in json.loads(Path(f_results).read_text())["u_max"].items() if v}
    bundle = {"measured": {}, "validation": None}
    core_slope = {}
    for cid in ("core-01", "core-02", "core-03"):
        if rows.get(cid):
            p, st = fit_e2e(rows[cid], label=cid)
            core_slope[cells[cid]["path"]] = p.b
            bundle["measured"][cid] = {"profile": p.to_json(), **st}
    for cid, rs in rows.items():
        if cid in bundle["measured"] or not rs or cells[cid]["path"].startswith("EDGE"):
            continue
        path = cells[cid]["path"]
        slope = core_slope.get(path) if path in core_slope else core_slope.get("SPLIT U-max(16)")
        p, st = fit_e2e(rs, slope=slope if path != "REMOTE-HF" else None, label=cid)
        bundle["measured"][cid] = {"profile": p.to_json(), **st}
    split_train = [r for cid, rs in rows.items() if cells[cid]["path"].startswith("SPLIT")
                   and cells[cid]["block"] != "heldout" for r in rs]
    remote_lin = {}
    for ld, cid in ((0, "core-03"), (8, "load-03"), (32, "load-04")):
        if rows.get(cid):
            ctx, dt = zip(*(decode_steps(r) for r in rows[cid]))
            b, a = np.polyfit(np.concatenate(ctx), np.concatenate(dt), 1)
            remote_lin[ld] = (float(a), float(b))
    if split_train:
        am = AdditiveModel().fit(split_train)
        bundle["additive"] = am.to_json()
        ho = {cid: rs for cid, rs in rows.items() if cells[cid]["block"] == "heldout" and rs}
        bundle["validation"] = validate(am, remote_lin, ho)
        bundle["composed"] = {}
        for M, (k, j) in umax.items():
            for st, ld in (("good", 0), ("contended32", 32)):
                for rtt in S.R_RTT:
                    if ld in am.r:
                        bundle["composed"][f"{M}|{st}|{rtt}"] = am.profile(k + j, rtt, 1000, ld,
                                                                          f"model U({k},{j})").to_json()
    bundle["umax"] = {str(m): list(v) for m, v in umax.items()}
    return bundle


def split_profiles_from_bundle(bundle):
    """Profiles for the R grid: the validated additive model, else only exactly measured cells."""
    prof = {}
    if bundle.get("validation", {}) and bundle["validation"]["valid"]:
        for key, p in bundle["composed"].items():
            M, st, rtt = key.split("|")
            prof[(int(M), st, None if rtt == "None" else int(rtt))] = Profile.from_json(p)
        return prof, "additive model (validated on held-out cells)"
    m = bundle["measured"]
    umax16 = bundle["umax"].get("16")
    if umax16:
        for cid, key in (("core-02", (16, "good", None)), ("rtt-02", (16, "good", 20)),
                         ("load-02", (16, "contended32", None))):
            if cid in m:
                prof[key] = Profile.from_json(m[cid]["profile"])
    return prof, "measured cells only (additive model not validated)"


def remote_profiles_from_bundle(bundle):
    m = bundle["measured"]
    out = {}
    for cid, key in (("core-03", ("good", None)), ("rtt-05", ("good", 20)), ("load-04", ("contended32", None))):
        if cid in m:
            out[key] = Profile.from_json(m[cid]["profile"])
    if ("good", None) in out and ("contended32", None) in out and ("good", 20) in out:
        # contended + 20 ms: contended profile shifted by the measured good-state RTT shift of TTFT
        shift = float(out[("good", 20)].pre(np.array([256.0]))[0] - out[("good", None)].pre(np.array([256.0]))[0])
        c = out[("contended32", None)]
        out[("contended32", 20)] = Profile(c.pre_n, c.pre_s + shift, c.a, c.b, "load-04 + rtt shift")
    return out


def edge_profiles_from_dir(d):
    out = {}
    for f in Path(d).glob("P-edge-*-cap*.jsonl"):
        name = f.stem[len("P-edge-"):]
        if name.startswith("U("):          # echo edge-segment profiles, not EDGE competitors
            continue
        model, cap = name.rsplit("-cap", 1)
        rows = request_rows(f)
        if rows:
            out[(model, int(cap))] = fit_e2e(rows, label=f.stem)[0]
    return out


# --------------------------------------------------------------------------- #
# Gates                                                                       #
# --------------------------------------------------------------------------- #
def server_lb_profile(gpu, k, j):
    """Datasheet lower bound on the server segment: weights + KV at peak HBM bandwidth, FLOPs at peak."""
    pk = S.GPU_PEAKS[gpu]
    cfg = S.QWEN3_32B_CONFIG
    p = MEM.param_counts(cfg)
    Ls = cfg["num_hidden_layers"] - k - j
    w = Ls * p["layer"] * 2
    kv = Ls * MEM.kv_bytes_per_token_layer(cfg)
    return roofline_profile(w, kv, 2 * Ls * p["layer"], pk["hbm_Bps"], pk["bf16_flops"], 1.0, 1.0,
                            f"{gpu} roofline LB U({k},{j})", first_token_step=False)


def scaled(p, f):
    """Profile with every time multiplied by f (f < 1 turns a measured segment into a safer lower bound)."""
    return Profile(p.pre_n, p.pre_s * f, p.a * f, p.b * f, f"{p.label} x{f:g}")


def upper_bound_gate(data, edge_seg, server_seg, edge_prof, eligible, W, umax):
    """Bound on every R cell with SPLIT time lower-bounded (PREREG §9.2).

    U_SPLIT is monotone non-increasing in T, so lower-bounding T upper-bounds U_SPLIT per query and per
    bootstrap replicate, hence Delta and its CI. Contention and RTT > 0 only add SPLIT time; the EDGE
    competitors are the same measured profiles R uses; REMOTE (open cells) is bounded below by U = 0.
    STOP is allowed only if no bound cell reaches Delta >= 3.0 pt: then no real cell can pass at 200 queries
    or trigger the §8.5 extension. A class with a U-max but no segment profile makes the gate INCOMPLETE.
    """
    missing = sorted(M for M, (k, j) in umax.items() if (M, k) not in edge_seg or (M, k) not in server_seg)
    split_prof = {(M, "good", None): edge_seg[(M, k)] + server_seg[(M, k)]
                  for M, (k, j) in umax.items() if M not in missing}
    remote_zero = {("good", None): Profile([64.0], [1e12], 1e12, 0.0, "REMOTE bounded below by U = 0")}
    grid = [(M, pv, "good", None, D) for M in S.R_MEMORY for pv in S.R_PRIVACY for D in S.R_DEADLINES_S]
    cells = evaluate_grid(data, split_prof, edge_prof, remote_zero, eligible, W, grid, "upper bound")
    s = summarize_cells(cells)
    deltas = [c["delta_pt"] for c in cells if c["status"] == "evaluated"]
    best = max(deltas) if deltas else None
    if missing:
        decision = (f"INCOMPLETE: no segment profile for class(es) {missing}; measure it, then re-run the gate "
                    "(neither STOP nor CONTINUE)")
    elif best is not None and best >= S.R_MARGIN_PT:
        decision = ("CONTINUE: R is not excluded by the bound. This is NOT evidence that SPLIT meets any latency "
                    "or deadline requirement.")
    else:
        decision = "STOP: R fails by bound (no bound cell reaches delta >= 3.0 pt; SPLIT removed)"
    return {"cells": cells, "summary": s, "missing_profiles": missing, "max_delta_bound_pt": best,
            "R_possible": decision.startswith(("CONTINUE", "INCOMPLETE")), "decision": decision}


# --------------------------------------------------------------------------- #
# Informational prior (spec sheets; S1 per-query data)                         #
# --------------------------------------------------------------------------- #
PRIOR_EDGE_GPU = {"name": "RTX 3090 (capped)", "hbm_Bps": 936e9, "bf16_flops": 71e12}
PRIOR_SERVER_GPU = "H200"
PRIOR_EFF_BW, PRIOR_EFF_FLOPS = 0.7, 0.5


def prior_profiles(link_rtt_native_ms=0.5, per_step_overhead_ms=0.0):
    cfg = S.QWEN3_32B_CONFIG
    p = MEM.param_counts(cfg)
    e, sv = PRIOR_EDGE_GPU, S.GPU_PEAKS[PRIOR_SERVER_GPU]
    edge = {}
    for m, (wbytes, kvt) in MEM.EDGE_COMPETITORS.items():
        active = {"Qwen3-8B-AWQ": 8.2e9, "Qwen3-14B-AWQ": 14.8e9, "Qwen3-32B-AWQ": 32.8e9, "Qwen3-8B": 8.2e9}[m]
        for M in S.MEMORY_CLASSES:
            kv = kvt * (0.5 if (m, M) in S.FP8_KV_AT else 1.0)
            edge[(m, M)] = roofline_profile(wbytes, kv, 2 * active, e["hbm_Bps"], e["bf16_flops"], PRIOR_EFF_BW,
                                            PRIOR_EFF_FLOPS, f"roofline {m}")
    split, umax = {}, {M: S.U_MAX_PREDICTED[M] for M in S.MEMORY_CLASSES}
    for M, (k, j) in umax.items():
        Le = k + j
        we = (p["embed"] * 0 + p["lm_head"] + Le * p["layer"]) * 2       # embedding row read is negligible
        ep = roofline_profile(we, Le * 4096, 2 * (Le * p["layer"] + p["lm_head"]), e["hbm_Bps"], e["bf16_flops"],
                              PRIOR_EFF_BW, PRIOR_EFF_FLOPS, "edge seg")
        Ls = 64 - Le
        spf = roofline_profile(Ls * p["layer"] * 2, Ls * 4096, 2 * Ls * p["layer"], sv["hbm_Bps"], sv["bf16_flops"],
                               PRIOR_EFF_BW, PRIOR_EFF_FLOPS, "server seg")
        for rtt in S.R_RTT:
            r_s = ((rtt if rtt is not None else link_rtt_native_ms) + per_step_overhead_ms) / 1e3
            link = Profile([64.0], [r_s], r_s, 0.0, f"link {rtt}")
            split[(M, "good", rtt)] = ep + spf + link
    rem = {}
    full = roofline_profile(p["total"] * 2, 64 * 4096, 2 * p["total"], sv["hbm_Bps"], sv["bf16_flops"], PRIOR_EFF_BW,
                            PRIOR_EFF_FLOPS, "remote")
    for rtt in S.R_RTT:
        r_s = (rtt if rtt is not None else link_rtt_native_ms) / 1e3
        rem[("good", rtt)] = full + Profile([64.0], [r_s], 0.0, 0.0, "rtt once")
    return split, edge, rem, umax


def prior_markdown(cells, summary, note):
    head = ["# E0-R prior from S1 per-query data + spec-sheet roofline (informational; NOT a result)", "",
            note, "", f"Summary: {summary['n_pass']}/{summary['n_evaluated']} evaluated cells would pass; "
            f"best cell Δ = {summary['best']['delta_pt']:+.1f} pt at M={summary['best']['M']}, "
            f"{summary['best']['privacy']}, D={summary['best']['D_s']} s." if summary["best"] else "", ""]
    return "\n".join(head) + cells_markdown(cells, "Cells (good state)")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def load_seg_profiles(d, keys, prefix):
    out = {}
    for f in Path(d).glob(f"{prefix}*.jsonl"):
        rows = request_rows(f)
        if rows and rows[0].get("k"):
            out[rows[0]["k"]] = fit_component(rows, keys, f.stem)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prior")
    p.add_argument("--out", default=str(S.E0_DIR / "frozen" / "R_prior_s1_roofline.md"))
    p.add_argument("--boot", type=int, default=S.R_BOOT)
    p.add_argument("--overhead-ms", type=float, default=0.0,
                   help="extra per-token SPLIT time (runtime/serialisation overhead) for a sensitivity row")
    for name in ("gate-ub1", "gate-ub2"):
        g = sub.add_parser(name)
        g.add_argument("--stage1-dir", required=True, help="edge profiles + echo edge-segment logs + F results")
        g.add_argument("--stage2-dir", help="server-segment profile logs (gate-ub2)")
        g.add_argument("--server-gpu", choices=list(S.GPU_PEAKS), default="H200")
        g.add_argument("--out", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--lat-dir", required=True)
    f.add_argument("--f-results", required=True)
    f.add_argument("--out", required=True)
    r = sub.add_parser("r")
    r.add_argument("--bundle", required=True)
    r.add_argument("--stage1-dir", required=True)
    r.add_argument("--out-dir", required=True)
    d = sub.add_parser("decide")
    d.add_argument("--f-results", required=True)
    d.add_argument("--lat-dir", nargs="+", required=True, help="Stage 1 profile dir and Stage 3 cell dir")
    d.add_argument("--q", required=True)
    d.add_argument("--r", required=True)
    d.add_argument("--out-dir", required=True)
    d.add_argument("--a3b-added", action="store_true", help="R was re-evaluated with Qwen3-30B-A3B (PREREG §8.4 a)")
    a = ap.parse_args(argv)
    data = S1Data()

    if a.cmd == "prior":
        W = boot_weights(data.strata, a.boot)
        split, edge, rem, umax = prior_profiles(per_step_overhead_ms=a.overhead_ms)
        grid = [(M, pv, "good", rt, D) for M in S.R_MEMORY for pv in S.R_PRIVACY for rt in S.R_RTT
                for D in S.R_DEADLINES_S]
        cells = evaluate_grid(data, split, edge, rem, S.EDGE_ELIGIBLE_DESK, W, grid, "prior")
        s = summarize_cells(cells)
        note = (f"Spec-sheet roofline at {PRIOR_EFF_BW} bandwidth / {PRIOR_EFF_FLOPS} compute efficiency for every "
                f"action (edge {PRIOR_EDGE_GPU['name']}, server {PRIOR_SERVER_GPU}); SPLIT link = RTT per token "
                f"plus {a.overhead_ms:g} ms runtime/serialisation overhead per token; predicted U-max; desk-check eligibility; quality and output lengths are the "
                "measured S1 per-query values. Good state only. With 0 ms overhead the prior favours SPLIT (no "
                "serialisation, no HF-eager overhead); the same efficiency for the edge competitors favours neither. Decides nothing.")
        Path(a.out).write_text(prior_markdown(cells, s, note))
        I.write_json(str(a.out).replace(".md", ".json"), {"summary": s, "cells": _json_cells(cells)})
        print(json.dumps({k: v for k, v in s.items() if k != "best"}), "\nbest:",
              json.dumps({k: s["best"][k] for k in ("M", "privacy", "rtt_ms", "D_s", "delta_pt", "ci95")}))
    elif a.cmd in ("gate-ub1", "gate-ub2"):
        W = boot_weights(data.strata)
        st1 = Path(a.stage1_dir)
        fres = json.loads((st1 / "f_results.json").read_text())
        if not fres.get("valid", True):
            sys.exit(f"Stage 1 invalid (harness defects {fres['harness_defects']}): fix and re-run Stage 1 (PREREG §5.6)")
        blocked = set(fres.get("no_split_pass_classes", []))   # R cannot pass there (§8.4 b): bound excludes them
        umax = {int(m): tuple(v) for m, v in fres["u_max"].items() if v and int(m) not in blocked}
        eligible = {int(m): v for m, v in fres["edge_eligible"].items()}
        edge_prof = edge_profiles_from_dir(st1)
        eseg_k = load_seg_profiles(st1, EDGE_KEYS, "P-edge-U(")
        f = S.UB_SEGMENT_DISCOUNT
        edge_seg = {(M, k): scaled(eseg_k[k], f) for M, (k, j) in umax.items() if k in eseg_k}
        if a.cmd == "gate-ub1":
            server_seg = {(M, k): server_lb_profile(a.server_gpu, k, j) for M, (k, j) in umax.items()}
        else:
            sseg_k = load_seg_profiles(a.stage2_dir, SERVER_KEYS, "P-server-U(")
            server_seg = {(M, k): scaled(sseg_k[k], f) for M, (k, j) in umax.items() if k in sseg_k}
        res = upper_bound_gate(data, edge_seg, server_seg, edge_prof, eligible, W, umax)
        I.write_json(a.out, {"gate": a.cmd, "decision": res["decision"], "summary": res["summary"],
                             "max_delta_bound_pt": res["max_delta_bound_pt"],
                             "missing_profiles": res["missing_profiles"], "segment_discount": f,
                             "cells": _json_cells(res["cells"])})
        Path(str(a.out).replace(".json", ".md")).write_text(
            f"# {a.cmd}: {res['decision']}\n\n" + cells_markdown(res["cells"], "Upper-bound cells"))
        print(res["decision"])
    elif a.cmd == "fit":
        I.write_json(a.out, fit_all(a.lat_dir, a.f_results))
        print(open(a.out).read()[:2000])
    elif a.cmd == "r":
        W = boot_weights(data.strata)
        bundle = json.loads(Path(a.bundle).read_text())
        st1 = Path(a.stage1_dir)
        fres = json.loads((st1 / "f_results.json").read_text())
        eligible = {int(m): v for m, v in fres["edge_eligible"].items()}
        sp, src = split_profiles_from_bundle(bundle)
        blocked = set(fres.get("no_split_pass_classes", []))
        sp = {key: v for key, v in sp.items() if key[0] not in blocked}
        cells = evaluate_grid(data, sp, edge_profiles_from_dir(st1), remote_profiles_from_bundle(bundle), eligible,
                              W, label=src)
        for c in cells:
            if c["M"] in blocked:
                c["status"] = ("blocked: a desk-eligible EDGE competitor failed its fit for a non-memory reason "
                               "(PREREG §8.4 b); no SPLIT pass at this class")
        s = summarize_cells(cells)
        out = Path(a.out_dir)
        I.write_json(out / "R_cells.json", {"source": src, "sample_n": len(data.keys), "summary": s,
                                            "cells": _json_cells(cells)})
        (out / "R_cells.md").write_text(f"Source of SPLIT profiles: {src}\n\n" + cells_markdown(cells, "E0-R cells"))
        print(json.dumps({k: v for k, v in s.items() if k != "best"}))
    elif a.cmd == "decide":
        decide(a)


def extension_triggered(rr):
    """PREREG §8.5 / E0 §4.4: R fails at 200 queries, yet its best cell has delta >= 3.0 pt with CI low <= 0.
    Fires at most once (never on the 400-query re-evaluation)."""
    best = rr["summary"].get("best")
    return bool(not rr["summary"]["R_pass"] and rr.get("sample_n", 200) == 200 and best is not None
                and best["delta_pt"] >= S.R_MARGIN_PT and best["ci95"][0] <= 0)


def decide(a):
    """F: a class passes iff it has a measured U-max; F additionally needs zero failed E0-L requests on ANY
    path (E0 §8 F, read literally; failed rows are never erased by a resume)."""
    fres = json.loads(Path(a.f_results).read_text())
    errs = {}
    for f in (f for d in a.lat_dir for f in Path(d).glob("*.jsonl")):
        for r in I.read_jsonl(f):
            if "prompt_id" in r and not r.get("ok"):
                errs.setdefault(f.stem, []).append(r.get("error"))
    F_classes = {m: v is not None for m, v in fres["u_max"].items()}
    F = any(F_classes.values()) and not errs
    q = json.loads(Path(a.q).read_text())
    rr = json.loads(Path(a.r).read_text())
    R = rr["summary"]["R_pass"]
    Q = q["pass"]
    ext = Q and F and extension_triggered(rr)
    if not Q or not F:
        cons = "SPLIT removed. F3 = EDGE/REMOTE; H-joint must pass on contention alone."
    elif ext:
        cons = ("PENDING: the pre-declared 400-query extension (PREREG §8.5) is triggered. Run it once with the "
                "pinned sample and the same profiles, thresholds and seeds; then re-run r and decide. No other change.")
    elif R and set(rr["summary"].get("pass_classes", [])) == {24} and not getattr(a, "a3b_added", False):
        cons = ("PENDING: every passing cell is at 24 GB. PREREG §8.4 (a): add Qwen3-30B-A3B as an EDGE competitor "
                "and re-evaluate before declaring anything.")
    elif not R:
        cons = "SPLIT removed for lack of a regime; report 'weight-quantised edge models dominate layer split'."
    elif rr["summary"]["open_pass"]:
        cons = "PASS including open cells: unexpected; re-check against the component floor and netem logs first."
    else:
        cons = ("SPLIT enters A as the no-raw-text quality tier (raw text stays local is an admissibility rule, "
                "never a privacy claim; report measured leakage).")
    if R and rr["summary"]["only_12GB"]:
        cons += " Regime restricted to memory-starved (12 GB) edges."
    if R and 24 in rr["summary"].get("pass_classes", []) and not getattr(a, "a3b_added", False):
        cons += " Any 24 GB passing cell is held pending PREREG §8.4 (a) (Qwen3-30B-A3B) and is not claimed."
    passing = set(rr["summary"].get("pass_classes", []))
    scope = None
    if Q and F and R:
        scope = ("physical 24 GB device (capped by 512 MiB)" if passing == {24} else
                 "12/16 GB classes are EMULATED on a capped 24 GB GPU: no claim about physical 12/16 GB devices "
                 "without a separate physical validation (PREREG §3.3)")
    res = {"Q": Q, "F": F, "F_classes": F_classes, "transport_or_oom_errors": errs, "R": R,
           "R_robust_to_multiplicity": rr["summary"]["pass_robust_to_multiplicity"],
           "R_sample_n": rr.get("sample_n", 200), "extension_triggered": ext,
           "E0": "PENDING" if cons.startswith("PENDING") else ("PASS" if (Q and F and R) else "FAIL"),
           "memory_class_scope": scope,
           "consequence": cons}
    out = Path(a.out_dir)
    I.write_json(out / "DECISION.json", res)
    (out / "DECISION.md").write_text(
        "# E0 decision\n\n" + "\n".join(f"- **{k}**: {v}" for k, v in res.items()) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
