"""POST HOC screen of the laboratory pair from archived records. CPU only. Exploratory: NOT a result of QWEN-SELECT.

It was run on 2026-11-20, before the tag, to decide whether a new experiment is justified and to fix its design. It
combines records of protocols that are not compatible with each other, under the assumptions printed in the output:

  quality and output lengths   S1 / E0-Q3 (pre-registered quality gate, 2026-11-08): vLLM on a rented H200, sampled
                               decoding, two seeds, 200 queries. File diagnostics/deployment_s1/results/s1_2026-11-08/q3_scores.jsonl
  split and unsplit timing     K16-HWFIT (exploratory, 2026-11-18): the laboratory pair, greedy decoding, 256 forced
                               tokens, ten prompts of 256 or 2,048 tokens. Medians typed in below from the run's summaries
  smaller models on the edge   never measured on the edge machine. Their time is ASSUMED to be the unsplit 32B model's
                               time multiplied by a ratio; the central ratio is the ratio of weight bytes

The decision rule and the bootstrap are E0's own code (deployment_e0/e0_r.py), imported unchanged.

  python -I posthoc_screen.py            # writes POSTHOC_SCREEN.md and posthoc_screen.json beside this file
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "deployment_e0"))
import e0_r as R  # noqa: E402
import e0_spec as E0  # noqa: E402

# K16-HWFIT medians (seconds). Split: reference mode, two hosts. Unsplit: the whole bf16 model on the edge machine.
K16 = {"split": {"ttft": (0.791, 5.685), "tpot": (0.2263, 0.2616)}, "unsplit": {"ttft": (0.442, 2.599), "tpot": (0.3024, 0.3633)}}
GB = {"Qwen3-32B": 65.5, "Qwen3-14B": 29.5, "Qwen3-8B": 16.4}                  # bf16 checkpoint sizes
QUALITY_PROXY = {"Qwen3-14B": "Qwen3-14B-AWQ", "Qwen3-8B": "Qwen3-8B"}         # S1 never ran Qwen3-14B in bf16: its 4-bit build stands in
RATIO = {m: GB[m] / GB["Qwen3-32B"] for m in GB}                               # central assumption for the edge
REMOTE_RATIO_14B = 0.19        # assumption: 40 layers at 0.72 of the 1.8 ms per layer measured on the server GPU for the 32B layers
DESIGN = {"deadlines": (30, 60, 120, 300), "cutoff": 300.0, "cap": 3072, "context": 4096, "chat_overhead": 12}


def profile(d, label):
    """E0's latency profile through the two measured prompt lengths (256 and 2,048 tokens, 256 output tokens)."""
    b = (d["tpot"][1] - d["tpot"][0]) / 1792.0
    return R.Profile([256.0, 2048.0], list(d["ttft"]), d["tpot"][0] - b * (256 + 128), b, label)


def main():
    data = R.S1Data()
    W = R.boot_weights(data.strata)
    P_S, P_E = profile(K16["split"], "split"), profile(K16["unsplit"], "unsplit")
    out = {"label": "POST HOC, exploratory; incompatible protocols combined under stated assumptions; not a QWEN-SELECT result",
           "k16_medians_s": K16, "ratios": RATIO, "remote_ratio_14B": REMOTE_RATIO_14B}
    L = ["# Post hoc screen of the laboratory pair (exploratory; not a result)", "",
         "Written 2026-11-20 from archived records, before the QWEN-SELECT tag. It combines S1 quality scores and output lengths",
         "(sampled decoding, another engine, another machine) with K16-HWFIT timings (greedy, 256 forced tokens, ten prompts).",
         "Times of smaller models on the edge are assumed, never measured. Nothing here validates anything.", ""]

    def rule(v_s, v_c):
        st = R.cell_stats(v_s, v_c, W)
        st["pass"] = bool(round(st["delta_pt"], 6) >= 3.0 and st["ci95"][0] > 0)
        return st

    def oc(action, prof, D):
        return R.ontime_correct(data, action, prof, D)[0]

    # ---- A. the same model, split against whole on the edge (all 200 queries, both modes, E0's deadlines)
    L += ["## A. Same model: split against the whole 32B model on the edge", "",
          "| D (s) | U split, non-thinking | U edge, non-thinking | U split, thinking | U edge, thinking | Delta (pt) | 95% CI | rule |", "|---|---|---|---|---|---|---|---|"]
    out["A_same_model"] = []
    for D in E0.R_DEADLINES_S:
        vs = {f"S|{md}": oc(("Qwen3-32B", md), P_S, D) for md in E0.MODES}
        vc = {f"E32|{md}": oc(("Qwen3-32B", md), P_E, D) for md in E0.MODES}
        st = rule(vs, vc)
        out["A_same_model"].append({"D": D, "delta_pt": st["delta_pt"], "ci95": st["ci95"], "pass": st["pass"], "U": {**st["U_split"], **st["U_comp"]}})
        L.append(f"| {D} | {100 * vs['S|nonthinking'].mean():.2f} | {100 * vc['E32|nonthinking'].mean():.2f} | {100 * vs['S|thinking'].mean():.2f} | "
                 f"{100 * vc['E32|thinking'].mean():.2f} | {st['delta_pt']:+.2f} | [{st['ci95'][0]:+.2f}, {st['ci95'][1]:+.2f}] | {'SPLIT' if st['pass'] else 'edge'} |")
    L += ["", "Reading: with the K16 medians the split's speed advantage per token turns into a utility advantage over the same model",
          "on the edge at 30 to 300 s. Thinking mode is never the split's better mode at these speeds.", ""]

    # ---- B. smaller models on the edge at assumed speeds
    def cells(models, ratios, modes=E0.MODES, deadlines=E0.R_DEADLINES_S):
        res = []
        for D in deadlines:
            vs = {f"S|{md}": oc(("Qwen3-32B", md), P_S, D) for md in modes}
            vc = {f"E:Qwen3-32B|{md}": oc(("Qwen3-32B", md), P_E, D) for md in modes}
            for m in models:
                for md in modes:
                    vc[f"E:{m}|{md}"] = oc((QUALITY_PROXY[m], md), R.scaled(P_E, ratios[m]), D)
            st = rule(vs, vc)
            res.append({"D": D, "delta_pt": st["delta_pt"], "ci95": st["ci95"], "pass": st["pass"], "best": st["comp_best"],
                        "U_split": st["U_split"][st["split_best"]], "U_best": st["U_comp"][st["comp_best"]]})
        return res
    L += ["## B. Smaller bf16 models on the edge, time assumed = unsplit 32B time x weight-bytes ratio", "",
          f"Ratios: 14B {RATIO['Qwen3-14B']:.2f}, 8B {RATIO['Qwen3-8B']:.2f}. Quality of the 14B model: its 4-bit build's S1 scores (a proxy).", "",
          "| Edge competitors | " + " | ".join(f"D={D}" for D in E0.R_DEADLINES_S) + " | cells selecting SPLIT |", "|---|" + "---|" * (len(E0.R_DEADLINES_S) + 1)]
    out["B_smaller_models"] = {}
    for label, models in (("32B and 8B", ["Qwen3-8B"]), ("32B, 14B and 8B", ["Qwen3-14B", "Qwen3-8B"])):
        cs = cells(models, RATIO)
        out["B_smaller_models"][label] = cs
        L.append(f"| {label} | " + " | ".join(f"{c['delta_pt']:+.2f} ({c['best'].split(':')[1].split('|')[0].replace('Qwen3-', '')})" for c in cs)
                 + f" | {[c['D'] for c in cs if c['pass']]} |")
    L += ["", "Each entry is Delta in points against the best competitor (named in brackets).", ""]
    # break-even: the slowest 14B (as a ratio of the unsplit 32B time) at which no cell selects the split
    grid = [round(x, 3) for x in np.arange(0.20, 1.001, 0.05)]
    be = []
    for r in grid:
        cs = cells(["Qwen3-14B", "Qwen3-8B"], dict(RATIO, **{"Qwen3-14B": r}))
        be.append({"ratio_14B": r, "max_delta_pt": max(c["delta_pt"] for c in cs), "select_split": [c["D"] for c in cs if c["pass"]]})
    out["B_break_even_14B"] = be
    first = next((b["ratio_14B"] for b in be if b["select_split"]), None)
    L += [f"Break-even: with the 8B model at its ratio, the split is first selected in some cell when the 14B model's time ratio reaches "
          f"**{first}** (central assumption {RATIO['Qwen3-14B']:.2f}). Below that, no cell selects the split.", ""]

    # ---- C. the design's cells: non-thinking, no LiveCodeBench, cap and cutoff of the protocol; with REMOTE
    keep = data.strata != R.BENCH_ORDER.index("livecodebench")
    Wk = R.boot_weights(data.strata[keep])

    def v(model, prof, D):
        a = (QUALITY_PROXY.get(model, model), "nonthinking")
        n_in = data.n_in[a][keep] + DESIGN["chat_overhead"]
        cap = np.minimum(DESIGN["cap"], DESIGN["context"] - n_in)[:, None]
        N = np.minimum(data.N[a][keep], cap)
        T = prof.T(n_in[:, None], N)
        ok = data.score[a][keep] * (data.N[a][keep] <= cap) * (T <= min(D, DESIGN["cutoff"]))
        return ok.mean(axis=1), np.minimum(T, DESIGN["cutoff"])
    profs = {"S": ("Qwen3-32B", P_S), "E32": ("Qwen3-32B", P_E), "E14": ("Qwen3-14B", R.scaled(P_E, RATIO["Qwen3-14B"])),
             "E8": ("Qwen3-8B", R.scaled(P_E, RATIO["Qwen3-8B"])), "R14": ("Qwen3-14B", R.scaled(P_E, REMOTE_RATIO_14B))}
    L += ["## C. The cells of the QWEN-SELECT design (prediction recorded before the run)", "",
          "Non-thinking mode, MMLU-Pro, GPQA and LiveMathBench of the S1 sample (160 queries, two sampled seeds), output cap and 300 s cutoff of the",
          "protocol. The experiment itself uses other queries, greedy decoding and measured times.", "",
          "| D (s) | U S | U E32 | U E14 | U E8 | U R14 | no-raw-text: Delta, CI, rule | open: Delta, CI, rule | same model: Delta, CI, rule |", "|---|---|---|---|---|---|---|---|---|"]
    out["C_design_cells"] = []
    for D in DESIGN["deadlines"]:
        u = {c: v(m, p, D)[0] for c, (m, p) in profs.items()}
        row = {"D": D, "U": {c: float(x.mean()) for c, x in u.items()}}
        txt = []
        for name, comp in (("no-raw-text", ("E32", "E14", "E8")), ("open", ("E32", "E14", "E8", "R14")), ("same-model", ("E32",))):
            st = R.cell_stats({"S": u["S"]}, {c: u[c] for c in comp}, Wk)
            ps = bool(round(st["delta_pt"], 6) >= 3.0 and st["ci95"][0] > 0)
            row[name] = {"delta_pt": st["delta_pt"], "ci95": st["ci95"], "pass": ps, "best": st["comp_best"]}
            txt.append(f"{st['delta_pt']:+.2f} [{st['ci95'][0]:+.2f}, {st['ci95'][1]:+.2f}] {'SPLIT' if ps else st['comp_best']}")
        out["C_design_cells"].append(row)
        L.append(f"| {D} | " + " | ".join(f"{100 * row['U'][c]:.2f}" for c in ("S", "E32", "E14", "E8", "R14")) + " | " + " | ".join(txt) + " |")
    L += ["", "Predicted outcome under these assumptions: no cell selects the split against the best edge configuration, and the split beats the",
          "same model on the edge in several cells. Both statements are at risk in the experiment.", ""]

    # ---- D. planning arithmetic for the GPU budget (not a result)
    L += ["## D. Planning arithmetic for the GPU budget", "", "| Block | expected hours for 160 queries | share of queries cut off at 300 s |", "|---|---|---|"]
    out["D_budget_hours"] = {}
    for c, (m, p) in profs.items():
        _, T = v(m, p, 300)
        h = float(T.mean(axis=1).sum() / 3600)
        out["D_budget_hours"][c] = h
        L.append(f"| {c} | {h:.1f} | {float((T >= DESIGN['cutoff']).mean()):.2f} |")
    L += ["", "The split block holds two GPUs. These figures rest on sampled output lengths; greedy lengths may differ.", ""]

    # ---- E. how often the rule would select the split with 160 queries and one pass (resampling the S1 queries)
    rng = np.random.default_rng(20261120)
    st_k = data.strata[keep]
    members = [np.flatnonzero(st_k == s) for s in np.unique(st_k)]
    L += ["## E. Sensitivity of the rule at 160 queries, one pass (resampled S1 queries; planning only)", "",
          "| Competitors | " + " | ".join(f"D={D}" for D in DESIGN["deadlines"]) + " |", "|---|" + "---|" * len(DESIGN["deadlines"])]
    out["E_selection_rate"] = {}
    for label, comp in (("E32 only (same model)", ("E32",)), ("E32 and E8", ("E32", "E8")), ("E32, E14 and E8", ("E32", "E14", "E8"))):
        rates = []
        for D in DESIGN["deadlines"]:
            a = {c: (QUALITY_PROXY.get(profs[c][0], profs[c][0]), "nonthinking") for c in ("S",) + comp}
            per_seed = {}
            for c in ("S",) + comp:
                m, p = profs[c]
                n_in = data.n_in[a[c]][keep] + DESIGN["chat_overhead"]
                cap = np.minimum(DESIGN["cap"], DESIGN["context"] - n_in)[:, None]
                T = p.T(n_in[:, None], np.minimum(data.N[a[c]][keep], cap))
                per_seed[c] = data.score[a[c]][keep] * (data.N[a[c]][keep] <= cap) * (T <= D)
            sel, reps = 0, 300
            for _ in range(reps):
                idx = np.concatenate([rng.choice(mm, size=len(mm), replace=True) for mm in members])
                seed = rng.integers(0, 2, size=len(idx))
                S_ = per_seed["S"][idx, seed]
                C_ = np.stack([per_seed[c][idx, seed] for c in comp], 1)
                d = 100 * (S_.mean() - C_.mean(0).max())
                b = rng.integers(0, len(idx), size=(400, len(idx)))
                lo = np.percentile(100 * (S_[b].mean(1) - C_[b].mean(1).max(1)), 2.5)
                sel += bool(round(d, 6) >= 3.0 and lo > 0)
            rates.append(sel / reps)
        out["E_selection_rate"][label] = rates
        L.append(f"| {label} | " + " | ".join(f"{r:.2f}" for r in rates) + " |")
    L += ["", "Each entry is the share of resampled workloads in which the rule selects the split.", ""]
    (HERE / "POSTHOC_SCREEN.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (HERE / "posthoc_screen.json").write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
