"""E0-Q: output equivalence of the split path (E0 §4.1-§4.2, §8 Q; PREREG §4).

Subcommands
  q2-keys     the frozen 30 Q2 prompts (CPU; output committed)
  cont-gen    greedy reference continuations (<= 4,096 tokens, thinking template), HF batched, server
  ref         teacher-forced reference top-K per position: single-host HF (Qwen3-32B or Qwen3-8B)
  cmp         teacher-forced comparison of a configuration against a reference directory:
                --config hf      single-host HF on this host (hardware floor, Qwen3-8B on the edge)
                --config split   edge runner -> segment server (SPLIT on real hosts, or the in-host
                                 floor when both processes share one server GPU)
  q2-ref      greedy free-run reference (HF generate, single host) with per-step top-1/top-2 margins
  q2-run      greedy free-run on the split path
  decide      apply the frozen Q1 and Q2 rules to the files above

Q1 metrics are pooled over every continuation position of every query:
  D  = share of positions where argmax_x != argmax_ref (full-vocabulary argmax on both sides);
  KL = mean coarse-grained KL(ref || x) over the partition {the reference's top-K tokens, "rest"}.
The coarse KL is a lower bound on the full-vocabulary KL (data processing) and is computed the same
way for the split path and for both floors, so the relative criterion is like-for-like.
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e0_instrument as I  # noqa: E402
import e0_spec as S  # noqa: E402

Q2_KEYS_JSON = S.E0_DIR / "frozen" / "q2_keys.json"


# --------------------------------------------------------------------------- #
# Frozen selections                                                           #
# --------------------------------------------------------------------------- #
def sample_rows():
    s1 = S.load_s1()
    return [json.loads(x) for x in open(s1.SAMPLE_JSONL)]


def q2_keys(sample=None):
    """First Q2_PER_BENCH[b] keys of a seeded permutation of each benchmark's S1 sample keys."""
    sample = sample or sample_rows()
    keys = []
    for b, n in S.Q2_PER_BENCH.items():
        ks = sorted((r["key"] for r in sample if r["dataset"] == b), key=lambda k: int(k.split(":")[1]))
        random.Random(f"{S.Q2_SEED}:{b}").shuffle(ks)
        keys += ks[:n]
    assert len(keys) == S.Q2_N
    return keys


# --------------------------------------------------------------------------- #
# Shared numerics                                                             #
# --------------------------------------------------------------------------- #
def head_chunks(torch, lm_head, hidden, positions, chunk=S.HEAD_CHUNK):
    """Yield (slice of `positions`, float32 log-probs [c, V]) for final normed hidden states [n, H]."""
    for a in range(0, len(positions), chunk):
        pos = positions[a:a + chunk]
        with torch.inference_mode():
            lp = torch.log_softmax(lm_head(hidden[pos]).float(), dim=-1)
        yield slice(a, a + len(pos)), lp


def ref_topk(torch, lm_head, hidden, positions, k=S.Q1_TOPK):
    import numpy as np
    n = len(positions)
    ids = np.zeros((n, k), dtype=np.int32)
    lps = np.zeros((n, k), dtype=np.float32)
    for sl, lp in head_chunks(torch, lm_head, hidden, positions):
        v, i = torch.topk(lp, k, dim=-1)
        ids[sl], lps[sl] = i.cpu().numpy(), v.cpu().numpy()
    return ids, lps


def compare_to_ref(torch, lm_head, hidden, positions, ref_ids, ref_lps):
    """Per-position (agree, coarse KL, x margin) of configuration x against the stored reference."""
    import numpy as np
    n = len(positions)
    agree = np.zeros(n, dtype=bool)
    kl = np.zeros(n, dtype=np.float64)
    xm = np.zeros(n, dtype=np.float32)
    for sl, lp in head_chunks(torch, lm_head, hidden, positions):
        top2 = torch.topk(lp, 2, dim=-1)
        x_arg = top2.indices[:, 0].cpu().numpy()
        xm[sl] = (top2.values[:, 0] - top2.values[:, 1]).cpu().numpy()
        rid = torch.as_tensor(ref_ids[sl], device=lp.device, dtype=torch.long)
        lx = torch.gather(lp, 1, rid).double().cpu().numpy()
        agree[sl] = x_arg == ref_ids[sl][:, 0]
        kl[sl] = coarse_kl(ref_lps[sl].astype(np.float64), lx)
    return agree, kl, xm


def coarse_kl(lp_ref, lp_x):
    """KL(ref || x) on the partition {each of the reference's top-K ids, rest}; arrays [n, K] (log-probs)."""
    import numpy as np
    pr, px = np.exp(lp_ref), np.exp(lp_x)
    head = (pr * (lp_ref - lp_x)).sum(axis=1)
    rr = np.clip(1.0 - pr.sum(axis=1), 1e-12, None)
    rx = np.clip(1.0 - px.sum(axis=1), 1e-12, None)
    return np.maximum(head + rr * (np.log(rr) - np.log(rx)), 0.0)


def is_degenerate(tokens, vocab_len, window=S.DEGEN_WINDOW, max_period=S.DEGEN_MAX_PERIOD):
    """Invalid token ids, or the last `window` tokens periodic with some period <= max_period."""
    if any(t < 0 or t >= vocab_len for t in tokens):
        return True, "invalid token id"
    if len(tokens) < window:
        return False, ""
    tail = tokens[-window:]
    for p in range(1, max_period + 1):
        if all(tail[i] == tail[i - p] for i in range(p, window)):
            return True, f"period {p}"
    return False, ""


def first_divergence(ref, x):
    for i, (a, b) in enumerate(zip(ref, x)):
        if a != b:
            return i
    return None if len(ref) == len(x) else min(len(ref), len(x))


# --------------------------------------------------------------------------- #
# Runs                                                                        #
# --------------------------------------------------------------------------- #
def _torch_setup():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    return torch


def _tokenizer(d):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(d)


def _hf_model(torch, model_dir, device):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                                             device_map={"": device})
    return m.eval()


def cont_gen(a):
    """HF batched greedy continuations; only used as fixed teacher-forcing text (PREREG §4.1)."""
    torch = _torch_setup()
    tok = _tokenizer(a.model_dir)
    tok.padding_side = "left"
    m = _hf_model(torch, a.model_dir, a.device)
    from e0_latency import templated_ids
    rows = sample_rows()[: a.limit] if a.limit else sample_rows()
    out = Path(a.out)
    done = {r["key"] for r in I.read_jsonl(out)}
    todo = [r for r in rows if r["key"] not in done]
    todo.sort(key=lambda r: len(templated_ids(tok, r["prompt"])))
    for i in range(0, len(todo), a.batch):
        batch = todo[i:i + a.batch]
        enc = [templated_ids(tok, r["prompt"]) for r in batch]
        L = max(map(len, enc))
        pad = tok.pad_token_id if tok.pad_token_id is not None else S.EOS_IDS[1]
        ids = torch.tensor([[pad] * (L - len(e)) + e for e in enc], device=a.device)
        att = torch.tensor([[0] * (L - len(e)) + [1] * len(e) for e in enc], device=a.device)
        with torch.inference_mode():
            g = m.generate(input_ids=ids, attention_mask=att, max_new_tokens=a.max_new, do_sample=False,
                           eos_token_id=list(S.EOS_IDS), pad_token_id=pad)
        for r, e, row in zip(batch, enc, g[:, L:].tolist()):
            cont = []
            for t in row:
                cont.append(t)
                if t in S.EOS_IDS:
                    break
            I.append_jsonl(out, {"key": r["key"], "prompt_ids": e, "cont_ids": cont,
                                 "eos": cont[-1] in S.EOS_IDS, "max_new": a.max_new})
        print(f"cont-gen {min(i + a.batch, len(todo))}/{len(todo)}", flush=True)


def _sequences(cont_file, limit=None):
    rows = I.read_jsonl(cont_file)
    rows = rows[:limit] if limit else rows
    for r in rows:
        ids = r["prompt_ids"] + r["cont_ids"]
        positions = list(range(len(r["prompt_ids"]) - 1, len(ids) - 1))   # predict each continuation token
        yield r["key"], ids, positions


def ref_run(a):
    import numpy as np
    torch = _torch_setup()
    m = _hf_model(torch, a.model_dir, a.device)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for key, ids, pos in _sequences(a.cont, a.limit):
        f = out / f"{key.replace(':', '_')}.npz"
        if f.exists():
            continue
        with torch.inference_mode():
            h = m.model(torch.tensor([ids], device=a.device)).last_hidden_state[0]
        rid, rlp = ref_topk(torch, m.lm_head, h, pos)
        np.savez_compressed(f, ids=rid, lps=rlp, positions=np.array(pos, dtype=np.int32))
    I.write_json(out / "env.json", I.environment(torch, {"role": "q1-ref", "model_dir": a.model_dir}))


def cmp_run(a):
    import numpy as np
    torch = _torch_setup()
    if a.config == "hf":
        m = _hf_model(torch, a.model_dir, a.device)

        def hidden(ids):
            with torch.inference_mode():
                return m.model(torch.tensor([ids], device=a.device)).last_hidden_state[0]
        lm_head = m.lm_head
    else:
        import split_runtime as R
        expected = S.QWEN3_32B_CONFIG if a.expect_32b else None
        sl = R.edge_slice(a.model_dir, a.k, a.j, a.device, torch.bfloat16, expected)
        host, port = a.server.split(":")
        mid = R.RemoteMiddle(host, int(port), R.hello_for(sl, a.model_id, a.k, a.j, S.L_F), a.device,
                             sl.cfg.hidden_size)
        runner = R.EdgeRunner(sl, a.k, a.j, mid, max_len=None)
        hidden, lm_head = runner.teacher_forced_hidden, sl.lm_head
    ref = Path(a.ref_dir)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for key, ids, pos in _sequences(a.cont, a.limit):
        f = out / f"{key.replace(':', '_')}.npz"
        if f.exists():
            continue
        r = np.load(ref / f"{key.replace(':', '_')}.npz")
        assert list(r["positions"]) == pos, key
        h = hidden(ids)
        ag, kl, xm = compare_to_ref(torch, lm_head, h, pos, r["ids"], r["lps"])
        np.savez_compressed(f, agree=ag, kl=kl, x_margin=xm, ref_margin=r["lps"][:, 0] - r["lps"][:, 1])
    I.write_json(out / "env.json", I.environment(torch, {"role": f"q1-cmp-{a.config}", "model_dir": a.model_dir,
                                                         "k": a.k, "j": a.j, "server": a.server}))


def q2_ref(a):
    torch = _torch_setup()
    tok = _tokenizer(a.model_dir)
    m = _hf_model(torch, a.model_dir, a.device)
    from e0_latency import templated_ids
    by_key = {r["key"]: r for r in sample_rows()}
    keys = json.loads(Q2_KEYS_JSON.read_text())["keys"][: a.limit or None]
    out = Path(a.out)
    done = {r["key"] for r in I.read_jsonl(out)}
    for key in keys:
        if key in done:
            continue
        ids = templated_ids(tok, by_key[key]["prompt"])
        with torch.inference_mode():
            t = torch.tensor([ids], device=a.device)
            g = m.generate(t, attention_mask=torch.ones_like(t), max_new_tokens=a.max_new, do_sample=False,
                           eos_token_id=list(S.EOS_IDS), output_scores=True, return_dict_in_generate=True)
        toks = g.sequences[0, len(ids):].tolist()
        margins = []
        for sc in g.scores:
            top = torch.topk(torch.log_softmax(sc[0].float(), -1), 2).values
            margins.append(float(top[0] - top[1]))
        I.append_jsonl(out, {"key": key, "prompt_ids": ids, "tokens": toks, "margins": margins[:len(toks)],
                             "nan": any(math.isnan(x) for x in margins)})


def q2_run(a):
    torch = _torch_setup()
    import split_runtime as R
    by_key = {r["key"]: r for r in I.read_jsonl(a.ref)}
    expected = S.QWEN3_32B_CONFIG if a.expect_32b else None
    sl = R.edge_slice(a.model_dir, a.k, a.j, a.device, torch.bfloat16, expected)
    host, port = a.server.split(":")
    mid = R.RemoteMiddle(host, int(port), R.hello_for(sl, a.model_id, a.k, a.j, S.L_F), a.device, sl.cfg.hidden_size)
    runner = R.EdgeRunner(sl, a.k, a.j, mid, max_len=S.L_F)
    out = Path(a.out)
    done = {r["key"] for r in I.read_jsonl(out)}
    for key, ref in by_key.items():
        if key in done:
            continue
        toks = runner.generate(ref["prompt_ids"], a.max_new, S.PREFILL_CHUNK, eos_ids=S.EOS_IDS, margins=True)
        I.append_jsonl(out, {"key": key, "tokens": toks, "margins": runner.margins,
                             "nan": any(math.isnan(x) for x in runner.margins), "k": a.k, "j": a.j})


# --------------------------------------------------------------------------- #
# Decision                                                                    #
# --------------------------------------------------------------------------- #
def q1_summary(cmp_dir):
    import numpy as np
    ag, kl, n_q = [], [], 0
    for f in sorted(Path(cmp_dir).glob("*.npz")):
        z = np.load(f)
        ag.append(z["agree"])
        kl.append(z["kl"])
        n_q += 1
    if not ag:
        return None
    a, k = np.concatenate(ag), np.concatenate(kl)
    return {"n_queries": n_q, "n_positions": int(a.size), "D_pp": 100.0 * (1.0 - a.mean()), "KL": float(k.mean())}


def q1_rule(split, pipeline, hw):
    d_floor = max(pipeline["D_pp"], hw["D_pp"])
    kl_floor = max(pipeline["KL"], hw["KL"])
    kl_thr = S.Q1_KL_FACTOR * kl_floor if kl_floor > 0 else S.Q1_KL_EPS
    ok_d = split["D_pp"] <= d_floor + S.Q1_D_MARGIN_PP
    ok_kl = split["KL"] <= kl_thr
    return {"D_split_pp": split["D_pp"], "D_threshold_pp": d_floor + S.Q1_D_MARGIN_PP, "KL_split": split["KL"],
            "KL_threshold": kl_thr, "pass_D": bool(ok_d), "pass_KL": bool(ok_kl), "pass": bool(ok_d and ok_kl)}


def q2_rule(ref_rows, x_rows, vocab_len):
    ref = {r["key"]: r for r in ref_rows}
    per, near, bad = [], 0, []
    for x in x_rows:
        r = ref[x["key"]]
        i = first_divergence(r["tokens"], x["tokens"])
        m = None if i is None or i >= len(r["margins"]) else r["margins"][i]
        ok_div = i is None or (m is not None and m < S.Q2_MARGIN_NATS)
        near += ok_div
        dx, why_x = is_degenerate(x["tokens"], vocab_len)
        dr, _ = is_degenerate(r["tokens"], vocab_len)
        dx = dx or x.get("nan", False)
        if dx and not dr:
            bad.append(x["key"])
        per.append({"key": x["key"], "first_div": i, "ref_margin": m, "near_tie_or_none": ok_div,
                    "x_degenerate": dx, "why": why_x, "ref_degenerate": dr})
    ok = len(x_rows) == S.Q2_N and near >= S.Q2_MIN_NEAR_TIE and not bad
    return {"n": len(x_rows), "near_tie_or_none": near, "required": S.Q2_MIN_NEAR_TIE,
            "degenerate_only_split": bad, "pass": bool(ok), "per_prompt": per}


def decide(a):
    q = {}
    hw = q1_summary(a.hw_dir)
    q["floors"] = {"hardware": hw, "pipeline": {}}
    q["Q1"], q["Q2"] = {}, {}
    ref2 = I.read_jsonl(a.q2_ref)
    for cfg, d, pd, f2 in zip(S.Q1_CONFIGS, a.split_dirs, a.pipeline_dirs, a.q2_runs):
        pipe = q1_summary(pd)          # in-host floor cut at the same layers as this configuration
        q["floors"]["pipeline"][cfg] = pipe
        sp = q1_summary(d)
        q["Q1"][cfg] = dict(q1_rule(sp, pipe, hw), summary=sp) if sp and pipe and hw else {"pass": False,
                                                                                       "missing": True}
        q["Q2"][cfg] = q2_rule(ref2, I.read_jsonl(f2), a.vocab_len)
    q["pass"] = all(q["Q1"][c]["pass"] and q["Q2"][c]["pass"] for c in S.Q1_CONFIGS)
    q["fix_attempt"] = a.fix_attempt
    q["consequence"] = ("Q PASS" if q["pass"] else
                        ("Q FAIL: one implementation fix allowed, then re-run Q1+Q2 once" if a.fix_attempt == 0
                         else "Q FAIL after the allowed fix: SPLIT removed (no activation compression)"))
    I.write_json(a.out, q)
    print(json.dumps({k: v for k, v in q.items() if k not in ("Q2",)}, indent=1, default=str))
    print({c: {k: v for k, v in q["Q2"][c].items() if k != "per_prompt"} for c in S.Q1_CONFIGS})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("q2-keys")
    k.add_argument("--out", default=str(Q2_KEYS_JSON))
    c = sub.add_parser("cont-gen")
    c.add_argument("--model-dir", required=True)
    c.add_argument("--device", default="cuda:0")
    c.add_argument("--out", required=True)
    c.add_argument("--batch", type=int, default=16)
    c.add_argument("--max-new", type=int, default=S.Q1_MAX_CONT_TOKENS)
    c.add_argument("--limit", type=int)
    r = sub.add_parser("ref")
    r.add_argument("--model-dir", required=True)
    r.add_argument("--device", default="cuda:0")
    r.add_argument("--cont", required=True)
    r.add_argument("--out-dir", required=True)
    r.add_argument("--limit", type=int)
    m = sub.add_parser("cmp")
    m.add_argument("--config", choices=["hf", "split"], required=True)
    m.add_argument("--model-dir", required=True)
    m.add_argument("--model-id", default=f"{S.SPLIT_REPO}@{S.SPLIT_REVISION}")
    m.add_argument("--device", default="cuda:0")
    m.add_argument("--cont", required=True)
    m.add_argument("--ref-dir", required=True)
    m.add_argument("--out-dir", required=True)
    m.add_argument("--server")
    m.add_argument("--k", type=int)
    m.add_argument("--j", type=int)
    m.add_argument("--expect-32b", type=lambda s: s != "0", default=True)
    m.add_argument("--limit", type=int)
    q = sub.add_parser("q2-ref")
    q.add_argument("--model-dir", required=True)
    q.add_argument("--device", default="cuda:0")
    q.add_argument("--out", required=True)
    q.add_argument("--max-new", type=int, default=S.Q2_NEW_TOKENS)
    q.add_argument("--limit", type=int)
    w = sub.add_parser("q2-run")
    w.add_argument("--model-dir", required=True)
    w.add_argument("--model-id", default=f"{S.SPLIT_REPO}@{S.SPLIT_REVISION}")
    w.add_argument("--device", default="cuda:0")
    w.add_argument("--ref", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--server", required=True)
    w.add_argument("--k", type=int, required=True)
    w.add_argument("--j", type=int, required=True)
    w.add_argument("--max-new", type=int, default=S.Q2_NEW_TOKENS)
    w.add_argument("--expect-32b", type=lambda s: s != "0", default=True)
    d = sub.add_parser("decide")
    d.add_argument("--pipeline-dirs", nargs=2, required=True, help="in-host floor dirs for U-min and U-max(16)")
    d.add_argument("--hw-dir", required=True)
    d.add_argument("--split-dirs", nargs=2, required=True, help="Q1 cmp dirs for U-min and U-max(16)")
    d.add_argument("--q2-ref", required=True)
    d.add_argument("--q2-runs", nargs=2, required=True, help="Q2 runs for U-min and U-max(16)")
    d.add_argument("--vocab-len", type=int, default=151669)
    d.add_argument("--fix-attempt", type=int, default=0, choices=[0, 1])
    d.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "q2-keys":
        I.write_json(a.out, {"seed": S.Q2_SEED, "per_bench": S.Q2_PER_BENCH, "keys": q2_keys()})
        print(open(a.out).read())
    else:
        {"cont-gen": cont_gen, "ref": ref_run, "cmp": cmp_run, "q2-ref": q2_ref, "q2-run": q2_run,
         "decide": decide}[a.cmd](a)


if __name__ == "__main__":
    main()
