"""Corrected CV Mission -- Task 4: raw-query END-TO-END CPU timing of the MiniLM router.

Timed path, per query (nothing precomputed or cached):

    raw query text
      -> all-MiniLM-L6-v2 WordPiece tokenizer (truncation at 256 tokens)
      -> all-MiniLM-L6-v2 BertModel forward pass (no pooler)
      -> attention-mask mean pooling
      -> L2 normalisation
      -> MLP-64 score-distillation head (sklearn MLPRegressor.predict)
      -> routing decision (score > 0 -> Qwen3-8B, else DeepSeek-R1-0528-Qwen3-8B)

Pooling / truncation (mean pooling, L2 norm, max_length=256) is the
sentence-transformers all-MiniLM-L6-v2 recipe; it is VERIFIED here against the
cached embeddings used by every CV result (min cosine is reported and asserted).

Router head: NOT a new experiment. The fold-0 MLP-64 student of
run_corrected_nonlinear_cv.py is rebuilt with that script's own factory and
seed (fold-local GTE LogReg teacher on outer-train non-tie rows, student on all
outer-train queries). Identity with the Task 1 fold-0 head is checked through
n_iter_ (Task 1 fold 0: 55 epochs). No metric is computed; no fold is re-run.

Timed queries: every outer-TEST query of frozen fold 0 (real benchmark
queries, held out from the head). Warm-up: queries from the fold-0 TRAIN
portion, never timed. Timing runs in a fresh spawned child process so the peak
RSS reflects only the online deployment path (tokenizer + MiniLM + head), not the
GTE embedding cache used to rebuild the head.

GTE-Qwen2-7B end-to-end timing is NOT attempted: see corrected_efficiency.csv.

Writes diagnostics/stratified_cv/minilm_end_to_end_timing.csv (new file; refuses
to overwrite).
"""

import multiprocessing as mp
import os
import pickle
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cv_eval as cv  # noqa: E402

OUT_CSV = cv.OUT / "minilm_end_to_end_timing.csv"

FOLD = 0
MAX_LENGTH = 256
N_WARMUP = 32
BATCH_SIZES = (1, 32)
N_THREADS = 4
TASK1_FOLD0_EPOCHS = 55  # corrected_nonlinear_cv.csv, MLP-64 student, fold 0


def build_head(train):
    """Rebuild the Task 1 fold-0 MLP-64 student (same code path, same seed)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import Normalizer
    import run_corrected_nonlinear_cv as rc

    gte_cache = cv.load_embedding_cache(cv.GTE_CACHE, cv.GTE_MODEL)
    minilm_cache = cv.load_embedding_cache(cv.MINILM_CACHE, cv.MINILM_MODEL)
    tr_nt_mask = (train["target"] != -1).to_numpy()
    Xg_tr = Normalizer(norm="l2").fit_transform(
        cv.embedding_matrix(train["query"].tolist(), gte_cache))
    teacher = LogisticRegression(C=1.0, penalty="l2", solver="liblinear",
                                 max_iter=2000, random_state=42 + FOLD, class_weight=None)
    teacher.fit(Xg_tr[tr_nt_mask], train.loc[tr_nt_mask, "target"].to_numpy())
    Xm_tr = Normalizer(norm="l2").fit_transform(
        cv.embedding_matrix(train["query"].tolist(), minilm_cache))
    head = rc.make_student_mlp(FOLD)
    head.fit(Xm_tr, teacher.decision_function(Xg_tr))
    return head, minilm_cache


def _proc_status_mb(field):
    # VmHWM/VmRSS from /proc are per address space and reset on exec, unlike
    # getrusage().ru_maxrss, which Linux carries across fork+exec from the parent.
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1]) / 1024.0  # kB -> MiB
    return float("nan")


def _rss_mb():
    return _proc_status_mb("VmRSS")


def _peak_rss_mb():
    return _proc_status_mb("VmHWM")


def child_timing(head_path, warmup_queries, test_queries, queue):
    """Fresh interpreter: load only the online stack, then time it."""
    rss_start = _rss_mb()
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, BertModel

    torch.set_num_threads(N_THREADS)
    rss_after_imports = _rss_mb()

    t0 = time.perf_counter()
    path = snapshot_download(cv.MINILM_MODEL, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(path)
    model = BertModel.from_pretrained(path, add_pooling_layer=False).eval()
    with open(head_path, "rb") as f:
        head = pickle.load(f)
    load_sec = time.perf_counter() - t0
    rss_after_load = _rss_mb()
    encoder_params = int(sum(p.numel() for p in model.parameters()))

    def route(queries, stage_times=None):
        t_a = time.perf_counter()
        enc = tok(queries, padding=True, truncation=True, max_length=MAX_LENGTH,
                  return_tensors="pt")
        t_b = time.perf_counter()
        with torch.inference_mode():
            hidden = model(**enc).last_hidden_state
        t_c = time.perf_counter()
        with torch.inference_mode():
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        emb_np = emb.numpy()
        t_d = time.perf_counter()
        score = head.predict(emb_np)
        decision = np.where(score > 0, cv.QWEN, cv.DS)
        t_e = time.perf_counter()
        if stage_times is not None:
            stage_times["tokenize"].append(t_b - t_a)
            stage_times["encoder"].append(t_c - t_b)
            stage_times["pool_norm"].append(t_d - t_c)
            stage_times["head_decision"].append(t_e - t_d)
        return emb_np, score, decision

    # ---- warm-up (fold-0 TRAIN queries, untimed) ----
    for bs in BATCH_SIZES:
        for i in range(0, N_WARMUP, bs):
            route(warmup_queries[i:i + bs])

    results = []
    emb_b1 = score_b1 = None
    for bs in BATCH_SIZES:
        stages = {k: [] for k in ("tokenize", "encoder", "pool_norm", "head_decision")}
        call_sec, embs, scores, n_tokens = [], [], [], []
        for i in range(0, len(test_queries), bs):
            batch = test_queries[i:i + bs]
            t0 = time.perf_counter()
            e, s, _ = route(batch, stages)
            call_sec.append(time.perf_counter() - t0)
            embs.append(e)
            scores.append(s)
        total = sum(call_sec)
        n = len(test_queries)
        row = {
            "batch_size": bs,
            "n_timed_queries": n,
            "n_timed_calls": len(call_sec),
            "n_warmup_queries_per_batch_size": N_WARMUP,
            "total_timed_sec": total,
            "amortized_ms_per_query": 1000.0 * total / n,
            "mean_ms_per_call": 1000.0 * statistics.mean(call_sec),
            "median_ms_per_call": 1000.0 * statistics.median(call_sec),
            "p95_ms_per_call": 1000.0 * float(np.percentile(call_sec, 95)),
            "max_ms_per_call": 1000.0 * max(call_sec),
            "queries_per_sec": n / total,
        }
        for k, v in stages.items():
            row[f"stage_{k}_ms_per_query"] = 1000.0 * sum(v) / n
        results.append(row)
        if bs == 1:
            emb_b1, score_b1 = np.concatenate(embs), np.concatenate(scores)

    lengths = np.array([len(x) for x in tok(test_queries, add_special_tokens=True, verbose=False)["input_ids"]])
    env = {
        "encoder_model": cv.MINILM_MODEL,
        "encoder_params_loaded": encoder_params,
        "router_head": "MLPRegressor(64, relu) score-distillation student (Task 1 fold 0)",
        "max_length_tokens": MAX_LENGTH,
        "pooling": "attention-mask mean pooling + L2 normalisation",
        "query_tokens_mean_untruncated": float(lengths.mean()),
        "query_tokens_median_untruncated": float(np.median(lengths)),
        "frac_queries_truncated_at_max_length": float((lengths > MAX_LENGTH).mean()),
        "device": "cpu",
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "model_load_sec": load_sec,
        "rss_mb_at_child_start": rss_start,
        "rss_mb_after_imports": rss_after_imports,
        "rss_mb_after_model_load": rss_after_load,
        "peak_rss_mb_online_process": _peak_rss_mb(),
    }
    queue.put((results, env, emb_b1, score_b1))


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def mem_total_gb():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal"):
            return int(line.split()[1]) / 1024 ** 2
    return float("nan")


def main():
    assert not OUT_CSV.exists(), f"refusing to overwrite {OUT_CSV}"
    pool = cv.load_full_pool_with_folds()
    train = pool[pool["fold"] != FOLD].copy()
    test = pool[pool["fold"] == FOLD].copy()

    head, minilm_cache = build_head(train)
    assert head.n_iter_ == TASK1_FOLD0_EPOCHS, (
        f"rebuilt head ran {head.n_iter_} epochs, Task 1 fold 0 ran {TASK1_FOLD0_EPOCHS}")
    head_params = int(sum(w.size for w in head.coefs_) + sum(b.size for b in head.intercepts_))
    head_disk_bytes = cv.model_size_bytes(head)

    test_queries = test["query"].tolist()
    warmup_queries = train["query"].sample(n=N_WARMUP, random_state=0).tolist()
    cached = cv.embedding_matrix(test_queries, minilm_cache)
    cached = cached / np.linalg.norm(cached, axis=1, keepdims=True)
    cached_score = head.predict(cached)
    del minilm_cache

    with tempfile.TemporaryDirectory() as tmp:
        head_path = os.path.join(tmp, "head.pkl")
        with open(head_path, "wb") as f:
            pickle.dump(head, f)
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(target=child_timing, args=(head_path, warmup_queries, test_queries, q))
        p.start()
        results, env, emb_b1, score_b1 = q.get()
        p.join()
        assert p.exitcode == 0, f"timing child exited with {p.exitcode}"

    # ---- validation: end-to-end path reproduces the cached-embedding path ----
    cos = np.sum(emb_b1 * cached, axis=1)
    agreement = float(((score_b1 > 0) == (cached_score > 0)).mean())
    max_score_diff = float(np.max(np.abs(score_b1 - cached_score)))
    assert cos.min() > 0.9999, f"end-to-end embeddings diverge from cache (min cos {cos.min()})"

    common = {
        "path": "raw query -> tokenizer -> MiniLM encoder -> mean pool -> L2 norm -> MLP-64 head -> decision",
        "timing_label": "END-TO-END raw-query CPU inference (single process, in-memory model)",
        "fold_of_timed_queries": FOLD,
        "router_head_params": head_params,
        "router_head_disk_bytes": head_disk_bytes,
        "head_epochs_check": int(head.n_iter_),
        "e2e_vs_cached_embedding_min_cosine": float(cos.min()),
        "e2e_vs_cached_decision_agreement": agreement,
        "e2e_vs_cached_max_abs_score_diff": max_score_diff,
        "cpu_model": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "mem_total_gb": mem_total_gb(),
        "platform": platform.platform(),
        **env,
    }
    out = pd.DataFrame([{**r, **common} for r in results])
    out.to_csv(OUT_CSV, index=False)

    for r in results:
        print(f"batch={r['batch_size']:>2}  n={r['n_timed_queries']}  "
              f"mean/call={r['mean_ms_per_call']:.2f}ms  median/call={r['median_ms_per_call']:.2f}ms  "
              f"amortized/query={r['amortized_ms_per_query']:.2f}ms  qps={r['queries_per_sec']:.1f}")
    print(f"peak RSS (online process) = {env['peak_rss_mb_online_process']:.0f} MB; "
          f"encoder params = {env['encoder_params_loaded']:,}; head params = {head_params:,}")
    print(f"validation: min cos vs cache = {cos.min():.6f}, decision agreement = {agreement:.4f}, "
          f"max |score diff| = {max_score_diff:.2e}")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
