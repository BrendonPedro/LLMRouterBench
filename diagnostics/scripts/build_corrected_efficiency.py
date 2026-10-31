"""Corrected CV Mission -- Task 3: corrected efficiency table (no experiment re-run).

Assembles diagnostics/stratified_cv/corrected_efficiency.csv purely from saved
artefacts:
    efficiency_results.csv          (original 5-fold efficiency rows)
    corrected_nonlinear_cv.csv      (Task 1: Ridge vs MLP-64 MiniLM students)
    gte_classifier_capacity.csv     (Task 2: GTE LogReg / MLP-64 / RBF-SVM)
    minilm_end_to_end_timing.csv    (Task 4: raw-query end-to-end MiniLM timing)
    local GTE-Qwen2-7B-instruct config.json (encoder parameter count)

Corrections applied relative to efficiency_results.csv:
  * Encoder, router-head, total-deployed and trainable parameters are separate
    columns. The old distillation "total_params" (3,970) summed the offline
    teacher head (3,585) and the online Ridge student (385); only 385 router-head
    parameters are deployed.
  * GTE / MiniLM / distillation "ms_per_query" divided the head inference time
    for ALL test rows (ties included, ~2,293) by the non-tie count (~467), and
    covered the router head on cached embeddings only. Here the same timings are
    divided by the rows actually scored and labelled router-head-only.
  * Cached-embedding timings are never labelled end-to-end. The only end-to-end
    neural-router timing is the measured MiniLM raw-query path.
  * GTE end-to-end latency is not estimated.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cv_eval as cv  # noqa: E402

OUT_CSV = cv.OUT / "corrected_efficiency.csv"

CACHED_LABEL = ("router-head-only cached-embedding inference, batch over the whole test fold, "
                "amortized per scored row; EXCLUDES tokenizer and encoder; NOT end-to-end latency")
GTE_NOT_MEASURED = ("NOT MEASURED on this machine: end-to-end GTE timing was not measured because "
                    "the 7B encoder cannot be fairly loaded within available memory")


def gte_encoder_params(config):
    """Qwen2Model parameters used for embedding (AutoModel; lm_head not used)."""
    h, inter, L, V = (config["hidden_size"], config["intermediate_size"],
                      config["num_hidden_layers"], config["vocab_size"])
    kv = config["num_key_value_heads"] * (h // config["num_attention_heads"])
    attn = (h * h + h) + 2 * (h * kv + kv) + h * h  # q,k,v with bias; o without
    mlp = 3 * h * inter                               # gate, up, down
    layer = attn + mlp + 2 * h                        # + 2 RMSNorm
    return V * h + L * layer + h                      # embed + layers + final norm


def mean_row(df, method, col_method="method"):
    r = df[(df[col_method] == method) & (df["fold"] == "mean")]
    assert len(r) == 1, method
    return r.iloc[0]


def std_row(df, method):
    return df[(df["method"] == method) & (df["fold"] == "std")].iloc[0]


def main():
    assert not OUT_CSV.exists(), f"refusing to overwrite {OUT_CSV}"
    eff = pd.read_csv(cv.OUT / "efficiency_results.csv").set_index("method")
    tn = pd.read_csv(cv.OUT / "corrected_nonlinear_cv.csv")
    cap = pd.read_csv(cv.OUT / "gte_classifier_capacity.csv")
    e2e = pd.read_csv(cv.OUT / "minilm_end_to_end_timing.csv").set_index("batch_size")

    from huggingface_hub import snapshot_download
    import json
    gte_cfg = json.loads((Path(snapshot_download(cv.GTE_MODEL, local_files_only=True,
                                                 allow_patterns=["config.json"]))
                          / "config.json").read_text())
    gte_params = gte_encoder_params(gte_cfg)
    gte_fp32_gb, gte_bf16_gb = gte_params * 4 / 1e9, gte_params * 2 / 1e9
    mem_gb = float(e2e.loc[1, "mem_total_gb"])
    gte_src = (f"derived from local config.json (Qwen2Model, no lm_head): {gte_params:,}; weights are "
               f"NOT present locally; fp32 weights ~{gte_fp32_gb:.1f} GB, bf16 ~{gte_bf16_gb:.1f} GB "
               f"vs {mem_gb:.1f} GiB machine RAM")

    mini_params = int(e2e.loc[1, "encoder_params_loaded"])
    mini_src = (f"verified from loaded BertModel(add_pooling_layer=False): {mini_params:,} "
                f"(checkpoint also stores an unused 147,840-param pooler; 22,713,216 incl. pooler)")
    n_all = float(mean_row(tn, "Frozen-MiniLM MLP-64 student (nonlinear; corrected)")["n_test_all_rows"])
    n_nt = float(mean_row(tn, "Frozen-MiniLM MLP-64 student (nonlinear; corrected)")["n_test_nontie"])

    b1, b32 = e2e.loc[1], e2e.loc[32]
    e2e_cols_measured = {
        "end_to_end_status": "MEASURED: raw query -> tokenizer -> MiniLM -> mean pool -> L2 norm -> MLP-64 -> decision (CPU)",
        "e2e_batch_size": 1,
        "e2e_n_timed_queries": int(b1["n_timed_queries"]),
        "e2e_mean_ms_per_query": b1["mean_ms_per_call"],
        "e2e_median_ms_per_query": b1["median_ms_per_call"],
        "e2e_p95_ms_per_query": b1["p95_ms_per_call"],
        "e2e_queries_per_sec": b1["queries_per_sec"],
        "e2e_batch32_amortized_ms_per_query": b32["amortized_ms_per_query"],
        "e2e_batch32_queries_per_sec": b32["queries_per_sec"],
        "e2e_encoder_stage_ms_per_query_batch1": b1["stage_encoder_ms_per_query"],
        "e2e_head_stage_ms_per_query_batch1": b1["stage_head_decision_ms_per_query"],
        "e2e_peak_rss_mb": b1["peak_rss_mb_online_process"],
        "e2e_environment": (f"{b1['cpu_model']}, {int(b1['torch_threads'])} torch threads, CPU only "
                            f"(cuda_available={b1['cuda_available']}), {mem_gb:.1f} GiB RAM, WSL2, "
                            f"torch {b1['torch_version']}, transformers {b1['transformers_version']}"),
    }
    mini_not_measured = {"end_to_end_status": ("not measured for this head (only the MLP-64 student path "
                                               "was timed end-to-end; encoder stage is shared)")}

    rows = []

    # ---- TF-IDF ----
    e = eff.loc["TF-IDF + LogReg"]
    rows.append({
        "method": "TF-IDF + LogReg", "macro_f1_mean": np.nan, "macro_f1_std": np.nan,
        "macro_f1_source": "router_cv_results.csv (not re-read here)",
        "encoder_model": "none (TF-IDF 1-2gram vectorizer, 20,000 features)",
        "encoder_params": 0, "encoder_params_source": "no neural encoder; IDF weights are fitted statistics, not parameters",
        "router_head_type": "LogisticRegression on TF-IDF", "router_head_params": int(e["total_params"]),
        "total_deployed_params": int(e["total_params"]), "total_deployed_params_status": "verified (saved run)",
        "trainable_router_params": int(e["trainable_params"]),
        "router_head_disk_bytes": int(e["model_disk_bytes"]),
        "router_head_disk_note": "pickle of vectorizer + LogReg",
        "cached_emb_head_train_sec_per_fold": e["train_sec_per_fold_mean"],
        "cached_emb_head_infer_ms_per_query": e["ms_per_query"],
        "cached_emb_rows_scored_per_fold": n_nt,
        "timing_scope_label": ("raw-text-to-decision (vectorizer transform + LogReg), batch over the "
                               "whole non-tie test fold, amortized; NOT single-query latency"),
        "end_to_end_status": "partially: raw-text input is timed but only as a whole-fold batch; no per-query latency",
        "online_encoder_required": "none (TF-IDF vectorizer)",
        "offline_teacher_required": "no", "teacher_required_at_deployment": "no",
        "previous_efficiency_row": "TF-IDF + LogReg", "previous_ms_per_query_as_reported": e["ms_per_query"],
        "relabel_note": "denominator correct (only non-tie rows scored); label as batch-amortized, not latency",
    })

    # ---- GTE rows ----
    gte_rows = [
        ("GTE LogReg + default cutoff", "GTE Direct LogReg", "LogisticRegression"),
        ("GTE LogReg + training-only threshold", "GTE + train-only threshold", "LogisticRegression + train-only cutoff"),
        ("GTE MLP-64 + training-only threshold", None, "MLPClassifier(64) + train-only cutoff"),
        ("GTE RBF-SVM + training-only threshold", None, "SVC(rbf) + train-only cutoff"),
    ]
    for method, prev, head_type in gte_rows:
        c, s = mean_row(cap, method), std_row(cap, method)
        head_params = int(round(c["head_params"]))
        trainable = head_params
        disk, disk_note = np.nan, "not measured"
        note = "new row (Task 2); head timing is cached-embedding only"
        prev_ms = np.nan
        if prev is not None:
            e = eff.loc[prev]
            disk, disk_note = int(e["model_disk_bytes"]), "pickle of LogReg"
            train_sec = e["train_sec_per_fold_mean"]
            infer_ms = 1000.0 * e["infer_sec_per_fold_mean"] / n_all
            rows_scored = n_all
            prev_ms = e["ms_per_query"]
            note = (f"old ms_per_query divided ALL-row ({n_all:.0f}) head time by non-tie count "
                    f"({n_nt:.0f}): inflated x{n_all / n_nt:.2f}; and it is router-head-only on cached "
                    f"GTE embeddings, not 7B-encoder latency; old deploy label correct that GTE 7B is needed online")
        else:
            train_sec = c["head_train_sec_cached_emb"]
            infer_ms = 1000.0 * c["head_infer_sec_cached_emb_test_nontie"] / c["n_test_nontie"]
            rows_scored = c["n_test_nontie"]
        if "SVM" in method:
            n_sv = (head_params - 1) / (gte_cfg["hidden_size"] + 1)
            trainable = int(round(n_sv + 1))
            disk_note = (f"not measured; head stores ~{n_sv:.0f} support vectors x 3,584 dims "
                         f"(={head_params:,} stored values, fold mean)")
        rows.append({
            "method": method, "macro_f1_mean": c["macro_f1"], "macro_f1_std": s["macro_f1"],
            "macro_f1_source": "gte_classifier_capacity.csv",
            "encoder_model": cv.GTE_MODEL, "encoder_params": gte_params, "encoder_params_source": gte_src,
            "router_head_type": head_type, "router_head_params": head_params,
            "total_deployed_params": gte_params + head_params,
            "total_deployed_params_status": "approximate: encoder count derived from config, not from loaded weights",
            "trainable_router_params": trainable,
            "router_head_disk_bytes": disk, "router_head_disk_note": disk_note,
            "cached_emb_head_train_sec_per_fold": train_sec,
            "cached_emb_head_infer_ms_per_query": infer_ms,
            "cached_emb_rows_scored_per_fold": rows_scored,
            "timing_scope_label": CACHED_LABEL,
            "end_to_end_status": GTE_NOT_MEASURED,
            "online_encoder_required": f"{cv.GTE_MODEL} (~{gte_params / 1e9:.2f}B params) per query",
            "offline_teacher_required": "no", "teacher_required_at_deployment": "no",
            "previous_efficiency_row": prev or "", "previous_ms_per_query_as_reported": prev_ms,
            "relabel_note": note,
        })

    # ---- MiniLM direct LogReg ----
    e = eff.loc["MiniLM Direct"]
    rows.append({
        "method": "MiniLM Direct LogReg", "macro_f1_mean": np.nan, "macro_f1_std": np.nan,
        "macro_f1_source": "router_cv_results.csv (not re-read here)",
        "encoder_model": cv.MINILM_MODEL, "encoder_params": mini_params, "encoder_params_source": mini_src,
        "router_head_type": "LogisticRegression", "router_head_params": int(e["total_params"]),
        "total_deployed_params": mini_params + int(e["total_params"]),
        "total_deployed_params_status": "verified (loaded encoder + saved head)",
        "trainable_router_params": int(e["trainable_params"]),
        "router_head_disk_bytes": int(e["model_disk_bytes"]), "router_head_disk_note": "pickle of LogReg",
        "cached_emb_head_train_sec_per_fold": e["train_sec_per_fold_mean"],
        "cached_emb_head_infer_ms_per_query": 1000.0 * e["infer_sec_per_fold_mean"] / n_all,
        "cached_emb_rows_scored_per_fold": n_all,
        "timing_scope_label": CACHED_LABEL, **mini_not_measured,
        "online_encoder_required": f"{cv.MINILM_MODEL} per query",
        "offline_teacher_required": "no", "teacher_required_at_deployment": "no",
        "previous_efficiency_row": "MiniLM Direct", "previous_ms_per_query_as_reported": e["ms_per_query"],
        "relabel_note": (f"old ms_per_query inflated x{n_all / n_nt:.2f} (all-row time / non-tie count); "
                         "head-only on cached embeddings, not end-to-end"),
    })

    # ---- MiniLM students (Task 1) ----
    e = eff.loc["Frozen-MiniLM distillation"]
    teacher_note = (f"YES (training only): GTE-Qwen2-7B embeddings of outer-train queries + "
                    f"LogReg teacher ({int(e['teacher_params']):,} params)")
    for method, head_type, prev in (
        ("Frozen-MiniLM Ridge student (linear; previous 'distillation' row)", "Ridge regression on teacher score",
         "Frozen-MiniLM distillation"),
        ("Frozen-MiniLM MLP-64 student (nonlinear; corrected)", "MLPRegressor(64, relu) on teacher score", None),
    ):
        c, s = mean_row(tn, method), std_row(tn, method)
        head_params = int(c["head_params"])
        is_mlp = prev is None
        row = {
            "method": method, "macro_f1_mean": c["macro_f1"], "macro_f1_std": s["macro_f1"],
            "macro_f1_source": "corrected_nonlinear_cv.csv",
            "encoder_model": cv.MINILM_MODEL, "encoder_params": mini_params, "encoder_params_source": mini_src,
            "router_head_type": head_type, "router_head_params": head_params,
            "total_deployed_params": mini_params + head_params,
            "total_deployed_params_status": "verified (loaded encoder + head)",
            "trainable_router_params": head_params,
            "router_head_disk_bytes": int(b1["router_head_disk_bytes"]) if is_mlp else int(e["model_disk_bytes"]),
            "router_head_disk_note": ("sklearn pickle of fold-0 head incl. Adam optimizer state; raw float64 "
                                      f"weights = {head_params * 8:,} bytes") if is_mlp else "pickle of Ridge",
            "cached_emb_head_train_sec_per_fold": c["head_train_sec_cached_emb"],
            "cached_emb_head_infer_ms_per_query": 1000.0 * c["head_infer_sec_cached_emb_full_test_fold"] / c["n_test_all_rows"],
            "cached_emb_rows_scored_per_fold": c["n_test_all_rows"],
            "timing_scope_label": CACHED_LABEL,
            "online_encoder_required": f"{cv.MINILM_MODEL} per query (GTE NOT needed online)",
            "offline_teacher_required": teacher_note, "teacher_required_at_deployment": "no",
            "previous_efficiency_row": prev or "",
            "previous_ms_per_query_as_reported": e["ms_per_query"] if prev else np.nan,
        }
        if is_mlp:
            row.update(e2e_cols_measured)
            row["relabel_note"] = ("new row; cached-head train time excludes the offline GTE teacher; "
                                   "end-to-end figures are the measured raw-query path")
        else:
            row.update(mini_not_measured)
            row["relabel_note"] = (f"old total_params {int(e['total_params'])} = teacher head "
                                   f"{int(e['teacher_params'])} + student {head_params}: only {head_params} "
                                   f"router-head params (+ MiniLM encoder) are deployed; old ms_per_query "
                                   f"inflated x{n_all / n_nt:.2f} and head-only (cached timing in this row is from the "
                                   f"Task 1 re-run, same run as its F1); the old 0.5922 'distillation' "
                                   f"row is this linear Ridge student, not the MLP-64")
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(out[["method", "macro_f1_mean", "encoder_params", "router_head_params", "total_deployed_params",
               "trainable_router_params", "cached_emb_head_infer_ms_per_query", "end_to_end_status"]]
          .to_string(index=False))
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
