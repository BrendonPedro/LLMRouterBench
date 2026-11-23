"""ROUTER-INTEG: the shared measurement recorder, the failure and fallback recorder, and the reproducibility manifest.

One row per request, whatever happened to it. A request that failed is a row with status "error": it is never dropped,
never retried silently and never turned into an answer. Rows are appended to JSONL files that are never rewritten.
Raw outputs stay outside every repository: they name hosts and devices.

Row fields (protocol, section 8): query id, router fold, exact model identifier, execution mode, hardware, precision and
quantisation, prompt and generation settings, sampling seed, generated output, time to first token, completion time,
output tokens, completion inside each deadline, stop or truncation reason, error, data-policy feasibility, run
identifier, phase (warmup / profile / shared / live_A / live_B) and attempt. Correctness is added by the scorer
(rint_score.py) in a separate file keyed by the row's id, so that no measurement row is ever edited.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rint_common as C  # noqa: E402
import rint_policy as P  # noqa: E402
import rint_spec as S  # noqa: E402

PHASES = ("warmup", "profile", "shared", "live_A", "live_B")
MEASURED_PHASES = ("shared", "live_A", "live_B")     # warm-up and profiling rows are never analysed as results


def settings(model, max_new, cutoff_s=None):
    """The prompt and generation settings of a request, as written into its row."""
    m = S.MODELS[model]
    return {"prompt_format": "one user turn, no system prompt, the checkpoint's chat template with its defaults, generation prompt added",
            "chat_template_kwargs": S.CHAT_TEMPLATE_KWARGS, "reasoning": "as the template and the model produce it (not disabled)",
            "temperature": S.TEMPERATURE, "top_p": S.TOP_P, "top_k": m["top_k"], "eos_ids": list(m["eos_ids"]),
            "context_tokens": S.CONTEXT_TOKENS, "max_new": int(max_new), "prefill_chunk": S.PREFILL_CHUNK,
            "cutoff_s": float(S.QUERY_CUTOFF_S if cutoff_s is None else cutoff_s), "decoding": "sampled, one draw per request, seeded",
            "batch": 1}


def header(freeze, site, role, facts, run_id, stage, dev=False, **extra):
    """What every file of a run starts with: the manifest of what produced it."""
    return dict({"schema": S.SCHEMA, "protocol": S.PROTOCOL, "experiment": S.NAME, "dev": bool(dev), "freeze": freeze, "run_id": run_id,
                 "stage": stage, "role": role, "host": facts, "site_sha256": site.get("_sha256"), "versions": C.versions(),
                 "utc_start": C.utc()}, **extra)


def row_id(phase, lane, key, config, attempt, n=0):
    return f"{phase}|{lane}|{key}|{config}|a{attempt}|n{n}"


def base_row(hdr, phase, lane, q, model, config, attempt, seed, max_new, model_identity=None, cutoff_s=None, n=0):
    mode = S.CONFIGS[config][1]
    st = settings(model, max_new, cutoff_s)
    return {"id": row_id(phase, lane, q["key"], config, attempt, n), "schema": S.SCHEMA, "protocol": S.PROTOCOL,
            "spec_hash": hdr["freeze"].get("spec_hash"), "run_id": hdr["run_id"], "dev": hdr["dev"], "phase": phase, "lane": lane,
            "attempt": attempt, "key": q["key"], "dataset": q["dataset"], "router_fold": q["fold"],
            "model": model, "model_name": S.MODELS[model]["name"], "model_repo": S.MODELS[model]["repo"],
            "model_revision": S.MODELS[model]["revision"], "model_identity": model_identity,
            "precision": "bfloat16 as published", "quantisation": "none", "config": config, "mode": mode,
            "hardware": {"edge_host_role": hdr["role"], "edge_gpus": hdr["host"].get("gpu_names"),
                         "compute": "edge GPU" if mode == S.EDGE else "server GPU (one), through the wire"},
            "settings": st, "settings_sha256": C.sha256_obj(st), "seed": int(seed), "n_prompt": q["n_prompt"][model],
            "ids_sha256": q["ids_sha256"][model],
            "data_policy_feasible": {pol: P.feasible(mode, pol)[0] for pol in (S.POLICY_A, S.POLICY_B)}, "utc_start": C.utc()}


def ok_row(row, res, text, extra=None):
    """Complete a row from a backend result. Raises if the backend ran another model than the row names."""
    if res.get("model") != row["model"]:
        raise RuntimeError(f"backend ran {res.get('model')}, the row names {row['model']}")
    if res.get("mode") != row["mode"]:
        raise RuntimeError(f"backend mode {res.get('mode')}, the row names {row['mode']}")
    st = res["stamps_ns"]
    total = st[-1] / 1e9
    row.update(status="ok", error=None, model_identity=res.get("model_identity") or row.get("model_identity"),
               backend_model=res["model"], tokens=[int(t) for t in res["tokens"]], n_out=len(res["tokens"]), text=text,
               stop=res["stop"], cutoff=bool(res["cutoff"]), truncated=res["stop"] in ("cutoff", "cap", "cancel"),
               ttft_s=st[0] / 1e9, total_s=total, per_token_s=(st[-1] - st[0]) / 1e9 / (len(st) - 1) if len(st) > 1 else None,
               stamps_s=[round(x / 1e9, 6) for x in st],
               ended_within_s={str(d): bool(total <= d) for d in S.DEADLINES_S},     # time only; deadline success also needs a correct answer
               n_prefill_steps=res.get("n_prefill_steps"), prefill_s=None if res.get("prefill_ns") is None else res["prefill_ns"] / 1e9,
               decode_forward_s_sum=None if res.get("decode_forward_ns_sum") is None else res["decode_forward_ns_sum"] / 1e9,
               decode_forward_s_median=None if res.get("decode_forward_ns_median") is None else res["decode_forward_ns_median"] / 1e9,
               utc_end=C.utc())
    if res.get("mock"):
        row["mock"] = True
    for k in ("bytes_out", "bytes_in", "tokens_after_cutoff", "server"):
        if k in res:
            row[k] = res[k]
    if res.get("server_last_token_ns") is not None:
        row["server_last_token_s"] = res["server_last_token_ns"] / 1e9
        row["wire_and_edge_s"] = total - res["server_last_token_ns"] / 1e9       # completion time not spent in the server's own clock
    row.update(extra or {})
    return row


def error_row(row, exc, extra=None, transport=False):
    """A request that failed. No output, no timing; counted as incorrect and late by every analysis.
    transport=True: the wire or the server was not reachable. Such a row is kept, the lane stops, and the request is
    repeated when the lane is resumed; the later row is the one that counts."""
    row.update(status="error", error={"type": type(exc).__name__, "message": str(exc)[:500]}, tokens=None, n_out=0, text=None,
               stop="error", cutoff=False, truncated=False, ttft_s=None, total_s=None, per_token_s=None,
               ended_within_s={str(d): False for d in S.DEADLINES_S}, transport_error=bool(transport), utc_end=C.utc())
    row.update(extra or {})
    return row


def event(hdr, kind, **body):
    """A failure, fallback or stop that is not itself a request: appended to events.jsonl of the run directory."""
    return dict({"schema": S.SCHEMA, "run_id": hdr["run_id"], "stage": hdr["stage"], "dev": hdr["dev"], "event": kind, "utc": C.utc()}, **body)


class Recorder:
    """Append-only writer of one lane's rows and of the run's events."""

    def __init__(self, run_dir, name, hdr):
        self.dir, self.name, self.hdr = Path(run_dir), name, hdr
        self.rows_path = self.dir / f"rows_{name}.jsonl"
        self.events_path = self.dir / "events.jsonl"
        self.dir.mkdir(parents=True, exist_ok=True)

    def existing(self):
        rows, torn = C.read_jsonl(self.rows_path)
        return rows, torn

    def write(self, row):
        C.append_jsonl(self.rows_path, row)
        if row.get("status") == "error":
            C.append_jsonl(self.events_path, event(self.hdr, "request_error", row_id=row["id"], key=row["key"], config=row["config"],
                                                   error=row["error"]))

    def event(self, kind, **body):
        C.append_jsonl(self.events_path, event(self.hdr, kind, **body))
