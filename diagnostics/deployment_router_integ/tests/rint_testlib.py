"""ROUTER-INTEG test fixtures (CPU only): tiny random Qwen3 checkpoints, tiny tokenizers, a fixture workload, site files.

Everything here is random or synthetic: nothing computed with it is a result. The tiny DeepSeek stand-in carries the
real checkpoint's position-scaling configuration (yarn), which is the one architectural setting in which the two real
configurations differ.
"""
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("RINT_QSEL_REPO", str(Path.home() / "research" / "LLMRouterBench-qwen-select"))
import rint_common as C  # noqa: E402
import rint_spec as S  # noqa: E402
import rint_workload as WL  # noqa: E402

PY = [sys.executable, "-I", "-B"]
RUN = str(HERE / "rint_run.py")
TMP = Path(tempfile.mkdtemp(prefix="rinttest_"))
SPECIALS = ["<unk>", "<bos>", "<eos>", "<end>", "<user>", "<assistant>"]
WORDS = [f"w{i}" for i in range(180)]
EOS = {S.DS: [2], S.QW: [2, 3]}
TEMPLATES = {
    S.DS: "<bos> {% for m in messages %}<user> {{ m['content'] }} <assistant>{% endfor %}",
    S.QW: "{% for m in messages %}<user> {{ m['role'] }} {{ m['content'] }} <eos> {% endfor %}{% if add_generation_prompt %}<user> assistant{% endif %}",
}
_CACHE = {}


def encoder_dir():
    """The real router encoder, if this machine has it (RINT_TEST_ENCODER_DIR or the Hugging Face cache)."""
    p = os.environ.get("RINT_TEST_ENCODER_DIR")
    if p and Path(p).is_dir():
        return p
    base = Path.home() / ".cache" / "huggingface" / "hub" / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots" / S.ROUTER_ENCODER_REVISION
    return str(base) if base.is_dir() else None


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def tiny_models():
    """{"DS": dir, "QW": dir}: random bfloat16 Qwen3 checkpoints with the real layer count and tiny width."""
    if "models" in _CACHE:
        return _CACHE["models"]
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import Qwen3Config, Qwen3ForCausalLM
    vocab = {t: i for i, t in enumerate(SPECIALS + WORDS)}
    out = {}
    for seed, name in enumerate(S.MODEL_ORDER):
        torch.manual_seed(100 + seed)
        extra = {"rope_scaling": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768,
                                  "attn_factor": 0.8782488562869419}, "max_position_embeddings": 131072} if name == S.DS else {"max_position_embeddings": 40960}
        cfg = Qwen3Config(torch_dtype=torch.bfloat16, vocab_size=len(vocab), hidden_size=64, intermediate_size=128,
                          num_hidden_layers=S.MODELS[name]["n_layers"], num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                          tie_word_embeddings=False, rope_theta=1000000.0, rms_norm_eps=1e-6, **extra)
        model = Qwen3ForCausalLM(cfg).to(torch.bfloat16).eval()
        model.lm_head.weight.data.mul_(40.0)
        model.lm_head.weight.data[[2, 3]] = 0.0     # the random models rarely end early; EOS handling has its own tests
        d = TMP / f"tiny{name}"
        model.save_pretrained(d, safe_serialization=True)
        tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
        tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tok.add_special_tokens(SPECIALS[:4])
        tok.save(str(d / "tokenizer.json"))
        (d / "tokenizer_config.json").write_text(json.dumps({
            "tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "<unk>", "bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<end>",
            "chat_template": TEMPLATES[name]}))
        out[name] = str(d)
    _CACHE["models"] = out
    return out


def fixture_rows(n_eval=8, n_profile=2):
    """Synthetic queries with benchmark names, folds, original outcomes and (if the encoder is here) real router scores."""
    import random
    rng = random.Random("rint-dev-workload")
    benches = ["gpqa", "medqa", "bbh", "meld"]
    classes = ["both_correct", "both_wrong", "ds_only", "qw_only"]
    sc = {"both_correct": (1, 1), "both_wrong": (0, 0), "ds_only": (1, 0), "qw_only": (0, 1)}
    rows = []
    for i in range(n_eval + n_profile):
        prompt = " ".join(rng.choice(WORDS) for _ in range(rng.choice([4, 9, 20, 44, 70])))
        ds, c4 = benches[i % 4], classes[(i // 2) % 4]
        a, b = sc[c4]
        rows.append({"key": f"{ds}:{900000 + i}", "dataset": ds, "index": 900000 + i, "fold": i % S.ROUTER_FOLDS,
                     "query_sha1": hashlib.sha1(prompt.encode()).hexdigest()[:16], "prompt": prompt, "ground_truth": "A",
                     "orig": {S.DS: {"score": a, "completion_tokens": 10, "prompt_tokens": -1}, S.QW: {"score": b, "completion_tokens": 10, "prompt_tokens": -1}},
                     "class4": c4, "router_score": 0.25 if i % 3 else -0.25, "router_choice": S.QW if i % 3 else S.DS})
    return rows[:n_eval], rows[n_eval:]


def dev_workload(name="wl", with_router=True):
    """(public path, private path, fold-map path). Router scores are the real router's on the fixture text when the
    encoder is available, so that the pipeline's router can be compared with a pinned decision."""
    key = ("wl", name, with_router)
    if key in _CACHE:
        return _CACHE[key]
    md = tiny_models()
    toks = {m: WL.load_tokenizer(md[m]) for m in S.MODEL_ORDER}
    ev, prof = fixture_rows()
    fmap = {r["key"]: [r["fold"], r["query_sha1"]] for r in ev + prof}
    fp = TMP / f"{name}_foldmap.json"
    fp.write_text(json.dumps(fmap))
    enc = encoder_dir()
    if with_router and enc:
        import rint_router as RR
        r = RR.Router(enc, threads=2, check=False, fold_map=RR.FoldMap(fp))
        for row in ev + prof:
            d = r.route(row["key"], row["prompt"])
            row["router_score"], row["router_choice"] = d["score"], d["choice"]
    lookup = {r["key"]: [r["fold"], r["dataset"], 0, 1] for r in ev + prof}
    wl, body = WL.assemble(ev, prof, toks, lookup, ev + prof)
    pp, pv = TMP / f"{name}_public.json", TMP / f"{name}_private.jsonl"
    pp.write_text(json.dumps(wl))
    pv.write_bytes(body)
    _CACHE[key] = (str(pp), str(pv), str(fp))
    return _CACHE[key]


def site(role, run_dir, port, extra_dev=None, name=None):
    pub, priv, fmap = dev_workload()
    d = {"role": role, "expected_hostname": socket.gethostname(), "qsel_repo": os.environ["RINT_QSEL_REPO"], "model_dirs": tiny_models(),
         "device": "cpu", "gpu_uuid": None, "remote_port": port, "token_file": str(TMP / "token"), "run_dir": str(run_dir),
         "workload_private": priv, "hash_cache": str(TMP / "hash_cache.json"), "router_encoder_dir": encoder_dir(),
         "ports_in_use_by_others": [1], "excluded_gpu_uuids": [],
         "dev": dict({"workload_public": pub, "fold_map": fmap, "eos_ids": EOS, "max_new": 12, "context": 256, "cutoff_s": 300.0, "chunk": 48,
                      "first_wait_s": 120, "idle_exit_s": 120, "warmup_new": 3, "limit_s": 900, "allotment_h": 1.0}, **(extra_dev or {}))}
    p = TMP / (name or f"site_{role}_{Path(run_dir).name}.json")
    p.write_text(json.dumps(d, indent=1))
    return str(p)


def run(*args, env=None, timeout=900):
    e = dict(os.environ, CUDA_VISIBLE_DEVICES="", TOKENIZERS_PARALLELISM="false", **(env or {}))
    return subprocess.run(PY + [RUN, *args], capture_output=True, text=True, env=e, timeout=timeout)


def start_server(site_file, stage="shared", models=None):
    e = dict(os.environ, CUDA_VISIBLE_DEVICES="", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    cmd = PY + [RUN, "serve-remote", "--dev", "--site", site_file, "--stage", stage] + (["--models", models] if models else [])
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=e)


def wait_ready(proc, t=240):
    end = time.time() + t
    while time.time() < end:
        line = proc.stdout.readline()
        if '"ready": true' in line:
            return json.loads(line)
        if not line and proc.poll() is not None:
            break
    raise AssertionError("server not ready: " + (proc.stdout.read() if proc.poll() is not None else "timeout"))


def stop_server(proc, t=120):
    import signal
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
    try:
        proc.wait(t)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise AssertionError("server did not stop")
    return proc.stdout.read()


def rows_of(run_dir, name):
    return C.read_jsonl(Path(run_dir) / f"rows_{name}.jsonl")[0]
