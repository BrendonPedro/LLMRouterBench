"""QWEN-SELECT test fixtures: tiny random Qwen3 checkpoints with a tiny tokenizer, a synthetic workload, site files.

The three models have the layer counts of the real ones (64, 40, 36), so the k = 16 cut is the real layer
allocation. Everything is random: nothing computed here is a result.
"""
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
import qsel_common as C  # noqa: E402
import qsel_runtime as RT  # noqa: E402
import qsel_spec as S  # noqa: E402
import qsel_workload as WL  # noqa: E402

PY = [sys.executable, "-I", "-B"]
BENCH = str(HERE / "qsel_bench.py")
TMP = Path(tempfile.mkdtemp(prefix="qseltest_"))
SPECIALS = ["<unk>", "<|im_start|>", "<|im_end|>", "<|endoftext|>", "user", "assistant"]
WORDS = [f"w{i}" for i in range(180)]
EOS = [2, 3]
_CACHE = {}


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def tiny_models():
    """{"32B": dir, "14B": dir, "8B": dir}: random bfloat16 Qwen3 checkpoints with the real layer counts."""
    if "models" in _CACHE:
        return _CACHE["models"]
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import Qwen3Config, Qwen3ForCausalLM
    vocab = {t: i for i, t in enumerate(SPECIALS + WORDS)}
    out = {}
    for seed, (name, m) in enumerate(S.MODELS.items()):
        torch.manual_seed(seed)
        cfg = Qwen3Config(torch_dtype=torch.bfloat16, vocab_size=len(vocab), hidden_size=64, intermediate_size=128, num_hidden_layers=m["n_layers"],
                          num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=32768, tie_word_embeddings=False,
                          rope_theta=10000.0, rms_norm_eps=1e-6)
        model = Qwen3ForCausalLM(cfg).to(torch.bfloat16).eval()
        model.lm_head.weight.data.mul_(40.0)            # peaked next-token distributions: the check set has decisive positions
        model.lm_head.weight.data[EOS] = 0.0            # the random models never end early; EOS handling has its own tests
        d = TMP / f"tiny{name}"
        model.save_pretrained(d, safe_serialization=True)
        tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
        tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tok.add_special_tokens(SPECIALS[:4])
        tok.save(str(d / "tokenizer.json"))
        (d / "tokenizer_config.json").write_text(json.dumps({
            "tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "<unk>", "eos_token": "<|im_end|>", "pad_token": "<|endoftext|>",
            "chat_template": "{% for m in messages %}<|im_start|> {{ m['role'] }} {{ m['content'] }} <|im_end|> {% endfor %}"
                             "{% if add_generation_prompt %}<|im_start|> assistant{% endif %}"}))
        (d / "generation_config.json").write_text(json.dumps({"eos_token_id": EOS}))
        out[name] = str(d)
    _CACHE["models"] = out
    return out


def dev_workload(n_per_bench=5, name="wl"):
    """(public dict path, private file path): a synthetic workload whose gold answer is what the 64-layer model
    writes, so the split and the whole 32B stand-in are 'correct' and the other models are not."""
    key = ("wl", n_per_bench, name)
    if key in _CACHE:
        return _CACHE[key]
    import random
    import torch
    _, R = RT.k16()
    R.setup_torch()
    md = tiny_models()
    tok = WL.load_tokenizer(md["32B"])
    hf = R.hf_model(md["32B"], "cpu")
    rng = random.Random("qsel-dev-workload")
    rows = []
    for b in S.BENCHMARKS:
        for i in range(n_per_bench):
            n_words = rng.choice([4, 9, 20, 44, 70])                           # some prompts are longer than one dev prefill chunk (48)
            prompt = " ".join(rng.choice(WORDS) for _ in range(n_words))
            ids = WL.tokenize(tok, prompt)
            res = RT.whole_generate(torch, hf, ids, 12, EOS, "cpu", 48, 1e9, R.E0RT._sync)
            rows.append({"key": f"{b}:{i + 1}", "dataset": b, "rank": i, "prompt": prompt, "answer": tok.decode(res["tokens"], skip_special_tokens=True)})
    pub = WL.build(rows, tok)
    pp, pv = TMP / f"{name}_public.json", TMP / f"{name}_private.jsonl"
    pp.write_text(json.dumps(pub))
    pv.write_bytes(WL.private_bytes(rows))
    _CACHE[key] = (str(pp), str(pv))
    return _CACHE[key]


def site(role, run_dir, ports, extra_dev=None, gpu_uuid=None, name=None, template=False):
    """A dev site file for CPU runs. ports = (split, remote)."""
    pub, priv = dev_workload()
    md = tiny_models()
    d = {"role": role, "expected_hostname": socket.gethostname(), "model_dirs": md if role == "edge" else {"32B": md["32B"], "14B": md["14B"]},
         "device": "cpu", "gpu_uuid": gpu_uuid, "split_port": ports[0], "remote_port": ports[1], "token_file": str(TMP / "token"),
         "run_dir": str(run_dir), "workload_private": priv, "hash_cache": str(TMP / "hash_cache.json"), "ports_in_use_by_others": [1],
         "excluded_gpu_uuids": [],
         "dev": dict({"workload_public": pub, "max_new": 12, "context": 256, "n_cont": 8, "cutoff_s": 300.0, "first_wait_s": 120, "idle_exit_s": 120},
                     **(extra_dev or {}))}
    p = TMP / (name or f"site_{role}_{Path(run_dir).name}{'_tpl' if template else ''}.json")
    p.write_text(json.dumps(d, indent=1))
    return str(p)


def bench(*args, env=None, timeout=900):
    e = dict(os.environ, CUDA_VISIBLE_DEVICES="", TOKENIZERS_PARALLELISM="false", **(env or {}))
    return subprocess.run(PY + [BENCH, *args], capture_output=True, text=True, env=e, timeout=timeout)


def start_server(cmd, site_file, env=None):
    e = dict(os.environ, CUDA_VISIBLE_DEVICES="", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1", **(env or {}))
    return subprocess.Popen(PY + [BENCH, cmd, "--dev", "--site", site_file], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=e)


def wait_ready(proc, t=180):
    """Blocks until the server prints its ready line."""
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
