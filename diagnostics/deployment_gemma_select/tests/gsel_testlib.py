"""Helpers for the GEMMA-SELECT CPU tests: a tiny random Gemma-4-shaped checkpoint, a site file, a way to run commands."""
import json
import os
import string
import subprocess
import sys
import tempfile
from pathlib import Path

GS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GS))
SPECIALS = ["<pad>", "<eos>", "<bos>", "<unk>", "<start_of_turn>", "<end_of_turn>"]
CHAT_TEMPLATE = ("{{ bos_token }}{% for m in messages %}<start_of_turn>{{ m['role'] }}\n{{ m['content'] }}<end_of_turn>\n{% endfor %}"
                 "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}")
LAYER_TYPES = ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "sliding_attention",
               "full_attention", "sliding_attention", "full_attention"]
TINY = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=8, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            global_head_dim=16, hidden_size_per_layer_input=8, num_kv_shared_layers=3, sliding_window=8, max_position_embeddings=8192,
            layer_types=LAYER_TYPES, final_logit_softcapping=30.0, initializer_range=0.5, pad_token_id=0, eos_token_id=1, bos_token_id=2)


def tmpdir(prefix="gsel_"):
    return Path(tempfile.mkdtemp(prefix=prefix))


def make_tiny_checkpoint(out_dir, seed=0):
    import torch
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig, PreTrainedTokenizerFast
    chars = sorted(set(string.printable) - set(" \t\n\r\x0b\x0c")) + [" ", "\n"]
    vocab = {s: i for i, s in enumerate(SPECIALS + chars)}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    tk.add_special_tokens(SPECIALS)
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<bos>", eos_token="<eos>", pad_token="<pad>", unk_token="<unk>")
    tok.chat_template = CHAT_TEMPLATE
    tok.save_pretrained(out_dir)
    torch.manual_seed(seed)
    cfg = Gemma4TextConfig(vocab_size=len(vocab), vocab_size_per_layer_input=len(vocab), **TINY)
    Gemma4ForCausalLM(cfg).eval().save_pretrained(out_dir)
    (Path(out_dir) / "generation_config.json").write_text(json.dumps({"eos_token_id": [1], "bos_token_id": 2}))
    return Path(out_dir)


def site_file(root, role, model_dir, port=1, **extra):
    root = Path(root)
    (root / "src").mkdir(parents=True, exist_ok=True)
    cred = root / f"cred_{role}"
    cred.write_text("x" * 64)
    cred.chmod(0o600)
    site = {"role": role, "expected_hostname": "test-host", "source_root": str(root / "src"), "model_dir": str(model_dir), "device": "cpu",
            "gpu_uuid": None, "port": port, "credential_file": str(cred), "client_id": "t", "run_dir": str(root / f"run_{role}"),
            "adapter_path": str(GS / "tests" / "double_adapter.py"), "ports_in_use_by_others": [9, 10]}
    site.update(extra)
    p = root / f"site_{role}_{len(list(root.glob('site_*')))}.json"
    p.write_text(json.dumps(site))
    return p


def bench(*args, expect=0):
    r = subprocess.run([sys.executable, "-I", "-B", str(GS / "gsel_bench.py"), *map(str, args)], capture_output=True, text=True,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    if expect is not None and r.returncode != expect:
        raise AssertionError(f"gsel_bench {' '.join(map(str, args))}: exit {r.returncode}, expected {expect}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return r
