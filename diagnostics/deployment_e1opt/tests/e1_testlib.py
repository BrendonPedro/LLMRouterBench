"""Shared fixtures for the E1-OPT CPU tests: a tiny random Qwen3 checkpoint, a fake two-machine inventory in
which one machine is THIS host (so the freeze checker's version comparison runs for real), roles, and a freeze
record written to a temporary path. Mocks cover control flow only; no scientific number is faked."""
import json
import socket
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import e1_spec as S  # noqa: E402
import e1_freeze  # noqa: E402
import e1_inventory as INV  # noqa: E402
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

TINY_CFG = dict(vocab_size=211, hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=4,
                num_key_value_heads=2, head_dim=16, max_position_embeddings=32768, tie_word_embeddings=False,
                rope_theta=10000.0, rms_norm_eps=1e-6)


def tiny_model_dir(dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    tmp = tempfile.mkdtemp(prefix="e1tiny_")
    cfg = Qwen3Config(torch_dtype=dtype, **TINY_CFG)
    m = Qwen3ForCausalLM(cfg).to(dtype).eval()
    m.save_pretrained(tmp, safe_serialization=True)
    return tmp


def check_ids(n=37, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, 211, (n,), generator=g).tolist()


def fake_gpu(index=0, name="FAKE-GPU-80GB", uuid="GPU-fake-0000", memory_mib=81920, cc=9.0):
    return {"index": index, "name": name, "uuid": uuid, "memory_mib": memory_mib, "driver": "999.99", "compute_cap": cc}


def fake_inventory(hostname, gpus, vllm=True, flash=True, route="10.0.0.9 dev eth0 src 10.0.0.1", versions=None, arch="x86_64"):
    return {"hostname": hostname, "utc": "2026-11-11T00:00:00Z", "uname": "fake", "arch": arch, "os": "fake", "cpu": "fake", "ram_gib": 256,
            "nvidia_smi_raw": "", "gpus": gpus, "gpu_count": len(gpus), "cuda_toolkit": "fake", "python": "3.12",
            "torch": {"version": "fake", "cuda": "12.4", "cuda_available": True, "device_count": len(gpus), "devices": []},
            "vllm": {"importable": vllm, "version": "0.8.5.post1"} if vllm else {"importable": False, "error": "no wheel"},
            "flash_attn": {"importable": flash, "version": "2.7.4", "has_flash_attn_with_kvcache": True} if flash else {"importable": False, "error": "no wheel for sm_121"},
            "pynvml": {"importable": True, "version": "12"}, "nics": [{"name": "eth0", "state": "UP", "speed_mbps": 10000}],
            "route_to_peer": route, "peer": "10.0.0.9", "tc_present": True, "tc_version": "tc utility", "sudo_noninteractive": False,
            "capsh": "", "cap_net_admin": "check manually", "unified_memory": "unknown",
            "versions": versions if versions is not None else e1_freeze.versions()}


def local_pair(local_is="edge"):
    """Inventory of THIS host (gpu uuid 'cpu' so CPU drivers pass the identity check) plus a fake peer."""
    me = socket.gethostname()
    peer = "peer-fake-host"
    local_gpu = fake_gpu(0, "cpu", "cpu", 131072, 9.0)
    peer_gpu = fake_gpu(0, "FAKE-PEER-GPU", "GPU-peer-0000", 81920 if local_is == "edge" else 24576, 9.0)
    # this host has no flash_attn wheel (CPU box): its frozen branch is the SDPA fallback, as the §4.2 rule derives
    inv = {me: fake_inventory(me, [local_gpu], flash=False), peer: fake_inventory(peer, [peer_gpu])}
    if local_is == "edge":
        roles = INV.assign_roles(inv[peer], inv[me], lab_server=peer, lab_edge=me)
    else:
        roles = INV.assign_roles(inv[me], inv[peer], lab_server=me, lab_edge=peer)
    assert roles["feasible"], roles
    return inv, roles


def write_freeze(tmpdir, local_is="edge", tag=None):
    inv, roles = local_pair(local_is)
    rec = e1_freeze.build(inv, roles, notes="test fixture")
    if tag:
        rec["git_tag"] = tag; rec["freeze_sha256"] = e1_freeze.freeze_sha256(rec)
    path = Path(tmpdir) / "runtime_freeze.json"
    e1_freeze.write(rec, path)
    (Path(tmpdir) / "roles.json").write_text(json.dumps(roles, indent=1))
    (Path(tmpdir) / "inventory.json").write_text(json.dumps(inv, indent=1, default=str))
    return path, rec, roles, inv


def tiny_workload(path, vocab=211):
    """E0 workload schema with short prompts whose ids fit the tiny vocabulary (dev/test only)."""
    import hashlib
    groups = {"short": (32, 2), "long": (64, 2), "depth": (96, 1)}
    prompts = []
    for g, (n_in, count) in groups.items():
        for i in range(count):
            gen = torch.Generator().manual_seed(100 + len(prompts))
            prompts.append({"id": f"{g}-{i}", "group": g, "n_in": n_in, "ids": torch.randint(1, vocab, (n_in,), generator=gen).tolist()})
    w = {"created_utc": "test", "tokenizer": "tiny", "gen_tokens": S.E0.GEN_TOKENS, "groups": {g: list(v) for g, v in groups.items()}, "prompts": prompts,
         "ids_sha256": hashlib.sha256(json.dumps([p["ids"] for p in prompts]).encode()).hexdigest()}
    Path(path).write_text(json.dumps(w))
    return str(path)
