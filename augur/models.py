"""Model configs without weights.

vLLM builds its config from a directory holding ``config.json``. We take that
file from the LLMServingSim fork's ``configs/model/<org>/<name>.json`` (the
same HF config the fork simulates), so no download or HF token is needed.
"""

from __future__ import annotations

import json
import os
import shutil


def model_dir(fork_root: str, model: str, cache_root: str) -> str:
    """Return a directory with ``config.json`` for ``model`` (e.g. 'meta-llama/Llama-3.1-8B')."""
    src = os.path.join(fork_root, "configs", "model", f"{model}.json")
    if not os.path.exists(src):
        raise FileNotFoundError(f"No model config for {model!r} in the fork: {src}")
    dst_dir = os.path.join(cache_root, model)
    dst = os.path.join(dst_dir, "config.json")
    os.makedirs(dst_dir, exist_ok=True)
    if not os.path.exists(dst) or open(dst).read() != open(src).read():
        shutil.copyfile(src, dst)
    return dst_dir


def estimate_weight_bytes(config_json: str, dtype_bytes: int = 2, tp: int = 1) -> int:
    """Parameter bytes per GPU for decoder-only dense and MoE models (estimate)."""
    c = json.load(open(config_json))
    c = c.get("text_config", c)
    h = c["hidden_size"]
    L = c["num_hidden_layers"]
    v = c["vocab_size"]
    nh = c["num_attention_heads"]
    nkv = c.get("num_key_value_heads", nh)
    hd = c.get("head_dim") or h // nh
    attn = h * nh * hd + 2 * h * nkv * hd + nh * hd * h
    n_exp = c.get("num_experts") or c.get("n_routed_experts") or 0
    if n_exp:
        inter = c.get("moe_intermediate_size") or c["intermediate_size"]
        mlp = n_exp * 3 * h * inter + h * n_exp  # experts + gate
        shared = c.get("n_shared_experts") or 0
        if shared:
            mlp += shared * 3 * h * inter
    else:
        mlp = 3 * h * c["intermediate_size"]
    embed = v * h * (1 if c.get("tie_word_embeddings") else 2)
    return (L * (attn + mlp) + embed) * dtype_bytes // tp
