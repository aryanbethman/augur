"""Model configs without weights.

vLLM builds its config from a directory holding ``config.json``. We take that
file from the LLMServingSim fork's ``configs/model/<org>/<name>.json`` (the
same HF config the fork simulates), so no download or HF token is needed.
"""

from __future__ import annotations

import json
import os


def model_dir(fork_root: str, model: str, cache_root: str, config_path: str | None = None,
              profile_path=None) -> str:
    """Return a directory with ``config.json`` for ``model`` (e.g. 'meta-llama/Llama-3.1-8B').

    The config comes from ``config_path`` if given (any HF ``config.json``),
    else from the augur profile at ``profile_path`` (which carries the model's
    config), else from the LLMServingSim fork's ``configs/model/<model>.json``."""
    dst_dir = os.path.join(cache_root, model)
    dst = os.path.join(dst_dir, "config.json")
    if config_path is None and profile_path is not None:
        import gzip
        with gzip.open(profile_path, "rt", encoding="utf-8") as f:
            text = json.dumps(json.load(f)["model_config"], indent=2, sort_keys=True) + "\n"
    else:
        src = config_path or os.path.join(fork_root, "configs", "model", f"{model}.json")
        if not os.path.exists(src):
            raise FileNotFoundError(f"No model config for {model!r}: {src} (pass --model-config <config.json>)")
        text = open(src).read()
    os.makedirs(dst_dir, exist_ok=True)
    if not os.path.exists(dst) or open(dst).read() != text:
        tmp = f"{dst}.{os.getpid()}"
        with open(tmp, "w") as f:
            f.write(text)
        os.replace(tmp, dst)  # atomic: other augur runs may be reading it
    return dst_dir


def estimate_weight_bytes(config_json: str, dtype_bytes: int = 2, tp: int = 1, ep: int = 1) -> int:
    """Parameter bytes per GPU for decoder-only dense and MoE models (estimate).

    As in vLLM: attention and dense layers are split by TP; routed experts are
    split by EP when expert parallelism is on (``ep`` = DP x TP), else by TP."""
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
    experts = 0
    if n_exp:
        inter = c.get("moe_intermediate_size") or c["intermediate_size"]
        experts = n_exp * 3 * h * inter
        mlp = h * n_exp  # gate
        shared = c.get("n_shared_experts") or 0
        if shared:
            mlp += shared * 3 * h * inter
    else:
        mlp = 3 * h * c["intermediate_size"]
    embed = v * h * (1 if c.get("tie_word_embeddings") else 2)
    expert_split = ep if ep > 1 else tp
    return ((L * (attn + mlp) + embed) // tp + L * experts // expert_split) * dtype_bytes
