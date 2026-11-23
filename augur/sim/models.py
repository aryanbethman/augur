"""Model configs without weights.

vLLM builds its config from a directory holding ``config.json``. We take that
file from ``--model-config``, from the augur profile in use (profiles embed the
model's config), from the LLMServingSim fork's ``configs/model/<org>/<name>.json``,
or from any profile shipped with augur for that model, so no download or HF
token is needed.
"""

from __future__ import annotations

import hashlib
import json
import os


def _embedded_config(profile_path) -> str:
    import gzip
    with gzip.open(profile_path, "rt", encoding="utf-8") as f:
        return json.dumps(json.load(f)["model_config"], indent=2, sort_keys=True) + "\n"


def model_dir(fork_root: str, model: str, cache_root: str, config_path: str | None = None,
              profile_path=None) -> str:
    """Return a directory with ``config.json`` for ``model`` (e.g. 'meta-llama/Llama-3.1-8B').

    The config comes from ``config_path`` if given (any HF ``config.json``),
    else from the augur profile at ``profile_path`` (which carries the model's
    config), else from the LLMServingSim fork's ``configs/model/<model>.json``
    if that exists, else from any profile shipped with augur for ``model``.
    Each distinct config gets its own folder, so concurrent runs of one model
    with different configs never read each other's."""
    fork_cfg = os.path.join(fork_root, "configs", "model", f"{model}.json")
    if config_path is not None:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"No model config for {model!r}: {config_path} does not exist")
        text = open(config_path).read()
    elif profile_path is not None:
        text = _embedded_config(profile_path)
    elif os.path.exists(fork_cfg):
        text = open(fork_cfg).read()
    else:
        from augur.cost.profiles import shipped
        from augur.cost.tables import bundled_profile
        have = sorted({m for _, m, _, _ in shipped()})
        hit = next(((hw, dt, tp) for hw, m, dt, tp in shipped() if m == model), None)
        if hit is None:
            raise FileNotFoundError(
                f"No model config for {model!r}: not in the fork ({fork_cfg}) and augur ships configs only for "
                f"{', '.join(have) or 'no models'}. Pass --model-config <config.json> (any Hugging Face config.json)")
        text = _embedded_config(bundled_profile(hit[0], model, hit[2], hit[1]))
    dst_dir = os.path.join(cache_root, model, hashlib.sha256(text.encode()).hexdigest()[:16])
    dst = os.path.join(dst_dir, "config.json")
    os.makedirs(dst_dir, exist_ok=True)
    if not os.path.exists(dst):
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


# estimate_reserve_bytes: CUDA context and library workspaces (non-torch memory: 0.15-0.17 GiB in 6 of 7 runs of
# Qwen3-1.7B on an RTX 4060 Ti), the CUDA-graph pool (0.12 GiB there, in vLLM's peak-activation figure), and how many
# fp32 [max_num_seqs, vocab] buffers the sampler's warm-up holds at once (fitted: 5 puts this model inside the range of
# its 7 runs, 0.65-0.97 GiB; it is one calibration point, not a fit across models).
_NON_TORCH_BYTES = 150 << 20
_CUDA_GRAPH_BYTES = 120 << 20
_SAMPLER_FP32_COPIES = 5


def estimate_reserve_bytes(config_json: str, max_num_batched_tokens: int, max_num_seqs: int, tp: int = 1) -> int:
    """Memory vLLM keeps besides weights and the KV cache (non-torch + peak activation) when no profile measured it:
    the forward pass of max_num_batched_tokens tokens (MLP up-projection and two hidden-state buffers, bf16) and the
    sampler's logits, plus the constants above."""
    c = json.load(open(config_json))
    c = c.get("text_config", c)
    h = c["hidden_size"]
    inter = (c["moe_intermediate_size"] * c["num_experts_per_tok"]
             if (c.get("num_experts") or c.get("n_routed_experts")) else c["intermediate_size"])
    forward = 2 * max_num_batched_tokens * (2 * inter // tp + 2 * h)
    sampler = _SAMPLER_FP32_COPIES * 4 * max_num_seqs * c["vocab_size"] // tp
    return _NON_TORCH_BYTES + _CUDA_GRAPH_BYTES + forward + sampler
