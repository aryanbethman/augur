"""EAGLE / EAGLE-3 / MTP drafters without weights: the draft-model config, its KV, and its step cost.

vLLM's own speculative-decoding path (lookahead slots, the scheduler's draft handling, rejection) runs unchanged;
what augur supplies is what a GPU would:

* ``draft_model_dir``: a ``config.json`` for the draft model, derived from the target's the way EAGLE heads are
  (one decoder layer of the target's width and head layout; vLLM wraps it in its EAGLEConfig and checks it as it
  would a real checkpoint).  MTP needs no draft directory: vLLM reads the MTP layers from the target's own config
  (models that define them, e.g. DeepSeek-V3).
* ``draft_kv_layers``: the draft model's attention layers, which take KV cache next to the target's.
* ``DraftCost``: the drafter's GPU time inside a step.  After the verification forward, the drafter runs k
  passes: the first over the same tokens as the verification step (it reads the target's hidden states at every
  position), the other k - 1 over one token per drafting request.  Each pass is priced from the target's own
  per-layer tables for ``layers`` decoder layers plus the final norm, ``lm_head`` and sampler, and, as the
  drafter's input projection (EAGLE ``fc``, MTP ``eh_proj``), 2 (EAGLE, MTP) or 3 (EAGLE-3) times the target's
  ``o_proj``.  That is an approximation: no drafter was profiled, so its layer is priced as a target layer, the
  projection as a multiple of ``o_proj``, and the drafter's own CUDA-graph padding is not modeled.
"""

from __future__ import annotations

import hashlib
import json
import os

from augur.cost import SeqWork, StepShape

METHODS = ("ngram", "eagle", "eagle3", "mtp")
PROJECTION_O_PROJ = {"eagle": 2, "eagle3": 3, "mtp": 2}  # drafter input projection, in target o_proj prices

# HF config fields that hold the number of MTP layers a target defines.
_MTP_FIELDS = ("num_nextn_predict_layers", "mtp_num_hidden_layers", "num_mtp_layers", "n_predict")


def mtp_layers(config: dict) -> int:
    """MTP layers a target config defines (0 when it defines none)."""
    c = config.get("text_config", config)
    return max((int(c.get(f) or 0) for f in _MTP_FIELDS), default=0)


def draft_config(target: dict, method: str) -> dict:
    """The draft model's config.json: the target's with one decoder layer (EAGLE heads have one)."""
    if method not in ("eagle", "eagle3"):
        raise ValueError(f"no draft config for method {method!r}")
    cfg = json.loads(json.dumps(target))
    text = cfg.get("text_config", cfg)
    if text.get("layer_types") or text.get("num_experts") or text.get("n_routed_experts"):
        raise ValueError("EAGLE heads are derived from dense, uniform-layer targets only "
                         "(this config has per-layer types or experts)")
    text["num_hidden_layers"] = 1
    return cfg


def draft_model_dir(target_dir: str, method: str, cache_root: str) -> str:
    """A directory with the draft model's config.json (created once per distinct config)."""
    with open(os.path.join(target_dir, "config.json"), encoding="utf-8") as f:
        target = json.load(f)
    text = json.dumps(draft_config(target, method), indent=2, sort_keys=True) + "\n"
    dst_dir = os.path.join(cache_root, "draft", method, hashlib.sha256(text.encode()).hexdigest()[:16])
    dst = os.path.join(dst_dir, "config.json")
    os.makedirs(dst_dir, exist_ok=True)
    if not os.path.exists(dst):
        tmp = f"{dst}.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, dst)  # atomic: other augur runs may be reading it
    return dst_dir


def speculative_config(method: str, k: int, target_dir: str, cache_root: str) -> dict:
    """vLLM's ``speculative_config`` for the method (the rest of vLLM's checks run on it as on a real model)."""
    if k < 1:
        raise ValueError("--num-speculative-tokens must be at least 1")
    if method == "ngram":
        return {"method": "ngram", "num_speculative_tokens": k, "prompt_lookup_max": 4, "prompt_lookup_min": 2}
    if method == "mtp":
        with open(os.path.join(target_dir, "config.json"), encoding="utf-8") as f:
            if not mtp_layers(json.load(f)):
                raise ValueError("--speculative-method mtp: the model's config defines no MTP layers "
                                 f"(none of {', '.join(_MTP_FIELDS)}); use eagle, which derives a head from any "
                                 "dense target")
        return {"method": "mtp", "num_speculative_tokens": k}
    _check_head_architecture(target_dir, method)
    return {"method": method, "model": draft_model_dir(target_dir, method, cache_root), "num_speculative_tokens": k}


def _check_head_architecture(target_dir: str, method: str) -> None:
    """vLLM has an EAGLE head class for some architectures only (EagleLlamaForCausalLM, Eagle3Qwen3ForCausalLM ...):
    say so in a line instead of vLLM's validation error listing every architecture it knows."""
    from vllm import ModelRegistry
    with open(os.path.join(target_dir, "config.json"), encoding="utf-8") as f:
        archs = json.load(f).get("architectures") or []
    prefix = "Eagle3" if method == "eagle3" else "Eagle"
    known = set(ModelRegistry.get_supported_archs())
    if not any(f"{prefix}{a}" in known for a in archs):
        have = sorted(a[len(prefix):] for a in known if a.startswith(prefix) and a != prefix
                      and not (method == "eagle" and a.startswith("Eagle3")))
        raise ValueError(f"vLLM 0.33 has no {method} head for {archs}; it has one for: {', '.join(have)}")


def draft_weight_bytes(target_config_path: str, method: str, tp: int = 1, dtype_bytes: int = 2) -> int:
    """Extra weights per GPU for the drafter (estimate): its decoder layers and input projection.  EAGLE heads
    share the target's embedding and lm_head; MTP modules are in the target's checkpoint."""
    with open(target_config_path, encoding="utf-8") as f:
        c = json.load(f)
    c = c.get("text_config", c)
    h, nh = c["hidden_size"], c["num_attention_heads"]
    nkv, hd = c.get("num_key_value_heads", nh), c.get("head_dim") or h // nh
    layers = 1 if method in ("eagle", "eagle3") else mtp_layers(c)
    attn = h * nh * hd + 2 * h * nkv * hd + nh * hd * h
    mlp = 3 * h * c["intermediate_size"] if not (c.get("num_experts") or c.get("n_routed_experts")) else 0
    proj = PROJECTION_O_PROJ[method] * h * h
    return (layers * (attn + mlp + proj) // tp) * dtype_bytes


class DraftCost:
    """The drafter's GPU time inside one engine step; ``step_ns(shape, rows)`` is added to the step's price.

    ``rows`` is how many requests get a draft after this step.  With a cost model that prices layers
    (``draft_pass_ns``, i.e. TableCost) each pass is priced from its tables; with another model a pass costs
    the same share of the step's price as the drafter's layers are of the target's."""

    def __init__(self, cost, method: str, k: int, layers: int, target_layers: int):
        self.cost, self.method, self.k, self.layers, self.target_layers = cost, method, k, layers, target_layers
        self.proj = PROJECTION_O_PROJ[method]
        self._native = hasattr(cost, "draft_pass_ns")
        self._cache: dict[tuple, int] = {}

    def step_ns(self, shape: StepShape, rows: int) -> int:
        if rows <= 0 or not shape.seqs:
            return 0
        kvs = [s.num_computed_before + s.num_new for s in shape.seqs][:rows]
        key = (tuple((s.num_new, s.num_computed_before, s.is_decode) for s in shape.seqs), shape.padded_tokens,
               shape.dummy, rows)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        total = self._pass(shape, rows)
        for j in range(1, self.k):  # passes 2..k: one new token per drafting request, one position further each
            one = StepShape(seqs=tuple(SeqWork(1, kv + j, True) for kv in kvs))
            total += self._pass(one, len(kvs))
        self._cache[key] = total
        return total

    def _pass(self, shape: StepShape, rows: int) -> int:
        if self._native:
            return self.cost.draft_pass_ns(shape, self.layers, rows, self.proj)
        return round(self.cost.step_ns(shape) * self.layers / max(self.target_layers, 1))


def draft_layer_count(vllm_config) -> int:
    """Attention layers of the draft model in vLLM's resolved config (0: no model-based drafter)."""
    sc = vllm_config.speculative_config
    if sc is None or sc.method not in ("eagle", "eagle3", "mtp") or sc.draft_model_config is None:
        return 0
    return int(sc.draft_model_config.get_num_layers(vllm_config.parallel_config))


def draft_kv_spec(vllm_config, target_specs):
    """(KV spec, layer count) of the draft model's attention layers, or None: they take cache next to the target's.

    The drafter's layers are built like the target's attention layer: MLA targets (DeepSeek MTP) give MLA layers, a
    dense full-attention target gives full-attention layers of the draft model's own head layout. Other kinds
    (sliding window, hybrid) raise: no drafter for them is modelled."""
    n = draft_layer_count(vllm_config)
    if not n:
        return None
    from vllm.v1.kv_cache_interface import AttentionSpec, FullAttentionSpec, MLAAttentionSpec
    target = next((spec for spec in target_specs.values() if isinstance(spec, AttentionSpec)), None)
    if isinstance(target, MLAAttentionSpec):
        return target, n
    if type(target) is not FullAttentionSpec:
        kind = "no attention layer" if target is None else type(target).__name__
        raise ValueError(f"{vllm_config.speculative_config.method} drafting is modelled for full-attention and MLA "
                         f"targets only; this model's first attention layer is {kind}")
    dm, pc = vllm_config.speculative_config.draft_model_config, vllm_config.parallel_config
    return FullAttentionSpec(block_size=target.block_size, num_kv_heads=dm.get_num_kv_heads(pc),
                             head_size=dm.get_head_size(), dtype=target.dtype), n
