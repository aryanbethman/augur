"""Step cost from a roofline: the model's shape and the GPU's peak FLOP/s and memory bandwidth, no profile.

For GPUs and models nobody has profiled.  A step is priced like ``TableCost`` prices it (same ops, same layer
structure, same CPU and TP collective terms) but each op's time comes from its FLOPs and bytes instead of a table.
With ``T`` = ``shape.forward_tokens``, ``R`` = number of sequences (0 for a dummy batch) and ``L`` =
``num_hidden_layers``::

    step_ns = embedding + L * (2 * layernorm + qkv_proj (+ qk_norm) + rotary_emb + attention + o_proj
                               + gate_up_proj + act_fn + down_proj)     (a MoE layer: moe_gate_routing + moe_experts)
            + final_layernorm + lm_head + sampler                            (R > 0)
            + the CPU terms of TableCost (token ids in, sampled ids out; at TP>1 its closed-form ring collectives)

Every op is ``max(FLOPs / (peak * eff_compute), bytes / (bandwidth * eff_memory)) + overhead``; sizes are per GPU
(TP shards the heads, the MLP's intermediate dimension and the vocabulary; the K/V heads are replicated once
TP exceeds their number):

* GEMMs (qkv, o, gate_up, down, lm_head, the MoE router): ``2*m*k*n`` FLOPs; weights ``k*n`` plus activations
  ``m*(k+n)`` in bf16.  ``m`` is T, or R for lm_head.
* MoE experts: ``2*T*top_k*3*h*I_moe/TP`` FLOPs at the GEMM rate; weights of the experts the batch touches,
  ``E * (1 - (1 - top_k/E)**T)`` of them in expectation under uniform routing.
* attention, one kernel for the step's prefill requests and one for its decodes (the profiles' mixed steps cost the
  sum of the two), each summed over its requests (``q`` new tokens after ``c`` cached ones): ``4*heads*head_dim*
  sum(q*(c + (q+1)/2))`` FLOPs (causal); bytes: the K/V of every request (``c + q`` tokens), q, the output and the
  K/V written.
* layernorm, qk_norm, rotary, act_fn, embedding: memory traffic only (``elem_bw_scale`` x bandwidth, which is above 1
  where the profilers' repeated runs kept the tensors in L2); the sampler: one pass over the bf16 logits.

``DEFAULT_PARAMS`` holds the parameters fitted from the profiles shipped in ``augur/profiles`` (how: ``python -m
augur.cost.roofline_fit``).  Attention at small batch, kernel launch gaps, tile and wave quantisation and
cache effects are not modelled; ``roofline_fit`` reports how far that leaves the model from the profiles.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from typing import NamedTuple

import numpy as np

from augur.cost import StepShape
from augur.cost.tables import GPU_FACTS, RingTP

BYTES = 2  # bf16 weights, activations and KV cache


@dataclass(frozen=True)
class RooflineParams:
    """Efficiencies and overheads shared by all GPUs.  A fraction is of the datasheet peak; overheads are per kernel."""

    gemm_flops_eff: float  # share of peak bf16 FLOP/s a large GEMM reaches
    gemm_mem_eff: float  # share of peak bandwidth that weight streaming (and the sampler's logits) reaches
    gemm_ovh_ns: float  # fixed time of a GEMM kernel
    elem_bw_scale: float  # effective bandwidth of elementwise kernels over the datasheet's (L2 hits)
    elem_ovh_ns: float  # fixed time of an elementwise kernel
    attn_flops_eff: float  # share of peak FLOP/s the attention kernel reaches
    attn_mem_eff: float  # share of peak bandwidth the decode attention kernel reaches reading the KV cache
    attn_pre_mem_eff: float  # the same for the prefill kernel (few queries per KV head: it streams the history slower)
    attn_ovh_ns: float  # fixed time of the attention kernel
    sampler_ovh_ns: float  # fixed time of the sampler

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "RooflineParams":
        return cls(**{f.name: float(d[f.name]) for f in fields(cls)})


# Fitted by ``python -m augur.cost.roofline_fit fit`` (2026-11-24) on the eight profiles shipped in augur/profiles:
# least squares on the log of predicted / measured over a grid of step shapes, soft-L1 loss; the module docstring of
# roofline_fit lists the measurements left out.  Rounded to three digits.  sampler_ovh_ns sits at its upper bound
# (30 us): the profiles' sampler times are irregular in the number of requests.
DEFAULT_PARAMS = RooflineParams(
    gemm_flops_eff=0.696, gemm_mem_eff=0.876, gemm_ovh_ns=5600.0, elem_bw_scale=2.02, elem_ovh_ns=2360.0,
    attn_flops_eff=0.535, attn_mem_eff=0.770, attn_pre_mem_eff=0.843, attn_ovh_ns=16300.0, sampler_ovh_ns=30000.0)


class Arch(NamedTuple):
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    inter: int  # dense MLP intermediate size
    vocab: int
    qk_norm: bool
    experts: int = 0  # routed experts per MoE layer (0: dense model)
    top_k: int = 0
    moe_inter: int = 0
    moe_layers: int = 0

    @classmethod
    def from_config(cls, path) -> "Arch":
        """From a Hugging Face ``config.json`` (dense or MoE decoder, GQA attention, no MLA or shared experts)."""
        with open(path, encoding="utf-8") as f:
            c = json.load(f)
        return cls.from_dict(c.get("text_config", c))

    @classmethod
    def from_dict(cls, c: dict) -> "Arch":
        if c.get("kv_lora_rank") or c.get("n_shared_experts") or c.get("shared_expert_intermediate_size"):
            raise NotImplementedError("--cost roofline prices GQA attention with routed experts only "
                                      "(no MLA, no shared experts)")
        if c.get("decoder_sparse_step", 1) != 1:
            raise NotImplementedError("--cost roofline: decoder_sparse_step other than 1")
        h, layers, heads = c["hidden_size"], c["num_hidden_layers"], c["num_attention_heads"]
        experts = c.get("num_experts") or c.get("n_routed_experts") or c.get("num_local_experts") or 0
        dense = len(c.get("mlp_only_layers") or []) + (c.get("first_k_dense_replace") or 0) if experts else layers
        return cls(hidden=h, layers=layers, heads=heads, kv_heads=c.get("num_key_value_heads") or heads,
                   head_dim=c.get("head_dim") or h // heads, inter=c["intermediate_size"], vocab=c["vocab_size"],
                   qk_norm=c.get("model_type") in ("qwen3", "qwen3_moe"), experts=experts,
                   top_k=c.get("num_experts_per_tok", 0) if experts else 0,
                   moe_inter=(c.get("moe_intermediate_size") or c["intermediate_size"]) if experts else 0,
                   moe_layers=layers - dense if experts else 0)


def attention_features(shape: StepShape) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """((P, K, Q, n) of the prefill requests, the same of the decode requests): causal query-key pairs, K/V tokens
    read, new tokens and requests."""
    acc = {True: [0, 0, 0, 0], False: [0, 0, 0, 0]}
    for s in shape.seqs:
        a = acc[s.is_decode]
        a[0] += s.num_new * s.num_computed_before + s.num_new * (s.num_new + 1) // 2
        a[1] += s.num_computed_before + s.num_new
        a[2] += s.num_new
        a[3] += 1
    return tuple(acc[False]), tuple(acc[True])


def _gemm(m, k, n, flops, bw, prm: RooflineParams):
    return (np.maximum(2.0 * m * k * n / (flops * prm.gemm_flops_eff),
                       (k * n + m * (k + n)) * BYTES / (bw * prm.gemm_mem_eff)) + prm.gemm_ovh_ns)


def op_ns(arch: Arch, flops: float, bw: float, tp: int, prm: RooflineParams, tokens, rows, pre, dec) -> dict:
    """Time (ns) of one occurrence of every op of a step, as arrays or floats over steps: ``flops`` FLOP/ns and ``bw``
    bytes/ns are the GPU's peaks; ``tokens`` T, ``rows`` R and the attention features ``pre``, ``dec`` of
    ``attention_features``.
    Ops with ``rows`` = 0 (lm_head, sampler) are 0 there."""
    h, hd = arch.hidden, arch.head_dim
    heads, kv_heads = arch.heads / tp, -(-arch.kv_heads // tp)
    t, r = np.asarray(tokens, dtype=float), np.asarray(rows, dtype=float)
    elem = lambda nbytes: nbytes * BYTES / (bw * prm.elem_bw_scale) + prm.elem_ovh_ns
    qk_width = (heads + kv_heads) * hd
    out = {"embedding": elem(2 * t * h), "layernorm": elem(4 * t * h), "final_layernorm": elem(4 * t * h),
           "qkv_proj": _gemm(t, h, qk_width + kv_heads * hd, flops, bw, prm),
           "rotary_emb": elem(2 * t * qk_width), "o_proj": _gemm(t, heads * hd, h, flops, bw, prm)}
    if arch.qk_norm:
        out["qk_norm"] = elem(2 * t * qk_width)

    def attention(mem_eff, pairs, kv, new, seqs):  # one kernel launch: the prefill requests, or the decode requests
        nbytes = (kv * 2 * kv_heads * hd + new * (2 * heads + 2 * kv_heads) * hd) * BYTES
        return np.where(np.asarray(seqs) > 0, np.maximum(4.0 * heads * hd * pairs / (flops * prm.attn_flops_eff),
                                                         nbytes / (bw * mem_eff)) + prm.attn_ovh_ns, 0.0)
    out["attention"] = attention(prm.attn_pre_mem_eff, *pre) + attention(prm.attn_mem_eff, *dec)
    out["gate_up_proj"] = _gemm(t, h, 2 * arch.inter // tp, flops, bw, prm)
    out["act_fn"] = elem(3 * t * (arch.inter // tp))
    out["down_proj"] = _gemm(t, arch.inter // tp, h, flops, bw, prm)
    if arch.experts:
        e, k, im = arch.experts, arch.top_k, arch.moe_inter // tp
        touched = e * (1.0 - (1.0 - k / e) ** t)
        # router GEMM + softmax/top-k; then sort, two grouped GEMMs, act and the weighted reduce
        out["moe_gate_routing"] = _gemm(t, h, e, flops, bw, prm) + 2 * prm.elem_ovh_ns
        out["moe_experts"] = (np.maximum(2.0 * t * k * 3 * h * im / (flops * prm.gemm_flops_eff),
                                         (touched * 3 * h * im + t * k * (3 * h + 6 * im)) * BYTES
                                         / (bw * prm.gemm_mem_eff)) + 2 * prm.gemm_ovh_ns + 3 * prm.elem_ovh_ns)
    live = r > 0
    out["lm_head"] = np.where(live, _gemm(r, h, arch.vocab // tp, flops, bw, prm), 0.0)
    out["sampler"] = np.where(live, r * arch.vocab * BYTES / (bw * prm.gemm_mem_eff) + prm.sampler_ovh_ns, 0.0)
    return out


def layer_counts(arch: Arch) -> dict[str, int]:
    """How many times each op runs in a step (the head ops once)."""
    dense, moe = arch.layers - arch.moe_layers, arch.moe_layers
    n = {"embedding": 1, "final_layernorm": 1, "lm_head": 1, "sampler": 1, "layernorm": 2 * arch.layers,
         "qkv_proj": arch.layers, "rotary_emb": arch.layers, "o_proj": arch.layers, "attention": arch.layers,
         "gate_up_proj": dense, "act_fn": dense, "down_proj": dense}
    if arch.qk_norm:
        n["qk_norm"] = arch.layers
    if moe:
        n.update(moe_gate_routing=moe, moe_experts=moe)
    return n


def gpu_peaks(hardware: str, tflops: float | None = None, bw_gbps: float | None = None) -> tuple[float, float]:
    """(peak bf16 FLOP/ns, bandwidth bytes/ns) of ``hardware`` from ``GPU_FACTS``; the arguments override."""
    facts = GPU_FACTS.get(hardware)
    tflops = tflops or (facts.bf16_tflops if facts else None)
    bw_gbps = bw_gbps or (facts and (facts.mem_bw_gbps or facts.datasheet_bw_gbps))
    if not tflops or not bw_gbps:
        known = ", ".join(h for h, f in GPU_FACTS.items() if f.bf16_tflops and (f.mem_bw_gbps or f.datasheet_bw_gbps))
        raise ValueError(f"--cost roofline needs the GPU's peak bf16 TFLOP/s and memory bandwidth; none known for "
                         f"--hardware {hardware} (known: {known}): give --peak-tflops and --npu-mem-bw")
    return tflops * 1e3, float(bw_gbps)


class RooflineCost(RingTP):
    """``step_ns(shape)`` for a dense or MoE model at TP=N from the roofline above.  ``cpu_mem_*`` and the TP link
    parameters mean what they do in ``TableCost``; ``npu_mem_bw`` (GB/s) and ``peak_tflops`` override ``GPU_FACTS``."""

    def __init__(self, config_json, hardware: str, tp: int = 1, params: RooflineParams = DEFAULT_PARAMS,
                 peak_tflops: float | None = None, cpu_mem_bw: float = 256.0, cpu_mem_latency_ns: int = 0,
                 link_bw: float = 16.0, link_latency_ns: int = 20_000, npu_mem_bw: float | None = None):
        self.arch = Arch.from_config(config_json)
        self.tp, self.params, self.hardware = tp, params, hardware
        self._flops, self._bw = gpu_peaks(hardware, peak_tflops, npu_mem_bw)
        self.cpu_mem_bw, self.cpu_mem_latency_ns = cpu_mem_bw, cpu_mem_latency_ns
        self._in_bytes_per_token = self._out_bytes_per_seq = 4  # int32 token ids in, sampled ids out
        self._counts = layer_counts(self.arch)
        if tp > 1:
            a = self.arch
            self._coll = {"all_reduce_count": 2 * a.layers + 1, "all_reduce_bytes_per_token": BYTES * a.hidden,
                          "all_gather_bytes_per_seq": BYTES * a.vocab // tp}
            if int(self._bw) < 1 or int(cpu_mem_bw) < 1 or link_bw <= 0 or link_latency_ns < 0:
                raise ValueError("TP>1 needs npu_mem_bw >= 1, cpu_mem_bw >= 1, link_bw > 0, link_latency_ns >= 0")
            self.link_bw, self.link_latency_ns, self.npu_mem_bw = link_bw, link_latency_ns, self._bw
        self.steps = 0

    def ops_ns(self, shape: StepShape) -> dict[str, float]:
        """Total time of each op over the whole step (its time x how often it runs); the GPU part of ``step_ns``."""
        rows = 0 if shape.dummy else len(shape.seqs)
        per = op_ns(self.arch, self._flops, self._bw, self.tp, self.params, max(shape.forward_tokens, 1), rows,
                    *attention_features(shape))
        return {name: float(per[name]) * n for name, n in self._counts.items()}

    def step_ns(self, shape: StepShape) -> int:
        # No cache: decode KV lengths grow every step, so a step's shape practically never repeats.
        self.steps += 1
        tokens, rows = shape.forward_tokens, 0 if shape.dummy else len(shape.seqs)
        ns = int(sum(self.ops_ns(shape).values()))
        if self.tp > 1:
            ns += self._comm_ns(tokens, rows)
        else:  # token ids in, sampled ids out: both in CPU memory, as TableCost
            ns += self.cpu_mem_latency_ns + int(self._in_bytes_per_token * tokens / self.cpu_mem_bw)
            ns += self.cpu_mem_latency_ns + int(self._out_bytes_per_seq * rows / self.cpu_mem_bw)
        return ns
