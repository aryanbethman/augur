"""Step cost from a compact profile file: no LLMServingSim code at run time.

For a dense model at TP=1 on one engine, the price of a step is a sum of table
lookups.  With ``T`` = ``shape.forward_tokens`` (real tokens, or the CUDA-graph
padded count when larger), ``R`` = number of sequences (0 for a dummy batch)
and ``L`` = ``num_hidden_layers``::

    step_ns = embedding[T]
            + L * (layernorm[T] + qkv_proj[T] (+ qk_norm[T]) + rotary_emb[T]
                   + attention(prefill_chunk, prefill_key, n_decode,
                               kv_mean, kv_max, decode_q_len)
                   + o_proj[T] + layernorm[T]
                   + gate_up_proj[T] + act_fn[T] + down_proj[T])
            + final_layernorm[T] + lm_head[R] + sampler[R]          (R > 0)
            + (cpu_mem_latency_ns + int(4 * T / cpu_mem_bw))   # token ids, CPU memory
            + (cpu_mem_latency_ns + int(4 * R / cpu_mem_bw))   # sampled ids, CPU memory

The layer order, ``L`` and the table values are read from the profile file.
``attention`` is skipped when the step has no prefill tokens and no decodes.
``lm_head`` and ``sampler`` are dropped when ``R == 0`` (an idle dummy forward).

* ``embedding``...``down_proj``: one-dimensional tables in integer ns (by tokens,
  or by sequences for ``lm_head``/``sampler``).  Below the first key the value is
  clamped, between keys it is linear, above the last key it is extrapolated
  linearly from the top two keys; the result is truncated to an int, at least 1.
* ``attention``: four-dimensional, over (prefill chunk tokens, prefill key,
  number of decodes, decode KV length).  The two table axes bracketing the
  chunk and the decode count give four corners, each a bilinear interpolation
  over (prefill key, KV length); a missing corner uses the nearest profiled
  (chunk, decodes) pair.  ``prefill_key`` is the query-weighted mean of
  ``history + chunk / 2``; the decode KV length is the mean over decodes.
* skew: with more than one decode of unequal KV length, the attention time is
  ``t_mean + alpha * (t_max - t_mean)`` where ``t_max`` is the lookup at the
  longest KV and ``alpha`` comes from a calibrated bucket table (decode count,
  prefill chunk and relative gap ``(t_max - t_mean) / t_mean``).

* shared KV (optional ``shared_kv`` entry of the profile): the attention table is measured with every decode on
  blocks of its own, so each KV byte comes from GPU memory.  Under prefix caching, requests of a batch share
  blocks (``SeqWork.shared_kv``); a block several of them read stays in the L2 cache and the kernel runs faster than
  the table says.  With ``u`` = distinct KV tokens / total KV tokens of the decodes and ``c(n)`` the profile's floor
  at ``n`` decodes (linear in n, clamped), the decode part of the attention time (the lookup less the lookup without
  decodes) is scaled by ``min(1, (u**p + c(n)**p) ** (1/p))``.  The profile's provenance says how ``p`` and ``c``
  were fitted.  Without the entry, or when nothing is shared, the lookup above is the price.

Mixture-of-experts models (``kind: "moe"``).  Every layer outside the MoE block is priced as above.  The MoE
block of a decoder layer costs a function of the forward token count ``T`` only (the fork's balanced gate
is deterministic in ``T``), so the profile stores it as a table of ``T`` and ``L`` copies are summed:

* TP=1, native components (``moe.exact``: ``moe_gate_routing`` and ``moe_experts``): one integer ns per
  ``T`` = 1..N, N the largest token count the fork's component tables measured (2048).  The fork builds
  these from a 3-D (tokens, activated experts, local assignments) surface; the converter evaluated its
  own code at every ``T``, so no triangulation (scipy) is needed at run time.  ``T`` = 0 and ``T`` > N
  are refused, as the fork refuses them.
* TP>1 without native components (``moe.balanced``: ``expert``): the fork's legacy whole-block table on
  the activated-expert count of its closed-form balanced gate::

      active(T) = min(E, max(1, round(E * (1 - ((E - k) / E) ** T))))
      expert(T) = interpolation over (active, T) of the ep=1 rows, as in ``_lookup_moe``

  The block is absent (0 ns) when ``T`` = 0.  Unlike the other tables it is not scaled by TP.
  This path has no collective of its own, but it changes what ASTRA-Sim charges for the others: the
  per-step communication above is lower by a few ns per MoE layer, M = the number of MoE layers
  (``collectives.moe_layers``), found by fitting ASTRA-Sim runs and checked on random shapes::

      TP2:  max(0, min(M, A, max(M, A) - B))
      TP4:  min(M, max(H + A - B, min(M - B, H + A, max(M - 2B, 2A + H - 2B))))      H = M - M // 2, A >= B

  (A and B as in the TP formula below.)  The TP4 form was fitted on a grid of A and B (0-100 ns) with M = 20,
  40, 41 and 48 (MoE blocks removed from real traces), then checked on random shapes and cluster settings.
  MoE at other TP sizes is refused.

Data parallelism with expert parallelism (``round_ns``; DP2 + EP2 of TP1 ranks, ``moe.dp``, the only DP layout the
fork has native MoE tables for).  Each rank runs its own trace: the embedding, then per decoder layer the layers
before the MoE block, ``moe_gate_routing`` (priced at the rank's token count T_r), three all-gathers (hidden states,
top-k weights, top-k ids; bytes = per-token sizes x the round's largest T), ``moe_experts`` (at the round's summed
count, per rank), a reduce-scatter (2 x largest T x hidden bytes), ``moe_finalize_copy`` (at T_r); then the head.
ASTRA-Sim runs both ranks together, which the closed form reproduces (fitted on probes, exact on 360 random rounds
over six cluster settings):

* start: both ranks read their token ids from the node's one CPU memory, the round's first member first (a FIFO:
  the second finishes at A_first + A_second);
* a collective of ring hop h = int(lat + msg / bw) and duration D (all-gather: h + 20; reduce-scatter of S bytes:
  msg = S / 2, h + 3 int(msg / HBM) + 20) entered at t_late >= t_early: the early rank leaves at t_late + D, the late
  one at t_late - min(t_late - t_early, h) + D (it already holds the early rank's data);
* end: the sampled ids (a dummy rank writes 0 bytes) are written to the CPU memory in finishing order.

Tensor parallelism (TP = N > 1, one node, dense and MoE models; ASTRA-Sim FullyConnected topology, ring
collectives).  Compute is priced exactly as above, on the profile's tpN tables.  The fork prices the
rest by running ASTRA-Sim; for a dense step it has this closed form (``_comm_ns``, with the ASTRA-Sim
source of each term in the comments there)::

    comm_ns = N * max(A, B) + min(A, B)  +  (2L+1) * AR(2*hidden*T)  +  AG(2*V*R)      (AG only if R > 0)
    A = cpu_lat + floor(4*T / int(cpu_bw)),   B = cpu_lat + floor(4*R / int(cpu_bw))

Guard: the formula is not exact when a CPU-memory access is long next to a ring step (the N NPUs queue
on one CPU-memory device and their staggered starts stop hiding behind the ring).  Measured against the
ASTRA-Sim binary (N = 2, 3, 4, 8): inexact once 2A > S_ar (N >= 3; A > S_ar at N = 2) or 2B > S_last, where
S_ar = floor(link_lat + ((2*hidden*T) // N) / bw) is one all-reduce ring step and S_last is one step of
the last collective (all-gather if R > 0, else all-reduce).  ``step_ns`` therefore prices a step only
if 2A <= S_ar and 2B <= S_last, and raises ``OutsideRegime`` otherwise (use --cost llmservingsim).

Every operation, including float evaluation order, rounding (Python ``round``,
half to even) and truncation, follows the fork's ``serving/core/trace_generator.py``
so that results are identical to the nanosecond.

Method and data: LLMServingSim (MIT licence, Copyright (c) 2024, CASYS Lab at
KAIST).  The profile files are produced from its profiler outputs by
``python -m augur.cost.convert``; only the standard library is needed here.
"""

from __future__ import annotations

import bisect
import gzip
import json
import logging
from pathlib import Path
from typing import NamedTuple

from augur.cost import StepShape

FORMAT_VERSION = 1
PROFILE_DIR = Path(__file__).resolve().parent.parent / "profiles"
log = logging.getLogger(__name__)


def bundled_profile(hardware: str, model: str, tp: int = 1, variant: str = "bf16") -> Path:
    """Path of the profile shipped in ``augur/profiles`` for (hardware, model, tp)."""
    path = PROFILE_DIR / hardware / model.replace("/", "--") / f"{variant}-tp{tp}.json.gz"
    if not path.is_file():
        have = sorted(str(p.relative_to(PROFILE_DIR)) for p in PROFILE_DIR.glob("*/*/*.json.gz"))
        raise FileNotFoundError(f"no bundled profile {path.relative_to(PROFILE_DIR)}; have: {have or 'none'}")
    return path


# --- lookups, copied operation for operation from the fork -----------------

def _interp1d(keys, values, query):
    if len(keys) == 1:
        return values[0]
    idx = bisect.bisect_right(keys, query)
    if idx == 0:
        return values[0]
    lo, hi = (len(keys) - 2, len(keys) - 1) if idx >= len(keys) else (idx - 1, idx)
    x0, y0, x1, y1 = keys[lo], values[lo], keys[hi], values[hi]
    if x1 == x0:
        return y0
    return y0 + (query - x0) / (x1 - x0) * (y1 - y0)


def _axis_bracket(values, query):
    """(lo, hi, t): clamp below the first value, extrapolate above the last."""
    n = len(values)
    if n == 1 or query <= values[0]:
        return 0, 0, 0.0
    idx = bisect.bisect_right(values, query)
    lo, hi = (n - 2, n - 1) if idx >= n else (idx - 1, idx)
    x0, x1 = values[lo], values[hi]
    if x1 == x0:
        return lo, hi, 0.0
    return lo, hi, (query - x0) / (x1 - x0)


def _prefill_key(chunks):
    """Query-weighted causal key length of (chunk, history) pairs."""
    total = 0
    work = 0.0
    for chunk, history in chunks:
        if chunk < 0 or history < 0:
            raise ValueError("prefill chunks and histories must be non-negative")
        total += chunk
        work += chunk * (history + chunk / 2.0)
    return work / total if total else 0.0


def _label(bins, labels, value):
    return next((lab for upper, lab in zip(bins[1:], labels) if value <= upper), labels[-1])


def _anchor_index(anchors, n):
    lo, hi = 0, len(anchors) - 1
    square = n * n
    while lo < hi:
        mid = (lo + hi) // 2
        if square <= anchors[mid] * anchors[mid + 1]:
            hi = mid
        else:
            lo = mid + 1
    return lo


# Price caches are cleared when full: prices are pure, and step shapes grow with the cluster (about 400 B a key;
# 375k keys at 256 GPUs), so an unbounded cache costs memory at scale.
CACHE_MAX = 1 << 17


class _Attention:
    """One kernel at one decode query length: slices keyed by (chunk, decodes)."""

    def __init__(self, spec):
        axes = spec["kd_axes"]
        self.slices = {}
        for s in spec["slices"]:
            kd = axes[s["kd"]]
            self.slices[(s["pc"], s["nd"])] = (s["kp"], [(kd, row) for row in s["ns"]])
        self.pairs = sorted(self.slices)
        self._memo = {}  # lookup() is pure and batch shapes repeat (several layers, steps and engines share one)
        self.pc_vals = sorted({pc for pc, _ in self.pairs})
        self.nd_vals = sorted({nd for _, nd in self.pairs})
        # largest profiled value on each axis; a lookup above it extrapolates
        self.limits = {"prefill chunk tokens": self.pc_vals[-1], "decodes": self.nd_vals[-1],
                       "decode KV length": max(max(a) for a in axes),
                       "prefill key length": max(max(kp) for kp, _ in self.slices.values())}

    def _slice(self, pc, nd, prefill_key, kv):
        found = self.slices.get((pc, nd))
        if found is None:
            return None
        kp_vals, rows = found
        lo_kp, hi_kp, t_kp = _axis_bracket(kp_vals, max(float(prefill_key), 0.0))

        def along_kv(row):
            keys, vals = row
            if len(keys) == 1:
                return vals[0]
            lo, hi, t = _axis_bracket(keys, kv)
            return vals[lo] + t * (vals[hi] - vals[lo])

        v_lo = along_kv(rows[lo_kp])
        if lo_kp == hi_kp:
            return v_lo
        v_hi = along_kv(rows[hi_kp])
        return v_lo + t_kp * (v_hi - v_lo)

    def lookup(self, prefill_chunk, prefill_key, n_decode, kv):
        key = (prefill_chunk, prefill_key, n_decode, kv)
        hit = self._memo.get(key)
        if hit is None:
            if len(self._memo) >= CACHE_MAX:
                self._memo.clear()
            hit = self._memo[key] = self._lookup(prefill_chunk, prefill_key, n_decode, kv)
        return hit

    def _lookup(self, prefill_chunk, prefill_key, n_decode, kv):
        kv = max(int(kv), 0)
        lo_pc, hi_pc, t_pc = _axis_bracket(self.pc_vals, max(int(prefill_chunk), 0))
        lo_nd, hi_nd, t_nd = _axis_bracket(self.nd_vals, max(int(n_decode), 0))

        def corner(pc, nd):
            v = self._slice(pc, nd, prefill_key, kv)
            if v is not None:
                return v
            near = min(self.pairs, key=lambda p: (p[0] - pc) ** 2 + (p[1] - nd) ** 2)
            return self._slice(near[0], near[1], prefill_key, kv) or 0.0

        c00 = corner(self.pc_vals[lo_pc], self.nd_vals[lo_nd])
        c01 = corner(self.pc_vals[lo_pc], self.nd_vals[hi_nd])
        c10 = corner(self.pc_vals[hi_pc], self.nd_vals[lo_nd])
        c11 = corner(self.pc_vals[hi_pc], self.nd_vals[hi_nd])
        v0 = c00 + t_nd * (c01 - c00)
        v1 = c10 + t_nd * (c11 - c10)
        return max(1, int(v0 + t_pc * (v1 - v0)))


class GpuFacts(NamedTuple):
    mem_gib: float  # device memory as torch / vLLM report it (GiB): what --gpu-memory-utilization is a share of
    mem_bw_gbps: float | None = None  # GPU memory bandwidth (GB/s); None: the profiles carry it or it is not needed
    bf16_tflops: float | None = None  # peak dense BF16 tensor-core TFLOP/s (what --cost roofline prices from)
    datasheet_bw_gbps: float | None = None  # datasheet bandwidth for --cost roofline where mem_bw_gbps is left unset


# What augur knows about each --hardware name.  mem_bw_gbps is for hardware whose profiles carry none because the
# fork's hardware_defaults has no entry for it; bf16_tflops (dense, fp32 accumulate) is for --cost roofline.
# Sources (NVIDIA product pages and datasheets read 2026-11-24; NVIDIA prints the sparse figure, dense is half of it):
# H100: SXM5 HBM3, 3.35 TB/s (nvidia.com/en-us/data-center/h100: 80 GB, 3.35 TB/s), BF16 1,979 sparse = 989.5 dense;
#   the value the fork's H100 cluster configs set (an H100 PCIe has 2.0 TB/s: pass npu_mem_bw for it); 79.19 GiB seen by
#   torch (the total in PyTorch OOM messages for H100 80GB; not measured on this host).
# H200: SXM, 141 GB HBM3e, 4.8 TB/s, same compute as H100 (nvidia.com/en-us/data-center/h200).
# B200: per GPU of an 8-GPU HGX/DGX B200 baseboard: BF16 18 PFLOPS dense / 8 (nvidia.com/en-us/data-center/hgx: 36
#   PFLOPS sparse), 64 TB/s / 8 = 8 TB/s HBM3e (nvidia.com/en-us/data-center/dgx-b200), 1,440 GB / 8 = 180 GB.
# A100: 80 GB SXM: 312 dense BF16 TFLOPS, 2,039 GB/s; A100_40GB (SXM): 312, 1,555 GB/s (NVIDIA A100 datasheet,
#   Jun 2021).
# L40S: 362.05 dense BF16 TFLOPS, 48 GB GDDR6, 864 GB/s (nvidia.com/en-us/data-center/l40s).
# A10: 125 dense BF16 TFLOPS, 24 GB GDDR6, 600 GB/s (nvidia.com/en-us/data-center/products/a10-gpu).
# mem_gib of the GPUs above except H100 and the RTX cards is the datasheet capacity x 0.99, the ratio torch reports
# for the H100 (79.19 of 80 GB): an estimate, not a torch reading; give --hbm-gb for the exact figure.
# RTX 4060 Ti: 128-bit GDDR6 at 18 Gb/s = 288 GB/s (also what --hbm-interference was fitted against); 16 GB card,
#   15.58 GiB seen by vLLM 0.33 ("Free memory on device (15.4/15.58 GiB)" in all 11 out/real/*/server.log).  BF16 44.1
#   TFLOP/s dense with fp32 accumulate: 136 tensor cores x 2.535 GHz boost x 128 FLOP/clock (the GeForce rate; core
#   count and clock from the public spec sheets; NVIDIA lists 353 AI TOPS = FP8 sparse, 353 / 8 = 44.1; flopper.io
#   lists 44).
# RTX PRO 6000 Blackwell Server Edition: 96 GB GDDR7 and 1,597 GB/s (NVIDIA datasheet, doc 3850600; 1,792 GB/s is the
#   Workstation Edition), 94.97 GiB seen by torch (the total in public PyTorch and vLLM logs for this card; not measured
#   on this host).  BF16 500 TFLOP/s dense = 1,000 sparse / 2 (the sparse figure is from flopper.io; the datasheet
#   gives only FP4).  Its mem_bw_gbps stays unset (the default pricer's behaviour is unchanged); roofline reads the
#   datasheet value from datasheet_bw_gbps.
GPU_FACTS = {
    "H100": GpuFacts(79.19, 3350.0, 989.5), "H200": GpuFacts(139.59, 4800.0, 989.5),
    "B200": GpuFacts(178.2, 8000.0, 2250.0), "A100": GpuFacts(79.2, 2039.0, 312.0),
    "A100_40GB": GpuFacts(39.6, 1555.0, 312.0), "L40S": GpuFacts(47.52, 864.0, 362.05),
    "A10": GpuFacts(23.76, 600.0, 125.0), "RTX4060Ti": GpuFacts(15.58, 288.0, 44.1),
    "RTXPRO6000": GpuFacts(94.97, None, 500.0, 1597.0)}


def gpu_mem_bw_gbps(hardware) -> float | None:
    facts = GPU_FACTS.get(hardware)
    return facts.mem_bw_gbps if facts else None


def measured_reserve_gib(mem: dict) -> float:
    """What vLLM 0.33 keeps from the KV cache besides the weights, from a ``vllm_memory`` record: non-torch memory
    plus peak activation (``reserve_gib`` when the record summarises several runs).  The CUDA-graph figure is not
    added: vLLM's "peak activation" already holds the graph estimate (VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS) and
    KV = gpu_memory_utilization * total - (weights + non-torch) - peak activation."""
    return mem.get("reserve_gib", mem["non_torch_gib"] + mem["peak_activation_gib"])


def profile_vllm_memory(profile_path) -> dict | None:
    """vLLM's memory accounting recorded in a profile's provenance (python -m augur.profile), None if it has none."""
    with gzip.open(profile_path, "rt", encoding="utf-8") as f:
        return json.load(f).get("provenance", {}).get("vllm_memory")


class OutsideRegime(ValueError):
    """A TP>1 step whose communication time the closed form does not reproduce exactly."""


class RingTP:
    """ASTRA-Sim's TP>1 collectives in closed form (the module docstring's ``comm_ns``), shared by ``TableCost`` and
    ``augur.cost.roofline.RooflineCost``.  Needs ``tp``, ``link_bw``, ``link_latency_ns``, ``npu_mem_bw``,
    ``cpu_mem_bw``, ``cpu_mem_latency_ns``, ``_coll`` (all_reduce_count, all_reduce_bytes_per_token,
    all_gather_bytes_per_seq, moe_layers) and ``_in_bytes_per_token`` / ``_out_bytes_per_seq``."""

    def _ring_ns(self, tokens: int, rows: int) -> tuple[int, int, int, int]:
        """ASTRA-Sim's ring at TP>1: (one hop and the whole of an all-reduce of ``tokens`` tokens, one hop and
        the whole of the logits all-gather of ``rows`` sequences; 0 for the whole without rows)."""
        n, lat, c = self.tp, self.link_latency_ns, self._coll
        bw = self.link_bw * (1 << 30) / 1000000000          # NetworkFunction.cpp: bw * (1<<30) / 1e9 bytes/ns
        local = int(self.npu_mem_bw) * 1000000000           # Sys.cc: local-mem-bw (json, int) * 1e9
        msg = (c["all_reduce_bytes_per_token"] * tokens) // n   # Ring.cc: chunk = data_size / N
        hop = int(lat + msg / bw)                           # BasicTopology.cpp: one FullyConnected hop
        # Ring.cc: 2(N-1) steps; reduce half is processed by the memory bus, 3 x size/local_bw (PacketBundle.cc);
        # MemBus + endpoint delay (10 ns, system.json) once per step and once for the first packet
        ar = 2 * (n - 1) * hop + (n - 1) * 3 * int(msg / local * 1e9) + 10 * (2 * (n - 1) + 1)
        ag_hop = int(lat + c["all_gather_bytes_per_seq"] * rows / bw) if rows else hop
        ag = (n - 1) * ag_hop + 10 * n if rows else 0      # N-1 steps (Ring.cc)
        return hop, ar, ag_hop, ag

    def _comm_ns(self, tokens: int, rows: int) -> int:
        n, lat, c = self.tp, self.link_latency_ns, self._coll
        cpu_bw = int(self.cpu_mem_bw)                       # AnalyticalMemory: uint64 mem_bw
        a = self.cpu_mem_latency_ns + int(self._in_bytes_per_token * tokens / cpu_bw)  # token ids in (CPU memory)
        b = self.cpu_mem_latency_ns + int(self._out_bytes_per_seq * rows / cpu_bw)     # sampled ids out
        hop, ar, ag_hop, ag = self._ring_ns(tokens, rows)
        if 2 * a > hop or 2 * b > ag_hop:
            raise OutsideRegime(
                f"TP={n} step (T={tokens}, R={rows}): a CPU-memory access (A={a} ns for the token ids, B={b} ns "
                f"for the sampled ids; cpu latency {self.cpu_mem_latency_ns} ns, bandwidth {self.cpu_mem_bw} GB/s) "
                f"is more than half a ring step (all-reduce {hop} ns, last collective {ag_hop} ns; link latency "
                f"{lat} ns, bandwidth {self.link_bw} GB/s); the closed form is not exact there. "
                f"Use --cost llmservingsim")
        total = n * max(a, b) + min(a, b) + c["all_reduce_count"] * ar   # CPU memory: one FIFO queue per node
        m = c.get("moe_layers", 0) if tokens > 0 else 0  # an empty step has no MoE block
        if m:  # see the MoE note in the module docstring
            total -= self._moe_comm_saving(m, a, b)
        return total + ag  # and the all-gather of the per-rank logits chunk

    def _moe_comm_saving(self, m: int, a: int, b: int) -> int:
        """How much less than the dense form ASTRA-Sim charges a MoE step at TP2 or TP4 (see the module docstring);
        ``m`` MoE layers, ``a`` and ``b`` the CPU-memory times of the token ids and the sampled ids."""
        if self.tp == 2:
            return max(0, min(m, a, max(m, a) - b))
        if b > a:  # never with 4-byte token ids and sampled ids (R <= T); the form was fitted for a >= b only
            raise OutsideRegime(f"MoE TP4 step with sampled-id time {b} ns above the token-id time {a} ns")
        h = m - m // 2
        return min(m, max(h + a - b, min(m - b, h + a, max(m - 2 * b, 2 * a + h - 2 * b))))


class TableCost(RingTP):
    """``step_ns(shape)`` for a dense or MoE model at TP=N, priced from a profile file.

    ``cpu_mem_bw`` (GB/s = bytes/ns) and ``cpu_mem_latency_ns`` price the two
    tensors that live in CPU memory, as in ``LLMServingSimCost``.  At TP>1 the
    collectives are priced from ``link_bw`` (GB/s), ``link_latency_ns`` and
    ``npu_mem_bw`` (GB/s; default: the profile's ``hardware`` entry, else
    ``GPU_FACTS`` for its hardware).  ``shared_kv=False`` ignores the profile's
    shared-KV model (decode attention priced as if no two requests shared a block).
    """

    def __init__(self, profile_path, cpu_mem_bw: float = 256.0, cpu_mem_latency_ns: int = 0,
                 link_bw: float = 16.0, link_latency_ns: int = 20_000, npu_mem_bw: float | None = None,
                 shared_kv: bool = True):
        with gzip.open(profile_path, "rt", encoding="utf-8") as f:
            p = json.load(f)
        if p.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"{profile_path}: profile format {p.get('format_version')!r}, expected {FORMAT_VERSION}")
        ident = p["identity"]
        if ident.get("kind") not in ("dense", "moe"):
            raise NotImplementedError(f"TableCost prices dense and MoE models; profile kind is {ident.get('kind')!r}")
        moe = p.get("moe")
        if ident["kind"] == "moe" and not moe:
            raise NotImplementedError("a MoE profile needs its 'moe' tables (python -m augur.cost.convert)")
        self.identity = ident
        self.tp = ident["tp"]
        if moe and self.tp not in (1, 2, 4):
            raise NotImplementedError(f"MoE at tp={self.tp}: the MoE communication correction was fitted at TP2 and "
                                      "TP4 only (use --cost llmservingsim)")
        if self.tp > 1:
            coll = p.get("collectives")
            if not coll or any(not isinstance(coll.get(k), int) for k in
                               ("all_reduce_count", "all_reduce_bytes_per_token", "all_gather_bytes_per_seq")):
                raise NotImplementedError(f"tp={self.tp} profile has no collective coefficients")
            npu = npu_mem_bw if npu_mem_bw is not None else (p.get("hardware") or {}).get("npu_mem_bw_gbps")
            if npu is None:
                npu = gpu_mem_bw_gbps(ident.get("hardware"))
            if npu is None:
                raise ValueError("TP>1 needs npu_mem_bw (GB/s): the profile has no default for this hardware")
            if int(npu) < 1 or int(cpu_mem_bw) < 1 or link_bw <= 0 or link_latency_ns < 0:
                raise ValueError("TP>1 needs npu_mem_bw >= 1, cpu_mem_bw >= 1, link_bw > 0, link_latency_ns >= 0")
            self._coll = coll
            self.link_bw, self.link_latency_ns, self.npu_mem_bw = link_bw, link_latency_ns, npu
        self.cpu_mem_bw = cpu_mem_bw
        self.cpu_mem_latency_ns = cpu_mem_latency_ns
        self._dense = {n: (t["keys"], t["ns"]) for n, t in p["dense"].items()}
        self._per_seq = {n: (t["keys"], t["ns"]) for n, t in p["per_sequence"].items()}
        self._attn = {layer: {int(q): _Attention(spec) for q, spec in by_q.items()}
                      for layer, by_q in p["attention"].items()}
        self._skew = p["skew"]
        self._shared = p.get("shared_kv") if shared_kv else None
        self._moe_exact = (moe or {}).get("exact", {})
        self._moe_balanced = (moe or {}).get("balanced", {})
        self._moe_dp = (moe or {}).get("dp", {})
        if self._moe_dp and self.tp == 1:  # DP rounds run collectives between the ranks: link and HBM bandwidth
            npu = npu_mem_bw if npu_mem_bw is not None else (p.get("hardware") or {}).get("npu_mem_bw_gbps")
            npu = gpu_mem_bw_gbps(ident.get("hardware")) if npu is None else npu
            if npu is None or int(npu) < 1 or int(cpu_mem_bw) < 1 or link_bw <= 0 or link_latency_ns < 0:
                raise ValueError("DP rounds need npu_mem_bw >= 1, cpu_mem_bw >= 1, link_bw > 0, link_latency_ns >= 0")
            self.link_bw, self.link_latency_ns, self.npu_mem_bw = link_bw, link_latency_ns, npu
        lay = p["layers"]
        self._prologue, self._head, self._repeat = lay["prologue"], lay["head"], lay["repeat"]
        self._block = {}
        for name in lay["block"]:
            self._block[name] = self._block.get(name, 0) + 1
        self._in_bytes_per_token = p["remote"]["input_bytes_per_token"]
        self._out_bytes_per_seq = p["remote"]["output_bytes_per_seq"]
        self._warned: set = set()
        self._cache: dict[tuple, int] = {}
        # Steps priced above the profile's largest value on some axis (linear extrapolation from the
        # top two grid points, not a measurement): which axes, per step shape, and how many steps.
        self._beyond: dict[tuple, tuple] = {}
        self._limits = {"tokens": min(keys[-1] for keys, _ in self._dense.values()),
                        "sequences": min(keys[-1] for keys, _ in self._per_seq.values())}
        self.steps = 0
        self.steps_beyond_profile = 0
        if self.tp > 1:
            self._comm_ns(1, 1)  # a setting outside the exact regime fails here, before a run starts

    # --- attention -------------------------------------------------------

    def _attention_table(self, layer, q):
        by_q = self._attn[layer]
        q = max(1, int(q))
        if q in by_q:
            return by_q[q]
        nearest = min(by_q, key=lambda k: (abs(k - q), k))
        if (layer, q) not in self._warned:
            self._warned.add((layer, q))
            log.warning("decode_q_len=%d was not profiled for %r (have %s); using %d", q, layer, sorted(by_q), nearest)
        return by_q[nearest]

    def _alpha(self, layer, n, pc, lev, q):
        sk = self._skew
        kernel = sk["kernels"].get(layer, {}).get(str(q)) if sk else None
        if kernel is None:
            return 0.0
        lo, hi = kernel["n_range"]
        if not lo <= n <= hi:
            return kernel["default"]
        cell = kernel["cells"].get(f"{_label(sk['pc_bins'], sk['pc_labels'], int(pc))}"
                                   f"|{_label(sk['lev_bins'], sk['lev_labels'], float(lev))}")
        if not cell or not cell["anchors"]:
            return kernel["default"]
        return cell["alpha"][_anchor_index(cell["anchors"], n)]

    @property
    def prices_sharing(self) -> bool:
        """Whether the profile has a shared-KV model: the executor then fills ``SeqWork.shared_kv``."""
        return bool(self._shared)

    def _shared_scale(self, n, kv_total, shared) -> float:
        """Factor on the decode part of the attention time when ``shared`` of the decodes' ``kv_total`` KV tokens
        are in blocks another decode of the step also reads (module docstring)."""
        spec = self._shared
        u = 1.0 - min(shared, kv_total) / kv_total
        p = spec["p"]
        floor = _interp1d(spec["nd"], spec["floor"], n)
        return min(1.0, (u ** p + floor ** p) ** (1.0 / p))

    def _attention_ns(self, layer, pc, key, n, kv_mean, kv_max, q, shared=0):
        table = self._attention_table(layer, q)
        t_mean = table.lookup(pc, key, n, kv_mean)
        if n <= 1 or kv_max == kv_mean:
            ns = max(1, int(round(t_mean)))
        else:
            t_max = table.lookup(pc, key, n, kv_max)
            lev = (t_max - t_mean) / t_mean if t_mean > 0 else 0.0
            alpha = self._alpha(layer, n, pc, lev, q)
            if alpha == 0.0 or t_max <= t_mean:  # dense kernels never saturate in key length
                ns = max(1, int(round(t_mean)))
            else:
                ns = max(1, int(round(t_mean + alpha * (t_max - t_mean))))
        if shared > 0 and n > 1 and kv_mean > 0 and self._shared:
            base = table.lookup(pc, key, 0, 0) if pc and 0 in table.nd_vals else 0  # the prefill chunk alone
            ns = max(1, int(round(base + (ns - base) * self._shared_scale(n, n * kv_mean, shared))))
        return ns

    def _beyond_axes(self, pre, n_dec, kv_max, dq, tokens, rows):
        """Axes on which this step is above the profile's largest value, as ((axis, value, limit), ...).

        Above its last key a table is extended linearly from its top two points. That is the fork's rule
        too, but it is a guess: a profile measured to 2,048 tokens of KV says little about a 20,000-token
        decode, and LLMServingSim versions that extrapolate differently disagree there by tens of percent.
        """
        found = []
        lim = self._limits
        if tokens > lim["tokens"]:
            found.append(("forward tokens", tokens, lim["tokens"]))
        if rows > lim["sequences"]:
            found.append(("sequences", rows, lim["sequences"]))
        chunk = sum(q for q, _ in pre)
        if chunk or n_dec:
            tables = [self._attention_table(layer, dq) for layer in self._attn]
            for axis, value in (("prefill chunk tokens", chunk), ("decodes", n_dec), ("decode KV length", kv_max),
                                ("prefill key length", _prefill_key(pre) if chunk else 0)):
                top = min(t.limits[axis] for t in tables)
                if value > top:
                    found.append((axis, value, top))
        for axis, value, top in found:
            if axis not in self._warned:
                self._warned.add(axis)
                log.warning("step priced beyond the profile: %s %s is above the profile's largest value %s; "
                            "prices there are a linear extrapolation, not a measurement (augur counts such "
                            "steps as steps_beyond_profile in its summary)", axis, value, top)
        return tuple(found)

    # --- MoE block, TP>1 legacy path ---------------------------------------

    def _moe_ns(self, name, tokens):
        """Latency of MoE layer ``name`` at ``tokens`` forward tokens; None when ``name`` is not a MoE layer."""
        table = self._moe_exact.get(name)
        if table is not None:
            if tokens < 1 or tokens > len(table):
                raise OutsideRegime(f"the MoE block is profiled for 1..{len(table)} forward tokens, this step has "
                                    f"{tokens} (the fork's component tables stop there too)")
            return table[tokens - 1]
        spec = self._moe_balanced.get(name)
        return None if spec is None else self._moe_balanced_ns(spec, tokens)

    @staticmethod
    def _moe_balanced_ns(spec, tokens):
        """The fork's ``_balanced_route_ep`` (one rank) followed by ``_lookup_moe``, operation for operation."""
        if tokens <= 0:
            return 0
        e, k = max(1, spec["experts"]), spec["top_k"]
        miss = ((e - min(k, e)) / e) ** tokens
        active = max(min(e, max(1, int(round(e * (1.0 - miss))))), 1)
        ae, rows = spec["activated"], spec["rows"]
        t = max(int(tokens), 1)
        idx = bisect.bisect_right(ae, active)
        if idx == 0 or (idx >= len(ae) and len(ae) < 2):
            lo = hi = 0
        elif idx >= len(ae):
            lo, hi = len(ae) - 2, len(ae) - 1
        else:
            lo, hi = idx - 1, idx
        val_lo = _interp1d(rows[lo]["keys"], rows[lo]["ns"], t)
        if lo == hi:
            return max(1, int(val_lo))
        val_hi = _interp1d(rows[hi]["keys"], rows[hi]["ns"], t)
        x0, x1 = ae[lo], ae[hi]
        out = val_lo if x1 == x0 else val_lo + (active - x0) / (x1 - x0) * (val_hi - val_lo)
        return max(1, int(out))

    # --- drafter passes (EAGLE / MTP) -----------------------------------------

    def _batch_key(self, shape: StepShape) -> tuple:
        """What a batch's layer prices depend on: (prefill chunks, decodes, decode query length, mean and max
        decode KV length, decode KV tokens shared with another decode; 0 unless the profile prices sharing)."""
        pre, dec, dq, shared = [], [], 0, 0
        for s in shape.seqs:  # one pass: this runs on every step
            if s.is_decode:
                dec.append(s.num_computed_before)
                if s.num_new > dq:
                    dq = s.num_new
                if s.shared_kv:
                    shared += min(s.shared_kv, s.num_computed_before)
            else:
                pre.append((s.num_new, s.num_computed_before))
        n_dec = len(dec)
        return (tuple(pre), n_dec, dq or 1, sum(dec) // n_dec if n_dec else 0, max(dec, default=0),
                shared if self._shared else 0)

    def _pricer(self, batch: tuple, tokens: int, rows: int):
        """price(layer name) for one batch (``_batch_key``) of ``tokens`` forward tokens and ``rows`` sequences."""
        pre, n_dec, dq, kv_mean, kv_max, shared = batch
        t, r = max(int(tokens), 1), max(int(rows), 1)

        def price(name):
            moe = self._moe_ns(name, tokens)
            if moe is not None:
                return moe
            if name in self._attn:
                chunk = sum(q for q, _ in pre)
                if chunk == 0 and n_dec == 0:  # no attention kernel is launched (fork: _emit_npu_attention)
                    return 0
                return self._attention_ns(name, chunk, _prefill_key(pre), n_dec, kv_mean, kv_max, dq, shared)
            if name in self._per_seq:
                keys, vals = self._per_seq[name]
                return max(1, int(_interp1d(keys, vals, r)))
            keys, vals = self._dense[name]
            return max(1, int(_interp1d(keys, vals, t)))
        return price

    def draft_pass_ns(self, shape: StepShape, layers: int = 1, rows: int | None = None, proj: int = 0) -> int:
        """One forward pass of a drafter, priced from this model's own per-layer tables.

        ``layers`` decoder layers (the profile's layer block, attention included, at ``shape``), ``proj`` times the
        ``o_proj`` price for the drafter's input projection (EAGLE ``fc``, MTP ``eh_proj``; an approximation),
        then the final norm, ``lm_head`` and sampler for ``rows`` sequences (default: the shape's).  No embedding
        (the drafter reads the target's), no CPU-memory terms.  Used by ``augur.sim.draft.DraftCost``."""
        batch = self._batch_key(shape)
        tokens = shape.forward_tokens
        rows = len(shape.seqs) if rows is None else rows
        key = ("draft", *batch, tokens, rows, layers, proj)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        price = self._pricer(batch, tokens, rows)
        ns = layers * sum(count * price(n) for n, count in self._block.items())
        if proj and "o_proj" in self._dense:
            ns += proj * price("o_proj")
        ns += sum(price(n) for n in self._head if rows > 0 or n not in self._per_seq)
        if self.tp > 1:  # the drafter's token ids never leave the GPU: no CPU-memory terms; two all-reduces a layer
            _, ar, _, ag = self._ring_ns(tokens, rows)
            ns += 2 * layers * ar + ag
        if len(self._cache) >= CACHE_MAX:
            self._cache.clear()
        self._cache[key] = ns
        return ns

    # --- step ------------------------------------------------------------

    def round_ns(self, shapes, first_rank: int = 0) -> list[int]:
        """One DP round (see the module docstring): each rank's time from the round's first handover to its last
        CPU-memory write, as LLMServingSim's ASTRA-Sim run of the round. Called by: Coordinator (DP groups)."""
        spec = self._moe_dp.get(str(len(shapes)))
        if spec is None or self.tp != 1:
            raise NotImplementedError(f"TableCost prices DP rounds of {sorted(self._moe_dp) or 'no'} TP1 ranks; this "
                                      f"profile at tp={self.tp} has none for {len(shapes)} (use --cost llmservingsim)")
        counts = [s.forward_tokens for s in shapes]
        top = spec["max_tokens"]
        if min(counts) < 1 or max(counts) > top:
            raise OutsideRegime(f"DP round with forward token counts {counts}: the MoE tables cover 1..{top} per rank")
        summed, largest = sum(counts), max(counts)
        cpu_bw, lat = int(self.cpu_mem_bw), self.cpu_mem_latency_ns
        # collectives: (ring hop, duration) of the all-gathers and of the reduce-scatter
        bw = self.link_bw * (1 << 30) / 1000000000          # NetworkFunction.cpp, as _ring_ns
        local = int(self.npu_mem_bw) * 1000000000
        gathers = []
        for per_token in spec["all_gather_bytes_per_token"]:
            h = int(self.link_latency_ns + per_token * largest / bw)
            gathers.append((h, h + 20))
        msg = (2 * largest * spec["reduce_scatter_bytes_per_token"]) // 2
        h = int(self.link_latency_ns + msg / bw)
        scatter = [(h, h + 3 * int(msg / local * 1e9) + 20)]
        segments, reads, writes = [], [], []
        for rank, shape in enumerate(shapes):
            rows = 0 if shape.dummy else len(shape.seqs)
            t = counts[rank]
            price = self._pricer(self._batch_key(shape), t, rows)
            tab = spec["ranks"][rank]
            gate, fin, experts = tab["moe_gate_routing"][t - 1], tab["moe_finalize_copy"][t - 1], tab["moe_experts"][summed - 2]
            pre = sum(price(n) for n in spec["pre"]) + gate
            head = sum(price(n) for n in spec["tail"] if rows > 0 or n not in self._per_seq)
            seg = [sum(price(n) for n in self._prologue) + pre]
            for layer in range(spec["layers"]):
                seg += [experts, fin + (pre if layer < spec["layers"] - 1 else head)]
            segments.append(seg)
            reads.append(lat + int(self._in_bytes_per_token * t / cpu_bw))
            writes.append(lat + int(self._out_bytes_per_seq * rows / cpu_bw))
        n = len(shapes)
        t = [0] * n
        busy = 0
        for r in [first_rank] + [r for r in range(n) if r != first_rank]:  # token ids in, first member first
            busy += reads[r]
            t[r] = busy
        for k in range(len(segments[0])):
            t = [t[r] + segments[r][k] for r in range(n)]
            for h, d in (gathers if k % 2 == 0 else scatter) if k < len(segments[0]) - 1 else ():
                late = max(range(n), key=lambda r: (t[r], r))
                end, gap = t[late], max(t) - min(t)
                t = [end + d if r != late else end - min(gap, h) + d for r in range(n)]
        busy = 0
        for r in sorted(range(n), key=lambda r: (t[r], r)):  # sampled ids out, in finishing order
            busy = max(busy, t[r]) + writes[r]
            t[r] = busy
        self.steps += n
        return t

    def step_ns(self, shape: StepShape) -> int:
        batch = self._batch_key(shape)
        pre, n_dec, dq, _, kv_max, _ = batch
        tokens = shape.forward_tokens
        rows = 0 if shape.dummy else len(shape.seqs)
        key = (*batch, tokens, rows)
        hit = self._cache.get(key)
        self.steps += 1
        if hit is not None:
            if key in self._beyond:
                self.steps_beyond_profile += 1
            return hit

        comm = self._comm_ns(tokens, rows) if self.tp > 1 else None
        price = self._pricer(batch, tokens, rows)
        beyond = self._beyond_axes(pre, n_dec, kv_max, dq, max(int(tokens), 1), max(int(rows), 1))
        if beyond:
            self._beyond[key] = beyond
            self.steps_beyond_profile += 1
        ns = sum(price(n) for n in self._prologue)
        ns += self._repeat * sum(count * price(n) for n, count in self._block.items())
        ns += sum(price(n) for n in self._head if rows > 0 or n not in self._per_seq)
        if comm is not None:
            ns += comm
        else:  # token ids in, sampled ids out: both in CPU memory ("REMOTE")
            ns += self.cpu_mem_latency_ns + int(self._in_bytes_per_token * tokens / self.cpu_mem_bw)
            ns += self.cpu_mem_latency_ns + int(self._out_bytes_per_seq * rows / self.cpu_mem_bw)
        if len(self._cache) >= CACHE_MAX:
            self._cache.clear()
        self._cache[key] = ns
        return ns
