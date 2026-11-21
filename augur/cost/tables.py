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
  per-step communication above is lower by ``max(0, min(M, A, max(M, A) - B))`` ns, M = the number of MoE
  layers (``collectives.moe_layers``), found by fitting ASTRA-Sim runs and checked on random shapes.
  It was fitted at TP2 only (at TP4 the residual has another form): a MoE profile at TP>2 is refused.

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


class _Attention:
    """One kernel at one decode query length: slices keyed by (chunk, decodes)."""

    def __init__(self, spec):
        axes = spec["kd_axes"]
        self.slices = {}
        for s in spec["slices"]:
            kd = axes[s["kd"]]
            self.slices[(s["pc"], s["nd"])] = (s["kp"], [(kd, row) for row in s["ns"]])
        self.pairs = sorted(self.slices)
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


# GPU memory bandwidth (GB/s) for hardware whose profiles carry none because the fork's
# hardware_defaults has no entry for it. H100: SXM5 HBM3, 3.35 TB/s, the value the fork's H100
# cluster configs set (an H100 PCIe has 2.0 TB/s: pass npu_mem_bw for it).
NPU_MEM_BW_GBPS = {"H100": 3350.0}


class OutsideRegime(ValueError):
    """A TP>1 step whose communication time the closed form does not reproduce exactly."""


class TableCost:
    """``step_ns(shape)`` for a dense or MoE model at TP=N, priced from a profile file.

    ``cpu_mem_bw`` (GB/s = bytes/ns) and ``cpu_mem_latency_ns`` price the two
    tensors that live in CPU memory, as in ``LLMServingSimCost``.  At TP>1 the
    collectives are priced from ``link_bw`` (GB/s), ``link_latency_ns`` and
    ``npu_mem_bw`` (GB/s; default: the profile's ``hardware`` entry, else
    ``NPU_MEM_BW_GBPS`` for its hardware).
    """

    def __init__(self, profile_path, cpu_mem_bw: float = 256.0, cpu_mem_latency_ns: int = 0,
                 link_bw: float = 16.0, link_latency_ns: int = 20_000, npu_mem_bw: float | None = None):
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
        if self.tp > 2 and moe:
            raise NotImplementedError(f"MoE at tp={self.tp}: the MoE communication correction was fitted at TP2 only "
                                      "(use --cost llmservingsim)")
        if self.tp > 1:
            coll = p.get("collectives")
            if not coll or any(not isinstance(coll.get(k), int) for k in
                               ("all_reduce_count", "all_reduce_bytes_per_token", "all_gather_bytes_per_seq")):
                raise NotImplementedError(f"tp={self.tp} profile has no collective coefficients")
            npu = npu_mem_bw if npu_mem_bw is not None else (p.get("hardware") or {}).get("npu_mem_bw_gbps")
            if npu is None:
                npu = NPU_MEM_BW_GBPS.get(ident.get("hardware"))
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
        self._moe_exact = (moe or {}).get("exact", {})
        self._moe_balanced = (moe or {}).get("balanced", {})
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

    def _attention_ns(self, layer, pc, key, n, kv_mean, kv_max, q):
        table = self._attention_table(layer, q)
        t_mean = table.lookup(pc, key, n, kv_mean)
        if n <= 1 or kv_max == kv_mean:
            return max(1, int(round(t_mean)))
        t_max = table.lookup(pc, key, n, kv_max)
        lev = (t_max - t_mean) / t_mean if t_mean > 0 else 0.0
        alpha = self._alpha(layer, n, pc, lev, q)
        if alpha == 0.0 or t_max <= t_mean:  # dense kernels never saturate in key length
            return max(1, int(round(t_mean)))
        return max(1, int(round(t_mean + alpha * (t_max - t_mean))))

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

    # --- TP>1 communication ----------------------------------------------

    def _comm_ns(self, tokens: int, rows: int) -> int:
        n, lat, c = self.tp, self.link_latency_ns, self._coll
        bw = self.link_bw * (1 << 30) / 1000000000          # NetworkFunction.cpp: bw * (1<<30) / 1e9 bytes/ns
        cpu_bw = int(self.cpu_mem_bw)                       # AnalyticalMemory: uint64 mem_bw
        local = int(self.npu_mem_bw) * 1000000000           # Sys.cc: local-mem-bw (json, int) * 1e9
        a = self.cpu_mem_latency_ns + int(self._in_bytes_per_token * tokens / cpu_bw)  # token ids in (CPU memory)
        b = self.cpu_mem_latency_ns + int(self._out_bytes_per_seq * rows / cpu_bw)     # sampled ids out
        msg = (c["all_reduce_bytes_per_token"] * tokens) // n   # Ring.cc: chunk = data_size / N
        hop = int(lat + msg / bw)                           # BasicTopology.cpp: one FullyConnected hop
        ag_hop = int(lat + c["all_gather_bytes_per_seq"] * rows / bw) if rows else hop
        if 2 * a > hop or 2 * b > ag_hop:
            raise OutsideRegime(
                f"TP={n} step (T={tokens}, R={rows}): a CPU-memory access (A={a} ns for the token ids, B={b} ns "
                f"for the sampled ids; cpu latency {self.cpu_mem_latency_ns} ns, bandwidth {self.cpu_mem_bw} GB/s) "
                f"is more than half a ring step (all-reduce {hop} ns, last collective {ag_hop} ns; link latency "
                f"{lat} ns, bandwidth {self.link_bw} GB/s); the closed form is not exact there. "
                f"Use --cost llmservingsim")
        # Ring.cc: 2(N-1) steps; reduce half is processed by the memory bus, 3 x size/local_bw (PacketBundle.cc);
        # MemBus + endpoint delay (10 ns, system.json) once per step and once for the first packet
        ar = 2 * (n - 1) * hop + (n - 1) * 3 * int(msg / local * 1e9) + 10 * (2 * (n - 1) + 1)
        total = n * max(a, b) + min(a, b) + c["all_reduce_count"] * ar   # CPU memory: one FIFO queue per node
        m = c.get("moe_layers", 0)
        if m:  # see the MoE note in the module docstring
            total -= max(0, min(m, a, max(m, a) - b))
        if rows:  # all-gather of the per-rank logits chunk, N-1 steps (Ring.cc)
            total += (n - 1) * ag_hop + 10 * n
        return total

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

    def _draft_comm_ns(self, tokens: int, rows: int, layers: int) -> int:
        """Collectives of ``layers`` decoder layers plus the logits all-gather at TP>1: the formula of
        ``_comm_ns`` without its CPU-memory terms (the drafter's token ids never leave the GPU) and with two
        all-reduces per layer."""
        n, lat, c = self.tp, self.link_latency_ns, self._coll
        bw = self.link_bw * (1 << 30) / 1000000000
        local = int(self.npu_mem_bw) * 1000000000
        msg = (c["all_reduce_bytes_per_token"] * tokens) // n
        hop = int(lat + msg / bw)
        ar = 2 * (n - 1) * hop + (n - 1) * 3 * int(msg / local * 1e9) + 10 * (2 * (n - 1) + 1)
        total = 2 * layers * ar
        if rows:
            ag_hop = int(lat + c["all_gather_bytes_per_seq"] * rows / bw)
            total += (n - 1) * ag_hop + 10 * n
        return total

    def draft_pass_ns(self, shape: StepShape, layers: int = 1, rows: int | None = None, proj: int = 0) -> int:
        """One forward pass of a drafter, priced from this model's own per-layer tables.

        ``layers`` decoder layers (the profile's layer block, attention included, at ``shape``), ``proj`` times the
        ``o_proj`` price for the drafter's input projection (EAGLE ``fc``, MTP ``eh_proj``; an approximation),
        then the final norm, ``lm_head`` and sampler for ``rows`` sequences (default: the shape's).  No embedding
        (the drafter reads the target's), no CPU-memory terms.  Used by ``augur.sim.draft.DraftCost``."""
        pre = tuple((s.num_new, s.num_computed_before) for s in shape.seqs if not s.is_decode)
        dec = [s.num_computed_before for s in shape.seqs if s.is_decode]
        n_dec = len(dec)
        dq = max((s.num_new for s in shape.seqs if s.is_decode), default=1) or 1
        kv_mean = sum(dec) // n_dec if n_dec else 0
        kv_max = max(dec, default=0)
        tokens = shape.forward_tokens
        rows = len(shape.seqs) if rows is None else rows
        key = ("draft", pre, n_dec, dq, kv_mean, kv_max, tokens, rows, layers, proj)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        t, r = max(int(tokens), 1), max(int(rows), 1)

        def price(name):
            if name in self._attn:
                chunk = sum(q for q, _ in pre)
                if chunk == 0 and n_dec == 0:
                    return 0
                return self._attention_ns(name, chunk, _prefill_key(pre), n_dec, kv_mean, kv_max, dq)
            if name in self._per_seq:
                keys, vals = self._per_seq[name]
                return max(1, int(_interp1d(keys, vals, r)))
            keys, vals = self._dense[name]
            return max(1, int(_interp1d(keys, vals, t)))

        ns = layers * sum(count * price(n) for n, count in self._block.items())
        if proj and "o_proj" in self._dense:
            ns += proj * price("o_proj")
        ns += sum(price(n) for n in self._head if rows > 0 or n not in self._per_seq)
        if self.tp > 1:
            ns += self._draft_comm_ns(tokens, rows, layers)
        self._cache[key] = ns
        return ns

    # --- step ------------------------------------------------------------

    def round_ns(self, shapes, first_rank: int = 0):
        raise NotImplementedError("TableCost does not price DP rounds (EP collectives)")

    def step_ns(self, shape: StepShape) -> int:
        pre = tuple((s.num_new, s.num_computed_before) for s in shape.seqs if not s.is_decode)
        dec = [s.num_computed_before for s in shape.seqs if s.is_decode]
        n_dec = len(dec)
        dq = max((s.num_new for s in shape.seqs if s.is_decode), default=1) or 1
        kv_mean = sum(dec) // n_dec if n_dec else 0
        kv_max = max(dec, default=0)
        tokens = shape.forward_tokens
        rows = 0 if shape.dummy else len(shape.seqs)
        key = (pre, n_dec, dq, kv_mean, kv_max, tokens, rows)
        hit = self._cache.get(key)
        self.steps += 1
        if hit is not None:
            if key in self._beyond:
                self.steps_beyond_profile += 1
            return hit

        comm = self._comm_ns(tokens, rows) if self.tp > 1 else None
        t = max(int(tokens), 1)
        r = max(int(rows), 1)

        def price(name):
            moe = self._moe_ns(name, tokens)
            if moe is not None:
                return moe
            if name in self._attn:
                chunk = sum(q for q, _ in pre)
                if chunk == 0 and n_dec == 0:  # no attention kernel is launched (fork: _emit_npu_attention)
                    return 0
                return self._attention_ns(name, chunk, _prefill_key(pre), n_dec, kv_mean, kv_max, dq)
            if name in self._per_seq:
                keys, vals = self._per_seq[name]
                return max(1, int(_interp1d(keys, vals, r)))
            keys, vals = self._dense[name]
            return max(1, int(_interp1d(keys, vals, t)))

        beyond = self._beyond_axes(pre, n_dec, kv_max, dq, t, r)
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
        self._cache[key] = ns
        return ns
