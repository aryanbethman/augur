"""Step cost from LLMServingSim's profiles (the fork's trace generator).

The step shape vLLM's scheduler chose becomes the fork's ``Batch``, the same
object its own scheduler builds (``Scheduler._build_batch``): a request with
one new token past its known tokens is a decode, anything else a prefill
chunk. ``generate_trace`` then looks every layer up in the profile bundle.

For a single engine without collectives (TP=1, EP=1) the step time is the sum
of the emitted layer latencies, which is what ASTRA-Sim computes for one NPU
running the trace serially. Collectives (TP>1, EP>1, PD transfers) go to
ASTRA-Sim in Phase 2.

Results are memoized on the exact inputs the trace generator reads, so the
cache never changes an answer.
"""

from __future__ import annotations

import os
import sys

from augur.cost import StepShape


class _Req:  # the trace generator counts batch.requests (lm_head rows) and logs their ids
    __slots__ = ("id",)

    def __init__(self, i: int) -> None:
        self.id = i


class LLMServingSimCost:
    def __init__(self, fork_root: str, hardware: str, model: str, tp: int = 1, ep: int = 1,
                 max_num_batched_tokens: int = 2048, dtype: str = "bfloat16",
                 cpu_mem_bw_gbps: float = 256.0, cpu_mem_latency_ns: int = 0,
                 cluster_config: str | None = None, dp: int = 1):
        self.fork_root = os.path.abspath(fork_root)
        if self.fork_root not in sys.path:
            sys.path.insert(0, self.fork_root)
        self._run_dir = os.path.join(self.fork_root, "serving")
        cwd = os.getcwd()
        os.chdir(self._run_dir)  # the fork resolves configs and profiles relative to serving/
        try:
            from serving.core import trace_generator as tg
            from serving.core.request import Batch
            from serving.core.config_builder import _mem_str
        finally:
            os.chdir(cwd)
        self._tg = tg
        self._Batch = Batch
        # The fork's default placement: weights and KV on the NPU, evictions to CPU.
        self._placement = {"default": {"weights": _mem_str("npu", 0), "kv_loc": _mem_str("npu", 0),
                                       "kv_evict_loc": _mem_str("cpu", 0)}, "block": [], "layer": {}}
        self.hardware = hardware
        self.model = model
        self.tp = tp
        self.ep = ep
        self.max_num_batched_tokens = max_num_batched_tokens
        self.dtype = dtype
        # Tensors placed in CPU memory ("REMOTE", e.g. the token ids the
        # embedding reads) cost what ASTRA-Sim's analytical memory charges:
        # latency + floor(bytes / bandwidth), bandwidth in GB/s = bytes/ns.
        self.cpu_mem_bw = cpu_mem_bw_gbps
        self.cpu_mem_latency_ns = cpu_mem_latency_ns
        # TP>1 (collectives): ASTRA-Sim prices everything that is not compute,
        # from LLMServingSim's cluster config (link bandwidth, latency, topology).
        self.oracle = None
        self.dp = dp
        if tp > 1 or ep > 1 or dp > 1:
            if cluster_config is None:
                raise ValueError("tp/ep/dp > 1 needs an LLMServingSim cluster config for ASTRA-Sim (--lss-cluster-config)")
            from augur.astra_oracle import AstraOracle
            self.oracle = AstraOracle(self.fork_root, cluster_config)
        self._cache: dict[tuple, int] = {}
        self._batch_id = 0
        self.lookups = 0
        self.hits = 0

    def _batch(self, shape: StepShape):
        q_list, k_list = [], []
        prefill_q, prefill_k, decode_k = [], [], []
        kv_len = 0
        for s in shape.seqs:
            q_list.append(s.num_new)
            k_list.append(s.num_computed_before)
            if s.is_decode:
                decode_k.append(s.num_computed_before)
                kv_len += s.num_computed_before
            else:
                prefill_q.append(s.num_new)
                prefill_k.append(s.num_computed_before)
        self._batch_id += 1
        # Dense/MoE layers run at the padded token count; attention keeps the
        # real query/KV lists (the fork's _pad_batch_to_max does the same).
        b = self._Batch(self._batch_id, self.model, shape.forward_tokens, kv_len, q_list, k_list,
                        len(prefill_q), len(decode_k), prefill_q, prefill_k, decode_k, 0, 0)
        # A DP dummy forward has no requests: no logits, no sampler.
        b.requests = [] if shape.dummy else [_Req(i) for i in range(len(shape.seqs))]
        return b

    def _trace(self, shape: StepShape):
        cwd = os.getcwd()
        os.chdir(self._run_dir)
        try:
            return self._tg.generate_trace(
                self._batch(shape), self.hardware, self.tp, 1, self.ep, self.ep,
                max_num_batched_tokens=self.max_num_batched_tokens, dtype=self.dtype,
                placement=self._placement, enable_block_copy=True,
            )
        finally:
            os.chdir(cwd)

    def round_ns(self, shapes: list[StepShape], first_rank: int = 0) -> list[int]:
        """One DP round: every rank's step (or dummy) runs together in one
        ASTRA-Sim run, joined by the EP collectives. Each rank's trace carries
        the round's token counts, as LLMServingSim's DP path generates it.
        Returns each rank's own finish time: the ranks meet at every
        collective, but differ in the work after the last one."""
        counts = tuple(s.forward_tokens for s in shapes)
        insts = self.oracle.cluster["instances"]
        members = []
        cwd = os.getcwd()
        os.chdir(self._run_dir)
        try:
            for rank, shape in enumerate(shapes):
                inst = insts[rank]
                trace = self._tg.generate_trace(
                    self._batch(shape), self.hardware, inst["tp_size"], inst.get("pp_size", 1),
                    inst["local_ep"], inst["ep_total"],
                    max_num_batched_tokens=self.max_num_batched_tokens, dtype=self.dtype,
                    placement=self._placement, enable_block_copy=True,
                    tp_dim=inst.get("tp_dim"), ep_dim=inst.get("ep_dim"),
                    dp_sum_total_len=sum(counts), dp_min_total_len=min(counts),
                    dp_token_counts=counts, dp_rank=rank,
                )
                rows = [r for r in trace.rows if not (len(r) == 11 and r[0] in ("kv_load", "kv_evict"))]
                members.append((rows, trace.header_line, rank))
        finally:
            os.chdir(cwd)
        self.lookups += 1
        return self.oracle.run_group(members, self.model, self.hardware, first_member=first_rank)

    def _step_ns_astra(self, shape: StepShape) -> int:
        """compute (sum of layer latencies) + ASTRA-Sim's time for the rest.

        Checked exact against a full ASTRA-Sim run of the same trace for
        decode, mixed and prefill steps at TP2."""
        self.lookups += 1
        dec = [s.num_computed_before for s in shape.seqs if s.is_decode]
        pre = tuple((s.num_new, s.num_computed_before) for s in shape.seqs if not s.is_decode)
        n_dec = len(dec)
        key = (pre, n_dec, (sum(dec) // n_dec) if n_dec else 0, max(dec, default=0), min(dec, default=0),
               shape.forward_tokens, len(shape.seqs))
        hit = self._cache.get(key)
        if hit is not None:
            self.hits += 1
            return hit
        trace = self._trace(shape)
        comp = sum(int(r[1]) for r in trace.rows if len(r) == 11 and r[0] not in ("kv_load", "kv_evict"))
        rows = [r for r in trace.rows if not (len(r) == 11 and r[0] in ("kv_load", "kv_evict"))]
        ns = comp + self.oracle.comm_ns(rows, trace.header_line, self.model, self.hardware, self.tp)
        self._cache[key] = ns
        return ns

    def trace_rows(self, shape: StepShape) -> list:
        cwd = os.getcwd()
        os.chdir(self._run_dir)
        try:
            trace = self._tg.generate_trace(
                self._batch(shape), self.hardware, self.tp, 1, self.ep, self.ep,
                max_num_batched_tokens=self.max_num_batched_tokens, dtype=self.dtype,
                placement=self._placement, enable_block_copy=True,
            )
        finally:
            os.chdir(cwd)
        return trace.rows

    def step_ns(self, shape: StepShape) -> int:
        if self.oracle is not None:
            return self._step_ns_astra(shape)
        self.lookups += 1
        dec = [s.num_computed_before for s in shape.seqs if s.is_decode]
        pre = tuple((s.num_new, s.num_computed_before) for s in shape.seqs if not s.is_decode)
        n_dec = len(dec)
        key = (pre, n_dec, (sum(dec) // n_dec) if n_dec else 0, max(dec, default=0), min(dec, default=0),
               shape.forward_tokens, len(shape.seqs))
        hit = self._cache.get(key)
        if hit is not None:
            self.hits += 1
            return hit
        cwd = os.getcwd()
        os.chdir(self._run_dir)
        try:
            trace = self._tg.generate_trace(
                self._batch(shape), self.hardware, self.tp, 1, self.ep, self.ep,
                max_num_batched_tokens=self.max_num_batched_tokens, dtype=self.dtype,
                placement=self._placement, enable_block_copy=True,  # same prices, ~10x faster
            )
        finally:
            os.chdir(cwd)
        ns = 0
        for row in trace.rows:
            # Skip EXPERT/PIM markers (one field). With EP=1 they wrap a single
            # layer row each, which runs serially like the rest.
            if len(row) != 11 or row[0] in ("kv_load", "kv_evict"):
                continue
            ns += int(row[1])
            for loc, size in ((row[2], row[3]), (row[4], row[5]), (row[6], row[7])):
                if loc.startswith("REMOTE"):
                    ns += self.cpu_mem_latency_ns + int(int(size) / self.cpu_mem_bw)
        self._cache[key] = ns
        return ns
