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
                 max_num_batched_tokens: int = 2048, dtype: str = "bfloat16"):
        if tp != 1 or ep != 1:
            raise NotImplementedError("TP/EP collectives need ASTRA-Sim (Phase 2); use tp=1, ep=1")
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
        b.requests = [_Req(i) for i in range(len(shape.seqs))]
        return b

    def step_ns(self, shape: StepShape) -> int:
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
        self._cache[key] = ns
        return ns
