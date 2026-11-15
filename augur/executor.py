"""SimExecutor: vLLM's Executor seam, with no GPU and no model.

vLLM's EngineCore builds its executor through ``distributed_executor_backend``
(a class is accepted as-is). Ours answers the start-up questions (KV-cache
spec, available memory, layouts) from the model config and the simulated
hardware, and answers ``execute_model`` by:

* emitting a token for every request whose known tokens are all computed after
  this step (partial prefill chunks emit nothing, as in vLLM's GPU runner), and
* asking the cost model how long the step takes.

Emitted tokens are the workload's recorded output IDs when it has them, so a
later turn of an agentic session hits the prefix cache on this turn's output,
exactly as in the recorded run.
"""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Callable

import random
import zlib

import torch

from vllm.config import CUDAGraphMode
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.executor.abstract import Executor
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput

from augur.cost import CostModel, SeqWork, StepShape
from augur.vllm_hooks import call_id

SYNTHETIC_TOKEN = 0  # emitted when the workload recorded no output IDs


@dataclass
class SimHardware:
    """What the executor needs to know about the simulated GPU."""

    kv_cache_bytes: int  # memory left for the KV cache on each GPU


@dataclass
class SimContext:
    """Shared between the engine driver and its executor (same process)."""

    cost: CostModel
    hardware: SimHardware
    # req_id -> recorded output token ids (may be shorter than the output, or empty)
    output_ids: dict[str, list[int]] = field(default_factory=dict)
    # Written by the executor after each step, read by the driver.
    last_step_ns: int = 0
    last_shape: StepShape | None = None
    last_new_req_ids: dict[str, int] = field(default_factory=dict)
    last_preempted_req_ids: list[str] = field(default_factory=list)
    # PD (KV connector) plumbing. The real connector's scheduler half runs in
    # vLLM; this stands in for its worker half.
    kv_block_bytes: int = 0  # KV bytes of one block on this rank, all layers
    new_recvs: list = field(default_factory=list)  # (req_id, blocks, remote_engine_id, remote_req_id)
    deliver_recving: set = field(default_factory=set)  # reported as finished_recving next step
    deliver_sending: set = field(default_factory=set)  # reported as finished_sending next step
    last_delivered: bool = False
    # DP ranks: the coordinator prices the whole round (all ranks together,
    # padded per vLLM's DP rule), so the executor only reports its shape.
    defer_cost: bool = False
    # Speculative decoding (sync): k draft tokens per request, accepted with
    # this rate each; drafts handed to vLLM's scheduler via take_draft_token_ids.
    spec_k: int = 0
    spec_accept_rate: float = 0.0
    pending_drafts: object = None
    # Async scheduling: every execute_model call is a launch onto the GPU
    # stream; the coordinator drains these durations (0 for an empty batch).
    launches: list = field(default_factory=list)
    launch_shapes: list = field(default_factory=list)  # DP async: each launch's shape (None if empty)
    # vLLM's per-step CPU work (schedule, input preparation, sampling,
    # output processing), which GPU profiles do not include: a fixed part
    # plus a part per scheduled request. Synchronous: added to the step.
    # Async: delays the batch's launch (overlaps the previous GPU batch).
    async_cpu: bool = False  # set for async engines: CPU time goes into the launch time
    cpu_fixed_ns: int = 0
    cpu_per_request_ns: int = 0
    last_cpu_ns: int = 0


@dataclass
class _ReqState:
    num_prompt: int
    num_output: int = 0


class SimWorker:
    """The worker-side methods EngineCore reaches through ``collective_rpc``."""

    def __init__(self, vllm_config, ctx: SimContext):
        self.vllm_config = vllm_config
        self.ctx = ctx
        self.reqs: dict[str, _ReqState] = {}
        # vLLM's own CUDA-graph dispatcher decides how far each forward pass is
        # padded, as in the GPU model runner (uniform decode: one token per request).
        self.cudagraph = CudagraphDispatcher(vllm_config)
        mode = vllm_config.compilation_config.cudagraph_mode or CUDAGraphMode.NONE
        self.cudagraph.initialize_cudagraph_keys(mode, uniform_decode_query_len=1 + ctx.spec_k)
        self._step = 0

    # ---- start-up -------------------------------------------------------
    def get_kv_cache_spec(self) -> dict[str, FullAttentionSpec]:
        mc = self.vllm_config.model_config
        pc = self.vllm_config.parallel_config
        num_layers = mc.get_num_layers(pc)
        num_kv_heads = mc.get_num_kv_heads(pc)
        head_size = mc.get_head_size()
        dtype = self.vllm_config.cache_config.cache_dtype
        torch_dtype = mc.dtype if dtype in ("auto", None) else _kv_dtype(dtype)
        block_size = self.vllm_config.cache_config.block_size
        spec = FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=torch_dtype,
        )
        self.ctx.kv_block_bytes = spec.page_size_bytes * num_layers
        return {f"model.layers.{i}.self_attn.attn": spec for i in range(num_layers)}

    def get_supported_kv_cache_layouts(self) -> list[str]:
        return ["NHD"]

    def set_kv_cache_layout(self, layout_name: str, interleave: int) -> None:
        return None

    def determine_available_memory(self) -> int:
        return self.ctx.hardware.kv_cache_bytes

    def initialize_from_config(self, kv_cache_configs) -> None:
        return None

    def get_kv_connector_handshake_metadata(self):
        return None

    def _replay_step(self):
        """(emitted per call id this step, scheduled per call id next step) when replaying."""
        cost = self.ctx.cost
        outs = getattr(cost, "outs", None)
        if not outs or cost.k >= len(outs):
            return None
        nxt = cost.toks[cost.k + 1] if cost.k + 1 < len(cost.toks) else {}
        return outs[cost.k], nxt

    def take_draft_token_ids(self):
        out, self.ctx.pending_drafts = self.ctx.pending_drafts, None
        return out

    # ---- steps -----------------------------------------------------------
    def execute_model(self, so) -> ModelRunnerOutput:
        for rid in so.finished_req_ids:
            self.reqs.pop(rid, None)
        if so.preempted_req_ids:
            for rid in so.preempted_req_ids:
                self.reqs.pop(rid, None)

        # KV connector: requests whose KV the worker starts pulling this step.
        self.ctx.new_recvs = []
        meta = so.kv_connector_metadata
        for rid, rm in (getattr(meta, "reqs_to_recv", None) or {}).items():
            n_local = sum(len(g) for g in rm.local_block_ids)
            n_done = sum(rm.local_num_computed_blocks) if rm.local_num_computed_blocks else 0
            r = rm.remote
            self.ctx.new_recvs.append((rid, max(n_local - n_done, 0),
                                       r.engine_id if r else None, r.request_id if r else None))
        kvo = None
        self.ctx.last_delivered = bool(self.ctx.deliver_recving or self.ctx.deliver_sending)
        if self.ctx.last_delivered:
            kvo = KVConnectorOutput(finished_sending=set(self.ctx.deliver_sending) or None,
                                    finished_recving=set(self.ctx.deliver_recving) or None)
            self.ctx.deliver_recving.clear()
            self.ctx.deliver_sending.clear()

        work: list[tuple[str, int, int]] = []  # (req_id, computed_before, num_new)
        # A new request's computed tokens at its first schedule are its prefix-cache hit.
        self.ctx.last_new_req_ids = {nr.req_id: nr.num_computed_tokens for nr in so.scheduled_new_reqs}
        self.ctx.last_preempted_req_ids = list(so.preempted_req_ids or ())
        for nr in so.scheduled_new_reqs:
            # With vLLM's v2 model runner (0.33's default) a preempted request
            # comes back as a new request whose prefill_token_ids hold its
            # prompt plus the outputs it already generated; it samples again
            # only once all of those are recomputed.
            n_prompt = len(nr.prompt_token_ids or ())
            n_known = len(nr.prefill_token_ids) if nr.prefill_token_ids else n_prompt
            self.reqs[nr.req_id] = _ReqState(num_prompt=n_prompt, num_output=n_known - n_prompt)
            work.append((nr.req_id, nr.num_computed_tokens, so.num_scheduled_tokens[nr.req_id]))
        cr = so.scheduled_cached_reqs
        for i, rid in enumerate(cr.req_ids):
            st = self.reqs.get(rid)
            if st is None or rid in cr.resumed_req_ids:
                # Resumed after preemption: the scheduler tells us how far it got.
                all_ids = cr.all_token_ids.get(rid) if cr.all_token_ids else None
                n_out = cr.num_output_tokens[i]
                n_prompt = (len(all_ids) - n_out) if all_ids is not None else st.num_prompt
                st = self.reqs[rid] = _ReqState(num_prompt=n_prompt, num_output=n_out)
            work.append((rid, cr.num_computed_tokens[i], so.num_scheduled_tokens[rid]))

        req_ids: list[str] = []
        sampled: list[list[int]] = []
        seqs: list[SeqWork] = []
        drafts_next: dict[str, list[int]] = {}
        spec = so.scheduled_spec_decode_tokens or {}
        self._step += 1
        for rid, before, num_new in work:
            st = self.reqs[rid]
            known = st.num_prompt + st.num_output
            n_draft = len(spec.get(rid, ()))
            # Decode vs prefill by query length, as vLLM's attention backends
            # split a batch: one token is a decode (a one-token last prefill
            # chunk runs as one), and so is 1 + k under speculative decoding,
            # where k draft tokens are verified alongside the real one.
            is_decode = num_new == 1 or (n_draft > 0 and num_new == 1 + n_draft)
            seqs.append(SeqWork(num_new=num_new, num_computed_before=before, is_decode=is_decode))
            req_ids.append(rid)
            if before + num_new >= known + n_draft:
                rec = self.ctx.output_ids.get(rid)
                # Speculative decoding: each draft is accepted with the given
                # rate until the first rejection, then one more token is
                # sampled. Content decides acceptance on a GPU; augur cannot
                # see content, so the rate is a parameter.
                accepted = 0
                replay = self._replay_step()
                if replay is not None:
                    # Replay: the logged run's acceptance (content-dependent).
                    accepted = max(replay[0].get(call_id(rid), 1) - 1, 0)
                elif n_draft:
                    rng = random.Random(zlib.crc32(f"{rid}/{self._step}".encode()))
                    while accepted < n_draft and rng.random() < self.ctx.spec_accept_rate:
                        accepted += 1
                n_emit = accepted + 1
                if rec:
                    n_emit = min(n_emit, max(len(rec) - st.num_output, 1))
                toks = [rec[st.num_output + i] if rec and st.num_output + i < len(rec) else SYNTHETIC_TOKEN
                        for i in range(n_emit)]
                st.num_output += n_emit
                sampled.append(toks)
                if self.ctx.spec_k and rec:
                    # The drafter proposes the request's next recorded tokens,
                    # so accepted drafts are the right tokens. Replay: as many
                    # drafts as the logged run scheduled next step (an ngram
                    # drafter proposes none when the prompt has no match).
                    n_next = self.ctx.spec_k
                    if replay is not None:
                        n_next = max(replay[1].get(call_id(rid), 1) - 1, 0)
                    nxt = rec[st.num_output:st.num_output + n_next]
                    if nxt:
                        drafts_next[rid] = list(nxt)
            else:
                sampled.append([])
        if self.ctx.spec_k:
            self.ctx.pending_drafts = DraftTokenIds(req_ids=list(drafts_next),
                                                    draft_token_ids=list(drafts_next.values()))

        if self.ctx.defer_cost:
            self.ctx.last_shape = StepShape(seqs=tuple(seqs))
            self.ctx.last_step_ns = 0
        else:
            num_tokens = sum(x.num_new for x in seqs)
            q = 1 + self.ctx.spec_k
            uniform_decode = bool(seqs) and all(x.num_new == q for x in seqs)
            _, desc = self.cudagraph.dispatch(num_tokens=num_tokens, uniform_decode=uniform_decode)
            shape = StepShape(seqs=tuple(seqs), padded_tokens=desc.num_tokens)
            self.ctx.last_shape = shape
            gpu = self.ctx.cost.step_ns(shape) if seqs else 0
            cpu = (self.ctx.cpu_fixed_ns + self.ctx.cpu_per_request_ns * len(seqs)) if seqs else 0
            self.ctx.last_cpu_ns = cpu
            self.ctx.last_step_ns = gpu + cpu if not self.ctx.async_cpu else gpu
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={r: i for i, r in enumerate(req_ids)},
            sampled_token_ids=sampled,
            kv_connector_output=kvo,
        )


def _kv_dtype(name: str) -> torch.dtype:
    if name.startswith("fp8"):
        return torch.float8_e4m3fn
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "half": torch.float16}[name]


def make_sim_executor(ctx: SimContext) -> type[Executor]:
    """An Executor subclass bound to one engine's SimContext."""

    class SimExecutor(Executor):
        uses_ray = False
        supports_pp = False
        _ctx = ctx

        def _init_executor(self) -> None:
            self.worker = SimWorker(self.vllm_config, self._ctx)

        def collective_rpc(self, method: str | Callable, timeout=None, args=(), kwargs=None, non_block=False):
            kwargs = kwargs or {}
            fn = getattr(self.worker, method) if isinstance(method, str) else (lambda *a, **k: method(self.worker, *a, **k))
            result: list[Any] = [fn(*args, **kwargs)]
            if non_block:
                fut: Future = Future()
                fut.set_result(result)
                return fut
            return result

        _pending = None

        def execute_model(self, scheduler_output, non_block=False):
            out = self.worker.execute_model(scheduler_output)
            self._ctx.launches.append(self._ctx.last_step_ns)
            if self._ctx.defer_cost:
                self._ctx.launch_shapes.append(
                    self._ctx.last_shape if scheduler_output.total_num_scheduled_tokens > 0 else None)
            if scheduler_output.total_num_scheduled_tokens > 0:
                # As vLLM's GPU runner: the forward returns None and the
                # output comes from sample_tokens.
                self._pending = out
                out = None
            if non_block:
                fut: Future = Future()
                fut.set_result(out)
                return fut
            return out

        def sample_tokens(self, grammar_output, non_block=False):
            out, self._pending = self._pending, None
            assert out is not None, "sample_tokens without a forward"
            if non_block:
                fut: Future = Future()
                fut.set_result(out)
                return fut
            return out

        def compile_or_warm_up_model(self) -> None:
            return None

        def check_health(self) -> None:
            return None

        def shutdown(self) -> None:
            return None

        @classmethod
        def supports_async_scheduling(cls) -> bool:
            return True

    return SimExecutor
