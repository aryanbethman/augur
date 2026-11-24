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

from vllm.config import CUDAGraphMode
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiKVConnectorMetadata,
    MultiKVConnectorWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import OffloadingConnectorMetadata
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.executor.abstract import Executor
from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput

from augur.cost import CostModel, SeqWork, StepShape
from augur.sim.draft import draft_kv_spec
from augur.sim.kvspec import layer_kv_specs
from augur.sim.ngram import Context, NgramDrafter
from augur.trace.workload import call_id

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
    recorded: set[str] = field(default_factory=set)  # req_ids whose output_ids are all recorded (not filler)
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
    spec_rate_only: bool = False  # ngram: use the rate even for recorded calls (else vLLM's drafter + exact check)
    # Per request verified with drafts, beyond the profile's price: rejection sampling and draft bookkeeping (real
    # ngram runs: ~100-200 us per request per step after attention is priced at the multi-token query length).
    # Added to the step's GPU time, so async scheduling does not hide it; not used when replaying logged durations.
    spec_ns_per_request: int = 0
    pending_drafts: object = None
    drafter: Any = None  # EAGLE / MTP: augur.sim.draft.DraftCost, the drafter's time added to each step
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
    step_time_scale: float = 1.0  # multiplies the cost model's GPU time, not the CPU time
    # KV offload to CPU (augur.sim.offload.SimOffloadWorker): set by the
    # coordinator before each step, the virtual time of this execute_model
    # call and when the GPU work queued before it is done.
    offload: Any = None
    # Contention (--contention on): this engine's view of the transfer fabric (augur.sim.fabric.Net), and
    # how much a step is slowed by the PCIe traffic beside it: t * (1 + k * rate / hbm_bw), the rate of the
    # GPU's PCIe flows sampled when the step starts; k is hbm_interference for a decode-only step and
    # hbm_interference_prefill for one with prefill tokens (compute-bound work feels the copies less).
    net: Any = None
    hbm_interference: float = 0.0
    hbm_interference_prefill: float = 0.0
    hbm_bw_gbps: float = 0.0

    def interference_factor(self, shape: StepShape | None = None) -> float:
        """1 + k * (bytes per ns now moving over this GPU's PCIe links) / HBM bandwidth for a step of ``shape``;
        1 without contention. Called by: Executor.execute_model, Coordinator._scaled."""
        k = self.hbm_interference_prefill if shape is not None and shape.num_prefill else self.hbm_interference
        if not k or self.net is None:
            return 1.0
        return 1.0 + k * self.net.interference_rate() / self.hbm_bw_gbps
    now_ns: int = 0
    gpu_free_ns: int = 0
    last_offload_wait_ns: int = 0  # the step's wait for flushed stores / sync loads (DP rounds add it to the CPU term)
    last_kv_report: bool = False  # this step's output carries a KV connector report
    # Tiered offload (augur.sim.tiers): the engine's secondary tiers, and the
    # virtual time of the scheduler-side work running now (the step's start
    # while it schedules; once execute_model has run, when vLLM applies the
    # step's outputs, which is when cascades to the tiers start).
    tiers: list = field(default_factory=list)
    tier_clock_ns: int = 0


@dataclass
class _ReqState:
    num_prompt: int
    num_output: int = 0
    context: Context | None = None  # ngram, token-faithful: prompt + emitted ids
    drafts: list[int] = field(default_factory=list)  # ngram, token-faithful: proposed after the last step
    blocks: list[int] = field(default_factory=list)  # KV block ids of the first cache group (a cost model that prices sharing)


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
        q = 1 + ctx.spec_k
        try:
            self.cudagraph.initialize_cudagraph_keys(mode, uniform_decode_query_len=q)
        except AssertionError:
            # A query length that does not divide the capture sizes (k = 2, 4, 5 ...): vLLM's v1 runner rounds the
            # sizes up to multiples of it (CompilationConfig.adjust_cudagraph_sizes_for_spec_decode).
            vllm_config.compilation_config.adjust_cudagraph_sizes_for_spec_decode(q)
            self.cudagraph = CudagraphDispatcher(vllm_config)
            self.cudagraph.initialize_cudagraph_keys(mode, uniform_decode_query_len=q)
        self._step = 0
        sc = vllm_config.speculative_config
        faithful = sc is not None and sc.method == "ngram" and not ctx.spec_rate_only
        self.ngram = NgramDrafter(vllm_config) if faithful else None

    def _track(self, rid: str, st: _ReqState, ids) -> None:
        """Token-faithful ngram: follow the request's real context (``ids``: its prompt and the outputs so far)."""
        if self.ngram is not None and rid in self.ctx.recorded:
            st.context = Context(ids, spare=len(self.ctx.output_ids[rid]) - st.num_output + self.ngram.k)

    # ---- start-up -------------------------------------------------------
    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        layers = layer_kv_specs(self.vllm_config)
        self.ctx.kv_block_bytes = sum(spec.page_size_bytes for spec in layers.values())
        # EAGLE / MTP layers take KV next to the target's.
        draft = draft_kv_spec(self.vllm_config, layers)
        if draft is not None:
            draft_spec, n_draft = draft
            self.ctx.kv_block_bytes += draft_spec.page_size_bytes * n_draft
            layers.update({f"draft.layers.{i}.self_attn.attn": draft_spec for i in range(n_draft)})
        return layers

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

    def _offload_output(self, meta, forward_ready: int, cpu_ns: int, gpu_ns: int | None,
                        kvo: KVConnectorOutput | None, slot: int | None) -> KVConnectorOutput | None:
        """Post-forward offload hooks: loads queue behind this forward, then the
        worker polls (as vLLM, right after launching the forward). kvo is what
        the NIXL side already reports this step (PD + offload), merged with
        the offload worker's report; slot is the offload connector's place in a
        MultiConnector's metadata (None when it is the only connector).
        gpu_ns None: a DP round, priced later (see SimOffloadWorker.post_forward)."""
        ctx = self.ctx
        if gpu_ns is None:
            forward_end = None
        else:
            launch = max(ctx.now_ns + cpu_ns, ctx.gpu_free_ns) if gpu_ns else max(forward_ready, ctx.gpu_free_ns)
            forward_end = launch + gpu_ns
        finished_recving, worker_meta = ctx.offload.post_forward(meta, forward_end_ns=forward_end,
                                                                 poll_ns=forward_ready)
        ctx.last_delivered = ctx.last_delivered or bool(finished_recving)
        if not finished_recving and worker_meta is None:
            return kvo
        if slot is not None and worker_meta is not None:
            # MultiConnector hands each child its own slot of the worker metadata.
            width = self._n_children
            worker_meta = MultiKVConnectorWorkerMetadata(
                metadata=tuple(worker_meta if i == slot else None for i in range(width)))
        recv = set(finished_recving) | (set(kvo.finished_recving) if kvo is not None and kvo.finished_recving else set())
        return KVConnectorOutput(finished_sending=kvo.finished_sending if kvo is not None else None,
                                 finished_recving=recv or None, kv_connector_worker_meta=worker_meta)

    @staticmethod
    def _split_connector_meta(meta):
        """(offload metadata or None, its slot in a MultiConnector's metadata or None, the metadata
        of every connector). A single connector's metadata is not wrapped."""
        if isinstance(meta, MultiKVConnectorMetadata):
            kids = list(meta.metadata)
            for i, m in enumerate(kids):
                if isinstance(m, OffloadingConnectorMetadata):
                    return m, i, kids
            return None, None, kids
        return (meta if isinstance(meta, OffloadingConnectorMetadata) else None), None, [meta]

    def take_draft_token_ids(self):
        out, self.ctx.pending_drafts = self.ctx.pending_drafts, None
        return out

    # ---- steps -----------------------------------------------------------
    def _shared_kv(self, decodes) -> dict[str, int]:
        """For the decodes of a step, ``(request id, tokens of computed KV)``: per request, the tokens of its KV that
        sit in blocks an earlier decode of the step also holds (the total does not depend on the order)."""
        size = self.vllm_config.cache_config.block_size
        seen: set[int] = set()
        out: dict[str, int] = {}
        for rid, before in decodes:
            blocks = set(self.reqs[rid].blocks[:before // size])  # only full blocks are ever shared
            if blocks:
                out[rid] = len(blocks & seen) * size
                seen |= blocks
        return out

    def execute_model(self, so) -> ModelRunnerOutput:
        for rid in so.finished_req_ids:
            self.reqs.pop(rid, None)
        if so.preempted_req_ids:
            for rid in so.preempted_req_ids:
                self.reqs.pop(rid, None)

        # KV connector: requests whose KV the worker starts pulling this step.
        self.ctx.new_recvs = []
        meta = so.kv_connector_metadata
        offload_meta, offload_slot, kids = self._split_connector_meta(meta)
        self._n_children = len(kids)
        for km in kids:
            for rid, rm in (getattr(km, "reqs_to_recv", None) or {}).items():
                n_local = sum(len(g) for g in rm.local_block_ids)
                n_done = sum(rm.local_num_computed_blocks) if rm.local_num_computed_blocks else 0
                r = rm.remote
                self.ctx.new_recvs.append((rid, max(n_local - n_done, 0),
                                           r.engine_id if r else None, r.request_id if r else None))
        kvo = None
        forward_ready = self.ctx.now_ns
        if offload_meta is not None:
            forward_ready = self.ctx.offload.pre_forward(offload_meta, self.ctx.now_ns, self.ctx.gpu_free_ns)
        self.ctx.last_delivered = bool(self.ctx.deliver_recving or self.ctx.deliver_sending)
        if self.ctx.last_delivered:
            kvo = KVConnectorOutput(finished_sending=set(self.ctx.deliver_sending) or None,
                                    finished_recving=set(self.ctx.deliver_recving) or None)
            self.ctx.deliver_recving.clear()
            self.ctx.deliver_sending.clear()

        work: list[tuple[str, int, int]] = []  # (req_id, computed_before, num_new)
        track = getattr(self.ctx.cost, "prices_sharing", False)  # the cost model prices KV blocks several requests read
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
            self.reqs[nr.req_id] = st = _ReqState(num_prompt=n_prompt, num_output=n_known - n_prompt)
            self._track(nr.req_id, st, nr.prefill_token_ids or nr.prompt_token_ids)
            if track and nr.block_ids:
                st.blocks = list(nr.block_ids[0])
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
                if all_ids is not None:
                    self._track(rid, st, all_ids)
            if track and cr.new_block_ids and cr.new_block_ids[i]:  # a resumed request's list is its whole table
                st.blocks.extend(cr.new_block_ids[i][0])
            work.append((rid, cr.num_computed_tokens[i], so.num_scheduled_tokens[rid]))

        req_ids: list[str] = []
        sampled: list[list[int]] = []
        seqs: list[SeqWork] = []
        drafts_next: dict[str, list[int]] = {}
        proposing: list[tuple[str, _ReqState]] = []  # token-faithful ngram: requests that emit this step
        spec = so.scheduled_spec_decode_tokens or {}
        self._step += 1
        shared = self._shared_kv([(rid, before) for rid, before, n in work
                                  if n == 1 or (spec.get(rid) and n == 1 + len(spec[rid]))]) if track else {}
        for rid, before, num_new in work:
            st = self.reqs[rid]
            known = st.num_prompt + st.num_output
            n_draft = len(spec.get(rid, ()))
            # Decode vs prefill by query length, as vLLM's attention backends
            # split a batch: one token is a decode (a one-token last prefill
            # chunk runs as one), and so is 1 + k under speculative decoding,
            # where k draft tokens are verified alongside the real one.
            is_decode = num_new == 1 or (n_draft > 0 and num_new == 1 + n_draft)
            seqs.append(SeqWork(num_new=num_new, num_computed_before=before, is_decode=is_decode,
                                shared_kv=shared.get(rid, 0)))
            req_ids.append(rid)
            if before + num_new >= known + n_draft:
                rec = self.ctx.output_ids.get(rid)
                # Speculative decoding: each draft is accepted with the given
                # rate until the first rejection, then one more token is
                # sampled. Content decides acceptance on a GPU; augur cannot
                # see content, so the rate is a parameter.
                accepted = 0
                replay = self._replay_step()
                faithful = st.context is not None and replay is None
                if replay is not None:
                    # Replay: the logged run's acceptance (content-dependent).
                    accepted = max(replay[0].get(call_id(rid), 1) - 1, 0)
                elif faithful:
                    # The drafts the runner's drafter made from the real context (the scheduler may hold placeholders,
                    # and fewer than were made) are accepted up to the first one that is not the next recorded id.
                    while (accepted < min(n_draft, len(st.drafts)) and st.num_output + accepted < len(rec)
                           and st.drafts[accepted] == rec[st.num_output + accepted]):
                        accepted += 1
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
                if faithful:
                    st.context.append(toks)
                    st.drafts = []
                    if st.num_output < len(rec):
                        proposing.append((rid, st))
                elif self.ctx.spec_k:
                    # The drafter proposes the request's next recorded tokens,
                    # so accepted drafts are the right tokens. It does not know
                    # max_tokens: near the end it still proposes k (vLLM stops
                    # the request when it reaches max_tokens). Replay: as many
                    # drafts as the logged run scheduled next step (an ngram
                    # drafter proposes none when the prompt has no match). A
                    # request missing from the next logged step (preempted,
                    # because its drafts needed KV the cache lacked) keeps k.
                    n_next = self.ctx.spec_k
                    if replay is not None and call_id(rid) in replay[1]:
                        n_next = max(replay[1][call_id(rid)] - 1, 0)
                    nxt = list((rec or ())[st.num_output:st.num_output + n_next])
                    nxt += [SYNTHETIC_TOKEN] * (n_next - len(nxt))
                    if nxt:
                        drafts_next[rid] = nxt
            else:
                sampled.append([])
        if proposing:
            for (rid, st), drafts in zip(proposing, self.ngram.propose([st.context for _, st in proposing])):
                st.drafts = drafts_next[rid] = drafts
        if self.ctx.spec_k:
            self.ctx.pending_drafts = DraftTokenIds(req_ids=list(drafts_next),
                                                    draft_token_ids=list(drafts_next.values()))

        self.ctx.last_offload_wait_ns = forward_ready - self.ctx.now_ns
        if self.ctx.defer_cost:
            self.ctx.last_shape = StepShape(seqs=tuple(seqs))
            self.ctx.last_step_ns = 0
            # DP: the round's GPU time comes later; the CPU time is known now.
            self.ctx.last_cpu_ns = (self.ctx.cpu_fixed_ns + self.ctx.cpu_per_request_ns * len(seqs)) if seqs else 0
            if offload_meta is not None:
                # A DP round is priced after every rank has stepped: the rank's
                # loads are committed then (sync), or start at the launch (async).
                kvo = self._offload_output(offload_meta, forward_ready, 0, None if not self.ctx.async_cpu else 0,
                                           kvo, offload_slot)
        else:
            num_tokens = sum(x.num_new for x in seqs)
            q = 1 + self.ctx.spec_k
            uniform_decode = bool(seqs) and all(x.num_new == q for x in seqs)
            _, desc = self.cudagraph.dispatch(num_tokens=num_tokens, uniform_decode=uniform_decode)
            shape = StepShape(seqs=tuple(seqs), padded_tokens=desc.num_tokens)
            self.ctx.last_shape = shape
            gpu = self.ctx.cost.step_ns(shape) if seqs else 0
            if seqs and self.ctx.drafter is not None:
                gpu += self.ctx.drafter.step_ns(shape, len(drafts_next))
            if self.ctx.step_time_scale != 1.0:
                gpu = round(gpu * self.ctx.step_time_scale)
            if self.ctx.spec_ns_per_request and not hasattr(self.ctx.cost, "durations"):
                gpu += self.ctx.spec_ns_per_request * sum(1 for x in seqs if x.is_decode and x.num_new > 1)
            if self.ctx.net is not None and not hasattr(self.ctx.cost, "durations"):
                gpu = round(gpu * self.ctx.interference_factor(shape))
            cpu = (self.ctx.cpu_fixed_ns + self.ctx.cpu_per_request_ns * len(seqs)) if seqs else 0
            if not hasattr(self.ctx.cost, "durations"):  # a replayed duration already holds its stalls
                cpu += forward_ready - self.ctx.now_ns  # offload waits (flushed stores, sync loads) hold the step
            self.ctx.last_cpu_ns = cpu
            self.ctx.last_step_ns = gpu + cpu if not self.ctx.async_cpu else gpu
            if offload_meta is not None:
                kvo = self._offload_output(offload_meta, forward_ready, cpu, gpu, kvo, offload_slot)
        self.ctx.last_kv_report = kvo is not None
        # Sync: this step's outputs land when it ends. Async: the update that
        # follows is for the batch launched before, done when the GPU frees.
        self.ctx.tier_clock_ns = (self.ctx.gpu_free_ns if self.ctx.async_cpu
                                  else self.ctx.now_ns + self.ctx.last_step_ns)
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={r: i for i, r in enumerate(req_ids)},
            sampled_token_ids=sampled,
            kv_connector_output=kvo,
        )


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
