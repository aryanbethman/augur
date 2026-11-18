"""SimEngine: one unmodified vLLM ``EngineCore`` whose executor is augur's
``SimExecutor`` (no GPU, no model weights). The coordinator drives it."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

from vllm.engine import arg_utils
from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore

from augur.cost import CostModel
from augur.sim.executor import SimContext, SimHardware, make_sim_executor
from augur.sim.offload import OffloadLink, SimOffloadWorker
from augur.sim.tiers import GIB, TierSpec, register_clock
from augur.trace.workload import Call

# The engine inside a synchronous step (for the step log's clock).
_stepping: list = [None]


@dataclass
class EngineSpec:
    model_dir: str
    model_name: str
    kv_cache_bytes: int
    max_model_len: int | None = None
    max_num_batched_tokens: int | None = None
    max_num_seqs: int | None = None
    block_size: int = 16
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    tensor_parallel_size: int = 1
    dtype: str = "bfloat16"
    enforce_eager: bool = False  # vLLM's default: CUDA graphs on, so steps are padded
    scheduler_cls: str | None = None  # e.g. augur.verify.hooks.LoggingScheduler
    num_gpu_blocks: int | None = None  # vLLM's num_gpu_blocks_override (match a real run's KV cache)
    kv_role: str | None = None  # "kv_producer" (prefill) / "kv_consumer" (decode) for PD
    data_parallel_size: int = 1  # >1: this engine is one rank of a DP group (lockstep rounds)
    data_parallel_rank: int = 0
    enable_expert_parallel: bool = False
    async_scheduling: bool = False  # vLLM 0.33's default is True
    speculative: dict | None = None  # vLLM speculative_config, e.g. {"method": "ngram", "num_speculative_tokens": 3}
    spec_accept_rate: float = 0.0
    kv_offload_gib: float | None = None  # vLLM --kv-offloading-size: CPU tier for the KV cache (native connector)
    kv_offload_link: OffloadLink | None = None  # GPU <-> CPU transfer cost (required with kv_offload_gib)
    kv_offload_tiers: tuple[TierSpec, ...] = ()  # secondary tiers behind the CPU one, in order (vLLM tiering)
    hbm_gb: float | None = None  # simulated GPU memory: unset batch limits are `vllm serve`'s for this GPU
    hardware: str = ""  # simulated GPU's name, which vLLM's batch defaults also look at


def _simulated_gpu_batch_defaults(spec: EngineSpec):
    """vLLM's own batch-limit table, read for the simulated GPU instead of the
    machine augur runs on. Called by: SimEngine.__init__."""
    name = spec.hardware.lower()
    gpu = SimpleNamespace(get_device_total_memory=lambda: int(spec.hbm_gb * GIB), get_device_name=lambda: name,
                          is_device_capability=lambda capability: "gb10" in name,
                          is_tpu=lambda: False, is_cpu=lambda: False)

    def defaults(world_size: int):
        with patch.object(arg_utils, "current_platform", gpu):
            return EngineArgs.get_batch_defaults(world_size)
    return defaults


class SimEngine:
    def __init__(self, engine_id: int, spec: EngineSpec, cost: CostModel):
        self.engine_id = engine_id
        self.spec = spec
        self.ctx = SimContext(cost=cost, hardware=SimHardware(kv_cache_bytes=spec.kv_cache_bytes))
        # Secondary tiers are built inside vLLM from plain config: they find
        # this engine's clock by key.
        clock_key = f"augur-{engine_id}-{id(self.ctx)}"
        register_clock(clock_key, self.ctx)
        tiered = bool(spec.kv_offload_tiers)
        args = EngineArgs(
            model=spec.model_dir,
            served_model_name=spec.model_name,
            skip_tokenizer_init=True,
            enforce_eager=spec.enforce_eager,
            dtype=spec.dtype,
            max_model_len=spec.max_model_len,
            max_num_batched_tokens=spec.max_num_batched_tokens,
            max_num_seqs=spec.max_num_seqs,
            block_size=spec.block_size,
            enable_prefix_caching=spec.enable_prefix_caching,
            enable_chunked_prefill=spec.enable_chunked_prefill,
            async_scheduling=spec.async_scheduling,
            scheduler_cls=spec.scheduler_cls,
            num_gpu_blocks_override=spec.num_gpu_blocks,
            speculative_config=spec.speculative,
            distributed_executor_backend=make_sim_executor(self.ctx),
            tensor_parallel_size=spec.tensor_parallel_size,
            kv_transfer_config=self._kv_transfer_config(engine_id, spec, clock_key),
            data_parallel_size=spec.data_parallel_size,
            data_parallel_rank=spec.data_parallel_rank if spec.data_parallel_size > 1 else None,
            enable_expert_parallel=spec.enable_expert_parallel,
            # Tiered offload configures the connector itself (above), as vLLM's
            # --kv-offloading-size cannot name secondary tiers.
            kv_offloading_size=None if tiered else spec.kv_offload_gib,
            kv_offloading_backend="native",
        )
        self.ctx.defer_cost = spec.data_parallel_size > 1
        if spec.speculative:
            self.ctx.spec_k = int(spec.speculative["num_speculative_tokens"])
            self.ctx.spec_accept_rate = spec.spec_accept_rate
        usage_context = None
        if spec.hbm_gb is not None:
            args.get_batch_defaults = _simulated_gpu_batch_defaults(spec)
            usage_context = UsageContext.OPENAI_API_SERVER  # what `vllm serve` passes
        self.vllm_config = args.create_engine_config(usage_context=usage_context)
        self.core = EngineCore(self.vllm_config, executor_class=self.vllm_config.parallel_config.distributed_executor_backend,
                               log_stats=False)
        self.max_model_len = self.vllm_config.model_config.max_model_len
        if spec.kv_offload_gib:
            # The connector's scheduler half is vLLM's; this is its worker half.
            # kv_block_bytes was set while EngineCore sized the KV cache.
            self.ctx.offload = SimOffloadWorker(spec.kv_offload_link, self.ctx.kv_block_bytes,
                                                replay_done_ns=getattr(cost, "kv_done", None))
        self.busy = False
        self.admitting = 0  # calls routed here that are still in the API server (--frontend-latency-ms)
        self.xfer_wake_at: int | None = None  # offload: the poll already queued for a finishing transfer
        self.last_progress_ns = 0  # last step that scheduled tokens, or when work arrived at an idle engine
        self.async_sched = self.core.batch_queue is not None
        self.ctx.async_cpu = self.async_sched
        # DP + async: rounds this rank's queued batches belong to (FIFO, as
        # vLLM's batch queue), popped outputs held until that round ends here.
        self.dp_fifo: "deque[int]" = deque()
        self.dp_held: dict[int, dict] = {}
        self.dp_blocked_on: int | None = None
        self.dp_next_round = 0
        self.gpu_free_at = 0  # async: when the GPU stream finishes its queued batches
        self.batch_finish: "deque[int]" = deque()  # async: finish time per queued batch, FIFO
        self.batch_reports: "deque[bool]" = deque()  # async: per queued batch, its KV connector worker reported something
        self.sched_cpu_s = 0.0
        self.steps = 0
        # Only for --lss-idle-quantum (diagnostic): when this engine went idle,
        # and the wake-up already queued.
        self.idle_since: int | None = None
        self.wake_at: int | None = None
        self.step_log: list[tuple] | None = None  # (start_ns, dur_ns, total, prefill_tokens, n_decode, n_prefill)

    @staticmethod
    def _kv_transfer_config(engine_id: int, spec: EngineSpec, clock_key: str):
        from vllm.config import KVTransferConfig
        if spec.kv_offload_tiers:
            # vLLM's OffloadingConnector with its tiering spec (augur's variant:
            # no /dev/shm buffer) and augur's simulated secondary tiers.
            extra = {"cpu_bytes_to_use": int(spec.kv_offload_gib * GIB),
                     "spec_name": "SimTieringOffloadingSpec", "spec_module_path": "augur.sim.tiers",
                     "secondary_tiers": [t.tier_config(clock_key) for t in spec.kv_offload_tiers]}
            return KVTransferConfig(kv_connector="OffloadingConnector", kv_role="kv_both",
                                    kv_connector_extra_config=extra)
        if spec.kv_role is None:
            return None
        # vLLM's real NIXL connector; only its scheduler half is built (our
        # executor stands in for the worker half and never loads NIXL).
        return KVTransferConfig(kv_connector="SimNixlConnector", kv_connector_module_path="augur.sim.nixl",
                                kv_role=spec.kv_role, engine_id=f"augur-{engine_id}")

    @property
    def vllm_engine_id(self) -> str:
        return f"augur-{self.engine_id}"

    def load(self) -> int:
        s = self.core.scheduler
        return len(s.waiting) + len(s.running) + self.admitting

    def add(self, call: Call, now_ns: int, request_id: str | None = None, max_tokens: int | None = None,
            kv_transfer_params: dict | None = None) -> None:
        n = call.output_toks if max_tokens is None else max_tokens
        sp = SamplingParams(max_tokens=n, ignore_eos=True, min_tokens=n,
                            output_kind=RequestOutputKind.DELTA, detokenize=False,
                            extra_args={"kv_transfer_params": kv_transfer_params} if kv_transfer_params else None)
        rid = request_id or call.call_id
        if not self.core.scheduler.get_num_unfinished_requests():
            self.last_progress_ns = now_ns
        req = EngineCoreRequest(
            request_id=rid, prompt_token_ids=call.prompt_ids, mm_features=None,
            sampling_params=sp, pooling_params=None, arrival_time=now_ns / 1e9,
            lora_request=None, cache_salt=None, data_parallel_rank=None,
        )
        self.ctx.output_ids[rid] = call.output_ids
        r, wave = self.core.preprocess_add_request(req)
        self.core.add_request(r, wave)

    def step(self):
        _stepping[0] = self
        t0 = time.perf_counter()
        outputs, executed = self.core.step()
        self.core.post_step(executed)  # as vLLM's busy loop: hands draft tokens to the scheduler
        self.ctx.launches.clear()
        self.sched_cpu_s += time.perf_counter() - t0
        self.steps += 1
        return outputs, executed, self.ctx.last_step_ns
