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

import torch

from vllm.v1.executor.abstract import Executor
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.outputs import ModelRunnerOutput

from augur.cost import CostModel, SeqWork, StepShape

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

    # ---- steps -----------------------------------------------------------
    def execute_model(self, so) -> ModelRunnerOutput:
        for rid in so.finished_req_ids:
            self.reqs.pop(rid, None)
        if so.preempted_req_ids:
            for rid in so.preempted_req_ids:
                self.reqs.pop(rid, None)

        work: list[tuple[str, int, int]] = []  # (req_id, computed_before, num_new)
        # A new request's computed tokens at its first schedule are its prefix-cache hit.
        self.ctx.last_new_req_ids = {nr.req_id: nr.num_computed_tokens for nr in so.scheduled_new_reqs}
        self.ctx.last_preempted_req_ids = list(so.preempted_req_ids or ())
        for nr in so.scheduled_new_reqs:
            self.reqs[nr.req_id] = _ReqState(num_prompt=len(nr.prompt_token_ids or ()))
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
        for rid, before, num_new in work:
            st = self.reqs[rid]
            known = st.num_prompt + st.num_output
            # A decode step computes the one token sampled last step.
            is_decode = num_new == 1 and before == known - 1 and st.num_output > 0
            seqs.append(SeqWork(num_new=num_new, num_computed_before=before, is_decode=is_decode))
            req_ids.append(rid)
            if before + num_new >= known:
                rec = self.ctx.output_ids.get(rid)
                tok = rec[st.num_output] if rec and st.num_output < len(rec) else SYNTHETIC_TOKEN
                st.num_output += 1
                sampled.append([tok])
            else:
                sampled.append([])

        shape = StepShape(seqs=tuple(seqs))
        self.ctx.last_shape = shape
        self.ctx.last_step_ns = self.ctx.cost.step_ns(shape) if seqs else 0
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={r: i for i, r in enumerate(req_ids)},
            sampled_token_ids=sampled,
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

        def execute_model(self, scheduler_output, non_block=False):
            out = self.worker.execute_model(scheduler_output)
            if non_block:
                fut: Future = Future()
                fut.set_result(out)
                return fut
            return out

        def sample_tokens(self, grammar_output, non_block=False):
            raise AssertionError("SimExecutor samples inside execute_model")

        def compile_or_warm_up_model(self) -> None:
            return None

        def check_health(self) -> None:
            return None

        def shutdown(self) -> None:
            return None

        @classmethod
        def supports_async_scheduling(cls) -> bool:
            return False

    return SimExecutor
