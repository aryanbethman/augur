"""The coordinator: one virtual clock driving real vLLM engines.

Each ``SimEngine`` wraps an unmodified vLLM ``EngineCore`` whose executor is
our ``SimExecutor``. The coordinator owns the event queue. At each instant it
first applies every event due then, in a fixed order (arrivals, then step
completions), and only then lets idle engines with work run their next step.

One ``EngineCore.step()`` runs vLLM's ``schedule()``, our executor and vLLM's
``update_from_output()`` back to back, in wall time. On the virtual clock the
step starts now and its outputs land ``step_ns`` later, when the coordinator
records token times and releases the next call of any session that finished.
Nothing is scheduled on that engine in between, as in synchronous vLLM.
"""

from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass, field

from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore

from augur.cost import CostModel
from augur.executor import SimContext, SimHardware, make_sim_executor
from augur.metrics import CallRecord, Recorder
from augur.router import Router
from augur.workload import Call, Session, Workload

# Event priorities at equal times (design doc: arrivals, transfers, step completions).
# CAUSED_ARRIVAL is a call released by a completion at this same instant (a
# session's next call after a zero tool wait). It cannot reach the engine
# before the engine has started its next step at that instant: in vLLM the
# client sees the finished output only after the step, and the engine loop
# schedules again right away. So it is applied after steps start.
ARRIVAL, TRANSFER_DONE, STEP_DONE, CAUSED_ARRIVAL = 0, 1, 2, 3


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
    kv_role: str | None = None  # "kv_producer" (prefill) / "kv_consumer" (decode) for PD


@dataclass
class PDLink:
    """KV pull from a prefill to a decode engine: latency + bytes / bandwidth.

    No contention between concurrent pulls yet (ASTRA-Sim congestion-aware
    mode is the planned upgrade). Defaults are the fork's single-node link."""

    bw_gbps: float = 16.0
    latency_ns: int = 20_000

    def transfer_ns(self, nbytes: int) -> int:
        return self.latency_ns + int(nbytes / self.bw_gbps)

class SimEngine:
    def __init__(self, engine_id: int, spec: EngineSpec, cost: CostModel):
        self.engine_id = engine_id
        self.spec = spec
        self.ctx = SimContext(cost=cost, hardware=SimHardware(kv_cache_bytes=spec.kv_cache_bytes))
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
            async_scheduling=False,
            distributed_executor_backend=make_sim_executor(self.ctx),
            tensor_parallel_size=spec.tensor_parallel_size,
            kv_transfer_config=self._kv_transfer_config(engine_id, spec),
        )
        self.vllm_config = args.create_engine_config()
        self.core = EngineCore(self.vllm_config, executor_class=self.vllm_config.parallel_config.distributed_executor_backend,
                               log_stats=False)
        self.max_model_len = self.vllm_config.model_config.max_model_len
        self.busy = False
        self.sched_cpu_s = 0.0
        self.steps = 0
        # Only for --lss-idle-quantum (diagnostic): when this engine went idle,
        # and the wake-up already queued.
        self.idle_since: int | None = None
        self.wake_at: int | None = None
        self.step_log: list[tuple] | None = None  # (start_ns, dur_ns, total, prefill_tokens, n_decode, n_prefill)

    @staticmethod
    def _kv_transfer_config(engine_id: int, spec: EngineSpec):
        if spec.kv_role is None:
            return None
        from vllm.config import KVTransferConfig
        # vLLM's real NIXL connector; only its scheduler half is built (our
        # executor stands in for the worker half and never loads NIXL).
        return KVTransferConfig(kv_connector="SimNixlConnector", kv_connector_module_path="augur.nixl_sim",
                                kv_role=spec.kv_role, engine_id=f"augur-{engine_id}")

    @property
    def vllm_engine_id(self) -> str:
        return f"augur-{self.engine_id}"

    def load(self) -> int:
        s = self.core.scheduler
        return len(s.waiting) + len(s.running)

    def add(self, call: Call, now_ns: int, request_id: str | None = None, max_tokens: int | None = None,
            kv_transfer_params: dict | None = None) -> None:
        n = call.output_toks if max_tokens is None else max_tokens
        sp = SamplingParams(max_tokens=n, ignore_eos=True, min_tokens=n,
                            output_kind=RequestOutputKind.DELTA, detokenize=False,
                            extra_args={"kv_transfer_params": kv_transfer_params} if kv_transfer_params else None)
        rid = request_id or call.call_id
        req = EngineCoreRequest(
            request_id=rid, prompt_token_ids=call.prompt_ids, mm_features=None,
            sampling_params=sp, pooling_params=None, arrival_time=now_ns / 1e9,
            lora_request=None, cache_salt=None, data_parallel_rank=None,
        )
        self.ctx.output_ids[rid] = call.output_ids
        r, wave = self.core.preprocess_add_request(req)
        self.core.add_request(r, wave)

    def step(self):
        t0 = time.perf_counter()
        outputs, executed = self.core.step()
        self.sched_cpu_s += time.perf_counter() - t0
        self.steps += 1
        return outputs, executed, self.ctx.last_step_ns


@dataclass(order=True)
class _Event:
    t: int
    prio: int
    seq: int
    kind: str = field(compare=False)
    payload: object = field(compare=False)


class Coordinator:
    def __init__(self, engines: list[SimEngine], router: Router, workload: Workload,
                 lss_idle_quantum_ns: int = 0, decode_router: Router | None = None,
                 pd_link: PDLink | None = None):
        self.engines = engines
        # PD disaggregation: the coordinator plays vLLM's PD proxy. A call goes
        # to a prefill engine with max_tokens=1 and do_remote_decode; the
        # kv_transfer_params that engine's NIXL scheduler returns go with the
        # full request to a decode engine, whose NIXL scheduler parks it until
        # its KV has been pulled. Colocated when no engine has a kv_role.
        self.prefill_engines = [e for e in engines if e.spec.kv_role == "kv_producer"]
        self.decode_engines = [e for e in engines if e.spec.kv_role == "kv_consumer"]
        self.pd = bool(self.prefill_engines)
        if self.pd and not self.decode_engines:
            raise ValueError("PD needs at least one decode engine")
        self.entry_engines = self.prefill_engines if self.pd else engines
        self.decode_router = decode_router
        self.pd_link = pd_link or PDLink()
        self._by_vllm_id = {e.vllm_engine_id: e for e in engines}
        self._xfers_started: set[str] = set()
        # LLMServingSim loop compatibility, for comparisons only (off by
        # default; none of this is vLLM behaviour):
        # - A fully idle cluster advances in fixed quanta (1 ms): a session call
        #   released while the engine is idle starts at the first quantum
        #   boundary after the engine went idle (at least one quantum later at
        #   TP1; at TP>1 a call released at the idle instant is not delayed).
        #   New workload entries jump the clock to their exact arrival.
        # - At TP>1 a zero-wait session call joins the step starting at its
        #   release instant.
        self.lss_idle_quantum_ns = lss_idle_quantum_ns
        self.router = router
        self.workload = workload
        self.recorder = Recorder()
        self.now = 0
        self._q: list[_Event] = []
        self._seq = itertools.count()
        self._num_ids = itertools.count()
        self._records: dict[str, CallRecord] = {}
        self._calls: dict[str, Call] = {}
        self._call_engine: dict[str, SimEngine] = {}
        self._sessions: dict[str, Session] = {}
        self._entries = workload.entries()
        self._last_entry_ns = -1
        self.wall_s = 0.0

    def _push(self, t: int, prio: int, kind: str, payload) -> None:
        heapq.heappush(self._q, _Event(t, prio, next(self._seq), kind, payload))

    def _pull_entry(self) -> None:
        """Stream the next top-level entry (flat request or session) from the file."""
        e = next(self._entries, None)
        if e is None:
            return
        t = e.arrival_ns
        if t < self._last_entry_ns:
            raise ValueError(f"workload not sorted by arrival: {t} after {self._last_entry_ns}")
        self._last_entry_ns = t
        self._push(t, ARRIVAL, "entry", e)

    @staticmethod
    def _cid(request_id: str) -> str:
        return request_id.split("@", 1)[0]  # PD request ids are <call>@p / <call>@d

    def _pick(self, router: Router, call: Call, engines: list[SimEngine]) -> SimEngine:
        eid = router.route(call, engines)
        return next(e for e in engines if e.engine_id == eid)

    def _release(self, call: Call, exact: bool = True) -> None:
        eng = self._pick(self.router, call, self.entry_engines)
        if exact and not eng.busy:
            eng.idle_since = None
        if len(call.prompt_ids) + call.output_toks > eng.max_model_len:
            raise ValueError(f"{call.call_id}: {len(call.prompt_ids)}+{call.output_toks} tokens exceed "
                             f"max_model_len {eng.max_model_len}")
        rec = CallRecord(num_id=next(self._num_ids), call_id=call.call_id, instance_id=eng.engine_id,
                         model=eng.spec.model_name, input_toks=len(call.prompt_ids), output_toks=call.output_toks,
                         arrival_ns=self.now, session_id=call.session_id, index=call.index)
        self._records[call.call_id] = rec
        self._calls[call.call_id] = call
        self._call_engine[call.call_id] = eng
        if self.pd:
            eng.add(call, self.now, request_id=call.call_id + "@p", max_tokens=1,
                    kv_transfer_params={"do_remote_decode": True})
        else:
            eng.add(call, self.now)

    def _on_arrival(self, ev: _Event) -> None:
        p = ev.payload
        if ev.kind == "entry":
            self._pull_entry()
            if isinstance(p, Session):
                self._sessions[p.session_id] = p
                self._release(self.workload.next_call(p, self.now))
            else:
                self._release(p)
        elif ev.kind == "session_call":
            s = self._sessions[p]
            self._release(self.workload.next_call(s, self.now), exact=False)
        elif ev.kind == "wake":
            p.wake_at = None
        elif ev.kind == "decode_dispatch":
            cid, params = p
            call = self._calls[cid]
            d = self._pick(self.decode_router, call, self.decode_engines)
            self._call_engine[cid] = d
            self._records[cid].instance_id = d.engine_id
            d.add(call, self.now, request_id=cid + "@d", kv_transfer_params=params)

    def _on_transfer_done(self, ev: _Event) -> None:
        d, d_rid, p, p_rid = ev.payload
        d.ctx.deliver_recving.add(d_rid)
        if p is not None and p_rid is not None:
            p.ctx.deliver_sending.add(p_rid)

    def _on_step_done(self, ev: _Event) -> None:
        eng, outputs = ev.payload
        eng.busy = False
        for eco in outputs.values():
            for o in eco.outputs:
                cid = self._cid(o.request_id)
                rec = self._records[cid]
                if o.request_id.endswith("@p"):
                    # The prefill engine's one token goes back to the proxy, not
                    # the client; the client's tokens come from the decode engine.
                    if o.finish_reason is not None:
                        eng.ctx.output_ids.pop(o.request_id, None)
                        rec.prefill_done_ns = self.now
                        if not o.kv_transfer_params:
                            raise RuntimeError(f"{o.request_id}: prefill engine returned no kv_transfer_params")
                        # The proxy forwards to decode once it has this response.
                        self._push(self.now, CAUSED_ARRIVAL, "decode_dispatch", (cid, o.kv_transfer_params))
                    continue
                rec.token_times.extend([self.now] * len(o.new_token_ids))
                if o.finish_reason is not None:
                    self._finish(cid, rec, eng)
        if eng.core.scheduler.get_num_unfinished_requests() == 0 and eng.idle_since is None:
            # Idle from the moment its last step completes, before any session
            # call released at this same instant (matters only for the quantum).
            eng.idle_since = self.now

    def _finish(self, call_id: str, rec: CallRecord, eng: SimEngine) -> None:
        rec.end_ns = self.now
        self.recorder.done.append(rec)
        call = self._calls.pop(call_id)
        del self._records[call_id]
        self._call_engine.pop(call_id, None)
        eng.ctx.output_ids.pop(call_id + "@d" if self.pd else call_id, None)
        if call.session_id is not None:
            s = self._sessions[call.session_id]
            s.calls_done += 1
            if s.next_index < len(s.sub_requests):
                prio = CAUSED_ARRIVAL if call.tool_wait_ns == 0 else ARRIVAL
                if prio == CAUSED_ARRIVAL and self.lss_idle_quantum_ns and eng.spec.tensor_parallel_size > 1:
                    # LLMServingSim compat only: at TP>1 the other ranks' reports
                    # at this instant route and schedule again, so the call joins
                    # the next step. Not possible in vLLM.
                    prio = ARRIVAL
                self._push(self.now + call.tool_wait_ns, prio, "session_call", s.session_id)
            else:
                s.finish_ns = self.now
                del self._sessions[s.session_id]

    def _start_steps(self) -> None:
        q = self.lss_idle_quantum_ns
        for eng in self.engines:
            if eng.busy:
                continue
            pending_kv = bool(eng.ctx.deliver_recving or eng.ctx.deliver_sending)
            if not eng.core.scheduler.has_requests() and not pending_kv:
                if eng.idle_since is None:
                    eng.idle_since = self.now
                continue
            if q and eng.idle_since is not None:
                # At TP>1 the other ranks' reports at the idle instant reopen
                # polling, so a call released right then is not delayed.
                k_min = 1 if eng.spec.tensor_parallel_size == 1 else 0
                k = max(k_min, -(-(self.now - eng.idle_since) // q))
                start = eng.idle_since + k * q
                if start > self.now:
                    if eng.wake_at != start:
                        eng.wake_at = start
                        self._push(start, ARRIVAL, "wake", eng)
                    continue
            eng.idle_since = None
            for _ in range(4):
                outputs, executed, step_ns = eng.step()
                self._start_transfers(eng)
                if executed or step_ns or not eng.ctx.last_delivered:
                    break
                # An empty step that only delivered finished KV transfers: vLLM
                # promotes those requests at the next schedule, right away.
                self._on_step_done(_Event(self.now, STEP_DONE, 0, "step_done", (eng, outputs)))
            for rid, hit in eng.ctx.last_new_req_ids.items():
                rec = self._records.get(self._cid(rid))
                if rec is not None and rec.first_sched_ns is None:
                    rec.first_sched_ns = self.now
                    rec.cached_tokens = hit
            for rid in eng.ctx.last_preempted_req_ids:
                rec = self._records.get(self._cid(rid))
                if rec is not None:
                    rec.num_preemptions += 1
            if not executed and step_ns == 0:
                # Nothing could be scheduled (e.g. KV cache full with nothing to preempt).
                # Apply any outputs now; the engine waits for the next event.
                self._on_step_done(_Event(self.now, STEP_DONE, 0, "step_done", (eng, outputs)))
                continue
            if eng.step_log is not None and eng.ctx.last_shape is not None:
                sh = eng.ctx.last_shape
                eng.step_log.append((self.now, step_ns, sh.total_tokens,
                                     sum(x.num_new for x in sh.seqs if not x.is_decode), sh.num_decode, sh.num_prefill))
            eng.busy = True
            self._push(self.now + step_ns, STEP_DONE, "step_done", (eng, outputs))

    def _start_transfers(self, eng: SimEngine) -> None:
        """The decode worker starts pulling KV in the step its scheduler parks
        the request (WAITING_FOR_REMOTE_KVS); finished_recving and the prefill
        side's finished_sending are reported at each engine's next step after
        the pull completes."""
        for rid, blocks, remote_engine, remote_rid in eng.ctx.new_recvs:
            if rid in self._xfers_started:
                continue
            self._xfers_started.add(rid)
            ns = self.pd_link.transfer_ns(blocks * eng.ctx.kv_block_bytes)
            self._push(self.now + ns, TRANSFER_DONE, "kv_pull",
                       (eng, rid, self._by_vllm_id.get(remote_engine), remote_rid))

    def run(self) -> Recorder:
        t0 = time.perf_counter()
        self._pull_entry()
        while self._q:
            self.now = self._q[0].t
            while self._q and self._q[0].t == self.now and self._q[0].prio < CAUSED_ARRIVAL:
                ev = heapq.heappop(self._q)
                if ev.prio == ARRIVAL:
                    self._on_arrival(ev)
                elif ev.prio == TRANSFER_DONE:
                    self._on_transfer_done(ev)
                else:
                    self._on_step_done(ev)
            self._start_steps()
            caused = False
            while self._q and self._q[0].t == self.now and self._q[0].prio == CAUSED_ARRIVAL:
                self._on_arrival(heapq.heappop(self._q))
                caused = True
            if caused:
                self._start_steps()  # an idle engine picks the call up at once
        if self._records:
            raise RuntimeError(f"{len(self._records)} calls never finished (e.g. {next(iter(self._records))})")
        self.wall_s = time.perf_counter() - t0
        return self.recorder
