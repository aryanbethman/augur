"""The coordinator: one virtual clock driving real vLLM engines.

Each ``SimEngine`` (augur.sim.engine) wraps an unmodified vLLM ``EngineCore`` whose executor is
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
from collections import deque
from dataclasses import dataclass, field

from augur.cost import adjust_gpu_ns
from augur.sim.engine import SimEngine, _stepping
from augur.sim.router import Router
from augur.trace.metrics import CallRecord, Recorder
from augur.trace.workload import Call, Session, Workload, strip_pd_suffix

_async_stepping: list = [None]  # engine inside step_with_batch_queue (for the step-log clock)


def install_step_log_clock(coord: "Coordinator") -> None:
    """Point augur.verify.hooks' clock at the virtual clock: a step's outputs are
    applied at its start in augur, but land at its end."""
    from augur.verify import hooks as vllm_hooks

    def clock(phase: str) -> int:
        if phase == "upd" and _async_stepping[0] is not None:
            # Async: the batch being applied is the oldest one on the GPU.
            fin = _async_stepping[0].batch_finish
            return max(coord.now, fin[0]) if fin else coord.now
        if phase == "upd" and _stepping[0] is not None:
            return coord.now + _stepping[0].ctx.last_step_ns
        return coord.now

    vllm_hooks.CLOCK = clock


# Event priorities at equal times (design doc: arrivals, transfers, step completions).
# CAUSED_ARRIVAL is a call released by a completion at this same instant (a
# session's next call after a zero tool wait). It cannot reach the engine
# before the engine has started its next step at that instant: in vLLM the
# client sees the finished output only after the step, and the engine loop
# schedules again right away. So it is applied after steps start.
ARRIVAL, TRANSFER_DONE, STEP_DONE, CAUSED_ARRIVAL = 0, 1, 2, 3

# An engine with unfinished requests that schedules no token for this long
# (simulated) is stuck: a run stops rather than spinning forever.
STALL_NS = 60 * 1_000_000_000


@dataclass
class PDLink:
    """KV pull from a prefill to a decode engine: latency + bytes / bandwidth.

    Concurrent pulls share bandwidth only in the fabric (--pd-contention decode,
    --contention on). Defaults are the fork's single-node link."""

    bw_gbps: float = 16.0
    latency_ns: int = 20_000
    contention: str = "none"  # "decode": pulls into one decode engine share its link fairly

    def transfer_ns(self, nbytes: int) -> int:
        return self.latency_ns + int(nbytes / self.bw_gbps)


@dataclass(order=True)
class _Event:
    t: int
    prio: int
    seq: int
    kind: str = field(compare=False)
    payload: object = field(compare=False)


def _empty(sch) -> bool:
    """True when the scheduler holds nothing, so ``has_requests()`` is False without asking.

    ``_start_steps`` polls every engine after every event, and at 64 engines the idle ones dominate its cost
    (has_requests is four calls deep in vLLM).  has_requests() is False if there are no live requests, no
    finished ids awaiting the next schedule, and no connector (which could report pending work).
    """
    return not sch.requests and not sch.finished_req_ids and sch.connector is None and sch.ec_connector is None


@dataclass
class _DpaRank:
    """One rank of an async DP group, running vLLM's DPEngineCoreProc loop: one forward per iteration."""

    e: SimEngine
    idx: int
    n: int = 0  # iterations in this wave (vLLM's step_counter)
    nfwd: int = 0  # forwards this rank has queued, real or dummy
    gpu_free: int = 0  # when its last queued forward ends
    fifo: deque = field(default_factory=deque)  # its batch queue: each batch's forward, None for an empty batch
    at: int | None = None  # when its next iteration starts
    asleep: bool = True  # no rank has work: the group waits for a request
    gen: object = None  # the iteration under way (a generator that waits on forwards and the sync)


class Coordinator:
    def __init__(self, engines: list[SimEngine], router: Router, workload: Workload,
                 lss_idle_quantum_ns: int = 0, decode_router: Router | None = None,
                 pd_link: PDLink | None = None, frontend_latency_ns: int = 0, delivery_latency_ns: int = 0,
                 frontend_ns_per_token: float = 0.0, topology=None):
        self.engines = engines
        # --contention on (augur.sim.fabric): every transfer is a flow through shared resources.
        self.topology = topology
        self.fabric = topology.fabric if topology is not None else None
        self._fabric_wake: int | None = None
        self.frontend_latency_ns = frontend_latency_ns
        # Replay of a real log: call id -> the engine request id the real run logged. Same strings, same hashes, so
        # the sets vLLM iterates (finished ids, ...) come out in the real order, which the CPU tier's eviction order
        # follows with KV offload (the real ids carry a random suffix per run, so only a replay can reuse them).
        self.request_ids: dict[str, str] = {}
        self._call_of_rid: dict[str, str] = {}
        self.frontend_ns_per_token = frontend_ns_per_token  # added per prompt token (tokenization, request parsing)
        self.delivery_latency_ns = delivery_latency_ns  # engine emits a token -> the client has it
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
        # DP groups (one for now): ranks step in lockstep rounds.
        self.dp_group = [e for e in engines if e.spec.data_parallel_size > 1]
        self.dp_rounds = 0
        self.dp_dummy_forwards = 0
        self._dp_held: list[tuple[int, SimEngine, Call]] = []  # replay: calls waiting for their round
        self._dp_pending: dict[int, tuple] = {}  # engine_id -> (outputs, shape or None)
        self._dp_ready_at: int | None = None  # --lss-loop-compat only
        # DP + async scheduling
        self._dpa = bool(self.dp_group) and self.dp_group[0].async_sched
        self._dpa_ranks = [_DpaRank(e, i) for i, e in enumerate(self.dp_group)] if self._dpa else []
        self._dpa_replay = self._dpa and all(hasattr(e.ctx.cost, "round_start") for e in self.dp_group)
        self._dpa_interval = self.dp_group[0].vllm_config.parallel_config.dp_sync_interval if self._dpa else 1
        self._dpa_regs: dict[int, dict] = {}  # forward -> rank -> (when queued, shape or None), until every rank has one
        self._dpa_res: dict[tuple, list] = {}  # a settled wait (forward or sync) -> the time it ends for each rank
        self._dpa_waiting: dict[tuple, list] = {}  # a wait -> the ranks blocked on it
        self._dpa_arrived: dict[tuple, dict] = {}  # sync -> rank -> when the rank got there
        self._dpa_wave = 0
        self._dpa_started = False
        self._entries_done = False
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
            self._entries_done = True
            return
        t = e.arrival_ns
        if t < self._last_entry_ns:
            raise ValueError(f"workload not sorted by arrival: {t} after {self._last_entry_ns}")
        self._last_entry_ns = t
        prio = CAUSED_ARRIVAL if getattr(e, "after_step_start", False) else ARRIVAL
        self._push(t, prio, "entry", e)

    def _cid(self, request_id: str) -> str:
        return self._call_of_rid.get(request_id) or strip_pd_suffix(request_id)

    def _pick(self, router: Router, call: Call, engines: list[SimEngine]) -> SimEngine:
        eid = router.route(call, engines, self.now)
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
        delay = self.frontend_latency_ns + int(self.frontend_ns_per_token * len(call.prompt_ids))
        if delay:
            # The API server's work before the engine sees the request (HTTP,
            # tokenization, hand-off); the call's times stay measured from
            # when the client sent it.
            eng.admitting += 1
            self._push(self.now + delay, ARRIVAL, "admit", (eng, call))
        else:
            self._admit(eng, call)

    def _admit(self, eng: SimEngine, call: Call) -> None:
        # Replay of a DP run (sync scheduling): the call joins the round the log says, not the one this clock puts it in.
        join = getattr(eng.ctx.cost, "add_round", {}).get(call.call_id, 0) if eng in self.dp_group and not self._dpa else 0
        if join > self.dp_rounds:
            self._dp_held.append((join, eng, call))
            return
        if self.pd:
            eng.add(call, self.now, request_id=call.call_id + "@p", max_tokens=1,
                    kv_transfer_params={"do_remote_decode": True})
        else:
            rid = self.request_ids.get(call.call_id)
            if rid is not None:
                self._call_of_rid[rid] = call.call_id
            eng.add(call, self.now, request_id=rid)

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
        elif ev.kind == "poll":
            p.xfer_wake_at = None
        elif ev.kind == "admit":
            p[0].admitting -= 1
            self._admit(*p)
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
        if eng.spec.kv_events:
            # vLLM published this step's KV events when it applied the outputs; the
            # router hears of them when the step lands.
            events = eng.take_kv_events()
            if events and hasattr(self.router, "observe"):
                self.router.observe(eng.engine_id, events)
        for eco in outputs.values():
            for o in eco.outputs:
                cid = self._cid(o.request_id)
                rec = self._records[cid]
                if o.request_id.endswith("@p"):
                    # The prefill engine's one token goes back to the proxy, not
                    # the client; the client's tokens come from the decode engine.
                    if o.finish_reason is not None:
                        eng.ctx.output_ids.pop(o.request_id, None)
                        eng.ctx.recorded.discard(o.request_id)
                        rec.prefill_done_ns = self.now
                        self.router.release(self._calls[cid], self.now)
                        if not o.kv_transfer_params:
                            raise RuntimeError(f"{o.request_id}: prefill engine returned no kv_transfer_params")
                        # The proxy forwards to decode once it has this response.
                        self._push(self.now, CAUSED_ARRIVAL, "decode_dispatch", (cid, o.kv_transfer_params))
                    continue
                if not rec.token_times and o.new_token_ids and hasattr(self.router, "first_token"):
                    self.router.first_token(self._calls[cid], self.now)
                rec.token_times.extend([self.now + self.delivery_latency_ns] * len(o.new_token_ids))
                if o.finish_reason is not None:
                    self._finish(cid, rec, eng)
        if eng.core.scheduler.get_num_unfinished_requests() == 0 and eng.idle_since is None:
            # Idle from the moment its last step completes, before any session
            # call released at this same instant (matters only for the quantum).
            eng.idle_since = self.now

    def _finish(self, call_id: str, rec: CallRecord, eng: SimEngine) -> None:
        # The client has the last token, and acts on it, delivery_latency_ns after the engine emits it.
        done_ns = self.now + self.delivery_latency_ns
        rec.end_ns = done_ns
        self.recorder.done.append(rec)
        call = self._calls.pop(call_id)
        (self.decode_router if self.pd else self.router).release(call, done_ns)
        del self._records[call_id]
        self._call_engine.pop(call_id, None)
        rid = self.request_ids.get(call_id, call_id)
        self._call_of_rid.pop(rid, None)
        eng.ctx.output_ids.pop(call_id + "@d" if self.pd else rid, None)
        eng.ctx.recorded.discard(call_id + "@d" if self.pd else rid)
        if call.session_id is not None:
            s = self._sessions[call.session_id]
            if s.next_index < len(s.sub_requests):
                prio = CAUSED_ARRIVAL if call.tool_wait_ns + self.delivery_latency_ns == 0 else ARRIVAL
                if prio == CAUSED_ARRIVAL and self.lss_idle_quantum_ns and eng.spec.tensor_parallel_size > 1:
                    # LLMServingSim compat only: at TP>1 the other ranks' reports
                    # at this instant route and schedule again, so the call joins
                    # the next step. Not possible in vLLM.
                    prio = ARRIVAL
                self._push(done_ns + call.tool_wait_ns, prio, "session_call", s.session_id)
            else:
                del self._sessions[s.session_id]

    def _dp_padded_tokens(self, wants: list[tuple[int, bool]]) -> list[int]:
        """vLLM's DP padding (coordinate_batch_across_dp): each rank dispatches
        locally; the ranks agree on the minimum CUDA-graph mode; if that mode
        keeps graphs, every rank pads to the largest padded size and
        re-dispatches. Uses each rank's own CudagraphDispatcher."""
        from vllm.config import CUDAGraphMode
        local = []
        for e, (n, uniform) in zip(self.dp_group, wants):
            mode, desc = e.core.model_executor.worker.cudagraph.dispatch(num_tokens=n, uniform_decode=uniform)
            local.append((mode, desc.num_tokens))
        synced = CUDAGraphMode(min(m.value for m, _ in local))
        if synced == CUDAGraphMode.NONE:
            return [p for _, p in local]
        target = max(p for _, p in local)
        out = []
        for e, (n, uniform) in zip(self.dp_group, wants):
            _, desc = e.core.model_executor.worker.cudagraph.dispatch(
                num_tokens=target, uniform_decode=uniform, valid_modes={synced})
            out.append(desc.num_tokens)
        return out

    def _start_dp_round(self, retry: int = 4) -> None:
        """Lockstep steps of the DP group, as vLLM's DP engine loop runs them.

        A rank schedules its next step as soon as its own previous forward
        finishes (it then blocks in the first collective). While any rank has
        work, a rank with nothing to schedule joins with a dummy forward so the
        EP collectives line up. The round starts once every rank has joined;
        each rank's outputs land when its own forward finishes."""
        g = self.dp_group
        for held in [h for h in self._dp_held if h[0] <= self.dp_rounds]:
            self._dp_held.remove(held)
            self._admit(held[1], held[2])
        # Ranks free at the same instant join in descending rank order (the
        # order ASTRA-Sim polls NPUs in LLMServingSim); idle ranks join last
        # with a dummy. The last to join is handed the round first, which
        # matters only for who waits on the shared CPU-memory device.
        for e in sorted(g, key=lambda e: -e.engine_id):
            if not e.busy and e.engine_id not in self._dp_pending and e.core.scheduler.has_requests():
                self._set_step_clock(e, gpu_free_ns=self.now)
                outputs, executed, _ = e.step()
                self._after_step(e)
                self._poll_when_transfers_finish(e)
                shape = e.ctx.last_shape if executed and e.ctx.last_shape is not None and e.ctx.last_shape.seqs else None
                self._dp_pending[e.engine_id] = (outputs, shape)
        if not any(p[1] is not None or p[0] for p in self._dp_pending.values()):
            if self._dp_pending and all(not e.busy for e in g):
                # Every rank stepped and nothing was scheduled anywhere: apply
                # the outputs; the group waits for the next event.
                delivered = False
                for e in g:
                    outputs, _ = self._dp_pending.pop(e.engine_id, ({}, None))
                    if e.ctx.offload is not None:
                        e.ctx.offload.commit_loads(self.now)
                    delivered = delivered or e.ctx.last_delivered
                    self._on_step_done(_Event(self.now, STEP_DONE, 0, "step_done", (e, outputs)))
                if delivered and retry:
                    # A rank only took delivery of finished KV (an offload load): vLLM
                    # promotes that request at its next schedule, right away.
                    self._start_dp_round(retry - 1)
            return
        for e in sorted(g, key=lambda e: -e.engine_id):
            if not e.busy and e.engine_id not in self._dp_pending:
                self._dp_pending[e.engine_id] = ({}, None)  # idle rank joins with a dummy
        if any(e.busy for e in g):
            return
        outs = [self._dp_pending[e.engine_id][0] for e in g]
        shapes = [self._dp_pending[e.engine_id][1] for e in g]
        q = self.lss_idle_quantum_ns
        if q and self._entries_done and not any(ev.kind == "session_call" for ev in self._q) and any(
                not e.core.scheduler.has_requests() for e in g):
            # LLMServingSim compat only: once no arrivals remain, an instance
            # with no requests is put to sleep ("done") and its dummy joins
            # each round only on ASTRA-Sim's 1 ms poll. vLLM's idle rank runs
            # its dummy at once.
            if self._dp_ready_at is None:
                self._dp_ready_at = self.now + q
                self._push(self._dp_ready_at, ARRIVAL, "dp_wake", None)
            if self.now < self._dp_ready_at:
                return
        self._dp_ready_at = None
        last_joined = list(self._dp_pending)[-1]
        first_rank = next(i for i, e in enumerate(g) if e.engine_id == last_joined)
        self._dp_pending.clear()
        final = self._round_shapes(shapes)
        replay = self._replay_rounds(g)
        per_rank = replay or self._scaled(g, g[0].ctx.cost.round_ns(final, first_rank=first_rank), final)
        # The ranks meet in the round's first collective, so the slowest rank's CPU time holds up all of them.
        cpu_ns = 0 if replay else max((e.ctx.cpu_fixed_ns + e.ctx.cpu_per_request_ns * len(s.seqs)
                                       + e.ctx.last_offload_wait_ns for e, s in zip(g, shapes) if s is not None), default=0)
        per_rank = [ns + cpu_ns for ns in per_rank]
        self._log_round(g, final, per_rank)
        for e, ns in zip(g, per_rank):
            if e.ctx.offload is not None:
                e.ctx.offload.commit_loads(self.now + ns)  # the round's forward ends with the step
        self.dp_rounds += 1
        for e, sh, ns in zip(g, final, per_rank):
            self._log_step(e, self.now, ns, sh)
        for e, outputs, ns in zip(g, outs, per_rank):
            e.busy = True
            self._push(self.now + ns, STEP_DONE, "step_done", (e, outputs))

    def _round_shapes(self, shapes) -> list:
        """A DP round's batches, padded to the tokens vLLM's DP ranks agree on; an idle rank (None) runs a
        dummy forward."""
        from augur.cost import SeqWork, StepShape
        wants = [(s.total_tokens, all(x.num_new == 1 for x in s.seqs)) if s else (1, True) for s in shapes]
        final = []
        for s, p in zip(shapes, self._dp_padded_tokens(wants)):
            if s is None:
                self.dp_dummy_forwards += 1
                final.append(StepShape(seqs=(SeqWork(1, 1, True),), padded_tokens=p, dummy=True))
            else:
                final.append(StepShape(seqs=s.seqs, padded_tokens=p))
        return final

    @staticmethod
    def _log_step(eng: SimEngine, start: int, ns: int, sh) -> None:
        """One --step-log row: start, duration and batch shape."""
        if eng.step_log is not None and sh is not None:
            eng.step_log.append((start, ns, sh.total_tokens, sum(x.num_new for x in sh.seqs if not x.is_decode),
                                 sh.num_decode, sh.num_prefill, sh.forward_tokens, int(sh.dummy)))

    @staticmethod
    def _replay_rounds(g) -> list[int] | None:
        """Replay of a DP run: each rank's logged duration for its next round (the
        logged round already holds padding, CPU time and offload waits)."""
        if not hasattr(g[0].ctx.cost, "next_round_ns"):
            return None
        return [e.ctx.cost.next_round_ns() for e in g]

    def _log_round(self, g, final, per_rank) -> None:
        """Each rank's step log gets the round it took part in, dummy forwards included
        (a real vLLM log has no such record; only augur's own DP logs carry it, for replay)."""
        for e, sh, ns in zip(g, final, per_rank):
            log = e.core.scheduler._logger() if hasattr(e.core.scheduler, "_logger") else None
            if log:
                log.write({"ev": "round", "t": self.now, "dur": ns, "dummy": bool(sh.dummy)})

    @staticmethod
    def _scaled(g, per_rank, shapes):
        ctx = g[0].ctx
        per_rank = [adjust_gpu_ns(ns, sh, ctx.step_time_scale, ctx.class_scales) for ns, sh in zip(per_rank, shapes)]
        if any(e.ctx.net is not None for e in g):  # a rank's PCIe traffic slows its own forward
            per_rank = [round(ns * e.ctx.interference_factor(sh)) for e, ns, sh in zip(g, per_rank, shapes)]
        return per_rank

    def _after_step(self, eng: SimEngine) -> None:
        self._start_transfers(eng)
        for rid, hit in eng.ctx.last_new_req_ids.items():
            rec = self._records.get(self._cid(rid))
            if rec is not None and rec.first_sched_ns is None:
                rec.first_sched_ns = self.now
                rec.cached_tokens = hit
        for rid in eng.ctx.last_preempted_req_ids:
            rec = self._records.get(self._cid(rid))
            if rec is not None:
                rec.num_preemptions += 1

    def _step_async(self, eng: SimEngine) -> None:
        """vLLM's async scheduling (EngineCore.step_with_batch_queue).

        Each call may schedule a batch and launch it; a launched batch starts
        when the GPU stream is free. A call that pops the oldest batch returns
        when that batch finishes, so the engine's CPU is blocked until then and
        the next schedule happens at that time. A batch scheduled while the
        previous one runs therefore sees arrivals only up to its own schedule
        time, one step stale, as in vLLM."""
        progressed = False
        for _ in range(10_000):
            if eng.core.scheduler.has_requests() and not self._transfer_due(eng) and self._replay_holds(eng):
                return  # replay: the logged engine scheduled its next batch later
            t0 = time.perf_counter()
            _async_stepping[0] = eng
            self._set_step_clock(eng, gpu_free_ns=max(self.now, eng.gpu_free_at))
            try:
                outputs, executed = eng.core.step_fn()
            finally:
                _async_stepping[0] = None
            self._check_progress(eng, executed)
            progressed = progressed or executed or eng.ctx.last_delivered
            eng.sched_cpu_s += time.perf_counter() - t0
            eng.steps += 1
            for ns in eng.ctx.launches:
                # The batch reaches the GPU once the CPU has prepared it.
                start = max(self.now + (eng.ctx.last_cpu_ns if ns else 0), eng.gpu_free_at)
                eng.gpu_free_at = start + ns
                eng.batch_finish.append(eng.gpu_free_at)
                eng.batch_reports.append(eng.ctx.last_kv_report)
                if ns:
                    self._log_step(eng, start, ns, eng.ctx.last_shape)
            launched = bool(eng.ctx.launches)
            eng.ctx.launches.clear()
            self._after_step(eng)
            if outputs is None:
                if launched:
                    continue  # queue not full: schedule the next batch now
                return  # nothing queued, nothing to schedule: idle
            done_at = max(self.now, eng.batch_finish.popleft())
            # What the batch's KV connector worker reported (KV arrived,
            # transfers done) reaches the scheduler only with this output.
            reported = eng.batch_reports.popleft()
            progressed = progressed or reported
            if done_at == self.now:
                self._on_step_done(_Event(self.now, STEP_DONE, 0, "step_done", (eng, outputs)))
                if not eng.core.scheduler.has_requests():
                    if eng.batch_finish:
                        continue  # vLLM's loop drains the queue at once
                    return
                if not progressed and not any(eng.batch_reports):
                    # Nothing computed or reported, and no queued batch carries
                    # a report (vLLM's loop would spin until it is applied):
                    # wait for the next event.
                    return
                progressed = False
                continue
            eng.busy = True
            self._push(done_at, STEP_DONE, "step_done", (eng, outputs))
            return
        raise RuntimeError("async engine loop did not settle")

    # ---- DP + async scheduling -------------------------------------------------
    # vLLM's DP loop (DPEngineCoreProc.run_busy_loop) with a batch queue of 2 runs per iteration: schedule a batch and
    # launch it; if the queue is full block on the oldest batch, whose output resolves when the forward after it
    # ends (its own if none follows); a rank that ran nothing then runs a dummy forward; every dp_sync_interval
    # iterations (and the first) the ranks meet in a CPU all-reduce. The next batch is therefore scheduled after
    # the forward, not during it. Forward j of every rank is one collective round, so it starts when the last rank
    # has queued it. In a replay the log's iteration times stand in for all of that.
    def _dpa_kick(self) -> None:
        """Start the iterations whose time has come; wake the group when work reaches it."""
        ranks = self._dpa_ranks
        if self._dpa_replay and not self._dpa_started:
            self._dpa_started = True
            for r in ranks:
                self._dpa_set_at(r, self._dpa_replay_at(r))
        for _ in range(10_000):
            started = False
            for r in ranks:
                if r.gen is not None:
                    continue
                if not self._dpa_replay:
                    work = self._dpa_work()
                    if r.asleep and work:
                        if all(x.asleep for x in ranks):
                            self._dpa_wave += 1  # the loop's step_counter starts over with a wave
                            for x in ranks:
                                x.n = 0
                        r.asleep, r.at = False, self.now
                    elif not r.asleep and not work and r.at is not None and r.at <= self.now:
                        r.asleep, r.at = True, None
                if r.at is None or r.at > self.now:
                    continue
                r.at = None
                started = True
                if self._dpa_replay:
                    self._dpa_replay_iteration(r)
                else:
                    r.gen = self._dpa_iteration(r)
                    self._dpa_resume(r, None)
            if not started:
                return
        raise RuntimeError("DP async rank loop did not settle")

    def _dpa_work(self) -> bool:
        """Some rank has requests or a queued batch, or a forward waits for a rank that has not queued it."""
        return bool(self._dpa_regs or self._dpa_arrived) or any(
            r.fifo or r.e.core.scheduler.has_requests() for r in self._dpa_ranks)

    def _dpa_set_at(self, r: _DpaRank, t: int | None) -> None:
        r.at = t
        if t is not None and t > self.now:
            self._push(t, ARRIVAL, "dpa_wake", None)

    def _dpa_engine_step(self, r: _DpaRank):
        """One call of the rank's EngineCore.step_with_batch_queue. Returns (outputs, executed, launched, shape, cpu_ns):
        the output of the batch it applied, if any; whether it launched a forward; the launch's shape (None for an
        empty batch); the CPU time the batch took to prepare."""
        e = r.e
        t0 = time.perf_counter()
        self._set_step_clock(e, gpu_free_ns=self.now)
        outputs, executed = e.core.step_fn()
        e.sched_cpu_s += time.perf_counter() - t0
        e.steps += 1
        self._poll_when_transfers_finish(e)
        launched = bool(e.ctx.launches)
        shape = e.ctx.launch_shapes[-1] if launched else None
        cpu = e.ctx.last_cpu_ns if shape is not None else 0
        e.ctx.launches.clear()
        e.ctx.launch_shapes.clear()
        self._after_step(e)
        return outputs, executed, launched, shape, cpu

    def _dpa_iteration(self, r: _DpaRank):
        """One iteration of the rank's loop, as a generator that yields what it waits for."""
        e, start = r.e, self.now
        r.n += 1
        outputs, executed, launched, shape, cpu = None, False, False, None, 0
        if e.core.scheduler.has_requests() or r.fifo:
            outputs, executed, launched, shape, cpu = self._dpa_engine_step(r)
        queued = start + cpu
        new = self._dpa_queue(r, queued, shape) if launched and shape is not None else None
        if launched:
            r.fifo.append(new)
        end, applied = (queued if executed else start), None
        if outputs is not None:
            oldest = r.fifo.popleft()
            wait = new if new is not None else oldest
            end = queued if oldest is None else max(queued, (yield from self._dpa_wait(r, ("fwd", wait))))
            self._push(end, STEP_DONE, "dpa_done", (e, outputs))
            applied = end
        if not executed:
            end = yield from self._dpa_wait(r, ("fwd", self._dpa_queue(r, end, None)))
        if r.n == 1 or r.n % self._dpa_interval == 0:
            key = ("sync", self._dpa_wave, r.n)
            arrived = self._dpa_arrived.setdefault(key, {})
            arrived[r.idx] = end
            if len(arrived) == len(self._dpa_ranks):
                del self._dpa_arrived[key]
                self._dpa_settle(key, [max(arrived.values())] * len(arrived))
            end = yield from self._dpa_wait(r, key)
        self._dpa_log_round(r, start, end, not executed, applied)
        r.gen = None
        self._dpa_set_at(r, end)

    def _dpa_wait(self, r: _DpaRank, key: tuple):
        res = self._dpa_res.get(key)
        if res is None:
            res = yield key
        return res[r.idx]

    def _dpa_resume(self, r: _DpaRank, value) -> None:
        try:
            key = r.gen.send(value)
        except StopIteration:
            return
        self._dpa_waiting.setdefault(key, []).append(r)

    def _dpa_settle(self, key: tuple, ends: list) -> None:
        self._dpa_res[key] = ends
        for r in self._dpa_waiting.pop(key, ()):
            self._dpa_resume(r, ends)

    def _dpa_queue(self, r: _DpaRank, when: int, shape) -> int:
        """The rank queues its next forward (a dummy if shape is None). Once every rank has, the round runs."""
        j = r.nfwd
        r.nfwd += 1
        regs = self._dpa_regs.setdefault(j, {})
        regs[r.idx] = (when, shape)
        if len(regs) == len(self._dpa_ranks):
            del self._dpa_regs[j]
            g = self.dp_group
            final = self._round_shapes([regs[i][1] for i in range(len(g))])
            per_rank = self._scaled(g, g[0].ctx.cost.round_ns(final), final)
            # Each rank's forward waits for its own previous one; the collectives hold the round for the slowest.
            begin = max([w for w, _ in regs.values()] + [x.gpu_free for x in self._dpa_ranks])
            for x, e, sh, ns in zip(self._dpa_ranks, g, final, per_rank):
                x.gpu_free = begin + ns
                self._log_step(e, begin, ns, sh)
            self.dp_rounds += 1
            self._dpa_res.pop(("fwd", j - 8), None)
            self._dpa_settle(("fwd", j), [x.gpu_free for x in self._dpa_ranks])
        return j

    def _dpa_log_round(self, r: _DpaRank, start: int, end: int, dummy: bool, applied: int | None) -> None:
        log = r.e.core.scheduler._logger() if hasattr(r.e.core.scheduler, "_logger") else None
        if log:
            # The batch update was logged when vLLM applied it, before its time was known: say it here.
            log.write({"ev": "round", "t": start, "dur": end - start, "dummy": dummy,
                       **({"upd": applied} if applied is not None else {})})

    def _dpa_replay_at(self, r: _DpaRank) -> int | None:
        """When the logged rank made its next iteration's schedule call (the iteration's start if it made none)."""
        c = r.e.ctx.cost
        if r.n >= len(c.rounds):
            return None
        return c.round_start[r.n] if c.round_sched[r.n] is None else c.round_sched[r.n]

    def _dpa_replay_iteration(self, r: _DpaRank) -> None:
        """Replay of a real run: the iteration lasts what the log says (its sync and dummy forward included), and
        the batch output it applies lands when the log says it was applied."""
        c = r.e.ctx.cost
        start = c.round_start[r.n]
        end = start + c.rounds[r.n]
        applied = None
        if c.round_sched[r.n] is not None:
            outputs = self._dpa_engine_step(r)[0]
            if outputs is not None:
                applied = end if c.round_upd[r.n] is None else c.round_upd[r.n]
                self._push(applied, STEP_DONE, "dpa_done", (r.e, outputs))
        self._dpa_log_round(r, start, end, c.round_dummy[r.n], applied)
        if r.idx == 0:
            self.dp_rounds += 1
        self.dp_dummy_forwards += c.round_dummy[r.n]
        r.n += 1
        self._dpa_set_at(r, self._dpa_replay_at(r))

    def _dpa_done(self, ev: _Event) -> None:
        self._on_step_done(ev)

    def _start_steps(self) -> None:
        q = self.lss_idle_quantum_ns
        if self._dpa:
            self._dpa_kick()
        elif self.dp_group:
            self._start_dp_round()
        for eng in self.engines:
            if eng.busy or eng.spec.data_parallel_size > 1:
                continue
            sch = eng.core.scheduler
            if eng.async_sched:
                if not _empty(sch) and sch.has_requests() or eng.batch_finish:
                    self._step_async(eng)
                    self._poll_when_transfers_finish(eng)
                    self._replay_spin(eng)
                continue
            pending_kv = bool(eng.ctx.deliver_recving or eng.ctx.deliver_sending)
            if (_empty(sch) or not sch.has_requests()) and not pending_kv:
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
            if self._replay_holds(eng) and not self._transfer_due(eng):
                continue
            for _ in range(4):
                self._set_step_clock(eng, gpu_free_ns=self.now)
                outputs, executed, step_ns = eng.step()
                self._start_transfers(eng)
                self._poll_when_transfers_finish(eng)
                if executed or step_ns or not eng.ctx.last_delivered or self._replay_holds(eng):
                    break
                # An empty step that only delivered finished KV transfers: vLLM
                # promotes those requests at the next schedule, right away.
                self._on_step_done(_Event(self.now, STEP_DONE, 0, "step_done", (eng, outputs)))
            self._check_progress(eng, executed)
            self._after_step(eng)
            if not executed and step_ns == 0:
                # Nothing could be scheduled (e.g. KV cache full with nothing to preempt).
                # Apply any outputs now; the engine waits for the next event.
                self._on_step_done(_Event(self.now, STEP_DONE, 0, "step_done", (eng, outputs)))
                continue
            self._log_step(eng, self.now, step_ns, eng.ctx.last_shape)
            eng.busy = True
            self._push(self.now + step_ns, STEP_DONE, "step_done", (eng, outputs))

    def _replay_holds(self, eng: SimEngine) -> bool:
        """Replay: the logged engine started its next step later than the
        earliest moment it could (its loop's wake-up); hold the engine until
        the logged time so the same arrivals are in view. A logged empty
        step is let through too: it can hand a KV connector new requests."""
        cost = eng.ctx.cost
        starts = getattr(cost, "starts", None)
        if not starts or cost.k >= len(starts) or self.now >= starts[cost.k]:
            return False
        # Async: when a step was just launched at this instant, the schedule logged now is that step's, not a next
        # one (a real engine blocked in a launch scheduled again only later, with the requests that arrived meanwhile).
        launched_now = eng.async_sched and cost.k > 0 and starts[cost.k - 1] == self.now
        at = cost.next_schedule_ns(self.now + 1 if launched_now else self.now)
        if at == self.now:
            return False
        at = starts[cost.k] if at is None else min(at, starts[cost.k])
        if eng.wake_at != at:
            eng.wake_at = at
            self._push(at, ARRIVAL, "wake", eng)
        return True

    def _replay_spin(self, eng: SimEngine) -> None:
        """Replay, async: the logged engine loop kept scheduling (empty
        batches too) while it had requests; each schedule shapes its batch
        queue. Wake at its next logged schedule."""
        next_schedule = getattr(eng.ctx.cost, "next_schedule_ns", None)
        if next_schedule is None or eng.busy or not eng.core.scheduler.has_requests():
            return
        at = next_schedule(self.now + 1)
        if at is not None and eng.wake_at != at:
            eng.wake_at = at
            self._push(at, ARRIVAL, "wake", eng)

    def _check_progress(self, eng: SimEngine, executed: bool) -> None:
        """Raise if the engine has had requests but scheduled no token for
        STALL_NS. Called by: _start_steps, _step_async (after each step)."""
        waiting = eng.core.scheduler.get_num_unfinished_requests()
        if executed or not waiting:
            eng.last_progress_ns = self.now
            return
        if self.now - eng.last_progress_ns > STALL_NS:
            raise RuntimeError(
                f"engine {eng.engine_id} scheduled no token for {STALL_NS // 1_000_000_000} s of simulated time "
                f"({eng.last_progress_ns / 1e9:.3f} s to {self.now / 1e9:.3f} s) while {waiting} requests wait. "
                "With tiered KV offload the likely cause is a request whose prefix needs more chunks than the "
                "CPU tier holds: vLLM's tiering keeps promoting them and evicting them again. Give "
                "--kv-offload-gib room for at least one request's context.")

    def _set_step_clock(self, eng: SimEngine, gpu_free_ns: int) -> None:
        """What the executor and the offload tiers need to know about time
        before an engine step: now, and when the GPU work queued before it is done."""
        eng.ctx.now_ns = eng.ctx.tier_clock_ns = self.now
        eng.ctx.gpu_free_ns = gpu_free_ns
        if self.fabric is not None:
            self.fabric.advance(self.now)  # flows other engines started at this instant count in this step

    @staticmethod
    def _transfer_sources(eng: SimEngine) -> list:
        """The engine's offload worker and secondary tiers (each has next_end / deliverable)."""
        return ([eng.ctx.offload] if eng.ctx.offload is not None else []) + eng.ctx.tiers

    def _transfer_due(self, eng: SimEngine) -> bool:
        """An offload transfer has finished: vLLM's engine loop, polling while
        transfers are pending, takes a step now to report it."""
        return any(src.deliverable(self.now) for src in self._transfer_sources(eng))

    def _poll_when_transfers_finish(self, eng: SimEngine) -> None:
        """vLLM keeps stepping while offload transfers are pending; the step
        that matters is the first after one finishes. Queue it."""
        if self.fabric is not None:
            self.fabric.advance(self.now)
        t = min((e for src in self._transfer_sources(eng) if (e := src.next_end()) is not None), default=None)
        if t is not None and t > self.now and eng.xfer_wake_at != t:
            eng.xfer_wake_at = t
            self._push(t, ARRIVAL, "poll", eng)

    def _start_transfers(self, eng: SimEngine) -> None:
        """The decode worker starts pulling KV in the step its scheduler parks
        the request (WAITING_FOR_REMOTE_KVS); finished_recving and the prefill
        side's finished_sending are reported at each engine's next step after
        the pull completes."""
        for rid, blocks, remote_engine, remote_rid in eng.ctx.new_recvs:
            if rid in self._xfers_started:
                continue
            self._xfers_started.add(rid)
            payload = (eng, rid, self._by_vllm_id.get(remote_engine), remote_rid)
            nbytes = blocks * eng.ctx.kv_block_bytes
            if self.fabric is not None:
                # GPU-direct RDMA: the pull leaves the source GPU's NIC and enters the destination's, through
                # the network (or, topology "decode-link", one link per decode engine), sharing them with every
                # other pull.
                src = payload[2]
                path = (self.topology.pd_path(src.engine_id, eng.engine_id) if src is not None
                        else [f"gpu{self.topology.gpu_of[eng.engine_id]}.nic"])
                self.fabric.submit(("pd", rid), None, path, nbytes, self.now, self.pd_link.latency_ns,
                                   payload=payload, notify=True,
                                   rounding="ceil" if self.pd_link.contention == "decode" else "floor")
            else:
                ns = self.pd_link.transfer_ns(nbytes)
                self._push(self.now + ns, TRANSFER_DONE, "kv_pull", payload)

    def _advance_fabric(self) -> None:
        """Run the fabric to now; the PD pulls it finished are delivered (as a TRANSFER_DONE event would)."""
        self.fabric.advance(self.now)
        for _, _, payload in self.fabric.pop_completed():
            self._on_transfer_done(_Event(self.now, TRANSFER_DONE, 0, "kv_pull", payload))

    def _arm_fabric(self) -> None:
        """Wake at the fabric's next event: a transfer ending frees bandwidth for the ones beside it."""
        self.fabric.advance(self.now)
        t = self.fabric.next_event()
        if t is not None and t != self._fabric_wake:
            self._fabric_wake = t
            self._push(max(t, self.now), TRANSFER_DONE, "fabric", None)

    def _on_fabric_event(self) -> None:
        """Some flow started, got going or ended: engines polling for their transfers re-time their wake-up."""
        self._fabric_wake = None
        for e in self.engines:
            if e.ctx.offload is not None or e.ctx.tiers:
                self._poll_when_transfers_finish(e)

    def run(self) -> Recorder:
        t0 = time.perf_counter()
        self._pull_entry()
        self._drain()
        if self._records:
            raise RuntimeError(f"{len(self._records)} calls never finished (e.g. {next(iter(self._records))})")
        self.wall_s = time.perf_counter() - t0
        return self.recorder

    def _drain(self, until: int | None = None) -> None:
        """Process every instant of the queue before ``until`` (all of it when None). A parallel worker's
        shard (augur.sim.parallel) runs the loop in windows."""
        while self._q and (until is None or self._q[0].t < until):
            self.now = self._q[0].t
            if self.fabric is not None:
                self._advance_fabric()
            while self._q and self._q[0].t == self.now and self._q[0].prio < CAUSED_ARRIVAL:
                ev = heapq.heappop(self._q)
                if ev.prio == ARRIVAL:
                    self._on_arrival(ev)
                elif ev.kind == "dpa_done":
                    self._dpa_done(ev)
                elif ev.prio == TRANSFER_DONE:
                    if ev.kind == "fabric":
                        self._on_fabric_event()
                    else:
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
            if self.fabric is not None:
                self._arm_fabric()
