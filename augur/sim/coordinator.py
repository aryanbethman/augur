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
from dataclasses import dataclass, field

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

    No contention between concurrent pulls yet (ASTRA-Sim congestion-aware
    mode is the planned upgrade). Defaults are the fork's single-node link."""

    bw_gbps: float = 16.0
    latency_ns: int = 20_000
    contention: str = "none"  # "decode": pulls into one decode engine share its link fairly

    def transfer_ns(self, nbytes: int) -> int:
        return self.latency_ns + int(nbytes / self.bw_gbps)


class SharedLink:
    """Processor sharing: n active transfers each get bandwidth / n.

    Remaining bytes are advanced whenever the set changes; the next completion
    is (re)scheduled as an event tagged with a version, so stale ones are
    ignored."""

    def __init__(self, bw_gbps: float):
        self.bw = bw_gbps  # bytes per ns
        self.active: dict = {}  # key -> [remaining bytes, payload]
        self.last = 0
        self.version = 0

    def advance(self, now: int) -> None:
        if self.active and now > self.last:
            share = (now - self.last) * self.bw / len(self.active)
            for v in self.active.values():
                v[0] -= share
        self.last = now

    def next_finish(self, now: int) -> int | None:
        if not self.active:
            return None
        least = min(v[0] for v in self.active.values())
        return now + max(0, int(-(-least * len(self.active) // self.bw)))


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
                 pd_link: PDLink | None = None, frontend_latency_ns: int = 0, delivery_latency_ns: int = 0):
        self.engines = engines
        self.frontend_latency_ns = frontend_latency_ns
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
        self._dp_pending: dict[int, tuple] = {}  # engine_id -> (outputs, shape or None)
        self._dp_ready_at: int | None = None  # --lss-loop-compat only
        # DP + async scheduling
        self._dpa = bool(self.dp_group) and self.dp_group[0].async_sched
        self._dpa_shapes: dict[int, dict] = {}  # round -> engine_id -> shape (None: empty)
        self._dpa_next_form = 0
        self._dpa_running: int | None = None  # round on the GPU
        self._dpa_left: set = set()  # ranks still running that round
        self._dpa_finished: dict[int, set] = {}  # round -> ranks done with it
        self._entries_done = False
        self._xfers_started: set[str] = set()
        self._links: dict[int, SharedLink] = {}
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

    @staticmethod
    def _cid(request_id: str) -> str:
        return strip_pd_suffix(request_id)

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
        if self.frontend_latency_ns:
            # The API server's work before the engine sees the request (HTTP,
            # tokenization, hand-off); the call's times stay measured from
            # when the client sent it.
            eng.admitting += 1
            self._push(self.now + self.frontend_latency_ns, ARRIVAL, "admit", (eng, call))
        else:
            self._admit(eng, call)

    def _admit(self, eng: SimEngine, call: Call) -> None:
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
                        self.router.release(self._calls[cid], self.now)
                        if not o.kv_transfer_params:
                            raise RuntimeError(f"{o.request_id}: prefill engine returned no kv_transfer_params")
                        # The proxy forwards to decode once it has this response.
                        self._push(self.now, CAUSED_ARRIVAL, "decode_dispatch", (cid, o.kv_transfer_params))
                    continue
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
        eng.ctx.output_ids.pop(call_id + "@d" if self.pd else call_id, None)
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

    def _start_dp_round(self) -> None:
        """Lockstep steps of the DP group, as vLLM's DP engine loop runs them.

        A rank schedules its next step as soon as its own previous forward
        finishes (it then blocks in the first collective). While any rank has
        work, a rank with nothing to schedule joins with a dummy forward so the
        EP collectives line up. The round starts once every rank has joined;
        each rank's outputs land when its own forward finishes."""
        g = self.dp_group
        # Ranks free at the same instant join in descending rank order (the
        # order ASTRA-Sim polls NPUs in LLMServingSim); idle ranks join last
        # with a dummy. The last to join is handed the round first, which
        # matters only for who waits on the shared CPU-memory device.
        for e in sorted(g, key=lambda e: -e.engine_id):
            if not e.busy and e.engine_id not in self._dp_pending and e.core.scheduler.has_requests():
                outputs, executed, _ = e.step()
                self._after_step(e)
                shape = e.ctx.last_shape if executed and e.ctx.last_shape is not None and e.ctx.last_shape.seqs else None
                self._dp_pending[e.engine_id] = (outputs, shape)
        if not any(p[1] is not None or p[0] for p in self._dp_pending.values()):
            if self._dp_pending and all(not e.busy for e in g):
                # Every rank stepped and nothing was scheduled anywhere: apply
                # the outputs; the group waits for the next event.
                for e in g:
                    outputs, _ = self._dp_pending.pop(e.engine_id, ({}, None))
                    self._on_step_done(_Event(self.now, STEP_DONE, 0, "step_done", (e, outputs)))
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
        wants = [(s.total_tokens, all(x.num_new == 1 for x in s.seqs)) if s else (1, True) for s in shapes]
        padded = self._dp_padded_tokens(wants)
        from augur.cost import SeqWork, StepShape
        final = []
        for s, p in zip(shapes, padded):
            if s is None:
                self.dp_dummy_forwards += 1
                final.append(StepShape(seqs=(SeqWork(1, 1, True),), padded_tokens=p, dummy=True))
            else:
                final.append(StepShape(seqs=s.seqs, padded_tokens=p))
        per_rank = g[0].ctx.cost.round_ns(final, first_rank=first_rank)
        per_rank = self._scaled(g, per_rank)
        # The ranks meet in the round's first collective, so the slowest rank's CPU time holds up all of them.
        cpu_ns = max((e.ctx.cpu_fixed_ns + e.ctx.cpu_per_request_ns * len(s.seqs)
                      for e, s in zip(g, shapes) if s is not None), default=0)
        per_rank = [ns + cpu_ns for ns in per_rank]
        self.dp_rounds += 1
        for e, sh, ns in zip(g, final, per_rank):
            if e.step_log is not None:
                e.step_log.append((self.now, ns, sh.total_tokens, sum(x.num_new for x in sh.seqs if not x.is_decode),
                                   sh.num_decode, sh.num_prefill, sh.forward_tokens, int(sh.dummy)))
        for e, outputs, ns in zip(g, outs, per_rank):
            e.busy = True
            self._push(self.now + ns, STEP_DONE, "step_done", (e, outputs))

    @staticmethod
    def _scaled(g, per_rank):
        f = g[0].ctx.step_time_scale
        return per_rank if f == 1.0 else [round(ns * f) for ns in per_rank]

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
                if ns and eng.step_log is not None and eng.ctx.last_shape is not None:
                    sh = eng.ctx.last_shape
                    eng.step_log.append((start, ns, sh.total_tokens,
                                         sum(x.num_new for x in sh.seqs if not x.is_decode), sh.num_decode,
                                         sh.num_prefill, sh.forward_tokens, 0))
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
    def _dpa_run_rank(self, e: SimEngine) -> None:
        """Run a rank's vLLM async loop (step_with_batch_queue) until it pops
        a batch (then it waits for that batch's round) or has nothing to do."""
        for _ in range(1000):
            if e.dp_blocked_on is not None:
                return
            if not e.core.scheduler.has_requests() and not e.dp_fifo:
                return
            t0 = time.perf_counter()
            outputs, executed = e.core.step_fn()
            e.sched_cpu_s += time.perf_counter() - t0
            e.steps += 1
            launched = bool(e.ctx.launches)
            for shape in e.ctx.launch_shapes:
                r = max(e.dp_next_round, self._dpa_next_form)
                e.dp_next_round = r + 1
                self._dpa_shapes.setdefault(r, {})[e.engine_id] = shape
                e.dp_fifo.append(r)
            e.ctx.launches.clear()
            e.ctx.launch_shapes.clear()
            self._after_step(e)
            if outputs is None:
                if launched:
                    continue
                return
            r = e.dp_fifo.popleft()
            if e.engine_id in self._dpa_finished.get(r, ()):
                self._on_step_done(_Event(self.now, STEP_DONE, 0, "step_done", (e, outputs)))
                continue
            e.dp_held[r] = outputs
            e.dp_blocked_on = r
            return
        raise RuntimeError("DP async rank loop did not settle")

    def _dpa_try_form(self) -> bool:
        """Start the next GPU round once the previous one is done on every rank.
        Returns True if it settled a round with no GPU work (all batches
        empty), so ranks waiting on it can go on."""
        if self._dpa_running is not None:
            return False
        x = self._dpa_next_form
        shapes = self._dpa_shapes.get(x, {})
        if not shapes:
            return False
        if all(s is None for s in shapes.values()):
            del self._dpa_shapes[x]
            self._dpa_next_form = x + 1
            for e in self.dp_group:
                self._dpa_finished.setdefault(x, set()).add(e.engine_id)
                if e.dp_blocked_on == x:
                    e.dp_blocked_on = None
                    self._on_step_done(_Event(self.now, STEP_DONE, 0, "step_done", (e, e.dp_held.pop(x))))
            return True
        g = self.dp_group
        from augur.cost import SeqWork, StepShape
        sh = [shapes.get(e.engine_id) for e in g]
        wants = [(s.total_tokens, all(w.num_new == 1 for w in s.seqs)) if s else (1, True) for s in sh]
        padded = self._dp_padded_tokens(wants)
        final = []
        for s, p in zip(sh, padded):
            if s is None:
                self.dp_dummy_forwards += 1
                final.append(StepShape(seqs=(SeqWork(1, 1, True),), padded_tokens=p, dummy=True))
            else:
                final.append(StepShape(seqs=s.seqs, padded_tokens=p))
        per_rank = g[0].ctx.cost.round_ns(final)
        per_rank = self._scaled(g, per_rank)
        self.dp_rounds += 1
        self._dpa_running = x
        self._dpa_left = {e.engine_id for e in g}
        self._dpa_next_form = x + 1
        del self._dpa_shapes[x]
        for e, ns in zip(g, per_rank):
            self._push(self.now + ns, STEP_DONE, "dpa_done", (e, x))
        return False

    def _dpa_done(self, ev: _Event) -> None:
        e, x = ev.payload
        self._dpa_finished.setdefault(x, set()).add(e.engine_id)
        self._dpa_left.discard(e.engine_id)
        if e.dp_blocked_on == x:
            e.dp_blocked_on = None
            self._on_step_done(_Event(self.now, STEP_DONE, 0, "step_done", (e, e.dp_held.pop(x))))
        if not self._dpa_left:
            self._dpa_running = None
            self._dpa_finished.pop(x - 2, None)

    def _start_steps(self) -> None:
        q = self.lss_idle_quantum_ns
        if self._dpa:
            for _ in range(1000):
                for e in self.dp_group:
                    self._dpa_run_rank(e)
                if not self._dpa_try_form():
                    break
        elif self.dp_group:
            self._start_dp_round()
        for eng in self.engines:
            if eng.busy or eng.spec.data_parallel_size > 1:
                continue
            if eng.async_sched:
                if eng.core.scheduler.has_requests() or eng.batch_finish:
                    self._step_async(eng)
                    self._poll_when_transfers_finish(eng)
                    self._replay_spin(eng)
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
                                     sum(x.num_new for x in sh.seqs if not x.is_decode), sh.num_decode, sh.num_prefill,
                                     sh.forward_tokens, 0))
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
        at = cost.next_schedule_ns(self.now)
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
            if self.pd_link.contention == "decode":
                # Pay the link latency, then join the decode engine's shared link.
                self._push(self.now + self.pd_link.latency_ns, TRANSFER_DONE, "kv_join", (nbytes, payload))
            else:
                ns = self.pd_link.transfer_ns(nbytes)
                self._push(self.now + ns, TRANSFER_DONE, "kv_pull", payload)

    def _link_of(self, eng: SimEngine) -> SharedLink:
        link = self._links.get(eng.engine_id)
        if link is None:
            link = self._links[eng.engine_id] = SharedLink(self.pd_link.bw_gbps)
        return link

    def _reschedule(self, eng: SimEngine, link: SharedLink) -> None:
        link.version += 1
        t = link.next_finish(self.now)
        if t is not None:
            self._push(t, TRANSFER_DONE, "kv_link", (eng, link.version))

    def _on_link_event(self, ev: _Event) -> None:
        if ev.kind == "kv_join":
            nbytes, payload = ev.payload
            eng = payload[0]
            link = self._link_of(eng)
            link.advance(self.now)
            link.active[payload[1]] = [float(nbytes), payload]
            self._reschedule(eng, link)
            return
        eng, version = ev.payload
        link = self._link_of(eng)
        if version != link.version:
            return  # superseded by a later change to this link
        link.advance(self.now)
        for key in [k for k, v in link.active.items() if v[0] <= 0.5]:
            _, payload = link.active.pop(key)
            self._on_transfer_done(_Event(self.now, TRANSFER_DONE, 0, "kv_pull", payload))
        self._reschedule(eng, link)

    def run(self) -> Recorder:
        t0 = time.perf_counter()
        self._pull_entry()
        while self._q:
            self.now = self._q[0].t
            while self._q and self._q[0].t == self.now and self._q[0].prio < CAUSED_ARRIVAL:
                ev = heapq.heappop(self._q)
                if ev.prio == ARRIVAL:
                    self._on_arrival(ev)
                elif ev.kind == "dpa_done":
                    self._dpa_done(ev)
                elif ev.prio == TRANSFER_DONE:
                    if ev.kind in ("kv_join", "kv_link"):
                        self._on_link_event(ev)
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
        if self._records:
            raise RuntimeError(f"{len(self._records)} calls never finished (e.g. {next(iter(self._records))})")
        self.wall_s = time.perf_counter() - t0
        return self.recorder
