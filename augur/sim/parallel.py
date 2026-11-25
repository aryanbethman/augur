"""Engines in worker processes, with the output of one process.

The main process keeps the workload, the router and the recorder, and runs the usual event queue for the events
that involve no engine: arrivals, session calls, and the end of the front end (a call reaching its engine).
Workers (forked, so they inherit the built engines) each own a share of the engines and run the coordinator's own
loop on them (``_Shard``). Main tells a worker which calls reach which engine and when, and asks it to advance to a
virtual time E, meaning: every instant before E done, none at E. The worker answers with what happened in
between: step landings, in the order the single-process queue would have applied them, and what the router needs.

Why the order is the same. Engines meet only through the router and through sessions, and both act on main:

* An event of the main queue gets the key of the instant that pushed it (time, phase, ...), which is the order of
  the single-process queue's sequence numbers. Landings are merged across workers by that same key.
* A completion at time t makes the router's release at t + delivery and a session's next call at
  t + delivery + tool wait. So once every completion before H is known, nothing unknown can reach the router or the
  queue before H + delivery: that is how far main may run ahead of the workers.
* ``Router.needs`` says what a placement reads. "none": nothing from the engines (round-robin, hashes, ...), so
  main routes a whole window at once. "completions": in-flight counts, which are exact inside that window.
  "engines": the engines' queue lengths and KV events, so the workers stop at every arrival instant and report.

An engine at an arrival's instant is seen as the single-process loop sees it: its events before that instant are
done, those at it are not (arrivals come first within an instant).
"""

from __future__ import annotations

import gc
import heapq
import itertools
import multiprocessing as mp
import time
import traceback
from types import SimpleNamespace
from array import array
from collections import defaultdict

from augur.sim.coordinator import ARRIVAL, Coordinator, _Event, install_step_log_clock
from augur.trace.metrics import CallRecord

INF = 1 << 62
WINDOW_NS = 1_000_000_000  # without sessions or in-flight counts to wait for, how much arrival time main ships at once


def unsupported(a) -> str | None:
    """Why this configuration cannot use --workers > 1, or None. Called by: cli.main."""
    # shortcut: DP and PD engines meet outside the router (a DP group's lockstep rounds, KV pulls), so main would have
    # to sync with them; lift by keeping a DP group in one worker and running ahead by the minimum KV transfer time.
    if a.dp > 1 or a.prefill_engines or a.decode_engines:
        return "--workers > 1 does not support --dp or prefill/decode engines yet"
    # shortcut: the fabric and the replayed logs are shared state across engines (and ASTRA-Sim is a process of its own).
    if a.contention == "on" or a.replay_durations or a.replay_arrivals or a.cost == "llmservingsim" or a.lss_loop_compat:
        return "--workers > 1 does not support --contention on, replay, --cost llmservingsim or --lss-loop-compat"
    return None


class RemoteEngine:
    """An engine of a worker as main sees it: the id, the load the router reads, and what the run's summary and step
    logs take from it (filled when the workers finish). ``first`` is one real engine of the run, for what is the same
    on all of them: config, cost model, block hasher."""

    def __init__(self, engine_id: int, first) -> None:
        self.engine_id, self.spec, self.max_model_len = engine_id, first.spec, first.max_model_len
        self.vllm_config, self.async_sched, self.block_hasher = first.vllm_config, first.async_sched, first.block_hasher
        self.ctx = SimpleNamespace(cost=first.ctx.cost, offload=None, output_ids={}, recorded=set())
        self.busy, self.idle_since, self.admitting = False, None, 0
        self.steps, self.sched_cpu_s, self.step_log = 0, 0.0, None

    @property
    def vllm_engine_id(self) -> str:
        return f"augur-{self.engine_id}"


class _NoWorkload:
    def entries(self):
        return iter(())


class _Tap:
    """The router as a worker sees it: remembers what a landing tells the router, for main to tell the real one."""

    def __init__(self, kv: bool) -> None:
        self.kv, self.events, self.firsts = kv, None, []

    def observe(self, engine_id, events) -> None:
        self.events = events

    def first_token(self, call, at_ns) -> None:
        if self.kv:
            self.firsts.append(call.call_id)

    def release(self, call, at_ns=0) -> None:
        pass


class _Shard(Coordinator):
    """The coordinator's loop over some of the engines. Calls arrive as "wadmit" events from main; a step landing is
    queued in ``landings`` as (time, phase, push time, engine, KV events, first tokens, finished calls) for main."""

    def __init__(self, engines, delivery_ns: int, kv: bool) -> None:
        self._tap = _Tap(kv)
        super().__init__(engines, self._tap, _NoWorkload(), delivery_latency_ns=delivery_ns)
        self._seq = itertools.count(1)  # 0 marks the landings the loop applies inline
        self.by_id = {e.engine_id: e for e in engines}
        self.queued = dict.fromkeys(self.by_id, 0)  # calls in the queue not yet in their engine
        self._pushed_at: dict[int, int] = {}
        self.landings: list = []
        self._done: list = []

    def _push(self, t, prio, kind, payload) -> None:
        if kind == "step_done":
            self._pushed_at[payload[0].engine_id] = self.now
        super()._push(t, prio, kind, payload)

    def _on_arrival(self, ev) -> None:
        if ev.kind != "wadmit":
            return super()._on_arrival(ev)
        eng, call = ev.payload
        self.queued[eng.engine_id] -= 1
        self._records[call.call_id] = CallRecord(num_id=0, call_id=call.call_id, instance_id=eng.engine_id,
                                                 model=eng.spec.model_name, input_toks=len(call.prompt_ids),
                                                 output_toks=call.output_toks, arrival_ns=0)
        self._calls[call.call_id] = call
        self._admit(eng, call)

    def _on_step_done(self, ev) -> None:
        eid = ev.payload[0].engine_id
        # A landing the loop applies inline (seq 0) comes after the queued ones of its instant, in engine order;
        # the queued ones in the order their steps were started.
        key = (ev.t, 3, 0, eid) if ev.seq == 0 else (ev.t, 2, self._pushed_at.pop(eid), eid)
        self._tap.events, self._tap.firsts, self._done = None, [], []
        super()._on_step_done(ev)
        if self._tap.events or self._tap.firsts or self._done:
            self.landings.append((*key, self._tap.events, self._tap.firsts, self._done))

    def _finish(self, call_id, rec, eng) -> None:
        self._done.append((call_id, rec.first_sched_ns, rec.cached_tokens, rec.num_preemptions,
                           array("q", rec.token_times)))
        del self._calls[call_id], self._records[call_id]
        eng.ctx.output_ids.pop(call_id, None)
        eng.ctx.recorded.discard(call_id)


def _serve(conn, build, ids, delivery_ns: int, kv: bool) -> None:
    try:
        engines = build(ids)
        shard = _Shard(engines, delivery_ns, kv)
        install_step_log_clock(shard)
        while True:
            msg = conn.recv()
            if msg[0] == "adv":
                _, until, admits, loads = msg
                for t, eid, call in admits:
                    shard.queued[eid] += 1
                    shard._push(t, ARRIVAL, "wadmit", (shard.by_id[eid], call))
                shard._drain(None if until >= INF else until)
                landings, shard.landings = shard.landings, []
                conn.send((landings, {e.engine_id: type(e).load(e) + shard.queued[e.engine_id] for e in engines}  # not main's patched load
                           if loads else None, shard._q[0].t if shard._q else INF))
                continue
            for e in engines:
                log = getattr(e.core.scheduler, "_log", None)
                if log is not None:
                    log.flush()  # a forked process does not run atexit
            cost = engines[0].ctx.cost
            conn.send([(e.engine_id, e.steps, e.sched_cpu_s, e.step_log,
                        e.ctx.offload.job_ns if e.ctx.offload is not None else None) for e in engines]
                      + [getattr(cost, "steps_beyond_profile", None)])
            return
    except BaseException as e:  # noqa: BLE001 - main re-raises it
        try:
            conn.send(("err", e))
        except Exception:
            conn.send(("err", RuntimeError(traceback.format_exc())))


class ParallelCoordinator(Coordinator):
    def __init__(self, *args, workers: int, build, **kw) -> None:
        super().__init__(*args, **kw)
        self._build = build  # ids -> the engines of that worker, built in it
        self.workers = min(workers, len(self.engines))
        self.needs = getattr(self.router, "needs", "engines")
        self._ctx = (0, 0, 0, 0)  # the instant being applied: what a pushed event is keyed by
        self._owner = {e.engine_id: e.engine_id % self.workers for e in self.engines}
        self._out: list[list] = [[] for _ in range(self.workers)]  # calls to ship, per worker
        self._outmin = INF
        self._next_event = [INF] * self.workers  # each worker's next queued event
        self._load = defaultdict(int)
        self._unsent = defaultdict(int)
        self._sessions_in_flight = 0
        self._limit = INF  # main may apply queue events before this
        for e in self.engines:  # what a router reads, as the workers last reported
            e.load = lambda e=e: self._load[e.engine_id] + e.admitting + self._unsent[e.engine_id]

    def _push(self, t, prio, kind, payload) -> None:
        heapq.heappush(self._q, _Event(t, prio, (*self._ctx, next(self._seq)), kind, payload))

    def _admit(self, eng, call) -> None:
        self._out[self._owner[eng.engine_id]].append((self.now, eng.engine_id, call))
        self._unsent[eng.engine_id] += 1
        self._outmin = min(self._outmin, self.now)

    def _release(self, call, exact: bool = True) -> None:
        if call.session_id is not None:
            # shortcut: with no delivery latency a session's next call can follow a completion at the same instant,
            # which only the single-process loop orders; lift by syncing workers at every completion then.
            if self.delivery_latency_ns == 0:
                raise ValueError("--workers > 1 needs --delivery-latency-ms > 0 for sessions")
            self._sessions_in_flight += 1
            self._limit = min(self._limit, self.now + self.delivery_latency_ns)
        super()._release(call, exact)

    def _finish(self, call_id, rec, eng) -> None:
        if self._calls[call_id].session_id is not None:
            self._sessions_in_flight -= 1
        super()._finish(call_id, rec, eng)

    def _apply_main_events(self, before: int) -> None:
        while self._q and self._q[0].t < min(before, self._limit):
            ev = heapq.heappop(self._q)
            self.now = ev.t
            self._ctx = (ev.t, 0, 0, 0)
            self._on_arrival(ev)

    def _advance(self, until: int) -> None:
        """Workers run every instant before `until`; their landings are applied here, in the single-process order."""
        for conn, out in zip(self._conns, self._out):
            conn.send(("adv", until, out, self.needs == "engines"))
        self._out = [[] for _ in range(self.workers)]
        self._unsent.clear()
        self._outmin = INF
        landings = []
        for i, conn in enumerate(self._conns):
            lands, loads, self._next_event[i] = self._recv(conn)
            landings += lands
            if loads:
                self._load.update(loads)
        landings.sort(key=lambda land: land[:4])
        for t, phase, at, eid, events, firsts, done in landings:
            self.now, self._ctx = t, (t, phase, at, eid)
            if events:
                self.router.observe(eid, events)
            for cid in firsts:
                self.router.first_token(self._calls[cid], t)
            for cid, first_sched, cached, preempted, times in done:
                rec = self._records[cid]
                rec.first_sched_ns, rec.cached_tokens, rec.num_preemptions = first_sched, cached, preempted
                rec.token_times = times.tolist()
                self._finish(cid, rec, self._call_engine[cid])

    @staticmethod
    def _recv(conn):
        msg = conn.recv()
        if isinstance(msg, tuple) and msg and msg[0] == "err":
            raise msg[1]
        return msg

    def _loop(self) -> None:
        delivery = self.delivery_latency_ns
        if self.needs == "completions" and not delivery:
            # shortcut: a router counting calls in flight needs a delivery latency to know how far it may run ahead.
            raise ValueError(f"--workers > 1 with --router {type(self.router).__name__} needs --delivery-latency-ms > 0")
        horizon = 0
        while True:
            main_next = self._q[0].t if self._q else INF
            first = min(main_next, min(self._next_event), self._outmin)
            if first >= INF:
                return
            horizon = max(horizon, first)
            ahead = delivery if self._sessions_in_flight or self.needs == "completions" else INF
            if self.needs == "engines":
                until = min(main_next, horizon + ahead)
                self._advance(until)
                horizon, self._limit = until, INF
                self._apply_main_events(until + 1)  # the instant `until` itself, now that the engines are there
                continue
            until = INF if main_next >= INF and ahead >= INF else horizon + min(ahead, WINDOW_NS)
            self._limit = until
            self._apply_main_events(INF)
            until = self._limit
            self._advance(until)
            horizon = until

    def _start(self) -> None:
        ctx = mp.get_context("fork")
        kv = hasattr(self.router, "first_token")
        self._conns, self._procs = [], []
        gc.freeze()  # the workers share the built engines copy-on-write
        for i in range(self.workers):
            mine = [e.engine_id for e in self.engines if self._owner[e.engine_id] == i]
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_serve, args=(child, self._build, mine, self.delivery_latency_ns, kv), daemon=True)
            p.start()
            child.close()
            self._conns.append(parent)
            self._procs.append(p)

    def _stop(self, collect: bool) -> None:
        for conn in self._conns:
            if collect:
                conn.send(("fin",))
        if collect:
            by_id = {e.engine_id: e for e in self.engines}
            beyond = None
            for conn in self._conns:
                *rows, extra = self._recv(conn)
                for eid, steps, cpu, log, jobs in rows:
                    e = by_id[eid]
                    e.steps, e.sched_cpu_s = steps, cpu
                    if log is not None:
                        e.step_log = log
                    if jobs is not None:
                        e.ctx.offload = SimpleNamespace(job_ns=jobs)
                if extra is not None:  # each worker counts from the value it inherited
                    base = self._beyond0
                    beyond = (beyond or base) + extra - base
            if beyond is not None:
                self.engines[0].ctx.cost.steps_beyond_profile = beyond
        for p in self._procs:
            p.join(5) if collect else p.terminate()

    def run(self):
        t0 = time.perf_counter()
        self._beyond0 = getattr(self.engines[0].ctx.cost, "steps_beyond_profile", None) or 0
        self._start()
        try:
            self._pull_entry()
            self._loop()
        except BaseException:
            self._stop(False)
            raise
        self._stop(True)
        if self._records:
            raise RuntimeError(f"{len(self._records)} calls never finished (e.g. {next(iter(self._records))})")
        self.wall_s = time.perf_counter() - t0
        return self.recorder
