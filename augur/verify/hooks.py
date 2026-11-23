"""Step logging for vLLM's own scheduler, in real vLLM and in augur alike.

Plug in with vLLM's ``scheduler_cls`` (``--scheduler-cls`` on ``vllm serve``):

    augur.verify.hooks.LoggingScheduler        (synchronous scheduling)
    augur.verify.hooks.LoggingAsyncScheduler   (async scheduling, vLLM's default)

These subclass vLLM's Scheduler / AsyncScheduler and only observe: every
decision is still vLLM's. Set ``AUGUR_STEP_LOG=<path>``; each process writes
``<path>.<pid>.jsonl`` with one line per event:

    {"ev": "add",   "t": ns, "rid": ..., "prompt": n}
    {"ev": "sched", "t": ns, "k": step, "tok": {rid: n}, "new": {rid: computed},
     "pre": [rid...]}
    {"ev": "upd",   "t": ns, "k": step, "out": {rid: n}, "fin": [rid...], "left": n,
     "jobs": [id...], "recv": [rid...], "xfer": {"load": [bytes, s, n], "store": [...]}}

(the last three only when the KV connector's worker reported something). A DP rank of a real server (vLLM's
DPEngineCoreProc, MoE models) also writes one record per loop iteration, dummy forwards included, as augur's own DP
runs do (see _log_dp_rounds):

    {"ev": "round", "t": ns, "dur": ns, "dummy": bool}

With several engines, each logs to ``<path>.e<id>.<pid>.jsonl``.

``t`` comes from ``CLOCK(phase)`` (monotonic ns in real vLLM; augur points it
at its virtual clock, where an update lands at the end of its step). ``new`` holds a request's computed tokens at its first schedule,
i.e. its prefix-cache hit. ``left`` is the number of unfinished requests after
the update.
"""

from __future__ import annotations

import atexit
import json
import multiprocessing
import os
import re
import time
from typing import Callable

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler

CLOCK: Callable[[str], int] = lambda phase: time.monotonic_ns()

# Several engines in one process (augur's multi-engine runs): the engine a scheduler
# being built belongs to. Its log is <path>.e<id>.<pid>.jsonl; None (a single engine)
# keeps <path>.<pid>.jsonl. Set by SimEngine before it builds its core. A real server's
# DP ranks are told apart by their process name instead.
NEXT_ENGINE_TAG: int | None = None


def _dp_rank_of_process() -> int | None:
    """The DP rank of this vLLM engine process (named EngineCore_DP<rank> when data_parallel_size > 1)."""
    m = re.fullmatch(r"EngineCore_DP(\d+)", multiprocessing.current_process().name)
    return int(m.group(1)) if m else None


class _StepLog:
    def __init__(self, path: str):
        self._f = open(f"{path}.{os.getpid()}.jsonl", "a", buffering=1 << 20)
        atexit.register(self._f.flush)

    def write(self, rec: dict) -> None:
        self._f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self._n = getattr(self, "_n", 0) + 1
        if self._n % 2000 == 0:  # a killed server keeps (almost) all of its log
            self._f.flush()

    def flush(self) -> None:
        self._f.flush()


class _LoggingMixin:
    _log: _StepLog | None = None
    _k = 0
    _k_of: dict | None = None  # id(scheduler_output) -> step number
    _round_t: int | None = None  # a real DP rank: start of the loop iteration under way (see _log_dp_rounds)
    _round_dummy = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        rank = _dp_rank_of_process()
        self._engine_tag = NEXT_ENGINE_TAG if rank is None else rank
        if rank is not None and os.environ.get("AUGUR_STEP_LOG"):
            _log_dp_rounds()

    def _logger(self) -> _StepLog | None:
        if self._log is None:
            path = os.environ.get("AUGUR_STEP_LOG")
            if path:
                tag = getattr(self, "_engine_tag", None)
                self._log = _StepLog(path if tag is None else f"{path}.e{tag}")
        return self._log

    def add_request(self, request) -> None:
        super().add_request(request)
        log = self._logger()
        if log:
            log.write({"ev": "add", "t": CLOCK("add"), "rid": request.request_id,
                       "prompt": request.num_prompt_tokens})

    def schedule(self, *args, **kwargs):
        t = CLOCK("sched")
        so = super().schedule(*args, **kwargs)
        log = self._logger()
        if log:
            self._k += 1
            if self._round_t is None:
                self._round_t = t
            if self._k_of is None:
                self._k_of = {}
            # Async scheduling updates batch k after scheduling k+1: tag the
            # update with the step of the output it belongs to.
            self._k_of[id(so)] = self._k
            log.write({
                "ev": "sched", "t": t, "k": self._k,
                "tok": dict(so.num_scheduled_tokens),
                "new": {nr.req_id: nr.num_computed_tokens for nr in so.scheduled_new_reqs},
                "pre": sorted(so.preempted_req_ids or ()),
            })
        return so

    def update_from_output(self, scheduler_output, model_runner_output):
        outputs = super().update_from_output(scheduler_output, model_runner_output)
        log = self._logger()
        if log:
            out, fin = {}, []
            for eco in outputs.values():
                for o in eco.outputs:
                    if o.new_token_ids:
                        out[o.request_id] = len(o.new_token_ids)
                    if o.finish_reason is not None:
                        fin.append(o.request_id)
            k = (self._k_of or {}).pop(id(scheduler_output), self._k)
            log.write({"ev": "upd", "t": CLOCK("upd"), "k": k, "out": out, "fin": sorted(fin),
                       "left": self.get_num_unfinished_requests(), **_kv_report(model_runner_output)})
            if fin or not self.get_num_unfinished_requests():
                log.flush()  # an idle engine may be killed before atexit runs
        return outputs


def _log_dp_rounds() -> None:
    """Make a real DP rank log its rounds, which the scheduler alone cannot see.

    vLLM's DP engine loop (DPEngineCoreProc.run_busy_loop, MoE models only: dense DP ranks are independent
    engines) steps in lockstep with its peers: an iteration where nothing was scheduled runs a dummy forward
    outside the scheduler (execute_dummy_batch), and each iteration ends in _has_global_unfinished_reqs. Both are
    wrapped here on DPEngineCoreProc (and only there, once per process) to log each iteration as a round: from its start (the previous iteration's end while the group runs, else its first
    schedule or dummy) to the end of its sync, which is where a rank waits for its peers."""
    from vllm.v1.engine.core import DPEngineCoreProc
    if getattr(DPEngineCoreProc, "_augur_rounds", False):
        return
    run_dummy = DPEngineCoreProc.execute_dummy_batch
    sync = DPEngineCoreProc._has_global_unfinished_reqs

    def execute_dummy_batch(self):
        sch = self.scheduler
        if sch._round_t is None:
            sch._round_t = CLOCK("round")
        sch._round_dummy = True
        return run_dummy(self)

    def has_global_unfinished_reqs(self, local_unfinished):
        running = sync(self, local_unfinished)
        sch = self.scheduler
        end = CLOCK("round")
        log = sch._logger()
        if log and sch._round_t is not None:
            log.write({"ev": "round", "t": sch._round_t, "dur": end - sch._round_t, "dummy": sch._round_dummy})
            if not local_unfinished:
                log.flush()  # an idle rank's dummy rounds go on until the server is killed
        sch._round_t, sch._round_dummy = (end if running else None), False
        return running

    DPEngineCoreProc.execute_dummy_batch = execute_dummy_batch
    DPEngineCoreProc._has_global_unfinished_reqs = has_global_unfinished_reqs
    DPEngineCoreProc._augur_rounds = True


def _kv_report(model_runner_output) -> dict:
    """What the KV connector's worker reported in this step's output: finished
    offload jobs, requests whose KV arrived, and bytes / seconds moved each
    way (offload). Empty when it reported nothing."""
    kvo = getattr(model_runner_output, "kv_connector_output", None)
    if kvo is None:
        return {}
    rec = {}
    worker_meta = kvo.kv_connector_worker_meta
    if hasattr(worker_meta, "metadata"):
        # A MultiConnector wraps each connector's metadata: the offload connector's is the one that logs.
        worker_meta = next((m for m in worker_meta.metadata if hasattr(m, "completed_jobs")), None)
    if getattr(worker_meta, "completed_jobs", None):
        rec["jobs"] = sorted(worker_meta.completed_jobs)
    if kvo.finished_recving:
        rec["recv"] = sorted(kvo.finished_recving)
    stats = getattr(worker_meta, "transfer_stats", None)
    if stats is not None and not stats.is_empty():
        rec["xfer"] = {"load": [stats.load.bytes, stats.load.time, len(stats.load.sizes)],
                       "store": [stats.store.bytes, stats.store.time, len(stats.store.sizes)]}
    return rec


class LoggingScheduler(_LoggingMixin, Scheduler):
    pass


class LoggingAsyncScheduler(_LoggingMixin, AsyncScheduler):
    pass
