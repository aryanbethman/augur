"""Step logging for vLLM's own scheduler, in real vLLM and in augur alike.

Plug in with vLLM's ``scheduler_cls`` (``--scheduler-cls`` on ``vllm serve``):

    augur.vllm_hooks.LoggingScheduler        (synchronous scheduling)
    augur.vllm_hooks.LoggingAsyncScheduler   (async scheduling, vLLM's default)

These subclass vLLM's Scheduler / AsyncScheduler and only observe: every
decision is still vLLM's. Set ``AUGUR_STEP_LOG=<path>``; each process writes
``<path>.<pid>.jsonl`` with one line per event:

    {"ev": "add",   "t": ns, "rid": ..., "prompt": n}
    {"ev": "sched", "t": ns, "k": step, "tok": {rid: n}, "new": {rid: computed},
     "pre": [rid...]}
    {"ev": "upd",   "t": ns, "k": step, "out": {rid: n}, "fin": [rid...], "left": n}

``t`` comes from ``CLOCK(phase)`` (monotonic ns in real vLLM; augur points it
at its virtual clock, where an update lands at the end of its step). ``new`` holds a request's computed tokens at its first schedule,
i.e. its prefix-cache hit. ``left`` is the number of unfinished requests after
the update.
"""

from __future__ import annotations

import atexit
import json
import os
import time
from typing import Callable

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler

CLOCK: Callable[[str], int] = lambda phase: time.monotonic_ns()


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

    def _logger(self) -> _StepLog | None:
        if self._log is None:
            path = os.environ.get("AUGUR_STEP_LOG")
            if path:
                self._log = _StepLog(path)
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
                       "left": self.get_num_unfinished_requests()})
            if fin:
                log.flush()
        return outputs


class LoggingScheduler(_LoggingMixin, Scheduler):
    pass


class LoggingAsyncScheduler(_LoggingMixin, AsyncScheduler):
    pass


def call_id(rid: str) -> str:
    """The workload call id inside a request id.

    augur's trace client sends ``augur.<call_id>.``, which real vLLM wraps as
    ``cmpl-augur.<call_id>.-0`` (plus any internal suffix). Inside augur the
    request id is the call id, with ``@p`` / ``@d`` for PD halves."""
    if "augur." in rid:
        return rid.split("augur.", 1)[1].split(".", 1)[0]
    return rid.split("@", 1)[0]
