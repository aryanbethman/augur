"""Scheduler policies as vLLM Scheduler subclasses, one file, loaded unchanged by ``vllm serve --scheduler-cls`` and by
``python -m augur.cli --scheduler-cls`` (``PYTHONPATH=examples``; the ``...AsyncScheduler`` names with async scheduling).

FCFS is vLLM's default and needs no class. Each policy below only decides the order of the waiting queue; batching,
chunked prefill, preemption and the KV manager stay vLLM's. They read only the Request, never the GPU or the model.

* ShortestPromptFirst   admit the shortest prompt first (the idea of examples/shortest_prompt_first.py).
* DeadlineFirst         earliest TTFT deadline first, deadline = arrival + SLO_BASE_S + SLO_PER_TOKEN_S * prompt tokens:
                        a long prompt is allowed a longer wait, so it overtakes short ones only once they are younger.
* AgingShortestFirst    shortest prompt first, but every admission of another request takes AGE tokens off the prompt
                        length of all those still waiting, so a long prompt cannot be passed over for ever.
"""

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.core.sched.scheduler import Scheduler

SLO_BASE_S, SLO_PER_TOKEN_S = 5.0, 0.01
AGE = 4.0


class _SortedQueue(FCFSRequestQueue):
    """Waiting queue kept sorted by ``key(request)``; ties keep arrival order. vLLM pops from the front and returns
    preempted or skipped requests there with ``prepend_request``, so the front is where the order is decided."""

    def key(self, request):
        raise NotImplementedError

    def add_request(self, request) -> None:
        k, i = self.key(request), len(self)
        while i and self.key(self[i - 1]) > k:
            i -= 1
        self.insert(i, request)


class ShortestPromptQueue(_SortedQueue):
    def key(self, request):
        return request.num_prompt_tokens


class DeadlineQueue(_SortedQueue):
    def key(self, request):
        return request.arrival_time + SLO_BASE_S + SLO_PER_TOKEN_S * request.num_prompt_tokens


class AgingQueue(FCFSRequestQueue):
    """Smallest ``prompt tokens - AGE * admissions since it began waiting`` is served next (clock-free)."""

    def __init__(self):
        super().__init__()
        self.pops, self.since = 0, {}

    def _best(self):
        return min(self, key=lambda r: r.num_prompt_tokens - AGE * (self.pops - self.since.get(r.request_id, self.pops)))

    def add_request(self, request) -> None:
        self.since.setdefault(request.request_id, self.pops)
        self.append(request)

    def peek_request(self):
        if not self:
            raise IndexError("peek from an empty queue")
        return self._best()

    def pop_request(self):
        r = self.peek_request()
        self.remove(r)
        self.pops += 1
        self.since.pop(r.request_id, None)
        return r

    def prepend_request(self, request) -> None:
        self.since.setdefault(request.request_id, self.pops)
        self.appendleft(request)

    def prepend_requests(self, requests) -> None:
        for r in requests:
            self.prepend_request(r)


def _scheduler(name, queue_cls, base):
    def init(self, *args, **kwargs):
        base.__init__(self, *args, **kwargs)
        self.waiting = queue_cls()
    return type(name, (base,), {"__init__": init, "__module__": __name__})


ShortestPromptFirstScheduler = _scheduler("ShortestPromptFirstScheduler", ShortestPromptQueue, Scheduler)
ShortestPromptFirstAsyncScheduler = _scheduler("ShortestPromptFirstAsyncScheduler", ShortestPromptQueue, AsyncScheduler)
DeadlineFirstScheduler = _scheduler("DeadlineFirstScheduler", DeadlineQueue, Scheduler)
DeadlineFirstAsyncScheduler = _scheduler("DeadlineFirstAsyncScheduler", DeadlineQueue, AsyncScheduler)
AgingShortestFirstScheduler = _scheduler("AgingShortestFirstScheduler", AgingQueue, Scheduler)
AgingShortestFirstAsyncScheduler = _scheduler("AgingShortestFirstAsyncScheduler", AgingQueue, AsyncScheduler)
