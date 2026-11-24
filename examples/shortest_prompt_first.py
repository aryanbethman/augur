"""Shortest-prompt-first admission: trying a scheduling idea in vLLM's real scheduler, without a GPU.

vLLM admits waiting requests in arrival order (FCFS). Here the waiting queue keeps them ordered by prompt length
instead, so a short prompt is not stuck behind a long one. Nothing else changes: batching, chunked prefill,
preemption and the KV-cache manager are vLLM's own. (Requests that already hold KV blocks, e.g. while an async KV load
finishes, wait in a separate queue that stays FCFS.) The idea touches one method, ``RequestQueue.add_request``.

Run it against vLLM's default on a recorded trace, from the augur checkout (the first command is the baseline):

    AUGUR_MODEL="--model Qwen/Qwen3-1.7B --hardware RTX4060Ti --max-num-seqs 128 --max-num-batched-tokens 2048 \\
        --max-model-len 8192 --workload ~/llmss-vllm-sim-20261114/workloads/sharegpt-qwen3-32b-300-sps10.jsonl"
    python -m augur.cli $AUGUR_MODEL
    PYTHONPATH=examples python -m augur.cli $AUGUR_MODEL \\
        --scheduler-cls shortest_prompt_first.ShortestPromptFirstAsyncScheduler

(vLLM 0.33 schedules asynchronously by default; with ``--no-async-scheduling`` name ``ShortestPromptFirstScheduler``.)
Compare the printed ``ttft_ms``, ``tpot_ms`` and ``preemptions``. On this trace (RTX 4060 Ti tables, 300 requests at
10 per second, well past what the GPU sustains) the mean TTFT falls from 53.8 s to 45.4 s, because short prompts no
longer wait behind long ones, while p99 TTFT rises from 125.6 s to 136.8 s: a long prompt keeps being passed over for
as long as shorter ones arrive.
"""

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request


class ShortestPromptQueue(FCFSRequestQueue):
    """A waiting queue ordered by prompt length; equal lengths keep arrival order.

    vLLM pops from the front, and puts preempted or skipped requests back there with ``prepend_request``, so the
    front is the only place the order is decided: new requests are inserted before the first longer prompt."""

    def add_request(self, request: Request) -> None:
        i = len(self)
        while i and self[i - 1].num_prompt_tokens > request.num_prompt_tokens:
            i -= 1
        self.insert(i, request)


class _ShortestPromptFirst:
    """Mixin: the scheduler's waiting queue is a ShortestPromptQueue."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.waiting = ShortestPromptQueue()


class ShortestPromptFirstScheduler(_ShortestPromptFirst, Scheduler):
    pass


class ShortestPromptFirstAsyncScheduler(_ShortestPromptFirst, AsyncScheduler):
    pass
