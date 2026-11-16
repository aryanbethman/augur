"""Per-call results, written in the LLMServingSim fork's request CSV format.

The first twelve columns match the fork's ``Scheduler.save_output`` exactly
(times in ns), so its analysis scripts read augur output unchanged. Extra
columns follow: session id, call index within the session, prefix-cache hit
tokens, and preemption count.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field

import numpy as np

FORK_COLUMNS = ['instance id', 'request id', 'model', 'input', 'output', 'arrival', 'end_time',
                'latency', 'queuing_delay', 'TTFT', 'TPOT', 'ITL']
EXTRA_COLUMNS = ['session_id', 'sub_request_index', 'cached_tokens', 'num_preemptions', 'prefill_done']


@dataclass
class CallRecord:
    num_id: int
    call_id: str
    instance_id: int
    model: str
    input_toks: int
    output_toks: int
    arrival_ns: int
    session_id: str | None = None
    index: int = 0
    first_sched_ns: int | None = None
    token_times: list[int] = field(default_factory=list)
    end_ns: int | None = None
    cached_tokens: int = 0
    num_preemptions: int = 0
    prefill_done_ns: int | None = None  # PD: when the prefill engine finished

    @property
    def ttft(self) -> int:
        return self.token_times[0] - self.arrival_ns

    @property
    def latency(self) -> int:
        return self.end_ns - self.arrival_ns

    @property
    def tpot(self) -> int:
        n = len(self.token_times)
        return 0 if n <= 1 else (self.latency - self.ttft) // (n - 1)

    @property
    def itl(self) -> list[int]:
        t = self.token_times
        return [b - a for a, b in zip(t, t[1:])]


class Recorder:
    def __init__(self) -> None:
        self.done: list[CallRecord] = []

    def write_csv(self, path: str) -> None:
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(FORK_COLUMNS + EXTRA_COLUMNS)
            for r in sorted(self.done, key=lambda r: r.num_id):
                w.writerow([r.instance_id, r.num_id, r.model, r.input_toks, len(r.token_times), r.arrival_ns,
                            r.end_ns, r.latency,
                            (r.first_sched_ns - r.arrival_ns) if r.first_sched_ns is not None else -1,
                            r.ttft, r.tpot, r.itl,
                            r.session_id or '', r.index, r.cached_tokens, r.num_preemptions,
                            r.prefill_done_ns if r.prefill_done_ns is not None else ''])

    def summary(self) -> dict:
        if not self.done:
            return {"calls": 0}
        ttft = np.array([r.ttft for r in self.done]) / 1e6
        tpot = np.array([r.tpot for r in self.done]) / 1e6
        itl = np.array([x for r in self.done for x in r.itl] or [0]) / 1e6
        out_tokens = sum(len(r.token_times) for r in self.done)
        start = min(r.arrival_ns for r in self.done)
        end = max(r.end_ns for r in self.done)
        span_s = max(end - start, 1) / 1e9
        in_tokens = sum(r.input_toks for r in self.done)
        cached = sum(r.cached_tokens for r in self.done)
        return {
            "calls": len(self.done),
            "sessions": len({r.session_id for r in self.done if r.session_id}),
            "sim_seconds": round(end / 1e9, 3),
            "ttft_ms": {"mean": ttft.mean(), "p50": np.median(ttft), "p99": np.percentile(ttft, 99)},
            "tpot_ms": {"mean": tpot.mean(), "p50": np.median(tpot), "p99": np.percentile(tpot, 99)},
            "itl_ms": {"mean": itl.mean(), "p99": np.percentile(itl, 99)},
            "output_tok_per_s": out_tokens / span_s,
            "prefix_hit_rate": cached / max(in_tokens, 1),
            "preemptions": sum(r.num_preemptions for r in self.done),
        }
