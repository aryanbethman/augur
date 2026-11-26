"""Per-call results, written in the LLMServingSim fork's request CSV format.

The first twelve columns match the fork's ``Scheduler.save_output`` exactly
(times in ns), so its analysis scripts read augur output unchanged. Extra
columns follow: session id, call index within the session, prefix-cache hit
tokens, and preemption count.
"""

from __future__ import annotations

import csv
import os
from array import array
from dataclasses import dataclass, field
from itertools import accumulate

import numpy as np

FORK_COLUMNS = ['instance id', 'request id', 'model', 'input', 'output', 'arrival', 'end_time',
                'latency', 'queuing_delay', 'TTFT', 'TPOT', 'ITL']
EXTRA_COLUMNS = ['session_id', 'sub_request_index', 'cached_tokens', 'num_preemptions', 'prefill_done']


def ensure_parent_dir(path: str | None) -> None:
    """Create the folder a result file will be written to. Called by: cli.main,
    bench.main, client.main, before the work that produces the file."""
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


_BIG = (1 << 32) - 1  # a gap that does not fit 32 bits (a preempted call's stall) is kept apart


class PackedTimes:
    """A finished call's token times as the first time and the gaps after it, 4 bytes a token instead of 8. Every
    finished call keeps its times to the end of the run, so this is what grows with the request count."""

    __slots__ = ("first", "gaps", "big")

    def __init__(self, first: int, gaps: array, big: dict) -> None:
        self.first, self.gaps, self.big = first, gaps, big  # big: index in gaps -> its real value

    def __len__(self) -> int:
        return len(self.gaps) + 1

    def deltas(self) -> list[int]:
        out = self.gaps.tolist()
        for i, g in self.big.items():
            out[i] = g
        return out

    def __iter__(self):
        return accumulate(self.deltas(), initial=self.first)

    def __getitem__(self, i: int) -> int:
        return self.first if i == 0 else list(self)[i]


def pack_times(t: array):
    """``t`` as PackedTimes, or unchanged when it has fewer than two times."""
    if len(t) < 2:
        return t
    gaps = [b - a for a, b in zip(t, t[1:])]
    big = {i: g for i, g in enumerate(gaps) if g >= _BIG}
    for i in big:
        gaps[i] = _BIG
    return PackedTimes(t[0], array("I", gaps), big)


class _Done(list):
    """Recorder.done: a call's times are packed as it is added, since none changes after it finishes."""

    def append(self, rec) -> None:
        rec.token_times = pack_times(rec.token_times)
        super().append(rec)


@dataclass(slots=True)  # a record per finished call, kept to the end of the run: no per-instance dict
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
    token_times: array = field(default_factory=lambda: array("q"))  # 8 B a token: a list of ints is 40 B, and every finished call keeps its own
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
        return t.deltas() if isinstance(t, PackedTimes) else [b - a for a, b in zip(t, t[1:])]


class Recorder:
    def __init__(self) -> None:
        self.done: list[CallRecord] = _Done()

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

    def _itl_ms(self) -> np.ndarray:
        """Every call's inter-token gaps in ms, in call order, filled in place: a Python list of them held 40 B a token
        (2.7 GB at 1,024 GPUs of ShareGPT, the run's peak). The mean must be taken before the percentile reorders it."""
        out = np.empty(sum(max(len(r.token_times) - 1, 0) for r in self.done))
        i = 0
        for r in self.done:
            t = r.token_times
            n = max(len(t) - 1, 0)
            if isinstance(t, PackedTimes) and not t.big:
                np.divide(np.frombuffer(t.gaps, dtype=np.uint32), 1e6, out=out[i:i + n], dtype=np.float64)
            else:
                out[i:i + n] = np.array(r.itl, dtype=np.int64) / 1e6
            i += n
        return out if len(out) else np.zeros(1)

    def summary(self) -> dict:
        if not self.done:
            return {"calls": 0}
        ttft = np.array([r.ttft for r in self.done]) / 1e6
        tpot = np.array([r.tpot for r in self.done]) / 1e6
        itl = self._itl_ms()
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
            "itl_ms": {"mean": itl.mean(), "p99": np.percentile(itl, 99, overwrite_input=True)},
            "output_tok_per_s": out_tokens / span_s,
            "prefix_hit_rate": cached / max(in_tokens, 1),
            "preemptions": sum(r.num_preemptions for r in self.done),
        }
