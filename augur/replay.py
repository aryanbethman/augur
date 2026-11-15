"""Replay a recorded vLLM run through augur, and compare scheduling decisions.

A step log (augur.vllm_hooks format) from real vLLM, or from augur itself,
fixes everything except vLLM's scheduler: when each request reached the
engine, and how long each step took. Replaying it through augur with those
arrival times and step durations must reproduce the log's decisions step for
step, since the scheduler is the same code. A mismatch is a bug in what augur
wraps around vLLM (executor, coordinator), not in timing.

    python -m augur.replay compare A.jsonl B.jsonl
    augur ... --workload trace.jsonl --replay-durations run.<pid>.jsonl --step-log-vllm out/replayed
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field

from augur.cost import StepShape
from augur.vllm_hooks import call_id
from augur.workload import Call, Session, Workload


def read_log(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def steps(events: list[dict]) -> list[tuple[dict, dict | None]]:
    """(sched, upd) pairs for steps that ran tokens, in order."""
    out, pending = [], {}
    for e in events:
        if e["ev"] == "sched":
            pending[e["k"]] = e
        elif e["ev"] == "upd":
            s = pending.pop(e["k"], None)
            if s is not None and s["tok"]:
                out.append((s, e))
    return out


@dataclass
class ReplayCost:
    """Step k costs what step k took in the log.

    If the engine still had unfinished requests after step k, the next step
    followed at once, so the duration runs to the next step's start (vLLM's
    loop overhead included). Otherwise it runs to step k's own update."""

    durations: list[int]
    k: int = 0

    @classmethod
    def from_log(cls, path: str) -> "ReplayCost":
        st = steps(read_log(path))
        d = []
        for i, (s, u) in enumerate(st):
            if u["left"] > 0 and i + 1 < len(st):
                d.append(st[i + 1][0]["t"] - s["t"])
            else:
                d.append(u["t"] - s["t"])
        return cls(d)

    def step_ns(self, shape: StepShape) -> int:
        if self.k >= len(self.durations):
            raise RuntimeError(f"replay ran past the {len(self.durations)} logged steps")
        d = self.durations[self.k]
        self.k += 1
        return d


@dataclass
class ReplayWorkload(Workload):
    """The trace's calls as flat requests, arriving when the log says they
    reached the engine (relative to the log's first event). Session chaining
    is off: the log already holds each call's real release time."""

    log_path: str = ""
    _arrivals: dict = field(default_factory=dict)
    _after_sched: set = field(default_factory=set)

    def __post_init__(self):
        ev = read_log(self.log_path)
        t0 = min(e["t"] for e in ev)
        last_sched_t = None
        for e in ev:
            if e["ev"] == "sched":
                last_sched_t = e["t"]
            elif e["ev"] == "add":
                cid = call_id(e["rid"])
                if cid in self._arrivals:
                    continue
                self._arrivals[cid] = e["t"] - t0
                # Logged after a step that started at this same instant: it
                # reached the engine after that schedule, so it must not join
                # that step (only augur's own logs have such exact ties).
                if e["t"] == last_sched_t:
                    self._after_sched.add(cid)

    def entries(self):
        calls: list[Call] = []
        for e in super().entries():
            if isinstance(e, Session):
                for i in range(len(e.sub_requests)):
                    calls.append(self.next_call(e, 0))
            else:
                calls.append(e)
        out = []
        for c in calls:
            t = self._arrivals.get(c.call_id)
            if t is None:
                continue  # not in this run (e.g. a shorter run of the trace)
            c.arrival_ns = t
            c.session_id = None  # released by time, not by the previous call
            c.after_step_start = c.call_id in self._after_sched
            out.append(c)
        out.sort(key=lambda c: (c.arrival_ns, c.call_id))
        missing = set(self._arrivals) - {c.call_id for c in out}
        if missing:
            raise ValueError(f"{len(missing)} logged requests are not in the trace, e.g. {next(iter(missing))}")
        yield from out


def _norm(step: dict) -> tuple:
    s = step
    return (tuple(sorted((call_id(r), n) for r, n in s["tok"].items())),
            tuple(sorted((call_id(r), n) for r, n in s["new"].items())),
            tuple(sorted(call_id(r) for r in s["pre"])))


def compare(a_path: str, b_path: str) -> dict:
    a = [s for s, _ in steps(read_log(a_path))]
    b = [s for s, _ in steps(read_log(b_path))]
    same = 0
    first = None
    for i, (x, y) in enumerate(zip(a, b)):
        if _norm(x) == _norm(y):
            same += 1
        elif first is None:
            first = i
    return {"steps_a": len(a), "steps_b": len(b), "identical_steps": same,
            "first_difference": first,
            "at_first_difference": None if first is None else {"a": _norm(a[first]), "b": _norm(b[first])}}


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 3 or argv[0] != "compare":
        print(__doc__)
        return 2
    r = compare(argv[1], argv[2])
    print(json.dumps({k: v for k, v in r.items() if k != "at_first_difference"}, indent=2))
    if r["first_difference"] is not None:
        print("first difference, step", r["first_difference"])
        print("  A:", str(r["at_first_difference"]["a"])[:400])
        print("  B:", str(r["at_first_difference"]["b"])[:400])
    return 0


if __name__ == "__main__":
    sys.exit(main())
