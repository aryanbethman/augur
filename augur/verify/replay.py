"""Replay a recorded vLLM run through augur, and compare scheduling decisions.

A step log (augur.verify.hooks format) from real vLLM, or from augur itself,
fixes everything except vLLM's scheduler: when each request reached the
engine, and how long each step took. Replaying it through augur with those
arrival times and step durations must reproduce the log's decisions step for
step, since the scheduler is the same code. A mismatch is a bug in what augur
wraps around vLLM (executor, coordinator), not in timing.

    python -m augur.verify.replay compare A.jsonl B.jsonl
    augur ... --workload trace.jsonl --replay-durations run.<pid>.jsonl --step-log-vllm out/replayed
"""

from __future__ import annotations

import bisect
import json
import sys
from dataclasses import dataclass, field

from augur.cost import StepShape
from augur.trace.workload import Call, Session, Workload, call_id


def read_log(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def is_async(events: list[dict]) -> bool:
    """True if a step was scheduled before the previous step's update."""
    pending = 0
    for e in events:
        if e["ev"] == "sched":
            pending += 1
            if pending > 1:
                return True
        elif e["ev"] == "upd":
            pending = max(pending - 1, 0)
    return False


def scheduled(events: list[dict]) -> list[dict]:
    """Every step that scheduled tokens, in order, with or without a logged update."""
    return sorted((e for e in events if e["ev"] == "sched" and e["tok"]), key=lambda e: e["k"])


def _unlogged_tail(events: list[dict]) -> int:
    """Scheduled steps whose update the log lacks (an async engine's last batch
    when the server was killed before flushing)."""
    upd = {e["k"] for e in events if e["ev"] == "upd"}
    return sum(1 for e in scheduled(events) if e["k"] not in upd)


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
    starts: list[int] = field(default_factory=list)  # each step's logged start, log time 0 = first event
    # Speculative decoding: tokens each request emitted in each step, and the
    # tokens it was scheduled in each step (1 + drafts for a decode), by call id.
    outs: list[dict] = field(default_factory=list)
    toks: list[dict] = field(default_factory=list)
    # KV offload: job id -> when the logged worker reported it finished (the
    # start of the step whose output carried it; the worker polls then).
    kv_done: dict[int, int] = field(default_factory=dict)
    # Every logged schedule, empty ones too (log time 0 = first event): an
    # empty step can still matter (it hands a KV connector new requests).
    sched_times: list[int] = field(default_factory=list)
    k: int = 0
    # DP (augur's own logs): this rank's duration in each round it took part in, dummy forwards too.
    rounds: list[int] = field(default_factory=list)
    round_k: int = 0

    def next_round_ns(self) -> int:
        if self.round_k >= len(self.rounds):
            raise RuntimeError(f"replay ran past the {len(self.rounds)} logged DP rounds")
        d = self.rounds[self.round_k]
        self.round_k += 1
        return d

    def next_schedule_ns(self, now_ns: int) -> int | None:
        """The first logged schedule at or after now_ns."""
        i = bisect.bisect_left(self.sched_times, now_ns)
        return self.sched_times[i] if i < len(self.sched_times) else None

    @classmethod
    def from_logs(cls, paths: list[str]) -> list["ReplayCost"]:
        """One cost per engine log. Several engines share one time zero: the
        earliest event of any log."""
        t0 = min(e["t"] for p in paths if p for e in read_log(p))
        return [cls.from_log(p, t0) if p else cls([]) for p in paths]

    @classmethod
    def from_log(cls, path: str, t0: int | None = None) -> "ReplayCost":
        ev = read_log(path)
        t0 = min(e["t"] for e in ev) if t0 is None else t0
        st = steps(ev)
        outs = [{call_id(r): n for r, n in u["out"].items()} for _, u in st]
        toks = [{call_id(r): n for r, n in s["tok"].items()} for s, _ in st]
        sched_t = {e["k"]: e["t"] for e in ev if e["ev"] == "sched"}
        kv_done = {jid: sched_t[e["k"]] - t0 for e in ev if e["ev"] == "upd" for jid in e.get("jobs", ())}
        sched_times = sorted(t - t0 for t in sched_t.values())
        rounds = [e["dur"] for e in ev if e["ev"] == "round"]
        if is_async(ev):
            # The GPU runs batches back to back: batch k starts when it was
            # scheduled or when batch k-1 finished, whichever is later, and
            # finishes when its update is applied.
            d, prev_finish = [], None
            for s, u in st:
                start = s["t"] if prev_finish is None else max(s["t"], prev_finish)
                d.append(u["t"] - start)
                prev_finish = u["t"]
            # A batch scheduled but never logged as finished still runs; price it
            # like the last one.
            for _ in range(_unlogged_tail(ev) if d else 0):
                d.append(d[-1])
            return cls(d, [s["t"] - t0 for s, _ in st], outs, toks, kv_done, sched_times, rounds=rounds)
        d = []
        for i, (s, u) in enumerate(st):
            if u["left"] > 0 and i + 1 < len(st):
                d.append(st[i + 1][0]["t"] - s["t"])
            else:
                d.append(u["t"] - s["t"])
        return cls(d, [s["t"] - t0 for s, _ in st], outs, toks, kv_done, sched_times, rounds=rounds)

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
    log_paths: list | None = None  # several engines' logs (their requests reach the cluster at the earliest add)
    _arrivals: dict = field(default_factory=dict)
    _after_sched: set = field(default_factory=set)

    def __post_init__(self):
        super().__post_init__()
        logs = [read_log(p) for p in (self.log_paths or [self.log_path]) if p]
        t0 = min(e["t"] for ev in logs for e in ev)
        for ev in logs:
            last_sched_t = None
            for e in ev:
                if e["ev"] == "sched":
                    last_sched_t = e["t"]
                elif e["ev"] == "add":
                    if e["rid"].endswith("@d"):
                        continue  # a decode engine's request follows its prefill (PD): not a client arrival
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


def resolve_logs(spec: str, n_engines: int) -> list[str]:
    """The step logs of an n-engine run, in engine order: one path (n = 1), a comma
    separated list, or the --step-log-vllm prefix of a run (<prefix>.e<id>.<pid>.jsonl)."""
    import glob
    if n_engines == 1:
        return [spec]
    if "," in spec:
        paths = spec.split(",")
        if len(paths) != n_engines:
            raise ValueError(f"--replay-durations lists {len(paths)} logs for {n_engines} engines")
        return paths
    paths = [_engine_log(spec, i) for i in range(n_engines)]
    if not any(paths):
        raise ValueError(f"--replay-durations {spec}: no logs {spec}.e<id>.<pid>.jsonl")
    return paths


def _engine_log(prefix: str, i: int) -> str:
    import glob
    found = sorted(glob.glob(f"{glob.escape(prefix)}.e{i}.*.jsonl"))
    if len(found) > 1:
        raise ValueError(f"{prefix}: expected one log {prefix}.e{i}.<pid>.jsonl, found {len(found)}")
    return found[0] if found else None  # an engine that was never given a request wrote no log


class ReplayRouter:
    """Places each call on the engine whose log holds its request: the entry router
    by a call's first (non-decode) add, the decode router by its @d add (PD).
    Called by: Coordinator._pick."""

    def __init__(self, where: dict[str, int]) -> None:
        self.where = where

    @classmethod
    def from_logs(cls, paths: list[str]) -> tuple["ReplayRouter", "ReplayRouter"]:
        entry, decode = {}, {}
        for i, p in enumerate(paths):
            for e in (read_log(p) if p else ()):
                if e["ev"] != "add":
                    continue
                (decode if e["rid"].endswith("@d") else entry).setdefault(call_id(e["rid"]), i)
        return cls(entry), cls(decode)

    def route(self, call, engines, now_ns: int) -> int:
        return self.where[call.call_id]

    def release(self, call, at_ns: int = 0) -> None:
        return None


def _norm(step: dict) -> tuple:
    s = step
    return (tuple(sorted((call_id(r), n) for r, n in s["tok"].items())),
            tuple(sorted((call_id(r), n) for r, n in s["new"].items())),
            tuple(sorted(call_id(r) for r in s["pre"])))


def compare(a_path: str | None, b_path: str | None) -> dict:
    a = scheduled(read_log(a_path)) if a_path else []  # None: an engine that wrote no log
    b = scheduled(read_log(b_path)) if b_path else []
    same = 0
    first = None
    for i, (x, y) in enumerate(zip(a, b)):
        if _norm(x) == _norm(y):
            same += 1
        elif first is None:
            first = i
    if first is None and len(a) != len(b):
        first = min(len(a), len(b))  # one log stops early: its missing step is the difference
    at = None if first is None else {"a": _norm(a[first]) if first < len(a) else None,
                                     "b": _norm(b[first]) if first < len(b) else None}
    return {"steps_a": len(a), "steps_b": len(b), "identical_steps": same,
            "first_difference": first, "at_first_difference": at}


def compare_runs(a_prefix: str, b_prefix: str) -> dict[int, dict]:
    """compare() for every engine of two multi-engine runs (their --step-log-vllm
    prefixes), by engine id."""
    import glob
    out = {}
    ids = sorted({int(p.rsplit(".e", 1)[1].split(".")[0]) for q in (a_prefix, b_prefix)
                  for p in glob.glob(f"{glob.escape(q)}.e*.jsonl")})
    for i in ids:
        out[i] = compare(_engine_log(a_prefix, i), _engine_log(b_prefix, i))
    return out


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 3 or argv[0] != "compare":
        print(__doc__)
        return 2
    import os
    if not os.path.exists(argv[1]):  # prefixes of two multi-engine runs: every engine's pair
        runs = compare_runs(argv[1], argv[2])
        if not runs:
            print(f"no engine logs {argv[1]}.e<id>.*.jsonl")
            return 2
        bad = 0
        for i, r in runs.items():
            print(f"engine {i}:", json.dumps({k: v for k, v in r.items() if k != "at_first_difference"}))
            if r["first_difference"] is not None:
                bad += 1
                print(f"  first difference, step {r['first_difference']}")
                print("    A:", str(r["at_first_difference"]["a"])[:300])
                print("    B:", str(r["at_first_difference"]["b"])[:300])
        return 1 if bad else 0
    r = compare(argv[1], argv[2])
    print(json.dumps({k: v for k, v in r.items() if k != "at_first_difference"}, indent=2))
    if r["first_difference"] is None:
        return 0
    print("first difference, step", r["first_difference"])
    print("  A:", str(r["at_first_difference"]["a"])[:400])
    print("  B:", str(r["at_first_difference"]["b"])[:400])
    return 1


if __name__ == "__main__":
    sys.exit(main())
