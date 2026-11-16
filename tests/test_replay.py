"""Step-log parsing: durations for sync and async runs, unlogged tails, compare."""

import json

from augur.verify.replay import ReplayCost, compare, is_async, scheduled
from augur.verify.stepcheck import real_steps


def sched(k, t, tok, new=None, pre=()):
    return {"ev": "sched", "t": t, "k": k, "tok": tok, "new": new or {}, "pre": list(pre)}


def upd(k, t, out, left, fin=()):
    return {"ev": "upd", "t": t, "k": k, "out": out, "fin": list(fin), "left": left}


def write(tmp_path, name, events):
    p = tmp_path / name
    p.write_text("".join(json.dumps(e) + "\n" for e in events))
    return str(p)


# Sync: schedule -> update -> schedule ...
SYNC = [
    {"ev": "add", "t": 0, "rid": "req_0", "prompt": 4},
    sched(0, 10, {"req_0": 4}, {"req_0": 0}), upd(0, 40, {"req_0": 1}, 1),
    sched(1, 45, {"req_0": 1}), upd(1, 60, {"req_0": 1}, 0, ["req_0"]),
]

# Async: step k+1 is scheduled before step k's update lands.
ASYNC = [
    {"ev": "add", "t": 0, "rid": "req_0", "prompt": 4},
    sched(0, 10, {"req_0": 4}, {"req_0": 0}),
    sched(1, 12, {"req_0": 1}),
    upd(0, 40, {"req_0": 1}, 1),
    sched(2, 42, {"req_0": 1}),
    upd(1, 55, {"req_0": 1}, 1),
    upd(2, 70, {"req_0": 1}, 0, ["req_0"]),
]


def test_is_async():
    assert not is_async(SYNC)
    assert is_async(ASYNC)


def test_sync_durations_run_to_next_start(tmp_path):
    # More work left after step 0: its duration includes the loop gap to step 1.
    rc = ReplayCost.from_log(write(tmp_path, "s.jsonl", SYNC))
    assert rc.durations == [45 - 10, 60 - 45]


def test_async_durations_start_after_previous_batch(tmp_path):
    # Batch k runs from max(scheduled, batch k-1 finished) to its update.
    rc = ReplayCost.from_log(write(tmp_path, "a.jsonl", ASYNC))
    assert rc.durations == [40 - 10, 55 - 40, 70 - 55]


def test_async_unlogged_tail_is_priced(tmp_path):
    # Server killed before the last batch's update was flushed: that batch
    # still ran, so the replay needs a duration for it.
    ev = ASYNC + [sched(3, 72, {"req_1": 4}, {"req_1": 0})]
    rc = ReplayCost.from_log(write(tmp_path, "t.jsonl", ev))
    assert rc.durations == [30, 15, 15, 15]
    assert len(scheduled(ev)) == 4


def test_stepcheck_async_measures_gpu_time():
    d = [m for _, m in real_steps(ASYNC)]
    assert d == [30, 15, 15]


def test_compare_maps_request_ids(tmp_path):
    real = [sched(0, 1, {"cmpl-augur.req_0.-0": 4}, {"cmpl-augur.req_0.-0": 0}), upd(0, 2, {}, 0)]
    sim = [sched(0, 9, {"req_0": 4}, {"req_0": 0}), upd(0, 10, {}, 0)]
    r = compare(write(tmp_path, "r.jsonl", real), write(tmp_path, "s.jsonl", sim))
    assert r["identical_steps"] == 1 and r["first_difference"] is None
    sim[0]["tok"] = {"req_0": 3}
    r = compare(write(tmp_path, "r.jsonl", real), write(tmp_path, "s.jsonl", sim))
    assert r["first_difference"] == 0
