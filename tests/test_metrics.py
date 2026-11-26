"""Recorder packs each finished call's token times; its summary must be what plain arrays gave."""

import random
from array import array

import numpy as np

from augur.trace.metrics import CallRecord, PackedTimes, Recorder


def _records(seed=0):
    rng = random.Random(seed)
    out = []
    for i, n in enumerate([0, 1, 2, 5, 40, 3, 200]):
        t, times = 1000 * i, array("q")
        for k in range(n):
            t += rng.randrange(1, 5 * 10**7) + (6 * 10**9 if k == 3 else 0)  # one gap past 32 bits (a preemption)
            times.append(t)
        out.append(CallRecord(num_id=i, call_id=f"c{i}", instance_id=0, model="m", input_toks=7, output_toks=n,
                              arrival_ns=1000 * i, token_times=times, end_ns=t + 5))
    return out


def test_packed_times_read_as_the_array():
    for rec in _records():
        raw = array("q", rec.token_times)
        Recorder().done.append(rec)
        got = rec.token_times
        assert len(got) == len(raw) and list(got) == list(raw)
        if len(raw) > 1:
            assert isinstance(got, PackedTimes) and got.deltas() == [b - a for a, b in zip(raw, raw[1:])]


def test_summary_matches_plain_lists():
    recs = [r for r in _records() if len(r.token_times) > 0]
    itl = np.array([b - a for r in recs for a, b in zip(r.token_times, r.token_times[1:])]) / 1e6
    rec = Recorder()
    for r in recs:
        rec.done.append(r)
    s = rec.summary()
    assert s["itl_ms"] == {"mean": itl.mean(), "p99": np.percentile(itl, 99)}
