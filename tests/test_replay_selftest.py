"""augur's own step log, replayed with its arrivals and step durations, must
reproduce every scheduling decision: sync, async, and speculative decoding.

Needs vLLM only: the workload is synthetic (conftest.py). About a minute per case.
"""

import glob
import json
import os
import subprocess
import sys

import pytest

from augur.verify.replay import compare

# Small KV cache and batch limits, so runs queue, chunk prefills and preempt.
BASE = ["--model", "Qwen/Qwen3-1.7B", "--cost", "fixed",
        "--num-gpu-blocks", "600", "--max-model-len", "8192", "--max-num-seqs", "16",
        "--max-num-batched-tokens", "512", "--fixed-step-ms", "8", "--fixed-per-token-us", "15",
        "--no-recorded-outputs", "--quiet"]


def augur(workload, *args):
    """Run augur on the workload; returns its printed JSON summary."""
    env = dict(os.environ, VLLM_LOGGING_LEVEL="WARNING")
    r = subprocess.run([sys.executable, "-m", "augur.cli", *BASE, "--workload", workload, *args], check=True,
                       env=env, capture_output=True, text=True)
    return json.loads(r.stdout[r.stdout.index("{"):])


@pytest.mark.parametrize("mode", [
    ["--no-async-scheduling"],
    ["--async-scheduling"],
    ["--no-async-scheduling", "--num-speculative-tokens", "3"],
    ["--async-scheduling", "--num-speculative-tokens", "3"],
], ids=["sync", "async", "spec-sync", "spec-async"])
def test_replay_reproduces_every_step(tmp_path, workload, mode):
    summary = augur(workload, *mode, "--step-log-vllm", str(tmp_path / "orig"))
    if mode == ["--no-async-scheduling"]:  # the workload must keep exercising the scheduler
        assert summary["preemptions"] > 0 and summary["prefix_hit_rate"] > 0 and summary["steps"] > 100, summary
    (orig,) = glob.glob(str(tmp_path / "orig.*.jsonl"))
    augur(workload, *mode, "--replay-durations", orig, "--step-log-vllm", str(tmp_path / "rep"))
    (rep,) = glob.glob(str(tmp_path / "rep.*.jsonl"))
    r = compare(orig, rep)
    assert r["steps_a"] > 100
    assert r["identical_steps"] == r["steps_a"] == r["steps_b"], r["at_first_difference"]


def test_async_replay_waits_for_a_logged_stall_before_the_next_schedule(tmp_path):
    """A real engine can be blocked inside a launch (the first step's one-off setup took 96 ms in a real run) and
    schedule its next batch only then, with every request that arrived meanwhile. The replay must hold the next
    schedule to its logged time, not run it at once on the requests it had, with a request's decode alone."""
    ms = 1_000_000
    wl = tmp_path / "wl.jsonl"
    wl.write_text("".join(json.dumps({"input_toks": 64, "output_toks": 3, "arrival_time_ns": t,
                                      "input_tok_ids": [1000 + 64 * i + j for j in range(64)]}) + "\n"
                          for i, t in enumerate([0, 40 * ms, 40 * ms + 10_000])))
    rid = lambda i: f"cmpl-augur.req_{i}.-0-abcd{i:04d}"  # noqa: E731
    new = lambda *ids: {rid(i): 0 for i in ids}  # noqa: E731
    one = lambda *ids: {rid(i): 1 for i in ids}  # noqa: E731
    ev = [
        {"ev": "add", "t": 0, "rid": rid(0), "prompt": 64},
        {"ev": "sched", "t": 50_000, "k": 1, "tok": {rid(0): 64}, "new": new(0), "pre": []},
        {"ev": "add", "t": 40 * ms, "rid": rid(1), "prompt": 64},
        {"ev": "add", "t": 40 * ms + 10_000, "rid": rid(2), "prompt": 64},
        {"ev": "sched", "t": 40 * ms + 100_000, "k": 2, "tok": {rid(0): 1, rid(1): 64, rid(2): 64}, "new": new(1, 2), "pre": []},
        {"ev": "upd", "t": 45 * ms, "k": 1, "out": one(0), "fin": [], "left": 1},
        {"ev": "sched", "t": 45 * ms + 50_000, "k": 3, "tok": one(0, 1, 2), "new": {}, "pre": []},
        {"ev": "upd", "t": 60 * ms, "k": 2, "out": one(0, 1, 2), "fin": [], "left": 3},
        {"ev": "sched", "t": 60 * ms + 50_000, "k": 4, "tok": one(1, 2), "new": {}, "pre": []},
        {"ev": "upd", "t": 75 * ms, "k": 3, "out": one(0, 1, 2), "fin": [rid(0)], "left": 2},
        {"ev": "upd", "t": 90 * ms, "k": 4, "out": one(1, 2), "fin": [rid(1), rid(2)], "left": 0},
    ]
    real = tmp_path / "real.jsonl"
    real.write_text("".join(json.dumps(e) + "\n" for e in ev))
    augur(str(wl), "--async-scheduling", "--replay-durations", str(real), "--step-log-vllm", str(tmp_path / "rep"))
    (rep,) = glob.glob(str(tmp_path / "rep.*.jsonl"))
    r = compare(str(real), rep)
    assert r["steps_a"] == 4
    assert r["identical_steps"] == r["steps_a"] == r["steps_b"], r["at_first_difference"]
