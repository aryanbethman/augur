"""Flags for vLLM options that change scheduling or the KV cache: --scheduler-cls, --scheduling-policy (with the
workload's per-request priority) and --kv-cache-dtype."""

import asyncio
import csv
import glob
import json
import os
import subprocess
import sys

import pytest
from vllm.v1.core.sched.scheduler import Scheduler

from augur.cli import main
from augur.trace.workload import Workload
from augur.verify.client import Client
from augur.verify.replay import compare

MODEL = "Qwen/Qwen3-1.7B"
BASE = ["--model", MODEL, "--cost", "fixed", "--max-model-len", "8192", "--max-num-seqs", "16",
        "--max-num-batched-tokens", "512", "--fixed-step-ms", "8", "--fixed-per-token-us", "15",
        "--no-recorded-outputs", "--no-async-scheduling", "--quiet"]


def augur(workload, *args, check=True):
    r = subprocess.run([sys.executable, "-m", "augur.cli", *BASE, "--workload", workload, *args], capture_output=True,
                       text=True, env=dict(os.environ, VLLM_LOGGING_LEVEL="WARNING"))
    if check:
        assert r.returncode == 0, r.stderr
    return r


def summary(r):
    return json.loads(r.stdout[r.stdout.index("{"):])


class CountingScheduler(Scheduler):
    """A user's scheduler: vLLM's decisions, counted."""
    calls = 0

    def schedule(self, *args, **kwargs):
        CountingScheduler.calls += 1
        return super().schedule(*args, **kwargs)


def test_custom_scheduler_class_is_used_and_still_step_logged(tmp_path, workload):
    CountingScheduler.calls = 0
    argv = [*BASE, "--workload", workload, "--num-entries", "3", "--num-gpu-blocks", "600",
            "--scheduler-cls", f"{__name__}.CountingScheduler", "--step-log-vllm", str(tmp_path / "steps")]
    assert main(argv) == 0
    assert CountingScheduler.calls > 0
    (log,) = glob.glob(str(tmp_path / "steps.*.jsonl"))
    scheduled = sum(json.loads(line)["ev"] == "sched" for line in open(log))
    assert scheduled == CountingScheduler.calls  # the logging wrapper composes with the user's class


@pytest.mark.parametrize("cls,error", [
    ("no_such_module.Sched", "cannot import"),
    (f"{__name__}.NoSuchClass", "cannot import"),
    ("collections.OrderedDict", "not a subclass of vLLM's Scheduler"),
    ("vllm.v1.core.sched.async_scheduler.AsyncScheduler", "async scheduler but the run is synchronous"),
])
def test_bad_scheduler_class_is_refused(cls, error, capsys):
    assert main([*BASE, "--workload", "unused", "--scheduler-cls", cls]) == 2
    assert error in capsys.readouterr().err


def test_sync_scheduler_in_an_async_run_is_refused(capsys):
    argv = [a for a in BASE if a != "--no-async-scheduling"]
    assert main([*argv, "--workload", "unused", "--scheduler-cls", f"{__name__}.CountingScheduler"]) == 2
    assert "synchronous scheduler but the run is async" in capsys.readouterr().err


def write_trace(path, rows):
    path.write_text("".join(json.dumps({"output_toks": 8, "input_toks": 64, **r}) + "\n" for r in rows))
    return str(path)


def finish_order(tmp_path, workload, *args):
    out = tmp_path / "run.csv"
    augur(workload, "--max-num-seqs", "1", "--num-gpu-blocks", "600", "--output", str(out), *args)
    with open(out) as f:
        return [int(r["request id"]) for r in sorted(csv.DictReader(f), key=lambda r: int(r["end_time"]))]


def test_priority_policy_reorders_the_queue(tmp_path):
    # One sequence at a time: request 0 runs; 1 (priority 5) and 2 (priority 0) wait behind it.
    wl = write_trace(tmp_path / "t.jsonl", [{"arrival_time_ns": 0}, {"arrival_time_ns": 1_000_000, "priority": 5},
                                           {"arrival_time_ns": 2_000_000, "priority": 0}])
    assert finish_order(tmp_path, wl, "--scheduling-policy", "fcfs") == [0, 1, 2]
    assert finish_order(tmp_path, wl, "--scheduling-policy", "priority") == [0, 2, 1]


def test_own_log_replay_is_exact_with_priorities(tmp_path, workload):
    with open(workload) as f:
        rows = [json.loads(line) for line in f]
    wl = write_trace(tmp_path / "prio.jsonl", [dict(r, priority=(i * 7) % 4) for i, r in enumerate(rows)])
    args = ["--num-gpu-blocks", "600", "--scheduling-policy", "priority"]
    augur(wl, "--num-gpu-blocks", "600", "--step-log-vllm", str(tmp_path / "fcfs"))
    augur(wl, *args, "--step-log-vllm", str(tmp_path / "orig"))
    augur(wl, *args, "--replay-durations", glob.glob(str(tmp_path / "orig.*.jsonl"))[0],
          "--step-log-vllm", str(tmp_path / "rep"))
    orig, rep, base = (glob.glob(str(tmp_path / f"{n}.*.jsonl"))[0] for n in ("orig", "rep", "fcfs"))
    r = compare(orig, rep)
    assert r["steps_a"] > 100 and r["identical_steps"] == r["steps_a"] == r["steps_b"], r["at_first_difference"]
    assert compare(base, orig)["identical_steps"] < r["steps_a"], "priorities changed nothing: the check is vacuous"


def test_client_sends_priority_only_when_the_workload_has_one(tmp_path):
    sent = []

    class Capture:
        def post(self, url, json):
            sent.append(json)
            raise TimeoutError

    wl = write_trace(tmp_path / "t.jsonl", [{"arrival_time_ns": 0, "priority": 3}, {"arrival_time_ns": 0}])
    workload = Workload(wl, filler_vocab=1001)
    client = Client("http://unused", "m", workload, None)
    for call in workload.entries():
        asyncio.run(client._send(Capture(), call))
    assert [b.get("priority") for b in sent] == [3, None]


def test_fp8_kv_cache_doubles_the_block_count(workload):
    blocks = {}
    for dtype in ("auto", "fp8", "fp8_e5m2"):
        r = augur(workload, "--num-entries", "1", "--kv-cache-gb", "1", "--kv-cache-dtype", dtype)
        blocks[dtype] = summary(r)["num_gpu_blocks"]
        assert ("priced as bf16" in r.stderr) == (dtype != "auto")
    assert blocks["fp8"] == blocks["fp8_e5m2"] == 2 * blocks["auto"]
