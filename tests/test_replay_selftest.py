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
