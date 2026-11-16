"""augur's own step log, replayed with its arrivals and step durations, must
reproduce every scheduling decision: sync, async, and speculative decoding.

Needs vLLM and the LLMServingSim fork (model configs, workloads); skipped
without them. About a minute per case.
"""

import glob
import os
import subprocess
import sys

import pytest

from augur.cost.profiles import default_fork
from augur.verify.replay import compare

FORK = default_fork()
WORKLOAD = os.path.join(FORK, "workloads", "sharegpt-qwen3-32b-300-sps10.jsonl")
pytestmark = pytest.mark.skipif(not os.path.exists(WORKLOAD), reason="LLMServingSim fork not found (AUGUR_FORK)")

# Small KV cache and batch limits, so runs queue, chunk prefills and preempt.
BASE = ["--model", "Qwen/Qwen3-1.7B", "--workload", WORKLOAD, "--num-entries", "60",
        "--num-gpu-blocks", "600", "--max-model-len", "8192", "--max-num-seqs", "16",
        "--max-num-batched-tokens", "512", "--fixed-step-ms", "8", "--fixed-per-token-us", "15",
        "--no-recorded-outputs", "--quiet"]


def augur(*args):
    env = dict(os.environ, VLLM_LOGGING_LEVEL="WARNING")
    subprocess.run([sys.executable, "-m", "augur.cli", *BASE, *args], check=True, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


@pytest.mark.parametrize("mode", [
    ["--no-async-scheduling"],
    ["--async-scheduling"],
    ["--no-async-scheduling", "--num-speculative-tokens", "3"],
    ["--async-scheduling", "--num-speculative-tokens", "3"],
], ids=["sync", "async", "spec-sync", "spec-async"])
def test_replay_reproduces_every_step(tmp_path, mode):
    augur(*mode, "--step-log-vllm", str(tmp_path / "orig"))
    (orig,) = glob.glob(str(tmp_path / "orig.*.jsonl"))
    augur(*mode, "--replay-durations", orig, "--step-log-vllm", str(tmp_path / "rep"))
    (rep,) = glob.glob(str(tmp_path / "rep.*.jsonl"))
    r = compare(orig, rep)
    assert r["steps_a"] > 100
    assert r["identical_steps"] == r["steps_a"] == r["steps_b"], r["at_first_difference"]
