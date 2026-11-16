"""--step-time-scale multiplies GPU step times only; 1.0 changes nothing.

Same small configuration as test_replay_selftest.py (needs the LLMServingSim fork; skipped without it).
"""

import csv
import os
import subprocess
import sys

import pytest

from augur.cost.profiles import default_fork

FORK = default_fork()
WORKLOAD = os.path.join(FORK, "workloads", "sharegpt-qwen3-32b-300-sps10.jsonl")
pytestmark = pytest.mark.skipif(not os.path.exists(WORKLOAD), reason="LLMServingSim fork not found (AUGUR_FORK)")

BASE = ["--model", "Qwen/Qwen3-1.7B", "--workload", WORKLOAD, "--num-entries", "60",
        "--num-gpu-blocks", "600", "--max-num-seqs", "16", "--max-model-len", "8192",
        "--max-num-batched-tokens", "512", "--fixed-step-ms", "8", "--fixed-per-token-us", "15",
        "--no-recorded-outputs", "--no-async-scheduling", "--step-cpu-us", "0", "--quiet"]


def augur(tmp_path, name, *args):
    env = dict(os.environ, VLLM_LOGGING_LEVEL="WARNING")
    out, log = tmp_path / f"{name}.csv", tmp_path / f"{name}-steps.csv"
    subprocess.run([sys.executable, "-m", "augur.cli", *BASE, "--output", str(out), "--step-log", str(log), *args],
                   check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(log) as f:
        return out, list(csv.DictReader(f))


def test_scale_multiplies_step_times(tmp_path):
    _, base = augur(tmp_path, "base", "--step-time-scale", "1.0")
    _, scaled = augur(tmp_path, "scaled", "--step-time-scale", "1.5")
    assert int(scaled[0]["dur_ns"]) == round(1.5 * int(base[0]["dur_ns"]))
    assert sum(int(r["dur_ns"]) for r in scaled) > sum(int(r["dur_ns"]) for r in base)


def test_scale_one_is_bit_identical(tmp_path):
    a, _ = augur(tmp_path, "plain")
    b, _ = augur(tmp_path, "one", "--step-time-scale", "1.0")
    assert a.read_bytes() == b.read_bytes()
