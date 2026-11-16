"""--step-time-scale multiplies GPU step times only; 1.0 changes nothing.

Same small configuration as test_replay_selftest.py, on the synthetic workload (conftest.py).
"""

import csv
import os
import subprocess
import sys

BASE = ["--model", "Qwen/Qwen3-1.7B", "--cost", "fixed",
        "--num-gpu-blocks", "600", "--max-num-seqs", "16", "--max-model-len", "8192",
        "--max-num-batched-tokens", "512", "--fixed-step-ms", "8", "--fixed-per-token-us", "15",
        "--no-recorded-outputs", "--no-async-scheduling", "--step-cpu-us", "0", "--quiet"]


def augur(tmp_path, workload, name, *args):
    env = dict(os.environ, VLLM_LOGGING_LEVEL="WARNING")
    out, log = tmp_path / f"{name}.csv", tmp_path / f"{name}-steps.csv"
    subprocess.run([sys.executable, "-m", "augur.cli", *BASE, "--workload", workload, "--output", str(out), "--step-log", str(log), *args],
                   check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(log) as f:
        return out, list(csv.DictReader(f))


def test_scale_multiplies_step_times(tmp_path, workload):
    _, base = augur(tmp_path, workload, "base", "--step-time-scale", "1.0")
    _, scaled = augur(tmp_path, workload, "scaled", "--step-time-scale", "1.5")
    assert int(scaled[0]["dur_ns"]) == round(1.5 * int(base[0]["dur_ns"]))
    assert sum(int(r["dur_ns"]) for r in scaled) > sum(int(r["dur_ns"]) for r in base)


def test_scale_one_is_bit_identical(tmp_path, workload):
    a, _ = augur(tmp_path, workload, "plain")
    b, _ = augur(tmp_path, workload, "one", "--step-time-scale", "1.0")
    assert a.read_bytes() == b.read_bytes()
