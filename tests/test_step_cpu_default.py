"""The default per-step CPU time (--step-cpu-us) is added to synchronous steps and not to async ones; 0 turns it off.

Same small configuration as test_step_time_scale.py, on the synthetic workload (conftest.py).
"""

import csv
import os
import subprocess
import sys

DEFAULT_US = 1300  # augur.cli's --step-cpu-us default (docs/accuracy.md section 2)

BASE = ["--model", "Qwen/Qwen3-1.7B", "--cost", "fixed",
        "--num-gpu-blocks", "600", "--max-num-seqs", "16", "--max-model-len", "8192",
        "--max-num-batched-tokens", "512", "--fixed-step-ms", "8", "--fixed-per-token-us", "15",
        "--no-recorded-outputs", "--quiet"]


def step_durations(tmp_path, workload, name, *args):
    env = dict(os.environ, VLLM_LOGGING_LEVEL="WARNING")
    log = tmp_path / f"{name}-steps.csv"
    subprocess.run([sys.executable, "-m", "augur.cli", *BASE, "--workload", workload, "--output", str(tmp_path / f"{name}.csv"),
                    "--step-log", str(log), *args], check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(log) as f:
        return [int(r["dur_ns"]) for r in csv.DictReader(f)]


def test_sync_step_pays_the_default_cpu_time(tmp_path, workload):
    off = step_durations(tmp_path, workload, "off", "--no-async-scheduling", "--step-cpu-us", "0")
    default = step_durations(tmp_path, workload, "default", "--no-async-scheduling")
    assert default[0] - off[0] == DEFAULT_US * 1000


def test_async_steps_hide_the_cpu_time(tmp_path, workload):
    """Behind a running batch the CPU work costs nothing: only a step launched from idle waits for it."""
    off = step_durations(tmp_path, workload, "off", "--async-scheduling", "--step-cpu-us", "0")
    default = step_durations(tmp_path, workload, "default", "--async-scheduling")
    assert default[0] == off[0]
    assert sorted(default)[len(default) // 2] == sorted(off)[len(off) // 2]


def test_override_replaces_the_default(tmp_path, workload):
    off = step_durations(tmp_path, workload, "off", "--no-async-scheduling", "--step-cpu-us", "0")
    big = step_durations(tmp_path, workload, "big", "--no-async-scheduling", "--step-cpu-us", "2500",
                         "--step-cpu-per-request-us", "100")
    assert big[0] - off[0] == 2_500_000 + 100_000  # the first step holds one request
