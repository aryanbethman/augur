"""--frontend-latency-ms delays when the engine sees a request, --delivery-latency-ms when the client
gets each token; both count in the calls times, which stay measured from the clients send.

Same small configuration as test_step_time_scale.py, on the synthetic workload (conftest.py).
"""

import ast
import csv
import os
import subprocess
import sys

BASE = ["--model", "Qwen/Qwen3-1.7B", "--cost", "fixed",
        "--num-gpu-blocks", "600", "--max-num-seqs", "16", "--max-model-len", "8192",
        "--max-num-batched-tokens", "512", "--fixed-step-ms", "8", "--fixed-per-token-us", "15",
        "--no-recorded-outputs", "--no-async-scheduling", "--step-cpu-us", "0", "--quiet"]


def augur(tmp_path, workload, name, n, frontend_ms, delivery_ms):
    env = dict(os.environ, VLLM_LOGGING_LEVEL="WARNING")
    out = tmp_path / f"{name}.csv"
    subprocess.run([sys.executable, "-m", "augur.cli", *BASE, "--workload", workload, "--num-entries", str(n), "--output", str(out),
                    "--frontend-latency-ms", str(frontend_ms), "--delivery-latency-ms", str(delivery_ms)],
                   check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(out) as f:
        return list(csv.DictReader(f))


def ints(rows, col):
    return [int(r[col]) for r in rows]


def itls(rows):
    return [ast.literal_eval(r["ITL"]) for r in rows]


def test_one_request(tmp_path, workload):
    base = augur(tmp_path, workload, "base", 1, 0, 0)
    front = augur(tmp_path, workload, "front", 1, 6.5, 0)
    deliv = augur(tmp_path, workload, "deliv", 1, 0, 8)
    assert len(base) == len(front) == len(deliv) == 1
    for rows, ns in ((front, 6_500_000), (deliv, 8_000_000)):
        assert ints(rows, "arrival") == ints(base, "arrival")
        assert ints(rows, "TTFT")[0] == ints(base, "TTFT")[0] + ns
        assert ints(rows, "latency")[0] == ints(base, "latency")[0] + ns
        assert itls(rows) == itls(base)


def test_delivery_shifts_every_call_by_exactly_the_delay(tmp_path, workload):
    base = augur(tmp_path, workload, "base", 60, 0, 0)
    deliv = augur(tmp_path, workload, "deliv", 60, 0, 8)
    assert len(base) == len(deliv) == 60
    assert ints(deliv, "arrival") == ints(base, "arrival")
    assert ints(deliv, "queuing_delay") == ints(base, "queuing_delay")
    assert ints(deliv, "TTFT") == [x + 8_000_000 for x in ints(base, "TTFT")]
    assert ints(deliv, "latency") == [x + 8_000_000 for x in ints(base, "latency")]
    assert ints(deliv, "end_time") == [x + 8_000_000 for x in ints(base, "end_time")]
    assert itls(deliv) == itls(base)
