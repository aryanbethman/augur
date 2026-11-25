"""--workers N gives the output of --workers 1, byte for byte (augur.sim.parallel)."""

import glob
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

BASE = ["--model", "Qwen/Qwen3-1.7B", "--engines", "6", "--num-gpu-blocks", "600", "--max-num-seqs", "16",
        "--max-model-len", "8192", "--max-num-batched-tokens", "512", "--step-cpu-us", "300", "--quiet"]
FIXED = ["--cost", "fixed", "--fixed-step-ms", "8", "--fixed-per-token-us", "15"]
TABLES = ["--hardware", "RTX4060Ti", "--cost", "tables"]
TIMING = ("wall_s", "vllm_step_cpu_ms_per_step")


@pytest.fixture(scope="module")
def sessions(tmp_path_factory):
    """Agent sessions whose turns extend the previous prompt, with tool waits from none to long."""
    rng = random.Random(1)
    path = tmp_path_factory.mktemp("sessions") / "sessions.jsonl"
    with open(path, "w") as f:
        for s in range(10):
            prompt, subs = [rng.randrange(1000, 100_000) for _ in range(300)], []
            for _ in range(5):
                out = [rng.randrange(1000, 100_000) for _ in range(30)]
                subs.append({"input_toks": len(prompt), "output_toks": len(out), "input_tok_ids": prompt,
                             "output_tok_ids": out, "tool_duration_ns": rng.choice([0, 20_000_000, 400_000_000])})
                prompt = prompt + out + [rng.randrange(1000, 100_000) for _ in range(80)]
            f.write(json.dumps({"session_id": f"s{s}", "arrival_time_ns": s * 30_000_000, "sub_requests": subs}) + "\n")
    return str(path)


def run(tmp_path, workers, workload, *args):
    d = tmp_path / f"w{workers}"
    d.mkdir()
    cmd = [sys.executable, "-m", "augur.cli", *BASE, *args, "--workload", workload, "--workers", str(workers),
           "--output", str(d / "out.csv"), "--step-log", str(d / "steps.csv"), "--step-log-vllm", str(d / "vllm")]
    p = subprocess.run(cmd, check=True, capture_output=True, text=True, env=dict(os.environ, VLLM_LOGGING_LEVEL="WARNING"))
    summary = json.loads(p.stdout[p.stdout.rindex("\n{") + 1:])
    for k in TIMING:
        summary.pop(k)
    logs = {os.path.basename(f).split(".")[1]: open(f, "rb").read() for f in glob.glob(str(d / "vllm.e*.jsonl"))}
    return (d / "out.csv").read_bytes(), (d / "steps.csv").read_bytes(), summary, logs


CASES = {
    "round-robin sync": ("flat", FIXED + ["--router", "round-robin", "--no-async-scheduling"]),
    "round-robin async tables": ("flat", TABLES + ["--router", "round-robin", "--async-scheduling"]),
    "power-of-two async sessions": ("sessions", FIXED + ["--router", "power-of-two", "--async-scheduling"]),
    "cache-aware sync sessions": ("sessions", FIXED + ["--router", "cache-aware", "--no-async-scheduling"]),
    "least-loaded sync sessions": ("sessions", FIXED + ["--router", "least-loaded", "--no-async-scheduling"]),
    "kv-aware async flat": ("flat", FIXED + ["--router", "kv-aware", "--async-scheduling"]),
    "sticky no delivery wait": ("sessions", FIXED + ["--router", "sticky-least-loaded", "--delivery-latency-ms", "0.5",
                                                       "--frontend-latency-ms", "0"]),
}


@pytest.mark.parametrize("case", CASES)
def test_three_workers_match_one_process(tmp_path, workload, sessions, case):
    kind, args = CASES[case]
    wl = workload if kind == "flat" else sessions
    with ThreadPoolExecutor(2) as pool:  # the two runs are independent: start-up of one overlaps the other
        one, three = pool.map(lambda w: run(tmp_path, w, wl, *args), (1, 3))
    assert three[0] == one[0]  # the result CSV
    assert three[1] == one[1]  # the step log
    assert three[2] == one[2]  # the printed summary
    assert three[3] == one[3] and one[3]  # vLLM's own per-engine step logs


def test_workers_refuse_what_they_cannot_split(tmp_path, workload):
    for extra in (["--dp", "2"], ["--prefill-engines", "1", "--decode-engines", "1"]):
        p = subprocess.run([sys.executable, "-m", "augur.cli", *BASE, *FIXED, *extra, "--workload", workload,
                            "--workers", "2"], capture_output=True, text=True)
        assert p.returncode == 2 and "--workers > 1 does not support" in p.stderr
