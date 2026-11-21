"""--profile-variant: the fork's profile folder pinned by name (e.g. bf16-tp4proj, a projected TP4 bundle).

The first tests need no fork; the last runs a short TP4/EP4 workload on the
fork's projected bundle and is skipped without it (AUGUR_FORK)."""

import json
import os
import subprocess
import sys

import pytest

from augur.cost.profiles import check, default_fork

FORK = default_fork()
QWEN = "Qwen/Qwen3-30B-A3B-Instruct-2507"
VARIANT = "bf16-tp4proj"
PERF = os.path.join(FORK, "profiler", "perf", "RTXPRO6000", QWEN)


def make(tmp_path, *rel):
    for r in rel:
        (tmp_path / "profiler" / "perf" / r).mkdir(parents=True)
    return str(tmp_path)


def test_check_accepts_model_tp_only_under_the_variant(tmp_path):
    fork = make(tmp_path, "RTXPRO6000/Qwen/Q/bf16/tp1", "RTXPRO6000/Qwen/Q/bf16-tp4proj/tp1",
                "RTXPRO6000/Qwen/Q/bf16-tp4proj/tp4")
    check(fork, "RTXPRO6000", "Qwen/Q", 4, "bf16-tp4proj")
    with pytest.raises(FileNotFoundError, match="tp4 in profile variant 'bf16'"):
        check(fork, "RTXPRO6000", "Qwen/Q", 4, "bf16")  # tp4 exists, but not under that folder
    with pytest.raises(FileNotFoundError, match="variant 'nope'"):
        check(fork, "RTXPRO6000", "Qwen/Q", 1, "nope")
    check(fork, "RTXPRO6000", "Qwen/Q", 1)  # no variant: unchanged


def run_augur(*args):
    return subprocess.run([sys.executable, "-m", "augur.cli", "--quiet", *args], capture_output=True, text=True,
                          env=dict(os.environ, VLLM_LOGGING_LEVEL="WARNING"))


def test_variant_needs_the_fork_pricer():
    r = run_augur("--model", QWEN, "--workload", "x", "--cost", "tables", "--profile-variant", VARIANT)
    assert r.returncode == 2 and "needs --cost llmservingsim" in r.stderr


@pytest.mark.skipif(not os.path.isdir(os.path.join(PERF, VARIANT)), reason=f"fork has no {VARIANT} profile")
def test_tp4_ep4_on_the_variant(tmp_path):
    inst = {"model_name": QWEN, "hardware": "RTXPRO6000", "npu_mem": {"mem_size": 96, "mem_bw": 1597, "mem_latency": 0},
            "num_npus": 4, "tp_size": 4, "ep_size": 4, "pd_type": None}
    cluster = tmp_path / "cluster.json"
    cluster.write_text(json.dumps({"num_nodes": 1, "link_bw": 16, "link_latency": 20000, "nodes": [{
        "num_instances": 1, "cpu_mem": {"mem_size": 512, "mem_bw": 256, "mem_latency": 0}, "instances": [inst]}]}))
    src = os.path.join(FORK, "workloads", "swe-bench-qwen3-30b-a3b-50-sps0.2.jsonl")
    if not os.path.exists(src):
        pytest.skip("fork has no SWE-bench workload")
    out, steps = tmp_path / "r.csv", tmp_path / "steps.csv"
    args = ["--model", QWEN, "--workload", src, "--num-entries", "1", "--tp", "4", "--enable-expert-parallel",
            "--hbm-gb", "96", "--max-model-len", "65536", "--hardware", "RTXPRO6000", "--cost", "auto",
            "--lss-cluster-config", str(cluster), "--output", str(out), "--step-log", str(steps)]
    # variant absent from the folder it names: a clear refusal, not a silent fallback to another bundle
    bad = run_augur(*args, "--profile-variant", "bf16")
    assert bad.returncode == 2 and "no profile" in bad.stderr
    ok = run_augur(*args, "--profile-variant", VARIANT)  # --cost auto must pick llmservingsim
    assert ok.returncode == 0, ok.stderr[-2000:]
    summary = json.loads(ok.stdout)
    assert summary["astra_runs"] > 0 and summary["steps"] > 0


@pytest.mark.skipif(not os.path.isdir(os.path.join(PERF, VARIANT)), reason=f"fork has no {VARIANT} profile")
def test_tp4_ep4_on_a_four_instance_config_prices_like_one_instance(tmp_path):
    """ASTRA-Sim did not finish one query on a 4-instance (16 NPU) TP4/EP4 config; augur prices engines
    from instance 0 alone (AstraOracle first_instance_only), so the run ends in seconds with the same
    result as the 1-instance config."""
    src = os.path.join(FORK, "workloads", "swe-bench-qwen3-30b-a3b-50-sps0.2.jsonl")
    if not os.path.exists(src):
        pytest.skip("fork has no SWE-bench workload")
    inst = {"model_name": QWEN, "hardware": "RTXPRO6000", "npu_mem": {"mem_size": 96, "mem_bw": 1597, "mem_latency": 0},
            "num_npus": 4, "tp_size": 4, "ep_size": 4, "pd_type": None}
    outs = {}
    for n in (1, 4):
        cluster = tmp_path / f"cluster{n}.json"
        cluster.write_text(json.dumps({"num_nodes": 1, "link_bw": 16, "link_latency": 20000, "nodes": [{
            "num_instances": n, "cpu_mem": {"mem_size": 512, "mem_bw": 256, "mem_latency": 0},
            "instances": [dict(inst) for _ in range(n)]}]}))
        out = tmp_path / f"r{n}.csv"
        r = subprocess.run([sys.executable, "-m", "augur.cli", "--quiet", "--model", QWEN, "--workload", src,
                            "--num-entries", "1", "--tp", "4", "--enable-expert-parallel", "--engines", str(n),
                            "--hbm-gb", "96", "--max-model-len", "65536", "--hardware", "RTXPRO6000",
                            "--cost", "llmservingsim", "--profile-variant", VARIANT,
                            "--lss-cluster-config", str(cluster), "--output", str(out)],
                           capture_output=True, text=True, timeout=300,
                           env=dict(os.environ, VLLM_LOGGING_LEVEL="WARNING"))
        assert r.returncode == 0, r.stderr[-2000:]
        outs[n] = out.read_bytes()
    assert outs[1] == outs[4]
