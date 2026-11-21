"""augur's command line: flags it must refuse, where results go, and defaults that must match `vllm serve`."""

import csv
import json
import os
import subprocess
import sys

import pytest

from augur.cli import main

MODEL = "Qwen/Qwen3-1.7B"
RUN = ["--model", MODEL, "--cost", "fixed", "--num-gpu-blocks", "600", "--max-num-seqs", "16", "--max-model-len", "8192",
       "--max-num-batched-tokens", "512", "--fixed-step-ms", "8", "--step-cpu-us", "0", "--quiet"]


def augur(*args):
    return subprocess.run([sys.executable, "-m", "augur.cli", *RUN, *args], capture_output=True, text=True,
                          env=dict(os.environ, VLLM_LOGGING_LEVEL="WARNING"))


@pytest.mark.parametrize("flags", [["--prefill-engines", "2"], ["--decode-engines", "2"]])
def test_pd_needs_both_engine_kinds(flags, capsys):
    assert main(["--model", MODEL, "--workload", "unused", *flags]) == 2
    assert "--prefill-engines and --decode-engines" in capsys.readouterr().err


def test_missing_result_folders_are_created(tmp_path, workload):
    out, steps = tmp_path / "a" / "b" / "run.csv", tmp_path / "c" / "steps.csv"
    r = augur("--workload", workload, "--num-entries", "2", "--output", str(out), "--step-log", str(steps))
    assert r.returncode == 0, r.stderr
    with open(out) as f:
        assert len(list(csv.DictReader(f))) == 2
    assert steps.exists()


def test_speculative_async_run_is_reported_as_async(workload):
    r = augur("--workload", workload, "--num-entries", "2", "--num-speculative-tokens", "3", "--async-scheduling")
    assert r.returncode == 0, r.stderr
    assert "this run is synchronous" not in r.stderr
    summary = json.loads(r.stdout[r.stdout.index("{"):])
    assert summary["async_scheduling"] is True and "notes" not in summary


def test_session_id_with_an_at_sign_runs(tmp_path):
    trace = tmp_path / "at.jsonl"
    subs = [{"input_toks": 32, "output_toks": 4}, {"input_toks": 48, "output_toks": 4}]
    trace.write_text(json.dumps({"session_id": "alice@lab", "arrival_time_ns": 0, "sub_requests": subs}) + "\n")
    out = tmp_path / "at.csv"
    r = augur("--workload", str(trace), "--output", str(out))
    assert r.returncode == 0, r.stderr
    with open(out) as f:
        assert [row["session_id"] for row in csv.DictReader(f)] == ["alice@lab", "alice@lab"]


def test_two_configs_of_one_model_do_not_share_a_folder(tmp_path):
    from augur.sim.models import model_dir
    one, two = tmp_path / "one.json", tmp_path / "two.json"
    one.write_text('{"num_hidden_layers": 32}')
    two.write_text('{"num_hidden_layers": 80}')
    cache = str(tmp_path / "cache")
    first, second = model_dir("", "org/m", cache, str(one)), model_dir("", "org/m", cache, str(two))
    assert first != second
    with open(os.path.join(first, "config.json")) as f:
        assert json.load(f)["num_hidden_layers"] == 32
    assert model_dir("", "org/m", cache, str(one)) == first


@pytest.mark.parametrize("hardware,hbm_gb,tokens,seqs", [("RTX4060Ti", 8, 2048, 256), ("H100", 80, 8192, 1024),
                                                         ("A100", 80, 2048, 256), ("B200", 192, 16384, 1024)])
def test_unset_batch_limits_are_vllm_serves_for_the_simulated_gpu(tmp_path, hardware, hbm_gb, tokens, seqs):
    from augur.cost import FixedCost
    from augur.sim.engine import EngineSpec, SimEngine
    from augur.sim.models import model_dir
    spec = EngineSpec(model_dir=model_dir("", MODEL, str(tmp_path)), model_name=MODEL, kv_cache_bytes=1 << 30,
                      max_model_len=8192, hbm_gb=hbm_gb, hardware=hardware)
    limits = SimEngine(0, spec, FixedCost()).vllm_config.scheduler_config
    assert (limits.max_num_batched_tokens, limits.max_num_seqs) == (tokens, seqs)


def test_batch_limit_flags_win_over_the_defaults(tmp_path):
    from augur.cost import FixedCost
    from augur.sim.engine import EngineSpec, SimEngine
    from augur.sim.models import model_dir
    spec = EngineSpec(model_dir=model_dir("", MODEL, str(tmp_path)), model_name=MODEL, kv_cache_bytes=1 << 30,
                      max_model_len=8192, hbm_gb=80, hardware="H100", max_num_seqs=128, max_num_batched_tokens=2048)
    limits = SimEngine(0, spec, FixedCost()).vllm_config.scheduler_config
    assert (limits.max_num_batched_tokens, limits.max_num_seqs) == (2048, 128)


def _latency(tmp_path, workload, name, *args):
    out = tmp_path / f"{name}.csv"
    r = augur("--workload", workload, "--num-entries", "6", "--output", str(out), "--no-async-scheduling", *args)
    assert r.returncode == 0, r.stderr
    with open(out) as f:
        return [int(x["latency"]) for x in csv.DictReader(f)]


def test_spec_us_per_request_slows_only_speculative_steps(tmp_path, workload):
    spec = ["--num-speculative-tokens", "3"]
    base = _latency(tmp_path, workload, "base", *spec, "--spec-us-per-request", "0")
    slow = _latency(tmp_path, workload, "slow", *spec, "--spec-us-per-request", "2000")
    assert sum(slow) > 1.2 * sum(base)
    # without speculation the flag has no effect
    assert _latency(tmp_path, workload, "p0", "--spec-us-per-request", "0") == \
        _latency(tmp_path, workload, "p2000", "--spec-us-per-request", "2000")
