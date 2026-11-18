"""augur bench's start-up checks, before any simulation."""

import os
import subprocess
import sys
import tempfile


def test_bench_without_its_tokenizer_says_how_to_get_one():
    env = dict(os.environ, HF_HUB_OFFLINE="1", VLLM_LOGGING_LEVEL="WARNING", PYTHONHASHSEED="0")
    r = subprocess.run([sys.executable, "-m", "augur.bench", "--model", "Qwen/Qwen3-1.7B",
                        "--tokenizer", "augur-test/not-a-model", "--dataset-name", "random", "--num-prompts", "2"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 2
    assert "augur-test/not-a-model" in r.stderr and "--tokenizer" in r.stderr
    assert "Traceback" not in r.stderr


def bench_cli(*args):
    env = dict(os.environ, HF_HUB_OFFLINE="1", VLLM_LOGGING_LEVEL="WARNING", PYTHONHASHSEED="0")
    return subprocess.run([sys.executable, "-m", "augur.cli", "bench", "--model", "Qwen/Qwen3-1.7B",
                           "--dataset-name", "random", "--num-prompts", "2", *args],
                          capture_output=True, text=True, env=env)


def test_augur_bench_is_a_command():
    r = bench_cli("--tokenizer", "augur-test/not-a-model")
    assert r.returncode == 2
    assert "cannot load the tokenizer" in r.stderr and "--workload" not in r.stderr


def test_bench_refuses_max_concurrency_it_cannot_model():
    r = bench_cli("--max-concurrency", "8")
    assert r.returncode == 2
    assert "--max-concurrency" in r.stderr and "Traceback" not in r.stderr


def test_bench_removes_its_working_folder(tmp_path, monkeypatch):
    from augur import bench
    seen = []

    def failed_run(args, augur_args, trace, *rest):
        seen.append(os.path.exists(trace))
        return 3

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr("vllm.tokenizers.get_tokenizer", lambda *a, **k: None)
    monkeypatch.setattr("vllm.benchmarks.datasets.datasets.get_samples", lambda args, tokenizer: [])
    monkeypatch.setattr(bench, "_arrival_offsets", lambda requests, args: [])
    monkeypatch.setattr(bench, "_run", failed_run)
    assert bench.main(["--model", "m", "--dataset-name", "random", "--num-prompts", "1"]) == 3
    assert seen == [True] and list(tmp_path.iterdir()) == []
