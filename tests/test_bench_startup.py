"""augur bench's start-up checks, before any simulation."""

import os
import subprocess
import sys


def test_bench_without_its_tokenizer_says_how_to_get_one():
    env = dict(os.environ, HF_HUB_OFFLINE="1", VLLM_LOGGING_LEVEL="WARNING", PYTHONHASHSEED="0")
    r = subprocess.run([sys.executable, "-m", "augur.bench", "--model", "Qwen/Qwen3-1.7B",
                        "--tokenizer", "augur-test/not-a-model", "--dataset-name", "random", "--num-prompts", "2"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 2
    assert "augur-test/not-a-model" in r.stderr and "--tokenizer" in r.stderr
    assert "Traceback" not in r.stderr
