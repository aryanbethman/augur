import pytest

from augur.cost.profiles import available, check


def make(tmp_path, *rel):
    for r in rel:
        (tmp_path / "profiler" / "perf" / r).mkdir(parents=True)
    return str(tmp_path)


def test_available_lists_profiles(tmp_path):
    fork = make(tmp_path, "RTX4060Ti/Qwen/Qwen3-1.7B/bfloat16/tp1",
                "H100/meta-llama/Llama-3.1-8B/bfloat16/tp2",
                "H100/Qwen/Qwen3-30B-A3B/bfloat16/tp1/moe_components/tp1")
    assert available(fork) == [
        ("H100", "Qwen/Qwen3-30B-A3B", "bfloat16", 1),
        ("H100", "meta-llama/Llama-3.1-8B", "bfloat16", 2),
        ("RTX4060Ti", "Qwen/Qwen3-1.7B", "bfloat16", 1),
    ]


def test_check_names_what_exists(tmp_path):
    fork = make(tmp_path, "RTX4060Ti/Qwen/Qwen3-1.7B/bfloat16/tp1")
    check(fork, "RTX4060Ti", "Qwen/Qwen3-1.7B", 1)
    with pytest.raises(FileNotFoundError, match="RTX4060Ti Qwen/Qwen3-1.7B bfloat16 tp1"):
        check(fork, "RTX4060Ti", "Qwen/Qwen3-1.7B", 2)


def test_check_without_fork(tmp_path):
    with pytest.raises(FileNotFoundError, match="AUGUR_FORK"):
        check(str(tmp_path), "H100", "x/y", 1)
