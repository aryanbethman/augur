"""augur.profile: everything that needs no GPU (arguments, grid, commands, resume markers, conversion of an
existing fork profile with the measured provenance)."""

import gzip
import json
import os
import subprocess

import pytest

from augur import profile as P
from augur.cost.profiles import default_fork

FORK = default_fork()
needs_fork = pytest.mark.skipif(not os.path.isdir(os.path.join(FORK, "workloads")),
                                reason="LLMServingSim fork not found (AUGUR_FORK)")

# What vLLM 0.33 logs at startup (gpu_worker.py, gpu_model_runner.py, kv_cache_utils.py), rank prefix included.
LOG = """\
(Worker_TP0 pid=1) INFO 11-21 10:00:00 [gpu_model_runner.py:5089] Model loading took 32.88 GiB memory and 12.5 seconds
(EngineCore pid=1) INFO 11-21 10:00:20 [gpu_worker.py:773] Available KV cache memory: 28.31 GiB
(EngineCore pid=1) INFO 11-21 10:00:20 [kv_cache_utils.py:1] GPU KV cache size: 371,008 tokens
(Worker_TP0 pid=1) INFO 11-21 10:00:30 [gpu_worker.py:1055] Free memory on device (79.1/79.65 GiB) on startup. Desired GPU memory utilization is (0.9, 71.69 GiB). Actual usage is 33.9 GiB for consumed memory (weights + non-torch), 4.02 GiB for peak activation, and 0.62 GiB for CUDAGraph memory. Replace gpu_memory_utilization config with `--kv-cache-memory=1` (1 GiB).
"""


def args(*extra, tmp=None):
    return P.build_parser().parse_args(["--model", "org/m", "--hardware", "H100", "--tp", "4", *extra])


def test_grid_reaches_the_requested_maxima():
    g = P.grid_preview(32768, 8192, 1024)
    ax = g["axes"]
    assert ax["attention decode KV"][0] == 0 and ax["attention decode KV"][-1] == 32768
    assert ax["dense tokens"][-1] == 8192 and ax["attention decodes"][-1] == 1024
    assert ax["attention decodes"][:5] == [1, 2, 3, 4, 6]  # sqrt(2) spacing, as the profiler
    assert P.geometric(100) == [1, 2, 4, 8, 16, 32, 64, 100]
    assert g["attention_shots_upper_bound"] == 15 * 17 * 20 * 17


def test_profiler_commands():
    a = args("--max-kv", "16384", "--no-skew")
    a.max_model_len = 16386
    cmd = P.profiler_command(a)
    assert cmd[1:5] == ["-m", "profiler", "profile", "org/m"]
    assert cmd[cmd.index("--tp") + 1] == "1,4"  # the profiler needs TP 1 in the sweep
    for flag, val in (("--attention-max-kv", "16384"), ("--max-num-batched-tokens", "8192"),
                      ("--max-num-seqs", "1024"), ("--max-model-len", "16386")):
        assert cmd[cmd.index(flag) + 1] == val
    assert "--skip-skew" in cmd and "--only-skew" not in cmd
    assert "--only-skew" in P.profiler_command(a, only_skew=True)
    one = args()
    one.tp, one.max_model_len = 1, 5
    assert P.profiler_command(one)[P.profiler_command(one).index("--tp") + 1] == "1"
    one.force, one.profiler_arg = True, ["--measurement-iterations", "5"]
    assert P.profiler_command(one)[-3:] == ["--force", "--measurement-iterations", "5"]


def test_parse_vllm_memory():
    m = P.parse_vllm_memory(LOG)
    assert m == {"weights_gib": 32.88, "available_kv_gib": 28.31, "kv_cache_tokens": 371008,
                 "free_at_startup_gib": 79.1, "gpu_total_gib": 79.65, "consumed_gib": 33.9,
                 "peak_activation_gib": 4.02, "cuda_graph_gib": 0.62, "non_torch_gib": 1.02}
    assert P.parse_vllm_memory("nothing here") == {}


def test_config_comes_from_the_local_cache_never_the_network(tmp_path, monkeypatch):
    fork = tmp_path / "fork"
    (fork / "profiler").mkdir(parents=True)
    hub = tmp_path / "hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    with pytest.raises(P.ProfileError, match="allow-download"):
        P.ensure_config(str(fork), "org/m", False, False)
    snap = hub / "models--org--m" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text('{"model_type": "llama"}')
    path, how = P.ensure_config(str(fork), "org/m", False, False)
    assert path.read_text() == '{"model_type": "llama"}' and "Hugging Face cache" in how
    assert P.ensure_config(str(fork), "org/m", False, False)[1] == "fork configs/model"
    path2, how2 = P.ensure_config(str(fork), "org/other", True, True)
    assert not path2.exists() and "downloaded" in how2


def test_resume_markers(tmp_path):
    w = P.Work(tmp_path)
    cmd = ["python", "-m", "profiler", "x"]
    assert not w.done("layers", cmd)
    w.finish("layers", cmd)
    assert w.done("layers", cmd) and not w.done("layers", cmd + ["--force"]) and not w.done("skew", cmd)


def test_cluster_config_needs_hardware_numbers():
    c = P.cluster_config(args())
    inst = c["nodes"][0]["instances"][0]
    assert (c["link_bw"], inst["npu_mem"]["mem_bw"], inst["npu_mem"]["mem_size"], inst["tp_size"]) == (900.0, 3350.0, 80, 4)
    odd = P.build_parser().parse_args(["--model", "m/x", "--hardware", "A6000", "--tp", "2"])
    with pytest.raises(P.ProfileError, match="--link-bw"):
        P.cluster_config(odd)
    odd = P.build_parser().parse_args(["--model", "m/x", "--hardware", "A6000", "--tp", "2", "--link-bw", "32",
                                       "--npu-mem-bw", "768", "--hbm-gb", "48"])
    assert P.cluster_config(odd)["link_bw"] == 32


def test_memory_stage_skips_without_enough_gpus(tmp_path, capsys):
    a = args()
    P._memory(a, [{"index": 0}], tmp_path, [], tmp_path / "s.json", tmp_path / "m.json", {}, P.Work(tmp_path))
    assert "needs 4 visible GPUs" in capsys.readouterr().out
    assert not (tmp_path / "m.json").exists()


def test_dry_run_runs_nothing(tmp_path, monkeypatch, capsys):
    fork = tmp_path / "fork"
    (fork / "profiler").mkdir(parents=True)
    (fork / "profiler" / "__main__.py").write_text("")
    (fork / "profiler" / "core").mkdir()
    (fork / "profiler" / "core" / "skew.py").write_text("")
    cfg = fork / "configs" / "model" / "org"
    cfg.mkdir(parents=True)
    (cfg / "m.json").write_text('{"model_type": "llama", "max_position_embeddings": 131072}')

    def boom(*a, **k):
        raise AssertionError("dry run executed a command")

    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)
    rc = P.main(["--model", "org/m", "--hardware", "H100", "--tp", "4", "--fork", str(fork), "--dry-run",
                 "--work-dir", str(tmp_path / "w")])
    out = capsys.readouterr().out
    assert rc == 0 and "dry run: nothing executed" in out
    for word in ("[memory] would run", "[layers] would run", "[skew] would run", "[convert] would run",
                 "--attention-max-kv 32768", "--tp 1,4", "bf16-tp4.json.gz"):
        assert word in out
    assert not (tmp_path / "w").exists()
    rc = P.main(["--model", "org/m", "--hardware", "H100", "--fork", str(fork), "--dry-run", "--max-kv", "999999"])
    assert rc == 1  # beyond the model's context


def test_rejects_bad_arguments(capsys):
    assert P.main(["--model", "a/b", "--hardware", "H100", "--stages", "bogus", "--dry-run"]) == 2
    assert P.main(["--model", "a/b", "--hardware", "H100", "--max-kv", "4096", "--max-model-len", "100"]) == 2


@needs_fork
def test_converts_an_existing_fork_profile_with_provenance(tmp_path, monkeypatch):
    """The convert stage end to end on the fork's H100 Llama-3.1-70B TP4 bundle (CPU only): self-check passes and
    the measured environment and memory numbers land in the profile's provenance."""
    work = tmp_path / "w"
    work.mkdir()
    (work / "memory-tp4.json").write_text(json.dumps(P.parse_vllm_memory(LOG)))
    out = tmp_path / "p.json.gz"
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    rc = P.main(["--model", "meta-llama/Llama-3.1-70B", "--hardware", "H100", "--tp", "4", "--stages", "convert",
                 "--no-skew", "--fork", FORK, "--work-dir", str(work), "--out", str(out)])
    assert rc == 0
    prof = json.loads(gzip.decompress(out.read_bytes()))
    assert prof["provenance"]["vllm_memory"]["weights_gib"] == 32.88
    assert prof["provenance"]["augur_profile"]["max_kv"] == 32768
    assert prof["identity"]["tp"] == 4
    # finished with the same arguments: the second run skips it
    out.unlink()
    assert P.main(["--model", "meta-llama/Llama-3.1-70B", "--hardware", "H100", "--tp", "4", "--stages", "convert",
                   "--no-skew", "--fork", FORK, "--work-dir", str(work), "--out", str(out)]) == 0
    assert not out.exists()


def test_decode_q_lens_reach_the_profiler():
    a = args("--decode-q-lens", "1,4")
    a.max_model_len = 100
    cmd = P.profiler_command(a)
    assert cmd[cmd.index("--attention-decode-q-lens") + 1] == "1,4"
    assert "--attention-decode-q-lens" not in P.profiler_command(args())
