"""--cost roofline: monotonicity, TP sharding, the fitted parameters reproducing the fit, regression rows (no GPU)."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from augur.cost import SeqWork, StepShape
from augur.cost.roofline import (DEFAULT_PARAMS, Arch, RooflineCost, RooflineParams, attention_features, gpu_peaks,
                                 op_ns)
from augur.cost.tables import GPU_FACTS, bundled_profile
from augur.sim.models import _embedded_config


@pytest.fixture(scope="module")
def configs(tmp_path_factory):
    """config.json of the shipped Llama-3.1-8B (dense) and Qwen3-30B-A3B (MoE) profiles."""
    d = tmp_path_factory.mktemp("cfg")
    out = {}
    for name, hw, model in (("llama", "RTXPRO6000", "meta-llama/Llama-3.1-8B"),
                            ("moe", "RTXPRO6000", "Qwen/Qwen3-30B-A3B-Instruct-2507"),
                            ("qwen", "RTX4060Ti", "Qwen/Qwen3-1.7B")):
        path = d / f"{name}.json"
        path.write_text(_embedded_config(bundled_profile(hw, model)))
        out[name] = str(path)
    return out


def decode(n, kv):
    return StepShape(tuple(SeqWork(1, kv, True) for _ in range(n)))


def prefill(chunk, history=0):
    return StepShape((SeqWork(chunk, history, False),))


def test_every_gpu_with_flops_has_a_bandwidth():
    for hw, f in GPU_FACTS.items():
        assert (f.bf16_tflops is None) or (f.mem_bw_gbps or f.datasheet_bw_gbps), hw
    assert {"A100", "L40S", "H200", "B200"} <= set(GPU_FACTS)


def test_unknown_gpu_is_refused_with_the_known_ones():
    with pytest.raises(ValueError, match="--peak-tflops"):
        gpu_peaks("NoSuchGpu")
    assert gpu_peaks("NoSuchGpu", 100.0, 500.0) == (100e3, 500.0)


@pytest.mark.parametrize("name", ["llama", "moe", "qwen"])
def test_monotone_in_work(configs, name):
    cost = RooflineCost(configs[name], "H100")
    for series in ([decode(n, 512) for n in (1, 2, 4, 16, 64, 128)], [decode(16, kv) for kv in (16, 256, 2048, 8192)],
                   [prefill(c) for c in (16, 64, 256, 1024, 2048)], [prefill(256, h) for h in (0, 512, 4096)],
                   [StepShape(decode(32, 700).seqs + prefill(c).seqs) for c in (32, 256, 1024)]):
        ns = [cost.step_ns(s) for s in series]
        assert ns == sorted(ns) and ns[0] < ns[-1], ns


def test_faster_gpu_is_faster(configs):
    a100, h100, b200 = (RooflineCost(configs["llama"], hw) for hw in ("A100", "H100", "B200"))
    for shape in (decode(64, 1024), prefill(2048)):
        assert b200.step_ns(shape) < h100.step_ns(shape) < a100.step_ns(shape)


def test_padding_and_dummy_steps(configs):
    cost = RooflineCost(configs["llama"], "H100")
    plain = decode(5, 100)
    padded = StepShape(plain.seqs, padded_tokens=8)
    assert cost.step_ns(padded) > cost.step_ns(plain)
    dummy = StepShape((), padded_tokens=8, dummy=True)
    assert 0 < cost.step_ns(dummy) < cost.step_ns(padded)  # layers only: no logits, no sampler, no attention


def test_tp_shards_the_work(configs):
    one = RooflineCost(configs["llama"], "H100", tp=1)
    two = RooflineCost(configs["llama"], "H100", tp=2, link_bw=400.0, link_latency_ns=1000)
    for shape in (prefill(2048), decode(128, 1024)):
        ops1, ops2 = one.ops_ns(shape), two.ops_ns(shape)
        for k in ("qkv_proj", "gate_up_proj", "down_proj", "attention", "lm_head"):
            assert ops2[k] < ops1[k], k
    # a bandwidth-bound GEMM (one token) streams half the weights per GPU: the time above the kernel overhead halves
    p = DEFAULT_PARAMS
    t1, t2 = (op_ns(one.arch, one._flops, one._bw, tp, p, 1, 1, ((0, 0, 0, 0)), ((1, 1, 1, 1)))["gate_up_proj"]
              for tp in (1, 2))
    assert (t1 - p.gemm_ovh_ns) / (t2 - p.gemm_ovh_ns) == pytest.approx(2.0, rel=0.02)
    assert two.step_ns(prefill(2048)) > sum(two.ops_ns(prefill(2048)).values())  # plus the ring all-reduces
    assert two._coll["all_reduce_count"] == 2 * one.arch.layers + 1


def test_kv_heads_replicate_beyond_their_number():
    arch = Arch(hidden=1024, layers=2, heads=16, kv_heads=2, head_dim=64, inter=4096, vocab=1000, qk_norm=False)
    pre, dec = (0, 0, 0, 0), (1, 1024, 1, 8)
    t = [op_ns(arch, 1e5, 1e3, tp, DEFAULT_PARAMS, 8, 8, pre, dec)["attention"] for tp in (1, 2, 4)]
    assert t[0] > t[1] > t[2] or t[1] == pytest.approx(t[2], rel=0.3)  # kv heads: 2 -> 1 -> 1 per GPU


def test_attention_features():
    shape = StepShape((SeqWork(4, 10, False), SeqWork(1, 99, True), SeqWork(1, 49, True)))
    pre, dec = attention_features(shape)
    assert pre == (4 * 10 + 10, 14, 4, 1)  # q (c + (q+1)/2) = 4*10 + 4*5/2
    assert dec == (99 + 1 + 49 + 1, 100 + 50, 2, 2)


def test_moe_touches_all_experts_at_large_batches(configs):
    cost = RooflineCost(configs["moe"], "H100")
    small, large = cost.ops_ns(prefill(1)), cost.ops_ns(prefill(2048))
    assert 0 < small["moe_experts"] < large["moe_experts"]
    assert "qk_norm" in large and "gate_up_proj" in large and large["gate_up_proj"] == 0  # no dense MLP in a MoE model


def test_refuses_what_it_does_not_price():
    with pytest.raises(NotImplementedError):
        Arch.from_dict({"hidden_size": 1, "num_hidden_layers": 1, "num_attention_heads": 1, "kv_lora_rank": 512})


def test_params_json_round_trip():
    assert RooflineParams.from_json(DEFAULT_PARAMS.to_json()) == DEFAULT_PARAMS


def test_default_params_reproduce_the_fit():
    from augur.cost.roofline_fit import fit, load_profiles
    refit = fit(load_profiles()).to_json()
    for k, v in DEFAULT_PARAMS.to_json().items():
        assert refit[k] == pytest.approx(v, rel=0.01), k


def test_regression_rows(configs):
    """Pins today's prices (fitted parameters, GPU_FACTS peaks); not a measurement."""
    qwen = RooflineCost(configs["qwen"], "RTX4060Ti")
    assert qwen.step_ns(decode(32, 1024)) == REGRESSION["qwen_decode"]
    assert qwen.step_ns(prefill(1024, 512)) == REGRESSION["qwen_prefill"]
    moe = RooflineCost(configs["moe"], "RTXPRO6000")
    assert moe.step_ns(StepShape(decode(8, 512).seqs + prefill(256).seqs)) == REGRESSION["moe_mixed"]


def test_cli_prices_a_run(tmp_path, workload):
    env = dict(os.environ, VLLM_LOGGING_LEVEL="WARNING")
    out = tmp_path / "r.csv"
    cmd = [sys.executable, "-m", "augur.cli", "--model", "Qwen/Qwen3-1.7B", "--cost", "roofline", "--hardware",
           "RTX4060Ti", "--num-gpu-blocks", "600", "--max-num-seqs", "16", "--max-model-len", "8192",
           "--max-num-batched-tokens", "512", "--no-recorded-outputs", "--no-async-scheduling", "--quiet",
           "--workload", workload, "--output", str(out)]
    subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert len(out.read_text().splitlines()) == 61
    unknown = [*cmd[:7], "--hardware", "NoSuchGpu", "--hbm-gb", "16", *cmd[9:]]
    bad = subprocess.run(unknown, env=env, capture_output=True, text=True)
    assert bad.returncode == 2 and "--peak-tflops" in bad.stderr


REGRESSION = {"qwen_decode": 32562790, "qwen_prefill": 112842250, "moe_mixed": 49811965}
