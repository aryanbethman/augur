"""EAGLE / EAGLE-3 / MTP drafters on vLLM's real speculative path, without weights: draft config, KV for the draft
layers, the drafter's priced passes, and replay of augur's own log.

Needs vLLM only (synthetic workload, conftest.py); the pricing tests use the shipped Llama-3.1 profiles.
"""

import glob
import json
import os
import subprocess
import sys

import pytest

from augur.cli import main
from augur.cost import SeqWork, StepShape
from augur.cost.tables import TableCost, bundled_profile
from augur.sim import draft as D
from augur.verify.replay import compare

LLAMA = "meta-llama/Llama-3.1-8B"
BASE = ["--model", LLAMA, "--cost", "fixed", "--num-gpu-blocks", "600", "--max-model-len", "8192", "--max-num-seqs", "16",
        "--max-num-batched-tokens", "512", "--fixed-step-ms", "8", "--fixed-per-token-us", "15",
        "--no-recorded-outputs", "--step-cpu-us", "0", "--quiet"]
TINY_MTP = {"architectures": ["DeepseekV3ForCausalLM"], "model_type": "deepseek_v3", "vocab_size": 129280,
            "hidden_size": 1024, "intermediate_size": 4096, "moe_intermediate_size": 512, "num_hidden_layers": 4,
            "num_attention_heads": 16, "num_key_value_heads": 16, "n_routed_experts": 8, "n_shared_experts": 1,
            "num_experts_per_tok": 2, "first_k_dense_replace": 1, "kv_lora_rank": 128, "q_lora_rank": 256,
            "qk_nope_head_dim": 64, "qk_rope_head_dim": 32, "v_head_dim": 64, "num_nextn_predict_layers": 1,
            "n_group": 1, "topk_group": 1, "routed_scaling_factor": 2.5, "max_position_embeddings": 8192,
            "rope_theta": 10000, "rms_norm_eps": 1e-6, "tie_word_embeddings": False, "torch_dtype": "bfloat16",
            "hidden_act": "silu", "scoring_func": "sigmoid", "topk_method": "noaux_tc", "norm_topk_prob": True}


def augur(workload, *args, base=BASE):
    env = dict(os.environ, VLLM_LOGGING_LEVEL="WARNING")
    r = subprocess.run([sys.executable, "-m", "augur.cli", *base, "--workload", workload, *args], env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout[r.stdout.index("{"):])


@pytest.mark.parametrize("method", ["eagle", "eagle3"])
@pytest.mark.parametrize("k", [2, 3])
def test_runs_complete_and_decisions_differ_from_no_spec(workload, method, k):
    plain = augur(workload, "--no-async-scheduling")
    spec = augur(workload, "--no-async-scheduling", "--speculative-method", method, "--num-speculative-tokens", str(k))
    assert spec["calls"] == plain["calls"] == 60
    assert spec["steps"] < plain["steps"]  # accepted drafts: several tokens per step
    assert spec["tpot_ms"]["mean"] < plain["tpot_ms"]["mean"]


def test_async_run_completes(workload):
    spec = augur(workload, "--async-scheduling", "--speculative-method", "eagle", "--num-speculative-tokens", "3")
    assert spec["calls"] == 60 and spec["async_scheduling"] is True


def test_kv_cache_holds_the_draft_layers(workload):
    """The same KV memory holds fewer blocks: every block also carries the drafter's layer (1 of 32 + 1)."""
    args = ["--no-async-scheduling", "--kv-cache-gb", "8"]
    base = [a for i, a in enumerate(BASE) if a != "--num-gpu-blocks" and BASE[i - 1] != "--num-gpu-blocks"]
    plain = augur(workload, *args, base=base)
    eagle = augur(workload, *args, "--speculative-method", "eagle", "--num-speculative-tokens", "3", base=base)
    ratio = eagle["num_gpu_blocks"] / plain["num_gpu_blocks"]
    assert abs(ratio - 32 / 33) < 0.002, (plain["num_gpu_blocks"], eagle["num_gpu_blocks"])


def test_mtp_runs_for_a_model_that_defines_mtp_layers(workload, tmp_path):
    cfg = tmp_path / "ds.json"
    cfg.write_text(json.dumps(TINY_MTP))
    base = ["--model", "test/tiny-ds", "--model-config", str(cfg), "--cost", "fixed", "--num-gpu-blocks", "600",
            "--max-model-len", "4096", "--max-num-seqs", "16", "--max-num-batched-tokens", "512", "--fixed-step-ms", "8",
            "--no-recorded-outputs", "--step-cpu-us", "0", "--quiet"]
    plain = augur(workload, "--no-async-scheduling", base=base)
    mtp = augur(workload, "--no-async-scheduling", "--speculative-method", "mtp", "--num-speculative-tokens", "2",
                base=base)
    assert mtp["calls"] == 60 and mtp["steps"] < plain["steps"]


@pytest.mark.parametrize("mode", [["--no-async-scheduling"], ["--async-scheduling"]], ids=["sync", "async"])
def test_replay_of_own_log_is_exact(tmp_path, workload, mode):
    spec = ["--speculative-method", "eagle", "--num-speculative-tokens", "3"]
    augur(workload, *mode, *spec, "--step-log-vllm", str(tmp_path / "orig"))
    (orig,) = glob.glob(str(tmp_path / "orig.*.jsonl"))
    augur(workload, *mode, *spec, "--replay-durations", orig, "--step-log-vllm", str(tmp_path / "rep"))
    (rep,) = glob.glob(str(tmp_path / "rep.*.jsonl"))
    r = compare(orig, rep)
    assert r["steps_a"] > 50 and r["identical_steps"] == r["steps_a"] == r["steps_b"], r["at_first_difference"]


def test_refusals(workload, tmp_path, capsys):
    run = ["--model", "Qwen/Qwen3-1.7B", "--cost", "fixed", "--workload", workload, "--num-speculative-tokens", "3"]
    assert main([*run, "--speculative-method", "eagle"]) == 2
    assert "no eagle head" in capsys.readouterr().err  # vLLM 0.33 has Eagle3 for Qwen3, not Eagle
    assert main([*run, "--speculative-method", "mtp"]) == 2
    assert "defines no MTP layers" in capsys.readouterr().err
    assert main(["--model", LLAMA, "--cost", "fixed", "--workload", workload, "--num-speculative-tokens", "3",
                 "--speculative-method", "eagle", "--dp", "2"]) == 2
    assert "--dp" in capsys.readouterr().err


def test_draft_config_is_one_layer_of_the_targets_shape():
    target = {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "num_hidden_layers": 32,
              "hidden_size": 4096, "num_attention_heads": 32, "intermediate_size": 14336}
    d = D.draft_config(target, "eagle")
    assert d["num_hidden_layers"] == 1 and d["hidden_size"] == 4096 and target["num_hidden_layers"] == 32
    with pytest.raises(ValueError, match="dense"):
        D.draft_config({**target, "num_experts": 8}, "eagle")
    assert D.mtp_layers({"num_nextn_predict_layers": 1}) == 1 and D.mtp_layers(target) == 0


def test_drafter_passes_are_priced_from_the_target_layers():
    cost = TableCost(bundled_profile("RTXPRO6000", LLAMA, 1))
    shape = StepShape(seqs=tuple(SeqWork(4, 400 + 50 * i, True) for i in range(8)))
    one = StepShape(seqs=tuple(SeqWork(1, 400 + 50 * i, True) for i in range(8)))
    d1, d3 = (D.DraftCost(cost, "eagle", k, 1, 32) for k in (1, 3))
    assert d1.step_ns(shape, 8) == cost.draft_pass_ns(shape, 1, 8, 2)
    # k passes: the first over the verified tokens, k - 1 over one token per request
    second = cost.draft_pass_ns(StepShape(seqs=tuple(SeqWork(1, 400 + 50 * i + 4 + 1, True) for i in range(8))), 1, 8, 2)
    assert d3.step_ns(shape, 8) > d1.step_ns(shape, 8) + second
    assert d3.step_ns(shape, 0) == 0
    # a layer is a share of the step; the vocabulary projection and sampler make a pass cost more than 1/32
    assert cost.step_ns(one) // 32 < cost.draft_pass_ns(one, 1, 8, 0) < cost.step_ns(one) // 4
    assert cost.draft_pass_ns(one, 2, 8, 0) > cost.draft_pass_ns(one, 1, 8, 0)
    assert cost.draft_pass_ns(one, 1, 8, 3) > cost.draft_pass_ns(one, 1, 8, 2) > cost.draft_pass_ns(one, 1, 8, 0)


def test_tensor_parallel_drafter_pays_collectives():
    cost = TableCost(bundled_profile("H100", "meta-llama/Llama-3.1-70B", 4), cpu_mem_bw=512, link_bw=900,
                     link_latency_ns=0, npu_mem_bw=3350)
    one = StepShape(seqs=tuple(SeqWork(1, 800, True) for _ in range(8)))
    base = sum(cost._draft_comm_ns(8, 8, n) for n in (1,))
    assert cost._draft_comm_ns(8, 8, 2) > base > 0
    assert cost.draft_pass_ns(one, 1, 8, 2) > base


def test_fallback_prices_a_pass_as_a_share_of_the_step():
    from augur.cost import FixedCost
    fixed = FixedCost(base_ns=32_000_000)
    d = D.DraftCost(fixed, "eagle", 3, 1, 32)
    shape = StepShape(seqs=(SeqWork(4, 100, True),))
    assert d.step_ns(shape, 1) == 3 * 1_000_000
