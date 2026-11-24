"""Prefill/decode and data-parallel runs: every call finishes, and each mode pays for what it should."""

import csv
import dataclasses
import json
import os
import subprocess
import sys

import pytest

from augur.cost import FixedCost
from augur.sim.coordinator import Coordinator
from augur.sim.engine import EngineSpec, SimEngine
from augur.sim.models import model_dir
from augur.sim.router import make_router
from augur.trace.workload import Workload

MODEL = "Qwen/Qwen3-1.7B"
SCHEDULING = {"sync": "--no-async-scheduling", "async": "--async-scheduling"}
PD = ["--model", MODEL, "--cost", "fixed", "--prefill-engines", "2", "--decode-engines", "2", "--num-gpu-blocks", "2000",
      "--max-num-seqs", "16", "--max-model-len", "8192", "--max-num-batched-tokens", "512", "--fixed-step-ms", "8",
      "--no-recorded-outputs", "--step-cpu-us", "0", "--quiet", "--num-entries", "20"]


def pd_run(tmp_path, workload, name, *args):
    out = tmp_path / f"{name}.csv"
    subprocess.run([sys.executable, "-m", "augur.cli", *PD, "--workload", workload, "--output", str(out), *args],
                   check=True, env=dict(os.environ, VLLM_LOGGING_LEVEL="WARNING"),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(out) as f:
        return list(csv.DictReader(f))


@pytest.mark.parametrize("contention", ["none", "decode"])
@pytest.mark.parametrize("mode", SCHEDULING)
def test_pd_run_finishes_every_call_on_a_decode_engine(tmp_path, workload, mode, contention):
    rows = pd_run(tmp_path, workload, "pd", SCHEDULING[mode], "--pd-contention", contention)
    assert len(rows) == 20
    assert {r["instance id"] for r in rows} <= {"2", "3"}
    for r in rows:
        assert int(r["arrival"]) <= int(r["prefill_done"]) <= int(r["arrival"]) + int(r["TTFT"])


def test_pd_pulls_sharing_a_decode_link_take_longer(tmp_path, workload):
    slow = ["--no-async-scheduling", "--pd-link-bw", "0.05", "--decode-engines", "1"]
    alone = pd_run(tmp_path, workload, "alone", *slow, "--pd-contention", "none")
    shared = pd_run(tmp_path, workload, "shared", *slow, "--pd-contention", "decode")
    assert sum(int(r["latency"]) for r in shared) > sum(int(r["latency"]) for r in alone)


# vLLM runs data parallel in one engine per rank for MoE models only.
MOE_MODEL = "Qwen/Qwen3-30B-A3B"
MOE_CONFIG = {"architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe", "hidden_size": 2048,
              "intermediate_size": 6144, "moe_intermediate_size": 768, "num_hidden_layers": 48,
              "num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 128, "num_experts": 128,
              "num_experts_per_tok": 8, "max_position_embeddings": 262144, "vocab_size": 151936,
              "torch_dtype": "bfloat16", "tie_word_embeddings": False}


class RoundCost(FixedCost):
    """Every DP round costs the same on every rank."""

    def round_ns(self, shapes, first_rank=0):
        return [self.base_ns] * len(shapes)


def dp_run(tmp_path, workload, async_scheduling, cpu_ns=0):
    config = tmp_path / "moe.json"
    config.write_text(json.dumps(MOE_CONFIG))
    spec = EngineSpec(model_dir=model_dir("", MOE_MODEL, str(tmp_path), str(config)), model_name=MOE_MODEL,
                      kv_cache_bytes=1 << 30, max_model_len=8192, max_num_batched_tokens=512, max_num_seqs=16, num_gpu_blocks=600,
                      async_scheduling=async_scheduling, data_parallel_size=2)
    cost = RoundCost(base_ns=8_000_000)
    engines = [SimEngine(i, dataclasses.replace(spec, data_parallel_rank=i), cost) for i in range(2)]
    for e in engines:
        e.ctx.cpu_fixed_ns = cpu_ns
        e.step_log = []
    coord = Coordinator(engines, make_router("round-robin"), Workload(workload, max_entries=12, recorded_outputs=False))
    return coord, engines, coord.run()


@pytest.mark.parametrize("cpu_ns", [0, 2_000_000])
def test_sync_dp_round_pays_the_step_cpu_time(tmp_path, workload, cpu_ns):
    coord, engines, rec = dp_run(tmp_path, workload, False, cpu_ns)
    assert len(rec.done) == 12 and coord.dp_rounds > 0
    assert {row[1] for e in engines for row in e.step_log} == {8_000_000 + cpu_ns}


def _span(engines):
    return max(row[0] + row[1] for e in engines for row in e.step_log)


def test_async_dp_round_hides_step_cpu_time_like_one_engine(tmp_path, workload):
    """GPU round 8 ms. CPU time under it is hidden (async) except at the start of a busy period; above it,
    the CPU is the bottleneck and every round is spaced by the CPU time. Sync pays it on every round."""
    span = {}
    for mode, cpu in [("a", 0), ("a", 2_000_000), ("a", 20_000_000), ("s", 0), ("s", 2_000_000)]:
        coord, engines, rec = dp_run(tmp_path, workload, mode == "a", cpu)
        assert len(rec.done) == 12
        span[mode, cpu] = _span(engines)
    assert span["a", 0] < span["a", 2_000_000] - 1_000_000 < span["s", 2_000_000] - 1_000_000
    assert span["a", 2_000_000] - span["a", 0] < (span["s", 2_000_000] - span["s", 0]) / 3
    assert span["a", 20_000_000] > span["a", 2_000_000] + 20_000_000


def test_async_dp_first_round_waits_for_the_cpu_when_the_gpu_is_idle(tmp_path, workload):
    _, engines, _ = dp_run(tmp_path, workload, True, 20_000_000)
    first_arrival = min(row[0] for e in engines for row in e.step_log)
    _, base, _ = dp_run(tmp_path, workload, True, 0)
    assert first_arrival == min(row[0] for e in base for row in e.step_log) + 20_000_000


def test_async_dp_run_finishes_every_call(tmp_path, workload):
    coord, _, rec = dp_run(tmp_path, workload, True)
    assert len(rec.done) == 12 and coord.dp_rounds > 0


def test_pd_engines_keep_no_recorded_ids_once_calls_finish(tmp_path):
    # Regression (2026-11-24): a prefill engine marked each request "recorded" for token-faithful speculation but only
    # dropped its output ids when the request left, so the set grew by one id per call for the whole run.
    path = tmp_path / "w.jsonl"
    with open(path, "w") as f:
        for i in range(6):
            f.write(json.dumps({"input_toks": 64, "output_toks": 4, "arrival_time_ns": i * 1_000_000,
                                "input_tok_ids": list(range(i * 100, i * 100 + 64)), "output_tok_ids": [7, 8, 9, 10]})
                    + "\n")
    spec = EngineSpec(model_dir=model_dir("", MODEL, str(tmp_path), None), model_name=MODEL, kv_cache_bytes=1 << 30,
                      max_model_len=8192, max_num_batched_tokens=512, max_num_seqs=16, num_gpu_blocks=600)
    engines = [SimEngine(i, dataclasses.replace(spec, kv_role=r), FixedCost(base_ns=8_000_000))
               for i, r in enumerate(["kv_producer", "kv_consumer"])]
    coord = Coordinator(engines, make_router("round-robin"), Workload(str(path)),
                        decode_router=make_router("round-robin"))
    assert len(coord.run().done) == 6
    assert [e.ctx.recorded for e in engines] == [set(), set()]
