"""Multi-engine step logs and replay: a run with several engines writes one log per engine, and replaying
those logs reproduces every engine's scheduling decisions step for step (and the calls' end times), for
round-robin engines, prefill/decode and data parallelism, with and without KV offload."""

import csv
import dataclasses
import glob
import json
import os
import random
import subprocess
import sys

import pytest

from augur.sim.coordinator import Coordinator, install_step_log_clock
from augur.sim.engine import EngineSpec, SimEngine
from augur.sim.models import model_dir
from augur.sim.offload import OffloadLink
from augur.sim.router import make_router
from augur.trace.workload import Workload
from augur.verify.replay import (ReplayCost, ReplayRouter, ReplayWorkload, compare, compare_runs, main,
                                 resolve_logs)
from test_pd_dp import MOE_CONFIG, MOE_MODEL, RoundCost

MODEL = "Qwen/Qwen3-1.7B"
BASE = ["--model", MODEL, "--cost", "fixed", "--num-gpu-blocks", "400", "--max-num-seqs", "16",
        "--max-model-len", "4096", "--max-num-batched-tokens", "1024", "--fixed-step-ms", "8",
        "--fixed-per-token-us", "15", "--step-cpu-us", "0", "--quiet"]
SCHEDULING = {"sync": "--no-async-scheduling", "async": "--async-scheduling"}


@pytest.fixture(scope="module")
def sessions(tmp_path_factory):
    """Agent sessions whose KV is usually evicted from the GPU between turns (so offload has work to do)."""
    rng = random.Random(2)
    path = tmp_path_factory.mktemp("replay") / "sessions.jsonl"
    with open(path, "w") as f:
        for s in range(10):
            prompt, subs = [rng.randrange(1000, 100_000) for _ in range(600)], []
            for _ in range(5):
                out = [rng.randrange(1000, 100_000) for _ in range(40)]
                subs.append({"input_toks": len(prompt), "output_toks": len(out), "input_tok_ids": prompt,
                             "output_tok_ids": out, "tool_duration_ns": 2_000_000_000})
                prompt = prompt + out + [rng.randrange(1000, 100_000) for _ in range(120)]
            f.write(json.dumps({"session_id": f"s{s}", "arrival_time_ns": s * 50_000_000, "sub_requests": subs}) + "\n")
    return str(path)


def run(tmp_path, workload, name, *extra):
    out = tmp_path / f"{name}.csv"
    subprocess.run([sys.executable, "-m", "augur.cli", *BASE, "--workload", workload, "--output", str(out), *extra],
                   check=True, env=dict(os.environ, VLLM_LOGGING_LEVEL="WARNING"),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(out) as f:
        return list(csv.DictReader(f))


def assert_replays_exactly(tmp_path, workload, *flags):
    """Run, replay the per-engine logs, and demand identical decisions on every engine and identical calls."""
    orig = run(tmp_path, workload, "orig", *flags, "--step-log-vllm", str(tmp_path / "orig"))
    rep = run(tmp_path, workload, "rep", *flags, "--replay-durations", str(tmp_path / "orig"),
              "--step-log-vllm", str(tmp_path / "rep"))
    results = compare_runs(str(tmp_path / "orig"), str(tmp_path / "rep"))
    assert len(results) >= 2
    for engine, r in results.items():
        assert r["steps_a"] == r["steps_b"] == r["identical_steps"] > 0, (engine, r)
        assert r["first_difference"] is None
    # (the clients' own timings differ: a replay has no front-end or delivery latency)
    where = lambda rows: sorted((int(r["input"]), int(r["output"]), r["instance id"], r["cached_tokens"], r["num_preemptions"]) for r in rows)
    assert where(orig) == where(rep) and len(orig) == 50
    return orig


@pytest.mark.parametrize("mode", SCHEDULING)
def test_round_robin_engines_replay_their_own_logs_exactly(tmp_path, sessions, mode):
    assert_replays_exactly(tmp_path, sessions, SCHEDULING[mode], "--engines", "2", "--router", "round-robin")


@pytest.mark.parametrize("mode", SCHEDULING)
def test_pd_replays_its_own_logs_exactly(tmp_path, sessions, mode):
    rows = assert_replays_exactly(tmp_path, sessions, SCHEDULING[mode], "--prefill-engines", "1",
                                  "--decode-engines", "1")
    assert {r["instance id"] for r in rows} == {"1"}


@pytest.mark.parametrize("mode", SCHEDULING)
def test_pd_with_offload_replays_its_own_logs_exactly(tmp_path, sessions, mode):
    assert_replays_exactly(tmp_path, sessions, SCHEDULING[mode], "--prefill-engines", "2", "--decode-engines", "2",
                           "--kv-offload-gib", "1")
    # offload jobs were logged and re-timed
    jobs = sum("jobs" in json.loads(line) for p in glob.glob(str(tmp_path / "orig.e*.jsonl")) for line in open(p))
    assert jobs > 0


@pytest.mark.parametrize("mode", SCHEDULING)
def test_round_robin_engines_with_offload_replay_exactly(tmp_path, sessions, mode):
    assert_replays_exactly(tmp_path, sessions, SCHEDULING[mode], "--engines", "2", "--router", "round-robin",
                           "--kv-offload-gib", "1")


def test_a_single_engine_keeps_its_log_name(tmp_path, sessions):
    run(tmp_path, sessions, "one", "--step-log-vllm", str(tmp_path / "one"))
    assert len(glob.glob(str(tmp_path / "one.*.jsonl"))) == 1 and not glob.glob(str(tmp_path / "one.e*"))


def test_each_engine_of_a_run_has_its_own_log(tmp_path, sessions):
    run(tmp_path, sessions, "two", "--engines", "2", "--router", "round-robin", "--step-log-vllm", str(tmp_path / "two"))
    assert [os.path.basename(p).split(".")[1] for p in resolve_logs(str(tmp_path / "two"), 2)] == ["e0", "e1"]


def test_compare_exits_nonzero_when_one_engine_differs(tmp_path, sessions, capsys):
    run(tmp_path, sessions, "a", "--engines", "2", "--router", "round-robin", "--step-log-vllm", str(tmp_path / "a"))
    run(tmp_path, sessions, "b", "--engines", "2", "--router", "round-robin", "--step-log-vllm", str(tmp_path / "b"))
    assert main(["compare", str(tmp_path / "a"), str(tmp_path / "b")]) == 0
    victim = resolve_logs(str(tmp_path / "b"), 2)[1]
    lines = open(victim).read().splitlines()
    i = next(i for i, line in enumerate(lines) if json.loads(line)["ev"] == "sched" and json.loads(line)["tok"])
    rec = json.loads(lines[i])
    rid = next(iter(rec["tok"]))
    rec["tok"][rid] += 1
    lines[i] = json.dumps(rec)
    open(victim, "w").write("\n".join(lines) + "\n")
    assert main(["compare", str(tmp_path / "a"), str(tmp_path / "b")]) == 1
    assert "engine 1" in capsys.readouterr().out


# ---- DP (in process: the CLI's fixed cost has no DP rounds) ----------------------------

def dp_cluster(tmp_path, workload, async_scheduling, offload, prefix, replay=None):
    config = tmp_path / "moe.json"
    config.write_text(json.dumps(MOE_CONFIG))
    os.environ["AUGUR_STEP_LOG"] = prefix
    spec = EngineSpec(model_dir=model_dir("", MOE_MODEL, str(tmp_path), str(config)), model_name=MOE_MODEL,
                      kv_cache_bytes=1 << 30, max_model_len=4096, max_num_batched_tokens=1024, max_num_seqs=16,
                      num_gpu_blocks=300, async_scheduling=async_scheduling, data_parallel_size=2, step_log_tag=True,
                      scheduler_cls="augur.verify.hooks." + ("LoggingAsyncScheduler" if async_scheduling
                                                             else "LoggingScheduler"),
                      kv_offload_gib=1 if offload else None,
                      kv_offload_link=OffloadLink(gbps=64, latency_ns=20_000) if offload else None)
    if replay:
        logs = resolve_logs(replay, 2)
        costs = ReplayCost.from_logs(logs)
        wl = ReplayWorkload(workload, log_path="", log_paths=logs, recorded_outputs=True)
        router = ReplayRouter.from_logs(logs)[0]
    else:
        costs = [RoundCost(base_ns=8_000_000)] * 2
        wl = Workload(workload, recorded_outputs=True)
        router = make_router("round-robin")
    engines = [SimEngine(i, dataclasses.replace(spec, data_parallel_rank=i), costs[i]) for i in range(2)]
    coord = Coordinator(engines, router, wl)
    install_step_log_clock(coord)
    rec = coord.run()
    for e in engines:
        e.core.scheduler._logger().flush()
    return rec


@pytest.mark.parametrize("offload", [False, True], ids=["plain", "offload"])
@pytest.mark.parametrize("mode", SCHEDULING)
def test_dp_replays_its_own_logs_exactly(tmp_path, sessions, mode, offload):
    orig = dp_cluster(tmp_path, sessions, mode == "async", offload, str(tmp_path / "orig"))
    rep = dp_cluster(tmp_path, sessions, mode == "async", offload, str(tmp_path / "rep"), replay=str(tmp_path / "orig"))
    assert len(orig.done) == len(rep.done) == 50
    for engine, r in compare_runs(str(tmp_path / "orig"), str(tmp_path / "rep")).items():
        assert r["steps_a"] == r["steps_b"] == r["identical_steps"] > 0, (engine, r)
    key = lambda rec: sorted((c.call_id, c.instance_id, c.first_sched_ns, c.end_ns, c.cached_tokens) for c in rec.done)
    assert key(orig) == key(rep)
    if offload:
        assert sum("jobs" in json.loads(line) for p in glob.glob(str(tmp_path / "orig.e*.jsonl"))
                   for line in open(p)) > 0
