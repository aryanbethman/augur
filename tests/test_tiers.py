"""Tiered KV offload (augur.sim.tiers): a simulated secondary tier (CXL or a remote DRAM pool) under vLLM's
own TieringOffloadingManager. First the tier on vLLM's job types and a hand-set clock, then whole runs
where the CPU tier is too small and evicted chunks come back from the secondary tier."""

import csv
import glob
import json
import os
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from vllm.v1.kv_offload.base import LookupResult, ReqContext
from vllm.v1.kv_offload.tiering.base import TransferJob

from augur.sim.tiers import SimMemoryTier, TierSpec, register_clock

CHUNK = 1_000_000  # bytes per chunk (the primary view's row)
NS = 10.0  # GB/s = bytes per ns: one chunk in 100 us
LAT = 5_000


class Clock(SimpleNamespace):
    pass


def make_tier(capacity_chunks=4, gbps=NS, latency_ns=LAT):
    clock = Clock(tier_clock_ns=0, tiers=[])
    key = f"test-{id(clock)}"
    register_clock(key, clock)
    view = memoryview(np.zeros((8, CHUNK), dtype=np.int8))
    tier = SimMemoryTier(offloading_spec=None, primary_kv_view=view, tier_type="SimMemoryTier", name="cxl",
                         capacity_bytes=capacity_chunks * CHUNK, gbps=gbps, latency_ns=latency_ns, clock=key)
    return tier, clock


_ids = iter(range(1, 1_000_000))


def job(keys, promotion=False):
    return TransferJob(job_id=next(_ids), keys=list(keys), chunk_ids=np.arange(len(keys), dtype=np.int32),
                       is_promotion=promotion, req_context=ReqContext(req_id="r"))


def finished(tier, clock, at):
    clock.tier_clock_ns = at
    return {r.job_id: r for r in tier.get_finished_jobs()}


def store(tier, clock, keys, at=0):
    clock.tier_clock_ns = at
    j = job(keys)
    tier.submit_store(j)
    return j


def test_a_stored_chunk_is_a_miss_until_its_write_completes():
    tier, clock = make_tier()
    j = store(tier, clock, [b"a", b"b"])
    assert tier.lookup(b"a", None) is LookupResult.MISS
    assert finished(tier, clock, LAT + 199_999) == {}
    done = finished(tier, clock, LAT + 200_000)
    assert done[j.job_id].success and done[j.job_id].transfer_bytes == 2 * CHUNK
    assert done[j.job_id].transfer_time == pytest.approx(205e-6)
    assert tier.lookup(b"a", None) is LookupResult.HIT


def test_a_load_takes_latency_plus_bytes_over_bandwidth():
    tier, clock = make_tier()
    store(tier, clock, [b"a", b"b", b"c"])
    finished(tier, clock, 10**9)
    clock.tier_clock_ns = 10**9
    j = job([b"a", b"b", b"c"], promotion=True)
    tier.submit_load(j)
    assert tier.next_end() == 10**9 + LAT + 300_000
    done = finished(tier, clock, 10**9 + LAT + 300_000)
    assert done[j.job_id].success and done[j.job_id].successful_keys is None


def test_writes_queue_behind_each_other_and_reads_do_not_wait_for_writes():
    tier, clock = make_tier(capacity_chunks=8)
    store(tier, clock, [b"a"])
    finished(tier, clock, 10**9)
    first = store(tier, clock, [b"b", b"c"], at=10**9)
    second = store(tier, clock, [b"d"], at=10**9)
    clock.tier_clock_ns = 10**9
    read = job([b"a"], promotion=True)
    tier.submit_load(read)
    done = finished(tier, clock, 10**9 + LAT + 100_000)
    assert set(done) == {read.job_id}  # the read ran beside the writes
    done = finished(tier, clock, 10**9 + 2 * LAT + 300_000)
    assert set(done) == {first.job_id, second.job_id}  # the second write started when the first ended


def test_a_full_tier_evicts_its_least_recently_used_chunk():
    tier, clock = make_tier(capacity_chunks=3)
    store(tier, clock, [b"a", b"b", b"c"])
    finished(tier, clock, 10**9)
    tier.touch([b"a"], None)
    store(tier, clock, [b"d"], at=10**9)
    finished(tier, clock, 2 * 10**9)
    assert [tier.lookup(k, None) for k in (b"a", b"b", b"c", b"d")] == [
        LookupResult.HIT, LookupResult.MISS, LookupResult.HIT, LookupResult.HIT]


def test_chunks_being_read_are_not_evicted_and_a_store_that_does_not_fit_fails():
    tier, clock = make_tier(capacity_chunks=2)
    store(tier, clock, [b"a", b"b"])
    finished(tier, clock, 10**9)
    clock.tier_clock_ns = 10**9
    tier.submit_load(job([b"a", b"b"], promotion=True))
    j = store(tier, clock, [b"c"], at=10**9)
    done = finished(tier, clock, 2 * 10**9)
    assert not done[j.job_id].success
    assert tier.lookup(b"c", None) is LookupResult.MISS
    assert tier.lookup(b"a", None) is LookupResult.HIT


def test_a_load_of_an_evicted_chunk_reports_the_chunks_that_did_load():
    tier, clock = make_tier(capacity_chunks=2)
    store(tier, clock, [b"a", b"b"])
    finished(tier, clock, 10**9)
    store(tier, clock, [b"c"], at=10**9)  # evicts a
    clock.tier_clock_ns = 10**9
    j = job([b"a", b"b"], promotion=True)
    tier.submit_load(j)
    done = finished(tier, clock, 2 * 10**9)
    assert not done[j.job_id].success and list(done[j.job_id].successful_keys) == [b"b"]


def test_storing_chunks_the_tier_already_holds_completes_at_once():
    tier, clock = make_tier()
    store(tier, clock, [b"a"])
    finished(tier, clock, 10**9)
    j = store(tier, clock, [b"a"], at=10**9)
    done = finished(tier, clock, 10**9)
    assert done[j.job_id].success and done[j.job_id].transfer_bytes == 0


def test_pending_work_and_next_end_follow_the_jobs_in_flight():
    tier, clock = make_tier()
    assert not tier.has_pending_work() and tier.next_end() is None
    store(tier, clock, [b"a"])
    assert tier.has_pending_work() and tier.next_end() == LAT + 100_000
    assert not tier.deliverable(LAT + 99_999) and tier.deliverable(LAT + 100_000)
    finished(tier, clock, LAT + 100_000)
    assert not tier.has_pending_work()


def test_drain_finishes_every_job_now():
    tier, clock = make_tier()
    j = store(tier, clock, [b"a", b"b"])
    clock.tier_clock_ns = 1_000
    tier.drain_jobs()
    assert j.job_id in finished(tier, clock, 1_000)
    assert tier.lookup(b"a", None) is LookupResult.HIT


@pytest.mark.parametrize("text, spec", [
    ("cxl:64:30", TierSpec("cxl", 64 * 2**30, 30.0, 0)),
    ("pool:0.5:12.5:250", TierSpec("pool", 2**29, 12.5, 250)),
])
def test_tier_spec_parses_name_gib_gbps_and_latency(text, spec):
    assert TierSpec.parse(text) == spec


@pytest.mark.parametrize("text", ["cxl", "cxl:64", "cxl:x:30", "cxl:0:30", "cxl:64:0", "cxl:64:30:-1",
                                  ":64:30", "cxl:64:30:1:2"])
def test_tier_spec_rejects_malformed_text(text):
    with pytest.raises(ValueError, match="NAME:GIB:GBPS"):
        TierSpec.parse(text)


# ---- whole runs -----------------------------------------------------------------

BASE = ["--model", "Qwen/Qwen3-1.7B", "--cost", "fixed",
        "--num-gpu-blocks", "400", "--max-num-seqs", "16", "--max-model-len", "4096",
        "--max-num-batched-tokens", "1024", "--fixed-step-ms", "8", "--fixed-per-token-us", "15",
        "--no-async-scheduling", "--step-cpu-us", "0", "--quiet"]
SMALL_CPU = ["--kv-offload-gib", "0.3"]  # 175 chunks of 1.75 MiB: one context fits, ten (~780 chunks) do not
TOO_SMALL_CPU = ["--kv-offload-gib", "0.1"]  # 58 chunks: less than one late turn's context (~78)
N_SESSIONS, N_TURNS = 10, 5


@pytest.fixture(scope="module")
def sessions(tmp_path_factory):
    """Agent sessions whose turns grow by 160 tokens and wait 2 s for a tool: a session's KV is usually
    evicted from the GPU, and with a small CPU tier from the CPU too, before its next turn."""
    rng = random.Random(3)
    path = tmp_path_factory.mktemp("tiers") / "sessions.jsonl"
    with open(path, "w") as f:
        for s in range(N_SESSIONS):
            prompt, subs = [rng.randrange(1000, 100_000) for _ in range(600)], []
            for _ in range(N_TURNS):
                out = [rng.randrange(1000, 100_000) for _ in range(40)]
                subs.append({"input_toks": len(prompt), "output_toks": len(out), "input_tok_ids": prompt,
                             "output_tok_ids": out, "tool_duration_ns": 2_000_000_000})
                prompt = prompt + out + [rng.randrange(1000, 100_000) for _ in range(120)]
            f.write(json.dumps({"session_id": f"s{s}", "arrival_time_ns": s * 50_000_000,
                                "sub_requests": subs}) + "\n")
    return str(path)


def augur(*args, env=None):
    env = dict(os.environ if env is None else env, VLLM_LOGGING_LEVEL="WARNING")
    return subprocess.run([sys.executable, "-m", "augur.cli", *BASE, *args], env=env,
                          capture_output=True, text=True)


def run(tmp_path, workload, name, *extra):
    out = tmp_path / f"{name}.csv"
    r = augur("--workload", workload, "--output", str(out), *extra)
    assert r.returncode == 0, r.stderr[-2000:]
    with open(out) as f:
        return list(csv.DictReader(f))


def cached(rows):
    return sum(int(r["cached_tokens"]) for r in rows)


def mean_latency(rows):
    return sum(int(r["latency"]) for r in rows) / len(rows)


def test_a_cxl_tier_brings_back_chunks_the_small_cpu_tier_evicted(tmp_path, sessions):
    cpu_only = run(tmp_path, sessions, "cpu_only", *SMALL_CPU)
    tiered = run(tmp_path, sessions, "tiered", *SMALL_CPU, "--kv-offload-tier", "cxl:2:30:250")
    assert len(cpu_only) == len(tiered) == N_SESSIONS * N_TURNS
    assert cached(tiered) > 1.5 * cached(cpu_only)


def test_a_slower_tier_makes_calls_slower(tmp_path, sessions):
    fast = run(tmp_path, sessions, "fast", *SMALL_CPU, "--kv-offload-tier", "cxl:2:100")
    slow = run(tmp_path, sessions, "slow", *SMALL_CPU, "--kv-offload-tier", "cxl:2:0.05")
    assert mean_latency(slow) > 1.2 * mean_latency(fast)


@pytest.mark.parametrize("sched", ["--no-async-scheduling", "--async-scheduling"])
def test_tiered_runs_replay_their_own_logs_exactly(tmp_path, sessions, sched):
    from augur.verify.replay import compare

    tier = ["--kv-offload-tier", "cxl:2:30:250"]
    run(tmp_path, sessions, "orig", *SMALL_CPU, *tier, sched, "--step-log-vllm", str(tmp_path / "orig"))
    orig = next(tmp_path.glob("orig.*.jsonl"))
    run(tmp_path, sessions, "rep", *SMALL_CPU, *tier, sched, "--replay-durations", str(orig),
        "--step-log-vllm", str(tmp_path / "rep"))
    r = compare(str(orig), str(next(tmp_path.glob("rep.*.jsonl"))))
    assert r["steps_a"] == r["steps_b"] == r["identical_steps"] > 0


def test_tiered_runs_repeat_exactly_without_a_hash_seed(tmp_path, sessions):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}
    outputs = []
    for i in range(3):
        out = tmp_path / f"r{i}.csv"
        r = augur("--workload", sessions, "--output", str(out), *SMALL_CPU, "--kv-offload-tier", "cxl:2:30",
                  "--async-scheduling", env=env)
        assert r.returncode == 0, r.stderr[-2000:]
        outputs.append(out.read_bytes())
    assert all(o == outputs[0] for o in outputs)


def test_tiers_leave_no_shared_memory_files(tmp_path, sessions):
    before = set(glob.glob("/dev/shm/vllm_offload_*"))
    run(tmp_path, sessions, "shm", *SMALL_CPU, "--kv-offload-tier", "cxl:2:30")
    assert set(glob.glob("/dev/shm/vllm_offload_*")) == before


def test_an_engine_that_stops_scheduling_is_reported_not_left_spinning(sessions):
    # Regression (2026-11-18): with a CPU tier smaller than one request's context, vLLM's tiering promotes
    # the request's chunks from the CXL tier and evicts them again to promote the rest, forever; the run
    # spun at 100% CPU. augur now stops after 60 s of simulated time without a token scheduled.
    r = subprocess.run([sys.executable, "-m", "augur.cli", *BASE, "--workload", sessions, *TOO_SMALL_CPU,
                        "--kv-offload-tier", "cxl:2:30:250"], capture_output=True, text=True, timeout=600,
                       env=dict(os.environ, VLLM_LOGGING_LEVEL="WARNING"))
    assert r.returncode != 0
    assert "scheduled no token for 60 s" in r.stderr and "--kv-offload-gib" in r.stderr


@pytest.mark.parametrize("args, message", [
    (["--kv-offload-tier", "cxl:2:30"], "--kv-offload-tier needs --kv-offload-gib"),
    (SMALL_CPU + ["--kv-offload-tier", "cxl:2"], "NAME:GIB:GBPS"),
    (SMALL_CPU + ["--kv-offload-tier", "cxl:2:30", "--kv-offload-tier", "cxl:4:10"], "named twice"),
])
def test_bad_tier_flags_are_refused_before_a_run(sessions, args, message):
    r = augur("--workload", sessions, *args)
    assert r.returncode == 2
    assert message in r.stderr
