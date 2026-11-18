"""KV offload to CPU (augur.sim.offload): vLLM's OffloadingConnector runs unmodified in the scheduler and
augur stands in for its GPU worker. First the worker's timing on vLLM's own job types, then whole runs
where a session's earlier turns are evicted from the GPU and come back from the CPU."""

import csv
import json
import os
import random
import subprocess
import sys

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    TransferJob,
)
from vllm.v1.kv_offload.base import GPULoadStoreSpec
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

from augur.sim.offload import OffloadLink, SimOffloadWorker

MS = 1_000_000
BLOCK_BYTES = 1_000_000  # one GPU block, all layers
LINK = OffloadLink(gbps=10.0, latency_ns=20_000)  # 10 GB/s = 10 bytes/ns: one block in 100 us + 20 us


def gpu(n: int) -> GPULoadStoreSpec:
    return GPULoadStoreSpec(list(range(n)), group_sizes=[n], block_indices=[0])


def store(req: str, blocks: int) -> TransferJob:
    return TransferJob(req_id=req, src_spec=gpu(blocks), dst_spec=CPULoadStoreSpec(list(range(blocks))))


def load(req: str, blocks: int, sync: bool = False) -> TransferJob:
    return TransferJob(req_id=req, src_spec=CPULoadStoreSpec(list(range(blocks))), dst_spec=gpu(blocks), sync=sync)


def meta(loads=None, stores=None, flush=None) -> OffloadingConnectorMetadata:
    return OffloadingConnectorMetadata(load_jobs=loads or {}, store_jobs=stores or {}, jobs_to_flush=flush or set())


def step(w: SimOffloadWorker, m, now: int, gpu_ns: int = 0):
    """One engine step as vLLM's GPU runner takes it: pre-forward hooks, the forward, post-forward hooks."""
    start = w.pre_forward(m, now_ns=now, gpu_free_ns=now)
    return w.post_forward(m, forward_end_ns=start + gpu_ns, poll_ns=start), start


def done(out):
    finished_recving, worker_meta = out
    return finished_recving, set(worker_meta.completed_jobs) if worker_meta else set()


def test_a_store_starts_at_the_next_step_and_is_reported_at_the_first_poll_after_it_ends():
    w = SimOffloadWorker(LINK, BLOCK_BYTES)
    assert done(step(w, meta(stores={1: store("a", 4)}), now=0, gpu_ns=10 * MS)[0]) == (set(), set())
    assert done(step(w, meta(), now=10 * MS)[0]) == (set(), set())  # submitted now, needs 420 us
    assert w.next_end() == 10 * MS + 420_000
    assert done(step(w, meta(), now=10 * MS + 419_999)[0]) == (set(), set())
    assert done(step(w, meta(), now=10 * MS + 420_000)[0]) == (set(), {1})
    assert w.next_end() is None


def test_an_async_load_starts_when_the_steps_forward_ends_and_finishes_its_request():
    w = SimOffloadWorker(LINK, BLOCK_BYTES)
    step(w, meta(loads={1: load("a", 2)}), now=0, gpu_ns=10 * MS)
    assert w.next_end() == 10 * MS + 220_000
    assert done(step(w, meta(), now=10 * MS + 220_000)[0]) == ({"a"}, {1})


def test_transfers_in_one_direction_queue_behind_each_other():
    w = SimOffloadWorker(LINK, BLOCK_BYTES)
    step(w, meta(loads={1: load("a", 2), 2: load("b", 2)}), now=0, gpu_ns=0)
    assert done(step(w, meta(), now=220_000)[0]) == ({"a"}, {1})
    assert done(step(w, meta(), now=440_000)[0]) == ({"b"}, {2})


def test_stores_and_loads_do_not_queue_behind_each_other():
    w = SimOffloadWorker(LINK, BLOCK_BYTES)
    step(w, meta(stores={1: store("a", 4)}), now=0)
    step(w, meta(loads={2: load("b", 2)}), now=0)  # the store is submitted here too
    assert done(step(w, meta(), now=420_000)[0]) == ({"b"}, {1, 2})


def test_a_flushed_store_holds_the_step_until_it_is_done():
    w = SimOffloadWorker(LINK, BLOCK_BYTES)
    step(w, meta(stores={1: store("a", 4)}), now=0)
    out, start = step(w, meta(flush={1}), now=MS)
    assert start == MS + 420_000
    assert done(out) == (set(), {1})


def test_a_sync_load_holds_the_step_and_finishes_no_request():
    w = SimOffloadWorker(LINK, BLOCK_BYTES)
    out, start = step(w, meta(loads={1: load("a", 3, sync=True)}), now=0)
    assert start == 320_000
    assert done(out) == (set(), {1})


def test_replayed_completion_times_override_the_link():
    w = SimOffloadWorker(LINK, BLOCK_BYTES, replay_done_ns={1: 5 * MS})
    step(w, meta(loads={1: load("a", 2)}), now=0)
    assert w.next_end() == 5 * MS
    assert done(step(w, meta(), now=5 * MS)[0]) == ({"a"}, {1})


def test_reports_bytes_and_time_moved():
    w = SimOffloadWorker(LINK, BLOCK_BYTES)
    step(w, meta(loads={1: load("a", 2)}), now=0)
    _, worker_meta = step(w, meta(), now=MS)[0]
    assert worker_meta.transfer_stats.load.bytes == 2 * BLOCK_BYTES
    assert worker_meta.transfer_stats.load.time == pytest.approx(220e-6)


# ---- whole runs -----------------------------------------------------------------

BASE = ["--model", "Qwen/Qwen3-1.7B", "--cost", "fixed",
        "--num-gpu-blocks", "400", "--max-num-seqs", "16", "--max-model-len", "4096",
        "--max-num-batched-tokens", "1024", "--fixed-step-ms", "8", "--fixed-per-token-us", "15",
        "--no-async-scheduling", "--step-cpu-us", "0", "--quiet"]
N_SESSIONS, N_TURNS = 10, 5


@pytest.fixture(scope="module")
def sessions(tmp_path_factory):
    """Agent sessions whose turns grow by 160 tokens and wait 2 s for a tool between turns: with 400 GPU
    blocks (6,400 tokens) for ten ~1,000-token contexts, a session's KV is usually evicted from the GPU
    before its next turn."""
    rng = random.Random(2)
    path = tmp_path_factory.mktemp("offload") / "sessions.jsonl"
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


def run(tmp_path, workload, name, *extra):
    out = tmp_path / f"{name}.csv"
    subprocess.run([sys.executable, "-m", "augur.cli", *BASE, "--workload", workload, "--output", str(out), *extra],
                   check=True, env=dict(os.environ, VLLM_LOGGING_LEVEL="WARNING"),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(out) as f:
        return list(csv.DictReader(f))


def cached(rows):
    return sum(int(r["cached_tokens"]) for r in rows)


def mean_latency(rows):
    return sum(int(r["latency"]) for r in rows) / len(rows)


def test_offload_brings_evicted_turns_back_from_cpu(tmp_path, sessions):
    gpu_only = run(tmp_path, sessions, "gpu_only")
    offload = run(tmp_path, sessions, "offload", "--kv-offload-gib", "1")
    assert len(gpu_only) == len(offload) == N_SESSIONS * N_TURNS
    assert cached(offload) > 2 * cached(gpu_only)


def test_a_slower_cpu_link_makes_calls_slower(tmp_path, sessions):
    fast = run(tmp_path, sessions, "fast", "--kv-offload-gib", "1", "--kv-offload-gbps", "100")
    slow = run(tmp_path, sessions, "slow", "--kv-offload-gib", "1", "--kv-offload-gbps", "0.2")
    assert mean_latency(slow) > 1.2 * mean_latency(fast)


def test_async_engine_schedules_again_within_a_step_of_a_load_being_reported(tmp_path, sessions):
    # Regression (2026-11-18): the async loop counted a delivery as progress when the batch carrying it
    # launched, not when its output reached the scheduler, so an otherwise idle engine stopped with the
    # loaded request ready (71 ms here, until an unrelated event woke it). Whether the request then runs
    # is vLLM's call (it waits when the GPU KV cache is full); the engine must at least schedule.
    run(tmp_path, sessions, "async", "--kv-offload-gib", "1", "--async-scheduling", "--step-log-vllm",
        str(tmp_path / "log"))
    events = [json.loads(line) for line in open(next(tmp_path.glob("log.*.jsonl")))]
    gaps = [next(x for x in events[i + 1:] if x["ev"] == "sched")["t"] - e["t"]
            for i, e in enumerate(events) if e.get("recv")]
    longest_step_ns = (8 + 1024 * 0.015) * MS
    assert gaps and max(gaps) <= longest_step_ns


def test_offload_runs_repeat_exactly_without_a_hash_seed(tmp_path, sessions):
    # Regression (2026-11-18): vLLM iterates sets of request ids (requests finished in a step), so the CPU
    # tier's eviction ranks, and later hits, followed Python's per-process string hashing. augur now pins
    # PYTHONHASHSEED at start-up.
    env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}
    env["VLLM_LOGGING_LEVEL"] = "WARNING"
    outputs = []
    for i in range(4):
        out = tmp_path / f"r{i}.csv"
        subprocess.run([sys.executable, "-m", "augur.cli", *BASE, "--workload", sessions, "--output", str(out),
                        "--kv-offload-gib", "1", "--async-scheduling"], check=True, env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        outputs.append(out.read_bytes())
    assert all(o == outputs[0] for o in outputs)


@pytest.mark.parametrize("sched", ["--no-async-scheduling", "--async-scheduling"])
def test_offload_runs_replay_their_own_logs_exactly(tmp_path, sessions, sched):
    from augur.verify.replay import compare

    log = tmp_path / "orig"
    run(tmp_path, sessions, "orig", "--kv-offload-gib", "1", sched, "--step-log-vllm", str(log))
    orig = next(tmp_path.glob("orig.*.jsonl"))
    rep = tmp_path / "rep"
    run(tmp_path, sessions, "rep", "--kv-offload-gib", "1", sched, "--replay-durations", str(orig),
        "--step-log-vllm", str(rep))
    r = compare(str(orig), str(next(tmp_path.glob("rep.*.jsonl"))))
    assert r["steps_a"] == r["steps_b"] == r["identical_steps"] > 0
