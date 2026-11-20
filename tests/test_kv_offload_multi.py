"""The vLLM Router's kv-aware policy (routes by the KV blocks each engine announces) and KV offload together
with prefill/decode (a MultiConnector of NIXL and the OffloadingConnector) and with data parallelism."""

import csv
import dataclasses
import json
import os
import random
import subprocess
import sys
from types import SimpleNamespace

import pytest
from vllm.distributed.kv_events import AllBlocksCleared, BlockRemoved, BlockStored

from augur.sim.coordinator import Coordinator
from augur.sim.engine import EngineSpec, SimEngine
from augur.sim.models import model_dir
from augur.sim.offload import OffloadLink
from augur.sim.router import KvAware, make_router
from augur.trace.workload import Call, Workload
from test_pd_dp import MOE_CONFIG, MOE_MODEL, RoundCost

MODEL = "Qwen/Qwen3-1.7B"
BLOCK = 16
N_SESSIONS, N_TURNS = 10, 5


@pytest.fixture(scope="module")
def sessions(tmp_path_factory):
    """Agent sessions whose turns grow by 160 tokens and wait 2 s for a tool: with 400 GPU blocks per engine
    a session's KV is usually evicted from the GPU before its next turn."""
    rng = random.Random(2)
    path = tmp_path_factory.mktemp("kvpd") / "sessions.jsonl"
    with open(path, "w") as f:
        for s in range(N_SESSIONS):
            prompt, subs = [rng.randrange(1000, 100_000) for _ in range(600)], []
            for _ in range(N_TURNS):
                out = [rng.randrange(1000, 100_000) for _ in range(40)]
                subs.append({"input_toks": len(prompt), "output_toks": len(out), "input_tok_ids": prompt,
                             "output_tok_ids": out, "tool_duration_ns": 2_000_000_000})
                prompt = prompt + out + [rng.randrange(1000, 100_000) for _ in range(120)]
            f.write(json.dumps({"session_id": f"s{s}", "arrival_time_ns": s * 50_000_000, "sub_requests": subs}) + "\n")
    return str(path)


# ---- kv-aware policy on hand-built engines and events ------------------------

class FakeEngine:
    """An engine whose block hashes are the prompt's block prefixes."""

    def __init__(self, engine_id, load=0):
        self.engine_id = engine_id
        self._load = load
        self.spec = SimpleNamespace(block_size=BLOCK)

    def load(self):
        return self._load

    def block_hasher(self, ids):
        return [hash(tuple(ids[:(i + 1) * BLOCK])) for i in range(len(ids) // BLOCK)]


ENGINES = [FakeEngine(i) for i in range(3)]


def call(cid, ids):
    return Call(call_id=cid, session_id=None, index=0, arrival_ns=0, prompt_ids=list(ids), output_toks=1,
                output_ids=[], tool_wait_ns=0)


def stored(ids, medium="GPU"):
    hashes = FakeEngine(0).block_hasher(ids)
    return BlockStored(block_hashes=hashes, parent_block_hash=None, token_ids=[], block_size=BLOCK, lora_id=None,
                       medium=medium, lora_name=None)


def removed(ids, medium="GPU", keep=0):
    return BlockRemoved(block_hashes=FakeEngine(0).block_hasher(ids)[keep:], medium=medium)


PROMPT = list(range(5000, 5000 + 40 * BLOCK))


def test_kv_aware_follows_the_engine_that_announced_the_prefix():
    r = make_router("kv-aware")
    r.observe(2, [stored(PROMPT)])
    assert r.route(call("a", PROMPT + [1, 2, 3]), ENGINES, 0) == 2


def test_kv_aware_falls_back_to_load_when_nothing_matches():
    r = make_router("kv-aware")
    engines = [FakeEngine(0, load=3), FakeEngine(1, load=1), FakeEngine(2, load=2)]
    assert r.route(call("a", range(100, 300)), engines, 0) == 1


def test_kv_aware_stops_following_after_blocks_are_removed_or_cleared():
    r = make_router("kv-aware")
    r.observe(1, [stored(PROMPT)])
    r.observe(1, [removed(PROMPT)])
    assert r.route(call("a", PROMPT), ENGINES, 0) == 0
    r.observe(2, [stored(PROMPT)])
    r.observe(2, [AllBlocksCleared()])
    assert r.route(call("b", PROMPT), ENGINES, 0) == 1  # engine 0 carries a, so the idle one wins the tie


def test_kv_aware_counts_only_the_run_from_the_first_block():
    r = make_router("kv-aware")
    r.observe(0, [stored(PROMPT)])
    r.observe(0, [removed(PROMPT, keep=0)])
    r.observe(0, [stored(PROMPT)])
    r.observe(0, [removed(PROMPT[:BLOCK], keep=0)])  # block 0 gone: nothing is reusable from engine 0
    r.observe(1, [stored(PROMPT[:5 * BLOCK])])
    assert r.route(call("a", PROMPT), ENGINES, 0) == 1


def test_kv_aware_credits_cpu_blocks_at_four_fifths_of_gpu_blocks():
    r = make_router("kv-aware")
    r.observe(0, [stored(PROMPT[:20 * BLOCK], "CPU")])  # CPU-only prefix of 20 blocks
    r.observe(1, [stored(PROMPT[:20 * BLOCK], "GPU")])
    assert r.route(call("a", PROMPT), ENGINES, 0) == 1  # GPU beats CPU
    r2 = make_router("kv-aware")
    r2.observe(0, [stored(PROMPT[:20 * BLOCK], "CPU")])
    assert r2.route(call("a", PROMPT), ENGINES, 0) == 0  # CPU beats nothing
    # GPU holds 10 blocks, CPU continues to 40: (10 + 30 * 0.8) beats 20 on GPU alone
    r3 = make_router("kv-aware")
    r3.observe(0, [stored(PROMPT[:10 * BLOCK], "GPU"), stored(PROMPT, "CPU")])
    r3.observe(1, [stored(PROMPT[:20 * BLOCK], "GPU")])
    assert r3.route(call("a", PROMPT + [7]), ENGINES, 0) == 0


def test_kv_aware_never_credits_the_last_token():
    r = KvAware()
    r.observe(0, [stored(PROMPT)])
    h = ENGINES[0].block_hasher(PROMPT)
    assert r._credited_tokens(0, h, len(PROMPT), BLOCK) == len(PROMPT) - BLOCK  # a whole-block prompt: last block recomputed


def test_kv_aware_active_prefill_steers_a_burst_away_from_a_loaded_engine():
    r = make_router("kv-aware")
    r.observe(0, [stored(PROMPT)])
    first = r.route(call("a", PROMPT + [1]), ENGINES, 0)
    assert first == 0
    big = call("b", PROMPT + [2])
    # engine 0 holds the prefix but carries a request in flight: cost (new + active) * (batch + 1)
    engines = [FakeEngine(0, load=1), FakeEngine(1), FakeEngine(2)]
    r.in_flight[0] = 1000  # (1 new + 1 active) * 1001 > 641 * 1
    assert r.route(big, engines, 0) in (1, 2)


# ---- kv-aware on real engines: vLLM's own events -----------------------------

def spec_for(tmp_path, **kw):
    return EngineSpec(model_dir=model_dir("", MODEL, str(tmp_path), None), model_name=MODEL, kv_cache_bytes=1 << 30,
                      max_model_len=4096, max_num_batched_tokens=1024, max_num_seqs=16, num_gpu_blocks=400, **kw)


def run_cluster(tmp_path, workload, router, n=2, **kw):
    from augur.cost import FixedCost
    spec = spec_for(tmp_path, kv_events=(router == "kv-aware"), **kw)
    engines = [SimEngine(i, spec, FixedCost(base_ns=8_000_000)) for i in range(n)]
    r = make_router(router)
    rec = Coordinator(engines, r, Workload(workload, recorded_outputs=False)).run()
    return r, rec


def test_kv_aware_keeps_a_session_on_the_engine_that_holds_its_blocks(tmp_path, sessions):
    _, rec = run_cluster(tmp_path, sessions, "kv-aware", n=4)
    where = {}
    for c in rec.done:
        where.setdefault(c.session_id, set()).add(c.instance_id)
    assert len(rec.done) == N_SESSIONS * N_TURNS
    assert all(len(v) == 1 for v in where.values())
    assert len({e for v in where.values() for e in v}) > 1  # and the sessions spread out


def test_kv_aware_hears_the_cpu_tier_when_offload_is_on(tmp_path, sessions):
    gpu_only, _ = run_cluster(tmp_path, sessions, "kv-aware", n=2)
    r, rec = run_cluster(tmp_path, sessions, "kv-aware", n=2, kv_offload_gib=1, kv_offload_link=OffloadLink(gbps=64))
    assert gpu_only.events_seen["GPU"] > 0 and not gpu_only.events_seen.get("CPU")
    assert r.events_seen["GPU"] > 0 and r.events_seen["CPU"] > 0
    assert len(rec.done) == N_SESSIONS * N_TURNS


def test_kv_aware_runs_repeat_exactly(tmp_path, sessions):
    runs = [[(c.call_id, c.instance_id, c.end_ns) for c in run_cluster(tmp_path, sessions, "kv-aware", n=3)[1].done]
            for _ in range(2)]
    assert runs[0] == runs[1]


# ---- KV offload with prefill/decode and with DP ------------------------------

PD = ["--model", MODEL, "--cost", "fixed", "--prefill-engines", "1", "--decode-engines", "1", "--num-gpu-blocks", "400",
      "--max-num-seqs", "16", "--max-model-len", "4096", "--max-num-batched-tokens", "1024", "--fixed-step-ms", "8",
      "--fixed-per-token-us", "15", "--step-cpu-us", "0", "--quiet"]
SCHEDULING = {"sync": "--no-async-scheduling", "async": "--async-scheduling"}


def pd_rows(tmp_path, workload, name, *extra):
    out = tmp_path / f"{name}.csv"
    subprocess.run([sys.executable, "-m", "augur.cli", *PD, "--workload", workload, "--output", str(out), *extra],
                   check=True, env=dict(os.environ, VLLM_LOGGING_LEVEL="WARNING"),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(out) as f:
        return list(csv.DictReader(f))


def cached(rows):
    return sum(int(r["cached_tokens"]) for r in rows)


@pytest.mark.parametrize("mode", SCHEDULING)
def test_pd_with_offload_brings_evicted_turns_back_from_the_cpu_tier(tmp_path, sessions, mode):
    plain = pd_rows(tmp_path, sessions, "plain", SCHEDULING[mode])
    both = pd_rows(tmp_path, sessions, "both", SCHEDULING[mode], "--kv-offload-gib", "1")
    assert len(plain) == len(both) == N_SESSIONS * N_TURNS
    assert {r["instance id"] for r in both} == {"1"}  # still served by the decode engine
    assert cached(both) > 2 * cached(plain)


def test_pd_with_offload_runs_repeat_exactly(tmp_path, sessions):
    a = pd_rows(tmp_path, sessions, "a", "--kv-offload-gib", "1", "--async-scheduling")
    b = pd_rows(tmp_path, sessions, "b", "--kv-offload-gib", "1", "--async-scheduling")
    assert a == b


def test_pd_with_offload_and_tiers_runs(tmp_path, sessions):
    rows = pd_rows(tmp_path, sessions, "tier", "--kv-offload-gib", "0.25", "--kv-offload-tier", "cxl:1:64")
    assert len(rows) == N_SESSIONS * N_TURNS


def test_multi_connector_worker_report_is_logged(tmp_path):
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiKVConnectorWorkerMetadata
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import OffloadingWorkerMetadata
    from vllm.v1.outputs import KVConnectorOutput

    from augur.verify.hooks import _kv_report
    inner = OffloadingWorkerMetadata()
    inner.mark_completed(7)
    out = SimpleNamespace(kv_connector_output=KVConnectorOutput(
        finished_recving={"r"}, kv_connector_worker_meta=MultiKVConnectorWorkerMetadata(metadata=(None, inner))))
    assert _kv_report(out) == {"jobs": [7], "recv": ["r"]}


def dp_run(tmp_path, workload, async_scheduling, offload):
    config = tmp_path / "moe.json"
    config.write_text(json.dumps(MOE_CONFIG))
    spec = EngineSpec(model_dir=model_dir("", MOE_MODEL, str(tmp_path), str(config)), model_name=MOE_MODEL,
                      kv_cache_bytes=1 << 30, max_model_len=4096, max_num_batched_tokens=1024, max_num_seqs=16,
                      num_gpu_blocks=300, async_scheduling=async_scheduling, data_parallel_size=2,
                      kv_offload_gib=1 if offload else None,
                      kv_offload_link=OffloadLink(gbps=64, latency_ns=20_000) if offload else None)
    cost = RoundCost(base_ns=8_000_000)
    engines = [SimEngine(i, dataclasses.replace(spec, data_parallel_rank=i), cost) for i in range(2)]
    coord = Coordinator(engines, make_router("round-robin"), Workload(workload, recorded_outputs=False))
    return coord, coord.run()


@pytest.mark.parametrize("mode", SCHEDULING)
def test_dp_with_offload_brings_evicted_turns_back_from_the_cpu_tier(tmp_path, sessions, mode):
    _, plain = dp_run(tmp_path, sessions, mode == "async", False)
    coord, both = dp_run(tmp_path, sessions, mode == "async", True)
    assert len(plain.done) == len(both.done) == N_SESSIONS * N_TURNS and coord.dp_rounds > 0
    assert sum(c.cached_tokens for c in both.done) > 1.5 * sum(c.cached_tokens for c in plain.done)


@pytest.mark.parametrize("mode", SCHEDULING)
def test_dp_with_offload_runs_repeat_exactly(tmp_path, sessions, mode):
    runs = [[(c.call_id, c.end_ns) for c in dp_run(tmp_path, sessions, mode == "async", True)[1].done]
            for _ in range(2)]
    assert runs[0] == runs[1]
