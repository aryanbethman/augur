"""Cluster routers (augur.sim.router): the vLLM Router's policies on hand-built calls and engines, then
whole runs through augur with agent sessions on four engines."""

import csv
import json
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict

import pytest

from augur.sim.router import ROUTERS, make_router
from augur.trace.workload import Call

S = 1_000_000_000  # ns per second


class Engine:
    def __init__(self, engine_id: int, load: int = 0):
        self.engine_id = engine_id
        self._load = load

    def load(self) -> int:
        return self._load


FOUR = [Engine(i) for i in range(4)]
PREFIX = list(range(10_000, 11_000))


def call(cid: str, ids, session: str | None = None) -> Call:
    return Call(call_id=cid, session_id=session, index=0, arrival_ns=0, prompt_ids=list(ids), output_toks=1,
                output_ids=[], tool_wait_ns=0)


# ---- cache-aware (the vLLM Router's default) ---------------------------------

def test_cache_aware_sends_a_shared_prefix_to_the_engine_holding_it():
    r = make_router("cache-aware")
    first = r.route(call("a", PREFIX + [1]), FOUR, 0)
    assert r.route(call("b", PREFIX + [2]), FOUR, 0) == first


def test_cache_aware_sends_an_unmatched_prompt_to_the_engine_with_fewest_requests_in_flight():
    r = make_router("cache-aware")
    assert r.route(call("a", range(100)), FOUR, 0) == 0
    assert r.route(call("b", range(500, 600)), FOUR, 0) == 1


def test_cache_aware_ignores_a_match_at_or_below_the_threshold():
    r = make_router("cache-aware", cache_threshold=0.5)
    r.route(call("a", PREFIX), FOUR, 0)
    half = PREFIX[:500] + list(range(500))  # matches exactly 500 / 1000
    assert r.route(call("b", half), FOUR, 0) == 1


def test_cache_aware_balances_load_once_in_flight_counts_are_imbalanced():
    r = make_router("cache-aware")
    for i in range(65):  # engine 0 reaches 65 in flight: 65 - 0 > 64 and 65 > 0 * 1.5
        assert r.route(call(f"c{i}", PREFIX + [i]), FOUR, 0) == 0
    assert r.route(call("x", PREFIX + [999]), FOUR, 0) == 1


def test_cache_aware_counts_a_request_until_it_is_released():
    r = make_router("cache-aware")
    calls = [call(f"c{i}", PREFIX + [i]) for i in range(65)]
    for c in calls:
        r.route(c, FOUR, 0)
    for c in calls:
        r.release(c)
    assert r.route(call("x", PREFIX + [999]), FOUR, 0) == 0


def test_cache_aware_evicts_the_oldest_prompts_beyond_the_tree_size_at_the_eviction_interval():
    def route_a_again(at_ns):
        r = make_router("cache-aware", max_tree_size=8 * 1500)  # 1,500 tokens (8 characters each)
        a = call("a", PREFIX)
        assert r.route(a, FOUR, 0) == 0
        r.release(a)
        assert r.route(call("b", range(1000)), FOUR, 1 * S) == 0  # engine 0 now holds 2,000 tokens
        return r.route(call("a2", PREFIX + [7]), FOUR, at_ns)

    assert route_a_again(119 * S) == 0  # before the 120 s eviction pass: a is still in engine 0's tree
    assert route_a_again(121 * S) == 1  # evicted (oldest): no match, so the least-loaded engine


# ---- power-of-two -------------------------------------------------------------

def test_power_of_two_with_two_engines_always_takes_the_one_with_fewer_in_flight():
    r = make_router("power-of-two")
    two = FOUR[:2]
    counts = defaultdict(int)
    for i in range(50):
        counts[r.route(call(f"c{i}", [i]), two, 0)] += 1
        assert abs(counts[0] - counts[1]) <= 1


def test_power_of_two_with_one_engine_takes_it():
    assert make_router("power-of-two").route(call("a", [1]), FOUR[2:3], 0) == 2


def test_power_of_two_is_reproducible_for_a_seed():
    def picks(seed):
        r = make_router("power-of-two", seed=seed)
        return [r.route(call(f"c{i}", [i]), FOUR, 0) for i in range(40)]

    assert picks(3) == picks(3)
    assert picks(3) != picks(4)


# ---- session affinity: consistent-hash, rendezvous-hash, sticky-least-loaded ---

@pytest.mark.parametrize("name", ["consistent-hash", "rendezvous-hash", "sticky-least-loaded"])
def test_session_policies_keep_every_call_of_a_session_on_one_engine(name):
    r = make_router(name)
    for s in range(30):
        picked = {r.route(call(f"s{s}#{i}", range(i * 10, i * 10 + 50), session=f"s{s}"), FOUR, i * S)
                  for i in range(10)}
        assert len(picked) == 1


@pytest.mark.parametrize("name", ["consistent-hash", "rendezvous-hash", "sticky-least-loaded"])
def test_session_policies_spread_sessions_over_every_engine(name):
    r = make_router(name)
    used = {r.route(call(f"s{s}#0", [s], session=f"s{s}"), FOUR, 0) for s in range(200)}
    assert used == {0, 1, 2, 3}


@pytest.mark.parametrize("name", ["consistent-hash", "rendezvous-hash"])
def test_hash_policies_move_only_the_sessions_of_a_removed_engine(name):
    r = make_router(name)
    before = {s: r.route(call(f"s{s}", [s], session=f"s{s}"), FOUR, 0) for s in range(200)}
    after = {s: r.route(call(f"s{s}", [s], session=f"s{s}"), FOUR[:3], 0) for s in range(200)}
    assert all(after[s] == before[s] for s in before if before[s] != 3)


def test_hash_policies_without_session_ids_hash_the_prompt():
    r = make_router("consistent-hash", session_ids=False)
    picked = {r.route(call(f"s#{i}", range(i * 10, i * 10 + 50), session="s"), FOUR, 0) for i in range(20)}
    assert len(picked) > 1
    same = [r.route(call(f"x{i}", range(50)), FOUR, 0) for i in range(5)]
    assert len(set(same)) == 1


def test_sticky_least_loaded_gives_a_new_session_the_engine_with_fewest_sessions():
    r = make_router("sticky-least-loaded")
    first = {r.route(call(f"s{s}", [s], session=f"s{s}"), FOUR, 0) for s in range(4)}
    assert first == {0, 1, 2, 3}


def test_sticky_least_loaded_stops_counting_a_session_idle_past_its_expiry():
    r = make_router("sticky-least-loaded", session_expiry_s=100)
    home = r.route(call("a#0", [1], session="a"), FOUR, 0)
    for s in range(3):  # one session on each other engine, still active at 150 s
        r.route(call(f"s{s}#0", [s], session=f"s{s}"), FOUR, 0)
        r.route(call(f"s{s}#1", [s], session=f"s{s}"), FOUR, 90 * S)
    assert r.route(call("n#0", [9], session="n"), FOUR, 150 * S) == home


def test_every_policy_is_listed():
    assert set(ROUTERS) == {"round-robin", "least-loaded", "random", "power-of-two", "cache-aware",
                            "consistent-hash", "rendezvous-hash", "sticky-least-loaded"}


# ---- whole runs: agent sessions on four engines ---------------------------------

BASE = ["--model", "Qwen/Qwen3-1.7B", "--cost", "fixed", "--engines", "4",
        "--num-gpu-blocks", "2000", "--max-num-seqs", "16", "--max-model-len", "8192",
        "--max-num-batched-tokens", "1024", "--fixed-step-ms", "8", "--fixed-per-token-us", "15",
        "--no-async-scheduling", "--step-cpu-us", "0", "--quiet"]
N_SESSIONS, N_TURNS = 12, 6


@pytest.fixture(scope="module")
def sessions(tmp_path_factory):
    """Agent sessions: each turn's prompt is the previous prompt, its output and new tokens, so a turn
    hits the prefix cache only on the engine that ran the session's earlier turns."""
    rng = random.Random(1)
    path = tmp_path_factory.mktemp("sessions") / "sessions.jsonl"
    with open(path, "w") as f:
        for s in range(N_SESSIONS):
            prompt, subs = [rng.randrange(1000, 100_000) for _ in range(400)], []
            for _ in range(N_TURNS):
                out = [rng.randrange(1000, 100_000) for _ in range(40)]
                subs.append({"input_toks": len(prompt), "output_toks": len(out), "input_tok_ids": prompt,
                             "output_tok_ids": out, "tool_duration_ns": 50_000_000})
                prompt = prompt + out + [rng.randrange(1000, 100_000) for _ in range(120)]
            f.write(json.dumps({"session_id": f"s{s}", "arrival_time_ns": s * 100_000_000,
                                "sub_requests": subs}) + "\n")
    return str(path)


def run(tmp_path, workload, router):
    out = tmp_path / f"{router}.csv"
    subprocess.run([sys.executable, "-m", "augur.cli", *BASE, "--workload", workload, "--router", router,
                    "--output", str(out)], check=True, env=dict(os.environ, VLLM_LOGGING_LEVEL="WARNING"),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(out) as f:
        return list(csv.DictReader(f))


def engines_per_session(rows):
    by = defaultdict(set)
    for r in rows:
        by[r["session_id"]].add(r["instance id"])
    return [len(v) for v in by.values()]


@pytest.mark.parametrize("router", ["consistent-hash", "rendezvous-hash", "sticky-least-loaded"])
def test_session_routers_keep_each_session_on_one_engine_end_to_end(tmp_path, sessions, router):
    rows = run(tmp_path, sessions, router)
    assert len(rows) == N_SESSIONS * N_TURNS
    assert engines_per_session(rows) == [1] * N_SESSIONS


def test_cache_aware_hits_the_prefix_cache_more_than_round_robin_end_to_end(tmp_path, sessions):
    def cached(rows):
        return sum(int(r["cached_tokens"]) for r in rows)

    rr = run(tmp_path, sessions, "round-robin")
    ca = run(tmp_path, sessions, "cache-aware")
    assert len(ca) == len(rr) == N_SESSIONS * N_TURNS
    assert max(engines_per_session(rr)) > 1
    assert engines_per_session(ca) == [1] * N_SESSIONS
    assert cached(ca) > cached(rr)


def test_least_loaded_spreads_a_burst_still_in_the_api_server(tmp_path):
    """Calls routed but not yet in an engine's scheduler (--frontend-latency-ms) count as that engine's load."""
    burst = tmp_path / "burst.jsonl"
    burst.write_text("".join(json.dumps({"input_toks": 64, "output_toks": 8, "arrival_time_ns": 0}) + "\n"
                             for _ in range(8)))
    rows = run(tmp_path, str(burst), "least-loaded")
    assert Counter(r["instance id"] for r in rows) == {"0": 2, "1": 2, "2": 2, "3": 2}
