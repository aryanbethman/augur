"""The transfer fabric (augur.sim.fabric): max-min sharing, latency, FIFO streams, equivalence with the
simulator's own link models, and whole runs with --contention on."""

import csv
import json
import os
import random
import subprocess
import sys

import pytest

from augur.sim.fabric import Fabric, Topology


def run_to_end(fab: Fabric) -> None:
    fab.advance(10**15)


def test_two_flows_on_one_link_share_it_equally():
    f = Fabric({"a": 10.0})
    f.submit("x", None, ["a"], 1000, 0)
    f.submit("y", None, ["a"], 1000, 0)
    f.advance(0)
    assert f.rate("x") == f.rate("y") == 5.0
    run_to_end(f)
    assert f.info("x") == f.info("y") == (0, 200)


def test_three_flows_on_two_links_share_the_bottleneck_max_min_fairly():
    f = Fabric({"a": 10.0, "b": 4.0})
    f.submit("long", None, ["a", "b"], 10_000, 0)
    f.submit("on_a", None, ["a"], 10_000, 0)
    f.submit("on_b", None, ["b"], 10_000, 0)
    f.advance(0)
    assert (f.rate("long"), f.rate("on_b"), f.rate("on_a")) == (2.0, 2.0, 8.0)
    assert f.rates_through("a") == 10.0 and f.rates_through("b") == 4.0


def test_flows_joining_and_leaving_change_the_rates_of_the_others():
    f = Fabric({"a": 10.0})
    f.submit("a1", None, ["a"], 1000, 0)  # alone: would end at 100
    f.advance(50)
    f.submit("b", None, ["a"], 200, 50)  # a1 has 500 left; both at 5/ns: b ends at 90
    f.advance(50)
    assert f.rate("a1") == f.rate("b") == 5.0
    run_to_end(f)
    assert f.info("b")[1] == 90
    assert f.info("a1")[1] == 120  # 300 left at 90, alone again at 10/ns


def test_a_flow_alone_takes_latency_plus_bytes_over_the_slowest_resource():
    for nbytes, latency in [(1_000_003, 0), (7, 20_000), (0, 500), (123_456_789, 20_000)]:
        f = Fabric({"fast": 256.0, "slow": 25.0, "mid": 64.0})
        f.submit("x", None, ["fast", "slow", "mid"], nbytes, 1000, latency_ns=latency)
        run_to_end(f)
        assert f.info("x") == (1000, 1000 + latency + int(nbytes / 25.0))


def test_a_flows_latency_is_a_delay_before_it_uses_bandwidth():
    f = Fabric({"a": 10.0})
    f.submit("x", None, ["a"], 1000, 0)
    f.submit("y", None, ["a"], 1000, 0, latency_ns=100)
    f.advance(50)
    assert f.rate("x") == 10.0 and f.rate("y") == 0.0  # y has not reached the link
    run_to_end(f)
    assert f.info("x")[1] == 100 and f.info("y")[1] == 200


def test_a_stream_runs_its_flows_one_after_another_and_not_before_earliest():
    f = Fabric({"a": 10.0})
    f.submit("s1", "lane", ["a"], 1000, 0)
    f.submit("s2", "lane", ["a"], 1000, 0)
    f.submit("s3", "lane", ["a"], 1000, 500)
    f.submit("other", "elsewhere", ["a"], 1000, 0)
    run_to_end(f)
    # the other stream's flow shares the link with s1 (both at 5), then s2
    assert f.info("s1") == (0, 200) and f.info("other") == (0, 200)
    assert f.info("s2") == (200, 300)
    assert f.info("s3") == (500, 600)


def test_projected_ends_are_the_ends_the_flows_then_get():
    rng = random.Random(1)
    f = Fabric({"a": 8.0, "b": 5.0, "c": 20.0})
    for i in range(12):
        path = rng.sample(["a", "b", "c"], rng.randint(1, 3))
        f.submit(i, f"s{i % 3}", path, rng.randint(100, 5000), rng.randint(0, 400), rng.choice([0, 30]))
    projected = {i: f.info(i) for i in range(12)}
    run_to_end(f)
    assert {i: f.info(i) for i in range(12)} == projected


def test_a_projection_over_linked_flows_only_gets_the_same_ends():
    """info() simulates only the flows linked to the one asked about (shared resources, PCIe duplex
    couplings, streams); across hosts the projected ends are still the ends the flows get."""
    rng = random.Random(3)
    topo = Topology.build([(i, 1) for i in range(4)], gpus_per_host=2, h2d_gbps=25.0, tiers={"cxl": 30.0})
    f = topo.fabric
    for i in range(40):
        net = topo.net(rng.randrange(4))
        path = rng.choice([net.store_path(), net.load_path(), net.tier_write_path("cxl"), net.tier_read_path("cxl")])
        f.submit(i, (net.gpu, rng.randrange(2)), path, rng.randint(1 << 10, 1 << 24), rng.randint(0, 10**5),
                 rng.choice([0, 500]))
    projected = {i: f.info(i) for i in range(40)}
    run_to_end(f)
    assert {i: f.info(i) for i in range(40)} == projected


def test_a_run_repeats_exactly():
    def once():
        rng = random.Random(7)
        f = Fabric({"a": 8.0, "b": 5.0})
        for i in range(30):
            f.submit(i, f"s{i % 4}", rng.sample(["a", "b"], rng.randint(1, 2)), rng.randint(1, 9000),
                     rng.randint(0, 2000), rng.choice([0, 17]))
            f.advance(rng.randint(0, 2000))
        run_to_end(f)
        return [f.info(i) for i in range(30)]
    assert once() == once()


def test_forced_end_replaces_the_flows_own_end():
    f = Fabric({"a": 10.0})
    f.submit("x", None, ["a"], 1000, 0, forced_end=500)
    f.submit("y", None, ["a"], 1000, 0, forced_end=3)
    run_to_end(f)
    assert f.info("x")[1] == 500 and f.info("y")[1] == 3


def test_cancel_frees_the_bandwidth():
    f = Fabric({"a": 10.0})
    f.submit("x", None, ["a"], 1000, 0)
    f.submit("y", None, ["a"], 1000, 0)
    f.advance(40)  # 800 left each
    f.cancel("y")
    run_to_end(f)
    assert f.info("x")[1] == 40 + 80


def test_a_resource_needs_bandwidth_and_a_flow_a_known_resource():
    with pytest.raises(ValueError):
        Fabric({"a": 0.0})
    f = Fabric({"a": 1.0})
    with pytest.raises(KeyError):
        f.submit("x", None, ["nope"], 1, 0)


# ---- processor sharing on one link, the reference for rounding="ceil" ------------

class SharedLink:
    """n active transfers each get bandwidth / n. Remaining bytes are advanced whenever the set changes; the next
    completion is (re)scheduled as an event tagged with a version, so stale ones are ignored. (The coordinator's
    --pd-contention decode model before it ran on the fabric.)"""

    def __init__(self, bw_gbps: float):
        self.bw = bw_gbps  # bytes per ns
        self.active: dict = {}  # key -> [remaining bytes, payload]
        self.last = 0

    def advance(self, now: int) -> None:
        if self.active and now > self.last:
            share = (now - self.last) * self.bw / len(self.active)
            for v in self.active.values():
                v[0] -= share
        self.last = now

    def next_finish(self, now: int) -> int | None:
        if not self.active:
            return None
        least = min(v[0] for v in self.active.values())
        return now + max(0, int(-(-least * len(self.active) // self.bw)))


def shared_link_ends(joins, bw):
    """coordinator.SharedLink driven as the coordinator drives it: a join and each completion is an event."""
    link, ends, pending = SharedLink(bw), {}, sorted(joins)
    t = 0
    version = 0
    events = [(at, "join", (k, n)) for at, k, n in pending]
    import heapq
    heapq.heapify(events)

    def reschedule(now):
        nonlocal version
        version += 1
        nf = link.next_finish(now)
        if nf is not None:
            heapq.heappush(events, (nf, "link", version))
    while events:
        t, kind, p = heapq.heappop(events)
        if kind == "join":
            link.advance(t)
            link.active[p[0]] = [float(p[1]), p[0]]
            reschedule(t)
        else:
            if p != version:
                continue
            link.advance(t)
            for k in [k for k, v in link.active.items() if v[0] <= 0.5]:
                del link.active[k]
                ends[k] = t
            reschedule(t)
    return ends


def test_one_link_in_the_fabric_reproduces_shared_link_to_the_ns():
    rng = random.Random(3)
    for trial in range(40):
        bw = rng.choice([0.05, 2.0, 16.0, 7.3])
        joins = sorted((rng.randint(0, 3_000_000), i, rng.choice([0, rng.randint(1000, 40_000_000)])) for i in range(rng.randint(1, 12)))
        expect = shared_link_ends(joins, bw)
        f = Fabric({"link": bw})
        for at, i, n in joins:
            f.advance(at)
            f.submit(i, None, ["link"], n, at, rounding="ceil")
        run_to_end(f)
        got = {i: f.info(i)[1] for _, i, _ in joins}
        assert got == expect, (trial, bw)


# ---- whole runs ----------------------------------------------------------------------

MODEL = "Qwen/Qwen3-1.7B"
BASE = ["--model", MODEL, "--cost", "fixed", "--num-gpu-blocks", "400", "--max-num-seqs", "16",
        "--max-model-len", "4096", "--max-num-batched-tokens", "1024", "--fixed-step-ms", "8",
        "--fixed-per-token-us", "15", "--step-cpu-us", "0", "--quiet"]


@pytest.fixture(scope="module")
def sessions(tmp_path_factory):
    rng = random.Random(2)
    path = tmp_path_factory.mktemp("fabric") / "sessions.jsonl"
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
    return out.read_bytes()


def rows(raw: bytes):
    return list(csv.DictReader(raw.decode().splitlines()))


@pytest.mark.parametrize("sched", ["--no-async-scheduling", "--async-scheduling"])
def test_contention_on_changes_nothing_when_one_transfer_is_ever_active(tmp_path, sessions, sched):
    # one engine, PCIe links of 25 GB/s under a 256 GB/s host: its stores and loads never need more than
    # their own link, so the fabric gives each the whole link, as the lanes do
    args = [sched, "--kv-offload-gib", "1", "--kv-offload-gbps", "25"]
    off = run(tmp_path, sessions, "off", *args)
    # (no duplex loss and no HBM interference: those are the effects of a second transfer / of a transfer)
    on = run(tmp_path, sessions, "on", *args, "--contention", "on", "--d2h-duplex-penalty", "0",
             "--hbm-interference", "0")
    assert off == on


def test_contention_on_slows_offload_when_host_memory_is_the_bottleneck(tmp_path, sessions):
    args = ["--no-async-scheduling", "--kv-offload-gib", "1", "--kv-offload-gbps", "25"]
    off = rows(run(tmp_path, sessions, "off", *args))
    slow = rows(run(tmp_path, sessions, "slow", *args, "--contention", "on", "--host-mem-gbps", "2"))
    assert len(off) == len(slow) == 50
    assert sum(int(r["latency"]) for r in slow) > 1.03 * sum(int(r["latency"]) for r in off)


def test_contention_runs_repeat_exactly(tmp_path, sessions):
    args = ["--async-scheduling", "--engines", "2", "--router", "round-robin", "--kv-offload-gib", "1",
            "--kv-offload-gbps", "25", "--contention", "on", "--host-mem-gbps", "30"]
    assert run(tmp_path, sessions, "a", *args) == run(tmp_path, sessions, "b", *args)


PD = ["--prefill-engines", "1", "--decode-engines", "2"]


def test_a_decode_link_fabric_reproduces_pd_contention_decode(tmp_path, sessions):
    fab = tmp_path / "pd.json"
    fab.write_text(json.dumps({"pd": "decode-link"}))
    slow = ["--pd-link-bw", "0.05", "--no-async-scheduling"]  # slow enough that pulls overlap
    old = run(tmp_path, sessions, "old", *PD, *slow, "--pd-contention", "decode")
    new = run(tmp_path, sessions, "new", *PD, *slow, "--pd-contention", "decode", "--contention", "on",
              "--fabric", str(fab))
    assert old == new


def test_pd_with_offload_under_contention_runs_and_pulls_share_the_nic(tmp_path, sessions):
    args = [*PD, "--kv-offload-gib", "1", "--kv-offload-gbps", "25", "--no-async-scheduling"]
    off = rows(run(tmp_path, sessions, "off", *args))
    on = rows(run(tmp_path, sessions, "on", *args, "--contention", "on"))
    slow = rows(run(tmp_path, sessions, "slow", *args, "--contention", "on", "--pd-link-bw", "0.05"))
    assert len(off) == len(on) == len(slow) == 50
    assert sum(int(r["latency"]) for r in slow) > sum(int(r["latency"]) for r in on)


def test_hbm_interference_slows_steps_only_while_transfers_run(tmp_path, sessions):
    args = ["--no-async-scheduling", "--kv-offload-gib", "1", "--kv-offload-gbps", "25", "--contention", "on",
            "--hbm-bw-gbps", "300"]
    zero = rows(run(tmp_path, sessions, "k0", *args))
    some = rows(run(tmp_path, sessions, "k2", *args, "--hbm-interference", "2"))
    assert sum(int(r["latency"]) for r in some) > sum(int(r["latency"]) for r in zero)


def test_topology_default_paths_and_packing():
    topo = Topology.build([(0, 2), (1, 2), (2, 4), (3, 1)], gpus_per_host=4, h2d_gbps=25.0, mem_gbps=100.0,
                          tiers={"cxl": 30.0})
    assert [topo.gpu_of[i] for i in range(4)] == [0, 2, 4, 8]
    assert topo.net(2).host == 1 and topo.net(3).host == 2  # engine 2 starts on host 1's first GPU
    assert topo.net(1).store_path() == ["gpu2.d2h", "host0.pcie_root", "host0.mem"]
    assert topo.net(1).load_path() == ["host0.mem", "host0.pcie_root", "gpu2.h2d"]
    assert topo.net(1).tier_write_path("cxl") == ["host0.mem", "host0.tier.cxl"]
    assert topo.pd_path(0, 1) == ["gpu0.nic", "net", "gpu2.nic"]
    assert topo.fabric.cap["host0.pcie_root"] == 100.0  # two GPUs' links, both directions


# ---- replay of own logs stays exact with contention on ---------------------------------------

@pytest.mark.parametrize("mode", ["--no-async-scheduling", "--async-scheduling"])
def test_own_log_replay_is_exact_with_contention_on_round_robin_offload(tmp_path, sessions, mode):
    from test_multi_replay import assert_replays_exactly
    assert_replays_exactly(tmp_path, sessions, mode, "--engines", "2", "--router", "round-robin",
                           "--kv-offload-gib", "1", "--kv-offload-gbps", "25", "--contention", "on",
                           "--host-mem-gbps", "30")


@pytest.mark.parametrize("mode", ["--no-async-scheduling", "--async-scheduling"])
def test_own_log_replay_is_exact_with_contention_on_pd_offload(tmp_path, sessions, mode):
    from test_multi_replay import assert_replays_exactly
    assert_replays_exactly(tmp_path, sessions, mode, "--prefill-engines", "2", "--decode-engines", "2",
                           "--kv-offload-gib", "1", "--kv-offload-gbps", "25", "--contention", "on",
                           "--host-mem-gbps", "30", "--hbm-bw-gbps", "300", "--hbm-interference", "1")


# ---- calibration against the RTX 4060 Ti measurements (scripts/contention_bench.py, 2026-11-21) -------------

H2D, D2H = 12.7, 13.15  # one GPU alone, GB/s, flat from one KV block to 64 MB


def copy_rates(n_gpus, directions, background=0.0, mem=256.0):
    """Steady rates of long copies: directions is a list of (gpu, 'h2d' | 'd2h')."""
    topo = Topology.build([(i, 1) for i in range(n_gpus)], h2d_gbps=H2D, d2h_gbps=D2H, mem_gbps=mem,
                          host_mem_background_gbps=background)
    for i, (g, d) in enumerate(directions):
        net = topo.net(g)
        topo.fabric.submit(i, None, net.load_path() if d == "h2d" else net.store_path(), 10**12, 0)
    topo.fabric.advance(0)
    return [topo.fabric.rate(i) for i in range(len(directions))]


def test_one_gpu_alone_moves_its_measured_rate_in_each_direction():
    assert copy_rates(1, [(0, "h2d")]) == [H2D] and copy_rates(1, [(0, "d2h")]) == [D2H]


def test_both_directions_on_one_gpu_cost_d2h_fifteen_percent_and_h2d_nothing():
    h2d, d2h = copy_rates(1, [(0, "h2d"), (0, "d2h")])
    assert h2d == H2D
    assert d2h == pytest.approx(11.1, abs=0.15)  # measured 13.2 -> 11.1


def test_two_h2d_streams_on_one_gpu_split_the_link():
    a, b = copy_rates(1, [(0, "h2d"), (0, "h2d")])
    assert a == b == pytest.approx(6.35)  # measured 6.3 each


def test_two_gpus_on_an_idle_host_do_not_slow_each_other():
    alone = copy_rates(1, [(0, "h2d"), (0, "d2h")])
    for dirs in ([(0, "h2d"), (1, "h2d")], [(0, "d2h"), (1, "d2h")]):
        both = copy_rates(2, dirs)
        assert both == copy_rates(1, [(0, dirs[0][1])]) + copy_rates(1, [(0, dirs[1][1])])  # additive
    four = copy_rates(2, [(0, "h2d"), (0, "d2h"), (1, "h2d"), (1, "d2h")])
    assert four[:2] == alone and four[2:] == alone


def test_a_loaded_host_makes_gpus_share_memory_bandwidth():
    # measured with four memory-streaming host processes: same-direction copies on two GPUs 14 GB/s in all
    # (GPU0 -11-13%, GPU1 -56%), all four directions 24 GB/s. The fabric shares host memory fairly, so it
    # reproduces the aggregates and the direction of the effect, not the GPU asymmetry.
    idle = sum(copy_rates(2, [(0, "h2d"), (1, "h2d")]))
    loaded = copy_rates(2, [(0, "h2d"), (1, "h2d")], background=256 - 14)
    assert sum(loaded) == pytest.approx(14.0) and sum(loaded) < idle
    four = copy_rates(2, [(0, "h2d"), (0, "d2h"), (1, "h2d"), (1, "d2h")], background=256 - 24)
    assert sum(four) == pytest.approx(24.0)
    assert sum(four) < sum(copy_rates(2, [(0, "h2d"), (0, "d2h"), (1, "h2d"), (1, "d2h")]))


def test_host_memory_background_must_leave_some_bandwidth():
    with pytest.raises(ValueError):
        Topology.build([(0, 1)], mem_gbps=100.0, host_mem_background_gbps=100.0)
