"""examples/kv_offload_policies.py: out-of-tree CPU-tier eviction policies, loaded the way a real vLLM 0.33 loads
them (kv_connector_extra_config eviction_policy + cache_policy_module_path), here inside augur. GPU-free."""

import json
import os
import subprocess
import sys
import textwrap

import pytest
from vllm.v1.kv_offload.base import ReqContext
from vllm.v1.kv_offload.cpu.policies.base import ChunkStatus
from vllm.v1.kv_offload.cpu.policies.factory import CachePolicyFactory

from examples.kv_offload_policies import ReuseFrequencyPolicy, ToolWaitPolicy, session_of
from tests.test_offload import BASE, sessions  # noqa: F401  (agent sessions that overflow the GPU)

MODULE = "examples.kv_offload_policies"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_the_session_is_read_from_augurs_and_real_vllms_request_ids():
    assert session_of("augur.astropy__astropy-12907-run0#3.") == "astropy__astropy-12907-run0"
    assert session_of("cmpl-augur.astropy__astropy-12907-run0#3.-0") == "astropy__astropy-12907-run0"
    assert session_of("cmpl-augur.astropy__astropy-12907-run0#3.-0-b103b954") == "astropy__astropy-12907-run0"  # a real log
    assert session_of("cmpl-augur.req_7.-0") == "req_7"
    assert session_of("some-other-id") == "some-other-id"


def test_vllm_resolves_the_plugin_by_name_and_module_path():
    assert CachePolicyFactory.get_cache_policy_cls("ToolWaitPolicy", MODULE) is ToolWaitPolicy
    assert CachePolicyFactory.get_cache_policy_cls("ReuseFrequencyPolicy", MODULE) is ReuseFrequencyPolicy


def key(i: int) -> bytes:
    return i.to_bytes(8, "big") + bytes(4)


def serve(policy, req_id: str, keys, reused=()):
    """One request: stores the keys it has not got (head to tail), then finishes."""
    ctx = ReqContext(req_id=req_id)
    for pos, k in enumerate(keys):
        ctx.set_offload_key_position(k, 16 * (pos + 1))
    for k in keys:
        if policy.get(k) is None:
            policy.insert(k, ChunkStatus(0), ctx)
            policy.get(k).ref_cnt = 0  # store done
            policy.mark_evictable(k)
    policy.on_request_finished((tuple(keys),), set(keys) - set(reused), set(reused), ctx)
    return ctx


def evicted(policy, n):
    return [k for k, _ in policy.evict(n, set())]


def test_tool_wait_evicts_a_running_sessions_blocks_before_a_waiting_sessions():
    p = ToolWaitPolicy(100)
    a, b = [key(i) for i in range(1, 4)], [key(i) for i in range(11, 14)]
    serve(p, "augur.a#0.", a)  # a finishes first: the oldest, and now waiting on a tool
    serve(p, "augur.b#0.", b)
    p.touch([b[0]], ReqContext(req_id="augur.b#1."))  # b's next call starts: b has a request in flight
    # LRU would take a's tail first (a is the oldest); b is running, so its blocks go first, tail before head
    assert evicted(p, 3) == [b[2], b[1], b[0]]
    assert evicted(p, 3) == [a[2], a[1], a[0]]


def test_tool_wait_is_lru_among_waiting_sessions():
    p = ToolWaitPolicy(100)
    a, b, c = [[key(10 * s + i) for i in range(2)] for s in (1, 2, 3)]
    for name, ks in (("a", a), ("b", b), ("c", c)):
        serve(p, f"augur.{name}#0.", ks)
    serve(p, "augur.a#1.", a, reused=a)  # a came back and finished again: now the most recent
    assert evicted(p, 4) == [b[1], b[0], c[1], c[0]]


def test_tool_wait_hands_priority_back_when_the_running_request_finishes():
    p = ToolWaitPolicy(100)
    a, b = [key(1), key(2)], [key(11), key(12)]
    serve(p, "augur.a#0.", a)
    serve(p, "augur.b#0.", b)
    run = ReqContext(req_id="augur.a#1.")
    for pos, k in enumerate(a):
        run.set_offload_key_position(k, 16 * (pos + 1))
    p.touch(a, run)  # a runs: its blocks go first
    p.on_request_finished((tuple(a),), set(), set(a), run)  # done: a waits again, as the freshest
    assert not p._inflight
    assert evicted(p, 4) == [b[1], b[0], a[1], a[0]]


def test_reuse_frequency_keeps_blocks_that_served_requests_over_newer_blocks_that_did_not():
    p = ReuseFrequencyPolicy(100)
    hot, cold = [key(1), key(2)], [key(11), key(12)]
    serve(p, "augur.a#0.", hot)
    serve(p, "augur.a#1.", hot, reused=hot)
    serve(p, "augur.b#0.", cold)  # newer, never reused
    assert evicted(p, 2) == [cold[1], cold[0]]
    assert evicted(p, 2) == [hot[1], hot[0]]


def test_evict_skips_protected_chunks_and_is_atomic():
    p = ToolWaitPolicy(100)
    ks = [key(i) for i in range(1, 5)]
    serve(p, "augur.a#0.", ks)
    assert [k for k, _ in p.evict(2, {ks[3]})] == [ks[2], ks[1]]
    assert p.evict(3, set()) is None  # ks[3] and ks[0] are left: nothing changed
    assert evicted(p, 2) == [ks[3], ks[0]]
    assert not p.chunks


@pytest.mark.parametrize("name", ["ToolWaitPolicy", "ReuseFrequencyPolicy"])
def test_augur_runs_the_plugin_as_the_cpu_tiers_policy(tmp_path, sessions, name):  # noqa: F811
    """A whole augur run with the plugin named on --kv-offload-extra-config: vLLM's CPUOffloadingManager calls its
    evict(), and it sees the sessions the workload ran in the request ids."""
    seen = tmp_path / "seen.json"
    script = textwrap.dedent(f"""
        import json, sys
        import examples.kv_offload_policies as m
        from augur.cli import main
        ev = {{"evict": 0, "sessions": set()}}
        cls = m.{name}
        orig_evict, orig_req = cls.evict, cls._request
        def evict(self, n, protected):
            ev["evict"] += n
            return orig_evict(self, n, protected)
        def _request(self, ctx):
            r = orig_req(self, ctx); ev["sessions"].add(r.session); return r
        cls.evict, cls._request = evict, _request
        rc = main(sys.argv[1:])
        json.dump({{"rc": rc, "evict": ev["evict"], "sessions": sorted(ev["sessions"])}}, open({str(seen)!r}, "w"))
    """)
    cfg = json.dumps({"eviction_policy": name, "cache_policy_module_path": MODULE})
    out = tmp_path / "out.csv"
    subprocess.run([sys.executable, "-c", script, *BASE, "--workload", sessions, "--output", str(out),
                    "--kv-offload-gib", "0.3", "--kv-offload-extra-config", cfg],
                   check=True, cwd=ROOT, env=dict(os.environ, VLLM_LOGGING_LEVEL="WARNING", PYTHONPATH=ROOT),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    got = json.load(open(seen))
    assert got["evict"] > 0  # 0.3 GiB is 175 blocks: ten ~1,000-token sessions overflow it
    assert got["sessions"] == [f"s{i}" for i in range(10)]
    assert len(open(out).readlines()) == 51  # header + 50 calls


def test_replicated_sessions_are_copies_with_their_own_tokens(sessions):  # noqa: F811
    from augur.trace.workload import Workload
    once = list(Workload(sessions).entries())
    wl = Workload(sessions, replicas=3, replica_jitter_s=1.0)
    many = list(wl.entries())
    assert len(many) == 3 * len(once)
    assert [s.arrival_ns for s in many] == sorted(s.arrival_ns for s in many)
    by_id = {s.session_id: s for s in many}
    assert set(by_id) == {f"{o.session_id}-r{k}" for o in once for k in range(3)}
    first = lambda s: wl.next_call(s, 0).prompt_ids  # noqa: E731
    for o in once:
        c0, c1, c2 = (first(by_id[f"{o.session_id}-r{k}"]) for k in range(3))
        assert c0 == first(o)  # copy 0 is the original
        assert len(c1) == len(c0) and all(len({x, y, z}) == 3 for x, y, z in zip(c0, c1, c2))  # a prefix shared by none
        assert by_id[f"{o.session_id}-r0"].arrival_ns == o.arrival_ns
        assert 0 <= by_id[f"{o.session_id}-r2"].arrival_ns - o.arrival_ns < 1_000_000_000
        assert all(t < 151643 for t in c2)
