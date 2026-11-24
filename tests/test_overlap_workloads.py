"""scripts/baselines/overlap_workloads.py: seeded workloads whose prompts overlap in part, and the prefix groups and
sessions BLIS's trace states for them (scripts/baselines/workloads.py, --blis-encoding best)."""

import csv
import hashlib
import json
import os
import random
import sys
from argparse import Namespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "baselines"))

import overlap_workloads as ow  # noqa: E402
import workloads as wk  # noqa: E402


def docs(n, lo, hi, seed):
    rng = random.Random(seed)
    return [[rng.randrange(1000, 150_000) for _ in range(rng.randint(lo, hi))] for _ in range(n)]


CHAT, SWE = docs(80, 300, 1500, 1), docs(12, 6000, 9000, 2)


def gen(pattern, seed=0, n=None):
    return ow.generate(pattern, seed, n, chat=CHAT, swe=SWE)


EXPECTED = {"tenants": "da9a09973e336940", "fewshot": "cec023fce1d0f5d5", "rag": "fd6d909416e6762e",
            "agent": "8abc5a676d67b203"}


def digest(lines):
    return hashlib.sha256(json.dumps(lines, separators=(",", ":")).encode()).hexdigest()[:16]


def flat_prompts(lines):
    return [d["input_tok_ids"] for d in lines]


def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def test_the_same_seed_gives_the_same_workload_and_another_seed_another():
    for p in ow.PATTERNS:
        n = 8 if p == "agent" else 60
        assert digest(gen(p, 0, n)) == digest(gen(p, 0, n))
        assert digest(gen(p, 0, n)) != digest(gen(p, 1, n))


def test_the_workloads_are_stable_across_runs():
    # a change to a generator changes every real run made from its files: these digests say when
    assert {p: digest(gen(p, 0, 8 if p == "agent" else 60)) for p in ow.PATTERNS} == EXPECTED


def test_sizes_arrivals_and_lengths_fit_the_engine():
    for p, (n, _rate) in ow.PATTERNS.items():
        lines = gen(p)
        assert len(lines) == n
        t = [d["arrival_time_ns"] for d in lines]
        assert t == sorted(t)
        for _no, _key, _sid, ids, n_out in ow.calls(lines):
            assert 1 <= len(ids) <= ow.MAX_PROMPT and 100 <= n_out <= 300
            assert all(1000 <= x < 150_000 for x in ids)
        for d in lines:
            for c in d.get("sub_requests", [d]):
                assert c["input_toks"] == len(c["input_tok_ids"])


def test_tenants_share_the_preamble_and_a_tenant_shares_its_system_prompt():
    prompts = flat_prompts(gen("tenants", n=200))
    assert min(lcp(prompts[0], p) for p in prompts) >= 768  # the preamble
    shared = sorted((lcp(a, b) for i, a in enumerate(prompts) for b in prompts[i + 1:]), reverse=True)
    assert shared[0] >= 768 + 200  # two requests of one tenant share the preamble and its system prompt
    # other tenants share the preamble and not much more
    assert sum(768 <= s < 768 + 16 for s in shared) > len(shared) // 4


def test_fewshot_prompts_lead_with_a_task_header_and_the_examples_they_have_in_common():
    prompts = flat_prompts(gen("fewshot", n=120))
    heads = {}
    for p in prompts:
        heads.setdefault(tuple(p[:150]), []).append(p)
    assert len(heads) == 3  # three tasks
    # two prompts of a task share the header and, often, more: leading examples
    longest = max(lcp(a, b) for g in heads.values() for i, a in enumerate(g) for b in g[i + 1:])
    assert longest > 1000
    assert ow.stats(gen("fewshot", n=120))["reusable_fraction"] > 0.2


def test_rag_documents_are_shared_only_behind_the_same_documents():
    prompts = flat_prompts(gen("rag", n=120))
    assert min(lcp(prompts[0], p) for p in prompts) >= 300
    # a prompt's tail (documents and question) is not reusable unless the documents before it are the same
    r = ow.reusable(prompts)
    assert 0 < sum(r) < sum(map(len, prompts))
    assert all(x % ow.BLOCK == 0 and x < len(p) for x, p in zip(r, prompts))


def test_agent_calls_continue_the_previous_prompt_or_keep_only_its_head():
    lines = gen("agent", n=10)
    head = lcp(lines[0]["sub_requests"][0]["input_tok_ids"], lines[1]["sub_requests"][0]["input_tok_ids"])
    assert head == 1000  # the system prompt is common to the sessions, the task is not
    cont = trimmed = 0
    for d in lines:
        subs = d["sub_requests"]
        assert subs[-1]["tool_duration_ns"] == 0 and all(c["tool_duration_ns"] > 0 for c in subs[:-1])
        for a, b in zip(subs, subs[1:]):
            n = lcp(a["input_tok_ids"], b["input_tok_ids"])
            if n == len(a["input_tok_ids"]):
                cont += 1
                # the previous output follows the previous prompt
                assert b["input_tok_ids"][n:n + a["output_toks"]] == a["output_tok_ids"]
            else:
                trimmed += 1
                assert 1000 + 600 <= n < len(a["input_tok_ids"])  # the system prompt and the task survive, then it differs
    assert cont > 0 and trimmed > 0


def test_reusable_counts_whole_blocks_against_the_best_earlier_prompt_and_never_the_last_token():
    a = list(range(100, 164))  # four blocks
    b = a[:40] + [9] * 30  # two shared blocks and 8 tokens
    c = list(a)  # all four blocks shared, but one token is always recomputed
    assert ow.reusable([a, b, c]) == [0, 32, 48]


def test_stats_state_the_reusable_tokens_and_what_prefix_groups_state_of_them():
    s = ow.stats(gen("tenants", n=100))
    assert s["calls"] == 100 and s["sessions"] == 0
    assert 0 < s["prefix_group_tokens"] <= s["reusable_tokens"] * 1.05
    assert s["prefix_group_tokens"] > s["greedy_group_tokens"]  # the improved encoding states more


def rows(prompts, outputs=None, sids=None):
    return [{"call_id": (f"{sids[i]}#{i}" if sids else f"req_{i}"), "blocks": wk._block_hashes(p), "input": len(p),
             "output": (outputs or [10] * len(prompts))[i], "arrival_s": float(i)} for i, p in enumerate(prompts)]


def blocks_of(token, n):  # n blocks of one value, so that equal values are equal blocks
    return [token] * (n * wk.BLOCK)


def test_encoding_two_tenants_behind_one_preamble_makes_a_group_per_tenant():
    pre = blocks_of(1, 4)
    a, b = blocks_of(2, 10), blocks_of(3, 6)
    prompts = [pre + a + [7] * 20, pre + b + [8] * 20, pre + a + [9] * 20, pre + b + [5] * 20, pre + a + [6] * 20]
    r = rows(prompts)
    meta = {}
    wk._map_prefixes(r, meta, "best")
    # tenant a (rows 0, 2, 4) and b (1, 3): each group states its tenant's whole system prompt
    assert [x["prefix_len"] for x in r] == [14 * 16, 10 * 16, 14 * 16, 10 * 16, 14 * 16]
    assert r[0]["prefix_group"] == r[2]["prefix_group"] == r[4]["prefix_group"] != r[1]["prefix_group"]
    # b's first request shares the preamble with a's, which the groups cannot say
    assert meta["prefix"]["blis_error_tokens"] == 4 * 16
    assert meta["prefix"]["blis_mapping_tokens"] == (2 * 14 + 10) * 16


def test_encoding_a_trimmed_round_is_a_new_session_with_the_head_as_its_prefix():
    head, turn1, turn2 = blocks_of(1, 5), blocks_of(2, 3), blocks_of(3, 4)
    p0 = head + turn1
    out0 = [4] * 16
    p1 = p0 + out0 + turn2  # continues
    p2 = head + blocks_of(9, 6)  # trimmed: only the head is shared
    r = rows([p0, p1, p2], outputs=[16, 16, 16], sids=["S", "S", "S"])
    for i, x in enumerate(r):
        x["call_id"] = f"S#{i}"
    meta = {}
    wk._map_prefixes(r, meta, "best")
    assert [(x["blis_sid"], x["blis_round"], x["blis_cont"]) for x in r] == [("S", 0, False), ("S", 1, True), ("S~1", 0, False)]
    assert r[2]["prefix_len"] == 5 * 16 and r[2]["prefix_group"] == r[0]["prefix_group"] != ""
    assert r[0]["prefix_len"] == 5 * 16  # the group's length is every member's


def test_best_encoding_writes_blis_csv_with_prefix_groups_and_split_sessions(tmp_path):
    head = blocks_of(1, 5)
    p0 = head + blocks_of(2, 3)
    p1 = p0 + [4] * 16 + blocks_of(3, 4)
    p2 = head + blocks_of(9, 6)
    r = rows([p0, p1, p2], outputs=[16, 16, 16], sids=["S", "S", "S"])
    for x in r:
        x.update(priority=0, tool_wait_s=1.0, frontend_s=0.01, delivery_s=0.008)
    meta = {"scheduling_policy": "fcfs", "prefix_caching": True, "spec_k": 0, "kv_offloading_gib": None,
            "max_num_batched_tokens": 2048, "max_num_seqs": 128, "max_model_len": 8192, "async_scheduling": False}
    wk._map_prefixes(r, meta, "best")
    json.dump({"requests": r}, open(tmp_path / "requests.json", "w"))
    json.dump(meta, open(tmp_path / "meta.json", "w"))
    wk.blis(Namespace(out=str(tmp_path)))
    got = list(csv.DictReader(open(tmp_path / "blis_replay.csv")))
    assert [(x["session_id"], x["round_index"], x["prefix_group"] != "", x["prefix_length"], x["input_tokens"],
             x["input_tokens_reset"]) for x in got] == [("S", "0", True, "80", "48", ""), ("S", "1", False, "0", "64", ""),
                                                        ("S~1", "0", True, "80", "96", "")]
    assert "accumulate" in open(tmp_path / "blis_replay.yaml").read()
    live = list(csv.DictReader(open(tmp_path / "blis.csv")))
    assert live[1]["think_time_us"] == "1018000" and live[2]["think_time_us"] == ""


def trace_lines(n, sessions=False):
    rng = random.Random(3)
    flat = [{"input_toks": 50, "output_toks": 20, "arrival_time_ns": i * 10**8, "input_tok_ids": CHAT[i % len(CHAT)][:50],
             "output_tok_ids": CHAT[i % len(CHAT)][50:70]} for i in range(n)]
    if not sessions:
        return flat
    return [{"session_id": f"s{i}", "arrival_time_ns": int(i * 5e9 + rng.random() * 1e9),
             "sub_requests": [{"input_toks": 50, "output_toks": 20, "tool_duration_ns": 10**9, "input_tok_ids": CHAT[i % len(CHAT)][:50]}]}
            for i in range(n)]


def test_heldout_workloads_are_seeded_and_stable():
    lines = {"bursty": gen_h("bursty", n=60), "longctx": gen_h("longctx", n=20), "peruser": gen_h("peruser", n=80)}
    assert {k: digest(v) for k, v in lines.items()} == HELDOUT_EXPECTED
    for k, v in lines.items():
        assert digest(gen_h(k, 0, len(v))) == digest(v) and digest(gen_h(k, 1, len(v))) != digest(v)


HELDOUT_EXPECTED = {"bursty": "88be458a17972409", "longctx": "cfcc396a7cb258a1", "peruser": "591826a04f2b6027"}


def gen_h(pattern, seed=0, n=None):
    return ow.generate(pattern, seed, n, chat=CHAT, swe=SWE, src=trace_lines(60))


def test_bursty_keeps_the_requests_and_draws_gamma_gaps_of_the_asked_rate_and_cv():
    src = trace_lines(60)
    out = ow.generate("bursty", 0, 60, 4.0, src=src)
    assert [{k: v for k, v in d.items() if k != "arrival_time_ns"} for d in out] == \
        [{k: v for k, v in d.items() if k != "arrival_time_ns"} for d in src]
    t = [d["arrival_time_ns"] for d in out]
    assert t == sorted(t)
    big = ow.generate("bursty", 0, 20000, 4.0, src=trace_lines(20000))
    t = [0] + [d["arrival_time_ns"] / 1e9 for d in big]
    gaps = [b - a for a, b in zip(t, t[1:])]
    mean = sum(gaps) / len(gaps)
    cv = (sum((g - mean) ** 2 for g in gaps) / len(gaps)) ** 0.5 / mean
    assert abs(1 / mean - 4.0) < 0.5 and abs(cv - ow.BURST_CV) < 0.4


def test_longctx_prompts_are_12k_to_24k_and_share_no_prefix():
    lines = ow.generate("longctx", 0, 30, chat=CHAT, swe=SWE)
    lens = [d["input_toks"] for d in lines]
    assert all(12000 <= n <= 23800 for n in lens) and len(set(lens)) > 20
    assert all(d["input_toks"] + d["output_toks"] <= 24576 for d in lines)
    assert max(lcp(a["input_tok_ids"], b["input_tok_ids"]) for i, a in enumerate(lines) for b in lines[i + 1:]) < 16


def test_swefast_scales_session_arrivals_and_nothing_else():
    src = trace_lines(30, sessions=True)
    out = ow.generate("swefast", 0, 30, 1.0, src=src)  # 5x the trace's 0.2 sessions per second
    for a, b in zip(src, out):
        assert b["arrival_time_ns"] == int(a["arrival_time_ns"] / 5)
        assert b["sub_requests"] == a["sub_requests"]
    assert len(ow.generate("swefast", 0, 10, 0.4, src=src)) == 10


def test_peruser_prompts_are_short_and_a_users_versions_agree_in_part():
    prompts = flat_prompts(ow.generate("peruser", 0, 300, chat=CHAT, swe=SWE))
    assert all(150 <= len(p) <= ow.PERUSER_MAX for p in prompts)
    assert min(lcp(prompts[0], p) for p in prompts) >= 96  # the shared preamble
    users = {}
    for p in prompts:
        users.setdefault(tuple(p[96:112]), []).append(p)
    assert 10 < len(users) <= 60
    # inside a user, pairs agree beyond the preamble: some share a whole prefix, some only the part the versions have in common
    agree = [lcp(a, b) - 96 for g in users.values() for i, a in enumerate(g) for b in g[i + 1:]]
    assert max(agree) > 150 and sum(0 < x < 150 for x in agree) > 0
