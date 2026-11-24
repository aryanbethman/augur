"""Decode attention over KV blocks that requests share (prefix caching): SeqWork.shared_kv, TableCost's shared_kv
model, the executor that fills it, augur.profile_shared, and augur.verify.prefix_share for real runs' logs."""

import gzip
import json
import os
import random
from types import SimpleNamespace

import pytest

from augur.cost import SeqWork, StepShape
from augur.cost.convert import random_shapes
from augur.cost.tables import TableCost, bundled_profile
from augur.profile_shared import ND_AXIS, apply, fit, relative_times, unique_fraction
from augur.verify import prefix_share
from augur.verify.stepcheck import real_steps

PROFILE = bundled_profile("RTX4060Ti", "Qwen/Qwen3-1.7B")


def load_profile(path=PROFILE):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def save_profile(p, path):
    path.write_bytes(gzip.compress(json.dumps(p, sort_keys=True, separators=(",", ":")).encode(), mtime=0))
    return path


@pytest.fixture(scope="module")
def cost():
    return TableCost(PROFILE)


@pytest.fixture(scope="module")
def plain(tmp_path_factory):
    """The same profile without its shared_kv entry (what a profile that was never measured for sharing prices)."""
    p = load_profile()
    del p["shared_kv"]
    return TableCost(save_profile(p, tmp_path_factory.mktemp("plain") / "plain.json.gz"))


def decodes(n, kv, shared=0):
    return StepShape(seqs=tuple(SeqWork(1, kv, True, shared_kv=shared) for _ in range(n)))


def test_the_shipped_profile_has_a_fitted_shared_kv_model(cost):
    spec = load_profile()["shared_kv"]
    assert cost.prices_sharing and spec["nd"] == ND_AXIS and len(spec["floor"]) == len(ND_AXIS)
    assert all(0.3 < c < 0.6 for c in spec["floor"])
    assert spec["floor"][1:] == sorted(spec["floor"][1:], reverse=True)  # bigger batches gain more
    prov = load_profile()["provenance"]["shared_kv"]
    assert prov["fit"]["shots"] > 300 and prov["fit"]["rmse"] < 0.03 and prov["measured"][0]["gpu"]


def test_nothing_shared_prices_as_before(cost, plain):
    rng = random.Random(7)
    for shape in random_shapes(rng, 400):
        assert cost.step_ns(shape) == plain.step_ns(shape)
    assert cost.step_ns(decodes(48, 3000)) == plain.step_ns(decodes(48, 3000))


def test_a_profile_without_the_model_ignores_shared_kv(plain):
    assert not plain.prices_sharing
    assert plain.step_ns(decodes(48, 3000, shared=2000)) == plain.step_ns(decodes(48, 3000))


def test_shared_blocks_make_the_decode_attention_cheaper_by_the_model(cost, plain):
    spec = load_profile()["shared_kv"]
    n, kv, shared = 48, 3000, 1600
    private = plain._attention_ns("attention", 0, 0.0, n, kv, kv, 1)
    u = 1 - shared / kv
    floor = spec["floor"][ND_AXIS.index(32)] + (spec["floor"][ND_AXIS.index(64)] - spec["floor"][ND_AXIS.index(32)]) * 0.5
    want = round(private * min(1.0, (u ** spec["p"] + floor ** spec["p"]) ** (1 / spec["p"])))
    assert cost._attention_ns("attention", 0, 0.0, n, kv, kv, 1, n * shared) == want
    assert want < private
    # more sharing, less time; never more than the private price
    prices = [cost._attention_ns("attention", 0, 0.0, n, kv, kv, 1, n * s) for s in range(0, kv, 250)]
    assert prices == sorted(prices, reverse=True) and prices[0] == private
    # a whole step: only attention changes, in each of the 28 layers
    gap = plain.step_ns(decodes(n, kv)) - cost.step_ns(decodes(n, kv, shared))
    assert gap == 28 * (private - cost._attention_ns("attention", 0, 0.0, n, kv, kv, 1, n * shared))


def test_the_measured_shot_is_not_far_from_the_model(cost):
    # 64 decodes of 4096 tokens sharing 3680 of them in one group: the benchmark measured 0.43-0.44 of the private time
    private = cost._attention_ns("attention", 0, 0.0, 64, 4096, 4096, 1)
    shared = cost._attention_ns("attention", 0, 0.0, 64, 4096, 4096, 1, 64 * 3680 - 3680)
    assert 0.40 < shared / private < 0.50


def test_one_decode_and_unknown_sharing_are_not_discounted(cost):
    assert cost.step_ns(decodes(1, 3000, 2000)) == cost.step_ns(decodes(1, 3000))
    # more shared tokens than a request has is clamped to what it has
    assert cost.step_ns(decodes(8, 500, 10 ** 6)) == cost.step_ns(decodes(8, 500, 500))


def test_a_prefill_chunk_keeps_its_own_price_when_decodes_share(cost):
    table = cost._attention_table("attention", 1)
    key = 1000 + 512 / 2
    chunk = table.lookup(512, key, 0, 0)
    private = cost._attention_ns("attention", 512, key, 32, 3000, 3000, 1)
    shared = cost._attention_ns("attention", 512, key, 32, 3000, 3000, 1, 32 * 1500)
    assert chunk < shared < private  # the decode part shrinks, the chunk's own attention stays


def test_the_engine_fills_shared_kv_from_block_ids(workload, monkeypatch):
    import augur.cli as cli
    seen = []
    step = TableCost.step_ns
    monkeypatch.setattr(TableCost, "step_ns", lambda self, shape: (seen.append(shape), step(self, shape))[1])
    argv = ["--model", "Qwen/Qwen3-1.7B", "--hardware", "RTX4060Ti", "--num-gpu-blocks", "600", "--max-num-seqs", "16",
            "--max-model-len", "8192", "--max-num-batched-tokens", "512", "--no-recorded-outputs",
            "--no-async-scheduling", "--quiet", "--workload", workload]
    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "WARNING")
    assert cli.main(argv + ["--output", os.devnull]) == 0
    shared = [s for shape in seen for s in shape.seqs if s.shared_kv]
    assert shared and all(s.shared_kv % 16 == 0 and s.shared_kv <= s.num_computed_before for s in shared)
    # a third of the synthetic requests start with the same 256 tokens: 16 blocks
    assert max(s.shared_kv for s in shared) == 256


def test_the_engine_does_not_look_at_blocks_without_a_model_for_them(workload, monkeypatch, tmp_path):
    import augur.cli as cli
    p = load_profile()
    del p["shared_kv"]
    plain = save_profile(p, tmp_path / "plain.json.gz")
    seen = []
    step = TableCost.step_ns
    monkeypatch.setattr(TableCost, "step_ns", lambda self, shape: (seen.append(shape), step(self, shape))[1])
    argv = ["--model", "Qwen/Qwen3-1.7B", "--hardware", "RTX4060Ti", "--profile", str(plain), "--num-gpu-blocks", "600",
            "--max-num-seqs", "16", "--max-model-len", "8192", "--max-num-batched-tokens", "512",
            "--no-recorded-outputs", "--no-async-scheduling", "--quiet", "--workload", workload, "--output", os.devnull]
    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "WARNING")
    assert cli.main(argv) == 0
    assert seen and not any(s.shared_kv for shape in seen for s in shape.seqs)


def test_no_shared_kv_prices_as_a_profile_without_the_model(workload, monkeypatch, tmp_path):
    # The baselines' control run (scripts/baselines/run.sh) must take the price pricer.py gives the other simulators,
    # which know no block ids.
    import augur.cli as cli
    p = load_profile()
    del p["shared_kv"]
    plain = save_profile(p, tmp_path / "plain.json.gz")
    argv = ["--model", "Qwen/Qwen3-1.7B", "--hardware", "RTX4060Ti", "--num-gpu-blocks", "600", "--max-num-seqs", "16",
            "--max-model-len", "8192", "--max-num-batched-tokens", "512", "--no-recorded-outputs",
            "--no-async-scheduling", "--quiet", "--workload", workload]
    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "WARNING")
    assert cli.main(argv + ["--profile", str(plain), "--output", str(tmp_path / "plain.csv")]) == 0
    assert cli.main(argv + ["--no-shared-kv", "--output", str(tmp_path / "off.csv")]) == 0
    assert cli.main(argv + ["--output", str(tmp_path / "on.csv")]) == 0
    off = (tmp_path / "off.csv").read_bytes()
    assert off == (tmp_path / "plain.csv").read_bytes() and off != (tmp_path / "on.csv").read_bytes()


# --- augur.profile_shared -------------------------------------------------------

def synthetic_rows(p=3.0, floors=(0.54, 0.54, 0.5, 0.5, 0.45, 0.42, 0.4), noise=0.0, seed=0):
    rng = random.Random(seed)
    rows = []
    for nd in (2, 4, 8, 16, 32, 64, 128):
        floor = dict(zip(ND_AXIS, floors))[nd]
        for kv in (1024, 3000):
            rows.append({"nd": nd, "kv": kv, "G": 1, "P": 0, "ns_per_layer": 1000 * nd * kv // 100})
            for g in (1, 2):
                if g > nd:
                    continue
                for frac in (0.1, 0.3, 0.5, 0.7, 0.9):
                    shared = int(kv * frac) // 16 * 16
                    u = unique_fraction(nd, kv, g, shared)
                    rel = min(1.0, (u ** p + floor ** p) ** (1 / p)) * (1 + rng.uniform(-noise, noise))
                    rows.append({"nd": nd, "kv": kv, "G": g, "P": shared,
                                 "ns_per_layer": round(1000 * nd * kv // 100 * rel)})
    return rows


def test_unique_fraction():
    assert unique_fraction(8, 1000, 1, 0) == 1.0 and unique_fraction(8, 1000, 8, 500) == 1.0
    assert unique_fraction(4, 1000, 1, 800) == (800 + 4 * 200) / 4000
    assert unique_fraction(4, 1000, 2, 800) == (2 * 800 + 4 * 200) / 4000


def test_fit_recovers_the_model_it_is_given():
    rows = synthetic_rows()
    assert len(relative_times(rows)) > 100
    got = fit(rows)
    assert got["p"] == 3.0 and got["rmse"] < 0.01
    assert all(abs(a - b) < 0.02 for a, b in zip(got["floor"], (0.54, 0.54, 0.5, 0.5, 0.45, 0.42, 0.4)))
    noisy = fit(synthetic_rows(noise=0.04, seed=3))
    assert 0.005 < noisy["rmse"] < 0.05 and abs(noisy["p"] - 3.0) <= 1.0


def test_apply_adds_the_entry_and_leaves_the_tables_alone(tmp_path):
    before = load_profile()
    del before["shared_kv"]
    del before["provenance"]["shared_kv"]
    path = save_profile(before, tmp_path / "p.json.gz")
    shots = tmp_path / "shots.csv"
    shots.write_text("nd,kv,G,P,ns_per_layer\n8,1024,1,0,100\n")
    apply(path, fit(synthetic_rows()), [shots], [{"gpu": "test"}])
    after = load_profile(path)
    for key in ("attention", "dense", "per_sequence", "skew", "layers", "identity", "model_config"):
        assert after[key] == before[key]
    assert after["shared_kv"]["nd"] == ND_AXIS
    assert after["provenance"]["shared_kv"]["measured"] == [{"gpu": "test"}]
    assert {k: v for k, v in after["provenance"].items() if k != "shared_kv"} == before["provenance"]
    assert TableCost(path).prices_sharing


# --- augur.verify.prefix_share ---------------------------------------------------

def write_workload(tmp_path, prompts):
    path = tmp_path / "w.jsonl"
    with open(path, "w") as f:
        for i, ids in enumerate(prompts):
            f.write(json.dumps({"input_toks": len(ids), "output_toks": 4, "arrival_time_ns": i, "input_tok_ids": ids}) + "\n")
    return str(path)


def test_prompt_chains_identify_blocks_by_everything_before_them(tmp_path):
    base = list(range(100, 164))
    a = base + [1] * 16
    b = base[:32] + [2] * 16 + base[48:]  # the same first two blocks, a different third, then the same tokens
    c = [9] + base[1:]                    # differs in the first token: no block in common
    chains = prefix_share.prompt_chains(write_workload(tmp_path, [a, b, c]))
    ra, rb, rc = chains["req_0"], chains["req_1"], chains["req_2"]
    assert len(ra) == 5 and ra[:2] == rb[:2] and ra[2] != rb[2] and ra[3] != rb[3]  # the fourth block differs too
    assert not set(ra) & set(rc)


def test_shared_tokens_counts_blocks_an_earlier_decode_holds():
    # (request id, its block ids, tokens of computed KV): only the full blocks of the computed KV count
    decodes = [("a", [1, 2, 3, 4, 5, 6, 7], 100), ("b", [1, 2, 3, 9, 10, 11, 12], 70), ("c", [1, 2, 20], 40),
               ("d", [30, 31], 0), ("e", [], 50)]
    # a holds 6 full blocks (100 // 16); b 4: its first 3 are a's; c 2: both a's; d none yet; e has no blocks
    assert prefix_share.shared_tokens(decodes, 16) == {"a": 0, "b": 48, "c": 32}


def test_real_steps_carry_the_shared_tokens():
    x, y = "cmpl-augur.x.-0", "cmpl-augur.y.-0"
    ev = [{"ev": "add", "t": 0, "rid": x, "prompt": 64}, {"ev": "add", "t": 0, "rid": y, "prompt": 64},
          {"ev": "sched", "t": 1, "k": 1, "tok": {x: 64}, "new": {x: 0}, "pre": []},
          {"ev": "upd", "t": 2, "k": 1, "out": {x: 1}, "fin": [], "left": 2},
          # y arrives and hits x's first three blocks
          {"ev": "sched", "t": 3, "k": 2, "tok": {x: 1, y: 16}, "new": {y: 48}, "pre": []},
          {"ev": "upd", "t": 4, "k": 2, "out": {x: 1, y: 1}, "fin": [], "left": 2},
          {"ev": "sched", "t": 5, "k": 3, "tok": {x: 1, y: 1}, "new": {}, "pre": []},
          {"ev": "upd", "t": 6, "k": 3, "out": {x: 1, y: 1}, "fin": [], "left": 2}]
    chains = {"x": [1, 2, 3, 4], "y": [1, 2, 3, 9]}
    assert all(q.shared_kv == 0 for s, _ in real_steps(ev) for q in s.seqs)
    steps = [s for s, _ in real_steps(ev, chains)]
    assert [q.shared_kv for q in steps[1].seqs] == [0, 0]  # one decode and a prefill chunk: nothing to share
    assert [(q.num_computed_before, q.shared_kv) for q in steps[2].seqs] == [(65, 0), (64, 48)]
