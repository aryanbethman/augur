"""MoE models priced from the shipped tables (Qwen3-30B-A3B on an RTX PRO 6000, TP1 and TP2; TP4 from the fork's
projected bf16-tp4proj bundle).

The first tests need no fork; the last ones compare with it (skipped without AUGUR_FORK): the MoE block at
every token count, random step shapes, and whole runs, which must write the same bytes as --cost llmservingsim."""

import gzip
import json
import os
import random
import subprocess
import sys

import pytest

from augur.cost import SeqWork, StepShape
from augur.cost.convert import cluster_params, random_shapes
from augur.cost.profiles import default_fork, shipped
from augur.cost.tables import OutsideRegime, TableCost, bundled_profile

FORK = default_fork()
needs_fork = pytest.mark.skipif(not os.path.isdir(os.path.join(FORK, "workloads")),
                                reason="LLMServingSim fork not found (AUGUR_FORK)")
QWEN = "Qwen/Qwen3-30B-A3B-Instruct-2507"
TP2_CONFIG = os.path.join(os.path.dirname(__file__), "data", "llmservingsim", "qwen3moe_tp2.json")
TP4_CONFIG = os.path.join(os.path.dirname(__file__), "data", "llmservingsim", "qwen3moe_tp4.json")
TP1, TP2 = (bundled_profile("RTXPRO6000", QWEN, tp) for tp in (1, 2))
TP4 = bundled_profile("RTXPRO6000", QWEN, 4, "bf16-tp4proj")


def run_augur(*args, fork=None):
    env = dict(os.environ, VLLM_LOGGING_LEVEL="WARNING", HF_HUB_OFFLINE="1")
    if fork is not None:
        env["AUGUR_FORK"] = fork
    return subprocess.run([sys.executable, "-m", "augur.cli", "--quiet", "--model", QWEN, *args],
                          capture_output=True, text=True, env=env)


def test_moe_profiles_are_shipped_and_small():
    assert {("RTXPRO6000", QWEN, "bf16", 1), ("RTXPRO6000", QWEN, "bf16", 2),
            ("RTXPRO6000", QWEN, "bf16-tp4proj", 4)} <= set(shipped())
    for path in (TP1, TP2, TP4):
        assert path.stat().st_size < 100_000
        with gzip.open(path, "rt") as f:
            p = json.load(f)
        assert p["identity"]["kind"] == "moe" and p["moe"]
    assert len(json.loads(gzip.decompress(TP1.read_bytes()))["moe"]["exact"]["moe_experts"]) == 2048


def test_tp1_moe_steps_and_limits():
    cost = TableCost(TP1)
    step = StepShape(seqs=(SeqWork(1, 100, True), SeqWork(1, 300, True), SeqWork(256, 0, False)))
    assert cost.step_ns(step) == cost.step_ns(step) > 10_000_000
    top = StepShape(seqs=(SeqWork(2048, 0, False),))
    assert cost.step_ns(top) > cost.step_ns(StepShape(seqs=(SeqWork(2047, 0, False),)))
    # the fork's component tables stop at 2048 tokens (it raises); so does the pricer
    for shape in (StepShape(seqs=(SeqWork(2049, 0, False),)), StepShape(seqs=(SeqWork(1, 5, True),), padded_tokens=4096)):
        with pytest.raises(OutsideRegime, match="2048"):
            cost.step_ns(shape)


def test_tp2_moe_has_no_block_for_an_empty_step_and_no_token_limit():
    cost = TableCost(TP2, **cluster_params(TP2_CONFIG))
    assert cost.step_ns(StepShape(seqs=(SeqWork(9000, 0, False),))) > cost.step_ns(StepShape(seqs=(SeqWork(2048, 0, False),)))
    assert cost.step_ns(StepShape()) < cost.step_ns(StepShape(padded_tokens=1))  # T = 0: the fork emits no expert row


def test_tp4_moe_saving_is_bounded_and_needs_a_block():
    cost = TableCost(TP4, **cluster_params(TP4_CONFIG))
    m = cost._coll["moe_layers"]
    for a in range(0, 120, 7):
        for b in range(0, a + 1, 5):
            assert m - m // 2 - b <= cost._moe_comm_saving(m, a, b) <= m
    with pytest.raises(OutsideRegime):
        cost._moe_comm_saving(m, 3, 4)
    # an empty step emits no MoE block, so nothing is saved (the fork prices it as the dense form)
    dense = TableCost(TP4, **cluster_params(TP4_CONFIG))
    dense._coll = dict(dense._coll, moe_layers=0)
    assert cost._comm_ns(0, 0) == dense._comm_ns(0, 0) and cost._comm_ns(1, 0) < dense._comm_ns(1, 0)


def test_moe_profile_at_other_tp_is_refused(tmp_path):
    p = json.loads(gzip.decompress(TP2.read_bytes()))
    q = dict(p, identity=dict(p["identity"], tp=8))
    f = tmp_path / "tp8.json.gz"
    f.write_bytes(gzip.compress(json.dumps(q).encode()))
    with pytest.raises(NotImplementedError, match="TP2 and TP4 only"):
        TableCost(f)
    q = dict(p)
    del q["moe"]
    f.write_bytes(gzip.compress(json.dumps(q).encode()))
    with pytest.raises(NotImplementedError, match="moe"):
        TableCost(f)


def test_no_scipy_or_fork_at_run_time(tmp_path):
    code = ("import sys\n"
            "from augur.cost import SeqWork, StepShape\n"
            "from augur.cost.tables import TableCost, bundled_profile\n"
            f"c = TableCost(bundled_profile('RTXPRO6000', {QWEN!r}))\n"
            "c.step_ns(StepShape(seqs=(SeqWork(1, 5, True), SeqWork(33, 0, False))))\n"
            "bad = [m for m in sys.modules if m.split('.')[0] in ('serving', 'profiler', 'pandas', 'scipy', 'yaml', 'numpy')]\n"
            "assert not bad, bad\nprint('ok')\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)))
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr[-2000:]


@pytest.mark.parametrize("tp,extra", [(1, []), (2, ["--tp", "2", "--lss-cluster-config", TP2_CONFIG]),
                                      (4, ["--tp", "4", "--lss-cluster-config", TP4_CONFIG,
                                           "--profile-variant", "bf16-tp4proj"])])
def test_short_run_from_shipped_tables_without_the_fork(tmp_path, workload, tp, extra):
    """--cost auto picks the shipped tables for DP=1 (the fork path does not exist here: AUGUR_FORK is empty)."""
    out = tmp_path / "run.csv"
    r = run_augur("--workload", workload, "--num-entries", "12", "--hbm-gb", "96", "--max-model-len", "8192",
                  "--max-num-seqs", "16", "--max-num-batched-tokens", "512", "--hardware", "RTXPRO6000",
                  "--output", str(out), *extra, fork=str(tmp_path / "no-fork"))
    assert r.returncode == 0, r.stderr[-2000:]
    summary = json.loads(r.stdout[r.stdout.index("{"):])
    assert summary["steps"] > 0 and "astra_runs" not in summary
    assert len(out.read_text().splitlines()) == 13


def test_dp_ep_stays_on_the_fork(tmp_path, workload):
    r = run_augur("--workload", workload, "--num-entries", "2", "--dp", "2", "--enable-expert-parallel",
                  "--hardware", "RTXPRO6000", fork=str(tmp_path / "no-fork"))
    assert r.returncode == 2 and "no step-time profile" in r.stderr


@needs_fork
def test_every_token_count_of_the_moe_block_against_the_fork():
    from augur.cost.convert import _moe_prices
    from augur.cost.llmservingsim import LLMServingSimCost
    for tp, path, cfg in ((1, TP1, None), (2, TP2, TP2_CONFIG)):
        ref = LLMServingSimCost(FORK, "RTXPRO6000", QWEN, tp=tp, cluster_config=cfg)
        mine = TableCost(path, **(cluster_params(cfg) if cfg else {}))
        names = {"moe_gate_routing", "moe_experts"} if tp == 1 else {"expert"}
        rng = random.Random(3)
        counts = list(range(1, 2049)) if tp == 1 else list(range(1, 2100)) + rng.sample(range(2100, 60000), 150)
        bad = [t for t in counts if _moe_prices(ref, t, names) != {n: mine._moe_ns(n, t) for n in names}]
        assert bad == []


@needs_fork
def test_live_against_fork_tp1():
    from augur.cost.llmservingsim import LLMServingSimCost
    ref = LLMServingSimCost(FORK, "RTXPRO6000", QWEN)
    cost = TableCost(TP1)
    checked = 0
    for s in random_shapes(random.Random(8675309), 2500):
        try:
            got = cost.step_ns(s)
        except OutsideRegime:
            with pytest.raises(ValueError):  # the fork refuses the same step (MoeCoverageError)
                ref.step_ns(s)
            continue
        assert ref.step_ns(s) == got
        checked += 1
    assert checked > 1000


@needs_fork
def test_live_against_fork_tp2():
    """120 fresh random shapes at TP2 (about 110 ASTRA-Sim runs)."""
    from augur.cost.llmservingsim import LLMServingSimCost
    ref = LLMServingSimCost(FORK, "RTXPRO6000", QWEN, tp=2, cluster_config=TP2_CONFIG)
    cost = TableCost(TP2, **cluster_params(TP2_CONFIG))
    shapes = list(random_shapes(random.Random(20261121), 120))
    assert [(s, ref.step_ns(s), cost.step_ns(s)) for s in shapes if ref.step_ns(s) != cost.step_ns(s)] == []


TP4_AGAINST_FORK = """
import random, sys
from augur.cost.convert import _moe_prices, cluster_params, random_shapes
from augur.cost.llmservingsim import LLMServingSimCost
from augur.cost.tables import TableCost
fork, path, cfg, qwen = sys.argv[1:5]
ref = LLMServingSimCost(fork, "RTXPRO6000", qwen, tp=4, cluster_config=cfg, profile_variant="bf16-tp4proj")
cost = TableCost(path, **cluster_params(cfg))
rng = random.Random(3)
counts = list(range(1, 2100)) + rng.sample(range(2100, 60000), 150)
bad = [t for t in counts if _moe_prices(ref, t, {"expert"}) != {"expert": cost._moe_ns("expert", t)}]
assert bad == [], bad[:5]
shapes = list(random_shapes(random.Random(20261123), 120))
bad = [(s, ref.step_ns(s), cost.step_ns(s)) for s in shapes if ref.step_ns(s) != cost.step_ns(s)]
assert bad == [], bad[:3]
print("ok")
"""


@needs_fork
def test_tp4_against_the_fork():
    """The MoE block at every token count, and 120 fresh random shapes (about 110 ASTRA-Sim runs), on the projected
    bundle. In a process of its own: the fork pins a --profile-variant per process (set_profile_variant_override),
    and every later fork lookup in this one would read that variant's folder."""
    r = subprocess.run([sys.executable, "-c", TP4_AGAINST_FORK, FORK, str(TP4), TP4_CONFIG, QWEN],
                       capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)))
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stderr[-3000:]


@needs_fork
@pytest.mark.parametrize("tp,extra", [(1, ["--hbm-gb", "96"]),
                                      (2, ["--kv-cache-gb", "30.95", "--tp", "2", "--lss-cluster-config", TP2_CONFIG]),
                                      (4, ["--tp", "4", "--lss-cluster-config", TP4_CONFIG,
                                           "--profile-variant", "bf16-tp4proj"])])
def test_whole_swe_bench_run_is_identical(tmp_path, tp, extra):
    """SWE-bench sessions (765 calls), sync: the CSV is the same bytes with the shipped tables as with the fork."""
    src = os.path.join(FORK, "workloads", "swe-bench-qwen3-30b-a3b-50-sps0.2.jsonl")
    if not os.path.exists(src):
        pytest.skip("fork has no SWE-bench workload")
    full = tmp_path / "swe.jsonl"
    with open(src) as f, open(full, "w") as g:
        for line in f:
            d = json.loads(line)
            for s in d["sub_requests"]:
                s["input_toks"] = len(s["input_tok_ids"])
                s["output_toks"] = len(s["output_tok_ids"]) if s.get("output_tok_ids") else s["output_toks"]
                s.setdefault("output_tok_ids", [])
            g.write(json.dumps(d) + "\n")
    def run(cost):
        out = tmp_path / f"{cost}.csv"
        r = run_augur("--cost", cost, "--hardware", "RTXPRO6000", "--max-model-len", "131072", "--workload", str(full),
                      "--max-num-seqs", "128", "--max-num-batched-tokens", "2048", "--step-cpu-us", "0",
                      "--frontend-latency-ms", "0", "--delivery-latency-ms", "0", "--lss-loop-compat",
                      "--no-async-scheduling", *extra, "--output", str(out))
        assert r.returncode == 0, r.stderr[-2000:]
        return out.read_bytes()
    mine = run("tables")
    # ASTRA-Sim runs in the fork's own serving/ folder, shared by every process using the fork: two concurrent
    # TP>1 runs can corrupt each other's step times, so a differing fork run is repeated before it counts
    for _ in range(3 if tp > 1 else 1):
        if run("llmservingsim") == mine:
            break
    else:
        pytest.fail("the CSV differs from the fork's in three runs")
    assert len(mine.decode().splitlines()) == 766
