"""TableCost against LLMServingSimCost, to the nanosecond.

The regression files in data/table_cost/ hold (shape, ns) pairs priced by the LLMServingSim
fork through LLMServingSimCost: 3,000 step shapes sampled from real augur runs of the test
workloads (sync and async), 3,000 random shapes per profile, and a few hundred priced with a
different CPU-memory bandwidth/latency (scripts/build_table_regression.py).  They run without
the fork.  With the fork present, fresh random shapes are compared live, and a whole run with
each cost model must write the same bytes.

TP>1 files (one per shipped tpN profile) hold 1,000 workload and 1,000 random shapes priced by the
fork with ASTRA-Sim, and the link, NPU and CPU-memory parameters used.
"""

import gzip
import hashlib
import json
import os
import random
import subprocess
import sys

import pytest

from augur.cost import SeqWork, StepShape
from augur.cost.convert import random_shapes, shape_from_row
from augur.cost.profiles import default_fork
from augur.cost.tables import OutsideRegime, TableCost, bundled_profile

DATA = os.path.join(os.path.dirname(__file__), "data", "table_cost")
FILES = sorted(f for f in os.listdir(DATA) if f.endswith(".json.gz"))
TP1_FILES = [f for f in FILES if "-tp" not in f]
TP_FILES = [f for f in FILES if "-tp" in f]
TP2_CONFIG = os.path.join(os.path.dirname(__file__), "data", "llmservingsim", "llama_tp2.json")
FORK = default_fork()
needs_fork = pytest.mark.skipif(not os.path.isdir(os.path.join(FORK, "workloads")),
                                reason="LLMServingSim fork not found (AUGUR_FORK)")


def load(name):
    with gzip.open(os.path.join(DATA, name), "rt") as f:
        return json.load(f)


def mismatches(cost, rows):
    return [(r[:5], r[5], got) for r in rows if (got := cost.step_ns(shape_from_row(r))) != r[5]]


@pytest.mark.parametrize("name", FILES)
def test_regression_exact(name):
    d = load(name)
    tp = d.get("tp", 1)
    path = bundled_profile(d["hardware"], d["model"], tp)
    assert hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest() == d["profile_sha256"], \
        "profile changed since the regression data was priced: rebuild it (scripts/build_table_regression.py)"
    if tp > 1:
        prm = d["params"]
        assert len(d["workload"]) == 1000 and len(d["random"]) == 1000
        cost = TableCost(path, **{k: prm[k] for k in ("link_bw", "link_latency_ns", "npu_mem_bw", "cpu_mem_bw",
                                                      "cpu_mem_latency_ns")})
        assert cost.tp == tp
        assert mismatches(cost, d["workload"]) == []
        assert mismatches(cost, d["random"]) == []
        return
    assert len(d["workload"]) > 1000 and len(d["random"]) >= 3000
    cost = TableCost(path)
    assert mismatches(cost, d["workload"]) == []
    assert mismatches(cost, d["random"]) == []
    cv = d["cpu_variant"]
    other = TableCost(path, cpu_mem_bw=cv["cpu_mem_bw"], cpu_mem_latency_ns=cv["cpu_mem_latency_ns"])
    assert mismatches(other, cv["rows"]) == []


@needs_fork
@pytest.mark.parametrize("name", TP1_FILES)
def test_live_against_fork(name):
    from augur.cost.llmservingsim import LLMServingSimCost
    d = load(name)
    ref = LLMServingSimCost(FORK, d["hardware"], d["model"])
    cost = TableCost(bundled_profile(d["hardware"], d["model"]))
    shapes = list(random_shapes(random.Random(987654321), 3000))

    def price(c, s):  # a MoE step beyond the fork's component tables (2048 tokens): the fork raises, the table refuses
        try:
            return c.step_ns(s)
        except (ValueError, OutsideRegime):
            return "refused"
    bad = [(s, price(ref, s), price(cost, s)) for s in shapes]
    bad = [b for b in bad if b[1] != b[2]]
    assert bad == []


@needs_fork
def test_live_against_fork_tp2():
    """150 fresh random shapes at TP2: TableCost against the fork with ASTRA-Sim (about 140 binary runs)."""
    from augur.cost.llmservingsim import LLMServingSimCost
    ref = LLMServingSimCost(FORK, "RTXPRO6000", "meta-llama/Llama-3.1-8B", tp=2, cluster_config=TP2_CONFIG)
    cost = TableCost(bundled_profile("RTXPRO6000", "meta-llama/Llama-3.1-8B", 2), link_bw=16, link_latency_ns=20000,
                     npu_mem_bw=1597, cpu_mem_bw=256, cpu_mem_latency_ns=0)
    shapes = list(random_shapes(random.Random(424242), 150))
    bad = [(s, ref.step_ns(s), cost.step_ns(s)) for s in shapes]
    assert [b for b in bad if b[1] != b[2]] == []


@needs_fork
def test_whole_run_is_identical_tp2(tmp_path):
    """augur --tp 2 end to end: same bytes out with the shipped tp2 profile as with the fork and ASTRA-Sim."""
    out = {}
    for cost in ("tables", "llmservingsim"):
        out[cost] = tmp_path / f"{cost}.csv"
        subprocess.run([sys.executable, "-m", "augur.cli", "--cost", cost, "--hardware", "RTXPRO6000",
                        "--model", "meta-llama/Llama-3.1-8B", "--tp", "2", "--hbm-gb", "96",
                        "--lss-cluster-config", TP2_CONFIG, "--max-num-seqs", "128", "--max-num-batched-tokens", "2048",
                        "--workload", os.path.join(FORK, "workloads", "sharegpt-llama-3.1-8b-300-sps10.jsonl"),
                        "--num-entries", "80", "--quiet", "--output", str(out[cost])],
                       check=True, env=dict(os.environ, VLLM_LOGGING_LEVEL="WARNING"),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert out["tables"].read_bytes() == out["llmservingsim"].read_bytes()


@needs_fork
def test_whole_run_is_identical(tmp_path):
    """augur end to end: same bytes out with the shipped profile as with the fork."""
    out = {}
    for cost in ("tables", "llmservingsim"):
        out[cost] = tmp_path / f"{cost}.csv"
        subprocess.run([sys.executable, "-m", "augur.cli", "--cost", cost, "--hardware", "RTX4060Ti",
                        "--model", "Qwen/Qwen3-1.7B", "--num-gpu-blocks", "1200", "--max-model-len", "8192",
                        "--max-num-seqs", "32", "--max-num-batched-tokens", "1024", "--no-recorded-outputs",
                        "--workload", os.path.join(FORK, "workloads", "sharegpt-qwen3-32b-300-sps10.jsonl"),
                        "--num-entries", "80", "--quiet", "--output", str(out[cost])],
                       check=True, env=dict(os.environ, VLLM_LOGGING_LEVEL="WARNING"),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert out["tables"].read_bytes() == out["llmservingsim"].read_bytes()


def test_shapes_cover_the_required_classes():
    shapes = list(random_shapes(random.Random(5), 20000))
    seqs = [len(s.seqs) for s in shapes]
    assert max(seqs) > 256 and any(n == 1 for n in seqs)
    assert any(s.dummy for s in shapes) and any(not s.seqs and not s.dummy for s in shapes)
    assert any(s.num_decode and any(q.num_new > 1 for q in s.seqs if q.is_decode) for s in shapes)
    assert any(any(q.num_new == 1 and not q.is_decode for q in s.seqs) for s in shapes)
    assert max(s.forward_tokens for s in shapes) >= 8192
    assert any(s.num_decode > 1 and len({q.num_computed_before for q in s.seqs if q.is_decode}) == 1 for s in shapes)
    assert max(q.num_computed_before for s in shapes for q in s.seqs) > 30000


def test_idle_and_empty_steps():
    cost = TableCost(bundled_profile("RTXPRO6000", "meta-llama/Llama-3.1-8B"))
    # no lm_head/sampler, no attention: the 4060Ti-independent value the fork gave in development
    assert cost.step_ns(StepShape(padded_tokens=64, dummy=True)) == 11016343
    assert cost.step_ns(StepShape(padded_tokens=64)) == 11016343
    shape = StepShape(seqs=(SeqWork(1, 100, True), SeqWork(1, 300, True), SeqWork(256, 0, False)))
    assert cost.step_ns(shape) == 15823218
    assert cost.step_ns(StepShape(seqs=shape.seqs, padded_tokens=512)) == 24601278


TP2 = bundled_profile("RTXPRO6000", "meta-llama/Llama-3.1-8B", 2)


def test_guard_refuses_long_cpu_memory_access():
    """A CPU-memory access longer than half a ring step is outside the verified regime: refused, with the values."""
    shape = StepShape(seqs=(SeqWork(1, 100, True),))
    ok = TableCost(TP2, cpu_mem_latency_ns=10_000)  # A = 10000 <= half of 20238
    assert ok.step_ns(shape) > 0
    # refused when the pricer is built, before a run starts
    with pytest.raises(OutsideRegime, match=r"A=11000 .*all-reduce 20238 ns.*--cost llmservingsim"):
        TableCost(TP2, cpu_mem_latency_ns=11_000)
    # the last collective (all-gather, 1 sequence) bounds the sampled-ids access too
    with pytest.raises(OutsideRegime):
        TableCost(TP2, link_bw=1000, link_latency_ns=0, cpu_mem_latency_ns=40)


def test_h100_tp_profile_defaults_to_h100_sxm_memory_bandwidth():
    """The fork has no H100 hardware default, so the H100 profiles carry none; augur supplies 3,350 GB/s
    (H100 SXM, the value the fork's H100 cluster configs use)."""
    h100 = bundled_profile("H100", "meta-llama/Llama-3.1-70B", 4)
    shape = StepShape(seqs=(SeqWork(1, 100, True), SeqWork(256, 0, False)))
    default = TableCost(h100)
    assert default.npu_mem_bw == 3350
    assert default.step_ns(shape) == TableCost(h100, npu_mem_bw=3350).step_ns(shape)
    assert TableCost(h100, npu_mem_bw=2000).npu_mem_bw == 2000


def test_profile_hardware_entry_wins_over_augurs_default(tmp_path):
    p = json.loads(gzip.decompress(TP2.read_bytes()))
    q = dict(p, identity=dict(p["identity"], hardware="H100"))  # RTXPRO6000 data, relabelled
    f = tmp_path / "relabelled.json.gz"
    f.write_bytes(gzip.compress(json.dumps(q).encode()))
    assert TableCost(f).npu_mem_bw == p["hardware"]["npu_mem_bw_gbps"]


def test_tp_profile_needs_its_parameters(tmp_path):
    p = json.loads(gzip.decompress(bundled_profile("H100", "meta-llama/Llama-3.1-70B", 4).read_bytes()))
    q = dict(p, identity=dict(p["identity"], hardware="XPU"))  # no profile entry, no augur default
    f = tmp_path / "unknown_hw.json.gz"
    f.write_bytes(gzip.compress(json.dumps(q).encode()))
    with pytest.raises(ValueError, match="npu_mem_bw"):
        TableCost(f)
    TableCost(f, npu_mem_bw=3350)
    p = json.loads(gzip.decompress(TP2.read_bytes()))
    del p["collectives"]
    f = tmp_path / "nocoll.json.gz"
    f.write_bytes(gzip.compress(json.dumps(p).encode()))
    with pytest.raises(NotImplementedError):
        TableCost(f)


def test_tp1_remote_term_unchanged():
    """TP=1 still adds latency + int(4 * tokens / bw) twice (float bandwidth, no TP communication)."""
    cost = TableCost(bundled_profile("RTXPRO6000", "meta-llama/Llama-3.1-8B"), cpu_mem_bw=3.7, cpu_mem_latency_ns=123)
    base = TableCost(bundled_profile("RTXPRO6000", "meta-llama/Llama-3.1-8B"), cpu_mem_bw=1e30)
    shape = StepShape(seqs=(SeqWork(1, 100, True), SeqWork(256, 0, False)))
    assert cost.step_ns(shape) - base.step_ns(shape) == 2 * 123 + int(4 * 257 / 3.7) + int(4 * 2 / 3.7)


def test_out_of_scope_is_refused(tmp_path):
    src = bundled_profile("RTXPRO6000", "meta-llama/Llama-3.1-8B")
    p = json.loads(gzip.decompress(src.read_bytes()))
    for edit, exc in (({"tp": 2}, NotImplementedError), ({"kind": "moe"}, NotImplementedError)):  # tp2: no collectives
        q = dict(p, identity=dict(p["identity"], **edit))
        f = tmp_path / "x.json.gz"
        f.write_bytes(gzip.compress(json.dumps(q).encode()))
        with pytest.raises(exc):
            TableCost(f)
    q = dict(p, format_version=99)
    f = tmp_path / "y.json.gz"
    f.write_bytes(gzip.compress(json.dumps(q).encode()))
    with pytest.raises(ValueError):
        TableCost(f)
    with pytest.raises(NotImplementedError):
        TableCost(src).round_ns([StepShape()])
    with pytest.raises(FileNotFoundError):
        bundled_profile("RTXPRO6000", "nope/none")


def test_no_fork_modules_and_cwd_unchanged(tmp_path):
    code = (
        "import os, sys\n"
        "from augur.cost import SeqWork, StepShape\n"
        "from augur.cost.tables import TableCost, bundled_profile\n"
        "c = TableCost(bundled_profile('RTX4060Ti', 'Qwen/Qwen3-1.7B'))\n"
        "cwd, path = os.getcwd(), list(sys.path)\n"
        "c.step_ns(StepShape(seqs=(SeqWork(1, 5, True), SeqWork(1, 90, True), SeqWork(33, 0, False))))\n"
        "assert os.getcwd() == cwd and sys.path == path\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('serving', 'profiler', 'pandas', 'scipy', 'yaml', 'numpy')]\n"
        "assert not bad, bad\n"
        "print('ok')\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)))
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr[-2000:]


def test_steps_beyond_the_profile_are_counted_and_warned(caplog):
    """The H100 Llama-3.1-70B profile stops at 2,048 tokens of KV; a longer decode extrapolates, and says so."""
    cost = TableCost(bundled_profile("H100", "meta-llama/Llama-3.1-70B", tp=4), cpu_mem_bw=512,
                     link_bw=900, link_latency_ns=0, npu_mem_bw=3350)
    inside = StepShape(seqs=(SeqWork(1, 1500, True),))
    beyond = StepShape(seqs=(SeqWork(1, 4552, True),))
    with caplog.at_level("WARNING", logger="augur.cost.tables"):
        a = cost.step_ns(inside)
        assert (cost.steps, cost.steps_beyond_profile) == (1, 0) and not caplog.records
        b = cost.step_ns(beyond)
        cost.step_ns(beyond)  # a repeated shape (cache hit) still counts
        cost.step_ns(inside)
    assert (cost.steps, cost.steps_beyond_profile) == (4, 2)
    assert [r.getMessage().split(" is above")[0] for r in caplog.records] == [
        "step priced beyond the profile: decode KV length 4552"]
    assert b > a
    big_chunk = StepShape(seqs=(SeqWork(8192, 0, False),))  # more tokens than the dense tables hold
    cost.step_ns(big_chunk)
    assert cost.steps_beyond_profile == 3


@needs_fork
def test_astra_runs_in_a_private_directory(monkeypatch):
    """ASTRA-Sim writes tmp__mem/remote_mem.json into its working directory, so concurrent runs sharing the
    fork's serving/ could read each other's CPU-memory config (a few ns off, or a crash). Each query runs in the
    oracle's own directory, and does not touch the fork's."""
    import subprocess
    from augur.cost.llmservingsim import LLMServingSimCost
    cost = LLMServingSimCost(FORK, "RTXPRO6000", "meta-llama/Llama-3.1-8B", tp=2, cluster_config=TP2_CONFIG)
    seen = []
    real = subprocess.Popen

    def spy(*a, **kw):
        seen.append(kw.get("cwd"))
        return real(*a, **kw)

    monkeypatch.setattr(subprocess, "Popen", spy)
    cost.step_ns(StepShape(seqs=(SeqWork(1, 100, True), SeqWork(64, 0, False))))
    assert seen and all(c == cost.oracle.workdir for c in seen)
    assert not os.path.exists(os.path.join(FORK, "serving", "tmp__mem", "remote_mem.json"))
