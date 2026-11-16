"""TableCost against LLMServingSimCost, to the nanosecond.

The regression files in data/table_cost/ hold (shape, ns) pairs priced by the LLMServingSim
fork through LLMServingSimCost: 3,000 step shapes sampled from real augur runs of the test
workloads (sync and async), 3,000 random shapes per profile, and a few hundred priced with a
different CPU-memory bandwidth/latency (scripts/build_table_regression.py).  They run without
the fork.  With the fork present, fresh random shapes are compared live, and a whole run with
each cost model must write the same bytes.
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
from augur.cost.tables import TableCost, bundled_profile

DATA = os.path.join(os.path.dirname(__file__), "data", "table_cost")
FILES = sorted(f for f in os.listdir(DATA) if f.endswith(".json.gz"))
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
    path = bundled_profile(d["hardware"], d["model"])
    assert hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest() == d["profile_sha256"], \
        "profile changed since the regression data was priced: rebuild it (scripts/build_table_regression.py)"
    assert len(d["workload"]) > 1000 and len(d["random"]) >= 3000
    cost = TableCost(path)
    assert mismatches(cost, d["workload"]) == []
    assert mismatches(cost, d["random"]) == []
    cv = d["cpu_variant"]
    other = TableCost(path, cpu_mem_bw=cv["cpu_mem_bw"], cpu_mem_latency_ns=cv["cpu_mem_latency_ns"])
    assert mismatches(other, cv["rows"]) == []


@needs_fork
@pytest.mark.parametrize("name", FILES)
def test_live_against_fork(name):
    from augur.cost.llmservingsim import LLMServingSimCost
    d = load(name)
    ref = LLMServingSimCost(FORK, d["hardware"], d["model"])
    cost = TableCost(bundled_profile(d["hardware"], d["model"]))
    shapes = list(random_shapes(random.Random(987654321), 3000))
    bad = [(s, ref.step_ns(s), cost.step_ns(s)) for s in shapes]
    bad = [b for b in bad if b[1] != b[2]]
    assert bad == []


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


def test_out_of_scope_is_refused(tmp_path):
    src = bundled_profile("RTXPRO6000", "meta-llama/Llama-3.1-8B")
    p = json.loads(gzip.decompress(src.read_bytes()))
    for edit, exc in (({"tp": 2}, NotImplementedError), ({"kind": "moe"}, NotImplementedError)):
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
