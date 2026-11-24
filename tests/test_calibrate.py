"""augur.verify.calibrate's fit, --step-class-scales and StepShape.step_class (no GPU)."""

import csv
import json
import os
import subprocess
import sys

import pytest

from augur.cost import SeqWork, StepShape, parse_class_scales
from augur.verify.calibrate import fit_scales, total_error

BASE = ["--model", "Qwen/Qwen3-1.7B", "--cost", "fixed",
        "--num-gpu-blocks", "600", "--max-num-seqs", "16", "--max-model-len", "8192",
        "--max-num-batched-tokens", "512", "--fixed-step-ms", "8", "--fixed-per-token-us", "15",
        "--no-recorded-outputs", "--no-async-scheduling", "--step-cpu-us", "0", "--quiet"]


def test_step_class():
    d, p = SeqWork(1, 10, True), SeqWork(64, 0, False)
    assert StepShape(seqs=(d, d)).step_class == "decode"
    assert StepShape(seqs=(p,)).step_class == "prefill"
    assert StepShape(seqs=(d, p)).step_class == "mixed"


def test_fit_recovers_scales_and_pools_rare_class():
    pred = [100.0] * 40 + [300.0] * 40 + [50.0] * 2
    cl = ["decode"] * 40 + ["mixed"] * 40 + ["prefill"] * 2
    meas = [p * s + 10 for p, s in zip(pred, [1.1] * 40 + [0.9] * 40 + [2.0] * 2)]
    fit = fit_scales(cl, meas, pred, cpu_ns=10, min_steps=30)
    assert fit["scales"]["decode"] == pytest.approx(1.1)
    assert fit["scales"]["mixed"] == pytest.approx(0.9)
    assert fit["scales"]["prefill"] == pytest.approx(fit["pooled"])  # two steps: the pooled scale
    assert fit["steps"] == {"decode": 40, "prefill": 2, "mixed": 40}
    err = total_error(cl, meas, pred, 10, fit["scales"])
    assert abs(err["decode"]) < 1e-9 and abs(err["mixed"]) < 1e-9
    assert abs(total_error(cl, meas, pred, 10, None)["decode"]) > 0.05


def test_parse_class_scales(tmp_path):
    assert parse_class_scales(None) == {}
    assert parse_class_scales("decode=1.5, mixed=0.5") == {"decode": 1.5, "mixed": 0.5}
    f = tmp_path / "s.json"
    f.write_text(json.dumps({"scales": {"decode": 1.2}, "pooled": 1.0}))
    assert parse_class_scales(str(f)) == {"decode": 1.2}
    with pytest.raises(ValueError):
        parse_class_scales("bogus=1")


def augur(tmp_path, workload, name, *args):
    env = dict(os.environ, VLLM_LOGGING_LEVEL="WARNING")
    log = tmp_path / f"{name}-steps.csv"
    subprocess.run([sys.executable, "-m", "augur.cli", *BASE, "--workload", workload, "--output", str(tmp_path / f"{name}.csv"),
                    "--step-log", str(log), *args], check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(log) as f:
        return list(csv.DictReader(f))


def test_class_scales_flag_scales_only_that_class(tmp_path, workload):
    base = augur(tmp_path, workload, "base")
    scaled = augur(tmp_path, workload, "scaled", "--step-class-scales", "decode=2.0")
    assert sum(int(r["dur_ns"]) for r in scaled) > sum(int(r["dur_ns"]) for r in base)
    assert int(scaled[0]["dur_ns"]) == int(base[0]["dur_ns"])  # the first step is a prefill: its class is not scaled


def test_class_scales_of_one_is_bit_identical(tmp_path, workload):
    augur(tmp_path, workload, "plain")
    augur(tmp_path, workload, "one", "--step-class-scales", "decode=1.0,prefill=1.0,mixed=1.0")
    assert (tmp_path / "plain.csv").read_bytes() == (tmp_path / "one.csv").read_bytes()
