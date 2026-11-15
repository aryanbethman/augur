"""How well does augur price real vLLM steps?

Rebuilds every step of a real run from its step log (augur.vllm_hooks,
synchronous scheduling): which requests ran, how many tokens each, and how
many tokens each already had in the KV cache. Pads it with vLLM's own CUDA-graph
dispatcher (same engine settings) and prices it with augur's cost model. The
measured duration is the step's schedule start to its update end, so it
includes vLLM's per-step CPU work; the profile prices GPU work only. The
difference is reported per step class and as a fit against batch size.

    python -m augur.stepcheck out/real/run/steps.<pid>.jsonl --model Qwen/Qwen3-1.7B \\
        --hardware RTX4060Ti --max-num-seqs 128 --max-num-batched-tokens 2048 --max-model-len 8192
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from augur.cost import SeqWork, StepShape
from augur.replay import is_async, read_log, steps


def real_steps(events):
    """[(shape without padding, measured ns)] for a synchronous step log."""
    if is_async(events):
        raise SystemExit("stepcheck needs a synchronous run (async step times overlap)")
    computed: dict[str, int] = {}
    out = []
    for s, u in steps(events):
        seqs = []
        for rid, n in s["tok"].items():
            if rid in s["new"]:
                computed[rid] = s["new"][rid]
            before = computed.get(rid, 0)
            seqs.append(SeqWork(num_new=n, num_computed_before=before, is_decode=n == 1))
            computed[rid] = before + n
        for rid in s["pre"]:
            computed[rid] = 0
        out.append((StepShape(seqs=tuple(seqs)), u["t"] - s["t"]))
    return out


def classify(shape: StepShape) -> str:
    if shape.num_prefill == 0:
        return "decode"
    return "prefill" if shape.num_decode == 0 else "mixed"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="augur.stepcheck")
    p.add_argument("log")
    p.add_argument("--fork", default=os.environ.get("AUGUR_FORK", os.path.expanduser("~/llmss-vllm-sim-20261114")))
    p.add_argument("--model", required=True)
    p.add_argument("--hardware", required=True)
    p.add_argument("--max-num-seqs", type=int, default=128)
    p.add_argument("--max-num-batched-tokens", type=int, default=2048)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--csv", default=None, help="per-step rows: class, tokens, seqs, padded, measured, predicted")
    a = p.parse_args(argv)

    from augur.engine import EngineSpec, SimEngine
    from augur.lss_cost import LLMServingSimCost
    from augur.models import model_dir
    from vllm.config import CUDAGraphMode

    mdir = model_dir(a.fork, a.model, os.path.expanduser("~/.cache/augur/models"))
    cost = LLMServingSimCost(fork_root=a.fork, hardware=a.hardware, model=a.model,
                             max_num_batched_tokens=a.max_num_batched_tokens)
    # Only for vLLM's CUDA-graph dispatcher under the same engine settings.
    eng = SimEngine(0, EngineSpec(model_dir=mdir, model_name=a.model, kv_cache_bytes=8 << 30,
                                  max_model_len=a.max_model_len, max_num_batched_tokens=a.max_num_batched_tokens,
                                  max_num_seqs=a.max_num_seqs), cost)
    disp = eng.core.model_executor.worker.cudagraph

    rows = []
    for shape, measured in real_steps(read_log(a.log)):
        uniform = all(x.num_new == 1 for x in shape.seqs)
        mode, desc = disp.dispatch(num_tokens=shape.total_tokens, uniform_decode=uniform)
        padded = StepShape(seqs=shape.seqs, padded_tokens=desc.num_tokens)
        rows.append((classify(shape), shape.total_tokens, len(shape.seqs), desc.num_tokens,
                     mode != CUDAGraphMode.NONE, measured, cost.step_ns(padded)))

    m = np.array([r[5] for r in rows], dtype=float)
    pr = np.array([r[6] for r in rows], dtype=float)
    report = {"steps": len(rows), "measured_total_s": m.sum() / 1e9, "predicted_total_s": pr.sum() / 1e9,
              "total_ratio": pr.sum() / m.sum()}
    for cls in ("decode", "prefill", "mixed"):
        idx = [i for i, r in enumerate(rows) if r[0] == cls]
        if not idx:
            continue
        mm, pp = m[idx], pr[idx]
        rel = (pp - mm) / mm
        report[cls] = {"steps": len(idx), "measured_mean_ms": mm.mean() / 1e6, "predicted_mean_ms": pp.mean() / 1e6,
                       "median_rel_err": float(np.median(rel)), "p10": float(np.percentile(rel, 10)),
                       "p90": float(np.percentile(rel, 90)), "median_gap_ms": float(np.median(mm - pp)) / 1e6}
    # The gap the profile does not explain, against batch size (requests).
    seqs = np.array([r[2] for r in rows], dtype=float)
    gap = m - pr
    slope, intercept = np.polyfit(seqs, gap, 1)
    report["gap_fit"] = {"intercept_ms": intercept / 1e6, "per_request_us": slope / 1e3,
                         "note": "measured - predicted ~ intercept + slope * requests in step"}
    print(json.dumps(report, indent=2))
    if a.csv:
        with open(a.csv, "w") as f:
            f.write("class,tokens,seqs,padded,cudagraph,measured_ns,predicted_ns\n")
            for r in rows:
                f.write(",".join(str(int(x)) if isinstance(x, (bool, np.bool_)) else str(x) for x in r) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
