"""How well does augur price real vLLM steps?

Rebuilds every step of a real run from its step log (augur.verify.hooks,
synchronous or async scheduling): which requests ran, how many tokens each, and how
many tokens each already had in the KV cache. Pads it with vLLM's own CUDA-graph
dispatcher (same engine settings) and prices it with augur's cost model (the profile shipped with augur
for --hardware and --model if there is one, else the fork's). The
measured duration is the step's schedule start to its update end, so it
includes vLLM's per-step CPU work; the profile prices GPU work only. The
difference is reported per step class and as a fit against batch size.

    python -m augur.verify.stepcheck out/real/run/steps.<pid>.jsonl --model Qwen/Qwen3-1.7B \\
        --hardware RTX4060Ti --max-num-seqs 128 --max-num-batched-tokens 2048 --max-model-len 8192
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from augur.cost import SeqWork, StepShape
from augur.cost.profiles import default_fork
from augur.verify.replay import is_async, read_log, steps


MAX_DRAFTS = 8  # a decode step carries at most this many draft tokens (num_speculative_tokens)


def real_steps(events):
    """[(shape without padding, measured ns)] for a step log."""
    # Async runs overlap scheduling with the GPU: a batch runs from when it was
    # scheduled or the previous batch finished, whichever is later, to its update.
    async_run = is_async(events)
    computed: dict[str, int] = {}
    prompt = {e["rid"]: e["prompt"] for e in events if e["ev"] == "add"}
    out = []
    prev_finish = None
    for s, u in steps(events):
        seqs = []
        emitted = u["out"]
        for rid, n in s["tok"].items():
            if rid in s["new"]:
                computed[rid] = s["new"][rid]
            before = computed.get(rid, 0)
            # A decode step is a request past its prompt: 1 token, or 1 + drafts under speculative decoding (where
            # only the emitted tokens, accepted drafts + 1, enter the KV cache; rejected drafts do not).
            decode = before >= prompt.get(rid, 0) and emitted.get(rid, 0) >= 1 and (n == 1 or n <= 1 + MAX_DRAFTS)
            seqs.append(SeqWork(num_new=n, num_computed_before=before, is_decode=decode or n == 1))
            computed[rid] = before + (emitted[rid] if decode and n > 1 else n)
        for rid in s["pre"]:
            computed[rid] = 0
        start = max(s["t"], prev_finish) if async_run and prev_finish is not None else s["t"]
        prev_finish = u["t"]
        out.append((StepShape(seqs=tuple(seqs)), u["t"] - start))
    return out


def classify(shape: StepShape) -> str:
    if shape.num_prefill == 0:
        return "decode"
    return "prefill" if shape.num_decode == 0 else "mixed"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="augur.verify.stepcheck")
    p.add_argument("log")
    p.add_argument("--fork", default=default_fork(), help="LLMServingSim fork, for models augur ships no profile for")
    p.add_argument("--model", required=True)
    p.add_argument("--hardware", required=True)
    p.add_argument("--max-num-seqs", type=int, default=128)
    p.add_argument("--max-num-batched-tokens", type=int, default=2048)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--csv", default=None, help="per-step rows: class, tokens, seqs, padded, measured, predicted")
    a = p.parse_args(argv)

    from augur.sim.engine import EngineSpec, SimEngine
    from augur.cost.llmservingsim import LLMServingSimCost
    from augur.cost.tables import TableCost, bundled_profile
    from augur.sim.models import model_dir
    from vllm.config import CUDAGraphMode

    try:
        try:  # as augur --cost auto: the profile shipped with augur, else the fork's
            profile = bundled_profile(a.hardware, a.model)
        except FileNotFoundError:
            profile = None
        mdir = model_dir(a.fork, a.model, os.path.expanduser("~/.cache/augur/models"), None, profile)
        cost = TableCost(profile) if profile else LLMServingSimCost(
            fork_root=a.fork, hardware=a.hardware, model=a.model, max_num_batched_tokens=a.max_num_batched_tokens)
    except FileNotFoundError as e:
        print(f"augur.verify.stepcheck: {e}", file=sys.stderr)
        return 2
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
