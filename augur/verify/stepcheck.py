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
from dataclasses import dataclass

import numpy as np

from augur.cost import SeqWork, StepShape
from augur.cost.profiles import default_fork
from augur.trace.workload import call_id
from augur.verify.prefix_share import prompt_chains, shared_tokens
from augur.verify.replay import is_async, read_log, steps


MAX_DRAFTS = 8  # a decode step carries at most this many draft tokens (num_speculative_tokens)


def real_steps(events, chains=None):
    """[(shape without padding, measured ns)] for a step log.  ``chains`` (``prefix_share.prompt_chains`` of the run's
    workload): fill ``SeqWork.shared_kv`` from the prompt blocks requests had in common."""
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
        for rid in s["new"]:
            if rid in s["tok"]:
                computed[rid] = s["new"][rid]
        # A decode step is a request past its prompt: 1 token, or 1 + drafts under speculative decoding (where
        # only the emitted tokens, accepted drafts + 1, enter the KV cache; rejected drafts do not).
        decode = {rid: computed.get(rid, 0) >= prompt.get(rid, 0) and emitted.get(rid, 0) >= 1
                  and (n == 1 or n <= 1 + MAX_DRAFTS) for rid, n in s["tok"].items()}
        shared = shared_tokens([(rid, chains.get(call_id(rid), ()), computed.get(rid, 0)) for rid, n in s["tok"].items()
                                if decode[rid] or n == 1]) if chains else {}
        for rid, n in s["tok"].items():
            before = computed.get(rid, 0)
            seqs.append(SeqWork(num_new=n, num_computed_before=before, is_decode=decode[rid] or n == 1,
                                shared_kv=shared.get(rid, 0)))
            computed[rid] = before + (emitted[rid] if decode[rid] and n > 1 else n)
        for rid in s["pre"]:
            computed[rid] = 0
        start = max(s["t"], prev_finish) if async_run and prev_finish is not None else s["t"]
        prev_finish = u["t"]
        out.append((StepShape(seqs=tuple(seqs)), u["t"] - start))
    return out


def classify(shape: StepShape) -> str:
    return shape.step_class


@dataclass
class PricedStep:
    shape: StepShape  # padded as the run's CUDA-graph dispatcher pads it
    graph: bool  # the step runs a CUDA graph
    measured_ns: int
    predicted_ns: int  # the cost model's GPU time for ``shape``
    xfer: dict  # offload transfers reported in the step: {"load": [bytes, s, n], "store": [...]}


def make_pricer(model, hardware, fork, max_num_seqs, max_num_batched_tokens, max_model_len, roofline=False,
                roofline_params=None):
    """(cost model, CUDA-graph dispatcher) of a run's engine settings. Raises FileNotFoundError without a profile.
    ``roofline``: price with ``--cost roofline`` (parameters from the JSON file ``roofline_params``, else the fitted
    defaults) instead of the profile."""
    from augur.sim.engine import EngineSpec, SimEngine
    from augur.cost.llmservingsim import LLMServingSimCost
    from augur.cost.tables import TableCost, bundled_profile
    from augur.sim.models import model_dir

    try:  # as augur --cost auto: the profile shipped with augur, else the fork's
        profile = bundled_profile(hardware, model)
    except FileNotFoundError:
        profile = None
    mdir = model_dir(fork, model, os.path.expanduser("~/.cache/augur/models"), None, None if roofline else profile)
    if roofline:
        from augur.cost.roofline import DEFAULT_PARAMS, RooflineCost, RooflineParams
        params = DEFAULT_PARAMS
        if roofline_params:
            with open(roofline_params, encoding="utf-8") as f:
                params = RooflineParams.from_json(json.load(f))
        cost = RooflineCost(os.path.join(mdir, "config.json"), hardware, params=params)
    else:
        cost = TableCost(profile) if profile else LLMServingSimCost(
            fork_root=fork, hardware=hardware, model=model, max_num_batched_tokens=max_num_batched_tokens)
    # Only for vLLM's CUDA-graph dispatcher under the same engine settings.
    eng = SimEngine(0, EngineSpec(model_dir=mdir, model_name=model, kv_cache_bytes=8 << 30, max_model_len=max_model_len,
                                  max_num_batched_tokens=max_num_batched_tokens, max_num_seqs=max_num_seqs), cost)
    return cost, eng.core.model_executor.worker.cudagraph


def price_steps(events, cost, disp, workload=None) -> list[PricedStep]:
    """Every step of a log, measured and priced. ``workload``: the file the run was driven by, which lets a cost model
    that prices shared KV blocks (``cost.prices_sharing``) see what requests had in common (augur.verify.prefix_share).
    Called by: main, augur.verify.breakdown."""
    from vllm.config import CUDAGraphMode

    xfers = [u.get("xfer") or {} for _, u in steps(events)]  # same order as real_steps
    out = []
    chains = prompt_chains(workload) if workload and getattr(cost, "prices_sharing", False) else None
    for (shape, measured), x in zip(real_steps(events, chains), xfers):
        uniform = all(x.num_new == 1 for x in shape.seqs)
        mode, desc = disp.dispatch(num_tokens=shape.total_tokens, uniform_decode=uniform)
        padded = StepShape(seqs=shape.seqs, padded_tokens=desc.num_tokens)
        out.append(PricedStep(padded, mode != CUDAGraphMode.NONE, measured, cost.step_ns(padded), x))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="augur.verify.stepcheck")
    p.add_argument("log")
    p.add_argument("--fork", default=default_fork(), help="LLMServingSim fork, for models augur ships no profile for")
    p.add_argument("--model", required=True)
    p.add_argument("--hardware", required=True)
    p.add_argument("--max-num-seqs", type=int, default=128)
    p.add_argument("--max-num-batched-tokens", type=int, default=2048)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--workload", default=None, help="the workload file the run was driven by: with it, requests that "
                                                    "shared prompt blocks are priced as sharing them (a profile with "
                                                    "a shared_kv model; see augur.verify.prefix_share)")
    p.add_argument("--csv", default=None, help="per-step rows: class, tokens, seqs, padded, measured, predicted, and the "
                                               "offload bytes loaded and stored by transfers reported in the step")
    p.add_argument("--cost", default="auto", choices=["auto", "roofline"],
                   help="auto: the profile; roofline: --cost roofline")
    p.add_argument("--roofline-params", default=None,
                   help="--cost roofline: parameters JSON (python -m augur.cost.roofline_fit)")
    a = p.parse_args(argv)

    try:
        cost, disp = make_pricer(a.model, a.hardware, a.fork, a.max_num_seqs, a.max_num_batched_tokens, a.max_model_len,
                                 a.cost == "roofline", a.roofline_params)
    except FileNotFoundError as e:
        print(f"augur.verify.stepcheck: {e}", file=sys.stderr)
        return 2
    rows = [(classify(p.shape), p.shape.total_tokens, len(p.shape.seqs), p.shape.padded_tokens, p.graph,
             p.measured_ns, p.predicted_ns, p.xfer.get("load", [0])[0], p.xfer.get("store", [0])[0])
            for p in price_steps(read_log(a.log), cost, disp, a.workload)]

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
            f.write("class,tokens,seqs,padded,cudagraph,measured_ns,predicted_ns,load_bytes,store_bytes\n")
            for r in rows:
                f.write(",".join(str(int(x)) if isinstance(x, (bool, np.bool_)) else str(x) for x in r) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
