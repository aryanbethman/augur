"""Calibrate augur's step prices on one real run: one GPU-time scale per step class.

augur prices a step from a GPU profile (augur.verify.stepcheck holds the per-step predicted and measured times). To ask
"what if I changed setting X?" of a real run, augur prices the steps that change; this fits a correction on the real run
first, and `--step-class-scales` applies it to the simulation of the changed setting.

Method. A step is decode (no prefill sequence), prefill (only prefill sequences) or mixed. A measured step is the
schedule-to-update time, which holds vLLM's CPU work; augur adds that as a constant (--step-cpu-us, 2000 by default,
none for async scheduling where it hides behind the previous batch) and prices the GPU work. Per class c, the scale is

    s_c = sum(measured - cpu) / sum(predicted)     over the class's steps

so that the class's total time matches the real run. A class with fewer than --min-steps steps (a ShareGPT run has two
pure-prefill steps) takes the scale of all steps pooled. Steps that waited on KV offload transfers are left out.
The cpu term is not fitted: it is a property of the host, and a fit of it (and of a per-request term) on one run
extrapolated worse to other batch sizes than leaving it.

    python -m augur.verify.calibrate out/real/run/steps.<pid>.jsonl --model Qwen/Qwen3-1.7B --hardware RTX4060Ti \\
        --max-num-seqs 128 --max-num-batched-tokens 2048 --max-model-len 8192 --output scales.json
    python -m augur.cli ... --step-class-scales scales.json          (also: augur.verify.breakdown ...)

It prints the scales, and the error of the total step time per class before and after, and for a check the same with
the scales fitted on the log's first half and applied to its second.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from augur.cost import STEP_CLASSES
from augur.cost.profiles import default_fork
from augur.verify.replay import is_async, read_log

MIN_STEPS = 30


def fit_scales(classes, measured_ns, predicted_ns, cpu_ns: float, min_steps: int = MIN_STEPS) -> dict:
    """{"scales": {class: s}, "steps": {class: n}, "pooled": s} from per-step class names and times. Pure."""
    cl = np.asarray(classes)
    gpu = np.asarray(measured_ns, dtype=float) - cpu_ns
    pred = np.asarray(predicted_ns, dtype=float)
    pooled = float(gpu.sum() / pred.sum())
    scales, steps = {}, {}
    for c in STEP_CLASSES:
        idx = cl == c
        steps[c] = int(idx.sum())
        scales[c] = float(gpu[idx].sum() / pred[idx].sum()) if steps[c] >= min_steps else pooled
    return {"scales": scales, "steps": steps, "pooled": pooled}


def total_error(classes, measured_ns, predicted_ns, cpu_ns: float, scales: dict | None) -> dict:
    """Relative error of the summed step time (predicted + cpu against measured) per class and over all. Pure."""
    cl = np.asarray(classes)
    m = np.asarray(measured_ns, dtype=float)
    p = np.asarray(predicted_ns, dtype=float) * np.array([(scales or {}).get(c, 1.0) for c in cl]) + cpu_ns
    out = {c: float(p[cl == c].sum() / m[cl == c].sum() - 1) for c in STEP_CLASSES if (cl == c).any()}
    out["all"] = float(p.sum() / m.sum() - 1)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="augur.verify.calibrate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="the real run's step log (steps.<pid>.jsonl)")
    ap.add_argument("--fork", default=default_fork(), help="LLMServingSim fork, for models augur ships no profile for")
    ap.add_argument("--model", required=True)
    ap.add_argument("--hardware", required=True)
    ap.add_argument("--max-num-seqs", type=int, default=128)
    ap.add_argument("--max-num-batched-tokens", type=int, default=2048)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--step-cpu-us", type=float, default=None,
                    help="CPU time per step that augur adds to the GPU price (default: augur's 2000 for a synchronous "
                         "run, 0 for an async one)")
    ap.add_argument("--min-steps", type=int, default=MIN_STEPS,
                    help="a class with fewer steps takes the scale of all steps")
    ap.add_argument("--output", default=None, help="write the scales here as JSON (augur --step-class-scales takes it)")
    a = ap.parse_args(argv)

    from augur.verify.stepcheck import classify, make_pricer, price_steps
    events = read_log(a.log)
    try:
        cost, disp = make_pricer(a.model, a.hardware, a.fork, a.max_num_seqs, a.max_num_batched_tokens, a.max_model_len)
    except FileNotFoundError as e:
        print(f"augur.verify.calibrate: {e}", file=sys.stderr)
        return 2
    cpu_ns = (a.step_cpu_us if a.step_cpu_us is not None else (0.0 if is_async(events) else 2000.0)) * 1e3
    priced = [p for p in price_steps(events, cost, disp) if not p.xfer]
    cl = [classify(p.shape) for p in priced]
    m = [p.measured_ns for p in priced]
    pr = [p.predicted_ns for p in priced]

    fit = fit_scales(cl, m, pr, cpu_ns, a.min_steps)
    half = len(cl) // 2
    check = fit_scales(cl[:half], m[:half], pr[:half], cpu_ns, a.min_steps)
    res = {**fit, "step_cpu_us": cpu_ns / 1e3, "log": a.log,
           "total_error": {"uncalibrated": total_error(cl, m, pr, cpu_ns, None),
                           "calibrated": total_error(cl, m, pr, cpu_ns, fit["scales"])},
           "first_half_fit": {"scales": check["scales"],
                              "second_half_error_uncalibrated": total_error(cl[half:], m[half:], pr[half:], cpu_ns, None),
                              "second_half_error_calibrated": total_error(cl[half:], m[half:], pr[half:], cpu_ns,
                                                                          check["scales"])}}
    print(json.dumps(res, indent=2))
    if a.output:
        with open(a.output, "w") as f:
            json.dump(res, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
