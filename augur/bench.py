"""`augur bench`: `vllm bench serve` without a GPU or a server.

Takes `vllm bench serve`'s own arguments (dataset, number of prompts, request
rate, burstiness, seed, percentiles, goodput, --save-result ...) and samples
requests and arrival times with vLLM's own benchmark code, so the same seed
gives the same requests at the same times. Instead of sending them to a
server, it runs them through augur (any augur engine flag, e.g. --hardware,
--tp, --num-gpu-blocks, passes through) and reports with vLLM's own
`calculate_metrics`, in the same format.

    augur bench --model Qwen/Qwen3-1.7B --dataset-name random \\
        --random-input-len 1024 --random-output-len 128 --num-prompts 200 \\
        --request-rate 4 --ignore-eos --hardware RTX4060Ti --num-gpu-blocks 5300

Outputs run to the requested length (as with --ignore-eos): augur generates
no text, so it cannot know where a model would stop.

The same trace is also run with step times scaled by 1 -/+ --sensitivity percent
and the spread of each metric is printed, since near saturation a 1% change in
step time can move TTFT by tens of percent.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import sys
import tempfile
import time

import numpy as np

from augur.trace.metrics import ensure_parent_dir


def _arrival_offsets(requests, args) -> list[float]:
    """vLLM's own get_request arrival schedule, on a fake clock (no sleeping)."""
    from vllm.benchmarks import serve

    now = [0.0]

    async def fake_sleep(dt):
        now[0] += dt

    offsets: list[float] = []

    async def collect():
        gen = serve.get_request(requests, args.request_rate, args.burstiness, args.ramp_up_strategy,
                                args.ramp_up_start_rps, args.ramp_up_end_rps)
        async for _ in gen:
            offsets.append(now[0])

    real_time, real_sleep = serve.time.time, serve.asyncio.sleep
    serve.time.time, serve.asyncio.sleep = (lambda: now[0]), fake_sleep
    try:
        asyncio.run(collect())
    finally:
        serve.time.time, serve.asyncio.sleep = real_time, real_sleep
    return offsets


def _run(args, augur_args, trace, requests, offsets, tokenizer, scale):
    """One augur run of the trace: (metrics, end time s, wall s), or an int exit code."""
    import contextlib
    import io

    from vllm.benchmarks import serve
    from vllm.benchmarks.lib.endpoint_request_func import RequestFuncOutput

    from augur.cli import main as augur_main
    out_csv = trace[:-len(".jsonl")] + f"-{scale}.csv"
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):  # augur's own summary; vLLM's report follows
        rc = augur_main(["--model", args.model, "--workload", trace, "--output", out_csv, "--quiet",
                         "--step-time-scale", repr(scale)] + augur_args)
    wall = time.perf_counter() - t0
    if rc:
        return rc

    by_id = {}
    with open(out_csv) as f:
        for row in csv.DictReader(f):
            by_id[int(row["request id"])] = row
    outputs, end = [], 0.0
    for i, r in enumerate(requests):
        row = by_id.get(i)
        o = RequestFuncOutput(prompt_len=r.prompt_len, start_time=offsets[i])
        if row is not None:
            itl = [x / 1e9 for x in json.loads(row["ITL"])]
            o.success, o.output_tokens = True, int(row["output"])
            o.ttft, o.itl, o.latency = int(row["TTFT"]) / 1e9, itl, int(row["latency"]) / 1e9
            end = max(end, int(row["end_time"]) / 1e9)
        outputs.append(o)
    selected = [float(x) for x in args.metric_percentiles.split(",")]
    goodput = serve.check_goodput_args(args)
    metrics, _ = serve.calculate_metrics(requests, outputs, end, tokenizer, selected, goodput)
    return metrics, end, wall


def _sens_values(m) -> dict:
    v = {"request_throughput": m.request_throughput, "mean_ttft_ms": m.mean_ttft_ms,
         "median_ttft_ms": m.median_ttft_ms}
    v.update({f"p{pct:g}_ttft_ms": x for pct, x in m.percentiles_ttft_ms if pct == 99})
    v.update({"mean_tpot_ms": m.mean_tpot_ms, "median_tpot_ms": m.median_tpot_ms, "mean_e2el_ms": m.mean_e2el_ms})
    return v


_SENS_NAMES = {"request_throughput": "Request throughput (req/s)", "mean_ttft_ms": "Mean TTFT (ms)",
               "median_ttft_ms": "Median TTFT (ms)", "p99_ttft_ms": "P99 TTFT (ms)",
               "mean_tpot_ms": "Mean TPOT (ms)", "median_tpot_ms": "Median TPOT (ms)",
               "mean_e2el_ms": "Mean E2EL (ms)"}


def main(argv=None) -> int:
    if argv is None:
        from augur.cli import pin_hash_seed
        pin_hash_seed("augur.bench")
    from vllm.benchmarks import serve
    from vllm.benchmarks.datasets.datasets import get_samples
    from vllm.tokenizers import get_tokenizer
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    p = FlexibleArgumentParser(prog="augur bench", description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    serve.add_cli_args(p)
    p.add_argument("--sensitivity", type=float, default=1.0,
                   help="also run with step times scaled by 1 -/+ this many percent and print how much each "
                        "metric moves (3 runs); 0 disables")
    p.add_argument("--step-log", default=None, help="augur's --step-log, for the reported run only")
    p.add_argument("--step-log-vllm", default=None, help="augur's --step-log-vllm, for the reported run only")
    p.add_argument("--step-time-scale", type=float, default=1.0,
                   help="augur's --step-time-scale: multiplies every GPU step time (applied to all sensitivity runs)")
    args, augur_args = p.parse_known_args(argv)
    if args.max_concurrency is not None:
        print("augur bench: --max-concurrency is not modeled: requests arrive on the --request-rate schedule alone",
              file=sys.stderr)
        return 2
    if not args.ignore_eos:
        print("augur bench: outputs always run to their requested length (as with --ignore-eos)", file=sys.stderr)

    name = args.tokenizer or args.model
    try:
        tokenizer = get_tokenizer(name, trust_remote_code=args.trust_remote_code)
    except OSError as e:
        print(f"augur bench: cannot load the tokenizer {name!r} ({str(e).splitlines()[0]}). The benchmark builds "
              "its requests with the model's tokenizer, as vllm bench serve does: let Hugging Face download it "
              "(unset HF_HUB_OFFLINE), or pass --tokenizer DIR with a local copy.", file=sys.stderr)
        return 2
    random.seed(args.seed)
    np.random.seed(args.seed)
    requests = get_samples(args, tokenizer)
    offsets = _arrival_offsets(requests, args)

    if "--cost" not in augur_args:
        augur_args = ["--cost", "auto"] + augur_args
    base = args.step_time_scale
    f = args.sensitivity / 100
    scales = [base * (1 - f), base, base * (1 + f)] if f > 0 else [base]
    runs = []
    with tempfile.TemporaryDirectory(prefix="augur-bench-") as work:
        trace = os.path.join(work, "requests.jsonl")
        with open(trace, "w") as out:
            for r, t in zip(requests, offsets):
                ids = r.prompt if isinstance(r.prompt, list) and r.prompt and isinstance(r.prompt[0], int) \
                    else tokenizer(r.prompt).input_ids
                out.write(json.dumps({"input_toks": len(ids), "output_toks": r.expected_output_len,
                                      "arrival_time_ns": int(t * 1e9), "input_tok_ids": ids}) + "\n")
        logs = [x for flag, v in (("--step-log", args.step_log), ("--step-log-vllm", args.step_log_vllm)) if v
                for x in (flag, v)]
        for sc in scales:
            r = _run(args, augur_args + (logs if sc == base else []), trace, requests, offsets, tokenizer, sc)
            if isinstance(r, int):
                return r
            runs.append(r)
    metrics, end, _ = runs[len(runs) // 2]
    wall = sum(r[2] for r in runs)

    print("{s:{c}^60}".format(s=" augur Benchmark Result ", c="="))
    rows = [("Successful requests:", metrics.completed), ("Benchmark duration (s):", f"{end:.2f}"),
            ("Total input tokens:", metrics.total_input), ("Total generated tokens:", metrics.total_output),
            ("Request throughput (req/s):", f"{metrics.request_throughput:.2f}"),
            ("Output token throughput (tok/s):", f"{metrics.output_throughput:.2f}"),
            ("Total token throughput (tok/s):", f"{metrics.total_token_throughput:.2f}")]
    for name in ("ttft", "tpot", "itl", "e2el"):
        rows.append((f"Mean {name.upper()} (ms):", f"{getattr(metrics, f'mean_{name}_ms'):.2f}"))
        rows.append((f"Median {name.upper()} (ms):", f"{getattr(metrics, f'median_{name}_ms'):.2f}"))
        for pct, v in getattr(metrics, f"percentiles_{name}_ms"):
            rows.append((f"P{pct:g} {name.upper()} (ms):", f"{v:.2f}"))
    for k, v in rows:
        print("{:<40} {:<10}".format(k, v))
    print("{:<40} {:<10}".format("augur wall time (s):", f"{wall:.2f}"))
    print("=" * 60)

    sens = None
    if f > 0:
        vals = [_sens_values(r[0]) for r in runs]
        spread = {k: 100 * (max(v[k] for v in vals) - min(v[k] for v in vals)) / vals[1][k] if vals[1][k] else 0.0
                  for k in vals[1]}
        print(f"\nSensitivity to step time +-{args.sensitivity:g}%")
        print("{:<28} {:>12} {:>12} {:>12} {:>9}".format("", f"x{scales[0]:.4g}", f"x{scales[1]:.4g}",
                                                       f"x{scales[2]:.4g}", "spread"))
        for k in vals[1]:
            print("{:<28} {:>12.2f} {:>12.2f} {:>12.2f} {:>8.1f}%".format(
                _SENS_NAMES[k], vals[0][k], vals[1][k], vals[2][k], spread[k]))
        worst = max((spread[k] for k in spread if "ttft" in k), default=0.0)
        if worst > 10:
            print(f"TTFT moves by up to {worst:.0f}% for a +-{args.sensitivity:g}% change in step time: the system "
                  "is near saturation, so read TTFT as a range, not a single number.")
        sens = {"percent": args.sensitivity,
                "runs": [{"step_time_scale": sc, **v} for sc, v in zip(scales, vals)]}

    if args.save_result:
        result = {"backend": "augur", "model_id": args.model, "num_prompts": args.num_prompts,
                  "request_rate": args.request_rate, "burstiness": args.burstiness, "duration": end,
                  "augur_wall_s": wall, "augur_args": augur_args}
        if sens:
            result["sensitivity"] = sens
        result.update({k: v for k, v in vars(metrics).items() if not k.startswith("_")})
        name = args.result_filename or f"augur-{args.request_rate}qps-{os.path.basename(args.model)}.json"
        path = os.path.join(args.result_dir or ".", name)
        ensure_parent_dir(path)
        with open(path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print("saved", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
