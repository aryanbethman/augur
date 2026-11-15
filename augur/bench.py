"""`augur bench`: `vllm bench serve` without a GPU or a server.

Takes `vllm bench serve`'s own arguments (dataset, number of prompts, request
rate, burstiness, seed, percentiles, goodput, --save-result ...) and samples
requests and arrival times with vLLM's own benchmark code, so the same seed
gives the same requests at the same times. Instead of sending them to a
server, it runs them through augur (any augur engine flag, e.g. --hardware,
--tp, --num-gpu-blocks, passes through) and reports with vLLM's own
`calculate_metrics`, in the same format.

    python -m augur.bench --model Qwen/Qwen3-1.7B --dataset-name random \\
        --random-input-len 1024 --random-output-len 128 --num-prompts 200 \\
        --request-rate 4 --ignore-eos --hardware RTX4060Ti --num-gpu-blocks 5300

Outputs run to the requested length (as with --ignore-eos): augur generates
no text, so it cannot know where a model would stop.
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


def main(argv=None) -> int:
    from vllm.benchmarks import serve
    from vllm.benchmarks.datasets.datasets import get_samples
    from vllm.benchmarks.lib.endpoint_request_func import RequestFuncOutput
    from vllm.tokenizers import get_tokenizer
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    p = FlexibleArgumentParser(prog="augur bench", description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    serve.add_cli_args(p)
    args, augur_args = p.parse_known_args(argv)
    if not args.ignore_eos:
        print("augur bench: outputs always run to their requested length (as with --ignore-eos)", file=sys.stderr)

    tokenizer = get_tokenizer(args.tokenizer or args.model, trust_remote_code=args.trust_remote_code)
    random.seed(args.seed)
    np.random.seed(args.seed)
    requests = get_samples(args, tokenizer)
    offsets = _arrival_offsets(requests, args)

    work = tempfile.mkdtemp(prefix="augur-bench-")
    trace = os.path.join(work, "requests.jsonl")
    with open(trace, "w") as f:
        for r, t in zip(requests, offsets):
            ids = r.prompt if isinstance(r.prompt, list) and r.prompt and isinstance(r.prompt[0], int) \
                else tokenizer(r.prompt).input_ids
            f.write(json.dumps({"input_toks": len(ids), "output_toks": r.expected_output_len,
                                "arrival_time_ns": int(t * 1e9), "input_tok_ids": ids}) + "\n")

    from augur.cli import main as augur_main
    if "--cost" not in augur_args:
        augur_args = ["--cost", "llmservingsim"] + augur_args
    out_csv = os.path.join(work, "calls.csv")
    t0 = time.perf_counter()
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):  # augur's own summary; vLLM's report follows
        rc = augur_main(["--model", args.model, "--workload", trace, "--output", out_csv, "--quiet"] + augur_args)
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

    if args.save_result:
        result = {"backend": "augur", "model_id": args.model, "num_prompts": args.num_prompts,
                  "request_rate": args.request_rate, "burstiness": args.burstiness, "duration": end,
                  "augur_wall_s": wall, "augur_args": augur_args}
        result.update({k: v for k, v in vars(metrics).items() if not k.startswith("_")})
        name = args.result_filename or f"augur-{args.request_rate}qps-{os.path.basename(args.model)}.json"
        path = os.path.join(args.result_dir or ".", name)
        with open(path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print("saved", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
