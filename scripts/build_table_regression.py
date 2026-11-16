"""Build tests/data/table_cost/<name>.json.gz: (shape, expected ns) pairs priced by the fork.

Needs the LLMServingSim fork.  Workload shapes come from scripts/collect_step_shapes.py runs
(augur --cost llmservingsim, every StepShape passed to step_ns); an evenly spaced sample of
them is kept, with random shapes on top.  Example:

  python scripts/build_table_regression.py --fork ~/llmss-vllm-sim-20261114 \
      --hardware RTXPRO6000 --model meta-llama/Llama-3.1-8B \
      --profile augur/profiles/RTXPRO6000/meta-llama--Llama-3.1-8B/bf16-tp1.json.gz \
      --workload-json llama_sync.json llama_async.json --out tests/data/table_cost/llama-3.1-8b-rtxpro6000.json.gz
"""
import argparse, gzip, hashlib, json, os, random, subprocess

from augur.cost.convert import random_shapes, shape_from_row, shape_to_row
from augur.cost.llmservingsim import LLMServingSimCost


def distinct_random(seed, n):
    seen, out = set(), []
    while len(out) < n:
        for s in random_shapes(random.Random(seed), n):
            row = shape_to_row(s)
            k = json.dumps(row)
            if k not in seen:
                seen.add(k); out.append(row)
                if len(out) == n:
                    break
        seed += 1000003
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fork", required=True); ap.add_argument("--hardware", required=True)
    ap.add_argument("--model", required=True); ap.add_argument("--profile", required=True)
    ap.add_argument("--workload-json", nargs="+", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--random", type=int, default=3000); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workload-sample", type=int, default=3000, help="keep this many workload shapes, evenly spaced")
    a = ap.parse_args()
    fork = os.path.abspath(os.path.expanduser(a.fork))
    ref = LLMServingSimCost(fork, a.hardware, a.model)
    ref2 = LLMServingSimCost(fork, a.hardware, a.model, cpu_mem_bw_gbps=3.7, cpu_mem_latency_ns=123)
    price = lambda c, row: c.step_ns(shape_from_row(row))
    runs, seen, workload = [], set(), []
    for path in a.workload_json:
        d = json.load(open(path))
        new = 0
        for row in d["rows"]:
            k = json.dumps(row)
            if k not in seen:
                seen.add(k); new += 1; workload.append(row + [price(ref, row)])
        runs.append({"argv": d["argv"], "step_calls": d["calls"], "distinct_shapes": len(d["rows"]), "new_after_dedupe": new})
    total = len(workload)
    if total > a.workload_sample:
        workload.sort(key=lambda r: json.dumps(r[:5]))
        workload = [workload[i * total // a.workload_sample] for i in range(a.workload_sample)]
    rnd = [r + [price(ref, r)] for r in distinct_random(a.seed, a.random)]
    cpu = [r + [price(ref2, r)] for r in distinct_random(a.seed + 7, 600)]
    out = {"format": 1, "profile_sha256": hashlib.sha256(gzip.decompress(open(a.profile, "rb").read())).hexdigest(),
           "hardware": a.hardware, "model": a.model,
           "fork_commit": subprocess.run(["git", "-C", fork, "rev-parse", "HEAD"], capture_output=True, text=True,
                                         env=dict(os.environ, GIT_OPTIONAL_LOCKS="0")).stdout.strip(),
           "row_layout": "[padded_tokens or -1, dummy, [chunk, history, ...], [decode kv, ...], decode q len, expected ns]",
           "workload_runs": runs, "workload_distinct_shapes": total, "workload": workload, "random_seed": a.seed, "random": rnd,
           "cpu_variant": {"cpu_mem_bw": 3.7, "cpu_mem_latency_ns": 123, "rows": cpu}}
    data = json.dumps(out, separators=(",", ":"), sort_keys=True).encode()
    with open(a.out, "wb") as f:
        f.write(gzip.compress(data, 9, mtime=0))
    print(a.out, len(workload), len(rnd), len(cpu), os.path.getsize(a.out))


main()
