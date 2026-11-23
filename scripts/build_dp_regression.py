"""Build tests/data/dp_rounds/<name>.json.gz: DP rounds priced by the LLMServingSim fork (ASTRA-Sim).

Rounds come from augur runs on the fork (--cost llmservingsim; every round_ns call is recorded, evenly sampled down
to --workload-sample) and from random shapes; a second set of random rounds is priced with other CPU-memory and link
settings. tests/test_moe_tables.py checks TableCost.round_ns against them without the fork.

  python scripts/build_dp_regression.py --fork ~/llmss-vllm-sim-20261114 --hardware RTXPRO6000 \
      --model Qwen/Qwen3-30B-A3B-Instruct-2507 --cluster-config CONFIG --out tests/data/dp_rounds/NAME.json.gz \
      --run "<augur arguments of a DP run>" [--run ...]
"""
import argparse, gzip, json, os, random, shlex, subprocess, sys

from augur.cost.convert import random_shapes, shape_to_row
from augur.cost.llmservingsim import LLMServingSimCost


def config_with(path, cpu_bw, cpu_lat, link_bw, link_lat, out):
    with open(path) as f:
        c = json.load(f)
    c["link_bw"], c["link_latency"] = link_bw, link_lat
    c["nodes"][0]["cpu_mem"]["mem_bw"], c["nodes"][0]["cpu_mem"]["mem_latency"] = cpu_bw, cpu_lat
    with open(out, "w") as f:
        json.dump(c, f)
    return out


def params(path):
    with open(path) as f:
        c = json.load(f)
    node = c["nodes"][0]
    return {"link_bw": c["link_bw"], "link_latency_ns": c["link_latency"],
            "npu_mem_bw": node["instances"][0]["npu_mem"]["mem_bw"],
            "cpu_mem_bw": node["cpu_mem"]["mem_bw"], "cpu_mem_latency_ns": node["cpu_mem"]["mem_latency"]}


def random_rounds(ref, n, seed, dp):
    rng = random.Random(seed)
    gen = random_shapes(rng, 10 ** 7)
    out = []
    while len(out) < n:
        shapes = []
        while len(shapes) < dp:
            s = next(gen)
            if 1 <= s.forward_tokens <= 2048:
                shapes.append(s)
        first = rng.randrange(dp)
        out.append([[shape_to_row(s) for s in shapes], first, ref.round_ns(shapes, first_rank=first)])
    return out


def recorded_rounds(argv, fork):
    """Run augur in a subprocess with round_ns recorded; returns [[rows, first, ns], ...] and the call count."""
    code = ("import json, sys\n"
            "from augur.cost.convert import shape_to_row\n"
            "from augur.cost.llmservingsim import LLMServingSimCost\n"
            "seen = []\n"
            "orig = LLMServingSimCost.round_ns\n"
            "def rec(self, shapes, first_rank=0):\n"
            "    ns = orig(self, shapes, first_rank)\n"
            "    seen.append([[shape_to_row(s) for s in shapes], first_rank, ns])\n"
            "    return ns\n"
            "LLMServingSimCost.round_ns = rec\n"
            "from augur.cli import main\n"
            "rc = main(sys.argv[2:])\n"
            "json.dump(seen, open(sys.argv[1], 'w'))\n"
            "sys.exit(rc)\n")
    out = f"/tmp/augur-dp-rounds-{os.getpid()}.json"
    env = dict(os.environ, AUGUR_FORK=fork, HF_HUB_OFFLINE="1")
    r = subprocess.run([sys.executable, "-c", code, out, *argv], env=env, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(r.stderr[-3000:])
    with open(out) as f:
        seen = json.load(f)
    os.unlink(out)
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fork", required=True); ap.add_argument("--hardware", required=True)
    ap.add_argument("--model", required=True); ap.add_argument("--cluster-config", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--workload-sample", type=int, default=1000); ap.add_argument("--random", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    fork = os.path.abspath(os.path.expanduser(a.fork))
    cfg = os.path.abspath(a.cluster_config)
    with open(cfg) as f:
        dp = sum(len(n["instances"]) for n in json.load(f)["nodes"])
    runs, workload = [], []
    for run in a.run:
        argv = shlex.split(run)
        seen = recorded_rounds(argv, fork)
        runs.append({"argv": argv, "rounds": len(seen)})
        workload += seen
    if len(workload) > a.workload_sample:
        workload = [workload[i * len(workload) // a.workload_sample] for i in range(a.workload_sample)]
    ref = LLMServingSimCost(fork, a.hardware, a.model, tp=1, cluster_config=cfg, dp=dp, ep=dp)
    rnd = random_rounds(ref, a.random, a.seed, dp)
    other = config_with(cfg, 3.7, 123, 25.0, 5000, f"/tmp/augur-dp-other-{os.getpid()}.json")
    p_other = params(other)
    ref2 = LLMServingSimCost(fork, a.hardware, a.model, tp=1, cluster_config=other, dp=dp, ep=dp,
                             cpu_mem_bw_gbps=p_other["cpu_mem_bw"], cpu_mem_latency_ns=p_other["cpu_mem_latency_ns"])
    rnd2 = random_rounds(ref2, max(200, a.random // 4), a.seed + 7, dp)
    os.unlink(other)
    out = {"format": 1, "hardware": a.hardware, "model": a.model, "dp": dp, "cluster_config": os.path.basename(cfg),
           "params": params(cfg), "row_layout": "[[shape row per rank], first rank, [ns per rank]]",
           "fork_commit": subprocess.run(["git", "-C", fork, "rev-parse", "HEAD"], capture_output=True, text=True,
                                         env=dict(os.environ, GIT_OPTIONAL_LOCKS="0")).stdout.strip(),
           "workload_runs": runs, "workload": workload, "random_seed": a.seed, "random": rnd,
           "other": {"params": p_other, "rounds": rnd2}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "wb") as f:
        f.write(gzip.compress(json.dumps(out, separators=(",", ":"), sort_keys=True).encode(), 9, mtime=0))
    print(a.out, len(workload), len(rnd), len(rnd2), os.path.getsize(a.out))


main()
