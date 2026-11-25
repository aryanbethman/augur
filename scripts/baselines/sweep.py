"""Does a simulator pick the right engine setting?  A real-vLLM sweep of one setting against every simulator's.

    python scripts/baselines/sweep.py real [--values 256 512 ...]      # one real vLLM run per value, GPU 0 (flock ~/gpu0.lock)
    python scripts/baselines/sweep.py sims [--jobs 2]                  # augur, BLIS, Frontier, Vidur per value (run.sh)
    python scripts/baselines/sweep.py table                            # metric per simulator, Kendall tau, best pick, regret
    python scripts/baselines/sweep.py throughput [--budgets N] [--rates N] [--parallel 4]   # configurations per hour

Setting swept: max-num-batched-tokens (the token budget of a step), on the tenants workload of overlap_workloads.py
(400 requests, 6/s, prompts that share a platform preamble and a tenant prompt: which simulators see the token ids
matters). Metric, fixed before the results: mean end-to-end latency over the requests, as the client measures it
(compare.py's ``abs.e2e`` mean); the best setting is the lowest. Every simulator is priced by pricer.py (run.sh), BLIS
gets BLIS_ENCODING=best; augur is run as a user would (no replayed durations). Real runs are out/real/sweep_mbt<N>,
simulators' out/baselines/sweep_mbt<N>.  ``throughput`` times the sweep's setting crossed with arrival rates (the
workload's arrivals scaled), the simulator runs pinned to cores 4-7 with ``taskset``.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import itertools
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
WORKLOAD = os.environ.get("SWEEP_WORKLOAD", os.path.expanduser("~/augur-wt/overlap/out/overlap/tenants.jsonl"))
VALUES = [256, 512, 1024, 2048, 4096, 8192]
SIMS = ("augur", "blis", "frontier", "vidur")  # augur: run.sh's, on the one price every simulator has
CORES = "4-7"


def real_dir(v):
    return os.path.join(ROOT, "out/real", f"sweep_mbt{v}")


def base_dir(name):
    return os.path.join(ROOT, "out/baselines", name)


def sh(cmd, env=None, **kw):
    return subprocess.run(cmd, shell=True, env={**os.environ, **(env or {})}, **kw)


def run_sh(name, rdir, wl, stages, env=None, pin=False):
    """scripts/baselines/run.sh NAME STAGES for a real run directory; returns wall seconds."""
    e = {"AUGUR_RUN_DIR": rdir, "AUGUR_WORKLOAD": wl, "BLIS_ENCODING": "best", **(env or {})}
    t = time.time()
    r = sh(f"{'taskset -c ' + CORES + ' ' if pin else ''}nice -n 10 {HERE}/run.sh {name} {stages}", e,
           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return time.time() - t, r.returncode


def cmd_real(a):
    for v in a.values:
        d = real_dir(v)
        if os.path.exists(f"{d}/client.csv") and os.path.exists(f"{d}/client.json"):
            continue
        r = sh(f"flock ~/gpu0.lock env MAXBATCH={v} {ROOT}/scripts/real_trace_run.sh sweep_mbt{v} {WORKLOAD}", cwd=ROOT)
        print(v, "exit", r.returncode, flush=True)


def cmd_sims(a):
    def one(v):
        n, d = f"sweep_mbt{v}", base_dir(f"sweep_mbt{v}")
        # augur as a user runs it (shared-KV decode term on), kept as augur_skv.csv; then the same-price run of
        # run.sh, with every simulator's results compared by compare.py
        run_sh(n, real_dir(v), WORKLOAD, "prepare augur", {"SHARED_KV": "on"})
        sh(f"mv {d}/augur.csv {d}/augur_skv.csv")
        t, rc = run_sh(n, real_dir(v), WORKLOAD, "prepare augur blis frontier vidur compare")
        return v, t, rc
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        for v, t, rc in ex.map(one, a.values):
            print(v, f"{t:.0f}s", "exit", rc, flush=True)


def kendall(x, y):
    """Kendall tau-a over all pairs (no ties in these metrics)."""
    s = [np.sign(x[i] - x[j]) * np.sign(y[i] - y[j]) for i, j in itertools.combinations(range(len(x)), 2)]
    return float(np.mean(s))


def cmd_table(a):
    global SIMS
    res = {}
    for v in a.values:
        r = json.load(open(f"{base_dir(f'sweep_mbt{v}')}/results.json"))[0]["sims"]
        res[v] = {k: r[k]["abs"]["e2e"][0] for k in ("real", *SIMS) if k in r}
        # augur as a user runs it: the client-side latency column of its output (same measure as compare's augur row)
        res[v]["augur_skv"] = float(np.mean([float(x["latency"]) for x in csv.DictReader(
            open(f"{base_dir(f'sweep_mbt{v}')}/augur_skv.csv"))])) / 1e9
        res[v]["augur_same_price"] = res[v].pop("augur")
    SIMS = ("augur_skv", "augur_same_price", "blis", "frontier", "vidur")
    sims = ("real", *SIMS)
    print("| max-num-batched-tokens | " + " | ".join(sims) + " |\n|" + " --- |" * (len(sims) + 1))
    for v in a.values:
        print(f"| {v} | " + " | ".join(f"{res[v][s]:.1f}" for s in sims) + " |")
    real = [res[a.values[i]]["real"] for i in range(len(a.values))]
    best = int(np.argmin(real))
    print("\n| simulator | Kendall tau | picks | real best | regret % |\n| --- | --- | --- | --- | --- |")
    for s in SIMS:
        y = [res[v][s] for v in a.values]
        p = int(np.argmin(y))
        print(f"| {s} | {kendall(real, y):+.2f} | {a.values[p]} | {a.values[best]} | "
              f"{100 * (real[p] / real[best] - 1):.1f} |")
    print("\nrank order (best first):", {s: [a.values[i] for i in np.argsort([res[v][s] for v in a.values])] for s in sims})


def cmd_throughput(a):
    """budgets x rates configurations; for each, augur's run and BLIS's run timed on their own (4 at a time,
    pinned to cores 4-7). augur reads the workload as it is (its time includes run.sh's writing of the other
    simulators' inputs, a few seconds); BLIS also needs ``prepare`` (its input conversion and prefix groups), counted."""
    ref = next(real_dir(v) for v in (2048, *VALUES) if os.path.exists(real_dir(v) + "/client.json"))  # its KV block count and step log
    budgets = np.unique(np.geomspace(256, 8192, a.budgets).round(-1 if a.budgets > 6 else 0).astype(int)).tolist()
    rates = np.linspace(0.5, 1.5, a.rates).tolist()  # multiples of the trace's 6 req/s
    tmp = os.path.join(ROOT, "out/baselines/sweep_tp")
    os.makedirs(tmp, exist_ok=True)
    wls, prep = {}, {}
    for r in rates:  # arrivals / r: the same prompts, sent r times as fast
        wls[r] = f"{tmp}/rate{r:.2f}.jsonl"
        with open(WORKLOAD) as f, open(wls[r], "w") as g:
            for line in f:
                e = json.loads(line)
                e["arrival_time_ns"] = int(e["arrival_time_ns"] / r)
                g.write(json.dumps(e) + "\n")
    cfgs = [(b, r) for r in rates for b in budgets]
    t_all = time.time()
    def one(c):
        b, r = c
        name = f"sweep_tp/b{b}_r{r:.2f}"
        env = {"MAX_BATCHED_TOKENS": str(b)}
        tp, _ = run_sh(name, ref, wls[r], "prepare", env, pin=True)
        ta, rca = run_sh(name, ref, wls[r], "augur", {**env, "SHARED_KV": "on"}, pin=True)
        tb, rcb = run_sh(name, ref, wls[r], "blis", env, pin=True)
        sh(f"rm -rf {base_dir(name)}")
        return {"budget": b, "rate": r, "prepare_s": tp, "augur_s": ta, "blis_s": tb, "rc": [rca, rcb]}
    with cf.ThreadPoolExecutor(a.parallel) as ex:
        out = list(ex.map(one, cfgs))
    wall = time.time() - t_all
    json.dump(out, open(f"{tmp}/throughput.json", "w"), indent=1)
    n = len(out)
    for k in ("prepare_s", "augur_s", "blis_s"):
        print(k, f"mean {np.mean([o[k] for o in out]):.1f} s")
    for k, cost in (("augur", "augur_s"), ("BLIS (prepare + run)", None)):
        t = np.mean([o["augur_s"] if cost else o["prepare_s"] + o["blis_s"] for o in out])
        print(f"{k}: {t:.1f} s per configuration, {a.parallel * 3600 / t:.0f} configurations per hour at {a.parallel} at a time")
    print(f"{n} configs, wall {wall:.0f} s with {a.parallel} at a time; failures {sum(any(o['rc']) for o in out)}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for n, f in (("real", cmd_real), ("sims", cmd_sims), ("table", cmd_table), ("throughput", cmd_throughput)):
        s = sub.add_parser(n)
        s.add_argument("--values", type=int, nargs="+", default=VALUES)
        s.set_defaults(f=f)
    sub.choices["sims"].add_argument("--jobs", type=int, default=2)
    sub.choices["throughput"].add_argument("--budgets", type=int, default=10)
    sub.choices["throughput"].add_argument("--rates", type=int, default=10)
    sub.choices["throughput"].add_argument("--parallel", type=int, default=4)
    a = p.parse_args(argv)
    a.f(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
