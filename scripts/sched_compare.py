"""Scheduler policies, real vLLM vs augur: mean / p90 TTFT and SLO attainment, and whether augur ranks them as real does.

usage: python scripts/sched_compare.py DIR... [--json FILE]
Each DIR is one policy: client.csv (real run, optional) and augur.csv (augur's run). The SLO is the one
examples/scheduler_policies.py's DeadlineFirst orders by: a request meets it when
TTFT <= SLO_BASE_S + SLO_PER_TOKEN_S * prompt tokens. Lower TTFT and higher attainment are better.

How the runs were made (ShareGPT-300 with arrival times x4, i.e. 2.5 requests/s; one policy per DIR):
  real:  ASYNC=1 flock ~/gpu0.lock scripts/real_trace_run.sh sched_NAME WORKLOAD "" --scheduler-cls MODULE.LoggingX
         where MODULE is a file with LoggingX = augur.verify.hooks.with_step_log(examples.scheduler_policies.XAsyncScheduler)
         (a real server needs a dotted path to the class that also writes the step log); FCFS takes no --scheduler-cls
  augur: python -m augur.cli --model Qwen/Qwen3-1.7B --hardware RTX4060Ti --num-gpu-blocks <the real server's>
         --max-model-len 8192 --async-scheduling --no-recorded-outputs --filler-vocab 150000 --workload WORKLOAD
         --scheduler-cls examples.scheduler_policies.XAsyncScheduler [--replay-durations steps.*.jsonl
         --replay-tokens tokens.jsonl --step-log-vllm PREFIX, then python -m augur.verify.replay compare REAL AUGUR]
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from scheduler_policies import SLO_BASE_S, SLO_PER_TOKEN_S  # noqa: E402

from augur.verify.compare import load  # noqa: E402


def stats(path):
    rows = list(load(path).values())
    t = np.array([float(r["TTFT"]) for r in rows]) / 1e9
    slo = SLO_BASE_S + SLO_PER_TOKEN_S * np.array([float(r["input"]) for r in rows])
    return {"mean": t.mean(), "p90": np.percentile(t, 90), "slo": float((t <= slo).mean())}


def rank(res, who, key):
    return sorted(res, key=lambda p: res[p][who][key], reverse=key == "slo")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--json")
    a = ap.parse_args()
    res = {}
    for d in a.dirs:
        r = {"augur": stats(os.path.join(d, "augur.csv"))}
        if os.path.exists(os.path.join(d, "client.csv")):
            r["real"] = stats(os.path.join(d, "client.csv"))
        rep = os.path.join(d.rstrip("/") + "_r2", "client.csv")  # a second real run of the policy, beside DIR
        if os.path.exists(rep):
            r["real_r2"] = stats(rep)
        res[os.path.basename(d.rstrip("/"))] = r
    print(f"{'policy':26}{'':7}{'mean TTFT s':>12}{'p90 s':>9}{'SLO met':>9}")
    for p, r in res.items():
        for who, x in r.items():  # real, augur, real_r2
            print(f"{p:26}{who:7}{x['mean']:12.2f}{x['p90']:9.2f}{100 * x['slo']:8.1f}%")
    for who in ("real", "augur"):  # (the second real run, real_r2, is reported below)
        if all(who in r for r in res.values()):
            for key in ("mean", "p90", "slo"):
                print(f"rank by {key:5} {who:6}", " > ".join(rank(res, who, key)))
    reps = {p: r for p, r in res.items() if "real_r2" in r}
    if reps:
        print("run-to-run noise of the real runs (spread = |run 1 - run 2| / mean of the two)")
        for p, r in reps.items():
            print(f"{p:26}" + "".join(f"  {k} {abs(r["real"][k] - r["real_r2"][k]) / ((r["real"][k] + r["real_r2"][k]) / 2) * 100:.1f}%"
                                      for k in ("mean", "p90", "slo")))
        for key in ("mean", "p90", "slo"):
            r2 = {p: {"real": r["real_r2"]} for p, r in reps.items()}
            print(f"rank by {key:5} real r2", " > ".join(rank(r2, "real", key)))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
