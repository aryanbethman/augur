"""Intake delay (client send -> engine arrival) vs prompt length, from scripts/frontend_intake.sh runs.

usage: python scripts/frontend_intake.py OUTDIR
Each result file len<L>_<k>.json has the client's send times (time.perf_counter, the step log's clock); the step
log's add events carry the request ids (cmpl-bench-<run>-<i>-...). Intake is arrival - send. A request that found
the engine busy also waits for the engine loop's next iteration (simulated separately by augur), so the fit uses
requests that found the engine idle (nothing in flight at send time): there intake is the API server's own work.
Fits intake_ms = a + b * prompt_tokens (least squares on idle-arrival requests; also on the per-length medians).
"""
import glob
import json
import re
import sys

import numpy as np

from augur.verify.replay import read_log

RID = re.compile(r"bench-([0-9a-f]{8})-(\d+)-")


def main(d):
    ev = read_log(glob.glob(f"{d}/steps.*.jsonl")[0])
    add = {}
    for e in ev:
        if e["ev"] == "add":
            m = RID.search(e["rid"])
            add[(m.group(1), int(m.group(2)))] = e["t"] / 1e9
    fin = np.sort([e["t"] / 1e9 for e in ev if e["ev"] == "upd" for _ in e["fin"]])
    adds = np.sort(list(add.values()))
    rows, res = [], {}
    for f in sorted(glob.glob(f"{d}/len*_*.json")):
        r = json.load(open(f))
        n = len(r["ttfts"])
        run = None
        for (h, i) in add:
            if i == 0 and abs(add[(h, 0)] - r["start_times"][0]) < 1.0:
                run = h
        if run is None:
            print("no match for", f); continue
        send = np.array(r["start_times"]); inl = np.array(r["input_lens"])
        arr = np.array([add[(run, i)] for i in range(n)])
        intake = (arr - send) * 1e3
        in_eng = np.searchsorted(adds, send, "right") - np.searchsorted(fin, send, "right")
        idle = in_eng == 0
        name = f.split("/")[-1][:-5]
        for i in range(n):
            rows.append((name, int(inl[i]), float(intake[i]), bool(idle[i])))
        q = lambda x, p: float(np.percentile(x, p)) if len(x) else None
        res[name] = dict(n=n, mean_input_tokens=float(inl.mean()), idle_n=int(idle.sum()),
                         idle_median_ms=q(intake[idle], 50), idle_mean_ms=float(intake[idle].mean()) if idle.any() else None,
                         idle_p10_ms=q(intake[idle], 10), idle_p90_ms=q(intake[idle], 90),
                         all_median_ms=q(intake, 50), all_mean_ms=float(intake.mean()), all_min_ms=float(intake.min()))
        print(f"{name:12} tokens {inl.mean():7.0f}  idle {idle.sum():3d}/{n}: median {res[name]['idle_median_ms']:.2f} "
              f"mean {res[name]['idle_mean_ms']:.2f} p10 {res[name]['idle_p10_ms']:.2f} p90 {res[name]['idle_p90_ms']:.2f} ms | "
              f"all: median {res[name]['all_median_ms']:.2f} mean {res[name]['all_mean_ms']:.2f} min {res[name]['all_min_ms']:.2f}")
    x = np.array([r[1] for r in rows if r[3]], float); y = np.array([r[2] for r in rows if r[3]])
    b, a = np.polyfit(x, y, 1)
    res["fit_idle"] = dict(a_ms=float(a), b_us_per_token=float(b * 1e3), over_256_to_4096_ms=float(b * (4096 - 256)))
    xa = np.array([r[1] for r in rows], float); ya = np.array([r[2] for r in rows])
    bb, aa = np.polyfit(xa, ya, 1)
    res["fit_all"] = dict(a_ms=float(aa), b_us_per_token=float(bb * 1e3), over_256_to_4096_ms=float(bb * (4096 - 256)))
    # bootstrap the slope of the idle fit
    rng = np.random.default_rng(0)
    sl = [np.polyfit(x[i], y[i], 1)[0] * 1e3 for i in (rng.integers(0, len(x), len(x)) for _ in range(500))]
    res["fit_idle"]["b_us_per_token_ci95"] = [float(np.percentile(sl, 2.5)), float(np.percentile(sl, 97.5))]
    print("fit idle:", res["fit_idle"]); print("fit all :", res["fit_all"])
    json.dump(res, open(f"{d}/intake.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
