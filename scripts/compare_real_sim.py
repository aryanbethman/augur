"""Real run vs simulator run: mean/median TTFT, TPOT, latency and prefix-hit rate, plus per-call errors.

usage: python scripts/compare_real_sim.py REAL.csv SIM.csv [SIM2.csv ...] [--labels real,augur,lss]
Both CSVs are in the fork format (the client writes the real one). Prefix-hit rate is sum(cached_tokens)/sum(input)
when the CSV has cached_tokens. Per-call errors use augur.verify.compare's alignment (session id + call index).
"""
import argparse

import numpy as np

from augur.verify.compare import align, load
from augur.verify.replay import read_log

METRICS = ("TTFT", "TPOT", "latency")


def col(rows, m):
    return np.array([float(r[m]) for r in rows]) / 1e6  # ns -> ms


def stats(rows):
    out = {}
    for m in METRICS:
        x = col(rows, m)
        out[m] = (float(x.mean()), float(np.median(x)))
    if rows and rows[0].get("cached_tokens", "") != "":
        inp = sum(float(r["input"]) for r in rows)
        out["hit"] = sum(float(r["cached_tokens"]) for r in rows) / inp if inp else 0.0
    return out


def log_hit_rate(path):
    """Prefix-hit rate from a step log: computed tokens at each request's first schedule / its prompt tokens."""
    ev = read_log(path)
    prompt = {e["rid"]: e["prompt"] for e in ev if e["ev"] == "add"}
    hit = {}
    for e in ev:
        if e["ev"] == "sched":
            for r, c in e["new"].items():
                hit.setdefault(r, c)
    tot = sum(prompt.values())
    return sum(hit.values()) / tot if tot else 0.0


def pair(real, sim):
    """Pair calls: by request id when lengths agree (the real client and augur number calls alike), else compare.align."""
    ids = sorted(set(real) & set(sim))
    if ids and all(real[i]["input"] == sim[i]["input"] and real[i]["output"] == sim[i]["output"] for i in ids):
        return real, sim
    return align(real, sim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("real")
    ap.add_argument("sims", nargs="+")
    ap.add_argument("--real-log", help="real run's step log: gives its prefix-hit rate (the client CSV has none)")
    ap.add_argument("--labels", default="", help="comma list: real label then one per sim")
    a = ap.parse_args()
    labels = a.labels.split(",") if a.labels else ["real"] + [f"sim{i}" for i in range(len(a.sims))]
    real = load(a.real)
    sr = stats(list(real.values()))
    if a.real_log:
        sr["hit"] = log_hit_rate(a.real_log)
    print(f"{'':10}{'TTFT mean':>11}{'median':>9}{'TPOT mean':>11}{'median':>9}{'lat mean':>10}{'median':>9}{'hit':>7}")

    def line(lbl, s):
        h = f"{100 * s['hit']:.1f}%" if "hit" in s else "n/a"
        print(f"{lbl:10}" + "".join(f"{s[m][0]:>{w}.1f}{s[m][1]:>{v}.1f}" for m, w, v in
                                    (("TTFT", 11, 9), ("TPOT", 11, 9), ("latency", 10, 9))) + f"{h:>7}")

    line(labels[0], sr)
    for lbl, p in zip(labels[1:], a.sims):
        sim = load(p)
        s = stats(list(sim.values()))
        line(lbl, s)
        ra, sa = pair(real, sim)
        ids = sorted(set(ra) & set(sa))
        err = []
        for m in METRICS:
            x = np.array([float(sa[i][m]) for i in ids])
            y = np.array([float(ra[i][m]) for i in ids])
            rel = np.abs(x - y) / np.maximum(np.abs(y), 1.0)
            err.append(f"{m}: mean {100 * (s[m][0] / sr[m][0] - 1):+.1f}% median {100 * (s[m][1] / sr[m][1] - 1):+.1f}% "
                       f"per-call median |err| {100 * np.median(rel):.1f}% p90 {100 * np.percentile(rel, 90):.1f}%")
        print(f"  {lbl} vs {labels[0]} ({len(ids)} calls): " + " | ".join(err))


if __name__ == "__main__":
    main()
