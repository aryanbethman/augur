"""Real runs vs augur for one CPU-tier policy each (scripts/real_offload_check.sh with POLICY=...): the numbers the
plugin case study compares, and whether augur ranks the policies as the real server did.

usage: python scripts/policy_compare.py OUT_DIR... [--bytes-per-token 114688] [--json FILE]
Each OUT_DIR holds client.csv (real), augur.csv (augur's normal run at the measured PCIe rate), steps.*.jsonl (real
step log) and augur_norm.*.jsonl (augur's). Per run: mean / p50 / p99 TTFT and latency (client side, ms), the share of
prompt tokens that were prefix hits (GPU or CPU tier), the share loaded from the CPU tier, and bytes moved each way.
Lower is better for TTFT and latency, higher for hit rates.
"""

import argparse
import glob
import json
import os

import numpy as np

from augur.verify.compare import load
from augur.verify.replay import read_log


def ms(rows, col):
    x = np.array([float(r[col]) for r in rows]) / 1e6
    return {"mean": float(x.mean()), "p50": float(np.median(x)), "p99": float(np.percentile(x, 99))}


def from_log(path, bytes_per_token):
    ev = read_log(path)
    prompt = {e["rid"]: e["prompt"] for e in ev if e["ev"] == "add"}
    computed = {}
    xfer = {"load": 0, "store": 0}
    for e in ev:
        if e["ev"] == "sched":
            for r, c in e["new"].items():
                computed.setdefault(r, c)
        for d, (b, _s, *_) in (e.get("xfer") or {}).items():
                xfer[d] += b
    tot = sum(prompt.values())
    return {"hit": sum(computed.values()) / tot, "cpu_hit": xfer["load"] / bytes_per_token / tot,
            "gb_loaded": xfer["load"] / 1e9, "gb_stored": xfer["store"] / 1e9}


def summarize(d, bytes_per_token):
    out = {}
    for who, csv_name, pat in (("real", "client.csv", "steps.*.jsonl"), ("augur", "augur.csv", "augur_norm.*.jsonl")):
        rows = list(load(os.path.join(d, csv_name)).values())
        r = {"ttft": ms(rows, "TTFT"), "latency": ms(rows, "latency")}
        logs = glob.glob(os.path.join(d, pat))
        if logs:
            r.update(from_log(logs[0], bytes_per_token))
        out[who] = r
    return out


def ranking(res, who, key):
    """Policies from best to worst; for hit rates the highest first."""
    reverse = key in ("hit", "cpu_hit")
    f = (lambda p: res[p][who][key]) if key in ("hit", "cpu_hit") else (lambda p: res[p][who][key[0]][key[1]])
    return sorted(res, key=f, reverse=reverse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--bytes-per-token", type=float, default=114688)  # Qwen3-1.7B: 28 layers x 2 x 8 x 128 x 2 bytes
    ap.add_argument("--json")
    a = ap.parse_args()
    res = {os.path.basename(d.rstrip("/")).split("_", 2)[2].replace("_kv9", ""): summarize(d, a.bytes_per_token)
           for d in a.dirs}  # swe50_offload<GIB>_<POLICY>[_kv9]
    print(f"{'policy':22}{'':7}{'TTFT mean':>10}{'p50':>9}{'p99':>9}{'lat mean':>10}{'p50':>9}{'p99':>9}"
          f"{'hit':>7}{'cpu hit':>8}{'GB load':>8}{'GB store':>9}")
    for p, r in res.items():
        for who in ("real", "augur"):
            x = r[who]
            print(f"{p:22}{who:7}{x['ttft']['mean']:10.0f}{x['ttft']['p50']:9.0f}{x['ttft']['p99']:9.0f}"
                  f"{x['latency']['mean']:10.0f}{x['latency']['p50']:9.0f}{x['latency']['p99']:9.0f}"
                  + (f"{100 * x['hit']:6.1f}%{100 * x['cpu_hit']:7.1f}%{x['gb_loaded']:8.1f}{x['gb_stored']:9.1f}"
                     if "hit" in x else ""))
        e = {k: 100 * (r["augur"]["ttft" if k == "ttft" else "latency"]["mean"]
                       / r["real"]["ttft" if k == "ttft" else "latency"]["mean"] - 1) for k in ("ttft", "latency")}
        print(f"{'':22}{'error':7}{e['ttft']:+9.1f}%{'':27}{e['latency']:+9.1f}%")
    print()
    for name, key in (("mean TTFT", ("ttft", "mean")), ("p99 TTFT", ("ttft", "p99")), ("mean latency", ("latency", "mean")),
                      ("p99 latency", ("latency", "p99")), ("prefix hit rate", "hit"), ("CPU-tier hit rate", "cpu_hit")):
        try:
            real, aug = ranking(res, "real", key), ranking(res, "augur", key)
        except KeyError:
            continue
        print(f"{name:18} real: {' < '.join(real) if key not in ('hit', 'cpu_hit') else ' > '.join(real)}"
              f"   augur: {' < '.join(aug) if key not in ('hit', 'cpu_hit') else ' > '.join(aug)}"
              f"   {'SAME' if real == aug else 'DIFFERENT'}")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
