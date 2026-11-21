"""What does the speculative-decoding step cost beyond the profile's price? Regress (measured - predicted) of a real
run's decode steps on the batch: requests, summed context tokens, scheduled tokens.

usage: python scripts/spec_gap_fit.py STEPLOG.jsonl STEPCHECK.csv
STEPCHECK.csv: `python -m augur.verify.stepcheck STEPLOG ... --csv` (one row per scheduled step, same order).
Only pure decode steps (every request is a decode) are used, where the gap is the drafting and verification overhead.
"""
import csv
import sys

import numpy as np

from augur.verify.replay import read_log, steps
from augur.verify.stepcheck import MAX_DRAFTS


def main(log, csv_path):
    ev = read_log(log)
    prompt = {e["rid"]: e["prompt"] for e in ev if e["ev"] == "add"}
    rows = list(csv.DictReader(open(csv_path)))
    computed, feats = {}, []
    for s, u in steps(ev):
        nreq = ntok = ctx = 0
        alldec = True
        for rid, n in s["tok"].items():
            if rid in s["new"]:
                computed[rid] = s["new"][rid]
            before = computed.get(rid, 0)
            em = u["out"].get(rid, 0)
            dec = before >= prompt.get(rid, 0) and em >= 1 and n <= 1 + MAX_DRAFTS
            alldec &= dec or n == 1
            nreq += 1; ntok += n; ctx += before
            computed[rid] = before + (em if dec and n > 1 else n)
        for rid in s["pre"]:
            computed[rid] = 0
        feats.append((nreq, ntok, ctx, alldec))
    assert len(feats) == len(rows), (len(feats), len(rows))
    X, y = [], []
    for (nreq, ntok, ctx, dec), r in zip(feats, rows):
        if dec and r["class"] == "decode":
            X.append((1, nreq, ctx / 1e3)); y.append((float(r["measured_ns"]) - float(r["predicted_ns"])) / 1e6)
    X, y = np.array(X, float), np.array(y)
    for name, cols in (("per request", [0, 1]), ("per 1k context tokens", [0, 2]), ("requests + context", [0, 1, 2])):
        beta, *_ = np.linalg.lstsq(X[:, cols], y, rcond=None)
        pred = X[:, cols] @ beta
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        lab = dict(zip(["intercept ms", "us/request", "us per context token"], [beta[0]] + [b * 1e3 if i == 1 else b * 1e3 / 1e3 * 1e3 for i, b in enumerate(beta[1:], 1)]))
        print(f"{name:24} R2 {r2:.3f}  intercept {beta[0]:+.2f} ms" + "".join(
            f"  {n} {b * 1e3:.1f} us" for n, b in zip((["per request"] if 1 in cols else []) + (["per 1k ctx"] if 2 in cols else []),
                                                     beta[1:])))
    print(f"{len(y)} decode steps; mean gap {y.mean():.1f} ms; mean requests {X[:,1].mean():.1f}; mean ctx {X[:,2].mean()*1e3:.0f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
