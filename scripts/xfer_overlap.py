"""Did offload transfers that overlapped another transfer run slower? From a real step log's xfer fields.

usage: python scripts/xfer_overlap.py STEPLOG.jsonl [--out FILE.json]
Each update event may carry xfer = {load|store: [bytes, GPU-timed seconds, jobs]} for the transfers that finished in it.
A record with both directions non-empty finished its load and its store in the same step (they ran on separate streams and
overlapped at least partly). Per direction and per size bucket (log2 of mean bytes per job), compare GB/s of records where
the other direction was also active with those where it was not (single-job records only, so bytes/seconds is one transfer).
Also reported: how GB/s depends on decode batch occupancy is NOT available here; GPU kernels' effect is in contention_bench.py.
"""
import json
import statistics
import sys

import numpy as np


def main(path, out=None):
    recs = []
    for line in open(path):
        e = json.loads(line)
        x = e.get("xfer")
        if e["ev"] == "upd" and x:
            recs.append(x)
    res = {"records": len(recs)}
    print(f"{len(recs)} update events with transfer stats")
    for d, o in (("load", "store"), ("store", "load")):
        rows = [(x[d][0], x[d][1], x[o][0] > 0) for x in recs if x[d][2] == 1 and x[d][1] > 0]
        tot_b = sum(x[d][0] for x in recs); tot_s = sum(x[d][1] for x in recs)
        res[d] = {"single_job_records": len(rows), "overall_gbps": tot_b / tot_s / 1e9, "buckets": []}
        print(f"\n{d}: overall {tot_b / tot_s / 1e9:.2f} GB/s over {tot_b / 1e9:.1f} GB; "
              f"{len(rows)} single-transfer records, {sum(r[2] for r in rows)} overlapped with {o}")
        print(f"  {'size':>9}  {'alone n':>8}{'GB/s':>8}  {'overlap n':>10}{'GB/s':>8}  {'ratio':>6}")
        edges = [0, 1 << 20, 4 << 20, 16 << 20, 64 << 20, 1 << 40]
        for lo, hi in zip(edges, edges[1:]):
            a = [b / s / 1e9 for b, s, ov in rows if lo <= b < hi and not ov]
            c = [b / s / 1e9 for b, s, ov in rows if lo <= b < hi and ov]
            if not a and not c:
                continue
            ma = statistics.median(a) if a else float("nan"); mc = statistics.median(c) if c else float("nan")
            lab = f"{lo >> 20}-{hi >> 20 if hi < 1 << 40 else 'inf'}MB"
            res[d]["buckets"].append({"size": lab, "alone_n": len(a), "alone_gbps": ma, "overlap_n": len(c), "overlap_gbps": mc})
            print(f"  {lab:>9}  {len(a):>8}{ma:>8.2f}  {len(c):>10}{mc:>8.2f}  {mc / ma if a and c else float('nan'):>6.2f}")
    if out:
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None)
