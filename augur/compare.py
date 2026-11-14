"""Compare two request CSVs in the fork's format (e.g. augur vs LLMServingSim).

Requests are matched on 'request id'. For the fork, ids are assigned in
release order; augur assigns them the same way, so flat workloads match
directly. For agentic workloads the fork numbers calls as it releases them,
which can differ between simulators once timing differs, so sessions are
matched on (arrival order of the session, call index) when both files carry
the extra columns, else by id.

Usage: python -m augur.compare A.csv B.csv
"""

from __future__ import annotations

import csv
import sys

import numpy as np

METRICS = ["TTFT", "TPOT", "latency", "end_time", "queuing_delay"]


def load(path: str) -> dict[int, dict]:
    with open(path) as f:
        return {int(r["request id"]): r for r in csv.DictReader(f)}


def align(a: dict[int, dict], b: dict[int, dict]) -> tuple[dict[int, dict], dict[int, dict]]:
    """Pair calls by (input, output) length and arrival order when ids disagree.

    Agentic workloads number calls in release order, which differs between
    simulators once their timings differ. Calls are then paired by their
    (input, output) lengths, the k-th such call by arrival in A with the k-th
    in B.
    """
    ids = set(a) & set(b)
    if all(a[i]["input"] == b[i]["input"] and a[i]["output"] == b[i]["output"] for i in ids):
        return a, b
    def groups(d):
        g: dict[tuple, list] = {}
        for r in sorted(d.values(), key=lambda r: int(r["arrival"])):
            g.setdefault((r["input"], r["output"]), []).append(r)
        return g
    ga, gb = groups(a), groups(b)
    na, nb, k = {}, {}, 0
    for key in ga.keys() & gb.keys():
        for x, y in zip(ga[key], gb[key]):
            na[k], nb[k] = x, y
            k += 1
    return na, nb


def summarize(a: dict[int, dict], b: dict[int, dict]) -> dict:
    a, b = align(a, b)
    ids = sorted(set(a) & set(b))
    out = {"matched": len(ids), "only_a": len(set(a) - set(b)), "only_b": len(set(b) - set(a))}
    for m in METRICS:
        x = np.array([float(a[i][m]) for i in ids])
        y = np.array([float(b[i][m]) for i in ids])
        denom = np.maximum(np.abs(y), 1.0)
        rel = np.abs(x - y) / denom
        out[m] = {
            "mean_a_ms": x.mean() / 1e6, "mean_b_ms": y.mean() / 1e6,
            "mean_rel_diff": float(np.abs(x.mean() - y.mean()) / max(abs(y.mean()), 1.0)),
            "per_req_median_rel_diff": float(np.median(rel)), "per_req_p99_rel_diff": float(np.percentile(rel, 99)),
            "exact": int((x == y).sum()),
        }
    out["output_tokens_equal"] = all(a[i]["output"] == b[i]["output"] for i in ids)
    return out


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    a, b = load(argv[0]), load(argv[1])
    s = summarize(a, b)
    print(f"matched {s['matched']} (only A {s['only_a']}, only B {s['only_b']}); "
          f"output lengths equal: {s['output_tokens_equal']}")
    print(f"{'metric':<14}{'mean A ms':>12}{'mean B ms':>12}{'mean diff':>11}{'req p50':>9}{'req p99':>9}{'exact':>7}")
    for m in METRICS:
        r = s[m]
        print(f"{m:<14}{r['mean_a_ms']:>12.2f}{r['mean_b_ms']:>12.2f}{100 * r['mean_rel_diff']:>10.2f}%"
              f"{100 * r['per_req_median_rel_diff']:>8.2f}%{100 * r['per_req_p99_rel_diff']:>8.2f}%{r['exact']:>7}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
