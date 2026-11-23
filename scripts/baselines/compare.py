"""Compare each simulator's run with the real vLLM run (and with augur) on the same workload.

    python scripts/baselines/compare.py out/baselines/sharegpt300 out/baselines/swe50 ... [--md out/baselines/results.md]

Each directory is one run (see run.sh): ``requests.json`` and ``meta.json`` from workloads.py, and
what each simulator wrote there. Rows are the calls in arrival order, the same in every simulator.

Per-request metrics: TTFT is the first token,
E2E the last, TPOT (E2E - TTFT) / (output tokens - 1). Real and augur numbers come from the step log
(a token exists when the step that produced it is updated); BLIS, Frontier and Vidur report them in
their own output. The runs with their own arrival process (augur's normal run, and BLIS, Frontier and Vidur given
the same trace times) are compared as the real client measured the real run (client.csv, or the JSON of vllm bench
serve): from the moment the request was sent, so a wait before the engine counts, with the front end (3.4 ms + 3 us per
prompt token) and the delivery (8 ms) that augur's normal run uses added to a simulator's engine-side numbers. The
``_replay`` rows are given the real run's arrival at the engine and compared with the real engine's own numbers (the
step log's add and update). For each statistic (mean, p50, p90, p99 over the requests) the table gives the
relative difference of a simulator's value from the reference's, in percent: ``vs real`` and ``vs
augur``. augur runs the real scheduler code on the same pricer, so ``vs augur`` isolates what a
re-implemented scheduler changes. ``|e|`` is the median over requests of the absolute relative error
of the request's own E2E against the reference's.

``<sim>_replay`` rows are the simulator run again with step k taking what step k took in the real run (and the
same arrivals): there the scheduler alone decides, and an exact copy of vLLM's reproduces the real steps.

Decisions are compared step by step against the real run's step log, and against augur's (``steps.<pid>.jsonl``) where a
simulator has per-step batches (augur's step log; the batch each simulator's pricer was asked to price
for BLIS, Frontier and Vidur). A step is the set of (request, tokens scheduled). ``same`` is the share
of the real run's steps that the simulator's step at the same position equals; ``first`` the position
(1-based) of the first step that differs, and ``prefix`` that position as a share of the real steps;
``found`` the share of real steps that occur anywhere in the simulator's steps (a count of equal
steps, which survives a divergence that merely shifts the sequence). ``pre`` is the number of
preemptions (a request evicted and recomputed): the real log's, augur's own, BLIS's report and
Frontier's and Vidur's restart counts. Against the real steps even augur diverges early: the pricer's step
times are not the real ones, so a request reaches the engine in a different step; read the simulators'
numbers against augur's, and against augur's own steps (same arrivals, same pricer, so a difference there is the
scheduler).
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import re
import sys

import numpy as np

STATS = (("mean", None), ("p50", 50), ("p90", 90), ("p99", 99))


def _stat(x, q):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if not len(x):
        return float("nan")
    return float(x.mean()) if q is None else float(np.percentile(x, q))


def _metrics(ttft, e2e, out):
    """Per-request arrays (seconds): ttft, tpot, e2e."""
    tpot = [(e - t) / (o - 1) if o and o > 1 and e is not None and t is not None else None
            for t, e, o in zip(ttft, e2e, out)]
    return {"ttft": ttft, "tpot": tpot, "e2e": e2e}


def _from_step_log(path, rows):
    """Metrics and step list of a step log in augur.verify.hooks format (real vLLM, or augur)."""
    from augur.trace.workload import call_id
    from augur.verify.replay import read_log, scheduled

    ev = read_log(path)
    idx = {r["call_id"]: i for i, r in enumerate(rows)}
    add, first, fin, out = {}, {}, {}, collections.Counter()
    rid_row = {}
    for e in ev:
        if e["ev"] == "add" and not e["rid"].endswith("@d"):
            i = idx.get(call_id(e["rid"]))
            if i is not None:
                add[i] = e["t"]
                rid_row[e["rid"]] = i
        elif e["ev"] == "upd":
            for rid, n in e["out"].items():
                i = rid_row.get(rid)
                if i is not None and n:
                    out[i] += n
                    first.setdefault(i, e["t"])
            for rid in e["fin"]:
                i = rid_row.get(rid)
                if i is not None:
                    fin[i] = e["t"]
    n = len(rows)
    ttft = [(first[i] - add[i]) / 1e9 if i in first else None for i in range(n)]
    e2e = [(fin[i] - add[i]) / 1e9 if i in fin else None for i in range(n)]
    steps = [{rid_row[r]: t for r, t in s["tok"].items() if r in rid_row} for s in scheduled(ev)]
    pre = sum(len(e["pre"]) for e in ev if e["ev"] == "sched")
    return _metrics(ttft, e2e, [out[i] for i in range(n)]), steps, pre


def _pre_events_log(path, rows):
    """(step, row) of every preemption in a step log: step is the 1-based position of the step the schedule starts."""
    from augur.trace.workload import call_id
    from augur.verify.replay import read_log

    ev = read_log(path)
    idx = {r["call_id"]: i for i, r in enumerate(rows)}
    rid_row = {e["rid"]: idx.get(call_id(e["rid"])) for e in ev if e["ev"] == "add"}
    out, n = [], 0
    for e in ev:
        if e["ev"] == "sched":
            out += [(n + 1, rid_row.get(r)) for r in e["pre"]]
            n += bool(e["tok"])
    return out


def _pre_events_steps(path, idx):
    """The same from a baseline's step file (BLIS writes the preempted request ids with each batch)."""
    if not os.path.exists(path):
        return None
    out = []
    for k, l in enumerate(open(path)):
        for r in json.loads(l).get("pre") or ():
            out.append((k + 1, _row_of(r, idx)))
    return out


def _preempt(events, real_events):
    if events is None:
        return None
    rows_ = {r for _, r in events}
    real_rows = {r for _, r in real_events}
    return {"events": len(events), "requests": len(rows_), "first_step": min((s for s, _ in events), default=None),
            "real_requests_also": len(rows_ & real_rows) / len(real_rows) if real_rows else None,
            "same_step_and_request": len(set(events) & set(real_events)) / len(real_events) if real_events else None}


_SESSION_RID = re.compile(r"session_(.*)_round_(\d+)_\d+$")


def _row_of(rid, idx):
    """The row of a baseline's request id: ``request_<row>``, or BLIS's ``session_<id>_round_<n>_<k>`` (a session's call)."""
    m = _SESSION_RID.match(rid)
    return idx[f"{m.group(1)}#{m.group(2)}"] if m else int(rid.removeprefix("request_"))


def _steps_file(path, idx=None):
    if not os.path.exists(path):
        return None
    return [{(_row_of(k, idx) if idx else int(k)): v for k, v in json.loads(l)["tok"].items()}
            for l in open(path) if l.strip()]


def _load_blis(d, rows, v=""):
    p = os.path.join(d, f"blis_result{v}.json")
    if not os.path.exists(p):
        return None
    j = json.load(open(p))
    ttft, e2e, out = [None] * len(rows), [None] * len(rows), [0] * len(rows)
    idx = {r["call_id"]: i for i, r in enumerate(rows)}
    for r in j["request_rows"]:
        i = _row_of(r["id"], idx)
        if r["fate"] == "completed":
            ttft[i], e2e[i], out[i] = r["ttft_us"] / 1e6, r["e2e_us"] / 1e6, r["output_tokens"]
    pre = j["reports"]["engine"]["cluster"]["preemptions"]
    return _metrics(ttft, e2e, out), _steps_file(os.path.join(d, f"blis_steps{v}.jsonl"), idx), pre


def _load_csv_sim(d, rows, pattern, ttft_col, scale, pre_col):
    f = sorted(glob.glob(os.path.join(d, pattern)))
    if not f:
        return None
    ttft, e2e, out, pre = [None] * len(rows), [None] * len(rows), [0] * len(rows), 0
    for r in csv.DictReader(open(f[-1])):
        i = int(float(r["Request Id"]))
        ttft[i], e2e[i] = float(r[ttft_col]) * scale, float(r["request_e2e_time"]) * scale
        out[i] = rows[i]["output"]
        pre += int(float(r.get(pre_col, 0) or 0))
    return ttft, e2e, out, pre


def _load_frontier(d, rows, v=""):
    x = _load_csv_sim(d, rows, f"frontier_out{v}/*/*/r/request_metrics.csv", "ttft", 1e-3, "request_total_preemption_count")  # ms
    if x is None:
        return None
    ttft, e2e, out, pre = x
    return _metrics(ttft, e2e, out), _steps_file(os.path.join(d, f"frontier_steps{v}.jsonl")), pre


def _load_vidur(d, rows, v=""):
    x = _load_csv_sim(d, rows, f"vidur_out{v}/*/request_metrics.csv", "prefill_e2e_time", 1.0, "request_num_restarts")  # s
    if x is None:
        return None
    ttft, e2e, out, pre = x
    return _metrics(ttft, e2e, out), _steps_file(os.path.join(d, f"vidur_steps{v}.jsonl")), pre


def _rel(a, b):
    return 100.0 * (a - b) / b if b else float("nan")


def _stat_errors(sim, ref):
    return {f"{m}_{n}": _rel(_stat(sim[m], q), _stat(ref[m], q)) for m in ("ttft", "tpot", "e2e") for n, q in STATS}


def _abs_err(sim, ref):
    e = [abs(s - r) / r for s, r in zip(sim["e2e"], ref["e2e"]) if s is not None and r]
    return 100.0 * float(np.median(e)) if e else float("nan")


def _decisions(sim_steps, real_steps):
    if sim_steps is None:
        return None
    key = lambda s: frozenset(s.items())  # noqa: E731
    n = len(real_steps)
    same = sum(1 for a, b in zip(real_steps, sim_steps) if a == b)
    first = None
    for k, (a, b) in enumerate(zip(real_steps, sim_steps)):
        if a != b:
            first = k + 1
            break
    else:
        first = min(n, len(sim_steps)) + 1 if n != len(sim_steps) else None
    have = collections.Counter(key(s) for s in sim_steps)
    found = 0
    for s in real_steps:
        k = key(s)
        if have[k] > 0:
            have[k] -= 1
            found += 1
    return {"real_steps": n, "sim_steps": len(sim_steps), "same": same / n, "first": first,
            "prefix": ((first - 1) / n) if first else 1.0, "found": found / n}


def _real_client(meta, rows):
    """The real run's metrics as its client measured them: from the request's send (client.csv, or the JSON of
    ``vllm bench serve``), so a wait before the engine (a connection cap, the API server) counts."""
    idx = {r["call_id"]: i for i, r in enumerate(rows)}
    n = len(rows)
    ttft, e2e = [None] * n, [None] * n
    d = meta["run_dir"]
    if os.path.exists(os.path.join(d, "client.csv")):
        for r in csv.DictReader(open(os.path.join(d, "client.csv"))):
            sid = r.get("session_id")
            cid = f"{sid}#{r['sub_request_index']}" if sid and sid != "None" else f"req_{r['request id']}"
            i = idx.get(cid)
            if i is not None and r["latency"] and r["TTFT"]:
                ttft[i], e2e[i] = float(r["TTFT"]) / 1e9, float(r["latency"]) / 1e9
    elif os.path.exists(os.path.join(d, "real.json")):
        j = json.load(open(os.path.join(d, "real.json")))
        for k, (t, lat, err) in enumerate(zip(j["ttfts"], j["latencies"], j["errors"])):
            i = idx.get(f"req_{k}")
            if i is not None and not err:
                ttft[i], e2e[i] = t, lat
    else:
        return None
    return _metrics(ttft, e2e, [r["real"]["out"] if r["real"] else 0 for r in rows])


def _client_side(m, rows):
    """An engine-side metric set put on the client's side as augur's normal run does: the front end before the
    engine and the delivery after it (TTFT and E2E; TPOT is unchanged)."""
    add = [r["frontend_s"] + r["delivery_s"] for r in rows]
    return {"ttft": [None if t is None else t + a for t, a in zip(m["ttft"], add)], "tpot": m["tpot"],
            "e2e": [None if t is None else t + a for t, a in zip(m["e2e"], add)]}


def compare(d):
    rows = json.load(open(os.path.join(d, "requests.json")))["requests"]
    meta = json.load(open(os.path.join(d, "meta.json")))
    real_ttft = [r["real"]["first_s"] - r["real"]["add_s"] if r["real"] and r["real"]["first_s"] is not None else None
                 for r in rows]
    real_e2e = [r["real"]["fin_s"] - r["real"]["add_s"] if r["real"] and r["real"]["fin_s"] is not None else None
                for r in rows]
    real_out = [r["real"]["out"] if r["real"] else 0 for r in rows]
    runs = {"real": (_metrics(real_ttft, real_e2e, real_out), None, None)}
    _, real_steps, real_pre = _from_step_log(meta["step_log"], rows)
    runs["real"] = (runs["real"][0], real_steps, real_pre)
    af = sorted(glob.glob(os.path.join(d, "augur_steps.*.jsonl")))
    if af:
        runs["augur"] = _from_step_log(af[-1], rows)
    for v in ("", "_replay"):
        for name, load in (("blis", _load_blis), ("frontier", _load_frontier), ("vidur", _load_vidur)):
            x = load(d, rows, v)
            if x is not None:
                runs[name + v] = x
    spec = None
    if meta["spec_k"] and af:
        from workloads import _acceptance
        from augur.verify.replay import read_log
        got = _acceptance(read_log(af[-1]), meta["spec_k"])
        spec = {"real_accepted_per_step": meta["acceptance"]["accepted_per_step"], "spec_p": meta["acceptance"]["spec_p"],
                "augur_accepted_per_step": got["mean_tokens_per_step"] - 1}
    res = {"run": os.path.basename(d.rstrip("/")), "spec": spec, "requests": len(rows), "async": meta["async_scheduling"],
           "spec_k": meta["spec_k"], "offload_gib": meta["kv_offloading_gib"], "sims": {}}
    pre_ev = {"real": _pre_events_log(meta["step_log"], rows)}
    if af:
        pre_ev["augur"] = _pre_events_log(af[-1], rows)
    for v in ("", "_replay"):
        e = _pre_events_steps(os.path.join(d, f"blis_steps{v}.jsonl"), {r["call_id"]: i for i, r in enumerate(rows)})
        if e is not None:
            pre_ev["blis" + v] = e
    real_eng = runs["real"][0]
    real_cl = _real_client(meta, rows) or _client_side(real_eng, rows)
    cl = {n: (m if n.endswith("_replay") else _client_side(m, rows)) for n, (m, _, _) in runs.items()}
    for name, (m, steps, pre) in runs.items():
        replay = name.endswith("_replay")
        # a run with its own arrival process is compared with the real run as its client measured it; a replay (real
        # arrivals at the engine) with the engine's own numbers
        mm, ref = (m, real_eng) if replay else (cl[name], real_cl)
        if name == "real":
            mm = real_cl
        r = {"abs": {k: [_stat(mm[k], q) for _, q in STATS] for k in mm}, "pre": pre,
             "completed": sum(1 for v in m["e2e"] if v is not None)}
        if name != "real":
            r["vs_real"] = _stat_errors(mm, ref)
            r["abs_err_real"] = _abs_err(mm, ref)
            if "augur" in runs and name != "augur" and not replay:
                r["vs_augur"] = _stat_errors(mm, cl["augur"])
                r["abs_err_augur"] = _abs_err(mm, cl["augur"])
            r["decisions"] = _decisions(steps, real_steps)
            if "augur" in runs and name != "augur" and not replay:
                r["decisions_augur"] = _decisions(steps, runs["augur"][1])
        r["preempt"] = _preempt(pre_ev.get(name), pre_ev["real"])
        res["sims"][name] = r
    res["sims"]["real"]["abs_engine"] = {k: [_stat(real_eng[k], q) for _, q in STATS] for k in real_eng}
    return res


def _fmt(x, spec="+.1f"):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, spec)


def render(results):
    out = []
    for res in results:
        mode = ("async" if res["async"] else "sync") + (f", ngram k={res['spec_k']}" if res["spec_k"] else "") + \
            (f", offload {res['offload_gib']} GiB" if res["offload_gib"] else "")
        out.append(f"### {res['run']} ({res['requests']} requests, {mode})\n")
        if res["spec"]:
            out.append(f"ngram acceptance, accepted drafts per decode step: real {res['spec']['real_accepted_per_step']:.3f}, "
                       f"augur at --spec-accept-rate {res['spec']['spec_p']} {res['spec']['augur_accepted_per_step']:.3f}\n")
        real = res["sims"]["real"]["abs"]
        out.append("real, as the client measured it (s): " + ", ".join(f"{m} " + "/".join(_fmt(v, ".3f") for v in real[m]) for m in ("ttft", "tpot", "e2e"))
                   + "  (mean/p50/p90/p99)\n")
        for ref in ("real", "augur"):
            hdr = ("| sim | TTFT mean | p50 | p90 | p99 | TPOT mean | p50 | p90 | p99 | E2E mean | p50 | p90 | p99 | \\|e\\| |\n"
                   "|---|" + "---|" * 13 + "\n")
            body = ""
            for name, r in res["sims"].items():
                e = r.get(f"vs_{ref}")
                if e is None:
                    continue
                cells = [_fmt(e[f"{m}_{n}"]) for m in ("ttft", "tpot", "e2e") for n, _ in STATS]
                body += f"| {name} | " + " | ".join(cells) + f" | {_fmt(r[f'abs_err_{ref}'], '.1f')} |\n"
            if body:
                out.append(f"Error of the statistic against {ref} (%):\n\n" + hdr + body)
        for ref in ("real", "augur"):
            hdr = "| sim | steps | same | first | prefix | found | pre |\n|---|---|---|---|---|---|---|\n"
            body = ""
            for name, r in res["sims"].items():
                d = r.get("decisions" if ref == "real" else "decisions_augur")
                if name == "real" or (ref == "augur" and name == "augur"):
                    continue
                if d is None:
                    body += f"| {name} | n/a | | | | | {r['pre']} |\n"
                else:
                    body += (f"| {name} | {d['sim_steps']} (ref {d['real_steps']}) | {100 * d['same']:.1f}% | "
                             f"{d['first'] if d['first'] else 'none'} | {100 * d['prefix']:.1f}% | "
                             f"{100 * d['found']:.1f}% | {r['pre']} ({ref} {res['sims'][ref]['pre']}) |\n")
            if body:
                out.append(f"Decisions against the {ref} run's steps:\n\n" + hdr + body)
        hdr = "| sim | preemptions | requests preempted | first at step | real's preempted requests also preempted | same step and request |\n|---|---|---|---|---|---|\n"
        body = ""
        for name, r in res["sims"].items():
            p = r.get("preempt")
            if p is None:
                body += f"| {name} | {r['pre']} | n/a (no per-step record) | | | |\n"
            else:
                body += (f"| {name} | {p['events']} | {p['requests']} | {p['first_step']} | "
                         f"{_fmt(None if p['real_requests_also'] is None else 100 * p['real_requests_also'], '.0f')}% | "
                         f"{_fmt(None if p['same_step_and_request'] is None else 100 * p['same_step_and_request'], '.0f')}% |\n")
        out.append("Preemptions (the real run's from its step log, not client.csv):\n\n" + hdr + body)
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="compare")
    p.add_argument("dirs", nargs="+")
    p.add_argument("--md", default=None)
    p.add_argument("--json", default=None)
    a = p.parse_args(argv)
    results = [compare(d) for d in a.dirs]
    text = render(results)
    print(text)
    if a.md:
        open(a.md, "w").write(text)
    if a.json:
        json.dump(results, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
