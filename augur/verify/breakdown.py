"""Where does augur's error against a real vLLM run come from?

Splits the end-to-end error (augur's full simulation minus the real run) of each metric (mean, p50, p90 and p99 of
TTFT, TPOT, ITL and request latency) into three parts that add up to it. A part is the difference of the metric of
two runs of the same requests, so total = (1) + (2) + (3) exactly.

    real     the run, as its client measured it (client.csv from augur.verify.client, or the JSON of
             `vllm bench serve --save-result`: the request ids of the step log carry the benchmark's request numbers)
    replay   augur replaying the real step log (--replay-durations): requests reach the engine when the log says and
             every step takes its logged duration, so decisions are the real ones (checked below)
    retimed  the logged schedule (same batches, same order) priced by augur: its step prices
             (augur.verify.stepcheck) and CPU terms (--step-cpu-us, --step-cpu-per-request-us,
             --spec-us-per-request, --step-time-scale). No decision is taken. Flat requests and a session's first call
             reach the engine when the log says; a session's later calls where augur's normal run puts them.
    arrivals augur's normal run (its own decisions and prices) of the trace with each flat request and each session's
             first call sent so that it reaches the engine when the log says
    sim      augur's normal run: its own decisions, prices, front end and arrival process (trace times, and a
             session's next call released by its previous one)

Client model. The replay knows when a request reaches the engine and when a token leaves it. It is put on the client's
side as augur's normal run is (the retimed and arrivals runs are on it by construction): a request is sent
--frontend-latency-ms + --frontend-us-per-token * prompt tokens before it reaches the engine, and each token and the
end of the request reach the client --delivery-latency-ms after the engine emits them (TTFT and latency grow by
both, TPOT and ITL by neither).

    (1) outside the step loop   (replay - real) + (sim - arrivals). What the engine's own timeline leaves out and
                                augur models as constants: the real front end, delivery, the client (e.g. a
                                connection cap queueing requests before the engine), and how requests reach the engine.
                                (replay - real) is the first term; (sim - arrivals) moves the first calls from the
                                real arrivals at the engine to augur's trace times. Both are in the CSV.
    (2) step time               retimed - replay. Logged step durations and the gaps between steps replaced by
                                augur's prices and its CPU terms (and, for sessions, later calls placed by augur's
                                loop instead of the log's).
    (3) decisions               arrivals - retimed. augur's batches, preemptions and prefix hits instead of the
                                logged ones, under the same arrivals.

Timeline of the retimed run. Step i is scheduled at the later of
  - the end of the step whose update the real engine had applied when it scheduled step i (the previous step; in async
    scheduling two back; read from the log), and
  - the arrival at the engine of every request first scheduled in it (the causality rule: a request cannot be
    scheduled before it arrives, so the engine idles until then, as augur's does),
then its batch is launched once the CPU has prepared it and the GPU is free (max(schedule + CPU, GPU free), as
augur's executor) and its update lands when the GPU finishes it; with synchronous scheduling that is schedule + CPU +
GPU. A session's later call is sent --delivery-latency-ms after the previous call's last token plus that call's tool
wait, as augur's own loop does, and reaches the engine a front end later: the session loop stays closed in the retimed
and arrivals runs, and only the replay (whose engine timeline is the log's) takes every arrival from the log. The
causality rule changes the real schedule when faster steps leave the engine waiting for a request the real engine
already had: those steps, their new requests and the idle time added are reported as "late". A later session call
that augur's loop sends after the step it was logged in leaves the engine idle in the same way.

Not priced by the retimed run: offload transfers and their stalls, contention, a drafter other than ngram (they are
in the runs with augur's decisions, so they land in (3)).

    python -m augur.verify.breakdown --log out/real/run/steps.<pid>.jsonl --real out/real/run/client.csv \\
        --workload trace.jsonl --model Qwen/Qwen3-1.7B --hardware RTX4060Ti [--csv out.csv] \\
        --num-gpu-blocks 6000 --max-model-len 8192 --no-recorded-outputs --filler-vocab 150000

Arguments it does not know go to augur, for every run of it; sync or async scheduling is taken from the log.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import glob
import io
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field

import numpy as np

from augur.cost import parse_class_scales
from augur.cost.profiles import default_fork
from augur.trace.workload import Workload, call_id
from augur.verify.replay import ReplayWorkload, compare, is_async, read_log, steps

METRICS = ("TTFT", "TPOT", "ITL", "latency")
STATS = ("mean", "p50", "p90", "p99")
PARTS = ("outside", "step_time", "decisions")


@dataclass
class Req:
    """A finished call (ns)."""
    ttft: int
    latency: int
    itl: list[int]


@dataclass
class CallInfo:
    session: str | None
    index: int  # position in its session (0 for flat requests)
    n_in: int
    n_out: int
    tool_wait_ns: int  # the session's wait after this call


def workload_calls(path: str) -> dict[str, CallInfo]:
    """Every call of the trace by call id, in file order."""
    out = {}
    for e in Workload(path).entries():
        if hasattr(e, "sub_requests"):
            for i, d in enumerate(e.sub_requests):
                out[f"{e.session_id}#{i}"] = CallInfo(e.session_id, i, int(d["input_toks"]), int(d["output_toks"]),
                                                      int(d.get("tool_duration_ns", 0) or 0))
        else:
            out[e.call_id] = CallInfo(None, 0, len(e.prompt_ids), e.output_toks, e.tool_wait_ns)
    return out


def from_csv(path: str, calls: dict[str, CallInfo], order: list[str] | None = None) -> dict[str, Req]:
    """Per-call results of a CSV in the fork's format. A call is named by its session and index; flat requests, by
    ``order`` (the call ids in the order the run numbered them), else the trace's. Raises ValueError if the
    lengths of a row disagree with the call it is matched to."""
    with open(path) as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["request id"]))
    flat = iter(order or [c for c, i in calls.items() if i.session is None])
    out = {}
    for r in rows:
        cid = f"{r['session_id']}#{r['sub_request_index']}" if r.get("session_id") else next(flat)
        if (calls[cid].n_in, calls[cid].n_out) != (int(r["input"]), int(r["output"])):
            raise ValueError(f"{path}: request {r['request id']} does not match call {cid} of the workload")
        out[cid] = Req(int(r["TTFT"]), int(r["latency"]), json.loads(r["ITL"]))
    return out


def from_bench(path: str) -> dict[str, Req]:
    """Per-call results of `vllm bench serve --save-result` (request i is the trace's req_i)."""
    with open(path) as f:
        d = json.load(f)
    return {f"req_{i}": Req(round(t * 1e9), round(lat * 1e9), [round(x * 1e9) for x in itl])
            for i, (t, lat, itl, err) in enumerate(zip(d["ttfts"], d["latencies"], d["itls"], d["errors"])) if not err}


@dataclass
class Timing:
    """augur's per-run timing terms (ns), from its own options."""
    cpu_ns: int
    cpu_per_request_ns: int
    spec_ns_per_request: int
    step_time_scale: float
    frontend_ns: int
    frontend_ns_per_token: float
    delivery_ns: int
    class_scales: dict = field(default_factory=dict)  # --step-class-scales


def timing(a: argparse.Namespace) -> Timing:
    return Timing(int(a.step_cpu_us * 1e3), int(a.step_cpu_per_request_us * 1e3), int(a.spec_us_per_request * 1e3),
                  a.step_time_scale, int(a.frontend_latency_ms * 1e6), a.frontend_us_per_token * 1e3,
                  int(a.delivery_latency_ms * 1e6), parse_class_scales(a.step_class_scales))


def frontend_ns(t: Timing, info: CallInfo) -> int:
    """Time from a client sending a call to the engine receiving it (the coordinator's own formula)."""
    return t.frontend_ns + int(t.frontend_ns_per_token * info.n_in)


def engine_arrivals(events: list[dict]) -> dict[str, int]:
    """When each call first reached the engine, from the log's first event."""
    t0 = min(e["t"] for e in events)
    out: dict[str, int] = {}
    for e in events:
        if e["ev"] == "add":
            out.setdefault(call_id(e["rid"]), e["t"] - t0)
    return out


def arrivals_trace(src: str, dst: str, calls: dict[str, CallInfo], events: list[dict], t: Timing) -> None:
    """Write the trace with each flat request and each session's first call sent a front end before it reached the
    engine in the log (shifted to start at 0, and held to its predecessor's time where the log reorders neighbours:
    augur reads a trace in arrival order). Later calls of a session keep their place in the loop."""
    added = engine_arrivals(events)
    with open(src) as f:
        rows = [(n, json.loads(line)) for n, line in enumerate(f) if line.strip()]
    rows = [(n, d) for n, d in rows if d.get("sub_requests", True)]  # augur skips empty sessions
    sent = []
    for n, d in rows:
        cid = f"{d.get('session_id', f'session_{n}')}#0" if "sub_requests" in d else f"req_{n}"
        sent.append(added[cid] - frontend_ns(t, calls[cid]))
    shift = -min(sent)
    last = 0
    with open(dst, "w") as f:
        for (n, d), c in zip(rows, sent):
            last = max(last, c + shift)
            d["arrival_time_ns"] = last
            f.write(json.dumps(d) + "\n")


def client_view(reqs: dict[str, Req], calls: dict[str, CallInfo], t: Timing) -> dict[str, Req]:
    """Engine-side results seen from the client, as augur's normal run measures them (see the module docstring)."""
    out = {}
    for c, r in reqs.items():
        extra = frontend_ns(t, calls[c]) + t.delivery_ns
        out[c] = Req(r.ttft + extra, r.latency + extra, r.itl)
    return out


def summarize(reqs: dict[str, Req]) -> dict[str, float]:
    """"<metric> <stat>" -> ms. TPOT: per request, over requests with more than one token; ITL: every gap of every
    request, pooled (vLLM's own definitions)."""
    r = list(reqs.values())
    x = {"TTFT": [q.ttft for q in r], "latency": [q.latency for q in r],
         "TPOT": [(q.latency - q.ttft) / len(q.itl) for q in r if q.itl], "ITL": [g for q in r for g in q.itl]}
    out = {}
    for m, v in x.items():
        v = np.asarray(v, dtype=float) / 1e6
        out.update({f"{m} mean": v.mean(), f"{m} p50": np.percentile(v, 50), f"{m} p90": np.percentile(v, 90),
                    f"{m} p99": np.percentile(v, 99)})
    return out


def parts(real: dict, replay: dict, retimed: dict, arrivals: dict, sim: dict) -> dict[str, dict[str, float]]:
    """Per "<metric> <stat>": the five values, the total error and its parts (ms); (1) is also split in two."""
    out = {}
    for k in real:
        o1, o2 = replay[k] - real[k], sim[k] - arrivals[k]
        out[k] = {"real": real[k], "replay": replay[k], "retimed": retimed[k], "arrivals": arrivals[k], "sim": sim[k],
                  "total": sim[k] - real[k], "outside": o1 + o2, "step_time": retimed[k] - replay[k],
                  "decisions": arrivals[k] - retimed[k], "outside_replay": o1, "outside_arrivals": o2}
    return out


@dataclass
class Late:
    """Where the causality rule changed the logged schedule: a request reached the engine after the step it was
    first scheduled in could have started, while the real engine already had it."""
    steps: int = 0
    requests: int = 0
    delay_ns: int = 0
    waits: int = 0  # steps that started on an arrival, late or not (an idle engine)


def retime(events: list[dict], priced: list, calls: dict[str, CallInfo], t: Timing) -> tuple[dict[str, Req], Late]:
    """The logged schedule on augur's timeline (see the module docstring), as its client sees it. ``priced`` is
    stepcheck.price_steps(events)."""
    t0 = min(e["t"] for e in events)
    pairs = steps(events)
    index = {s["k"]: i for i, (s, _) in enumerate(pairs)}
    dep: dict[int, int] = {}  # step -> the step whose update the real engine had applied when it scheduled it
    applied = -1
    for e in events:
        if e["ev"] == "sched" and e["k"] in index:
            dep[index[e["k"]]] = applied
        elif e["ev"] == "upd" and e["k"] in index:
            applied = index[e["k"]]
    added = engine_arrivals(events)
    # When each call is sent: flat requests and a session's first call a front end before the log's arrival; a
    # session's later calls once the previous one is delivered and its tool wait is over.
    client = {c: a - frontend_ns(t, calls[c]) for c, a in added.items() if calls[c].index == 0}
    tokens: dict[str, list[int]] = {}
    done: dict[str, int] = {}
    seen: set[str] = set()
    late, ends, gpu_free = Late(), [], 0
    for i, ((s, u), p) in enumerate(zip(pairs, priced)):
        new = [c for c in map(call_id, s["tok"]) if c not in seen]
        seen.update(new)
        if any(c not in client for c in new):
            raise ValueError(f"step {s['k']}: a call is scheduled before the call that releases it finished")
        arrive = {c: client[c] + frontend_ns(t, calls[c]) for c in new}
        after = ends[dep[i]] if dep[i] >= 0 else 0
        start = max([after, *arrive.values()])
        if start > after:
            late.waits += 1
            # Late: the real engine already had the request when it applied the update this step waited for.
            wait = [c for c in new if arrive[c] > after and dep[i] >= 0 and added[c] <= pairs[dep[i]][1]["t"] - t0]
            if wait:
                late.steps += 1
                late.requests += len(wait)
                late.delay_ns += start - after
        seqs = p.shape.seqs
        gpu = round(p.predicted_ns * t.step_time_scale * t.class_scales.get(p.shape.step_class, 1.0)) + t.spec_ns_per_request * sum(
            1 for x in seqs if x.is_decode and x.num_new > 1)
        launch = max(start + t.cpu_ns + t.cpu_per_request_ns * len(seqs), gpu_free)
        gpu_free = launch + gpu
        ends.append(gpu_free)
        for rid, n in u["out"].items():
            tokens.setdefault(call_id(rid), []).extend([gpu_free + t.delivery_ns] * n)
        for rid in u["fin"]:
            c = call_id(rid)
            done[c] = gpu_free + t.delivery_ns
            nxt = f"{calls[c].session}#{calls[c].index + 1}"
            if calls[c].session is not None and nxt in calls:
                client[nxt] = done[c] + calls[c].tool_wait_ns
    out = {}
    for c, end in done.items():
        tt = tokens[c]
        out[c] = Req(tt[0] - client[c], end - client[c], [b - a for a, b in zip(tt, tt[1:])])
    return out, late


def run_augur(argv: list[str]) -> None:
    """One augur run, its summary discarded. Raises RuntimeError if it fails."""
    from augur.cli import main as augur_main

    with contextlib.redirect_stdout(io.StringIO()):
        rc = augur_main(argv)
    if rc:
        raise RuntimeError(f"augur {' '.join(argv)} exited with {rc}")


def only(prefix: str) -> str:
    (path,) = glob.glob(f"{glob.escape(prefix)}.*.jsonl")
    return path


def pct(r: dict[str, float], k: str) -> float:
    """A value as % of the real one (nan when the real value is 0, e.g. the median ITL under speculative decoding)."""
    return 100 * r[k] / r["real"] if r["real"] else float("nan")


def table(name: str, p: dict[str, dict[str, float]]) -> list[str]:
    """Rows of the breakdown: the real value (ms) and the total error and its parts, in % of it."""
    lines = [f"{name:<14}{'real ms':>10}{'total %':>10}" + "".join(f"{n + ' %':>13}" for n in PARTS)]
    for m in METRICS:
        for st in STATS:
            r = p[f"{m} {st}"]
            lines.append(f"{m + ' ' + st:<14}{r['real']:>10.2f}" + "".join(
                f"{pct(r, k):>{w}.1f}" for k, w in (("total", 10), *((n, 13) for n in PARTS))))
    return lines


def main(argv=None) -> int:
    if argv is None:
        from augur.cli import pin_hash_seed
        pin_hash_seed("augur.verify.breakdown")
    ap = argparse.ArgumentParser(prog="augur.verify.breakdown", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True, help="the real run's step log (steps.<pid>.jsonl)")
    ap.add_argument("--real", required=True, help="the real run's per-call results: augur.verify.client CSV, or "
                                                  "`vllm bench serve --save-result` JSON")
    ap.add_argument("--workload", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--hardware", default="RTX4060Ti")
    ap.add_argument("--max-num-seqs", type=int, default=128)
    ap.add_argument("--max-num-batched-tokens", type=int, default=2048)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--fork", default=default_fork())
    ap.add_argument("--name", default=None, help="label of the run (default: its folder)")
    ap.add_argument("--csv", default=None, help="write every value and part (ms and %% of the real value) here")
    ap.add_argument("--work", default=None, help="keep the augur runs' outputs here (default: discard)")
    ap.add_argument("--cost", default="auto", choices=["auto", "roofline"],
                    help="auto: augur's own choice for the runs and the profile for the retimed prices; roofline: "
                         "--cost roofline for all of them")
    ap.add_argument("--roofline-params", default=None,
                    help="--cost roofline: parameters JSON (augur.cost.roofline_fit)")
    a, rest = ap.parse_known_args(argv)
    if a.cost == "roofline":
        rest = ["--cost", "roofline", *(["--roofline-params", a.roofline_params] if a.roofline_params else []), *rest]

    from augur.cli import _parser
    from augur.verify.stepcheck import make_pricer, price_steps

    events = read_log(a.log)
    base = ["--model", a.model, "--hardware", a.hardware, "--max-num-seqs", str(a.max_num_seqs),
            "--max-num-batched-tokens", str(a.max_num_batched_tokens), "--max-model-len", str(a.max_model_len),
            "--fork", a.fork, "--quiet",
            "--async-scheduling" if is_async(events) else "--no-async-scheduling", *rest]
    t = timing(_parser().parse_args([*base, "--workload", a.workload]))
    name = a.name or os.path.basename(os.path.dirname(os.path.abspath(a.log)))
    calls = workload_calls(a.workload)
    real = from_bench(a.real) if a.real.endswith(".json") else from_csv(a.real, calls)
    order = [c.call_id for c in ReplayWorkload(path=a.workload, log_path=a.log).entries()]

    with tempfile.TemporaryDirectory(prefix="augur-breakdown-") as tmp:
        work = a.work or tmp
        os.makedirs(work, exist_ok=True)
        arrivals_trace(a.workload, f"{work}/arrivals.jsonl", calls, events, t)
        same = {}
        for run, flags in (("replay", ["--workload", a.workload, "--replay-durations", a.log]),
                           ("arrivals", ["--workload", f"{work}/arrivals.jsonl"]), ("sim", ["--workload", a.workload])):
            run_augur([*base, *flags, "--step-log-vllm", f"{work}/{run}", "--output", f"{work}/{run}.csv"])
            same[run] = compare(a.log, only(f"{work}/{run}"))
        replay = client_view(from_csv(f"{work}/replay.csv", calls, order), calls, t)
        arrivals = from_csv(f"{work}/arrivals.csv", calls)
        sim = from_csv(f"{work}/sim.csv", calls)
    cost, disp = make_pricer(a.model, a.hardware, a.fork, a.max_num_seqs, a.max_num_batched_tokens, a.max_model_len,
                             a.cost == "roofline", a.roofline_params)
    priced = price_steps(events, cost, disp)
    retimed, late = retime(events, priced, calls, t)

    ids = sorted(set(real) & set(replay) & set(retimed) & set(arrivals) & set(sim))
    p = parts(*(summarize({c: d[c] for c in ids}) for d in (real, replay, retimed, arrivals, sim)))
    print(f"{name}: {len(ids)} requests (real {len(real)}, replay {len(replay)}, retimed {len(retimed)}, "
          f"arrivals {len(arrivals)}, sim {len(sim)})")
    print("  steps of the log reproduced: " + ", ".join(
        f"{k} {v['identical_steps']} of {v['steps_a']} ({v['steps_b']} run)" for k, v in same.items()))
    print(f"  retimed: {len(priced)} steps, {late.waits} started on an arrival, {late.steps} late ({late.requests} "
          f"requests, {late.delay_ns / 1e6:.1f} ms of idle time added)")
    print("\n".join(table(name, p)))
    if a.csv:
        cols = ("real", "replay", "retimed", "arrivals", "sim", "total", *PARTS, "outside_replay", "outside_arrivals")
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run", "metric", "stat", *(f"{c}_ms" for c in cols[:5]), *(f"{c}_ms" for c in cols[5:]),
                        *(f"{c}_pct" for c in cols[5:])])
            for k, r in p.items():
                m, st = k.split()
                w.writerow([name, m, st, *(f"{r[c]:.4f}" for c in cols), *(f"{pct(r, c):.3f}" for c in cols[5:])])
    return 0


if __name__ == "__main__":
    sys.exit(main())
