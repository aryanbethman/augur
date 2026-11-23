"""augur.verify.breakdown on a constructed log: the retimed timeline is known by hand, the parts add up to the total,
and augur's own run, taken as the "real" one, has no error to split. No GPU."""

import csv
import glob
import io
import json
from contextlib import redirect_stdout

import pytest

from augur.cli import main as augur_main
from augur.cost import SeqWork, StepShape
from augur.verify import breakdown as B
from augur.verify.stepcheck import PricedStep

MS = 1_000_000
ZERO = B.Timing(cpu_ns=0, cpu_per_request_ns=0, spec_ns_per_request=0, step_time_scale=1.0, frontend_ns=0,
                frontend_ns_per_token=0.0, delivery_ns=0)


def priced(ns):
    return PricedStep(StepShape(seqs=(SeqWork(num_new=1, num_computed_before=5, is_decode=True),)), False, 0, ns, {})


def flat(n, fe=0):
    return {f"req_{i}": B.CallInfo(None, 0, 4, 2, 0) for i in range(n)}


def log(rows):
    """rows: (kind, id, t_ms[, request, finished]) -> events; a step has one request and one token out."""
    ev = []
    for r in rows:
        kind, t = r[0], r[2] * MS
        if kind == "add":
            ev.append({"ev": "add", "t": t, "rid": f"augur.{r[1]}.", "prompt": 4})
        elif kind == "sched":
            ev.append({"ev": "sched", "t": t, "k": r[1], "tok": {f"augur.{r[3]}.": 1}, "new": {}, "pre": []})
        else:
            fin = [f"augur.{r[3]}."] if r[4] else []
            ev.append({"ev": "upd", "t": t, "k": r[1], "out": {f"augur.{r[3]}.": 1}, "fin": fin, "left": 0})
    return ev


def test_sync_retime_waits_for_arrival_then_runs_back_to_back():
    # req_0 arrives at 0 and takes two steps; req_1 reaches the engine at 60 ms, long after the engine went idle.
    ev = log([("add", "req_0", 0), ("sched", 1, 0, "req_0"), ("upd", 1, 9, "req_0", False),
              ("sched", 2, 9, "req_0"), ("upd", 2, 18, "req_0", True),
              ("add", "req_1", 60), ("sched", 3, 61, "req_1"), ("upd", 3, 70, "req_1", True)])
    t = B.Timing(1 * MS, 0, 0, 1.0, 0, 0.0, 0)
    out, late = B.retime(ev, [priced(4 * MS)] * 3, flat(2), t)
    assert out["req_0"].ttft == 5 * MS and out["req_0"].latency == 10 * MS and out["req_0"].itl == [5 * MS]
    assert out["req_1"].ttft == 5 * MS  # starts at its own arrival, not at the logged 61 ms
    assert (late.waits, late.steps) == (1, 0)  # req_1 found the engine idle, as in the real run


def test_a_request_the_real_engine_already_had_is_late():
    # The real engine was busy with req_0 (step 1 ends at 100 ms) when req_1 arrived at 10 ms, so step 2 was
    # scheduled at once after it; faster steps finish at 5 ms, and req_1 only arrives at 10 ms.
    ev = log([("add", "req_0", 0), ("sched", 1, 0, "req_0"), ("add", "req_1", 10), ("upd", 1, 100, "req_0", True),
              ("sched", 2, 100, "req_1"), ("upd", 2, 110, "req_1", True)])
    out, late = B.retime(ev, [priced(5 * MS)] * 2, flat(2), ZERO)
    assert late.steps == 1 and late.requests == 1 and late.delay_ns == 5 * MS
    assert out["req_1"].latency == 5 * MS


def test_async_step_waits_for_the_update_it_followed_and_the_gpu():
    # Step 2 was scheduled before step 1's update (async: it depends on nothing); its batch queues behind step 1.
    ev = log([("add", "req_0", 0), ("add", "req_1", 0), ("sched", 1, 0, "req_0"), ("sched", 2, 1, "req_1"),
              ("upd", 1, 10, "req_0", True), ("upd", 2, 20, "req_1", True)])
    out, _ = B.retime(ev, [priced(10 * MS)] * 2, flat(2), ZERO)
    assert out["req_0"].latency == 10 * MS and out["req_1"].latency == 20 * MS


def test_a_session_call_is_sent_when_augurs_loop_sends_it_not_when_the_log_did():
    c = {"s#0": B.CallInfo("s", 0, 4, 2, 7 * MS), "s#1": B.CallInfo("s", 1, 4, 2, 0)}
    ev = log([("add", "s#0", 0), ("sched", 1, 0, "s#0"), ("upd", 1, 9, "s#0", True),
              ("add", "s#1", 99), ("sched", 2, 99, "s#1"), ("upd", 2, 120, "s#1", True)])
    t = B.Timing(0, 0, 0, 1.0, 2 * MS, 0.0, 1 * MS)
    out, late = B.retime(ev, [priced(4 * MS)] * 2, c, t)
    # s#0 is sent 2 ms before it reaches the engine, runs 4 ms and is delivered 1 ms later. s#1 is sent 7 ms after
    # that, reaches the engine 2 ms later, runs 4 ms and is delivered 1 ms later: not at the logged 99 ms.
    assert out["s#0"].ttft == 2 * MS + 4 * MS + 1 * MS and out["s#1"].ttft == 2 * MS + 4 * MS + 1 * MS
    assert late.steps == 0


def test_the_arrivals_trace_sends_first_calls_a_front_end_before_the_log_and_stays_sorted(tmp_path):
    src, dst = tmp_path / "w.jsonl", tmp_path / "a.jsonl"
    rows = [{"input_toks": 4, "output_toks": 2, "arrival_time_ns": 0},
            {"session_id": "s", "arrival_time_ns": 5, "sub_requests": [{"input_toks": 4, "output_toks": 2}] * 2},
            {"input_toks": 4, "output_toks": 2, "arrival_time_ns": 9}]
    src.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ev = log([("add", "req_0", 10), ("add", "s#0", 30), ("add", "req_2", 25), ("add", "s#1", 40)])
    c = {"req_0": B.CallInfo(None, 0, 4, 2, 0), "s#0": B.CallInfo("s", 0, 4, 2, 0), "s#1": B.CallInfo("s", 1, 4, 2, 0),
         "req_2": B.CallInfo(None, 0, 4, 2, 0)}
    B.arrivals_trace(str(src), str(dst), c, ev, B.Timing(0, 0, 0, 1.0, 2, 0.0, 0))
    out = [json.loads(line)["arrival_time_ns"] for line in dst.read_text().splitlines()]
    # the engine got them 0, 20 and 15 ms into the log, 2 ns after they were sent (shift 2 ns): the third is held to the second
    assert out == [0, 20 * MS, 20 * MS]


def test_the_client_sees_the_front_end_and_delivery_but_not_in_tpot():
    t = B.Timing(0, 0, 0, 1.0, 2 * MS, 1000.0, 1 * MS)  # 2 ms + 1 us per prompt token, 1 ms delivery
    (r,) = B.client_view({"a": B.Req(10 * MS, 20 * MS, [5 * MS, 5 * MS])}, {"a": B.CallInfo(None, 0, 1000, 3, 0)}, t).values()
    assert (r.ttft, r.latency, r.itl) == (14 * MS, 24 * MS, [5 * MS, 5 * MS])


def test_the_parts_add_up_to_the_total():
    base = B.summarize({"a": B.Req(10 * MS, 100 * MS, [5 * MS, 7 * MS]), "b": B.Req(30 * MS, 300 * MS, [9 * MS])})
    s = [{k: v * f for k, v in base.items()} for f in (1, 0.5, 0.8, 1.1, 1.5)]
    for r in B.parts(*s).values():
        assert r["total"] == pytest.approx(sum(r[n] for n in B.PARTS))
        assert r["total"] == pytest.approx(r["sim"] - r["real"])
        assert r["outside"] == pytest.approx(r["outside_replay"] + r["outside_arrivals"])


def test_summarize_uses_vllm_definitions():
    s = B.summarize({"a": B.Req(2 * MS, 12 * MS, [5 * MS, 5 * MS]), "b": B.Req(4 * MS, 4 * MS, [])})
    assert s["TTFT mean"] == 3 and s["latency p50"] == 8
    assert s["TPOT mean"] == 5 and s["ITL mean"] == 5  # b has one token: no TPOT, no gaps


@pytest.mark.parametrize("mode", ["--no-async-scheduling", "--async-scheduling"], ids=["sync", "async"])
def test_augurs_own_run_has_no_error_to_split(tmp_path, workload, mode):
    """Taken as the real run, augur's own run (its log, its CSV) must be reproduced by the replay, the retimed
    timeline and the run with its arrivals, once put on the client's side: every part is zero."""
    args = ["--model", "Qwen/Qwen3-1.7B", "--hardware", "RTX4060Ti", "--num-gpu-blocks", "600", "--max-model-len", "8192",
            "--max-num-seqs", "16", "--max-num-batched-tokens", "512", "--no-recorded-outputs", "--quiet", mode]
    own = str(tmp_path / "own.csv")
    with redirect_stdout(io.StringIO()):
        assert augur_main([*args, "--workload", workload, "--output", own, "--step-log-vllm", str(tmp_path / "own")]) == 0
    (log_path,) = glob.glob(str(tmp_path / "own.*.jsonl"))
    with redirect_stdout(io.StringIO()):
        assert B.main(["--log", log_path, "--real", own, "--workload", workload, "--model", "Qwen/Qwen3-1.7B",
                       "--max-num-seqs", "16", "--max-num-batched-tokens", "512", "--num-gpu-blocks", "600",
                       "--no-recorded-outputs", "--csv", str(tmp_path / "b.csv")]) == 0
    with open(tmp_path / "b.csv") as f:
        rows = list(csv.DictReader(f))
    assert rows
    for r in rows:
        for c in ("total", "outside", "step_time", "decisions"):
            assert float(r[f"{c}_ms"]) == pytest.approx(0, abs=1e-3), (r["metric"], r["stat"], c)
