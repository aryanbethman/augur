"""examples/shortest_prompt_first.py: a custom scheduler for --scheduler-cls changes the order requests are served."""

import csv
import os
import sys
from types import SimpleNamespace

import pytest

from augur.cli import main

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from shortest_prompt_first import ShortestPromptQueue  # noqa: E402

BASE = ["--model", "Qwen/Qwen3-1.7B", "--cost", "fixed", "--num-gpu-blocks", "600", "--max-model-len", "8192",
        "--max-num-batched-tokens", "512", "--fixed-step-ms", "8", "--fixed-per-token-us", "15",
        "--no-recorded-outputs", "--no-async-scheduling", "--quiet", "--num-entries", "12",
        "--max-num-seqs", "1"]  # one request at a time: the queue's order is the service order


def test_queue_orders_by_prompt_length_and_keeps_arrival_order_among_equals():
    queue = ShortestPromptQueue()
    for name, n in [("a", 50), ("b", 30), ("c", 80), ("d", 30), ("e", 10)]:
        queue.add_request(SimpleNamespace(request_id=name, num_prompt_tokens=n))
    assert [queue.pop_request().request_id for _ in range(5)] == ["e", "b", "d", "a", "c"]


def test_preempted_request_goes_back_to_the_front():
    queue = ShortestPromptQueue()
    queue.add_request(SimpleNamespace(request_id="short", num_prompt_tokens=5))
    queue.prepend_request(SimpleNamespace(request_id="long", num_prompt_tokens=500))
    assert queue.peek_request().request_id == "long"


def finish_order(tmp_path, workload, *args):
    out = tmp_path / "out.csv"
    assert main([*BASE, "--workload", workload, "--output", str(out), *args]) == 0
    with open(out) as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["end_time"]))
    return [int(r["request id"]) for r in rows]


@pytest.mark.parametrize("cls,mode", [("ShortestPromptFirstScheduler", "--no-async-scheduling"),
                                      ("ShortestPromptFirstAsyncScheduler", "--async-scheduling")])
def test_scheduler_serves_short_prompts_first(tmp_path, workload, cls, mode):
    fcfs = finish_order(tmp_path, workload, mode)
    spf = finish_order(tmp_path, workload, mode, "--scheduler-cls", f"shortest_prompt_first.{cls}")
    assert sorted(fcfs) == sorted(spf) == list(range(12))
    assert spf != fcfs
