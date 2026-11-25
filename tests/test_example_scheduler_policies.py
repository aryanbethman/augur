"""examples/scheduler_policies.py: the three waiting-queue orders, and that augur loads each scheduler class."""

import os
import sys
from types import SimpleNamespace

import pytest

from augur.cli import main

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
import scheduler_policies as sp  # noqa: E402


def req(name, n, t=0.0):
    return SimpleNamespace(request_id=name, num_prompt_tokens=n, arrival_time=t)


def order(queue, reqs):
    for r in reqs:
        queue.add_request(r)
    return [queue.pop_request().request_id for _ in reqs]


def test_shortest_prompt_first():
    assert order(sp.ShortestPromptQueue(), [req("a", 50), req("b", 30), req("c", 80), req("d", 30)]) == list("bdac")


def test_deadline_first_lets_a_long_prompt_wait_longer_but_not_for_ever():
    # deadline = arrival + 5 s + 10 ms per prompt token: "long" (1000 tokens) is due at 15 s
    assert order(sp.DeadlineQueue(), [req("long", 1000, 0), req("late", 100, 5)]) == ["late", "long"]  # due at 11 s
    assert order(sp.DeadlineQueue(), [req("long", 1000, 0), req("later", 100, 20)]) == ["long", "later"]  # due 26 s


def test_aging_serves_a_long_prompt_after_enough_admissions():
    q = sp.AgingQueue()
    q.add_request(req("long", 400))
    served = []
    for i in range(200):  # a short prompt arrives after every admission, so SPF would never serve "long"
        q.add_request(req(f"s{i}", 10))
        served.append(q.pop_request().request_id)
        if "long" in served:
            break
    assert "long" in served and len(served) < 200


def test_preempted_request_goes_back_to_the_front_of_a_sorted_queue():
    q = sp.DeadlineQueue()
    q.add_request(req("short", 5))
    q.prepend_request(req("long", 500))
    assert q.peek_request().request_id == "long"


@pytest.mark.parametrize("name", ["ShortestPromptFirst", "DeadlineFirst", "AgingShortestFirst"])
def test_augur_runs_the_scheduler(tmp_path, workload, name):
    out = tmp_path / "out.csv"
    assert main(["--model", "Qwen/Qwen3-1.7B", "--cost", "fixed", "--num-gpu-blocks", "600", "--max-model-len", "8192",
                 "--fixed-step-ms", "8", "--fixed-per-token-us", "15", "--no-recorded-outputs", "--quiet",
                 "--num-entries", "12", "--workload", workload, "--output", str(out),
                 "--scheduler-cls", f"scheduler_policies.{name}AsyncScheduler"]) == 0
    assert len(out.read_text().splitlines()) == 13
