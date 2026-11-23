"""Run Frontier with augur's step price (the same-step-time test).

Frontier's dummy execution-time mode returns one fixed value for every component of a step. This
wrapper keeps that mode and scales the value, per batch, so that the step takes what
``pricer.py`` says: augur's GPU time on the batch padded to vLLM's CUDA-graph size, plus the CPU
term. Everything else Frontier would add (its own CPU overheads, launch stalls) is inside the
dummy components and therefore inside that total. The wrapper checks each step's total against
the price and stops when they differ by more than 0.1 ms.

    PYTHONPATH=<Frontier> AUGUR_WT=<worktree> <Frontier>/.venv/bin/python \\
        scripts/baselines/frontier_predictor.py --step-log out/baselines/swe50/frontier_steps.jsonl \\
        --pricer-args "--max-model-len 24576" -- <frontier.main arguments>

``--step-log`` gets one JSON line per priced batch: ``tok`` and ``computed`` by request (Frontier's
request id, the row number in frontier.csv), tokens scheduled and tokens already in the KV cache.
Frontier schedules synchronously (no async scheduling), so the CPU term always adds to the step.
Needs Frontier's Python environment; the price server runs in augur's (``AUGUR_PYTHON``, default
``~/augur/.venv/bin/python``).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pricer import Client  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--step-log", required=True)
    p.add_argument("--pricer-args", default="")
    a, rest = p.parse_known_args()
    rest = rest[1:] if rest and rest[0] == "--" else rest
    client = Client(args=shlex.split(a.pricer_args), pythonpath=os.environ.get("AUGUR_WT"))
    log = open(a.step_log, "w")
    memo: dict[int, float] = {}
    state = {"k": 0}

    from frontier.execution_time_predictor.sklearn_execution_time_predictor import SklearnExecutionTimePredictor as P

    orig = P.predict_stage_execution_time

    def predict(self, batch, stage_id, cluster_type, *args, **kw):
        want_ms = memo.get(batch.id)
        if want_ms is None:
            reqs = [(n, c, int(n == 1)) for n, c in zip(batch.num_tokens, batch.num_context_tokens)]
            gpu, cpu = client.price(reqs)
            want_ms = (gpu + cpu) / 1e6
            memo[batch.id] = want_ms
            state["k"] += 1
            log.write(json.dumps({"k": state["k"], "tok": {str(r.id): n for r, n in zip(batch.requests, batch.num_tokens)},
                                  "computed": {str(r.id): c for r, c in zip(batch.requests, batch.num_context_tokens)},
                                  "ns": gpu + cpu}) + "\n")
            log.flush()
        self._dummy_execution_time = 1.0
        unit = orig(self, batch, stage_id, cluster_type, *args, **kw)
        per_ms = unit.total_time + unit.forward_launch_stall_time(unit.model_time)  # seconds per 1 ms of dummy value
        self._dummy_execution_time = want_ms * 1e-3 / per_ms
        out = orig(self, batch, stage_id, cluster_type, *args, **kw)
        got = out.total_time + out.forward_launch_stall_time(out.model_time)
        if abs(got - want_ms * 1e-3) > 1e-4:
            raise RuntimeError(f"Frontier step {got * 1e3:.3f} ms, price {want_ms:.3f} ms: the dummy value is not linear")
        return out

    P.predict_stage_execution_time = predict
    sys.argv = [sys.argv[0]] + rest
    from frontier.main import main as frontier_main
    frontier_main()
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
