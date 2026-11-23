"""Run Vidur with augur's step price (the same-step-time test).

Vidur's execution-time predictor is replaced (no profiling data, no model training) by one that
asks ``pricer.py`` for each batch: augur's GPU time on the batch padded to vLLM's CUDA-graph size,
plus the CPU term. The step's ``ExecutionTime`` carries that total in its attention-decode field and
the CPU term in ``schedule_time``, and every other component is zero, so Vidur adds nothing of its
own to ``total_time``. Vidur's scheduler picks the batch; it schedules synchronously.

    PYTHONPATH=<vidur> AUGUR_WT=<worktree> <vidur>/.venv/bin/python scripts/baselines/vidur_predictor.py \\
        --step-log out/baselines/swe50/vidur_steps.jsonl --pricer-args "--max-model-len 24576" -- \\
        <vidur.main arguments>

``--step-log`` gets one JSON line per priced batch: ``tok`` and ``computed`` by request (Vidur's
request id, the row number in vidur.csv), tokens scheduled and tokens already in the KV cache.
The price server runs in augur's Python (``AUGUR_PYTHON``, default ``~/augur/.venv/bin/python``).
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
    state = {"k": 0}

    from vidur.entities import ExecutionTime
    from vidur.execution_time_predictor import BaseExecutionTimePredictor
    from vidur.execution_time_predictor.execution_time_predictor_registry import ExecutionTimePredictorRegistry
    from vidur.types import ExecutionTimePredictorType

    class AugurPredictor(BaseExecutionTimePredictor):
        def get_execution_time(self, batch, pipeline_stage):
            reqs = [(n, r.num_processed_tokens, int(n == 1)) for r, n in zip(batch.requests, batch.num_tokens)]
            gpu, cpu = client.price(reqs)
            state["k"] += 1
            log.write(json.dumps({"k": state["k"], "tok": {str(r.id): n for r, n in zip(batch.requests, batch.num_tokens)},
                                  "computed": {str(r.id): c for r, (_, c, _) in zip(batch.requests, reqs)},
                                  "ns": gpu + cpu}) + "\n")
            log.flush()
            zero = 0.0
            # ExecutionTime takes milliseconds and multiplies the per-layer block by the layer count
            per_layer_ms = gpu / 1e6 / self._num_layers_per_pipeline_stage
            return ExecutionTime(self._num_layers_per_pipeline_stage, zero, zero, per_layer_ms, zero, zero, zero, zero,
                                 zero, zero, zero, zero, zero, zero, zero, cpu / 1e6, zero, zero, zero, zero)

    AugurPredictor.__abstractmethods__ = frozenset()  # every price comes from get_execution_time
    for t in ExecutionTimePredictorType:
        ExecutionTimePredictorRegistry._registry[t] = AugurPredictor
    sys.argv = [sys.argv[0]] + rest
    from vidur.main import main as vidur_main
    vidur_main()
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
