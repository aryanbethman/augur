import json
import random

import pytest

N_REQUESTS = 60


@pytest.fixture(scope="session")
def workload(tmp_path_factory):
    """A deterministic synthetic trace (augur's flat-request format), ~10 req/s, for tests that only need an
    engine. Lengths are spread widely: under the tests' small configuration (600 KV blocks, 16 sequences,
    512-token batches) the run queues, chunks prefills and preempts; a third of the requests share a
    256-token prefix. Every request fits max_model_len 8192."""
    rng = random.Random(0)
    prefix = [rng.randrange(1000, 100_000) for _ in range(256)]
    path = tmp_path_factory.mktemp("workload") / "synthetic.jsonl"
    t = 0.0
    with open(path, "w") as f:
        for i in range(N_REQUESTS):
            t += rng.expovariate(10.0)
            n_in = int(min(3000, max(32, rng.lognormvariate(6.5, 1.0))))
            head = []
            if i % 3 == 0:
                n_in = max(n_in, 272)
                head = prefix
            ids = head + [rng.randrange(1000, 100_000) for _ in range(n_in - len(head))]
            n_out = int(min(1000, max(8, rng.lognormvariate(5.0, 0.9))))
            f.write(json.dumps({"input_toks": n_in, "output_toks": n_out, "arrival_time_ns": int(t * 1e9),
                                "input_tok_ids": ids}) + "\n")
    return str(path)
