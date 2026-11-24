"""Does augur.verify.prefix_share rebuild a real run's shared KV right?  Per step, the KV tokens that decodes hold in
blocks another decode of the step also holds: augur's engine replaying the real log (vLLM's own block ids) against
prefix_share's rebuild from the workload's prompts (what stepcheck, calibrate and breakdown price real steps with).

    python scripts/prefix_share_check.py RUN_DIR WORKLOAD [MAX_MODEL_LEN]

RUN_DIR is a synchronous real run (scripts/real_trace_run.sh: steps.*.jsonl, tokens.jsonl, server.log).  Prints the
share of steps where the two agree and the shared tokens of each.  Checked 2026-11-24 on the stress ShareGPT run
(520 KV blocks, 225 preemptions; 31,931 steps) and the trimmed agent sessions (5,560 steps): equal in every step.
"""

import glob
import os
import re
import sys

import numpy as np

import augur.cli as cli
from augur.verify import prefix_share
from augur.verify.replay import ReplayCost, read_log
from augur.verify.stepcheck import real_steps


def main(run_dir: str, workload: str, max_model_len: str = "8192") -> int:
    log = glob.glob(os.path.join(run_dir, "steps.*.jsonl"))[0]
    server = open(os.path.join(run_dir, "server.log")).read()
    blocks = int(re.findall(r"GPU KV cache size: ([0-9,]+)", server)[-1].replace(",", "")) // 16
    engine = []
    replayed = ReplayCost.step_ns

    def record(self, shape):
        engine.append(sum(s.shared_kv for s in shape.seqs))
        return replayed(self, shape)

    ReplayCost.prices_sharing = True  # the executor reads the block ids only for a cost model that prices sharing
    ReplayCost.step_ns = record
    rc = cli.main(["--model", "Qwen/Qwen3-1.7B", "--hardware", "RTX4060Ti", "--num-gpu-blocks", str(blocks),
                   "--max-num-seqs", "128", "--max-num-batched-tokens", "2048", "--max-model-len", max_model_len,
                   "--no-recorded-outputs", "--filler-vocab", "150000", "--no-async-scheduling", "--quiet",
                   "--workload", workload, "--replay-durations", log,
                   "--replay-tokens", os.path.join(run_dir, "tokens.jsonl"), "--output", os.devnull])
    if rc:
        return rc
    rebuilt = np.array([sum(s.shared_kv for s in shape.seqs)
                        for shape, _ in real_steps(read_log(log), prefix_share.prompt_chains(workload))])
    engine = np.array(engine)
    if len(engine) != len(rebuilt):
        print(f"the replay ran {len(engine)} priced steps, the log has {len(rebuilt)}: not an exact replay")
        return 1
    print(f"steps {len(engine)}, equal {np.mean(engine == rebuilt):.4f}; shared KV tokens: engine {engine.sum()}, "
          f"rebuilt {rebuilt.sum()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
