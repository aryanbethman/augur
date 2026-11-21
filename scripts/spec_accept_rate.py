"""Acceptance rate of a real speculative-decoding run, from its step log, in the terms of LLMServingSim's
--spec-acceptance-rate: accepted draft tokens / (decode steps x num_speculative_tokens), so the mean tokens per
decode step is 1 + rate * N. Also prints the rate over steps that carried drafts only.

usage: python scripts/spec_accept_rate.py STEPLOG.jsonl N
"""
import sys
from collections import defaultdict

from augur.verify.replay import read_log, steps


def main(path, n):
    sched = defaultdict(int)
    emitted = defaultdict(int)
    dec_steps = drafted_steps = drafts = accepted = 0
    tok_dec = 0
    for s, u in steps(read_log(path)):
        if u is None:
            continue
        for r, t in s["tok"].items():
            o = u["out"].get(r, 0)
            if t <= n + 1 and o >= 1:  # a decode step (prefill chunks are larger)
                dec_steps += 1
                tok_dec += o
                if t > 1:
                    drafted_steps += 1
                    drafts += t - 1
                    accepted += o - 1
    print(f"decode steps {dec_steps}, with drafts {drafted_steps} ({100 * drafted_steps / dec_steps:.1f}%), "
          f"mean tokens per decode step {tok_dec / dec_steps:.3f}")
    print(f"acceptance among drafted tokens {accepted / max(drafts, 1):.3f}")
    print(f"LLMServingSim marginal rate (accepted / (decode steps x N)): {(tok_dec / dec_steps - 1) / n:.3f}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
