"""Run augur --cost llmservingsim and record every distinct StepShape passed to step_ns.

  python scripts/collect_step_shapes.py OUT.json <augur arguments, with --cost llmservingsim>

Input for scripts/build_table_regression.py."""
import json, sys, os
from augur.cost.llmservingsim import LLMServingSimCost
from augur.cost.convert import shape_to_row
out = sys.argv[1]; argv = sys.argv[2:]
seen = {}; calls = [0]
orig = LLMServingSimCost.step_ns
def rec(self, shape):
    calls[0] += 1
    dec = [s.num_computed_before for s in shape.seqs if s.is_decode]
    pre = tuple((s.num_new, s.num_computed_before) for s in shape.seqs if not s.is_decode)
    dq = max((s.num_new for s in shape.seqs if s.is_decode), default=1)
    key = (pre, len(dec), dq, sum(dec)//len(dec) if dec else 0, max(dec, default=0), shape.forward_tokens, 0 if shape.dummy else len(shape.seqs))
    if key not in seen: seen[key] = shape_to_row(shape)
    return orig(self, shape)
LLMServingSimCost.step_ns = rec
from augur.cli import main
rc = main(argv)
json.dump({"argv": argv, "calls": calls[0], "rows": list(seen.values())}, open(out, "w"))
print("rc", rc, "calls", calls[0], "distinct", len(seen))
