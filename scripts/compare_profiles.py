"""Step prices of two profiles on a few shapes. usage: python scripts/compare_profiles.py A.json.gz B.json.gz"""
import sys, logging
logging.disable(logging.WARNING)
from augur.cost import SeqWork, StepShape
from augur.cost.tables import TableCost
a = TableCost(sys.argv[1]); b = TableCost(sys.argv[2])
S = lambda n, c, d: SeqWork(num_new=n, num_computed_before=c, is_decode=d)
shapes = {
 "decode b=1 kv=512": [S(1, 512, True)],
 "decode b=16 kv=1024": [S(1, 1024, True)] * 16,
 "decode b=64 kv=1024": [S(1, 1024, True)] * 64,
 "decode b=128 kv=2048": [S(1, 2048, True)] * 128,
 "prefill 512": [S(512, 0, False)],
 "prefill 2048": [S(2048, 0, False)],
 "prefill 1024 @kv2048": [S(1024, 2048, False)],
 "mixed 32 dec kv1024 + 512 chunk": [S(1, 1024, True)] * 32 + [S(512, 0, False)],
 "spec q=4 b=16 kv=1024": [S(4, 1024, True)] * 16,
 "spec q=4 b=48 kv=1200": [S(4, 1200, True)] * 48,
 "spec q=4 b=48 kv=3000": [S(4, 3000, True)] * 48,
 "spec q=4 b=128 kv=1200": [S(4, 1200, True)] * 128,
}
print(f"{'shape':34}{'shipped ms':>11}{'pilot ms':>10}{'ratio':>8}")
for k, v in shapes.items():
    x = a.step_ns(StepShape(seqs=tuple(v))) / 1e6; y = b.step_ns(StepShape(seqs=tuple(v))) / 1e6
    print(f"{k:34}{x:>11.3f}{y:>10.3f}{y / x:>8.3f}")
