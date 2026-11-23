"""Fit --hbm-interference K on a real offload run (scripts/real_offload_check.sh).

Replays the run's step log through augur with the fabric on (transfers end when the log says), takes each step's
time-averaged PCIe rate on GPU 0, and compares the real step's measured / predicted time (stepcheck, the profile prices
GPU work alone) with steps that saw no transfer, per step class: K = (slowdown) / (rate / HBM bandwidth).

usage: python scripts/fit_hbm_interference.py STEPCHECK.csv RATES_OUT.json -- <augur.cli arguments of the run with
       --replay-durations STEPLOG --contention on --hbm-bw-gbps 288 --step-log STEPS.csv>
STEPCHECK.csv: python -m augur.verify.stepcheck STEPLOG ... --csv STEPCHECK.csv. The HBM bandwidth below is the RTX
4060 Ti's (288 GB/s).
"""
import bisect, csv, json, sys
import numpy as np
from augur.sim import fabric

check_csv, rates_out = sys.argv[1], sys.argv[2]
argv = sys.argv[sys.argv.index("--") + 1:]
timeline = []  # (t, bytes/ns through gpu0.h2d + gpu0.d2h)
orig = fabric.Fabric._recompute
def rec(self, touched):
    orig(self, touched)
    r = self.rates_through("gpu0.h2d") + self.rates_through("gpu0.d2h")
    if timeline and timeline[-1][0] == self.time:
        timeline[-1] = (self.time, r)
    else:
        timeline.append((self.time, r))
fabric.Fabric._recompute = rec
from augur.cli import main
rc = main(argv)
step_log = argv[argv.index("--step-log") + 1]
steps = list(csv.DictReader(open(step_log)))
ts = [t for t, _ in timeline]

def avg_rate(a, b):
    if b <= a or not timeline:
        return 0.0
    i = max(0, bisect.bisect_right(ts, a) - 1)
    tot, t = 0.0, a
    while t < b:
        r = timeline[i][1] if timeline[i][0] <= t else 0.0
        nxt = ts[i + 1] if i + 1 < len(ts) else b
        end = min(b, nxt)
        if end > t:
            tot += r * (end - t)
            t = end
        i += 1
        if i >= len(ts):
            tot += timeline[-1][1] * (b - t) if t < b else 0
            break
    return tot / (b - a)

keys = list(steps[0].keys())
start_k = next(k for k in keys if "start" in k); dur_k = next(k for k in keys if "dur" in k)
rates = [avg_rate(int(s[start_k]), int(s[start_k]) + int(s[dur_k])) for s in steps]
chk = list(csv.DictReader(open(check_csv)))
# align: both lists are in step order; augur's step log leaves out some steps (match on total tokens)
pairs, j = [], 0
for i, c in enumerate(chk):
    for jj in range(j, min(j + 8, len(steps))):  # resync after a few differing steps
        if int(steps[jj]["total_tokens"]) == int(c["tokens"]):
            pairs.append((i, jj)); j = jj + 1
            break
print("steps: augur", len(steps), "stepcheck", len(chk), "aligned", len(pairs), "timeline points", len(timeline),
      file=sys.stderr)
json.dump({"pairs": pairs, "rates_gbps": rates}, open(rates_out, "w"))
for cls in ("decode", "mixed"):
    x = np.array([rates[j] / 288.0 for i, j in pairs if chk[i]["class"] == cls])
    y = np.array([int(chk[i]["measured_ns"]) / int(chk[i]["predicted_ns"]) for i, j in pairs if chk[i]["class"] == cls])
    base = np.median(y[x == 0]) if (x == 0).any() else 1.0
    busy = x > 0
    k_ols = np.polyfit(x, y / base, 1)[0]
    # robust: median of (y/base - 1) / x over steps with real traffic (x > 1% of HBM)
    sel = x > 0.01
    k_med = float(np.median((y[sel] / base - 1) / x[sel])) if sel.any() else float("nan")
    print(f"{cls}: {len(y)} steps, {busy.sum()} with PCIe traffic, baseline meas/pred {base:.4f}, "
          f"K (OLS slope) {k_ols:.2f}, K (median ratio, rate > 1% HBM: {sel.sum()} steps) {k_med:.2f}")
    for lo, hi in ((0, 1e-9), (1e-9, 0.02), (0.02, 0.04), (0.04, 0.06), (0.06, 1)):
        m = (x >= lo) & (x < hi) if lo else (x == 0)
        if m.any():
            print(f"   rate/HBM in [{lo:.2f},{hi:.2f}): {m.sum():6d} steps, median meas/pred / baseline "
                  f"{np.median(y[m]) / base:.4f}, mean rate/HBM {x[m].mean():.4f}")
sys.exit(rc)
