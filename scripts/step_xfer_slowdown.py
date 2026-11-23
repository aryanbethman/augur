"""Are real steps slower while offload transfers run? From `augur.verify.stepcheck --csv` of a run with KV offload.

usage: python scripts/step_xfer_slowdown.py STEPCHECK.csv
A step whose update reported finished loads or stores overlapped those transfers (at least their end). Per step class,
measured / predicted (the profile prices GPU work alone) is compared between steps with and without transfers; the
difference is what transfers added to the step. augur's --hbm-interference K predicts 1 + K x PCIe GB/s / HBM GB/s
(RTX 4060 Ti: 12.7 / 288 x 1.4 = +6.2% one way).
"""
import csv
import statistics
import sys


def main(path):
    rows = list(csv.DictReader(open(path)))
    print(f"{'class':<8}{'transfers':<11}{'steps':>7}{'median meas/pred':>18}{'mean meas/pred':>16}")
    for cls in ("decode", "mixed", "prefill"):
        for label, keep in (("none", lambda r: r["load_bytes"] == r["store_bytes"] == "0"),
                            ("load", lambda r: r["load_bytes"] != "0" and r["store_bytes"] == "0"),
                            ("store", lambda r: r["store_bytes"] != "0" and r["load_bytes"] == "0"),
                            ("both", lambda r: r["load_bytes"] != "0" and r["store_bytes"] != "0")):
            ratio = [int(r["measured_ns"]) / int(r["predicted_ns"]) for r in rows if r["class"] == cls and keep(r)]
            if ratio:
                print(f"{cls:<8}{label:<11}{len(ratio):>7}{statistics.median(ratio):>18.3f}"
                      f"{statistics.fmean(ratio):>16.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
