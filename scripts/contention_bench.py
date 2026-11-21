"""PCIe / memory-bandwidth contention microbenchmarks for one GPU (augur design-doc phase 2).

(a) pinned host<->device copy bandwidth: H2D alone, D2H alone, both at once on separate streams, per chunk size
    (one Qwen3-1.7B KV block, and 1, 4, 16, 64 MB)
(b) two concurrent H2D copies on separate streams vs one alone
(c) a memory-bound decode-like kernel (bf16 GEMV/small-batch GEMM sized like Qwen3-1.7B layers, plus a big-tensor
    read) timed alone and while an H2D (and D2H) copy of --bulk-mb runs in the background

All timings use CUDA events; medians of repeats after warmups. Output: a table on stdout and --out JSON.
usage: CUDA_VISIBLE_DEVICES=0 python scripts/contention_bench.py [--model Qwen/Qwen3-1.7B] [--out out/contention.json]

Two-GPU mode (--two-gpu, needs both GPUs visible; copies only, no kernels): do two GPUs copying at once contend on the
shared host side? One thread per (GPU, direction) with its own pinned host buffer and stream, started together after a
barrier; cases: each GPU alone, both H2D, both D2H, GPU0 H2D + GPU1 D2H, all four directions, optionally with
--host-load N processes streaming memcpy on the host (host-memory contention). Prints per-GPU and aggregate GB/s.
usage: python scripts/contention_bench.py --two-gpu [--host-load 4] [--out out/contention_2gpu.json]
"""
import argparse
import json
import statistics
import sys
import threading
import time

import torch

DEV = torch.device("cuda")


def kv_block_bytes(model: str, block_size: int = 16) -> int:
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(model, local_files_only=True)
    hd = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    per_tok = 2 * c.num_hidden_layers * c.num_key_value_heads * hd * 2  # K and V, bf16
    return per_tok * block_size


def timed(fn, reps, warm=3):
    """median seconds of fn() between CUDA events on the current stream; fn must enqueue its work."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize()
        out.append(a.elapsed_time(b) / 1e3)
    return statistics.median(out), out


def copies(nbytes, reps, h2d_streams=0, d2h_streams=0, n_per=8):
    """Each stream copies n_per chunks of nbytes (one launch loop per stream); returns per-direction GB/s
    (bytes moved / wall time between the first start event and the last end event)."""
    pin = lambda: torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    dev = lambda: torch.empty(nbytes, dtype=torch.uint8, device=DEV)
    jobs = []
    for _ in range(h2d_streams):
        jobs.append(("h2d", torch.cuda.Stream(), pin(), dev()))
    for _ in range(d2h_streams):
        jobs.append(("d2h", torch.cuda.Stream(), pin(), dev()))
    res = {"h2d": [], "d2h": []}
    for it in range(reps + 3):
        torch.cuda.synchronize()
        evs = []
        gate = torch.cuda.Event(); gate.record()  # all streams start after this point
        for d, s, h, g in jobs:
            s.wait_event(gate)
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(s):
                a.record(s)
                for _ in range(n_per):
                    (g.copy_(h, non_blocking=True) if d == "h2d" else h.copy_(g, non_blocking=True))
                b.record(s)
            evs.append((d, a, b))
        torch.cuda.synchronize()
        if it < 3:
            continue
        for d in ("h2d", "d2h"):
            mine = [(a, b) for dd, a, b in evs if dd == d]
            if mine:
                origin = mine[0][0]  # events share one timeline: measure from the first stream's start
                dur = (max(origin.elapsed_time(b) for _, b in mine) - min(origin.elapsed_time(a) for a, _ in mine)) / 1e3
                res[d].append(len(mine) * n_per * nbytes / dur / 1e9)
    return {d: statistics.median(v) for d, v in res.items() if v}


def kernel_fns(hidden, inter, batch):
    """decode-like memory-bound work: one transformer layer's MLP + attention-projection GEMMs, bf16, M=batch."""
    ws = [torch.randn(hidden, hidden, device=DEV, dtype=torch.bfloat16) for _ in range(4)]
    wg = torch.randn(inter, hidden, device=DEV, dtype=torch.bfloat16)
    wu = torch.randn(inter, hidden, device=DEV, dtype=torch.bfloat16)
    wd = torch.randn(hidden, inter, device=DEV, dtype=torch.bfloat16)
    x = torch.randn(batch, hidden, device=DEV, dtype=torch.bfloat16)

    def layer():
        h = x
        for w in ws:
            h = h @ w.T
        h = ((h @ wg.T) * (h @ wu.T)) @ wd.T
        return h

    def run(n=28):  # one forward = 28 layers
        for _ in range(n):
            layer()
    layer_bytes = (4 * hidden * hidden + 3 * hidden * inter) * 2
    return run, 28 * layer_bytes


def _host_load(n, stop):
    import multiprocessing as mp
    import numpy as np

    def burn(ev):
        a = np.ones(1 << 28, dtype=np.uint8); b = np.empty_like(a)  # 256 MB each: far beyond the caches
        while not ev.is_set():
            np.copyto(b, a)
    evs = [mp.Event() for _ in range(n)]
    ps = [mp.Process(target=burn, args=(e,), daemon=True) for e in evs]
    for p in ps:
        p.start()
    stop.extend(evs)
    return ps


def two_gpu(a):
    """Concurrent copy cases over GPU 0 and GPU 1; see the module docstring."""
    n_gpu = torch.cuda.device_count()
    if n_gpu < 2:
        sys.exit("--two-gpu needs two visible GPUs")
    R = {"devices": [torch.cuda.get_device_name(i) for i in (0, 1)], "cases": {}}
    cases = [("g0 H2D alone", [(0, "h2d")]), ("g0 D2H alone", [(0, "d2h")]), ("g1 H2D alone", [(1, "h2d")]),
             ("g1 D2H alone", [(1, "d2h")]), ("both H2D", [(0, "h2d"), (1, "h2d")]),
             ("both D2H", [(0, "d2h"), (1, "d2h")]), ("g0 H2D + g1 D2H", [(0, "h2d"), (1, "d2h")]),
             ("g0 D2H + g1 H2D", [(0, "d2h"), (1, "h2d")]),
             ("all four", [(0, "h2d"), (0, "d2h"), (1, "h2d"), (1, "d2h")])]
    stop, procs = [], []
    for load in ([0, a.host_load] if a.host_load else [0]):
        if load:
            procs = _host_load(load, stop)
            time.sleep(1.0)
        print(f"\nhost memcpy load: {load} processes  (uptime: {open('/proc/loadavg').read().split()[0]} load avg)")
        print(f"{'chunk':>9}  {'case':<18}{'per-stream GB/s':<34}{'sum GB/s':>10}")
        for cname, nb in (("1.84MB", kv_block_bytes(a.model)), ("64MB", 64 << 20)):
            n_per = max(4, (64 << 20) // nb)
            for case, streams in cases:
                bufs = []
                for g, d in streams:
                    with torch.cuda.device(g):
                        bufs.append((g, d, torch.cuda.Stream(g), torch.empty(nb, dtype=torch.uint8, pin_memory=True),
                                     torch.empty(nb, dtype=torch.uint8, device=f"cuda:{g}")))
                reps = []
                for it in range(a.reps + 2):
                    bar = threading.Barrier(len(bufs)); t = [None] * len(bufs)

                    def work(i):
                        g, d, s, h, dev = bufs[i]
                        with torch.cuda.device(g), torch.cuda.stream(s):
                            for _ in range(2):  # warm this stream's path
                                dev.copy_(h, non_blocking=True) if d == "h2d" else h.copy_(dev, non_blocking=True)
                            s.synchronize(); bar.wait(); t0 = time.perf_counter(); done = 0
                            while time.perf_counter() - t0 < 1.0:  # fixed window: every stream overlaps all the others
                                for _ in range(n_per):
                                    dev.copy_(h, non_blocking=True) if d == "h2d" else h.copy_(dev, non_blocking=True)
                                s.synchronize(); done += n_per
                            t[i] = (done * nb, time.perf_counter() - t0)
                    th = [threading.Thread(target=work, args=(i,)) for i in range(len(bufs))]
                    [x.start() for x in th]; [x.join() for x in th]
                    if it >= 2:
                        rates = [b / e / 1e9 for b, e in t]
                        reps.append((rates, sum(rates)))
                per = [statistics.median(r[0][i] for r in reps) for i in range(len(bufs))]
                agg = statistics.median(r[1] for r in reps)
                lab = ", ".join(f"g{g} {d} {v:.2f}" for (g, d, *_), v in zip(bufs, per))
                print(f"{cname:>9}  {case:<18}{lab:<34}{agg:>10.2f}")
                R["cases"][f"{load}load/{cname}/{case}"] = {"per_stream": per, "aggregate": agg}
        if load:
            for e in stop:
                e.set()
    with open(a.out, "w") as f:
        json.dump(R, f, indent=1)
    print("wrote", a.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--out", default="out/contention.json")
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--bulk-mb", type=int, default=512)
    ap.add_argument("--two-gpu", action="store_true", help="two-GPU copy contention mode (see module docstring)")
    ap.add_argument("--host-load", type=int, default=0, help="two-GPU mode: also run with N host memcpy processes")
    a = ap.parse_args()
    if a.two_gpu:
        return two_gpu(a)
    torch.cuda.init()
    print("device:", torch.cuda.get_device_name(0), file=sys.stderr)
    blk = kv_block_bytes(a.model)
    sizes = [("kv_block", blk), ("1MB", 1 << 20), ("4MB", 4 << 20), ("16MB", 16 << 20), ("64MB", 64 << 20)]
    R = {"device": torch.cuda.get_device_name(0), "kv_block_bytes": blk, "copy": {}, "two_h2d": {}, "kernel": {}}

    print(f"\n(a) copy bandwidth GB/s, pinned (n chunks per stream sized so each run moves >= 256 MB)")
    print(f"{'chunk':>10}{'bytes':>12}{'h2d alone':>11}{'d2h alone':>11}{'h2d (both)':>12}{'d2h (both)':>12}"
          f"{'h2d loss':>10}{'d2h loss':>10}")
    for name, nb in sizes:
        n_per = max(8, (256 << 20) // nb)
        h = copies(nb, a.reps, 1, 0, n_per)["h2d"]
        d = copies(nb, a.reps, 0, 1, n_per)["d2h"]
        both = copies(nb, a.reps, 1, 1, n_per)
        R["copy"][name] = {"bytes": nb, "h2d": h, "d2h": d, "h2d_both": both["h2d"], "d2h_both": both["d2h"]}
        print(f"{name:>10}{nb:>12}{h:>11.2f}{d:>11.2f}{both['h2d']:>12.2f}{both['d2h']:>12.2f}"
              f"{100 * (1 - both['h2d'] / h):>9.1f}%{100 * (1 - both['d2h'] / d):>9.1f}%")

    print("\n(b) two concurrent H2D streams (GB/s per stream; aggregate)")
    print(f"{'chunk':>10}{'one stream':>12}{'per stream (x2)':>17}{'aggregate':>11}")
    for name, nb in sizes:
        n_per = max(8, (256 << 20) // nb)
        one = copies(nb, a.reps, 1, 0, n_per)["h2d"]
        two = copies(nb, a.reps, 2, 0, n_per)["h2d"]  # median of per-direction total = aggregate over both
        R["two_h2d"][name] = {"one": one, "aggregate_two": two}
        print(f"{name:>10}{one:>12.2f}{two / 2:>17.2f}{two:>11.2f}")

    print(f"\n(c) memory-bound kernel slowdown while a background copy of {a.bulk_mb} MB chunks runs")
    hidden, inter = 2048, 6144  # Qwen3-1.7B
    bulk = a.bulk_mb << 20
    hpin = torch.empty(bulk, dtype=torch.uint8, pin_memory=True)
    gb = torch.empty(bulk, dtype=torch.uint8, device=DEV)
    hpin2 = torch.empty(bulk, dtype=torch.uint8, pin_memory=True)
    gb2 = torch.empty(bulk, dtype=torch.uint8, device=DEV)
    cs, cs2 = torch.cuda.Stream(), torch.cuda.Stream()
    big = torch.empty(2 << 30, dtype=torch.uint8, device=DEV)  # 2 GiB read
    cases = [(f"decode fwd B={b}", *kernel_fns(hidden, inter, b)) for b in (1, 4, 16, 64)]
    cases.append(("2GiB sum-read", lambda: big.view(torch.int32).sum(), 2 << 30))
    print(f"{'kernel':>18}{'GB read':>9}{'alone ms':>10}{'+H2D ms':>9}{'slow':>7}{'+D2H ms':>9}{'slow':>7}"
          f"{'+both ms':>10}{'slow':>7}")
    for name, fn, rbytes in cases:
        alone, _ = timed(fn, a.reps)
        row = {"read_bytes": rbytes, "alone_ms": alone * 1e3}
        for tag, dirs in (("h2d", ("h2d",)), ("d2h", ("d2h",)), ("both", ("h2d", "d2h"))):
            t = []
            warm = 2
            for it in range(a.reps + warm):
                torch.cuda.synchronize()
                gate = torch.cuda.Event()
                a0, a1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                gate.record()
                for d, s, h, g in (("h2d", cs, hpin, gb), ("d2h", cs2, hpin2, gb2)):
                    if d in dirs:
                        s.wait_event(gate)
                        with torch.cuda.stream(s):
                            for _ in range(max(1, int(alone * 30e9 * 2 / bulk) + 1)):
                                (g.copy_(h, non_blocking=True) if d == "h2d" else h.copy_(g, non_blocking=True))
                a0.record(); fn(); a1.record()
                torch.cuda.synchronize()
                if it >= warm:
                    t.append(a0.elapsed_time(a1) / 1e3)
            row[tag + "_ms"] = statistics.median(t) * 1e3
            row[tag + "_slow_pct"] = 100 * (statistics.median(t) / alone - 1)
        R["kernel"][name] = row
        print(f"{name:>18}{rbytes / 1e9:>9.2f}{row['alone_ms']:>10.3f}{row['h2d_ms']:>9.3f}{row['h2d_slow_pct']:>6.1f}%"
              f"{row['d2h_ms']:>9.3f}{row['d2h_slow_pct']:>6.1f}%{row['both_ms']:>10.3f}{row['both_slow_pct']:>6.1f}%")
    with open(a.out, "w") as f:
        json.dump(R, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
