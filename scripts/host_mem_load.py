"""Load host memory: N processes copying 256 MB buffers in a loop (memory-bound, beyond the CPU caches).

usage: python scripts/host_mem_load.py N        (runs until interrupted or terminated)
Used by scripts/contention_bench.py (--host-load) and scripts/real_offload_check.sh (HOST_LOAD=N).
"""
import multiprocessing as mp
import signal
import sys


def _burn(stop):
    import numpy as np
    a = np.ones(1 << 28, dtype=np.uint8)
    b = np.empty_like(a)
    while not stop.is_set():
        np.copyto(b, a)


def start(n):
    """Start ``n`` copying processes; returns a function that stops them. Called by: contention_bench, main."""
    stop = mp.Event()
    procs = [mp.Process(target=_burn, args=(stop,), daemon=True) for _ in range(n)]
    for p in procs:
        p.start()

    def halt():
        stop.set()
        for p in procs:
            p.join(timeout=10)
    return halt


def main():
    halt = start(int(sys.argv[1]))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        signal.pause()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        halt()


if __name__ == "__main__":
    main()
