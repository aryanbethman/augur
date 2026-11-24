"""How much faster is decode attention when requests share KV blocks?  Measure it and add the fit to a profile.

The profile's attention table is measured with every request on blocks of its own, so each KV byte is read from GPU
memory.  With prefix caching, requests of a batch point at the same physical blocks (a shared system prompt, a
few-shot header, a document); a block several of them read is served from the L2 cache after the first read and the
kernel finishes sooner.  ``augur.cost.tables`` prices that with a small model of the decode part of the attention
time; this module measures the data it is fitted to and writes the fit into a profile.

    python -m augur.profile_shared measure --out shots.csv [--grid main|second|third]     (GPU 0: one process, no TP)
    python -m augur.profile_shared fit shots.csv [more.csv ...] [--profile augur/profiles/<hw>/<model>/bf16-tp1.json.gz]

``measure`` times vLLM's own paged-attention kernel (flash-attn v2, as ``--attention-backend FLASH_ATTN`` runs it)
on one decode batch at a time: ``nd`` requests of KV length ``kv``; ``G`` groups of requests whose first ``P`` tokens
are the same blocks and whose remaining tokens are private; ``P = 0`` is the profile's own layout.  Four caches
rotate between calls, so another layer's traffic has evicted the L2 by the time a layer is visited again, as in a
real forward pass.  The model's head sizes are ``--q-heads/--kv-heads/--head-dim`` (default Qwen3-1.7B: 16/8/128).
``fit`` fits  ``t / t_private = min(1, (u**p + c(nd)**p) ** (1/p))``  with ``u`` = distinct KV tokens / total KV
tokens, one ``c`` per batch size on an axis of powers of two (read linearly in between), and one ``p``; with
``--profile`` it stores the result as the profile's ``shared_kv`` entry and records the measurement in its provenance
(the tables are left as they are).  Run ``measure`` under the GPU lock of a shared host.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import gzip
import hashlib
import json
import statistics
import sys
from pathlib import Path

BLOCK = 16
ROTATE = 4  # caches visited in turn: enough traffic between two visits to one layer to flush the L2
ND_AXIS = [2, 4, 8, 16, 32, 64, 128]
GRIDS = {  # (decodes, KV lengths, groups, shared fractions)
    "main": ((16, 32, 48, 64), (1024, 2048, 3072, 4096), (1, 4, 16), (0.25, 0.5, 0.75, 0.9)),
    "second": ((2, 4, 8, 12, 24, 40, 56, 96, 128), (1536, 3000), (1, 4, 16), (0.4, 0.8)),
    "third": ((8, 16, 48, 96), (3000, 2048), (1, 4), (0.06, 0.12, 0.2)),
}


def cases(grid: str) -> list[tuple[int, int, int, int]]:
    """(nd, kv, G, P) shots of a grid: every (nd, kv) once with nothing shared, then each sharing layout."""
    nds, kvs, groups, fracs = GRIDS[grid]
    out = []
    for nd in nds:
        for kv in kvs:
            out.append((nd, kv, 1, 0))
            for g in groups:
                if g <= nd:
                    out += [(nd, kv, g, int(kv * f) // BLOCK * BLOCK) for f in fracs]
    return out


def unique_fraction(nd: int, kv: int, groups: int, shared: int) -> float:
    """Distinct KV tokens over total KV tokens of a shot (1 when no two requests share a block)."""
    if shared == 0 or groups >= nd:
        return 1.0
    return (groups * shared + nd * (kv - shared)) / (nd * kv)


# --- measurement (GPU) -------------------------------------------------------

def time_shot(nd, kv, groups, shared, heads, reps=6) -> float:
    """Median ns per layer call of one decode batch (see the module docstring)."""
    import torch
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    hq, hkv, dim = heads
    dev = "cuda:0"
    tail = (kv - shared + BLOCK - 1) // BLOCK
    shared_blocks = shared // BLOCK
    nblocks = 1 + groups * shared_blocks + nd * tail
    caches = [tuple(torch.randn(nblocks, BLOCK, hkv, dim, device=dev, dtype=torch.bfloat16) for _ in range(2))
              for _ in range(ROTATE)]
    table = torch.zeros(nd, shared_blocks + tail, dtype=torch.int32)
    private = 1 + groups * shared_blocks
    for i in range(nd):
        table[i, :shared_blocks] = torch.arange(1 + i % groups * shared_blocks, 1 + (i % groups + 1) * shared_blocks)
        table[i, shared_blocks:] = torch.arange(private + i * tail, private + (i + 1) * tail)
    table = table.to(dev)
    q = torch.randn(nd, hq, dim, device=dev, dtype=torch.bfloat16)
    out = torch.empty_like(q)
    cu = torch.arange(nd + 1, dtype=torch.int32, device=dev)
    used = torch.full((nd,), kv, dtype=torch.int32, device=dev)

    def sweep():
        for k, v in caches:
            flash_attn_varlen_func(q, k, v, 1, cu, kv, seqused_k=used, block_table=table, causal=True, out=out,
                                   fa_version=2)

    for _ in range(3):
        sweep()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        sweep()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1e6 / ROTATE)
    return statistics.median(times)


def measure(out: Path, grid: str, heads: tuple[int, int, int]) -> None:
    import torch
    import vllm

    torch.manual_seed(0)
    with open(out, "w", newline="") as f:
        f.write("nd,kv,G,P,ns_per_layer\n")
        for nd, kv, g, p in cases(grid):
            try:
                ns = time_shot(nd, kv, g, p, heads)
            except torch.cuda.OutOfMemoryError:  # a private batch too big for this GPU: no row
                torch.cuda.empty_cache()
                continue
            f.write(f"{nd},{kv},{g},{p},{ns:.0f}\n")
            f.flush()
    meta = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "vllm": vllm.__version__,
            "grid": grid, "heads": list(heads), "rotating_caches": ROTATE,
            "at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1, sort_keys=True))


# --- fit (CPU) ----------------------------------------------------------------

def read_shots(paths) -> list[dict]:
    rows = []
    for path in paths:
        with open(path, newline="") as f:
            rows += [{k: int(v) for k, v in r.items()} for r in csv.DictReader(f)]
    return rows


def relative_times(rows) -> list[tuple[int, float, float]]:
    """(nd, u, t / t_private) for every shot that shares something, t_private being the same batch with
    nothing shared (the mean of the rows that measured it; shots without one are dropped)."""
    base: dict[tuple[int, int], list[int]] = {}
    for r in rows:
        if r["P"] == 0:
            base.setdefault((r["nd"], r["kv"]), []).append(r["ns_per_layer"])
    out = []
    for r in rows:
        u = unique_fraction(r["nd"], r["kv"], r["G"], r["P"])
        if u < 1 and (r["nd"], r["kv"]) in base:
            out.append((r["nd"], u, r["ns_per_layer"] / statistics.mean(base[(r["nd"], r["kv"])])))
    return out


def _node(nd: int) -> int:
    """The axis point nearest to ``nd`` (on a log scale) that the shot informs."""
    import math
    return min(ND_AXIS, key=lambda a: abs(math.log2(a) - math.log2(nd)))


def fit(rows, exponents=tuple(1.5 + 0.25 * i for i in range(27))) -> dict:
    """Fit ``min(1, (u**p + c**p) ** (1/p))`` to ``relative_times``: the exponent ``p`` and a floor ``c`` per axis
    point (grid search, least squares; shots go to the nearest axis point in log nd)."""
    import numpy as np

    data = relative_times(rows)
    if not data:
        raise ValueError("no shot with shared blocks and a private reference")
    nd = np.array([d[0] for d in data])
    u = np.array([d[1] for d in data])
    rel = np.array([d[2] for d in data])
    node = np.array([_node(n) for n in nd])
    cands = np.arange(0.2, 0.95, 0.0025)
    best = None
    for p in exponents:
        floors, err = [], 0.0
        for a in ND_AXIS:
            sel = node == a
            if not sel.any():
                floors.append(None)
                continue
            pred = np.minimum(1.0, (u[sel][:, None] ** p + cands[None, :] ** p) ** (1.0 / p))
            sse = ((pred - rel[sel][:, None]) ** 2).sum(axis=0)
            i = int(np.argmin(sse))
            floors.append(round(float(cands[i]), 4))
            err += float(sse[i])
        if best is None or err < best[0]:
            best = (err, p, floors)
    _, p, floors = best
    known = [(a, c) for a, c in zip(ND_AXIS, floors) if c is not None]
    axis = [a for a, _ in known]
    cs = [c for _, c in known]
    pred = np.minimum(1.0, (u ** p + np.interp(nd, axis, cs) ** p) ** (1.0 / p))
    err = np.abs(pred - rel)
    return {"p": p, "nd": axis, "floor": cs,
            "shots": int(len(data)), "rmse": round(float(np.sqrt(((pred - rel) ** 2).mean())), 4),
            "mean_abs_err": round(float(err.mean()), 4), "max_abs_err": round(float(err.max()), 4)}


def apply(profile: Path, result: dict, shots: list[Path], metas: list[dict]) -> None:
    """Store ``result`` as the profile's ``shared_kv`` entry and note the measurement in its provenance."""
    with gzip.open(profile, "rt", encoding="utf-8") as f:
        p = json.load(f)
    p["shared_kv"] = {"p": result["p"], "nd": result["nd"], "floor": result["floor"]}
    p.setdefault("provenance", {})["shared_kv"] = {
        "what": "decode attention with requests sharing KV blocks, relative to the same batch on private blocks: "
                "t / t_private = min(1, (u**p + c(nd)**p) ** (1/p)), u = distinct / total KV tokens "
                "(augur.profile_shared). The attention, dense, per-sequence and skew tables are not changed.",
        "fit": {k: result[k] for k in ("shots", "rmse", "mean_abs_err", "max_abs_err")},
        "measured": metas,
        "shots_sha256": [hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in shots],
    }
    data = json.dumps(p, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    profile.write_bytes(gzip.compress(data, compresslevel=9, mtime=0))  # as augur.cost.convert writes it


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="augur.profile_shared", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure", help="time shared and private decode batches on GPU 0")
    m.add_argument("--out", required=True)
    m.add_argument("--grid", choices=sorted(GRIDS), default="main")
    m.add_argument("--q-heads", type=int, default=16)
    m.add_argument("--kv-heads", type=int, default=8)
    m.add_argument("--head-dim", type=int, default=128)
    f = sub.add_parser("fit", help="fit the model to measured shots; with --profile, store it there")
    f.add_argument("shots", nargs="+")
    f.add_argument("--profile", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "measure":
        measure(Path(a.out), a.grid, (a.q_heads, a.kv_heads, a.head_dim))
        return 0
    result = fit(read_shots(a.shots))
    print(json.dumps(result))
    if a.profile:
        metas = []
        for s in a.shots:
            side = Path(s).with_suffix(".json")
            metas.append(json.loads(side.read_text()) if side.is_file() else {"file": Path(s).name})
        apply(Path(a.profile), result, [Path(s) for s in a.shots], metas)
    return 0


if __name__ == "__main__":
    sys.exit(main())
