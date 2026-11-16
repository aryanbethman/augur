"""Build a TableCost profile from an LLMServingSimCost profile directory.

    python -m augur.cost.convert --fork DIR --hardware H --model M [--tp 1] --out FILE

The only module allowed to import the LLMServingSim fork.  Table values are
taken from the fork's own loaded tables (so the stored integers are exactly what
the fork computes with), written as deterministic gzip JSON, reloaded and
compared value by value, and spot-checked against ``LLMServingSimCost`` on random
shapes.  MoE profiles and tp > 1 are refused.

Also holds the random step-shape generator and the shape codec shared with the
tests (stdlib only at import time).

Method and data: LLMServingSim (MIT licence, Copyright (c) 2024, CASYS Lab at KAIST).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time

from augur.cost import SeqWork, StepShape

LICENSE_NOTICE = ("Derived from LLMServingSim (https://github.com/casys-kaist/LLMServingSim), "
                  "MIT License, Copyright (c) 2024, CASYS Lab at KAIST")


class Refused(Exception):
    """The profile is outside what TableCost prices."""


# --- shape codec and generator (shared with tests) --------------------------

def shape_to_row(shape: StepShape) -> list:
    """[padded or -1, dummy, [chunk, history, ...], [decode kv, ...], decode q]; prefills then decodes."""
    pre, dec, qs = [], [], set()
    for s in shape.seqs:
        if s.is_decode:
            dec.append(s.num_computed_before)
            qs.add(s.num_new)
        else:
            pre += [s.num_new, s.num_computed_before]
    if len(qs) > 1:
        raise ValueError("decodes with different query lengths are not encodable")
    return [-1 if shape.padded_tokens is None else shape.padded_tokens, int(shape.dummy), pre, dec,
            qs.pop() if qs else 1]


def shape_from_row(row) -> StepShape:
    padded, dummy, pre, dec, q = row[:5]
    seqs = tuple(SeqWork(pre[i], pre[i + 1], False) for i in range(0, len(pre), 2))
    seqs += tuple(SeqWork(q, kv, True) for kv in dec)
    return StepShape(seqs=seqs, padded_tokens=None if padded < 0 else padded, dummy=bool(dummy))


_ND = [1, 2, 3, 4, 6, 8, 11, 16, 23, 24, 32, 40, 45, 48, 56, 64, 72, 80, 88, 91, 96, 104, 112, 120, 128, 136,
       144, 152, 160, 168, 176, 181, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256]
_KV = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 24574]
_PC = [1, 16, 32, 64, 128, 256, 512, 1024, 2048]


def _kv(rng):
    m = rng.random()
    if m < 0.08:
        return 0
    if m < 0.25:
        return rng.randint(0, 64)
    if m < 0.40:
        return max(0, rng.choice(_KV) + rng.choice((-1, 0, 0, 1)))
    if m < 0.55:
        return rng.randint(0, 2048)
    if m < 0.90:
        return int(math.exp(rng.uniform(0, math.log(40001))))
    return rng.randint(0, 40000)


def _n_dec(rng):
    m = rng.random()
    if m < 0.30:
        return rng.choice(_ND)
    if m < 0.45:
        return max(1, rng.choice(_ND) + rng.choice((-1, 1)))
    if m < 0.60:
        return rng.randint(1, 256)
    if m < 0.92:
        return rng.randint(1, 12)
    return rng.randint(257, 400)


def _decode_kvs(rng, n):
    m = rng.random()
    if m < 0.22:
        return [_kv(rng)] * n
    if m < 0.40:
        base = _kv(rng)
        return [max(0, base + rng.randint(-3, 3)) for _ in range(n)]
    if m < 0.55:
        short, long_, k = _kv(rng), _kv(rng), rng.randint(1, n)
        return [long_] * k + [short] * (n - k)
    if m < 0.70:
        lo = _kv(rng)
        return [lo + rng.randint(0, rng.choice((8, 256, 4096))) for _ in range(n)]
    return [_kv(rng) for _ in range(n)]


def _prefills(rng, k):
    out = []
    for _ in range(k):
        m = rng.random()
        if m < 0.10:
            chunk = 1
        elif m < 0.25:
            chunk = rng.randint(2, 16)
        elif m < 0.50:
            chunk = rng.randint(17, 512)
        elif m < 0.60:
            chunk = max(1, rng.choice(_PC) + rng.choice((-1, 0, 0, 1)))
        elif m < 0.90:
            chunk = rng.randint(513, 2048)
        else:
            chunk = rng.randint(2049, 8192)
        out.append((chunk, 0 if rng.random() < 0.45 else _kv(rng)))
    return out


def random_shapes(rng: random.Random, n: int):
    """Yield ``n`` step shapes covering decode-only, prefill-only, mixed, dummy, empty and
    speculative (decode query length > 1) steps, equal and unequal KV, padded and not."""
    for _ in range(n):
        kind = rng.choices(("decode", "prefill", "mixed", "dummy", "empty", "spec"), (28, 14, 38, 4, 1, 6))[0]
        pre, dec, q = [], [], 1
        if kind in ("decode", "mixed", "spec"):
            dec = _decode_kvs(rng, _n_dec(rng))
        if kind in ("prefill", "mixed"):
            pre = _prefills(rng, rng.choice((1, 1, 1, 2, 2, 3, 4, 5, 8, 16, 64, 200)) if rng.random() < 0.9
                            else rng.randint(1, 6))
        if kind == "spec":
            q = rng.randint(2, 5)
        total = sum(c for c, _ in pre) + q * len(dec)
        m = rng.random()
        if kind == "dummy" or kind == "empty":
            pad = rng.choice((None, 0, 1, 7, 64, 512, 4096, rng.randint(1, 9000)))
        elif m < 0.45:
            pad = None
        elif m < 0.75:
            pad = total + rng.randint(0, max(1, total // 4))
        elif m < 0.90:
            pad = max(1, 1 << math.ceil(math.log2(max(total, 1)))) if rng.random() < 0.7 else total + rng.randint(0, 600)
        else:
            pad = rng.choice((4096, 8192, rng.randint(1, 9000)))
        seqs = [SeqWork(c, h, False) for c, h in pre] + [SeqWork(q, kv, True) for kv in dec]
        yield StepShape(seqs=tuple(seqs), padded_tokens=pad, dummy=(kind == "dummy"))


# --- reading the fork --------------------------------------------------------

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git(fork, *args):
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    return subprocess.run(["git", "-C", fork, *args], capture_output=True, text=True, env=env).stdout.strip()


def _probe(cost, tokens, seqs, dummy=False):
    shape = StepShape(seqs=tuple(seqs), padded_tokens=tokens if tokens else None, dummy=dummy)
    return [r for r in cost.trace_rows(shape) if len(r) == 11 and r[0] not in ("kv_load", "kv_evict")]


def _layers_and_remote(cost, config):
    """Layer order and the REMOTE byte rule, read from traces the fork generates."""
    layers = config["num_hidden_layers"]
    mixed = lambda t, r: ([SeqWork(1, 50 + i, True) for i in range(r - 1)] + [SeqWork(t - (r - 1), 10, False)])
    rows = _probe(cost, 0, mixed(37, 3))
    names = [r[0] for r in rows]
    if names[0] != "embedding" or names[-3:] != ["final_layernorm", "lm_head", "sampler"]:
        raise Refused(f"unexpected prologue/head order: {names[:2]} ... {names[-3:]}")
    mid = names[1:-3]
    block = mid[:len(mid) // layers]
    if len(mid) % layers or mid != block * layers:
        raise Refused("decoder layers do not repeat one block")
    for t, r in ((1, 1), (37, 3), (300, 17), (2048, 128), (5000, 2)):
        rows = _probe(cost, 0, mixed(t, r))
        if [x[0] for x in rows] != names:
            raise Refused("layer sequence depends on the batch")
        remote = [(i, f) for i, x in enumerate(rows) for f in (2, 4, 6) if x[f].startswith("REMOTE")]
        if remote != [(0, 2), (len(rows) - 1, 6)]:
            raise Refused(f"REMOTE tensors other than embedding input / sampler output: {remote}")
        if int(rows[0][3]) != 4 * t or int(rows[-1][7]) != 4 * r:
            raise Refused("REMOTE sizes are not 4 bytes per token / per sequence")
    idle = _probe(cost, 64, [], dummy=True)
    if [x[0] for x in idle] != [n for n in names[:-2] if n != "attention"] or idle[-1][6][:6] != "REMOTE" or idle[-1][7] != "0":
        raise Refused("idle (dummy) batch does not drop lm_head and sampler as expected")
    return {"prologue": names[:1], "block": block, "repeat": layers, "head": names[-3:]}


def _attention_spec(by_q):
    out = {}
    for q, tbl in sorted(by_q.items()):
        axes, slices = [], []
        for (pc, nd), sl in sorted(tbl["slices"].items()):
            keysets = {tuple(r["keys"]) for r in sl["rows"]}
            if len(keysets) != 1:
                raise Refused(f"slice ({pc},{nd}) has several KV axes")
            ks = next(iter(keysets))
            if list(ks) not in axes:
                axes.append(list(ks))
            slices.append({"pc": pc, "nd": nd, "kd": axes.index(list(ks)), "kp": list(sl["prefill_key_vals"]),
                           "ns": [list(r["values"]) for r in sl["rows"]]})
        out[str(q)] = {"kd_axes": axes, "slices": slices}
    return out


def _skew_spec(db, tp):
    fit = (db["meta"].get("skew_fit") or {})
    entries = fit.get("per_tp") or {}
    entry = entries.get(tp) or entries.get(str(tp))
    if not fit.get("enabled") or not entry:
        return None
    cal, kernels = entry["calibration"], {}
    for kernel, rng in cal["n_range_by_kernel"].items():
        layer, qs = kernel.split("|")
        kernels.setdefault(layer, {})[qs[2:]] = {"n_range": list(rng), "default": cal["alpha_default_by_kernel"][kernel],
                                                 "cells": {}}
    for name, part in cal["slices"].items():
        layer, qs, pc, lev = name.split("|")
        kernels[layer][qs[2:]]["cells"][f"{pc}|{lev}"] = {"anchors": list(part["anchors"]), "alpha": list(part["alpha"])}
    ax = cal["axes"]
    return {"pc_bins": ax["pc_bins"], "pc_labels": ax["pc_labels"], "lev_bins": ax["lev_bins"],
            "lev_labels": ax["lev_labels"], "kernels": kernels}


def load_fork(fork, hardware, model, tp):
    """(LLMServingSimCost, perf_db, config) for the profile, refusing out-of-scope ones."""
    if tp != 1:
        raise Refused(f"tp={tp}: TableCost prices TP=1 only")
    from augur.cost.llmservingsim import LLMServingSimCost
    cost = LLMServingSimCost(fork, hardware, model, tp=1)
    cost.step_ns(StepShape(seqs=(SeqWork(1, 10, True),)))  # loads the profile into the fork's cache
    tg = cost._tg
    config = tg.get_config(model)
    if tg.utils_num_experts(config):
        raise Refused(f"{model} is a MoE model: TableCost prices dense models only")
    db = tg._perf_db_cache[(hardware, model, tg.resolve_variant(config))]
    tables = tg._tp_tables(db, 1)
    extra = set(tables) - {"dense", "per_sequence", "attention_by_layer"}
    if extra:
        raise Refused(f"profile has tables TableCost does not price: {sorted(extra)}")
    if db["architecture"].get("component_layouts") or db["component_sizes"].get(1) or "TPU" in hardware:
        raise Refused("profile uses component layouts / contextual sizes / TPU rules")
    if db["key_saturation_by_tp"][1] is not None:
        raise Refused("sparse (key-saturating) attention is not supported")
    if len(set(db["layer_stack"])) > 1:
        raise Refused("heterogeneous layer stack")
    return cost, db, config, tables, tg


def build_profile(fork, hardware, model, tp=1) -> dict:
    fork = os.path.abspath(fork)
    cost, db, config, tables, tg = load_fork(fork, hardware, model, tp)
    variant = db["variant"]
    seq = _layers_and_remote(cost, config)
    used = set(seq["prologue"] + seq["block"] + seq["head"])
    attn_names = {n for n in used if n in tables["attention_by_layer"]}
    per_seq = {n: t for n, t in tables["per_sequence"].items() if n in used}
    dense = {n: t for n, t in tables["dense"].items() if n in used}
    if used != attn_names | set(per_seq) | set(dense) or len(attn_names) + len(per_seq) + len(dense) != len(used):
        raise Refused("layer names are not covered by exactly one table each")
    rel = os.path.join("profiler", "perf", hardware, model, variant, f"tp{tp}")
    cfg_rel = os.path.join("configs", "model", model + ".json")
    sources = {}
    for name in ("dense.csv", "per_sequence.csv", "attention.csv", "skew_fit.csv"):
        path = os.path.join(fork, rel, name)
        if os.path.isfile(path):
            sources[name] = _sha256(path)
    with open(os.path.join(fork, cfg_rel), encoding="utf-8") as f:
        model_config = json.load(f)
    meta = db["meta"]
    prov = {"license": LICENSE_NOTICE, "source_profile": rel, "source_config": cfg_rel,
            "source_sha256": sources, "fork_commit": _git(fork, "rev-parse", "HEAD"),
            "fork_sources_modified": bool(_git(fork, "status", "--porcelain", "--", rel, cfg_rel))}
    for key in ("vllm_version", "profiled_at", "gpu", "profiler_version"):
        if meta.get(key):
            prov[key] = meta[key]
    return {
        "format_version": 1,
        "identity": {"hardware": hardware, "model": model, "variant": variant, "tp": tp, "kind": "dense",
                     "architecture": meta.get("architecture")},
        "provenance": prov,
        "layers": seq,
        "dense": {n: {"keys": t["keys"], "ns": t["values"]} for n, t in sorted(dense.items())},
        "per_sequence": {n: {"keys": t["keys"], "ns": t["values"]} for n, t in sorted(per_seq.items())},
        "attention": {n: _attention_spec(tables["attention_by_layer"][n]) for n in sorted(attn_names)},
        "skew": _skew_spec(db, tp),
        "remote": {"input_bytes_per_token": 4, "output_bytes_per_seq": 4},
        "model_config": model_config,
    }


def write_profile(profile, out):
    data = json.dumps(profile, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    blob = gzip.compress(data, compresslevel=9, mtime=0)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "wb") as f:
        f.write(blob)
    os.replace(tmp, out)
    return len(data), len(blob)


# --- self-check --------------------------------------------------------------

def _same(a, b, path=""):
    """Strict equality: same types and values; list order matters, dict order does not."""
    if type(a) is not type(b):
        return [f"{path}: type {type(a).__name__} != {type(b).__name__}"]
    if isinstance(a, dict):
        if set(a) != set(b):  # key order is not semantic (the file is written with sorted keys)
            return [f"{path}: keys differ"]
        return [m for k in a for m in _same(a[k], b[k], f"{path}/{k}")]
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += _same(x, y, f"{path}[{i}]")
            if len(out) > 5:
                break
        return out
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def compare_tables(path, db, tables, cost_tg) -> list[str]:
    """Reload ``path`` and compare every stored value with the fork's loaded tables."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        p = json.load(f)
    bad = []
    for cat, key in (("dense", "dense"), ("per_sequence", "per_sequence")):
        got = {n: {"keys": t["keys"], "values": t["ns"]} for n, t in p[cat].items()}
        want = {n: t for n, t in tables[key].items() if n in got}
        bad += _same(dict(sorted(got.items())), dict(sorted(want.items())), cat)
        if set(got) != set(tables[key]):
            bad.append(f"{cat}: names {sorted(got)} != fork {sorted(tables[key])}")
    for layer, by_q in p["attention"].items():
        fork_q = tables["attention_by_layer"][layer]
        if sorted(map(int, by_q)) != sorted(fork_q):
            bad.append(f"attention {layer}: q values differ")
        for q, spec in by_q.items():
            want = fork_q[int(q)]["slices"]
            got = {(s["pc"], s["nd"]): {"prefill_key_vals": s["kp"],
                                        "rows": [{"keys": spec["kd_axes"][s["kd"]], "values": r} for r in s["ns"]]}
                   for s in spec["slices"]}
            bad += _same(dict(sorted(got.items())), dict(sorted(want.items())), f"attention/{layer}/{q}")
    want_skew = _skew_spec(db, p["identity"]["tp"])
    bad += _same(p["skew"], want_skew, "skew")
    fit = (db["meta"].get("skew_fit") or {}).get("per_tp") or {}
    entry = fit.get(1) or fit.get("1")
    if p["skew"]:
        cal = entry["calibration"]
        cells = sum(len(k["cells"]) for by_q in p["skew"]["kernels"].values() for k in by_q.values())
        if cells != len(cal["slices"]):
            bad.append("skew: cell count differs from fork")
        for kernel, rng in cal["n_range_by_kernel"].items():
            layer, qs = kernel.split("|")
            k = p["skew"]["kernels"][layer][qs[2:]]
            if k["n_range"] != list(rng) or k["default"] != cal["alpha_default_by_kernel"][kernel]:
                bad.append(f"skew {kernel}: range/default differ")
        for name, part in cal["slices"].items():
            layer, qs, pc, lev = name.split("|")
            c = p["skew"]["kernels"][layer][qs[2:]]["cells"][f"{pc}|{lev}"]
            bad += _same(c, {"anchors": part["anchors"], "alpha": part["alpha"]}, f"skew/{name}")
        ax = cal["axes"]
        for k in ("pc_bins", "pc_labels", "lev_bins", "lev_labels"):
            bad += _same(p["skew"][k], ax[k], f"skew/{k}")
    return bad


def spot_check(path, fork, hardware, model, n, seed, **cpu):
    from augur.cost.llmservingsim import LLMServingSimCost
    from augur.cost.tables import TableCost
    ref = LLMServingSimCost(fork, hardware, model, cpu_mem_bw_gbps=cpu.get("bw", 256.0),
                            cpu_mem_latency_ns=cpu.get("lat", 0))
    mine = TableCost(path, cpu_mem_bw=cpu.get("bw", 256.0), cpu_mem_latency_ns=cpu.get("lat", 0))
    shapes = list(random_shapes(random.Random(seed), n))
    t0 = time.perf_counter()
    want = [ref.step_ns(s) for s in shapes]
    t1 = time.perf_counter()
    got = [mine.step_ns(s) for s in shapes]
    t2 = time.perf_counter()
    bad = [(s, w, g) for s, w, g in zip(shapes, want, got) if w != g]
    return bad, (t1 - t0) / n, (t2 - t1) / n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m augur.cost.convert", description=__doc__.splitlines()[0])
    ap.add_argument("--fork", required=True, help="LLMServingSim fork checkout")
    ap.add_argument("--hardware", required=True)
    ap.add_argument("--model", required=True, help="e.g. meta-llama/Llama-3.1-8B")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--spot-checks", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args(argv)
    fork = os.path.abspath(a.fork)
    try:
        profile = build_profile(fork, a.hardware, a.model, a.tp)
    except Refused as e:
        print(f"convert: refused: {e}", file=sys.stderr)
        return 2
    raw, packed = write_profile(profile, a.out)
    print(f"wrote {a.out}: {raw} bytes JSON, {packed} bytes gzip")

    cost, db, config, tables, tg = load_fork(fork, a.hardware, a.model, a.tp)
    bad = compare_tables(a.out, db, tables, tg)
    print(f"table comparison: {'OK' if not bad else 'MISMATCH'}")
    for m in bad[:10]:
        print("  ", m)
    failures = len(bad)
    for label, cpu in (("default CPU memory", {}), ("bw=3.7 latency=123", {"bw": 3.7, "lat": 123})):
        n = a.spot_checks if not cpu else max(500, a.spot_checks // 4)
        mism, t_ref, t_mine = spot_check(a.out, fork, a.hardware, a.model, n, a.seed, **cpu)
        print(f"spot check ({label}): {n} shapes, {len(mism)} mismatches; "
              f"fork {t_ref * 1e6:.0f} us/shape, TableCost {t_mine * 1e6:.0f} us/shape")
        for s, w, g in mism[:3]:
            print("  ", shape_to_row(s), "fork", w, "table", g)
        failures += len(mism)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
