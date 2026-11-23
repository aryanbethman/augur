"""Build a TableCost profile from an LLMServingSimCost profile directory.

    python -m augur.cost.convert --fork DIR --hardware H --model M [--tp N --cluster-config FILE] --out FILE

The only module allowed to import the LLMServingSim fork.  Table values are
taken from the fork's own loaded tables (so the stored integers are exactly what
the fork computes with), written as deterministic gzip JSON, reloaded and
compared value by value, and spot-checked against ``LLMServingSimCost`` on random
shapes.

MoE models: the MoE block of a decoder layer is a function of the forward token count only (the fork's
balanced gate is deterministic in it), so with native component tables (TP=1) the profile stores the
fork's own price at every token count 1..N (the fork builds it from a scipy Delaunay surface; this
module reads it back from the fork's traces, so TableCost needs no scipy), and for the legacy whole-block
table (TP>1) the ep=1 rows and the closed form of the balanced gate.  Every token count is then compared
with the fork's trace.

At TP = N > 1 the profile holds the tables the fork uses at that TP (``tp_stable`` layers resolved as the
fork does), the collective coefficients (derived from the fork's traces at several (T, R), which must be
exactly linear) and, when the fork defines one, the NPU memory bandwidth default.  The cluster config is
needed for the self-check, which prices random shapes with ASTRA-Sim (one binary run per distinct (T, R)).

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


def _layers_and_remote(cost, config, tmax=None):
    """Layer order and the REMOTE byte rule, read from traces the fork generates (token counts up to ``tmax``)."""
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
        if tmax is not None and t > tmax:
            continue
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


def _resolved(tg, db, tp, tables):
    """The dense / per-sequence tables the fork reads at ``tp``: ``tp_stable`` layers come from tp1 when it exists."""
    out = {"dense": {}, "per_sequence": {}, "attention_by_layer": tables["attention_by_layer"]}
    for cat in ("dense", "per_sequence"):
        names = set(tables[cat])
        if tp > 1 and 1 in db["available_tps"]:
            names |= {n for n in tg._tp_tables(db, 1)[cat] if tg._tp_stable(db, cat, n)}
        for n in sorted(names):
            out[cat][n] = tg._tp_tables(db, tg._effective_tp(db, cat, n, tp))[cat][n]
    return out


def load_fork(fork, hardware, model, tp, cluster_config=None, variant=None):
    """(LLMServingSimCost, perf_db, config) for the profile, refusing out-of-scope ones."""
    from augur.cost.llmservingsim import LLMServingSimCost
    if tp > 1 and not cluster_config:
        raise Refused(f"tp={tp} needs --cluster-config (the fork's cost model is built with it)")
    cost = LLMServingSimCost(fork, hardware, model, tp=tp, cluster_config=cluster_config, profile_variant=variant)
    cost.trace_rows(StepShape(seqs=(SeqWork(1, 10, True),)))  # loads the profile into the fork's cache (no ASTRA run)
    tg = cost._tg
    config = tg.get_config(model)
    db = tg._perf_db_cache[(hardware, model, tg.resolve_variant(config))]
    tables = tg._tp_tables(db, tp)
    extra = set(tables) - {"dense", "per_sequence", "attention_by_layer"} - ({"moe"} if tg.utils_num_experts(config) else set())
    if extra:
        raise Refused(f"profile has tables TableCost does not price: {sorted(extra)}")
    if db["architecture"].get("component_layouts") or db["component_sizes"].get(tp) or "TPU" in hardware:
        raise Refused("profile uses component layouts / contextual sizes / TPU rules")
    if db["key_saturation_by_tp"][tp] is not None:
        raise Refused("sparse (key-saturating) attention is not supported")
    if len(set(db["layer_stack"])) > 1:
        raise Refused("heterogeneous layer stack")
    return cost, db, config, _resolved(tg, db, tp, tables), tg


def _collectives(cost, layers, tmax=None):
    """(all-reduce count, bytes per forward token, all-gather bytes per sequence) read from the fork's traces;
    every sample must agree exactly (linear in T and R, no all-gather without sequences)."""
    mixed = lambda t, r: [SeqWork(1, 50 + i, True) for i in range(r - 1)] + [SeqWork(t - (r - 1), 10, False)]
    seen = set()
    for t, r in ((1, 1), (37, 3), (300, 17), (2048, 128), (5000, 2), (64, 64), (4096, 1), (9, 9)):
        if tmax is not None and t > tmax:
            continue
        rows = _probe(cost, 0, mixed(t, r))
        kinds = [(x[8], int(x[9])) for x in rows if x[8] != "NONE"]
        if any(k not in ("ALLREDUCE", "ALLGATHER") for k, _ in kinds):
            raise Refused(f"collectives other than plain ALLREDUCE / ALLGATHER: {sorted({k for k, _ in kinds})}")
        ar = [b for k, b in kinds if k == "ALLREDUCE"]
        ag = [b for k, b in kinds if k == "ALLGATHER"]
        if len(set(ar)) != 1 or len(ag) != 1 or ar[0] % t or ag[0] % r:
            raise Refused(f"collective sizes at T={t}, R={r} are not one size per kind, proportional to T / R")
        seen.add((len(ar), ar[0] // t, ag[0] // r))
    idle = _probe(cost, 64, [], dummy=True)
    ar = [int(x[9]) for x in idle if x[8] == "ALLREDUCE"]
    if len({c for c, _, _ in seen}) != 1 or len(seen) != 1 or [x for x in idle if x[8] == "ALLGATHER"] or \
            ar != [64 * next(iter(seen))[1]] * next(iter(seen))[0]:
        raise Refused(f"collectives are not linear in (T, R): {sorted(seen)}")
    n, per_token, per_seq = next(iter(seen))
    return {"all_reduce_count": n, "all_reduce_bytes_per_token": per_token, "all_gather_bytes_per_seq": per_seq}


def _npu_mem_bw_default(fork, hardware):
    """The NPU memory bandwidth the fork fills into a cluster config that omits it (None: no default)."""
    cwd = os.getcwd()
    os.chdir(os.path.join(fork, "serving"))
    try:
        from serving.core.hardware_defaults import _defaults_for
        entry = ((_defaults_for(hardware).get("npu_mem") or {}).get("mem_bw"))
    finally:
        os.chdir(cwd)
    return entry["value"] if isinstance(entry, dict) and "value" in entry else None


MOE_LAYERS = {"moe_gate_routing", "moe_experts", "expert"}
MOE_TOKEN_LIMIT = 1 << 16


def _moe_prices(cost, tokens, names):
    """{name: ns} of the MoE layers in the fork's trace for one prefill of ``tokens`` tokens; every decoder
    layer must carry the same value (the fork emits the block once and copies it)."""
    rows = cost.trace_rows(StepShape(seqs=(SeqWork(tokens, 0, False),)))
    out = {}
    for r in rows:
        if len(r) == 11 and r[0] in names:
            if out.setdefault(r[0], int(r[1])) != int(r[1]):
                raise Refused(f"MoE layers differ within one step at T={tokens}")
    return out


def _moe_section(cost, tg, db, config, names):
    """The profile's ``moe`` tables (see ``augur.cost.tables``)."""
    if names == {"expert"}:  # legacy whole-block table (TP>1 bundles without native components)
        by_ep = tg._tp_tables(db, 1 if 1 in db["available_tps"] else db["available_tps"][0]).get("moe")
        if not by_ep or 1 not in by_ep:
            raise Refused("legacy MoE table has no ep=1 rows")
        tbl = by_ep[1]
        return {"balanced": {"expert": {
            "experts": tg.utils_num_experts(config), "top_k": tg._stack_module().expert_top_k(config),
            "activated": list(tbl["activated_experts_vals"]),
            "rows": [{"keys": list(r["keys"]), "ns": list(r["values"])} for r in tbl["rows"]]}}}
    table = {n: [] for n in names}
    for t in range(1, MOE_TOKEN_LIMIT + 1):
        try:
            prices = _moe_prices(cost, t, names)
        except Exception as e:  # the fork's MoeCoverageError: the component tables stop here
            if type(e).__name__ != "MoeCoverageError":
                raise
            break
        for n in names:
            table[n].append(prices[n])
    if len(table[next(iter(names))]) in (0, MOE_TOKEN_LIMIT):
        raise Refused("MoE component tables: no usable token range")
    return {"exact": table}


DP_MOE = ("moe_gate_routing", "moe_experts", "moe_finalize_copy")


def _dp_rank_rows(ref, counts, rank):
    """One rank's trace rows in a DP round whose ranks run prefills of ``counts`` tokens (the fork's DP path)."""
    from augur.cost import SeqWork, StepShape
    shapes = [StepShape(seqs=(SeqWork(c, 0, False),)) for c in counts]
    inst = ref.oracle.cluster["instances"][rank]
    cwd = os.getcwd()
    os.chdir(ref._run_dir)
    try:
        tr = ref._tg.generate_trace(ref._batch(shapes[rank]), ref.hardware, inst["tp_size"], inst.get("pp_size", 1),
                                    inst["local_ep"], inst["ep_total"], max_num_batched_tokens=ref.max_num_batched_tokens,
                                    dtype=ref.dtype, placement=ref._placement, enable_block_copy=True,
                                    tp_dim=inst.get("tp_dim"), ep_dim=inst.get("ep_dim"),
                                    dp_sum_total_len=sum(counts), dp_min_total_len=min(counts),
                                    dp_token_counts=tuple(counts), dp_rank=rank)
    finally:
        os.chdir(cwd)
    return [r for r in tr.rows if not (len(r) == 11 and r[0] in ("kv_load", "kv_evict"))]


def _dp_layout(rows):
    """(MoE prices {name: ns}, per-layer names before the MoE block, all-gather sizes, reduce-scatter size) of a DP
    rank's trace; refuses a trace whose layers differ or whose block is not gate | all-gathers | experts |
    reduce-scatter | finalize."""
    prices, pre, ag, rs, layer, seq = {}, None, None, None, [], []
    for r in rows:
        if len(r) == 11:
            if r[0] in DP_MOE and prices.setdefault(r[0], int(r[1])) != int(r[1]):
                raise Refused("DP MoE layers differ within one step")
            seq.append(r[0])
        elif r[0].startswith("EXPERT END") and r[0] != "EXPERT END NONE 0":
            f = r[0].split()[2:]
            kinds = [(f[k].split(":")[0], int(f[k + 1])) for k in range(0, len(f), 2)]
            if all(k == "ALLGATHER" for k, _ in kinds):
                if ag not in (None, [b for _, b in kinds]):
                    raise Refused("DP all-gathers differ between layers")
                ag = [b for _, b in kinds]
                seq.append("|AG")
            elif [k for k, _ in kinds] == ["REDUCESCATTER"]:
                if rs not in (None, kinds[0][1]):
                    raise Refused("DP reduce-scatters differ between layers")
                rs = kinds[0][1]
                seq.append("|RS")
            else:
                raise Refused(f"unexpected DP collectives {kinds}")
    try:
        g = seq.index("moe_gate_routing")
    except ValueError:
        raise Refused("no moe_gate_routing in the DP trace") from None
    pre = seq[1:g]  # after the embedding
    unit = pre + ["moe_gate_routing", "|AG", "moe_experts", "|RS", "moe_finalize_copy"]
    body = seq[1:]
    n = body.count("moe_experts")
    if body[:len(unit) * n] != unit * n:
        raise Refused("DP trace layers are not [pre, gate, all-gathers, experts, reduce-scatter, finalize] each")
    return prices, pre, ag, rs, n, body[len(unit) * n:]


def _dp_section(fork, hardware, model, dp_cluster_config, variant=None):
    """DP2 + EP2 (TP1 ranks) tables for ``TableCost.round_ns``: per rank, the gate and finalize at its own token
    count (1..2048) and the experts at the round's summed count (2..4096), read off the fork's traces, and the
    collectives' bytes per token. Every value is checked to depend on those counts only."""
    from augur.cost.llmservingsim import LLMServingSimCost
    with open(dp_cluster_config, encoding="utf-8") as f:
        c = json.load(f)
    insts = [i for node in c["nodes"] for i in node["instances"]]
    if len(c["nodes"]) != 1 or len(insts) != 2 or any(i.get("tp_size", 1) != 1 or i.get("ep_size") != 2
                                                      or (i.get("pp_size") or 1) != 1 for i in insts):
        raise Refused("--dp-cluster-config: one node with two TP1 instances of ep_size 2 (DP2 + EP2)")
    ref = LLMServingSimCost(fork, hardware, model, tp=1, cluster_config=os.path.abspath(dp_cluster_config), dp=2,
                            ep=2, profile_variant=variant)
    top = 2048
    ranks = [{n: [None] * (top if n != "moe_experts" else 2 * top - 1) for n in DP_MOE} for _ in range(2)]
    layout = None
    for rank in (0, 1):
        def counts(mine, other):
            return (mine, other) if rank == 0 else (other, mine)
        for t in range(1, top + 1):
            prices, *lay = _dp_layout(_dp_rank_rows(ref, counts(t, 1), rank))
            if layout is None:
                layout = lay
            ranks[rank]["moe_gate_routing"][t - 1] = prices["moe_gate_routing"]
            ranks[rank]["moe_finalize_copy"][t - 1] = prices["moe_finalize_copy"]
            ranks[rank]["moe_experts"][t - 1] = prices["moe_experts"]  # summed count t + 1
        for u in range(2, top + 1):
            prices, *_ = _dp_layout(_dp_rank_rows(ref, counts(top, u), rank))
            ranks[rank]["moe_experts"][top + u - 2] = prices["moe_experts"]  # summed count top + u
    pre, ag, rs, layers, tail = layout
    probe = _dp_layout(_dp_rank_rows(ref, (37, 5), 0))
    if len(ag) != 3 or any(b % 37 for b in probe[2]) or probe[3] % (2 * 37):
        raise Refused("DP collective sizes are not proportional to the round's largest token count")
    return {"2": {"tp": 1, "ep": 2, "pre": pre, "layers": layers, "tail": tail,
                  "all_gather_bytes_per_token": [b // 37 for b in probe[2]],
                  "reduce_scatter_bytes_per_token": probe[3] // (2 * 37), "max_tokens": top,
                  "ranks": ranks}}


def check_dp_section(profile, fork, hardware, model, dp_cluster_config, n=400, seed=11, variant=None):
    """Random (count, count) pairs: every MoE row of both ranks' traces against the profile's DP tables. Returns
    the mismatches."""
    from augur.cost.llmservingsim import LLMServingSimCost
    ref = LLMServingSimCost(fork, hardware, model, tp=1, cluster_config=os.path.abspath(dp_cluster_config), dp=2,
                            ep=2, profile_variant=variant)
    spec = profile["moe"]["dp"]["2"]
    rng = random.Random(seed)
    bad = []
    for _ in range(n):
        cs = (rng.randint(1, 2048), rng.randint(1, 2048))
        for rank in (0, 1):
            prices, pre, ag, rs, layers, tail = _dp_layout(_dp_rank_rows(ref, cs, rank))
            want = {"moe_gate_routing": spec["ranks"][rank]["moe_gate_routing"][cs[rank] - 1],
                    "moe_finalize_copy": spec["ranks"][rank]["moe_finalize_copy"][cs[rank] - 1],
                    "moe_experts": spec["ranks"][rank]["moe_experts"][sum(cs) - 2]}
            if prices != want or ag != [b * max(cs) for b in spec["all_gather_bytes_per_token"]] or \
                    rs != 2 * max(cs) * spec["reduce_scatter_bytes_per_token"] or pre != spec["pre"]:
                bad.append((cs, rank, prices, want))
    return bad


def build_profile(fork, hardware, model, tp=1, cluster_config=None, variant=None, provenance=None,
                  dp_cluster_config=None) -> dict:
    fork = os.path.abspath(fork)
    cost, db, config, tables, tg = load_fork(fork, hardware, model, tp, cluster_config, variant)
    variant = db["variant"]
    is_moe = bool(tg.utils_num_experts(config))
    seq = _layers_and_remote(cost, config, tmax=1000 if is_moe else None)
    used = set(seq["prologue"] + seq["block"] + seq["head"])
    attn_names = {n for n in used if n in tables["attention_by_layer"]}
    per_seq = {n: t for n, t in tables["per_sequence"].items() if n in used}
    dense = {n: t for n, t in tables["dense"].items() if n in used}
    moe_names = (used & MOE_LAYERS) if is_moe else set()
    if used != attn_names | set(per_seq) | set(dense) | moe_names or \
            len(attn_names) + len(per_seq) + len(dense) + len(moe_names) != len(used):
        raise Refused("layer names are not covered by exactly one table each")
    if is_moe and (moe_names not in ({"moe_gate_routing", "moe_experts"}, {"expert"}) or
                   seq["block"].count("expert") + seq["block"].count("moe_experts") != 1):
        raise Refused(f"unexpected MoE layers {sorted(moe_names)}")
    rel = os.path.join("profiler", "perf", hardware, model, variant, f"tp{tp}")
    cfg_rel = os.path.join("configs", "model", model + ".json")
    sources = {}
    for name in ("dense.csv", "per_sequence.csv", "attention.csv", "skew_fit.csv"):
        path = os.path.join(fork, rel, name)
        if os.path.isfile(path):
            sources[name] = _sha256(path)
    rel1 = os.path.join("profiler", "perf", hardware, model, variant, "tp1")
    for name in ("moe.csv", os.path.join("moe_components", "dp1_ep1_rank0_sp0", "components.csv")):
        path = os.path.join(fork, rel1 if is_moe else rel, name)
        if is_moe and os.path.isfile(path):
            sources[os.path.join("tp1", name)] = _sha256(path)
    with open(os.path.join(fork, cfg_rel), encoding="utf-8") as f:
        model_config = json.load(f)
    meta = db["meta"]
    prov = {"license": LICENSE_NOTICE, "source_profile": rel, "source_config": cfg_rel,
            "source_sha256": sources, "fork_commit": _git(fork, "rev-parse", "HEAD"),
            "fork_sources_modified": bool(_git(fork, "status", "--porcelain", "--", rel, cfg_rel))}
    for key in ("vllm_version", "profiled_at", "gpu", "profiler_version"):
        if meta.get(key):
            prov[key] = meta[key]
    prov.update(provenance or {})  # e.g. what augur.profile measured: environment, vLLM memory accounting
    extra = {}
    if tp > 1:
        stable = sorted(n for cat in ("dense", "per_sequence") for n in tables[cat]
                        if 1 in db["available_tps"] and tg._effective_tp(db, cat, n, tp) == 1)
        prov["tables_from_tp1"] = stable
        extra["collectives"] = _collectives(cost, seq, tmax=1000 if is_moe else None)
        if is_moe:  # decoder layers whose collective-free MoE block ASTRA-Sim prices 1 ns short (see tables.py)
            extra["collectives"]["moe_layers"] = seq["repeat"] * (seq["block"].count("expert")
                                                                  + seq["block"].count("moe_experts"))
        bw = _npu_mem_bw_default(fork, hardware)
        if bw is not None:
            extra["hardware"] = {"npu_mem_bw_gbps": bw}
    if is_moe:
        extra["moe"] = _moe_section(cost, tg, db, config, moe_names)
        if dp_cluster_config:
            if tp != 1:
                raise Refused("DP tables are built from the TP1 profile (--tp 1)")
            dp = _dp_section(fork, hardware, model, dp_cluster_config, variant)
            if dp["2"]["tail"] != seq["head"] or dp["2"]["layers"] != seq["repeat"]:
                raise Refused("DP trace layers differ from the single-engine trace's")
            extra["moe"]["dp"] = dp
            bw = _npu_mem_bw_default(fork, hardware)  # the reduce-scatter's reduction runs at it
            if bw is not None:
                extra["hardware"] = {"npu_mem_bw_gbps": bw}
            for name in ("dp2_ep2_rank0_sp0", "dp2_ep2_rank1_sp0"):
                path = os.path.join(fork, rel1, "moe_components", name, "components.csv")
                if os.path.isfile(path):
                    sources[os.path.join("tp1", "moe_components", name, "components.csv")] = _sha256(path)
            prov["dp_cluster_config"] = os.path.basename(dp_cluster_config)
    return {**extra, **{
        "format_version": 1,
        "identity": {"hardware": hardware, "model": model, "variant": variant, "tp": tp,
                     "kind": "moe" if is_moe else "dense",
                     "architecture": meta.get("architecture")},
        "provenance": prov,
        "layers": seq,
        "dense": {n: {"keys": t["keys"], "ns": t["values"]} for n, t in sorted(dense.items())},
        "per_sequence": {n: {"keys": t["keys"], "ns": t["values"]} for n, t in sorted(per_seq.items())},
        "attention": {n: _attention_spec(tables["attention_by_layer"][n]) for n in sorted(attn_names)},
        "skew": _skew_spec(db, tp),
        "remote": {"input_bytes_per_token": 4, "output_bytes_per_seq": 4},
        "model_config": model_config,
    }}


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
    tp = p["identity"]["tp"]
    for cat, key in (("dense", "dense"), ("per_sequence", "per_sequence")):
        got = {n: {"keys": t["keys"], "values": t["ns"]} for n, t in p[cat].items()}
        want = {n: t for n, t in tables[key].items() if n in got}
        bad += _same(dict(sorted(got.items())), dict(sorted(want.items())), cat)
        if not set(got) <= set(tables[key]) or (tp == 1 and set(got) != set(tables[key])):
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
    tp = p["identity"]["tp"]
    entry = fit.get(tp) or fit.get(str(tp))
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


def compare_moe(path, cost, cluster_config=None, seed=7):
    """Every token count of the MoE layers, stored table against the fork's trace: all of 1..N for tables
    (N+2 for the legacy closed form, plus random counts far above), and T = 0 where the fork prices it."""
    from augur.cost.tables import OutsideRegime, TableCost
    with gzip.open(path, "rt", encoding="utf-8") as f:
        p = json.load(f)
    moe = p.get("moe")
    if not moe:
        return []
    mine = TableCost(path, **(cluster_params(cluster_config) if p["identity"]["tp"] > 1 else {}))
    names = set(moe.get("exact", {})) | set(moe.get("balanced", {}))
    if "exact" in moe:
        top = len(next(iter(moe["exact"].values())))
        counts = list(range(1, top + 1))
    else:
        rng = random.Random(seed)
        counts = list(range(1, 4200)) + sorted(rng.sample(range(4200, 200000), 600))
    bad = []
    for t in counts:
        want = _moe_prices(cost, t, names)
        got = {n: mine._moe_ns(n, t) for n in names}
        if want != got:
            bad.append((t, want, got))
    if "exact" in moe:
        for t in (0, top + 1, top + 2000):
            try:
                mine._moe_ns(next(iter(names)), t)
            except OutsideRegime:
                continue
            bad.append((t, "table priced a count the fork refuses", None))
    return bad[:10]


TEMPLATE_SYSTEM = {"scheduling-policy": "LIFO", "endpoint-delay": 10, "active-chunks-per-dimension": 1,
                   "preferred-dataset-splits": 1, "all-reduce-implementation": ["ring"],
                   "all-gather-implementation": ["ring"], "reduce-scatter-implementation": ["ring"],
                   "all-to-all-implementation": ["ring"], "collective-optimization": "localBWAware",
                   "boost-mode": 0}


def cluster_params(path, default_npu_bw=None):
    """TableCost keyword arguments that mirror a fork cluster config (single node, one instance)."""
    with open(path, encoding="utf-8") as f:
        c = json.load(f)
    node = c["nodes"][0]
    inst = node["instances"][0]
    return dict(link_bw=c["link_bw"], link_latency_ns=c["link_latency"],
                npu_mem_bw=(inst.get("npu_mem") or {}).get("mem_bw", default_npu_bw),
                cpu_mem_bw=node["cpu_mem"]["mem_bw"], cpu_mem_latency_ns=node["cpu_mem"]["mem_latency"])


def spot_check_tp(path, fork, hardware, model, tp, cluster_config, n, seed, variant=None):
    """Random shapes: TableCost against LLMServingSimCost (ASTRA-Sim) with the same cluster config.
    Returns (mismatches, refused by the guard, distinct ASTRA runs)."""
    from augur.cost.llmservingsim import LLMServingSimCost
    from augur.cost.tables import OutsideRegime, TableCost
    ref = LLMServingSimCost(fork, hardware, model, tp=tp, cluster_config=cluster_config, profile_variant=variant)
    with open(ref.oracle.cluster["system_config_path"], encoding="utf-8") as f:
        system = json.load(f)
    if {k: v for k, v in system.items() if k != "local-mem-bw"} != TEMPLATE_SYSTEM:
        raise Refused(f"ASTRA-Sim system config differs from the one the closed form was verified for: {system}")
    mine = TableCost(path, **cluster_params(cluster_config))
    bad, refused = [], 0
    for s in random_shapes(random.Random(seed), n):
        try:
            got = mine.step_ns(s)
        except OutsideRegime:
            refused += 1
            continue
        try:
            want = ref.step_ns(s)
        except ValueError:  # the fork refuses it too
            bad.append((s, "fork refuses", got))
            continue
        if want != got:
            bad.append((s, want, got))
    return bad, refused, ref.oracle.runs


def spot_check(path, fork, hardware, model, n, seed, variant=None, **cpu):
    from augur.cost.llmservingsim import LLMServingSimCost
    from augur.cost.tables import TableCost
    ref = LLMServingSimCost(fork, hardware, model, cpu_mem_bw_gbps=cpu.get("bw", 256.0),
                            cpu_mem_latency_ns=cpu.get("lat", 0), profile_variant=variant)
    mine = TableCost(path, cpu_mem_bw=cpu.get("bw", 256.0), cpu_mem_latency_ns=cpu.get("lat", 0))
    from augur.cost.tables import OutsideRegime
    shapes = list(random_shapes(random.Random(seed), n))

    def price(c, s):  # a step either side refuses (the MoE component tables stop at 2048 tokens) is compared as a refusal
        try:
            return c.step_ns(s)
        except (ValueError, OutsideRegime) as e:
            return "refused"
    t0 = time.perf_counter()
    want = [price(ref, s) for s in shapes]
    t1 = time.perf_counter()
    got = [price(mine, s) for s in shapes]
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
    ap.add_argument("--cluster-config", default=None,
                    help="fork cluster config JSON (needed for --tp > 1: link, NPU and CPU memory parameters of the self-check)")
    ap.add_argument("--spot-checks", type=int, default=None,
                    help="random shapes priced against the fork (default 2000 at TP=1; 200 at TP>1, where each "
                         "distinct (tokens, sequences) pair costs one ASTRA-Sim binary run)")
    ap.add_argument("--profile-variant", default=None,
                    help="the fork's profile folder (e.g. bf16-tp4proj); name the output <variant>-tp<N>.json.gz")
    ap.add_argument("--dp-cluster-config", default=None,
                    help="MoE at --tp 1: also build the DP2 + EP2 round tables from this fork cluster config (one node, "
                         "two TP1 instances with ep_size 2) and check them against the fork's traces")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--provenance-json", default=None,
                    help="JSON object merged into the profile's provenance (augur.profile writes the measured "
                         "environment and vLLM's memory accounting this way)")
    a = ap.parse_args(argv)
    fork = os.path.abspath(a.fork)
    if a.spot_checks is None:
        a.spot_checks = 2000 if a.tp == 1 else 200
    try:
        with open(a.provenance_json, encoding="utf-8") if a.provenance_json else open(os.devnull) as f:
            extra_prov = json.load(f) if a.provenance_json else None
        profile = build_profile(fork, a.hardware, a.model, a.tp, a.cluster_config, a.profile_variant,
                                provenance=extra_prov, dp_cluster_config=a.dp_cluster_config)
    except Refused as e:
        print(f"convert: refused: {e}", file=sys.stderr)
        return 2
    raw, packed = write_profile(profile, a.out)
    print(f"wrote {a.out}: {raw} bytes JSON, {packed} bytes gzip")

    cost, db, config, tables, tg = load_fork(fork, a.hardware, a.model, a.tp, a.cluster_config, a.profile_variant)
    bad = compare_tables(a.out, db, tables, tg)
    print(f"table comparison: {'OK' if not bad else 'MISMATCH'}")
    for m in bad[:10]:
        print("  ", m)
    failures = len(bad)
    mbad = compare_moe(a.out, cost, a.cluster_config)
    if json.loads(gzip.open(a.out, "rt").read()).get("moe"):
        print(f"MoE block, every token count against the fork's trace: {'OK' if not mbad else 'MISMATCH'}")
        for m in mbad[:5]:
            print("  ", m)
    failures += len(mbad)
    if a.dp_cluster_config:
        dbad = check_dp_section(profile, fork, a.hardware, a.model, a.dp_cluster_config, variant=a.profile_variant)
        print(f"DP MoE tables, random round counts against the fork's traces: {'OK' if not dbad else 'MISMATCH'}")
        for m in dbad[:5]:
            print("  ", m)
        failures += len(dbad)
    if a.tp > 1:
        try:
            mism, refused, runs = spot_check_tp(a.out, fork, a.hardware, a.model, a.tp, a.cluster_config,
                                                a.spot_checks, a.seed, a.profile_variant)
        except Refused as e:
            print(f"convert: refused: {e}", file=sys.stderr)
            return 2
        print(f"spot check (cluster config): {a.spot_checks} shapes, {len(mism)} mismatches, "
              f"{refused} outside the guard's regime, {runs} ASTRA-Sim runs")
        for s, w, g in mism[:3]:
            print("  ", shape_to_row(s), "fork", w, "table", g)
        print("collectives:", json.dumps(profile["collectives"]), "hardware:", json.dumps(profile.get("hardware")))
        return 1 if failures + len(mism) else 0
    for label, cpu in (("default CPU memory", {}), ("bw=3.7 latency=123", {"bw": 3.7, "lat": 123})):
        n = a.spot_checks if not cpu else max(500, a.spot_checks // 4)
        mism, t_ref, t_mine = spot_check(a.out, fork, a.hardware, a.model, n, a.seed, a.profile_variant, **cpu)
        print(f"spot check ({label}): {n} shapes, {len(mism)} mismatches; "
              f"fork {t_ref * 1e6:.0f} us/shape, TableCost {t_mine * 1e6:.0f} us/shape")
        for s, w, g in mism[:3]:
            print("  ", shape_to_row(s), "fork", w, "table", g)
        failures += len(mism)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
