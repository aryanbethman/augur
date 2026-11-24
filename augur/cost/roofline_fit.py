"""Fit and check ``--cost roofline`` against the measured profiles shipped with augur (no GPU).

    python -m augur.cost.roofline_fit fit [--exclude HARDWARE_OR_PROFILE ...] [--out params.json]
    python -m augur.cost.roofline_fit loo [--by profile|gpu]

The data are the profiles' own tables evaluated by ``TableCost`` on a grid of step shapes (decode batches of 1-128
requests at 64-2048 tokens of context; single and double prefill chunks of 16-2048 tokens over 0-1024 of history;
decodes plus a prefill chunk), per op and per step.  A step here is the GPU work of the layers only: ``TableCost``'s
CPU and TP collective terms are the same closed forms for both pricers and are left out so they do not flatter the
error.

Fit: ``scipy.optimize.least_squares`` over the log of the ``RooflineParams`` with a soft-L1 loss (scale 0.25) on the log
ratio predicted / measured.  Each profile has weight 1; within it each op counts by its share of the step's measured
time and the step total counts as much as all ops together.

Data left out, because the measurements are not physical or are copies: the H100 profiles' attention table (prefill
chunk 64 to 2048 all take 5.5 us at the same key length, below the FLOP time of the larger ones), sampler (1 ns) and
elementwise ops (layernorm 18 us at one token and 287 us at 2048, 7x the bandwidth time of the tensor, against 1.6-2.4
us and 0.3-0.5x elsewhere: unfused eager kernels with launch gaps, not what vLLM runs); an H100 step is therefore its
GEMMs and lm_head.  Also the attention table of every TP>1 profile (the RTX PRO 6000's TP2 and TP4 tables equal their
TP1 table within 0.3%, as if the heads were not sharded, so they cannot test the model's sharding); and the MoE
``expert`` block of the TP>1 profiles (one table, not scaled by TP, the same at TP2 and TP4).
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from augur.cost import SeqWork, StepShape
from augur.cost.profiles import shipped
from augur.cost.roofline import (Arch, DEFAULT_PARAMS, RooflineParams, attention_features, gpu_peaks, layer_counts,
                                 op_ns)
from augur.cost.tables import TableCost, bundled_profile

FIELDS = [f for f in RooflineParams.__dataclass_fields__]
LOWER = dict(gemm_flops_eff=0.05, gemm_mem_eff=0.05, gemm_ovh_ns=100.0, elem_bw_scale=0.2, elem_ovh_ns=100.0,
             attn_flops_eff=0.05, attn_mem_eff=0.05, attn_pre_mem_eff=0.05, attn_ovh_ns=100.0, sampler_ovh_ns=100.0)
UPPER = dict(gemm_flops_eff=1.0, gemm_mem_eff=1.0, gemm_ovh_ns=5e4, elem_bw_scale=8.0, elem_ovh_ns=5e4,
             attn_flops_eff=1.0, attn_mem_eff=1.0, attn_pre_mem_eff=1.0, attn_ovh_ns=5e4, sampler_ovh_ns=3e4)
SOFT_L1_SCALE = 0.25
# The H100 profiles' attention, sampler and elementwise tables (norms, rotary, act_fn, embedding) are left out: see the
# module docstring.
H100_UNUSABLE = {"attention", "sampler", "layernorm", "final_layernorm", "rotary_emb", "act_fn", "embedding"}


def shapes() -> list[tuple[str, StepShape]]:
    """(class, shape) grid: decode batches, prefill chunks, decodes + a prefill chunk."""
    out = []
    for n in (1, 2, 4, 8, 16, 32, 64, 128):
        for kv in (64, 256, 1024, 2047):
            out.append(("decode", StepShape(tuple(SeqWork(1, kv, True) for _ in range(n)))))
    for c in (16, 64, 256, 512, 1024, 2048):
        for hist in (0, 512, 1024):
            out.append(("prefill", StepShape((SeqWork(c, hist, False),))))
    for c in (128, 256, 512):
        out.append(("prefill", StepShape((SeqWork(c, 0, False), SeqWork(c, 256, False)))))
    for n in (8, 32, 64):
        for kv in (512, 1024):
            for c in (64, 256, 1024):
                for hist in (0, 512):
                    out.append(("mixed", StepShape(tuple(SeqWork(1, kv, True) for _ in range(n))
                                                   + (SeqWork(c, hist, False),))))
    return out


@dataclass
class Profile:
    name: str
    hardware: str
    tp: int
    arch: Arch
    flops: float
    bw: float
    classes: np.ndarray
    feats: dict  # tokens, rows, pairs, kv, new, seqs: arrays over shapes
    measured: dict  # op -> time of one occurrence, array over shapes (ops we can compare only)
    counts: dict

    def predict(self, prm: RooflineParams) -> dict:
        f = self.feats
        per = op_ns(self.arch, self.flops, self.bw, self.tp, prm, f["tokens"], f["rows"], f["pre"], f["dec"])
        return {k: np.asarray(per[k], dtype=float) * np.ones(len(self.classes)) for k in self.measured}

    def step(self, per_op: dict) -> np.ndarray:
        return sum(self.counts[k] * v for k, v in per_op.items())


def load_profiles() -> list[Profile]:
    grid = shapes()
    out = []
    for hw, model, variant, tp in shipped():
        path = bundled_profile(hw, model, tp, variant)
        cost = TableCost(path, npu_mem_bw=1000.0) if tp > 1 else TableCost(path)
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            arch = Arch.from_dict(json.load(fh)["model_config"])
        keep = []
        for cls, shape in grid:
            pre, n_dec, dq, _, kv_max = cost._batch_key(shape)
            if not cost._beyond_axes(pre, n_dec, kv_max, dq, shape.forward_tokens, len(shape.seqs)):
                keep.append((cls, shape))
        rows = []
        measured: dict[str, list] = {}
        for _, shape in keep:
            price = cost._pricer(cost._batch_key(shape), shape.forward_tokens, len(shape.seqs))
            rows.append(attention_features(shape))
            for name in set(cost._prologue + cost._head + list(cost._block)):
                measured.setdefault(name, []).append(price(name))
        # the ops the roofline prices and the measurements can be compared on
        skip = {"expert"} | (H100_UNUSABLE if hw == "H100" else set()) | ({"attention"} if tp > 1 else set())
        counts = layer_counts(arch)
        names = [n for n in measured if n in counts and n not in skip]
        feats = {part: tuple(np.array([r[i] for r in rows], dtype=float).T) for i, part in enumerate(("pre", "dec"))}
        feats["tokens"] = np.array([s.forward_tokens for _, s in keep], dtype=float)
        feats["rows"] = np.array([len(s.seqs) for _, s in keep], dtype=float)
        flops, bw = gpu_peaks(hw)
        short = model.split("/")[-1]
        out.append(Profile(f"{hw} {short} tp{tp}", hw, tp, arch, flops, bw, np.array([c for c, _ in keep]), feats,
                           {n: np.array(measured[n], dtype=float) for n in names}, {n: counts[n] for n in names}))
    return out


def to_params(theta: np.ndarray) -> RooflineParams:
    return RooflineParams.from_json(dict(zip(FIELDS, np.exp(theta))))


def residuals(prm: RooflineParams, profiles: list[Profile]) -> np.ndarray:
    out = []
    for p in profiles:
        pred, n = p.predict(prm), len(p.classes)
        meas_step, pred_step = p.step(p.measured), p.step(pred)
        for k, meas in p.measured.items():
            share = p.counts[k] * meas / meas_step
            out.append(np.sqrt(share / n) * np.log(np.maximum(pred[k], 1.0) / np.maximum(meas, 1.0)))
        out.append(np.log(pred_step / meas_step) / np.sqrt(n))
    return np.concatenate(out)


def fit(profiles: list[Profile]) -> RooflineParams:
    start = DEFAULT_PARAMS.to_json()
    bounds = (np.log([LOWER[f] for f in FIELDS]), np.log([UPPER[f] for f in FIELDS]))
    best = None
    for scale in (1.0, 0.5, 2.0):  # three starts: the overheads at once and scaled
        x0 = np.log([min(max(start[f] * (scale if f.endswith("ovh_ns") else 1.0), LOWER[f] * 1.01), UPPER[f] * 0.99)
                     for f in FIELDS])
        r = least_squares(lambda x: residuals(to_params(x), profiles), x0, loss="soft_l1", f_scale=SOFT_L1_SCALE,
                          ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=2000, bounds=bounds)
        if best is None or r.cost < best.cost:
            best = r
    return to_params(best.x)


def errors(p: Profile, prm: RooflineParams) -> dict:
    """Predicted / measured - 1 per shape for the step and per op, from the roofline with ``prm``."""
    pred = p.predict(prm)
    out = {"step": p.step(pred) / p.step(p.measured) - 1}
    out.update({k: pred[k] / p.measured[k] - 1 for k in p.measured})
    return out


def summary(err: np.ndarray) -> tuple[float, float, float]:
    """(median absolute, p90 absolute, median signed) percent error."""
    a = np.abs(err) * 100
    return float(np.median(a)), float(np.percentile(a, 90)), float(np.median(err) * 100)


def loo(profiles: list[Profile], by: str) -> list[tuple[str, RooflineParams, Profile]]:
    """Fit without each profile (or each GPU) and return (held-out name, params, held-out profile)."""
    groups = sorted({p.hardware for p in profiles}) if by == "gpu" else [p.name for p in profiles]
    out = []
    for g in groups:
        train = [p for p in profiles if (p.hardware if by == "gpu" else p.name) != g]
        for p in profiles:
            if (p.hardware if by == "gpu" else p.name) == g:
                out.append((p.name, fit(train), p))
    return out


def print_loo(profiles: list[Profile], by: str) -> None:
    print(f"leave one {by} out: |predicted / measured - 1| in %, median / p90 (signed median) per step class")
    print(f"{'held out':<36} {'decode':<22} {'prefill':<22} {'mixed':<22}")
    per_op: dict[str, list] = {}
    for name, prm, p in loo(profiles, by):
        e = errors(p, prm)
        cells = []
        for cls in ("decode", "prefill", "mixed"):
            m = p.classes == cls
            cells.append("%5.1f / %5.1f (%+5.1f)" % summary(e["step"][m]))
        print(f"{name:<36} " + " ".join(cells))
        for k, v in e.items():
            if k != "step":
                per_op.setdefault(k, []).append(v)
    print("\nper op, all held-out profiles pooled: median / p90 (signed median)")
    for k, v in sorted(per_op.items()):
        print(f"  {k:<18} %5.1f / %5.1f (%+5.1f)" % summary(np.concatenate(v)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="augur.cost.roofline_fit", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--exclude", nargs="*", default=[], help="hardware names or profile names to leave out")
    f.add_argument("--out", default=None, help="write the parameters here (JSON, for augur --roofline-params)")
    l = sub.add_parser("loo")
    l.add_argument("--by", choices=["profile", "gpu"], default="profile")
    a = ap.parse_args(argv)
    profiles = load_profiles()
    if a.cmd == "loo":
        print_loo(profiles, a.by)
        return 0
    train = [p for p in profiles if p.hardware not in a.exclude and p.name not in a.exclude]
    prm = fit(train)
    text = json.dumps(prm.to_json(), indent=1)
    print(text)
    for p in profiles:
        print(f"{p.name:<36} in fit={p in train}  step error median/p90 %5.1f / %5.1f"
              % summary(errors(p, prm)["step"])[:2], file=sys.stderr)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
