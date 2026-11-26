"""Does a simulator pick the right engine setting?  A real-vLLM sweep of one setting against every simulator's.

    python scripts/baselines/sweep.py real [--values 256 512 ...]      # one real vLLM run per value, GPU 0 (flock ~/gpu0.lock)
    python scripts/baselines/sweep.py sims [--jobs 2]                  # augur, BLIS, Frontier, Vidur per value (run.sh)
    python scripts/baselines/sweep.py table                            # metric per simulator, Kendall tau, best pick, regret
    python scripts/baselines/sweep.py spec-real|spec-sims|spec-table [--workloads code rag chat] [--ks 0 1 ...]   # ngram depth
    python scripts/baselines/sweep.py throughput [--budgets N] [--rates N] [--parallel 4]   # configurations per hour

Setting swept: max-num-batched-tokens (the token budget of a step), on the tenants workload of overlap_workloads.py
(400 requests, 6/s, prompts that share a platform preamble and a tenant prompt: which simulators see the token ids
matters). Metric, fixed before the results: mean end-to-end latency over the requests, as the client measures it
(compare.py's ``abs.e2e`` mean); the best setting is the lowest. Every simulator is priced by pricer.py (run.sh), BLIS
gets BLIS_ENCODING=best; augur is run as a user would (no replayed durations). Real runs are out/real/sweep_mbt<N>,
simulators' out/baselines/sweep_mbt<N>.  ``throughput`` times the sweep's setting crossed with arrival rates (the
workload's arrivals scaled), the simulator runs pinned to cores 4-7 with ``taskset``.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import itertools
import json
import glob
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
WORKLOAD = os.environ.get("SWEEP_WORKLOAD", os.path.expanduser("~/augur-wt/overlap/out/overlap/tenants.jsonl"))
VALUES = [256, 512, 1024, 2048, 4096, 8192]
SIMS = ("augur", "blis", "frontier", "vidur")  # augur: run.sh's, on the one price every simulator has
CORES = os.environ.get("SWEEP_CORES", "4-7")
# ngram speculation sweep: workload -> (trace, entries kept, max model length)
_WL = os.path.expanduser("~/llmss-vllm-sim-20261114/workloads")
SPEC_WL = {"code": (f"{_WL}/swe-bench-qwen3-30b-a3b-50-sps0.2.jsonl", 20, 24576),
           "rag": (os.path.join(os.path.dirname(WORKLOAD), "rag.jsonl"), 250, 8192),
           "chat": (f"{_WL}/sharegpt-qwen3-32b-300-sps10.jsonl", 300, 8192)}
SPEC_K = [0, 1, 2, 3, 4, 5]


def real_dir(v):
    return os.path.join(ROOT, "out/real", f"sweep_mbt{v}")


def base_dir(name):
    return os.path.join(ROOT, "out/baselines", name)


def sh(cmd, env=None, **kw):
    return subprocess.run(cmd, shell=True, env={**os.environ, **(env or {})}, **kw)


def run_sh(name, rdir, wl, stages, env=None, pin=False):
    """scripts/baselines/run.sh NAME STAGES for a real run directory; returns wall seconds."""
    e = {"AUGUR_RUN_DIR": rdir, "AUGUR_WORKLOAD": wl, "BLIS_ENCODING": "best", **(env or {})}
    t = time.time()
    r = sh(f"{'taskset -c ' + CORES + ' ' if pin else ''}nice -n 10 {HERE}/run.sh {name} {stages}", e,
           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return time.time() - t, r.returncode


def cmd_real(a):
    for v in a.values:
        d = real_dir(v)
        if os.path.exists(f"{d}/client.csv") and os.path.exists(f"{d}/client.json"):
            continue
        r = sh(f"flock ~/gpu0.lock env MAXBATCH={v} {ROOT}/scripts/real_trace_run.sh sweep_mbt{v} {WORKLOAD}", cwd=ROOT)
        print(v, "exit", r.returncode, flush=True)


def cmd_sims(a):
    def one(v):
        n, d = f"sweep_mbt{v}", base_dir(f"sweep_mbt{v}")
        # augur as a user runs it (shared-KV decode term on), kept as augur_skv.csv; then the same-price run of
        # run.sh, with every simulator's results compared by compare.py
        run_sh(n, real_dir(v), WORKLOAD, "prepare augur", {"SHARED_KV": "on"})
        sh(f"mv {d}/augur.csv {d}/augur_skv.csv")
        t, rc = run_sh(n, real_dir(v), WORKLOAD, "prepare augur blis frontier vidur compare")
        return v, t, rc
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        for v, t, rc in ex.map(one, a.values):
            print(v, f"{t:.0f}s", "exit", rc, flush=True)


def kendall(x, y):
    """Kendall tau-a over all pairs (no ties in these metrics)."""
    s = [np.sign(x[i] - x[j]) * np.sign(y[i] - y[j]) for i, j in itertools.combinations(range(len(x)), 2)]
    return float(np.mean(s))


def cmd_table(a):
    global SIMS
    res = {}
    for v in a.values:
        r = json.load(open(f"{base_dir(f'sweep_mbt{v}')}/results.json"))[0]["sims"]
        res[v] = {k: r[k]["abs"]["e2e"][0] for k in ("real", *SIMS) if k in r}
        # augur as a user runs it: the client-side latency column of its output (same measure as compare's augur row)
        res[v]["augur_skv"] = float(np.mean([float(x["latency"]) for x in csv.DictReader(
            open(f"{base_dir(f'sweep_mbt{v}')}/augur_skv.csv"))])) / 1e9
        res[v]["augur_same_price"] = res[v].pop("augur")
    SIMS = ("augur_skv", "augur_same_price", "blis", "frontier", "vidur")
    sims = ("real", *SIMS)
    print("| max-num-batched-tokens | " + " | ".join(sims) + " |\n|" + " --- |" * (len(sims) + 1))
    for v in a.values:
        print(f"| {v} | " + " | ".join(f"{res[v][s]:.1f}" for s in sims) + " |")
    real = [res[a.values[i]]["real"] for i in range(len(a.values))]
    best = int(np.argmin(real))
    print("\n| simulator | Kendall tau | picks | real best | regret % |\n| --- | --- | --- | --- | --- |")
    for s in SIMS:
        y = [res[v][s] for v in a.values]
        p = int(np.argmin(y))
        print(f"| {s} | {kendall(real, y):+.2f} | {a.values[p]} | {a.values[best]} | "
              f"{100 * (real[p] / real[best] - 1):.1f} |")
    print("\nrank order (best first):", {s: [a.values[i] for i in np.argsort([res[v][s] for v in a.values])] for s in sims})


def cmd_throughput(a):
    """budgets x rates configurations; for each, augur's run and BLIS's run timed on their own (4 at a time,
    pinned to cores 4-7). augur reads the workload as it is (its time includes run.sh's writing of the other
    simulators' inputs, a few seconds); BLIS also needs ``prepare`` (its input conversion and prefix groups), counted."""
    ref = next(real_dir(v) for v in (2048, *VALUES) if os.path.exists(real_dir(v) + "/client.json"))  # its KV block count and step log
    budgets = np.unique(np.geomspace(256, 8192, a.budgets).round(-1 if a.budgets > 6 else 0).astype(int)).tolist()
    rates = np.linspace(0.5, 1.5, a.rates).tolist()  # multiples of the trace's 6 req/s
    tmp = os.path.join(ROOT, "out/baselines/sweep_tp")
    os.makedirs(tmp, exist_ok=True)
    wls, prep = {}, {}
    for r in rates:  # arrivals / r: the same prompts, sent r times as fast
        wls[r] = f"{tmp}/rate{r:.2f}.jsonl"
        with open(WORKLOAD) as f, open(wls[r], "w") as g:
            for line in f:
                e = json.loads(line)
                e["arrival_time_ns"] = int(e["arrival_time_ns"] / r)
                g.write(json.dumps(e) + "\n")
    cfgs = [(b, r) for r in rates for b in budgets]
    t_all = time.time()
    def one(c):
        b, r = c
        name = f"sweep_tp/b{b}_r{r:.2f}"
        env = {"MAX_BATCHED_TOKENS": str(b)}
        tp, _ = run_sh(name, ref, wls[r], "prepare", env, pin=True)
        ta, rca = run_sh(name, ref, wls[r], "augur", {**env, "SHARED_KV": "on"}, pin=True)
        tb, rcb = run_sh(name, ref, wls[r], "blis", env, pin=True)
        sh(f"rm -rf {base_dir(name)}")
        return {"budget": b, "rate": r, "prepare_s": tp, "augur_s": ta, "blis_s": tb, "rc": [rca, rcb]}
    with cf.ThreadPoolExecutor(a.parallel) as ex:
        out = list(ex.map(one, cfgs))
    wall = time.time() - t_all
    json.dump(out, open(f"{tmp}/throughput.json", "w"), indent=1)
    n = len(out)
    for k in ("prepare_s", "augur_s", "blis_s"):
        print(k, f"mean {np.mean([o[k] for o in out]):.1f} s")
    for k, cost in (("augur", "augur_s"), ("BLIS (prepare + run)", None)):
        t = np.mean([o["augur_s"] if cost else o["prepare_s"] + o["blis_s"] for o in out])
        print(f"{k}: {t:.1f} s per configuration, {a.parallel * 3600 / t:.0f} configurations per hour at {a.parallel} at a time")
    print(f"{n} configs, wall {wall:.0f} s with {a.parallel} at a time; failures {sum(any(o['rc']) for o in out)}")


def spec_wl(w):
    """The workload's first N entries as a file of its own, so that real run and simulators send the same requests."""
    src, n, _ = SPEC_WL[w]
    dst = os.path.join(ROOT, "scratch/spec_wl", f"{w}.jsonl")
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        sh(f"head -n {n} {src} > {dst}")
    return dst


def spec_dir(w, k, tag=""):
    return os.path.join(ROOT, "out/real", f"spec_{w}_k{k}{tag}")


def cmd_spec_real(a):
    for w, k in itertools.product(a.workloads, a.ks):
        d = spec_dir(w, k, a.tag)
        if os.path.exists(f"{d}/client.json") and os.path.getsize(f"{d}/client.json"):
            continue
        cfg = "" if k == 0 else ("--speculative-config '" + json.dumps(
            {"method": "ngram", "num_speculative_tokens": k, "prompt_lookup_max": 4, "prompt_lookup_min": 2}) + "'")
        t = time.time()
        r = sh(f"flock ~/gpu0.lock env MAXLEN={SPEC_WL[w][2]} {ROOT}/scripts/real_trace_run.sh spec_{w}_k{k}{a.tag} "
               f"{spec_wl(w)} '' {cfg}", cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(w, k, "exit", r.returncode, f"{time.time() - t:.0f}s", flush=True)


def cmd_spec_sims(a):
    """Per (workload, k): augur token-faithful (outputs of the k=0 real run; ``_own``: the run's own outputs) and BLIS
    at the run's measured acceptance (oracle); then BLIS at one acceptance length, the mean over all speculating runs."""
    def go(w, k, sfx, env, stages):
        n = f"spec_{w}_k{k}{sfx}"
        run_sh(n, spec_dir(w, k), spec_wl(w), stages, {"SHARED_KV": "on", "BLIS_ENCODING": "greedy" if w == "code" else "best", **env}, pin=True)  # (best breaks BLIS on sessions)
        sh(f"rm -f {base_dir(n)}/*_steps*.jsonl")  # regenerable
    cfgs = list(itertools.product(a.workloads, a.ks))
    def one(c):
        w, k = c
        go(w, k, "", {"AUGUR_TOKENS": f"{spec_dir(w, 0)}/tokens.jsonl"}, "prepare augur blis compare")
        go(w, k, "_own", {"AUGUR_TOKENS": f"{spec_dir(w, k)}/tokens.jsonl"}, "prepare augur compare")
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(one, cfgs))
    al = [json.load(open(f"{base_dir(f'spec_{w}_k{k}')}/meta.json"))["acceptance"]["mean_tokens_per_step"]
          for w, k in itertools.product(SPEC_WL, SPEC_K[1:]) if os.path.exists(f"{base_dir(f'spec_{w}_k{k}')}/meta.json")]
    fixed = float(np.mean(al))
    json.dump({"fixed_acceptance_length": fixed, "runs": len(al)}, open(base_dir("spec_fixed.json"), "w"))
    print("fixed acceptance length", fixed, "over", len(al), "runs", flush=True)
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(lambda c: go(*c[:2], "_fix", {"ACCEPT_LEN": str(min(fixed, c[1] + 1))}, "prepare blis compare")
                    if c[1] else None, cfgs))


def cmd_spec_table(a):
    cols = {"real": ("", "real"), "augur": ("", "augur"), "augur_own": ("_own", "augur"), "blis_fixed": ("_fix", "blis"),
            "blis_oracle": ("", "blis")}
    def get(w, k, c, m):
        sfx, sim = cols[c]
        if c in ("blis_fixed",) and k == 0:
            sfx = ""
        return json.load(open(f"{base_dir(f'spec_{w}_k{k}{sfx}')}/results.json"))[0]["sims"][sim]["abs"][m][0]
    for m in ("tpot", "e2e"):
        for w in a.workloads:
            print(f"\n{w}, mean {m} (s)\n| k | " + " | ".join(cols) + " |\n|" + " --- |" * (len(cols) + 1))
            res = {c: [get(w, k, c, m) for k in a.ks] for c in cols}
            for i, k in enumerate(a.ks):
                print(f"| {k} | " + " | ".join(f"{res[c][i]:.4f}" for c in cols) + " |")
            real = res["real"]
            for c in list(cols)[1:]:
                p = int(np.argmin(res[c]))
                print(f"{c}: picks k={a.ks[p]} (real best k={a.ks[int(np.argmin(real))]}), regret "
                      f"{100 * (real[p] / min(real) - 1):.1f}%, tau {kendall(real, res[c]):+.2f}")
    print("\nacceptance (mean tokens per decode request-step; accepted / scheduled drafts)")
    for w in a.workloads:
        for k in a.ks[1:]:
            ac = json.load(open(f"{base_dir(f'spec_{w}_k{k}')}/meta.json"))["acceptance"]
            print(f"{w} k={k}: {ac['mean_tokens_per_step']:.3f}  {ac['accepted_per_scheduled']:.3f}")
    print(open(base_dir("spec_fixed.json")).read())


def cmd_spec_noise(a):
    """Run-to-run noise of the real runs (the first and every ``_r*`` repeat of a cell, from client.csv) and whether each
    simulator's pick of k holds within it: a pick holds when its real mean (over the runs) is within the cell spreads
    (max - min over runs, the larger of pick's and best's) of the best k's. The simulators' values are the first-run sweep's."""
    col = {"tpot": "TPOT", "e2e": "latency"}
    def real(w, k, m):
        out = []
        for t in ("", "_r2", "_r3"):
            f = f"{spec_dir(w, k, t)}/client.csv"
            if os.path.exists(f) and os.path.exists(f"{spec_dir(w, k, t)}/client.json") and os.path.getsize(f"{spec_dir(w, k, t)}/client.json"):
                out.append(float(np.mean([float(r[col[m]]) for r in csv.DictReader(open(f))])) / 1e9)
        return out
    cols = {"augur": ("", "augur"), "blis_fixed": ("_fix", "blis"), "blis_oracle": ("", "blis")}
    for m in ("tpot", "e2e"):
        for w in a.workloads:
            runs = {k: real(w, k, m) for k in a.ks}
            mean = {k: float(np.mean(v)) for k, v in runs.items()}
            spread = {k: (max(v) - min(v)) if len(v) > 1 else float("nan") for k, v in runs.items()}
            print(f"\n{w}, mean {m} (s), real runs\n| k | runs | mean | spread % of mean |\n| --- | --- | --- | --- |")
            for k in a.ks:
                print(f"| {k} | {' / '.join(f'{x:.4f}' for x in runs[k])} | {mean[k]:.4f} | {100 * spread[k] / mean[k]:.1f} |")
            sp = [100 * spread[k] / mean[k] for k in a.ks if spread[k] == spread[k]]
            best = min(a.ks, key=mean.get)
            print(f"real best (mean of runs) k={best}; spread median {np.median(sp):.1f}%, max {max(sp):.1f}%")
            for c, (sfx, sim) in cols.items():
                v = {k: json.load(open(f"{base_dir(f'spec_{w}_k{k}' + ('' if k == 0 else sfx))}/results.json"))[0]["sims"][sim]["abs"][m][0] for k in a.ks}
                pk = min(a.ks, key=v.get)
                gap = mean[pk] - mean[best]
                tol = max(x for x in (spread[pk], spread[best], 0) if x == x)
                print(f"{c}: picks k={pk}, real gap to best {100 * gap / mean[best]:.1f}% vs spread {100 * tol / mean[best]:.1f}% -> "
                      f"{'holds within noise' if gap <= tol else 'outside noise'}")


# Router sweep: simulated only. N engines get the tenants pattern in cells of 400 requests (each cell its own seed:
# its own platform preamble and 24 tenants), ROUTER_PER_ENGINE requests per engine arriving at ROUTER_RATE per engine.
# augur sees the token ids; BLIS gets the best prefix-group encoding of each cell (workloads._map_prefixes "best").
ROUTER_PER_ENGINE, ROUTER_RATE, MML = 12, 6.0, 8192
BLOCKS = int(os.environ.get("ROUTER_BLOCKS", 5334))  # 5,480 for the real 2-engine anchor (its servers report that)
ROUTERS = {"round-robin": ("round-robin", "round-robin.yaml"), "cache-aware": ("cache-aware", "default.yaml"),
           "kv-aware": ("kv-aware", "guides/precise-prefix-cache-routing.yaml")}  # augur name, BLIS's nearest configuration
BLIS_PIPE = os.path.expanduser("~/speedrun/blis-ext-count")  # a BLIS build whose engines share one pricer process
PY = os.path.expanduser("~/augur/.venv/bin/python")


def router_dir(n, *more):
    return os.path.join(ROOT, "out/router", f"n{n}", *more)


def cmd_router_gen(a):
    sys.path.insert(0, HERE)
    import overlap_workloads as ow
    import workloads as wl
    for n in a.engines:
        d = router_dir(n)
        if os.path.exists(f"{d}/wl.jsonl"):
            continue
        os.makedirs(f"{d}/scen", exist_ok=True)
        cells = max(1, round(ROUTER_PER_ENGINE * n / 400))
        lines = []
        for c in range(cells):
            L = ow.generate("tenants", seed=c, n=400, rate=n * ROUTER_RATE / cells)
            rows = [{"call_id": f"req_{i}", "blocks": wl._block_hashes(x["input_tok_ids"]), "input": x["input_toks"],
                     "output": x["output_toks"]} for i, x in enumerate(L)]
            wl._map_prefixes(rows, {}, "best")
            lines += [(x, f"c{c}_{r['prefix_group']}" if r["prefix_group"] else "", r["prefix_len"]) for x, r in zip(L, rows)]
        lines.sort(key=lambda t: t[0]["arrival_time_ns"])
        with open(f"{d}/wl.jsonl", "w") as f, open(f"{d}/blis.csv", "w") as g:
            g.write("request_id,client_id,tenant_id,slo_class,session_id,round_index,prefix_group,prefix_length,streaming,"
                    "input_tokens,output_tokens,text_tokens,image_tokens,audio_tokens,video_tokens,reason_ratio,model,"
                    "deadline_us,arrival_time_us,first_chunk_time_us,last_chunk_time_us,status\n")
            for i, (x, grp, plen) in enumerate(lines):
                f.write(json.dumps(x, separators=(",", ":")) + "\n")
                fe = FRONT_S + FRONT_PER_TOKEN_S * x["input_toks"]  # augur's front end, which BLIS lacks
                g.write(f"{i},c,,,,0,{grp},{plen},false,{x['input_toks'] - plen},{x['output_toks']},0,0,0,0,0,,0,"
                        f"{round(x['arrival_time_ns'] / 1e3 + fe * 1e6)},0,0,ok\n")
        open(f"{d}/blis.yaml", "w").write("trace_version: 4\ntime_unit: microseconds\nmode: converted\nworkload_seed: 1\n")
        json.dump({"engines": n, "cells": cells, "requests": len(lines)}, open(f"{d}/meta.json", "w"))
        print(n, "engines:", len(lines), "requests", flush=True)


FRONT_S, FRONT_PER_TOKEN_S, DELIVERY_S = 3.4e-3, 3e-6, 8e-3


def router_scenario(n, router_file):
    cfg = subprocess.run([f"{os.path.expanduser('~/baselines')}/inference-sim/blis", "router-configs", "--file", router_file],
                         capture_output=True, text=True, check=True).stdout
    cfg = cfg.replace("blockSizeTokens: 64", "blockSizeTokens: 16")  # the simulated engines' KV block size
    d = router_dir(n, "scen")
    open(f"{d}/{os.path.basename(router_file)}", "w").write(cfg)
    open(f"{d}/scenario_{os.path.basename(router_file)}", "w").write(f"""kind: Scenario
name: router
engine: {{kind: vllm, version: "0.33.0"}}
model: qwen3-14b
coefficients: [cost-model-primitives, cost-model-collectives, cost-model-attention, cost-model-recurrent, cost-model-memory]
cluster:
  hardware: h100
  nodes: 1
  gpus_per_node: {n}
---
kind: Deployment
name: router
pools:
  - role: colocated
    nodes: 1
    parallel: {{tp: 1, pp: 1, dp: 1}}
    engine:
      block_size: 16
      max_num_batched_tokens: 2048
      max_num_seqs: 128
      max_model_len: {MML}
      gpu_memory_utilization: 0.85
      async_scheduling: false
router:
  endpoint_picker_config: {os.path.basename(router_file)}
""")
    return f"scenario_{os.path.basename(router_file)}"


def cmd_router_run(a):
    for n, name in itertools.product(a.engines, a.routers):
        d = router_dir(n)
        aug, blis_cfg = ROUTERS[name]
        taskset = f"taskset -c {CORES} nice -n 10"
        if "augur" in a.sims and not os.path.exists(f"{d}/augur_{name}.csv"):
            t = time.time()
            r = sh(f"{taskset} {PY} -m augur.cli --model Qwen/Qwen3-1.7B --hardware RTX4060Ti --cost tables --engines {n} "
                   f"--router {aug} --workers {a.workers} --num-gpu-blocks {BLOCKS} --max-num-seqs 128 --max-num-batched-tokens 2048 "
                   f"--max-model-len {MML} --no-async-scheduling --step-cpu-us 2000 --filler-vocab 150000 --quiet "
                   f"--workload {d}/wl.jsonl --output {d}/augur_{name}.csv > {d}/augur_{name}.log 2>&1",
                   {"VLLM_LOGGING_LEVEL": "ERROR", "PYTHONPATH": ROOT})
            print(n, "augur", name, "exit", r.returncode, f"{time.time() - t:.0f}s", flush=True)
        if "blis" in a.sims and not os.path.exists(f"{d}/blis_{name}.json"):
            t = time.time()
            scen = router_scenario(n, blis_cfg)
            r = sh(f"{taskset} {BLIS_PIPE} replay --scenarios {d}/scen --scenario {scen} --trace-header {d}/blis.yaml "
                   f"--trace-data {d}/blis.csv --num-instances {n} --output {d}/blis_{name}.json > {d}/blis_{name}.log 2>&1",
                   {"BLIS_NUM_GPU_BLOCKS": str(BLOCKS), "BLIS_PRICER_CMD":
                    f"env PYTHONPATH={ROOT} VLLM_LOGGING_LEVEL=ERROR {PY} {HERE}/pricer.py serve --max-model-len {MML}"})
            print(n, "blis", name, "exit", r.returncode, f"{time.time() - t:.0f}s", flush=True)


def router_metrics(n, name, sim):
    """Mean and p99 TTFT, mean and p99 E2E (s) as a client sees them, and the prefix-hit share of prompt tokens (augur only)."""
    d = router_dir(n)
    wl = [json.loads(l)["input_toks"] for l in open(f"{d}/wl.jsonl")]
    if sim == "augur":
        rows = list(csv.DictReader(open(f"{d}/augur_{name}.csv")))
        t = np.array([float(r["TTFT"]) for r in rows]) / 1e9
        e = np.array([float(r["latency"]) for r in rows]) / 1e9
    else:
        rr = sorted((r for r in json.load(open(f"{d}/blis_{name}.json"))["request_rows"] if r["fate"] == "completed"),
                    key=lambda r: int(str(r["id"]).split("_")[-1]))
        idx = [int(str(r["id"]).split("_")[-1]) for r in rr]
        fe = np.array([FRONT_S + FRONT_PER_TOKEN_S * wl[i] for i in idx])
        t = np.array([r["ttft_us"] for r in rr]) / 1e6 + fe  # client latency: the front end and the delivery BLIS lacks
        e = np.array([r["e2e_us"] for r in rr]) / 1e6 + fe + DELIVERY_S
        t = t + 0  # (ttft_us counts from the request's arrival at the engine; the front end is before it)
    return [float(np.mean(t)), float(np.percentile(t, 99)), float(np.mean(e)), float(np.percentile(e, 99)), len(t)]


def cmd_router_table(a):
    print("| engines | router | augur TTFT mean / p99 | augur E2E mean / p99 | BLIS TTFT mean / p99 | BLIS E2E mean / p99 |\n| --- | --- | --- | --- | --- | --- |")
    for n in a.engines:
        res = {}
        for name in a.routers:
            row = []
            for sim in ("augur", "blis"):
                try:
                    m = router_metrics(n, name, sim)
                except FileNotFoundError:
                    m = None
                res[(name, sim)] = m
                row.append("-" if m is None else f"{m[0]:.2f} / {m[1]:.2f} | {m[2]:.2f} / {m[3]:.2f} ({m[4]})")
            print(f"| {n} | {name} | " + " | ".join(row) + " |")
        for sim in ("augur", "blis"):
            ok = {k: v[2] for (k, s_), v in res.items() if s_ == sim and v}
            if ok:
                print(f"  {sim} ranking by mean E2E (best first) at {n}: " + " < ".join(f"{k} {v:.2f}" for k, v in sorted(ok.items(), key=lambda kv: kv[1])))


# Real two-engine anchor: no vLLM Router binary here (and no downloads), so the client places each request with augur's
# own port of the router policy (augur/sim/router.py) over two real `vllm serve` engines, GPU 0 and GPU 1. kv-aware is
# fed by the servers' real KV events (zmq); in-flight counts are the client's. Round-robin needs nothing.
def real_router_dir(n, name):
    return os.path.join(ROOT, "out/real", f"router_n{n}_{name}")


def cmd_router_real(a):
    import asyncio
    import collections
    import signal
    import threading
    import msgspec
    import zmq
    from vllm.distributed.kv_events import KVEventBatch
    from vllm.utils.hashing import get_hash_fn_by_name
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash, maybe_convert_block_hash
    from types import SimpleNamespace
    from augur.sim.router import make_router
    from augur.trace.workload import Workload
    from augur.verify.client import Client
    n = 2
    for name in a.routers:
        d = real_router_dir(n, name)
        if os.path.exists(f"{d}/client.csv"):
            continue
        os.makedirs(d, exist_ok=True)
        srv = []
        for i in range(n):
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(i), "HF_HUB_OFFLINE": "1", "PYTHONHASHSEED": "0",
                   "AUGUR_STEP_LOG": f"{d}/steps.e{i}", "PYTHONPATH": ROOT}
            cfg = json.dumps({"enable_kv_cache_events": True, "publisher": "zmq", "endpoint": f"tcp://*:{5560 + i}"})
            srv.append(subprocess.Popen(
                [os.path.expanduser("~/augur/.venv/bin/vllm"), "serve", "Qwen/Qwen3-1.7B", "--port", str(8011 + i),
                 "--seed", "0", "--max-num-seqs", "128", "--max-num-batched-tokens", "2048", "--max-model-len", str(MML),
                 "--gpu-memory-utilization", "0.85", "--no-async-scheduling", "--scheduler-cls",
                 "augur.verify.hooks.LoggingScheduler", "--kv-events-config", cfg],
                env=env, stdout=open(f"{d}/server{i}.log", "w"), stderr=subprocess.STDOUT))
        stop = threading.Event()
        try:
            for i in range(n):
                for _ in range(300):
                    if sh(f"curl -sf http://localhost:{8011 + i}/health", stdout=subprocess.DEVNULL).returncode == 0:
                        break
                    assert srv[i].poll() is None, f"server {i} died"
                    time.sleep(2)
            fn = get_hash_fn_by_name("sha256")
            init_none_hash(fn)
            hasher = get_request_block_hasher(16, fn)
            q = collections.deque()

            def listen(i):
                sock = zmq.Context.instance().socket(zmq.SUB)
                sock.connect(f"tcp://localhost:{5560 + i}")
                sock.setsockopt(zmq.SUBSCRIBE, b"")
                dec = msgspec.msgpack.Decoder(KVEventBatch)
                while not stop.is_set():
                    if sock.poll(200):
                        q.append((i, dec.decode(sock.recv_multipart()[-1]).events))
            for i in range(n):
                threading.Thread(target=listen, args=(i,), daemon=True).start()
            time.sleep(1)

            class E:
                def __init__(s, i):
                    s.engine_id, s.n, s.spec = i, 0, SimpleNamespace(block_size=16)

                def load(s):
                    return s.n

                def block_hasher(s, ids):
                    req = SimpleNamespace(all_token_ids=list(ids), num_tokens=len(ids), block_hashes=[], mm_features=None,
                                          lora_request=None, cache_salt=None, prompt_embeds=None)
                    return [maybe_convert_block_hash(h) for h in hasher(req)]
            engines = [E(i) for i in range(n)]
            router = make_router(ROUTERS[name][0])
            placed = {}

            class RouterClient(Client):
                async def _send(s, http, call):
                    while q:
                        i, ev = q.popleft()
                        if hasattr(router, "observe"):
                            router.observe(i, ev)
                    e = router.route(call, engines, s._now())
                    placed[call.call_id] = e
                    engines[e].n += 1
                    s.url = f"http://localhost:{8011 + e}/v1/completions"
                    try:
                        await Client._send(s, http, call)
                    finally:
                        engines[e].n -= 1
                        router.release(call, s._now())

            c = RouterClient("http://localhost:8011", "Qwen/Qwen3-1.7B",
                             Workload(f"{router_dir(n)}/wl.jsonl", filler_vocab=150000), f"{d}/tokens.jsonl")
            if hasattr(router, "first_token"):
                c.on_first_token = lambda call: router.first_token(call, c._now())
            rec = asyncio.run(c.run())
            rec.write_csv(f"{d}/client.csv")
            json.dump({"placement": placed, "failed": len(c.failed)}, open(f"{d}/placement.json", "w"))
            print(name, "done", len(rec.done), "failed", len(c.failed), flush=True)
        finally:
            stop.set()
            for p_ in srv:
                p_.send_signal(signal.SIGINT)
            for p_ in srv:
                p_.wait()
            time.sleep(5)

def _hits(logs):
    """Prefix-hit share of prompt tokens and requests per engine, from step logs (a request's computed tokens at its first schedule)."""
    out = []
    for f in logs:
        prompt, hit = {}, {}
        for line in open(f):
            e = json.loads(line)
            if e["ev"] == "add":
                prompt[e["rid"]] = e["prompt"]
            elif e["ev"] == "sched":
                hit.update({r: c for r, c in e["new"].items() if r not in hit})
        out.append((sum(hit.values()), sum(prompt.values()), len(prompt)))
    return out


def cmd_router_real_table(a):
    """Real vs augur vs BLIS at 2 engines: client TTFT / E2E, prefix-hit share, placement, and augur's replay of the real run."""
    n = 2
    d0 = router_dir(n)
    print("| router | who | TTFT mean / p99 (s) | E2E mean (s) | prefix hit | engine 0 / 1 requests | same engine as real |\n| --- | --- | --- | --- | --- | --- | --- |")
    for name in a.routers:
        rd = real_router_dir(n, name)
        rows = {c: np.array([[float(r["TTFT"]), float(r["latency"])] for r in csv.DictReader(open(f))]) / 1e9
                for c, f in (("real", f"{rd}/client.csv"), ("augur", f"{d0}/augur_{name}.csv")) if os.path.exists(f)}
        place = {"real": {k: v for k, v in json.load(open(f"{rd}/placement.json"))["placement"].items()}}
        logs = sorted(glob.glob(f"{rd}/steps.e*.jsonl"))
        hit = {"real": _hits(logs)}
        for who in ("augur",):
            rr = list(csv.DictReader(open(f"{d0}/augur_{name}.csv")))
            place[who] = {f"req_{r['request id']}": int(r["instance id"]) for r in rr}  # calls are req_<index> in router-gen's trace
            hit[who] = [(sum(int(r["cached_tokens"]) for r in rr if int(r["instance id"]) == i),
                         sum(int(r["input"]) for r in rr if int(r["instance id"]) == i), 0) for i in range(n)]
        b = json.load(open(f"{d0}/blis_{name}.json"))
        place["blis"] = {f"req_{r['id'].split('_')[-1]}": int(r["instance"].split("_")[-1]) for r in b["request_rows"]}
        wl = [json.loads(l)["input_toks"] for l in open(f"{d0}/wl.jsonl")]
        te = np.array([[r["ttft_us"] / 1e6 + FRONT_S + FRONT_PER_TOKEN_S * wl[int(r["id"].split("_")[-1])],
                        r["e2e_us"] / 1e6 + FRONT_S + FRONT_PER_TOKEN_S * wl[int(r["id"].split("_")[-1])] + DELIVERY_S]
                       for r in b["request_rows"] if r["fate"] == "completed"])
        rows["blis"] = te
        hit["blis"] = None
        for who in ("real", "augur", "blis"):
            m = rows[who]
            share = (f"{100 * sum(h[0] for h in hit[who]) / sum(h[1] for h in hit[who]):.1f}%" if hit[who]
                     else f"{100 * b['reports']['engine']['cluster']['cache_hit_rate']:.1f}% (cluster)")
            pl = place[who]
            agree = "-" if who == "real" else f"{100 * np.mean([pl[k] == place['real'][k] for k in place['real']]):.0f}%"
            print(f"| {name} | {who} | {m[:, 0].mean():.2f} / {np.percentile(m[:, 0], 99):.2f} | {m[:, 1].mean():.2f} | {share} | "
                  f"{sum(v == 0 for v in pl.values())} / {sum(v == 1 for v in pl.values())} | {agree} |")
        if os.path.exists(f"{rd}/replay.json"):
            print("  augur replay of the real run, steps identical per engine:", open(f"{rd}/replay.json").read().strip())


def cmd_router_real_replay(a):
    """augur replays each real 2-engine run (the logged durations, arrivals and placement), then compares each engine's steps."""
    n = 2
    for name in a.routers:
        rd = real_router_dir(n, name)
        logs = sorted(glob.glob(f"{rd}/steps.e*.jsonl"))
        blocks = [int(re.findall(r"GPU KV cache size: ([0-9,]+)", open(f"{rd}/server{i}.log").read())[-1].replace(",", "")) // 16
                  for i in range(n)]
        if len(set(blocks)) != 1:
            print("note: engines have different KV sizes", blocks)
        sh(f"rm -f {rd}/augur_replay.*.jsonl")
        sh(f"taskset -c {CORES} nice -n 10 {PY} -m augur.cli --model Qwen/Qwen3-1.7B --engines {n} --num-gpu-blocks {blocks[0]} "
           f"--max-num-seqs 128 --max-num-batched-tokens 2048 --max-model-len {MML} --no-async-scheduling --no-recorded-outputs "
           f"--filler-vocab 150000 --quiet --workload {router_dir(n)}/wl.jsonl --replay-durations {rd}/steps "
           f"--step-log-vllm {rd}/augur_replay --output {rd}/augur_replay.csv > {rd}/augur_replay.log 2>&1",
           {"VLLM_LOGGING_LEVEL": "ERROR", "PYTHONPATH": ROOT})
        res = []
        for i in range(n):
            f = sorted(glob.glob(f"{rd}/augur_replay.e{i}.*.jsonl"))
            r = json.loads(sh(f"{PY} -m augur.verify.replay compare {logs[i]} {f[0]}", capture_output=True, text=True,
                              env={"PYTHONPATH": ROOT}).stdout)
            res.append(f"e{i} {r['identical_steps']} of {r['steps_a']}")
        open(f"{rd}/replay.json", "w").write(", ".join(res) + "\n")
        print(name, ", ".join(res), flush=True)


# KV size sweep: --num-gpu-blocks-override on the tenants workload (None: the engine's own 5,334 blocks)
KV_BLOCKS = [1000, 1800, 2800, 4000, 5334]


def kv_dir(b, tag=""):
    return os.path.join(ROOT, "out/real", f"sweep_kv{b}{tag}")


def cmd_kv_real(a):
    for b in a.blocks:
        d = kv_dir(b, a.tag)
        if os.path.exists(f"{d}/client.json") and os.path.getsize(f"{d}/client.json"):
            continue
        t = time.time()
        r = sh(f"flock ~/gpu0.lock {ROOT}/scripts/real_trace_run.sh sweep_kv{b}{a.tag} {WORKLOAD} '' --num-gpu-blocks-override {b}",
               cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("kv", b, "exit", r.returncode, f"{time.time() - t:.0f}s", flush=True)


def cmd_kv_sims(a):
    def one(b):
        n = f"sweep_kv{b}"
        run_sh(n, kv_dir(b), WORKLOAD, "prepare augur blis compare", {"SHARED_KV": "on", "NUM_GPU_BLOCKS": str(b)}, pin=True)
        sh(f"rm -f {base_dir(n)}/*_steps*.jsonl")
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(one, a.blocks))


def cmd_kv_table(a):
    sims = ("real", "augur", "blis")
    for m in ("e2e", "ttft"):
        res = {b: {k: json.load(open(f"{base_dir(f'sweep_kv{b}')}/results.json"))[0]["sims"][k]["abs"][m][0] for k in sims}
               for b in a.blocks}
        print(f"\nmean {m} (s)\n| blocks | " + " | ".join(sims) + " | preemptions real / augur / BLIS |\n|" + " --- |" * (len(sims) + 2))
        for b in a.blocks:
            r = json.load(open(f"{base_dir(f'sweep_kv{b}')}/results.json"))[0]["sims"]
            print(f"| {b} | " + " | ".join(f"{res[b][k]:.2f}" for k in sims) + " | " +
                  " / ".join(str(r[k].get("pre", "-")) for k in sims) + " |")
        real = [res[b]["real"] for b in a.blocks]
        for k in sims[1:]:
            y = [res[b][k] for b in a.blocks]
            p = int(np.argmin(y))
            print(f"{k}: picks {a.blocks[p]} (real best {a.blocks[int(np.argmin(real))]}), regret {100 * (real[p] / min(real) - 1):.1f}%, "
                  f"tau {kendall(real, y):+.2f}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for n, f in (("real", cmd_real), ("sims", cmd_sims), ("table", cmd_table), ("throughput", cmd_throughput)):
        s = sub.add_parser(n)
        s.add_argument("--values", type=int, nargs="+", default=VALUES)
        s.set_defaults(f=f)
    sub.choices["sims"].add_argument("--jobs", type=int, default=2)
    sub.choices["throughput"].add_argument("--budgets", type=int, default=10)
    sub.choices["throughput"].add_argument("--rates", type=int, default=10)
    sub.choices["throughput"].add_argument("--parallel", type=int, default=4)
    for n, f in (("spec-real", cmd_spec_real), ("spec-sims", cmd_spec_sims), ("spec-table", cmd_spec_table), ("spec-noise", cmd_spec_noise)):
        s = sub.add_parser(n)
        s.add_argument("--workloads", nargs="+", default=list(SPEC_WL))
        s.add_argument("--ks", type=int, nargs="+", default=SPEC_K)
        s.set_defaults(f=f)
    sub.choices["spec-real"].add_argument("--tag", default="", help="suffix of the run name (a repeat)")
    sub.choices["spec-sims"].add_argument("--jobs", type=int, default=3)
    for n, f in (("router-gen", cmd_router_gen), ("router-run", cmd_router_run), ("router-table", cmd_router_table)):
        s = sub.add_parser(n)
        s.add_argument("--engines", type=int, nargs="+", default=[64, 256, 1024])
        s.set_defaults(f=f)
    for n in ("router-run", "router-table"):
        sub.choices[n].add_argument("--routers", nargs="+", default=list(ROUTERS))
    sub.choices["router-run"].add_argument("--sims", nargs="+", default=["augur", "blis"])
    sub.choices["router-run"].add_argument("--workers", type=int, default=3)
    for n, f in (("router-real-table", cmd_router_real_table), ("router-real-replay", cmd_router_real_replay)):
        s = sub.add_parser(n)
        s.add_argument("--routers", nargs="+", default=list(ROUTERS))
        s.set_defaults(f=f)
    s = sub.add_parser("router-real")
    s.add_argument("--routers", nargs="+", default=list(ROUTERS))
    s.set_defaults(f=cmd_router_real)
    for n, f in (("kv-real", cmd_kv_real), ("kv-sims", cmd_kv_sims), ("kv-table", cmd_kv_table)):
        s = sub.add_parser(n)
        s.add_argument("--blocks", type=int, nargs="+", default=KV_BLOCKS)
        s.set_defaults(f=f)
    sub.choices["kv-real"].add_argument("--tag", default="")
    sub.choices["kv-sims"].add_argument("--jobs", type=int, default=3)
    a = p.parse_args(argv)
    a.f(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
