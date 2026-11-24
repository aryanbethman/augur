"""Simulator speed test: augur against BLIS on the same replicated workload at 16-1,024 single-GPU engines.

Each workload is built from the repo's ShareGPT (300 requests) or SWE-bench (50 sessions) files, cycled in a seeded
random order so that every GPU sees the same load: ``--per-gpu`` calls (ShareGPT) or sessions (SWE-bench) per GPU, arriving
as a Poisson process at ``--rate`` per GPU per second. Only lengths are kept (no token ids, no prefix caching in either
simulator, so no sharing to encode), the router is round-robin in both, and the engine is vLLM's default
(max-num-seqs 128, 2,048 batched tokens). One directory holds both simulators' inputs::

    python scripts/baselines/speed.py gen  OUT --kind sharegpt|swe --gpus N [--per-gpu R --rate L]
    python scripts/baselines/speed.py run  OUT --sim augur|blis|blis-pipe [--cores 4,5] [--rep K] [--results F]
    python scripts/baselines/speed.py table F...                      # median and spread over repetitions
    python scripts/baselines/speed.py profile OUT [--top 40]          # pyinstrument of augur: self time by code, inclusive by function

``run`` is one measurement: the simulator under ``taskset -c CORES /usr/bin/time -v`` (wall, user+sys CPU, peak RSS),
with the load average read before and after, appended to ``--results`` as a JSON line with the simulated time and the
steps the simulator took. ``blis`` is BLIS's own latency model (qwen3-14b on an H100 from its catalog: no Qwen3-1.7B
or RTX 4060 Ti there, so its steps differ from augur's); ``blis-pipe`` takes augur's step price through
pricer.py, as run.sh does, so its steps match augur's and the difference is the pipe.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time

HOME = os.path.expanduser("~")
BASE = os.environ.get("BASELINES", f"{HOME}/baselines")
PY = os.environ.get("AUGUR_PYTHON", f"{HOME}/augur/.venv/bin/python")
SRC = os.environ.get("AUGUR_FORK", f"{HOME}/llmss-vllm-sim-20261114") + "/workloads"
BASE_FILES = {"sharegpt": "sharegpt-qwen3-32b-300-sps10.jsonl", "swe": "swe-bench-qwen3-30b-a3b-50-sps0.2.jsonl"}
# engine settings per kind: the real runs' (block counts of out/baselines/sharegpt300 and swe50)
ENGINE = {"sharegpt": dict(max_model_len=8192, blocks=5334), "swe": dict(max_model_len=24576, blocks=5480)}
FRONT_S, FRONT_PER_TOKEN_S, DELIVERY_S = 3.4e-3, 3e-6, 8e-3  # augur's default front end and delivery, which BLIS lacks
SEQS, BATCHED = 128, 2048
SHARED = os.environ.get("BLIS_SHARED", f"{HOME}/speedrun/blis-ext-count")


def _base(kind: str) -> list[dict]:
    rows = []
    with open(os.path.join(SRC, BASE_FILES[kind])) as f:
        for line in f:
            d = json.loads(line)
            if "sub_requests" in d:
                d["sub_requests"] = [{k: s[k] for k in ("input_toks", "output_toks", "tool_duration_ns")}
                                     for s in d["sub_requests"]]
            else:
                d = {k: d[k] for k in ("input_toks", "output_toks")}
            rows.append(d)
    return rows


def gen(a) -> int:
    kind, n = a.kind, a.gpus
    per = a.per_gpu or (12 if kind == "sharegpt" else 1)
    rate = a.rate or (1.0 if kind == "sharegpt" else 0.1)
    rng = random.Random(a.seed)
    base = _base(kind)
    count = n * per
    order = []
    while len(order) < count:
        o = list(range(len(base)))
        rng.shuffle(o)
        order += o
    t, entries = 0.0, []
    for i in range(count):
        t += rng.expovariate(rate * n)
        d = dict(base[order[i]])
        d["arrival_time_ns"] = int(t * 1e9)
        if kind == "swe":
            d["session_id"] = f"s{i}"
        entries.append(d)
    os.makedirs(os.path.join(a.out, "scen"), exist_ok=True)
    with open(os.path.join(a.out, "wl.jsonl"), "w") as f:
        for d in entries:
            f.write(json.dumps(d) + "\n")
    # BLIS: sessions are closed loop (think_time = the tool wait + delivery + the next call's front end)
    sessions = kind == "swe"
    cols = ("request_id,client_id,tenant_id,slo_class,session_id,round_index,prefix_group,prefix_length,streaming,"
            "input_tokens,output_tokens,text_tokens,image_tokens,audio_tokens,video_tokens,reason_ratio,model,"
            "deadline_us,arrival_time_us,first_chunk_time_us,last_chunk_time_us,status"
            + (",think_time_us,input_tokens_reset" if sessions else "") + "\n")
    rows, calls = [], 0
    for d in entries:
        if not sessions:
            fe = FRONT_S + FRONT_PER_TOKEN_S * d["input_toks"]
            rows.append((d["arrival_time_ns"] / 1e9 + fe, f"{{i}},c,,,,0,,0,false,{d['input_toks']},{d['output_toks']},"
                         f"0,0,0,0,0,,0,{{arr}},0,0,ok\n"))
            calls += 1
            continue
        prev, at = None, d["arrival_time_ns"] / 1e9
        for k, s in enumerate(d["sub_requests"]):
            fe = FRONT_S + FRONT_PER_TOKEN_S * s["input_toks"]
            if prev is None:
                n_in, reset, tt, at = s["input_toks"], "", "", at + fe
            else:
                grown = prev["input_toks"] + prev["output_toks"]
                n_in, reset = (s["input_toks"] - grown, "") if s["input_toks"] >= grown else (0, str(s["input_toks"]))
                wait = prev["tool_duration_ns"] / 1e9 + DELIVERY_S + fe
                tt, at = str(round(wait * 1e6)), at + wait  # nominal: the loop sets the real arrival
            rows.append((at, f"{{i}},c,,,{d['session_id']},{k},,0,false,{n_in},{s['output_toks']},0,0,0,0,0,,0,{{arr}},0,0,"
                         f"ok,{tt},{reset}\n"))
            prev = s
            calls += 1
    rows.sort(key=lambda r: r[0])  # stable: a session's rounds stay in order
    with open(os.path.join(a.out, "blis.csv"), "w") as f:
        f.write(cols)
        for i, (arr, fmt) in enumerate(rows):
            f.write(fmt.replace("{i}", str(i)).replace("{arr}", str(round(arr * 1e6))))
    with open(os.path.join(a.out, "blis.yaml"), "w") as f:
        f.write("trace_version: 4\ntime_unit: microseconds\nmode: converted\nworkload_seed: 1\n"
                + ("session_context_growth: accumulate\n" if sessions else ""))
    e = ENGINE[kind]
    with open(os.path.join(a.out, "scen", "scenario.yaml"), "w") as f:
        f.write(f"""kind: Scenario
name: speed
engine: {{kind: vllm, version: "0.33.0"}}
model: qwen3-14b
coefficients: [cost-model-primitives, cost-model-collectives, cost-model-attention, cost-model-recurrent, cost-model-memory]
cluster:
  hardware: h100
  nodes: 1
  gpus_per_node: {n}
---
kind: Deployment
name: speed
pools:
  - role: colocated
    nodes: 1
    parallel: {{tp: 1, pp: 1, dp: 1}}
    engine:
      block_size: 16
      max_num_batched_tokens: {BATCHED}
      max_num_seqs: {SEQS}
      max_model_len: {e['max_model_len']}
      gpu_memory_utilization: 0.85
      async_scheduling: false
      enable_prefix_caching: false
router:
  endpoint_picker_config: round-robin.yaml
""")
    cfg = subprocess.run([f"{BASE}/inference-sim/blis", "router-configs", "--file", "round-robin.yaml"],
                         capture_output=True, text=True, check=True).stdout
    open(os.path.join(a.out, "scen", "round-robin.yaml"), "w").write(cfg)
    json.dump(dict(kind=kind, gpus=n, per_gpu=per, rate=rate, seed=a.seed, entries=len(entries), calls=calls,
                   **e), open(os.path.join(a.out, "meta.json"), "w"))
    print(f"{kind} {n} GPUs: {len(entries)} entries, {calls} calls, {os.path.getsize(os.path.join(a.out, 'wl.jsonl')) >> 10} KiB")
    return 0


def _load() -> float:
    with open("/proc/loadavg") as f:
        return float(f.read().split()[0])


def _cmd(sim: str, out: str, m: dict, startup: bool = False, count: bool = False) -> tuple[list[str], dict, str]:
    """The command, its extra environment and where its result goes."""
    n = m["gpus"]
    if sim == "augur":
        c = [PY, "-m", "augur.cli", "--model", "Qwen/Qwen3-1.7B", "--hardware", "RTX4060Ti", "--cost", "tables",
             "--engines", str(n), "--router", "round-robin", "--num-gpu-blocks", str(m["blocks"]),
             "--max-num-seqs", str(SEQS), "--max-num-batched-tokens", str(BATCHED), "--max-model-len",
             str(m["max_model_len"]), "--no-prefix-caching", "--no-async-scheduling", "--step-cpu-us", "2000",
             "--no-recorded-outputs", "--filler-vocab", "150000", "--quiet", "--workload", f"{out}/wl.jsonl",
             "--output", f"{out}/augur.csv"] + (["--num-entries", "1"] if startup else [])
        return c, {"VLLM_LOGGING_LEVEL": "ERROR", "PYTHONPATH": os.getcwd()}, f"{out}/augur.csv"
    # blis: the plain patched binary (its own latency model); blis-pipe: a build whose engines share one pricer process
    # (blis-ext starts one per engine, 1.2 GB each) and that counts steps (BLIS_STEP_COUNT_FILE)
    c = [SHARED if sim == "blis-pipe" or count else f"{BASE}/inference-sim/blis-ext", "replay", "--scenarios", f"{out}/scen", "--scenario", "scenario.yaml",
         "--trace-header", f"{out}/blis.yaml", "--trace-data", f"{out}/blis.csv", "--num-instances", str(n),
         "--output", f"{out}/blis_{sim}.json"]
    if m["kind"] == "swe":
        c += ["--session-mode", "closed-loop"]
    env = {"BLIS_NUM_GPU_BLOCKS": str(m["blocks"])}  # the real runs KV pool, as augur
    if sim == "blis-pipe":
        env["BLIS_STEP_COUNT_FILE"] = f"{out}/steps_blis-pipe.txt"  # rewritten every 64 steps: short by at most 63
        env["BLIS_PRICER_CMD"] = (f"/usr/bin/time -v -o {out}/time_pricer.txt env PYTHONPATH={os.getcwd()} VLLM_LOGGING_LEVEL=ERROR {PY} "
                                  f"{os.path.dirname(os.path.abspath(__file__))}/pricer.py serve "
                                  f"--max-model-len {m['max_model_len']}")
    return c, env, f"{out}/blis_{sim}.json"


def run(a) -> int:
    m = json.load(open(os.path.join(a.out, "meta.json")))
    if a.count:  # untimed: how many steps BLIS prices (its own model has no step log)
        cmd, env, _ = _cmd(a.sim, a.out, m, count=True)
        f = os.path.join(a.out, f"steps_{a.sim}.txt")
        env["BLIS_STEP_COUNT_FILE"] = f
        subprocess.run(cmd, env={**os.environ, **env}, capture_output=True, check=True)
        print(a.sim, "steps", open(f).read().strip())
        return 0
    cmd, env, res = _cmd(a.sim, a.out, m, a.startup)
    tfile = os.path.join(a.out, f"time_{a.sim}.txt")
    full = ["taskset", "-c", a.cores, "/usr/bin/time", "-v", "-o", tfile] + cmd
    l0, t0 = _load(), time.time()
    p = subprocess.run(full, env={**os.environ, **env}, capture_output=True, text=True)
    l1, wall = _load(), time.time() - t0
    if p.returncode:
        print(p.stderr[-2000:], file=sys.stderr)
        return p.returncode
    tt = open(tfile).read()

    def g(pat):
        return float(re.search(pat, tt).group(1))
    user, sysc, rss = g(r"User time \(seconds\): ([\d.]+)"), g(r"System time \(seconds\): ([\d.]+)"), g(r"Maximum resident set size \(kbytes\): (\d+)")
    h = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)", tt).group(1).split(":")
    elapsed = sum(float(x) * 60 ** i for i, x in enumerate(reversed(h)))
    pcpu = prss = 0.0
    if a.sim == "blis-pipe":  # the pricer is the simulator's grandchild, which time -v does not count: it is timed itself
        pf = os.path.join(a.out, "time_pricer.txt")
        for _ in range(100):
            if os.path.exists(pf) and "Maximum resident" in open(pf).read():
                break
            time.sleep(0.1)
        pt = open(pf).read()
        pcpu = (float(re.search(r"User time \(seconds\): ([\d.]+)", pt).group(1))
                + float(re.search(r"System time \(seconds\): ([\d.]+)", pt).group(1)))
        prss = float(re.search(r"Maximum resident set size \(kbytes\): (\d+)", pt).group(1))
        os.remove(pf)
    r = dict(sim=a.sim + ("-startup" if a.startup else ""), kind=m["kind"], gpus=m["gpus"], rep=a.rep, cores=a.cores, calls=m["calls"], wall_s=elapsed,
             user_s=user, sys_s=sysc, pricer_cpu_s=pcpu, cpu_s=user + sysc + pcpu, rss_mb=(rss + prss) / 1024, pricer_rss_mb=prss / 1024, load_before=l0, load_after=l1)
    if a.sim == "augur":
        s = json.loads(p.stdout[p.stdout.index("{"):p.stdout.rindex("}") + 1]) if "{" in p.stdout else {}
        r["steps"], r["sim_s"] = s.get("steps"), s.get("sim_seconds")
        r["augur_wall_s"] = s.get("wall_s")
        r["done"] = s.get("calls")
    else:
        d = json.load(open(res))
        r["sim_s"] = d["window"]["duration_s"]
        r["done"] = d["requests"]["completed"]
        r["timed_out"] = d["requests"]["timed_out"]
        sf = os.path.join(a.out, f"steps_{a.sim}.txt")
        r["steps"] = int(open(sf).read()) if os.path.exists(sf) and open(sf).read().strip() else None
    if a.results:
        with open(a.results, "a") as f:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(r))
    return 0


def table(a) -> int:
    runs = [json.loads(l) for fn in a.files for l in open(fn)]
    groups: dict = {}
    for r in runs:
        groups.setdefault((r["kind"], r["gpus"], r["sim"]), []).append(r)

    def ms(xs, f="{:.1f}"):
        return f.format(statistics.median(xs)) + (f" ({f.format(min(xs))}-{f.format(max(xs))})" if len(xs) > 1 else "")
    print("workload gpus sim n | wall s | cpu s | rss MB | sim/wall | steps | steps/cpu-s | load before..after")
    for (k, n, s), rs in sorted(groups.items()):
        st = [r["steps"] for r in rs if r.get("steps")]
        print(k, n, s, len(rs), "|", ms([r["wall_s"] for r in rs]), "|", ms([r["cpu_s"] for r in rs]), "|",
              ms([r["rss_mb"] for r in rs], "{:.0f}"), "|", ms([r["sim_s"] / r["wall_s"] for r in rs if r.get("sim_s")], "{:.2f}"), "|",
              f"{statistics.median(st):.0f}" if st else "-", "|",
              ms([r["steps"] / r["cpu_s"] for r in rs if r.get("steps")], "{:.0f}") if st else "-", "|",
              f"{min(r['load_before'] for r in rs):.1f}-{max(r['load_after'] for r in rs):.1f}")
    return 0


def profile(a) -> int:
    """pyinstrument (sampling) of augur in process: where the time of the whole run goes, by file and by function."""
    from pyinstrument import Profiler
    m = json.load(open(os.path.join(a.out, "meta.json")))
    cmd, env, _ = _cmd("augur", a.out, m)
    os.environ.update(env)
    argv = cmd[3:]
    argv[argv.index("--output") + 1] = f"{a.out}/augur_prof.csv"
    from augur import cli
    pr = Profiler(interval=a.interval)
    t0 = time.time()
    pr.start()
    cli.main(argv)
    pr.stop()
    wall = time.time() - t0
    root = pr.last_session.root_frame()
    groups = [  # first match wins
        ("vLLM scheduler (v1/core/sched)", "vllm/v1/core/sched"),
        ("vLLM KV-cache manager and block pool", "vllm/v1/core/"),
        ("vLLM request, engine core, other", "vllm/"),
        ("augur coordinator", "augur/sim/coordinator"),
        ("augur router", "augur/sim/router"),
        ("augur cost model (pricer)", "augur/cost"),
        ("augur engine, executor, other sim", "augur/"),
        ("numpy, torch", "site-packages/numpy"), ("numpy, torch", "site-packages/torch"),
    ]
    own, incl = {}, {}

    def walk(f, stack, path):
        path = f.file_path if f.function != "[self]" else path  # pyinstrument's [self] leaf is its parent's own time
        kids = f.children
        g = next((n for n, k in groups if k in (path or "")), "other (stdlib, third party, builtins)")
        if f.function == "[self]" or not kids:
            own[g] = own.get(g, 0.0) + f.time
        key = f"{f.function} ({os.path.basename(f.file_path or '')})"
        if key not in stack and f.function != "[self]":
            incl[key] = incl.get(key, 0.0) + f.time
        for c in kids:
            walk(c, stack | {key}, path)

    def find(f, fn, file):
        if f.function == fn and (f.file_path or "").endswith(file):
            return [f]
        return [x for c in f.children for x in find(c, fn, file)]
    sys.setrecursionlimit(100000)
    runs = find(root, "run", "coordinator.py")
    builds = find(root, "_build_engines", "cli.py")
    loop = sum(f.time for f in runs)
    print(f"{a.out}: profiled {root.time:.1f} s of {wall:.1f} s wall; engine setup {sum(f.time for f in builds):.1f} s, "
          f"event loop (Coordinator.run) {loop:.1f} s, the rest is imports and output")
    for f in runs:
        walk(f, frozenset(), "")
    print("\nevent loop: self time by code")
    for g, t in sorted(own.items(), key=lambda x: -x[1]):
        print(f"  {100 * t / loop:5.1f}%  {t:7.1f} s  {g}")
    print("\nevent loop: inclusive time of the top functions (a function inside another counts in both)")
    for k, t in sorted(incl.items(), key=lambda x: -x[1])[:a.top]:
        print(f"  {100 * t / loop:5.1f}%  {t:7.1f} s  {k}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="speed")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("out")
    g.add_argument("--kind", choices=("sharegpt", "swe"), required=True)
    g.add_argument("--gpus", type=int, required=True)
    g.add_argument("--per-gpu", type=int, default=None, help="calls (sharegpt) or sessions (swe) per GPU; default 12 / 1")
    g.add_argument("--rate", type=float, default=None, help="arrivals per GPU per second; default 1.0 / 0.1")
    g.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("run")
    r.add_argument("out")
    r.add_argument("--sim", choices=("augur", "blis", "blis-pipe"), required=True)
    r.add_argument("--cores", default="4,5")
    r.add_argument("--rep", type=int, default=0)
    r.add_argument("--results", default=None)
    r.add_argument("--count", action="store_true", help="BLIS only, untimed: count its steps into OUT/steps_SIM.txt")
    r.add_argument("--startup", action="store_true", help="augur on its first entry only: the fixed cost of import and engine setup")
    t = sub.add_parser("table")
    t.add_argument("files", nargs="+")
    pf = sub.add_parser("profile")
    pf.add_argument("out")
    pf.add_argument("--top", type=int, default=40)
    pf.add_argument("--interval", type=float, default=0.002)
    a = p.parse_args(argv)
    return {"gen": gen, "run": run, "table": table, "profile": profile}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
