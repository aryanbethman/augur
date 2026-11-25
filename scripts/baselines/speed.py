"""Simulator speed test: augur against BLIS on the same replicated workload at 16-1,024 single-GPU engines.

Each workload is built from the repo's ShareGPT (300 requests) or SWE-bench (50 sessions) files, cycled in a seeded
random order so that every GPU sees the same load: ``--per-gpu`` calls (ShareGPT) or sessions (SWE-bench) per GPU, arriving
as a Poisson process at ``--rate`` per GPU per second. Only lengths are kept (no token ids, no prefix caching in either
simulator, so no sharing to encode), the router is round-robin in both, and the engine is vLLM's default
(max-num-seqs 128, 2,048 batched tokens). One directory holds both simulators' inputs::

    python scripts/baselines/speed.py gen  OUT --kind sharegpt|swe --gpus N [--per-gpu R --rate L]
    python scripts/baselines/speed.py run  OUT --sim augur|blis|blis-pipe [--cores 4,5] [--rep K] [--results F]
    python scripts/baselines/speed.py table F...                      # median and spread over repetitions
    python scripts/baselines/speed.py profile OUT [--top 40]          # pyinstrument of augur: self time by code and function, inclusive by function
    python scripts/baselines/speed.py real LOGDIR                     # busy share and batch size of a real run's steps.*.jsonl

The default load is production-like: ShareGPT at the real sharegpt300 run's 10 req/s per GPU (100 calls per GPU, 10 s of
arrivals), SWE-bench at the real swe50 run's 0.2 sessions/s per GPU (``PER_GPU``). The 2026-11-24 runs used
``--per-gpu 12 --rate 1.0`` (ShareGPT) and ``--per-gpu 1 --rate 0.1`` (SWE-bench); the same seed rebuilds those inputs.

``run`` is one measurement: the simulator under ``taskset -c CORES /usr/bin/time -v`` (wall, user+sys CPU, peak RSS),
with the load average read before and after, appended to ``--results`` as a JSON line with the simulated time and the
steps the simulator took, the share of simulated time each engine is busy (mean and min over engines), the mean batch
size (requests per step) and, for augur, the CPU of start-up (imports and engine setup, up to the event loop) apart from
the event loop's, and steps per steady-state CPU second. augur gets these from ``child`` (this script, run in its place:
it counts steps as they are logged); BLIS from its Result (per-instance busy_share; batch size is output tokens per step,
which leaves out a request still in prefill, and needs the step count of its ``--count`` build, run once after the timed
run; its start-up CPU is not separable). ``blis`` is BLIS's own latency model (qwen3-14b on an H100 from its catalog: no Qwen3-1.7B
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
# default load per GPU: the real runs' rates (sharegpt300 10 req/s, swe50 0.2 sessions/s) and the entries per GPU
PER_GPU = {"sharegpt": 100, "swe": 12}
RATE = {"sharegpt": 10.0, "swe": 0.2}
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
    per = a.per_gpu or PER_GPU[kind]
    rate = a.rate or RATE[kind]
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
                   last_arrival_ns=entries[-1]["arrival_time_ns"], **e), open(os.path.join(a.out, "meta.json"), "w"))
    print(f"{kind} {n} GPUs: {len(entries)} entries, {calls} calls, {os.path.getsize(os.path.join(a.out, 'wl.jsonl')) >> 10} KiB")
    return 0


def child(extra: str, window_ns: int, argv: list[str]) -> int:
    """augur's CLI in this process, plus what ``run`` reports: per engine the steps, busy time (the logged step
    durations; all, and up to ``window_ns``, the last arrival), requests per step, and the process CPU at the start of
    the event loop (imports and engine setup before it) and at its end. Written to ``extra`` as JSON."""
    from augur import cli
    from augur.sim.coordinator import Coordinator

    class Agg:  # stands in for an engine's --step-log list (rows: start, dur, total, prefill tokens, n_decode, n_prefill, ...)
        def __init__(s):
            s.n = s.busy = s.win = s.reqs = 0

        def append(s, row):
            st, d = row[0], row[1]
            s.n += 1
            s.busy += d
            s.win += max(0, min(st + d, window_ns) - st)
            s.reqs += row[4] + row[5]
    engines, cpu = [], {}
    build, run = cli._build_engines, Coordinator.run

    def build_agg(*x, **k):
        engines[:] = build(*x, **k)
        for e in engines:
            e.step_log = Agg()
        return engines

    def timed_run(self):
        cpu["loop_start"] = time.process_time()
        r = run(self)
        cpu["loop_end"] = time.process_time()
        return r
    cli._build_engines, Coordinator.run = build_agg, timed_run
    rc = cli.main(argv)
    json.dump(dict(cpu, engines=[[e.engine_id, e.step_log.n, e.step_log.busy, e.step_log.win, e.step_log.reqs]
                                 for e in engines]), open(extra, "w"))
    return rc


def _load() -> float:
    with open("/proc/loadavg") as f:
        return float(f.read().split()[0])


def _cmd(sim: str, out: str, m: dict, startup: bool = False, count: bool = False) -> tuple[list[str], dict, str]:
    """The command, its extra environment and where its result goes."""
    n = m["gpus"]
    if sim == "augur":
        c = [PY, os.path.abspath(__file__), "child", f"{out}/augur_extra.json", str(m.get("last_arrival_ns", 1 << 62)), "--model", "Qwen/Qwen3-1.7B", "--hardware", "RTX4060Ti", "--cost", "tables",
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
    p = subprocess.Popen(full, env={**os.environ, **env}, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         start_new_session=True)
    while True:  # watch the resident memory of the whole process group: BLIS's grows with the GPUs
        try:
            p.stdout_s, p.stderr_s = p.communicate(timeout=5)
            break
        except subprocess.TimeoutExpired:
            kb = sum(int(x) for x in subprocess.run(["ps", "-o", "rss=", "--sid", str(p.pid)], capture_output=True,
                                                    text=True).stdout.split())
            if kb / 2**20 > a.max_rss_gb:
                os.killpg(p.pid, 9)
                p.communicate()
                r = dict(sim=a.sim, kind=m["kind"], gpus=m["gpus"], rep=a.rep, killed_rss_gb=kb / 2**20,
                         elapsed_s=time.time() - t0, load_before=l0, load_after=_load())
                if a.results:
                    open(a.results, "a").write(json.dumps(r) + "\n")
                print(json.dumps(r))
                return 3
    p.stdout, p.stderr = p.stdout_s, p.stderr_s
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
        x = json.load(open(os.path.join(a.out, "augur_extra.json")))
        es = [e for e in x["engines"] if e[1]]
        T, W = r["sim_s"] * 1e9, min(m["last_arrival_ns"], r["sim_s"] * 1e9)
        b = [e[2] / T for e in es] + [0.0] * (m["gpus"] - len(es))  # an engine that never stepped is idle
        r["busy_mean"], r["busy_min"] = statistics.fmean(b), min(b)
        bw = [e[3] / W for e in es] + [0.0] * (m["gpus"] - len(es))
        r["busy_win_mean"], r["busy_win_min"] = statistics.fmean(bw), min(bw)  # up to the last arrival
        r["batch_mean"] = sum(e[4] for e in es) / max(sum(e[1] for e in es), 1)
        if not a.startup:
            r["startup_cpu_s"], r["steady_cpu_s"] = x["loop_start"], x["loop_end"] - x["loop_start"]
            r["steady_steps_per_cpu_s"] = r["steps"] / r["steady_cpu_s"]
    else:
        d = json.load(open(res))
        r["sim_s"] = d["window"]["duration_s"]
        r["done"] = d["requests"]["completed"]
        r["timed_out"] = d["requests"]["timed_out"]
        sf = os.path.join(a.out, f"steps_{a.sim}.txt")
        if not os.path.exists(sf) and not a.no_count:  # untimed, once: the step count of BLIS's own model
            c2, e2, _ = _cmd(a.sim, a.out, m, count=True)
            e2["BLIS_STEP_COUNT_FILE"] = sf
            subprocess.run(["nice", "-n", "10"] + c2, env={**os.environ, **e2}, capture_output=True, check=True)
        r["steps"] = int(open(sf).read()) if os.path.exists(sf) and open(sf).read().strip() else None
        ins = d["reports"]["engine"]["instances"].values()
        b = [i["measured_window"]["busy_share"] for i in ins if "measured_window" in i]
        r["busy_mean"], r["busy_min"] = statistics.fmean(b), min(b)  # first arrival to last completion
        if r["steps"]:  # output tokens per step: a request still in prefill emits none, so a little below requests per step
            r["batch_mean"] = d["reports"]["engine"]["cluster"]["total_output_tokens"] / r["steps"]
    if a.results:
        with open(a.results, "a") as f:
            f.write(json.dumps(r) + "\n")
    print(json.dumps(r))
    return 0


def table(a) -> int:
    runs = [json.loads(l) for fn in a.files for l in open(fn)]
    groups: dict = {}
    for r in runs:
        if "killed_rss_gb" in r:
            print(r["kind"], r["gpus"], r["sim"], f"killed at {r['killed_rss_gb']:.0f} GB resident after {r['elapsed_s']:.0f} s")
        else:
            groups.setdefault((r["kind"], r["gpus"], r["sim"]), []).append(r)

    def ms(xs, f="{:.1f}"):
        return f.format(statistics.median(xs)) + (f" ({f.format(min(xs))}-{f.format(max(xs))})" if len(xs) > 1 else "") if xs else "-"

    def col(rs, k, f="{:.1f}"):
        return ms([r[k] for r in rs if r.get(k) is not None], f)
    print("workload gpus sim n | wall s | cpu s | startup/steady cpu s | rss MB | sim s | busy mean/min | batch | steps | steps/cpu-s | steady steps/cpu-s | load before..after")
    for (k, n, s), rs in sorted(groups.items()):
        print(k, n, s, len(rs), "|", col(rs, "wall_s"), "|", col(rs, "cpu_s"), "|",
              col(rs, "startup_cpu_s"), "/", col(rs, "steady_cpu_s"), "|", col(rs, "rss_mb", "{:.0f}"), "|", col(rs, "sim_s"), "|",
              col(rs, "busy_mean", "{:.2f}"), "/", col(rs, "busy_min", "{:.2f}"), "|", col(rs, "batch_mean"), "|",
              col(rs, "steps", "{:.0f}"), "|",
              ms([r["steps"] / r["cpu_s"] for r in rs if r.get("steps")], "{:.0f}"), "|", col(rs, "steady_steps_per_cpu_s", "{:.0f}"), "|",
              f"{min(r['load_before'] for r in rs):.1f}-{max(r['load_after'] for r in rs):.1f}")
    return 0


def real(a) -> int:
    """A real run's step log (steps.*.jsonl: sched and upd events): busy share and requests per step. Busy is the time from a
    step's schedule to its update (as augur's logged step duration), over first arrival to last update; also over the
    arrival window (up to the last arrival), the steady part."""
    import glob
    for fn in sorted(glob.glob(os.path.join(a.dir, "steps.*.jsonl"))):
        sched, steps, arrivals = {}, [], []  # steps: (start, end, requests)
        for line in open(fn):
            e = json.loads(line)
            if e["ev"] == "add":
                arrivals.append(e["t"])
            elif e["ev"] == "sched":
                sched[e["k"]] = (e["t"], len(e["tok"]))
            elif e["ev"] == "upd" and e["k"] in sched:
                t0, n = sched.pop(e["k"])
                steps.append((t0, e["t"], n))
        t_a, t_b, t_w = arrivals[0], steps[-1][1], arrivals[-1]
        busy = sum(e - s for s, e, _ in steps) / (t_b - t_a)
        win = [(s, min(e, t_w), n) for s, e, n in steps if s < t_w]
        bw = sum(e - s for s, e, _ in win) / (t_w - t_a)
        print(f"{fn}: {len(steps)} steps over {(t_b - t_a) / 1e9:.1f} s ({(t_w - t_a) / 1e9:.1f} s of arrivals); busy {busy:.3f}; "
              f"requests per step {statistics.fmean(n for *_, n in steps):.2f}; in the arrival window busy {bw:.3f}, "
              f"requests per step {statistics.fmean(n for *_, n in win):.2f}, max {max(n for *_, n in steps)}")
    return 0


def profile(a) -> int:
    """pyinstrument (sampling) of augur in process: where the time of the whole run goes, by file and by function."""
    from pyinstrument import Profiler
    m = json.load(open(os.path.join(a.out, "meta.json")))
    cmd, env, _ = _cmd("augur", a.out, m)
    os.environ.update(env)
    argv = cmd[cmd.index("--model"):]
    if a.prefix_caching:  # the workloads carry no token ids, so there is nothing to share, but every block is still hashed
        argv.remove("--no-prefix-caching")
    argv[argv.index("--output") + 1] = f"{a.out}/augur_prof.csv"
    from augur import cli
    pr = Profiler(interval=a.interval)
    t0 = time.time()
    pr.start()
    cli.main(argv)
    pr.stop()
    wall = time.time() - t0
    root = pr.last_session.root_frame()

    def where(f):
        return (f.file_path or "").replace(HOME, "~")

    def category(f):  # the innermost frame that matches decides; a frame that matches nothing takes its caller's
        p, fn = where(f), f.function
        if "kv_cache_utils" in p and "hash" in fn.lower() or "hash" in fn.lower() and "/vllm/" in p or "vllm/utils/hashing" in p:
            return "prefix-cache hashing"
        if "make_stats" in fn or "vllm/v1/metrics" in p or "vllm/logger" in p or fn == "record_function_or_nullcontext":
            return "vLLM stats, logging, profiling hooks"
        if fn in ("update_from_output", "_update_request_with_output", "check_stop", "append_output_token_ids"):
            return "vLLM output processing (update_from_output)"
        if "vllm/v1/core/sched" in p:
            return "vLLM scheduler"
        if "vllm/v1/core/" in p:
            return "vLLM KV-cache manager and block pool"
        if "/vllm/" in p:
            return "vLLM request, engine core, other"
        if "prefix_share" in p:
            return "augur prefix_share (called by the executor)"
        if "augur/sim/executor" in p:
            return "augur executor"
        if "augur/cost" in p:
            return "augur pricer (augur/cost)"
        if "augur/sim/coordinator" in p or "augur/sim/router" in p or "augur/sim/workload" in p:
            return "augur event loop (coordinator, router, workload)"
        if "augur/" in p:
            return "augur engine wrapper, other"
        return None
    own, selfk, incl = {}, {}, {}

    def walk(f, stack, cat, parent):
        c = category(f) or cat
        key = f"{f.function} ({os.path.basename(f.file_path or '')})"
        if f.function == "[self]" or not f.children:
            own[c] = own.get(c, 0.0) + f.time
            k = parent if f.function == "[self]" or not f.file_path else key  # a built-in's own time is its caller's
            selfk[k] = selfk.get(k, 0.0) + f.time
        if key not in stack and f.function != "[self]":
            incl[key] = incl.get(key, 0.0) + f.time
        for ch in f.children:
            walk(ch, stack | {key}, c, key)

    def find(f, fn, file):
        if f.function == fn and (f.file_path or "").endswith(file):
            return [f]
        return [x for c in f.children for x in find(c, fn, file)]
    sys.setrecursionlimit(100000)
    runs = find(root, "run", "coordinator.py")
    builds = find(root, "_build_engines", "cli.py")
    loop, setup = sum(f.time for f in runs), sum(f.time for f in builds)
    print(f"{a.out}: profiled {root.time:.1f} s of {wall:.1f} s wall; engine setup {setup:.1f} s, "
          f"event loop (Coordinator.run) {loop:.1f} s, the rest ({root.time - loop - setup:.1f} s) is imports and output")
    for f in runs:
        walk(f, frozenset(), "augur event loop (coordinator, router, workload)", "run (coordinator.py)")
    print(f"\nevent loop: self time by code, % of the event loop (engine start-up is {100 * setup / loop:.1f}% of it)")
    for g, t in sorted(own.items(), key=lambda x: -x[1]):
        print(f"  {100 * t / loop:5.1f}%  {t:7.1f} s  {g}")
    print("\nevent loop: top self-time functions (a built-in's time is its caller's)")
    for k, t in sorted(selfk.items(), key=lambda x: -x[1])[:a.top]:
        print(f"  {100 * t / loop:5.1f}%  {t:7.1f} s  {k}")
    print("\nevent loop: inclusive time of the top functions (a function inside another counts in both)")
    for k, t in sorted(incl.items(), key=lambda x: -x[1])[:a.top]:
        print(f"  {100 * t / loop:5.1f}%  {t:7.1f} s  {k}")
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["child"]:  # speed.py child EXTRA_JSON WINDOW_NS AUGUR_ARGS...: what run times for augur
        return child(argv[1], int(argv[2]), argv[3:])
    p = argparse.ArgumentParser(prog="speed")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("out")
    g.add_argument("--kind", choices=("sharegpt", "swe"), required=True)
    g.add_argument("--gpus", type=int, required=True)
    g.add_argument("--per-gpu", type=int, default=None, help=f"calls (sharegpt) or sessions (swe) per GPU; default {PER_GPU}")
    g.add_argument("--rate", type=float, default=None, help=f"arrivals per GPU per second; default {RATE}")
    g.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("run")
    r.add_argument("out")
    r.add_argument("--sim", choices=("augur", "blis", "blis-pipe"), required=True)
    r.add_argument("--cores", default="4,5")
    r.add_argument("--rep", type=int, default=0)
    r.add_argument("--results", default=None)
    r.add_argument("--max-rss-gb", type=float, default=80, help="kill the run past this resident memory (all its processes)")
    r.add_argument("--no-count", action="store_true", help="BLIS: skip the untimed step-counting run after the timed one")
    r.add_argument("--count", action="store_true", help="BLIS only, untimed: count its steps into OUT/steps_SIM.txt")
    r.add_argument("--startup", action="store_true", help="augur on its first entry only: the fixed cost of import and engine setup")
    t = sub.add_parser("table")
    t.add_argument("files", nargs="+")
    rl = sub.add_parser("real")
    rl.add_argument("dir", help="a real run's directory: its steps.*.jsonl")
    pf = sub.add_parser("profile")
    pf.add_argument("out")
    pf.add_argument("--top", type=int, default=40)
    pf.add_argument("--prefix-caching", action="store_true", help="augur with prefix caching on (the speed runs have it off)")
    pf.add_argument("--interval", type=float, default=0.002)
    a = p.parse_args(argv)
    return {"gen": gen, "run": run, "table": table, "profile": profile, "real": real}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
