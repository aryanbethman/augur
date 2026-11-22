"""augur command line: run a workload on simulated vLLM engines."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

from augur.cost.profiles import default_fork
from augur.sim.router import ROUTERS, make_router


def pin_hash_seed(module: str) -> None:
    """Re-run this program once with PYTHONHASHSEED=0 if it is unset.

    vLLM iterates sets of request ids (e.g. the requests finished in a step),
    so with KV offload the order the CPU tier ranks blocks for eviction, and
    so later hits, depends on Python's string hashing, which is random per
    process unless the seed is fixed at start-up. Pinned, a run repeats
    exactly; a real vLLM run to compare with should use the same seed.
    Called by: cli.main, bench.main (as programs only)."""
    if os.environ.get("PYTHONHASHSEED") is not None:
        return
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable, "-m", module, *sys.argv[1:]])


def _scheduler_cls(a) -> str | None:
    if not a.step_log_vllm:
        return None
    os.environ["AUGUR_STEP_LOG"] = a.step_log_vllm
    return ("augur.verify.hooks.LoggingAsyncScheduler" if a.async_scheduling
            else "augur.verify.hooks.LoggingScheduler")


def _tp_params(a, cpu_bw, cpu_lat) -> dict:
    """TableCost keyword arguments for TP>1: flags, or the cluster config file (then a flag that contradicts it is an error)."""
    flags = {"link_bw": a.tp_link_bw, "link_latency_ns": a.tp_link_latency_ns, "npu_mem_bw": a.npu_mem_bw,
             "cpu_mem_bw": cpu_bw, "cpu_mem_latency_ns": cpu_lat}
    if a.lss_cluster_config:
        with open(a.lss_cluster_config, encoding="utf-8") as f:
            c = json.load(f)
        nodes = c.get("nodes") or []
        insts = nodes[0].get("instances") or [] if len(nodes) == 1 else []
        inst = insts[0] if len(insts) == 1 else {}
        bad = []
        if len(nodes) != 1 or len(insts) != 1:
            bad.append("not one node with one instance")
        if inst.get("num_npus") != a.tp or inst.get("tp_size") != a.tp:
            bad.append(f"num_npus/tp_size are not {a.tp}")
        if inst.get("pd_type") or inst.get("dp_group") or (inst.get("pp_size") or 1) != 1:
            bad.append("PD, DP or PP instance")
        if "collective_links" in c or "cxl_mem" in c:
            bad.append("collective_links / cxl_mem")
        for k in ("link_bw", "link_latency"):
            if not isinstance(c.get(k), (int, float)):
                bad.append(f"{k} missing or not a number (the fork would take it from hardware.yaml)")
        if bad:
            raise ValueError(f"--lss-cluster-config {a.lss_cluster_config}: {'; '.join(bad)}: "
                             "the closed-form TP pricer does not cover it (use --cost llmservingsim)")
        cpu = nodes[0].get("cpu_mem") or {}
        fromfile = {"link_bw": c["link_bw"], "link_latency_ns": c["link_latency"],
                    "npu_mem_bw": (inst.get("npu_mem") or {}).get("mem_bw"),
                    "cpu_mem_bw": cpu.get("mem_bw"), "cpu_mem_latency_ns": cpu.get("mem_latency")}
        names = {"link_bw": "--tp-link-bw", "link_latency_ns": "--tp-link-latency-ns", "npu_mem_bw": "--npu-mem-bw",
                 "cpu_mem_bw": "--cpu-mem-bw", "cpu_mem_latency_ns": "--cpu-mem-latency-ns"}
        for k, v in fromfile.items():
            if v is None:
                continue
            if flags[k] is not None and float(flags[k]) != float(v):
                raise ValueError(f"{names[k]} {flags[k]} contradicts {a.lss_cluster_config} ({v})")
            flags[k] = v
    defaults = {"link_bw": 16.0, "link_latency_ns": 20_000, "cpu_mem_bw": 256.0, "cpu_mem_latency_ns": 0}
    return {k: defaults.get(k) if v is None else v for k, v in flags.items()}


def _offload_link(a, cpu_bw, cpu_lat):
    """GPU <-> CPU KV transfer cost: --kv-offload-* flags, else the CPU memory's
    (--cpu-mem-* flags, else the cluster config's cpu_mem, else the fork's
    defaults), the numbers LLMServingSim prices its KV loads with."""
    from augur.sim.offload import OffloadLink
    if a.lss_cluster_config and (cpu_bw is None or cpu_lat is None):
        with open(a.lss_cluster_config, encoding="utf-8") as f:
            cpu = ((json.load(f).get("nodes") or [{}])[0]).get("cpu_mem") or {}
        cpu_bw = cpu.get("mem_bw") if cpu_bw is None else cpu_bw
        cpu_lat = cpu.get("mem_latency") if cpu_lat is None else cpu_lat
    bw = a.kv_offload_gbps if a.kv_offload_gbps is not None else (cpu_bw if cpu_bw is not None else 256.0)
    lat = a.kv_offload_latency_ns if a.kv_offload_latency_ns is not None else (cpu_lat or 0)
    return OffloadLink(gbps=float(bw), latency_ns=int(lat))


def _offload_tiers(a) -> tuple:
    """--kv-offload-tier values as TierSpecs, in order. Raises ValueError with
    the reason on a malformed or misplaced tier. Called by: main."""
    if not a.kv_offload_tier:
        return ()
    if not a.kv_offload_gib:
        raise ValueError("--kv-offload-tier needs --kv-offload-gib (the CPU tier every secondary tier sits behind)")
    from augur.sim.tiers import TierSpec
    tiers = tuple(TierSpec.parse(t) for t in a.kv_offload_tier)
    names = [t.name for t in tiers]
    twice = sorted({n for n in names if names.count(n) > 1})
    if twice:
        raise ValueError(f"--kv-offload-tier: {', '.join(twice)} named twice")
    return tiers


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="augur")
    p.add_argument("--fork", default=default_fork(),
                   help="LLMServingSim fork checkout (model configs, profiles); "
                        "not needed for the models and GPUs augur ships profiles for")
    p.add_argument("--model", required=True, help="e.g. meta-llama/Llama-3.1-8B")
    p.add_argument("--model-config", default=None,
                   help="the model's HF config.json (default: the --cost tables profile's, else the fork's "
                        "configs/model/<model>.json, else the config of a profile shipped with augur)")
    p.add_argument("--workload", required=True, help="JSONL in the fork's format (flat and/or agentic)")
    p.add_argument("--num-entries", type=int, default=None, help="first N requests/sessions only")
    p.add_argument("--engines", type=int, default=1, help="colocated engines (ignored with --prefill-engines)")
    p.add_argument("--prefill-engines", type=int, default=0, help="PD: prefill engines (NIXL kv_producer)")
    p.add_argument("--decode-engines", type=int, default=0, help="PD: decode engines (NIXL kv_consumer)")
    p.add_argument("--async-scheduling", action=argparse.BooleanOptionalAction, default=True,
                   help="vLLM async scheduling, on by default as in vLLM 0.33: schedule step k+1 while step k runs")
    p.add_argument("--num-speculative-tokens", type=int, default=0,
                   help="speculative decoding: draft tokens per step")
    p.add_argument("--speculative-method", choices=["ngram", "eagle", "eagle3", "mtp"], default="ngram",
                   help="drafter: ngram (prompt lookup, no draft cost), eagle / eagle3 (a head derived from the "
                        "target's config, priced as one target layer per pass), mtp (the target's own MTP layers; "
                        "models that define them)")
    p.add_argument("--spec-us-per-request", type=float, default=100.0,
                   help="speculative decoding: extra step time per request verified with drafts (rejection sampling, "
                        "draft bookkeeping), on top of the profile; 100 us (step fits say 160-210, whole-run fits 0-100) on an RTX 4060 Ti "
                        "(vLLM 0.33), after attention is priced at the multi-token query length (needs a profile made "
                        "with --attention-decode-q-lens, else attention is priced as a one-token decode). Ignored "
                        "with --replay-durations")
    p.add_argument("--spec-accept-rate", type=float, default=0.7,
                   help="probability each draft token is accepted (content decides this on a GPU)")
    p.add_argument("--dp", type=int, default=1, help="DP ranks in one lockstep group (vLLM data_parallel_size)")
    p.add_argument("--enable-expert-parallel", action="store_true", help="MoE experts split across DP x TP ranks")
    p.add_argument("--pd-link-bw", type=float, default=16.0, help="PD KV pull bandwidth, GB/s")
    p.add_argument("--pd-link-latency-ns", type=int, default=20_000)
    p.add_argument("--pd-contention", choices=["none", "decode"], default="none",
                   help="decode: concurrent KV pulls into a decode engine share its link bandwidth")
    p.add_argument("--router", default="least-loaded", choices=list(ROUTERS),
                   help="engine for each call (and, with PD, the decode engine): least-loaded (fewest requests in "
                        "the engine, as LLMServingSim) and round-robin, or a vLLM Router policy: random, "
                        "power-of-two, cache-aware (the vLLM Router's default), consistent-hash, rendezvous-hash, "
                        "sticky-least-loaded, kv-aware (routes by the KV blocks each engine holds, from its KV events) (augur/sim/router.py)")
    p.add_argument("--router-seed", type=int, default=0, help="random, power-of-two: random seed")
    p.add_argument("--router-cache-threshold", type=float, default=0.3,
                   help="cache-aware: share of the prompt an engine must already hold to be preferred "
                        "(vLLM Router --cache-threshold)")
    p.add_argument("--router-balance-abs-threshold", type=int, default=64,
                   help="cache-aware: in-flight counts are imbalanced (prefix ignored) when max - min exceeds "
                        "this and max exceeds min times --router-balance-rel-threshold")
    p.add_argument("--router-balance-rel-threshold", type=float, default=1.5)
    p.add_argument("--router-eviction-interval-s", type=float, default=120.0,
                   help="cache-aware: how often each engine's prefix tree is cut back to --router-max-tree-size")
    p.add_argument("--router-max-tree-size", type=int, default=1 << 25,
                   help="cache-aware: characters each engine's prefix tree keeps (the router counts 8 per token)")
    p.add_argument("--router-session-expiry-s", type=float, default=7200.0,
                   help="sticky-least-loaded: a session idle this long gets an engine afresh "
                        "(VLLM_ROUTER_SLL_SESSION_EXPIRATION_IN_S)")
    p.add_argument("--router-session-ids", action=argparse.BooleanOptionalAction, default=True,
                   help="consistent-hash, rendezvous-hash, sticky-least-loaded: clients send each call's session "
                        "id (x-session-id); without, those policies key a call on its prompt")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--tp-link-bw", type=float, default=None,
                   help="--cost tables at TP>1: GPU-to-GPU link bandwidth for the collectives, GB/s (default 16; "
                        "with --lss-cluster-config: the file's link_bw)")
    p.add_argument("--tp-link-latency-ns", type=int, default=None,
                   help="--cost tables at TP>1: link latency per hop (default 20000; or the file's link_latency)")
    p.add_argument("--npu-mem-bw", type=float, default=None,
                   help="--cost tables at TP>1: GPU memory bandwidth, GB/s (default: the profile's, if the fork "
                        "defines one for the hardware, else augur's for it (H100: 3350, SXM); or the file's "
                        "npu_mem.mem_bw)")
    p.add_argument("--hbm-gb", type=float, default=80.0, help="simulated GPU memory")
    p.add_argument("--mem-util", type=float, default=0.9, help="vLLM's --gpu-memory-utilization")
    p.add_argument("--kv-reserve-gib", type=float, default=0.0,
                   help="memory vLLM keeps besides weights (peak activations, non-torch, CUDA-graph pool); "
                        "~0.9 GiB for Qwen3-1.7B on an RTX 4060 Ti. KV = hbm*mem_util - weights - this")
    p.add_argument("--kv-cache-gb", type=float, default=None, help="override the KV-cache size per GPU")
    p.add_argument("--num-gpu-blocks", type=int, default=None,
                   help="exact KV-cache block count (vLLM num_gpu_blocks_override), e.g. from a real run's log")
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--max-num-batched-tokens", type=int, default=None,
                   help="default: what `vllm serve` picks for the simulated GPU (--hbm-gb, --hardware)")
    p.add_argument("--max-num-seqs", type=int, default=None,
                   help="default: what `vllm serve` picks for the simulated GPU (--hbm-gb, --hardware)")
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--no-prefix-caching", action="store_true")
    p.add_argument("--no-chunked-prefill", action="store_true")
    p.add_argument("--frontend-latency-ms", type=float, default=3.4,
                   help="API-server time before the engine sees a request (HTTP, hand-off), independent of prompt "
                        "length; counted in TTFT and latency. Plus --frontend-us-per-token per prompt token. "
                        "Defaults measured with vLLM 0.33 on an i7-12700 with text prompts of 256 to 4,096 tokens "
                        "(6.5 ms at 1,024); the further wait for the engine's next loop iteration is "
                        "simulated, not part of this. Ignored with --replay-durations and --replay-arrivals")
    p.add_argument("--frontend-us-per-token", type=float, default=3.0,
                   help="API-server time per prompt token (tokenization, request parsing), added to "
                        "--frontend-latency-ms: 3.0 us fits 256, 1,024 and 4,096-token prompts at light load "
                        "(vLLM 0.33, i7-12700; intake 3.5 / 5.9 / 14 ms, median). Ignored with --replay-durations "
                        "and --replay-arrivals")
    p.add_argument("--delivery-latency-ms", type=float, default=8.0,
                   help="the client receives each token this long after the engine emits it (output "
                        "processing, detokenization, streaming, client parse): token times, TTFT, end time "
                        "and latency grow by this, inter-token latencies do not, and a session's next call "
                        "is released this much later. Engine behaviour is unchanged. Default measured with "
                        "vLLM 0.33 on an i7-12700. Ignored with --replay-durations and --replay-arrivals")
    p.add_argument("--step-cpu-us", type=float, default=2000.0,
                   help="vLLM's per-step CPU time, which GPU profiles do not hold (sync: added to the step; "
                        "async: delays the launch, hidden behind the previous batch). Default measured with "
                        "vLLM 0.33 on an i7-12700; 0 for GPU time only. Ignored with --replay-durations")
    p.add_argument("--step-time-scale", type=float, default=1.0,
                   help="multiply every GPU step time from the cost model by this (sensitivity runs: 0.99, 1.01); "
                        "not the CPU term, KV transfers or front-end latency. Ignored with --replay-durations")
    p.add_argument("--step-cpu-per-request-us", type=float, default=0.0,
                   help="vLLM's per-step CPU time per scheduled request")
    p.add_argument("--filler-vocab", type=int, default=None,
                   help="fill unrecorded prompt tokens with in-vocabulary ids below this (match augur.verify.client runs)")
    p.add_argument("--no-recorded-outputs", action="store_true",
                   help="emit unique output ids instead of the trace's recorded outputs (as a real model does)")
    p.add_argument("--step-log-vllm", default=None,
                   help="write vLLM-scheduler step events (augur.verify.hooks format) to <path>.<pid>.jsonl; "
                        "with several engines, one log each: <path>.e<engine id>.<pid>.jsonl")
    p.add_argument("--contention", choices=["off", "on"], default="off",
                   help="on: every KV transfer (offload stores and loads, tier writes and reads, PD pulls) is a flow "
                        "through shared resources (PCIe links, root complex, host memory, tiers, NICs) and concurrent "
                        "flows share bandwidth max-min fairly (augur/sim/fabric.py). off: each transfer has its link "
                        "to itself, as before")
    p.add_argument("--fabric", default=None, metavar="FILE",
                   help="--contention on: JSON topology overrides (see augur.sim.fabric.Topology.from_file)")
    p.add_argument("--gpus-per-host", type=int, default=8, help="--contention on: GPUs sharing a host's PCIe root and memory")
    p.add_argument("--host-mem-gbps", type=float, default=None,
                   help="--contention on: host memory bandwidth shared by every transfer on the host (default: --cpu-mem-bw)")
    p.add_argument("--pcie-root-gbps", type=float, default=None,
                   help="--contention on: a host's PCIe root bandwidth shared by every GPU transfer on it (default two GPUs' links, both directions)")
    p.add_argument("--d2h-duplex-penalty", type=float, default=0.15, metavar="P",
                   help="--contention on: a GPU's D2H capacity drops by this fraction while an H2D copy runs on it "
                        "(RTX 4060 Ti measured 0.15; H2D loses about 1%%)")
    p.add_argument("--host-mem-background-gbps", type=float, default=0.0,
                   help="--contention on: host memory bandwidth other work takes (a loaded host); the default "
                        "assumes an idle host")
    p.add_argument("--hbm-interference", type=float, default=None, metavar="K",
                   help="--contention on: a step runs t * (1 + K * r / HBM bandwidth) where r is what the GPU's PCIe "
                        "flows move when the step starts (default 1.4, measured on an RTX 4060 Ti: +7%% under an H2D "
                        "copy, +5%% D2H, +11-12%% both, at 288 GB/s; 0: none). Not applied in replays")
    p.add_argument("--hbm-bw-gbps", type=float, default=None,
                   help="GPU memory bandwidth for --hbm-interference (default: --npu-mem-bw, else the fork's for --hardware)")
    p.add_argument("--replay-durations", default=None,
                   help="price step k with the duration recorded in this vLLM step log (see augur.verify.replay). "
                        "With several engines: the --step-log-vllm prefix of the run, or one log per engine, "
                        "comma separated in engine order")
    p.add_argument("--replay-arrivals", default=None,
                   help="requests reach the engine when this vLLM step log says, steps priced by --cost "
                        "(separates arrival timing from step pricing)")
    p.add_argument("--enforce-eager", action="store_true", help="no CUDA graphs, so no step padding")
    p.add_argument("--cost", default="auto", choices=["fixed", "tables", "llmservingsim", "auto"],
                   help="step prices. auto (default): tables when augur ships a profile for --hardware, "
                        "--model and --tp (no DP/EP), else llmservingsim. "
                        "tables: a profile shipped with augur (dense models, any TP it ships, one node, no DP/EP; "
                        "no fork needed). At TP>1 a step whose CPU-memory accesses are long next to a ring step "
                        "(see augur/cost/tables.py) is refused. "
                        "llmservingsim: the fork's profiles and ASTRA-Sim (MoE, TP>1, DP). "
                        "fixed: one constant step time (--fixed-step-ms, --fixed-per-token-us), no profile")
    p.add_argument("--profile-variant", default=None, metavar="NAME",
                   help="--cost llmservingsim: read the fork's profiles from the folder perf/<hardware>/<model>/NAME/ "
                        "(e.g. bf16-tp4proj, a projected TP4 bundle) instead of the one the model's dtype names; "
                        "the fork's --profile-variant. --cost auto then uses llmservingsim (tables do not know variants)")
    p.add_argument("--profile", default=None, help="profile file for --cost tables (default: the one shipped "
                                                   "for --hardware and --model; python -m augur.cost.profiles lists them)")
    p.add_argument("--fixed-step-ms", type=float, default=None, help="--cost fixed only: base step time (default 25)")
    p.add_argument("--fixed-per-token-us", type=float, default=None,
                   help="--cost fixed only: added per batched token (default 0)")
    p.add_argument("--hardware", default="RTXPRO6000", help="profile hardware for --cost tables / llmservingsim")
    p.add_argument("--lss-cluster-config", default=None,
                   help="LLMServingSim cluster config for ASTRA-Sim (needed for --cost llmservingsim at --tp > 1). With "
                        "--cost tables at --tp > 1 its link_bw, link_latency, npu_mem.mem_bw and cpu_mem values "
                        "are used (so one file drives either pricer); a contradicting flag is an error")
    p.add_argument("--cpu-mem-bw", type=float, default=None, help="GB/s, the fork's cpu_mem.mem_bw (default 256)")
    p.add_argument("--cpu-mem-latency-ns", type=int, default=None, help="default 0")
    p.add_argument("--kv-offload-gib", type=float, default=None,
                   help="KV offload to CPU memory, GiB per engine: vLLM's --kv-offloading-size (native "
                        "OffloadingConnector; its scheduler half runs unmodified, augur/sim/offload.py stands in "
                        "for its GPU worker)")
    p.add_argument("--kv-offload-gbps", type=float, default=None,
                   help="GPU <-> CPU KV transfer bandwidth, GB/s each way (default: the CPU memory's, "
                        "--cpu-mem-bw or the cluster config's cpu_mem.mem_bw, as LLMServingSim prices its KV "
                        "loads; a PCIe link is usually far slower, so set it for real hardware)")
    p.add_argument("--kv-offload-latency-ns", type=int, default=None,
                   help="per GPU <-> CPU transfer (default: --cpu-mem-latency-ns or the cluster config's "
                        "cpu_mem.mem_latency)")
    p.add_argument("--kv-offload-tier", action="append", default=None, metavar="NAME:GIB[:GBPS[:LATENCY_NS]]",
                   help="a secondary KV tier behind the CPU one (e.g. CXL memory: cxl:64 or cxl:64:60:250), as vLLM's "
                        "offload tiering runs it: stores cascade to it, hits in it are promoted to the CPU "
                        "tier, then loaded to the GPU. Capacity in GiB, bandwidth in GB/s each way, latency per "
                        "transfer. 'cxl:64' alone takes 60 GB/s and 250 ns from LLMServingSim's cxl_mem. "
                        "Repeat for more tiers (in order); needs --kv-offload-gib")
    p.add_argument("--output", default=None, help="request CSV (fork format)")
    p.add_argument("--step-log", default=None, help="per-step CSV: start, duration, batch shape")
    p.add_argument("--lss-loop-compat", action="store_true",
                   help="comparison only: reproduce LLMServingSim's main-loop timing artifacts "
                        "(1 ms idle quantum, TP>1 same-instant admission); not vLLM behaviour")
    p.add_argument("--quiet", action="store_true")
    return p


def _pick_cost(a, cpu_flags) -> str | None:
    """--cost auto: the shipped tables when augur has a profile for --hardware, --model and --tp (no DP/EP and a TP
    config the closed form covers), else the fork, whose profile must exist. Sets ``a.cost``; returns the tables'
    profile (None for the fork). Raises FileNotFoundError. Called by: main."""
    auto = a.cost == "auto"
    profile = None
    if a.cost in ("tables", "auto"):
        from augur.cost.tables import bundled_profile
        plain = a.dp == 1 and not a.enable_expert_parallel and not a.profile_variant
        if a.cost == "tables" or a.profile:
            profile = a.profile or bundled_profile(a.hardware, a.model, a.tp)
        elif plain:
            try:
                profile = bundled_profile(a.hardware, a.model, a.tp)
                if a.tp > 1:
                    _tp_params(a, *cpu_flags)  # a config the closed form does not cover: use the fork
            except FileNotFoundError:
                pass
            except ValueError:
                profile = None
        a.cost = "tables" if profile else "llmservingsim"
    if a.cost == "llmservingsim" and not a.replay_durations:
        from augur.cost.profiles import check, shipped
        try:
            check(a.fork, a.hardware, a.model, a.tp, a.profile_variant)
        except FileNotFoundError:
            if not auto:
                raise
            have = "\n".join(f"  {h} {m} {d} tp{t}" for h, m, d, t in shipped())
            raise FileNotFoundError(
                f"no step-time profile for {a.model} on {a.hardware} (tp={a.tp}, dp={a.dp}). "
                f"Profiles shipped with augur (--cost tables, DP=1 and no expert parallelism):\n{have}\n"
                "For another model or GPU, point augur at an LLMServingSim fork that has the profile "
                "(--fork DIR or AUGUR_FORK=DIR; python -m augur.cost.profiles --fork DIR lists them), "
                "or use --cost fixed (a constant step time, no profile).") from None
    return profile


def _build_engines(a, spec, cost, replay_costs, topology) -> list:
    """The run's engines: DP ranks, prefill + decode engines, or colocated ones, with augur's per-step terms.
    Called by: main."""
    from augur.sim.engine import SimEngine
    net_of = (lambda i: topology.net(i)) if a.contention == "on" else (lambda i: None)
    if a.dp > 1:
        engines = [SimEngine(i, dataclasses.replace(spec, data_parallel_size=a.dp, data_parallel_rank=i,
                                                    enable_expert_parallel=a.enable_expert_parallel),
                             replay_costs[i] if replay_costs else cost, net_of(i))
                   for i in range(a.dp)]
    elif a.prefill_engines:
        roles = ["kv_producer"] * a.prefill_engines + ["kv_consumer"] * a.decode_engines
        engines = [SimEngine(i, dataclasses.replace(spec, kv_role=r), replay_costs[i] if replay_costs else cost, net_of(i))
                   for i, r in enumerate(roles)]
    else:
        engines = [SimEngine(i, spec, replay_costs[i] if replay_costs else cost, net_of(i)) for i in range(a.engines)]
    if not a.replay_durations:  # a logged step's duration already holds its CPU time
        for e in engines:
            e.ctx.cpu_fixed_ns = int(a.step_cpu_us * 1e3)
            e.ctx.cpu_per_request_ns = int(a.step_cpu_per_request_us * 1e3)
            e.ctx.spec_ns_per_request = int(a.spec_us_per_request * 1e3)
            e.ctx.step_time_scale = a.step_time_scale
    if a.step_log:
        for e in engines:
            e.step_log = []
    return engines


def _summary(a, rec, coord, engines, cost, topology, kv_bytes) -> dict:
    """The JSON summary main prints after a run. Called by: main."""
    summary = rec.summary()
    steps = sum(e.steps for e in engines)
    cpu = sum(e.sched_cpu_s for e in engines)
    if hasattr(cost, "oracle") and cost.oracle is not None:
        summary["astra_runs"] = cost.oracle.runs
    if coord.dp_group:
        summary["dp_rounds"] = coord.dp_rounds
        summary["dp_dummy_forwards"] = coord.dp_dummy_forwards
    summary["async_scheduling"] = engines[0].spec.async_scheduling
    jobs = [x for e in engines if e.ctx.offload is not None for x in e.ctx.offload.job_ns]
    if jobs:
        def dist(ns):
            ns = sorted(ns)
            return {"n": len(ns), "mean_ms": round(sum(ns) / len(ns) / 1e6, 3),
                    "p50_ms": round(ns[len(ns) // 2] / 1e6, 3), "p99_ms": round(ns[min(len(ns) - 1, int(0.99 * len(ns)))] / 1e6, 3)}
        summary["offload_transfers"] = {kind: dist([ns for is_load, ns in jobs if is_load == (kind == "load")])
                                        for kind in ("store", "load") if any(is_load == (kind == "load") for is_load, _ in jobs)}
    if a.contention == "on":
        summary["fabric"] = {"hosts": len({n for n in topology.gpu_of.values()}) and
                             max(topology.gpu_of.values()) // topology.gpus_per_host + 1,
                             "rate_recomputations": topology.fabric.recomputes}
    if hasattr(cost, "steps_beyond_profile"):
        summary["steps_beyond_profile"] = cost.steps_beyond_profile
    summary.update({"engines": len(engines), "steps": steps, "wall_s": round(coord.wall_s, 2),
                    "vllm_step_cpu_ms_per_step": round(1e3 * cpu / max(steps, 1), 3),
                    "kv_cache_gib_per_engine": round(kv_bytes / 2**30, 2),
                    "num_gpu_blocks": engines[0].vllm_config.cache_config.num_gpu_blocks})
    return summary


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        pin_hash_seed("augur.cli")
    args = sys.argv[1:] if argv is None else argv
    if args[:1] == ["bench"]:
        from augur.bench import main as bench_main
        return bench_main(args[1:])
    a = _parser().parse_args(argv)
    if a.profile_variant and a.cost in ("tables", "fixed"):
        print(f"augur: --profile-variant selects a profile folder of the fork: it needs --cost llmservingsim "
              f"(or auto), not --cost {a.cost}", file=sys.stderr)
        return 2
    if a.profile_variant and a.profile:
        print("augur: --profile (a --cost tables file) and --profile-variant (a fork folder) exclude each other",
              file=sys.stderr)
        return 2

    cpu_flags = (a.cpu_mem_bw, a.cpu_mem_latency_ns)
    a.cpu_mem_bw = 256.0 if a.cpu_mem_bw is None else a.cpu_mem_bw
    a.cpu_mem_latency_ns = 0 if a.cpu_mem_latency_ns is None else a.cpu_mem_latency_ns
    if a.cost != "fixed" and (a.fixed_step_ms is not None or a.fixed_per_token_us is not None):
        print(f"augur: --fixed-step-ms and --fixed-per-token-us only apply with --cost fixed (this run: --cost {a.cost})",
              file=sys.stderr)
        return 2
    if bool(a.prefill_engines) != bool(a.decode_engines):
        print("augur: PD needs both --prefill-engines and --decode-engines", file=sys.stderr)
        return 2
    num_engines = a.dp if a.dp > 1 else (a.prefill_engines + a.decode_engines or a.engines)
    a.num_engines = num_engines
    try:
        a.kv_offload_tiers = _offload_tiers(a)
    except ValueError as e:
        print(f"augur: {e}", file=sys.stderr)
        return 2
    if a.quiet:
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    from augur.trace.metrics import ensure_parent_dir
    for path in (a.output, a.step_log, a.step_log_vllm):
        ensure_parent_dir(path)  # before the run: results are written at its end
    from augur.cost import FixedCost
    from augur.sim.coordinator import Coordinator
    from augur.sim.engine import EngineSpec
    from augur.sim.models import estimate_weight_bytes, model_dir
    from augur.trace.workload import Workload

    cache_root = os.path.expanduser("~/.cache/augur/models")
    try:
        profile = _pick_cost(a, cpu_flags)
        mdir = model_dir(a.fork, a.model, cache_root, a.model_config, profile)
    except FileNotFoundError as e:
        print(f"augur: {e}", file=sys.stderr)
        return 2
    if a.kv_cache_gb is not None:
        kv_bytes = int(a.kv_cache_gb * 2**30)
    else:
        ep = a.dp * a.tp if a.enable_expert_parallel else 1
        weights = estimate_weight_bytes(os.path.join(mdir, "config.json"), tp=a.tp, ep=ep)
        if a.num_speculative_tokens and a.speculative_method != "ngram":
            from augur.sim.draft import draft_weight_bytes
            weights += draft_weight_bytes(os.path.join(mdir, "config.json"), a.speculative_method, tp=a.tp)
        kv_bytes = int(a.hbm_gb * 2**30 * a.mem_util) - weights - int(a.kv_reserve_gib * 2**30)
        if kv_bytes <= 0:
            print(f"model weights ({weights / 2**30:.1f} GiB/GPU) do not fit in {a.hbm_gb} GB", file=sys.stderr)
            return 2

    replay_costs = None
    if a.replay_durations:
        from augur.verify.replay import ReplayCost, resolve_logs
        try:
            replay_logs = resolve_logs(a.replay_durations, a.num_engines)
        except ValueError as e:
            print(f"augur: {e}", file=sys.stderr)
            return 2
        replay_costs = ReplayCost.from_logs(replay_logs)
        cost = replay_costs[0]
    elif a.cost == "fixed":
        step_ms = 25.0 if a.fixed_step_ms is None else a.fixed_step_ms
        cost = FixedCost(base_ns=int(step_ms * 1e6), per_token_ns=int((a.fixed_per_token_us or 0.0) * 1e3))
    elif a.cost == "tables":
        from augur.cost.tables import TableCost
        if a.dp != 1 or a.enable_expert_parallel:
            print("augur: --cost tables prices TP without DP/EP only (use --cost llmservingsim)", file=sys.stderr)
            return 2
        try:
            kw = _tp_params(a, *cpu_flags) if a.tp > 1 else dict(cpu_mem_bw=a.cpu_mem_bw,
                                                                   cpu_mem_latency_ns=a.cpu_mem_latency_ns)
            cost = TableCost(profile, **kw)
            if cost.tp != a.tp:
                raise ValueError(f"profile {profile} is tp={cost.tp}, --tp is {a.tp}")
        except (NotImplementedError, ValueError) as e:
            hint = "" if "--cost llmservingsim" in str(e) else " (use --cost llmservingsim)"
            print(f"augur: {e}{hint}", file=sys.stderr)
            return 2
    else:
        from augur.cost.llmservingsim import LLMServingSimCost
        cost = LLMServingSimCost(fork_root=a.fork, hardware=a.hardware, model=a.model, tp=a.tp,
                                 max_num_batched_tokens=a.max_num_batched_tokens or 2048,
                                 cpu_mem_bw_gbps=a.cpu_mem_bw, cpu_mem_latency_ns=a.cpu_mem_latency_ns,
                                 cluster_config=a.lss_cluster_config, dp=a.dp, profile_variant=a.profile_variant,
                                 ep=a.dp * a.tp if a.enable_expert_parallel else 1)

    speculative = None
    if a.num_speculative_tokens:
        from augur.sim.draft import speculative_config
        try:
            speculative = speculative_config(a.speculative_method, a.num_speculative_tokens, mdir, cache_root)
        except ValueError as e:
            print(f"augur: {e}", file=sys.stderr)
            return 2
        if a.dp > 1:
            print("augur: speculative decoding is not priced with --dp yet", file=sys.stderr)
            return 2
    spec = EngineSpec(model_dir=mdir, model_name=a.model, kv_cache_bytes=kv_bytes,
                      max_model_len=a.max_model_len, max_num_batched_tokens=a.max_num_batched_tokens,
                      max_num_seqs=a.max_num_seqs, block_size=a.block_size,
                      enable_prefix_caching=not a.no_prefix_caching,
                      enable_chunked_prefill=not a.no_chunked_prefill, tensor_parallel_size=a.tp,
                      enforce_eager=a.enforce_eager,
                      async_scheduling=a.async_scheduling,
                      speculative=speculative,
                      spec_accept_rate=a.spec_accept_rate,
                      scheduler_cls=_scheduler_cls(a), num_gpu_blocks=a.num_gpu_blocks,
                      kv_offload_gib=a.kv_offload_gib,
                      kv_offload_link=_offload_link(a, *cpu_flags) if a.kv_offload_gib else None,
                      kv_offload_tiers=a.kv_offload_tiers, hbm_gb=a.hbm_gb, hardware=a.hardware,
                      kv_events=a.router == "kv-aware",
                      step_log_tag=bool(a.step_log_vllm) and a.num_engines > 1)
    topology = None
    if a.contention == "on":
        from augur.sim.fabric import Topology
        topo_args = dict(gpus_per_host=a.gpus_per_host,
                         h2d_gbps=spec.kv_offload_link.gbps if spec.kv_offload_link else 25.0,
                         mem_gbps=a.host_mem_gbps if a.host_mem_gbps is not None else a.cpu_mem_bw,
                         root_gbps=a.pcie_root_gbps, nic_gbps=a.pd_link_bw,
                         d2h_duplex_penalty=a.d2h_duplex_penalty, host_mem_background_gbps=a.host_mem_background_gbps,
                         tiers={t.name: t.gbps for t in a.kv_offload_tiers})
        tps = [(i, a.tp) for i in range(a.num_engines)]
        try:
            topology = (Topology.from_file(a.fabric, tps, **topo_args) if a.fabric
                        else Topology.build(tps, **topo_args))
        except (OSError, ValueError, KeyError) as e:
            print(f"augur: --fabric: {e}", file=sys.stderr)
            return 2
    elif a.pd_contention == "decode" and a.prefill_engines:
        from augur.sim.fabric import Topology  # each decode engine's pulls share its pd.<engine> link
        topology = Topology.build([(i, a.tp) for i in range(a.num_engines)], nic_gbps=a.pd_link_bw,
                                  pd_mode="decode-link")
    engines = _build_engines(a, spec, cost, replay_costs, topology)
    if a.contention == "on" and a.hbm_interference != 0:
        hbm_bw = a.hbm_bw_gbps or a.npu_mem_bw
        if hbm_bw is None:
            try:
                from augur.cost.convert import _npu_mem_bw_default
                hbm_bw = _npu_mem_bw_default(a.fork, a.hardware)
            except Exception:  # no fork on this machine
                hbm_bw = None
        if not hbm_bw:
            if a.hbm_interference is not None:
                print("augur: --hbm-interference needs the GPU memory bandwidth: give --hbm-bw-gbps", file=sys.stderr)
                return 2
            print("augur: no GPU memory bandwidth known for --hardware %s: steps are not slowed by PCIe traffic "
                  "(give --hbm-bw-gbps to model it)" % a.hardware, file=sys.stderr)
        else:
            k = 1.4 if a.hbm_interference is None else a.hbm_interference
            for e in engines:
                e.ctx.hbm_interference, e.ctx.hbm_bw_gbps = k, float(hbm_bw)
    if hasattr(cost, "max_num_batched_tokens"):  # the fork's traces are built for the engine's real budget
        cost.max_num_batched_tokens = engines[0].vllm_config.scheduler_config.max_num_batched_tokens
    from augur.sim.coordinator import PDLink
    replaying = bool(a.replay_durations or a.replay_arrivals)  # arrival times are engine-side: no client latencies
    if replaying:
        from augur.verify.replay import ReplayWorkload
        workload = ReplayWorkload(a.workload, max_entries=a.num_entries,
                                  log_path=a.replay_durations or a.replay_arrivals,
                                  log_paths=replay_logs if replay_costs and a.num_engines > 1 else None,
                                  recorded_outputs=not a.no_recorded_outputs, filler_vocab=a.filler_vocab)
    else:
        workload = Workload(a.workload, max_entries=a.num_entries, recorded_outputs=not a.no_recorded_outputs,
                            filler_vocab=a.filler_vocab)
    router_options = dict(seed=a.router_seed, cache_threshold=a.router_cache_threshold,
                          balance_abs_threshold=a.router_balance_abs_threshold,
                          balance_rel_threshold=a.router_balance_rel_threshold,
                          eviction_interval_s=a.router_eviction_interval_s, max_tree_size=a.router_max_tree_size,
                          session_expiry_s=a.router_session_expiry_s, session_ids=a.router_session_ids)
    entry_router = make_router(a.router, **router_options)
    decode_router = make_router("least-loaded" if a.router == "kv-aware" else a.router, **router_options)
    if replay_costs and a.num_engines > 1:
        # Each call goes where its log says (the placing policy is not what is replayed).
        from augur.verify.replay import ReplayRouter
        entry_router, decode_router = ReplayRouter.from_logs(replay_logs)
    coord = Coordinator(engines, entry_router, workload,
                        lss_idle_quantum_ns=1_000_000 if a.lss_loop_compat else 0,
                        decode_router=decode_router, topology=topology,
                        pd_link=PDLink(bw_gbps=a.pd_link_bw, latency_ns=a.pd_link_latency_ns, contention=a.pd_contention),
                        frontend_latency_ns=0 if replaying else int(a.frontend_latency_ms * 1e6),
                        frontend_ns_per_token=0.0 if replaying else a.frontend_us_per_token * 1e3,
                        delivery_latency_ns=0 if replaying else int(a.delivery_latency_ms * 1e6))
    if a.replay_durations and not (a.prefill_engines or a.decode_engines):
        # the logged engines' request ids (see Coordinator.request_ids); every engine's log when there are several
        coord.request_ids = {c: r for rc in (replay_costs or [cost]) for c, r in rc.rids.items()}
    if a.step_log_vllm:
        from augur.sim.coordinator import install_step_log_clock
        install_step_log_clock(coord)
    from augur.cost.tables import OutsideRegime
    try:
        rec = coord.run()
    except OutsideRegime as e:
        print(f"augur: {e}", file=sys.stderr)
        return 2
    if a.output:
        rec.write_csv(a.output)
    if a.step_log:
        import csv as _csv
        with open(a.step_log, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["engine", "start_ns", "dur_ns", "total_tokens", "prefill_tokens", "n_decode", "n_prefill",
                        "forward_tokens", "dummy"])
            for e in engines:
                for row in e.step_log:
                    w.writerow([e.engine_id, *row])
    print(json.dumps(_summary(a, rec, coord, engines, cost, topology, kv_bytes), indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
