"""augur command line: run a workload on simulated vLLM engines."""

from __future__ import annotations

import argparse
import json
import os
import sys

from augur.cost.profiles import default_fork


def _scheduler_cls(a) -> str | None:
    if not a.step_log_vllm:
        return None
    os.environ["AUGUR_STEP_LOG"] = a.step_log_vllm
    return ("augur.verify.hooks.LoggingAsyncScheduler" if a.async_scheduling
            else "augur.verify.hooks.LoggingScheduler")


def main(argv: list[str] | None = None) -> int:
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
                   help="speculative decoding (ngram method): draft tokens per step")
    p.add_argument("--spec-accept-rate", type=float, default=0.7,
                   help="probability each draft token is accepted (content decides this on a GPU)")
    p.add_argument("--dp", type=int, default=1, help="DP ranks in one lockstep group (vLLM data_parallel_size)")
    p.add_argument("--enable-expert-parallel", action="store_true", help="MoE experts split across DP x TP ranks")
    p.add_argument("--pd-link-bw", type=float, default=16.0, help="PD KV pull bandwidth, GB/s")
    p.add_argument("--pd-link-latency-ns", type=int, default=20_000)
    p.add_argument("--pd-contention", choices=["none", "decode"], default="none",
                   help="decode: concurrent KV pulls into a decode engine share its link bandwidth")
    p.add_argument("--router", default="least-loaded")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--hbm-gb", type=float, default=80.0, help="simulated GPU memory")
    p.add_argument("--mem-util", type=float, default=0.9, help="vLLM's --gpu-memory-utilization")
    p.add_argument("--kv-reserve-gib", type=float, default=0.0,
                   help="memory vLLM keeps besides weights (peak activations, non-torch, CUDA-graph pool); "
                        "~0.9 GiB for Qwen3-1.7B on an RTX 4060 Ti. KV = hbm*mem_util - weights - this")
    p.add_argument("--kv-cache-gb", type=float, default=None, help="override the KV-cache size per GPU")
    p.add_argument("--num-gpu-blocks", type=int, default=None,
                   help="exact KV-cache block count (vLLM num_gpu_blocks_override), e.g. from a real run's log")
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--max-num-batched-tokens", type=int, default=None)
    p.add_argument("--max-num-seqs", type=int, default=None)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--no-prefix-caching", action="store_true")
    p.add_argument("--no-chunked-prefill", action="store_true")
    p.add_argument("--frontend-latency-ms", type=float, default=6.5,
                   help="API-server time before the engine sees a request (HTTP, tokenization, hand-off); "
                        "counted in TTFT and latency. Default measured with vLLM 0.33 on an i7-12700 for "
                        "1,024-token text prompts; the further wait for the engine's next loop iteration is "
                        "simulated, not part of this. Ignored with --replay-durations and --replay-arrivals")
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
                   help="write vLLM-scheduler step events (augur.verify.hooks format) to <path>.<pid>.jsonl")
    p.add_argument("--replay-durations", default=None,
                   help="price step k with the duration recorded in this vLLM step log (see augur.verify.replay)")
    p.add_argument("--replay-arrivals", default=None,
                   help="requests reach the engine when this vLLM step log says, steps priced by --cost "
                        "(separates arrival timing from step pricing)")
    p.add_argument("--enforce-eager", action="store_true", help="no CUDA graphs, so no step padding")
    p.add_argument("--cost", default="auto", choices=["fixed", "tables", "llmservingsim", "auto"],
                   help="step prices. auto (default): tables when augur ships a profile for --hardware and "
                        "--model (TP=1, no DP/EP), else llmservingsim. "
                        "tables: a profile shipped with augur (dense models, TP=1; no fork needed). "
                        "llmservingsim: the fork's profiles and ASTRA-Sim (MoE, TP>1, DP). "
                        "fixed: one constant step time (--fixed-step-ms, --fixed-per-token-us), no profile")
    p.add_argument("--profile", default=None, help="profile file for --cost tables (default: the one shipped "
                                                   "for --hardware and --model; python -m augur.cost.profiles lists them)")
    p.add_argument("--fixed-step-ms", type=float, default=None, help="--cost fixed only: base step time (default 25)")
    p.add_argument("--fixed-per-token-us", type=float, default=None,
                   help="--cost fixed only: added per batched token (default 0)")
    p.add_argument("--hardware", default="RTXPRO6000", help="profile hardware for --cost tables / llmservingsim")
    p.add_argument("--lss-cluster-config", default=None,
                   help="LLMServingSim cluster config for ASTRA-Sim (needed for --tp > 1)")
    p.add_argument("--cpu-mem-bw", type=float, default=256.0, help="GB/s, the fork's cpu_mem.mem_bw")
    p.add_argument("--cpu-mem-latency-ns", type=int, default=0)
    p.add_argument("--output", default=None, help="request CSV (fork format)")
    p.add_argument("--step-log", default=None, help="per-step CSV: start, duration, batch shape")
    p.add_argument("--lss-loop-compat", action="store_true",
                   help="comparison only: reproduce LLMServingSim's main-loop timing artifacts "
                        "(1 ms idle quantum, TP>1 same-instant admission); not vLLM behaviour")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)

    if a.cost != "fixed" and (a.fixed_step_ms is not None or a.fixed_per_token_us is not None):
        print(f"augur: --fixed-step-ms and --fixed-per-token-us only apply with --cost fixed (this run: --cost {a.cost})",
              file=sys.stderr)
        return 2
    if a.quiet:
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    from augur.cost import FixedCost
    from augur.sim.coordinator import Coordinator
    from augur.sim.engine import EngineSpec, SimEngine
    from augur.sim.models import estimate_weight_bytes, model_dir
    from augur.sim.router import ROUTERS
    from augur.trace.workload import Workload

    cache_root = os.path.expanduser("~/.cache/augur/models")
    auto = a.cost == "auto"
    try:
        profile = None
        if a.cost in ("tables", "auto"):
            from augur.cost.tables import bundled_profile
            plain = a.tp == 1 and a.dp == 1 and not a.enable_expert_parallel
            if a.cost == "tables" or a.profile:
                profile = a.profile or bundled_profile(a.hardware, a.model)
            elif plain:
                try:
                    profile = bundled_profile(a.hardware, a.model)
                except FileNotFoundError:
                    pass
            a.cost = "tables" if profile else "llmservingsim"
        if a.cost == "llmservingsim" and not a.replay_durations:
            from augur.cost.profiles import check, shipped
            try:
                check(a.fork, a.hardware, a.model, a.tp)
            except FileNotFoundError:
                if not auto:
                    raise
                have = "\n".join(f"  {h} {m} {d} tp{t}" for h, m, d, t in shipped())
                raise FileNotFoundError(
                    f"no step-time profile for {a.model} on {a.hardware} (tp={a.tp}, dp={a.dp}). "
                    f"Profiles shipped with augur (dense, TP=1):\n{have}\n"
                    "For another model or GPU, point augur at an LLMServingSim fork that has the profile "
                    "(--fork DIR or AUGUR_FORK=DIR; python -m augur.cost.profiles --fork DIR lists them), "
                    "or use --cost fixed (a constant step time, no profile).") from None
        mdir = model_dir(a.fork, a.model, cache_root, a.model_config, profile)
    except FileNotFoundError as e:
        print(f"augur: {e}", file=sys.stderr)
        return 2
    if a.kv_cache_gb is not None:
        kv_bytes = int(a.kv_cache_gb * 2**30)
    else:
        ep = a.dp * a.tp if a.enable_expert_parallel else 1
        weights = estimate_weight_bytes(os.path.join(mdir, "config.json"), tp=a.tp, ep=ep)
        kv_bytes = int(a.hbm_gb * 2**30 * a.mem_util) - weights - int(a.kv_reserve_gib * 2**30)
        if kv_bytes <= 0:
            print(f"model weights ({weights / 2**30:.1f} GiB/GPU) do not fit in {a.hbm_gb} GB", file=sys.stderr)
            return 2

    if a.replay_durations:
        from augur.verify.replay import ReplayCost
        cost = ReplayCost.from_log(a.replay_durations)
    elif a.cost == "fixed":
        step_ms = 25.0 if a.fixed_step_ms is None else a.fixed_step_ms
        cost = FixedCost(base_ns=int(step_ms * 1e6), per_token_ns=int((a.fixed_per_token_us or 0.0) * 1e3))
    elif a.cost == "tables":
        from augur.cost.tables import TableCost
        try:
            cost = TableCost(profile, cpu_mem_bw=a.cpu_mem_bw, cpu_mem_latency_ns=a.cpu_mem_latency_ns)
        except NotImplementedError as e:
            print(f"augur: {e} (use --cost llmservingsim)", file=sys.stderr)
            return 2
        if a.tp != 1 or a.dp != 1 or a.enable_expert_parallel:
            print("augur: --cost tables prices TP=1 without DP/EP only (use --cost llmservingsim)", file=sys.stderr)
            return 2
    else:
        from augur.cost.llmservingsim import LLMServingSimCost
        cost = LLMServingSimCost(fork_root=a.fork, hardware=a.hardware, model=a.model, tp=a.tp,
                                 max_num_batched_tokens=a.max_num_batched_tokens or 2048,
                                 cpu_mem_bw_gbps=a.cpu_mem_bw, cpu_mem_latency_ns=a.cpu_mem_latency_ns,
                                 cluster_config=a.lss_cluster_config, dp=a.dp,
                                 ep=a.dp * a.tp if a.enable_expert_parallel else 1)

    spec = EngineSpec(model_dir=mdir, model_name=a.model, kv_cache_bytes=kv_bytes,
                      max_model_len=a.max_model_len, max_num_batched_tokens=a.max_num_batched_tokens,
                      max_num_seqs=a.max_num_seqs, block_size=a.block_size,
                      enable_prefix_caching=not a.no_prefix_caching,
                      enable_chunked_prefill=not a.no_chunked_prefill, tensor_parallel_size=a.tp,
                      enforce_eager=a.enforce_eager,
                      async_scheduling=a.async_scheduling,
                      speculative=({"method": "ngram", "num_speculative_tokens": a.num_speculative_tokens,
                                    "prompt_lookup_max": 4, "prompt_lookup_min": 2}
                                   if a.num_speculative_tokens else None),
                      spec_accept_rate=a.spec_accept_rate,
                      scheduler_cls=_scheduler_cls(a), num_gpu_blocks=a.num_gpu_blocks)
    notes = []
    if a.async_scheduling and a.num_speculative_tokens:
        notes.append("speculative decoding is modeled with synchronous scheduling only; this run is synchronous")
        print("augur: " + notes[-1], file=sys.stderr)
    import dataclasses
    if a.dp > 1:
        engines = [SimEngine(i, dataclasses.replace(spec, data_parallel_size=a.dp, data_parallel_rank=i,
                                                    enable_expert_parallel=a.enable_expert_parallel), cost)
                   for i in range(a.dp)]
    elif a.prefill_engines:
        roles = ["kv_producer"] * a.prefill_engines + ["kv_consumer"] * a.decode_engines
        engines = [SimEngine(i, dataclasses.replace(spec, kv_role=r), cost) for i, r in enumerate(roles)]
    else:
        engines = [SimEngine(i, spec, cost) for i in range(a.engines)]
    if not a.replay_durations:  # a logged step's duration already holds its CPU time
        for e in engines:
            e.ctx.cpu_fixed_ns = int(a.step_cpu_us * 1e3)
            e.ctx.cpu_per_request_ns = int(a.step_cpu_per_request_us * 1e3)
            e.ctx.step_time_scale = a.step_time_scale
    if a.step_log:
        for e in engines:
            e.step_log = []
    from augur.sim.coordinator import PDLink
    replaying = bool(a.replay_durations or a.replay_arrivals)  # arrival times are engine-side: no client latencies
    if replaying:
        from augur.verify.replay import ReplayWorkload
        workload = ReplayWorkload(a.workload, max_entries=a.num_entries,
                                  log_path=a.replay_durations or a.replay_arrivals,
                                  recorded_outputs=not a.no_recorded_outputs, filler_vocab=a.filler_vocab)
    else:
        workload = Workload(a.workload, max_entries=a.num_entries, recorded_outputs=not a.no_recorded_outputs,
                            filler_vocab=a.filler_vocab)
    coord = Coordinator(engines, ROUTERS[a.router](), workload,
                        lss_idle_quantum_ns=1_000_000 if a.lss_loop_compat else 0,
                        decode_router=ROUTERS[a.router](),
                        pd_link=PDLink(bw_gbps=a.pd_link_bw, latency_ns=a.pd_link_latency_ns, contention=a.pd_contention),
                        frontend_latency_ns=0 if replaying else int(a.frontend_latency_ms * 1e6),
                        delivery_latency_ns=0 if replaying else int(a.delivery_latency_ms * 1e6))
    if a.step_log_vllm:
        from augur.sim.coordinator import install_step_log_clock
        install_step_log_clock(coord)
    rec = coord.run()
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
    summary = rec.summary()
    steps = sum(e.steps for e in engines)
    cpu = sum(e.sched_cpu_s for e in engines)
    if hasattr(cost, "oracle") and cost.oracle is not None:
        summary["astra_runs"] = cost.oracle.runs
    if coord.dp_group:
        summary["dp_rounds"] = coord.dp_rounds
        summary["dp_dummy_forwards"] = coord.dp_dummy_forwards
    summary["async_scheduling"] = spec.async_scheduling
    if notes:
        summary["notes"] = notes
    summary.update({"engines": len(engines), "steps": steps, "wall_s": round(coord.wall_s, 2),
                    "vllm_step_cpu_ms_per_step": round(1e3 * cpu / max(steps, 1), 3),
                    "kv_cache_gib_per_engine": round(kv_bytes / 2**30, 2),
                    "num_gpu_blocks": engines[0].vllm_config.cache_config.num_gpu_blocks})
    print(json.dumps(summary, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
