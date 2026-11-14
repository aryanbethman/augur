"""augur command line: run a workload on simulated vLLM engines."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="augur")
    p.add_argument("--fork", default=os.environ.get("AUGUR_FORK", os.path.expanduser("~/llmss-vllm-sim-20261114")),
                   help="LLMServingSim fork checkout (model configs, profiles)")
    p.add_argument("--model", required=True, help="e.g. meta-llama/Llama-3.1-8B")
    p.add_argument("--workload", required=True, help="JSONL in the fork's format (flat and/or agentic)")
    p.add_argument("--num-entries", type=int, default=None, help="first N requests/sessions only")
    p.add_argument("--engines", type=int, default=1)
    p.add_argument("--router", default="least-loaded")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--hbm-gb", type=float, default=80.0, help="simulated GPU memory")
    p.add_argument("--mem-util", type=float, default=0.9)
    p.add_argument("--kv-cache-gb", type=float, default=None, help="override the KV-cache size per GPU")
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--max-num-batched-tokens", type=int, default=None)
    p.add_argument("--max-num-seqs", type=int, default=None)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--no-prefix-caching", action="store_true")
    p.add_argument("--enforce-eager", action="store_true", help="no CUDA graphs, so no step padding")
    p.add_argument("--cost", default="fixed", choices=["fixed", "llmservingsim"])
    p.add_argument("--fixed-step-ms", type=float, default=25.0)
    p.add_argument("--fixed-per-token-us", type=float, default=0.0)
    p.add_argument("--hardware", default="RTXPRO6000", help="profile hardware for --cost llmservingsim")
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

    if a.quiet:
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    from augur.cost import FixedCost
    from augur.engine import Coordinator, EngineSpec, SimEngine
    from augur.models import estimate_weight_bytes, model_dir
    from augur.router import ROUTERS
    from augur.workload import Workload

    cache_root = os.path.expanduser("~/.cache/augur/models")
    mdir = model_dir(a.fork, a.model, cache_root)
    if a.kv_cache_gb is not None:
        kv_bytes = int(a.kv_cache_gb * 2**30)
    else:
        weights = estimate_weight_bytes(os.path.join(mdir, "config.json"), tp=a.tp)
        kv_bytes = int(a.hbm_gb * 2**30 * a.mem_util) - weights
        if kv_bytes <= 0:
            print(f"model weights ({weights / 2**30:.1f} GiB/GPU) do not fit in {a.hbm_gb} GB", file=sys.stderr)
            return 2

    if a.cost == "fixed":
        cost = FixedCost(base_ns=int(a.fixed_step_ms * 1e6), per_token_ns=int(a.fixed_per_token_us * 1e3))
    else:
        from augur.lss_cost import LLMServingSimCost
        cost = LLMServingSimCost(fork_root=a.fork, hardware=a.hardware, model=a.model, tp=a.tp,
                                 max_num_batched_tokens=a.max_num_batched_tokens or 2048,
                                 cpu_mem_bw_gbps=a.cpu_mem_bw, cpu_mem_latency_ns=a.cpu_mem_latency_ns,
                                 cluster_config=a.lss_cluster_config)

    spec = EngineSpec(model_dir=mdir, model_name=a.model, kv_cache_bytes=kv_bytes,
                      max_model_len=a.max_model_len, max_num_batched_tokens=a.max_num_batched_tokens,
                      max_num_seqs=a.max_num_seqs, block_size=a.block_size,
                      enable_prefix_caching=not a.no_prefix_caching, tensor_parallel_size=a.tp,
                      enforce_eager=a.enforce_eager)
    engines = [SimEngine(i, spec, cost) for i in range(a.engines)]
    if a.step_log:
        for e in engines:
            e.step_log = []
    coord = Coordinator(engines, ROUTERS[a.router](), Workload(a.workload, max_entries=a.num_entries),
                        lss_idle_quantum_ns=1_000_000 if a.lss_loop_compat else 0)
    rec = coord.run()
    if a.output:
        rec.write_csv(a.output)
    if a.step_log:
        import csv as _csv
        with open(a.step_log, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["engine", "start_ns", "dur_ns", "total_tokens", "prefill_tokens", "n_decode", "n_prefill"])
            for e in engines:
                for row in e.step_log:
                    w.writerow([e.engine_id, *row])
    summary = rec.summary()
    steps = sum(e.steps for e in engines)
    cpu = sum(e.sched_cpu_s for e in engines)
    if hasattr(cost, "oracle") and cost.oracle is not None:
        summary["astra_runs"] = cost.oracle.runs
    summary.update({"engines": a.engines, "steps": steps, "wall_s": round(coord.wall_s, 2),
                    "vllm_step_cpu_ms_per_step": round(1e3 * cpu / max(steps, 1), 3),
                    "kv_cache_gib_per_engine": round(kv_bytes / 2**30, 2),
                    "num_gpu_blocks": engines[0].vllm_config.cache_config.num_gpu_blocks})
    print(json.dumps(summary, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
