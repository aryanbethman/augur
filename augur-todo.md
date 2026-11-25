# augur to-do

Goal: augur replaces running vLLM on GPUs for serving research. It runs vLLM's own control plane, unmodified, on a
virtual clock. Focus now: scale, speed and research value, then a simulator others can install and reuse.

Sections are in priority order, and so are the items inside each. Status: [ ] open, [~] in progress.
Finished work lives in docs/accuracy.md and git history, not here.

Rule for every change: output stays byte-identical unless the change is meant to move it, the six real-GPU replays
stay exact, and the full test suite passes.

Rule for large-scale runs (items 1, 3, 5, 6, 17): engines must be loaded as in production, so time goes to real
scheduling work, not start-up, idle engines or a draining tail. Use scripts/baselines/speed.py's replicated ShareGPT
and SWE-bench workloads (load per GPU stays fixed as GPUs grow), at the real single-GPU runs' load: ShareGPT at
10 req/s per GPU (the real sharegpt300 run's rate), SWE-bench at the real swe50 run's 0.2 sessions/s, scaled per GPU,
with enough calls per GPU for several minutes of simulated steady state. The 2026-11-24 speed runs were too light:
12 calls per GPU at 1 req/s (SWE-bench: one session per GPU), 32-137 s simulated, and start-up was 53% of augur's
CPU at 16 GPUs (5% at 1,024). The October LLMServingSim scale runs are lighter still (50 SWE-bench sessions over up
to 256 GPUs). Report per run: share of simulated time each engine is busy, mean batch size against the real run's,
and start-up CPU separately from the steady state.

## 1. Scale, speed and utility (no new hardware)

Why: BLIS spends 2.2-13x less CPU than augur, because every augur step runs vLLM's Python scheduler and KV manager.
A Go port will stay cheaper per step. But BLIS's simulation loop runs on one core and needs up to 8x augur's memory
(29.6 GB at 1,024 GPUs), so augur can win on time to finish a run and on size, with every engine still unmodified vLLM.

1. [ ] **Profile a 1,024-GPU run** (ShareGPT and SWE-bench, the setup of scripts/baselines/speed.py). Split the CPU
   between vLLM's scheduler, KV manager, prefix-cache hashing and output processing, and augur's executor, pricer,
   event loop and engine start-up. This sets the targets of items 2 and 3.
2. [ ] **Less CPU per step, without touching vLLM's code.** Candidates: vLLM's stats and logging off, a cheaper
   prefix-cache hash setting (changes hash values, not decisions), skip output processing augur never reads, cache in
   the pricer, trim augur's event loop. Target: 1.5-2x less CPU. Matters most for sweeps, which already run one
   process per configuration.
3. [ ] **Engines in parallel worker processes, exact.** Each worker runs unmodified vLLM engines; workers sync only
   where engines interact.
   - Stateless routers (round-robin, random, hash): engines are independent between arrivals.
   - Load- and cache-aware routers: every worker advances to the next arrival, completions are gathered, the router
     decides, repeat (~12k sync points against ~1.3M steps at 1,024 GPUs).
   - Sessions: a next turn comes at least front end + delivery (~11 ms) after a completion, a safe window to run ahead.
   - A DP group stays inside one worker; PD runs ahead by the minimum KV transfer time.

   Done when: output is byte-identical to one process on every replay and scale test, and 1,024 GPUs finishes in
   under 30 s of wall time (now 5.7 min; BLIS takes 152 CPU s on one core).
4. [ ] **Async DP exact.** vLLM 0.33's DP loop with async scheduling (DPEngineCoreProc.run_busy_loop,
   step_with_batch_queue, batch queue of 2) applies a batch's output only after the next forward ends, so scheduling
   is not hidden behind the GPU. augur's async DP path (Coordinator._dpa_*) schedules the next batch at once, as
   single-engine async does: replay of the two real async DP runs differs from step 1 (370/1,985 and 462/2,187
   identical steps), and augur hides CPU time the real loop pays. Fix: model that loop in _dpa_* (a rank with nothing
   runs a dummy forward synchronously; all ranks sync at step 1 and every dp_sync_interval steps; forward j of every
   rank pairs into round j). The async DP tests' "hides CPU time" expectation changes with it. Async + DP is how vLLM
   is deployed at scale, so the scale results rest on this.
5. [ ] **Scale study against BLIS**, 16 to 16,384 GPUs: wall time, CPU and peak memory, and the size at which BLIS
   outgrows a normal host's memory. Report CPU per step as well as wall time.
6. [ ] **Sweeps that pick the right configuration.**
   - Throughput: a 100-configuration sweep (TP x DP split, batch-token budget, KV size, router) on one host,
     configurations per hour for augur and BLIS.
   - Correctness: sweep one setting with real runs on the 4060 Ti. Does augur pick the real best setting where the
     ported simulators do not?
7. [ ] **LLMServingSim at a higher session rate**, where its scheduler and vLLM's can diverge (at 0.2 sessions/s on
   async SWE-bench 50 they did not: both within 1% of the real means).

## 2. Standalone and reusable

8. [ ] **Install and run anywhere.** A clean `pip install` (needs vllm 0.33.0 and its dependencies from the network),
   a Dockerfile, CI for the tests that need no fork, a setup script for the LLMServingSim fork and ASTRA-Sim
   (still manual), and a LICENSE (needs the owner's choice). The wheel already builds offline and runs from outside
   the repo with a shipped profile.
9. [ ] **Profile your own GPU, end to end.** `python -m augur.profile --model M --hardware H --tp N` is written and its
   dry run and conversion are tested, but its GPU stages have never run. Run it on the 4060 Ti first (also shakes out
   the memory probe).
10. [ ] **Push fork commit e8625e0** (feature/vllm-sim-20261114), which the shipped 4060 Ti profile's provenance points
    at. Needs the owner, from a machine with access.
11. [ ] **vLLM version study.** Move augur to the next vLLM release: lines changed, the six real replays rerun, a
    decision change that only the new code produces, and what BLIS's ports took for each release (lines, days after
    the release) from its history. Needs the new vLLM wheel (download).
12. [ ] **Third-party and unreleased code on day one.** LMCache's connector unmodified, replayed against a real LMCache
    run, and a vLLM release candidate or open pull request run without changes to augur. Needs downloads.

## 3. Needs rented GPUs

13. [ ] **H100 profiles** for Llama-3.1-70B TP4 out to 32k KV tokens (the shipped one stops at 2,048 KV tokens and
    2,048 tokens per step; augur warns and reports steps_beyond_profile). About 4-12 GPU hours.
14. [ ] **MoE with wider expert parallelism** than DP2 + EP2 of TP1 ranks: profile those layouts, then price them from
    tables. Today they fall back to the fork's legacy EP table (unverified; its collective rule was derived for two
    ranks, and DP rounds through ASTRA-Sim cost ~25 ms each). MoE TP8 has no profile to fit against.
15. [ ] **Real prefill/decode replay** (one prefill and one decode engine; needs NIXL), and multi-GPU timing on
    GPUs without a clock lock (GPU 1 here is locked).
16. [ ] **The per-step CPU term on a second host.** 1.3 ms is one host's value (i7-12700, vLLM 0.33).
17. [ ] **Engines across hosts** (after item 3): the same sync over the network, for 100k+ GPUs.

## 4. Parked

No work planned; kept so the findings are not lost.

- Pipeline parallelism; EAGLE with DP; MoE MTP pricing (needs per-layer MoE); profiling a real EAGLE head.
- Contention: refit K and host memory on more hosts and GPUs and on two-socket hosts; a transfer-bound run; the
  measured per-GPU asymmetry of copy bandwidth under CPU memory load is not reproduced.
- Tiered offload: a real-hardware check (CXL or a second memory pool); storage tiers (fs/obj) as more than a slow
  memory tier.
- vLLM's tiering livelocks when one request's prefix needs more chunks than the CPU tier holds (promote, evict,
  promote, ...); augur stops such a run after 60 s simulated without a token. Report upstream? Owner's call.
- Accuracy refinements:
  - a delivery default per scheduling mode (the per-user run's TTFT is +16%, half a step counted twice);
  - the q=4 attention table looks low (ngram error grows ~0.2 us per context token);
  - stepcheck builds its dispatcher without the speculative config (0.2%);
  - the shared-KV term on other GPUs, at query length > 1 and for prefill chunks;
  - more than one run per policy in the plugin study, a second workload, and kv-aware routing at scale;
  - the breakdown's step/decision split is not meaningful on always-busy runs.
- Front end: token-id prompts (the trace client) and other hosts.
- Ask Revati's authors for their code.
