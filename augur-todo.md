# augur to-do

End goal: augur replaces running vLLM on GPUs for researchers without
hardware, so they can do serving research faster and cheaper without running
inference experiments. Every item is judged by that.

Order: top to bottom. Status: [ ] open, [~] in progress, [x] done.

## Current state (2026-11-23)
Merged on 2026-11-21: kv-aware router; KV offload with PD and DP (vLLM's MultiConnector); one step log per engine and
exact replay of multi-engine runs (round-robin, PD, DP, with and without offload, sync and async); MoE
(Qwen3-30B-A3B) priced from shipped tables at TP1/TP2; DP + async pays the CPU term; `cxl` tier preset; warning
and `steps_beyond_profile` when a run leaves the profile's range; ASTRA-Sim runs in a private directory per process
(concurrent runs no longer corrupt each other); `python -m augur.profile` (profiling pipeline, GPU stages not yet
run); ~10% faster step loop at 256 GPUs. Earlier: vLLM Router policies, KV offload to CPU and tiers (2026-11-18),
audit fixes (2026-11-18), --profile-variant (2026-11-19).

Contention-aware co-simulation (design doc) is merged (2026-11-21), off by default (`--contention on`).

2026-11-23 (branch feature/next-features-20261123): RTX 4060 Ti profile with q=1,2,4 attention (fork e8625e0, not
pushed); MoE TP4 and DP2 + EP2 on the tables; contention checked against two real offload runs and
--hbm-interference refitted (decode 1.1, prefill 0.1); exact replay of the sync 4 GiB-KV offload run.

## Verification runs, once the GPUs and CPUs are free
1. [x] Rerun the PD async comparisons after the async loop fix: async PD moved 0.5% mean TTFT; sync identical
2. [x] 256-GPU scale run with the table pricer: 20 s, 1.5 GB RSS, byte-identical to the ASTRA-priced run
3. [x] Real-GPU check of CPU offload: replay exact sync 11,396/11,396, async 11,136/11,136 (after the request-id
   fix); end to end −3 to −4% mean TTFT; PCIe 12.7 GB/s
4. [x] Real-GPU runs where the schedulers differ: async SWE-bench, both simulators within 1% mean; ngram
   speculation, both 25-35% too fast (see section 4)
5. `vllm bench serve` vs `augur bench` end to end at other prompt lengths (256 and 4,096 done for TTFT)
6. Wheel pip-installed into a clean venv: needs vllm 0.33.0 and its dependencies from the network (uv cache empty)
7. `python -m augur.profile` for H100 Llama-3.1-70B TP4 with KV to 32k (~4-12 h of GPU), then redo the arrival
   study; a short pilot on the RTX 4060 Ti first to shake out the memory probe

## Plan without new hardware (agreed 2026-11-23; order = priority)
Replaces the earlier next-features list (its items are marked "next-features N"; storage tiers wait for paper 2).
GPU 1 of the test host is clock-locked, which spoils timing but not scheduling decisions, so both 4060 Tis serve
for decision checks. The test host has ~14 GB of disk free.

Fixes
1. [x] KV size (c7ab353): GPU_FACTS in cost/tables.py gives each --hardware its memory (--hbm-gb overrides); the
   reserve is vLLM 0.33's (non-torch + peak activation, which holds the CUDA-graph estimate) from the profile's
   measured vllm_memory, else estimated from the model shape with a warning. Against the 7 real 4060 Ti logs:
   −2.4% to +1.0% (estimate: −0.2% to +3.3%). Open: the H100 and RTX PRO 6000 totals are from public logs, not
   measured here; ngram runs' larger reserve (0.79-1.36 GiB) is not modeled
2. [x] Flags (2a0dc58): --scheduler-cls (wrapped so --step-log-vllm still logs), --scheduling-policy priority with a
   per-request "priority" in workloads (the client sends it when non-zero; untested against a real server),
   --kv-cache-dtype fp8 (block count doubles; prices stay bf16, warned)
3. [ ] Real DP runs replayable: log DP dummy rounds, which run outside the scheduler (section 4)

Additions
4. [ ] Same-step-time test: Vidur, BLIS (inference-sim), Frontier and LLMServingSim priced with augur's step times
   on the real runs; what error is left comes from their copies of vLLM's scheduler. Decides whether decision
   fidelity leads the paper
5. [x] Error breakdown (206a5a4): `python -m augur.verify.breakdown`, results in docs/accuracy.md section 3. Outside
   the step loop, step time and decisions each within ~2% of the mean on the non-speculative runs; ngram decisions
   dominate their tails (median TTFT −24%): acceptance is a rate, not token content
6. [ ] Custom-scheduler example through --scheduler-cls (e.g. shortest job first)
7. [ ] KV cache spec from vLLM's own model definition (meta device) for sliding-window, MLA and hybrid models;
   decisions checked on a small sliding-window model on the 4060 Ti
8. [ ] Pipeline parallelism; EAGLE with DP; MoE MTP pricing. Next-features 2 and 4
9. [ ] Packaging: Dockerfile, fork and ASTRA-Sim setup script, CI for the tests that need no fork, LICENSE.
   Next-features 6

Real runs on the test host
10. [ ] Multi-GPU decision replay on GPU 0 + GPU 1: TP2, DP2 (after 3), one prefill + one decode engine (needs NIXL)
11. [ ] Winner-changes pilot: sweep one setting with real runs; does augur pick the real best setting while the
    copied schedulers do not?
12. [ ] Held-out accuracy: 2-3 workloads not used for any fit (another chat set, long context to 32k, bursty
    arrivals)
13. [ ] More single-GPU decision coverage: priority scheduling, heavy preemption, prefix caching off, other chunk
    sizes, SWE-bench at a higher session rate
14. [ ] LMCache connector unmodified, replayed against a real LMCache run. Next-features 3
15. [ ] vLLM upgrade study: move to the next vLLM release, count the lines changed, rerun the six real replays

Needs the repo owner: push fork commit e8625e0 (the shipped 4060 Ti profile's provenance points at it); pick a
license; ask Revati's authors for code; decide whether to report the tiering livelock (section 4) upstream.

## Next: what only running vLLM's own code can do (added 2026-11-23; after the plan above)
Why: BLIS (inference-sim) ports vLLM 0.33's scheduler to Go and checks it step by step against vLLM's classes. With
real step durations it reproduced the first 835 of 3,402 steps of ShareGPT 300 and came within 0.3% end to end, so
accuracy alone will not set augur apart from it on ordinary workloads. These items rest on what a port cannot do.
The paper's lead claim is still open: lead with these (accuracy against ports as support), or lead with accuracy
against ports, decided after the heavy-preemption runs (plan item 13).

16. [ ] Token-faithful simulation: with recorded outputs, the simulated model emits the real tokens; vLLM's own ngram
    proposer is verified against them (exact acceptance per request and step for greedy decoding, instead of
    --spec-accept-rate), and prefix hits on generated text are the real ones (identical generations share blocks).
    Check on the existing ngram runs, whose decision error is −24% median TTFT today (plan item 5). No GPU
17. [ ] Plugin case study: a KV eviction or offload policy (or a scheduler) written as a vLLM plugin, ranked against
    alternatives at hundreds of GPUs in augur, then the same code run on the 4060 Ti to confirm the ranking; BLIS
    would need it rewritten in Go first
18. [ ] vLLM version study (plan item 15, moved up): the lines changed to move augur to the next release, a decision
    change between the two versions that only the new code produces, and what each of BLIS's ports took (lines,
    days after the vLLM release) from its history
19. [ ] Third-party and unreleased code on day one: LMCache unmodified (plan item 14) and a vLLM release candidate or
    open pull request run without changes to augur
20. [ ] Workloads where a port's inputs lose information: prompts that overlap in part (shared system prompts,
    few-shot, RAG), which BLIS encodes as prefix groups; plus the heavy-preemption runs of plan item 13. Measure
    BLIS's error against augur's on each
21. [ ] Exact counterfactuals on a real trace: replay a real log step for step, change one setting, report the
    effect; contrast with a port, which reproduces 57% of the steps of the same run
22. [ ] Where BLIS is ahead: a pluggable timing model for GPUs nobody has profiled (a roofline pricer, or an adapter
    to BLIS's timing, giving "vLLM's real decisions under any timing model"); measure augur's speed against BLIS's
    at 256-1,024 GPUs and report it honestly

Done 2026-11-23: contention checked against real contended runs (K refitted), q=1,2,4 shipped profile, MoE TP4 and
DP2 + EP2 on the tables, the sync offload replay difference fixed.

## 0. Prove augur runs vLLM properly (top priority)
- [x] Logging scheduler (augur.vllm_hooks), plugged in via `scheduler_cls`
- [x] Replay (augur.replay); self-test exact on ShareGPT, SWE-bench, heavy
      preemption; in scripts/validate.sh (commit 6f19bc2)
- [x] Trace client for `vllm serve` (augur.client) — untested against a real server
- [x] On GPU, synchronous: ShareGPT 3,402/3,402 and SWE-bench 11,030/11,030 steps
      identical (commit b9d3878); found + fixed the resumed-request bug
- [x] On GPU, async scheduling (vLLM's default): 3,250/3,250 steps (f03a7f5)

## 1. Timing accuracy against real vLLM on GPUs
- [x] Profile Qwen3-1.7B on the RTX 4060 Ti (GPU 0; fork 8bf9682). GPU 1 of the
      test host is clock-locked at 510 MHz (needs root to fix)
- [x] Skew sweep (fork 2a2321d): step ratio 0.951→0.965 ShareGPT, 0.950→0.973 SWE; e2e unchanged
- [x] Knee TTFT diagnosed (d954e4a): real run replays 778/778; with real arrivals, 1% step bias -> 23% median TTFT (backlog). Real run-to-run noise 8%
- [x] Sub-1% step pricing: not reachable (see section 6); the profile's sign flips between runs
- [x] Load-dependent front-end delay: it is the engine loop's wait, already simulated (section 6)
- [x] Per-step (augur.stepcheck): predicted/measured 0.95 on ShareGPT and SWE-bench
- [~] End to end: real `vllm bench serve` vs `augur bench`, random 1024/128, 2/6/12 req/s

## 2. CPU overhead in virtual time
- [x] `--step-cpu-us` (sync 3 ms: ShareGPT e2e within 0.5%); async hides it;
      `--frontend-latency-ms` (15 ms closes light-load TTFT)

## 3. Use it like vLLM
- [x] `augur bench`: `vllm bench serve` args and report, on augur (c8dbd75)
- [~] Packaging (bench runs now replayable: c9e57e9): `python -m augur.profiles` + clear missing-profile error done; fork/ASTRA setup still manual

## 4. Feature coverage
- [x] Speculative decoding: ngram, sync, acceptance rate; replay from logs (f14aa07, d16fcb0)
- [x] Speculative decoding with async scheduling (runs; not checked against real async yet)
- [x] Real-GPU replay of an ngram run (sync): 1,880/1,880 after drafter fix (e0f0a18)
- [x] Real-GPU replay of an async ngram run: 1,722/1,722 (818a204)
- [x] Speculative decoding: EAGLE / EAGLE-3 / MTP (2026-11-21, 7aa5e3c): --speculative-method eagle|eagle3|mtp runs
      vLLM's own path with a synthetic draft config (one target-width layer; MTP from the target's own layers), the
      draft layers in the KV cache, and k drafter passes priced from the target's layer tables. Own-log replay exact.
      Approximate: acceptance is a rate; the drafter is a target layer plus 2x/3x o_proj; verification attention is
      priced at query length 1 unless profiled (augur.profile --decode-q-lens). Also fixed: k with 1+k not in
      {2, 4, 8} crashed the CUDA-graph dispatcher (ngram too)
- [ ] EAGLE: profile a real EAGLE head; MoE MTP pricing (needs per-layer MoE); eagle with DP
- [x] KV offload to CPU (2026-11-18, branch feature/router-offload-20261118): vLLM's OffloadingConnector
      unmodified, augur/sim/offload.py as its GPU worker (store/load/sync-load FIFO lanes, latency + bytes/bw,
      default the CPU memory's as LLMServingSim); --kv-offload-gib/-gbps/-latency-ns; step log records offload
      jobs, replay re-times them. Own-log replay exact (sync and async); runs repeat exactly (PYTHONHASHSEED pinned)
- [x] Real-GPU check of CPU offload (2026-11-21, f394170): replay exact sync 11,396/11,396 and async 11,136/11,136
      once replay reuses the logged request ids (vLLM's CPU-tier eviction follows set order of request ids); end to
      end mean TTFT/TPOT/latency −3.2/−1.6/−2.9% sync, −4.0/−1.3/−3.2% async; prefix hits within 1.5 points
- [x] ngram speculation step cost (2026-11-21, d5d6b1b): the gap is verification attention with 1+k queries priced
      as a one-token decode (1.4-1.9x), plus 160-210 us per request per step; the ngram proposer is 0.1-0.25 ms.
      --spec-us-per-request (default 100); stepcheck now prices spec steps as decodes. With a q=1,4 profile the real
      runs come to −1.3% (async) and +0.6% (sync) mean latency
- [x] Ship q=2,4 attention for RTX4060Ti Qwen3-1.7B (2026-11-23). The shipped profile timed attention only at 1
      query per request, so ngram verification (1+k queries, 1.4-1.9x dearer) was priced as a plain decode: real
      ngram runs came out −17.8% (async) and −21.6% (sync) mean latency. A whole new profile (2026-11-21,
      --decode-q-lens 1,2,4) drifted up to 5% elsewhere and made stepcheck worse, so the fork's RTX4060Ti bundle got
      only its q=2,4 attention and skew rows (fork e8625e0, local on anjuna3, not pushed: q=1 rows and skew cells
      unchanged, skew refit with refit-skew); augur's profile is converted from it (exact against the fork), the
      regression data rebuilt (workload shapes now include an ngram run). Non-spec prices identical; ngram runs
      +8.9/−1.0/+4.7% async, +2.2/−5.4/−0.9% sync (mean TTFT/TPOT/latency). 276 tests pass
- [ ] Push the fork commit e8625e0 (feature/vllm-sim-20261114) from a machine with access; refit
      --spec-us-per-request on a quiet host and another GPU
- [x] Offload: GPU-to-CPU copies lose 15% when a load overlaps; decode is 7% slower during a copy: modeled by the
      fabric (--d2h-duplex-penalty, --hbm-interference)
- [x] Contention-aware co-simulation (2026-11-21, feature/contention-20261121): augur/sim/fabric.py, `--contention
      on`, `--fabric`, `--host-mem-gbps`, `--host-mem-background-gbps`, `--pcie-root-gbps`, `--d2h-duplex-penalty`,
      `--hbm-interference`. Off by default and byte-identical (six real-GPU replays exact); own-log replay exact
      with it on. Matters little on an idle host; strongly for PD pulls on shared NICs and for a busy host memory
- [x] Contention against real contended runs (2026-11-23, one 4060 Ti, SWE-bench 50 sessions, 16 GiB CPU tier, GPU
      KV cut to 4 GiB so 138 GB comes back from CPU per run; async and sync). Links ran at idle rates (fabric agrees).
      Real steps beside a transfer are slower: decode +4.4/+5.6% one way, steps with prefill +0.1-0.8%. Fitted
      --hbm-interference 1.1 for decode-only steps and the new --hbm-interference-prefill 0.1 (was 1.4 for all, from
      a GEMV microbenchmark: async e2e +2.4% TTFT, now −0.1%). RTX4060Ti now has a 288 GB/s default (interference
      was silently off for it). Scripts: real_offload_check.sh KV_GIB / HOST_LOAD, fit_hbm_interference.py,
      step_xfer_slowdown.py, host_mem_load.py; stepcheck --csv records each step's transfer bytes
- [x] Sync replay of the 4 GiB-KV offload run (2026-11-23): was 24,191 / 24,195 steps. vLLM's worker submits a step's
      async loads after the forward and polls once its CPU has launched the copies, so a 127-block load was reported
      in its own step (71.5 ms against ~31). Replay now reports a job in its issuing step when the log did: all four
      real offload runs replay exactly. The launch's CPU time is not priced (0.07% of the sync run's time)
- [ ] Contention: K and host memory on more hosts and GPUs (fitted on the same two runs, one 4060 Ti); two-socket
      hosts; real contended runs at several GPUs per host (GPU 1 here is clock-locked); a transfer-bound run (here
      transfers overlapped ~8% of steps); poll-time waits use projected flow ends; host memory shares fairly and
      does not reproduce the measured per-GPU asymmetry
- [~] Multi-GPU offload: two GPUs do not contend on an idle host, but under memory-bound CPU load aggregate copy
      bandwidth falls 25-33% (2026-11-21, two-GPU benchmark): the fabric models it as shared host memory
      (--host-mem-background-gbps); the per-GPU asymmetry is not reproduced
- [x] Tiered KV offload (2026-11-18): vLLM's TieringOffloadingManager unmodified (cascade CPU -> tiers,
      promotion tier -> CPU -> GPU), augur/sim/tiers.py SimMemoryTier as the tier (capacity, LRU, FIFO
      write/read lanes, latency + bytes/bw on the virtual clock), CPU tier buffer anonymous (no /dev/shm).
      --kv-offload-tier NAME:GIB:GBPS[:LATENCY_NS], repeatable — memory tiers (paper 2). Own-log replay exact
- [ ] Tiered offload: real-hardware check (needs a CXL box or a second memory pool)
- [x] Tiered offload: `--kv-offload-tier cxl:GIB` takes LLMServingSim's cxl_mem defaults, 60 GB/s and 250 ns per
      device (fork configs/cluster/single_node_cxl_instance.json; example values, not a measured device) (3b05161)
- [ ] Tiered offload: storage tiers (fs/obj) modeled as more than a slow memory tier
- [x] KV offload with PD or DP (2026-11-21, 90924c8): MultiConnector (NIXL + OffloadingConnector) for PD; DP ranks run
      the offload worker (sync: loads committed when the round is priced; async: loads start at launch, store-flush
      waits ignored). Own-log replay exact (21b01a2)
- [x] Multi-engine step logs and replay (2026-11-21, 21b01a2): one log per engine (<prefix>.e<id>.<pid>.jsonl),
      --replay-durations with a run prefix, DP rounds logged by augur and replayed, `replay compare` on run prefixes.
      Own-log replay exact for round-robin, PD, PD+offload, DP, DP+offload, sync and async
- [ ] Real-GPU multi-engine replay: real vLLM logs carry no DP dummy rounds, so a real DP run cannot be replayed
      round by round; PD and multi-engine real logs not tried (needs GPU 1)
- [ ] KV offload: LMCache connector
- [ ] vLLM's tiering livelocks when one request's prefix needs more chunks than the CPU tier holds (promote,
      evict, promote ...); augur now stops such a run after 60 s simulated without a token (STALL_NS).
      Worth reporting upstream?
- [ ] Pipeline parallelism
- [x] Async scheduling with DP (68f24df)
- [x] PD M x N engines; shared-link KV pull contention (55da69c)
- [x] vLLM Router policies (2026-11-18): random, power-of-two, cache-aware (router default; per-engine token
      prefix trees, LRU eviction), consistent-hash, rendezvous-hash, sticky-least-loaded, with the router's
      defaults; in-flight counted until the stream ends. --router-* flags
- [x] vLLM Router kv-aware policy (2026-11-21, 90924c8): scoring ported from the router's kv_aware.rs (seen only in
      part), index from vLLM's own KV events (GPU and offload CPU tier). Ties by lowest engine id (assumed). With PD
      only prefill placement uses it. Not checked against the real router
- [x] Async loop fix (2026-11-18) reran: async PD moved 0.5% mean TTFT (207.8 -> 206.8 ms), sync PD identical

## 5. Profiles
- [x] augur.profile pilot (2026-11-21): Qwen3-1.7B on the 4060 Ti, small grid, 19 min; check, memory, layers and
      convert all work; prices within 0.2% of the shipped profile; memory numbers match the real servers' except peak
      activation (0.82 vs 0.50-0.63 GiB)
- [~] Profiling pipeline (2026-11-21, c2e506c): `python -m augur.profile --model M --hardware H --tp N [--max-kv
      --max-batch-tokens]` runs the fork's profiler (layers, skew), records vLLM's memory accounting (weights,
      non-torch, peak activation, CUDA graphs, KV size) from a dummy-weight start-up, converts with the self-check;
      stages resume; --dry-run. Dry run and conversion tested, GPU stages not yet run
- [ ] Profile ranges: the shipped H100 Llama-3.1-70B profile stops at 2,048 KV tokens and 2,048 tokens per step;
      augur warns and reports steps_beyond_profile (30af200). Re-profile to 32k KV with augur.profile

## 6. Standalone, and closer to hardware than LLMServingSim (proposed 2026-11-16, order = priority)
- [x] Step-level profiler: TESTED AND DROPPED (2026-11-16, out/stepfit/). Held out, direct fits are worse than
      the layer profile (totals 0.56-1.15 vs 0.94-1.01). Async GPU pricing is already at the run-to-run floor:
      decode 0.986/1.012, mixed 0.992/0.992 on two runs; identical steps vary 2.5% within a run
- [x] Sync gap is vLLM's CPU time: --step-cpu-us now defaults to the measured 2 ms. Sync ShareGPT
      -3.3% -> +0.2% TTFT, SWE-bench -4.4% -> +0.5%; async unchanged (sweep in out/cpu_sweep/)
- [ ] CPU term is one host's value (i7-12700, vLLM 0.33): needs a second host to know how it varies
- [x] Knee metrics as a range: augur bench --sensitivity (default +-1%); 6 req/s median TTFT 735-1,293 ms
      covers both real runs (1,289 and 1,186)
- [x] Own profile format + pricer (c980fcb): augur/cost/tables.py + shipped profiles (55 KB each) for
      Qwen3-1.7B/4060Ti and Llama-3.1-8B/RTXPRO6000; identical to the fork to the ns (whole runs byte-identical);
      augur bench runs with no fork. --cost tables / auto
- [x] Standalone for shipped profiles (1aa37c2): augur defaults to --cost auto; model config from the profile;
      26 of 36 tests run with no fork (the other 10 compare against it); offline-built wheel runs the benchmark
- [x] augur bench without its tokenizer (offline, not cached): clear error with the way out (--tokenizer DIR or
      a Hugging Face download) instead of a traceback (2026-11-18)
- [ ] Standalone, remaining: the wheel builds offline and runs from outside the repo (shipped profile, entry point,
      package data; 2026-11-21), but a clean `pip install` needs vllm 0.33.0 and its dependencies from the network
- [x] Tables for MoE (2026-11-21, b60b4cc): Qwen3-30B-A3B-Instruct-2507, RTX PRO 6000, TP1 and TP2, DP=1; byte-
      identical to the fork on SWE-bench (765/765), 1.6-2.2x faster whole runs. MoE block stored per token count (no
      scipy); TP2 via the fork's legacy table plus a fitted ASTRA-Sim communication correction
- [x] MoE TP4 on the tables (2026-11-23): at TP4 ASTRA-Sim charges a MoE step min(M, max(H+A-B, min(M-B, H+A,
      max(M-2B, 2A+H-2B)))) ns less than the dense form (H = M - M//2; nothing at T=0, where no MoE block is emitted),
      fitted on a grid of A, B with M = 20/40/41/48 and exact on 2,000 random shapes over 8 cluster settings.
      Shipped for the fork's projected bundle only: --tp 4 --profile-variant bf16-tp4proj (--cost tables/auto now
      read shipped <variant>-tp<N> profiles). Whole SWE-bench run byte-identical to the fork, 9.0 s vs 21.2 s
- [x] DP+EP on the tables (2026-11-23): DP2 + EP2 of TP1 ranks, the one DP layout the fork has native MoE tables
      for (moe_components dp2_ep2_rank0/1). The TP1 profile carries per-rank gate / finalize (by the rank's tokens)
      and experts (by the round's sum) read off the fork's traces, and TableCost.round_ns a closed form of ASTRA-Sim's
      round: token-id reads FIFO on the node's CPU memory (first member first); at each collective (hop h, duration
      D) the early rank leaves at t_late + D, the late one at t_late - min(gap, h) + D; sampled-id writes in
      finishing order. Exact on 480 random rounds over 6 cluster settings and 1,000 rounds from fork runs; ShareGPT
      300 DP2+EP2 byte-identical sync and async, 8.4 s vs 34.8 s; SWE-bench 11.1 s vs 159 s; with KV offload
      12 s vs 103 s. --cost auto/tables use it for --dp 2 --enable-expert-parallel
- [ ] DP+EP beyond DP2 + EP2 of TP1 ranks: the fork has no native MoE tables for them and falls back to its legacy
      EP table ("DP/backend matching is not verified"; DP4+EP4: one all-gather on the dispatch marker, a whole-block
      expert row, a reduce-scatter), so profile those layouts first; the collective rule was derived for two ranks.
      MoE TP8: no bundle to fit against
- [x] TP>1 without the ASTRA binary (afa6a3b): closed form of the ring collectives in tables.py, identical to
      the fork on whole runs (8B TP2, 70B TP4, 4 x TP4); shipped 8B tp2, 70B tp4/tp8. Refuses settings where the
      form is not exact (2A > all-reduce step or 2B > last collective step)
- [x] H100 profiles default to 3,350 GB/s GPU memory bandwidth (H100 SXM, as the fork's H100 cluster configs)
      when the profile has none (tables.NPU_MEM_BW_GBPS; 2026-11-18)
- [ ] TP follow-ups: the guard rests on ~300 probes, not a derivation; the inexact corner (large CPU-memory
      latency) is not understood; multi-node out of scope. 256-GPU run redone with the table pricer (2026-11-21):
      20 s, 1.5 GB RSS, byte-identical to the ASTRA-priced run
- [x] Real-GPU runs where the schedulers differ (2026-11-21): async SWE-bench 50 (replay 11,214/11,214): augur
      −0.8/+0.1/−0.6%, LLMServingSim −0.4/−0.6/−0.4% mean TTFT/TPOT/latency; at 0.2 sessions/s they do not diverge
- [ ] Same at a higher session rate, where vLLM's and LLMServingSim's schedulers can differ
- [x] Front-end model (456dcac): constant 6.5 ms intake + 8 ms delivery as defaults; the load-dependent
      part is the engine loop, which augur reproduces (775/778 steps from client send times). 2 req/s
      TTFT -11.7% -> -1.2%. Trace client's 100-connection cap removed (untested on a server)
- [x] Front end by prompt length (2026-11-21): 3.0 ms + 3.0 us per token measured at 256/1,024/4,096 tokens;
      --frontend-latency-ms 3.4 + --frontend-us-per-token 3.0 (6.5 ms at 1,024 tokens as before)
- [ ] Front end: token-id prompts (the trace client) and other hosts
- [x] Run-to-run noise: covered by the +-1% sensitivity range in augur bench
- [x] Automatic KV size from memory accounting (done 2026-11-23, plan item 1). Looked at 2026-11-18: in the 7
      real 4060 Ti logs (same model and config) vLLM's own reserve moves 0.65-0.97 GiB run to run (peak
      activation alone 0.50-0.82), so one model/GPU cannot fit a formula. Better done in the profiling pipeline:
      record weights / non-torch / peak activation / CUDA-graph memory when a GPU is profiled. augur.profile now
      records them (2026-11-21); using them to size the KV cache is next
- [ ] Multi-GPU against real hardware (TP>1, PD): blocked on GPU 1's clock lock

## Also open
- [x] Repo restructured by role (sim/ cost/ trace/ verify/), pytest tests, docs/accuracy.md (960750c);
      ngram self-test found + fixed a replay preemption bug (52e39e0)
- [x] LLMServingSim comparison is a pytest test with its reference data in the repo (ccae81a); 23 tests pass from a fresh clone
- [x] LLMServingSim 256-GPU run on the current branch: matches augur; 2h11m vs 44 s
- [x] LLMServingSim template store bounded (fork 879b69d); identical output at 16 NPUs
- [x] 51.6 GB peak RSS at 256 NPUs: that was LLMServingSim's; augur is 1.5 GB at 256 GPUs (2026-11-21)
- [x] ASTRA-Sim determinism (2026-11-21, 825d797): ASTRA-Sim writes ./tmp__mem/remote_mem.json in its cwd, which was
      the fork's shared serving/ folder, so concurrent runs read each other's file (a few ns off) or crashed. Each
      augur process now runs it in its own directory; concurrent runs are byte-identical to the table pricer.
      The fork's own LLMServingSim has the same race when two of its runs share a checkout
- [x] Step loop skips idle engines' has_requests chain (a84b71a): ~10% faster at 256 GPUs, byte-identical
- [x] Commit the PROGRESS.md notes (speed explanation, crash fix)
- [~] Faster DP rounds: per round 25 ms = chakra graph conversion 13 ms, ASTRA-Sim 6 ms, trace generation 4 ms
      (2026-11-21). A cache cannot help (no round repeats; 5.7% of per-rank traces do). Per-rank worker processes
      gave byte-identical output but no measurable speed-up on the loaded host. DP2 + EP2 now prices from tables
      (2026-11-23); other layouts still pay this
- [x] --profile-variant (2026-11-19): price from a named fork profile folder, e.g. the projected Qwen3-30B-A3B
      RTXPRO6000 bf16-tp4proj (rebuilt in the vllm-sim fork's format from its measured tp1/tp2; not the a4053bc
      bundle, which that fork cannot load)
- [x] Arrival study TPOT gap (2026-11-21): the H100 Llama-3.1-70B profile covers KV up to 2,048 tokens and 2,048
      tokens per step; the sessions reach 33k, so 99.9% of steps are priced by extrapolation. LLMServingSim a4053bc
      extrapolates its v0 attention surfaces (64-token grid, round-up); the fork and augur extrapolate attention.csv
      from its last doubling interval: lookups differ by -60% to +9x at 4-40k KV. In range (ShareGPT, KV <= 1,947)
      augur, the fork and a4053bc agree to 0.2% median TPOT. Next: profile to 32k+ KV, redo the study
- [x] Qwen3-30B-A3B TP4/EP4 on a 4-instance cluster config: covered by a5e524d (11 s, byte-identical to the
      1-instance config); regression test 4233534
- [x] DP + async pays --step-cpu-us as single-engine async does (81f7176): hidden behind the previous round,
      delays a round only when the GPU would wait
