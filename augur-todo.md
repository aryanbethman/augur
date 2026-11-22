# augur to-do

End goal: augur replaces running vLLM on GPUs for researchers without
hardware, so they can do serving research faster and cheaper without running
inference experiments. Every item is judged by that.

Order: top to bottom. Status: [ ] open, [~] in progress, [x] done.

## Current state (2026-11-21)
Merged on 2026-11-21: kv-aware router; KV offload with PD and DP (vLLM's MultiConnector); one step log per engine and
exact replay of multi-engine runs (round-robin, PD, DP, with and without offload, sync and async); MoE
(Qwen3-30B-A3B) priced from shipped tables at TP1/TP2; DP + async pays the CPU term; `cxl` tier preset; warning
and `steps_beyond_profile` when a run leaves the profile's range; ASTRA-Sim runs in a private directory per process
(concurrent runs no longer corrupt each other); `python -m augur.profile` (profiling pipeline, GPU stages not yet
run); ~10% faster step loop at 256 GPUs. Earlier: vLLM Router policies, KV offload to CPU and tiers (2026-11-18),
audit fixes (2026-11-18), --profile-variant (2026-11-19).

Contention-aware co-simulation (design doc) is merged (2026-11-21), off by default (`--contention on`).

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

## Next features (proposed order, no runs needed)
1. Contention: validate against real contended runs  2. Shipped profiles with decode query lengths 1,2,4  3. MoE TP4 and DP+EP on the tables

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
- [ ] Ship q=2,4 attention for RTX4060Ti Qwen3-1.7B (2026-11-22). Problem: the shipped profile times attention only
      at 1 query per request, so ngram verification (1+k queries, 1.4-1.9x dearer) is priced as a plain decode and
      real ngram runs come out too fast (mean latency −17.8% async, −21.6% sync; ShareGPT 300, GPU 0).
      Re-profiled with --decode-q-lens 1,2,4 (anjuna3 out/profile_qlens, fork perf/RTX4060Ti_qlens). The full new
      profile fixes ngram (+5.6% / −0.2%) but its other tables moved up to 5% from last week's (prefill 2048
      −4.9%), and stepcheck predicted/measured got worse: ShareGPT 0.965→0.961, SWE 0.973→0.955, bench
      0.990→0.982. So build a hybrid: the shipped profile plus only the q=2,4 attention and skew tables (local
      branch profile-qlens on anjuna3, 2762287). Non-spec pricing is unchanged, six replays exact, ngram runs
      +8.9/−1.0/+4.7% async, +2.2/−5.4/−0.9% sync (mean TTFT/TPOT/latency).
      Blocker: test_regression_exact and test_live_against_fork need augur and the fork to price identically, and
      the fork's RTX4060Ti data has no q=2,4 tables. Plan: add the q=2,4 rows to the fork's attention.csv,
      skew.csv, skew_fit.csv and merge meta.yaml (attention_grid.decode_q_lens, sampling targets, skew
      case_families and fits; keep q=1 entries); commit and push the fork; regenerate augur's profile from it with
      convert; rebuild tests/data/table_cost/qwen3-1.7b-rtx4060ti with build_table_regression.py; full suite +
      replays; update docs/accuracy.md section 3. Also: refit --spec-us-per-request on a quiet host and another GPU
- [x] Offload: GPU-to-CPU copies lose 15% when a load overlaps; decode is 7% slower during a copy: modeled by the
      fabric (--d2h-duplex-penalty, --hbm-interference)
- [x] Contention-aware co-simulation (2026-11-21, feature/contention-20261121): augur/sim/fabric.py, `--contention
      on`, `--fabric`, `--host-mem-gbps`, `--host-mem-background-gbps`, `--pcie-root-gbps`, `--d2h-duplex-penalty`,
      `--hbm-interference`. Off by default and byte-identical (six real-GPU replays exact); own-log replay exact
      with it on. Matters little on an idle host; strongly for PD pulls on shared NICs and for a busy host memory
- [ ] Contention: calibrate --hbm-interference (1.4, one 4060 Ti) and host memory on more hosts; two-socket hosts;
      real contended offload/PD runs at several GPUs per host; poll-time waits use projected flow ends; host memory
      shares fairly and does not reproduce the measured per-GPU asymmetry
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
- [ ] MoE TP4 (the TP2 correction does not carry over: 24-48 ns/step residual), DP+EP and EP on the tables
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
- [~] Automatic KV size from memory accounting (now a manual --kv-reserve-gib). Looked at 2026-11-18: in the 7
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
- [ ] Faster DP rounds: per round 25 ms = chakra graph conversion 13 ms, ASTRA-Sim 6 ms, trace generation 4 ms
      (2026-11-21). A cache cannot help (no round repeats; 5.7% of per-rank traces do). Per-rank worker processes
      gave byte-identical output but no measurable speed-up on the loaded host
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
