# How close augur is to real vLLM

augur's error has two independent sources, checked separately:

1. **Decisions.** Which requests run in each step, with how many tokens,
   which prefix hits, which preemptions. augur runs vLLM's own scheduler, so
   these should be vLLM's exactly. Checked by replaying real vLLM runs step
   for step.
2. **Step times.** How long each step takes on the GPU. These come from the
   GPU profile (LLMServingSim's profiler), so this error is the profile's.
   Checked per step and end to end against real runs.

Hardware for all real runs: one RTX 4060 Ti, vLLM 0.33.0, Qwen3-1.7B,
`max_num_seqs` 128, 2,048-token budget, prefix caching and chunked prefill on.

## 1. Decisions: identical to vLLM

Real `vllm serve` runs with augur's logging scheduler
(`--scheduler-cls augur.verify.hooks.LoggingScheduler`, or
`LoggingAsyncScheduler`; `AUGUR_STEP_LOG=<path>`), then the same trace through
augur with the real run's arrival times and step durations
(`--replay-durations <log>`), then `python -m augur.verify.replay compare`:

| Real run | Steps with identical decisions |
| --- | --- |
| ShareGPT, 300 requests at 10 req/s (overloaded; preemptions), sync | 3,402 / 3,402 |
| SWE-bench, 50 agent sessions / 765 calls, 24k context, sync | 11,030 / 11,030 |
| ShareGPT 300, async scheduling (vLLM's default) | 3,250 / 3,250 |
| ShareGPT 300, ngram speculative decoding k=3, sync | 1,880 / 1,880 |
| ShareGPT 300, ngram speculative decoding k=3, async | 1,722 / 1,722 |
| `vllm bench serve`, random 1,024/128 at 6 req/s, async | 778 / 778 |
| SWE-bench 50 sessions, async | 11,214 / 11,214 |
| SWE-bench 50 sessions, KV offload to a 16 GiB CPU tier, sync | 11,396 / 11,396 |
| SWE-bench 50 sessions, KV offload to a 16 GiB CPU tier, async | 11,136 / 11,136 |
| ShareGPT 300, ngram k=3, sync, rerun on GPU 0 | 1,653 / 1,653 |
| ShareGPT 300, ngram k=3, async, rerun on GPU 0 | 1,760 / 1,760 |

Replay takes the KV block count from the server log (it varies run to run)
and uses `--no-recorded-outputs`; runs from `augur.verify.client` add
`--filler-vocab 150000` so unrecorded prompt tokens get in-vocabulary ids.
With timing fixed from the log, any mismatch is a bug in what augur wraps
around vLLM (executor, coordinator), not in the timing model. Every
mismatch found so far was such a bug, now fixed (a resumed preempted
request's prefill length; async log ordering; ngram drafts near
`max_tokens`; with KV offload, request ids: vLLM iterates sets of request
ids, the CPU tier's eviction order follows that order, and the real engine's
ids carry a random per-run suffix, so replay now reuses the logged ids).
`tests/test_replay_selftest.py` replays augur's own logs for sync, async and
speculative runs; `tests/test_multi_replay.py` for several engines,
prefill/decode and DP, with and without offload.

## 2. Step times: the profile

Profile: Qwen3-1.7B TP1, RTX 4060 Ti, with the attention skew sweep (fork
2a2321d). `python -m augur.verify.stepcheck` rebuilds every real step's batch
from the step log, pads it with vLLM's CUDA-graph dispatcher, prices it and
compares with the measured step:

| Real run | Predicted / measured, all steps | Median step error: decode / mixed / prefill |
| --- | --- | --- |
| ShareGPT 300, sync | 0.965 | −2.8% / −3.7% / −11.8% |
| SWE-bench sessions, sync | 0.973 | −5.3% / +0.2% / −11.1% |
| random 1,024/128 at 6 req/s, async | 0.990 | −0.5% / −0.9% / — |

The profile prices GPU work only, and for GPU work it is about as accurate as
the hardware is repeatable. On the two async runs with step logs (where the
measured step is close to pure GPU time) predicted / measured is 1.012 and
0.986 for decode steps and 0.992 for mixed steps on both: within ~1%, with no
stable sign to calibrate out. Identical steps within one real run vary by
about 2.5% (std / median).

Fitting step times directly on logged steps instead (linear models on batch
size, padded tokens, KV lengths; fitted on one run, scored on another) was
worse than the profile: held-out totals between 0.56 and 1.15, because a
workload's steps cover too little of the shape space to identify the terms.
The layer profile stays.

**vLLM's CPU time.** A synchronous step also holds vLLM's own work (schedule,
process outputs): 1–3 ms longer than the profile's price at every batch size.
`--step-cpu-us` adds it, default 2,000 (the best end-to-end fit on both sync
workloads below; measured with vLLM 0.33 on an i7-12700, and it will differ on
other host CPUs). Under async scheduling vLLM hides this work behind the
previous batch, and results do not move with it.

## 3. End to end

`augur bench` against real `vllm bench serve` with the same arguments
(random dataset, 1,024-token prompts, 128-token outputs, 300 requests, seed 0,
`--ignore-eos`, async scheduling). Error of augur, KV size from each real
server:

| Rate | Throughput | Mean TTFT | Median TTFT | Mean TPOT |
| --- | --- | --- | --- | --- |
| 2 req/s (light) | +0.1% | −1.2% | −0.9% | −1.9% |
| 6 req/s (knee) | +0.8% | −11.2% | −28.1% | −1.4% |
| 12 req/s (overloaded) | +2.0% | −3.4% | −2.6% | −2.1% |

The same traces replayed against the real server, augur run normally (not
replayed), mean TTFT / TPOT / latency:

| Run | Error | Median per-call error (absolute) |
| --- | --- | --- |
| ShareGPT 300, sync | +0.2 / −0.2 / 0.0% | 0.7 / 0.9 / 0.4% |
| SWE-bench sessions (765 calls), sync | +0.4 / 0.0 / +0.2% | 7.3 / 5.9 / 4.9% |
| ShareGPT 300, async | +1.7 / +1.1 / +1.5% | 2.6 / 1.2 / 1.5% |

Without the CPU term (`--step-cpu-us 0`) the sync rows are −3.3 / −3.4 / −3.3%
and −4.4 / −2.9 / −3.8%. SWE-bench's per-call errors stay larger because
sessions compound small timing differences.

The two ShareGPT runs were made with a trace client that capped itself at 100
open connections (fixed since), so most of their calls queued in the client
first; both systems were throughput-bound, and the per-call times still agree.

**The API server and the client.** augur simulates the engine, and adds two
constants around it, measured on the 6 req/s run by lining up the client's
send times with the engine's step log (same clock):

- `--frontend-latency-ms`, default 3.4, plus `--frontend-us-per-token`,
  default 3.0: client send to the engine process (HTTP, tokenization,
  hand-off); 6.5 ms for a 1,024-token prompt. Constant to within about 1 ms for
  298 of 300 requests at 1,024 tokens. With the engine idle at light load,
  median intake was 3.5 / 5.9 / 14 ms for 256 / 1,024 / 4,096-token text
  prompts (fit: 3.0 ms + 3.0 us per token, 95% interval 2.6-3.5 us). With it,
  median TTFT at 4,096 tokens moves from −1.3% to +0.9%; at 256 tokens augur
  stays about 7 ms high for another reason.
- `--delivery-latency-ms`, default 8: engine emits a token to the client
  having it (output processing, detokenization, streaming, client parse). It
  does not accumulate: mean client inter-token latency equals the engine's.

The rest of the delay before a request is scheduled, 63 ms on average in that
run and growing with load, is not the API server: the engine takes new
requests only between steps. augur's engine loop reproduces that wait.
Replaying the real run with each request arriving at its client send time
plus a constant 6 ms, instead of at its logged engine arrival, still gives
775 of 778 steps with identical decisions (726 with no delay, 549 with 15 ms).
Without the two constants light-load TTFT reads 12% low (−11.7% mean at 2
req/s). Both were measured with vLLM 0.33 on an i7-12700 with text prompts;
other hosts will differ, and token-id prompts (the trace client) are untested.

**KV offload against real vLLM.** SWE-bench 50 sessions with a 16 GiB CPU
tier, Qwen3-1.7B, RTX 4060 Ti; augur at the measured PCIe rate (12.7 GB/s
CPU to GPU, 12.1-13.0 GB/s GPU to CPU). Error of augur:

| Run | Mean TTFT / TPOT / latency | Median per-call TTFT / TPOT / latency (absolute) | Prefix-hit rate, real vs augur |
| --- | --- | --- | --- |
| sync | −3.2 / −1.6 / −2.9% | 12.6 / 6.5 / 7.4% | 47.8% vs 48.5% |
| async | −4.0 / −1.3 / −3.2% | 12.9 / 5.7 / 6.9% | 51.5% vs 53.0% |

**Contention on one GPU** (`scripts/contention_bench.py`, RTX 4060 Ti):
copies run at 12.7 GB/s to the GPU and 13.1 GB/s from it, flat from one KV
block (1.84 MB) to 64 MB. With both directions at once, GPU-to-CPU copies lose
14-15% and CPU-to-GPU copies about 1%; two CPU-to-GPU streams split the link
evenly. A memory-bound decode-like forward is 7% slower while a 512 MB copy to
the GPU runs, 5% with a copy from it, 11-12% with both. The real offload logs
agree: stores that finished in the same step as a load ran 15% slower (10.4 vs
12.1-12.3 GB/s); loads were unaffected. On this trace transfers overlap about
5% of the run, so contention costs under 0.5% of step time here; it matters
when transfers are active most of the time, or when GPUs share a host.

**Two GPUs offloading at once** (`scripts/contention_bench.py --two-gpu`).
The test host has two RTX 4060 Ti on one PCIe host bridge with no switch: GPU0
on the CPU's root port, GPU1 behind the chipset, at about half GPU0's copy
rate (6.6 GB/s each way; its clock lock does not affect copies). With the host
otherwise idle, the two GPUs do not slow each other in any combination of
directions: each keeps its single-GPU rate within 1%, 35.8 GB/s in total with
all four directions active. The only slowdown is the single-GPU one (a
GPU-to-host copy beside a host-to-GPU copy on the same GPU loses 15-17%). With
four processes streaming host memory, they contend: both GPUs copying the same
way fall to 14 GB/s in total (GPU1 −56%, GPU0 −11 to −13%), and all four
directions to 24 GB/s (−33%). Offload on a host busy with memory-bound CPU
work, or through a chipset slot, should not be priced at the single-GPU rate.
Measured with other users' jobs on the host (load 9-15); this host only.

**Speculative decoding timing.** ngram k=3, ShareGPT 300, rerun on GPU 0
(host load 33-41): both simulators are far too fast. Mean TPOT error: augur
−32% async and −35% sync, LLMServingSim −26% and −29%. Real steps take about
2.3x the priced time, about 1 ms per request per step more. Neither simulator
prices the drafter or verification overhead; decisions still replay exactly.

**Knee TTFT** is the hardest metric. With the real engine arrivals held fixed
(`--replay-arrivals`), median TTFT is 1,111 ms with the real step times and
855 ms with the profile's, although every step class is priced within ~1%. At
the knee a request's TTFT is set by a backlog built over tens of seconds, so
a 1% step difference becomes a ~23% median TTFT difference. Two identical
real runs differed by 8% in median TTFT, and augur's own result moves by a few
points with changes that should not matter (−26.7% to −30.6% in an earlier sweep across CPU-term
values that leave every other metric alone). A point estimate here is not
meaningful, from augur or from one real run; step pricing is already at the
hardware's repeatability (section 2).

So `augur bench` reports a range: it reruns the trace with step times at −1%
and +1% (`--sensitivity`, default 1; the profile's accuracy) and prints how
far each metric moves. On this benchmark:

| Rate | Median TTFT at −1% / 0 / +1% | Spread | Real runs |
| --- | --- | --- | --- |
| 2 req/s | 123 / 124 / 125 ms | 1.8% | 125 ms |
| 6 req/s (knee) | 750 / 927 / 1,308 ms | 60% | 1,289 and 1,186 ms |
| 12 req/s | 9,547 / 9,778 / 9,902 ms | 3.6% | 10,043 ms |

A large spread marks an operating point where no single number, simulated or
measured, is reliable.

## 4. Against LLMServingSim

**Same real runs, same profile and KV size** (error vs real vLLM):

| Run | Metric | LLMServingSim | augur |
| --- | --- | --- | --- |
| ShareGPT 300, sync | mean TTFT / TPOT / latency | −3.3 / −3.4 / −3.4% | +0.2 / −0.2 / 0.0% (−3.3 / −3.4 / −3.3% without the CPU term) |
| ShareGPT 300, async | mean TTFT / TPOT / latency | +1.4 / +0.9 / +1.2% | +1.7 / +1.1 / +1.5% |
| SWE-bench sessions (765 calls), sync | mean TTFT / TPOT / latency | −4.5 / −2.1 / −3.6% | +0.4 / 0.0 / +0.2% |
| SWE-bench sessions, sync | session duration, median per-session error | 3.7% | 0.9% |
| SWE-bench sessions, async | mean TTFT / TPOT / latency | −0.4 / −0.6 / −0.4% | −0.8 / +0.1 / −0.6% |
| SWE-bench sessions, async | median per-call TTFT / TPOT / latency (absolute) | 8.7 / 8.2 / 6.1% | 9.1 / 7.0 / 6.1% |
| ShareGPT 300, ngram k=3, async | mean TPOT | −25.5% | −32.0% |
| random 2 req/s | mean TTFT / TPOT | −11.9 / −1.9% | −1.2 / −1.9% |
| random 6 req/s | mean / median TTFT, TPOT | −1.1 / −20.8%, +1.2% | −11.2 / −28.1%, −1.4% |
| random 12 req/s | mean TTFT, TPOT, throughput | −1.2%, +0.4%, +0.6% | −3.4%, −2.1%, +2.0% |

At 0.2 sessions/s the async SWE-bench engine is rarely saturated and the two
schedulers agree. Neither simulator prices ngram drafting (section 3).
On sync runs and at light load augur is closer: LLMServingSim has no term for
vLLM's CPU time, the API server or the client. On loaded async runs the two
are equally close. LLMServingSim's async runs come out
~1% slower than augur's (cause not yet known); the real 6 and 12 req/s runs
were also ~1% slower than the profile, so it lands closer there, and further
on async ShareGPT, where the real run was faster than the profile. Its
scheduler is its own reimplementation; augur's decisions are vLLM's
(section 1).

**Same simulated hardware, scheduler only** (`tests/test_vs_llmservingsim.py`,
RTX PRO 6000 profiles; `--lss-loop-compat` reproduces LLMServingSim's loop
timing, which is not vLLM behaviour; CPU, front-end and delivery terms at 0). Calls with identical TTFT, TPOT, latency and end
time, to the ns:

| Case | Identical calls |
| --- | --- |
| ShareGPT, Llama-3.1-8B, TP1 and TP2 | 300 / 300 |
| SWE-bench, Qwen3-30B-A3B MoE, TP1 and TP2 | 765 / 765 |
| ShareGPT, Qwen3-30B-A3B, DP2 + EP2 | 300 / 300 |
| ShareGPT, 2 engines, round-robin | TPOT 300 / 300; others 150 / 300 |
| ShareGPT, async scheduling | 128 / 300 TTFT |

Where they differ, LLMServingSim departs from vLLM:

- **Idle loop.** An idle LLMServingSim cluster advances in 1 ms quanta, and a
  second idle instance sees a new arrival only at ASTRA-Sim's next event (the
  2-engine offset above). vLLM admits at once.
- **Async scheduling.** When a request finishes in step k, vLLM (building k+1
  during k) cannot free its slot until k's output is processed, so a waiting
  request starts at k+2. LLMServingSim admits it at k+1.
- **PD disaggregation.** LLMServingSim's TTFT is the prefill finishing. In vLLM
  the first token comes from the decode engine after the KV pull and a
  one-token recompute: mean TTFT 152 ms sync and 207 ms async against
  LLMServingSim's 56 ms (2.7x and 3.7x; 1 prefill + 1 decode engine,
  Llama-3.1-8B, max_num_seqs 1024). augur's prefill finishes at 56.2 ms, as
  LLMServingSim's does. The async-loop fix of 2026-11-18 moved async PD by 0.5%
  mean TTFT and left sync PD identical. LLMServingSim's decode instance also
  ignores `max_num_seqs`.
- **Profile range.** LLMServingSim's session-scale branch (a4053bc) prices
  Llama-3.1-70B H100 attention from the bundle's v0 surfaces; the fork and
  augur read attention.csv from the same bundle. Inside the profiled range
  (KV up to 2,048 tokens) the two agree to 0.2% median TPOT; beyond it they
  extrapolate differently, by tens of percent per lookup. On sessions with
  contexts up to 33k, call-for-call TPOT is 18% apart (median) with identical
  scheduling. Neither extrapolation has been measured; augur now warns and
  reports `steps_beyond_profile` when a run leaves the profile's range.

## 5. Speed

Wall time simulating Llama-3.1-70B TP4 on H100s (ShareGPT 750
requests at 10 req/s, 89 s simulated):

| GPUs | augur (shipped tables) | augur (ASTRA-Sim pricing) | LLMServingSim |
| --- | --- | --- | --- |
| 16 (4 × TP4) | 10.6 s, 1.2 GB | 27 s | 8 min 06 s |
| 256 (64 × TP4) | 20 s, 1.5 GB | 44 s | 2 h 11 min |

Both simulators agree on this run (TPOT 750/750 identical at 256).
LLMServingSim runs every step through ASTRA-Sim. augur prices the collectives
of a dense model on one node itself (a closed form of ASTRA-Sim's ring
all-reduce and all-gather, identical to the binary to the ns on every tested
configuration), so these runs need no ASTRA-Sim and give byte-identical output
to the ASTRA-priced ones. At 256 GPUs about 80% of the time is vLLM's own
scheduler, KV-cache manager and output processing, 10% pricing and 13%
augur's event loop; about 6 s of the 20 s is start-up (building 64 engines).
Wall times are on a shared host (load 5-12).

ASTRA-priced runs are byte-identical to the table pricer and to each other,
including when several run at once: each augur process runs ASTRA-Sim in its
own directory. Before 2026-11-21 they shared the fork's serving/ folder through
ASTRA-Sim's tmp__mem file, which could shift concurrent runs by a few ns or
crash them. MoE (Qwen3-30B-A3B, TP1 and TP2) is priced from tables too:
byte-identical to the fork on the SWE-bench runs, 8.7 s against 14.1 s (TP1)
and 8.6 s against 19.4 s (TP2) per whole run. A DP round still costs about
25 ms through the fork (chakra graph conversion 13 ms, ASTRA-Sim 6 ms, trace
generation 4 ms).

Multi-GPU timing has not been compared with real hardware: the collective
model is ASTRA-Sim's (ring over point-to-point links), not a measurement. On the 4060 Ti
benchmarks above augur takes 1–4 s per run against 55–152 s for the real one.

## Modeled, not yet checked against real vLLM

- **KV offload to CPU** (`--kv-offload-gib`), now partly checked. vLLM's OffloadingConnector
  scheduler half runs unmodified; augur stands in for its GPU worker
  (augur/sim/offload.py). Checked against augur's own logs only: sync and
  async runs replay step for step, and runs repeat exactly. Transfer time is
  latency + bytes / bandwidth on three FIFO lanes (stores, loads, sync loads)
  that do not share bandwidth. Checked against real runs since 2026-11-21
  (section 3: exact replay, end to end within 3-4% mean TTFT);
  `scripts/real_offload_check.sh` reruns the check.
- **Secondary KV tiers** (`--kv-offload-tier`). vLLM's TieringOffloadingManager
  runs unmodified (stores cascade from the CPU tier to every secondary tier;
  hits there are promoted to the CPU tier, then loaded to the GPU). augur's
  `SimMemoryTier` stands in for the tier itself: capacity with LRU eviction,
  a write lane and a read lane, each FIFO at latency + bytes / bandwidth on
  the virtual clock. Checked against augur's own logs (exact replay, sync and
  async). Nothing real to compare with yet. `--kv-offload-tier cxl:GIB` takes
  LLMServingSim's cxl_mem defaults, 60 GB/s and 250 ns per device (the fork's
  example values, not a measured device).
- **vLLM Router policies** (`--router`). Ported from the router's source
  (src/policies, main of 2026-11-18): random, power-of-two, cache-aware
  (its default; per-engine token prefix trees), consistent-hash,
  rendezvous-hash, sticky-least-loaded. Its hash function is replaced by
  BLAKE2b (engine URLs are synthetic, so placement is statistically, not
  bit-for-bit, the same); every engine is healthy.
- **kv-aware router** (`--router kv-aware`). Scoring ported from the vLLM
  Router's kv_aware.rs (only part of its source was read): cost = (new prefill
  + active prefill) x (batch + 1), GPU blocks credited 1.0 and CPU-tier blocks
  0.8, the last token never credited. The index is fed by vLLM's own KV events
  (block pool and offload connector) when each step lands, with no transport
  delay. Ties go to the lowest engine id (the router picks at random). Block
  hashes are vLLM's own; the router's hashing was not checked. With PD only
  prefill placement uses it.
- **KV offload with PD and DP.** vLLM's MultiConnector of NIXL and the
  OffloadingConnector (NIXL first: remote KV wins, else the CPU tier). DP ranks
  run the offload worker: sync rounds commit loads when the round is priced;
  async rounds start loads at launch and ignore store-flush waits (both
  approximations). Checked by exact replay of augur's own logs (every engine,
  sync and async), not against hardware.

## Not modeled yet

- EAGLE/MTP drafter cost (ngram speculation is modeled).
- Pipeline parallelism; LMCache; storage tiers (files, object stores) as
  anything but a slower memory tier.
- API-server time as a function of prompt length (one constant now).
- MoE at TP > 2 and DP + EP on the shipped tables (they use the fork).
- Bandwidth contention between transfers (offload lanes, tiers, PD pulls on
  shared PCIe, host memory or NICs) and between transfers and compute: in
  progress (feature/contention-20261121).
