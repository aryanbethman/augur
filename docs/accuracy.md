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
| ShareGPT 300, DP2 (two engines, GPU 0 + 1), sync | 1,872 / 1,872 and 2,145 / 2,145 |
| ShareGPT 300, DP2, async | 1,961 / 1,961 and 2,107 / 2,107 |
| ShareGPT 300, TP2 (GPU 0 + 1), sync | 2,262 / 2,262 |
| ShareGPT 300, TP2, async | 2,265 / 2,265 |
| ShareGPT 300, small MoE (8 layers, 16 experts, dummy weights), DP2 + EP, sync | 1,919 / 1,919 and 2,151 / 2,151 |
| ShareGPT 300, small MoE, DP2 + EP, async | 1,985 / 1,985 and 2,187 / 2,187 |
| ShareGPT 300, KV cut to 520 blocks (228 preemptions), sync | 31,931 / 31,931 |
| ShareGPT 300, KV cut to 520 blocks (225 preemptions), async | 31,985 / 31,985 |
| SWE-bench 50 sessions, KV cut to 1,550 blocks, sync | 39,044 / 39,044 |
| ShareGPT 300, priority scheduling (seeded priorities 0-3), sync | 3,164 / 3,164 |
| ShareGPT 300, prefix caching off, 512-token budget, sync | 3,635 / 3,635 |

The two-GPU runs use GPU 1, whose clock is locked at 510 MHz: that slows its steps, not its decisions. In vLLM 0.33
only MoE models run DP ranks in lockstep (dense DP ranks are independent engines), so the lockstep rounds, with 241
and 9 dummy forwards on the two ranks, come from a small MoE with random weights; Qwen3-30B-A3B does not fit two
16 GB GPUs. The logging scheduler records those rounds (vLLM runs a dummy forward outside the scheduler), and replay
places each call in the round the log says. Multi-GPU replays take the real run's output tokens
(`--replay-tokens`): identical prompts decoded greedily generate identical tokens, whose blocks then share prefix-cache
hits after a preemption.

Async DP has replayed exactly since 2026-11-25. Before that it differed from the first step (370 and 462 identical
steps). vLLM 0.33's DP loop (DPEngineCoreProc, batch queue of 2) does not hide scheduling behind the GPU:
- A rank schedules batch k and launches it, then waits for batch k-1. That batch's output resolves only when the
  forward after it ends.
- A rank with nothing to run runs a dummy forward.
- All ranks sync at the first iteration and every 16 iterations.

augur used to schedule the next batch at once, as single-engine async does. Now each rank iterates as that loop
does, and replay plays each rank's logged iterations. Normal async DP runs now pay the step's CPU time as sync runs
do. On a DP2 + EP2 Qwen3-30B-A3B ShareGPT run, mean TTFT is +11% (1,475 → 1,639 ms) and mean latency +3.4% against
the earlier model. Two gaps remain, neither affecting replay:
- An idle DP group sleeps in augur, where vLLM keeps running dummy rounds until the next sync.
- An empty batch's output lands about 0.35 ms later in the real logs (2 cases).

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

**KV cache size.** Decisions also depend on how many KV blocks the engine
has. Without `--num-gpu-blocks`, augur sizes the cache as vLLM 0.33 does
(GPU memory × utilization − weights − non-torch − peak activation), with the
reserve vLLM measured at start-up (median of the seven real non-speculative
runs, shipped with the 4060 Ti profile). Against vLLM's "GPU KV cache size" in
those seven runs (85,552 tokens for all of them in augur): −2.4% to +1.0%. vLLM's
own reserve moved 0.65-0.97 GiB from run to run, worth about 3.5% of the cache.
With no measured reserve, augur's estimate from the model shape gives −0.2% to
+3.3%. Weights: 3.20 GiB estimated, 3.22 GiB loaded. Speculative runs keep
more: in the four real ngram runs (k=3) peak activation is 0.13 GiB above that
of the plain runs in the same memory mode (0.63 against 0.50 GiB, 0.95 against
0.82), the sampler's logits for the draft positions. augur adds 2.4 bytes per
draft logit (max_num_seqs x k x vocabulary; 0.13 GiB here) to the measured or
estimated reserve (since 2026-11-24), which puts its ngram cache at 84,320
tokens against the real runs' 81,104-86,400. One model and one k.

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

**Decodes that share KV blocks.** The profile times decode attention with every request on its own blocks. With
prefix caching, requests in one batch read the same physical blocks, which then come from L2 rather than DRAM: on
the partly-overlapping-prompt runs of section 5 the profile overpriced decode steps by 60% at the median (tenants;
predicted / measured 1.44 overall), rising with the number of decodes and the shared fraction. Timing vLLM's paged
decode kernel on GPU 0 with private and aliased blocks (`python -m augur.profile_shared measure`; 308 shots, cold
L2 between layers, repeated within 0.2% on average) reproduces the profile with nothing shared and gives 0.44 of the
private time for 64 decodes sharing 90% of 4,096 tokens. The decode part of attention is now scaled by
min(1, (u³ + c(n)³)^(1/3)), u the distinct over total KV tokens of the step's decodes, c(n) 0.54 at 2 decodes to
0.40 at 128 (fit rmse 0.025), from the block ids vLLM's KV manager assigns; it ships as a `shared_kv` entry with the
shots in the profile, and unshared prices are unchanged. Fitted on the microbenchmark alone, it brings stepcheck's
predicted / measured to 1.007 / 0.985 / 0.998 / 0.968 on the tenants, few-shot, RAG and agent runs (from 1.435 /
1.145 / 1.136 / 1.057); ShareGPT moves by at most 0.002, SWE-bench and the benchmark run not at all. Measured on the
RTX 4060 Ti (32 MB L2) with query length 1 only; prefill chunks are not discounted. Only a simulator that knows
which blocks each request holds can price this.

**vLLM's CPU time.** A synchronous step also holds vLLM's own work (schedule,
process outputs): 1–3 ms longer than the profile's price at every batch size.
`--step-cpu-us` adds it, default 1,300 (since 2026-11-24; 2,000 before, and the
tables in this file made before that date used 2,000). Fitted on 15 sync runs
of 4 workload families: the constant left out one run at a time stays within
1,209-1,307 us, and no batch term (per request, per new or finished request,
per token, per context token) generalizes from run to run, where an offset of
0.7-2.2 ms between runs is larger than any batch dependence. Against 2,000, the
mean step-time error over the 15 runs falls from 2.1 to 1.3% (worst 5.8 →
3.4%) and the mean end-to-end error from 4.6 / 2.2 / 2.5% to 3.1 / 1.5 / 1.7%
(mean TTFT / TPOT / latency); ShareGPT 300 and SWE-bench sync, where 2,000 was
tuned, move from about −0.2 to −1.3% latency, and the 16 GiB offload run from
−3.0 to −5.4%. Measured with vLLM 0.33 on an i7-12700; other host CPUs will
differ. Under async scheduling vLLM hides this work behind the previous batch
(the next batch is scheduled at least 13 ms before the previous one ends), and
results do not move with it.

**GPUs with no profile: `--cost roofline`.** Each operation takes max(FLOPs / (peak × efficiency), bytes /
(bandwidth × efficiency)) plus a per-kernel overhead, from the model's config and the GPU's datasheet peaks
(`GPU_FACTS`); ten parameters shared by all GPUs, fitted to the shipped profiles (`python -m
augur.cost.roofline_fit fit`). It is a coarse fallback. With every profile of one GPU held out of the fit, median
step error (GPU work only) is 3-13% for decode on the RTX PRO 6000 and H100 (GEMMs only: the H100 profile's
attention and elementwise tables are not usable) but 13 / 47 / 26% for decode / prefill / mixed on the RTX 4060
Ti, whose GEMMs reach 97% of datasheet peak against 77-80% elsewhere, which no shared efficiency captures. End to
end on the real 4060 Ti runs with it held out: mean TTFT / TPOT / latency +27 / +22 / +25% (ShareGPT sync), +42 /
+29 / +37% (SWE-bench), and +268% mean TTFT at the 6 req/s knee, where slow steps grow the queue. The breakdown
puts that error in step time (decisions within 2.3%), so comparisons that turn on scheduling hold under it.

One-point calibration for a new GPU (`python -m augur.cost.roofline_fit onepoint`, since 2026-11-24): time one decode
step (32 requests at 1,024 tokens of context) and one 1,024-token prefill chunk on it; their ratios to the roofline
are the decode and prefill scales of `--step-class-scales`, mixed steps taking the geometric mean. Checked on the
shipped profiles with each GPU held out of the fit (`roofline_fit onepoint-check`), median step error over the 8
profiles goes from 7.8 / 8.5 / 11.9% to 2.7 / 5.8 / 9.6% (decode / prefill / mixed); on the RTX 4060 Ti from
13.1 / 46.8 / 26.4% to 5.1 / 2.5 / 10.0%. It does not help everywhere: mixed steps of the RTX PRO 6000's
Qwen3-30B-A3B at TP2 go from 4.3% to 16.6%, as the decode and prefill scales differ there (0.71, 0.98).
Sliding-window layers (Mistral, Qwen2 past `max_window_layers`, `sliding_attention` in `layer_types`) are priced
over the window only; a config with a window but no layer layout is refused.

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

**Speculative decoding timing** (ngram k=3, ShareGPT 300, RTX 4060 Ti,
GPU 0; acceptance from the real log: 1.25 drafts accepted per step, augur's
per-position `--spec-accept-rate` 0.62). With the shipped profile, which holds
attention only at query length 1, a verification step is priced as a
one-token decode, and augur comes out 24-29% too fast (mean TTFT / TPOT /
latency async −24 / −26 / −25%, sync −28 / −29 / −28%). The gap grows with
context and with the number of requests, not per step: FlashAttention with
1 + k queries per request costs 1.4-1.9x a one-token decode at the same
context (0.35-0.5 us per context token), while vLLM's ngram proposer takes only
0.1-0.25 ms per step on the CPU, and async does not hide the gap. Pricing
attention from a profile made with `--decode-q-lens 1,4` (a 19-minute
`augur.profile` run) brings step totals to 0.92 (async) and 0.87 (sync) of the
measured, and the runs to +1.9 / −5.8 / −1.3% (async) and −3.7 / −9.6 / −6.1%
(sync). A further 160-210 us per request per step remains (rejection sampling,
bookkeeping); `--spec-us-per-request` (default 100) adds it: +10.4 / +0.7 /
+6.3% async, +3.6 / −3.8 / +0.6% sync. Decisions replay exactly (1,760 / 1,760
and 1,653 / 1,653 steps).

The shipped profile now has attention at query lengths 1, 2 and 4 (since
2026-11-23; fork e8625e0): the shipped q=1 tables and skew fit unchanged,
plus the q=2 and q=4 attention and skew tables of the 2026-11-21 re-profile
(query lengths in between use the nearer, ties going down). Prices of steps
without speculation are identical to before. On the same two runs, with the
defaults (`--spec-us-per-request` 100), mean TTFT / TPOT / latency:

| Run | Shipped profile before (q=1) | Now (q=1,2,4) |
| --- | --- | --- |
| ngram k=3, async | −16.3 / −20.4 / −17.8% | +8.9 / −1.0 / +4.7% |
| ngram k=3, sync | −21.0 / −23.3 / −21.6% | +2.2 / −5.4 / −0.9% |

A wholly new profile from that run is no better: its other tables moved up
to 5% from the shipped ones (run-to-run drift of the GPU) and step checks on
the non-speculative runs got worse (predicted / measured 0.965 → 0.961
ShareGPT, 0.973 → 0.955 SWE-bench), so only the q=2,4 tables were taken.

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

**Held-out workloads.** Four real runs on workloads none of augur's terms were fitted on (generators in
`scripts/baselines/overlap_workloads.py`, seeded), each replayed step for step (12,952, 6,316, 10,068 and 4,354
steps), with augur's normal run against the real client, mean TTFT / TPOT / latency, and stepcheck's
predicted / measured:

| Workload | augur error | Steps priced |
| --- | --- | --- |
| ShareGPT with bursty arrivals (gamma, CV 4, ~1 req/s), async | +1.5 / +3.1 / +3.0% | 1.019 |
| 100 prompts of 12k-24k tokens, sync | +4.1 / +0.2 / +2.1% | 0.984 |
| SWE-bench 50 sessions arriving 5x faster, sync | +1.1 / +0.4 / +0.9% | 0.989 |
| Short per-user prompts (60 users, partly shared prefixes, 5 req/s), sync | +23.0 / +7.9 / +8.1% | 0.959 |

On the per-user run the GPU price is close but the fixed 2 ms CPU term overprices its short (~19 ms) decode steps;
the same workload at 10 req/s, where the engine never idles, is +6.4 / +1.9 / +2.0%. With the 1.3 ms term (since
2026-11-24) the per-user run is +20.4 / +3.2 / +3.4%. Why (2026-11-24, from its logs): vLLM's CPU time per step there
is 0.69 ms (median, steps less the profile's price; 1.4-1.9 ms on ShareGPT, SWE-bench and the long prompts), and the
engine is busy 99.7% of the time, so the term's 0.6 ms on 19 ms steps is a 3% throughput loss. At the run's own 0.69
ms TPOT and latency come to −1.0 / −0.8%, but TTFT stays +16%. No per-step form fits every run better than the
constant: with each run held out, a term per ms of GPU time gives mean / worst step-time error 1.8 / 4.5% across the
15 sync runs, and a term proportional to GPU time alone 2.5 / 6.1%, against 1.3 / 3.4% for the constant. The rest of
the TTFT gap is outside the engine. The engine takes a request only between steps (its log's add is 0.01 ms before
the next schedule), so the real time from send to add holds the wait for the step in flight: 14.9 ms mean, spread
over a step (p10 6.2, p90 22.4 ms), while delivery from the logged first token to the client is about 0 ms. augur
simulates that wait in its engine and also adds its front end (3.4 ms + 3 us per token) and 8 ms of delivery, which
was measured on an async run, where tokens leave the engine only after the next launch. Half a step (~7-10 ms of a
44 ms TTFT) is counted twice. A delivery default per scheduling mode would fix it; it needs checking on every sync
and async run first. Long-TTFT runs do not show it (it is under 1% of their TTFT). Making the bursty run replay
exactly needed a fix: after a launch that stalled the real engine's loop, replay now waits for the next logged
schedule.

**What if one setting changed?** Pairs of real runs of the same workload that differ in one setting test this held
out: augur is calibrated on run A (`python -m augur.verify.calibrate`: one GPU-time scale per step class, so each
class's total matches A; `--step-class-scales` applies it) and predicts run B. Mean absolute error over ten metrics
(mean / p50 / p99 TTFT, TPOT, latency, and throughput), for B itself and for the effect B / A, augur uncalibrated /
calibrated / BLIS given the same calibrated price:

| Change from A | Error of B | Error of the effect B / A |
| --- | --- | --- |
| ShareGPT 300: fcfs → priority | 2.2 / 1.8 / 1.8% | 2.2 / 1.7 / 1.6% |
| ShareGPT 300: KV cut to 520 blocks | 4.9 / 4.4 / 4.4% | 5.0 / 4.3 / 4.1% |
| ShareGPT 300: prefix caching off, 512-token budget | 0.8 / 0.9 / 0.9% | 0.5 / 0.6 / 0.6% |
| SWE-bench 50: KV cut to 1,550 blocks | 2.6 / 3.0 / 3.8% | 4.1 / 3.2 / 3.2% |

Large effects come out at their size: at 520 blocks mean TTFT grows 5.1x (real 4.9x) and throughput falls to 0.32x
(real 0.33x); priority scheduling's p99 TPOT grows 2.25x (real 2.23x). Calibration moves the error by about half a
point here because the profile is already within 1-3% on the base runs; what is left is the per-step CPU time at
batch sizes the base run never had. With the same price BLIS predicts the same effects, as its scheduler is exact
on these runs (section 5). The ShareGPT base was recorded with the client's old 100-connection cap; simulating it
as recorded or unthrottled differs by under 1%.

Calibration needs `--workload` when the profile prices shared KV blocks (the RTX 4060 Ti one does). Until
2026-11-24 it fitted the scales without that discount while augur's runs applied it, so the discount counted twice:
on the overlapping-prompt tenants run the decode scale came out 0.62 instead of 0.96 and step time 29% short. On the
ShareGPT and SWE-bench runs above sharing moves prices by at most 0.2%, so the table stands.

**Where the error comes from** (`python -m augur.verify.breakdown`). For each
real run, augur's error (its normal run minus the real one) splits exactly into
three parts, from three more runs of the same requests: the real log replayed
with its step durations (decisions identical); the logged schedule re-timed
with augur's step prices and CPU terms (no decisions taken); and augur's own
run with the first calls reaching the engine when they did in the real run.

- **Outside the step loop:** front end, delivery, the client, the arrival process.
- **Step time:** the re-timed schedule minus the replay.
- **Decisions:** augur's own batches, preemptions and prefix hits minus the re-timed schedule.

Every simulated run is measured on the client's side, and a session's later
calls follow augur's own loop. Mean TTFT / TPOT / latency, % of the real value:

| Run | Total | Outside | Step time | Decisions |
| --- | --- | --- | --- | --- |
| ShareGPT 300, sync | +0.4 / −0.4 / +0.1 | +0.2 / −0.2 / 0.0 | +0.3 / −0.2 / +0.1 | −0.1 / 0.0 / 0.0 |
| ShareGPT 300, async | +1.7 / +1.1 / +1.5 | −0.1 / 0.0 / 0.0 | +1.7 / +0.9 / +1.3 | +0.1 / +0.2 / +0.2 |
| ngram k=3, sync (GPU 0) | −10.2 / −14.7 / −11.9 | 0.0 / 0.0 / 0.0 | −7.3 / −7.5 / −7.4 | −2.9 / −7.1 / −4.5 |
| ngram k=3, async (GPU 0) | −4.2 / −11.1 / −7.1 | −0.2 / −0.2 / −0.2 | −4.2 / −4.3 / −4.2 | +0.1 / −6.6 / −2.7 |
| SWE-bench 50, sync | +0.1 / −0.3 / −0.2 | −0.8 / +0.4 / −0.4 | +0.2 / −0.3 / 0.0 | +0.7 / −0.4 / +0.2 |
| SWE-bench 50, async | −0.8 / +0.1 / −0.6 | −0.7 / +0.4 / −0.3 | +0.2 / −0.3 / 0.0 | −0.3 / 0.0 / −0.4 |
| SWE-bench 50, offload, sync | −3.4 / −2.3 / −3.0 | +1.2 / +0.2 / +0.9 | −2.2 / −2.0 / −2.2 | −2.3 / −0.5 / −1.7 |
| SWE-bench 50, offload, async | −2.8 / −0.2 / −2.1 | −0.5 / +0.6 / −0.1 | −1.5 / −1.5 / −1.5 | −0.8 / +0.7 / −0.4 |
| bench 1,024/128 at 6 req/s | −10.0 / −1.4 / −2.6 | −0.5 / 0.0 / −0.1 | −7.0 / −0.7 / −1.6 | −2.5 / −0.6 / −0.9 |

Without speculation, step time and decisions each stay within about 2% of the
mean. The ngram runs are the exception. Their step-time error (−4 to −7.5%)
is the price of the verification steps. With acceptance drawn from a rate
(the table above), decisions dominate their tails (median TTFT −24 / −22%, p99
TPOT −16 / −14%).

**Token-faithful speculation.** When a call's output tokens are known (recorded
in the trace, or `--replay-tokens` from a real run), the simulated model emits
them, and the drafts are those vLLM 0.33's V2 runner makes (a numpy mirror of
its two Triton n-gram kernels, checked against the kernels run in Triton's CPU
interpreter), verified greedily against the real next tokens. Over every logged
decode step of the two real ngram runs, it reproduces each request's accepted
count exactly: 88,640 / 88,640 request-steps (sync) and 88,562 / 88,562 (async);
vLLM's CPU NgramProposer, which takes the earliest of equal matches where the
kernel takes the latest, matches 89%. With it, the decisions part of the two
runs falls to +0.1 / +0.2 / +0.1% (sync) and +0.6 / +0.3 / +0.4% (async) of
mean TTFT / TPOT / latency (from +9.2 / +2.3 / +6.4% and +12.9 / +3.4 / +8.8%
with the rate), every p50/p90/p99 within ±1.5%, and the error left is step
time: mean latency −7.6% (sync) and −3.7% (async). The rate-based totals were
closer only because the two errors cancelled. Since 2026-11-24 `--spec-us-per-request` is 180 us, not 100 (the two
runs' verification steps fit 213 sync, 144 async; each fit applied to the other run leaves −5.2 / +3.1% of step
time): step time is −3.1% (sync) and +1.2% (async) of mean latency, and the totals are −2.7 / −3.5 / −3.1% and
+2.2 / +1.1 / +1.7% mean TTFT / TPOT / latency. What is left grows with context, which points at the q=4 attention
table. In the offload
runs, transfer stalls are not re-timed, so they count under decisions.
On the five runs at the end of the section 1 table (KV cut down, priority, prefix caching off), decisions stay
within ±1.3% of every mean and step time carries the error: mean latency +5.1 / +1.2% with 520 KV blocks (sync /
async), +3.4% on SWE-bench with 1,550 blocks, +2.0% with priority, 0.0% with prefix caching off and a 512-token
budget. The re-timed schedule had to wait for a request the real engine already had
in one step of three runs (14-447 ms), and in no other run. p50/p90/p99 rows
are in the tool's CSV.

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
| ShareGPT 300, ngram k=3, async (q=1 profile) | mean TPOT | −25.5% | −26.3% |
| random 2 req/s | mean TTFT / TPOT | −11.9 / −1.9% | −1.2 / −1.9% |
| random 6 req/s | mean / median TTFT, TPOT | −1.1 / −20.8%, +1.2% | −11.2 / −28.1%, −1.4% |
| random 12 req/s | mean TTFT, TPOT, throughput | −1.2%, +0.4%, +0.6% | −3.4%, −2.1%, +2.0% |

At 0.2 sessions/s the async SWE-bench engine is rarely saturated and the two
schedulers agree. Higher session rates do not separate them either. These are SWE-bench 50 runs at 0.5, 1.0 and
2.0 sessions/s, sync and async: six real runs (2026-11-25), each replayed exactly (9,995-10,591 steps). augur is
within −0.3 to +2.2% of real mean TTFT, TPOT and latency, and LLMServingSim within −1.6 to +3.4%:

| Rate | augur, sync | LLMServingSim, sync | augur, async | LLMServingSim, async |
| --- | --- | --- | --- | --- |
| 0.5 sessions/s | +0.8 / +0.2 / +0.8% | −1.2 / −0.9 / −1.0% | +1.0 / +2.2 / +1.2% | +1.7 / +3.4 / +1.8% |
| 1.0 sessions/s | +0.9 / −0.3 / +0.7% | −0.6 / −1.6 / −0.8% | +1.3 / +1.6 / +1.2% | +0.2 / +1.6 / +0.6% |
| 2.0 sessions/s | 0.0 / +0.3 / −0.1% | −0.9 / −1.3 / −1.3% | +0.8 / +1.0 / +0.8% | +0.7 / +2.1 / +1.0% |

Cells are mean TTFT / TPOT / latency, off real.
- Real vLLM is saturated from 0.5 sessions/s: mean TTFT is 24-26 s and the run lasts 870-900 s at every rate.
  With 24k-token contexts only a few requests fit in the KV cache, so the batch is set by KV, not by the rate. A
  smaller KV cache, not a higher rate, is what would stress the two schedulers.
- Preemptions (real / augur / LLMServingSim) are 12-20 / 13-20 / 15-24.
- The exception is augur at 2.0 async, with 7 against 20 real. Replay is exact there, so step time is the cause; not
  yet traced.
- Percentiles are within about ±5%, except p99 TPOT: +6 to +10% for augur, up to +10% for LLMServingSim.
- LLMServingSim takes 21-35 s a run (304 MB), augur 7-11 s. With speculation and a q=1 profile both price verification
steps as one-token decodes (section 3); the fork reads the shipped q=2,4
tables as augur does.
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
| SWE-bench, Qwen3-30B-A3B MoE, TP1, TP2 and TP4 (projected bundle) | 765 / 765 |
| ShareGPT, Qwen3-30B-A3B, DP2 + EP2 (fork or shipped tables) | 300 / 300 |
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

## 5. Against BLIS, Frontier and Vidur, with the same step times

`scripts/baselines/` gives each simulator augur's step price (the same function, checked equal to
`augur.verify.stepcheck` on all 3,402 and 11,030 steps of two runs), the same KV block count, batch limits and CPU
term, and zeroes their own CPU, front-end and host-overhead terms, so what differs is how each decides. Simulators:
BLIS (inference-sim fe42284, a Go port of vLLM 0.33's scheduler with a per-step oracle against vLLM's classes;
patched to take the price and block count from outside), Frontier (db5a85b, a discrete-event simulator mirroring
vLLM v1's scheduler) and Vidur (abae7f6, its sarathi scheduler; its vllm scheduler cannot run prompts longer than
the token budget). Frontier and Vidur have no async scheduling, speculative decoding or KV offload. BLIS refuses ngram
speculation with async scheduling, takes a mean acceptance length, and takes prefix sharing as prefix groups.

**With the real arrivals and the real step durations**, only the scheduler and memory manager differ. E2E mean
against the real engine (TTFT in brackets), first step that differs from the real log, steps identical:

| Run | augur | BLIS | Frontier | Vidur |
| --- | --- | --- | --- | --- |
| ShareGPT 300, sync | 0.0, none, 100% | +0.3 [+1], 836, 57% | 0.0 [+4], 100, 14% | +3.8 [+26], 51, 8% |
| ShareGPT 300, async | 0.0, none, 100% | +0.5 [+3], 10, 43% | n/a | n/a |
| SWE-bench 50, sync | 0.0, none, 100% | +6.3 [+9], 134, 33% | −7.5 [−15], 312, 31% | +180 [+284], 134, 14% |
| bench 1,024/128 at 6 req/s | 0.0, none, 100% | +0.9 [+8], 12, 45% | n/a | n/a |
| ngram k=3, sync | 0.0, none, 100% | +8.3 [+13], 70, 5% | n/a | n/a |

BLIS's first difference on ShareGPT comes from its input, not its scheduler. A request first scheduled at step 836
shares 1,072 prompt tokens with an earlier one, so vLLM schedules 830 tokens. BLIS takes prefix sharing as groups
with one shared length each; this request's group gets the group minimum of 416 tokens, so BLIS schedules 1,486. On
SWE-bench, step 134 is a round its session format cannot express (a prompt shorter than the previous prompt plus
output; 64 of 715 later calls). Frontier preempts early (two requests at step 100 that vLLM keeps running).

**Each simulator's own run** (normal arrivals, augur's price), E2E mean against the real client, % (TTFT in
brackets); preemptions counted from the step logs:

| Run | augur | BLIS | Frontier | Vidur | Preemptions real / augur / BLIS / Frontier / Vidur |
| --- | --- | --- | --- | --- | --- |
| ShareGPT 300, sync | +0.1 [0] | +0.1 [0] | +3.9 [+5] | +4.1 [+7] | 177 / 178 / 177 / 247 / 202 |
| ShareGPT 300, async | +1.5 [+2] | +1.2 [+2] | n/a | n/a | 191 / 184 / 171 / – / – |
| SWE-bench 50, sync | −0.2 [0] | +0.8 [+2] | +3.7 [+5] | +205 [+319] | 18 / 12 / 14 / 403 / 48 |
| SWE-bench 50, async | −0.6 [−1] | +0.3 [+1] | n/a | n/a | 18 / 23 / 16 / – / – |
| bench 1,024/128 at 6 req/s | −2.6 [−10] | −2.1 [−9] | +5.1 [+16] | +9.0 [+31] | 17 / 18 / 21 / 14 / 17 |
| ngram k=3, sync (GPU 0) | −1.0 [+2] | −0.6 [+2] | n/a | n/a | 159 / 148 / 157 / – / – |
| ngram k=3, async (GPU 0) | +4.4 [+9] | refused | n/a | n/a | 159 / 150 / – / – / – |
| SWE-bench 50, offload 16 GiB, sync | −3.3 [−4] | +10.3 [+13] | n/a | n/a | 19 / 20 / 16 / – / – |
| SWE-bench 50, offload 16 GiB, async | −3.2 [−4] | +12.2 [+15] | n/a | n/a | 14 / 16 / 16 / – / – |

Frontier and Vidur ran bench 1,024/128 (an async real run) with sync semantics. BLIS's offload rows differ by its
own transfer queueing model (the link rate is the same 12.7 GB/s). Of the real run's preempted requests, BLIS
preempts 84% on ShareGPT sync (augur 97%) and 6% on SWE-bench sync. So on these runs, which have mild memory
pressure (14-191 preemptions), a faithful port such as BLIS lands within about 1% end to end on plain runs, while
Frontier and Vidur miss by 4-205%. `scripts/baselines/run.sh RUN` reruns a row (`AUGUR_RUN_DIR` and
`AUGUR_WORKLOAD` take any other real run); the simulators' checkouts live outside the repo (`~/baselines`).

**Under memory pressure, priority and prefix caching off** (the last five runs of the section 1 table). Replay with
the real step durations: E2E mean against the real engine (TTFT in brackets), first differing step, steps
identical; then preemptions real / BLIS / Frontier / Vidur in the simulators' own runs:

| Run | BLIS | Frontier | Vidur | Preemptions |
| --- | --- | --- | --- | --- |
| ShareGPT, 520 KV blocks, sync (31,931 steps) | 0.0 [0.0], none, 100% | −0.7 [−0.8], 98, 18% | +7.7 [+8.0], 98, 6% | 228 / 218 / 319 / 696 |
| ShareGPT, 520 KV blocks, async | 0.0 [0.0], 9, 92% | n/a | n/a | 225 / 221 / – / – |
| SWE-bench 50, 1,550 KV blocks, sync | +5.1 [+4.7], 134, 31% | −9.9 [−12.0], 325, 16% | +1,011 [+1,104], 134, 3% | 25 / 29 / 580 / 128 |
| ShareGPT, priority, sync | 0.0 [0.0], none, 100% | 0.0 [−2.3], 93, 11% | n/a | 169 / 172 / 220 / – |
| ShareGPT, prefix caching off, 512-token budget | 0.0 [0.0], none, 100% | −0.1 [+0.3], 179, 19% | +4.3 [+1.7], 203, 11% | 158 / 162 / 520 / 240 |

BLIS reproduces every step of the sync, priority and prefix-off runs, all 228, 169 and 158 preemptions included:
it is a faithful copy of vLLM 0.33's scheduler. Priority needed a patch to its host (`blis_priority.patch`; as
checked out it sends every request at priority 0 and refuses a priority policy). With the cache this small,
prefix sharing nearly vanishes (0.7% hits), and with it the input that BLIS's prefix groups cannot express. Its
remaining differences are all outside the scheduler: what its inputs cannot say (prefix sharing beyond one length
per group, session prompts that are not the previous prompt plus output) and timing (async step 9: a request that
arrived 10 us before vLLM's next schedule, in the 69 us of host time between steps that BLIS does not model).
Frontier preempts 1.3-23x as often as vLLM and Vidur 1.5-5x, and neither follows vLLM's steps past 98-325.

**Prompts that overlap in part** (`scripts/baselines/overlap_workloads.py`, seeded; prompts assembled from disjoint
ShareGPT and SWE-bench token ids so that they share tokens only where the pattern says): multi-tenant system prompts
under a common preamble (400 requests), few-shot prompts drawing different example subsets (350), RAG prompts with
shared documents (350), and agent sessions that share a system prompt and trim their context in ~20% of rounds (28
sessions, 330 calls). Real runs on GPU 0 (sync, normal KV size), prefix-hit rate 0.65 / 0.38 / 0.37 / 0.68; augur
replays them exactly (1,578, 1,636, 1,887 and 5,560 steps). BLIS gets the best prefix-group encoding we could build
(`BLIS_ENCODING=best`: groups chosen by local search to minimise mis-stated tokens; it states 42-67% of prompt tokens
as shared against a true 43-69%; the simple encoding states 9-35%). With the real step durations, E2E mean against
the real engine (TTFT in brackets), steps identical:

| Workload | augur | BLIS (best encoding) | BLIS (simple) | Frontier | Vidur |
| --- | --- | --- | --- | --- | --- |
| Tenants | 0.0, 100% | +21.6 [+48.2], 4% | +45.9, 3% | −0.1 [−0.6], 67% | +93.0, 2% |
| Few-shot | 0.0, 100% | +11.0 [+22.1], 7% | +34.1, 5% | −0.1 [−0.3], 77% | +44.0, 4% |
| RAG | 0.0, 100% | +12.1 [+20.6], 3% | +23.8, 3% | −0.1 [−0.5], 65% | +48.3, 2% |
| Agent sessions | 0.0, 100% | +26.8 [+1,101], 59% | +53.8 [+2,255], 52% | +0.5 [+20.4], 85% | +104.5, 27% |

BLIS diverges at step 1-12 of every run, each time at the first request of a prefix group that shares a preamble,
task header or system prompt with another group, which one length per group cannot say. On the agent sessions it
also re-shares previous outputs the trimmed rounds dropped, and schedules 784k tokens against vLLM's 489k. Frontier,
which takes block hashes, gets the sharing right and lands within 1% end to end, but still leaves vLLM's steps at
step 156-351 (one admission or decode a step apart). Only the real scheduler on the real tokens reproduces every
step. In each simulator's own run (augur's prices, augur's arrivals) BLIS is within 0-3% of augur's E2E mean on the
three flat workloads (TTFT +5-9%) and +49% on the agent sessions. These comparisons used augur's price before the
shared-KV term of section 2, so against the real client every simulator was 15-85% slow here; with that term augur's
own runs are within 1.2-5% of real mean latency (tenants +5.0%, few-shot +1.5%, RAG +3.5%, agent sessions +1.2%).

**Picking a setting** (`scripts/baselines/sweep.py`, 2026-11-25). The sweep varies max-num-batched-tokens over
six values on the tenants workload (400 requests at 6/s). Each value has one real run on GPU 0 and a run of every
simulator. The metric, fixed before the results, is mean end-to-end latency, in seconds:

| Budget | Real | augur | augur, shared price | BLIS (best encoding) | Frontier | Vidur |
| --- | --- | --- | --- | --- | --- | --- |
| 256 | 58.9 | 66.3 | 113.7 | 115.7 | 113.8 | 112.7 |
| 512 | 52.5 | 57.1 | 102.0 | 97.1 | 103.6 | 123.8 |
| 1,024 | 44.5 | 47.1 | 83.2 | 81.6 | 84.5 | 122.8 |
| 2,048 | 40.8 | 42.9 | 75.7 | 75.7 | 76.0 | 129.5 |
| 4,096 | 40.3 | 42.2 | 72.7 | 73.2 | 73.3 | 115.0 |
| 8,192 | 39.8 | 42.1 | 72.6 | 72.7 | 73.7 | 111.0 |

"augur" is augur as a user runs it, with the shared-KV term of section 2. "Shared price" is the price every other
simulator gets, without that term.
- **Picking the best:** the real curve falls monotonically and flattens. augur, BLIS and Vidur pick the real best,
  8,192. Kendall tau is 1.0 for augur and BLIS, but 0.2 for Vidur, whose pick is luck. Frontier swaps the near-tie
  at the top (tau 0.87, regret 1.3%).
- **Absolute error:** only augur with the shared-KV term is close, at 3-13% above real. The rest are 70-90% above,
  because a simulator without vLLM's block ids cannot price decodes that share KV.
- **Limits:** a monotone setting cannot separate the schedulers. A setting whose best lies inside the range, such
  as KV size, is next. There is one real run per value, so run-to-run noise is not measured.
- **Throughput:** a 100-configuration sweep (10 budgets × 10 arrival-rate scales), 4 runs at a time on 4 cores, does
  414 configurations an hour on augur and 431 on BLIS. Both used the same Python pricer, so this compares the
  schedulers, not BLIS's native latency model.

## 6. A KV offload policy written as a vLLM plugin

vLLM 0.33's CPU offloading manager takes its eviction policy by name from the connector's extra config
(`eviction_policy`, `cache_policy_module_path`), so a policy can live outside vLLM. `examples/kv_offload_policies.py`
holds two, written once and loaded unchanged by a real `vllm serve` (`--kv-transfer-config`) and by augur
(`--kv-offload-extra-config`): ToolWaitPolicy evicts the blocks of sessions with a request in flight first (their KV
is also on the GPU) and those of sessions waiting on a tool last; ReuseFrequencyPolicy evicts the blocks that served
the fewest requests first. Against vLLM's own LRU and ARC: SWE-bench 50 sessions, Qwen3-1.7B, one RTX 4060 Ti, sync,
GPU KV pinned to 9 GiB (5,266 blocks), an 8 GiB CPU tier (`scripts/real_offload_check.sh` with `KV_GIB`,
`OFFLOAD_GIB`, `POLICY`). One real run per policy; augur replays each step for step with the plugin loaded (11,384,
11,269, 11,390 and 11,337 steps). augur's own run, mean TTFT / latency in s and prefix-hit rate, real vs augur:

| Policy | Mean TTFT | Mean latency | Prefix hits | CPU-tier hits |
| --- | --- | --- | --- | --- |
| ToolWaitPolicy (plugin) | 17.89 / 17.75 | 29.94 / 29.61 | 33.8 / 36.4% | 21.8 / 23.1% |
| ReuseFrequencyPolicy (plugin) | 18.50 / 18.83 | 30.87 / 31.23 | 27.3 / 26.8% | 14.6 / 13.8% |
| LRU (vLLM's default) | 19.71 / 19.84 | 32.52 / 32.68 | 21.6 / 20.8% | 9.3 / 8.2% |
| ARC | 19.99 / 19.92 | 32.93 / 32.75 | 19.5 / 20.2% | 7.8 / 7.7% |

augur's mean TTFT is within 2% for every policy and it ranks them as the real runs do on mean and p99 TTFT, mean
latency and both hit rates (ToolWait 9.3% ahead of LRU on real mean TTFT, ReuseFrequency 6%); on p99 latency it swaps
LRU and ARC, which the real runs separate by 0.1%. With 16, 64 and 256 engines (session-affine routing, 50 sessions
per engine, `--replicate-sessions`) augur keeps the same order and gaps (ToolWait ~9%, ReuseFrequency ~5% ahead of
LRU); at that load per engine each engine behaves like the validated one, so this shows consistency at scale, not a
check of it. Limits: one workload, model and GPU, one run per policy, and margins only a few times augur's 1-2%
error. The ranking belongs to this configuration (a CPU tier small for the working set, long tool waits). A port such
as BLIS would need each policy rewritten first.

## 7. Speed

Wall time simulating Llama-3.1-70B TP4 on H100s (ShareGPT 750
requests at 10 req/s, 89 s simulated):

| GPUs | augur (shipped tables) | augur (ASTRA-Sim pricing) | LLMServingSim |
| --- | --- | --- | --- |
| 16 (4 × TP4) | 10.6 s, 1.2 GB | 27 s | 8 min 06 s |
| 256 (64 × TP4) | 20 s, 1.5 GB | 44 s | 2 h 11 min |

Both simulators agree on this run (TPOT 750/750 identical at 256; rechecked 2026-11-24 with the per-hardware
memory and estimated reserve defaults, 21 s; it needs the run's cluster config, without which TPOT is 2x).
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
crash them. MoE (Qwen3-30B-A3B, TP1, TP2 and the projected TP4) is priced
from tables too: byte-identical to the fork on the SWE-bench runs, 8.7 s
against 14.1 s (TP1), 8.6 s against 19.4 s (TP2) and 9.0 s against 21.2 s
(TP4) per whole run. DP2 + EP2 rounds of TP1 ranks are priced from tables as
well (a closed form of ASTRA-Sim's two-rank all-gathers and reduce-scatters,
tables.py): exact on 480 random rounds over six cluster settings and on 1,000
rounds taken from fork runs. Whole runs on DP2 + EP2 write the same bytes as
through the fork: ShareGPT 300 sync and async (8.4 s against 34.8 s),
SWE-bench sessions sync (11.1 s against 159 s), and ShareGPT with KV offload
to a 16 GiB CPU tier from a 2 GiB GPU cache, sync and async (12 s against
103 s). Other DP layouts still cost about 25 ms a round through the fork
(chakra graph conversion 13 ms, ASTRA-Sim 6 ms, trace generation 4 ms); for
them the fork has no native MoE tables and falls back to its legacy EP table,
which it flags as unverified.

Multi-GPU timing has not been compared with real hardware: the collective
model is ASTRA-Sim's (ring over point-to-point links), not a measurement. On the 4060 Ti
benchmarks above augur takes 1–4 s per run against 55–152 s for the real one.

**Against BLIS** (`scripts/baselines/speed.py`; constant load per GPU, one GPU per engine, round-robin, prefix
caching off, each run pinned to two P-core threads of the i7-12700 host, 3 runs (2 at 1,024 GPUs), CPU seconds
median, startup included). augur prices Qwen3-1.7B on the 4060 Ti tables; BLIS uses its own latency model with the
nearest model in its catalog (qwen3-14b on H100), so its step counts differ by up to ~15%:

| Workload | GPUs | augur CPU s / RSS | BLIS CPU s / RSS | augur steps per CPU s | BLIS steps per CPU s |
| --- | --- | --- | --- | --- | --- |
| ShareGPT (12 requests per GPU) | 16 | 14.5 / 1.4 GB | 1.1 / 0.2 GB | 1,441 | 19,371 |
| | 256 | 77.7 / 1.9 GB | 18.4 / 3.0 GB | 4,363 | 18,772 |
| | 1,024 | 341 / 3.5 GB | 152 / 29.6 GB | 3,934 | 9,016 |
| SWE-bench (1 session per GPU) | 16 | 14.8 / 1.4 GB | 1.7 / 0.2 GB | 2,780 | 19,981 |
| | 256 | 80.0 / 1.7 GB | 25.5 / 1.5 GB | 7,200 | 18,620 |
| | 1,024 | 618 / 2.6 GB | 192 / 7.4 GB | 3,719 | 9,805 |

BLIS is 3-13x faster at 16-256 GPUs and 2.2-3.2x at 1,024, where it needs up to 8x augur's memory. augur's time goes
to vLLM's own scheduler and KV manager (~60% of the event loop at 256 GPUs on ShareGPT; the table pricer 5-15%;
on SWE-bench also building the filler prompt ids of long prompts). 1,024 GPUs take 5.7 min (ShareGPT) and 10 min
(SWE-bench) of wall time. Given augur's price through a pipe, BLIS's scheduler costs 1.2-2.2x less CPU than augur.
The pipe rows used a BLIS build with one shared pricer process whose source is not in the repo, and no longer on
the test host either (only its binary): those rows cannot be rebuilt from the repo until that patch is recovered.
The engines in those runs were lightly loaded, and start-up was half of augur's CPU at 16 GPUs. speed.py's defaults
are now the production load below. The table above rebuilds with `gen --per-gpu 12 --rate 1.0` (ShareGPT) and
`--per-gpu 1 --rate 0.1` (SWE-bench).

**At production load** (2026-11-25, one run per point on a shared host, so the ratios are rough). Each GPU runs at
the real single-GPU run's rate: ShareGPT has 100 calls per GPU at 10 req/s, SWE-bench 12 sessions per GPU at
0.2/s. Engines are busy 80-91% of simulated time in augur. At the real run's length augur's batch matches the real
one (57.0 against 58.7 requests a step); at 100 calls per GPU it is 42, because ramp-up and drain weigh more.

| Workload | GPUs | augur CPU s (start-up + run) / RSS | BLIS CPU s / RSS |
| --- | --- | --- | --- |
| ShareGPT | 16 | 6.9 + 18.3 / 1.6 GB | 3.8 / 0.6 GB |
| | 64 | 7.3 + 82 / 2.1 GB | 16 / 2.7 GB |
| | 256 | 8.6 + 380 / 4.3 GB | 100 / 20.6 GB |
| | 1,024 | 14.6 + 1,808 / 13.7 GB | did not finish: stopped at 81.7 GB after 309 s |
| SWE-bench | 16 | 7.4 + 26.8 / 1.5 GB | 6.9 / 0.5 GB |
| | 64 | 7.7 + 118.5 / 1.8 GB | 31.9 / 2.3 GB |
| | 256 | 9.0 + 537 / 3.0 GB | 178 / 10.6 GB |
| | 1,024 | 14.0 + 2,765 / 8.2 GB | 1,575 / 66.7 GB |

- **CPU:** BLIS spends 4-7x less CPU at 16-64 GPUs and 1.8x less at 1,024 on SWE-bench.
- **Memory:** BLIS's grows with the run. At 1,024 GPUs it needs 8x augur's on SWE-bench, and on ShareGPT more than
  this 125 GB host allows.

Where augur's time goes at 1,024 GPUs (pyinstrument, % of the run after start-up, ShareGPT / SWE-bench):

| Part | ShareGPT | SWE-bench |
| --- | --- | --- |
| vLLM KV manager | 18% | 12% |
| vLLM scheduler | 12% | 10% |
| vLLM output processing | 12% | 6% |
| Other vLLM request and engine-core code | 10% | 8% |
| augur executor | 17% | 11% |
| Counting shared KV blocks | 12% | 15% |
| Pricer | 6% | 19% |
| Event loop | 10% | 16% |

Counting shared KV blocks runs even with prefix caching off, where nothing can be shared. vLLM's stats and logging
are already off.

**Parallel workers** (`--workers N`, 2026-11-25):
- **How it works:**
  - Workers are forked processes, each holding every N-th engine.
  - The main process keeps the workload, router and recorder.
  - A completion can reach the router or a session no earlier than the delivery latency after it. So workers advance
    in windows bounded by that latency, or by 1 s of arrivals when the router reads neither completions nor engine
    state and no session call is in flight.
  - Routers that read engine state (least-loaded, kv-aware) stop the workers at every arrival.
- **Byte-identical to one process:**
  - ShareGPT and SWE-bench at 64 and 256 GPUs at production load (5,878-25,600 calls).
  - ShareGPT at 1,024 GPUs (61,440 calls) with 4, 8 and 16 workers.
  - Power-of-two and least-loaded at 64 GPUs.
  - A test of every router kind, sync and async, with sessions.
- **Timing:** 1,024 GPUs, ShareGPT at 10 req/s per GPU, 60 calls per GPU (1.1M steps), on 8 cores:

| Router | Workers | Wall | CPU | Memory (summed PSS) |
| --- | --- | --- | --- | --- |
| round-robin | 1 | 1,150 s | 1,151 s | 9.9 GB |
| round-robin | 4 | 288 s | 1,032 s | 13.8 GB |
| round-robin | 8 | 222 s | 1,505 s | 13.8 GB |
| round-robin | 16 | 222 s | 1,457 s | 13.9 GB |
| power-of-two | 8 | 252 s | 1,424 s | 14.2 GB |
| least-loaded | 8 | 299 s | 1,577 s | 13.9 GB |

- **Speed-up:** 5.2x at 8 workers, with no gain past 8 on 8 cores. The main process is the likely limit; it has not
  been profiled.
- **Start-up:** building 1,024 engines still takes 33 s on one core before the fork.
- **Not yet supported:** DP, prefill/decode engines, `--contention on`, replay and the LLMServingSim pricer are
  refused with workers.

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

- **EAGLE / EAGLE-3 / MTP** (`--speculative-method`). vLLM's scheduling,
  lookahead slots and KV for the draft layers are vLLM's own, driven by a
  synthetic draft config (no weights). The drafter's time is an approximation:
  k passes priced as one target decoder layer plus 2-3x o_proj, final norm,
  lm_head and sampler. Acceptance is a rate parameter. Verification attention
  is priced at query length 1 unless the profile has longer decode queries.
  Not checked against a real EAGLE head.

- **Transfer contention** (`--contention on`, off by default). Every KV
  transfer (offload stores and loads, tier writes and reads, PD pulls) is a
  flow through shared resources: each GPU's PCIe link per direction, the host's
  PCIe root, host memory, each tier and each GPU's NIC, shared max-min fairly
  (augur/sim/fabric.py); offload lanes stay FIFO streams, and completion times
  reach vLLM's connectors as before, so its scheduler reacts. One flow alone
  costs what the uncontended model charges, and a single shared link
  reproduces the PD link model to the ns. Calibrated on the RTX 4060 Ti host
  (section 3): 12.7 / 13.15 GB/s per direction, GPU-to-host 15% slower while
  host-to-GPU runs (`--d2h-duplex-penalty`), two GPUs independent on an idle
  host, host memory shared under CPU load (`--host-mem-background-gbps`;
  aggregates of 14 and 24 GB/s reproduced, the per-GPU asymmetry is not). A
  step beside transfers takes t x (1 + K x PCIe rate / GPU memory
  bandwidth), sampled at its start, with K = 1.1 for decode-only steps
  (`--hbm-interference`) and 0.1 for steps with prefill tokens
  (`--hbm-interference-prefill`), fitted on real vLLM steps (below). With
  contention off, outputs and the real-GPU replays are unchanged; with it on,
  own-log replay is exact.

  Checked against real contended runs (2026-11-23, one RTX 4060 Ti,
  `KV_GIB=4 scripts/real_offload_check.sh`): SWE-bench 50 sessions with a
  16 GiB CPU tier and the GPU KV cache cut to 4 GiB (37,440 tokens), so
  138 GB is loaded back and 341-348 GB stored per run. Transfers ran at the
  idle-host rates (12.6-12.8 GB/s loads, 12.3-13.0 stores): the link itself
  was not contended, and the fabric agrees (the same transfers with and
  without it). What the real runs do show is the GPU slowing down beside
  transfers. Replaying each run through the fabric gives every real step's
  PCIe rate; against steps with none, measured / predicted rose by
  4.4% (async) and 5.6% (sync) for decode steps beside a one-way transfer
  (8.9-11% with both directions), and by 0.1-0.8% for steps with prefill
  (`scripts/fit_hbm_interference.py`): K = 0.99 / 1.25 for decode and
  0.17 / 0.03 with prefill. The microbenchmark's 1.4 (a memory-bound GEMV
  kernel) was too high for real steps, and for compute-bound ones by far.
  End to end, mean TTFT / TPOT / latency (K fitted on these runs, so this is
  a calibration, not a held-out check):

  | Run | Contention off | On, K = 1.4 (before) | On, fitted K |
  | --- | --- | --- | --- |
  | async | −1.1 / −0.8 / −1.0% | +2.4 / −0.2 / +2.0% | −0.1 / −0.8 / −0.2% |
  | sync | −0.6 / −1.0 / −0.5% | −0.9 / −1.2 / −0.7% | −0.8 / −0.2 / −0.7% |

  Decisions: both replay exactly (24,373 / 24,373 async, 24,195 / 24,195
  sync). The sync run first differed in 4 steps: vLLM's worker submits a
  step's async loads after the forward and polls only once its CPU has
  launched the copies, so a large load (127 blocks, 233 MB) was reported in
  the step that issued it, which took 71.5 ms against ~31. Replay now reports
  such a job in its issuing step too. That CPU time is not priced in normal
  runs: steps reporting loads ran 0.7 s (sync) and 0.3 s (async) longer than
  priced over runs of 990 and 940 s. Four host
  processes streaming memory barely slow one GPU's copies (H2D −2%, D2H 0%;
  the losses of 11-56% need both GPUs copying), so a loaded-host run on one
  GPU was not made. Before 2026-11-23 `--hardware RTX4060Ti` had no GPU
  memory bandwidth, so interference was off for it unless `--hbm-bw-gbps` was
  given; it now defaults to 288 GB/s.

  What it changes, on SWE-bench sessions compressed 500x on 8 engines per
  host (Qwen3-1.7B, 16 GiB CPU tier each): with 100 GB/s of host memory and
  PCIe root, about 5% of scheduling steps differ and latency moves 0.1%; with
  host memory busy (6 GB/s free) offload transfers take 3-5x longer, half the
  steps differ and mean TTFT rises 3.2%; with prefill/decode (4+4) and KV
  pulls sharing NICs at 12.5 GB/s, 82% of steps differ and p99 TPOT rises from
  27.4 to 29.8 ms. Wall time +2 to +7%. Not checked against contended real
  runs at 8 GPUs per host (these numbers predate the fitted K).

## Not modeled yet

- Pipeline parallelism; LMCache; storage tiers (files, object stores) as
  anything but a slower memory tier.
- API-server time as a function of prompt length (one constant now).
- On the shipped tables: MoE at TP 8, and DP layouts other than DP2 + EP2 of
  TP1 ranks (they use the fork). TP4 is shipped for the projected bundle only
  (bf16-tp4proj).
