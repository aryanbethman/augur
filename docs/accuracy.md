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

Replay takes the KV block count from the server log (it varies run to run)
and uses `--no-recorded-outputs`; runs from `augur.verify.client` add
`--filler-vocab 150000` so unrecorded prompt tokens get in-vocabulary ids.
With timing fixed from the log, any mismatch is a bug in what augur wraps
around vLLM (executor, coordinator), not in the timing model. Every
mismatch found so far was such a bug, now fixed (a resumed preempted
request's prefill length; async log ordering; ngram drafts near
`max_tokens`). `tests/test_replay_selftest.py` replays augur's own logs for
sync, async and speculative runs.

## 2. Step times: the profile

Profile: Qwen3-1.7B TP1, RTX 4060 Ti, with the attention skew sweep (fork
4bbee2f). `python -m augur.verify.stepcheck` rebuilds every real step's batch
from the step log, pads it with vLLM's CUDA-graph dispatcher, prices it and
compares with the measured step:

| Real run | Predicted / measured, all steps | Median step error: decode / mixed / prefill |
| --- | --- | --- |
| ShareGPT 300, sync | 0.965 | −2.8% / −3.7% / −11.8% |
| SWE-bench sessions, sync | 0.973 | −5.3% / +0.2% / −11.1% |
| random 1,024/128 at 6 req/s, async | 0.990 | −0.5% / −0.9% / — |

The profile prices GPU work only. A synchronous step also holds vLLM's CPU
work, about 1.5–3 ms per step here; `--step-cpu-us` adds it. Under async
scheduling vLLM hides that CPU work behind the previous batch, so it barely
matters there.

## 3. End to end

`augur bench` against real `vllm bench serve` with the same arguments
(random dataset, 1,024-token prompts, 128-token outputs, 300 requests, seed 0,
`--ignore-eos`, async scheduling). Error of augur, KV size from each real
server:

| Rate | Throughput | Mean TTFT | Median TTFT | Mean TPOT |
| --- | --- | --- | --- | --- |
| 2 req/s (light) | +0.1% | −11.8% (−1.0% with `--frontend-latency-ms 15`) | −12.2% (−0.2%) | −1.9% |
| 6 req/s (knee) | +0.7% | −10.8% | −26.7% | −1.1% |
| 12 req/s (overloaded) | +2.0% | −3.6% | −2.8% | −2.1% |

ShareGPT 300, the same trace replayed against the real server, augur run
normally (not replayed): sync, mean TTFT / TPOT / latency −3.3 / −3.4 / −3.3%
(+2.1 / +1.4 / +1.8% with `--step-cpu-us 3000`, which now overshoots;
~1.5–2 ms fits); async +1.7 / +1.1 / +1.4%. Per-request median errors
1–3%. SWE-bench sessions, sync: means within 1.6–2.9%, per-call medians 7–9%
(sessions compound small timing differences).

**Light load TTFT** is mostly the API server (HTTP, tokenizing the prompt),
which augur does not simulate; `--frontend-latency-ms` adds a fixed delay.
The real delay grows with load (about 29 ms early in the 6 req/s run, 60–80
ms later).

**Knee TTFT** is the hardest metric. With the real engine arrivals held fixed
(`--replay-arrivals`), median TTFT is 1,111 ms with the real step times and
855 ms with the profile's, although every step class is priced within ~1%. At
the knee a request's TTFT is set by a backlog built over tens of seconds, so
a 1% step bias becomes a ~23% median TTFT bias. Two identical real runs
differed by 8% in median TTFT. Getting this right needs sub-1% step pricing.

## 4. Against LLMServingSim

**Same real runs, same profile and KV size** (error vs real vLLM):

| Run | Metric | LLMServingSim | augur |
| --- | --- | --- | --- |
| ShareGPT 300, sync | mean TTFT / TPOT / latency | −3.3 / −3.4 / −3.4% | −3.3 / −3.4 / −3.3% |
| ShareGPT 300, async | mean TTFT / TPOT / latency | +1.4 / +0.9 / +1.2% | +1.7 / +1.1 / +1.4% |
| random 2 req/s | mean TTFT / TPOT | −11.9 / −1.9% | −11.8 / −1.9% |
| random 6 req/s | mean / median TTFT, TPOT | −1.1 / −20.8%, +1.2% | −10.8 / −26.7%, −1.1% |
| random 12 req/s | mean TTFT, TPOT, throughput | −1.2%, +0.4%, +0.6% | −3.6%, −2.1%, +2.0% |

On aggregates the two are equally close. LLMServingSim's async runs come out
~1% slower than augur's, which offsets the profile's ~1% low bias at 6–12
req/s (cause not yet known). Its scheduler is its own reimplementation;
augur's decisions are vLLM's (section 1).

**Same simulated hardware, scheduler only** (`scripts/validate.sh`, RTX PRO
6000 profiles; `--lss-loop-compat` reproduces LLMServingSim's loop timing,
which is not vLLM behaviour). Calls with identical TTFT, TPOT, latency and end
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
  one-token recompute (2.7x longer here). LLMServingSim's decode instance also
  ignores `max_num_seqs`.

## 5. Speed

Wall time simulating Llama-3.1-70B TP4 on H100s (ShareGPT 750
requests at 10 req/s, 89 s simulated):

| GPUs | augur | LLMServingSim |
| --- | --- | --- |
| 16 (4 × TP4) | 27 s | 8 min 06 s |
| 256 (64 × TP4) | 44 s | 2 h 11 min |

Both agree on this run (TPOT 750/750 identical at 256). augur prices a step as
the sum of profiled layer latencies plus ASTRA-Sim's time for the step's
collectives, cached on that pattern (checked exact against full ASTRA-Sim
runs); LLMServingSim runs every step through ASTRA-Sim. On the 4060 Ti
benchmarks above augur takes 1–4 s per run against 55–152 s for the real one.

## Not modeled yet

- EAGLE/MTP drafter cost (ngram speculation is modeled).
- Pipeline parallelism; KV offload connectors (LMCache, CPU offload).
- vLLM Router policies (augur has round-robin and least-loaded).
- Load-dependent API-server delay.
- DP rounds are one full ASTRA-Sim run each (slow at large DP).
