# Progress

## M1: one engine on virtual time (2026-11-14) — done

vLLM 0.33.0 `EngineCore` runs unmodified with `SimExecutor` (passed as a class
through `distributed_executor_backend`). No GPU, no weights, no tokenizer:
the model's `config.json` comes from the fork. Requests enter as token IDs
through `EngineCore.preprocess_add_request`, which skips vLLM's input
processor and its vocabulary check, so relabeled IDs from
`replicate-sessions --token-policy isolate-copies` are accepted.

Checks, fixed step cost of 25 ms:

| Run | Result |
| --- | --- |
| ShareGPT, Llama-3.1-8B, 50 requests | 944 steps in 0.32 s wall; vLLM step CPU 0.32 ms/step |
| Same, 300 requests, KV cache cut to 563 blocks | 231 preemptions; every request emits exactly its `output_toks` |
| SWE-bench, Qwen3-30B-A3B, 50 sessions (765 calls), 96 GB GPU | 11,485 steps in 2.5 s wall; 0.16 ms/step; 88% of prompt tokens hit vLLM's prefix cache |
| SWE-bench run repeated | byte-identical CSV |

## M2: the fork's workloads — done (same commit)

Flat and agentic JSONL, streamed. A session's call `i+1` is released at call
`i`'s completion plus call `i`'s `tool_duration_ns`, as in the fork's router.
Recorded `input_tok_ids` drive vLLM's prefix cache; tokens past them get ids
unique to the request (the fork leaves that tail unhashed). The simulated model
emits the recorded `output_tok_ids`.

Not yet covered: async scheduling (vLLM 0.33's default; we run it off), the
`replicate-sessions` scaled files (the format is the same, untested at scale).

## M3: step times from LLMServingSim's profiles (2026-11-14) — done

`augur/lss_cost.py` turns vLLM's chosen batch into the fork's `Batch` and
prices it with the fork's own `generate_trace` (profile lookups per layer).
TP=1, EP=1 only: the step time is the sum of layer latencies, which is what
ASTRA-Sim computes for one NPU. Block copy is on (identical prices, about 10x
faster). Prices are memoized on the exact inputs the trace generator reads.

The forward pass is padded the way vLLM pads it: vLLM 0.33's own
`CudagraphDispatcher` (CUDA graphs on, as in vLLM's default) gives the padded
token count, and dense/MoE layers are priced at it. Without this, augur ran
more, smaller steps than LLMServingSim (which models the same padding).

## M4: comparison with LLMServingSim's own scheduler (2026-11-14) — done

Same workload, profiles, KV size (both 36,578 / 20,158 blocks), `max_num_seqs`
128, 2,048-token budget, chunked prefill and prefix caching on, synchronous
scheduling, RTX PRO 6000 profiles. LLMServingSim at commit 696cf86 (our branch
`feature/vllm-sim-20261114`), analytical network. `python -m augur.compare`:

Calls with identical TTFT, TPOT, latency and end time (to the ns), and mean
differences:

| Workload | augur default | augur `--lss-idle-quantum-ms 1` | Wall: augur / LLMServingSim |
| --- | --- | --- | --- |
| ShareGPT, Llama-3.1-8B, 300 req | **300/300 identical** | 300/300 identical | 3.2 s / 6.7 s |
| SWE-bench, Qwen3-30B-A3B, 50 sessions, token IDs as long as `input_toks` | TTFT 1.1%, TPOT 0.05%, latency 0.03% | **765/765 identical** | 6.6 s / 33 s |
| SWE-bench, bundled file (token IDs ~141 short per call) | TTFT 2.8%, TPOT 0.7%, latency 0.7% | TTFT 2.9%, TPOT 0.7% | 7.2 s / 33 s |

So once LLMServingSim's idle loop is reproduced, augur running real vLLM
reproduces LLMServingSim's own scheduler exactly on both workloads. Every
remaining difference is one of these, none of them vLLM behaviour:

- **Idle quantum (LLMServingSim).** A fully idle cluster advances in 1 ms
  quanta in ASTRA-Sim's frontend, so a session call released while the engine
  is idle starts at the next quantum after the engine went idle (at least 1 ms
  later). New workload entries jump the clock exactly. vLLM has no such delay,
  so augur's default leaves it out; `--lss-idle-quantum-ms 1` reproduces it
  for comparison only.
- **Prefix hashing when token IDs are shorter than `input_toks`.** The fork
  hashes recorded output ids as if they followed the recorded prompt ids
  directly; in a real token sequence the prompt's unrecorded tail sits between
  them, so vLLM sees fewer hits (88.1% vs 89.4%). With consistent ids both give
  91.66%. Newer generator output should not have this gap; check before use.
- **`queuing_delay` column.** LLMServingSim sets it at every prefill chunk
  until the first token, so for multi-chunk prompts it is the time to the
  *last* chunk. augur reports the time to the first schedule, as vLLM's
  queued-time metric does. TTFT and the other columns are unaffected.

How the gap was closed (TTFT was 2.4–5.8% apart):

1. **Single session:** CPU-memory reads were missing from augur's step price.
   The embedding reads token ids from CPU memory ("REMOTE"); ASTRA-Sim charges
   latency + floor(bytes / GB/s). Added (`--cpu-mem-bw`, default 256 as in the
   fork's configs). With the idle quantum on, all 20 calls became identical.
2. **Concurrency, ordering at equal times:** a call released by a completion
   with zero tool wait arrives at the instant the step ends. augur let it join
   the next step; it cannot in vLLM (the client sees the output only after the
   step, and the engine schedules again at once). augur now applies such
   caused arrivals after steps start. This is a correction to augur, not a
   comparison switch, and it refines the design doc's tie-break rule.
3. **Decode vs prefill:** a one-token final prefill chunk was priced as a
   prefill. vLLM's attention backends split a batch by query length, so it is
   a decode. Fixed; ShareGPT became exact.

## M5: many engines behind a router (2026-11-14) — done

No code changes were needed: the coordinator already ran N engines. Checked
against LLMServingSim with round-robin routing (both sides), ShareGPT 300:

| Engines | TPOT identical | TTFT mean diff | Note |
| --- | --- | --- | --- |
| 2 | 300/300 | 0.03% | engine 0 identical; engine 1 offset by a constant 35,385 ns |
| 4 | 193/300 | 0.36% | engine 0 identical; others offset by a constant from their first request |

The offsets come from LLMServingSim's loop, not vLLM: an idle instance with a
new arrival waits for its pass deadline, and the deadline appears to be seen
only when ASTRA-Sim next processes an event. Here that was 35 µs after the
arrival, which was neither a step boundary nor on a 1 ms grid. Inferred from
the code comments and data, not traced through ASTRA-Sim. augur admits at the
arrival time, as vLLM does.

Scale check: 32 engines (Llama-3.1-8B, TP1) in one process, 750 ShareGPT
requests, 163,414 steps: 24 s wall, 1.4 GB RSS, 0.14 ms of vLLM CPU per step.
The design doc's top risk (vLLM global state blocking many engines in one
process) did not show up at 32.

Not done: the fork's LOAD routing policy (augur's least-loaded is a
placeholder for the vLLM Router), and a head-to-head wall-time benchmark on
the fork's matched 16/256-NPU configs (Phase 3).

## M6: TP>1 with ASTRA-Sim collectives (2026-11-14) — done

`augur/astra_oracle.py` drives LLMServingSim's own ASTRA-Sim binary with the
fork's config builder, graph generator and controller protocol. A step costs

    sum(compute latencies) + ASTRA-Sim(the same trace, compute rows at 1 ns) - n_rows

The second term covers collectives and CPU-memory reads. It is cached on the
collective and CPU-memory rows only (LOCAL tensor sizes do not change it,
checked), so a 300-request run needs about 350 ASTRA-Sim calls of ~20 ms.
The split was checked exact against full ASTRA-Sim runs of decode, mixed and
prefill steps, dense and MoE. Two details had to match LLMServingSim: the
batch goes to the instance's first NPU before the others (they share the CPU
memory device), and a step runs from the first handover to the last NPU
finishing.

MoE weights under TP without EP: LLMServingSim shards expert weights only by
EP, so at TP2/EP1 every rank holds all experts (30.95 GiB KV per rank for
Qwen3-30B-A3B). vLLM shards experts across TP ranks in that case, which
augur's weight estimate follows. Comparisons pin `--kv-cache-gb 30.95`.

`--lss-idle-quantum-ms` became `--lss-loop-compat` (comparison only). It now
also covers what LLMServingSim's loop does at TP>1: a call released at the
idle instant is not delayed, and a zero-wait session call joins the step that
starts at its release (the other ranks' reports route and schedule again).

`scripts/validate.sh` reruns every comparison (about 1 minute):

| Case | Identical calls (TTFT, TPOT, latency, end) |
| --- | --- |
| ShareGPT, Llama-3.1-8B, TP1 | 300/300 |
| ShareGPT, 2 engines, round-robin | TPOT 300/300; 150/300 others (engine 1's 35 µs idle offset, M5) |
| ShareGPT, Llama-3.1-8B, TP2 | 300/300 |
| SWE-bench, Qwen3-30B-A3B, TP1 | 765/765 |
| SWE-bench, Qwen3-30B-A3B MoE, TP2 | 765/765 |

Wall time: TP2 ShareGPT 6.6 s vs LLMServingSim 13 s; TP2 SWE-bench 12.5 s vs 52 s.

## M7: prefill/decode disaggregation (2026-11-14) — done

vLLM's own NIXL connector runs on both engines, scheduler half as shipped
(`kv_producer` on prefill, `kv_consumer` on decode). `augur/nixl_sim.py`
subclasses it only to skip the ZeroMQ side-channel handshake, which binds a
port at engine start and collides when engines share a process. The worker
half is never built; the executor stands in for it:

- The coordinator plays vLLM's PD proxy (as in vLLM's toy proxy): prefill
  request with `max_tokens=1` and `do_remote_decode`; the `kv_transfer_params`
  the prefill engine's NIXL scheduler returns (remote block ids, engine id,
  ...) go with the full request to a decode engine.
- The decode scheduler parks it in `WAITING_FOR_REMOTE_KVS`. The executor
  sees `reqs_to_recv` in that step's connector metadata and starts a pull of
  the uncomputed blocks: `latency + bytes / bandwidth` (`--pd-link-bw`,
  `--pd-link-latency-ns`; defaults are the fork's 16 GB/s, 20 µs; no
  contention between pulls yet).
- When the pull ends, the decode engine reports `finished_recving` and the
  prefill engine `finished_sending` at their next step, as vLLM's workers do.
  vLLM then promotes the request, recomputes the last prompt token and
  samples the first output token on the decode engine. Pinned prefill blocks
  are released (checked: all blocks free at the end).
- TTFT is the decode engine's first token, which is what the client of vLLM's
  proxy sees. `prefill_done` is a new CSV column.

Against LLMServingSim's PD config (1 prefill + 1 decode, ShareGPT 300):

| | augur | LLMServingSim |
| --- | --- | --- |
| Prefill done (augur) vs TTFT (LLMServingSim), mean | 56.02 ms | 56.20 ms (p50 diff 0.30%) |
| TPOT, `max_num_seqs` 1024 | 30.62 ms | 30.64 ms (0.09%) |
| Latency, `max_num_seqs` 1024 | 19,990.6 ms | 19,912.4 ms (0.39%) |
| TTFT, `max_num_seqs` 1024 | 152.5 ms | 56.2 ms |
| TTFT, `max_num_seqs` 128 | 3,014 ms | 56.2 ms |

The TTFT gap is behaviour LLMServingSim leaves out, not an error in either:

- LLMServingSim's PD TTFT is the prefill finishing. In vLLM the client's first
  token comes from the decode engine after the pull, the decode worker
  reporting it at a step boundary, and the one-token recompute: 96 ms mean
  here (p99 145 ms), about 2.7x LLMServingSim's TTFT.
- LLMServingSim's decode instance adds requests straight to its running set,
  bypassing `max_num_seqs`. vLLM's decode scheduler respects it, so at
  `max_num_seqs` 128 requests queue on the decode engine for seconds.
- LLMServingSim sends KV per layer inside the prefill step (slightly longer
  prefill steps); vLLM's NIXL decode side pulls after the prefill.

## M8: DP + EP MoE lockstep (2026-11-14) — done

`--dp N --enable-expert-parallel`: N engines, each a real vLLM `EngineCore`
built with `data_parallel_size=N`, its own rank and expert parallelism on
(configs build in one process; vLLM's DP engine loop is what augur's
coordinator reproduces):

- **Lockstep**, as vLLM's DP engine loop runs it: a rank schedules its next
  step as soon as its own forward finishes; while any rank has work, a rank
  with nothing to schedule joins with a dummy forward (vLLM
  `execute_dummy_batch`: 1 token, uniform decode, no logits or sampler). A
  round starts once every rank has joined.
- **DP padding** is vLLM's `coordinate_batch_across_dp`, using each rank's
  own CUDA-graph dispatcher: agree on the minimum graph mode; if it keeps
  graphs, every rank pads to the largest size and re-dispatches.
- **Round time**: every rank's trace (with the round's token counts, as
  LLMServingSim's DP path generates them) runs together in one ASTRA-Sim
  run, joined by the EP all-to-alls. Each rank's outputs land at its own
  finish time: the ranks meet at every collective but differ in the work
  after the last one.
- Weight estimate: routed experts split by EP (DP x TP) when expert
  parallelism is on, as in vLLM.

Against LLMServingSim's `single_node_moe_dp_ep_instance` (Qwen3-30B-A3B,
2 ranks, TP1, EP2, KV pinned to its 56.53 GiB/rank):

| Workload | Result |
| --- | --- |
| ShareGPT 300, `--lss-loop-compat` | **300/300 identical**; 1,753 rounds, 204 dummy forwards |
| SWE-bench 50 sessions, `--lss-loop-compat` | means within 0.6% (TTFT 113.05 vs 113.79 ms, TPOT 26.33 vs 26.49 ms); not call-for-call |

Getting ShareGPT exact needed three LLMServingSim details, two of which are
real behaviour and one a loop artifact:

1. Per-rank finish times (real; was: whole round ended at the slowest rank).
2. Handover order: the last rank to join gets the round first; ranks free at
   the same instant join in descending rank order, idle ranks last. Only
   decides who waits on the shared CPU-memory device (ns).
3. Compat only: once no arrivals remain, LLMServingSim puts an instance with
   no requests to sleep, and its dummy joins each round on ASTRA-Sim's 1 ms
   poll. vLLM's idle rank runs its dummy at once.

SWE-bench sessions in DP mode hit more of LLMServingSim's polling: a busy
rank's round starts 1 ms late when the other rank is idle and joins as the
dummy (e.g. session 12907's second call). Reproducing that exactly would
only make a comparison switch match; not pursued.

Speed: each DP round is a full ASTRA-Sim run (no cache yet). ShareGPT 300:
32 s (LLMServingSim 28 s); SWE-bench 50 sessions, 11,000 rounds: 242 s
(LLMServingSim 157 s). Phase 3 needs a cheaper round price (e.g. split the
round into per-segment compute maxima plus a cached comm-only ASTRA run;
to be checked exact first).

## M9: async scheduling (2026-11-14) — done

vLLM 0.33's default, now augur's default too (`--no-async-scheduling` for
synchronous). The engine runs vLLM's own `EngineCore.step_with_batch_queue`
(batch queue of 2):

- The executor now behaves like vLLM's GPU runner: `execute_model` returns
  None and the output comes from `sample_tokens`. Each launch's duration is
  recorded.
- Each engine has a GPU stream: a launched batch starts when the stream is
  free. A call that pops the oldest batch returns at that batch's finish
  time, so the CPU is blocked until then and schedules again at that moment.
  A batch scheduled while the previous one runs sees only arrivals up to its
  schedule time, and placeholder tokens for the running step, as in vLLM.
- Async with DP is not modeled yet: `--dp` runs fall back to synchronous and
  say so in the summary.

Against LLMServingSim's `--async-scheduling` (ShareGPT 300, Llama-3.1-8B):
step-for-step identical for 531 steps (14.6 s), then they part. TTFT mean
6,575.9 vs 6,559.8 ms (0.25%), TPOT 31.53 vs 31.54, latency 0.02%. The cause
is in LLMServingSim: its async mode only delays arrivals by a step. When a
request reaches its last token in step k, vLLM's scheduler (building k+1
during k) leaves it out but cannot free its slot until k's output is
processed, so a waiting request starts at k+2. LLMServingSim frees the slot
and admits it at k+1.

What async costs, from augur (vLLM's real scheduler), vs synchronous:

| Workload | TTFT sync → async | TPOT |
| --- | --- | --- |
| ShareGPT 300, Llama-3.1-8B | 6,517 → 6,576 ms | 31.52 → 31.53 ms |
| SWE-bench 50 sessions, Qwen3-30B-A3B | 102.7 → 137.4 ms (+34%) | 36.29 → 36.15 ms |
| PD 1+1, ShareGPT, `max_num_seqs` 1024 | 152.5 → 207.8 ms (+36%) | 30.62 → 30.60 ms |

Agent follow-up calls and PD handoffs each wait one more step, because the
engine has already scheduled its next step when they arrive. Runs remain
deterministic (checked). `scripts/validate.sh` pins synchronous mode for the
exact cases and adds the async case.

## Phase 3, first results: 256 GPUs (2026-11-16)

**Oracle hang fixed.** ASTRA-Sim polls only an instance's first and last NPU
(`--start/--end-npu-ids`); the NPUs between never report. The oracle waited
on all NPUs, so at TP4 it blocked forever while ASTRA-Sim spun. It now hands
the workload to and waits on those two NPUs, as LLMServingSim's loop does.
TP2 worked before only because both its NPUs are first and last.

Also: the ASTRA-Sim start-up graph is built once per oracle;
`--no-chunked-prefill` added.

**Matched scale runs** (`baseline_scale_16/256`: Llama-3.1-70B, TP4, H100,
900 GB/s links; ShareGPT 750 requests at 10 req/s; no prefix caching, no
chunked prefill, round-robin, synchronous). The H100 profile is converted
from v0, not measured (fork commit 82506e5). One single-instance TP4 cluster
config prices every engine (they are identical).

| GPUs | augur wall | LLMServingSim wall | Simulated | Agreement |
| --- | --- | --- | --- | --- |
| 16 (4 x TP4) | 27 s (415 ASTRA calls) | 8 min 06 s (same branch, shared-template) | 89.7 s | means 0.00%; TPOT 721/750 identical; engine 0 identical (others: the M5 idle-instance offset) |
| 256 (64 x TP4) | 44 s (213 ASTRA calls, 1.5 GB) | 42–49 min (Sept. run, older branch) | 89.2 s | TPOT 0.4%, latency 0.24%, TTFT 11.6% (different code version) |

augur runs 256 GPUs faster than real time on this workload: 0.22 ms of wall
per engine step, all-in. LLMServingSim's 256-GPU run on the current branch
fails ("ASTRA closed stdout before a READY record", after 3 min); not
investigated.

The load here is light (~30 steps/s per engine). At full load (~50
steps/s per engine at 20 ms steps) 256 GPUs would be ~3,200 steps per
simulated second, ~0.7 s of wall per simulated second at the same cost per
step: still within the design doc's target (1 simulated hour in at most 1
wall hour).

### Why augur is faster than LLMServingSim

Same inputs and the same numbers, priced differently:

- LLMServingSim runs every batch through ASTRA-Sim: an event-driven
  simulation of the whole step graph (80 layers of compute nodes and
  collectives on every NPU), 173,261 times at 256 GPUs. In its September
  256-GPU run that was 2,349 s of 2,541 s of CPU (92%), even with shared
  templates removing most graph conversion.
- augur prices a step as the sum of its layer latencies (the same profile
  lookups LLMServingSim feeds ASTRA-Sim; compute rows run serially, so the
  sum is what ASTRA-Sim would compute) plus ASTRA-Sim's time for the step's
  collectives and CPU-memory reads, cached on that pattern. Checked exact
  against full ASTRA-Sim runs (M6). At 256 GPUs: 33,036 distinct step
  shapes among 173,261 steps priced in Python, and 213 ASTRA-Sim runs.
- vLLM's scheduler itself costs ~0.1 ms per step.

Under cProfile (inflates 44 s to 127 s): fork trace generation 33.6 s, the
ASTRA cache key 19.6 s, vLLM scheduler 12.9 s, ASTRA-Sim runs 7.3 s. The
cache key walks every row of every step; keying on the step's collective
sizes directly would remove most of that.

### LLMServingSim's 256-GPU crash on the current branch, fixed

`template binding refers to unknown template` (ASTRA-Sim abort), after 3
minutes. ASTRA-Sim parses only the asking NPU's bindings of a
shared-template bundle, then evicts unreferenced templates and reports
`TEMPLATE_RELEASE`. LLMServingSim re-sends the same stored batch graph when
the instance's other polled NPU asks, so a template evicted in between was
referenced without its definition. At 16 GPUs the 128-entry cache rarely
evicts; at 256 it evicts constantly. Fix (fork commit 7da69d6): the
controller keeps every template definition it sent and re-attaches any a
bundle references that ASTRA-Sim no longer holds. `baseline_scale_16` with
an 8-entry cache now finishes and is byte-identical to the 128-entry run.

## Verification tooling: does augur run vLLM properly? (2026-11-16)

- `augur.vllm_hooks.LoggingScheduler` / `LoggingAsyncScheduler`: subclasses of
  vLLM's own schedulers that only log (plug in with `scheduler_cls`; set
  `AUGUR_STEP_LOG`). Same log format from real `vllm serve` and from augur
  (`--step-log-vllm`): request adds, each step's decisions (tokens per
  request, prefix hits, preemptions), each update's outputs.
- `augur.replay`: rerun a logged run through augur with the logged arrival
  times and step durations (`--replay-durations`), then
  `python -m augur.replay compare A B` checks the decisions step for step.
  With timing fixed from the log, any mismatch is a bug in what augur wraps
  around vLLM, not in the timing model.
- `augur.client`: replays a trace against `vllm serve` with the same token-id
  prompts, exact output lengths and request ids `augur.<call_id>.`; records
  per-call times and the generated token ids. `--no-recorded-outputs` makes
  augur emit unique output ids, as a real model generating its own text does.

Self-test (augur's own log replayed through augur), now in
`scripts/validate.sh`: ShareGPT 2,406/2,406 steps identical, SWE-bench
sessions 8,924/8,924, ShareGPT with 228 preemptions 28,652/28,652. Needed:
an add logged after a step that started at the same instant must not join
that step (exact ties only occur in augur's virtual time).

**Against real vLLM on a GPU** (vLLM 0.33 `vllm serve`, Qwen3-1.7B, one RTX
4060 Ti, synchronous scheduling, `max_num_seqs` 128, 2,048-token budget,
prefix caching on; `augur.client` replaying the trace; augur replaying the
server's step log with `--num-gpu-blocks` from the server log,
`--no-recorded-outputs` and `--filler-vocab 150000`):

| Workload | Steps with identical decisions |
| --- | --- |
| ShareGPT, 20 requests | 1,143 / 1,143 |
| ShareGPT, 300 requests at 10 req/s (overloaded; preemptions) | 3,402 / 3,402 |
| SWE-bench, 50 sessions / 765 calls, 24k context | 11,030 / 11,030 |
| ShareGPT, 300 requests, **async scheduling** (vLLM's default) | 3,250 / 3,250 |

What it took:

- **Real bug found:** with vLLM 0.33's default v2 model runner, a preempted
  request returns as a *new* request whose `prefill_token_ids` hold its
  prompt plus the outputs it already generated. augur counted only the
  prompt and sampled one step early after every resume. augur's own
  self-tests could not see this (both sides shared the bug).
- Replay starts each step at its logged start (real vLLM's loop wakes ~50 µs
  after an arrival), and takes the KV block count from each run's server log
  (it varies run to run: 5,294 / 5,334 / 5,480 blocks).
- Async: the step logger tags each update with the step of the scheduler
  output it belongs to (updates trail schedules by one), and replay gives a
  batch the GPU time from when it could start (its schedule, or the previous
  batch finishing) to its update.
- Unrecorded prompt tokens (the bundled SWE-bench file's ids are short) get
  in-vocabulary ids seeded by call id (`--filler-vocab`); augur's default
  out-of-vocabulary fillers made real vLLM reject every call.

This shows augur makes vLLM's decisions when step times are right.

## Step-time accuracy against real vLLM (2026-11-16)

Profile: Qwen3-1.7B TP1 on the RTX 4060 Ti, fork profiler with vLLM 0.33
(fork commit 4e1e890; the profiler needed a fix: 0.33 passes attention
metadata to `build()` positionally). No skew sweep yet. `augur.stepcheck`
rebuilds every real step's shape from the step log, pads it with vLLM's
CUDA-graph dispatcher, prices it, and compares with the measured step
(schedule start to update end, synchronous runs):

| Real run | Steps | Predicted / measured total | Median step error: decode / mixed / prefill |
| --- | --- | --- | --- |
| ShareGPT 300 | 3,402 | 0.951 | −4.6% / −4.1% / −11.8% |
| SWE-bench sessions | 11,030 | 0.950 | −7.0% / −1.9% / −11.1% |

augur is a little fast: the profile prices GPU work, and the measured step
also holds vLLM's CPU work. ShareGPT's gap is ~1.5–3 ms per step at any
batch size (CPU work); SWE-bench's grows with batch size and long, uneven KV
lengths (likely the attention skew correction the sweep would add).
`--step-cpu-us` / `--step-cpu-per-request-us` model the CPU part (sync: added
to the step; async: delays the batch's launch, hiding it behind the previous
GPU batch); not calibrated by default.

## End to end: `augur bench` vs real `vllm bench serve` (2026-11-16)

Same arguments on both sides (vLLM random dataset, 1,024-token prompts,
128-token outputs, 300 requests, seed 0, `--ignore-eos`; vLLM 0.33 defaults,
so async scheduling; `max_num_seqs` 128, 2,048-token budget). augur uses the
4060 Ti profile and each real server's KV block count. Error of augur:

| Rate | Throughput | Mean TTFT | Median TTFT | Mean TPOT | augur wall / real |
| --- | --- | --- | --- | --- | --- |
| 2 req/s | +0.1% | −11.5% → **−0.6%** | −11.8% → **+0.2%** | −1.8% | 3.6 s / 152 s |
| 6 req/s (knee) | +0.6% | −10.9% → −10.1% | −26.6% → −25.5% | −1.0% | 1.1 s / 58 s |
| 12 req/s (overloaded) | +1.9% | −3.3% | −2.6% | −2.0% | 1.1 s / 55 s |

(→ with `--frontend-latency-ms 15`.) Calibration over a grid:

- `--frontend-latency-ms` (new): API-server time before the engine sees a
  request (HTTP, tokenizing a 1,024-token text prompt, hand-off). 15 ms
  closes the light-load TTFT gap on this host.
- `--step-cpu-us` barely matters under async scheduling (0–3 ms per step
  moves no metric by more than ~1%): vLLM hides its CPU work behind the
  GPU. It matters for synchronous runs (the 1.5–3 ms per step above).
- Left: augur's GPU time is ~2% low (throughput +1.9% when saturated), which
  queueing turns into the TTFT error at the knee. The profile's skew sweep
  (attention with uneven KV lengths) is the open item.

**Synchronous runs**, augur (profile + `--step-cpu-us 3000`) vs the real
synchronous runs' client-side results, run normally (not replayed):
ShareGPT 300: mean TTFT +0.4%, TPOT +0.1%, latency +0.2% (per-request median
errors under 0.7%; without the CPU term all were ~−4.9%). SWE-bench sessions:
means within 1.6–2.9% (from 5.9–7.8%), per-call medians 7–9% (sessions
compound small timing differences).

**PD contention** (`--pd-contention decode`, new): concurrent KV pulls into a
decode engine share its link (processor sharing; each pays the link latency
first). 2 prefill + 1 decode at 2 GB/s: mean TTFT 231 → 284 ms. PD also runs
with M x N engines (2+2, 1+3 checked: all calls complete, all blocks freed).

**Speculative decoding** (`--num-speculative-tokens k`, ngram method,
synchronous): vLLM's own scheduler schedules and verifies drafts; augur's
executor accepts each draft with `--spec-accept-rate` until the first
rejection and proposes the request's next recorded tokens as drafts
(`take_draft_token_ids`, called through `EngineCore.post_step` as vLLM's busy
loop does). Llama-3.1-8B ShareGPT, k=3: TPOT 31.5 ms without; 41.4 / 19.0 /
10.8 ms at 0 / 70 / 90% acceptance; output lengths exact. In replay,
acceptance and draft counts come from the log (self-test 1,040/1,040), so a
real speculative run can be checked step for step. Speculation also runs
under async scheduling (k=3 at 70%: TPOT 18.99 ms as in sync, TTFT 956 vs
882 ms). Not yet: EAGLE/MTP drafter cost.

**DP + EP under async scheduling** (vLLM's default): each rank runs vLLM's
`step_with_batch_queue`; rounds stay lockstep across ranks (an idle rank joins
with a dummy batch) and are priced jointly. Qwen3-30B-A3B ShareGPT, DP2/EP2:
TTFT 1,458 vs 1,412 ms sync, TPOT 50.06 vs 50.23 ms, same 1,753 rounds.

**KV size without a real server:** `--kv-reserve-gib` subtracts what vLLM keeps
besides weights (activations, non-torch, CUDA-graph pool; ~0.9 GiB for
Qwen3-1.7B on the 4060 Ti) and lands within 2.5% of real vLLM's KV size.

**256 GPUs, same branch** (LLMServingSim with the template fix): augur and
LLMServingSim agree (TPOT 750/750 identical; mean TTFT and latency equal),
augur 44 s vs LLMServingSim 2 h 11 min. That LLMServingSim run peaked at
51.6 GB RSS (Sept: 560 MiB); the fix's template store is now an LRU (fork),
re-measurement pending.

**Hardware note:** GPU 1 on anjuna3 runs at a locked 510 MHz (max 3,105) with
no throttle reason: 8.6 vs 44.0 TFLOP/s on the same GEMM. A first profile
made there priced steps at 2.9–3.6x. Only GPU 0 is usable for timing.

## Next

- Phase 3: a loaded 256-GPU run to confirm the speed target; DP rounds
  (one full ASTRA-Sim run each) are the slow path; async with DP.
- PD: pull contention (ASTRA-Sim congestion-aware), multiple P/D engines.
