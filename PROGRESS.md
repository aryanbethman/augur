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

## Next

- M8: DP+EP MoE lockstep. M9: async scheduling.
- PD: pull contention (ASTRA-Sim congestion-aware), multiple P/D engines.
