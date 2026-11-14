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

## Next

- Phase 2: several engines behind a router; TP/EP collectives and PD KV
  transfers through ASTRA-Sim; async scheduling.
