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

| Workload | TTFT mean | TPOT mean | Latency mean | Per-call latency, p50 diff | Wall: augur / LLMServingSim |
| --- | --- | --- | --- | --- | --- |
| ShareGPT, Llama-3.1-8B, 300 req | 6,517.35 vs 6,517.08 ms | 31.52 vs 31.52 | 0.00% | 0.00% | 3.2 s / 6.7 s |
| SWE-bench, Qwen3-30B-A3B, 50 sessions | 119.5 vs 122.4 ms (2.4%) | 46.75 vs 46.43 (0.7%) | 0.6% | 0.9% | 7.2 s / 33 s |
| Same, token IDs as long as `input_toks` | 96.7 vs 102.7 ms (5.8%) | 36.34 vs 36.29 (0.15%) | 0.06% | 1.0% | 6.6 s / 33 s |

A single SWE-bench session run alone gives identical decode step times (to
the ns) and identical first-call TTFT.

Known differences, all in LLMServingSim's loop rather than vLLM:

- **Idle pass.** When every instance is idle, LLMServingSim advances through
  an ASTRA-Sim "pass" before releasing a session's next call, adding up to
  1 ms of queuing. vLLM has no such delay.
- **Unexplained, open:** on SWE-bench, 103 of 765 calls queue more than 1 ms
  in LLMServingSim but are scheduled at once in augur. This is the remaining
  TTFT gap. Not yet traced.
- **Token IDs shorter than `input_toks`** (the bundled upstream SWE-bench file:
  ids are ~141 tokens short per call). The fork hashes recorded output ids as
  if they followed the recorded prompt ids directly; in a real token sequence
  the prompt's unrecorded tail sits between them, so vLLM sees fewer hits
  (88.1% vs 89.4%). With consistent ids both give 91.66%.

## Next

- Phase 2: several engines behind a router; TP/EP collectives and PD KV
  transfers through ASTRA-Sim; async scheduling.
