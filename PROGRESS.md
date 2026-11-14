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

## Next

- M3: step times from LLMServingSim's profiles in place of the fixed cost.
- M4: compare against LLMServingSim's own scheduler on the same workloads.
