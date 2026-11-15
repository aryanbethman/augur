# augur

Serving research without GPUs: augur runs vLLM's real control plane
(scheduler, KV-cache manager, prefix caching, chunked prefill, preemption,
async scheduling, NIXL prefill/decode, DP/EP) unmodified on a virtual clock,
with no GPU and no model weights. Step and transfer times come from measured
GPU profiles and LLMServingSim's ASTRA-Sim hardware model.
Design doc: https://claude.ai/artifact/X669zyopvMUTTc4wJR9x5p

## Use it like vLLM

`augur bench` takes `vllm bench serve`'s arguments and prints its report:

```bash
python -m augur.bench --model Qwen/Qwen3-1.7B --dataset-name random \
  --random-input-len 1024 --random-output-len 128 --num-prompts 300 \
  --request-rate 6 --ignore-eos --seed 0 \
  --hardware RTX4060Ti --num-gpu-blocks 5300 --max-num-seqs 128 --frontend-latency-ms 15
```

Requests and arrival times are sampled with vLLM's own benchmark code (same
seed, same requests), and the report is vLLM's `calculate_metrics`. Any augur
engine flag passes through. Outputs always run to their requested length.

For the LLMServingSim fork's traces (ShareGPT, SWE-bench agent sessions,
scaled sessions), run augur directly:

```bash
augur --cost llmservingsim --hardware RTXPRO6000 --hbm-gb 96 \
  --model meta-llama/Llama-3.1-8B --workload trace.jsonl --output out/run.csv
```

Multi-GPU engines (`--tp`, with `--lss-cluster-config` for ASTRA-Sim),
many engines (`--engines`, `--router`), prefill/decode
(`--prefill-engines/--decode-engines`), DP+EP MoE (`--dp`,
`--enable-expert-parallel`), async scheduling (default, as in vLLM 0.33).

## Check it against real vLLM

| Tool | What it does |
| --- | --- |
| `augur.vllm_hooks.LoggingScheduler` / `LoggingAsyncScheduler` | vLLM's own schedulers, logging every step (`--scheduler-cls` on `vllm serve`, `AUGUR_STEP_LOG=<path>`) |
| `augur.client` | Replays a trace against `vllm serve` with the same token-id prompts and exact output lengths |
| `augur ... --replay-durations <log>` + `python -m augur.replay compare` | Reruns a logged run with its arrival times and step durations; scheduling decisions must match step for step |
| `augur.stepcheck` | Prices every step of a real run and compares with its measured time |
| `scripts/validate.sh` | All comparisons against stored LLMServingSim outputs, plus a replay self-test |

Results so far are in `PROGRESS.md`.

## Layout

| Module | Role |
| --- | --- |
| `augur/executor.py` | `SimExecutor`: vLLM's executor seam with no GPU; KV spec from the model config; emits tokens and asks the cost model for step time |
| `augur/engine.py` | `SimEngine` (an unmodified `EngineCore`) and the `Coordinator` (event queue, virtual clock, routing, PD proxy, DP rounds) |
| `augur/lss_cost.py`, `augur/astra_oracle.py` | Step prices from LLMServingSim profiles; collectives and DP rounds through ASTRA-Sim |
| `augur/nixl_sim.py` | vLLM's NIXL connector with its network handshake off |
| `augur/workload.py`, `augur/router.py`, `augur/metrics.py`, `augur/models.py` | Traces, routers, per-call CSV, model configs |
| `augur/bench.py`, `augur/client.py`, `augur/replay.py`, `augur/stepcheck.py`, `augur/vllm_hooks.py`, `augur/compare.py` | Front end and verification tools above |

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[llmservingsim]"
```

Profiles, model configs and ASTRA-Sim come from the LLMServingSim fork
checkout (`--fork` or `AUGUR_FORK`; default `~/llmss-vllm-sim-20261114`).
`--model-config` takes any HF `config.json` instead of the fork's.
