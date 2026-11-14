# augur

Runs vLLM's real control plane (scheduler, KV-cache manager, prefix caching,
chunked prefill, preemption) on a virtual clock, with no GPU and no model
weights. Step and transfer times come from LLMServingSim's hardware model.
Design doc: https://claude.ai/artifact/X669zyopvMUTTc4wJR9x5p

## Layout

| Module | Role (design doc name) |
| --- | --- |
| `augur/executor.py` | `SimExecutor`, vLLM's executor seam: KV spec and memory from the model config; each step emits recorded output tokens and asks the cost model for the step time |
| `augur/engine.py` | `SimEngine` (one unmodified `EngineCore`) and the `Coordinator` (event queue, virtual clock) |
| `augur/workload.py` | Streams the LLMServingSim fork's JSONL: flat requests and agentic sessions (SWE-bench, scaled sessions) |
| `augur/router.py` | Cluster routers (round-robin, least-loaded); vLLM Router later |
| `augur/cost.py` | Cost-model interface and a fixed-cost model for plumbing tests |
| `augur/lss_cost.py` | Step times from LLMServingSim's profiles via the fork's trace generator (TP=1, EP=1 for now) |
| `augur/compare.py` | Per-call comparison of two request CSVs (augur vs LLMServingSim) |
| `augur/metrics.py` | Per-call CSV in the fork's format, plus session and prefix-cache columns |
| `augur/models.py` | Model `config.json` taken from the fork; weight-size estimate |

## Setup (anjuna3)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
```

The LLMServingSim fork checkout defaults to `~/llmss-vllm-sim-20261114`
(`--fork` or `AUGUR_FORK` to change it).

## Run

```bash
.venv/bin/augur --model meta-llama/Llama-3.1-8B \
  --workload ~/llmss-vllm-sim-20261114/workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl \
  --output out/sharegpt.csv

.venv/bin/augur --model Qwen/Qwen3-30B-A3B-Instruct-2507 --hbm-gb 96 --max-model-len 131072 \
  --workload ~/llmss-vllm-sim-20261114/workloads/swe-bench-qwen3-30b-a3b-50-sps0.2.jsonl \
  --output out/swebench.csv
```

Profile-based timing (needs the `llmservingsim` extra: pandas, pyyaml, scipy):

```bash
.venv/bin/augur --cost llmservingsim --hardware RTXPRO6000 --hbm-gb 96 \
  --model meta-llama/Llama-3.1-8B --workload <file>.jsonl \
  --output out/run.csv --step-log out/steps.csv
.venv/bin/python -m augur.compare out/run.csv <llmservingsim>.csv
```

## Status

See `PROGRESS.md`.
