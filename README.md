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
| `augur.verify.hooks.LoggingScheduler` / `LoggingAsyncScheduler` | vLLM's own schedulers, logging every step (`--scheduler-cls` on `vllm serve`, `AUGUR_STEP_LOG=<path>`) |
| `augur.verify.client` | Replays a trace against `vllm serve` with the same token-id prompts and exact output lengths |
| `augur ... --replay-durations <log>` + `python -m augur.verify.replay compare` | Reruns a logged run with its arrival times and step durations; scheduling decisions must match step for step |
| `augur.verify.stepcheck` | Prices every step of a real run and compares with its measured time |
| `scripts/validate.sh` | All comparisons against stored LLMServingSim outputs, plus a replay self-test |

How close it gets, and how that was measured: [docs/accuracy.md](docs/accuracy.md).

## Layout

```
augur/
  cli.py, bench.py     the `augur` and `augur bench` commands
  sim/                 the simulator
    coordinator.py     virtual clock, event queue, routing, PD proxy, DP rounds
    engine.py          SimEngine: an unmodified vLLM EngineCore
    executor.py        SimExecutor: vLLM's executor seam with no GPU; asks the cost model for step time
    router.py          cluster routers
    nixl.py            vLLM's NIXL connector with its network handshake off
    models.py          model configs without weights
  cost/                step pricing (interface in __init__.py)
    llmservingsim.py   step prices from LLMServingSim profiles
    astra.py           collectives and DP rounds through ASTRA-Sim
    profiles.py        which GPU profiles exist
  trace/               workloads in (fork JSONL), per-call results out (fork CSV)
  verify/              checking against real vLLM (table above)
tests/                 pytest; the replay self-tests need the fork
scripts/validate.sh    scheduler comparison against stored LLMServingSim outputs
docs/accuracy.md       current accuracy against real vLLM and LLMServingSim
```

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[llmservingsim]"
.venv/bin/python -m pytest      # tests
```

Profiles, model configs and ASTRA-Sim come from the LLMServingSim fork
checkout (`--fork` or `AUGUR_FORK`; default `~/llmss-vllm-sim-20261114`).
`python -m augur.cost.profiles` lists the GPU / model / TP profiles it holds.
`--model-config` takes any HF `config.json` instead of the fork's.
