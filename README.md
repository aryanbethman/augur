# augur

Serving research without GPUs: augur runs vLLM's real control plane
(scheduler, KV-cache manager, prefix caching, chunked prefill, preemption,
async scheduling, NIXL prefill/decode, DP/EP) unmodified on a virtual clock,
with no GPU and no model weights. Step times come from measured GPU profiles;
multi-GPU collectives and transfers from LLMServingSim's ASTRA-Sim.
Design doc: https://claude.ai/artifact/X669zyopvMUTTc4wJR9x5p

## Use it like vLLM

`augur bench` takes `vllm bench serve`'s arguments and prints its report:

```bash
python -m augur.bench --model Qwen/Qwen3-1.7B --dataset-name random \
  --random-input-len 1024 --random-output-len 128 --num-prompts 300 \
  --request-rate 6 --ignore-eos --seed 0 \
  --hardware RTX4060Ti --num-gpu-blocks 5300 --max-num-seqs 128
```

Requests and arrival times are sampled with vLLM's own benchmark code (same
seed, same requests), and the report is vLLM's `calculate_metrics`. Any augur
engine flag passes through. Outputs always run to their requested length.

After the report it prints how far each metric moves when step times change by
±1% (`--sensitivity`, 0 to turn off). Near saturation TTFT can move by tens of
percent, on real hardware too; read it as a range there.

For traces (one JSON line per request or agent session; the format is described
in `augur/trace/workload.py` and is LLMServingSim's), run augur directly:

```bash
augur --hardware RTXPRO6000 --hbm-gb 96 --model meta-llama/Llama-3.1-8B \
  --workload trace.jsonl --output out/run.csv
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
| `pytest` | Replay self-tests, and the scheduler against LLMServingSim's stored outputs (`tests/test_vs_llmservingsim.py`) |

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
    tables.py          from profiles shipped with augur: dense models, TP=1, no fork needed
    convert.py         builds such a profile from an LLMServingSim profile
    llmservingsim.py   through the LLMServingSim fork: MoE, TP>1, DP
    astra.py           collectives and DP rounds through ASTRA-Sim
    profiles.py        which GPU profiles exist
  profiles/            shipped profiles: <hardware>/<org>--<model>/<dtype>-tp1.json.gz
  trace/               workloads in (fork JSONL), per-call results out (fork CSV)
  verify/              checking against real vLLM (table above)
tests/                 pytest; replay self-tests and LLMServingSim comparison need the fork
  data/llmservingsim/  LLMServingSim reference outputs for the comparison
scripts/               rebuild test data from the fork
docs/accuracy.md       current accuracy against real vLLM and LLMServingSim
```

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[test]"
.venv/bin/python -m pytest      # 26 tests; 10 more with the LLMServingSim fork
```

That is all for the GPUs and models augur ships profiles for
(`python -m augur.cost.profiles` lists them): `augur` and `augur bench` use
them by default (`--cost auto`), and the profile carries the model's config.
Their step prices are identical to the LLMServingSim fork's, to the nanosecond
(method and profile data: LLMServingSim, MIT licence, CASYS Lab at KAIST).
`augur bench` also needs the model's tokenizer, which vLLM fetches from
Hugging Face as usual.

MoE models, TP > 1, DP and other profiles go through a checkout of the fork
(`--fork` or `AUGUR_FORK`, default `~/llmss-vllm-sim-20261114`; install the
`llmservingsim` extra): its profiles, model configs and ASTRA-Sim.
`python -m augur.cost.convert` turns one of its dense TP=1 profiles into a
shipped one. `--model-config` takes any Hugging Face `config.json`, and
`--cost fixed` runs any model with a constant step time.
