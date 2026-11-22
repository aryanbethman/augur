# augur

Serving research without GPUs: augur runs vLLM's real control plane
(scheduler, KV-cache manager, prefix caching, chunked prefill, preemption,
async scheduling, NIXL prefill/decode, DP/EP) unmodified on a virtual clock,
with no GPU and no model weights. Step times come from measured GPU profiles;
multi-GPU collectives and transfers from LLMServingSim's ASTRA-Sim.

## Use it like vLLM

`augur bench` takes `vllm bench serve`'s arguments and prints its report:

```bash
augur bench --model Qwen/Qwen3-1.7B --dataset-name random \
  --random-input-len 1024 --random-output-len 128 --num-prompts 300 \
  --request-rate 6 --ignore-eos --seed 0 \
  --hardware RTX4060Ti --num-gpu-blocks 5300 --max-num-seqs 128
```

Requests and arrival times are sampled with vLLM's own benchmark code (same
seed, same requests), and the report is vLLM's `calculate_metrics`. Any augur
engine flag passes through. Outputs always run to their requested length.
`--max-concurrency` is refused: augur does not model a client-side cap yet.

Unset `--max-num-seqs` and `--max-num-batched-tokens` take the values `vllm
serve` picks for the simulated GPU (`--hbm-gb`, `--hardware`), from vLLM's own
table: 256 / 2,048 below 70 GB, 1,024 / 8,192 from 70 GB (A100s stay at 256 /
2,048), 1,024 / 16,384 from 160 GB.

After the report it prints how far each metric moves when step times change by
±1% (`--sensitivity`, 0 to turn off). Near saturation TTFT can move by tens of
percent, on real hardware too; read it as a range there.

For traces (one JSON line per request or agent session; the format is described
in `augur/trace/workload.py` and is LLMServingSim's), run augur directly:

```bash
augur --hardware RTXPRO6000 --hbm-gb 96 --model meta-llama/Llama-3.1-8B \
  --workload trace.jsonl --output out/run.csv
```

Multi-GPU engines (`--tp`; link and memory settings from `--tp-link-bw`,
`--tp-link-latency-ns`, `--npu-mem-bw` or an LLMServingSim cluster config,
`--lss-cluster-config`), many engines (`--engines`, `--router`: least-loaded,
round-robin, or the vLLM Router's random, power-of-two, cache-aware,
consistent-hash, rendezvous-hash, sticky-least-loaded and kv-aware policies; kv-aware
routes by the KV blocks each engine announces in vLLM's own KV events, GPU and CPU
tier), prefill/decode
(`--prefill-engines/--decode-engines`), DP+EP MoE (`--dp`,
`--enable-expert-parallel`), async scheduling (default, as in vLLM 0.33), speculative decoding
(`--speculative-method ngram|eagle|eagle3|mtp --num-speculative-tokens k
--spec-accept-rate p`, not with `--dp`; EAGLE and MTP run vLLM's drafter path without weights,
the drafter priced from the target's per-layer profile), KV
offload to CPU memory (`--kv-offload-gib`, vLLM's `--kv-offloading-size`;
transfer cost from `--kv-offload-gbps` / `--kv-offload-latency-ns`, by default
the CPU memory's, as LLMServingSim prices KV loads), and secondary tiers behind
the CPU one through vLLM's offload tiering (`--kv-offload-tier
NAME:GIB:GBPS[:LATENCY_NS]`, repeatable; `cxl:GIB` takes LLMServingSim's CXL
numbers, 60 GB/s and 250 ns). KV offload also runs with prefill/decode (vLLM's
MultiConnector of NIXL and the OffloadingConnector) and with DP.

`--contention on` turns every KV transfer (offload, tiers, prefill/decode
pulls) into a flow through shared resources (PCIe links, the host's PCIe root
and memory, tiers, NICs) with max-min fair sharing, the PCIe duplex loss and
GPU-memory interference with the step beside it; vLLM's scheduler sees the
resulting completion times. `--fabric FILE` describes the hardware; flags set
its bandwidths.

`--step-log-vllm PREFIX` writes vLLM's step log (one file per engine when there
are several), and `--replay-durations PREFIX` replays it: single engines,
round-robin engines, prefill/decode and DP, with and without offload, must make
the same decisions step for step. When a run goes beyond the range its profile
measured (KV length, tokens per step), augur warns and counts the steps
(`steps_beyond_profile` in the summary): their prices are extrapolated.

augur pins `PYTHONHASHSEED` (0 unless set): with KV offload, vLLM's CPU-tier
eviction order follows Python's hash order. Give a real server the same value
when comparing with it.

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
    router.py          cluster routers, including the vLLM Router's policies
    nixl.py            vLLM's NIXL connector with its network handshake off
    offload.py         KV offload to CPU: the GPU-worker half of vLLM's OffloadingConnector
    tiers.py           simulated secondary KV tiers (CXL, memory pools) for vLLM's offload tiering
    models.py          model configs without weights
  cost/                step pricing (interface in __init__.py)
    tables.py          from profiles shipped with augur: dense models on one node, no fork needed
    convert.py         builds such a profile from an LLMServingSim profile
    llmservingsim.py   through the LLMServingSim fork: MoE, DP/EP
    astra.py           collectives and DP rounds through ASTRA-Sim
    profiles.py        which GPU profiles exist
  profile.py           `python -m augur.profile`: profile a GPU and model, convert, record provenance
  profiles/            shipped profiles: <hardware>/<org>--<model>/<dtype>-tp<N>.json.gz
  trace/               workloads in (fork JSONL), per-call results out (fork CSV)
  verify/              checking against real vLLM (table above)
tests/                 pytest; replay self-tests and LLMServingSim comparison need the fork
  data/llmservingsim/  LLMServingSim reference outputs for the comparison
scripts/               rebuild test data from the fork
docs/accuracy.md       current accuracy against real vLLM and LLMServingSim
augur-todo.md          open work, in priority order
```

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[test]"
.venv/bin/python -m pytest      # about 215 tests; those comparing with LLMServingSim need its fork
```

That is all for the GPUs and models augur ships profiles for
(`python -m augur.cost.profiles` lists them): `augur` and `augur bench` use
them by default (`--cost auto`), and the profile carries the model's config.
Their step prices are identical to the LLMServingSim fork's, to the nanosecond
(method and profile data: LLMServingSim, MIT licence, CASYS Lab at KAIST).
`augur bench` also needs the model's tokenizer, which vLLM fetches from
Hugging Face as usual.

With TP > 1 the shipped profiles price the step's collectives with a closed
form of ASTRA-Sim's ring all-reduce and all-gather on one node, also identical
to the fork; a setting outside the range where that form is exact (CPU-memory
access longer than half a ring step) is refused, not approximated.

Qwen3-30B-A3B (MoE) is shipped for the RTX PRO 6000 at TP1 and TP2, priced
from tables as well. DP/EP, MoE at TP > 2 and other profiles go through a checkout of the fork
(`--fork` or `AUGUR_FORK`, default `~/llmss-vllm-sim-20261114`; install the
`llmservingsim` extra): its profiles, model configs and ASTRA-Sim.
`python -m augur.cost.convert` turns one of its dense profiles into a shipped
one. `--model-config` takes any Hugging Face `config.json`, and
`--cost fixed` runs any model with a constant step time.

To add a GPU or model, run `python -m augur.profile --model M --hardware NAME
--tp N` on that GPU: it profiles with the LLMServingSim fork's profiler (KV up
to `--max-kv` tokens, `--max-batch-tokens` per step), records vLLM's memory
accounting, converts the result into `augur/profiles/` and prints what to
commit. `--dry-run` shows the commands first.
