# augur to-do

End goal: augur replaces running vLLM on GPUs for researchers without
hardware, so they can do serving research faster and cheaper without running
inference experiments. Every item is judged by that.

Order: top to bottom. Status: [ ] open, [~] in progress, [x] done.

## 0. Prove augur runs vLLM properly (top priority)
- [x] Logging scheduler (augur.vllm_hooks), plugged in via `scheduler_cls`
- [x] Replay (augur.replay); self-test exact on ShareGPT, SWE-bench, heavy
      preemption; in scripts/validate.sh (commit 6f19bc2)
- [x] Trace client for `vllm serve` (augur.client) — untested against a real server
- [x] On GPU, synchronous: ShareGPT 3,402/3,402 and SWE-bench 11,030/11,030 steps
      identical (commit b9d3878); found + fixed the resumed-request bug
- [x] On GPU, async scheduling (vLLM's default): 3,250/3,250 steps (f03a7f5)

## 1. Timing accuracy against real vLLM on GPUs
- [x] Profile Qwen3-1.7B on the RTX 4060 Ti (GPU 0; fork 4e1e890). GPU 1 of the
      test host is clock-locked at 510 MHz (needs root to fix)
- [x] Skew sweep (fork 4bbee2f): step ratio 0.951→0.965 ShareGPT, 0.950→0.973 SWE; e2e unchanged
- [x] Knee TTFT diagnosed (d954e4a): real run replays 778/778; with real arrivals, 1% step bias -> 23% median TTFT (backlog). Real run-to-run noise 8%
- [x] Sub-1% step pricing: not reachable (see section 6); the profile's sign flips between runs
- [x] Load-dependent front-end delay: it is the engine loop's wait, already simulated (section 6)
- [x] Per-step (augur.stepcheck): predicted/measured 0.95 on ShareGPT and SWE-bench
- [~] End to end: real `vllm bench serve` vs `augur bench`, random 1024/128, 2/6/12 req/s

## 2. CPU overhead in virtual time
- [x] `--step-cpu-us` (sync 3 ms: ShareGPT e2e within 0.5%); async hides it;
      `--frontend-latency-ms` (15 ms closes light-load TTFT)

## 3. Use it like vLLM
- [x] `augur bench`: `vllm bench serve` args and report, on augur (c8dbd75)
- [~] Packaging (bench runs now replayable: c9e57e9): `python -m augur.profiles` + clear missing-profile error done; fork/ASTRA setup still manual

## 4. Feature coverage
- [x] Speculative decoding: ngram, sync, acceptance rate; replay from logs (f14aa07, d16fcb0)
- [x] Speculative decoding with async scheduling (runs; not checked against real async yet)
- [x] Real-GPU replay of an ngram run (sync): 1,880/1,880 after drafter fix (e0f0a18)
- [x] Real-GPU replay of an async ngram run: 1,722/1,722 (818a204)
- [ ] Speculative decoding: EAGLE/MTP drafter cost
- [ ] KV offload connectors (LMCache, CPU offload) — memory tiers (paper 2)
- [ ] Pipeline parallelism
- [x] Async scheduling with DP (68f24df)
- [x] PD M x N engines; shared-link KV pull contention (55da69c)
- [ ] Real vLLM Router policies

## 5. Profiles
- [ ] Profiling pipeline so new models / hardware can be added by whoever has a GPU

## 6. Standalone, and closer to hardware than LLMServingSim (proposed 2026-11-16, order = priority)
- [x] Step-level profiler: TESTED AND DROPPED (2026-11-16, out/stepfit/). Held out, direct fits are worse than
      the layer profile (totals 0.56-1.15 vs 0.94-1.01). Async GPU pricing is already at the run-to-run floor:
      decode 0.986/1.012, mixed 0.992/0.992 on two runs; identical steps vary 2.5% within a run
- [x] Sync gap is vLLM's CPU time: --step-cpu-us now defaults to the measured 2 ms. Sync ShareGPT
      -3.3% -> +0.2% TTFT, SWE-bench -4.4% -> +0.5%; async unchanged (sweep in out/cpu_sweep/)
- [ ] CPU term is one host's value (i7-12700, vLLM 0.33): needs a second host to know how it varies
- [x] Knee metrics as a range: augur bench --sensitivity (default +-1%); 6 req/s median TTFT 735-1,293 ms
      covers both real runs (1,289 and 1,186)
- [x] Own profile format + pricer (c980fcb): augur/cost/tables.py + shipped profiles (55 KB each) for
      Qwen3-1.7B/4060Ti and Llama-3.1-8B/RTXPRO6000; identical to the fork to the ns (whole runs byte-identical);
      augur bench runs with no fork. --cost tables / auto
- [x] Standalone for shipped profiles (1aa37c2): augur defaults to --cost auto; model config from the profile;
      26 of 36 tests run with no fork (the other 10 compare against it); offline-built wheel runs the benchmark
- [ ] Standalone, remaining: the wheel was run unpacked, not pip-installed into a clean venv (needs a vLLM
      download); augur bench needs the model's tokenizer from Hugging Face; scratch scripts that
      relied on the old --cost fixed default
- [ ] Tables for MoE (needs the 3-D expert surface; scipy Delaunay in the fork) and more shipped profiles
- [x] TP>1 without the ASTRA binary (afa6a3b): closed form of the ring collectives in tables.py, identical to
      the fork on whole runs (8B TP2, 70B TP4, 4 x TP4); shipped 8B tp2, 70B tp4/tp8. Refuses settings where the
      form is not exact (2A > all-reduce step or 2B > last collective step)
- [ ] TP follow-ups: H100 profiles have no default GPU memory bandwidth (need --npu-mem-bw or a cluster config);
      the guard rests on ~300 probes, not a derivation; the inexact corner (large CPU-memory latency) is not
      understood; 256-GPU scale run not yet redone with the table pricer; multi-node topologies out of scope
- [ ] Waiting for a free GPU: real-GPU runs where the schedulers differ (preemption-heavy,
      spec decoding, agent sessions under async): augur vs LLMServingSim per request
- [x] Front-end model (456dcac): constant 6.5 ms intake + 8 ms delivery as defaults; the load-dependent
      part is the engine loop, which augur reproduces (775/778 steps from client send times). 2 req/s
      TTFT -11.7% -> -1.2%. Trace client's 100-connection cap removed (untested on a server)
- [ ] Front end: prompt-length dependence of the 6.5 ms (only 1,024-token text prompts measured); other hosts
- [x] Run-to-run noise: covered by the +-1% sensitivity range in augur bench
- [ ] Automatic KV size from memory accounting (now a manual --kv-reserve-gib)
- [ ] Multi-GPU against real hardware (TP>1, PD): blocked on GPU 1's clock lock

## Also open
- [x] Repo restructured by role (sim/ cost/ trace/ verify/), pytest tests, docs/accuracy.md (960750c);
      ngram self-test found + fixed a replay preemption bug (52e39e0)
- [x] LLMServingSim comparison is a pytest test with its reference data in the repo (ccae81a); 23 tests pass from a fresh clone
- [x] LLMServingSim 256-GPU run on the current branch: matches augur; 2h11m vs 44 s
- [x] LLMServingSim template store bounded (fork 41d4b94); identical output at 16 NPUs
- [ ] 51.6 GB peak RSS at 256 NPUs not explained by the store at 16: rerun 256 to measure
- [x] Commit the PROGRESS.md notes (speed explanation, crash fix)
- [ ] Faster DP rounds; cheaper ASTRA cache key
