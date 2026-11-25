# augur to-do

Goal: augur replaces running vLLM on GPUs for serving research. It runs vLLM's own control plane, unmodified, on a
virtual clock. Focus now: scale, speed and research value, then a simulator others can install and reuse.

Sections are in priority order, and so are the items inside each. Status: [ ] open, [~] in progress.
Finished work lives in docs/accuracy.md and git history, not here.

Rule for every change: output stays byte-identical unless the change is meant to move it, the six real-GPU replays
stay exact, and the full test suite passes.

Rule for large-scale runs (items 1, 2, 3, 13): engines must be loaded as in production, so time goes to real
scheduling work, not start-up, idle engines or a draining tail. scripts/baselines/speed.py's defaults do this:
replicated ShareGPT at 10 req/s per GPU and SWE-bench at 0.2 sessions/s per GPU (the real single-GPU runs' rates).
Report each run's busy share, mean batch against the real run's, and start-up CPU apart from the run
(docs/accuracy.md section 7).

## 1. Scale, speed and utility (no new hardware)

Where things stand (docs/accuracy.md section 7):
- **CPU:** at production load BLIS spends 4-7x less CPU than augur at 16-64 GPUs and 1.8x less at 1,024.
- **Memory:** BLIS's memory grows with the run. At 1,024 GPUs it needs 8x augur's on SWE-bench, and more than 125 GB
  on ShareGPT, so it does not finish there.
- **Parallel workers:** `--workers 8` runs 1,024 GPUs 5.2x faster than one process, with byte-identical output.

1. [ ] **Less CPU per step, without touching vLLM's code.** The 1,024-GPU profile gives the targets, as share of
   CPU on ShareGPT / SWE-bench:
   - Skip counting shared KV blocks (augur/verify/prefix_share.shared_tokens, from the executor) when prefix caching
     is off or the profile has no shared-KV term: 12 / 15%.
   - Cache pricer results: pricing is 6 / 19%, so roughly 3-4 / 10-14% saved.
   - Skip output processing augur never reads: about 5-8 / 3% (check_stop and token appends stay).
   - Trim the event loop: about 3-5 / 5-8%.

   Expected about 1.3x on ShareGPT and 1.5x on SWE-bench. vLLM's stats and logging are already off, so they add
   nothing.
2. [ ] **Faster parallel workers.**
   - Build engines inside the workers: start-up is 33 s on one core at 1,024 GPUs.
   - Profile the main process: there is no gain past 8 workers, and CPU grows from 1,032 to 1,505 s between 4 and 8
     workers.
   - Then lift the limits: a DP group inside one worker; PD running ahead by the minimum KV transfer time; KV offload
     (untested with workers); sessions with zero delivery latency.

   Target: 1,024 GPUs at production load in under a minute of wall time (222 s with 8 workers now).
3. [ ] **Scale study against BLIS** (after 1 and 2): 16 to 16,384 GPUs at production load on a quiet host, 3 runs a
   point. Report wall time with workers, CPU, peak memory, and the size at which BLIS outgrows the host.
4. [ ] **A sweep whose best setting lies inside the range.** The batch-token sweep was monotone, so it could not
   separate the simulators: augur, BLIS and Vidur picked the real best, Frontier missed by 1.3%. Sweep KV size (or
   max-num-seqs) on an overlapping-prompt workload with `scripts/baselines/sweep.py`, and repeat real runs to measure
   noise. Launching this sweep was blocked by a permission check on 2026-11-25; it needs the owner's go-ahead.

## 2. Standalone and reusable

5. [ ] **Install and run anywhere.** A clean `pip install` (needs vllm 0.33.0 and its dependencies from the network),
   a Dockerfile, CI for the tests that need no fork, a setup script for the LLMServingSim fork and ASTRA-Sim
   (still manual), and a LICENSE (needs the owner's choice). The wheel already builds offline and runs from outside
   the repo with a shipped profile.
6. [ ] **Profile your own GPU, end to end.** `python -m augur.profile --model M --hardware H --tp N` is written and its
   dry run and conversion are tested, but its GPU stages have never run. Run it on the 4060 Ti first (also shakes out
   the memory probe).
7. [ ] **Push fork commit e8625e0** (feature/vllm-sim-20261114), which the shipped 4060 Ti profile's provenance points
   at. Needs the owner, from a machine with access.
8. [ ] **vLLM version study.** Move augur to the next vLLM release: lines changed, the six real replays rerun, a
   decision change that only the new code produces, and what BLIS's ports took for each release (lines, days after
   the release) from its history. Needs the new vLLM wheel (download).
9. [ ] **Third-party and unreleased code on day one.** LMCache's connector unmodified, replayed against a real LMCache
   run, and a vLLM release candidate or open pull request run without changes to augur. Needs downloads.

## 3. Needs rented GPUs

10. [ ] **H100 profiles** for Llama-3.1-70B TP4 out to 32k KV tokens (the shipped one stops at 2,048 KV tokens and
    2,048 tokens per step; augur warns and reports steps_beyond_profile). About 4-12 GPU hours.
11. [ ] **MoE with wider expert parallelism** than DP2 + EP2 of TP1 ranks: profile those layouts, then price them from
    tables. Today they fall back to the fork's legacy EP table (unverified; its collective rule was derived for two
    ranks, and DP rounds through ASTRA-Sim cost ~25 ms each). MoE TP8 has no profile to fit against.
12. [ ] **Real prefill/decode replay** (one prefill and one decode engine; needs NIXL), and multi-GPU timing on
    GPUs without a clock lock (GPU 1 here is locked).
13. [ ] **The per-step CPU term on a second host.** 1.3 ms is one host's value (i7-12700, vLLM 0.33).
14. [ ] **Engines across hosts** (after item 2): the same sync over the network, for 100k+ GPUs.

## 4. Parked

No work planned; kept so the findings are not lost.

- Pipeline parallelism; EAGLE with DP; MoE MTP pricing (needs per-layer MoE); profiling a real EAGLE head.
- Contention: refit K and host memory on more hosts and GPUs and on two-socket hosts; a transfer-bound run; the
  measured per-GPU asymmetry of copy bandwidth under CPU memory load is not reproduced.
- Tiered offload: a real-hardware check (CXL or a second memory pool); storage tiers (fs/obj) as more than a slow
  memory tier.
- vLLM's tiering livelocks when one request's prefix needs more chunks than the CPU tier holds (promote, evict,
  promote, ...); augur stops such a run after 60 s simulated without a token. Report upstream? Owner's call.
- Accuracy refinements:
  - a delivery default per scheduling mode (the per-user run's TTFT is +16%, half a step counted twice);
  - the q=4 attention table looks low (ngram error grows ~0.2 us per context token);
  - stepcheck builds its dispatcher without the speculative config (0.2%);
  - the shared-KV term on other GPUs, at query length > 1 and for prefill chunks;
  - more than one run per policy in the plugin study, a second workload, and kv-aware routing at scale;
  - the breakdown's step/decision split is not meaningful on always-busy runs;
  - SWE-bench at 2.0 sessions/s async: augur preempts 7 times against real's 20 (replay exact, so step time);
  - async DP: an idle DP group sleeps where vLLM runs dummy rounds until the next sync.
- LLMServingSim's scheduler against vLLM's under a cut KV cache: higher session rates did not separate them
  (docs/accuracy.md section 4), because KV, not the rate, sets the batch.
- Front end: token-id prompts (the trace client) and other hosts.
- Ask Revati's authors for their code.
