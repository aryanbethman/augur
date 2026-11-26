# augur to-do

Goal: augur replaces running vLLM on GPUs for serving research. It runs vLLM's own control plane, unmodified, on a
virtual clock. Focus now: scale, speed and research value, then a simulator others can install and reuse.

Sections are in priority order, and so are the items inside each. Status: [ ] open, [~] in progress.
Finished work lives in docs/accuracy.md and git history, not here.

Rule for every change: output stays byte-identical unless the change is meant to move it, the six real-GPU replays
stay exact, and the full test suite passes.

Rule for large-scale runs (items 1, 2, 4, 15): engines must be loaded as in production, so time goes to real
scheduling work, not start-up, idle engines or a draining tail. scripts/baselines/speed.py's defaults do this:
replicated ShareGPT at 10 req/s per GPU and SWE-bench at 0.2 sessions/s per GPU (the real single-GPU runs' rates).
Report each run's busy share, mean batch against the real run's, and start-up CPU apart from the run
(docs/accuracy.md section 7).

## 1. Scale, speed and utility (no new hardware)

Where things stand (docs/accuracy.md sections 5-7):
- **Speed:** with `--workers 8`, every 256-GPU run finishes inside BLIS's CPU time: ShareGPT 67-87 s and SWE-bench
  118 s, against 100 and 178 s. Measured on a busy host. BLIS still spends less CPU per step.
- **Memory:** augur needs 5.9 GB at 1,024 GPUs on ShareGPT, which projects to ~78 GB at 16,384. BLIS's memory grows with
  the run: it needs more than 125 GB on ShareGPT at 1,024 GPUs and does not finish.
- **Utility:**
  - BLIS with one fixed speculation acceptance rate picks the wrong ngram depth on code, RAG and chat, outside
    run-to-run noise. augur's picks hold within noise.
  - Scheduler policies written once rank the same in real vLLM and augur, over two real runs each.
  - On 2 real engines augur places 95-100% of requests where real round-robin and cache-aware routing did; BLIS
    places 50-54%.

BLIS's scheduler port is exact on our runs, so better scheduling logic is not a claim augur can make. augur wins
where the answer depends on token content and vLLM's own block ids (overlapping prompts, shared-KV pricing,
speculation acceptance), on memory at scale, and on running a researcher's code unchanged. It will not win CPU per
step. Compiling vLLM's scheduler with Cython was tried and gave 5-7% (not adopted; docs/accuracy.md section 7).

1. [ ] **Scale study against BLIS:** 16 to 16,384 GPUs at production load on a quiet host, 3 runs a point.
   - Report wall time with workers, CPU, peak memory, and the size at which BLIS outgrows the host.
   - Confirm the 78 GB projection with real 4,096- and 16,384-GPU runs.
   - Measure CPU per step and workers together; they have only been measured apart.
2. [ ] **The main process in worker runs.** On SWE-bench at 256 GPUs its own work is ~40 of 112 s, on the critical path:
   drawing filler token ids (~17 s) and pickling long prompts (~9 s). Draw ids in the workers, or ship prompts once.
   Then lift the limits:
   - a DP group inside one worker;
   - PD running ahead by the minimum KV transfer time;
   - KV offload (untested with workers);
   - least-loaded still stops every worker at each arrival.
3. [ ] **Structured outputs** (grammar-constrained decoding) in augur, running vLLM's structured-output manager.
   BLIS's Go code has no structured outputs, grammars or xgrammar (it does have LoRA and multimodal, so those do not
   set augur apart). The work is pricing the grammar compile on the virtual clock.
4. [ ] **Router choice at scale, for real.** 2 real engines anchor placement (augur matches 95-100%), but GPU 1's clock
   lock hides timing, and augur and BLIS rank the routers differently at 64 and 1,024 engines. Settling which router
   is best needs unlocked GPUs (rented, item 13).

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
8. [ ] **BLIS's porting lag, from its git history** (no download): for each vLLM release BLIS followed, the Go lines
   changed and the days between the vLLM release and BLIS's port.
9. [ ] **augur's vLLM upgrade.** Move augur to the next vLLM release: lines changed, the six real replays rerun, and
   a decision change that only the new code produces. Needs the new vLLM wheel (download).
10. [ ] **Third-party and unreleased code on day one.** LMCache's connector unmodified, replayed against a real LMCache
    run, and a vLLM release candidate or open pull request run without changes to augur. Needs downloads.

## 3. Needs rented GPUs

11. [ ] **H100 profiles** for Llama-3.1-70B TP4 out to 32k KV tokens (the shipped one stops at 2,048 KV tokens and
    2,048 tokens per step; augur warns and reports steps_beyond_profile). About 4-12 GPU hours.
12. [ ] **MoE with wider expert parallelism** than DP2 + EP2 of TP1 ranks: profile those layouts, then price them from
    tables. Today they fall back to the fork's legacy EP table (unverified; its collective rule was derived for two
    ranks, and DP rounds through ASTRA-Sim cost ~25 ms each). MoE TP8 has no profile to fit against.
13. [ ] **Real prefill/decode replay** (one prefill and one decode engine; needs NIXL), and multi-GPU timing on
    GPUs without a clock lock (GPU 1 here is locked).
14. [ ] **The per-step CPU term on a second host.** 1.3 ms is one host's value (i7-12700, vLLM 0.33).
15. [ ] **Engines across hosts** (after items 1 and 2): the same sync over the network, for 100k+ GPUs.

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
