"""augur's scheduler against LLMServingSim's, on the same simulated hardware.

Same workloads, RTX PRO 6000 profiles and KV sizes on both sides; augur with
``--lss-loop-compat`` (LLMServingSim's loop timing, not vLLM behaviour) for
the synchronous cases, and ``--step-cpu-us 0 --frontend-latency-ms 0 --frontend-us-per-token 0
--delivery-latency-ms 0`` (LLMServingSim has no CPU term, API server or client). The reference CSVs in data/llmservingsim/ are LLMServingSim's own
outputs (fork branch feature/vllm-sim-20261114, analytical network,
synchronous scheduling unless noted), with the per-token ITL column dropped.
Each case asserts how many calls have the identical TTFT, TPOT, latency and
end time, to the ns.
Where that is not every call, the difference is LLMServingSim's
(docs/accuracy.md, section 4): its idle loop for 2 engines, its async slot
release for async.

Needs the LLMServingSim fork (profiles, ASTRA-Sim, workloads); skipped
without it. About 3 minutes.
"""

import json
import os
import subprocess
import sys

import pytest

from augur.cost.profiles import default_fork
from augur.verify.compare import load, summarize

FORK = default_fork()
WORKLOADS = os.path.join(FORK, "workloads")
DATA = os.path.join(os.path.dirname(__file__), "data", "llmservingsim")
pytestmark = pytest.mark.skipif(not os.path.isdir(WORKLOADS), reason="LLMServingSim fork not found (AUGUR_FORK)")

SHAREGPT = os.path.join(WORKLOADS, "sharegpt-llama-3.1-8b-300-sps10.jsonl")
QWEN = "Qwen/Qwen3-30B-A3B-Instruct-2507"


@pytest.fixture(scope="module")
def swe_fullids(tmp_path_factory):
    """The fork's SWE-bench sessions with input_toks/output_toks set to the
    recorded token-id lengths, so both simulators hash the same prefixes."""
    src = os.path.join(WORKLOADS, "swe-bench-qwen3-30b-a3b-50-sps0.2.jsonl")
    dst = tmp_path_factory.mktemp("swe") / "swe50_fullids.jsonl"
    with open(src) as f, open(dst, "w") as g:
        for line in f:
            d = json.loads(line)
            for s in d["sub_requests"]:
                s["input_toks"] = len(s["input_tok_ids"])
                if s.get("output_tok_ids"):
                    s["output_toks"] = len(s["output_tok_ids"])
                else:
                    s["output_tok_ids"] = []
            g.write(json.dumps(d) + "\n")
    return str(dst)


def llama(*extra):
    return ["--model", "meta-llama/Llama-3.1-8B", "--hbm-gb", "96", "--workload", SHAREGPT, *extra]


def qwen_swe(*extra):
    return ["--model", QWEN, "--max-model-len", "131072", "--workload", "{swe}", *extra]


SYNC = ["--lss-loop-compat", "--no-async-scheduling"]
CASES = [
    # name, reference, augur args, identical (TTFT, TPOT, latency, end_time), calls
    ("sharegpt_tp1", "sharegpt300", SYNC + llama(), (300, 300, 300, 300), 300),
    ("sharegpt_tp1_2eng", "sharegpt300_2inst_rr", SYNC + llama("--engines", "2", "--router", "round-robin"),
     (150, 300, 150, 150), 300),
    ("sharegpt_tp2", "sharegpt300_tp2",
     SYNC + llama("--tp", "2", "--lss-cluster-config", os.path.join(DATA, "llama_tp2.json")), (300, 300, 300, 300), 300),
    ("swebench_tp1", "swe50_fullids", SYNC + qwen_swe("--hbm-gb", "96"), (765, 765, 765, 765), 765),
    ("swebench_tp2", "swe50_fullids_tp2",
     SYNC + qwen_swe("--kv-cache-gb", "30.95", "--tp", "2", "--lss-cluster-config", os.path.join(DATA, "qwen3moe_tp2.json")),
     (765, 765, 765, 765), 765),
    ("sharegpt_dp2_ep", "qwen_sharegpt300_dpep",
     SYNC + ["--model", QWEN, "--workload", os.path.join(WORKLOADS, "sharegpt-qwen3-30b-a3b-300-sps10.jsonl"),
             "--kv-cache-gb", "56.53", "--dp", "2", "--enable-expert-parallel", "--router", "round-robin",
             "--lss-cluster-config", os.path.join(FORK, "configs", "cluster", "single_node_moe_dp_ep_instance.json")],
     (300, 300, 300, 300), 300),
    ("sharegpt_tp1_async", "sharegpt300_async", ["--async-scheduling"] + llama(),
     (128, 1, 1, 1), 300),
]


@pytest.mark.parametrize("name,ref,args,identical,calls", CASES, ids=[c[0] for c in CASES])
def test_matches_llmservingsim(tmp_path, swe_fullids, name, ref, args, identical, calls):
    out = tmp_path / f"{name}.csv"
    args = [a.replace("{swe}", swe_fullids) for a in args]
    subprocess.run([sys.executable, "-m", "augur.cli", "--cost", "llmservingsim", "--hardware", "RTXPRO6000",
                    "--max-num-seqs", "128", "--max-num-batched-tokens", "2048", "--step-cpu-us", "0",
                    "--frontend-latency-ms", "0", "--frontend-us-per-token", "0", "--delivery-latency-ms", "0", "--quiet", *args,
                    "--output", str(out)],
                   check=True, env=dict(os.environ, VLLM_LOGGING_LEVEL="WARNING"),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    s = summarize(load(str(out)), load(os.path.join(DATA, f"{ref}.csv")))
    assert s["matched"] == calls
    got = tuple(s[m]["exact"] for m in ("TTFT", "TPOT", "latency", "end_time"))
    assert got == identical
