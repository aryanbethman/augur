#!/usr/bin/env bash
# Re-run augur against stored LLMServingSim outputs and print the comparison.
# The LLMServingSim reference CSVs live in out/lss/ (produced by the fork on the
# same configs; see docs/accuracy.md, section 4). Run from the augur repo root on anjuna3.
set -euo pipefail
W=${AUGUR_FORK:-$HOME/llmss-vllm-sim-20261114}/workloads
PY=.venv/bin/python
export VLLM_LOGGING_LEVEL=WARNING
mkdir -p out/validate
COMMON="--cost llmservingsim --hardware RTXPRO6000 --max-num-seqs 128 --max-num-batched-tokens 2048 --lss-loop-compat --no-async-scheduling"
LLAMA="--model meta-llama/Llama-3.1-8B --hbm-gb 96 --workload $W/sharegpt-llama-3.1-8b-300-sps10.jsonl"
QWEN="--model Qwen/Qwen3-30B-A3B-Instruct-2507 --max-model-len 131072 --workload out/lss/swe50_fullids.jsonl"

run() {  # name, reference csv, augur args...
  local name=$1 ref=$2; shift 2
  .venv/bin/augur $COMMON "$@" --output out/validate/$name.csv > out/validate/$name.json 2>/dev/null
  printf '%-28s ' "$name"
  $PY - "$name" "$ref" <<'EOF'
import sys
sys.path.insert(0, ".")
from augur.verify.compare import load, summarize
s = summarize(load(f"out/validate/{sys.argv[1]}.csv"), load(sys.argv[2]))
n = s["matched"]
print(" ".join(f"{m}={s[m]['exact']}/{n}" for m in ("TTFT", "TPOT", "latency", "end_time")))
EOF
}

run sharegpt_tp1        out/lss/sharegpt300.csv             $LLAMA
run sharegpt_tp1_2eng   out/lss/sharegpt300_2inst_rr.csv    $LLAMA --engines 2 --router round-robin
run sharegpt_tp2        out/lss/sharegpt300_tp2.csv         $LLAMA --tp 2 --lss-cluster-config out/lss/llama_tp2.json
run swebench_tp1        out/lss/swe50_fullids.csv           $QWEN --hbm-gb 96
run swebench_tp2        out/lss/swe50_fullids_tp2.csv       $QWEN --kv-cache-gb 30.95 --tp 2 --lss-cluster-config out/lss/qwen3moe_tp2.json
run sharegpt_dp2_ep       out/lss/qwen_sharegpt300_dpep.csv  --model Qwen/Qwen3-30B-A3B-Instruct-2507 --workload $W/sharegpt-qwen3-30b-a3b-300-sps10.jsonl --kv-cache-gb 56.53 --dp 2 --enable-expert-parallel --router round-robin --lss-cluster-config ${AUGUR_FORK:-$HOME/llmss-vllm-sim-20261114}/configs/cluster/single_node_moe_dp_ep_instance.json

# Async scheduling (vLLM's default). Not call-for-call: LLMServingSim's async
# mode frees a finishing request's slot one step earlier than vLLM (docs/accuracy.md).
COMMON="--cost llmservingsim --hardware RTXPRO6000 --max-num-seqs 128 --max-num-batched-tokens 2048"
run sharegpt_tp1_async      out/lss/sharegpt300_async.csv     $LLAMA --async-scheduling

# Replay self-test: augur's own step log, replayed with its arrival times and
# step durations, must reproduce every scheduling decision (augur.verify.replay).
rm -f out/validate/replay_orig.* out/validate/replay_rep.*
R="--cost llmservingsim --hardware RTXPRO6000 --hbm-gb 96 --max-model-len 131072 --max-num-seqs 128 --max-num-batched-tokens 2048 --model Qwen/Qwen3-30B-A3B-Instruct-2507 --no-async-scheduling --quiet --workload out/lss/swe50_fullids.jsonl"
.venv/bin/augur $R --step-log-vllm out/validate/replay_orig --output out/validate/replay_orig.csv > /dev/null 2>&1
LOG=$(ls out/validate/replay_orig.*.jsonl)
.venv/bin/augur $R --replay-durations "$LOG" --step-log-vllm out/validate/replay_rep --output out/validate/replay_rep.csv > /dev/null 2>&1
printf '%-28s ' replay_selftest_swebench
$PY -m augur.verify.replay compare "$LOG" out/validate/replay_rep.*.jsonl 2>/dev/null | $PY -c "import json,sys; r=json.loads(''.join(l for l in sys.stdin if not l.startswith(('first','  A','  B')))); print(f\"steps={r['identical_steps']}/{r['steps_a']}\")"
