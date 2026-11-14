#!/usr/bin/env bash
# Re-run augur against stored LLMServingSim outputs and print the comparison.
# The LLMServingSim reference CSVs live in out/lss/ (see PROGRESS.md for how
# they were produced). Run from the augur repo root on anjuna3.
set -euo pipefail
W=${AUGUR_FORK:-$HOME/llmss-vllm-sim-20261114}/workloads
PY=.venv/bin/python
export VLLM_LOGGING_LEVEL=WARNING
mkdir -p out/validate
COMMON="--cost llmservingsim --hardware RTXPRO6000 --max-num-seqs 128 --max-num-batched-tokens 2048 --lss-loop-compat"
LLAMA="--model meta-llama/Llama-3.1-8B --hbm-gb 96 --workload $W/sharegpt-llama-3.1-8b-300-sps10.jsonl"
QWEN="--model Qwen/Qwen3-30B-A3B-Instruct-2507 --max-model-len 131072 --workload out/lss/swe50_fullids.jsonl"

run() {  # name, reference csv, augur args...
  local name=$1 ref=$2; shift 2
  .venv/bin/augur $COMMON "$@" --output out/validate/$name.csv > out/validate/$name.json 2>/dev/null
  printf '%-28s ' "$name"
  $PY - "$name" "$ref" <<'EOF'
import sys
sys.path.insert(0, ".")
from augur.compare import load, summarize
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
