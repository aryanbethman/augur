#!/usr/bin/env bash
# Front-end intake delay vs prompt length: real `vllm serve` (async, logging scheduler) + `vllm bench serve` random
# prompts at light load, one run per input length against one server. Analyse with scripts/frontend_intake.py.
# usage: [LENS="256 1024 4096 1024"] [RATE=1] [N=100] [OUT=out/frontend_len] scripts/frontend_intake.sh   (GPU 0)
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=${AUGUR_VENV:-$HOME/augur/.venv}
out=${OUT:-out/frontend_len}; mkdir -p "$out"; rm -f "$out"/steps.*.jsonl
export HF_HUB_OFFLINE=1 AUGUR_STEP_LOG=$out/steps PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0
port=8013
"$VENV/bin/vllm" serve Qwen/Qwen3-1.7B --port $port --seed 0 \
  --max-num-seqs 128 --max-num-batched-tokens 2048 --max-model-len 8192 --gpu-memory-utilization 0.85 \
  --async-scheduling --scheduler-cls augur.verify.hooks.LoggingAsyncScheduler > "$out/server.log" 2>&1 &
srv=$!
trap 'kill $srv 2>/dev/null; wait $srv 2>/dev/null || true' EXIT
for i in $(seq 1 300); do
  curl -sf http://localhost:$port/health > /dev/null && break
  kill -0 $srv 2>/dev/null || { echo "server died"; tail -30 "$out/server.log"; exit 1; }
  sleep 2
done
uptime > "$out/uptime_start.txt"
k=0
for len in ${LENS:-256 1024 4096 1024}; do
  k=$((k + 1))
  "$VENV/bin/vllm" bench serve --base-url http://localhost:$port --model Qwen/Qwen3-1.7B --dataset-name random \
    --random-input-len "$len" --random-output-len 32 --num-prompts "${N:-100}" --request-rate "${RATE:-1}" \
    --ignore-eos --seed "$k" --save-result --save-detailed --result-dir "$out" --result-filename "len${len}_$k.json" \
    > "$out/len${len}_$k.txt" 2>&1
done
uptime > "$out/uptime_end.txt"
kill -INT $srv; wait $srv 2>/dev/null || true; trap - EXIT
ls "$out"
