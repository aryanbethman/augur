#!/usr/bin/env bash
# Real vLLM run of a trace with augur's logging scheduler, driven by augur.verify.client (GPU 0 by default).
# usage: [ASYNC=1] [MAXBATCH=2048] [GPU=0] [PORT=8011] [MAXLEN=8192] [MODEL=Qwen/Qwen3-1.7B] scripts/real_trace_run.sh NAME WORKLOAD [NUM_ENTRIES] [extra vllm serve args...]
#   Stress: extra args such as --num-gpu-blocks-override N, --scheduling-policy priority, --no-enable-prefix-caching.
#   DP or TP: GPU=0,1 and --data-parallel-size 2 / --tensor-parallel-size 2 as extra args.
#   output in out/real/NAME: steps.<pid>.jsonl (a DP run: steps.e<rank>.<pid>.jsonl per rank), client.csv/json, tokens.jsonl,
#   server.log, uptime_*.txt
#   The SWE-bench trace needs MAXLEN=24576. The client's --filler-vocab defaults to 150000, as for out/real/swe50.
set -euo pipefail
name=$1; workload=$2; n=${3:-}; shift 3 || shift $#
cd "$(dirname "$0")/.."
VENV=${AUGUR_VENV:-$HOME/augur/.venv}
out=out/real/$name; mkdir -p "$out"; rm -f "$out"/steps.*.jsonl
export CUDA_VISIBLE_DEVICES=${GPU:-0} HF_HUB_OFFLINE=1 AUGUR_STEP_LOG=$out/steps PYTHONPATH=$PWD
port=${PORT:-8011}; MODEL=${MODEL:-Qwen/Qwen3-1.7B}
if [ "${ASYNC:-0}" = 1 ]; then sched="--async-scheduling --scheduler-cls augur.verify.hooks.LoggingAsyncScheduler"
else sched="--no-async-scheduling --scheduler-cls augur.verify.hooks.LoggingScheduler"; fi
"$VENV/bin/vllm" serve "$MODEL" --port "$port" --seed 0 \
  --max-num-seqs 128 --max-num-batched-tokens ${MAXBATCH:-2048} --max-model-len "${MAXLEN:-8192}" \
  --gpu-memory-utilization 0.85 $sched "$@" > "$out/server.log" 2>&1 &
srv=$!
trap 'kill $srv 2>/dev/null; wait $srv 2>/dev/null || true' EXIT
for i in $(seq 1 300); do
  curl -sf "http://localhost:$port/health" > /dev/null && break
  kill -0 $srv 2>/dev/null || { echo "server died"; tail -30 "$out/server.log"; exit 1; }
  sleep 2
done
uptime > "$out/uptime_start.txt"
"$VENV/bin/python" -m augur.verify.client --url "http://localhost:$port" --model "$MODEL" \
  --workload "$workload" ${n:+--num-entries $n} \
  --output "$out/client.csv" --tokens-out "$out/tokens.jsonl" > "$out/client.json"
uptime > "$out/uptime_end.txt"
kill -INT $srv; wait $srv 2>/dev/null || true; trap - EXIT
grep -E "GPU KV cache size" "$out/server.log" | tail -1
