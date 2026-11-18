#!/usr/bin/env bash
# Real vLLM run with KV offload to CPU (GPU 0), then augur's checks against it. Not run yet: needs a free GPU.
# usage: [ASYNC=1] [AUGUR_VENV=.venv] [AUGUR_FORK=~/llmss-vllm-sim-20261114] scripts/real_offload_check.sh
#   (~15 min of GPU per run; 16 GiB pinned host memory)
#   1. vllm serve Qwen3-1.7B, 16 GiB CPU tier, augur's logging scheduler (this checkout's hooks: they log offload jobs),
#      the 50-session SWE-bench trace through augur.verify.client
#   2. augur replay of that log: scheduling decisions must match step for step
#   3. measured GPU<->CPU bandwidth from the log's transfer stats
#   4. augur run normally at that bandwidth vs the real run (TTFT/TPOT/latency, prefix hits)
set -euo pipefail
cd "$(dirname "$0")/.."
async=${ASYNC:-0}
name=swe50_offload16$([ "$async" = 1 ] && echo _async)
out=out/real/$name
mkdir -p "$out"; rm -f "$out"/steps.*.jsonl
VENV=${AUGUR_VENV:-.venv}
W=${AUGUR_FORK:-$HOME/llmss-vllm-sim-20261114}/workloads/swe-bench-qwen3-30b-a3b-50-sps0.2.jsonl
P=$VENV/bin/python
# PYTHONPATH: the server must import this checkout's augur.verify.hooks. PYTHONHASHSEED: augur pins 0; vLLM's
# offload eviction order depends on it, so the real server must use the same.
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 AUGUR_STEP_LOG=$out/steps PYTHONPATH=$PWD PYTHONHASHSEED=0
sched=$([ "$async" = 1 ] && echo "--async-scheduling --scheduler-cls augur.verify.hooks.LoggingAsyncScheduler" \
                         || echo "--no-async-scheduling --scheduler-cls augur.verify.hooks.LoggingScheduler")
"$VENV/bin/vllm" serve Qwen/Qwen3-1.7B --port 8013 --seed 0 \
  --max-num-seqs 128 --max-num-batched-tokens 2048 --max-model-len 24576 --gpu-memory-utilization 0.85 \
  --kv-offloading-size 16 --kv-offloading-backend native $sched > "$out/server.log" 2>&1 &
srv=$!
trap 'kill $srv 2>/dev/null; wait $srv 2>/dev/null || true' EXIT
for i in $(seq 1 300); do
  curl -sf http://localhost:8013/health > /dev/null && break
  kill -0 $srv 2>/dev/null || { echo "server died"; tail -30 "$out/server.log"; exit 1; }
  sleep 2
done
$P -m augur.verify.client --url http://localhost:8013 --model Qwen/Qwen3-1.7B --workload "$W" \
  --filler-vocab 150000 --output "$out/client.csv" --tokens-out "$out/tokens.jsonl" > "$out/client.json"
kill $srv; wait $srv 2>/dev/null || true; trap - EXIT

log=$(ls $out/steps.*.jsonl)
blocks=$(( $(grep -oE "GPU KV cache size: [0-9,]+" $out/server.log | tail -1 | grep -oE "[0-9,]+$" | tr -d ,) / 16 ))
amode=$([ "$async" = 1 ] && echo --async-scheduling || echo --no-async-scheduling)
common=(--model Qwen/Qwen3-1.7B --hardware RTX4060Ti --num-gpu-blocks $blocks --max-model-len 24576 --max-num-seqs 128
        --max-num-batched-tokens 2048 $amode --no-recorded-outputs --filler-vocab 150000 --workload "$W"
        --kv-offload-gib 16 --quiet)

# 2. decisions
$P -m augur.cli "${common[@]}" --replay-durations "$log" --step-log-vllm $out/augur_replay --output $out/augur_replay.csv
$P -m augur.verify.replay compare "$log" $(ls $out/augur_replay.*.jsonl) | tee $out/replay_compare.json

# 3. bandwidth the real transfers achieved (bytes / GPU-timed seconds, per direction)
gbps=$($P - "$log" <<'EOF'
import json, sys
tot = {"load": [0, 0.0], "store": [0, 0.0]}
for line in open(sys.argv[1]):
    e = json.loads(line)
    for d, (b, s, _) in (e.get("xfer") or {}).items():
        tot[d][0] += b; tot[d][1] += s
rates = {d: (b / s / 1e9 if s else None) for d, (b, s) in tot.items()}
json.dump({"bytes": {d: v[0] for d, v in tot.items()}, "gbps": rates}, open(sys.argv[1].rsplit("/", 1)[0] + "/xfer.json", "w"))
print(round(min(r for r in rates.values() if r), 2))
EOF
)
echo "measured GPU<->CPU GB/s (slower direction): $gbps"

# 4. end to end at that bandwidth
$P -m augur.cli "${common[@]}" --kv-offload-gbps "$gbps" --output $out/augur.csv
echo "done: $out"
