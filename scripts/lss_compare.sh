#!/usr/bin/env bash
# LLMServingSim (the fork) on the RTX 4060 Ti / Qwen3-1.7B profile, sized to the KV cache of a real run, then
# augur's compare_real_sim.py: real vs augur (normal run) vs LLMServingSim.
# usage: scripts/lss_compare.sh REAL_DIR WORKLOAD ASYNC(0|1) MAXLEN [extra args for BOTH simulators, e.g. --num-speculative-tokens 3
#        for augur; the fork takes --num-speculative-tokens N --spec-acceptance-rate R, see LSS_EXTRA]
#   REAL_DIR: out/real/<name> (server.log, steps.*.jsonl, client.csv from scripts/real_trace_run.sh)
#   LSS_EXTRA: extra args for the fork only (e.g. "--num-speculative-tokens 3 --spec-acceptance-rate 0.4")
#   Needs the fork (AUGUR_FORK, default ~/llmss-vllm-sim-20261114). Output: REAL_DIR/augur.csv, lss.csv, lss.log
# The fork sizes its KV cache from --npu-memory-utilization: on this profile 0.85 gives 91,888 tokens and each 0.01
# adds 1,498, so the utilization is set to the real run's "GPU KV cache size".
set -uo pipefail
real=$(realpath "$1"); wl=$(realpath "$2"); async=$3; maxlen=$4; shift 4
cd "$(dirname "$0")/.."
here=$PWD
VENV=${AUGUR_VENV:-$HOME/augur/.venv}
FORK=${AUGUR_FORK:-$HOME/llmss-vllm-sim-20261114}
export HF_HUB_OFFLINE=1 PYTHONPATH=$here VLLM_LOGGING_LEVEL=WARNING
T=$(grep -oE "GPU KV cache size: [0-9,]+" "$real/server.log" | tail -1 | grep -oE "[0-9,]+$" | tr -d ,)
u=$(python3 -c "print(round(0.85 - (91888 - $T) / 149818, 4))")
amode=--no-async-scheduling; [ "$async" = 1 ] && amode=--async-scheduling
echo "real KV tokens $T -> fork --npu-memory-utilization $u"
"$VENV/bin/python" -m augur.cli --model Qwen/Qwen3-1.7B --hardware RTX4060Ti --num-gpu-blocks $((T / 16)) \
  --max-model-len "$maxlen" --max-num-seqs 128 --max-num-batched-tokens 2048 $amode --no-recorded-outputs --quiet \
  --workload "$wl" --output "$real/augur.csv" "$@" > "$real/augur.log" 2>&1
# the fork resolves --dataset/--output relative to its root with a ../ prefix: pass paths relative to it
rel() { realpath --relative-to="$FORK" "$1"; }
( cd "$FORK" && PYTHONPATH=$HOME/llmss-shared-templates-upstream-20261112/.benchmark-python \
  /usr/bin/time -v "$VENV/bin/python" -m serving --cluster-config "$here/scripts/lss_4060.json" \
  --dataset "$(rel "$wl")" --output "$(rel "$real/lss.csv")" --max-num-seqs 128 --max-num-batched-tokens 2048 \
  --block-size 16 --npu-memory-utilization "$u" $amode ${LSS_EXTRA:-} > "$real/lss.log" 2>&1 )
echo "fork rc=$?"; grep -E "Elapsed|Maximum resident" "$real/lss.log"
grep -E "Instance \[0\] :" "$real/lss.log" | head -1
"$VENV/bin/python" scripts/compare_real_sim.py "$real/client.csv" "$real/augur.csv" "$real/lss.csv" \
  --real-log "$(ls "$real"/steps.*.jsonl | head -1)" --labels real,augur,lss
