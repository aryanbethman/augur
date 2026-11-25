#!/usr/bin/env bash
# Same-step-time test: one real vLLM run through augur and the three baseline simulators, all priced by
# scripts/baselines/pricer.py. usage:  scripts/baselines/run.sh RUN [STAGE...]
#   RUN    smoke sharegpt300 sharegpt300_async sharegpt300_spec sharegpt300_spec_async swe50 swe50_async
#          swe50_offload16 swe50_offload16_async bench_r6   (the real runs under $AUGUR_OUT, default ~/augur/out)
#          any other real run: AUGUR_RUN_DIR=<run dir> AUGUR_WORKLOAD=<workload jsonl> run.sh NAME [STAGE...]  (NAME is the
#          directory under out/baselines/; the run dir holds steps.*.jsonl and, for the settings, server.log). The
#          settings come from its server.log; these env vars state or override them: MAX_MODEL_LEN MAX_BATCHED_TOKENS
#          MAX_NUM_SEQS NUM_GPU_BLOCKS PREFIX_CACHING=on|off SCHEDULING_POLICY=fcfs|priority ASYNC_SCHEDULING=sync|async
#          SHARED_KV=on (augur only: the profile's shared-KV decode term, off by default so that every simulator has one price)
#          ACCEPT_LEN=<tokens per step> (BLIS only: one speculative acceptance length instead of the run's measured one)
#          AUGUR_TOKENS=<tokens.jsonl> (augur only: emit these output ids, so that ngram drafts and acceptance come from
#          the content, instead of unique ids drawn at --spec-accept-rate)
#          BLIS_ENCODING=greedy|best (how BLIS gets prefix groups and sessions; greedy, the default, is the earlier runs').
#          The workload's per-request "priority" goes to BLIS (blis_priority.patch, built into
#          inference-sim-prio/blis-ext-prio) and Frontier; Vidur has none, so it is not run. Prefix caching off:
#          BLIS and Frontier switch it off, Vidur has none.
#          PRICER_ARGS='--step-class-scales scales.json' gives BLIS's pricer a calibration (augur.verify.calibrate).
#   STAGE  prepare augur blis frontier vidur replay compare   (default: all of them; frontier and vidur only for runs
#          they can take: they have no async scheduling, speculative decoding or KV offload, so those are skipped
#          unless a stage is named: run.sh bench_r6 prepare augur blis frontier vidur compare)
# "replay" runs blis, frontier and vidur again with step k taking what step k took in the real run (pricer.py
# --durations), outputs *_replay: with the same arrivals an exact copy of the scheduler reproduces the real steps.
# Results go to out/baselines/RUN/. Simulator checkouts are in $BASELINES (default ~/baselines): vidur, Frontier,
# inference-sim (with its Go module cache in inference-sim/.gopath, Go in ~/opt/go). BLIS is built once from
# inference-sim and blis_vllm.patch (the step price and the KV block count taken from outside) into inference-sim/blis-ext.
set -uo pipefail
W=$(cd "$(dirname "$0")/../.." && pwd)
S=$W/scripts/baselines
OUTR=${AUGUR_OUT:-$HOME/augur/out}
BASE=${BASELINES:-$HOME/baselines}
PY=${AUGUR_PYTHON:-$HOME/augur/.venv/bin/python}
WL=${AUGUR_FORK:-$HOME/llmss-vllm-sim-20261114}/workloads
export VLLM_LOGGING_LEVEL=ERROR PYTHONPATH=$W
run=$1; shift
SG=$WL/sharegpt-qwen3-32b-300-sps10.jsonl; SW=$WL/swe-bench-qwen3-30b-a3b-50-sps0.2.jsonl
case $run in
  smoke) dir=$OUTR/real/smoke; wl=$SG;;
  sharegpt300*) dir=$OUTR/real/$run; wl=$SG;;
  swe50*) dir=$OUTR/real/$run; wl=$SW;;
  bench_r6) dir=$OUTR/bench/rand1024_r6_log; wl=$dir/requests.jsonl;;
  *) if [ -n "${AUGUR_RUN_DIR:-}" ] && [ -n "${AUGUR_WORKLOAD:-}" ]; then dir=$AUGUR_RUN_DIR; wl=$AUGUR_WORKLOAD
     else echo "unknown run $run (or set AUGUR_RUN_DIR and AUGUR_WORKLOAD)" >&2; exit 2; fi;;
esac
# the run's settings stated from outside (else prepare reads them from its server.log)
PREP=""
for kv in max-model-len:MAX_MODEL_LEN max-num-batched-tokens:MAX_BATCHED_TOKENS max-num-seqs:MAX_NUM_SEQS \
          num-gpu-blocks:NUM_GPU_BLOCKS prefix-caching:PREFIX_CACHING scheduling-policy:SCHEDULING_POLICY \
          async-scheduling:ASYNC_SCHEDULING blis-encoding:BLIS_ENCODING; do
  v=${kv#*:}; [ -n "${!v:-}" ] && PREP="$PREP --${kv%%:*} ${!v}"
done
O=$W/out/baselines/$run
get() { $PY $S/workloads.py get $O "$1"; }
stages=${*:-prepare augur blis frontier vidur replay compare}
has() { [[ " $stages " == *" $1 "* ]]; }
explicit=$([ -n "$*" ] && echo 1 || echo 0)
V=""; PX=""   # variant suffix of the outputs, and extra pricer.py arguments (replay)

prepare() {
  $PY $S/workloads.py prepare $dir $wl $O $PREP > /dev/null && for s in vidur frontier blis; do $PY $S/workloads.py $s $O; done
  # what was run: the simulators' commits and BLIS's engine module, Go, and this checkout
  { echo "{"
    for d in vidur Frontier inference-sim; do echo "\"$d\": \"$(git -C $BASE/$d rev-parse HEAD)\","; done
    echo "\"blis_vllm\": \"$(basename $(ls -d $BASE/inference-sim/.gopath/pkg/mod/github.com/inference-sim/blis-vllm@*))\","
    echo "\"go\": \"$($HOME/opt/go/bin/go version)\", \"augur\": \"$(git -C $W rev-parse HEAD)\""
    echo "}"; } > $O/versions.json
}

augur() {
  local extra=""
  [ "$(get prefix_caching)" = false ] && extra="--no-prefix-caching"
  [ "$(get scheduling_policy)" = priority ] && extra="$extra --scheduling-policy priority"
  [ "$(get async_scheduling)" = true ] && extra="$extra --async-scheduling" || extra="$extra --no-async-scheduling"
  local k=$(get spec_k); [ "$k" != 0 ] && extra="$extra --num-speculative-tokens $k --spec-accept-rate $(get acceptance.spec_p)"
  local g=$(get kv_offloading_gib); [ -n "$g" ] && extra="$extra --kv-offload-gib $g --kv-offload-gbps 12.7"
  rm -f $O/augur_steps.*.jsonl
  $PY -m augur.cli --model Qwen/Qwen3-1.7B --hardware RTX4060Ti --cost tables --num-gpu-blocks $(get num_gpu_blocks) \
    --max-num-seqs $(get max_num_seqs) --max-num-batched-tokens $(get max_num_batched_tokens) \
    --max-model-len $(get max_model_len) $extra --step-cpu-us 2000 $([ "${SHARED_KV:-off}" = on ] || echo --no-shared-kv) $([ -n "${AUGUR_TOKENS:-}" ] && echo "--replay-tokens $AUGUR_TOKENS" || echo --no-recorded-outputs) --filler-vocab 150000 --quiet \
    --workload $wl --num-entries $(get num_entries) --step-log-vllm $O/augur_steps --output $O/augur.csv > $O/augur.log 2>&1
}

blis_build() {
  [ -x $BASE/inference-sim/blis-ext ] && return
  local mod=$(ls -d $BASE/inference-sim/.gopath/pkg/mod/github.com/inference-sim/blis-vllm@*/)
  rm -rf $BASE/blis-vllm-ext && cp -r $mod $BASE/blis-vllm-ext && chmod -R u+w $BASE/blis-vllm-ext
  (cd $BASE/blis-vllm-ext && patch -p2 < $S/blis_vllm.patch) || return 1
  cd $BASE/inference-sim && cp go.mod $BASE/go.ext.mod && cp go.sum $BASE/go.ext.sum &&
    echo "replace github.com/inference-sim/blis-vllm => $BASE/blis-vllm-ext" >> $BASE/go.ext.mod &&
    PATH=$HOME/opt/go/bin:$PATH GOPATH=$PWD/.gopath GOCACHE=${TMPDIR:-/tmp}/blis-gocache GOTOOLCHAIN=local GOFLAGS=-mod=mod \
      GOPROXY=off go build -p 4 -modfile=$BASE/go.ext.mod -o blis-ext main.go
  rm -rf ${TMPDIR:-/tmp}/blis-gocache
}

# BLIS's host sends every request at priority 0 and refuses a priority policy: a copy of inference-sim with
# blis_priority.patch (a numeric SLO class is the request's priority), built beside it in inference-sim-prio.
blis_prio_build() {
  local d=$BASE/inference-sim-prio
  [ -x $d/blis-ext-prio ] && return
  mkdir -p $d && (cd $BASE/inference-sim && git archive HEAD) | tar -x -C $d && (cd $d && patch -p1 < $S/blis_priority.patch) || return 1
  (cd $d && PATH=$HOME/opt/go/bin:$PATH GOPATH=$BASE/inference-sim/.gopath GOCACHE=${TMPDIR:-/tmp}/blis-gocache GOTOOLCHAIN=local \
    GOFLAGS=-mod=mod GOPROXY=off go build -p 4 -modfile=$BASE/go.ext.mod -o blis-ext-prio main.go)
  rm -rf ${TMPDIR:-/tmp}/blis-gocache
}

blis() {
  blis_build || return 1
  local bin=$BASE/inference-sim/blis-ext
  [ "$(get scheduling_policy)" = priority ] && { blis_prio_build || return 1; bin=$BASE/inference-sim-prio/blis-ext-prio; }
  local mml=$(get max_model_len) flags="" k=$(get spec_k)
  # sessions: closed loop (think times in the trace) in the normal run, recorded arrivals in the replay
  if grep -q accumulate $O/blis$V.yaml; then
    [ -n "$V" ] && flags="--session-mode fixed-accumulate" || flags="--session-mode closed-loop"
  fi
  if [ "$k" != 0 ]; then
    flags="$flags --speculative-acceptance-length ${ACCEPT_LEN:-$(get acceptance.mean_tokens_per_step)}"; export BLIS_SPEC_K=$k
  fi
  [ "$(get async_scheduling)" = true ] && export BLIS_PRICER_NO_CPU=1 || unset BLIS_PRICER_NO_CPU
  # KV offload transfers as augur's control prices them: the PCIe rate measured on the real runs (12.7 GB/s), and
  # Qwen3-1.7B's block (28 layers x 2 x 8 KV heads x 128 x 2 bytes x 16 tokens)
  [ -n "$(get kv_offloading_gib)" ] && export BLIS_OFFLOAD_GBPS=12.7 BLIS_OFFLOAD_BLOCK_BYTES=1835008
  rm -f $O/blis_steps$V.jsonl
  BLIS_PRICER_CMD="env PYTHONPATH=$W VLLM_LOGGING_LEVEL=ERROR $PY $S/pricer.py serve --max-model-len $mml $PX ${PRICER_ARGS:-}" \
  BLIS_NUM_GPU_BLOCKS=$(get num_gpu_blocks) BLIS_STEP_LOG=$O/blis_steps$V.jsonl \
    $bin replay --scenarios $O/scen --scenario scenario.yaml --trace-header $O/blis$V.yaml \
      --trace-data $O/blis$V.csv $flags --output $O/blis_result$V.json > $O/blis$V.log 2>&1
  unset BLIS_SPEC_K BLIS_PRICER_NO_CPU BLIS_OFFLOAD_GBPS BLIS_OFFLOAD_BLOCK_BYTES
}

# Frontier and Vidur are given Llama-2-7B on an A100 to name a model and a chip: the price and the KV block count
# come from outside, so only the scheduler and the memory manager are theirs.
frontier() {
  local mml=$(get max_model_len)
  rm -rf $O/frontier_out$V
  (cd $BASE/Frontier && . .venv/bin/activate && PYTHONPATH=$PWD WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 AUGUR_WT=$W \
    python $S/frontier_predictor.py --step-log $O/frontier_steps$V.jsonl \
      --pricer-args "--max-model-len $mml --max-num-seqs $(get max_num_seqs) --max-num-batched-tokens $(get max_num_batched_tokens) $PX" -- \
      --simulation_mode online --sys_arch co-location --cc_backend_config_type analytical --cluster_config_num_replicas 1 \
      --replica_config_device a100 --replica_config_model_name meta-llama/Llama-2-7b-hf \
      --replica_config_attn_tensor_parallel_size 1 --replica_config_num_pipeline_stages 1 \
      --replica_scheduler_config_type vllm_v1 --vllm_v1_scheduler_config_batch_size_cap $(get max_num_seqs) \
      --vllm_v1_scheduler_config_max_tokens_in_batch $(get max_num_batched_tokens) \
      --vllm_v1_scheduler_config_max_model_len $mml --vllm_v1_scheduler_config_block_size 16 \
      --vllm_v1_scheduler_config_num_blocks_mode explicit --vllm_v1_scheduler_config_num_blocks $(get num_gpu_blocks) \
      --vllm_v1_scheduler_config_watermark_blocks_fraction 0 --vllm_v1_scheduler_config_enable_chunked_prefill \
      --$([ "$(get prefix_caching)" = false ] && echo no-)vllm_v1_scheduler_config_enable_prefix_caching \
      --vllm_v1_scheduler_config_scheduling_policy $(get scheduling_policy) --decode_cuda_graph_mode none \
      --request_generator_config_type trace_replay --trace_request_generator_config_trace_file $O/frontier$V.csv \
      --trace_request_generator_config_max_tokens $mml \
      --random_forrest_execution_time_predictor_config_enable_dummy_mode \
      --random_forrest_execution_time_predictor_config_dummy_execution_time_ms 1.0 \
      --metrics_config_output_dir $O/frontier_out$V --metrics_config_run_id r --metrics_config_write_metrics \
      --metrics_config_store_request_metrics --metrics_config_store_batch_metrics --no-metrics_config_store_plots \
      --no-metrics_config_enable_chrome_trace --no-metrics_config_write_json_trace) > $O/frontier$V.log 2>&1
}

# Vidur: its sarathi scheduler (chunked prefill; no priority, so a priority run is skipped, and no prefix cache, so a
# run with it off is the same to it). Its vllm scheduler cannot take these runs: prompts above its
# max_tokens_in_batch are never scheduled, and its padded batches fall outside the predictor's token grid.
vidur() {
  [ "$(get scheduling_policy)" = priority ] && { echo "vidur has no priority scheduling: not run" >&2; return 0; }
  local mml=$(get max_model_len)
  rm -rf $O/vidur_out$V
  (cd $BASE/vidur && . .venv/bin/activate && PYTHONPATH=$PWD WANDB_MODE=disabled AUGUR_WT=$W \
    python $S/vidur_predictor.py --step-log $O/vidur_steps$V.jsonl \
      --pricer-args "--max-model-len $mml --max-num-seqs $(get max_num_seqs) --max-num-batched-tokens $(get max_num_batched_tokens) $PX" -- \
      --replica_config_model_name meta-llama/Llama-2-7b-hf --replica_config_device a100 \
      --replica_config_network_device a100_pairwise_nvlink --cluster_config_num_replicas 1 \
      --replica_scheduler_config_type sarathi --sarathi_scheduler_config_batch_size_cap $(get max_num_seqs) \
      --sarathi_scheduler_config_chunk_size $(get max_num_batched_tokens) --sarathi_scheduler_config_block_size 16 \
      --sarathi_scheduler_config_num_blocks $(get num_gpu_blocks) --sarathi_scheduler_config_watermark_blocks_fraction 0 \
      --request_generator_config_type trace_replay --trace_request_generator_config_trace_file $O/vidur$V.csv \
      --trace_request_generator_config_max_tokens $mml --metrics_config_output_dir $O/vidur_out$V \
      --no-metrics_config_store_plots --no-metrics_config_enable_chrome_trace --metrics_config_store_request_metrics \
      --metrics_config_store_batch_metrics) > $O/vidur$V.log 2>&1
}

# the replay stage: Frontier and Vidur are synchronous, so not for an async real run
replay() {
  V=_replay; PX="--durations $(get step_log)"
  blis
  if [ "$(get async_scheduling)" = false ] && [ "$(get spec_k)" = 0 ] && [ -z "$(get kv_offloading_gib)" ]; then frontier; vidur; fi
  V=""; PX=""
}

compare() { $PY $S/compare.py $O --md $O/results.md --json $O/results.json > /dev/null; }

has prepare && prepare
[ -f $O/meta.json ] || { echo "run prepare first" >&2; exit 1; }
has augur && augur && $PY $S/workloads.py arrivals $O && for s in vidur frontier blis; do $PY $S/workloads.py $s $O; done
has blis && blis
# no async scheduling, speculative decoding or KV offload in Frontier and Vidur
plain=$([ "$(get spec_k)" = 0 ] && [ -z "$(get kv_offloading_gib)" ] && [ "$(get async_scheduling)" = false ] && echo 1 || echo 0)
{ has frontier && { [ $explicit = 1 ] || [ $plain = 1 ]; }; } && frontier
{ has vidur && { [ $explicit = 1 ] || [ $plain = 1 ]; }; } && vidur
has replay && replay
has compare && compare
exit 0
