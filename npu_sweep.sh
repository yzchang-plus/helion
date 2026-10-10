#!/bin/bash
# npu-dev 验收 sweep（rebase 后全量回归）
# 用户执行; 所有输出追加到 npu_sweep.log; per-op 日志 /tmp/npu_dev_sweep/<op>.log
# 用法: bash npu_sweep.sh
set -u
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR" || exit 1

LOG="$SCRIPT_DIR/npu_sweep.log"
exec >> "$LOG" 2>&1
echo "=== $(date '+%F %T') npu_sweep start (branch: $(git rev-parse --abbrev-ref HEAD), head: $(git rev-parse --short HEAD)) ==="

OUT=/tmp/npu_dev_sweep
mkdir -p "$OUT"
SUMMARY=$OUT/summary.txt
: > "$SUMMARY"

# ---- Step 0: 自动路由冒烟（不设 HELION_BACKEND，期望 NPU 上自动选 ascend）----
echo "--- Step 0: auto-routing smoke (no HELION_BACKEND) ---"
env -u HELION_BACKEND \
  HELION_LOGS=all \
  HELION_AUTOTUNE_EFFORT=none \
  ASCEND_RT_VISIBLE_DEVICES=0 \
  TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_smoke \
  timeout 900 python examples/add.py > "$OUT/smoke_add.log" 2>&1
SMOKE_CODE=$?
rm -rf /tmp/inductor_smoke
if [ "$SMOKE_CODE" -eq 0 ] && grep -q "using the 'ascend' backend" "$OUT/smoke_add.log"; then
  echo "ROUTING_SMOKE=PASS (auto triton->ascend routing confirmed, add.py exit 0)"
else
  echo "ROUTING_SMOKE=FAIL (exit=$SMOKE_CODE); see $OUT/smoke_add.log"
  grep -i "backend" "$OUT/smoke_add.log" | tail -5
fi

# ---- Step 1: 全量 sweep（环境与 PR4 基线完全一致）----
export HELION_BACKEND=ascend
export HELION_AUTOTUNE_EFFORT=none
export HELION_AUTOTUNE_IGNORE_ERRORS=1
export HELION_FORCE_AUTOTUNE=1
export ASCEND_LAUNCH_BLOCKING=1
export TRITON_BENCH_METHOD=npu
export HELION_NPU_MAX_TENSOR_NUMEL=4096
export HELION_NPU_UB_BUDGET_ELEMENTS=1024

# 顶层 examples/ 58 个 = PR4 基线 57 + upstream 新增 sparse_attn_indexer
EXAMPLES="add aot_compile_example aot_example attention batch_softmax \
bf16xint16_gemm blackwell_attention bmm broadcast_matmul concatenate \
cross_entropy embedding epilogue_subtiling exp flex_attention fp8_attention \
fp8_gemm fp8_matmul fused_linear_jsd gather_gemv gdn_fwd_h geglu grouped_gemm \
grpo_loss int4_gemm jagged_dense_add jagged_dense_bmm jagged_hstu_attn \
jagged_hstu_attn_2 jagged_layer_norm jagged_mean jagged_softmax jagged_sum \
jsd kl_div layer_norm long_sum low_mem_dropout mamba2_chunk_scan \
mamba2_chunk_state matmul matmul_layernorm matmul_split_k moe_matmul_ogs \
nvfp4_gemm nvfp4_gemv rms_norm rope se_block segment_reduction softmax \
sparse_attn_indexer split_k_barrier squeeze_and_excitation_net sum swiglu \
welford xsa"

# 子目录 examples/ 16 个 (PR4 基线未覆盖): acfs 1 + linear 8 + distributed 7
# distributed/ 入口硬断言 DEVICE.type=="cuda" 且需 torchrun+nccl, NPU 上预期快速失败
NESTED_EXAMPLES="acfs/softmax_acf \
linear/example_delta_rule linear/example_full_gla linear/example_gated_delta_rule \
linear/example_kda linear/example_mamba2_ssd linear/example_retention \
linear/example_simple_gla linear/example_vanilla_linear_attn \
distributed/all_gather_matmul distributed/all_reduce \
distributed/allreduce_bias_rmsnorm distributed/fp8_matmul_reduce_scatter \
distributed/fp8_scaled_all_gather_matmul distributed/matmul_reduce_scatter \
distributed/two_dim_parallel_matmul"

# 无 PR4 基线的算子 (新增/子目录), 失败不计为 REGRESSED
NO_BASELINE="sparse_attn_indexer acfs/softmax_acf \
linear/example_delta_rule linear/example_full_gla linear/example_gated_delta_rule \
linear/example_kda linear/example_mamba2_ssd linear/example_retention \
linear/example_simple_gla linear/example_vanilla_linear_attn \
distributed/all_gather_matmul distributed/all_reduce \
distributed/allreduce_bias_rmsnorm distributed/fp8_matmul_reduce_scatter \
distributed/fp8_scaled_all_gather_matmul distributed/matmul_reduce_scatter \
distributed/two_dim_parallel_matmul"

# FIFO 工作队列: token 为卡号 0..7, 每卡一个 worker 串行领任务
QUEUE=$(mktemp -u /tmp/sweep_queue.XXXXXX)
mkfifo "$QUEUE"
exec 3<>"$QUEUE"
rm -f "$QUEUE"
for c in 0 1 2 3 4 5 6 7; do echo $c >&3; done

run_one() {
  local ex=$1 card=$2   # ex 可为 "add" 或 "acfs/softmax_acf"
  local flat=${ex//\//_}
  # 每卡独立 inductor 缓存, 跑前清空（与基线方法学一致, 且并行互不干扰）
  local cdir=/tmp/inductor_card${card}
  rm -rf "$cdir"
  TORCHINDUCTOR_CACHE_DIR="$cdir" \
  ASCEND_RT_VISIBLE_DEVICES=$card \
    timeout -k 60 1800 python examples/${ex}.py > "$OUT/${flat}.log" 2>&1
  local code=$?
  if [ "$code" -eq 124 ]; then code="TIMEOUT"; fi
  flock "$SUMMARY" bash -c "echo \"${ex}|EXIT=${code}|CARD=${card}\" >> \"$SUMMARY\""
  echo $card >&3
}

echo "--- Step 1: full sweep ($(echo $EXAMPLES $NESTED_EXAMPLES | wc -w) examples, 8 cards) ---"
for ex in $EXAMPLES $NESTED_EXAMPLES; do
  read -u 3 card
  run_one "$ex" "$card" &
done
wait

echo "=== $(date '+%F %T') sweep finished ==="
echo ""
echo "===== SUMMARY (sorted by example name) ====="
sort "$SUMMARY"
echo ""
PASS=$(grep -c "|EXIT=0|" "$SUMMARY" || true)
FAIL=$(grep -vc "|EXIT=0|" "$SUMMARY" || true)
echo "PASS=$PASS FAIL=$FAIL TOTAL=$((PASS+FAIL))"
echo ""
echo "===== Comparison vs PR4 baseline (45 pass / 12 fail) ====="
# 基线 12 个失败
BASELINE_FAILS="blackwell_attention flex_attention fp8_attention fp8_gemm fp8_matmul nvfp4_gemv nvfp4_gemm jsd int4_gemm rope layer_norm squeeze_and_excitation_net"
echo "-- Baseline failures and their current result:"
for ex in $BASELINE_FAILS; do
  line=$(grep "^${ex}|" "$SUMMARY" || echo "${ex}|MISSING")
  echo "  $line"
done
echo "-- NEW failures (passed at baseline, failing now) --"
for line in $(grep -v "|EXIT=0|" "$SUMMARY" | cut -d'|' -f1); do
  if ! echo " $BASELINE_FAILS " | grep -q " $line " \
     && ! echo " $NO_BASELINE " | grep -q " $line "; then
    echo "  REGRESSED: $(grep "^${line}|" "$SUMMARY")"
  fi
done
echo "-- NEW passes (failed at baseline, passing now) --"
for ex in $BASELINE_FAILS; do
  if grep -q "^${ex}|EXIT=0|" "$SUMMARY"; then
    echo "  FIXED: $ex"
  fi
done
echo "-- No-baseline examples (new upstream + subdirs; failures are NOT regressions):"
for ex in $NO_BASELINE; do
  echo "  $(grep "^${ex}|" "$SUMMARY" || echo "${ex}|MISSING")"
done
echo ""

# ---- Step 2: pretuned_kernels/ (24 个; PR4 基线未覆盖; AOT heuristic 仅 cuda sm90/sm100) ----
PT_SUMMARY=$OUT/pretuned_summary.txt
: > "$PT_SUMMARY"

PT_KERNELS=""
for dir in pretuned_kernels/*/; do
  name=$(basename "$dir")
  if [ -f "pretuned_kernels/${name}/${name}.py" ]; then
    PT_KERNELS="$PT_KERNELS $name"
  else
    echo "pretuned: SKIP $name (no ${name}.py entry point)"
  fi
done

run_pretuned() {
  local name=$1 card=$2
  local cdir=/tmp/inductor_pt_card${card}
  rm -rf "$cdir"
  TORCHINDUCTOR_CACHE_DIR="$cdir" \
  ASCEND_RT_VISIBLE_DEVICES=$card \
    timeout -k 60 1800 python "pretuned_kernels/${name}/${name}.py" \
      > "$OUT/pretuned_${name}.log" 2>&1
  local code=$?
  if [ "$code" -eq 124 ]; then code="TIMEOUT"; fi
  flock "$PT_SUMMARY" bash -c "echo \"${name}|EXIT=${code}|CARD=${card}\" >> \"$PT_SUMMARY\""
  echo $card >&3
}

echo "--- Step 2: pretuned_kernels sweep ($(echo $PT_KERNELS | wc -w) kernels, 8 cards; no PR4 baseline) ---"
for name in $PT_KERNELS; do
  read -u 3 card
  run_pretuned "$name" "$card" &
done
wait

echo "===== PRETUNED KERNELS SUMMARY ====="
sort "$PT_SUMMARY"
PT_PASS=$(grep -c "|EXIT=0|" "$PT_SUMMARY" || true)
PT_FAIL=$(grep -vc "|EXIT=0|" "$PT_SUMMARY" || true)
echo "PRETUNED: PASS=$PT_PASS FAIL=$PT_FAIL TOTAL=$((PT_PASS+PT_FAIL))"
echo ""
echo "NOTE: benchmarks/ skipped (tritonbench not installed); notebooks/ skipped (jupyter not installed)."
echo "SWEEP COMPLETE"
