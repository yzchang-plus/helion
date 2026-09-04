#!/bin/bash
# npu-dev 验收 sweep 第二轮:
#  A) examples/linear 8 个 (第一轮用文件路径跑法错误, 须 -m 模块方式)
#  B) pretuned_kernels 24 个 (第一轮 _bench.py 硬编码 torch.cuda 全灭; 已打补丁)
#  C) epilogue_subtiling 单独长跑 (第一轮 30min 超时; 实测非挂死而是极慢)
# 用户执行; 输出追加到 npu_sweep2.log
set -u
cd /workspace/work/helion || exit 1

LOG=/workspace/work/helion/npu_sweep2.log
exec >> "$LOG" 2>&1
echo "=== $(date '+%F %T') npu_sweep2 start (head: $(git rev-parse --short HEAD)) ==="

OUT=/tmp/npu_dev_sweep2
mkdir -p "$OUT"

export HELION_BACKEND=ascend
export HELION_AUTOTUNE_EFFORT=none
export HELION_AUTOTUNE_IGNORE_ERRORS=1
export HELION_FORCE_AUTOTUNE=1
export ASCEND_LAUNCH_BLOCKING=1
export TRITON_BENCH_METHOD=npu
export HELION_NPU_MAX_TENSOR_NUMEL=4096
export HELION_NPU_UB_BUDGET_ELEMENTS=1024

# FIFO 卡队列 0..7
QUEUE=$(mktemp -u /tmp/sweep2_queue.XXXXXX)
mkfifo "$QUEUE"
exec 3<>"$QUEUE"
rm -f "$QUEUE"
for c in 0 1 2 3 4 5 6 7; do echo $c >&3; done

LINEAR="example_delta_rule example_full_gla example_gated_delta_rule example_kda \
example_mamba2_ssd example_retention example_simple_gla example_vanilla_linear_attn"

PT_KERNELS=""
for dir in pretuned_kernels/*/; do
  name=$(basename "$dir")
  [ -f "pretuned_kernels/${name}/${name}.py" ] && PT_KERNELS="$PT_KERNELS $name"
done

LSUM=$OUT/linear_summary.txt
PSUM=$OUT/pretuned_summary.txt
: > "$LSUM"; : > "$PSUM"

run_linear() {
  local mod=$1 card=$2
  local cdir=/tmp/inductor_l2_card${card}
  rm -rf "$cdir"
  TORCHINDUCTOR_CACHE_DIR="$cdir" \
  ASCEND_RT_VISIBLE_DEVICES=$card \
    timeout -k 60 1800 python -m examples.linear.${mod} \
      > "$OUT/linear_${mod}.log" 2>&1
  local code=$?
  [ "$code" -eq 124 ] && code="TIMEOUT"
  flock "$LSUM" bash -c "echo \"linear/${mod}|EXIT=${code}|CARD=${card}\" >> \"$LSUM\""
  echo $card >&3
}

run_pretuned() {
  local name=$1 card=$2
  local cdir=/tmp/inductor_p2_card${card}
  rm -rf "$cdir"
  TORCHINDUCTOR_CACHE_DIR="$cdir" \
  ASCEND_RT_VISIBLE_DEVICES=$card \
    timeout -k 60 1800 python "pretuned_kernels/${name}/${name}.py" \
      > "$OUT/pretuned_${name}.log" 2>&1
  local code=$?
  [ "$code" -eq 124 ] && code="TIMEOUT"
  flock "$PSUM" bash -c "echo \"${name}|EXIT=${code}|CARD=${card}\" >> \"$PSUM\""
  echo $card >&3
}

echo "--- Step A: linear examples via -m ($(echo $LINEAR | wc -w) modules) ---"
for mod in $LINEAR; do
  read -u 3 card
  run_linear "$mod" "$card" &
done
wait

echo "--- Step B: pretuned_kernels with device-agnostic harness ($(echo $PT_KERNELS | wc -w) kernels) ---"
for name in $PT_KERNELS; do
  read -u 3 card
  run_pretuned "$name" "$card" &
done
wait

echo "--- Step C: epilogue_subtiling standalone (60min timeout, card 0) ---"
rm -rf /tmp/inductor_epi
TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_epi \
ASCEND_RT_VISIBLE_DEVICES=0 \
  timeout -k 60 3600 python examples/epilogue_subtiling.py > "$OUT/epilogue_subtiling_long.log" 2>&1
EPI=$?
[ "$EPI" -eq 124 ] && EPI="TIMEOUT"
echo "epilogue_subtiling|EXIT=${EPI}"

echo ""
echo "===== LINEAR SUMMARY ====="
sort "$LSUM"
echo "LINEAR: PASS=$(grep -c '|EXIT=0|' "$LSUM" || true) FAIL=$(grep -vc '|EXIT=0|' "$LSUM" || true)"
echo ""
echo "===== PRETUNED SUMMARY (rerun, patched harness) ====="
sort "$PSUM"
echo "PRETUNED: PASS=$(grep -c '|EXIT=0|' "$PSUM" || true) FAIL=$(grep -vc '|EXIT=0|' "$PSUM" || true)"
echo ""
echo "===== EPILOGUE ====="
echo "epilogue_subtiling|EXIT=${EPI}"
echo "SWEEP2 COMPLETE"
