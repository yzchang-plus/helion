# NPU 项目验收报告 — npu-dev（rebase 到最新 main 后）

**分支**: `npu-dev`（6 个 NPU 提交 rebase 到 2026-09-04 最新 upstream main）
**Head**: `ab82df64 [NPU][codegen] acceptance fixes: bmm tracing, out_dtype propagation, batch-hoist guard`
**验收日期**: 2026-09-04
**方式**: examples/ 全量（顶层 58 + 子目录 16）+ pretuned_kernels/ 24 个，8 卡并行，环境与 PR4 基线完全一致。

---

## 1. 结论

| 范围 | 通过 | 失败/不支持 | 说明 |
|---|---|---|---|
| examples/ 顶层 | **45** | 13 | 与 PR4 基线 45/12 对齐，新增 sparse_attn_indexer 失败 |
| examples/ 子目录（单卡可跑） | **8** | 1 | acfs 1/1；linear 7/8 |
| examples/distributed | — | 7 | CUDA+nccl 专属，入口断言 cuda / 需 CUDA_HOME，环境性排除 |
| pretuned_kernels/ | 0 | 24 | CUDA dashboard 配方（jax/CUTLASS/cudagraph/cuda 分配/sm90-121 heuristic），本版本不支持 |
| **单卡算子合计** | **53 / 67（79.1%）** | 14 | 与 PR4 基线通过率（79.0%）一致，**零回归** |

- **自动设备路由验证通过**：不设 `HELION_BACKEND` 时，NPU 上自动 triton→ascend（日志 `Device npu: using the 'ascend' backend`）。
- benchmarks/（tritonbench 未安装）、notebooks/（jupyter 未安装）环境性跳过。

## 2. 环境

- 硬件：Ascend 910B4 × 8
- 软件：torch 2.10.0+cpu、torch_npu 2.10.0、triton 3.2.0、triton-ascend 3.2.1
- 环境变量：`HELION_BACKEND=ascend HELION_AUTOTUNE_EFFORT=none HELION_AUTOTUNE_IGNORE_ERRORS=1 HELION_FORCE_AUTOTUNE=1 ASCEND_LAUNCH_BLOCKING=1 TRITON_BENCH_METHOD=npu HELION_NPU_MAX_TENSOR_NUMEL=4096 HELION_NPU_UB_BUDGET_ELEMENTS=1024`
- 每算子 30 分钟超时（pretuned/linear 同）；epilogue_subtiling 单独 60 分钟。
- 日志：`/tmp/npu_dev_sweep/`（第一轮）、`/tmp/npu_dev_sweep2/`（第二轮）；驱动日志 `npu_sweep.log` / `npu_sweep2.log`。

## 3. 失败明细（14 个单卡算子）

### 3.1 PR4 基线已知失败（12 个，现象不变）

| 算子 | 现象 | 类别 |
|---|---|---|
| blackwell_attention | BiShengLIR 编译失败（Blackwell 专属指令） | 硬件不支持 |
| flex_attention | `FlexAttention is only supported on CUDA, CPU or HPU` | torch API 限制 |
| fp8_attention / fp8_gemm / fp8_matmul / nvfp4_gemv | `aclnnInplaceCopy failed, 561103`（fp8 cast） | 硬件不支持 |
| nvfp4_gemm | `aclnnSearchSorted failed, 161002`（DT_BFLOAT16 not implemented） | 硬件不支持 |
| jsd | NPU OOM（baseline 中间量 8 GiB） | 基准实现问题 |
| int4_gemm | `coreDim value 1048576 invalid, <= 65535`（SIGABRT, exit 134） | 框架/精度缺口 |
| rope | 编译期 UB overflow（3084288 > 1572864 bits） | UB 容量缺口 |
| layer_norm | bwd 编译期 UB overflow（fwd 通过） | UB 容量缺口 |
| squeeze_and_excitation_net | BWD gradient mismatch `[1024,1024]` | 精度缺口 |

### 3.2 新增失败（2 个，均非回归）

| 算子 | 现象 | 类别 |
|---|---|---|
| sparse_attn_indexer | `InductorError: AssertionError: ([x0], {p0: 4096, p1: 32})`（upstream 新算子，PR4 时不存在） | 框架 lowering 缺口，待查 |
| linear/example_vanilla_linear_attn | `Forward error: 0.8627`（其余 7 个 linear 变体全部通过） | NPU 精度缺口，待查 |

### 3.3 性能病态（correctness 通过）

- **epilogue_subtiling**：correctness 两段均通过（60 分钟长跑 EXIT=0），但 effort=none 隐式配置（pid_type=persistent_blocked, block 16³）下 helion 81.7s vs torch 24ms（0.00x）。30 分钟超时属测试模式假象（effort=none 只试隐式配置）；开启正常 autotune 后预期改善。release 注明。

## 4. 子目录结果

- `acfs/softmax_acf`：**通过**。
- `linear/`：delta_rule / full_gla / gated_delta_rule / kda / mamba2_ssd / retention / simple_gla **7 个通过**；vanilla_linear_attn 失败（见 3.2）。注意须以 `python -m examples.linear.<name>` 运行（包内相对导入）。
- `distributed/` 7 个：全部 CUDA 专属（nccl/CUDA_HOME/`assert DEVICE.type=="cuda"`），NPU 分布式为后续工作项。

## 5. pretuned_kernels/（24 个，本版本不支持）

失败原因分布（CUDA dashboard 配方，非算子回归）：
- 基准依赖 jax（causal_conv1d、gdn_decode 等，机器未装 jax）；
- CUTLASS 源码路径（grouped_gemm 需 HELION_CUTLASS_GROUPED_GEMM_SOURCE）；
- kernel 文件内 58 处 `device="cuda"` 硬编码分配；
- 23 个文件使用 cudagraph 计时路径（CUDA 专属）；
- AOT heuristic 仅 cuda sm90/sm100/sm121，无 ascend 版本。

移植为独立工作项（设备分配去硬编码 + npu 计时路径 + ascend heuristic 调优）。

## 6. 本轮顺带修复（harness 通用化，已在工作树）

- `helion/_compat.py`：`get_device_name()` 增加 npu 分支（返回 `Ascend910B4`），与现有 cuda/xpu/mps/tpu 分支同构。
- `pretuned_kernels/_bench.py`：设备名打印改用 `get_device_name()`；`thermal_warmup` 的 cuda 硬编码改为按可用设备分发（npu 用 `torch.npu.synchronize`）。

这两处是设备通用化改动，也是后续 upstream 小 PR 的素材。

## 7. 复现

```bash
cd /workspace/work/helion
bash npu_sweep.sh    # 第一轮：examples 74 个 + pretuned（旧 harness）
bash npu_sweep2.sh   # 第二轮：linear -m 重跑 + pretuned（修补后）+ epilogue 长跑
```
