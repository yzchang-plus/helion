# 代码改动记录（按分支区分；待用户在另一环境应用后提交）

> 本环境只做验证和草稿。每条改动含文件、位置、内容、原因、验证结果。
> 附带产物：`npu-dev-backend-hooks.patch`（npu-dev 改动的完整 git diff，可 `git apply`）。

---

# 第一部分：backend-generic-mechanisms 分支（社区 PR 分支）的改动

分支状态：本地 `backend-generic-mechanisms`，HEAD `fdbc4a43`（4 个通用机制提交），
基于 `6a1bddbe`。以下改动为**未提交的工作区改动**，需要应用到该分支后再提 PR。

> **快捷应用方式**：G1–G8 的完整 diff 已导出为
> **`backend-generic-mechanisms-fixes.patch`**（9 文件，+143/−56，基线 `fdbc4a43`），
> 在另一环境的 backend-generic-mechanisms 分支上执行
> `git apply backend-generic-mechanisms-fixes.patch` 即可一次到位。
> 下文各条目是逐项说明（含原因与验证），供 review 时对照。

## G1. `helion/_compat.py` L572 — import 恢复 main 的括号多行形式

- **改动**：
  ```diff
  -        from .fb.mtia_tunables import supports_mtia_tunables as _fb_supports_mtia  # pyrefly: ignore [missing-import]
  +        from .fb.mtia_tunables import (  # pyrefly: ignore [missing-import]
  +            supports_mtia_tunables as _fb_supports_mtia,
  +        )
  ```
- **原因**：分支提交（`be831546` 触碰 `_compat.py` 时）把 main 的括号多行 import 压成了
  91 字符单行，超过 88 列限制，`ruff check` 报 I001，会挡 CI。
- **验证**：`ruff check --no-fix` → All checks passed；`ruff format --check` → 497 文件通过。

---

## autoreview 对 4 个提交的 review（2026-10-12）→ G2–G8 修复

> 运行方式：`IS_SANDBOX=1 ./scripts/autoreview.py --no-codex "the 4 commits from
> 6a1bddbe to HEAD (git diff 6a1bddbe..HEAD) on branch backend-generic-mechanisms"`
> （log：`.logs/autoreview/20261012-121957-175833/`）。结论非 LGTM，共 6 个实质发现。
> 其中 5 个已修复（G2–G6），2 个 nit 已顺手修（G7–G8），1 个暂不修（见"未修复项"）。
> 以下全部是**未提交的工作区改动**。

### G2. hf32 被 settings 接受却在 `map_dot_precision` 被静默丢弃（发现 1）

- **`helion/_compiler/backend.py`**：`map_dot_precision` 的映射表加 `"hf32": "hf32"`。
  原状态：`DotPrecision` Literal 含 `"hf32"`（fdbc4a43），但 base 映射没有该键 → 返回
  `""` → `matmul_utils.py` 两处消费方都把 `""` 当"省略 input_precision"→ 用户显式设置
  hf32 是静默 no-op。修复后：CUDA 上会得到 triton 的显式 assert（好过静默），
  声明 `allowed_dot_input_precisions` 含 hf32 的 out-of-tree driver（如 ascend）则真正生效。
- **`helion/runtime/settings.py`**：`dot_precision` 的 settings 文档串补 `'hf32'`
  （原串在 fdbc4a43 后过期）。

### G3. 显式 `backend="triton"` 被设备路由静默改写（发现 2）

- **`helion/runtime/settings.py`**：新增非 dataclass 字段 `_backend_explicit`
  （仿照已有 `_cute_structural_origins` 的模式）：`__init__` 里在 fallback 之前记录
  `"backend" in settings`；`__setstate__` 里对旧序列化对象兜底为 True（与
  `_cute_structural_origins` 的兜底注释同理：旧序列只含生效值，视为显式）。
- **`helion/_compiler/compile_environment.py`**：路由条件从
  `backend_name == "triton" and not os.environ.get("HELION_BACKEND")` 改为
  再加 `and not settings._backend_explicit`。原状态：`@helion.kernel(backend="triton")`
  或 `Settings(backend="triton")` 与"没设置"无法区分，在注册了 out-of-tree 后端的
  设备上被静默改路由。
- **验证**：`Settings()._backend_explicit is False`、`Settings(backend="triton")._backend_explicit
  is True`、deepcopy/pickle round-trip 保持。

### G4. `_make_l2_cache_clearer` 仍会崩（发现 3，be831546 修漏的）

- **`helion/autotuner/benchmarking.py`**：在 `active.get_empty_cache_for_benchmark()`
  之前加 hasattr 守卫（缺 `clear_cache`/`get_empty_cache_for_benchmark` 任一 → 返回
  no-op），与 `safe_clear_cache()` 契约一致。原状态：4 个 generic bench 在
  `torch.cuda.is_available()==True` 而 driver 无该原语（如 CUDA+NPU 混装机器跑
  ascend）时仍会 AttributeError。

### G5. `safe_clear_cache` 收窄异常 + 缓冲区只分配一次（发现 4）

- **`helion/_compat.py`**：重构为：
  - 异常只包住 driver 解析段（`ImportError/AttributeError/RuntimeError`——
    RuntimeError 覆盖 triton `_create_driver` 在 0 或 ≥2 个 active driver 时抛出的错，
    这是原宽 except 真正兜住的情况，纯 CPU/混装机器依赖它）；flush 本身的失败
    （如分配 flush buffer 的 OOM）不再被吞，向上传播。
  - 新增 `_BenchmarkCacheDriver` Protocol + `_driver_flush_cache`：flush buffer
    按 driver 只分配一次（**身份比较**而非 dict/lru_cache 键——driver 对象和测试
    mock 的 SimpleNamespace 可能不可哈希），避免每次 timed iteration 都分配、
    首次分配落进 timed 估计循环。
- **注意**：test_benchmark_worker 有个测试 mock driver 为 SimpleNamespace，
  第一次用 lru_cache 实现会 TypeError: unhashable——已因此改成身份比较的
  模块级 `(driver, buffer)` 槽。另外 pyrefly 会报 `LazyProxy` 不满足 Protocol
  参数（运行时靠 `__getattr__` 转发，静态不可见）——`_driver_flush_cache` 参数
  类型用 `object`，函数体内 `cast("_BenchmarkCacheDriver", driver)`；全量
  pyrefly 与基线零差异。

### G6. clamp 与 `max_contiguous` hint 冲突 + 三处重复（发现 5）

- **`helion/_compiler/indexing_strategy.py`**：
  - 新增模块级 `_clamps_masked_offsets(indexing)` 和 `_clamped_offset_expr(indexing, offset)`
    （含 docstring：仅 pointer indexing strategy 走这里）。
  - `codegen_load`：clamp 提前到 contiguity hint **之前**，且 clamping 时**跳过**
    `max_contiguous` hint（选了 reviewer 两个方案中保守的一个：`tl.where` 产生的
    值序列不连续，clamp 后再断言 contiguity 不健全；原顺序 hint 则被 where 包住
    永远传播不出去，等于死代码）。附注释说明。
  - `codegen_store`/`codegen_atomic`：内联 clamp 块替换为 `_clamped_offset_expr`
    一行（消三处重复；atomic 处顺带删掉只为此而取的 `backend` 局部变量）。
- **`helion/_compiler/backend.py`**：`clamp_masked_pointer_offsets` docstring 补
  "仅 pointer indexing strategy 生效，block-ptr/descriptor 不受影响，启用它的后端
  应同时把那些 indexing mode 从搜索空间排除"。

### G7. `statement_from_string` docstring 去重（nit）

- **`helion/_compiler/ast_extension.py`**：单数版 docstring 改为一句话引用
  `statements_from_string`（placeholder 语法和 f-string 转义规则只在复数版保留）；
  复数版删掉对单数版的反向引用（避免循环引用）。原单数版示例还在用
  `expr_from_string`，且两份逐字重复。

### G8. `tile_strategy.py` num_warps 延迟时的保守路径加注释（nit）

- **`helion/_compiler/tile_strategy.py`** ~L7386：`effective_num_warps(config) == 1`
  的分支加一行注释：deferred num_warps（None）走下面的保守多 warp 路径，安全但
  放弃这个单 warp 流水线优化。

### 新增测试：launcher 对 None 的 num_warps/num_stages 省略（nit"无测试"的部分回应）

- **`test/test_runtime.py`**：`TestTritonLauncher` 新增
  `test_none_num_warps_and_stages_are_omitted_from_run`——None 时 `run` kwargs 里
  **不含**这两个键（而非传 None），直接覆盖 4339313d 的 launcher 侧机制。
  复用同 class 已有 FakeJITFunction 模式。注意：无 `_minimum_resident_programs`
  时 launcher 只调一次 `run`（warmup 只有 residency 路径才走）。

### 未修复项（有意，供 PR 决策）

- **发现 6**（deferred num_warps + `triton_minimum_resident_programs` 只在 launch 时
  报错而非 config 时拒绝）：launch-time RuntimeError 信息清晰（"require an explicit
  num_warps"），config 时拒绝需要动 ConfigSpec 管线、扩大 PR 范围。建议作为
  follow-up，PR 描述里可提一句。
- **nit**：progress-bar 改动（stderr console + 非 tty 跳过）混在 be831546 里，
  与提交标题无关。改历史需要 rebase，由你在另一环境决定：要么 PR 描述说明，
  要么 rebase 拆出独立提交。
- **nit**：新机制测试仍不全（设备路由、clamp codegen、`inline_constexpr_at_module_level`、
  `statements_from_string` 多语句模板等）。launcher None 省略已补（见上）；其余属
  PR 范围决策，reviewer 建议路由/AST helper 均 CPU 可测。
- **观察**（非缺陷）：in-tree 只有 TritonBackend 声明 `device_types`，stock 安装
  路由永远不触发；pallas(tpu)/metal 用户仍需 HELION_BACKEND。是否给 in-tree
  非 CUDA 后端声明 `device_types` 是后续决策。

## 通用分支验证结论（2026-10-12，本 NPU 机器）

| 检查 | 结果 | 结论 |
|---|---|---|
| `ruff check` / `ruff format` | G1–G8 全部在位后全绿（498 文件） | ✅ |
| pytest 4 文件（codegen_comments / benchmark_worker / benchmarking_batched_replay / config_api）+ test_runtime | G2–G8 后：test_runtime+config_api+codegen_comments 101 passed / 3 skipped；benchmark 两文件 76 passed / 5 skipped / 2 failed | 2 个失败在 **main 上同样失败**（已实测对照），系本机 torch 无 CUDA 构建、`_testing.DEVICE` 兜底 cuda 所致，**非分支回归**；CUDA CI 上会正常跑 |
| 相关模块扩展 sweep（settings/indexing/clamp/routing/ast_extension 关键字） | 改动前后完全一致（113 failed / 48 passed，均为本机环境失败，stash 对照实测） | ✅ 零新增失败 |
| pyrefly 全量（`scripts/pyrefly_check.sh`） | 改动前后均 493 errors（本机 cutlass/jax 缺失等环境误报；中途发现 1 个新增 LazyProxy/Protocol 报错，已用 cast 修复后归零） | ✅ 与基线零差异 |
| codespell（`lint.sh check` 的第 3 项） | 用户安装 codespell 2.4.3 后 `pre-commit run codespell --all-files` → Passed | ✅ |
| `Settings` 冒烟 | `_backend_explicit` 默认 False / 显式 True；deepcopy、pickle round-trip 保持 | ✅ |
| `./scripts/autoreview.py --no-codex`（未提交改动） | G1 被确认与 main 字节一致、ruff+pyrefly 干净 | ✅ |
| `./scripts/autoreview.py --no-codex`（4 个提交，TARGET 模式） | 6 实质发现 + 若干 nit → G2–G8 修复完毕 | ✅ 已处理；**建议提 PR 前再跑一次确认 LGTM** |
| examples/add.py（无 HELION_BACKEND） | 与 main 相同报错（无 CUDA） | ✅ 与 main 行为一致 |
| `./lint.sh check` 4 项（全量） | ruff format ✅ / ruff check ✅ / pyrefly ✅（零新增）/ codespell ✅ | ✅ **本机全部通过** |

**修复后复跑 autoreview（可选但推荐）**——G2–G8 应用后确认无新问题：

```
IS_SANDBOX=1 ./scripts/autoreview.py --no-codex
```
（默认模式即覆盖 G1–G8 这批未提交改动；4 个提交本身的问题已在上轮修复。）

**提 PR 前建议自查**：初版 review 曾点名 `safe_clear_cache` 的宽 `except Exception: pass`
会掩盖真实失败——该问题已由 **G5** 修复（异常收窄到 driver 解析段，flush 失败传播，
docstring 写明 no-op 的适用条件：driver 缺原语 / 无法解析 active driver）。此段仅作历史
记录。

---

# 第二部分：npu-dev 分支的改动（已保存，工作区已还原）

> 这批改动已于 2026-10-12 应用到 npu-dev 工作区验证通过后，用 `git restore .` 还原。
> 完整 diff 保存在 **`npu-dev-backend-hooks.patch`**（31 文件，+223/-97，基线 npu-dev @ `74510e3f`），
> 在 npu-dev 上 `git apply npu-dev-backend-hooks.patch` 即可恢复。
> 注意：其中与通用分支 `4339313d`（num_warps/num_stages 延迟）同主题的文件，
> 未来 npu-dev rebase 到通用分支上时会有少量交叠，属预期。
>
> **2026-10-12 v2（autoreview 发现的 2 个缺陷已修复，patch 已重新生成）：**
>
> 1. **抑制注释位置（30 个 import 语句）**：v1 patch 的括号化 import 把
>    `# pyrefly: ignore` / `# type: ignore` 放在**内层行**——pyrefly 把
>    missing-import 报在语句首行，内层注释压不住（reviewer 实测验证）。
>    修复：22 处直接恢复 main 原文（main 本来就是正确的首行注释形式）；
>    8 处 main 上没有对应形式（`_compat.py` 的 torch_npu 导入 ×2、
>    `template_buffer.py`、`autotuner_heuristics/cute.py`、`type_info.py`、
>    `loops.py`、`test_remote_copy.py`），手动把注释挪到首行
>    （`from x import (  # type: ignore[...]` 形式，ruff format 稳定）。
> 2. **搜索空间/降级不对称**（`config_spec.py`）：`valid_indexing_types()` /
>    `valid_atomic_indexing_types()` 先查机器级 `supports_tensor_descriptor()`，
>    混装机器（CUDA+NPU）上 `HELION_BACKEND=ascend` 时 `tensor_descriptor`
>    会被加进 ascend 搜索空间，而 downgrade 会把它降级——空间与降级不一致。
>    修复：两个方法的结果再过一遍 `backend.supports_indexing_mode()` 过滤，
>    使搜索空间与 downgrade 行为按构造一致。
>
> v2 验证（临时 worktree，npu-dev @ `74510e3f` + patch v2）：
> `ruff check` / `ruff format --check` 全绿；`pyrefly check helion/_compat.py`
> 仅剩 2 处原有 triton-ascend 环境误报（非 patch 触碰行），torch_npu/mtia
> 报错已被首行注释正确压制；pytest 4 文件 170 passed / 10 skipped（与 v1
> 验证结果一致）；`git apply --check` 于干净 npu-dev 工作区通过。

## A. 移除 5 处 `hasattr(torch, "npu")` 全局探测，改为后端钩子（修复 test_config_api 在 NPU 环境的 6 个失败）

**根因**：npu-dev 在通用代码里用 `hasattr(torch, "npu") and torch.npu.is_available()` 做全局
设备探测，环境装了 torch_npu 就会改变行为——即使构造的是 CUDA/cute 的 ConfigSpec。

### A1. `helion/runtime/config.py` — 整体回退 main 版

- `git restore --source=main`：删 `import torch`、删 `Config.__init__` 的 NPU 条件块
  （往 config dict 塞 `num_warps/num_stages=None`）、property 恢复 `-> int`。
- 原因：破坏 `from_json(to_json())` round-trip。None 延迟语义移到 A2/A4 钩子。

### A2. `helion/_compiler/backend.py` — 钩子调整/新增

- `effective_num_warps` 文档改通用措辞（与通用分支 `4339313d` 文案一致）
- 新增 `effective_num_stages(config) -> int | None`（默认 `config.num_stages`）
- 新增 `supports_indexing_mode(mode) -> bool`（默认 True；管 normalize 降级，区别于管搜索空间的
  `supports_block_ptr_indexing`）
- 新增 `allowed_pid_types() -> tuple[str, ...] | None`（默认 None = 不限制）

### A3. `helion/_compiler/triton/backend.py`

- `launcher_keyword_args`：`f"num_stages={config.num_stages}"` → `f"num_stages={self.effective_num_stages(config)}"`

### A4. `helion/_compiler/ascend/backend.py` — AscendBackend 声明 NPU 能力

```python
def effective_num_warps(self, config: Config) -> int | None:
    # triton-ascend binds the warp count itself; defer to the driver's
    # own default unless the config explicitly pins a value.
    return cast("int | None", config.get("num_warps"))

def effective_num_stages(self, config: Config) -> int | None:
    # triton-ascend binds the pipeline depth itself; defer to the
    # driver's own default unless the config explicitly pins a value.
    return cast("int | None", config.get("num_stages"))

def supports_block_ptr_indexing(self) -> bool:
    # triton-ascend claims block_ptr support but can hit device-side
    # faults (e.g. unaligned UUB access); exclude it from the search
    # space and downgrade pinned values.
    return False

def supports_indexing_mode(self, mode: str) -> bool:
    # Only pointer indexing is guaranteed to compile on triton-ascend.
    return mode == "pointer"

def allowed_pid_types(self) -> tuple[str, ...] | None:
    # Ascend grids cap the non-leading core dimensions at 65535; drop
    # the "xyz" layout whose y/z dimensions can exceed that limit.
    return ("flat", "persistent_blocked", "persistent_interleaved")
```
另加 `from typing import cast`、TYPE_CHECKING 下 `from ..runtime.config import Config`。

**语义对照**（与原 hack 逐一等价）：num_warps/stages 未指定→None 延迟、显式→透传；
indexing pinned block_ptr/tensor_descriptor→"pointer"；pid_type pinned xyz→"flat"。

### A5. `helion/autotuner/config_spec.py` — 四处探测改钩子

1. `get_valid_eviction_policies`：删 triton 分支 NPU 探测（恢复 main）。AscendBackend
   `name="ascend"` 本就走末尾 `("",)` 兜底，NPU 行为不变
2. `ConfigSpec.__init__`：NPU pid 探测 → `backend.allowed_pid_types()` 非 None 则过滤
3. `valid_indexing_types`：删 NPU 探测；NPU 由 `supports_block_ptr_indexing()->False` 得 `("pointer",)`
4. `downgrade_unsupported_indexing`：删 NPU-only guard → 按 `backend.supports_indexing_mode(v)`
   逐值降级 "pointer"；CUDA 默认钩子恒 True 永不降级（不误伤 H100 pinned block_ptr）
5. `normalize` 的 pid_type 改写：NPU 探测 → `backend.allowed_pid_types()` 非 None 且值不在其中才降级

### A 组验证（npu-dev 工作区，改动在位时实测）

- pytest 4 文件：**170 passed / 10 skipped**（改动前 6 个失败）
- `ruff check` / `ruff format`：全绿
- NPU 端到端：`HELION_BACKEND=ascend` 下 add.py、matmul.py exit 0；`HELION_PRINT_OUTPUT_CODE=1`
  确认生成代码 `num_warps=None, num_stages=None`（延迟给 triton-ascend driver）

## B. npu-dev 的 ruff 修复（随 patch 保存）

- 31 文件 import 重排（I001 自动修复）
- 3 个 kda benchmark 文件恢复 main 原版（ruff 修 I001 时会误删行尾 `# noqa: E402` 引发
  E402/I001 死循环；且 npu-dev 合并时把这些 main 的括号多行 import 压成了超长单行——
  与通用分支 G1 同源同类的问题）

## npu-dev 遗留项（未处理，建议后续按同样模式抽取）

config_spec.py 等处仍有 ~12 处 NPU 探测：`config_spec.py` L4453/4481/4631/5225/5304/6523/6694、
`compile_environment.py` L1167、`matmul_utils.py` L379、`type_info.py` L184/957、
`triton/backend.py` L549/561（multibuffer/auto-bind env 覆盖与 `_triton_config_*` withholding）。
`_compat.py` 的 `register_npu_backend`/`_register_interface_for_device` 为 torch_npu 注册，留在 npu 侧合理。

---

# 环境备忘（非代码改动）

- 本机已 `pip install`（经用户许可）：ruff 0.15.14、pyrefly 1.1.1、hypothesis 6.169.0、
  codespell 2.4.3（pre-commit 4.6.2 原已安装）
- 本机 torch 为 CPU 构建 + torch_npu（无 CUDA），`_testing.DEVICE` 在无 NPU 支持的分支上兜底 cuda
  → 依赖 CUDA 设备的测试在本机必失败（main 亦然），此类失败不是回归
- pyrefly 全量检查属 CUDA/社区环境（cutlass/jax 本机正确地不存在）
- `autoreview.py` 需以 `IS_SANDBOX=1` 运行（root 下嵌套 claude 的
  `--dangerously-skip-permissions` 会被拒绝；用户已确认此用法）
- **仓库卫生（autoreview 提示，防误提交）**：`CHANGES_TO_APPLY.md`、
  `npu-dev-backend-hooks.patch`、`npu_sweep_report.md`、
  `npu_sweep.log.broken-env-20261010`、`fusion_result.json` 均为未跟踪的内部产物且
  不在 `.gitignore` 内——本分支若 `git add .` 会把内部工作笔记带进公开 PR。
  已加入 `.git/info/exclude`（本地生效，不进版本库，无碍他环境）；
  `fusion_result.json` 是孤儿运行产物（9 行，无引用），建议删除。
  注意 `.git/info/exclude` 是本机配置，**其他环境提交前需自查这些文件名**。