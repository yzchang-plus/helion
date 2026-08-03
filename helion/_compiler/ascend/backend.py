"""Ascend NPU backend (triton-ascend) and NPU-specific codegen hooks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..triton.backend import TritonBackend

if TYPE_CHECKING:
    import sympy


class AscendBackend(TritonBackend):
    """Triton backend targeting Ascend NPU via triton-ascend."""

    @property
    def name(self) -> str:
        return "ascend"

    @property
    def codegen_name(self) -> str:
        return "triton"

    @property
    def library_imports(self) -> dict[str, str]:
        return {
            "math": "import math",
            "operator": "import operator",
            "torch": "import torch",
            "helion": "import helion",
            "hl": "import helion.language as hl",
            "triton": "import triton",
            "tl": "import triton.language as tl",
            "triton_helpers": "from torch._inductor.runtime import triton_helpers",
            "tl_math": "from torch_npu._inductor.npu_triton_helpers import math as tl_math",
            "libdevice": "from torch_npu._inductor.npu_triton_helpers import libdevice",
            "_default_launcher": "from helion.runtime import default_launcher as _default_launcher",
            "fast_dividef": "from triton.language.extra.libdevice import fast_dividef",
            "fast_expf": "from triton.language.extra.libdevice import fast_expf",
        }

    def classify_autotune_exception(self, err: BaseException) -> str | None:
        msg = f"{type(err).__name__}: {err}"
        if "BlockPtrAnalysis" in msg or "addptrRes.hasOneUse" in msg:
            return "debug"
        return super().classify_autotune_exception(err)

    @property
    def max_tensor_numel(self) -> int | None:
        from .config import _npu_max_tensor_numel

        return _npu_max_tensor_numel()

    def barrier_semaphore_dtype(self) -> torch.dtype:
        return torch.int32

    def grid_barrier_stmt(self, sem_arg: str) -> str:
        """Ascend grid barrier via atomic semaphore (triton-ascend has no x_grid_barrier)."""
        return (
            "if True:\n"
            "    tl.debug_barrier()\n"
            "    _bar_expected = tl.num_programs(0).to(tl.int32)\n"
            f"    _bar_inc = (0x80000000 - (_bar_expected - 1)) if tl.program_id(0) == 0 else 1\n"
            f"    _bar_old = tl.atomic_add({sem_arg}, _bar_inc, sem='release')\n"
            "    for _bar_i in tl.range(0, 1 << 18):\n"
            f"        _bar_cur = tl.atomic_add({sem_arg}, 0, sem='acquire')\n"
            "        _bar_flipped = ((_bar_old ^ _bar_cur) & 0x80000000) != 0\n"
            "        if _bar_flipped:\n"
            "            pass\n"
            "    tl.debug_barrier()"
        )

    def inline_constexpr_at_module_level(self) -> bool:
        return False

    def sympy_printer_expr(self, expr: "sympy.Expr") -> str:
        from ..triton.printer import ascend_texpr

        return ascend_texpr(expr)

    # -- NPU-specific config / codegen hooks (moved out of TritonBackend) ----

    def supports_config_key(self, key: str) -> bool:
        if key in ("num_warps", "num_stages"):
            return False
        return super().supports_config_key(key)

    def clamp_masked_pointer_offsets(self) -> bool:
        """Clamp masked offsets to >= 0 (Ascend MTE requires non-negative)."""
        return self.name != "tileir"

    def force_tile_mask(self) -> bool:
        """Force explicit masks for all tiles on NPU (pointer indexing safety)."""
        return self.name != "tileir"

    def customize_ast(self, hf) -> None:  # type: ignore[override]
        """NPU AST rewrites to fit the 192 KB UB (split-K, jagged 2D, norm two-pass, gdn sub-tile)."""
        if self.name == "tileir":
            return
        from .ast_split_k import split_k_matmuls
        from .ast_jagged_2d import rewrite_jagged_3d_bmm
        from .ast_batch_hoist import hoist_bmm_batch
        from .ast_norm_twopass import rewrite_norm_bwd
        from .ast_gdn_subtile import rewrite_gdn_fwd_h

        arg_names = [a.arg for a in hf.args.args]
        split_k_matmuls(hf.body)
        rewrite_jagged_3d_bmm(hf.body)
        hoist_bmm_batch(hf.body)
        rewrite_norm_bwd(hf.body, arg_names)
        rewrite_gdn_fwd_h(hf.body, arg_names)
