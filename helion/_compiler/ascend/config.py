"""Ascend NPU autotuner config helpers (env-tunable caps)."""

from __future__ import annotations

import os


def _npu_ub_budget_elements() -> int:
    """Max prod(block_sizes)*reduction_loops on Ascend (UB=192KB). Env: HELION_NPU_UB_BUDGET_ELEMENTS."""
    v = os.environ.get("HELION_NPU_UB_BUDGET_ELEMENTS", "").strip()
    try:
        return int(v) if v else 2048
    except ValueError:
        return 2048


def _npu_max_tensor_numel() -> int:
    """Per-tile max tensor numel on Ascend (UB=192KB). Env: HELION_NPU_MAX_TENSOR_NUMEL."""
    v = os.environ.get("HELION_NPU_MAX_TENSOR_NUMEL", "").strip()
    try:
        return int(v) if v else 8192
    except ValueError:
        return 8192


def _npu_default_reduction_loop() -> int:
    """Default reduction chunk on Ascend (must compile as baseline). Env: HELION_NPU_DEFAULT_REDUCTION_LOOP."""
    v = os.environ.get("HELION_NPU_DEFAULT_REDUCTION_LOOP", "").strip()
    try:
        return int(v) if v else 16
    except ValueError:
        return 16
