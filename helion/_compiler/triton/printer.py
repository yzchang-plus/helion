"""Triton-backend sympy expression printer.

``HelionTritonPrinter`` customises inductor's ``TritonPrinter``; it is also the
base class for the Cute and Pallas printers
(``helion/_compiler/{cute,pallas}/printer.py``).  Moved out of
``device_function.py`` so each backend owns its own printer.
"""

from __future__ import annotations

import sympy
from torch._inductor.codegen.triton import TritonPrinter


class HelionTritonPrinter(TritonPrinter):
    """Custom Triton printer that does the following:

    - Avoids wrapping float literals in tl.full().
     Inductor's default TritonPrinter prints SymPy Float as a 0-D Triton value
     via tl.full([], <val>, tl.float64). We override this to emit the raw numeric
     literal, letting downstream type promotion and casts handle dtype.

    - Avoids triton_helpers.div_floor_integer(...) calls when both operands are
      provably non-negative integers. TritonPrinter by default converts
      floor(u1/2) to triton_helpers.div_floor_integer(...). We override this to
      emit u1 // 2 only when the numerator is known to be non-negative and the
      denominator is a positive integer, so that we keep helper calls for cases
      that rely on floor semantics with mixed signs.
    """

    def _print_Float(self, expr: sympy.Expr) -> str:
        return str(expr)

    def _print_ToFloat(self, expr: sympy.Expr) -> str:
        assert expr.func.__name__ == "ToFloat" and len(expr.args) == 1
        # pyrefly: ignore [missing-attribute]
        return f"{self._print(expr.args[0])} + 0.0"

    def _print_FloorDiv(self, expr: sympy.Expr) -> str:
        from ..device_function import DeviceFunction

        lhs, rhs = expr.args
        # Only use // operator when:
        # 1. RHS is an integer constant
        # 2. LHS is a constexpr argument (autotune parameter like block size)
        # This ensures TMA descriptors get compile-time constants while preserving
        if (
            isinstance(rhs, sympy.Integer)
            and getattr(lhs, "name", None) in DeviceFunction.current()._constexpr_args
        ):
            # pyrefly: ignore [missing-attribute]
            lhs_str = self._print(lhs)
            # pyrefly: ignore [missing-attribute]
            rhs_str = self._print(rhs)
            if not (lhs.is_Integer or lhs.is_Symbol):
                lhs_str = f"({lhs_str})"
            return f"{lhs_str} // {rhs_str}"
        return super()._print_FloorDiv(expr)


def texpr(expr: sympy.Expr) -> str:
    return HelionTritonPrinter().doprint(expr)


class HelionAscendPrinter(HelionTritonPrinter):
    """Ascend NPU printer: emits simple ``//`` for FloorDiv/CleanDiv/CeilDiv.

    triton-ascend's BiShengIR does not recognise triton_helpers.div_floor_integer
    / clean_div helpers; for index calculations (lhs >= 0, rhs > 0) plain ``//``
    is correct and compiles.
    """

    def _print_FloorDiv(self, expr: sympy.Expr) -> str:
        lhs, rhs = expr.args
        lhs_str = self._print(lhs)
        rhs_str = self._print(rhs)
        if not (lhs.is_Integer or lhs.is_Symbol):
            lhs_str = f"({lhs_str})"
        if not (rhs.is_Integer or rhs.is_Symbol):
            rhs_str = f"({rhs_str})"
        return f"({lhs_str} // {rhs_str})"

    def _print_CleanDiv(self, expr: sympy.Expr) -> str:
        lhs, rhs = expr.args
        lhs_str = self._print(lhs)
        rhs_str = self._print(rhs)
        if not (lhs.is_Integer or lhs.is_Symbol):
            lhs_str = f"({lhs_str})"
        if not (rhs.is_Integer or rhs.is_Symbol):
            rhs_str = f"({rhs_str})"
        return f"({lhs_str} // {rhs_str})"

    def _print_CeilDiv(self, expr: sympy.Expr) -> str:
        lhs, rhs = expr.args
        lhs_str = self._print(lhs)
        rhs_str = self._print(rhs)
        if not (lhs.is_Integer or lhs.is_Symbol):
            lhs_str = f"({lhs_str})"
        if not (rhs.is_Integer or rhs.is_Symbol):
            rhs_str = f"({rhs_str})"
        # Standard ceiling division: (a + b - 1) // b
        return f"(({lhs_str} + {rhs_str} - 1) // {rhs_str})"

    def _print_PythonMod(self, expr: sympy.Expr) -> str:
        lhs, rhs = expr.args
        lhs_str = self._print(lhs)
        rhs_str = self._print(rhs)
        if not (lhs.is_Integer or lhs.is_Symbol):
            lhs_str = f"({lhs_str})"
        if not (rhs.is_Integer or rhs.is_Symbol):
            rhs_str = f"({rhs_str})"
        # Use simple % without sign checks (assumes lhs >= 0, rhs > 0, valid
        # for index calculations).
        return f"({lhs_str} % {rhs_str})"


def ascend_texpr(expr: sympy.Expr) -> str:
    """Convert SymPy expression to Ascend NPU-compatible Triton code."""
    return HelionAscendPrinter().doprint(expr)
