"""NPU-only: two-pass feature-dim tiling for rms_norm bwd (whole-n load overflows UB)."""

from __future__ import annotations

import ast
import os

from ..ast_extension import statements_from_string


def rewrite_norm_bwd(body: list[ast.stmt], arg_names: list[str]) -> bool:
    """Rewrite an eligible rms_norm_bwd kernel body in place.

    (layer_norm_bwd is NOT rewritten: its two-pass fixes the UB overflow but the
    bwd grad_w still fails a precision check vs torch_npu's native fp16 -- helion
    is in fact more accurate, so this is a reference/tolerance issue, not a
    kernel bug.  Leaving layer_norm untouched rather than shipping a half-fix.)"""
    if os.environ.get("HELION_NPU_DISABLE_NORM_TWOPASS"):
        return False
    if _is_rms_norm_bwd(body, arg_names):
        _replace_body(body, _emit_rms_twopass(arg_names))
        return True
    return False


def _replace_body(body: list[ast.stmt], src_lines: list[str]) -> None:
    new = statements_from_string("\n".join(src_lines))
    body.clear()
    body.extend(new)


# ---------------------------------------------------------------------------
# Structural helpers
# ---------------------------------------------------------------------------


def _for_mb_cta(body: list[ast.stmt]) -> ast.For | None:
    """``for mb_cta in hl.tile(SIZE, block_size=...)``."""
    for s in body:
        if not isinstance(s, ast.For) or not isinstance(s.iter, ast.Call):
            continue
        f = s.iter.func
        if not (isinstance(f, ast.Attribute) and f.attr == "tile"):
            continue
        if isinstance(f.value, ast.Name) and f.value.id == "hl":
            if any(
                isinstance(k, ast.keyword) and k.arg == "block_size"
                for k in s.iter.keywords
            ):
                return s
    return None


def _inner_mb_loop(mb_cta: ast.For) -> ast.For | None:
    for s in mb_cta.body:
        if not isinstance(s, ast.For) or not isinstance(s.iter, ast.Call):
            continue
        f = s.iter.func
        if not (isinstance(f, ast.Attribute) and f.attr == "tile"):
            continue
        a = s.iter.args
        if (
            len(a) == 2
            and isinstance(a[0], ast.Attribute)
            and a[0].attr == "begin"
            and isinstance(a[1], ast.Attribute)
            and a[1].attr == "end"
        ):
            return s
    return None


def _is_int(node: ast.AST, value: int) -> bool:
    """True if *node* is an integer literal evaluating to *value* (handles
    ``-1`` which the AST encodes as ``UnaryOp(USub, Constant(1))``)."""
    return (isinstance(node, ast.Constant) and node.value == value) or (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
        and -node.operand.value == value
    )


def _has_call_with_dim(node: ast.AST, attr: str, dim: int) -> bool:
    for n in ast.walk(node):
        if not (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == attr
        ):
            continue
        for a in n.args:
            if _is_int(a, dim):
                return True
        for k in n.keywords:
            if k.arg == "dim" and _is_int(k.value, dim):
                return True
    return False


def _loads_2d_whole(node: ast.AST, tensor: str) -> bool:
    """``tensor[idx, :]`` (2D, last axis whole slice), in a Load context."""
    for n in ast.walk(node):
        if not (
            isinstance(n, ast.Subscript)
            and isinstance(n.value, ast.Name)
            and n.value.id == tensor
        ):
            continue
        if not isinstance(n.ctx, ast.Load):
            continue
        sl = n.slice
        if isinstance(sl, ast.Tuple) and len(sl.elts) == 2:
            last = sl.elts[-1]
            if (
                isinstance(last, ast.Slice)
                and last.lower is None
                and last.upper is None
            ):
                return True
    return False


# ---------------------------------------------------------------------------
# rms_norm_bwd: args = (grad_out, x, weight, rsqrt)
# ---------------------------------------------------------------------------


def _is_rms_norm_bwd(body: list[ast.stmt], args: list[str]) -> bool:
    # 4 tensor args (compute_bias_grad-less rms bwd): grad_out, x, weight, rsqrt
    if len(args) != 4:
        return False
    grad_out, x, weight, rsqrt = args
    mb_cta = _for_mb_cta(body)
    if mb_cta is None:
        return False
    inner = _inner_mb_loop(mb_cta)
    if inner is None:
        return False
    # mb_cta iterates x.size(0)
    iter_expr = mb_cta.iter
    it = (
        iter_expr.args[0]
        if isinstance(iter_expr, ast.Call) and iter_expr.args
        else None
    )
    if not (
        isinstance(it, ast.Call)
        and isinstance(it.func, ast.Attribute)
        and it.func.attr == "size"
        and isinstance(it.func.value, ast.Name)
        and it.func.value.id == x
    ):
        return False
    # inner body: stores grad_x[mb, :], uses .mean(-1), loads rsqrt[mb, :]
    has_gx = _has_grad_x_store(inner)
    has_mean = _has_call_with_dim(inner, "mean", -1)
    has_rsqrt = _loads_2d_whole(inner, rsqrt)
    has_x = _loads_2d_whole(inner, x) or _loads_2d_whole(inner, grad_out)
    return has_gx and has_mean and has_rsqrt and has_x


def _has_grad_x_store(node: ast.AST) -> bool:
    for n in ast.walk(node):
        if (
            isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Subscript)
            and isinstance(n.targets[0].value, ast.Name)
            and n.targets[0].value.id.startswith("grad_x")
        ):
            return True
    return False


def _emit_rms_twopass(args: list[str]) -> list[str]:
    grad_out, x, weight, rsqrt = args
    return [
        f"m_block = hl.register_block_size({x}.size(0))",
        f"grad_x = torch.empty_like({x})",
        f"grad_weight = {x}.new_empty([({x}.size(0) + m_block - 1) // m_block, *{weight}.shape], dtype=torch.float32)",
        f"weight_shape = hl.specialize({weight}.size(0))",
        f"for mb_cta in hl.tile({x}.size(0), block_size=m_block):",
        "    for tile_n in hl.tile(weight_shape):",
        f"        grad_w_m = {weight}.new_zeros(tile_n, dtype=torch.float32)",
        "        for mb in hl.tile(mb_cta.begin, mb_cta.end):",
        f"            x_m = {x}[mb, tile_n].to(torch.float32)",
        f"            do_m = {grad_out}[mb, tile_n].to(torch.float32)",
        f"            rsqrt_m = {rsqrt}[mb, :].to(torch.float32)",
        "            grad_w_m += (x_m * do_m * rsqrt_m).sum(0)",
        "        grad_weight[mb_cta.id, tile_n] = grad_w_m",
        "    for mb in hl.tile(mb_cta.begin, mb_cta.end):",
        f"        rsqrt_m = {rsqrt}[mb, :].to(torch.float32)",
        "        mean_term = hl.zeros([mb], dtype=torch.float32)",
        "        for tile_n in hl.tile(weight_shape):",
        f"            x_m = {x}[mb, tile_n].to(torch.float32)",
        f"            do_m = {grad_out}[mb, tile_n].to(torch.float32)",
        f"            w_m = {weight}[tile_n].to(torch.float32)",
        "            mean_term += (w_m[None, :] * do_m * x_m).sum(-1)",
        "        mean_term = mean_term / weight_shape",
        "        for tile_n in hl.tile(weight_shape):",
        f"            x_m = {x}[mb, tile_n].to(torch.float32)",
        f"            do_m = {grad_out}[mb, tile_n].to(torch.float32)",
        f"            w_m = {weight}[tile_n].to(torch.float32)",
        "            {gx}[mb, tile_n] = (w_m[None, :] * do_m * rsqrt_m - x_m * rsqrt_m ** 3 * mean_term[:, None]).to({x}.dtype)".format(
            gx="grad_x", x=x
        ),
        f"return (grad_x, grad_weight.sum(0).to({weight}.dtype))",
    ]
