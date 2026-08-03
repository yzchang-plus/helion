"""NPU-only: split whole-loaded large-K 2D matmuls into split-K loops."""

from __future__ import annotations

import ast
from typing import TypeVar

from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string

T = TypeVar("T", bound=ast.AST)


def split_k_matmuls(body: list[ast.stmt]) -> bool:
    """Rewrite eligible whole-K 2D matmuls in ``body`` into split-K loops."""
    name_to_subscript = _build_name_to_subscript(body)
    return _transform_body(body, name_to_subscript)


def _build_name_to_subscript(body: list[ast.stmt]) -> dict[str, ast.Subscript]:
    """Map ``name -> Subscript`` for assignments ``name = tensor[...]``."""
    out: dict[str, ast.Subscript] = {}
    for stmt in body:
        if not isinstance(stmt, ast.Assign):
            continue
        if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
            continue
        if isinstance(stmt.value, ast.Subscript):
            out[stmt.targets[0].id] = stmt.value
    return out


def _transform_body(body: list[ast.stmt], name_to_subscript: dict) -> bool:
    changed = False
    i = 0
    while i < len(body):
        stmt = body[i]
        if isinstance(stmt, (ast.For, ast.While)):
            inner_map = _build_name_to_subscript(stmt.body)
            if _transform_body(stmt.body, {**name_to_subscript, **inner_map}):
                changed = True
        replacement = _maybe_split_k(stmt, name_to_subscript)
        if replacement is not None:
            body[i : i + 1] = replacement
            i += len(replacement)
            changed = True
        else:
            i += 1
    return changed


def _maybe_split_k(
    stmt: ast.stmt, name_to_subscript: dict
) -> list[ast.stmt] | None:
    """If ``stmt`` contains an eligible whole-K 2D matmul, return the split-K
    loop statements + the rewritten ``stmt`` (BinOp replaced by the acc)."""
    target_binop = _find_matmul(stmt)
    if target_binop is None:
        return None
    info = _analyze_binop(target_binop, name_to_subscript)
    if info is None:
        return None
    new_stmt = _ReplaceBinOp(target_binop, info["acc_cast_expr"]).visit(stmt)
    ast.copy_location(new_stmt, stmt)
    for s in info["loop_stmts"]:
        ast.copy_location(s, stmt)
    ast.fix_missing_locations(new_stmt)
    for s in info["loop_stmts"]:
        ast.fix_missing_locations(s)
    return [*info["loop_stmts"], new_stmt]


def _find_matmul(stmt: ast.stmt) -> ast.BinOp | None:
    for node in ast.walk(stmt):
        if (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.MatMult)
            and isinstance(node.left, (ast.Subscript, ast.Name))
            and isinstance(node.right, (ast.Subscript, ast.Name))
        ):
            return node
    return None


def _resolve_subscript(
    operand: ast.AST, name_to_subscript: dict
) -> ast.Subscript | None:
    if isinstance(operand, ast.Subscript):
        return operand
    if isinstance(operand, ast.Name):
        return name_to_subscript.get(operand.id)
    return None


def _is_whole_slice(idx: ast.AST) -> bool:
    return (
        isinstance(idx, ast.Slice)
        and idx.lower is None
        and idx.upper is None
        and idx.step is None
    )


def _is_full_tensor_load(idx_list: list[ast.AST]) -> bool:
    """True if every axis of the subscript is a whole ``:`` slice -- i.e. the
    whole tensor is loaded (the case that overflows NPU UB)."""
    return bool(idx_list) and all(_is_whole_slice(i) for i in idx_list)


def _size_or_index(name: str, idx: ast.AST, axis: int) -> str:
    """Source string for an accumulator dimension.

    A whole-slice axis becomes ``name.size(axis)`` (the full extent); a tile
    index is used directly (``hl.zeros`` accepts tile indices via
    ``tiles_as_sizes``), e.g. ``a[:, tile_k]`` -> N = ``tile_k`` not
    ``a.size(1)``.
    """
    if _is_whole_slice(idx):
        return f"{name}.size({axis})"
    return ast.unparse(idx)


def _index_list(subscript: ast.Subscript) -> list[ast.AST]:
    sl = subscript.slice
    if isinstance(sl, ast.Tuple):
        return list(sl.elts)
    return [sl]


def _analyze_binop(binop: ast.BinOp, name_to_subscript: dict) -> dict | None:
    a_sub = _resolve_subscript(binop.left, name_to_subscript)
    b_sub = _resolve_subscript(binop.right, name_to_subscript)
    if a_sub is None or b_sub is None:
        return None
    if not isinstance(a_sub.value, ast.Name) or not isinstance(b_sub.value, ast.Name):
        return None
    a_idx = _index_list(a_sub)
    b_idx = _index_list(b_sub)
    if len(a_idx) != 2 or len(b_idx) != 2:
        return None  # v1: 2D only
    # mm(A, B): A's contraction = last axis, B's contraction = 2nd-to-last.
    a_contract = len(a_idx) - 1  # = 1 for 2D
    b_contract = len(b_idx) - 2  # = 0 for 2D
    if not _is_whole_slice(a_idx[a_contract]) or not _is_whole_slice(b_idx[b_contract]):
        return None
    # Only split when a whole-K contraction is actually fed by a *full-tensor*
    # load (every axis ``:``), e.g. se_block's ``w[:, :]`` (2 MB, overflows UB
    # regardless of config).  A partial load such as squeeze's ``a[:, tile_k]``
    # (only the contraction axis is whole) fits in UB and the split-K loop
    # triggers triton-ascend non-determinism there -- so leave it alone.
    if not (_is_full_tensor_load(a_idx) or _is_full_tensor_load(b_idx)):
        return None
    a_name = a_sub.value.id
    b_name = b_sub.value.id
    m_index = a_idx[1 - a_contract]  # A's non-contraction index (a tile index)
    n_index = b_idx[1 - b_contract]  # B's non-contraction index
    m_axis = 1 - a_contract
    n_axis = 1 - b_contract  # = 1 for 2D
    # Accumulator M/N sizes: a tile index is used directly (hl.zeros accepts
    # tile indices via tiles_as_sizes); a whole slice becomes ``.size(axis)``.
    # e.g. se_block ``w[:, :]`` -> N = w.size(1); squeeze ``a[:, tile_k]`` ->
    # N = tile_k (the block index, not the full dim).
    m_src = _size_or_index(a_name, m_index, m_axis)
    n_size_src = _size_or_index(b_name, n_index, n_axis)
    k_size_src = f"{a_name}.size({a_contract})"  # K is always whole-loaded (slice)
    # Build reload subscripts as source strings (avoids hand-constructing
    # ExtendedAST nodes, which require a _location kwarg).  The contraction
    # axis index is replaced with the loop variable ``_sk_k``.
    a_idx_srcs = [ast.unparse(ix) for ix in a_idx]
    a_idx_srcs[a_contract] = "_sk_k"
    b_idx_srcs = [ast.unparse(ix) for ix in b_idx]
    b_idx_srcs[b_contract] = "_sk_k"
    a_reload_src = f"{a_name}[{', '.join(a_idx_srcs)}]"
    b_reload_src = f"{b_name}[{', '.join(b_idx_srcs)}]"
    init_stmt = statement_from_string(
        f"_sk_acc = hl.zeros([{m_src}, {n_size_src}], dtype=torch.float32)"
    )
    for_stmt = statement_from_string(
        f"for _sk_k in hl.tile({k_size_src}): "
        f"_sk_acc = _sk_acc + ({a_reload_src} @ {b_reload_src})"
    )
    cast_expr = expr_from_string(f"_sk_acc.to({a_name}.dtype)")
    return {
        "loop_stmts": [init_stmt, for_stmt],
        "acc_cast_expr": cast_expr,
        "a_name": a_name,
        "b_name": b_name,
    }


class _ReplaceBinOp(ast.NodeTransformer):
    def __init__(self, target: ast.BinOp, replacement: ast.AST) -> None:
        self.target = target
        self.replacement = replacement

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        if node is self.target:
            return self.replacement
        return self.generic_visit(node)
