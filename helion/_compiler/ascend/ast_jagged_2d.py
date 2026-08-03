"""NPU-only: rewrite jagged 3D bmm to 2D (triton-ascend can't lower 3D tl.dot)."""

from __future__ import annotations

import ast

from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string


def rewrite_jagged_3d_bmm(body: list[ast.stmt]) -> bool:
    """Rewrite ``hl.jagged_tile`` + 3D matmul to 2D on NPU. Returns True if changed."""
    changed = False
    i = 0
    while i < len(body):
        stmt = body[i]
        if isinstance(stmt, ast.For):
            # recurse into for-loop bodies
            if rewrite_jagged_3d_bmm(stmt.body):
                changed = True
            # check if this is the outer hl.tile + inner hl.jagged_tile pattern
            info = _match_pattern(stmt)
            if info is not None:
                body[i] = _rewrite_loop(stmt, info)
                changed = True
        i += 1
    return changed


def _match_pattern(outer_for: ast.For) -> dict | None:
    """Match ``for tile_b in hl.tile(B): ... for tile_l in hl.jagged_tile(seq_len): ...``
    AND the body contains a matmul (torch.matmul / torch.bmm / @).  Only
    transform when there's a 3D bmm to fix -- jagged kernels without matmul
    (jagged_sum, jagged_mean, etc.) must not be touched."""
    # outer loop must be `for X in hl.tile(...)`
    tile_b_name = _get_tile_loop_target(outer_for, "tile")
    if tile_b_name is None:
        return None
    # find inner hl.jagged_tile loop
    for stmt in outer_for.body:
        if not isinstance(stmt, ast.For):
            continue
        tile_l_name = _get_tile_loop_target(stmt, "jagged_tile")
        if tile_l_name is None:
            continue
        # Only transform if the body contains a matmul (the 3D bmm that
        # triggers the malloc).  Without matmul, jagged_tile works fine.
        if not _body_has_matmul(outer_for):
            return None
        return {
            "tile_b_name": tile_b_name,
            "tile_l_name": tile_l_name,
            "outer_for": outer_for,
            "inner_for": stmt,
        }
    return None


def _body_has_matmul(node: ast.AST) -> bool:
    """Check if the AST contains a matmul (torch.matmul, torch.bmm, or @)."""
    for child in ast.walk(node):
        if isinstance(child, ast.BinOp) and isinstance(child.op, ast.MatMult):
            return True
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
            if child.func.attr in ("matmul", "bmm", "mm"):
                return True
    return False


def _get_tile_loop_target(for_node: ast.For, func_name: str) -> str | None:
    """Return the loop target name if iter is hl.tile or hl.jagged_tile."""
    if not isinstance(for_node.target, ast.Name):
        return None
    iter_node = for_node.iter
    if not isinstance(iter_node, ast.Call):
        return None
    func = iter_node.func
    if isinstance(func, ast.Attribute):
        if func.attr != func_name:
            return None
    elif isinstance(func, ast.Name):
        if func.id != func_name:
            return None
    else:
        return None
    return for_node.target.id


def _rewrite_loop(outer_for: ast.For, info: dict) -> ast.For:
    """Rewrite the outer hl.tile + inner hl.jagged_tile to hl.grid + hl.tile + mask."""
    tile_b = info["tile_b_name"]
    tile_l = info["tile_l_name"]
    inner_for = info["inner_for"]

    # 1. Change outer iter: hl.tile(B) -> hl.grid(B)
    new_outer_iter = _replace_call_func(outer_for.iter, "grid")
    # 2. Change outer target: tile_b -> b
    new_outer_target = ast.Name(id="b", ctx=ast.Store())

    # 3. Change inner iter: hl.jagged_tile(seq_len) -> hl.tile(seq_len)
    new_inner_iter = _replace_call_func(inner_for.iter, "tile")
    # 4. Change inner target: tile_l -> tile_l (keep name, it's fine)

    # 5. Build mask statement: mask_l = tile_l.index < seq_len
    #    seq_len is the argument to the original hl.jagged_tile call
    seq_len_expr = _get_call_arg(inner_for.iter)
    mask_stmt = statement_from_string(
        f"_mask_l = {tile_l}.index < {_unparse(seq_len_expr)}"
    )

    # 6. Rewrite: visit OUTER body first (populates _scalars from starts=...),
    #    THEN inner body (which uses starts[:, None] that needs scalar removal).
    rewriter = _JaggedBodyRewriter(tile_b, "b", tile_l)
    new_outer_body = []
    inner_inserted = False
    for stmt in outer_for.body:
        if stmt is inner_for:
            # Visit inner body now (scalars are populated from outer body above)
            new_inner_body = [rewriter.visit(s) for s in inner_for.body]
            new_inner_body = [mask_stmt] + new_inner_body
            new_outer_body.append(
                ast.For(
                    target=inner_for.target,
                    iter=new_inner_iter,
                    body=new_inner_body,
                    orelse=[],
                )
            )
            inner_inserted = True
        else:
            new_outer_body.append(rewriter.visit(stmt))

    new_for = ast.For(
        target=new_outer_target,
        iter=new_outer_iter,
        body=new_outer_body,
        orelse=[],
    )
    ast.copy_location(new_for, outer_for)
    ast.fix_missing_locations(new_for)
    # Wrap as ExtendedAST (helion requires ExtendedAST nodes, not standard ast)
    return statement_from_string(ast.unparse(new_for))


def _replace_call_func(call_node: ast.Call, new_name: str) -> ast.Call:
    """Replace the function name of a Call node (hl.tile -> hl.grid etc.)."""
    new = ast.Call(
        func=ast.Attribute(
            value=ast.Name(id="hl", ctx=ast.Load()),
            attr=new_name,
            ctx=ast.Load(),
        ),
        args=call_node.args,
        keywords=call_node.keywords,
    )
    ast.copy_location(new, call_node)
    return new


def _get_call_arg(call_node: ast.Call) -> ast.AST:
    """Get the first positional arg of a Call."""
    return call_node.args[0] if call_node.args else ast.Constant(value=None)


def _unparse(node: ast.AST) -> str:
    return ast.unparse(node)


class _JaggedBodyRewriter(ast.NodeTransformer):
    """Rewrite jagged kernel body: replace batch block var with scalar,
    drop the batch dimension from shapes and subscripts."""

    def __init__(self, old_batch: str, new_batch: str, tile_l: str) -> None:
        self.old_batch = old_batch  # e.g. "tile_b"
        self.new_batch = new_batch  # e.g. "b"
        self.tile_l = tile_l  # e.g. "tile_len" (kept as-is)
        # Names that become scalar after the transform (indexed with scalar b)
        self._scalars: set[str] = set()

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id == self.old_batch:
            return ast.Name(id=self.new_batch, ctx=node.ctx)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        # tile_b.index -> b (scalar, no .index needed)
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == self.old_batch
            and node.attr == "index"
        ):
            return ast.Name(id=self.new_batch, ctx=ast.Load())
        return self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        node = self.generic_visit(node)
        # Track scalars: target = tensor[b] (scalar index) -> target is scalar
        if (
            isinstance(node.value, ast.Subscript)
            and isinstance(node.value.slice, ast.Tuple)
        ):
            # multi-dim index, not scalar
            pass
        elif isinstance(node.value, ast.Subscript):
            idx = node.value.slice
            if isinstance(idx, ast.Name) and idx.id == self.new_batch:
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        self._scalars.add(t.id)
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        # hl.zeros / hl.full: remove the batch dim (old_batch -> b) from shape list
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in ("zeros", "full")
            and node.args
            and isinstance(node.args[0], ast.List)
        ):
            shape_list = node.args[0]
            # Remove elements that are Name("b") (the scalar batch, which was tile_b)
            shape_list.elts = [
                e for e in shape_list.elts
                if not (isinstance(e, ast.Name) and e.id == self.new_batch)
            ]
        # Shift unsqueeze dim: removing batch dim (dim 0) shifts all dims down.
        # unsqueeze(1) -> unsqueeze(0), unsqueeze(2) -> unsqueeze(1), etc.
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "unsqueeze"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, int)
            and node.args[0].value > 0
        ):
            node.args[0] = ast.Constant(value=node.args[0].value - 1)
        # Add mask to hl.load / hl.store that use jagged_indices
        func_name = None
        if isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        elif isinstance(node.func, ast.Name):
            func_name = node.func.id
        if func_name in ("load", "store"):
            node = self._add_mask_if_needed(node)
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        node = self.generic_visit(node)
        sl = node.slice
        if not isinstance(sl, ast.Tuple):
            return node
        elts = sl.elts
        # 3+ element subscripts: remove first dim if trivial (batch dim)
        if len(elts) >= 3 and _is_none_or_slice_none(elts[0]):
            elts = elts[1:]
            node.slice = elts[0] if len(elts) == 1 else ast.Tuple(elts=elts, ctx=sl.ctx)
            return node
        # 2-element subscripts
        if len(elts) == 2:
            first, second = elts[0], elts[1]
            # [None, :] -> None is batch broadcast dim, remove -> value
            if _is_none_const(first) and _is_slice_none(second):
                return node.value
            # [:, None] on scalar -> remove -> value
            if _is_slice_none(first) and _is_none_const(second):
                if isinstance(node.value, ast.Name) and node.value.id in self._scalars:
                    return node.value
        return node

    def _add_mask_if_needed(self, node: ast.Call) -> ast.Call:
        """Add _mask_l[:, None] as the mask argument to hl.load/hl.store."""
        has_mask = len(node.args) >= 3 or any(
            kw.arg in ("mask", "extra_mask") for kw in node.keywords
        )
        if has_mask:
            return node
        # 2D mask: _mask_l[:, None] broadcasts to match the load/store shape
        mask_expr = expr_from_string("_mask_l[:, None]")
        node.args.append(mask_expr)
        return node


def _is_none_or_slice_none(node: ast.AST) -> bool:
    """Check if node is None (NoneConst) or Slice(None, None, None)."""
    return _is_none_const(node) or _is_slice_none(node)


def _is_none_const(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _is_slice_none(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Slice)
        and node.lower is None
        and node.upper is None
        and node.step is None
    )
