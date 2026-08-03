"""NPU-only: hoist 3D bmm batch dim to a grid loop (3D bmm overflows UB)."""

from __future__ import annotations

import ast

from ..ast_extension import statement_from_string


def hoist_bmm_batch(body: list[ast.stmt]) -> bool:
    """Hoist 3D bmm batch dim to a grid loop. Returns True if changed."""
    # Find H = hl.specialize(...) at the kernel top level first
    h_name = _find_specialize_in_body(body)
    if h_name is None:
        return False
    changed = False
    for stmt in body:
        if isinstance(stmt, ast.For):
            if _hoist_in_for_loop(stmt, h_name):
                changed = True
            if hoist_bmm_batch(stmt.body):
                changed = True
    return changed


def _find_specialize_in_body(body: list[ast.stmt]) -> str | None:
    """Find ``H = hl.specialize(...)`` in the body (top-level assignments)."""
    for stmt in body:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        if not isinstance(stmt.targets[0], ast.Name):
            continue
        if not isinstance(stmt.value, ast.Call):
            continue
        func = stmt.value.func
        attr = None
        if isinstance(func, ast.Attribute):
            attr = func.attr
        elif isinstance(func, ast.Name):
            attr = func.id
        if attr == "specialize":
            return stmt.targets[0].id
    return None


def _hoist_in_for_loop(for_node: ast.For, h_name: str) -> bool:
    """If this for-loop's body has torch.bmm, wrap in ``for h in hl.grid(H):``."""
    if not _body_has_bmm(for_node):
        return False
    _rewrite_to_2d(for_node, h_name)
    return True


def _body_has_bmm(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
            if child.func.attr == "bmm":
                return True
    return False


def _rewrite_to_2d(for_node: ast.For, h_name: str) -> None:
    """Wrap for_node.body in ``for h in hl.grid(H):`` and rewrite 3D -> 2D."""
    rewriter = _BmmBatchRewriter(h_name, "h")
    new_body = [rewriter.visit(stmt) for stmt in for_node.body]
    grid_loop = ast.For(
        target=ast.Name(id="h", ctx=ast.Store()),
        iter=ast.Call(
            func=ast.Attribute(value=ast.Name(id="hl", ctx=ast.Load()), attr="grid", ctx=ast.Load()),
            args=[ast.Name(id=h_name, ctx=ast.Load())],
            keywords=[],
        ),
        body=new_body,
        orelse=[],
    )
    ast.fix_missing_locations(grid_loop)
    # Wrap as ExtendedAST (helion requires ExtendedAST, not standard ast)
    grid_loop = statement_from_string(ast.unparse(grid_loop))
    for_node.body = [grid_loop]


class _BmmBatchRewriter(ast.NodeTransformer):
    """Rewrite 3D bmm body to 2D: index batch dim with scalar h."""

    def __init__(self, h_name: str, loop_var: str) -> None:
        self.h_name = h_name
        self.loop_var = loop_var

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        func_name = None
        if isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        # torch.bmm -> torch.matmul (3D -> 2D)
        if func_name == "bmm":
            node.func.attr = "matmul"
        # .transpose(0, 1) -> remove (return the value being transposed)
        if func_name == "transpose":
            if (
                len(node.args) == 2
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == 0
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == 1
            ):
                return node.func.value  # the object being .transpose(0,1)'d
        # hl.zeros / hl.full: remove H from shape list
        if func_name in ("zeros", "full") and node.args:
            if isinstance(node.args[0], ast.List):
                node.args[0].elts = [
                    e
                    for e in node.args[0].elts
                    if not (isinstance(e, ast.Name) and e.id == self.h_name)
                ]
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        node = self.generic_visit(node)
        sl = node.slice
        if not isinstance(sl, ast.Tuple):
            return node
        elts = sl.elts
        # mask[None, :, :] -> mask (remove the [None, :, :] subscript)
        if len(elts) >= 3 and _is_none_const(elts[0]):
            return node.value
        # tensor[tile, :, :] -> tensor[tile, h, :] (replace 2nd Slice(None) with h)
        if len(elts) >= 3:
            for i in range(len(elts) - 1):
                if _is_slice_none(elts[i]):
                    elts[i] = ast.Name(id=self.loop_var, ctx=ast.Load())
                    break
        return node


def _is_slice_none(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Slice)
        and node.lower is None
        and node.upper is None
        and node.step is None
    )


def _is_none_const(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is None
