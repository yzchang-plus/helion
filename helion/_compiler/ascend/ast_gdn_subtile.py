"""NPU-only: sub-tile gdn_fwd_h chunk dim (chunk=256 buffers overflow UB)."""

from __future__ import annotations

import ast
import os

from ..ast_extension import statements_from_string


def rewrite_gdn_fwd_h(body: list[ast.stmt], arg_names: list[str]) -> bool:
    """Rewrite an eligible ``gdn_fwd_h`` kernel body to sub-tile the chunk."""
    if os.environ.get("HELION_NPU_DISABLE_GDN_SUBTILE"):
        return False
    if not _is_gdn_fwd_h(body, arg_names):
        return False
    _replace_body(body, _emit_gdn_subtile(arg_names))
    return True


def _replace_body(body: list[ast.stmt], src_lines: list[str]) -> None:
    new = statements_from_string("\n".join(src_lines))
    body.clear()
    body.extend(new)


def _is_gdn_fwd_h(body: list[ast.stmt], args: list[str]) -> bool:
    # args = (k, w, u, g, chunk_size)
    if len(args) != 5:
        return False
    k, w, u, g, chunk_size = args
    # Must contain: for t_i in hl.tile(seqlen, block_size=chunk_size) with an
    # hl.dot inside, and a store h[...t_i.id...] = b_h.to(...).
    has_chunk_loop = False
    has_dot = False
    has_h_store = False
    for s in body:
        for n in ast.walk(s):
            if isinstance(n, ast.For) and isinstance(n.iter, ast.Call):
                f = n.iter.func
                if isinstance(f, ast.Attribute) and f.attr == "tile":
                    if any(
                        isinstance(kw, ast.keyword) and kw.arg == "block_size"
                        for kw in n.iter.keywords
                    ):
                        has_chunk_loop = True
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "dot":
                has_dot = True
            if (
                isinstance(n, ast.Assign)
                and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Subscript)
                and isinstance(n.targets[0].value, ast.Name)
                and n.targets[0].value.id == "h"
                and isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Attribute)
                and n.value.func.attr == "to"
            ):
                has_h_store = True
    return has_chunk_loop and has_dot and has_h_store


def _emit_gdn_subtile(args: list[str]) -> list[str]:
    k, w, u, g, chunk_size = args
    return [
        "batch, seqlen, nheads, dhead = {k}.shape".format(k=k),
        "dhead = hl.specialize(dhead)",
        "chunk_size = hl.specialize(chunk_size)",
        "dstate = {u}.shape[-1]".format(u=u),
        "acc_dtype = torch.float32",
        "dtype = {k}.dtype".format(k=k),
        "nchunks = (seqlen + chunk_size - 1) // chunk_size",
        "h = torch.empty(batch, nchunks, nheads, dhead, dstate, dtype=dtype, device={k}.device)".format(k=k),
        "block_v = hl.register_block_size(dstate)",
        "for tile_b, tile_h, tile_v in hl.tile([batch, nheads, dstate], block_size=[1, 1, block_v]):",
        "    i_b = tile_b.id",
        "    i_h = tile_h.id",
        "    b_h = hl.zeros([dhead, tile_v], dtype=acc_dtype)",
        "    for t_i in hl.tile(seqlen, block_size=chunk_size):",
        "        h[i_b, t_i.id, i_h, :, tile_v] = b_h.to(dtype)",
        "        t_i_last = min(t_i.begin + chunk_size, seqlen) - 1",
        "        b_g_last = {g}[i_b, t_i_last, i_h].to(acc_dtype)".format(g=g),
        "        b_h_pre = b_h",
        "        b_h = b_h * torch.exp(b_g_last)",
        "        for tile_c in hl.tile(t_i.begin, t_i.end):",
        "            b_w = {w}[i_b, tile_c, i_h, :]".format(w=w),
        "            c_h = b_h_pre.to(dtype)",
        "            b_v = hl.dot(b_w, c_h, out_dtype=acc_dtype)",
        "            p_v = {u}[i_b, tile_c, i_h, tile_v].to(acc_dtype)".format(u=u),
        "            b_v = p_v - b_v",
        "            m_t = tile_c.index < seqlen",
        "            b_g = {g}[i_b, tile_c, i_h].to(acc_dtype)".format(g=g),
        "            b_v *= torch.where(m_t, torch.exp(b_g_last - b_g), 0)[:, None]",
        "            b_v = b_v.to(dtype)",
        "            p_k = {k}[i_b, tile_c, i_h, :]".format(k=k),
        "            b_h = hl.dot(p_k.T, b_v, acc=b_h)",
        "return h",
    ]
