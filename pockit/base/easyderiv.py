import numba as nb

from typing import Iterable, Self
from pockit.base.vectypes import *


@nb.njit("boolean(int32, int32)")
def _less(a, b):
    if a < 0:
        if b < 0:
            return a < b
        else:
            return False
    else:
        if b < 0:
            return True
        else:
            return a < b


class Node:
    """Node in the computation graph to generate derivatives."""

    g: VecFloat
    """Local gradient value."""
    g_i: VecInt
    """Local gradient indices."""
    G: ListJit
    """Global gradient values."""
    G_i: ListJit
    """Global gradient indices."""
    h: VecFloat
    """Local Hessian value."""
    h_i_row: VecInt
    """Local Hessian row indices."""
    h_i_col: VecInt
    """Local Hessian column indices."""
    H: ListJit
    """Global Hessian values."""
    H_i_row: ListJit
    """Global Hessian row indices."""
    H_i_col: ListJit
    """Global Hessian column indices."""
    l: int
    """Length of the variables of the node."""
    args: list[Self]
    """List of ``Node``s of the function's arguments."""

    def __init__(self):
        self.g = np.empty((0, 1), dtype=np.float64)
        self.g_i = np.empty(0, dtype=np.int32)
        self.G = ListJit.empty_list(nb.float64[:])
        self.G_i = ListJit.empty_list(nb.int32[:])

        self.h = np.empty((0, 1), dtype=np.float64)
        self.h_i_row = np.empty(0, dtype=np.int32)
        self.h_i_col = np.empty(0, dtype=np.int32)
        self.H = ListJit.empty_list(nb.float64[:])
        self.H_i_row = ListJit.empty_list(nb.int32[:])
        self.H_i_col = ListJit.empty_list(nb.int32[:])

        self.l = 1
        self.args = []
        self._D_fct = []

    def set_G(self, G: Iterable[VecFloat]) -> None:
        """Set the node's global gradient values."""
        self.G = ListJit.empty_list(nb.float64[:])
        for G_ in G:
            self.G.append(G_)

    def set_G_i(self, G_i: Iterable[VecInt]) -> None:
        """Set the node's global gradient indices."""
        self.G_i = ListJit.empty_list(nb.int32[:])
        for G_i_ in G_i:
            self.G_i.append(G_i_)

    def set_H(self, H: Iterable[VecFloat]) -> None:
        """Set the node's global Hessian values."""
        self.H = ListJit.empty_list(nb.float64[:])
        for H_ in H:
            self.H.append(H_)

    def set_H_i_row(self, H_i_row: Iterable[VecInt]) -> None:
        """Set the node's global Hessian row indices."""
        self.H_i_row = ListJit.empty_list(nb.int32[:])
        for H_i_row_ in H_i_row:
            self.H_i_row.append(H_i_row_)

    def set_H_i_col(self, H_i_col: Iterable[VecInt]) -> None:
        """Set the node's global Hessian column indices."""
        self.H_i_col = ListJit.empty_list(nb.int32[:])
        for H_i_col_ in H_i_col:
            self.H_i_col.append(H_i_col_)


@nb.njit
def composite_gradient_i(arg_G_i: ListJit, l: int) -> ListJit:
    """Composite the global gradient indices of a node from the arguments'
    global gradient indices."""
    node_G_i = ListJit.empty_list(nb.int32[:])
    for G_i in arg_G_i:
        for a_i in G_i:
            if len(a_i) == 1 and l > 1:
                node_G_i.append(np.full(l, a_i[0], dtype=np.int32))
            else:
                node_G_i.append(a_i)
    return node_G_i


def forward_gradient_i(nodes: list[Node]) -> None:
    """Compute the global gradient indices of the nodes iteratively."""
    for node in nodes:
        if not len(node.g_i):
            continue
        arg_G_i = ListJit([node.args[i].G_i for i in node.g_i])
        node.G_i = composite_gradient_i(arg_G_i, node.l)


@nb.njit
def composite_gradient_v(arg_G: ListJit, node_g: VecFloat, l: int) -> ListJit:
    """Composite the global gradient values of a node from the arguments'
    global gradient values."""
    node_G = ListJit.empty_list(nb.float64[:])
    for G, n_g in zip(arg_G, node_g):
        for a_g in G:
            if len(a_g) == 1 and l > 1:
                node_G.append(np.full(l, a_g[0], dtype=np.float64) * n_g)
            else:
                node_G.append(a_g * n_g)
    return node_G


def forward_gradient_v(nodes: list[Node]) -> None:
    """Compute the global gradient values of the nodes iteratively."""
    for node in nodes:
        if not len(node.g_i):
            continue
        arg_G = ListJit([node.args[i].G for i in node.g_i])
        node.G = composite_gradient_v(arg_G, node.g, node.l)


@nb.njit
def composite_hessian_i_h(
    arg_H_i_row: ListJit, arg_H_i_col: ListJit, l: int
) -> tuple[ListJit, ListJit]:
    """Composite the global Hessian indices, the ``g-H`` part."""
    node_H_i_row = ListJit.empty_list(nb.int32[:])
    node_H_i_col = ListJit.empty_list(nb.int32[:])
    for H_i_row, H_i_col in zip(arg_H_i_row, arg_H_i_col):
        for a_i_row, a_i_col in zip(H_i_row, H_i_col):
            if len(a_i_row) == 1 and l > 1:
                node_H_i_row.append(np.full(l, a_i_row[0], dtype=np.int32))
                node_H_i_col.append(np.full(l, a_i_col[0], dtype=np.int32))
            else:
                node_H_i_row.append(a_i_row)
                node_H_i_col.append(a_i_col)
    return node_H_i_row, node_H_i_col


@nb.njit
def composite_hessian_i_g(
    arg_G_i_row: ListJit, arg_G_i_col: ListJit, diag: ListJit, l: int
) -> tuple[ListJit, ListJit]:
    """Composite the global Hessian indices, the ``h-G`` part."""
    node_H_i_row = ListJit.empty_list(nb.int32[:])
    node_H_i_col = ListJit.empty_list(nb.int32[:])
    for G_i_row, G_i_col, is_diag in zip(arg_G_i_row, arg_G_i_col, diag):
        for a_i_row in G_i_row:
            for a_i_col in G_i_col:
                a_i_row_2 = a_i_row.copy()
                a_i_col_2 = a_i_col.copy()
                if len(a_i_row) == 1 and l > 1:
                    a_i_row_2 = np.full(l, a_i_row[0], dtype=np.int32)
                if len(a_i_col) == 1 and l > 1:
                    a_i_col_2 = np.full(l, a_i_col[0], dtype=np.int32)
                if len(a_i_row) == 1 and len(a_i_col) > 1:
                    a_i_row_2 = np.full_like(a_i_col, a_i_row[0], dtype=np.int32)
                if len(a_i_col) == 1 and len(a_i_row) > 1:
                    a_i_col_2 = np.full(len(a_i_row), a_i_col[0], dtype=np.int32)

                if is_diag:
                    if not _less(a_i_row[0], a_i_col[0]):
                        node_H_i_row.append(a_i_row_2)
                        node_H_i_col.append(a_i_col_2)
                else:
                    if _less(a_i_row[0], a_i_col[0]):
                        node_H_i_row.append(a_i_col_2)
                        node_H_i_col.append(a_i_row_2)
                    else:
                        node_H_i_row.append(a_i_row_2)
                        node_H_i_col.append(a_i_col_2)
    return node_H_i_row, node_H_i_col


def forward_hessian_i(nodes: list[Node]) -> None:
    """Compute the global Hessian indices of the nodes iteratively."""
    for node in nodes:
        if not node.args:
            continue
        node.H_i_row = ListJit.empty_list(nb.int32[:])
        node.H_i_col = ListJit.empty_list(nb.int32[:])

        arg_H_i_row = ListJit()
        arg_H_i_col = ListJit()
        for i in node.g_i:
            arg_H_i_row.append(node.args[i].H_i_row)
            arg_H_i_col.append(node.args[i].H_i_col)
        if arg_H_i_row:
            H_i_h_row, H_i_h_col = composite_hessian_i_h(
                arg_H_i_row, arg_H_i_col, node.l
            )
            node.H_i_row.extend(H_i_h_row)
            node.H_i_col.extend(H_i_h_col)

        arg_G_i_row = ListJit()
        arg_G_i_col = ListJit()
        diag = ListJit()
        for i_row, i_col in zip(node.h_i_row, node.h_i_col):
            arg_G_i_row.append(node.args[i_row].G_i)
            arg_G_i_col.append(node.args[i_col].G_i)
            diag.append(i_row == i_col)
        if arg_G_i_row:
            H_i_g_row, H_i_g_col = composite_hessian_i_g(
                arg_G_i_row, arg_G_i_col, diag, node.l
            )
            node.H_i_row.extend(H_i_g_row)
            node.H_i_col.extend(H_i_g_col)


@nb.njit
def composite_hessian_v_h(arg_H: ListJit, node_g: VecFloat, l: int) -> ListJit:
    """Composite the global Hessian values, the ``g-H`` part."""
    node_H = ListJit.empty_list(nb.float64[:])
    for H, n_g in zip(arg_H, node_g):
        for a_h in H:
            if len(a_h) == 1 and l > 1:
                node_H.append(np.full(l, a_h[0], dtype=np.float64) * n_g)
            else:
                node_H.append(a_h * n_g)
    return node_H


@nb.njit
def composite_hessian_v_g(
    arg_G_i_row: ListJit,
    arg_G_i_col: ListJit,
    arg_G_row: ListJit,
    arg_G_col: ListJit,
    diag: ListJit,
    node_h: VecFloat,
    l: int,
) -> ListJit:
    """Composite the global Hessian values, the ``h-G`` part."""
    node_H = ListJit.empty_list(nb.float64[:])
    for G_i_row, G_i_col, G_row, G_col, is_diag, n_h in zip(
        arg_G_i_row, arg_G_i_col, arg_G_row, arg_G_col, diag, node_h
    ):
        for a_i_row, a_g_row in zip(G_i_row, G_row):
            for a_i_col, a_g_col in zip(G_i_col, G_col):
                a_g_row_2 = a_g_row.copy()
                a_g_col_2 = a_g_col.copy()
                if len(a_g_row) == 1 and l > 1:
                    a_g_row_2 = np.full(l, a_g_row[0], dtype=np.float64)
                if len(a_g_col) == 1 and l > 1:
                    a_g_col_2 = np.full(l, a_g_col[0], dtype=np.float64)
                if is_diag:
                    if not _less(a_i_row[0], a_i_col[0]):
                        node_H.append(a_g_row_2 * a_g_col_2 * n_h)
                else:
                    if a_i_row[0] == a_i_col[0]:
                        node_H.append(a_g_row_2 * a_g_col_2 * n_h * 2)
                    else:
                        node_H.append(a_g_row_2 * a_g_col_2 * n_h)
    return node_H


def forward_hessian_v(nodes: list[Node]) -> None:
    """Compute the global Hessian values of the nodes iteratively."""
    for node in nodes:
        if not node.args:
            continue
        node.H = ListJit.empty_list(nb.float64[:])
        arg_H = ListJit([node.args[i].H for i in node.g_i])
        if arg_H:
            node.H.extend(composite_hessian_v_h(arg_H, node.g, node.l))

        arg_G_i_row = ListJit()
        arg_G_i_col = ListJit()
        arg_G_row = ListJit()
        arg_G_col = ListJit()
        diag = ListJit()
        for i_row, i_col, n_h in zip(node.h_i_row, node.h_i_col, node.h):
            arg_G_i_row.append(node.args[i_row].G_i)
            arg_G_i_col.append(node.args[i_col].G_i)
            arg_G_row.append(node.args[i_row].G)
            arg_G_col.append(node.args[i_col].G)
            diag.append(i_row == i_col)
        if arg_G_row:
            node.H.extend(
                composite_hessian_v_g(
                    arg_G_i_row, arg_G_i_col, arg_G_row, arg_G_col, diag, node.h, node.l
                )
            )
