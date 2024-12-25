# Copyright (c) 2024 Yilin Zou
import functools
import os
import tempfile
import importlib.util
import sys
from typing import Callable, Optional

import numpy as np
import sympy as sp
from sympy.codegen.rewriting import create_expand_pow_optimization
from sympy.printing.lambdarepr import lambdarepr

from .vectypes import *


@functools.lru_cache
def _deco_basic(num_free_symbol: int, parallel: bool, fastmath: bool, cache: bool) -> str:
    """Decorator for basic functions."""
    tail = ""
    # maybe numba bugs, do not compile if set (TODO: why?)
    # if parallel:
    #     tail += ', target="parallel"'
    if fastmath:
        tail += ", fastmath=True"
    if cache:
        tail += ", cache=True"
    return '@numba.vectorize("float64({})"{})\n'.format(
        ", ".join(["float64"] * num_free_symbol), tail
    )


@functools.lru_cache
def _deco_F(parallel: bool, fastmath: bool, cache: bool) -> str:
    """Decorator for the aggregate function F (value)."""
    tail = ""
    if parallel:
        tail += ", parallel=True"
    if fastmath:
        tail += ", fastmath=True"
    if cache:
        tail += ", cache=True"
    return '@numba.njit("float64[:](float64[:], int32)"{})\n'.format(tail)


@functools.lru_cache
def _deco_GH(parallel: bool, fastmath: bool, cache: bool) -> str:
    """Decorator for the aggregate functions G (gradient) and H (hessian)."""
    tail = ""
    if parallel:
        tail += ", parallel=True"
    if fastmath:
        tail += ", fastmath=True"
    if cache:
        tail += ", cache=True"
    return '@numba.njit("float64[:, :](float64[:], int32)"{})\n'.format(tail)


def _load_module(path: str):
    spec = importlib.util.spec_from_file_location("module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fnv1a_hash(text: str) -> int:
    # FNV-1a hash parameters
    FNV_PRIME = 0x01000193
    FNV_OFFSET_BASIS = 0x811C9DC5
    
    # Convert string to bytes using UTF-8 encoding
    byte_array = text.encode('utf-8')
    
    # Compute hash
    hash_value = FNV_OFFSET_BASIS
    for byte in byte_array:
        hash_value ^= byte
        hash_value = (hash_value * FNV_PRIME) & 0xFFFFFFFF
        
    return hash_value


class FastFunc:
    """JITed, vectorized functions with automatic differentiation to compute
    value, gradient, and hessian.

    FastFunc takes a Sympy expression and a list of Sympy symbols as
    function arguments, generates JITed, vectorized functions for value,
    gradient, and hessian. Gradient and hessian are generated by
    automatic differentiation and in sparse format, i.e., only non-zero
    elements are computed.
    """

    F: Callable[[VecFloat, np.int32], VecFloat]
    """Vectorized function to compute value."""
    G: Callable[[VecFloat, np.int32], VecFloat]
    """Vectorized function to compute gradient."""
    H: Callable[[VecFloat, np.int32], VecFloat]
    """Vectorized function to compute hessian."""
    G_index: VecInt
    """Indices of non-zero elements for gradient."""
    H_index: tuple[VecInt, VecInt]
    """Indices of non-zero elements for hessian.

    The first element is the row index, the second element is the column
    index. Only the lower triangular part of the hessian matrix is
    stored.
    """

    def __init__(
        self,
        function: int | float | sp.Expr,
        args: list[sp.Symbol],
        simplify: bool = False,
        parallel: bool = False,
        fastmath: bool = False,
        *,
        cache: Optional[str] = None,
    ) -> None:
        """Suppose the expression of the input function is ``f(a_1, a_2, ...,
        a_n)``, with ``n`` arguments.

        ``(a_1, a_2, ..., a_n)`` input as the second argument ``args``. The generated functions ``F``, ``G``, and ``H``
        take two arguments ``x`` and ``k``, where ``x`` is a 1D array of length ``n * k``, and ``k`` is an integer.
        The first ``n`` elements of ``x`` are the values of ``a_1`` at ``k`` different points, the next ``n`` elements are
        the values of ``a_2``, and so on. The return value of ``F`` is a 1D array of length ``k``, where the ``i``-th
        element is the value of ``f(a_1, a_2, ..., a_n)`` at the ``i``-th point. The return value of ``G`` is a 2D array
        of shape ``(len(G_index), k)``, where ``G_index`` is the indices of non-zero elements in the gradient matrix.
        The return value of ``H`` is a 2D array of shape ``(len(H_index[0]), k)``, where ``H_index`` is the indices of
        non-zero elements in the lower triangular part of the hessian matrix.

        If ``simplify`` is ``True``, every symbolic expression will be simplified (by :func:`sympy.simplify`) before
        being compiled. This will slow down the speed of compilation.

        If ``parallel`` is ``True``, the ``parallel`` flag will be passed to the Numba JIT compiler,
        which will generate parallel code for multicore CPUs.
        This will slow down the speed of compilation and sometimes the speed of execution.

        If ``fastmath`` is ``True``, the ``fastmath`` flag will be passed to the Numba JIT compiler,
        see [Numba](https://numba.pydata.org/numba-doc/latest/user/performance-tips.html#fastmath)
        and [LLVM](https://llvm.org/docs/LangRef.html#fast-math-flags) documentations for details.

        Args:
            function: :class:`Sympy.Expr` of the function.
            args: :class:`sympy.Symbol` s of the function arguments.
            simplify: Whether to use :func:`Sympy.simplify` to simplify expressions before compilation.
            parallel: Whether to use Numba ``parallel`` mode.
            fastmath: Whether to use Numba ``fastmath`` mode.
        """
        self._function = sp.sympify(function)
        self._args = args

        # random valid Python name to avoid conflict
        self._valid_args = [
            sp.Symbol("xbN4dRhrnN_{}".format(i)) for i in range(len(args))
        ]
        for i in range(len(args)):
            self._function = self._function.subs(args[i], self._valid_args[i])
        self._args = self._valid_args

        self._hash = "# hash {}\n".format(_fnv1a_hash(str(self._function) + str(len(self._args))))

        if cache is not None and os.path.isfile(cache):
            with open(cache, "r") as file:
                hash_file = file.readline()
                if hash_file == self._hash:
                    # The file is auto-generated and matches the current function
                    self._load(cache)
                    return
                elif not hash_file.startswith("# hash"):
                    # The file is not auto-generated (no hash found)
                    # Load the user-provided file directly
                    self._load(cache)
                    return

        self._simplify = simplify
        self._parallel = parallel
        self._fastmath = fastmath
        self._cache = cache

        self._value_index = []
        self._grad_index = []
        self._hess_index = []

        self._value_consts = None
        self._grad_consts = {}
        self._hess_consts = {}

        self.__expand_opt = create_expand_pow_optimization(3)

        basic_str = self._gen_basic_funcs()
        aggregate_str = self._gen_aggregate_funcs()

        self.G_index = self._gen_G_index()
        self.H_index_row, self.H_index_col = self._gen_H_index()
        self._compile(basic_str + aggregate_str, cache)

    def _load(self, cache: str) -> None:
        module = _load_module(cache)
        self.F = module.F
        self.G = module.G
        self.H = module.H
        self.G_index = module.G_index
        self.H_index_row = module.H_index_row
        self.H_index_col = module.H_index_col

    def _compile(self, code: str, cache: Optional[str]) -> None:
        if cache is not None:
            path = cache
            file = open(path, "w")
            file.write(self._hash)
        else:
            file = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
            path = file.name
        file.write("from numpy import *\n")
        file.write("import numba\n\n")
        file.write(code)
        if cache is not None:
            file.write("G_index = array([{}], dtype=int32)\n".format(", ".join(map(str, self.G_index))))
            file.write("H_index_row = array([{}], dtype=int32)\n".format(", ".join(map(str, self.H_index_row))))
            file.write("H_index_col = array([{}], dtype=int32)\n".format(", ".join(map(str, self.H_index_col))))
        file.close()

        module = _load_module(path)

        self.F = module.F
        self.G = module.G
        self.H = module.H

        if cache is None:
            os.unlink(path)

    def _free_symbols(self, expr: sp.Expr) -> list[int]:
        return sorted([self._args.index(sym) for sym in expr.free_symbols])

    def _gen_basic_funcs(self) -> str:
        result_str = ""
        func = self._function

        # value
        if self._simplify:
            func = sp.simplify(func)
        if func.is_constant(simplify=self._simplify):
            self._value_consts = float(func)
            return ""

        free_symbols_value = self._free_symbols(func)
        self._value_index = free_symbols_value
        deco_str = _deco_basic(len(free_symbols_value), self._parallel, self._fastmath, self._cache is not None)
        func_str = lambdarepr(self.__expand_opt(func)).replace("math.", "")
        param_str = ", ".join([self._args[i].name for i in free_symbols_value])
        result_str += (
            deco_str
            + "def f({}):\n".format(param_str)
            + "    return "
            + func_str
            + "\n\n"
        )

        for j in free_symbols_value:  # row of hessian
            # gradient
            grad = sp.diff(func, self._args[j])
            if self._simplify:
                grad = sp.simplify(grad)
            if grad == 0:
                continue
            if grad.is_constant(simplify=self._simplify):
                self._grad_consts[j] = float(grad)
                self._grad_index.append((j, []))
                continue

            free_symbols_grad = self._free_symbols(grad)
            self._grad_index.append((j, free_symbols_grad))
            deco_str = _deco_basic(
                len(free_symbols_grad), self._parallel, self._fastmath, self._cache is not None
            )
            grad_str = lambdarepr(self.__expand_opt(grad)).replace("math.", "")
            param_str = ", ".join([self._args[k].name for k in free_symbols_grad])
            result_str += (
                deco_str
                + "def df_d{}({}):\n".format(j, param_str)
                + "    return "
                + grad_str
                + "\n\n"
            )

            # hessian
            for k in free_symbols_value:  # col of hessian
                if j < k:
                    break
                hess = sp.diff(grad, self._args[k])
                if self._simplify:
                    hess = sp.simplify(hess)
                if hess == 0:
                    continue
                if hess.is_constant(simplify=self._simplify):
                    self._hess_consts[(j, k)] = float(hess)
                    self._hess_index.append((j, k, []))
                    continue

                free_symbols_hess = self._free_symbols(hess)
                self._hess_index.append((j, k, free_symbols_hess))
                deco_str = _deco_basic(
                    len(free_symbols_hess), self._parallel, self._fastmath, self._cache is not None
                )
                hess_str = lambdarepr(self.__expand_opt(hess)).replace("math.", "")
                param_str = ", ".join([self._args[l].name for l in free_symbols_hess])
                result_str += (
                    deco_str
                    + "def d2f_d{}d{}({}):\n".format(j, k, param_str)
                    + "    return "
                    + hess_str
                    + "\n\n"
                )
        return result_str

    def _gen_aggregate_funcs(self) -> str:
        result_str = ""
        # value
        result_str += _deco_F(self._parallel, self._fastmath, self._cache is not None)
        result_str += "def F(x, n):\n"
        result_str += "    r = zeros(n, dtype=float64)\n"
        if self._value_consts is not None:
            result_str += f"    r[:] = {self._value_consts}\n"
        else:
            param_str = ", ".join(
                [f"x[{j} * n:{j + 1} * n]" for j in self._value_index]
            )
            result_str += "    r = f({})\n".format(param_str)
        result_str += "    return r\n\n"

        # gradient
        result_str += _deco_GH(self._parallel, self._fastmath, self._cache is not None)
        result_str += "def G(x, n):\n"
        result_str += f"    r = zeros(({len(self._grad_index)}, n), dtype=float64)\n"
        for row, (x, fs) in enumerate(self._grad_index):  # fs for free symbols
            if x in self._grad_consts:
                result_str += f"    r[{row}] = {self._grad_consts[x]}\n"
            else:
                param_str = ", ".join([f"x[{j} * n:{j + 1} * n]" for j in fs])
                result_str += "    r[{}] = df_d{}({})\n".format(row, x, param_str)
        result_str += "    return r\n\n"

        # hessian
        result_str += _deco_GH(self._parallel, self._fastmath, self._cache is not None)
        result_str += "def H(x, n):\n"
        result_str += f"    r = zeros(({len(self._hess_index)}, n), dtype=float64)\n"
        for row, (x, y, fs) in enumerate(self._hess_index):
            if (x, y) in self._hess_consts:
                result_str += f"    r[{row}] = {self._hess_consts[(x, y)]}\n"
            else:
                param_str = ", ".join([f"x[{j} * n:{j + 1} * n]" for j in fs])
                result_str += "    r[{}] = d2f_d{}d{}({})\n".format(
                    row, x, y, param_str
                )
        result_str += "    return r\n\n"

        return result_str

    def _gen_G_index(self) -> VecInt:
        return np.array([x for x, _ in self._grad_index], dtype=np.int32)

    def _gen_H_index(self) -> tuple[VecInt, VecInt]:
        return np.array([x for x, _, _ in self._hess_index], dtype=np.int32), np.array(
            [y for _, y, _ in self._hess_index], dtype=np.int32
        )
