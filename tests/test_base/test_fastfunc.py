# Copyright (c) 2024 Yilin Zou
import subprocess
import sys

import numpy as np
import sympy as sp

from pockit.base.fastfunc import FastFunc


def test_constant_function():
    ff = FastFunc(1, [])
    v = np.zeros(100, dtype=np.float64)
    assert np.allclose(ff.F(v, 10), np.ones(10, dtype=np.float64))
    assert np.allclose(ff.G(v, 10), np.zeros((0, 10), dtype=np.float64))
    assert np.allclose(ff.H(v, 10), np.zeros((0, 10), dtype=np.float64))
    assert np.allclose(ff.G_index, np.zeros(0, dtype=np.float64))
    assert np.allclose(ff.H_index_row, np.zeros(0, dtype=np.float64))
    assert np.allclose(ff.H_index_col, np.zeros(0, dtype=np.float64))

    x = sp.symbols("x")
    ff = FastFunc(1, [x])
    v = np.zeros(100, dtype=np.float64)
    assert np.allclose(ff.F(v, 10), np.ones(10, dtype=np.float64))
    assert np.allclose(ff.G(v, 10), np.zeros((0, 10), dtype=np.float64))
    assert np.allclose(ff.H(v, 10), np.zeros((0, 10), dtype=np.float64))
    assert np.allclose(ff.G_index, np.zeros(0, dtype=np.float64))
    assert np.allclose(ff.H_index_row, np.zeros(0, dtype=np.float64))
    assert np.allclose(ff.H_index_col, np.zeros(0, dtype=np.float64))


def test_derivative():
    x, y = sp.symbols("x, y")
    ff = FastFunc(x + y**2, [x, y])  # constant / non-constant, respectively
    t = np.arange(10, dtype=np.float64)
    v_x = np.sin(t)
    v_y = t * 2
    v = np.concatenate((v_x, v_y))
    assert np.allclose(ff.F(v, 10), v_x + v_y**2)
    assert np.allclose(ff.G(v, 10), np.vstack([np.ones(10, dtype=np.float64), 2 * v_y]))
    assert np.allclose(ff.H(v, 10), np.zeros((0, 10), dtype=np.float64))
    assert np.allclose(ff.G_index, np.array([0, 1]))
    assert np.allclose(ff.H_index_row, np.zeros(0, dtype=np.float64))
    assert np.allclose(ff.H_index_col, np.zeros(0, dtype=np.float64))


def test_hessian():
    x, y = sp.symbols("x, y")
    ff = FastFunc(x * y + y**3, [x, y])
    t = np.arange(10, dtype=np.float64)
    v_x = np.sin(t)
    v_y = t * 2
    v = np.concatenate((v_x, v_y))
    assert np.allclose(ff.F(v, 10), v_x * v_y + v_y**3)
    assert np.allclose(ff.G(v, 10), np.vstack([v_y, 3 * v_y**2 + v_x]))
    assert np.allclose(ff.H(v, 10), np.vstack([np.ones(10, dtype=np.float64), 6 * v_y]))
    assert np.allclose(ff.G_index, np.array([0, 1]))
    assert np.allclose(ff.H_index_row, np.array([1, 1]))
    assert np.allclose(ff.H_index_col, np.array([0, 1]))  # (r, c) must in order

    ff = FastFunc(x**2 * y + y**3, [x, y])
    t = np.arange(10, dtype=np.float64)
    v_x = np.sin(t)
    v_y = t * 2
    v = np.concatenate((v_x, v_y))
    assert np.allclose(ff.F(v, 10), v_x**2 * v_y + v_y**3)
    assert np.allclose(ff.G(v, 10), np.vstack([2 * v_x * v_y, 3 * v_y**2 + v_x**2]))
    assert np.allclose(ff.H(v, 10), np.vstack([2 * v_y, 2 * v_x, 6 * v_y]))
    assert np.allclose(ff.G_index, np.array([0, 1]))
    assert np.allclose(ff.H_index_row, np.array([0, 1, 1]))
    assert np.allclose(ff.H_index_col, np.array([0, 0, 1]))  # (r, c) must in order


def test_configs():
    x, y, z = sp.symbols("x, y, z")
    FastFunc(x + y**2, [x, y, z], fastmath=True)
    FastFunc(x + y**2, [x, y, z], simplify=True)


def test_cached_modules_have_unique_stable_names(tmp_path):
    x = sp.symbols("x")
    cache_1 = str(tmp_path / "function_1.py")
    cache_2 = str(tmp_path / "function_2.py")

    ff_1 = FastFunc(x, [x], cache=cache_1)
    ff_2 = FastFunc(x**2, [x], cache=cache_2)
    ff_1_reloaded = FastFunc(x, [x], cache=cache_1)

    assert ff_1.F.py_func.__module__ != ff_2.F.py_func.__module__
    assert ff_1.F.py_func.__module__ == ff_1_reloaded.F.py_func.__module__

    value = np.arange(5, dtype=np.float64)
    assert np.allclose(ff_1_reloaded.F(value, 5), value)
    assert np.allclose(ff_2.F(value, 5), value**2)

    subprocess.run(
        [
            sys.executable,
            "-c",
            f"""
import numpy as np
import sympy as sp
from pockit.base.fastfunc import FastFunc

x = sp.symbols("x")
value = np.arange(5, dtype=np.float64)
ff_1 = FastFunc(x, [x], cache={cache_1!r})
ff_2 = FastFunc(x**2, [x], cache={cache_2!r})
assert ff_1.F.py_func.__module__ != ff_2.F.py_func.__module__
assert np.allclose(ff_1.F(value, 5), value)
assert np.allclose(ff_2.F(value, 5), value**2)
""",
        ],
        check=True,
    )
