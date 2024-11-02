# Copyright (c) 2024 Yilin Zou
import numpy as np
import pytest

from pockit.base.discretizationbase import lr_nc
from pockit.base.variablebase import (
    V_interpolation,
    D_interpolation,
    BatchIndexArray,
    VariableBase,
)


def test_V_interpolation():
    t_old = np.linspace(0, 2 * np.pi, 11)
    y_old = np.sin(t_old)

    t_new = t_old
    y_new = np.sin(t_new)
    V = V_interpolation(t_old, t_new)
    assert np.allclose(V @ y_old, y_new)

    t_new = t_old[1:]
    y_new = np.sin(t_new)
    V = V_interpolation(t_old, t_new)
    assert np.allclose(V @ y_old, y_new)

    t_new = np.linspace(0, 2 * np.pi, 7)
    y_new = np.sin(t_new)
    V = V_interpolation(t_old, t_new)
    assert np.allclose(V @ y_old, y_new)


def test_D_interpolation():
    t_old = np.linspace(0, 2 * np.pi, 14)
    y_old = np.sin(t_old)

    t_new = t_old
    y_new = np.cos(t_new)
    D = D_interpolation(t_old, t_new)
    assert np.allclose(D @ y_old, y_new)

    t_new = t_old[1:]
    y_new = np.cos(t_new)
    D = D_interpolation(t_old, t_new)
    assert np.allclose(D @ y_old, y_new)

    t_new = np.linspace(0, 2 * np.pi, 7)
    y_new = np.cos(t_new)
    D = D_interpolation(t_old, t_new)
    assert np.allclose(D @ y_old, y_new)


def test_batch_index_array():
    data = np.arange(10)
    l_ind = np.array([0, 1, 3, 6])
    r_ind = np.array([1, 3, 6, 10])
    batch_index_array = BatchIndexArray(data, l_ind, r_ind)
    assert np.allclose(batch_index_array[0], [0])
    assert np.allclose(batch_index_array[1], [1, 2])
    assert np.allclose(batch_index_array[2], [3, 4, 5])
    assert np.allclose(batch_index_array[3], [6, 7, 8, 9])
    with pytest.raises(IndexError):
        _ = batch_index_array[4]
    assert len(batch_index_array) == 4


class PhaseMinimal:
    def __init__(self, n_x, n_u, mesh, num_point):
        self.L_v = sum(num_point)
        self.n_x = n_x
        self.n_u = n_u
        self.n = n_x + n_u
        self.l_v, self.r_v = lr_nc(np.full(self.n, self.L_v, dtype=np.int32))

        self._mesh = mesh
        self._num_point = num_point
        self.N = len(num_point)

        self.t_x = np.concatenate(
            [np.linspace(mesh[i], mesh[i + 1], num_point[i]) for i in range(self.N)]
        )
        self.l_x, self.r_x = lr_nc(num_point)
        self.t_u, self.l_u, self.r_u = self.t_x, self.l_x, self.r_x


class VariableMinimal(VariableBase):
    def _assemble_x(self, V_interval):
        pass

    def _assemble_u(self, V_interval):
        pass


class TestVariableBase:
    N = 10
    mesh = np.linspace(0, 1, N + 1)
    num_point = np.full(N, 3, dtype=np.int32)
    p = PhaseMinimal(1, 1, mesh, num_point)
    v = VariableMinimal(p, np.zeros(p.r_v[-1] + 2))
    v.t_f = 10

    def test_interval_partition(self):
        t = np.array([0.0, 0.05, 0.3, 0.3, 0.35, 1.0])
        interval_info_c = self.v._interval_partition(t)
        assert np.allclose(interval_info_c[0], [0.0, 0.05])  # 0 - 0.1
        assert np.allclose(interval_info_c[1], [])  # 0.1 - 0.2
        assert np.allclose(
            interval_info_c[2],
            [
                0.3,
            ],
        )  # 0.2 - 0.3
        assert np.allclose(interval_info_c[3], [0.3, 0.35])  # 0.3 - 0.4
        for i in range(4, 9):
            assert np.allclose(interval_info_c[i], [])
        assert np.allclose(interval_info_c[9], [1.0])  # 0.9 - 1.0

    def test_guard_t(self):
        t = np.array([0.0, 0.05, 0.3, 0.25, 1.0])
        with pytest.raises(ValueError):
            self.v._guard_t(t)
        t = np.array([0.0, 0.05, 0.30000000001, 0.3, 0.35, 1.0])
        self.v._guard_t(t)

        t = np.array([-1.0, 0.0, 1.0])
        with pytest.raises(ValueError):
            self.v._guard_t(t)
        t = np.array([1.0, 11.0])
        with pytest.raises(ValueError):
            self.v._guard_t(t)

        t = np.linspace(0, 1, 7)
        assert np.allclose(self.v._guard_t(t), t / 10)

    def test_assemble_c(self):
        num_point = np.array([2, 3, 4], dtype=np.int32)
        V_interval = [np.full((2, 2), 1), np.full((0, 3), 2), np.full((1, 4), 3)]
        assert np.allclose(
            self.v._assemble_c(num_point, V_interval).toarray(),
            np.array(
                [[1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0], [0, 0, 0, 3, 3, 3, 3]]
            ),
        )

    def test_assemble_nc(self):
        num_point = np.array([2, 3, 4], dtype=np.int32)
        V_interval = [np.full((2, 2), 1), np.full((0, 3), 2), np.full((1, 4), 3)]
        assert np.allclose(
            self.v._assemble_nc(V_interval).toarray(),
            np.array(
                [
                    [1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 3, 3, 3, 3],
                ]
            ),
        )
