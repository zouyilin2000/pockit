# Copyright (c) 2024 Yilin Zou
import numpy as np

from pockit.base.discretizationbase import lr_c, lr_nc


def test_lr_c():
    num_point = np.array([2, 3, 4, 5], dtype=np.int32)
    l_c, r_c = lr_c(num_point)
    assert np.allclose(l_c, [0, 1, 3, 6])
    assert np.allclose(r_c, [2, 4, 7, 11])


def test_lr_nc():
    num_point = np.array([2, 3, 4, 5], dtype=np.int32)
    l_nc, r_nc = lr_nc(num_point)
    assert np.allclose(l_nc, [0, 2, 5, 9])
    assert np.allclose(r_nc, [2, 5, 9, 14])
