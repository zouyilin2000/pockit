# Copyright (c) 2024 Yilin Zou
import numpy as np

from pockit.optimizer import scipy


def test_reflection():
    # row >= col
    row = np.array([3, 2, 2, 0, 2, 2, 1])
    col = np.array([3, 2, 0, 0, 1, 2, 0])
    func = lambda _: np.arange(7) ** 2
    func_2 = scipy._reflection(func, row, col, 4)
    res = np.array([[9, 36, 4, 0], [36, 0, 16, 0], [4, 16, 1 + 25, 0], [0, 0, 0, 0]])
    assert np.all(func_2(None).toarray() == res)
