# Copyright (c) 2024 Yilin Zou
from pockit.radau.discretization import *


def test_xw_lgr():
    x, w = xw_lgr(1)
    assert np.allclose(x, [-1.0])
    assert np.allclose(w, [2.0])

    x, w = xw_lgr(2)
    assert np.allclose(x, [-1.0, 0.333333])
    assert np.allclose(w, [0.5, 1.5])

    x, w = xw_lgr(3)
    assert np.allclose(x, [-1.0, -0.289898, 0.689898])
    assert np.allclose(w, [0.222222, 1.02497, 0.752806])

    x, w = xw_lgr(4)
    assert np.allclose(x, [-1.0, -0.575319, 0.181066, 0.822824])
    assert np.allclose(w, [0.125, 0.657689, 0.776387, 0.440924])

    x, w = xw_lgr(5)
    assert np.allclose(x, [-1.0, -0.72048, -0.167181, 0.446314, 0.885792])
    assert np.allclose(w, [0.08, 0.446208, 0.623653, 0.562712, 0.287427])


def test_T_lgr():
    x, _ = xw_lgr(10)
    x_1 = np.concatenate([x, [1.0]])
    y_1 = x_1**2 + 100
    assert np.allclose(T_lgr(10) @ y_1, y_1[:-1] - y_1[-1])

    x, _ = xw_lgr(10)
    x_1 = np.concatenate([x, [1.0]])
    y_1 = np.cos(x_1) - np.pi
    assert np.allclose(T_lgr(10) @ y_1, y_1[:-1] - y_1[-1])

    x, _ = xw_lgr(20)
    x_1 = np.concatenate([x, [1.0]])
    y_1 = np.exp(x_1) * 10
    assert np.allclose(T_lgr(20) @ y_1, y_1[:-1] - y_1[-1])


def test_I_lgr():
    x, _ = xw_lgr(10)
    y = 2 * x
    I_y = x**2 - 1**2
    assert np.allclose(I_lgr(10) @ y, I_y)

    x, _ = xw_lgr(10)
    y = np.cos(x)
    I_y = np.sin(x) - np.sin(1)
    assert np.allclose(I_lgr(10) @ y, I_y)

    x, _ = xw_lgr(20)
    y = np.exp(x) * 10
    I_y = np.exp(x) * 10 - np.exp(1) * 10
    assert np.allclose(I_lgr(20) @ y, I_y)


def test_P_lgr():
    x, _ = xw_lgr(10)
    y = (x + 1) ** 2
    P = P_lgr(10)
    poly = np.poly1d(P @ y)
    assert np.allclose(poly(x), y)


def test_V_lgr_x_aug():
    x, _ = xw_lgr(10)
    x_aug, _ = xw_lgr(11)
    x_1 = np.concatenate([x, [1.0]])
    y_1 = x_1**2 + 100
    y_aug = x_aug**2 + 100
    assert np.allclose(V_lgr_x_aug(10) @ y_1, y_aug)

    x, _ = xw_lgr(10)
    x_aug, _ = xw_lgr(11)
    x_1 = np.concatenate([x, [1.0]])
    y_1 = np.cos(x_1) - np.pi
    y_aug = np.cos(x_aug) - np.pi
    assert np.allclose(V_lgr_x_aug(10) @ y_1, y_aug)

    x, _ = xw_lgr(20)
    x_aug, _ = xw_lgr(21)
    x_1 = np.concatenate([x, [1.0]])
    y_1 = np.exp(x_1) * 10
    y_aug = np.exp(x_aug) * 10
    assert np.allclose(V_lgr_x_aug(20) @ y_1, y_aug)


def test_V_lgr_u_aug():
    x, _ = xw_lgr(10)
    x_aug, _ = xw_lgr(11)
    y = x**2 + 100
    y_aug = x_aug**2 + 100
    assert np.allclose(V_lgr_u_aug(10) @ y, y_aug)

    x, _ = xw_lgr(10)
    x_aug, _ = xw_lgr(11)
    y = np.cos(x) - np.pi
    y_aug = np.cos(x_aug) - np.pi
    assert np.allclose(V_lgr_u_aug(10) @ y, y_aug)

    x, _ = xw_lgr(20)
    x_aug, _ = xw_lgr(21)
    y = np.exp(x) * 10
    y_aug = np.exp(x_aug) * 10
    assert np.allclose(V_lgr_u_aug(20) @ y, y_aug)


def test_T_lgr_aug():
    x, _ = xw_lgr(10)
    x_aug, _ = xw_lgr(11)
    x_1 = np.concatenate([x, [1.0]])
    y_1 = x_1**2 + 100
    y_aug = x_aug**2 + 100 - (1**2 + 100)
    assert np.allclose(T_lgr_aug(10) @ y_1, y_aug)

    x, _ = xw_lgr(10)
    x_aug, _ = xw_lgr(11)
    x_1 = np.concatenate([x, [1.0]])
    y_1 = np.cos(x_1) - np.pi
    y_aug = np.cos(x_aug) - np.pi - (np.cos(1) - np.pi)
    assert np.allclose(T_lgr_aug(10) @ y_1, y_aug)

    x, _ = xw_lgr(20)
    x_aug, _ = xw_lgr(21)
    x_1 = np.concatenate([x, [1.0]])
    y_1 = np.exp(x_1) * 10
    y_aug = np.exp(x_aug) * 10 - np.exp(1) * 10
    assert np.allclose(T_lgr_aug(20) @ y_1, y_aug)


class TestDiscretizationRadau:
    mesh = np.array([0.0, 0.1, 1.0], dtype=np.float64)
    num_point = np.array([2, 3], dtype=np.int32)
    n_x = 1
    n_u = 1
    d = Discretization(mesh, num_point, n_x, n_u)

    def test_l_v(self):
        assert np.allclose(self.d.l_v, [0, 6])

    def test_r_v(self):
        assert np.allclose(self.d.r_v, [6, 11])

    def test_t_m(self):
        x_2, _ = xw_lgr(2)
        x_3, _ = xw_lgr(3)
        x = np.concatenate([(x_2 + 1) / 2 * 0.1, 0.1 + (x_3 + 1) / 2 * 0.9])
        assert np.allclose(self.d.t_m, x)

    def test_w_m(self):
        _, w_2 = xw_lgr(2)
        _, w_3 = xw_lgr(3)
        w = np.concatenate([w_2 / 2 * 0.1, w_3 / 2 * 0.9])
        assert np.allclose(self.d.w_m, w)

    def test_l_m(self):
        assert np.allclose(self.d.l_m, [0, 2])

    def test_r_m(self):
        assert np.allclose(self.d.r_m, [2, 5])

    def test_L_m(self):
        assert np.allclose(self.d.L_m, 5)

    def test_f_v2m(self):
        v = np.arange((self.d.L_m + 1) * 1 + self.d.L_m * 1)
        assert np.allclose(self.d.f_v2m(v), [0, 1, 2, 3, 4, 6, 7, 8, 9, 10])

    def test_T_v(self):
        v = np.arange(self.d.L_m + 1)
        assert np.allclose(self.d.T_v @ v, [-2, -1, -3, -2, -1])

    def test_I_m(self):
        y = 2 * self.d.t_m
        I_y = self.d.t_m**2
        I_y[:2] -= 0.1**2
        I_y[2:] -= 1.0**2
        assert np.allclose(self.d.I_m @ y, I_y)

    def test_l_d(self):
        assert np.allclose(self.d.l_d, [0])

    def test_r_d(self):
        assert np.allclose(self.d.r_d, [5])

    def test_t_m_aug(self):
        x_3, _ = xw_lgr(3)
        x_4, _ = xw_lgr(4)
        x = np.concatenate([(x_3 + 1) / 2 * 0.1, 0.1 + (x_4 + 1) / 2 * 0.9])
        assert np.allclose(self.d.t_m_aug, x)

    def test_l_m_aug(self):
        assert np.allclose(self.d.l_m_aug, [0, 3])

    def test_r_m_aug(self):
        assert np.allclose(self.d.r_m_aug, [3, 7])

    def test_L_m_aug(self):
        assert np.allclose(self.d.L_m_aug, 7)

    def test_w_aug(self):
        _, w_3 = xw_lgr(2)
        _, w_4 = xw_lgr(3)
        assert np.allclose(self.d.w_aug[0], w_3)
        assert np.allclose(self.d.w_aug[1], w_4)

    def test_V_xu_aug(self):
        v_x = np.concatenate([self.d.t_m, [1.0]]) + 2
        v_u = self.d.t_m * 2
        v = np.concatenate([v_x, v_u])

        v_x_aug = self.d.t_m_aug + 2
        v_u_aug = self.d.t_m_aug * 2
        v_aug = np.concatenate([v_x_aug, v_u_aug])

        assert np.allclose(self.d.V_xu_aug @ v, v_aug)

    def test_T_x_aug(self):
        v_x = np.concatenate([self.d.t_m, [1.0]]) ** 2
        v_x_aug = self.d.t_m_aug**2
        v_x_aug[:3] -= 0.1**2
        v_x_aug[3:] -= 1.0**2
        assert np.allclose(self.d.T_x_aug @ v_x, v_x_aug)

    def test_I_m_aug(self):
        m_aug = 2 * self.d.t_m_aug
        I_m_aug = self.d.t_m_aug**2
        I_m_aug[:3] -= 0.1**2
        I_m_aug[3:] -= 1.0**2
        assert np.allclose(self.d.I_m_aug @ m_aug, I_m_aug)

    def test_t_x(self):
        assert np.allclose(self.d.t_x, np.concatenate([self.d.t_m, [1.0]]))

    def test_t_u(self):
        assert np.allclose(self.d.t_u, self.d.t_m)

    def test_l_x(self):
        assert np.allclose(self.d.l_x, [0, 2])

    def test_r_x(self):
        assert np.allclose(self.d.r_x, [3, 6])

    def test_l_u(self):
        assert np.allclose(self.d.l_u, [0, 2])

    def test_r_u(self):
        assert np.allclose(self.d.r_u, [2, 5])
