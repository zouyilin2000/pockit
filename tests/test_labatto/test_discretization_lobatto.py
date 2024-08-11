from pockit.lobatto.discretization import *


def test_xw_lgl():
    x, w = xw_lgl(1)
    assert np.allclose(x, [0.0])
    assert np.allclose(w, [2.0])

    x, w = xw_lgl(2)
    assert np.allclose(x, [-1.0, 1.0])
    assert np.allclose(w, [1.0, 1.0])

    x, w = xw_lgl(3)
    assert np.allclose(x, [-1.0, 0.0, 1.0])
    assert np.allclose(w, [1 / 3, 4 / 3, 1 / 3])

    x, w = xw_lgl(4)
    assert np.allclose(x, [-1.0, -1 / np.sqrt(5), 1 / np.sqrt(5), 1])
    assert np.allclose(w, [1 / 6, 5 / 6, 5 / 6, 1 / 6])

    x, w = xw_lgl(5)
    assert np.allclose(x, [-1.0, -np.sqrt(3 / 7), 0, np.sqrt(3 / 7), 1])
    assert np.allclose(w, [1 / 10, 49 / 90, 32 / 45, 49 / 90, 1 / 10])

    x, w = xw_lgl(10)
    assert np.allclose(
        x,
        [
            -1.0,
            -0.9195339081664588138289,
            -0.7387738651055050750031,
            -0.4779249498104444956612,
            -0.1652789576663870246262,
            0.1652789576663870246262,
            0.4779249498104444956612,
            0.7387738651055050750031,
            0.9195339081664588138289,
            1.0,
        ],
    )
    assert np.allclose(
        w,
        [
            0.02222222222222222222222,
            0.1333059908510701111262,
            0.2248893420631264521195,
            0.2920426836796837578756,
            0.3275397611838974566565,
            0.3275397611838974566565,
            0.292042683679683757876,
            0.224889342063126452119,
            0.133305990851070111126,
            0.02222222222222222222222,
        ],
    )


def test_T_lgl():
    x, _ = xw_lgl(10)
    y = x**2 + 100
    assert np.allclose(T_lgl(10) @ y, y[:-1] - y[-1])

    x, _ = xw_lgl(10)
    y = np.cos(x) - np.pi
    assert np.allclose(T_lgl(10) @ y, y[:-1] - y[-1])

    x, _ = xw_lgl(20)
    y = np.exp(x) * 10
    assert np.allclose(T_lgl(20) @ y, y[:-1] - y[-1])


def test_I_lgl():
    x, _ = xw_lgl(10)
    y = 2 * x
    I_y = x**2 - 1**2
    assert np.allclose(I_lgl(10) @ y, I_y[:-1])

    x, _ = xw_lgl(10)
    y = np.cos(x)
    I_y = np.sin(x) - np.sin(1)
    assert np.allclose(I_lgl(10) @ y, I_y[:-1])

    x, _ = xw_lgl(20)
    y = np.exp(x) * 10
    I_y = np.exp(x) * 10 - np.exp(1) * 10
    assert np.allclose(I_lgl(20) @ y, I_y[:-1])


def test_P_lgl():
    x, _ = xw_lgl(10)
    y = (x + 1) ** 2
    P = P_lgl(10)
    poly = np.poly1d(P @ y)
    assert np.allclose(poly(x), y)


def test_V_lgl_aug():
    x, _ = xw_lgl(10)
    x_aug, _ = xw_lgl(11)
    y = x**2 + 100
    y_aug = x_aug**2 + 100
    assert np.allclose(V_lgl_aug(10) @ y, y_aug)

    x, _ = xw_lgl(10)
    x_aug, _ = xw_lgl(11)
    y = np.cos(x) - np.pi
    y_aug = np.cos(x_aug) - np.pi
    assert np.allclose(V_lgl_aug(10) @ y, y_aug)

    x, _ = xw_lgl(20)
    x_aug, _ = xw_lgl(21)
    y = np.exp(x) * 10
    y_aug = np.exp(x_aug) * 10
    assert np.allclose(V_lgl_aug(20) @ y, y_aug)


def test_T_lgl_aug():
    x, _ = xw_lgl(10)
    x_aug, _ = xw_lgl(11)
    y_1 = x**2 + 100
    y_aug = x_aug**2 + 100 - (1**2 + 100)
    assert np.allclose(T_lgl_aug(10) @ y_1, y_aug[:-1])

    x, _ = xw_lgl(10)
    x_aug, _ = xw_lgl(11)
    y_1 = np.cos(x) - np.pi
    y_aug = np.cos(x_aug) - np.pi - (np.cos(1) - np.pi)
    assert np.allclose(T_lgl_aug(10) @ y_1, y_aug[:-1])

    x, _ = xw_lgl(20)
    x_aug, _ = xw_lgl(21)
    y_1 = np.exp(x) * 10
    y_aug = np.exp(x_aug) * 10 - np.exp(1) * 10
    assert np.allclose(T_lgl_aug(20) @ y_1, y_aug[:-1])


class TestDiscretizationRadau:
    mesh = np.array([0.0, 0.1, 1.0], dtype=np.float64)
    num_point = np.array([2, 3], dtype=np.int32)
    n_x = 1
    n_u = 1
    d = Discretization(mesh, num_point, n_x, n_u)

    def test_l_v(self):
        assert np.allclose(self.d.l_v, [0, 4])

    def test_r_v(self):
        assert np.allclose(self.d.r_v, [4, 8])

    def test_t_m(self):
        x_2, _ = xw_lgl(2)
        x_3, _ = xw_lgl(3)
        x = np.concatenate([(x_2[:-1] + 1) / 2 * 0.1, 0.1 + (x_3 + 1) / 2 * 0.9])
        assert np.allclose(self.d.t_m, x)

    def test_w_m(self):
        _, w_2 = xw_lgl(2)
        _, w_3 = xw_lgl(3)
        w = np.zeros(4)
        w[:2] += w_2 / 2 * 0.1
        w[1:] += w_3 / 2 * 0.9
        assert np.allclose(self.d.w_m, w)

    def test_l_m(self):
        assert np.allclose(self.d.l_m, [0, 1])

    def test_r_m(self):
        assert np.allclose(self.d.r_m, [2, 4])

    def test_L_m(self):
        assert np.allclose(self.d.L_m, 4)

    def test_f_v2m(self):
        v = np.arange(self.d.L_m * 2)
        assert np.allclose(self.d.f_v2m(v), [0, 1, 2, 3, 4, 5, 6, 7])

    def test_T_v(self):
        v = np.arange(self.d.L_m)
        assert np.allclose(self.d.T_v @ v, [-1, -2, -1])

    def test_I_m(self):
        y = 2 * self.d.t_m
        I_y = self.d.t_m**2
        I_y_2 = np.zeros(3)
        I_y_2[0] = I_y[0] - I_y[1]
        I_y_2[1] = I_y[1] - I_y[3]
        I_y_2[2] = I_y[2] - I_y[3]
        assert np.allclose(self.d.I_m @ y, I_y_2)

    def test_l_d(self):
        assert np.allclose(self.d.l_d, [0])

    def test_r_d(self):
        assert np.allclose(self.d.r_d, [3])

    def test_t_m_aug(self):
        x_3, _ = xw_lgl(3)
        x_4, _ = xw_lgl(4)
        x = np.concatenate([(x_3[:-1] + 1) / 2 * 0.1, 0.1 + (x_4 + 1) / 2 * 0.9])
        assert np.allclose(self.d.t_m_aug, x)

    def test_l_m_aug(self):
        assert np.allclose(self.d.l_m_aug, [0, 2])

    def test_r_m_aug(self):
        assert np.allclose(self.d.r_m_aug, [3, 6])

    def test_L_m_aug(self):
        assert np.allclose(self.d.L_m_aug, 6)

    def test_w_aug(self):
        _, w_3 = xw_lgl(2)
        _, w_4 = xw_lgl(3)
        assert np.allclose(self.d.w_aug[0], w_3)
        assert np.allclose(self.d.w_aug[1], w_4)

    def test_V_xu_aug(self):
        v_x = self.d.t_m + 2
        v_u = self.d.t_m * 2
        v = np.concatenate([v_x, v_u])

        v_x_aug = self.d.t_m_aug + 2
        v_u_aug = self.d.t_m_aug * 2
        v_aug = np.concatenate([v_x_aug, v_u_aug])

        assert np.allclose(self.d.V_xu_aug @ v, v_aug)

    def test_T_x_aug(self):
        x = self.d.t_m * 2
        f_x_aug = self.d.t_m_aug * 2
        t_x_aug = np.zeros(self.d.L_m_aug - 1)
        t_x_aug[:2] = f_x_aug[:2] - f_x_aug[2]
        t_x_aug[2:] = f_x_aug[2:-1] - f_x_aug[-1]
        dot_res = self.d.T_x_aug.dot(x)
        assert np.allclose(dot_res, t_x_aug)

    def test_I_m_aug(self):
        m_aug = self.d.t_m_aug * 2
        I_m_aug = self.d.t_m_aug**2
        I_m_aug_2 = np.zeros(self.d.L_m_aug - 1)
        I_m_aug_2[:2] = I_m_aug[:2] - I_m_aug[2]
        I_m_aug_2[2:] = I_m_aug[2:-1] - I_m_aug[-1]
        dot_res = self.d.I_m_aug.dot(m_aug)
        assert np.allclose(dot_res, I_m_aug_2)

    def test_t_x(self):
        assert np.allclose(self.d.t_x, self.d.t_m)

    def test_t_u(self):
        assert np.allclose(self.d.t_u, self.d.t_m)

    def test_l_x(self):
        assert np.allclose(self.d.l_x, [0, 1])

    def test_r_x(self):
        assert np.allclose(self.d.r_x, [2, 4])

    def test_l_u(self):
        assert np.allclose(self.d.l_u, [0, 1])

    def test_r_u(self):
        assert np.allclose(self.d.r_u, [2, 4])
