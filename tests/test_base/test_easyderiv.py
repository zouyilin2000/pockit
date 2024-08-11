from pockit.base.easyderiv import *


def test_easyderiv():
    v1 = Node()
    v2 = Node()
    v3 = Node()
    v1.l = 2
    v1.set_G_i([np.array([0, 1], dtype=np.int32)])
    v1.set_G([np.ones_like(v1.G_i[0], dtype=np.float64)])
    v2.l = 2
    v2.set_G_i([np.array([2, 3], dtype=np.int32)])
    v2.set_G([np.ones_like(v2.G_i[0], dtype=np.float64)])
    v3.set_G_i([np.array([4], dtype=np.int32)])
    v3.set_G([np.ones_like(v3.G_i[0], dtype=np.float64)])

    base_nodes = [v1, v2, v3]

    I1 = Node()
    I2 = Node()
    F = Node()

    I1.l = 2
    I1.args = [v1, v2]
    I1.g_i = np.array([0, 1], dtype=np.int32)
    I2.l = 2
    I2.args = [v2, v3]
    I2.g_i = np.array([0, 1], dtype=np.int32)
    I2.h_i_row = np.array([1], dtype=np.int32)
    I2.h_i_col = np.array([0], dtype=np.int32)
    F.l = 2
    F.args = [I1, I2]
    F.g_i = np.array([0, 1], dtype=np.int32)
    F.h_i_row = np.array([0, 1], dtype=np.int32)
    F.h_i_col = np.array([0, 0], dtype=np.int32)

    dependent_nodes = [I1, I2, F]

    v1.V = np.array([1, 2], dtype=np.float64)
    v2.V = np.array([3, 4], dtype=np.float64)
    v3.V = np.array([5], dtype=np.float64)
    I1.V = v1.V + v2.V
    I2.V = v2.V * v3.V
    F.V = I1.V**2 * I2.V

    I1.g = np.array(
        [np.array([1, 1], dtype=np.float64), np.array([1, 1], dtype=np.float64)]
    )
    I2.g = np.array([np.full_like(I2.V, v3.V[0], dtype=np.float64), v2.V])
    F.g = np.array(
        [
            2 * I1.V * I2.V,
            I1.V**2,
        ]
    )

    forward_gradient_i(base_nodes + dependent_nodes)
    forward_gradient_v(base_nodes + dependent_nodes)

    I2.h = np.array([np.full(I2.l, 1, dtype=np.float64)])
    F.h = np.array([2 * I2.V, 2 * I1.V])

    forward_hessian_phase_i(base_nodes + dependent_nodes)
    forward_hessian_phase_v(base_nodes + dependent_nodes)

    x = np.arange(5, dtype=np.float64) + 1

    def f(x):
        return np.array(
            [(x[0] + x[2]) ** 2 * x[2] * x[4], (x[1] + x[3]) ** 2 * x[3] * x[4]]
        )

    # finite difference
    G_fd = np.zeros((2, 5), dtype=np.float64)
    eps = 1e-6
    for i in range(5):
        x[i] += eps
        fp = f(x)
        x[i] -= 2 * eps
        fm = f(x)
        x[i] += eps
        G_fd[:, i] = (fp - fm) / (2 * eps)

    G_graph = np.zeros((2, 5), dtype=np.float64)
    for i, v in zip(F.G_i, F.G):
        for r_, c_, v_ in zip(np.arange(2), i, v):
            G_graph[r_, c_] += v_

    assert np.allclose(G_fd, G_graph)

    H_fd_1 = np.zeros((5, 5), dtype=np.float64)
    H_fd_2 = np.zeros((5, 5), dtype=np.float64)
    eps = 5e-4
    for i in range(5):
        for j in range(i + 1):
            x[i] += eps
            x[j] += eps
            fpp = f(x)
            x[j] -= 2 * eps
            fpm = f(x)
            x[i] -= 2 * eps
            fmm = f(x)
            x[j] += 2 * eps
            fmp = f(x)
            x[i] += eps
            x[j] -= eps
            H_fd_1[i, j] = (fpp[0] - fpm[0] - fmp[0] + fmm[0]) / (4 * eps**2)
            H_fd_2[i, j] = (fpp[1] - fpm[1] - fmp[1] + fmm[1]) / (4 * eps**2)

    H_graph_1 = np.zeros((5, 5), dtype=np.float64)
    H_graph_2 = np.zeros((5, 5), dtype=np.float64)
    for r_, c_, v in zip(F.H_i_row, F.H_i_col, F.H):
        H_graph_1[r_[0], c_[0]] += v[0]
        H_graph_2[r_[1], c_[1]] += v[1]

    assert np.allclose(H_fd_1, H_graph_1)
    assert np.allclose(H_fd_2, H_graph_2)
