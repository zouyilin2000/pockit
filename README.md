# pockit: Python Optimal Control KIT

**😃 Welcome to pockit's project page**

## Introduction

**Pockit** is a Python package for solving optimal control problems numerically. It combines advanced techniques to deliver a powerful, user-friendly, and fast solution.

- 💪 **Powerful:** Pockit is designed to solve multi-phase optimal control problems with support for path, algebraic, and boundary condition constraints.
- 🔢 **User-friendly:** Pockit features a [SymPy](https://www.sympy.org/)-based interface that makes defining and solving problems intuitive.
- ⚡ **Fast:** Pockit achieves high performance through symbolic differentiation (with [SymPy](https://www.sympy.org/)), Just-In-Time compilation (with [Numba](https://numba.pydata.org/)), vectorization, and other advanced techniques.

For more information, visit the [documentation](https://pockit.pages.dev) and [API reference](https://pockit-api.pages.dev/).

总之，pockit 利用伪谱法数值求解最优控制问题，挺好用的。请访问 [pockit 文档](https://pockit.pages.dev) 和 [API 文档](https://pockit-api.pages.dev) 获取详细信息。

## Issues and Bug Reports

For any issues, questions, or bug reports, please open a GitHub issue at the [project's issue tracker](https://github.com/zouyilin2000/pockit/issues). 

如果有任何问题，最好直接在 GitHub Issues 上讨论，这样讨论比较集中而且大家都能看到。中文、英文都可以，现在自动翻译已经比较好用，可以自行翻译。

## Support the Project

If you find pockit helpful, please consider starring the project on GitHub. Thank you!

To support the project's development financially, you can make a contribution via GitHub Sponsors. Your support is greatly appreciated!

如果有帮助，请帮忙点下 Star，谢谢 🙏。如果你很有钱，可以考虑赞助项目 😂。

## Citation

If you use pockit in your research, please cite the following works:

```bibtex
@misc{zou2025vectorizedsparsesecondorderforward,
    title={Vectorized Sparse Second-Order Forward Automatic Differentiation for Optimal Control Direct Methods}, 
    author={Yilin Zou and Fanghua Jiang},
    year={2025},
    eprint={2506.11537},
    archivePrefix={arXiv},
    primaryClass={eess.SY},
    url={https://arxiv.org/abs/2506.11537}, 
}

@misc{zou2025reexamininglegendregausslobattopseudospectralmethods,
      title={Re-examining the Legendre-Gauss-Lobatto Pseudospectral Methods for Optimal Control}, 
      author={Yilin Zou and Fanghua Jiang},
      year={2025},
      eprint={2507.01660},
      archivePrefix={arXiv},
      primaryClass={eess.SY},
      url={https://arxiv.org/abs/2507.01660}, 
}
```

We sincerely appreciate your support.

## LQR in 1 Minute
```python
from pockit.lobatto import System, constant_guess
from pockit.optimizer import ipopt
import matplotlib.pyplot as plt

# LQR problem:
# min ∫_0^1 (q * x^2 + r * u^2) dt + s * x_f^2 / 2
# s.t. x' = a * x + b * u, x(0) = 1

# Set parameters
a, b, s, q, r = -1, 1, 1, 1, 0.1

# Set up the system
system = System(["x_f"])  # the system has one free parameter x_f
x_f, = system.s  # extract the Sympy symbol x_f
phase = system.new_phase(["x"], ["u"])  # the phase has one state x and one control u
x, = phase.x  # extract the Sympy symbol x
u, = phase.u  # extract the Sympy symbol u
phase.set_dynamics([a * x + b * u])  # x' = a * x + b * u
phase.set_integral([q * x ** 2 + r * u ** 2])  # I = ∫ q * x^2 + r * u^2 dt
phase.set_boundary_condition([1], [x_f], 0, 1)  # x(0) = 1, x(t_f) = x_f, t_0 = 0, t_f = 1
phase.set_discretization(10, 10)  # 10 subintervals with 10 collocation points in each subinterval
system.set_phase([phase])  # bind the phase to the system
system.set_objective(phase.I[0] + s * x_f ** 2 / 2)  # define the objective function

# Solve the problem
guess_p = constant_guess(phase, 0)  # initial guess for the phase
guess_s = [0.]  # initial guess for the free parameter x_f
# var_p, var_s: the optimal solution for the phase and the free parameter
[var_p, var_s], info = ipopt.solve(system, [guess_p, guess_s])

# Print the results
print("status:", info["status_msg"].decode())
print("objective:", info["obj_val"])  # 0.2319139744522318

# Plot the results
plt.plot(var_p.t_x, var_p.x[0], label="x")  # only one state variable and one control variable, 
plt.plot(var_p.t_u, var_p.u[0], label="u")  # so the indices are 0 for both
plt.legend()
plt.minorticks_on()
plt.grid(linestyle='--')
plt.show()
```
![Result of the LQR Problem](images/lqr_readme.png)
