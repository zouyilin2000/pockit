# pockit: Python Optimal Control KIT

[![PyPI](https://img.shields.io/pypi/v/pockit-optimal-control.svg)](https://pypi.org/project/pockit-optimal-control/)
[![Python](https://img.shields.io/pypi/pyversions/pockit-optimal-control.svg)](https://pypi.org/project/pockit-optimal-control/)
[![CI](https://github.com/zouyilin2000/pockit/actions/workflows/ci.yml/badge.svg)](https://github.com/zouyilin2000/pockit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/zouyilin2000/pockit/blob/main/LICENSE)

Pockit is a SymPy-native toolkit for research in numerical optimal control. It
is designed for workflows where models evolve: change dynamics, objectives,
constraints, free parameters, or discretization settings, then solve again and
compare results without manually maintaining derivative code. Pockit is
distributed under the permissive MIT License.

Pockit 是一个面向数值最优控制研究的 SymPy 原生工具包，适合模型持续变化的工作流：
调整动力学、目标函数、约束、自由参数或离散设置后即可重新求解并比较结果，无需手工
维护导数代码。Pockit 使用宽松的 MIT 许可证发布。

## Overview / 概览

- **SymPy-native modeling:** Write dynamics, objectives, and constraints as
  symbolic expressions that remain readable and easy to modify.
- **Research-oriented iteration:** Vary equations, parameters, phases, and mesh
  settings without rewriting low-level callbacks or derivative routines.
- **Flexible problem structure:** Build multiphase systems with path, algebraic,
  boundary, and integral constraints.
- **Open and extensible:** The MIT License permits use, modification, and
  redistribution in academic, personal, and commercial work.
- **Numerical pipeline:** Use Legendre-Gauss-Lobatto (LGL) or
  Legendre-Gauss-Radau (LGR) pseudospectral methods, symbolic differentiation,
  Numba-compiled vectorized evaluation, and Ipopt or SciPy solver backends.

- **SymPy 原生建模：** 使用清晰、易修改的符号表达式编写动力学、目标函数和约束。
- **面向研究迭代：** 调整方程、参数、阶段和网格设置时，无需重写底层回调或导数程序。
- **灵活的问题结构：** 支持包含路径、代数、边界和积分约束的多阶段系统。
- **开放且便于扩展：** MIT 许可证允许在学术、个人和商业项目中使用、修改和再发布。
- **数值计算流程：** 支持 Legendre-Gauss-Lobatto（LGL）和
  Legendre-Gauss-Radau（LGR）伪谱法、符号微分、Numba 编译的向量化计算，以及
  Ipopt 和 SciPy 求解器后端。

## Installation / 安装

Pockit requires Python 3.11 or later. The core package is available from PyPI;
the `ipopt` and `examples` extras add the Ipopt backend and plotting dependencies
used by the examples.

`cyipopt` may require a native Ipopt installation when a compatible wheel is not
available. See the
[`cyipopt` installation guide](https://cyipopt.readthedocs.io/stable/install.html)
for platform-specific instructions.

Pockit 需要 Python 3.11 或更高版本。基础包可从 PyPI 安装；`ipopt` 和 `examples`
可选依赖分别提供 Ipopt 后端和示例所需的绘图依赖。

如果当前平台没有兼容的 `cyipopt` wheel，则可能需要先安装原生 Ipopt 库。具体步骤请参考
[`cyipopt` 安装指南](https://cyipopt.readthedocs.io/stable/install.html)。

| Purpose / 用途 | Command / 命令 |
| --- | --- |
| Core package / 基础包 | `pip install pockit-optimal-control` |
| Ipopt and examples / Ipopt 与示例依赖 | `pip install "pockit-optimal-control[ipopt,examples]"` |

## Examples / 示例

The repository includes **33 maintained examples across 23 application areas**.
They range from classical benchmarks to multiphase aerospace, robotics,
chemistry, energy, epidemiology, neuroscience, quantum control, finance, and
scientific machine learning. Each standalone program follows a consistent
build, solve, and plot structure, making it practical to start from a nearby
problem and change its equations or parameters for new research.

Browse the
[complete example catalog](https://github.com/zouyilin2000/pockit/blob/main/examples/README.md),
or run any script with `--no-show`, `--save [PATH]`, and, where available,
`--quick`.

仓库包含覆盖 **23 个应用领域的 33 个维护示例**，既有经典数值算例，也包括多阶段航天、
机器人、化学、能源、流行病学、神经科学、量子控制、金融和科学机器学习等问题。每个独立
程序均采用一致的建模、求解和绘图结构，便于从相近问题出发，修改方程或参数开展新的研究。

可查看
[完整示例目录](https://github.com/zouyilin2000/pockit/blob/main/examples/README.md)，
也可使用 `--no-show`、`--save [PATH]` 以及部分示例提供的 `--quick` 参数运行脚本。

## LQR in 1 Minute / 一分钟 LQR

This minimal example shows how symbolic expressions map directly to an
optimal-control model, solution, and plot.

这个最小示例展示符号表达式如何直接对应到最优控制模型、求解过程和结果图。

```python
import matplotlib.pyplot as plt

from pockit.lobatto import System, constant_guess
from pockit.optimizer import ipopt

# min integral_0^1 (q * x^2 + r * u^2) dt + s * x_f^2 / 2
# subject to x' = a * x + b * u and x(0) = 1
a, b, s, q, r = -1, 1, 1, 1, 0.1

system = System(["x_f"])
x_f, = system.s
phase = system.new_phase(["x"], ["u"])
x, = phase.x
u, = phase.u

phase.set_dynamics([a * x + b * u])
phase.set_integral([q * x**2 + r * u**2])
phase.set_boundary_condition([1], [x_f], 0, 1)
phase.set_discretization(10, 10)

system.set_phase([phase])
system.set_objective(phase.I[0] + s * x_f**2 / 2)

guess_p = constant_guess(phase, 0)
guess_s = [0.0]
[var_p, var_s], info = ipopt.solve(system, [guess_p, guess_s])

print("status:", info["status_msg"].decode())
print("objective:", info["obj_val"])

plt.plot(var_p.t_x, var_p.x[0], label="x")
plt.plot(var_p.t_u, var_p.u[0], label="u")
plt.legend()
plt.minorticks_on()
plt.grid(linestyle="--")
plt.show()
```

![LQR solution / LQR 求解结果](https://raw.githubusercontent.com/zouyilin2000/pockit/main/images/lqr_readme.png)

Change the coefficients, symbolic equations, constraints, or mesh settings to
turn this example into a new experiment.

可以继续修改系数、符号方程、约束或网格设置，将这个示例扩展为新的实验。

## Real-time C++: naive-planner / 实时 C++：naive-planner

Pockit prioritizes model exploration and repeated research iterations. When a
model needs a compact C++20 runtime for real-time or performance-sensitive use,
see [naive-planner](https://github.com/zouyilin2000/naive_planner).

naive-planner also uses symbolic model definitions and generates sparse
analytic derivatives and C++ evaluation code. Because both projects follow a
symbolic modeling approach, equations remain recognizable and a stabilized
pockit research model is typically straightforward to migrate. The projects
serve different execution needs: pockit preserves modeling flexibility while
naive-planner focuses on real-time performance.

Pockit 优先服务于模型探索和反复研究迭代。当模型需要用于实时或性能敏感场景的精简
C++20 运行时时，可以使用
[naive-planner](https://github.com/zouyilin2000/naive_planner)。

naive-planner 同样使用符号化模型定义，并生成稀疏解析导数和 C++ 求值代码。两个项目
采用相近的符号建模思路，因此方程形式易于对应，已稳定的 pockit 研究模型通常可以方便地
迁移。二者分别服务于不同的执行需求：pockit 保留研究阶段的建模灵活性，naive-planner
专注于实时性能。

## Documentation and Support / 文档与支持

Detailed guides are available in the
[documentation](https://pockit.pages.dev), with generated interfaces in the
[API reference](https://pockit-api.pages.dev/) and runnable programs in
[`examples`](https://github.com/zouyilin2000/pockit/tree/main/examples).

Questions and bug reports are welcome in the
[GitHub issue tracker](https://github.com/zouyilin2000/pockit/issues). Both
Chinese and English are welcome.

You can support the project by starring it on GitHub or through
[GitHub Sponsors](https://github.com/sponsors/zouyilin2000).

使用指南请参阅 [pockit 文档](https://pockit.pages.dev)，接口定义请参阅
[API 文档](https://pockit-api.pages.dev/)，完整程序可在
[`examples`](https://github.com/zouyilin2000/pockit/tree/main/examples)
目录中查看。

问题和错误报告请提交到 [GitHub Issues](https://github.com/zouyilin2000/pockit/issues)，
中文和英文均可。

可以通过在 GitHub 上为项目点 Star，或通过
[GitHub Sponsors](https://github.com/sponsors/zouyilin2000) 支持项目维护。

## License / 许可证

Pockit is distributed under the permissive
[MIT License](https://github.com/zouyilin2000/pockit/blob/main/LICENSE), allowing
use, modification, and redistribution in research and commercial projects.

Pockit 使用宽松的
[MIT 许可证](https://github.com/zouyilin2000/pockit/blob/main/LICENSE) 发布，
允许在科研和商业项目中使用、修改和再发布。
