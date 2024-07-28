from typing import Iterable, Optional, Type

from pockit.base.systembase import SystemBase
from pockit.base.variablebase import VariableBase
from pockit.base.vectypes import *


def _preprocess(
    system: SystemBase,
    guess: VariableBase | list[VariableBase | Iterable[float]],
    optimizer_options: Optional[dict] = None,
) -> tuple[VecFloat, bool, dict]:
    if not system.ok:
        raise ValueError("system is not fully configured")
    if optimizer_options is None:
        optimizer_options = {}

    guess_is_variable = isinstance(guess, VariableBase)
    if guess_is_variable:
        guess = [guess]

    if not system.n_s and len(guess) != system.n_p:
        raise ValueError("len(guess) must be equal to the number of phases")
    elif system.n_s and len(guess) != system.n_p + 1:
        raise ValueError(
            "len(guess) must be equal to the number of phases + 1 (for static variables)"
        )

    x_0 = np.zeros(system.L)
    for i in range(system.n_p):
        x_0[system.l_p[i] : system.r_p[i]] = guess[i].data
    if system.n_s > 0:
        x_0[system.l_s : system.r_s] = np.array(list(guess[-1]), dtype=np.float64)

    return x_0, guess_is_variable, optimizer_options


def _postprocess(
    Variable: Type[VariableBase],
    system: SystemBase,
    x: VecFloat,
    guess_is_variable: bool,
) -> VariableBase | list[VariableBase | Iterable[float]]:
    result = []
    s = x[system.l_s : system.r_s]
    for i in range(system.n_p):
        p = system.p[i]
        x_ = x[system.l_p[i] : system.r_p[i]]
        for j in range(p.n_x):
            x_[p.l_v[j]] = p._value_boundary_condition(p.info_bc_0[j], x_[p.l_v[j]], s)
            x_[p.r_v[j] - 1] = p._value_boundary_condition(
                p.info_bc_f[j], x_[p.r_v[j] - 1], s
            )
        x_[-2] = p._value_boundary_condition(p.info_t_0, x_[-2], s)
        x_[-1] = p._value_boundary_condition(p.info_t_f, x_[-1], s)
        result.append(Variable(system.p[i], x[system.l_p[i] : system.r_p[i]]))
    if system.n_s > 0:
        result.append(s)

    if guess_is_variable:
        return result[0]
    return result
