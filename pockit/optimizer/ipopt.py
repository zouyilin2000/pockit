from typing import Any, Iterable, Optional

import cyipopt

from ._common import _preprocess, _postprocess
from pockit.base.systembase import SystemBase
from pockit.base.variablebase import VariableBase


def solve(
    system: SystemBase,
    guess: VariableBase | list[VariableBase | Iterable[float]],
    optimizer_options: Optional[dict] = None,
) -> tuple[VariableBase | list[VariableBase | Iterable[float]], Any]:
    """Solve the system using [IPOPT](https://github.com/coin-or/Ipopt).

    If the system has only one phase and no static variables, ``guess`` can
    be a single ``Variable`` object. Otherwise, ``guess`` should be a list of
    ``Variable`` objects, one for each ``Phase``, followed by an array
    as values of static variables.

    Optimizer options should be a dictionary of options to pass to Ipopt.
    See [Ipopt documentation](https://coin-or.github.io/Ipopt/OPTIONS.html) for available options. 
    Options will be passed verbatimly.

    Args:
        system: ``System`` to solve.
        guess: Guess to the solution.
        optimizer_options: Options to pass to IPOPT.

    Returns:
        The value returned by IPOPT parsed as the same format as ``guess`` 
        (a single ``Variable`` object or a list of ``Variable`` objects and a array for static values),
        and the raw output returned by IPOPT.
    """
    x_0, guess_is_variable, optimizer_options = _preprocess(
        system, guess, optimizer_options
    )

    solver = cyipopt.Problem(
        n=int(system.L),
        m=len(system.c_lb),
        problem_obj=system,
        lb=system.v_lb,
        ub=system.v_ub,
        cl=system.c_lb,
        cu=system.c_ub,
    )
    for k, v in optimizer_options.items():
        solver.add_option(k, v)

    x, info = solver.solve(x_0)

    if guess_is_variable:
        Variable = type(guess)
    else:
        Variable = type(guess[0])

    result = _postprocess(Variable, system, x, guess_is_variable)
    return result, info
