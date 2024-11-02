# Copyright (c) 2024 Yilin Zou
from typing import Iterable, Optional, Any

import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from scipy.sparse import coo_array

from ._common import _preprocess, _postprocess
from pockit.base.systembase import SystemBase
from pockit.base.variablebase import VariableBase


def _reflection(func, row, col, n):
    diag_i = []
    diag_rc = []
    for i, (r, c) in enumerate(zip(row, col)):
        if r == c:
            diag_i.append(i)
            diag_rc.append(r)
    diag_i = np.array(diag_i, dtype=np.int32)
    diag_rc = np.array(diag_rc, dtype=np.int32)

    def full_csr_matrix(*args):
        data = func(*args)
        coo_half = coo_array((data, (row, col)), shape=(n, n))
        coo_diag = coo_array((data[diag_i], (diag_rc, diag_rc)), shape=(n, n))
        return coo_half + coo_half.T - coo_diag

    return full_csr_matrix


def solve(
    system: SystemBase,
    guess: VariableBase | list[VariableBase | Iterable[float]],
    optimizer_options: Optional[dict] = None,
) -> tuple[VariableBase | list[VariableBase | Iterable[float]], Any]:
    """Solve the system using trust-constr method of
    :func:`scipy.optimize.minimize`.

    If the system has only one phase and no static variables, ``guess`` can
    be a single ``Variable`` object. Otherwise, ``guess`` should be a list of
    ``Variable`` objects, one for each ``Phase``, followed by an array
    as values of static variables.

    Optimizer options should be a dictionary of options to pass to :func:`scipy.optimize.minimize`.
    See [Scipy documentation](https://docs.scipy.org)
    for available options. Options will be passed verbatimly.

    Args:
        system: ``System`` to solve.
        guess: Guess to the solution.
        optimizer_options: Options to pass to :func:`scipy.optimize.minimize`.

    Returns:
        The value returned by :func:`scipy.optimize.minimize` parsed as the same format as ``guess``
        (a single ``Variable`` object or a list of ``Variable`` objects and a array for static values),
        and the raw output returned by :func:`scipy.optimize.minimize`.
    """
    x_0, guess_is_variable, optimizer_options = _preprocess(
        system, guess, optimizer_options
    )

    num_cons = len(system.c_lb)
    objective_hessian = _reflection(
        system.hessian_o, *system.hessianstructure_o(), system.L
    )
    constraints_jacobian = lambda x: coo_array(
        (system.jacobian(x), system.jacobianstructure()), shape=(num_cons, system.L)
    )
    constraints_hessian = _reflection(
        system.hessian_c, *system.hessianstructure_c(), system.L
    )

    bounds = Bounds(system.v_lb, system.v_ub)
    constraints = NonlinearConstraint(
        system.constraints,
        system.c_lb,
        system.c_ub,
        jac=constraints_jacobian,
        hess=constraints_hessian,
    )

    res = minimize(
        system.objective,
        x_0,
        method="trust-constr",
        jac=system.gradient,
        hess=objective_hessian,
        constraints=constraints,
        bounds=bounds,
        options=optimizer_options,
    )

    if guess_is_variable:
        Variable = type(guess)
    else:
        Variable = type(guess[0])

    result = _postprocess(Variable, system, res.x, guess_is_variable)
    return result, res
