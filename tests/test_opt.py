import random as rd

import numpy as np
from numpy.testing import assert_allclose

from simpleopt.opt import problem


def test_scalar_steepest_descent():
    """Test steepest descent algorithm for minimizing a scalar valued function f(x)."""
    # define function and gradient
    f = lambda x: x[0] ** 2 + x[1] ** 2
    gradf = lambda x: np.array([2 * x[0], 2 * x[1]])

    # define optimization problem
    prob = problem(f, gradf, dim=2, prob_type="min")

    # run steepest descent algorithm with default parameters
    sol, info = prob.steepest_descent()

    # check that the algorithm converged
    assert info["converged"]

    # check that the solution is correct
    assert_allclose(sol, np.zeros((2,)), atol=1e-6)
