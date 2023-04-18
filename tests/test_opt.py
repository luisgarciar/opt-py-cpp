import numpy as np
from numpy.testing import assert_allclose

from simpleopt.opt import Problem


def test_steepest_descent():
    """Test steepest descent algorithm for minimizing a scalar valued function f(x)."""

    # define function and gradient
    def f(x):
        return x[0] ** 2 + x[1] ** 2

    def grad(x):
        return np.array([2 * x[0], 2 * x[1]])

    # define optimization problem
    prob = Problem(f, grad, dim=2, prob_type="min", method="steepest_descent")

    # run the steepest descent algorithm with default parameters
    sol, info = prob.solve()

    # check that the algorithm converged
    assert info["converged"]

    # check that the solution is correct
    assert_allclose(sol, np.zeros((2,)), atol=1e-6)


def test_conjugate_gradient():
    """Test conjugate gradient algorithm for minimizing a scalar valued function f(x)."""

    # define function and gradient
    def f(x):
        return x[0] ** 2 + x[1] ** 2

    def grad(x):
        return np.array([2 * x[0], 2 * x[1]])

    # define optimization problem
    prob = Problem(f, grad, dim=2, prob_type="min", method="conjugate_gradient")

    # run conjugate gradient algorithm with default parameters
    sol, info = prob.solve()

    # check that the algorithm converged
    assert info["converged"]

    # check that the solution is correct
    assert_allclose(sol, np.zeros((2,)), atol=1e-6)
