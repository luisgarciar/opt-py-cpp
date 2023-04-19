import numpy as np
from numpy.testing import assert_allclose
from simpleopt.opt import Problem


def test_steepest_descent():
    """Test steepest descent algorithm for minimizing a scalar valued function f(x)."""
    dim = 5
    vec = np.array([1, 2, 3, 4, 5])
    A = np.diag(vec)
    b = np.ones((dim,))
    # Exact solution
    sol_exact = np.linalg.solve(A, -b)

    def f(x):
        return 0.5 * (x.T @ (A @ x)) + b.T @ x

    def grad(x):
        return A @ x + b

    # define optimization problem
    x0 = np.ones((dim,))
    prob = Problem(f, grad, dim, prob_type="min", method="steepest_descent")
    # run conjugate gradient algorithm with default parameters
    sol, info = prob.solve(x0=x0, gtol=1e-8, maxiter=1000)
    # check that the algorithm converged
    # assert info["converged"]
    # check that the solution is correct
    assert_allclose(sol, sol_exact, atol=1e-6)


def test_conjugate_gradient():
    """Test conjugate gradient algorithm for minimizing a scalar valued function f(x)."""

    dim = 5
    vec = np.array([1, 2, 3, 4, 5])
    A = np.diag(vec)
    b = np.ones((dim,))
    # Exact solution
    sol_exact = np.linalg.solve(A, -b)

    def f(x):
        return 0.5 * (x.T @ (A @ x)) + b.T @ x

    def grad(x):
        return A @ x + b

    # define optimization problem
    x0 = np.zeros((dim,))
    prob = Problem(f, grad, dim, prob_type="min", method="conjugate_gradient")
    # run conjugate gradient algorithm
    sol, info = prob.solve(x0=x0, gtol=1e-8, maxiter=1000)
    # check that the algorithm converged
    assert info["converged"]
    # check that the solution is correct
    assert_allclose(sol, sol_exact, atol=1e-6)
