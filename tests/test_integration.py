import numpy as np
import pytest
import quad
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as nps
from numpy.testing import assert_allclose
from simpleopt import opt


def test_integration_steepest_descent():
    """Test of opt.solve and quad.function with normal inputs"""
    dim = 5
    vec = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    A = np.diag(vec)
    b = np.ones((dim,), dtype=np.float64)
    # Exact solution
    exact_sol = np.linalg.solve(A, -b)

    # Define quadratic function f(x) = 0.5 * x.T @ A @ x + b.T @ x using quad module
    f = quad.function(A, b)

    # Define optimization problem
    prob = opt.Problem(f.eval, f.grad, dim, prob_type="min", method="steepest_descent")

    # Solve optimization problem with the steepest descent method
    x0 = np.zeros((dim,)).astype(np.float64)
    sol, info = prob.solve(x0=x0, gtol=1e-8, maxiter=1000)

    # Check that the algorithm converged and that the solution is correct
    assert info["converged"]
    assert_allclose(sol, exact_sol, atol=1e-6, equal_nan=True)


def test_integration_conj_grad():
    """Test of opt.solve and quad.function with normal inputs"""
    dim = 5
    vec = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    A = np.diag(vec)
    b = np.ones((dim,), dtype=np.float64)
    # Exact solution
    exact_sol = np.linalg.solve(A, -b)

    # Define quadratic function f(x) = 0.5 * x.T @ A @ x + b.T @ x using quad module
    f = quad.function(A, b)
    # Define optimization problem
    prob = opt.Problem(f.eval, f.grad, dim, prob_type="min", method="conjugate_gradient")
    # Solve optimization problem with the conjugate gradient algorithm
    x0 = np.zeros((dim,)).astype(np.float64)
    sol2, info2 = prob.solve(x0=x0, gtol=1e-6, maxiter=50)

    # Check that the algorithm converged and that the solution is correct
    assert info2["converged"]
    assert_allclose(sol2, exact_sol, atol=1e-6, equal_nan=True)


# @given(ndim=st.integers(min_value=5, max_value=6), data=st.data())
# def test_integration_random_input(ndim, data):
#     """Test of  opt.solve and quad.function with random inputs
#     of random dimension ndim"""
#     # Define random strategy for A and b
#     strategy1 = nps.arrays(
#         dtype=np.float64,
#         shape=(ndim, ndim),
#     )
#     strategy2 = nps.arrays(dtype=np.float64, shape=(ndim,))
#     A = data.draw(strategy1)
#     b = data.draw(strategy2)
#     I = np.eye(ndim)
#
#     f = function(A.T @ A + I, b)
#     exact_sol = np.linalg.solve((A.T @ A + I), -b)
#
#     # Define optimization problem
#     prob = Problem(
#         f.eval, f.grad, dim=ndim, prob_type="min", method="conjugate_gradient"
#     )
#
#     # Solve optimization problem with conjugate gradient method
#     sol, info = prob.solve(maxiter=20, gtol=1e-6)

#     assert_allclose(sol, exact_sol, atol=1e-4, equal_nan=True) or not info["converged"]
