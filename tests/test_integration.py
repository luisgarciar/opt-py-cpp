import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as nps
from numpy.testing import assert_allclose
from quad import function
from simpleopt.opt import Problem


def test_integration_normal_input():
    """Test of opt.solve and quad.function with normal inputs"""
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
    b = np.array([1, 2, 3], dtype=np.float64)
    I = np.eye(3, dtype=np.float64)

    # Define function and exact solution
    # f(x) = 0.5*(x^T(A^T A + I)x) - b^T x

    f = function(A.T @ A + I, -b)
    exact_sol = np.linalg.solve(A.T @ A + I, b)

    # Define optimization problem
    prob = Problem(f.eval, f.grad, dim=3, prob_type="min", method="steepest_descent")

    # Solve optimization problem with conjugate gradient method
    sol, info = prob.solve(maxiter=20, gtol=1e-6)

    assert_allclose(sol, exact_sol, atol=1e-6, equal_nan=True)


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
