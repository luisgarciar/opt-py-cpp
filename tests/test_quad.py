import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as nps
from numpy.testing import assert_allclose
from quad import Function


def test_quad_function():
    """Test of quadfunction defined in src-cpp/quad.cpp with bindings in
    src-cpp/bind/bindings.cpp"""

    # define function with numpy arrays
    A = np.asarray([[1, 2], [3, 4]], dtype=np.float64)
    x = np.asarray([1, 2], dtype=np.float64)
    b = A @ x
    # f(x) = 0.5*(x^T A x) + b^T x
    f = Function(A, b)

    # check that the function value is correct
    assert_allclose(f.eval(x), 40.5, atol=1e-6)

    # check that the gradient is correct
    print(f.grad(x))
    assert_allclose(f.grad(x), np.asarray([10, 22]), atol=1e-6)


def test_quad_function_invalid_input():
    """Test of quad.function with inputs of invalid dimensions"""
    A = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    b = np.asarray([1, 2], dtype=np.float64)

    with pytest.raises(ValueError, match="Input matrix must be a square matrix"):
        Function(A, b)

    A = np.asarray([[1, 2], [3, 4]], dtype=np.float64)
    b = np.asarray([1, 2, 3], dtype=np.float64)

    with pytest.raises(
        ValueError, match="Input vector must be compatible with the matrix"
    ):
        Function(A, b)

    A = np.asarray([[1, 2], [3, 4]], dtype=np.float64)
    b = np.asarray([1, 2], dtype=np.float64)
    x = np.asarray([1, 2, 3], dtype=np.float64)

    with pytest.raises(
        ValueError,
        match="Input vector must be compatible with the quadratic function",
    ):
        Function(A, b).eval(x)

    with pytest.raises(
        ValueError,
        match="Input vector must be compatible with the quadratic function",
    ):
        Function(A, b).grad(x)


@given(ndim=st.integers(min_value=2, max_value=4), data=st.data())
def test_random(ndim, data):
    """Test of quad.function with random inputs"""
    strategy1 = nps.arrays(
        dtype=np.float64,
        shape=(ndim, ndim),
    )
    strategy2 = nps.arrays(dtype=np.float64, shape=(ndim,))

    A = data.draw(strategy1)
    b = data.draw(strategy2)
    x = data.draw(strategy2)
    f = Function(A, b)
    q = 0.5 * (x.T @ (A @ x)) + b.T @ x
    assert_allclose(f.eval(x), q, atol=1e-6)
    assert_allclose(f.grad(x), A @ x + b, atol=1e-6, equal_nan=True)
