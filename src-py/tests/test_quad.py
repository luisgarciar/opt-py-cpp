import random as rd

import numpy as np
from numpy.testing import assert_allclose

from quad import function


def test_quad_function():
    """Test quad function from src-cpp/quad.cpp with bindings in
    src-cpp/bind/bindings.cpp"""

    # define function with numpy arrays
    A = np.asarray([[1, 2], [3, 4]])
    x = np.asarray([1, 2])
    b = A @ x
    # f(x) = 0.5*(x^T A x) + b^T x
    f = function(A, b)

    # check that the function value is correct
    assert_allclose(f(x), 40.5, atol=1e-6)

    # check that the gradient is correct
    assert_allclose(f.grad(x), np.asarray([5, 10]), atol=1e-6)