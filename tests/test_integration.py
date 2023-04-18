import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as nps
from numpy.testing import assert_allclose
from quad import function


@given(ndim=st.integers(min_value=2, max_value=4), data=st.data())
def test_integration_random_input(ndim, data):
    """Test of  opt.solve and quad.function with random inputs"""
    strategy1 = nps.arrays(
        dtype=np.float64,
        shape=(ndim, ndim),
    )
    strategy2 = nps.arrays(dtype=np.float64, shape=(ndim,))

    A = data.draw(strategy1)
    b = data.draw(strategy2)
    x = data.draw(strategy2)
    f = function(A, b)
    q = 0.5 * (x.T @ (A @ x)) + b.T @ x
    assert_allclose(f.eval(x), q, atol=1e-6)
    assert_allclose(f.grad(x), A @ x + b, atol=1e-6, equal_nan=True)
