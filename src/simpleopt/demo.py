import numpy as np
from simpleopt.opt import Problem
from quad import Function
from numpy.testing import assert_allclose


# Test data for the quadratic function
dim = 5
vec = np.array([1, 2, 3, 4, 5], dtype=np.float64)
A = np.diag(vec)
b = np.ones((dim,), dtype=np.float64)

# Exact solution
sol_exact = np.linalg.solve(A, -b)

# Define function f(x) = 0.5 * x^T A x + b^T x
f = Function(A, b)

# Define optimization problem
prob2 = Problem(f.eval, f.grad, dim, prob_type="min", method="conjugate_gradient")
# Solve optimization problem with the conjugate gradient algorithm
x0 = np.zeros((dim,)).astype(np.float64)
sol2, info2 = prob2.solve(x0=x0, gtol=1e-6, maxiter=50)
num_iter = info2["iter_count"]

# Check that the algorithm converged and that the solution is correct
print("The exact solution is: ", sol_exact)
print("The solution found by the conjugate gradient algorithm is: ", sol2)
