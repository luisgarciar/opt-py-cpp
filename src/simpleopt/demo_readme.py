import simpleopt
import numpy as np
import quad

A = np.array([[6.0, 2.0], [2.0, 6.0]], dtype=np.float64)
b = np.array([1, 1], dtype=np.float64)
f = quad.Function(A, b)
x0 = np.array([1, 1], dtype=np.float64)
max_iter = 100

problem = simpleopt.opt.Problem(f.eval, f.grad, dim=2, prob_type="min", method='sd')
sol, info = problem.solve(x0, max_iter=max_iter)
