"""Optimization routines for maximizing/minimizing a scalar valued function f(x)."""
# from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from numpy.testing import assert_allclose
from numpy.typing import ArrayLike, NDArray


class Problem:
    """A class for formulating an optimization problem."""

    def __init__(
        self: object,
        f: Callable,
        gradf: Callable,
        dim: int,
        prob_type: str = "min",
        method: str = None,
    ):
        """Initialize the problem class.

        :param self: Optimization problem
        :type self: object
        :param f: Function to be optimized
        :type f: callable
        :param gradf: Gradient of the function to be optimized
        :type gradf: callable
        :param dim: Dimension of the optimization problem
        :type dim: int
        :param prob_type: Type of optimization problem ('min' or 'max')
        :type prob_type: str
        :param method: Optimization method to use ('sd' for steepest descent or 'cg' for conjugate gradient)
        :type method: str

        """

        self.f = f
        self.gradf = gradf
        self.dim = dim

        if prob_type not in ["min", "max"]:
            raise ValueError("prob_type must be either 'min' or 'max'")

        self.prob_type = prob_type
        self.method = method
        self.info = {}
        self.solution = None

    def solve(
        self: object,
        x0: NDArray = None,
        gtol: float = 1e-6,
        alpha: float = 1,
        maxiter: int = None,
    ):
        """Solve the optimization problem using the method specified in the constructor.

        :param self: Optimization problem
        :type self: object
        :param x0: Initial point
        :type x0: NDArray
        :param gtol: Tolerance for the gradient
        :type gtol: float
        :param alpha: Step size
        :type alpha: float
        :param maxiter: Maximum number of iterations
        :type maxiter: int
        :return: sol, info :  Optimal point and additional information
        """
        if x0 is None:
            x = np.random.rand(self.dim).astype(np.float64)
        else:
            x = np.asarray(x0).flatten().astype(np.float64)

        if maxiter is None:
            maxiter = self.dim * 150

        if self.method == "sd":
            return steepest_descent(self, x0=x, gtol=gtol, alpha=alpha, maxiter=maxiter)
        elif self.method == "cg":
            return conjugate_gradient(
                self, x0=x, gtol=gtol, alpha=alpha, maxiter=maxiter
            )
        else:
            raise NotImplementedError("Method not implemented")


def line_search(
    f: Callable,
    direction: NDArray,
    x: NDArray,
    alpha: float = 1.0,
    beta: float = 0.5,
    maxiter: int = 100,
):
    """Line search algorithm with Armijo condition for finding the step size
     that minimizes a function at a given point towards a given direction.

    :param f: Function to be minimized
    :type f: callable
    :param direction: Direction of search
    :type direction: NDArray
    :param x: Current point
    :type x: NDArray
    :param alpha: Initial step size
    :type alpha: float
    :param beta: Step size reduction factor
    :type beta: float
    :param maxiter: Maximum number of iterations
    :type maxiter: int
    :return: alpha, converged: Optimal step size and boolean variable indicating whether the algorithm converged

    """
    iter_count = 0
    t = alpha

    while f(x + t * direction) > (
        f(x) - beta * t * np.linalg.norm(direction) ** 2
    ):  # Armijo condition
        t = t * beta
        iter_count += 1
        if iter_count > maxiter or t < 1e-20:
            converged = False
            return t, converged
    return t, True


def steepest_descent(
    problem: object,
    x0: NDArray = None,
    gtol: float = 1e-6,
    alpha: float = 1.0,
    maxiter: int = None,
):
    """Steepest descent method for minimizing a scalar valued function f(x) of given grasdient with Armijo line search.

    :param problem: Optimization problem
    :type problem: object of class Problem
    :param x0: Initial point
    :type x0: NDArray
    :param alpha: Step size
    :type alpha: float
    :param gtol: Tolerance for stopping the algorithm when the norm of the gradient is less than tol
    :type gtol: float
    :param maxiter: Maximum number of iterations
    :type maxiter: int
    :return:
    sol, info : Optimal point and dictionary info. info["converged"] indicates whether the algorithm converged,
    info["iter_fvalues"]contains the function values at each iteration, info["iter_count"] is the number of iterations.
    the number of iterations.
    """
    if x0 is None:
        x = np.random.rand(problem.dim).astype(np.float64)
    else:
        x = np.asarray(x0).flatten().astype(np.float64)

    if maxiter is None:
        maxiter = 1000 * problem.dim
    else:
        maxiter = int(maxiter)

    iter_count = 0
    info = {}
    iter_fvalues = np.zeros(maxiter, dtype=float)
    info["iter_fvalues"] = iter_fvalues
    info["converged"] = False

    if problem.prob_type == "max":

        def func(x):
            return -problem.f(x)

        def grad(x):
            return -problem.gradf(x)

    else:
        func = problem.f
        grad = problem.gradf

    while iter_count < maxiter:
        # compute gradient and function value at current point
        gradient = grad(x)
        iter_fvalues[iter_count] = problem.f(x)

        # check if gradient is small enough to stop
        if np.linalg.norm(gradient) < gtol:
            info["converged"] = True
            break

        # set search direction as negative gradient
        direction = -gradient
        # set step size using line search
        step_size, converged = line_search(func, direction, x)
        if converged:  # if line search converged, update x
            x = x + step_size * direction
        else:  # if line search did not converge, stop
            break  # TODO: maybe try to reduce step size and continue
            info["converged"] = False
        iter_count += 1

    info["iter_count"] = iter_count
    return x, info


def conjugate_gradient(
    problem,
    x0: NDArray = None,
    gtol: float = 1e-6,
    alpha: float = 1,
    maxiter=None,
) -> Tuple[NDArray, dict]:
    """Conjugate gradient method with Fletcher-Reeves rule and Armijo line search for minimizing a  scalar valued
    function f(x).

    :param problem: Object of class Problem
    :type problem: object
    :param x0: Initial point
    :type x0: NDArray
    :param gtol: Tolerance for stopping the algorithm when the norm of the gradient is less than tol
    :type gtol: float
    :param alpha: Step size
    :type alpha: float
    :param maxiter: Maximum number of iterations
    :type maxiter: int
    """

    # Sanitize input and initialize variables
    if x0 is None:
        x = np.random.rand(problem.dim)
    else:
        x = np.asarray(x0).flatten()

    if maxiter is None:
        maxiter = problem.dim * 150

    iter_count = 0
    iter_fvalues = np.zeros(maxiter, dtype=float)
    info = {}
    info["iter_fvalues"] = iter_fvalues
    info["converged"] = False

    if problem.prob_type == "max":

        def func(x):
            return -problem.f(x)

        def grad(x):
            return -problem.gradf(x)

    else:
        func = problem.f
        grad = problem.gradf

    # initialize conjugate gradient direction
    grad0 = grad(x)
    direction = -grad0

    while iter_count < maxiter:
        # Do line search to find step size
        step_size, converged = line_search(func, direction, x)
        if converged:  # if line search converged, update x
            x = x + step_size * direction
        else:  # if line search did not converge, stop and return
            info["converged"] = False
            iter_count += 1
            break  # TODO: maybe try to reduce step size and continue

        # compute new gradient and function value, check stopping criterion
        grad1 = -grad(x)
        iter_fvalues[iter_count] = func(x)
        if np.linalg.norm(grad1) < gtol:
            info["converged"] = True
            break

        # set step size using Fletcher-Reeves rule
        beta = np.dot(grad1, grad1) / np.dot(grad0, grad0)
        # update descent direction
        direction = grad1 + beta * direction
        iter_count += 1

    info["iter_count"] = iter_count
    info["fvalues"] = iter_fvalues[:iter_count]
    return x, info


if __name__ == "__main__":
    dim = 5
    vec = np.array([3, 1, 3, 1, 3], dtype=np.float64)
    A = np.diag(vec)
    b = np.ones((dim,), dtype=np.float64)
    # Exact solution
    sol_exact = np.linalg.solve(A, -b)

    def f(x):
        return 0.5 * (x.T @ (A @ x)) + b.T @ x

    def grad(x):
        return A @ x + b

    # define optimization problem
    x0 = np.zeros((dim,))
    prob1 = Problem(f, grad, dim, prob_type="min", method="cg")
    # run conjugate gradient algorithm with default parameters
    sol1, info1 = prob1.solve(x0=x0, gtol=1e-8, maxiter=50)
    # check that the algorithm converged
    assert info1["converged"]
    # check that the solution is correct
    assert_allclose(sol1, sol_exact, atol=1e-6)

    # define optimization problem
    # Define quadratic function f(x) = 0.5 * x.T @ A @ x + b.T @ x using quad module
    import quad

    f = quad.function(A, b)
    # Define optimization problem
    prob2 = Problem(f.eval, f.grad, dim, prob_type="min", method="cg")
    # Solve optimization problem with the conjugate gradient algorithm
    x0 = np.zeros((dim,)).astype(np.float64)
    sol2, info2 = prob2.solve(x0=x0, gtol=1e-6, maxiter=50)

    # Check that the algorithm converged and that the solution is correct
    assert info2["converged"]
    assert_allclose(sol2, sol_exact, atol=1e-6, equal_nan=True)
