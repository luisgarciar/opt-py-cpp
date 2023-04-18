"""Optimization routines for maximizing/minimizing a scalar valued function f(x)."""

from typing import Callable, Tuple

import numpy as np
from numpy.testing import assert_allclose
from numpy.typing import ArrayLike, NDArray


class Problem:
    """A class for formulating an optimization problem.
    Sets up an optimization problem by defining the type of optimization problem (minimization or maximization),
    objective function, gradient, initial point, tolerance, maximum number of iterations, step size, and
    choice of optimization method.
    """

    def __init__(
        self: object,
        f: Callable,
        gradf: Callable,
        dim: int,
        prob_type: str = "min",
        method: str = None,
    ) -> object:
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
        :param method: Optimization method to use ('steepest_descent' or 'conjugate_gradient')
        :type method: str
        :return: self: Optimization problem
        :rtype: object
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
        x0: ArrayLike = None,
        gtol: float = 1e-6,
        alpha: float = 1,
        maxiter: int = None,
    ) -> Tuple[ArrayLike, dict]:
        """Solve the optimization problem using the method specified in the constructor.

        :raises [NotImplementedError]: Method not implemented

        :return:
        sol, info Tuple(NdArray, dict): Optimal point and additional information
        """
        if x0 is None:
            x = np.random.rand(self.dim).astype(np.float64)
        else:
            x = np.asarray(x0).flatten().astype(np.float64)

        if maxiter is None:
            maxiter = self.dim * 150

        if self.method == "steepest_descent":
            return steepest_descent(
                self, x0=x, gtol=gtol, alpha=alpha, maxiter=maxiter
            )
        elif self.method == "conjugate_gradient":
            return conjugate_gradient(
                self, x0=x, gtol=gtol, alpha=alpha, maxiter=maxiter
            )
        else:
            raise NotImplementedError("Method not implemented")


def line_search(
    f: Callable,
    direction: ArrayLike,
    x: ArrayLike,
    alpha: float = 1.0,
    beta: float = 0.5,
    maxiter: int = 100,
):
    """Line search algorithm with Armijo condition for finding the step size alpha
     that minimizes the function f(x + alpha * direction).
    :param f: Function to be minimized
    :type f: callable
    :param direction: Direction of search
    :type direction: ArrayLike
    :param x: Current point
    :type x: ArrayLike
    :param alpha: Initial step size
    :type alpha: float
    :param beta: Step size reduction factor
    :type beta: float
    :param maxiter: Maximum number of iterations
    :type maxiter: int

    :return:
    alpha, converged : Optimal step size and boolean variable indicating whether the algorithm converged
    :rtype: Tuple(float, Bool)
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
    x0: ArrayLike = None,
    gtol: float = 1e-6,
    alpha: float = 1.0,
    maxiter: int = None,
) -> Tuple[ArrayLike, dict]:
    """Steepest descent method for minimizing a scalar valued function f(x) of given gradient with Armijo line search.
    :param problem: Optimization problem
    :type problem: object of class Problem
    :param x0: Initial point
    :type x0: ArrayLike
    :param alpha: Step size
    :type alpha: float
    :param gtol: Tolerance for stopping the algorithm when the norm of the gradient is less than tol
    :type gtol: float
    :param maxiter: Maximum number of iterations
    :type maxiter: int

    :return:
    sol, info : Optimal point and additional information
        info = {"converged": bool, "iter_fvalues": np.ndarray, "iter_count": ¡nt}
        converged : bool : True if the algorithm converged
        iter_fvalues : np.ndarray : Function values at each iteration
        iter_count : int : Number of iterations

    :rtype: Tuple(ArrayLike, dict)
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
    problem, x0: ArrayLike = None, gtol: float = 1e-6, alpha: float = 1, maxiter=None
) -> Tuple[ArrayLike, dict]:
    """Conjugate gradient method with Fletcher-Reeves rule and Armijo line search for minimizing a  scalar valued
    function f(x).
    :param problem: Object of class Problem
    :type problem: object
    :param x0: Initial point
    :type x0: ArrayLike
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

        #set step size using Fletcher-Reeves rule
        beta = np.dot(grad1, grad1) / np.dot(grad0, grad0)
        # update descent direction
        direction = grad1 + beta * direction
        iter_count += 1

    info["iter_count"] = iter_count
    info["fvalues"] = iter_fvalues[:iter_count]
    return x, info


if __name__ == "__main__":
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
    # run conjugate gradient algorithm with default parameters
    sol, info = prob.solve(x0=x0, gtol=1e-8, maxiter=1000)
    # check that the algorithm converged
    assert info["converged"]
    # check that the solution is correct
    assert_allclose(sol, sol_exact, atol=1e-6)


