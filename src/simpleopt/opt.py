"""Optimization routines for maximizing/minimizing a scalar valued function f(x)."""

from typing import Callable, Tuple

import numpy as np
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
            x = np.random.rand(self.dim)
        else:
            x = np.asarray(x0).flatten()

        if maxiter is None:
            maxiter = self.dim * 150

        if self.method == "steepest_descent":
            return steepest_descent(
                self, x0=x0, gtol=gtol, alpha=alpha, maxiter=maxiter
            )
        elif self.method == "conjugate_gradient":
            return conjugate_gradient(
                self, x0=x0, gtol=gtol, alpha=alpha, maxiter=maxiter
            )
        else:
            raise NotImplementedError("Method not implemented")


def steepest_descent(
    problem: object,
    x0: ArrayLike = None,
    gtol: float = 1e-6,
    alpha: float = 1.0,
    maxiter: int = None,
) -> Tuple[ArrayLike, dict]:
    """Steepest descent method for minimizing a scalar valued function f(x).
    :param self: Optimization problem
    :type self: object
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
    :rtype: Tuple(ArrayLike, dict)
    """
    if x0 is None:
        x = np.random.rand(problem.dim)
    else:
        x = np.asarray(x0).flatten()

    if maxiter is None:
        maxiter = 1000 * problem.dim
    else:
        maxiter = int(maxiter)

    iter_count = 0
    iter_fvalues = np.zeros(maxiter, dtype=float)
    problem.info["iter_fvalues"] = iter_fvalues
    problem.info["converged"] = False
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
            problem.info["converged"] = True
            break
        # set search direction as negative gradient
        direction = -gradient
        # set step size
        step_size = alpha
        # perform line search to find optimal step size
        while func(x + step_size * direction) > func(x) + step_size * 0.1 * np.dot(
            gradient, direction
        ):
            step_size *= 0.5
        # update x with the step
        x = x + step_size * direction
        iter_count += 1

    problem.info["iter_count"] = iter_count
    problem.solution = x
    return x, problem.info


def conjugate_gradient(
    self, x0: ArrayLike = None, gtol: float = 1e-6, alpha: float = 1, maxiter=None
) -> Tuple[ArrayLike, dict]:
    """Conjugate gradient method of Fletcher-Reeves for minimizing a convex scalar valued function f(x).
    :param self: Optimization problem
    :type self: object
    :param x0: Initial point
    :type x0: ArrayLike
    :param gtol: Tolerance for stopping the algorithm when the norm of the gradient is less than tol
    :type gtol: float
    :param alpha: Step size
    :type alpha: float
    :param maxiter: Maximum number of iterations
    :type maxiter: int
    """

    #
    if x0 is None:
        x = np.random.rand(self.dim)
    else:
        x = np.asarray(x0).flatten()

    if maxiter is None:
        maxiter = self.dim * 150

    iter_count = 0
    iter_fvalues = np.zeros(maxiter, dtype=float)
    self.info["iter_fvalues"] = iter_fvalues
    self.info["converged"] = False
    if self.prob_type == "max":

        def func(x):
            return -self.f(x)

        def grad(x):
            return -self.gradf(x)

    else:
        func = self.f
        grad = self.gradf

    while iter_count < maxiter:
        # compute gradient and function value at current point
        gradient0 = grad(x)
        iter_fvalues[iter_count] = self.f(x)
        # check if gradient is small enough to stop
        if np.linalg.norm(gradient0) < gtol:
            self.info["converged"] = True
            break

        # set descent direction using Fletcher-Reeves formula
        if iter_count == 0:
            direction = -gradient0
        else:
            beta = np.dot(gradient0, gradient0) / np.dot(gradient1, gradient1)
            direction = -gradient0 + beta * direction

        # set step size
        step_size = alpha
        # perform line search to find optimal step size
        while func(x + step_size * direction) > func(x) + step_size * 0.1 * np.dot(
            gradient0, direction
        ):
            step_size *= 0.5

        # update x with the step size and direction
        x = x + step_size * direction
        iter_count += 1
        gradient1 = gradient0

    self.info["iter_count"] = iter_count
    self.info["fvalues"] = iter_fvalues[:iter_count]
    self.solution = x
    return x, self.info


if __name__ == "__main__":
    # define objective function and gradient
    def f(x):
        return x[0] ** 2 + x[1] ** 2

    def gradf(x):
        return np.array([2 * x[0], 2 * x[1]])

    # define optimization problem
    prob = Problem(f, gradf, dim=2, method="steepest_descent")
    # solve optimization problem
    sol, info = prob.solve()
