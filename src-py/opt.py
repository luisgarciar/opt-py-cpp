"""Optimization algorithms for maximizing/minimizing a scalar valued function f(x)."""

from typing import Callable, Tuple

import numpy as np
from numpy.typing import ArrayLike


class problem:
    """A class for formulating an optimization problem.
    Sets up an optimization problem by defining the type of optimization problem (minimization or maximization),
    objective function, gradient, initial point, tolerance, maximum number of iterations, step size, and
    choice of optimization method.
    """


def __init__(
    self: object,
    f: Callable,
    gradf: Callable,
    x0: ArrayLike,
    prob_type: str = "min",
    tol: float = 1e-6,
    maxiter: int = 1000,
    method: str = None,
) -> object:
    """Initialize the problem class.
    :param self:
    :param f (callable): Function to be optimized
    :param gradf (callable): Gradient of the function to be optimized
    :param x0 (NdArray): Initial point
    :param prob_type (str): Type of optimization problem ('min' or 'max')
    :param tol (float): Tolerance for stopping the algorithm
    :param maxiter (int): Maximum number of iterations
    :param method (str): Optimization method to use
    """
    self.f = f
    self.gradf = gradf
    self.x0 = x0
    self.prob_type = prob_type
    self.tol = tol
    self.maxiter = maxiter
    self.method = method


def steepest_descent(self: object, alpha: float = 1.0) -> Tuple[ArrayLike, dict]:
    """Steepest descent algorithm for minimizing a scalar valued function f(x).
    :param self (object): Optimization problem
    :param alpha (float): Step size
    :return:
    sol, info Tuple(ArrayLike, dict): Optimal point and additional information
    """

    x = self.x0
    iter_count = 0
    iter_fvalues = np.zeros((self.maxiter,), dtype=float)

    while iter_count < self.maxiter:
        # compute gradient and function value at current point
        gradient = self.gradf(x)
        iter_fvalues[iter_count] = self.f(x)
        # check if gradient is small enough to stop
        if np.linalg.norm(gradient) < self.tol:
            break
        # set search direction as negative gradient
        direction = -gradient
        # set step size
        step_size = alpha
        # perform line search to find optimal step size
        while self.f(x + step_size * direction) > self.f(x) + step_size * 0.1 * np.dot(
            gradient, direction
        ):
            step_size *= 0.5
        # update x with the step
        x = x + step_size * direction
        iter_count += 1
    return x, {iter_count, iter_fvalues}


def solve(self: object) -> tuple:
    """Solve the optimization problem.
    :param self (object): Optimization problem
    :return:
    sol, info Tuple(NdArray, dict): Optimal point and additional information
    """
    if self.method == "steepest_descent":
        return self.steepest_descent()

    else:
        raise NotImplementedError("Method not implemented")


if __name__ == "main":
    # define objective function
    f = lambda x: x[0] ** 2 + x[1] ** 2
    gradf = lambda x: np.array([2 * x[0], 2 * x[1]])
    # define optimization problem
    prob = problem(f, gradf, x0=np.array([1.0, 1.0]), method="steepest_descent")
    # solve optimization problem
    sol, info = prob.solve()
    print(sol)
