"""Optimization routines for maximizing/minimizing a scalar valued function f(x)."""

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
        :param self: Optimization problem
        :type self: object
        :param f: Function to be optimized
         :type f: callable
        :param gradf: Gradient of the function to be optimized
        :type gradf: callable
        :param x0: Initial point
        :type x0: NdArray
        :param prob_type: Type of optimization problem ('min' or 'max')
        :type prob_type: str
        :param tol: Tolerance for stopping the algorithm
        :type tol: float
        :param maxiter: Maximum number of iterations
        :type maxiter: int
        :param method: Optimization method to use
        :type method: str

        :return: self: Optimization problem
        :rtype: object
        """

        self.f = f
        self.gradf = gradf
        self.x0 = x0
        self.prob_type = prob_type
        self.tol = tol
        self.maxiter = maxiter
        self.method = method
        self.info = {}
        self.solution = None

    def steepest_descent(self: object, alpha: float = 1.0) -> Tuple[ArrayLike, dict]:
        """Steepest descent method for minimizing a scalar valued function f(x).
        :param self: Optimization problem
        :type self: object
        :param alpha: Step size
        :type alpha: float

        :return:
        sol, info : Optimal point and additional information
        :rtype: Tuple(NdArray, dict)
        """
        x = self.x0
        iter_count = 0
        iter_fvalues = np.zeros((self.maxiter,), dtype=float)
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

        while iter_count < self.maxiter:
            # compute gradient and function value at current point
            gradient = grad(x)
            iter_fvalues[iter_count] = self.f(x)
            # check if gradient is small enough to stop
            if np.linalg.norm(gradient) < self.tol:
                self.info["converged"] = True
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

        self.info["iter_count"] = iter_count
        self.solution = x
        return x, self.info

    def solve(self: object) -> Tuple[ArrayLike, dict]:
        """Solve the optimization problem using the specified method.

        :raises [NotImplementedError]: Method not implemented

        :return:
        sol, info Tuple(NdArray, dict): Optimal point and additional information
        """
        if self.method == "steepest_descent":
            return self.steepest_descent()

        else:
            raise NotImplementedError("Method not implemented")


if __name__ == "__main__":
    # define objective function and gradient
    def f(x):
        return x[0] ** 2 + x[1] ** 2


    def gradf(x):
        return np.array([2 * x[0], 2 * x[1]])


    # define optimization problem
    prob = problem(f, gradf, x0=np.array([1.0, 1.0]), method="steepest_descent")
    # solve optimization problem
    sol, info = prob.solve()
    print(sol)
