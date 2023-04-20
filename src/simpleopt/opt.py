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
        max_iter: int = None,
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
        :param max_iter: Maximum number of iterations
        :type max_iter: int
        :return: sol: Optimal point
        :rtype: NDArray
        :return info :  Dictionary with additional information about the optimization solve.
        :rtype info: dict
        """
        if self.method == "sd":
            optimizer = SteepestDescent(self, x0, gtol, alpha, max_iter)
        elif self.method == "cg":
            optimizer = ConjugateGradient(self, x0, gtol, alpha, max_iter)
        else:
            raise ValueError("Method must be either 'sd' or 'cg'")

        info = {
            "iter_count": optimizer.iter_count,
            "converged": optimizer.converged,
            "iter_fvalues": optimizer.iter_fvalues,
        }

        return optimizer.solution, info
        self.iter_fvalues = None


class _Optimizer:
    """Base class for optimization algorithms."""

    def __init__(
        self: object,
        problem: Problem,
        x0: NDArray = None,
        gtol: float = 1e-6,
        alpha: float = 1,
        max_iter=None,
    ):
        """Instantiate the optimizer with the problem to be solved and parameters.
        :param problem: Object of class Problem
        :type problem: object
        :param x0: Initial point
        :type x0: NDArray
        :param gtol: Tolerance for stopping the algorithm when the norm of the gradient is less than tol
        :type gtol: float
        :param alpha: Step size
        :type alpha: float
        :param max_iter: Maximum number of iterations
        :type max_iter: int

        After instantiating the Optimizer, the object contains the following attributes:
          * ``sol``: the approximate solution ``sol``
          *  ``converged``: True if the algorithm converged, False otherwise.
          * ``iter_count``: Number of iterations performed.
          * ``iter_fvalues``: List of function values at each iteration.
        """
        # Sanitize input and initialize variables
        if x0 is None:
            self.x0 = np.random.rand(problem.dim)
        else:
            self.x0 = np.asarray(x0).flatten()

        if max_iter is None:
            self.max_iter = problem.dim * 150
        else:
            self.max_iter = max_iter

        self.problem = problem
        self.iter_count = 0
        self.iter_fvalues = None
        self.solution = None
        self.converged = False
        self.gtol = gtol
        self.alpha = alpha

        if problem.prob_type == "max":

            def func(x):
                return -problem.f(x)

            def grad(x):
                return -problem.gradf(x)

            self.func = func
            self.grad = grad
        else:
            self.func = problem.f
            self.grad = problem.gradf

            # Call the abstract method _solve
        self._solve()

    def _solve(self):
        """Abstract method that solves the optimization problem."""
        if self.__class__ is _Optimizer:
            raise NotImplementedError(
                "_solve has to be overridden by " "the derived optimizer class."
            )

    def _line_search(
        self,
        direction: NDArray,
        x: NDArray,
        alpha: float = 1.0,
        beta: float = 0.5,
        max_iter: int = 100,
    ):

        """Line search algorithm with Armijo condition for finding the step size
         that minimizes a function at a given point towards a given direction.

        :param direction: Direction of search
        :type direction: NDArray
        :param x: Current point
        :type x: NDArray
        :param alpha: Initial step size
        :type alpha: float
        :param beta: Step size reduction factor
        :type beta: float
        :param max_iter: Maximum number of iterations
        :type max_iter: int
        :return: alpha: Optimal step size
        :rtype alpha: float
        :return: converged: Boolean variable indicating whether the algorithm converged
        :rtype converged: bool
        """
        iter_count = 0
        t = alpha

        ff = self.func

        # Find  t such that the Armijo condition is satisfied
        # ff(x + t * direction) <= ff(x) + alpha * t * ||grad f(x)||^2
        while ff(x + t * direction) > (
            ff(x) - beta * t * np.linalg.norm(direction) ** 2
        ):
            t = t * beta
            iter_count += 1

            if iter_count > max_iter or t < 1e-20:
                converged = False
                return t, converged

        return t, True


class SteepestDescent(_Optimizer):
    """Steepest descent algorithm with Armijo line search for solving unconstrained optimization problems."""

    def __init__(self, problem, x0, gtol, alpha, max_iter):
        """All parameters of :py:class:`_Optimizer` are valid in this optimization solver."""
        super().__init__(problem, x0, gtol, alpha, max_iter)

    def _solve(
        self,
    ):
        """Solve the optimization problem using the steepest descent algorithm."""
        x = self.x0
        iter_count = 0

        self.iter_fvalues = np.zeros(self.max_iter)
        self.iter_fvalues[iter_count] = self.func(x)

        while self.iter_count < self.max_iter:
            gradient = self.grad(x)
            self.iter_fvalues[iter_count] = self.func(x)

            # check if gradient is small enough to stop
            if np.linalg.norm(gradient) < self.gtol:
                self.converged = True
                break

            # set search direction as negative gradient
            direction = -gradient
            # set step size using line search
            step_size, converged = self._line_search(direction, x)

            if converged:  # if line search converged, update x
                x = x + step_size * direction
            else:
                self.converged = False
                break

            iter_count += 1

        self.iter_count = iter_count
        self.solution = x


class ConjugateGradient(_Optimizer):
    """Conjugate Gradient algorithm with Fletcher-Reeves rule and Armijo line search for solving unconstrained
    optimization problems."""

    def __init__(self, problem, x0, gtol, alpha, max_iter):
        """
        All parameters of :py:class:`_Optimizer` are valid in this optimization solver.
        """
        super().__init__(problem, x0, gtol, alpha, max_iter)

    def _solve(self):
        """Solve the optimization problem using the conjugate gradient algorithm."""
        x = self.x0
        iter_count = 0

        # initialize function values
        self.iter_fvalues = np.zeros(self.max_iter)
        self.iter_fvalues[iter_count] = self.func(x)

        # initialize conjugate gradient direction
        grad0 = self.grad(x)
        direction = -grad0

        while iter_count < self.max_iter:
            step_size, converged = self._line_search(direction, x)

            if converged:  # if line search converged, update x
                x = x + step_size * direction
            else:  # if line search did not converge, stop and return
                self.converged = False
                break
            # compute new gradient and function value, check stopping criterion
            grad1 = -self.grad(x)
            self.iter_fvalues[iter_count] = self.func(x)

            if np.linalg.norm(grad1) < self.gtol:
                self.converged = True
                break

            # set step size using Fletcher-Reeves rule
            beta = np.dot(grad1, grad1) / np.dot(grad0, grad0)
            # update descent direction
            direction = grad1 + beta * direction
            iter_count += 1

        self.iter_count = iter_count
        self.iter_fvalues = self.iter_fvalues[:iter_count]
        self.solution = x


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
    sol1, info1 = prob1.solve(x0=x0, gtol=1e-8, max_iter=50)
    # check that the algorithm converged
    assert info1["converged"]
    # check that the solution is correct
    assert_allclose(sol1, sol_exact, atol=1e-6)

    # define optimization problem
    # Define quadratic function f(x) = 0.5 * x.T @ A @ x + b.T @ x using quad module
    import quad

    f = quad.Function(A, b)
    # Define optimization problem
    prob2 = Problem(f.eval, f.grad, dim, prob_type="min", method="cg")
    # Solve optimization problem with the conjugate gradient algorithm
    x0 = np.zeros((dim,)).astype(np.float64)
    sol2, info2 = prob2.solve(x0=x0, gtol=1e-6, max_iter=50)

    # Check that the algorithm converged and that the solution is correct
    assert info2["converged"]
    assert_allclose(sol2, sol_exact, atol=1e-6, equal_nan=True)
