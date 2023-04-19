//
// Created by Luis - Work on 10.04.23.
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "quad_function.hpp"

namespace py = pybind11;

PYBIND11_MODULE(quad, m
){
py::options options;
options.
disable_function_signatures ();     // disable *default* function signatures in the docstrings
options.
disable_enum_members_docstring (); // disable *default* enum members docstrings
m.doc () = R"pbdoc(
        .. currentmodule:: quad
        .. autosummary::
           :toctree: _generate
           eval
           grad
    )pbdoc";
// optional module docstring
py::class_<quadFunction>(m, "Function", "Class for representing a function of the form f(x) = 0.5*(x.T @ A @ x) + b.T @ x")
.
def(py::init<py::EigenDRef < Eigen::MatrixXd>, py::EigenDRef<Eigen::VectorXd>>
(), R"pbdoc(Constructor for a function of the form f(x) = 0.5*(x.T @ A @ x) + b.T @ x
    :param matrix: Matrix A (dtype: float64)
    :type problem: NDArray
    :param vector: Vector b (dtype: float64)
	:type vector: NDArray
    )pbdoc"))
.def_property("matrix", &quadFunction::get_mat, &quadFunction::set_mat)
.def_property("vector", &quadFunction::get_vec, &quadFunction::set_vec)
.def("eval", &quadFunction::eval, R"pbdoc(Evaluates the quadratic function at the given point
    :param x: Point at which to evaluate the function (dtype: float64)
    :type problem: NDArray
    )pbdoc",
py::arg("x"), py::return_value_policy::take_ownership)
.def("grad", &quadFunction::grad, R"pbdoc(Evaluates the gradient of the quadratic function at the given point
    :param x: Point at which to evaluate the function (dtype: float64)
    :type problem: NDArray
    )pbdoc", py::arg("x"),
py::return_value_policy::take_ownership);
}

