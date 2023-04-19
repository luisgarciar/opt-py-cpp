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
m.
doc () = "Module for defining quadratic functions of the form "
		 " f(x) = 0.5*(x.T @ A @ x) + b.T @ x"; // optional module docstring
py::class_<quadFunction>(m,
"Function", "A function of the form f(x) = 0.5*(x.T @ A @ x) + b.T @ x")
.
def(py::init<py::EigenDRef < Eigen::MatrixXd>, py::EigenDRef<Eigen::VectorXd>>
())
.def_property("matrix", &quadFunction::get_mat, &quadFunction::set_mat)
.def_property("vector", &quadFunction::get_vec, &quadFunction::set_vec)
.def("eval", &quadFunction::eval, "Evaluates the quadratic function at the given point",
py::arg("x"), py::return_value_policy::take_ownership)
.def("grad", &quadFunction::grad, "Evaluates the gradient of the quadratic function at the given point", py::arg("x"),
py::return_value_policy::take_ownership);
}

