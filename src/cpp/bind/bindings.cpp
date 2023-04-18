//
// Created by Luis - Work on 10.04.23.
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "quad_function.hpp"

namespace py = pybind11;

PYBIND11_MODULE(quad, m){

py::class_<quadFunction>(m, "function")
.def(py::init< py::EigenDRef<Eigen::MatrixXd>, py::EigenDRef<Eigen::VectorXd>>())
.def("set_matrix", &quadFunction::set_mat, py::arg("mat"))
.def("set_vector", &quadFunction::set_vec, py::arg("vec"))
.def("get_matrix", &quadFunction::get_mat, py::return_value_policy::reference_internal )
.def("get_vector", &quadFunction::get_vec, py::return_value_policy::reference_internal )
.def("eval", &quadFunction::eval, py::arg("x"), py::return_value_policy::reference_internal )
.def("grad", &quadFunction::grad, py::arg("x"), py::return_value_policy::reference_internal );
}
