//
// Created by Luis - Work on 10.04.23.
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "quad_function.hpp"

namespace py = pybind11;

PYBIND11_MODULE(quad, m){

py::class_<quadFunction>(m, "function")
.def(py::init< py::EigenDRef<Eigen::MatrixXd>, py::EigenDRef<Eigen::VectorXd>>())
.def("set_mat", &quadFunction::set_mat, py::arg("mat_in"))
.def("set_b", &quadFunction::set_b, py::arg("b_in"))
.def("get_mat", &quadFunction::get_mat, py::return_value_policy::reference_internal )
.def("get_b", &quadFunction::get_b, py::return_value_policy::reference_internal )
.def("eval", &quadFunction::eval, py::arg("x"), py::return_value_policy::reference_internal )
.def("grad", &quadFunction::grad, py::arg("x"), py::return_value_policy::reference_internal );
}
