//
// Created by Luis - Work on 10.04.23.
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "simple_class.hpp"

namespace py = pybind11;

PYBIND11_MODULE(simp, m){
py::class_<simpleClass>(m, "simple")
.def(py::init<Eigen::MatrixXd>())
.def("multiply", &simpleClass::multiply);
}
