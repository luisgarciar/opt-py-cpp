//
// Created by Luis - Work on 10.04.23.
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "quad_function.hpp"

namespace py = pybind11;

PYBIND11_MODULE(quad, m) {
	m.doc() = "pybind11 example plugin"; // optional module docstring

	py::class_<quadFunction>(m, "function")
		.def(py::init<MatrixXd, VectorXd>())
		.def("eval", &quadFunction::eval)
		.def("grad", &quadFunction::grad);
}
