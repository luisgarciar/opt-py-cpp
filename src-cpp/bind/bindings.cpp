//
// Created by Luis - Work on 10.04.23.
#include <pybind11/pybind11.h>
#include "quad_function.hpp"

namespace py = pybind11;

PYBIND11_MODULE(quad_func, qf) {
	qf.doc() = "pybind11 example plugin"; // optional module docstring

	py::class_<quad_function>(qf, "quad_function")
		.def(py::init<MatrixXd, VectorXd>())
		.def("eval", &quad_function::eval)
		.def("grad", &quad_function::grad);
}
