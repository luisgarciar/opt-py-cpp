// quad_function.hpp
// Description: This file contains the declaration of the class quad_function
//              for representing quadratic functions of the form
//              f(x) = 0.5 * x^T * mat * x + b^T * x
//				where mat is a symmetric matrix and b is a vector
// Author: Luis Garcia Ramos
// Date: 09.04.23
//

#ifndef _QUAD_FUNCTION_HPP_
#define _QUAD_FUNCTION_HPP_
#include <Eigen/Dense>   // Eigen Library for Linear Algebra Operations
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

// We use py:EigenDRef to avoid copying the data and
// compatibility with default row ordering in
// numpy arrays
// See https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html


class quadFunction {
  // Class for a quadratic Function of the form
  // f(x) = 0.5 * (x^T * mat * x) + b^T * x
 private:
  py::EigenDRef<Eigen::MatrixXd> mat;
  py::EigenDRef<Eigen::VectorXd> b;

 public:
  // Constructor
  quadFunction (py::EigenDRef<Eigen::MatrixXd> mat_in, py::EigenDRef<Eigen::VectorXd> b_in);

  // Setter Functions
  void set_mat (py::EigenDRef<Eigen::MatrixXd> mat_in);
  void set_b (py::EigenDRef<Eigen::VectorXd> b_in);

  // Getter Functions
  Eigen::MatrixXd get_mat () const;
  Eigen::VectorXd get_b () const;

  // Evaluation Functions
  double eval (py::EigenDRef<Eigen::VectorXd> x) const;  // Evaluate the function at x

  Eigen::VectorXd grad (py::EigenDRef<Eigen::VectorXd> x) const; // Evaluate the gradient at x

};

#endif //_QUAD_FUNCTION_HPP_
