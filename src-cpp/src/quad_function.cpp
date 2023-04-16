// quad_function.cpp
// Description: This file contains the implementation of the class quad_function
//              for representing quadratic functions of the form
//              f(x) = 0.5 * x^T * mat * x + b^T * x
//				where mat is a symmetric matrix and b is a vector
// Author: Luis Garcia Ramos
// Date: 09.04.23
//

#include "../include/quad_function.hpp"
#include <iostream>


quadFunction::quadFunction (const py::EigenDRef<Eigen::MatrixXd> mat_in, const py::EigenDRef<Eigen::VectorXd> b_in)
	:
	mat (mat_in), b (b_in)
{

  if (mat.rows () != mat.cols ())
	{
	  std::cout << "Error: Incorrect dimensions of the input matrix" << std::endl;
	}

  // Check if the dimensions of the input vector are compatible with the matrix
  if (b.rows () != mat.rows ())
	{
	  std::cout << "Error: Incompatible dimensions of the input matrix and vector" << std::endl;
	}
}

void quadFunction::set_mat (const py::EigenDRef<Eigen::MatrixXd> mat_in)
// Setter Method for the matrix mat
{
  // Check if the dimensions of the input matrix are correct
  if (mat_in.rows () != mat_in.cols ())
	{
	  std::cout << "Error: Incorrect dimensions of the input matrix" << std::endl;
	}
  mat = mat_in;
}

void quadFunction::set_b (const py::EigenDRef<Eigen::VectorXd> b_in)
{// Setter Method for the vector b
  // Check if the dimensions of the input vector are correct
  if (b_in.rows () != mat.rows ())
	{
	  std::cout << "Error: Incorrect dimensions of the input vector" << std::endl;
	}
  b = b_in;
}

py::EigenDRef<Eigen::MatrixXd> quadFunction::get_mat () const
{// Getter Method for the matrix mat
  return mat;
}

py::EigenDRef<Eigen::VectorXd> quadFunction::get_b () const
{// Getter Method for the vector b
  return b;
}

double quadFunction::eval (const py::EigenDRef<Eigen::VectorXd> x) const
{// Method for evaluating the function f(x) = 0.5 * x^T * mat * x + b^T * x
  // Check if the dimensions of the input vector are correct
  if (x.rows () != mat.rows ())
	{
	  std::cout << "Error: Incorrect dimensions of the input vector" << std::endl;
	  return 0;
	}

  // Evaluate the function at x
  return 0.5 * double ((x.transpose () * (mat * x))) + double ((b.transpose () * x));
}

py::EigenDRef<Eigen::VectorXd> quadFunction::grad (py::EigenDRef<Eigen::VectorXd> x) const
{ // Method for the gradient of the function f(x) = 0.5 * x^T * mat * x + b^T * x
  // grad(f)(x) = mat * x + b
  // Check if the dimensions of the input vector are correct
  if (x.rows () != mat.rows ())
	{
	  std::cout << "Error: The dimensions of the input vector are not correct" << std::endl;
	}
  // Evaluate the gradient at x
  return mat ;
}

