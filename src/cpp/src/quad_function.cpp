// quad_function.cpp
// Description: This file contains the implementation of the class quad_function
//              for representing quadratic functions of the form
//              f(x) = 0.5 * x^T * mat * x + vec^T * x
//				where mat is a matrix and vec is a vector
// Author: Luis Garcia Ramos
// Date: 09.04.23
//

#include "../include/quad_function.hpp"
#include <iostream>

quadFunction::quadFunction (py::EigenDRef<Eigen::MatrixXd> mat_in, py::EigenDRef<Eigen::VectorXd> vec_in)
	:
	mat (mat_in), vec (vec_in)
{

  set_mat (mat_in);
  set_vec (vec_in);

}

void quadFunction::set_mat (py::EigenDRef<Eigen::MatrixXd> mat_in)
// Setter Method for the matrix mat
{
  // Check if the dimensions of the input matrix are correct
  if (mat_in.rows () != mat_in.cols ())
	{
	  //std::cout << "Error: Incorrect dimensions of the input matrix" << std::endl;
	  throw std::invalid_argument ("Input matrix must be a square matrix");
	}
  else
	{
	  mat = mat_in;
	}
}

void quadFunction::set_vec (py::EigenDRef<Eigen::VectorXd> vec_in)
{// Setter Method for the vector b
  // Check if the dimensions of the input vector are correct
  if (vec_in.rows () != mat.rows ())
	{
	  //std::cout << "Error: Incorrect dimensions of the input vector" << std::endl;
	  throw std::invalid_argument ("\"Input vector must be compatible with the matrix");
	}
  else
	{
	  vec = vec_in;
	}
}

Eigen::MatrixXd quadFunction::get_mat () const
{// Getter Method for the matrix mat
  return mat;
}

Eigen::VectorXd quadFunction::get_vec () const
{// Getter Method for the vector b
  return vec;
}

double quadFunction::eval (py::EigenDRef<Eigen::VectorXd> x) const
{// Method for evaluating the function f(x) = 0.5 * x^T * mat * x + b^T * x
  // Check if the dimensions of the input vector are correct
  if (x.rows () != mat.rows ())
	{
	  //std::cout  "Error: Incorrect dimensions of the input vector" << std::endl;
	  throw std::invalid_argument ("Input vector must be compatible with the quadratic function");
	}
  else
	{

	  // Evaluate the function at x
	  return 0.5 * double ((x.transpose () * (mat * x))) + double ((vec.transpose () * x));
	}
}

Eigen::VectorXd quadFunction::grad (py::EigenDRef<Eigen::VectorXd> x) const
{ // Method for the gradient of the function f(x) = 0.5 * x^T * mat * x + vec^T * x
  // grad(f)(x) = mat * x + vec
  // Check if the dimensions of the input vector are correct
  if (x.rows () != mat.rows ())
	{
	  throw std::invalid_argument ("Input vector must be compatible with the quadratic function");
	}
	else
	{
	  // Evaluate the gradient at x
	  return (mat * x) + vec;
	}
}

