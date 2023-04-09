//
// Created by Luis - Work on 09.04.23.
//

#include "quad_function.hpp"

quad_function::quad_function (const MatrixXd mat_in, const VectorXd b_in)
{
  // Constructor
  mat = mat_in;
  b = b_in;
}

void quad_function::set_mat (const MatrixXd mat_in)
{
  mat = mat_in;
}

void quad_function::set_b (const VectorXd b_in)
{
  b = b_in;
}

MatrixXd quad_function::get_mat () const
{
  return mat;
}

VectorXd quad_function::get_b () const
{
  return b;
}

double quad_function::eval (const VectorXd x) const
{
  // Check if the dimensions of the input vector are correct
  if (x.rows () != mat.rows ())
	{
	  std::cout << "Error: The dimensions of the input vector are not correct" << std::endl;
	  return 0;
	}

  // Evaluate the function at x
  return 0.5 * double ((x.transpose () * (mat * x))) + double ((b.transpose () * x));
}

VectorXd quad_function::grad (const VectorXd x) const
{
  // Check if the dimensions of the input vector are correct
  if (x.rows () != mat.rows ())
	{
	  std::cout << "Error: The dimensions of the input vector are not correct" << std::endl;
	}
  // Evaluate the gradient at x
  return mat * x + b;
}

