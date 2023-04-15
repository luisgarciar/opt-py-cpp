//
// Created by Luis - Work on 15.04.23.
//

#include "simple_class.hpp"
#include <iostream>

//simple_class constructor

simpleClass::simpleClass(Eigen::MatrixXd mat_in):
  mat(mat_in)
{
  // Check if the dimensions of the input matrix are correct
  if (mat.rows () != mat.cols ())
	{
	  std::cout << "Error: Incorrect dimensions of the input matrix" << std::endl;
	}
}

Eigen::MatrixXd simpleClass::multiply ()
{
  return 2*mat;
}

