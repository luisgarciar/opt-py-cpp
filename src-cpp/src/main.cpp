// main.cpp
// Description: This file illustrates the use of the	class quad_function
//              for representing quadratic functions of the form
//              f(x) = 0.5 * x^T * A * x + b^T * x
// Author: Luis Garcia Ramos
// Date: 09.04.23
//

#include <iostream>
#include <Eigen/Dense>
#include "quad_function.cpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;

int main()
{ // Main function

  // Create a matrix mat and vectors x and b
  MatrixXd mat(2,2);
  mat(0,0) = 1;
  mat(0,1) = 2;
  mat(1,0) = 3;
  mat(1,1) = 4;

  VectorXd x(2,1);
  x(0,0) = 1;
  x(1,0) = 2;

  VectorXd b(2,1);
  b = mat*x;

  // Print the values of the matrix and vector
  std::cout << "A = " << mat << std::endl;
  std::cout << "x = " << x << std::endl;
  std::cout << "b = " << b << std::endl;

  // Create an object f of the class quad_function
  // with the matrix mat and vector b
  quad_function f(mat, b);

  // Evaluate the function f and the gradient grad(f) at x
  double fx;
  Eigen::VectorXd gradfx(2,1);
  fx = f.eval(x);
  gradfx = f.grad(x);

  // Print the values of the function and the gradient
  std::cout << "For the function f(x) = x^T  * A * x + b^T * x we have: " << std::endl;
  std::cout << "f(x) = " << fx << std::endl;
  std::cout << "grad(f)(x) = " << gradfx << std::endl;

  return 0;
}