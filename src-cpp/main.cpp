#include <iostream>
#include <Eigen/Dense>
#include "quad_function.cpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;

int main()
{
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

  std::cout << b << std::endl;

  quad_function f(mat, b);
  double fx;
  Eigen::VectorXd gradfx(2,1);
  fx = f.eval(x);
  gradfx = f.grad(x);

  return 0;
}