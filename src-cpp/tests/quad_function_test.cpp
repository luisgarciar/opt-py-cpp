// test_quad_function.cpp
// Description: This file contains the implementation of the tests for
// 				the methods of the class quad_function
// 				QuadFunction.eval and QuadFunction.grad
// Author: Luis Garcia Ramos
// Date: 09.04.23


#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "../include/quad_function.hpp"


TEST_CASE("Computation of quadratic function f", "[quad_function]")
{
  // Data and definition of the quadratic function
  MatrixXd mat (2, 2);
  mat (0, 0) = 1;
  mat (0, 1) = 2;
  mat (1, 0) = 3;
  mat (1, 1) = 4;

  VectorXd x (2, 1);
  x (0, 0) = 1;
  x (1, 0) = 2;

  VectorXd b (2, 1);
  b = mat * x;

  quad_function f (mat, b);

  double eps = 1e-6; // Relative tolerance for the comparison of floating point numbers

  // Expected value for the function and evaluation of the function
  double fx_expected = 40.5;
  double fx = f.eval (x);

  // Comparison of the computed f(x) with the expected value (using Matchers)
  REQUIRE_THAT(fx, Catch::Matchers::WithinAbs(fx_expected, eps));
}

TEST_CASE("Computation of gradient of quadratic function f", "[gradient]")
{
  // Data and definition of the quadratic function
  MatrixXd mat (2, 2);
  mat (0, 0) = 1;
  mat (0, 1) = 2;
  mat (1, 0) = 3;
  mat (1, 1) = 4;

  VectorXd x (2, 1);
  x (0, 0) = 1;
  x (1, 0) = 2;

  VectorXd b (2, 1);
  b = mat * x;

  quad_function f (mat, b);

  double eps = 1e-6; // Relative tolerance for the comparison of floating point numbers

  // Expected values for the gradient
  Eigen::VectorXd gradfx_expected (2, 1);
  gradfx_expected (0, 0) = 10;
  gradfx_expected (1, 0) = 22;

  // Evaluation of the gradient
  double fx = f.eval (x);
  Eigen::VectorXd gradfx (2, 1);
  gradfx = f.grad (x);

  // Comparison of the gradient grad(f)(x) with the expected value
  // (using the function isApprox for Eigen Vectors)
  CHECK (gradfx.isApprox (gradfx_expected, eps));

}




