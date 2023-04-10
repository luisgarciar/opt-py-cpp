// test_quad_function.cpp
// Description: This file contains the implementation of the tests for
// 				the methods of the class quad_function
// 				QuadFunction.eval and QuadFunction.grad
// Author: Luis Garcia Ramos
// Date: 09.04.23


#include <catch2/catch_test_macros.hpp>
#include "../include/quad_function.hpp"

#define CATCH_CONFIG_MAIN

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

  // Evaluation of the function and the gradient
  double fx;
  double eps = 1e-6; // Relative tolerance for the comparison of floating point numbers

  // Expected values for the function and the gradient
  Eigen::VectorXd gradfx (2, 1);
  double fx_expected = 40.5;
  Eigen::VectorXd gradfx_expected (2, 1);
  gradfx_expected (0, 0) = 6;  // Correct value =5
  gradfx_expected (1, 0) = 11;

  // Evaluation of the function and the gradient
  fx = f.eval (x);
  gradfx = f.grad (x);

  // Comparison of the computed f(x) with the expected value (using Floating Point Matchers)
  // Comparison of the gradient grad(f)(x) with the expected value
  // (using the function isApprox for Eigen Vectors)
  REQUIRE(f.eval (x), Catch::Matchers::WithinRel (FloatingPoint fx_expected, FloatingPoint eps));
  ASSERT_TRUE (gradfx.isApprox (gradfx_expected, eps));

}
