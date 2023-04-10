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
#include <iostream>
#include <Eigen/Dense>   // Eigen Library for Linear Algebra Operations

using Eigen::MatrixXd;
using Eigen::VectorXd;


class quad_function {
  // Class for a quadratic Function of the form
  // f(x) = 0.5 * (x^T * mat * x) + b^T * x
 private:
  MatrixXd mat;
  VectorXd b;
  int dim;

 public:
  // Constructor
  quad_function(const MatrixXd mat_in, const VectorXd b_in);

  // Setter Functions
  void set_mat(const MatrixXd mat_in);
  void set_b(const VectorXd b_in);

  // Getter Functions
  MatrixXd get_mat() const;
  VectorXd get_b() const;

  // Evaluation Functions
  double eval(const VectorXd x) const;  // Evaluate the function at x

  VectorXd grad(const VectorXd x) const;	// Evaluate the gradient at x

};

#endif //_QUAD_FUNCTION_HPP_
