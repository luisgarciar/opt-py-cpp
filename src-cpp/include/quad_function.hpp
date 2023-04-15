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


class quadFunction {
  // Class for a quadratic Function of the form
  // f(x) = 0.5 * (x^T * mat * x) + b^T * x
 private:
  Eigen::Ref<Eigen::MatrixXd> mat;
  Eigen::Ref<Eigen::VectorXd> b;

 public:
  // Constructor
  quadFunction (Eigen::Ref<Eigen::MatrixXd> mat_in, Eigen::Ref<Eigen::VectorXd> b_in);

  // Setter Functions
  void set_mat (Eigen::Ref<Eigen::MatrixXd> mat_in);
  void set_b (Eigen::Ref<Eigen::VectorXd> b_in);

  // Getter Functions
  Eigen::Ref<Eigen::MatrixXd> get_mat () const;
  Eigen::Ref<Eigen::VectorXd> get_b () const;

  // Evaluation Functions
  double eval (Eigen::Ref<Eigen::VectorXd> x) const;  // Evaluate the function at x

  Eigen::Ref<Eigen::VectorXd> grad (Eigen::Ref<Eigen::VectorXd> x) const; // Evaluate the gradient at x

};

#endif //_QUAD_FUNCTION_HPP_
