//
// Created by Luis - Work on 15.04.23.
//
#ifndef _SIMPLE_CLASS_HPP_
#define _SIMPLE_CLASS_HPP_

#include <Eigen/Dense>

class simpleClass{
  // This class is used to represent a matrix
 private:
  Eigen::MatrixXd mat;

 public:
  simpleClass (Eigen::MatrixXd mat_in);
  Eigen::MatrixXd multiply();
};

#endif //_SIMPLE_CLASS_HPP_
