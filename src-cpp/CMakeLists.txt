cmake_minimum_required(VERSION 3.24)
project(funcs)

set(CMAKE_CXX_STANDARD 17)
set(EIGEN_DIR "/usr/local/include/eigen3")
if(NOT EXISTS ${EIGEN_DIR})
    message(FATAL_ERROR "Please check that the set Eigen directory is valid!")
endif()
include_directories(${EIGEN_DIR})

add_library(quad_function quad_function.hpp quad_function.cpp)
add_executable(main main.cpp)
