cmake_minimum_required(VERSION 3.24)
project(quadratic_function)

set(CMAKE_CXX_STANDARD 17)

# Set the directory of the Eigen library
set(EIGEN_DIR "/usr/local/include/eigen3")
if(NOT EXISTS ${EIGEN_DIR})
    message(FATAL_ERROR "Please check that the set Eigen directory is valid!")
endif()
include_directories(${EIGEN_DIR})

# Set the directory of the Catch2 library (for testing)
find_package(Catch2 3 REQUIRED)
# These tests can use the Catch2-provided main
add_executable(tests tests/test_quad_function.cpp)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)

# Include CTest and add the tests
include(CTest)
add_test(NAME TestQuadFunction COMMAND tests/test_quad_function)


# Add the library quad_function and the executable main
add_library(quad_function quad_function.hpp quad_function.cpp)
add_executable(main main.cpp)
