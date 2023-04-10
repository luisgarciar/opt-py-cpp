cmake_minimum_required(VERSION 3.20)
project(quadratic_function)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the directory of the Eigen library
set(EIGEN_DIR "/usr/local/include/eigen3")
if(NOT EXISTS ${EIGEN_DIR})
    message(FATAL_ERROR "Please check that the set Eigen directory is valid!")
endif()
include_directories(${EIGEN_DIR})

# Add the subdirectory containing the pybind11 library
add_subdirectory(../extern/pybind11)

#Add the subdirectory containing the bindings
add_subdirectory(bind)

# Add the library quad_function and the executable main
add_library(quad_function quad_function.hpp quad_function.cpp)
add_executable(main main.cpp bind/bindings.cpp)

#option(BUILD_TESTS "Build the tests" OFF)
#if(BUILD_TESTS) # Find the Catch2 library (for testing) and add the tests with CTest
#    find_package(Catch2 3 REQUIRED)
#    include(CTest)
#    # These tests are run with the Catch2-provided main function
#    add_executable(tests tests/test_quad_function.cpp)
#    target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)
#endif()


