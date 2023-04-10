cmake_minimum_required(VERSION 3.2)
project(quadratic_function LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.3 REQUIRED CONFIG)

# Add the library quad_function and the executable main
add_library(quad_function STATIC include/quad_function.hpp src/quad_function.cpp)
add_executable(main src/main.cpp)

#Link the libraries to the executables
target_link_libraries(quad_function PUBLIC Eigen3::Eigen)
target_link_libraries(main quad_function)

#Enable Testing with Catch2
enable_testing()
find_package(Catch2 3 REQUIRED)
include(CTest)

#These tests are run with the Catch2-provided main function
add_executable(tests tests/test_quad_function.cpp)
target_link_libraries(tests PRIVATE quad_function Catch2::Catch2WithMain)

add_test(
        NAME catch2_test
        COMMAND $<TARGET_FILE:tests> --success
)




#option(BUILD_TESTS "Build the tests" OFF)
#if(BUILD_TESTS) # Find the Catch2 library (for testing) and add the tests with CTest
#    find_package(Catch2 3 REQUIRED)
#    include(CTest)
#    # These tests are run with the Catch2-provided main function
#    add_executable(tests tests/test_quad_function.cpp)
#    target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)
#endif()


