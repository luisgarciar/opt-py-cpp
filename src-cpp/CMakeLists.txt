cmake_minimum_required(VERSION 3.11)



#Add the subdirectories with source code and tests
add_subdirectory(src)
add_subdirectory(bind)
add_subdirectory(tests)

enable_testing()

#Link the libraries to the executables
#target_link_libraries(quad_function PUBLIC Eigen3::Eigen)
#target_link_libraries(main quad_function)

#Enable Testing with Catch2
#enable_testing()
#find_package(Catch2 3 REQUIRED)
#include(CTest)
#include(Catch)

##Fetch the library Catch2
#include(FetchContent)
#FetchContent_Declare(
#        Catch2
#        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
#        GIT_TAG        v3.0.0
#)
#FetchContent_MakeAvailable(Catch2)

#These tests are run with the Catch2-provided main function
#add_executable(tests tests/test_quad_function.cpp)
#target_link_libraries(tests PRIVATE quad_function Catch2::Catch2WithMain)
#
#add_test(
#        NAME catch2_test
#        COMMAND $<TARGET_FILE:tests> --success
#)

#option(BUILD_TESTS "Build the tests" OFF)
#if(BUILD_TESTS) # Find the Catch2 library (for testing) and add the tests with CTest
#    find_package(Catch2 3 REQUIRED)
#    include(CTest)
#    # These tests are run with the Catch2-provided main function
#    add_executable(tests tests/test_quad_function.cpp)
#    target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)
#endif()


