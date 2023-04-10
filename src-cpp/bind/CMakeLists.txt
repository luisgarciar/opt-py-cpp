# CMakeLists.txt file for the python bindings  (OLD)
#find_package(PythonLibs 3 REQUIRED)
#find_package(pybind11 REQUIRED)
#
#set(PYBIND_PATH ../extern/pybind11)
#include_directories(${PYBIND_PATH})
#
#if(NOT EXISTS "${PYBIND_PATH}/CMakeLists.txt")
#    message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
#endif()
#
#pybind11_add_module(bindings  bindings.cpp)
#

# CMakeLists.txt file for the python bindings
cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 17)

project(bindings)

include_directories("${CMAKE_SOURCE_DIR}/include")
include_directories("${CMAKE_SOURCE_DIR}/bindings")

file (GLOB SOURCE_FILES "src/*.cpp")
file (GLOB HEADER_FILES "*.hpp")
file (GLOB PYTHON_FILES "bindings/*.cpp" "bindings/*.hpp")

find_package(pybind11 REQUIRED)
pybind11_add_module(bindings
        ${SOURCE_FILES}
        ${HEADER_FILES}
        ${PYTHON_FILES}
        )

target_link_libraries(bindings PUBLIC quad_function)

