# CMakeLists.txt file for the python bindings
include_directories("${PROJECT_SOURCE_DIR}/include")

# Fetch pybind11 from github
include(FetchContent)
FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11
        GIT_TAG v2.10.0
)

# Make pybind11 available for other targets
FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()

# Create the python module defined in bindings.cpp
pybind11_add_module(
        py_quad_fcn
        bindings.cpp
)

target_link_libraries(py_quad_fcn PRIVATE quad_function pybind11::module)
