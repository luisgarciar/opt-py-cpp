# CMakeLists.txt file for the python bindings
find_package(PythonLibs 3 REQUIRED)
find_package(pybind11 REQUIRED)
add_subdirectory(../../extern/pybind11/include/pybind11)
pybind11_add_module(bindings bindings.cpp)

