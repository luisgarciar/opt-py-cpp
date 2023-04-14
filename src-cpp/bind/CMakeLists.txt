# CMakeLists.txt file for the python bindings

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

# Directory where the header files are located
include_directories("${PROJECT_SOURCE_DIR}/src-cpp/include")
# Create the python module defined in bindings.cpp
pybind11_add_module(quad SHARED  bindings.cpp )

#Link the module to the pybind library and the quad_function library
target_link_libraries(quad PRIVATE quad_function pybind11::module)

# Copy the module to the python directory
install(TARGETS quad DESTINATION CMAKE_SOURCE_DIR/src-python)



#install(TARGETS quad DESTINATION .)