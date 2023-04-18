# CMakeLists.txt file for the python bindings
find_package(Eigen3 3.3 REQUIRED)

include(FetchContent)
FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11
        GIT_TAG v2.10.4
)

# Make pybind11 available for other targets
FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()
FetchContent_MakeAvailable(pybind11)

# Directory where the header files are located
include_directories("${CMAKE_SOURCE_DIR}/src/cpp/include")
# Create the python module defined in bindings.cpp
pybind11_add_module(quad SHARED bindings.cpp)
#Link the module to the pybind11 library and the quad_function library
target_link_libraries(quad PUBLIC quad_function pybind11::module)



#install(TARGETS quad DESTINATION .)
#install(TARGETS quad DESTINATION CMAKE_SOURCE_DIR)
#install(TARGETS simp DESTINATION CMAKE_SOURCE_DIR/src-python)
# Not working -- fetch pybind11 from github if not found
## Check if pybind11 is already installed, otherwise fetch it from github
#if (NOT pybind11_FOUND)
#    MESSAGE(STATUS "pybind11 not found, fetching it")
#    include(FetchContent)
#    FetchContent_Declare(
#            pybind11
#            GIT_REPOSITORY https://github.com/pybind/pybind11
#            GIT_TAG v2.10.0
#    )
#    FetchContent_GetProperties(pybind11)
#    if(NOT pybind11_POPULATED)
#        FetchContent_Populate(pybind11)
#        add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
#    endif()
#    FetchContent_MakeAvailable(pybind11)
#endif()
