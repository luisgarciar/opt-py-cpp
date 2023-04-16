
find_package(Eigen3 3.3 REQUIRED)
message(STATUS "Eigen3 found (include: ${EIGEN3_INCLUDE_DIR})")

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



# Add library and link with Eigen
add_library(quad_function STATIC ../include/quad_function.hpp quad_function.cpp)
target_include_directories(quad_function PUBLIC ../include)
target_link_libraries(quad_function PUBLIC Eigen3::Eigen pybind11::module)

add_library(simple_class STATIC ../include/simple_class.hpp simple_class.cpp)
target_include_directories(simple_class PUBLIC ../include)
target_link_libraries(simple_class PUBLIC Eigen3::Eigen)


# Add the executable main and link the library quad_function and the executable main
#add_executable(main main.cpp ../include/simple_class.hpp)
#target_link_libraries(main quad_function)