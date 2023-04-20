
find_package(Eigen3 3.3 REQUIRED)
message(STATUS "Eigen3 found (include: ${EIGEN3_INCLUDE_DIR})")

# Add library and link with Eigen
add_library(quad_function SHARED ../include/quad_function.hpp quad_function.cpp)
target_include_directories(quad_function PUBLIC ../include)
target_link_libraries(quad_function PUBLIC Eigen3::Eigen pybind11::module)


# Add the executable main and link the library quad_function and the executable main
#add_executable(main main.cpp ../include/simple_class.hpp)
#target_link_libraries(main quad_function)