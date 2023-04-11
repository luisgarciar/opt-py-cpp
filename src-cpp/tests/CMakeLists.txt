#Enable Testing with Catch2
enable_testing()
find_package(Catch2 3 REQUIRED)

##Fetch the library Catch2
#include(FetchContent)
#FetchContent_Declare(
#        Catch2
#        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
#        GIT_TAG        v3.0.0
#)
#FetchContent_MakeAvailable(Catch2)

#These tests are run with the Catch2-provided main function
add_executable(quad_tests quad_function_test.cpp)
target_link_libraries(quad_tests PRIVATE quad_function Catch2::Catch2WithMain)

#Let Catch2 discover and generate the tests
include(CTest)
include(Catch)
#catch_discover_tests(unit_tests)

add_test(
        NAME catch2_test
        COMMAND $<TARGET_FILE:quad_tests> --success
)
