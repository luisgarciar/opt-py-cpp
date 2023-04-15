#Enable Testing with Catch2
enable_testing()
find_package(Catch2 3 REQUIRED)

if (NOT Catch2_FOUND)
    MESSAGE(STATUS "Catch2 not found, fetching it")
    include(FetchContent)
    FetchContent_Declare(
            Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG        v3.0.0
    )
    FetchContent_GetProperties(Catch2)
    if(NOT Catch2_POPULATED)
        FetchContent_Populate(Catch2)
        add_subdirectory(${Catch2_SOURCE_DIR} ${Catch2_BINARY_DIR})
    endif()
    FetchContent_MakeAvailable(Catch2)
endif()


#These tests are run with the Catch2-provided main function
add_executable(quad_tests quad_function_test.cpp)
target_link_libraries(quad_tests PRIVATE quad_function Catch2::Catch2WithMain)

#Use Catch2 to generate the tests
include(CTest)
include(Catch)

add_test(
        NAME catch2_test
        COMMAND $<TARGET_FILE:quad_tests> --success
)
