cmake_minimum_required(VERSION 3.11)

#Add the subdirectories with source code and tests
add_subdirectory(src)
add_subdirectory(bind)
add_subdirectory(tests)

enable_testing()

