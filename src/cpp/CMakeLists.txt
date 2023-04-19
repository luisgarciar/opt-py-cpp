cmake_minimum_required(VERSION 3.11)

#Add the subdirectories with source code and bindings
add_subdirectory(src)
add_subdirectory(bind)

#Add the subdirectory with the tests
# add_subdirectory(tests)   # not needed for now
set()