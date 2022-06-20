project(demo_1)
cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_FLAGS "-Wl,--no-as-needed")

set(LINK_WHAT_YOU_USE TRUE)

add_library(op SHARED op.cc)

add_library(new_op SHARED new_op.cc)

add_executable(main main.cc)

target_link_libraries(main PUBLIC op new_op)