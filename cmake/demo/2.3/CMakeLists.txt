project(demo_2.3)
cmake_minimum_required(VERSION 3.16)

add_library(fun SHARED fun.cc)

add_library(test SHARED test.cc)

target_link_libraries(test  LINK_PRIVATE fun)

add_executable(main main.cc)

target_link_libraries(main test)

