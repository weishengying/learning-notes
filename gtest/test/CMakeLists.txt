project(Gtest_demo)

cmake_minimum_required(VERSION 3.10)

# locate gtest
set(GTEST_INCLUDE_DIRS /usr/local/include/gtest/gtest.h)
include_directories(${GTEST_INCLUDE_DIRS})

# compile object
add_executable(runTests demo.cc)
set(GTEST_IBRARYS /usr/local/lib/libgtest.a)
target_link_libraries(runTests ${GTEST_IBRARYS} pthread)