project(ctest_demo)
cmake_minimum_required(VERSION 3.16)

add_executable(sum_integers sum_integers.cpp sum_integers_test.cpp)
target_link_libraries(sum_integers gtest pthread)

enable_testing()
add_test(NAME test 
          COMMAND sum_integers)