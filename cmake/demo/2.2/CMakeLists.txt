project(demo_2.2)
cmake_minimum_required(VERSION 3.16)

add_library(fun SHARED fun.cc)

add_library(test SHARED test.cc)

# 加入下面这句话后可以观察 fun 和 test 之间的相对 build 顺序
# add_dependencies(fun test) 

add_executable(main main.cc)
target_link_libraries(main test fun)

