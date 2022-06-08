project(demo_2.1)

add_library(fun fun.cc)
target_compile_options(fun PRIVATE "-fPIC")

add_library(test SHARED test.cc)
target_link_libraries(test fun)

add_executable(main main.cc)
target_link_libraries(main test)
