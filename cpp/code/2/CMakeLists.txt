project(demo_1)

add_library(var SHARED fun.cc var.cc)

add_executable(main main.cc)
# target_link_libraries(main test)