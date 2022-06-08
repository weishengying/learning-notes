project(demo_2.4)
cmake_minimum_required(VERSION 3.16)

add_library(fun SHARED fun.cc)

add_library(test SHARED test.cc)

target_link_libraries(test fun)

set(LINK_FLAGS
        "-Wl,--version-script ${CMAKE_CURRENT_SOURCE_DIR}/a.map")
set_target_properties(test PROPERTIES LINK_FLAGS "${LINK_FLAGS}")
set_target_properties(fun PROPERTIES LINK_FLAGS "${LINK_FLAGS}")

add_executable(main main.cc)

target_link_libraries(main test)

