project(test)
cmake_minimum_required(VERSION 3.16)
#[[ 1. ==function command使用==
function(test arg1 arg2)
  message(STATUS "arguments number = " ${ARGC})
  message(STATUS "arg1 = " ${arg1})
  message(STATUS "arg2 = " ${arg2})
  message(STATUS "arg1 = " ${ARGV0})
  message(STATUS "arg2 = " ${ARGV1})
  foreach(arg ${ARGV})
    message(STATUS "arg = " ${arg})
  endforeach(arg ${})
  message(STATUS "ARGN = " ${ARGN})
endfunction(test)

test(1 2 3) 
#]]


#[[ 2. ==FetchContent模块使用demo==
include(FetchContent)

set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_BASE_DIR ${CMAKE_BINARY_DIR}/googletest)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.8.0
)

FetchContent_GetProperties(googletest)
if (NOT googletest_POPULATED)
  FetchContent_Populate(googletest)
  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
endif()

message(STATUS "googletest_SOURCE_DIR: " ${googletest_SOURCE_DIR})
message(STATUS "googletest_BINARY_DIR: " ${googletest_BINARY_DIR})
#]]


#[[ 3. ==cmake_parse_arguments command使用==
function(MY_INSTALL)
    set(options OPTIONAL FAST)
    set(oneValueArgs DESTINATION RENAME)
    set(multiValueArgs TARGETS CONFIGURATIONS)
    cmake_parse_arguments(MY_INSTALL "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN} )
    message(STATUS "MY_INSTALL_OPTIONAL : " ${MY_INSTALL_OPTIONAL})
    message(STATUS "MY_INSTALL_FAST : " ${MY_INSTALL_FAST})
    message(STATUS "MY_INSTALL_DESTINATION : " ${MY_INSTALL_DESTINATION})
    message(STATUS "MY_INSTALL_RENAME : " ${MY_INSTALL_RENAME})
    message(STATUS "MY_INSTALL_TARGETS : " ${MY_INSTALL_TARGETS})
    message(STATUS "MY_INSTALL_CONFIGURATIONS : " ${MY_INSTALL_CONFIGURATIONS})
    message(STATUS "MY_INSTALL_UNPARSED_ARGUMENTS : " ${MY_INSTALL_UNPARSED_ARGUMENTS})
endfunction(MY_INSTALL)
my_install(TARGETS foo bar DESTINATION bin OPTIONAL blub)

function(gather_srcs SRC_GROUP)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs "SRCS")
    cmake_parse_arguments(prefix "" "" "${multiValueArgs}" ${ARGN})
    foreach(cpp ${prefix_SRCS})
       set(${SRC_GROUP} "${${SRC_GROUP}};${CMAKE_CURRENT_SOURCE_DIR}/${cpp}" CACHE INTERNAL "" )
    endforeach()
endfunction()
gather_srcs(infrt_src SRCS infrt_api.cc)
message(STATUS "infrt_src : " ${infrt_src})

# ]]
