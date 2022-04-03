project(llvm_test)
cmake_minimum_required(VERSION 3.16)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

set(THIRD_PARTY_PATH ${CMAKE_SOURCE_DIR}/third-party)

include(llvm)

add_subdirectory(IR)