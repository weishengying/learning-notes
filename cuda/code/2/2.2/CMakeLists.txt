project(code_1)
cmake_minimum_required(VERSION 3.16)

FIND_PACKAGE(CUDA REQUIRED)

set(CUDA_PATH /usr/local/cuda)
include_directories(${CUDA_PATH}/include)

function(compile_demo target source)
  cuda_add_executable(${target} ${source})
  target_link_libraries(${target} ${CUDA_PATH}/lib64/libcudart.so)
endfunction(compile_demo)

compile_demo(matrix_add matrix_add.cu)
