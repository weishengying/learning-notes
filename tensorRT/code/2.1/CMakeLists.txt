project(tensorrt_demo)
cmake_minimum_required(VERSION 3.16)

set(TRT_PATH /weishengying/DownLoads/TensorRT-8.2.3.0)
set(CUDA_PATH /usr/local/cuda)

include_directories(${TRT_PATH}/include)
include_directories(${CUDA_PATH}/include)
function(compile_demo target source)
  add_executable(${target} ${source})
  target_link_libraries(${target} ${CUDA_PATH}/lib64/libcudart.so)
  target_link_libraries(${target} ${TRT_PATH}/lib/libnvinfer.so)
  target_link_libraries(${target} ${TRT_PATH}/lib/libnvinfer_plugin.so)
  #target_link_libraries(${target} /usr/lib/x86_64-linux-gnu/libcudnn.so)
endfunction(compile_demo)

compile_demo(test test.cc)


FIND_PACKAGE(CUDA REQUIRED)
enable_language(CUDA)
add_library(custom_plugin LeakRelu_Plugin.cu)

target_link_libraries(test custom_plugin)
