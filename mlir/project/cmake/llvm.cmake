include(FetchContent)

set(LLVM_DOWNLOAD_URL https://paddle-inference-dist.bj.bcebos.com/infrt/llvm_b5149f4e66a49a98b67e8e2de4e24a4af8e2781b.tar.gz)
set(LLVM_MD5 022819bb5760817013cf4b8a37e97d5e)

FetchContent_Declare(external_llvm
  URL ${LLVM_DOWNLOAD_URL}
  URL_MD5 ${LLVM_MD5}
  PREFIX ${THIRD_PARTY_PATH}/llvm
  SOURCE_DIR ${THIRD_PARTY_PATH}/install/llvm
)

FetchContent_GetProperties(external_llvm)
if(NOT external_llvm_POPULATED)
  FetchContent_Populate(external_llvm)
endif()

set(LLVM_DIR ${THIRD_PARTY_PATH}/install/llvm/lib/cmake/llvm)
set(MLIR_DIR ${THIRD_PARTY_PATH}/install/llvm/lib/cmake/mlir)

list(APPEND CMAKE_MODULE_PATH "${LLVM_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_DIR}")
include(AddLLVM)
include(TableGen)
include(AddMLIR)
include(MLIRConfig)

include_directories(${THIRD_PARTY_PATH}/install/llvm/include)
message(STATUS ${THIRD_PARTY_PATH}/install/llvm/include)