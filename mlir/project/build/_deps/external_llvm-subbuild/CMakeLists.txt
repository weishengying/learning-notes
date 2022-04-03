# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.16.0)

# We name the project and the target for the ExternalProject_Add() call
# to something that will highlight to the user what we are working on if
# something goes wrong and an error message is produced.

project(external_llvm-populate NONE)

include(ExternalProject)
ExternalProject_Add(external_llvm-populate
                     "UPDATE_DISCONNECTED" "False" "URL" "https://paddle-inference-dist.bj.bcebos.com/infrt/llvm_b5149f4e66a49a98b67e8e2de4e24a4af8e2781b.tar.gz" "URL_MD5" "022819bb5760817013cf4b8a37e97d5e" "PREFIX" "/weishengying/learning-notes/mlir/project/third-party/llvm"
                    SOURCE_DIR          "/weishengying/learning-notes/mlir/project/third-party/install/llvm"
                    BINARY_DIR          "/weishengying/learning-notes/mlir/project/build/_deps/external_llvm-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
)
