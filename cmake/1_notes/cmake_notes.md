# 常用cmake操作
  1.查看所有的target：  cmake --build . --target help  
                    make help
  2.cmake过程中提示找不多pythonlib ：
    [参考](https://stackoverflow.com/questions/24174394/cmake-is-not-able-to-find-python-libraries)
    -DPYTHON_INCLUDE_DIR=$(python3.7 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")  \
    -DPYTHON_LIBRARY=$(python3.7 -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
## 常用cmake函数
  1. cmake自定义函数：fuction [function](https://cmake.org/cmake/help/v3.10/command/function.html?highlight=function)
  2. cmake FetchContent 模块  [FetchContent](https://cmake.org/cmake/help/v3.16/module/FetchContent.html)
  3. cmake_parse_arguments： [cmake_parse_arguments](https://cmake.org/cmake/help/v3.10/command/cmake_parse_arguments.html?highlight=cmake_parse_arguments)
    