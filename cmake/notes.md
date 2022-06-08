
# 1.常用cmake操作
1.查看所有的target：  
```shell
cmake --build . --target help  
make help
```

2.cmake过程中提示找不多pythonlib ：

[参考](https://stackoverflow.com/questions/24174394/cmake-is-not-able-to-find-python-libraries)

手动设置下面两个 cmake 变量即可， 或者修改 cmake find_package 指定的搜索路径。
```shell
-DPYTHON_INCLUDE_DIR=$(python3.7 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")  \
-DPYTHON_LIBRARY=$(python3.7 -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR')")
```
     
# 2.常见 cmake 问题
## 2.1 动态库可以链接静态库吗

> 静态库：在程序编译时会被链接到⽬标代码中，程序运⾏时可独立运行，将不再需要该静态库。
> 动态库：在程序编译时并不会被链接到⽬标代码中，⽽是在程序运⾏是才被载⼊，因此在程序运⾏时还需要动态库存在。

目录 demo/2.1 中文件结构如下所示；
```shell
├── CMakeLists.txt
├── fun.cc
├── fun.h
├── main.cc
├── test.cc
└── test.h
```

依赖关系为 ： main 函数调用 test 函数， test 函数调用 fun 函数。

cmake 代码如下：
```shell
project(demo_2.1)

add_library(fun fun.cc)

add_library(test SHARED test.cc)
target_link_libraries(test fun)

add_executable(main main.cc)
target_link_libraries(main test)
```
fun 为静态库， test 为动态库，并依赖 fun， 可执行文件 main 依赖 test 动态库。 

编译时会报错 ：

  /usr/local/bin/ld: libfun.a(fun.cc.o): relocation R_X86_64_32 against `.rodata' can not be used when making a shared object; recompile with -fPIC

正确的方式为： 如果你的静态库可能会被动态库使用，那么静态库编译的时候就也需要 `-fPIC` 选项。
即:
```shell
project(demo_2.1)

add_library(fun fun.cc)
target_compile_options(fun PRIVATE "-fPIC")

add_library(test SHARED test.cc)
target_link_libraries(test fun)

add_executable(main main.cc)
target_link_libraries(main test)

```
