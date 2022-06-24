
# 1.简单 cmake 操作

**安装 cmake**
[cmake仓库代码]https://gitlab.kitware.com/cmake/cmake
[cmake官方release](https://github.com/Kitware/CMake/releases)

## 1.1 查看所有的target
```shell
cmake --build . --target help  
make help
```

## 1.2 cmake过程中提示找不多pythonlib

[参考](https://stackoverflow.com/questions/24174394/cmake-is-not-able-to-find-python-libraries)

手动设置下面两个 cmake 变量即可， 或者修改 cmake find_package 指定的搜索路径。
```shell
-DPYTHON_INCLUDE_DIR=$(python3.7 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")  \
-DPYTHON_LIBRARY=$(python3.7 -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR')")
```
     
## 1.3 cmake指定编译器
1. 使用环境变量
```shell
CC=/usr/bin/gcc-4.2 CXX=/usr/bin/g++-4.2
```
2. use cmake -D
推荐使用这种方式
```shell
cmake -G "Your Generator" -DCMAKE_CXX_COMPILER=/usr/local/gcc-8.2/bin/g++  -DCMAKE_C_COMPILER=/usr/local/gcc-8.2/bin/gcc path/to/your/source
```

## 1.4 cmake生成依赖关系图
```shell
# 先安装 dot 工具
sudo apt install graphviz

# 生成 cmake 依赖关系 dot 图
cd build
cmake --graphviz=foo.dot ..

# 根据 dot 图生成图片
dot -Tpng foo.dot -o foo.png
```

# 2.编译过程中的常见报错
## 2.1 cc1: all warnings being treated as errors**
在 CMakeCache.txt 中设置
```shell
# Flags used by the CXX compiler during all build types.
CMAKE_CXX_FLAGS:STRING=-Wno-error

# Flags used by the C compiler during all build types.
CMAKE_C_FLAGS:STRING=-Wno-error
```
注： 

-Werror ：所有警告当错误报。 

-Werror=  将指定的警告转换为错误。

反过来：

-Wno-error取消编译选项-Werror


## 2.2 编译提示[-Werror=maybe-unintialized]报错
在 CMakeCache.txt 中设置
```shell
CMAKE_CXX_FLAGS:STRING=-Wno-error -Wno-error=maybe-unintialized
CMAKE_C_FLAGS:STRING=-Wno-error -Wno-error=maybe-unintialized
```

或 cmake 之前设置环境变量 
```shell
export CFLAGS="-Wno-error"
export CXXFLAGS="-Wno-error"
```

# 3.常见 cmake 操作
## 3.1 动态库可以链接静态库吗

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

## 3.2 add_dependencies 的应用
> 注意： add_dependencies 仅仅保证 target 之间 build 的相对顺序， 并不会形成 `依赖关系`！

[cmake add_dependencies](https://cmake.org/cmake/help/v3.16/command/add_dependencies.html?highlight=add_dependencie#command:add_dependencies)

## 3.3 依赖的传递
以 2.1 中的 demo 为例， cmake 改为如下：
```cpp
project(demo_2.1)
cmake_minimum_required(VERSION 3.16)

add_library(fun SHARED fun.cc)

add_library(test SHARED test.cc)

target_link_libraries(test fun)

add_executable(main main.cc)

target_link_libraries(main test)
```

可执行文件 main 链接到 test 库， test 库链接到 fun 库， 那么在 main 中可以直接使用 fun 函数吗。

答案是可以的， 从 build/CMakeFiles/main.dir/link.txt， 可以查看可执行文件 main 的链接关系：
```shell
/usr/bin/c++     CMakeFiles/main.dir/main.cc.o  -o main  -Wl,-rpath,/weishengying/learning-notes/cmake/demo/2.3/build libtest.so libfun.so 
```

可以看出 main 同时链接到了 libtest.so libfun.so 这两个库， 这是因为链接的传递性。

test ---> fun,  然后 main --> test, 由于传递性， test 同时 --> fun。

可以通过 `LINK_PRIVATE` 关键字， 阻止链接关系的传递性。
```shell
target_link_libraries(test LINK_PRIVATE fun)
```
这样在 main 中就无法使用 fun 函数。
```shell
/usr/local/bin/ld: CMakeFiles/main.dir/main.cc.o: undefined reference to symbol '_Z3funv'
```

## 3.4 动态库符号的 global， local 属性
在上述的 main test fun 示例中， main 中调用 test 函数， test 中调用 fun 函数， main 中也可以直接调用 fun 函数。
现在只想让用户在 main 中调用 test 函数， 禁止用户在 main 中调用 fun 函数， 有什么办法呢。

方法如下：我们在 fun.so中 将 fun 函数符号设为 local， 这样用户就无法使用了。

```cpp
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
```

这样在 main 中使用 fun 函数的话， 编译会报错：
```shell
/usr/local/bin/ld: CMakeFiles/main.dir/main.cc.o: in function `main':
main.cc:(.text+0x5): undefined reference to `fun()'
```

其原理是通过 
```cpp
set(LINK_FLAGS
        "-Wl,--version-script ${CMAKE_CURRENT_SOURCE_DIR}/a.map")
set_target_properties(test PROPERTIES LINK_FLAGS "${LINK_FLAGS}")
set_target_properties(fun PROPERTIES LINK_FLAGS "${LINK_FLAGS}")
```
和 a.map
```shell
{
	global:
		*test*;
	local:
		*;
};
```
来控制 libtest.so 和 libfun.so 中函数符号的可见性。

```shell
nm -C libtest.so 
0000000000004028 b completed.7344
                 w __cxa_finalize@@GLIBC_2.2.5
0000000000001050 t deregister_tm_clones
00000000000010c0 t __do_global_dtors_aux
0000000000003da8 d __do_global_dtors_aux_fini_array_entry
0000000000004020 d __dso_handle
0000000000003db0 d _DYNAMIC
0000000000001114 t _fini
0000000000001100 t frame_dummy
0000000000003da0 d __frame_dummy_init_array_entry
00000000000020a0 r __FRAME_END__
0000000000004000 d _GLOBAL_OFFSET_TABLE_
                 w __gmon_start__
0000000000002000 r __GNU_EH_FRAME_HDR
0000000000001000 t _init
                 w _ITM_deregisterTMCloneTable
                 w _ITM_registerTMCloneTable
0000000000001080 t register_tm_clones
0000000000004028 d __TMC_END__
                 U fun()

nm -C libfun.so 
0000000000004040 b completed.7344
                 U __cxa_atexit@@GLIBC_2.2.5
                 w __cxa_finalize@@GLIBC_2.2.5
0000000000001080 t deregister_tm_clones
00000000000010f0 t __do_global_dtors_aux
0000000000003db0 d __do_global_dtors_aux_fini_array_entry
0000000000004038 d __dso_handle
0000000000003db8 d _DYNAMIC
00000000000011c8 t _fini
0000000000001130 t frame_dummy
0000000000003da0 d __frame_dummy_init_array_entry
0000000000002100 r __FRAME_END__
0000000000004000 d _GLOBAL_OFFSET_TABLE_
00000000000011b0 t _GLOBAL__sub_I_fun.cc
                 w __gmon_start__
0000000000002010 r __GNU_EH_FRAME_HDR
0000000000001000 t _init
                 w _ITM_deregisterTMCloneTable
                 w _ITM_registerTMCloneTable
00000000000010b0 t register_tm_clones
0000000000004040 d __TMC_END__
0000000000001135 t fun()
```
> 一个`T`, 一个`t`;

[nm指令参考](https://man7.org/linux/man-pages/man1/nm.1.html)

> The symbol type.  At least the following types are used; others are, as well, depending on the object file format.  If
  lowercase, the symbol is usually local; if uppercase, the symbol is global (external).  There are however a few lowercase symbols that are shown for special global symbols ("u", "v" and "w").