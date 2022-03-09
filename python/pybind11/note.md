# pybind11使用入门
[官方文档](https://pybind11.readthedocs.io/en/latest/)

## 简介
pybind11 is a lightweight header-only library that exposes C++ types in Python and vice versa, mainly to create Python bindings of existing C++ code.
注意，只是一个header-only library，所以我们主需要头文件即可！！！

## 安装
前面说了，使用pybind11只需要一堆头文件，下面介绍两种方法获取头文件

### 方式一
第一种方式，当然是直接clone源码
```shell
git clone https://github.com/pybind/pybind11.git
```
克隆成功后进入文件夹 cd pybind11/include/pybind11 即可看到所需的头文件。


### 方式二
直接通过python3 pip安装
[参考文档](https://pybind11.readthedocs.io/en/latest/installing.html)
```shell
python3 -m pip install pybind11
```
安装成功后在 cd /usr/local/lib/python3.8/dist-packages/pybind11/include 目录下有所需要的pybind11头文件

## 使用
Creating bindings for a simple function
```cpp
#include <pybind11/pybind11.h>

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function that adds two numbers");
}
```
### 编译

1、如果通过python pip安装的pybind，编译命令如下
```shell
g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) example.cpp -o example$(python3-config --extension-suffix)
```

等同于
```shell
g++ -O3 -Wall -shared -std=c++11 -fPIC -I/usr/include/python3.8 -I/usr/local/lib/python3.8/dist-packages/pybind11/include example.cpp -o example.cpython-38-aarch64-linux-gnu.so
```
编译生成example.cpython-38-aarch64-linux-gnu.so
当然也可以直接
```shell
g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) example.cpp -o example.so
```
不需要那么多后缀，但是建议后缀，能提示用户这个so编译的python版本、架构、系统等信息。


2、如果是clone的源码
按下面指令编译
```shell
g++ -O3 -Wall -shared -std=c++11 -fPIC -I/usr/include/python3.8 -I./pybind11/include/ example.cpp -o example.cpython-38-aarch64-linux-gnu.so
```
和上面的不同仅仅是指定的头文件路径不同： -I/usr/local/lib/python3.8/dist-packages/pybind11/include   -I./pybind11/include
一个是python module的安装路径，一个是clone的代码路径。

### 使用
在有example.cpython-38-aarch64-linux-gnu.so文件的路径下（否则需要自己设置import 路径），打开python解释器
```shell
$ python3
Python 3.8.10 (default, Nov 26 2021, 20:14:08)
[GCC 9.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import example
>>> example.add(1, 2)
3
>>>
```