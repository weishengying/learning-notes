
# 常用cmake操作
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
     
# 常见 cmake 问题
1. **动态库可以链接静态库吗**

