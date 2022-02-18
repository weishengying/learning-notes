# target_link_libraries 和 link_libraries的区别
## target_link_libraries : 
[官方文档](https://cmake.org/cmake/help/v3.10/command/target_link_libraries.html?highlight=target_link_libraries)

简介：
同一二进制（.out文件）在不同环境下执行的时候可能因为环境下具有的动态库不同而不能正常执行，可以用ldd（list, dynamic, dependencies的缩写， 意思是， 列出动态库依赖关系）命令看二进制依赖的动态库文件。
使用方法：ldd加 二进制文件可执行文件。
静态库和动态库共存时，cmake会默认先链接动态库，如果要强制使用静态库，在CMakeLists.txt中如此直接指明：
target_link_libraries(main ${CMAKE_SOURCE_DIR}/libbingitup.a)  #强制使用静态库
target_link_libraries(myProject comm)     # 连接libhello.so库，默认优先链接动态库
target_link_libraries(myProject libcomm.a)  # 显示指定链接静态库
target_link_libraries(myProject libcomm.so) # 显示指定链接动态库
target_link_libraries(myProject libcomm.so)　　#这些库名写法都可以。
target_link_libraries(myProject -lcomm) # 连接libhello.so库，默认优先链接动态库

CMAKE/gcc中库的链接顺序是**从右往左**进行，所以要把最基础实现的库放在最后，这样左边的lib就可以调用右边的lib中的代码。同时，当一个函数的实现代码在多个lib都存在时，最左边的lib代码最后link，所以也将最终保存下来。

target_link_libraries里库文件的顺序符合gcc链接顺序的规则，即被依赖的库放在依赖它的库的后面，比如target_link_libraries(hello A B.a C.so)。

   在上面的命令中，libA.so可能依赖于libB.a和libC.so，如果顺序有错，链接时会报错。还有一点，B.a会告诉CMake优先使用静态链接库libB.a，C.so会告诉CMake优先使用动态链接库libC.so，也可直接使用库文件的相对路径或绝对路径。使用绝对路径的好处在于，当依赖的库被更新时，make的时候也会重新链接。在链接命令中给出所依赖的库时，需要注意库之间的依赖顺序，依赖其它库的库一定要放到被依赖库的前面，这样才能真正避免undefined reference的错误。
