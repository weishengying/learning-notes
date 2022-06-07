# 常见问题
1、输出重定向
```shell
2>&1
```

2、 运行时遇到库找不到的错误
如：
  python3: error while loading shared libraries: libpython3.7m.so.1.0: cannot open shared object file: No such file or directory.

这是因为：

编译安装完成后，没有将 python/lib 下的文件放入默认库 /usr/lib 或 /lib 中，导致初始化时无法加载库文件。

修改 /etc/ld.so.conf.d 目录下的 libc.conf 或者 python3.conf 即可，将/usr/python/lib 作为一行插入，保存退出。然后运行 ldconfig 命令。

linux 下的共享库机制采用了类似于高速缓存的机制，将库信息保存在 /etc/ld.so.cache 里边，程序连接的时候首先从这个文件里边查找，然后再到 ld.so.conf 的路径里边去详细找，这就是为什么修改了 conf 文件要重新运行一下 ldconfig 的原因。

# vim 
## vim配置文件设置
[参考教程](https://www.cnblogs.com/wenxingxu/p/9510796.html)
vim全局配置文件：/etc/vim/vimrc
当前用户配置文件： ～/.vimrc
常用配置如下：
" add tab space
set ts=2
set softtabstop=2
set shiftwidth=2
set expandtab
set autoindent
set showmatch
set nu
syntax on
 
# tar命令
[参考博客](https://www.cnblogs.com/ftl1012/p/9255795.html)
tar命令主要参数
-A 新增压缩文件到已存在的压缩
-c 建立新的压缩文件
-d 记录文件的差别
-r 添加文件到已经压缩的文件
-u 添加改变了和现有的文件到已经存在的压缩文件
-x 从压缩的文件中提取文件
-t 显示压缩文件的内容
-z 支持gzip压缩/解压文件
-j 支持bzip2压缩/解压文件
-Z 支持compress解压文件
-v 显示操作过程
-l 文件系统边界设置
-k 保留原有文件不覆盖
-m 保留文件不被覆盖
-W 确认压缩文件的正确性
-f 需要解压或者生成的压缩包文件名

## 压缩
tar cf hhh.tar hhh         # 仅仅打包
tar jcf hhh.tar.bz2 hhh    # 压缩打包
tar czf hhh.tar.gz hhh     # 压缩打包
zip hhh.zip.gz hhh         # 压缩打包
gzip messages              # 仅压缩文件【默认删除源文件】

## 解压
tar xf hhh.tar hhh
tar jxf hhh.tar.bz2 hhh
tar xzf hhh.tar.gz hhh
unzip hhh.zip hhh

## tar包和tar.gz包的区别
.tar只是将文件打包，文件的大小没什么变化，一般用tar -cvf filename.tar filename格式；.tar.gz是加入了gzip的压缩命令，会将文件压缩存放，可以有效压缩文件的大小，以便于缩短传输时间或者释放磁盘空间，一般用tar -czvf filename.tar.gz filename。同样的解包的时候使用 tar -xvf filename.tar和tar -xzvf filename.tar.gz。

# 创建软连接
ln -s 源文件 目标文件 （目标文件是即将创建的新文件）