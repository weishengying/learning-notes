# 报错fatal error: Python.h: No such file or directory
解决：安装python-dev
[参考](https://stackoverflow.com/questions/21530577/fatal-error-python-h-no-such-file-or-directory)
如： apt-get install python3-dev

# python安装pip
[参考](https://www.runoob.com/w3cnote/python-pip-install-usage.html)
```shell
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py   # 下载安装脚本
sudo python3 get-pip.py    # 运行安装脚本
```