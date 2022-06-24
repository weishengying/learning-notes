[参考文档]https://github.com/protocolbuffers/protobuf/blob/master/src/README.md) 

# 源码编译
```shell
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf
git submodule update --init --recursive
./autogen.sh

./configure
make -j$(nproc) # $(nproc) ensures it uses all cores for compilation
make check
sudo make install
sudo ldconfig # refresh shared library cache.
```

# C++简单使用
[参考教程](https://developers.google.com/protocol-buffers/docs/cpptutorial)

## Defining Your Protocol Format

```protobuf
syntax = "proto3";

package tutorial;

message Person {
  optional string name = 1;
  optional int32 id = 2;
  optional string email = 3;

  enum PhoneType {
    MOBILE = 0;
    HOME = 1;
    WORK = 2;
  }

  message PhoneNumber {
    optional string number = 1;
    optional PhoneType type = 2 [default = HOME];
  }

  repeated PhoneNumber phones = 4;
}

message AddressBook {
  repeated Person people = 1;
}
```

## Compiling Your Protocol Buffers

```shell
protoc    -I=$SRC_DIR    --cpp_out=$DST_DIR     $SRC_DIR/addressbook.proto
```

## The Protocol Buffer API
参考官方文档
[文档](https://developers.google.com/protocol-buffers/docs/cpptutorial#parsing-and-serialization)

## Writing A Message
参考官方文档

## Reading A Message
参考官方文档

# 总结
protobuf提供一个自定义储存格式的工具。通过.proto文件定义数据存储方式，然后利用protoc工具生成读写定义数据的接口！