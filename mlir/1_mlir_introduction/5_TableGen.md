# MLIR Dialect的零件生产者 -- TableGen
    上一篇文章组建了MLIR的整个生产线，将MLIRGen模块看作是生产线的履带、将Dialect模块看作是生产线的机械臂、将TableGen模块看作是生产线的零件提供者。TableGen模块实际上是使用Operation Definition Specification (ODS)框架进行自动化生成代码，整个框架是基于TableGen规则来完成相应功能。

我们在组装完生产线之后发现，我们距离生产出MLIR只差“零件”了，那么“零件”从哪来呢？当然有一种方法就是手工打磨“零件”，也就是为每一个Dialect Operation手写类。但是这都2020年了，手工打磨的方式太过低效了，我们要使用自动化的“数控机床”--TableGen，定义一系列的TableGen规则，从而自动化地生产“零件”。

## 什么是TableGen
我们把TableGen看作是为Dialect提供零件的“数控机床”，那么它在MLIR体系中是如何发挥作用的呢？TableGen本身是一种声明性编程语言，在此处它用于描述MLIR中Operation的类的定义，在源代码中它以.td文件的形式存在，在编译时会自动生成C++的相应文件，给Dialect模块文件提供支持。

如果我们使用手动编写的方式，在针对不同编译目标时，我们需要在一系列不同文件中编写一些相同的代码，这就造成了冗余的开发。而使用TableGen，我们只需要修改.td文件即可实现批量修改，也就解决了上述问题。下面我们就来看看，在Toy语言程序例子中，.td文件是由什么组成的，而TableGen又是怎么发挥的作用。

### 1. 定义一个和Toy Dialect的链接

首先我们需要在.td文件中定义一个TableGen和Dialect的链接，它负责把在Dialect中定义的所有Operation整合起来：

```
// Provide a definition of the 'toy' dialect in the ODS framework so that we
// can define our operations.
def Toy_Dialect : Dialect {
  let name = "toy";
  let cppNamespace = "toy";
} 
```

### 2. 创建一个Toy Dialect Operation的基类

构造所有Dialect Operation的基类Toy_Op，所有的Operation类都将基于此类进行构造：

```
// Base class for toy dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Toy_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Toy_Dialect, mnemonic, traits>;

```
### 3. 创建Toy Dialect各种Operation的类

所有定义的Operation的类都继承自上述基类，以TransposeOp为例，使用TableGen的规则 定义参数、值、builder、verifier等元素：

```
def TransposeOp : Toy_Op<"transpose"> {
  let summary = "transpose operation";

  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor);

  // Allow building a TransposeOp with from the input operand.
  let builders = [
    OpBuilder<"Builder *b, OperationState &state, Value input">
  ];

  // Invoke a static verify method to verify this transpose operation.
  let verifier = [{ return ::verify(*this); }];
}

```

## TableGen生成C++代码
在编写完TableGen描述之后，我们可以使用mlir-tblgen工具来生成C++代码，操作步骤如下：

```shell
$ cd llvm-project/build
$ bin/mlir-tblgen -gen-op-defs ../mlir/examples/toy/Ch2/include/toy/Ops.td -I ../mlir/include/
```
执行命令后，在终端输出生成的C++代码，代码较长就不整个展示了，下面展示一下TransposeOp 的部分C++代码：

```cpp
//===----------------------------------------------------------------------===//
// toy::TransposeOp definitions
//===----------------------------------------------------------------------===//

......

StringRef TransposeOp::getOperationName() {
  return "toy.transpose";
}

......

void TransposeOp::build(Builder *tblgen_builder, OperationState &tblgen_state, Type resultType0, Value input) {
  tblgen_state.addOperands(input);
  tblgen_state.addTypes(resultType0);
}

......
```

在编译时，TableGen将会发挥作用，把.td文件生成为C++的文件，而上述生成的代码将给Dialect模块提供支持。

## 总结一下

![](../images/TableGen_1.png)

到此为止，MLIR生产线的组成部分MLIRGen模块、Dialect模块和TableGen模块，都进行了简单的介绍，下一步将会选择我所感兴趣的Open Project题目进行研究。

本文参考自MLIR官方文档Toy语言教程的章节2及其源代码，如有错误纰漏，欢迎大家批评指正。

