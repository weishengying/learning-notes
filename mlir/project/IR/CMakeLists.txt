add_mlir_dialect(infrt_base infrt)

set(LLVM_TARGET_DEFINITIONS infrt_base.td)
mlir_tablegen(infrt_opsAttributes.h.inc -gen-attrdef-decls -dialect=infrt)
mlir_tablegen(infrt_opsAttributes.cpp.inc -gen-attrdef-defs -dialect=infrt)
add_public_tablegen_target(MLIRinfrt_opsAttributesIncGen)
add_dependencies(mlir-headers MLIRinfrt_opsAttributesIncGen)