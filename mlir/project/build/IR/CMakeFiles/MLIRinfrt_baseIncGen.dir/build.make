# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/cmake-3.16.0-Linux-x86_64/bin/cmake

# The command to remove a file.
RM = /home/cmake-3.16.0-Linux-x86_64/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /weishengying/learning-notes/mlir/project

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /weishengying/learning-notes/mlir/project/build

# Utility rule file for MLIRinfrt_baseIncGen.

# Include the progress variables for this target.
include IR/CMakeFiles/MLIRinfrt_baseIncGen.dir/progress.make

IR/CMakeFiles/MLIRinfrt_baseIncGen: IR/infrt_base.h.inc
IR/CMakeFiles/MLIRinfrt_baseIncGen: IR/infrt_base.h.inc
IR/CMakeFiles/MLIRinfrt_baseIncGen: IR/infrt_base.cpp.inc
IR/CMakeFiles/MLIRinfrt_baseIncGen: IR/infrt_base.cpp.inc
IR/CMakeFiles/MLIRinfrt_baseIncGen: IR/infrt_baseTypes.h.inc
IR/CMakeFiles/MLIRinfrt_baseIncGen: IR/infrt_baseTypes.h.inc
IR/CMakeFiles/MLIRinfrt_baseIncGen: IR/infrt_baseTypes.cpp.inc
IR/CMakeFiles/MLIRinfrt_baseIncGen: IR/infrt_baseTypes.cpp.inc
IR/CMakeFiles/MLIRinfrt_baseIncGen: IR/infrt_baseDialect.h.inc
IR/CMakeFiles/MLIRinfrt_baseIncGen: IR/infrt_baseDialect.h.inc
IR/CMakeFiles/MLIRinfrt_baseIncGen: IR/infrt_baseDialect.cpp.inc
IR/CMakeFiles/MLIRinfrt_baseIncGen: IR/infrt_baseDialect.cpp.inc


IR/infrt_base.h.inc: ../third-party/install/llvm/bin/mlir-tblgen
IR/infrt_base.h.inc: ../IR/infrt_base.td
IR/infrt_base.h.inc: ../IR/infrt_ops.td
IR/infrt_base.h.inc: ../third-party/install/llvm/include/llvm/CodeGen/SDNodeProperties.td
IR/infrt_base.h.inc: ../third-party/install/llvm/include/llvm/CodeGen/ValueTypes.td
IR/infrt_base.h.inc: ../third-party/install/llvm/include/llvm/Frontend/Directive/DirectiveBase.td
IR/infrt_base.h.inc: ../third-party/install/llvm/include/llvm/Frontend/OpenACC/ACC.td
IR/infrt_base.h.inc: ../third-party/install/llvm/include/llvm/Frontend/OpenMP/OMP.td
IR/infrt_base.h.inc: ../third-party/install/llvm/include/llvm/IR/Attributes.td
IR/infrt_base.h.inc: ../third-party/install/llvm/include/llvm/IR/AttributesAMDGPU.td
IR/infrt_base.h.inc: ../third-party/install/llvm/include/llvm/IR/Intrinsics.td
IR/infrt_base.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsAArch64.td
IR/infrt_base.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsAMDGPU.td
IR/infrt_base.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsARM.td
IR/infrt_base.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsBPF.td
IR/infrt_base.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsHexagon.td
IR/infrt_base.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsHexagonDep.td
IR/infrt_base.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsMips.td
IR/infrt_base.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsNVVM.td
IR/infrt_base.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsPowerPC.td
IR/infrt_base.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsRISCV.td
IR/infrt_base.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsSystemZ.td
IR/infrt_base.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsVE.td
IR/infrt_base.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsVEVL.gen.td
IR/infrt_base.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsWebAssembly.td
IR/infrt_base.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsX86.td
IR/infrt_base.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsXCore.td
IR/infrt_base.h.inc: ../third-party/install/llvm/include/llvm/Option/OptParser.td
IR/infrt_base.h.inc: ../third-party/install/llvm/include/llvm/TableGen/Automaton.td
IR/infrt_base.h.inc: ../third-party/install/llvm/include/llvm/TableGen/SearchableTable.td
IR/infrt_base.h.inc: ../third-party/install/llvm/include/llvm/Target/GenericOpcodes.td
IR/infrt_base.h.inc: ../third-party/install/llvm/include/llvm/Target/GlobalISel/Combine.td
IR/infrt_base.h.inc: ../third-party/install/llvm/include/llvm/Target/GlobalISel/RegisterBank.td
IR/infrt_base.h.inc: ../third-party/install/llvm/include/llvm/Target/GlobalISel/SelectionDAGCompat.td
IR/infrt_base.h.inc: ../third-party/install/llvm/include/llvm/Target/GlobalISel/Target.td
IR/infrt_base.h.inc: ../third-party/install/llvm/include/llvm/Target/Target.td
IR/infrt_base.h.inc: ../third-party/install/llvm/include/llvm/Target/TargetCallingConv.td
IR/infrt_base.h.inc: ../third-party/install/llvm/include/llvm/Target/TargetInstrPredicate.td
IR/infrt_base.h.inc: ../third-party/install/llvm/include/llvm/Target/TargetItinerary.td
IR/infrt_base.h.inc: ../third-party/install/llvm/include/llvm/Target/TargetPfmCounters.td
IR/infrt_base.h.inc: ../third-party/install/llvm/include/llvm/Target/TargetSchedule.td
IR/infrt_base.h.inc: ../third-party/install/llvm/include/llvm/Target/TargetSelectionDAG.td
IR/infrt_base.h.inc: ../IR/infrt_base.td
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/weishengying/learning-notes/mlir/project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building infrt_base.h.inc..."
	cd /weishengying/learning-notes/mlir/project/build/IR && ../../third-party/install/llvm/bin/mlir-tblgen -gen-op-decls -I /weishengying/learning-notes/mlir/project/IR /weishengying/learning-notes/mlir/project/IR/infrt_base.td --write-if-changed -o /weishengying/learning-notes/mlir/project/build/IR/infrt_base.h.inc

IR/infrt_base.cpp.inc: ../third-party/install/llvm/bin/mlir-tblgen
IR/infrt_base.cpp.inc: ../IR/infrt_base.td
IR/infrt_base.cpp.inc: ../IR/infrt_ops.td
IR/infrt_base.cpp.inc: ../third-party/install/llvm/include/llvm/CodeGen/SDNodeProperties.td
IR/infrt_base.cpp.inc: ../third-party/install/llvm/include/llvm/CodeGen/ValueTypes.td
IR/infrt_base.cpp.inc: ../third-party/install/llvm/include/llvm/Frontend/Directive/DirectiveBase.td
IR/infrt_base.cpp.inc: ../third-party/install/llvm/include/llvm/Frontend/OpenACC/ACC.td
IR/infrt_base.cpp.inc: ../third-party/install/llvm/include/llvm/Frontend/OpenMP/OMP.td
IR/infrt_base.cpp.inc: ../third-party/install/llvm/include/llvm/IR/Attributes.td
IR/infrt_base.cpp.inc: ../third-party/install/llvm/include/llvm/IR/AttributesAMDGPU.td
IR/infrt_base.cpp.inc: ../third-party/install/llvm/include/llvm/IR/Intrinsics.td
IR/infrt_base.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsAArch64.td
IR/infrt_base.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsAMDGPU.td
IR/infrt_base.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsARM.td
IR/infrt_base.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsBPF.td
IR/infrt_base.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsHexagon.td
IR/infrt_base.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsHexagonDep.td
IR/infrt_base.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsMips.td
IR/infrt_base.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsNVVM.td
IR/infrt_base.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsPowerPC.td
IR/infrt_base.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsRISCV.td
IR/infrt_base.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsSystemZ.td
IR/infrt_base.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsVE.td
IR/infrt_base.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsVEVL.gen.td
IR/infrt_base.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsWebAssembly.td
IR/infrt_base.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsX86.td
IR/infrt_base.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsXCore.td
IR/infrt_base.cpp.inc: ../third-party/install/llvm/include/llvm/Option/OptParser.td
IR/infrt_base.cpp.inc: ../third-party/install/llvm/include/llvm/TableGen/Automaton.td
IR/infrt_base.cpp.inc: ../third-party/install/llvm/include/llvm/TableGen/SearchableTable.td
IR/infrt_base.cpp.inc: ../third-party/install/llvm/include/llvm/Target/GenericOpcodes.td
IR/infrt_base.cpp.inc: ../third-party/install/llvm/include/llvm/Target/GlobalISel/Combine.td
IR/infrt_base.cpp.inc: ../third-party/install/llvm/include/llvm/Target/GlobalISel/RegisterBank.td
IR/infrt_base.cpp.inc: ../third-party/install/llvm/include/llvm/Target/GlobalISel/SelectionDAGCompat.td
IR/infrt_base.cpp.inc: ../third-party/install/llvm/include/llvm/Target/GlobalISel/Target.td
IR/infrt_base.cpp.inc: ../third-party/install/llvm/include/llvm/Target/Target.td
IR/infrt_base.cpp.inc: ../third-party/install/llvm/include/llvm/Target/TargetCallingConv.td
IR/infrt_base.cpp.inc: ../third-party/install/llvm/include/llvm/Target/TargetInstrPredicate.td
IR/infrt_base.cpp.inc: ../third-party/install/llvm/include/llvm/Target/TargetItinerary.td
IR/infrt_base.cpp.inc: ../third-party/install/llvm/include/llvm/Target/TargetPfmCounters.td
IR/infrt_base.cpp.inc: ../third-party/install/llvm/include/llvm/Target/TargetSchedule.td
IR/infrt_base.cpp.inc: ../third-party/install/llvm/include/llvm/Target/TargetSelectionDAG.td
IR/infrt_base.cpp.inc: ../IR/infrt_base.td
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/weishengying/learning-notes/mlir/project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building infrt_base.cpp.inc..."
	cd /weishengying/learning-notes/mlir/project/build/IR && ../../third-party/install/llvm/bin/mlir-tblgen -gen-op-defs -I /weishengying/learning-notes/mlir/project/IR -I/weishengying/learning-notes/mlir/project/build/IR /weishengying/learning-notes/mlir/project/IR/infrt_base.td --write-if-changed -o /weishengying/learning-notes/mlir/project/build/IR/infrt_base.cpp.inc

IR/infrt_baseTypes.h.inc: ../third-party/install/llvm/bin/mlir-tblgen
IR/infrt_baseTypes.h.inc: ../IR/infrt_base.td
IR/infrt_baseTypes.h.inc: ../IR/infrt_ops.td
IR/infrt_baseTypes.h.inc: ../third-party/install/llvm/include/llvm/CodeGen/SDNodeProperties.td
IR/infrt_baseTypes.h.inc: ../third-party/install/llvm/include/llvm/CodeGen/ValueTypes.td
IR/infrt_baseTypes.h.inc: ../third-party/install/llvm/include/llvm/Frontend/Directive/DirectiveBase.td
IR/infrt_baseTypes.h.inc: ../third-party/install/llvm/include/llvm/Frontend/OpenACC/ACC.td
IR/infrt_baseTypes.h.inc: ../third-party/install/llvm/include/llvm/Frontend/OpenMP/OMP.td
IR/infrt_baseTypes.h.inc: ../third-party/install/llvm/include/llvm/IR/Attributes.td
IR/infrt_baseTypes.h.inc: ../third-party/install/llvm/include/llvm/IR/AttributesAMDGPU.td
IR/infrt_baseTypes.h.inc: ../third-party/install/llvm/include/llvm/IR/Intrinsics.td
IR/infrt_baseTypes.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsAArch64.td
IR/infrt_baseTypes.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsAMDGPU.td
IR/infrt_baseTypes.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsARM.td
IR/infrt_baseTypes.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsBPF.td
IR/infrt_baseTypes.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsHexagon.td
IR/infrt_baseTypes.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsHexagonDep.td
IR/infrt_baseTypes.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsMips.td
IR/infrt_baseTypes.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsNVVM.td
IR/infrt_baseTypes.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsPowerPC.td
IR/infrt_baseTypes.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsRISCV.td
IR/infrt_baseTypes.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsSystemZ.td
IR/infrt_baseTypes.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsVE.td
IR/infrt_baseTypes.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsVEVL.gen.td
IR/infrt_baseTypes.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsWebAssembly.td
IR/infrt_baseTypes.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsX86.td
IR/infrt_baseTypes.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsXCore.td
IR/infrt_baseTypes.h.inc: ../third-party/install/llvm/include/llvm/Option/OptParser.td
IR/infrt_baseTypes.h.inc: ../third-party/install/llvm/include/llvm/TableGen/Automaton.td
IR/infrt_baseTypes.h.inc: ../third-party/install/llvm/include/llvm/TableGen/SearchableTable.td
IR/infrt_baseTypes.h.inc: ../third-party/install/llvm/include/llvm/Target/GenericOpcodes.td
IR/infrt_baseTypes.h.inc: ../third-party/install/llvm/include/llvm/Target/GlobalISel/Combine.td
IR/infrt_baseTypes.h.inc: ../third-party/install/llvm/include/llvm/Target/GlobalISel/RegisterBank.td
IR/infrt_baseTypes.h.inc: ../third-party/install/llvm/include/llvm/Target/GlobalISel/SelectionDAGCompat.td
IR/infrt_baseTypes.h.inc: ../third-party/install/llvm/include/llvm/Target/GlobalISel/Target.td
IR/infrt_baseTypes.h.inc: ../third-party/install/llvm/include/llvm/Target/Target.td
IR/infrt_baseTypes.h.inc: ../third-party/install/llvm/include/llvm/Target/TargetCallingConv.td
IR/infrt_baseTypes.h.inc: ../third-party/install/llvm/include/llvm/Target/TargetInstrPredicate.td
IR/infrt_baseTypes.h.inc: ../third-party/install/llvm/include/llvm/Target/TargetItinerary.td
IR/infrt_baseTypes.h.inc: ../third-party/install/llvm/include/llvm/Target/TargetPfmCounters.td
IR/infrt_baseTypes.h.inc: ../third-party/install/llvm/include/llvm/Target/TargetSchedule.td
IR/infrt_baseTypes.h.inc: ../third-party/install/llvm/include/llvm/Target/TargetSelectionDAG.td
IR/infrt_baseTypes.h.inc: ../IR/infrt_base.td
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/weishengying/learning-notes/mlir/project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building infrt_baseTypes.h.inc..."
	cd /weishengying/learning-notes/mlir/project/build/IR && ../../third-party/install/llvm/bin/mlir-tblgen -gen-typedef-decls -typedefs-dialect=infrt -I /weishengying/learning-notes/mlir/project/IR -I/weishengying/learning-notes/mlir/project/build/IR -I/weishengying/learning-notes/mlir/project/build/IR /weishengying/learning-notes/mlir/project/IR/infrt_base.td --write-if-changed -o /weishengying/learning-notes/mlir/project/build/IR/infrt_baseTypes.h.inc

IR/infrt_baseTypes.cpp.inc: ../third-party/install/llvm/bin/mlir-tblgen
IR/infrt_baseTypes.cpp.inc: ../IR/infrt_base.td
IR/infrt_baseTypes.cpp.inc: ../IR/infrt_ops.td
IR/infrt_baseTypes.cpp.inc: ../third-party/install/llvm/include/llvm/CodeGen/SDNodeProperties.td
IR/infrt_baseTypes.cpp.inc: ../third-party/install/llvm/include/llvm/CodeGen/ValueTypes.td
IR/infrt_baseTypes.cpp.inc: ../third-party/install/llvm/include/llvm/Frontend/Directive/DirectiveBase.td
IR/infrt_baseTypes.cpp.inc: ../third-party/install/llvm/include/llvm/Frontend/OpenACC/ACC.td
IR/infrt_baseTypes.cpp.inc: ../third-party/install/llvm/include/llvm/Frontend/OpenMP/OMP.td
IR/infrt_baseTypes.cpp.inc: ../third-party/install/llvm/include/llvm/IR/Attributes.td
IR/infrt_baseTypes.cpp.inc: ../third-party/install/llvm/include/llvm/IR/AttributesAMDGPU.td
IR/infrt_baseTypes.cpp.inc: ../third-party/install/llvm/include/llvm/IR/Intrinsics.td
IR/infrt_baseTypes.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsAArch64.td
IR/infrt_baseTypes.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsAMDGPU.td
IR/infrt_baseTypes.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsARM.td
IR/infrt_baseTypes.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsBPF.td
IR/infrt_baseTypes.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsHexagon.td
IR/infrt_baseTypes.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsHexagonDep.td
IR/infrt_baseTypes.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsMips.td
IR/infrt_baseTypes.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsNVVM.td
IR/infrt_baseTypes.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsPowerPC.td
IR/infrt_baseTypes.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsRISCV.td
IR/infrt_baseTypes.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsSystemZ.td
IR/infrt_baseTypes.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsVE.td
IR/infrt_baseTypes.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsVEVL.gen.td
IR/infrt_baseTypes.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsWebAssembly.td
IR/infrt_baseTypes.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsX86.td
IR/infrt_baseTypes.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsXCore.td
IR/infrt_baseTypes.cpp.inc: ../third-party/install/llvm/include/llvm/Option/OptParser.td
IR/infrt_baseTypes.cpp.inc: ../third-party/install/llvm/include/llvm/TableGen/Automaton.td
IR/infrt_baseTypes.cpp.inc: ../third-party/install/llvm/include/llvm/TableGen/SearchableTable.td
IR/infrt_baseTypes.cpp.inc: ../third-party/install/llvm/include/llvm/Target/GenericOpcodes.td
IR/infrt_baseTypes.cpp.inc: ../third-party/install/llvm/include/llvm/Target/GlobalISel/Combine.td
IR/infrt_baseTypes.cpp.inc: ../third-party/install/llvm/include/llvm/Target/GlobalISel/RegisterBank.td
IR/infrt_baseTypes.cpp.inc: ../third-party/install/llvm/include/llvm/Target/GlobalISel/SelectionDAGCompat.td
IR/infrt_baseTypes.cpp.inc: ../third-party/install/llvm/include/llvm/Target/GlobalISel/Target.td
IR/infrt_baseTypes.cpp.inc: ../third-party/install/llvm/include/llvm/Target/Target.td
IR/infrt_baseTypes.cpp.inc: ../third-party/install/llvm/include/llvm/Target/TargetCallingConv.td
IR/infrt_baseTypes.cpp.inc: ../third-party/install/llvm/include/llvm/Target/TargetInstrPredicate.td
IR/infrt_baseTypes.cpp.inc: ../third-party/install/llvm/include/llvm/Target/TargetItinerary.td
IR/infrt_baseTypes.cpp.inc: ../third-party/install/llvm/include/llvm/Target/TargetPfmCounters.td
IR/infrt_baseTypes.cpp.inc: ../third-party/install/llvm/include/llvm/Target/TargetSchedule.td
IR/infrt_baseTypes.cpp.inc: ../third-party/install/llvm/include/llvm/Target/TargetSelectionDAG.td
IR/infrt_baseTypes.cpp.inc: ../IR/infrt_base.td
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/weishengying/learning-notes/mlir/project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building infrt_baseTypes.cpp.inc..."
	cd /weishengying/learning-notes/mlir/project/build/IR && ../../third-party/install/llvm/bin/mlir-tblgen -gen-typedef-defs -typedefs-dialect=infrt -I /weishengying/learning-notes/mlir/project/IR -I/weishengying/learning-notes/mlir/project/build/IR -I/weishengying/learning-notes/mlir/project/build/IR -I/weishengying/learning-notes/mlir/project/build/IR /weishengying/learning-notes/mlir/project/IR/infrt_base.td --write-if-changed -o /weishengying/learning-notes/mlir/project/build/IR/infrt_baseTypes.cpp.inc

IR/infrt_baseDialect.h.inc: ../third-party/install/llvm/bin/mlir-tblgen
IR/infrt_baseDialect.h.inc: ../IR/infrt_base.td
IR/infrt_baseDialect.h.inc: ../IR/infrt_ops.td
IR/infrt_baseDialect.h.inc: ../third-party/install/llvm/include/llvm/CodeGen/SDNodeProperties.td
IR/infrt_baseDialect.h.inc: ../third-party/install/llvm/include/llvm/CodeGen/ValueTypes.td
IR/infrt_baseDialect.h.inc: ../third-party/install/llvm/include/llvm/Frontend/Directive/DirectiveBase.td
IR/infrt_baseDialect.h.inc: ../third-party/install/llvm/include/llvm/Frontend/OpenACC/ACC.td
IR/infrt_baseDialect.h.inc: ../third-party/install/llvm/include/llvm/Frontend/OpenMP/OMP.td
IR/infrt_baseDialect.h.inc: ../third-party/install/llvm/include/llvm/IR/Attributes.td
IR/infrt_baseDialect.h.inc: ../third-party/install/llvm/include/llvm/IR/AttributesAMDGPU.td
IR/infrt_baseDialect.h.inc: ../third-party/install/llvm/include/llvm/IR/Intrinsics.td
IR/infrt_baseDialect.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsAArch64.td
IR/infrt_baseDialect.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsAMDGPU.td
IR/infrt_baseDialect.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsARM.td
IR/infrt_baseDialect.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsBPF.td
IR/infrt_baseDialect.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsHexagon.td
IR/infrt_baseDialect.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsHexagonDep.td
IR/infrt_baseDialect.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsMips.td
IR/infrt_baseDialect.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsNVVM.td
IR/infrt_baseDialect.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsPowerPC.td
IR/infrt_baseDialect.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsRISCV.td
IR/infrt_baseDialect.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsSystemZ.td
IR/infrt_baseDialect.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsVE.td
IR/infrt_baseDialect.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsVEVL.gen.td
IR/infrt_baseDialect.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsWebAssembly.td
IR/infrt_baseDialect.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsX86.td
IR/infrt_baseDialect.h.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsXCore.td
IR/infrt_baseDialect.h.inc: ../third-party/install/llvm/include/llvm/Option/OptParser.td
IR/infrt_baseDialect.h.inc: ../third-party/install/llvm/include/llvm/TableGen/Automaton.td
IR/infrt_baseDialect.h.inc: ../third-party/install/llvm/include/llvm/TableGen/SearchableTable.td
IR/infrt_baseDialect.h.inc: ../third-party/install/llvm/include/llvm/Target/GenericOpcodes.td
IR/infrt_baseDialect.h.inc: ../third-party/install/llvm/include/llvm/Target/GlobalISel/Combine.td
IR/infrt_baseDialect.h.inc: ../third-party/install/llvm/include/llvm/Target/GlobalISel/RegisterBank.td
IR/infrt_baseDialect.h.inc: ../third-party/install/llvm/include/llvm/Target/GlobalISel/SelectionDAGCompat.td
IR/infrt_baseDialect.h.inc: ../third-party/install/llvm/include/llvm/Target/GlobalISel/Target.td
IR/infrt_baseDialect.h.inc: ../third-party/install/llvm/include/llvm/Target/Target.td
IR/infrt_baseDialect.h.inc: ../third-party/install/llvm/include/llvm/Target/TargetCallingConv.td
IR/infrt_baseDialect.h.inc: ../third-party/install/llvm/include/llvm/Target/TargetInstrPredicate.td
IR/infrt_baseDialect.h.inc: ../third-party/install/llvm/include/llvm/Target/TargetItinerary.td
IR/infrt_baseDialect.h.inc: ../third-party/install/llvm/include/llvm/Target/TargetPfmCounters.td
IR/infrt_baseDialect.h.inc: ../third-party/install/llvm/include/llvm/Target/TargetSchedule.td
IR/infrt_baseDialect.h.inc: ../third-party/install/llvm/include/llvm/Target/TargetSelectionDAG.td
IR/infrt_baseDialect.h.inc: ../IR/infrt_base.td
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/weishengying/learning-notes/mlir/project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building infrt_baseDialect.h.inc..."
	cd /weishengying/learning-notes/mlir/project/build/IR && ../../third-party/install/llvm/bin/mlir-tblgen -gen-dialect-decls -dialect=infrt -I /weishengying/learning-notes/mlir/project/IR -I/weishengying/learning-notes/mlir/project/build/IR -I/weishengying/learning-notes/mlir/project/build/IR -I/weishengying/learning-notes/mlir/project/build/IR -I/weishengying/learning-notes/mlir/project/build/IR /weishengying/learning-notes/mlir/project/IR/infrt_base.td --write-if-changed -o /weishengying/learning-notes/mlir/project/build/IR/infrt_baseDialect.h.inc

IR/infrt_baseDialect.cpp.inc: ../third-party/install/llvm/bin/mlir-tblgen
IR/infrt_baseDialect.cpp.inc: ../IR/infrt_base.td
IR/infrt_baseDialect.cpp.inc: ../IR/infrt_ops.td
IR/infrt_baseDialect.cpp.inc: ../third-party/install/llvm/include/llvm/CodeGen/SDNodeProperties.td
IR/infrt_baseDialect.cpp.inc: ../third-party/install/llvm/include/llvm/CodeGen/ValueTypes.td
IR/infrt_baseDialect.cpp.inc: ../third-party/install/llvm/include/llvm/Frontend/Directive/DirectiveBase.td
IR/infrt_baseDialect.cpp.inc: ../third-party/install/llvm/include/llvm/Frontend/OpenACC/ACC.td
IR/infrt_baseDialect.cpp.inc: ../third-party/install/llvm/include/llvm/Frontend/OpenMP/OMP.td
IR/infrt_baseDialect.cpp.inc: ../third-party/install/llvm/include/llvm/IR/Attributes.td
IR/infrt_baseDialect.cpp.inc: ../third-party/install/llvm/include/llvm/IR/AttributesAMDGPU.td
IR/infrt_baseDialect.cpp.inc: ../third-party/install/llvm/include/llvm/IR/Intrinsics.td
IR/infrt_baseDialect.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsAArch64.td
IR/infrt_baseDialect.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsAMDGPU.td
IR/infrt_baseDialect.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsARM.td
IR/infrt_baseDialect.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsBPF.td
IR/infrt_baseDialect.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsHexagon.td
IR/infrt_baseDialect.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsHexagonDep.td
IR/infrt_baseDialect.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsMips.td
IR/infrt_baseDialect.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsNVVM.td
IR/infrt_baseDialect.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsPowerPC.td
IR/infrt_baseDialect.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsRISCV.td
IR/infrt_baseDialect.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsSystemZ.td
IR/infrt_baseDialect.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsVE.td
IR/infrt_baseDialect.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsVEVL.gen.td
IR/infrt_baseDialect.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsWebAssembly.td
IR/infrt_baseDialect.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsX86.td
IR/infrt_baseDialect.cpp.inc: ../third-party/install/llvm/include/llvm/IR/IntrinsicsXCore.td
IR/infrt_baseDialect.cpp.inc: ../third-party/install/llvm/include/llvm/Option/OptParser.td
IR/infrt_baseDialect.cpp.inc: ../third-party/install/llvm/include/llvm/TableGen/Automaton.td
IR/infrt_baseDialect.cpp.inc: ../third-party/install/llvm/include/llvm/TableGen/SearchableTable.td
IR/infrt_baseDialect.cpp.inc: ../third-party/install/llvm/include/llvm/Target/GenericOpcodes.td
IR/infrt_baseDialect.cpp.inc: ../third-party/install/llvm/include/llvm/Target/GlobalISel/Combine.td
IR/infrt_baseDialect.cpp.inc: ../third-party/install/llvm/include/llvm/Target/GlobalISel/RegisterBank.td
IR/infrt_baseDialect.cpp.inc: ../third-party/install/llvm/include/llvm/Target/GlobalISel/SelectionDAGCompat.td
IR/infrt_baseDialect.cpp.inc: ../third-party/install/llvm/include/llvm/Target/GlobalISel/Target.td
IR/infrt_baseDialect.cpp.inc: ../third-party/install/llvm/include/llvm/Target/Target.td
IR/infrt_baseDialect.cpp.inc: ../third-party/install/llvm/include/llvm/Target/TargetCallingConv.td
IR/infrt_baseDialect.cpp.inc: ../third-party/install/llvm/include/llvm/Target/TargetInstrPredicate.td
IR/infrt_baseDialect.cpp.inc: ../third-party/install/llvm/include/llvm/Target/TargetItinerary.td
IR/infrt_baseDialect.cpp.inc: ../third-party/install/llvm/include/llvm/Target/TargetPfmCounters.td
IR/infrt_baseDialect.cpp.inc: ../third-party/install/llvm/include/llvm/Target/TargetSchedule.td
IR/infrt_baseDialect.cpp.inc: ../third-party/install/llvm/include/llvm/Target/TargetSelectionDAG.td
IR/infrt_baseDialect.cpp.inc: ../IR/infrt_base.td
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/weishengying/learning-notes/mlir/project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building infrt_baseDialect.cpp.inc..."
	cd /weishengying/learning-notes/mlir/project/build/IR && ../../third-party/install/llvm/bin/mlir-tblgen -gen-dialect-defs -dialect=infrt -I /weishengying/learning-notes/mlir/project/IR -I/weishengying/learning-notes/mlir/project/build/IR -I/weishengying/learning-notes/mlir/project/build/IR -I/weishengying/learning-notes/mlir/project/build/IR -I/weishengying/learning-notes/mlir/project/build/IR -I/weishengying/learning-notes/mlir/project/build/IR /weishengying/learning-notes/mlir/project/IR/infrt_base.td --write-if-changed -o /weishengying/learning-notes/mlir/project/build/IR/infrt_baseDialect.cpp.inc

MLIRinfrt_baseIncGen: IR/CMakeFiles/MLIRinfrt_baseIncGen
MLIRinfrt_baseIncGen: IR/infrt_base.h.inc
MLIRinfrt_baseIncGen: IR/infrt_base.cpp.inc
MLIRinfrt_baseIncGen: IR/infrt_baseTypes.h.inc
MLIRinfrt_baseIncGen: IR/infrt_baseTypes.cpp.inc
MLIRinfrt_baseIncGen: IR/infrt_baseDialect.h.inc
MLIRinfrt_baseIncGen: IR/infrt_baseDialect.cpp.inc
MLIRinfrt_baseIncGen: IR/CMakeFiles/MLIRinfrt_baseIncGen.dir/build.make

.PHONY : MLIRinfrt_baseIncGen

# Rule to build all files generated by this target.
IR/CMakeFiles/MLIRinfrt_baseIncGen.dir/build: MLIRinfrt_baseIncGen

.PHONY : IR/CMakeFiles/MLIRinfrt_baseIncGen.dir/build

IR/CMakeFiles/MLIRinfrt_baseIncGen.dir/clean:
	cd /weishengying/learning-notes/mlir/project/build/IR && $(CMAKE_COMMAND) -P CMakeFiles/MLIRinfrt_baseIncGen.dir/cmake_clean.cmake
.PHONY : IR/CMakeFiles/MLIRinfrt_baseIncGen.dir/clean

IR/CMakeFiles/MLIRinfrt_baseIncGen.dir/depend:
	cd /weishengying/learning-notes/mlir/project/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /weishengying/learning-notes/mlir/project /weishengying/learning-notes/mlir/project/IR /weishengying/learning-notes/mlir/project/build /weishengying/learning-notes/mlir/project/build/IR /weishengying/learning-notes/mlir/project/build/IR/CMakeFiles/MLIRinfrt_baseIncGen.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : IR/CMakeFiles/MLIRinfrt_baseIncGen.dir/depend

