file(REMOVE_RECURSE
  "CMakeFiles/MLIRinfrt_baseIncGen"
  "infrt_base.cpp.inc"
  "infrt_base.h.inc"
  "infrt_baseDialect.cpp.inc"
  "infrt_baseDialect.h.inc"
  "infrt_baseTypes.cpp.inc"
  "infrt_baseTypes.h.inc"
  "infrt_opsAttributes.cpp.inc"
  "infrt_opsAttributes.h.inc"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/MLIRinfrt_baseIncGen.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
