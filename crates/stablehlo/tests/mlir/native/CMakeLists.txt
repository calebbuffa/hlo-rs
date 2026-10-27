add_llvm_executable(stablehlo-verify main.cpp)
llvm_update_compile_flags(stablehlo-verify)
target_include_directories(stablehlo-verify PRIVATE ${MLIR_INCLUDE_DIRS})
target_link_libraries(stablehlo-verify PRIVATE
  MLIROptLib
  StablehloOps
  MLIRFuncDialect
  MLIRQuantDialect
  MLIRSparseTensorDialect
)
mlir_check_all_link_libraries(stablehlo-verify)
