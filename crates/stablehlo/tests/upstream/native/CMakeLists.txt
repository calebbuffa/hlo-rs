cmake_minimum_required(VERSION 3.20)

include(AddLLVM)

find_package(Git REQUIRED)
execute_process(
  COMMAND "${GIT_EXECUTABLE}" --no-pager rev-parse HEAD
  WORKING_DIRECTORY "${LLVM_EXTERNAL_STABLEHLO_SOURCE_DIR}"
  OUTPUT_VARIABLE stablehlo_revision
  OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY
)
if(NOT stablehlo_revision STREQUAL "25605c9f48a9aec305876d60d26b028398d298d2")
  message(FATAL_ERROR "The validator requires the pinned StableHLO checkout")
endif()
execute_process(
  COMMAND "${GIT_EXECUTABLE}" --no-pager diff --quiet HEAD --
  WORKING_DIRECTORY "${LLVM_EXTERNAL_STABLEHLO_SOURCE_DIR}"
  COMMAND_ERROR_IS_FATAL ANY
)
file(READ "${LLVM_EXTERNAL_STABLEHLO_SOURCE_DIR}/build_tools/llvm_version.txt" llvm_revision)
string(STRIP "${llvm_revision}" llvm_revision)
if(NOT llvm_revision STREQUAL "2b35fb9283a44aae92cb72a41d077ee8de2be7ad")
  message(FATAL_ERROR "The validator requires the pinned LLVM dependency")
endif()

if(MSVC)
  # The shared MLIR PCH includes cmath before StableHLO enables M_PI.
  set_source_files_properties(
    "${LLVM_EXTERNAL_STABLEHLO_SOURCE_DIR}/stablehlo/transforms/ChloLegalizeToStablehlo.cpp"
    TARGET_DIRECTORY obj.StablehloPasses
    PROPERTIES SKIP_PRECOMPILE_HEADERS ON
  )
endif()

add_llvm_executable(stablehlo-rust-validator validator.cpp)
target_compile_features(stablehlo-rust-validator PRIVATE cxx_std_17)
target_include_directories(stablehlo-rust-validator PRIVATE
  "${LLVM_MAIN_SRC_DIR}/../mlir/include"
  "${LLVM_BINARY_DIR}/tools/mlir/include"
  "${LLVM_EXTERNAL_STABLEHLO_SOURCE_DIR}"
  "${LLVM_BINARY_DIR}/tools/stablehlo"
)
target_link_libraries(stablehlo-rust-validator PRIVATE
  MLIRArithDialect
  MLIRComplexDialect
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRQuantDialect
  MLIRShapeDialect
  MLIRSparseTensorDialect
  MLIRTensorDialect
  StablehloOps
  StablehloSerialization
  VhloOps
)
