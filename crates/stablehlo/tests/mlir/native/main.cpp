#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "stablehlo/dialect/StablehloOps.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::quant::QuantDialect,
                  mlir::sparse_tensor::SparseTensorDialect,
                  mlir::stablehlo::StablehloDialect>();
  return mlir::failed(mlir::MlirOptMain(
      argc, argv, "Pinned StableHLO parser and verifier\n", registry));
}
