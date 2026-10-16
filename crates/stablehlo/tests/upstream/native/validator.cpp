#include <iostream>
#include <string>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "stablehlo/dialect/Serialization.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/Version.h"
#include "stablehlo/dialect/VhloOps.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

namespace {
constexpr llvm::StringLiteral kStablehloRevision =
    "25605c9f48a9aec305876d60d26b028398d298d2";
constexpr llvm::StringLiteral kLlvmRevision =
    "2b35fb9283a44aae92cb72a41d077ee8de2be7ad";

int hexDigit(char value) {
  if (value >= '0' && value <= '9')
    return value - '0';
  if (value >= 'a' && value <= 'f')
    return value - 'a' + 10;
  if (value >= 'A' && value <= 'F')
    return value - 'A' + 10;
  return -1;
}

bool decode(llvm::StringRef hex, std::string &bytes) {
  if (hex.size() % 2 != 0)
    return false;
  bytes.clear();
  bytes.reserve(hex.size() / 2);
  for (size_t i = 0; i < hex.size(); i += 2) {
    int high = hexDigit(hex[i]);
    int low = hexDigit(hex[i + 1]);
    if (high < 0 || low < 0)
      return false;
    bytes.push_back(static_cast<char>((high << 4) | low));
  }
  return true;
}

std::string encode(llvm::StringRef bytes) {
  constexpr char digits[] = "0123456789abcdef";
  std::string hex;
  hex.reserve(bytes.size() * 2);
  for (unsigned char byte : bytes) {
    hex.push_back(digits[byte >> 4]);
    hex.push_back(digits[byte & 15]);
  }
  return hex;
}

void printVersion() {
  llvm::outs() << "VERSION\tnative\t"
               << mlir::vhlo::Version::getCurrentVersion().toString() << '\t'
               << kStablehloRevision << '\t' << kLlvmRevision << '\n';
  llvm::outs().flush();
}

void printDiagnostic(const mlir::Diagnostic &diagnostic,
                     llvm::raw_ostream &stream) {
  diagnostic.getLocation().print(stream);
  stream << ": ";
  diagnostic.print(stream);
  stream << '\n';
  for (const auto &note : diagnostic.getNotes())
    printDiagnostic(note, stream);
}
} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  llvm::StringRef mode = argc == 2 ? argv[1] : "";
  if (mode == "--version") {
    printVersion();
    return 0;
  }
  if (mode != "--verify" && mode != "--verify-generic" &&
      mode != "--serialize" && mode != "--deserialize" && mode != "--bytecode") {
    llvm::errs()
        << "usage: stablehlo-rust-validator "
           "--version|--verify|--verify-generic|--serialize|--deserialize|--bytecode\n";
    return 2;
  }
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::complex::ComplexDialect,
                  mlir::func::FuncDialect, mlir::quant::QuantDialect,
                  mlir::shape::ShapeDialect,
                  mlir::sparse_tensor::SparseTensorDialect,
                  mlir::tensor::TensorDialect,
                  mlir::stablehlo::StablehloDialect, mlir::vhlo::VhloDialect>();
  printVersion();
  std::string line;
  size_t index = 0;
  while (std::getline(std::cin, line)) {
    std::string request;
    if (!decode(line, request)) {
      llvm::errs() << "invalid hex request at index " << index << '\n';
      return 2;
    }
    mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
    context.allowUnregisteredDialects(false);
    std::string diagnostics;
    llvm::raw_string_ostream diagnosticStream(diagnostics);
    mlir::ScopedDiagnosticHandler handler(
        &context, [&](mlir::Diagnostic &diag) {
          printDiagnostic(diag, diagnosticStream);
          return mlir::success();
        });
    llvm::StringRef source(request);
    llvm::StringRef target;
    if (mode == "--serialize" || mode == "--bytecode") {
      auto separator = source.find('\t');
      if (separator == llvm::StringRef::npos) {
        llvm::errs() << "missing target version at index " << index << '\n';
        return 2;
      }
      target = source.take_front(separator);
      source = source.drop_front(separator + 1);
    }
    auto module =
        mode == "--deserialize"
            ? mlir::stablehlo::deserializePortableArtifact(source, &context)
            : mlir::parseSourceString<mlir::ModuleOp>(source, &context);
    bool accepted = module && mlir::succeeded(mlir::verify(*module));
    std::string result;
    if (accepted) {
      llvm::raw_string_ostream stream(result);
      if (mode == "--bytecode") {
        int64_t version;
        if (target.getAsInteger(10, version) || version < 0 || version > 6) {
          llvm::errs() << "invalid bytecode version at index " << index << '\n';
          return 2;
        }
        mlir::BytecodeWriterConfig config("stablehlo-native-reference");
        config.setDesiredBytecodeVersion(version);
        accepted = mlir::succeeded(mlir::writeBytecodeToFile(*module, stream, config));
      } else if (mode == "--serialize") {
        accepted = mlir::succeeded(mlir::stablehlo::serializePortableArtifact(
            *module, target, stream));
      } else if (mode == "--verify-generic") {
        module->print(stream, mlir::OpPrintingFlags().printGenericOpForm().enableDebugInfo());
      } else {
        module->print(stream);
      }
    }
    if (!accepted && diagnostics.empty()) {
      llvm::errs() << "upstream failed without a diagnostic at index " << index
                   << '\n';
      return 2;
    }
    // Artifacts are hex inside the UTF-8 response, matching the Python adapter.
    if (accepted && (mode == "--serialize" || mode == "--bytecode"))
      result = encode(result);
    llvm::outs() << index++ << '\t' << (accepted ? "OK" : "ERROR") << '\t'
                 << encode(accepted ? result : diagnostics) << '\n';
    llvm::outs().flush();
  }
  if (std::cin.bad()) {
    llvm::errs() << "cannot read requests\n";
    return 2;
  }
  return 0;
}
