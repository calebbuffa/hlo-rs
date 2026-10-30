# pliron-dialects

Rust dialects and compiler tools for [pliron](https://crates.io/crates/pliron),
covering StableHLO, ClientHLO, OpenXLA HLO, MLIR arithmetic/math/tensor/linalg,
and IREE's linalg extensions.

| Crate                                             | Use it for                                                                    | Guide                                            |
| ------------------------------------------------- | ----------------------------------------------------------------------------- | ------------------------------------------------ |
| [`stablehlo`](https://crates.io/crates/stablehlo) | Constructing and verifying StableHLO IR; opt-in MLIR interop                  | [Guide](crates/stablehlo/README.md)              |
| [`clienthlo`](https://crates.io/crates/clienthlo) | Client HLO operations that complement StableHLO                               | [Guide](crates/clienthlo/README.md)              |
| [`xlahlo`](https://crates.io/crates/xlahlo)       | OpenXLA HLO IR, metadata, inference, and verification                         | [Guide](crates/xlahlo/README.md)                 |
| `pliron-mlir-builtin`                             | Shared MLIR builtin types, attributes, and optional textual printing          | [Guide](crates/pliron-mlir-builtin/README.md)    |
| `pliron-arith`                                    | MLIR `arith` operations; opt-in MLIR interop                                  | [Guide](crates/pliron-arith/README.md)           |
| `pliron-math`                                     | MLIR `math` operations; opt-in MLIR interop                                   | [Guide](crates/pliron-math/README.md)            |
| `pliron-tensor`                                   | MLIR `tensor` operations, slicing, reshaping and padding; opt-in MLIR interop | [Guide](crates/pliron-tensor/README.md)          |
| `pliron-linalg`                                   | MLIR `linalg` operations; opt-in MLIR interop                                 | [Guide](crates/pliron-linalg/README.md)          |
| `pliron-iree-linalg-ext`                          | IREE `iree_linalg_ext` operations; opt-in MLIR interop                        | [Guide](crates/pliron-iree-linalg-ext/README.md) |

## License

The workspace default is Apache-2.0. See [`LICENSE`](LICENSE) and each crate's
manifest and bundled `LICENSE` file for its applicable terms.
