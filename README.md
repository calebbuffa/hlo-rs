# pliron-dialects

Rust dialects and compiler tools for [pliron](https://crates.io/crates/pliron),
covering StableHLO, ClientHLO, OpenXLA HLO, MLIR arithmetic/math/tensor/linalg,
and IREE's linalg extensions.

| Crate                                             | Use it for                                                   | Guide                               |
| ------------------------------------------------- | ------------------------------------------------------------ | ----------------------------------- |
| [`stablehlo`](https://crates.io/crates/stablehlo) | Constructing and verifying StableHLO IR; opt-in MLIR interop | [Guide](crates/stablehlo/README.md) |
| [`clienthlo`](https://crates.io/crates/clienthlo) | Client HLO operations that complement StableHLO              | [Guide](crates/clienthlo/README.md) |
| [`xlahlo`](https://crates.io/crates/xlahlo)       | OpenXLA HLO IR, metadata, inference, and verification        | [Guide](crates/xlahlo/README.md)    |
| `pliron-arith`  | MLIR `arith` operations; opt-in MLIR interop                      | [Guide](crates/pliron-arith/README.md)  |
| `pliron-math`   | MLIR `math` operations; opt-in MLIR interop                       | [Guide](crates/pliron-math/README.md)   |
| `pliron-tensor` | MLIR `tensor` operations, slicing, reshaping and padding; opt-in MLIR interop | [Guide](crates/pliron-tensor/README.md) |
| `pliron-linalg` | MLIR `linalg` operations; opt-in MLIR interop                      | [Guide](crates/pliron-linalg/README.md) |
| `pliron-iree-linalg-ext` | IREE `iree_linalg_ext` operations; opt-in MLIR interop   | [Guide](crates/pliron-iree-linalg-ext/README.md) |

## Micra-to-IREE compatibility

The five `pliron-*` crates model tensor frontend IR with native Rust builders,
verification, and optional textual MLIR export. They reuse `stablehlo` 0.3
types rather than defining another tensor representation. Default features are
empty; `mlir-interop` does not link a native compiler or runtime.

| Dialect | Operations | Source revision |
| --- | ---: | --- |
| `arith` | 56 | LLVM `f1ee8b38f40fd1f0e9596197b8e690eb9b57ec0c` |
| `math` | 46 | Same LLVM revision |
| `linalg` | 78 | Same LLVM revision |
| `tensor` | 21 | Same LLVM revision |
| `iree_linalg_ext` | 24 | IREE `29114a017c104882cab055a49758e8f7055ea0d5` |

Prior native validation used compiler
`3.14.0 @ 5f72e26ac2e941ccf40ead9e30c438993282de9d`, distinct from the IREE
source pin: 683 emitted fixtures covering all 225 operations parsed, verified,
and survived native printing/reparsing. Two mixed-dialect modules compiled for
`vmvx`, with five numerical cases covering static and dynamic inputs.
These results describe the exercised fixtures, not all shapes, encodings,
targets, floating-point edge cases, or compiler revisions.

Use complete-module verification for SSA scope/dominance, then parse, verify,
compile, and execute representative modules with the deployment compiler.
Runtime-dependent preconditions remain the caller's responsibility. Memref
forms, reverse MLIR import for these dialects, and local bufferization/tiling
interfaces are outside this frontend contract; IREE owns downstream lowering.

## Release preparation

See the [publishing guide](docs/publishing.md) for standalone archive checks,
feature/documentation checks, and dependency-ordered publication. The release
checker never publishes crates.

## License

The workspace default is Apache-2.0. See [`LICENSE`](LICENSE) and each crate's
manifest and bundled `LICENSE` file for its applicable terms.
