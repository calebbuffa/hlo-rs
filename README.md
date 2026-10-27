# hlo-rs

Rust dialects and compiler tools for [pliron](https://crates.io/crates/pliron),
covering StableHLO, ClientHLO, and OpenXLA HLO.

| Crate                                             | Use it for                                                   | Guide                               |
| ------------------------------------------------- | ------------------------------------------------------------ | ----------------------------------- |
| [`stablehlo`](https://crates.io/crates/stablehlo) | Constructing and verifying StableHLO IR; opt-in MLIR interop | [Guide](crates/stablehlo/README.md) |
| [`clienthlo`](https://crates.io/crates/clienthlo) | Client HLO operations that complement StableHLO              | [Guide](crates/clienthlo/README.md) |
| [`xlahlo`](https://crates.io/crates/xlahlo)       | OpenXLA HLO IR, metadata, inference, and verification        | [Guide](crates/xlahlo/README.md)    |

## Getting started

Add the crate for the dialect you use to your `Cargo.toml`, then follow its
guide for a complete example. For example:

```toml
[dependencies]
stablehlo = "0.2"
```

The `clienthlo` crate depends on `stablehlo`.

StableHLO MLIR assembly is available as `stablehlo::mlir` with the
`mlir-interop` feature. Use `mlir::import` and `mlir::export` for modules;
advanced helpers live in `mlir::assembly`. Export uses `pliron-mlir-interop`; the
feature is disabled by default. MLIR assembly is distinct from pliron's
internal parser/printer format.

## Requirements

Rust 1.95 or newer is required.

## License

The workspace default is Apache-2.0. See [`LICENSE`](LICENSE) and each crate's
manifest and bundled `LICENSE` file for its applicable terms.
