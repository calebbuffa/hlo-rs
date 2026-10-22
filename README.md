# hlo-rs

Rust dialects and compiler tools for [pliron](https://crates.io/crates/pliron),
covering StableHLO and OpenXLA HLO. Choose the crate that matches the IR or
compiler stage you need:

| Crate                                                     | Use it for                                                      | Guide                                     |
| --------------------------------------------------------- | --------------------------------------------------------------- | ----------------------------------------- |
| [`stablehlo`](https://crates.io/crates/stablehlo)         | Constructing and verifying StableHLO IR                         | [Guide](crates/stablehlo/README.md)       |
| `stablehlo-codec`                                         | StableHLO MLIR types, attributes, and supported module assembly | [Guide](crates/stablehlo-codec/README.md) |
| [`clienthlo`](https://crates.io/crates/clienthlo)         | Client HLO operations that complement StableHLO                 | [Guide](crates/clienthlo/README.md)       |
| [`xlahlo`](https://crates.io/crates/xlahlo)               | OpenXLA HLO IR, metadata, inference, and verification           | [Guide](crates/xlahlo/README.md)          |
| `xlahlo-codec`                                            | HLO shape/layout assembly and While backend-config JSON         | [Guide](crates/xlahlo-codec/README.md)    |
| [`xlahlo-cubecl`](https://crates.io/crates/xlahlo-cubecl) | Lowering supported xlahlo modules into CubeCL execution plans   | [Guide](crates/xlahlo-cubecl/README.md)   |

The dialect crates provide typed IR construction, inference, and verification;
they do not provide a general MLIR text parser or guarantee that every
operation can be executed by a backend. Backend support and limitations are
documented by the relevant crate.

## Getting started

Add the crate for the dialect you use to your `Cargo.toml`, then follow its
guide for a complete example. For example:

```toml
[dependencies]
stablehlo = "0.1"
pliron = "0.19"
```

The `clienthlo` crate depends on `stablehlo`. The `xlahlo-cubecl` crate depends
on `xlahlo` and returns an execution plan; the caller provides device memory,
launches kernels, and synchronizes execution.

Each codec depends on its corresponding IR crate. External format adapters
are separate from the dialects' pliron parser/printer registrations; no IR
crate depends on a codec. See the codec guides for their current coverage.

## Requirements

Rust 1.95 or newer is required.

## License

The workspace default is Apache-2.0. See [`LICENSE`](LICENSE) and each crate's
manifest and bundled `LICENSE` file for its applicable terms.
