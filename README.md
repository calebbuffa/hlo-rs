# hlo-rs

Rust dialects and compiler tools for [pliron](https://crates.io/crates/pliron),
covering StableHLO and OpenXLA HLO. Choose the crate that matches the IR or
compiler stage you need:

| Crate | Use it for | Guide |
| --- | --- | --- |
| [`stablehlo`](https://crates.io/crates/stablehlo) | Constructing and verifying StableHLO IR | [Guide](crates/stablehlo/README.md) |
| [`clienthlo`](https://crates.io/crates/clienthlo) | Client HLO operations that complement StableHLO | [Guide](crates/clienthlo/README.md) |
| [`xlahlo`](https://crates.io/crates/xlahlo) | OpenXLA HLO IR, metadata, inference, and verification | [Guide](crates/xlahlo/README.md) |
| [`xlahlo-cubecl`](https://crates.io/crates/xlahlo-cubecl) | Lowering supported xlahlo modules into CubeCL execution plans | [Guide](crates/xlahlo-cubecl/README.md) |

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

## Requirements

Rust 1.95 or newer is required.

## License

Apache-2.0. See [`LICENSE-APACHE`](LICENSE-APACHE).
