# hlo-rs

Rust implementations of StableHLO and XLA HLO, maintained as independent crates
in one Cargo workspace:

| Crate                                     | Scope                                                               |
| ----------------------------------------- | ------------------------------------------------------------------- |
| [`stablehlo`](crates/stablehlo/README.md) | StableHLO typed IR, verification, inference, text and MLIR bytecode |
| [`xlahlo`](crates/xlahlo/README.md)       | XLA HLO data model, graph operations, verification and interchange  |

Neither crate is a backend, runtime, or a complete conformant release yet.

## Development

From this directory:

```bash
cargo check --workspace --all-targets --locked
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
```

XLA HLO generates protobuf bindings and its opcode catalog from `extern\xla`
during Cargo builds; initialize that submodule before building.

## Reference sources and native tools

Upstream source for reference and testing lives in `extern`. Initialize sources when needed with:

```bash
git submodule update --init --recursive
```

To deliberately update the upstream sources:

```bash
git submodule update --remote --merge
```
