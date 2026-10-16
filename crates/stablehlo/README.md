# stablehlo

Typed Rust IR, verification, shape inference, text assembly and MLIR bytecode
for [StableHLO](https://github.com/openxla/stablehlo).
No optimizer, backend lowering, execution runtime or C++/MLIR dependency.

**Experimental: not yet a conformant StableHLO release.** The target is
StableHLO **1.22.0**. See [conformance status](docs/CONFORMANCE.md) for evidence,
known differences and release blockers.

## Use

Requires Rust 1.95 or newer. From another package in this repository:

```toml
[dependencies]
stablehlo = { path = "../stablehlo" }
```

Parse and verify a module using the same IR used by programmatic construction:

```rust
use stablehlo::text::{parse_module, print_module};

let module = parse_module(r#"
    module {
        func.func @add(%lhs: tensor<2xf32>, %rhs: tensor<2xf32>) -> tensor<2xf32> {
            %sum = stablehlo.add %lhs, %rhs : tensor<2xf32>
            return %sum : tensor<2xf32>
        }
    }
"#)?;
let assembly = print_module(&module)?;
# Ok::<(), stablehlo::text::TextError>(())
```

Parsing verifies the module; printing normalizes it to generic MLIR assembly.
Unsupported syntax or encodings return errors rather than dropping metadata.

## Capabilities and limits

| Area         | Available                                                                                                 |
| ------------ | --------------------------------------------------------------------------------------------------------- |
| IR           | Typed multi-result SSA, functions, calls, regions, tuples, tokens and source locations                    |
| Verification | Schemas for all 118 operation identities and many operation-specific shape/type constraints               |
| Inference    | Shared shape, bound and element inference used by the verifier                                            |
| Assembly     | Generic and custom StableHLO parsing; generic printing; aliases, quantization, affine and sparse metadata |
| Bytecode     | Rust reading/writing at MLIR format versions 0-6, including supported native dialect payloads             |

Operation-name coverage is **not** complete constraint coverage. General MLIR
interchange remains partial. VHLO conversion and portable-artifact
serialization/deserialization are not implemented; discovering an artifact's
version does not validate its body. Some user-controlled traversals still need
conversion to explicit stacks.

The library uses the pure-Rust `rustc_apfloat` dependency. Native validators
and Python execution adapters are optional test tools, never library dependencies.

## Development

From the `hlo-rs` workspace root:

```powershell
cargo test -p stablehlo --locked
cargo clippy -p stablehlo --all-targets --locked -- -D warnings
cargo doc -p stablehlo --no-deps
```

Ordinary tests need no upstream checkout or native tools.
[Conformance audits](docs/CONFORMANCE.md#reproducing-the-audits) use exact pinned
sources under `extern`; [native validator setup](tests/upstream/native/README.md)
documents the optional build.

## License

Apache-2.0 or MIT.
