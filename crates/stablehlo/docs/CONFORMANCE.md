# StableHLO conformance

**Not ready for a conformant public release.** This document describes the
implemented subset, independent evidence and remaining work. It is not a
percentage-complete estimate or a claim that every accepted program is valid.

## Reference

| Source | Exact revision |
| --- | --- |
| StableHLO 1.22.0 | `25605c9f48a9aec305876d60d26b028398d298d2` |
| Its LLVM/MLIR dependency | `2b35fb9283a44aae92cb72a41d077ee8de2be7ad` |

Upstream repositories belong under the workspace's `extern` directory.
Audits check the exact revision and fixture contents; a different submodule
HEAD is not an automatic upgrade of these pins.

## Coverage and release blockers

| Area | Implemented | Still required |
| --- | --- | --- |
| IR | Typed multi-result SSA, functions/declarations, direct calls, nested regions, lexical visibility, metadata and locations | General MLIR symbol tables and multi-block control-flow interchange |
| Operations | Operand/result and named-attribute schemas for all 118 identities; many relational constraints | Independent constraint-by-constraint audit, including quantization, encodings and bounds |
| Inference | Shared result inference for arithmetic, shape operations, linear algebra, windows, indexing, control flow and collectives | Remaining operation-result rules and complete encoding/bound propagation |
| Text | Generic/custom StableHLO assembly, aliases, literals, affine and sparse metadata | Remaining general MLIR syntax and broader independent roundtrips |
| Bytecode | Format versions 0-6; textual fallback and supported builtin, StableHLO and uniform Quant payloads; operation properties | Unmodeled builtin/Quant payloads, resources, successors and custom use-list ordering |
| Portable artifacts | Version parsing, compatibility policy and producer-header discovery | Rust VHLO conversion and portable serialization/deserialization |
| Robustness | Explicit stacks on several nested parsing, printing and validation paths; depth/expansion limits | Eliminate remaining recursive cross-kind parsing, region parsing and other IR traversals |
| Semantics | Independent execution checks for selected programs | Broad matching-version execution coverage |

There is one canonical Rust IR for construction, text and bytecode.
The crate does not implement optimization, backend lowering or execution.
Unsupported interchange features return errors.

## What has been checked

These are completed audits of specific corpora, not exhaustive dialect coverage.
External-tool audits are opt-in and are not rerun by ordinary `cargo test`.

| Evidence | Result | Test source |
| --- | --- | --- |
| Pinned operation fixtures | 195 positive chunks accepted and round-tripped; 4 classified positive rejections; 414 diagnostic-bearing chunks rejected | `tests/reference_fixtures.rs` |
| Independent 1.22 native parser/verifier | All 195 emitted positives verify and reparse in Rust; all 414 negatives contain every expected diagnostic pattern | `tests/upstream_interop.rs` |
| Bidirectional MLIR bytecode | 195 accepted modules at 7 format versions: 1,365 Rust-written modules verify natively, and 1,365 native-written modules import in Rust | `tests/bytecode_writer.rs`, `tests/bytecode_native.rs` |
| Native StableHLO payloads | All 22 custom attribute codes exercised at all 7 versions; represented module structure preserved | `tests/bytecode_native.rs`, `tests/upstream_support/structure.rs` |
| Portable header discovery | 8 native-generated historical/current artifacts inspected; native deserialization returns Rust-readable assembly | `tests/portable_versions.rs` |
| Independent CPU execution | 29 runtime-input programs; 203 executions after Rust bytecode writing at all 7 versions; exact output types, shapes and bytes | `tests/execution_semantics.rs`, `tests/bytecode_writer.rs` |

The native parser/verifier uses the exact 1.22/LLVM pins above and registers
the required supporting dialects. It is a test-only C++ driver, not part of
the Rust implementation. Bytecode structure comparisons ignore SSA spelling
and explicit versus implicit unknown locations, not operation metadata.

Execution uses the separately pinned jaxlib 0.11.2 CPU client, whose bundled
StableHLO is **1.20.0**. Its assembly-only audit accepts 180 of the 195 positive
chunks; the other 15 are explicit Quant/SparseTensor registration or WEAKORDER
version limitations. Of 414 negatives, 403 include the expected diagnostics;
11 have explicit Quant/WEAKORDER tool limitations. These older-tool results
are not matching-1.22 conformance evidence.

Portable serialization in the native audit runs in upstream C++, not Rust.
The Rust writer's `stablehlo-rust` producer does not denote a VHLO portable
artifact. Header discovery alone does not validate artifact contents.

## Known differences and limitations

When the pinned specification and executable reference disagree, tests record
the choice explicitly rather than weakening validation to match a fixture.
References below are paths within the pinned upstream sources.

| Case | Current behavior and reference |
| --- | --- |
| `all_gather_variadic` positive fixture | Rejected: its second result grows by 2 for a group of 4. Retains `docs/spec.md`, all_gather C6; `TypeInference.cpp::verifyAllGatherOp` only checks divisibility. |
| `convolution_operand_element_type_i4` positive fixture | Rejected: i4 operands produce i8 outside the specified exception. Retains convolution C27; the C++ verifier omits this nonquantized result-element check. |
| Two `create_sort_*` positive fixtures | Rejected because `hlo_test_create_sort.build` belongs to a test-only dialect, not StableHLO. |
| Dynamic shapes and bounds | Jointly contradictory witnesses are rejected even when the C++ verifier's pairwise comparisons accept them. This is not a complete nonlinear dimension-product solver. |
| Bitcast conversion | Enforces the integral minor-dimension width equation even for dynamic dimensions. See bitcast_convert C1-C2 and `TypeInference.cpp::verifyBitcastConvertOp`. |
| RNG shape | Follows the executable integer/index vector schema and runtime-shape inference rather than the prose table's constant-si64 restriction. See `StablehloOps.td` and `TypeInference.cpp::inferRngOp`. |
| Bit-generator state | Enforces the written rank-one ui64 and algorithm-length contracts. The pinned C++ verifier checks only compatible state shapes. |
| Zero-sized collective exchange | Retains the C++ verifier's positive-exchange restriction, although the written size equalities do not prohibit zero. |
| Linear algebra | Triangular solve follows the schema's float/complex domain, not prose allowing quantization. Cholesky retains C1's identical baseline types rather than the inconsistent f32-to-f64 example. |
| Zero-length IRFFT | Requires zero-length input per FFT C5; the pinned C++ helper also accepts one input element. |
| Async regions | Requires isolated block arguments per `StablehloOps.cpp`, not outer-value captures shown in some specification examples. |
| Sparse/affine metadata | No complete MLIR expression canonicalization or Presburger analysis. Sparse inverse maps must match the inferred represented tree; malformed/incomplete blocked mappings reject explicitly. |
| Complex metadata | Printing an f32 signaling NaN returns an error if widening would change its payload. |
| Selected entry function | `stablehlo_rust.entry` is a Rust interchange convention, not an upstream StableHLO attribute. |

Dedicated regression suites live in `tests/`: `reshape_constraints.rs`,
`bitcast_operations.rs`, `rng_inference.rs`, `rng_bit_generator.rs`,
`linear_algebra_inference.rs`, `fft_inference.rs`, `async_operations.rs`,
`affine_attributes.rs`, `sparse_encodings.rs` and `complex_attributes.rs`.
Operation-specific `*_inference.rs` tests cover the shared inference rules.
Generated shape tests enumerate small independent witnesses; they are not
proofs of every possible shape.

### Stack safety

Explicit work stacks cover direct alias chains, nested attribute containers,
dense literal lists, affine parsing/printing/evaluation, location-instance
traversal, shared attribute/location verification, type verification and
tuple/future/function type printing. Regressions exercise nesting near the
128-level limit without lowering that limit to conceal stack exhaustion.

This is **not a crate-wide recursion-free guarantee**. Type/attribute and
location/attribute parsing/printing can still form recursive cross-kind calls;
nested operation/region parsing and other traversals also remain. Dependency
decoding uses cycle detection and cumulative expansion budgets because a depth
limit alone does not prevent exponential work.

## Reproducing the audits

Ordinary checks, from the `hlo-rs` workspace root:

```powershell
cargo test -p stablehlo --locked
cargo clippy -p stablehlo --all-targets --locked -- -D warnings
```

For source/native audits, run from `crates\stablehlo`. First ensure
`extern\stablehlo` is at the exact pin; do not silently change submodule revisions.
Build the optional validator using its [setup guide](../tests/upstream/native/README.md).

```powershell
$env:STABLEHLO_REFERENCE_DIR = (Resolve-Path ..\..\extern\stablehlo).Path
$env:STABLEHLO_UPSTREAM_NATIVE = (Resolve-Path ..\..\target\reference-1.22\build\bin\stablehlo-rust-validator.exe).Path
cargo test --test reference_fixtures -- --ignored --nocapture
cargo test --test upstream_interop native -- --ignored --nocapture
cargo test --test bytecode_native -- --ignored --nocapture
cargo test --test portable_versions matching_native -- --ignored --nocapture
```

Additional ignored tests require
`STABLEHLO_UPSTREAM_PYTHON` pointing to an isolated environment installed from
`tests\upstream\requirements.txt`. See the relevant test's ignore message
before running it. Native binaries retained from the pre-workspace layout may
still run, but their old CMake caches must not be reused.
