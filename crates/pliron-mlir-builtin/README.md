# pliron-mlir-builtin

Pure-Rust representations of MLIR builtin types and attributes for
[pliron](https://crates.io/crates/pliron).

## Installation

```toml
[dependencies]
pliron-mlir-builtin = "0.1"
```

Default features are empty. The optional `mlir-interop` feature provides
textual MLIR printing through `pliron-mlir-interop`; it does not link native
MLIR or IREE.

## Types and attributes

The crate provides ranked/unranked tensors, fixed/scalable vectors, index,
complex and additional floating-point types, tuples, function signatures,
ranked/unranked buffers, and `none`. It also provides affine expressions/maps
and integer sets, exact scalar bits, dense/sparse elements, dense arrays,
symbol references, layouts, byte strings, opaque attributes, and shared
integer-array storage.

```rust
use pliron_mlir_builtin::{
    pliron::{common_traits::Verify, context::Context},
    types::{ElementType, TensorType},
};

let ctx = &Context::new();
let ty = TensorType::static_shape(ctx, ElementType::F32, [4, -1]);
ty.deref(ctx).verify(ctx)?;
# Ok::<(), pliron_mlir_builtin::Error>(())
```

Constructors intern representations without verifying them. Verify before
relying on invariants. All handles and nested attributes must belong to the
same context.

## Dialect composition

StableHLO re-exports these exact Rust types at its existing public paths.
Registration identifiers inherited from StableHLO are intentionally retained
for compatibility with existing pliron text. They are internal representation
names, not MLIR assembly spellings; use `mlir-interop` to emit MLIR.

Dialect-specific quantization, tokens, futures, bounds, and sparse encodings
remain in their owning dialect. `interfaces` lets those dialects extend
tensor-element, value-type, and encoding verification without creating reverse
dependencies. Foreign textual representations use the framework's
`ToMlirType`/`ToMlirAttr` interfaces. Unknown representations fail explicitly.

For composition with HLO operations, use `stablehlo` 0.4 and `clienthlo` 0.3.
Older StableHLO releases define the pre-extraction representations themselves
and must not be linked alongside this crate.

## License

Original code is Apache-2.0; upstream-derived portions retain Apache-2.0 with
LLVM Exceptions. See `LICENSE`, `LICENSE-LLVM`, and `NOTICE`.
