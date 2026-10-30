# pliron-tensor

Native [pliron](https://crates.io/crates/pliron) operations for MLIR's `tensor`
dialect. Tensor types reuse `pliron-mlir-builtin`, rather than introducing
another tensor representation.

The crate represents all 21 operations in the pinned `TensorOps.td`:
`bitcast`, `cast`, `concat`, `dim`, `empty`, `extract`, `extract_slice`,
`from_elements`, `gather`, `generate`, `insert`, `insert_slice`, `rank`,
`reshape`, `expand_shape`, `collapse_shape`, `pad`, `parallel_insert_slice`,
`scatter`, `splat`, and `yield`.

## API contract

Constructors allocate IR; use `Verify::verify` before consuming manually edited
IR. `generate` and `pad` also provide fallible `build` constructors that populate
and verify their regions. `MixedIndex` expresses static and dynamic sizes,
offsets, strides, and padding. Static storage uses MLIR's dynamic marker
(`i64::MIN`), while dynamic tensor type dimensions use `-1`.

`parallel_insert_slice` requires a parent implementing `InParallelOpInterface`;
external dialects can implement that marker on their parallel-combining op.
This crate does not implement SCF, bufferization, canonicalization, or tiling.

Region builders are not transactional: a failure can leave allocated operations
in the context. Operand accessors assume valid arity. Local verification does
not establish whole-module SSA scope/dominance or runtime slice bounds.
Use `Diagnostic::from_error` for structured dialect errors.

## Installation

```toml
[dependencies]
pliron-tensor = "0.1"
```

Default features are empty.

## Example

Construct a statically shaped, uninitialized tensor:

```rust
use pliron_tensor::{
    ops::EmptyOp,
    pliron::{builtin::types::FP32Type, common_traits::Verify, context::Context},
    pliron_mlir_builtin::types::TensorType,
};

let ctx = &mut Context::new();
let ty = TensorType::from_element(ctx, FP32Type::get(ctx).into(), vec![4, 8]);
let empty = EmptyOp::new(ctx, ty, vec![]);
empty.verify(ctx)?;
assert!(empty.result(ctx).is_some());
# Ok::<(), pliron_tensor::Error>(())
```

`empty` supplies shape and storage, not initialized element values. For dynamic
dimensions, use `-1` in the tensor type and supply one SSA `index` size operand
per dynamic dimension, in dimension order.

## MLIR interoperability

Enable `mlir-interop` for verified MLIR generic-form export on every operation.

```toml
[dependencies]
pliron-tensor = { version = "0.1", features = ["mlir-interop"] }
```

Compose these operations in a `stablehlo::ops::ModuleOp` and export the complete
module with `stablehlo::mlir::export(ctx, &module)` after explicitly adding
`stablehlo` 0.4 with `mlir-interop`. Tensor has no StableHLO or linalg dependency.
Export rejects unknown,
mistyped, or misplaced attributes. This is textual export, not an inverse
importer or a native MLIR binding.

## License

Original code is Apache-2.0; upstream-derived portions retain Apache-2.0 with
LLVM Exceptions. The package includes `LICENSE`, `LICENSE-LLVM`, and `NOTICE`.
