# pliron-linalg

Native [pliron](https://crates.io/crates/pliron) operations for the MLIR
[`linalg`](https://mlir.llvm.org/docs/Dialects/Linalg/) dialect on tensors,
covering the upstream operation catalog for frontend emission.

- **Region operations**: `linalg.generic` (affine indexing maps,
  `parallel`/`reduction` iterator types, optional `doc`/`library_call`),
  `linalg.map`, `linalg.reduce`, and the body operations `linalg.yield` and
  `linalg.index`. Bodies are built from
  [`pliron-arith`](https://crates.io/crates/pliron-arith) and
  [`pliron-math`](https://crates.io/crates/pliron-math) operations.
- **Contractions**: `linalg.matmul`, `linalg.batch_matmul`,
  `linalg.batch_reduce_matmul` (with optional user-defined broadcast/transpose
  indexing maps and `cast`), `linalg.contract`, `linalg.scaled_contract`.
- **Elementwise**: `linalg.elementwise` with every
  `#linalg.elementwise_kind` (unary, binary, and ternary).
- **Data movement**: `linalg.transpose`, `linalg.broadcast`, `linalg.pack`,
  `linalg.unpack` (static or dynamic inner tiles, optional padding).
- **Other**: `linalg.softmax`, `linalg.winograd_filter_transform`,
  `linalg.winograd_input_transform`, `linalg.winograd_output_transform`.
- **Named operations** generated from MLIR's `LinalgNamedStructuredOps.yaml`
  (in [`ops::named`]): `copy`, `fill`, `fill_rng_2d`, `dot`, `matvec`,
  `vecmat`, `batch_matvec`, `batch_vecmat`, `mmt4d`, `batch_mmt4d`,
  `quantized_matmul`, `quantized_batch_matmul`, every `conv_*`,
  `depthwise_conv_*` and `pooling_*` variant, with `strides`, `dilations` and
  `cast` attributes where upstream defines them.

Operations use destination-passing style on ranked tensors from the
[`pliron-mlir-builtin`](https://crates.io/crates/pliron-mlir-builtin) crate: each result has the
type of its `outs` operand. Indexing maps reuse the `pliron-mlir-builtin` affine-map
attribute. Operations whose computation is implied by their name store no body
region.

IREE's `iree_linalg_ext` dialect lives in the separate
[`pliron-iree-linalg-ext`](https://crates.io/crates/pliron-iree-linalg-ext)
crate.

## Installation

```toml
[dependencies]
pliron-linalg = "0.1"
```

Default features are empty. Enable `mlir-interop` for textual MLIR export.

## Example

Construct a matrix multiplication with an explicit destination tensor:

```rust
use pliron_linalg::{
    ops::MatmulOp,
    pliron::{
        basic_block::BasicBlock, builtin::types::FP32Type,
        common_traits::Verify, context::Context, op::Op,
    },
    pliron_mlir_builtin::types::TensorType,
};

let ctx = &mut Context::new();
let f32 = FP32Type::get(ctx).into();
let lhs = TensorType::from_element(ctx, f32, vec![4, 8]);
let rhs = TensorType::from_element(ctx, f32, vec![8, 16]);
let output = TensorType::from_element(ctx, f32, vec![4, 16]);
let block = BasicBlock::new(ctx, None, vec![lhs, rhs, output]);
let args: Vec<_> = block.deref(ctx).arguments().collect();
let matmul = MatmulOp::new(ctx, args[0], args[1], args[2]);
matmul.get_operation().insert_at_back(block, ctx);
matmul.verify(ctx)?;
assert_eq!(matmul.init(ctx), args[2]);
# Ok::<(), pliron_linalg::Error>(())
```

The destination is an accumulator: multiplication adds to its existing values.
Supply a zero-filled destination when a plain matrix product is intended;
`tensor.empty` alone does not initialize its elements.

## MLIR interoperability

Enable the `mlir-interop` feature to implement `ToMlirOp` for every operation.
Region operations print in MLIR's generic form; all other operations print in
their upstream custom form, whose parser rebuilds the implicit body region.

```toml
[dependencies]
pliron-linalg = { version = "0.1", features = ["mlir-interop"] }
```

Export rejects attributes whose types or owning operation do not match,
including attributes consumed by custom assembly instead of printed in a
dictionary. This crate does not implement memref forms or local
bufferization/tiling passes.

Compose operations inside a `stablehlo::ops::ModuleOp` and export with
`stablehlo::mlir::export(ctx, &module)`. The upstream parser reconstructs implicit
named-operation bodies; the local pliron representation stores no such body.
This crate does not import linalg assembly. Add `stablehlo` 0.4 explicitly if
using its module/function containers; linalg has no StableHLO dependency.

## API contract

`new` allocates without verification. Region operations also provide `build`,
which populates the single-block body, appends `linalg.yield`, and verifies it.
A failed `build` can leave allocated operations in the context; it is not a
transaction. Operand accessors assume valid arity. Use whole-module verification
for SSA scope/dominance and `Diagnostic::from_error` for structured dialect errors.

## License

Original code is Apache-2.0; upstream-derived portions retain Apache-2.0 with
LLVM Exceptions. The package includes `LICENSE`, `LICENSE-LLVM`, and `NOTICE`.
