# pliron-arith

Native [pliron](https://crates.io/crates/pliron) operations for the MLIR
[`arith`](https://mlir.llvm.org/docs/Dialects/ArithOps/) dialect.

- constants: `arith.constant`
- floating-point: `addf`, `subf`, `mulf`, `divf`, `remf`, `negf`,
  `flush_denormals`, `maximumf`, `minimumf`, `maxnumf`, `minnumf`,
  `maximumnumf`, `minimumnumf`
- integer and index: `addi`, `subi`, `muli`, `divsi`, `divui`, `ceildivsi`,
  `ceildivui`, `floordivsi`, `remsi`, `remui`, `andi`, `ori`, `xori`, `shli`,
  `shrsi`, `shrui`, `maxsi`, `minsi`, `maxui`, `minui`
- extended integer: `addui_extended`, `subui_extended`, `mulsi_extended`,
  `mului_extended`
- comparison and selection: `cmpf`, `cmpi`, `select`
- conversions: `extf`, `truncf`, `convertf`, `scaling_extf`, `scaling_truncf`,
  `extsi`, `extui`, `trunci`, `sitofp`, `uitofp`, `fptosi`, `fptoui`,
  `index_cast`, `index_castui`, `bitcast`

Operands may be scalars, ranked or unranked tensors, or vectors. Elementwise
operations require matching container shapes, as upstream MLIR does. Scalar,
tensor, and vector types are shared with the
[`pliron-mlir-builtin`](https://crates.io/crates/pliron-mlir-builtin) crate so that values flow
between dialects without conversion.

Fast-math flags (`#arith.fastmath`), integer overflow flags
(`#arith.overflow`), rounding modes, and the `exact` and `nneg` unit flags are
stored as typed attributes and default to their upstream defaults when absent.

```toml
[dependencies]
pliron-arith = "0.1"
```

Default features are empty. Enable `mlir-interop` for textual MLIR export.

## Example

Create a floating-point addition using typed block arguments:

```rust
use pliron_arith::{
    ops::AddFOp,
    pliron::{
        basic_block::BasicBlock, builtin::types::FP32Type,
        common_traits::Verify, context::Context, op::Op,
    },
};

let ctx = &mut Context::new();
let f32 = FP32Type::get(ctx).into();
let block = BasicBlock::new(ctx, None, vec![f32, f32]);
let args: Vec<_> = block.deref(ctx).arguments().collect();
let add = AddFOp::new(ctx, args[0], args[1]);
add.get_operation().insert_at_back(block, ctx);
add.verify(ctx)?;
assert_eq!(add.lhs(ctx), args[0]);
# Ok::<(), pliron_arith::Error>(())
```

This is a low-level block example. For compiler input, place the block's
operations in a function inside a `stablehlo::ops::ModuleOp`, then export the
complete module with `stablehlo::mlir::export(ctx, &module)`.
For that composition, add `stablehlo` 0.4 with `mlir-interop` explicitly;
arith itself does not depend on StableHLO.

## MLIR interoperability

Enable the `mlir-interop` feature to implement `ToMlirOp` for every operation.
Operations print in MLIR's generic form with upstream property names
(`fastmath`, `overflowFlags`, `roundingmode`, `isExact`, `nonNeg`, `predicate`,
`value`). Export produces text, not a native MLIR object or compiled program.

```toml
[dependencies]
pliron-arith = { version = "0.1", features = ["mlir-interop"] }
```

Export validates attribute types and operation ownership, even when a flag's
default value would otherwise be omitted, and rejects duplicate translated
properties. The repository's native-IREE compatibility gate covers every
operation, with scalar, tensor, fixed-vector, and scalable-vector fixtures for
elementwise operation families.
Compatibility is tied to the compiler revision recorded by that gate, not to
all possible IREE versions or target-specific lowering combinations.

## API contract

`new` allocates an operation without verifying it. Verification does not execute
arithmetic or prove value-dependent properties such as a nonzero divisor.
Operand accessors assume the declared arity. Optional attribute accessors use
defaults when absent; export rejects stored attributes with the wrong type or
owner. Use `Diagnostic::from_error` to inspect structured dialect errors.

## License

Original code is Apache-2.0; upstream-derived portions retain Apache-2.0 with
LLVM Exceptions. The package includes `LICENSE`, `LICENSE-LLVM`, and `NOTICE`.
