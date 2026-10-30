# pliron-math

Native [pliron](https://crates.io/crates/pliron) operations for the MLIR
[`math`](https://mlir.llvm.org/docs/Dialects/MathOps/) dialect.

The crate covers every `math` operation as of the LLVM revision IREE pins
(`extern/llvm-project`), complementing [`pliron-arith`](https://crates.io/crates/pliron-arith):

- rounding and sign: `absf`, `ceil`, `floor`, `round`, `roundeven`, `trunc`,
  `copysign`, `clampf`
- exponential and logarithmic: `exp`, `exp2`, `expm1`, `log`, `log2`, `log10`,
  `log1p`, `powf`, `fpowi`, `sqrt`, `rsqrt`, `cbrt`
- trigonometric and hyperbolic: `sin`, `cos`, `sincos`, `tan`, `asin`, `acos`,
  `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`
- special functions: `erf`, `erfc`, `fma`
- classification: `isnan`, `isinf`, `isfinite`, `isnormal`
- integer: `absi`, `ctlz`, `cttz`, `ctpop`, `ipowi`

Operands may be scalars, ranked or unranked tensors, or vectors, using the
types supplied by `pliron-mlir-builtin` and shared with `pliron-arith`. Floating-point operations
that support fast-math flags carry `#arith.fastmath`, defaulting to `none`.
Classification and integer operations do not accept those flags.

## Installation

```toml
[dependencies]
pliron-math = "0.1"
```

Default features are empty. Enable `mlir-interop` for textual MLIR export.

## Example

```rust
use pliron_math::{
    ops::ExpOp,
    pliron::{
        basic_block::BasicBlock, builtin::types::FP32Type,
        common_traits::Verify, context::Context, op::Op,
    },
};

let ctx = &mut Context::new();
let f32 = FP32Type::get(ctx).into();
let block = BasicBlock::new(ctx, None, vec![f32]);
let input = block.deref(ctx).arguments().next().unwrap();
let exp = ExpOp::new(ctx, input);
exp.get_operation().insert_at_back(block, ctx);
exp.verify(ctx)?;
assert_eq!(exp.operand(ctx), input);
# Ok::<(), pliron_math::Error>(())
```

This constructs IR; it does not calculate the exponential in Rust.

## MLIR interoperability

Enable the `mlir-interop` feature to implement `ToMlirOp` for every operation.
Operations print in MLIR's generic form with the upstream `fastmath` property.

```toml
[dependencies]
pliron-math = { version = "0.1", features = ["mlir-interop"] }
```

Compose these operations with `pliron-arith` inside linalg regions or function
bodies. Export a complete `stablehlo::ops::ModuleOp` using
`stablehlo::mlir::export(ctx, &module)`, then parse and compile it with IREE.
Shared types are re-exported through `pliron_math::pliron_mlir_builtin`.
StableHLO 0.4 is an optional application-level composition choice, not a dependency.

## API contract

`new` allocates without verification. Call `Verify::verify` before consuming
manually edited IR; operand accessors assume the declared arity. Verification
checks IR contracts, not runtime domains or numerical accuracy.
Export rejects unknown, mistyped, or misplaced attributes, even when their
values would otherwise be omitted as defaults.
Use `Diagnostic::from_error` for structured dialect errors.

## License

Original code is Apache-2.0; upstream-derived portions retain Apache-2.0 with
LLVM Exceptions. The package includes `LICENSE`, `LICENSE-LLVM`, and `NOTICE`.
