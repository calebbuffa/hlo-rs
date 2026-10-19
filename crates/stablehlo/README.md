# stablehlo

`stablehlo` is a StableHLO dialect for [pliron](https://crates.io/crates/pliron).

## Example

Build and verify an elementwise addition:

```rust
use stablehlo::{
    ops::{AddOp, ConstantOp},
    pliron::{common_traits::Verify, context::Context, op::Op, r#type::Typed},
    types::{tensor, ElementType},
};

let mut context = Context::default();
let ty = tensor(&context, &[4], ElementType::F32);
let lhs = ConstantOp::new(&mut context, ty, vec![0x3f80_0000], true);
let rhs = ConstantOp::new(&mut context, ty, vec![0x4000_0000], true);
let lhs = lhs.get_operation().deref(&context).get_result(0);
let rhs = rhs.get_operation().deref(&context).get_result(0);
let sum = AddOp::new_inferred(&mut context, lhs, rhs).expect("compatible operands");
sum.verify(&context).expect("valid addition");
```

The crate re-exports pliron at `stablehlo::pliron` so downstream code can use the matching IR APIs without a second direct dependency declaration.

## Coverage

All 118 StableHLO operations are implemented under `ops`, grouped by constants, control flow, elementwise, indexing, reductions, collectives, tensor, shape, and compute operations. The `inference` module provides checked result-type inference. `verification` contains the fast module verifier and operation-family verification.

The `types` module includes named and arbitrary-width integer and floating-point formats, complex elements, ranked and unranked tensors, vectors, tuple/function/token types, buffers, and quantization types. The `attributes` and `sparse` modules provide typed StableHLO metadata, affine expressions, dense and sparse element values, layouts, and sparse tensor encodings. Integer payloads retain exact bit patterns.

Types are interned in a pliron context. Use `TensorType::get` for native element handles or `tensor` for a convenience constructor. Operation builders accept explicit result types or provide checked inference constructors. Builders do not replace verification: verify operations or call `verification::verify_module_fast` before using a complete module.

## License

See the workspace license file.
