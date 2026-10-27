# stablehlo

`stablehlo` implements the StableHLO dialect for
[pliron](https://crates.io/crates/pliron).

## Public API

The public surface is organized by purpose: `ops` for operation authoring,
`types` and `attributes` for native IR records, `inference` for standalone
shape/type rules, `verification` for complete-IR checks, and `integer` and
`sparse` for their corresponding value representations. The optional `mlir`
module provides interoperability. `pliron`, `Error`, and `Result` are re-exported
so callers use the same framework types as the dialect.

Use typed operation accessors instead of internal dictionary storage names.
Parser state, verifier implementation modules, enum-spelling machinery, and
constructor adapters are private. Enum parsing errors are exposed as
`attributes::EnumParseError`. Public attribute/type fields remain available
for explicit IR construction; their invariants must still be verified.

Both dialect crates reject undocumented public items, unreachable public
declarations, and public signatures containing private types at compile time.

## Example: `((x + y) * 2.0).exp()`

This builds a rank-1 `tensor<4xf32>` function, checks each operation, and verifies the complete module:

```rust
use pliron::{
    common_traits::Verify,
    context::Context,
    op::Op,
    r#type::Typed,
};
use stablehlo::{
    ops::{AddOp, ConstantOp, ExponentialOp, FuncOp, ModuleOp, MultiplyOp, ReturnOp},
    types::{function_type, tensor, ElementType},
    verification::verify_module_fast,
};

let mut context = Context::new();
let module = ModuleOp::new(&mut context);
let tensor_type = tensor(&context, &[4], ElementType::F32);
let signature = function_type(&context, vec![tensor_type, tensor_type], vec![tensor_type]);
let function = FuncOp::new(&mut context, "pipeline".into(), signature)
    .expect("valid StableHLO function signature");
let block = function.entry_block(&context).expect("function has an entry block");
let arguments = function.arguments(&context);
let x = arguments[0];
let y = arguments[1];

let add = AddOp::new_inferred(&mut context, x, y).expect("matching tensor shapes");
let sum = add.get_operation().deref(&context).get_result(0);
add.verify(&context).expect("valid addition");
add.get_operation().insert_at_back(block, &context);

let two = ConstantOp::new(
    &mut context,
    tensor_type,
    vec![2.0f32.to_bits() as u64],
    true,
);
let two_value = two.get_operation().deref(&context).get_result(0);
two.verify(&context).expect("valid splat constant");
two.get_operation().insert_at_back(block, &context);

let multiply = MultiplyOp::new_inferred(&mut context, sum, two_value)
    .expect("matching tensor shapes");
let product = multiply.get_operation().deref(&context).get_result(0);
multiply.verify(&context).expect("valid multiplication");
multiply.get_operation().insert_at_back(block, &context);

let exponential = ExponentialOp::new_inferred(&mut context, product)
    .expect("valid exponential input");
let result = exponential.get_operation().deref(&context).get_result(0);
exponential.verify(&context).expect("valid exponential");
exponential.get_operation().insert_at_back(block, &context);

let ret = ReturnOp::new(&mut context, result);
ret.verify(&context).expect("valid return");
ret.get_operation().insert_at_back(block, &context);
function
    .get_operation()
    .insert_at_back(module.body_block(&context).expect("module has a body block"), &context);
verify_module_fast(&context, module.get_operation()).expect("valid StableHLO module");
```

## Construction and verification

Explicit operation constructors preserve the result types and attributes supplied
by the caller. They create IR; they do not certify that the operation satisfies
StableHLO's constraints. Inferred constructors return `Result` and check the
type and shape requirements documented for each operation. Insert the resulting
operations into a computation and verify the complete module before consuming it.

Fallible IR APIs use `stablehlo::Result<T>` and `stablehlo::Error`, re-exports of
pliron's shared result and error envelope. Text parsing keeps
small, specific standard error types; those errors also convert into the shared
envelope with `?`, preserving their typed payloads and rejected input where
provided by the parser.

Use `Diagnostic::from_error(&error)` rather than matching error text. It traverses
both standard source chains and framework payloads; pliron's envelope does not
itself expose its payload through `std::error::Error::source`. A result of `None`
means the error has a different payload, not that it can be ignored. Leaf parser
errors can be inspected through `error.err.downcast_ref`.
`RecoveryCategory` distinguishes invalid input, unsupported features, and internal
invariant failures. Correct the reported constraint or select a supported
representation before retrying; report invariant failures as implementation bugs.
Recovery is an application decision, not an automatic retry or rollback guarantee.

```rust
use stablehlo::{attributes::Precision, Error, Result};

fn precision(text: &str) -> Result<Precision> {
    Ok(text.parse()?)
}

let error: Error = precision("invalid").unwrap_err();
let details = error.err
    .downcast_ref::<stablehlo::attributes::EnumParseError>()
    .expect("the parser preserves its typed error");
assert_eq!(details.enum_name(), "precision");
assert_eq!(details.spelling(), "invalid");
```

Type handles are interned in a context. Handles and values passed together must
belong to that context. Public raw attributes and type records permit explicit IR
construction; verify independently constructed records before using them.

## MLIR interoperability

Enable the optional feature to export authored pliron IR as StableHLO MLIR:

```toml
[dependencies]
stablehlo = { version = "0.3", features = ["mlir-interop"] }
```

```rust
# #[cfg(feature = "mlir-interop")]
# fn example() -> Result<(), stablehlo::mlir::TextError> {
use stablehlo::{mlir, pliron::context::Context};

let mut context = Context::new();
let module = mlir::import(&mut context, "module {}")?;
let text: String = mlir::export(&context, &module)?;
# Ok(())
# }
```

## License

Apache-2.0. See the `LICENSE` file distributed with this crate.
