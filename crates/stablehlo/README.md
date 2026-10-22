# stablehlo

`stablehlo` StableHLO dialect for [pliron](https://crates.io/crates/pliron).

Public APIs are grouped under `attributes`, `types`, `integer`, `sparse`,
`ops`, `inference` and `verification`. Dictionary keys used in explicit
lookups are centralized in `attributes::keys`; generated operation attributes
retain their typed accessors, and serialized attribute names are unchanged.
Domain directories use `mod.rs` roots, and integration tests are grouped under
matching domain directories.

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

## License

Apache-2.0. See the `LICENSE` file distributed with this crate.
