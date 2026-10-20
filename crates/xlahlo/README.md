# xlahlo

`xlahlo` is an OpenXLA HLO dialect for [pliron](https://crates.io/crates/pliron).

## Example

Construct a computation that adds a vector parameter to itself:

```rust
use xlahlo::{
    attributes::Index,
    ops::{ComputationOp, ReturnOp, control::ParameterOp, elementwise::AddOp},
    pliron::{
        basic_block::BasicBlock, builtin::types::FunctionType,
        context::Context, identifier::Identifier, op::Op,
    },
    types::{ElementType, TensorType},
    verification::verify,
};

let mut ctx = Context::new();
xlahlo::register(&mut ctx);
let ty = TensorType::get(&ctx, ElementType::F32, vec![16])?;
let signature = FunctionType::get(&ctx, vec![ty], vec![ty]).into();
let computation = ComputationOp::new(&mut ctx, Identifier::new("main"), signature);
let body = BasicBlock::new(&mut ctx, None, vec![]);
body.insert_at_back(computation.body(&ctx), &ctx);

let parameter = ParameterOp::new(&mut ctx, ty, vec![], Index(0));
parameter.get_operation().insert_at_back(body, &ctx);
let x = parameter.result(&ctx);
let add = AddOp::new(&mut ctx, ty, vec![x, x]);
add.get_operation().insert_at_back(body, &ctx);
let sum = add.result(&ctx);
ReturnOp::new(&mut ctx, sum).get_operation().insert_at_back(body, &ctx);

verify(&ctx, computation.get_operation())?;
# Ok::<(), xlahlo::pliron::result::Error>(())
```

Constructors accept explicit result types and typed attributes. They do not
implicitly verify the graph; use the checked `inference` functions when computing
result types, then verify before consuming the IR.

## API organization

| Namespace                                             | Contents                                                               |
| ----------------------------------------------------- | ---------------------------------------------------------------------- |
| `types`, `primitive`                                  | Interned IR types, `ElementType` and the complete `PrimitiveType` enum |
| `shape`                                               | Shape construction, borrowed queries, layouts and allocation sizes     |
| `literal`                                             | Typed literal storage, borrowed sub-literals and logical-bit equality  |
| `attributes`                                          | Geometry, precision, comparison and instruction metadata               |
| `sharding`, `replica_groups`, `original_value`        | Distributed assignments, groups and provenance                         |
| `opcode`                                              | Opcode names, operand arities and capabilities                         |
| `ops`                                                 | `ComputationOp`, `ReturnOp` and `HloOpInterface`                       |
| `ops::{elementwise,movement,numerical,control}`       | Concrete instruction families                                          |
| `inference::{elementwise,movement,numerical,control}` | Checked shape inference                                                |
| `ir`                                                  | Module configuration, entry points, aliases, donors and schedules      |
| `verification`                                        | Whole-graph verification and caller policies                           |
| `effects`                                             | Native pliron `SideEffects` analysis over a module snapshot            |

## Errors and handle safety

Fallible APIs return `pliron::result::Result<T>`. Its `Error` retains the error
category, source location and backtrace; `error.err.downcast_ref::<Diagnostic>()`
exposes typed dialect details without parsing message text. `Diagnostic` is
available at the crate root. Whole-graph instruction failures include the
operation name and a source diagnostic. Framework-generated parser and interface
errors retain their native payloads. Use pliron's `Printable::disp` to format
errors with context-aware locations.

Invalid input returns an error without consuming the context, so callers can
repair the IR and verify again. Production dialect code denies Clippy's
`unwrap_used` and `expect_used` lints. As with pliron itself, handles must belong
to the supplied context and remain live; dereferencing erased or foreign-context
handles is outside this contract. Constructors and SSA-result accessors assume
their documented structural preconditions.

## License

See the workspace license file.
