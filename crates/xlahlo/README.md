# xlahlo

`xlahlo` is an OpenXLA HLO dialect for [pliron](https://crates.io/crates/pliron).

## Installation

```toml
[dependencies]
xlahlo = "0.1"
```

Requires Rust 1.95 or later. The crate re-exports its compatible pliron version
as `xlahlo::pliron`; use that path when constructing contexts, operations, and
type handles to avoid mixing incompatible framework versions.

## Rust API organization

The crate-root facade follows HLO's Rust domains rather than OpenXLA's C++
namespace hierarchy. Public APIs have one canonical path:

| Module         | Responsibility                                                  |
| -------------- | --------------------------------------------------------------- |
| `attributes`   | HLO attribute types and attribute keys                          |
| `inference`    | Checked shape and operation inference                           |
| `ir`           | Graph metadata, provenance, and graph transformations           |
| `literal`      | Literal storage and `literal::semantics`                        |
| `ops`          | Operation types grouped by semantic category                    |
| `opcode`       | HLO opcode model                                                |
| `scalar`       | `scalar::semantics`, the native scalar API                      |
| `shape`        | Dimensions, layouts, and shape construction and queries         |
| `sharding`     | Sharding types and replica-group concepts                       |
| `types`        | Element types, primitive types, and primitive properties        |
| `verification` | Verification entry points and options                           |
| `evaluator`    | Pure native execution for supported operations                  |
| `analysis`     | Effects, instruction classification, and `analysis::while_loop` |

Numerical kernels, parser plumbing, and focused implementation modules remain
private. The public facade selectively re-exports supported APIs at their
domain paths without compatibility aliases at the crate root.

Shape construction and queries are functions in the `shape` module, such as
`shape::array`, `shape::dimensions`, and `shape::with_layout`. A shape is an
interned pliron type handle, not a separate namespace object. Transformations
return a new interned handle and leave their input unchanged.

`ElementType` and `PrimitiveType` expose allocation-free `as_str` methods and
implement `Display` and `FromStr` for exact lowercase HLO names. Primitive
identifiers use checked `TryFrom<i32>` conversion; converting a structural
primitive to an array element is also checked. Dialect attribute printing is a
separate framework representation and retains its registered names and spellings.

## Construction, verification, and execution

Explicit operation constructors build detached IR and retain the supplied
result type; they do not imply successful verification. Use `Verify::verify`
for local instruction checks and `verification::verify` for module-wide
ownership, call, channel, and shape constraints. Type and literal constructors
that return `Result` validate their documented invariants before interning or
returning the value. Public raw attribute fields remain available for IR
construction and must be verified before use.

Fallible IR APIs use `xlahlo::Result<T>` and `xlahlo::Error`, re-exports of
pliron's shared result and error envelope. All public leaf parsing and numeric
conversion errors, as well as `evaluator::EvaluationError`, convert into this
envelope with `?` while retaining their typed payloads. Callers do not need to
flatten errors to strings or introduce a different result type for every helper.

Use `Diagnostic::from_error(&error)` rather than matching diagnostic text. It
traverses standard source chains and framework `err` payloads; pliron's envelope
does not itself expose its payload through `std::error::Error::source`. `None`
means the error has another payload, not success. Leaf conversion and evaluator
errors can be inspected through `error.err.downcast_ref`.

`Diagnostic::recovery_category` distinguishes invalid input from unsupported
features. `EvaluationError::recovery_category` additionally distinguishes missing
values or floating-point context from an explicit loop execution limit. Correct
the input, supply the missing state, select a supported execution strategy, or
adjust the configured limit before retrying. Recovery is not an automatic retry
or a blanket rollback guarantee; graph-edit APIs document their staging policy.

Evaluation exposes explicit known, unknown, and unvisited value states.
Unsupported operations return errors; the evaluator is not a complete OpenXLA
execution backend.

## API migration

The Rust API intentionally uses semantic names rather than mirroring C++
utility classes. Shape operations are available directly under `shape`:
`Shape::array(...)` becomes `shape::array(...)`. Nonmutating dimension
transformations use `shape::with_dimension_bound` and
`shape::with_dynamic_dimension`, not setter names.

Category enums use `ComparisonKind`, `ChannelKind`, `FftKind`, `PaddingKind`,
`ShardingKind`, and `DimLevelKind`. Genuine type descriptors retain names such
as `ElementType`, `PrimitiveType`, and `TensorType`. These Rust names do not
change external dialect attribute identifiers.

For numeric element formats, replace `ElementType::from_hlo_name(text)` with
`text.parse::<ElementType>()`, and use `as_str()` or `to_string()` instead of
`hlo_name()`. Parsing is case-sensitive. `PrimitiveType` includes the `Invalid`
sentinel for representing primitive identifiers, but array construction rejects
it and every structural primitive.

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

## License

Apache-2.0. See the `LICENSE` file distributed with this crate.
