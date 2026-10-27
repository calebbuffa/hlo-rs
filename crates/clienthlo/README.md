# CHLO

`clienthlo` is a pliron implementation of OpenXLA's CHLO (client HLO)
dialect. CHLO adds client-oriented, higher-level tensor operations alongside
StableHLO operations in StableHLO functions.

All operations use the `chlo.` dialect namespace.

## API modules

The public API is organized into `attributes`, `inference`, `ops`, and `passes`.
`clienthlo::stablehlo` and `clienthlo::pliron` re-export the underlying dialect
and IR framework. Applications can depend on `clienthlo` alone; StableHLO
operations are not flattened into the CHLO namespace.
Integration tests are grouped by API domain under `tests/`.

The public modules expose authoring records, standalone inference rules, and the
legalization pass, not their implementations. Tensor snapshots, element-domain
checks, broadcast storage keys, and lowering builders are private. Inspect tensor
metadata through `stablehlo::types::{TensorType, UnrankedTensorType}` and use typed
operation accessors to change attributes. Enum conversion errors are available as
`attributes::{EnumParseError, EnumValueError}`; the implementation module is private.
Public attribute fields are value records for explicit construction, not a
substitute for verification. Both crates enforce documentation and visibility
boundaries with compiler lints.

The checked-in upstream CHLO catalog has 51 operations, 32 operation properties,
four dialect attribute kinds, and 15 enum cases. All have Rust representations.
Tests compare compiled operation identifiers, native property storage keys,
attribute parameters, and enum spellings/encodings against resolved upstream
TableGen definitions. These are representation guards, not exhaustive proofs of
verifier or inference equivalence.

## Building and verifying operations

`new` constructors accept explicit result types and allocate IR without checking
it. This is useful for parsers, transformations, and partially constructed IR,
but callers must run `pliron::common_traits::Verify::verify` before relying on
the operation's validity.

`new_inferred` constructors derive result types from operands and attributes.
They check the constraints required for inference, but do not universally perform
full verification. For example, broadcasting can produce an unranked result
when the output shape cannot be established; success does not prove a runtime
broadcast is valid. Constructors that also verify their output document this
explicitly. Fallible construction is not transactional: an error after IR
allocation can leave allocated objects in the context.

Operand and result accessors require the operation's declared arity; malformed
IR with missing values can panic. Typed attribute accessors usually return `None`
for missing or incorrectly typed attributes. Some accessors apply documented
dialect defaults. Raw attribute setters do not revalidate the operation.

```rust
use clienthlo::{ops::{BroadcastAddOp, ErfOp}, pliron, stablehlo};
use pliron::{common_traits::Verify, context::Context, op::Op, r#type::Typed};
use stablehlo::{ops::ConstantOp, types::{ElementType, tensor}};

let mut ctx = Context::new();
let ty = tensor(&ctx, &[4], ElementType::F32);
let x = ConstantOp::new(&mut ctx, ty, vec![0x3f80_0000], true);
let x = x.get_operation().deref(&ctx).get_result(0);
let erf = ErfOp::new_inferred(&mut ctx, x)?;
let sum = BroadcastAddOp::new_inferred(&mut ctx, x, x, None)?;
erf.verify(&ctx)?;
sum.verify(&ctx)?;
assert_eq!(erf.result(&ctx).get_type(&ctx), sum.get_operation().deref(&ctx).get_result(0).get_type(&ctx));
# Ok::<(), pliron::result::Error>(())
```

## Legalizing CHLO to StableHLO

`passes::ChloToStablehloPass` is a standard pliron pass, implemented with its
dialect-conversion driver and rewriter. It recursively rewrites CHLO operations,
leaves existing non-CHLO operations in place, and verifies the whole result.
Success guarantees **no CHLO remains**. Replacements use only StableHLO, with no
new `arith`, `tensor`, or `shape` operations or backend custom calls. Existing
foreign operations are not legalized by this pass.

```rust
use clienthlo::{
    ops::BroadcastAddOp,
    passes::ChloToStablehloPass,
    pliron::{
        context::Context, op::Op,
        pass::{AnalysisManager, Pass, Passes},
    },
    stablehlo::{
        ops::{FuncOp, ModuleOp, ReturnOp},
        types::{ElementType, function_type, tensor},
    },
};

let mut ctx = Context::new();
let module = ModuleOp::new(&mut ctx);
let ty = tensor(&ctx, &[4], ElementType::F32);
let scalar = tensor(&ctx, &[], ElementType::F32);
let signature = function_type(&ctx, vec![ty, scalar], vec![ty]);
let function = FuncOp::new(&mut ctx, "main".into(), signature)?;
function.get_operation().insert_at_back(module.body_block(&ctx).unwrap(), &ctx);
let block = function.entry_block(&ctx).unwrap();
let args = function.arguments(&ctx);
let sum = BroadcastAddOp::new_inferred(&mut ctx, args[0], args[1], None)?;
let result = sum.get_operation().deref(&ctx).get_result(0);
sum.get_operation().insert_at_back(block, &ctx);
ReturnOp::new(&mut ctx, result).get_operation().insert_at_back(block, &ctx);

let mut pipeline = Passes::default();
pipeline.add_pass(ChloToStablehloPass);
pipeline.run(module.get_operation(), &mut ctx, &mut AnalysisManager::default())?;
# #[cfg(feature = "mlir-interop")]
let text = clienthlo::stablehlo::mlir::export(&ctx, &module)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Enable `clienthlo`'s optional `mlir-interop` feature to enable
`clienthlo::stablehlo::mlir`. It forwards to StableHLO's feature without adding a
second interoperability implementation. Mixed CHLO/StableHLO modules must be
legalized successfully before StableHLO MLIR export.

## Attributes and enum encodings

Comparison directions, comparison semantics, and precision requests are Rust
enums. `Display`, `FromStr`, and the IR printer use their canonical, case-sensitive
CHLO symbols rather than Rust variant names. For example, `ComparisonDirection::Eq`
prints as `EQ` and `ComparisonKind::NoType` prints as `NOTYPE`.
The Rust name `ComparisonKind` does **not** change the CHLO wire name:
the attribute remains `chlo.comparison_type`.

Conversion to `u64` is lossless. Conversion from `u64` rejects undefined encodings
with `EnumValueError`, which implements `std::error::Error` and exposes the enum
name and rejected value. Text parsing returns `EnumParseError`; its `enum_name()`
and `spelling()` accessors preserve the rejected input for diagnostics.

```rust
use clienthlo::attributes::{
    ComparisonDirection, Precision, precision_config, read_precision_config,
};

let direction = ComparisonDirection::Eq;
assert_eq!(direction.to_string(), "EQ");
assert_eq!("EQ".parse::<ComparisonDirection>()?, direction);
assert_eq!(ComparisonDirection::try_from(u64::from(direction))?, direction);

let config = precision_config([Precision::High, Precision::Highest]);
assert_eq!(read_precision_config(&config), vec![Precision::High, Precision::Highest]);
# Ok::<(), Box<dyn std::error::Error>>(())
```

`read_precision_config` is infallible because its input is already a typed
attribute. By contrast, `RaggedDotOp::precision_config` returns an `Option` to
distinguish an absent attribute from an explicitly stored empty list.
`TopKOp::is_stable` returns a `bool`, applying CHLO's `true` default when no typed
boolean attribute is available.

## Inference contracts

The `inference` module exposes individual shape and type rules, not a general IR
validator. Negative dimension sizes represent dynamic dimensions. A rank-zero
tensor has an empty shape; an unranked tensor has no known rank.

Default binary broadcasting aligns trailing axes using NumPy rules. Explicit
broadcast dimensions map the lower-rank operand's axes into the higher-rank
operand; on equal ranks, the left operand is mapped into the right. Shape-only
helpers do not check element domains, and explicit-mapping helpers do not check
axis uniqueness or ordering. Shape-returning helpers return `None` for
incompatible dimensions. Operation builders can infer an unranked type for an
indeterminate or incompatible broadcast; their inference success does not
replace operation verification.

## Diagnosing invalid IR

Inference and verification use `clienthlo::Result<T>` and `clienthlo::Error`,
re-exports of pliron's shared result and error envelope. Use
`Diagnostic::from_error(&error)` to inspect a domain violation or an operation
constraint failure without matching display text. This helper traverses standard
source chains and framework payloads, including wrapped errors. A result of
`None` means a different component or a leaf parser produced the error; it does
not mean success. Operation diagnostics
include a rendered location when available; inference may use a placeholder
because no source operation exists yet.

`Diagnostic::recovery_category()` returns `RecoveryCategory::InvalidInput`
for inference and verification violations and `RecoveryCategory::Unsupported`
for missing legalization support. Correct invalid types, attributes, or structure;
unsupported lowering requires an implemented decomposition or a supported
representation, not an unchanged retry. `InternalInvariant` is reserved for
implementation failures and is not emitted by current diagnostic variants.

Enum parsing and numeric conversion retain their small standard error types.
Both convert into the shared envelope with `?` without losing their typed
payloads, so callers do not need custom `map_err` adapters or a separate result
type for every helper. Inspect a leaf payload through `error.err.downcast_ref`.
The framework envelope does not expose its payload through the standard `source`
method; use that payload or `Diagnostic::from_error` explicitly. Recovery requires
correcting the reported input, not retrying unchanged IR, and does not imply that
earlier IR mutations were rolled back.

## License

Apache-2.0. See the `LICENSE` file distributed with this crate.
