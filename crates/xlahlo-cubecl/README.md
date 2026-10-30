# xlahlo-cubecl

`xlahlo-cubecl` verifies native `xlahlo` modules, lowers supported operations to
CubeCL IR, and produces owned, caller-executable `ExecutionPlan`s. It is a
compiler and execution-plan library, not a device runtime: compilation does not
allocate device memory, upload data, or launch kernels.

## Getting started

The crate requires Rust 1.95 or newer. Version 0.1 uses `xlahlo` 0.1, Pliron
0.19, and exactly CubeCL 0.11.0. Applications using the extension traits should
use compatible versions of these dependencies so their IR types agree.

```toml
[dependencies]
xlahlo-cubecl = "0.1"
xlahlo = "0.1"
pliron = { version = "0.19", default-features = false }
```

This complete example compiles an identity module. The returned tensor aliases
the input, so no kernel dispatch or new allocation is necessary.

```rust
use pliron::{
    basic_block::BasicBlock,
    builtin::{ops::ModuleOp, types::FunctionType},
    context::Context,
    identifier::Identifier,
    op::Op,
};
use xlahlo::{
    attributes::Index,
    ops::{ComputationOp, ReturnOp, control::ParameterOp},
    shape,
    types::ElementType,
};
use xlahlo_cubecl::{ModuleCompiler, OutputContract};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut context = Context::new();
    xlahlo::register(&mut context);
    let ty = shape::array(&context, ElementType::F32, vec![4])?;
    let signature = FunctionType::get(&context, vec![ty], vec![ty]).into();
    let module = ModuleOp::new(&mut context, Identifier::new("identity"));
    xlahlo::ir::set_entry(
        &context, module.get_operation(), Identifier::new("main"),
    )?;
    let container = module.get_operation().deref(&context)
        .get_region(0).deref(&context).get_entry_block().unwrap();
    let main = ComputationOp::new(&mut context, Identifier::new("main"), signature);
    main.get_operation().insert_at_back(container, &context);
    let block = BasicBlock::new(&mut context, None, vec![]);
    block.insert_at_back(main.body(&context), &context);
    let parameter = ParameterOp::new(&mut context, ty, vec![], Index(0));
    parameter.get_operation().insert_at_back(block, &context);
    let value = parameter.result(&context);
    ReturnOp::new(&mut context, value)
        .get_operation().insert_at_back(block, &context);

    let compiler = ModuleCompiler::builder().build();
    let plan = compiler.compile(&context, module.get_operation())?;
    assert!(!plan.has_device_work());
    assert!(matches!(
        plan.output_contract(0)?,
        OutputContract::Buffer { alias: true, .. }
    ));
    Ok(())
}
```

The context access above unwraps the body block created by `ModuleOp::new`;
fallible shape construction, entry selection, and compilation propagate errors.
For existing programs, pass the verified native module root and its owning
context to `ModuleCompiler::compile`. The compiler performs native verification,
lowering, plan validation, and generated-kernel validation before returning.

## Supported contracts and limits

Support depends on the complete operation contract, not just its opcode.
Successful XLAHLO verification does not imply backend support or device support.

| Area | Current lowering scope |
| --- | --- |
| Tensor storage | Static shapes, dense buffers, checked logical views, explicit entry-layout realization, constants, and empty tensors. Runtime-dependent dimensions are rejected rather than guessed. |
| Scalar computation | Admitted arithmetic, comparisons, conversions, selection, and scalar DAG fusion. Supported primitives and numerical attributes vary by operation. |
| Movement and indexing | Admitted reshape, transpose, reverse, slice, broadcast, copy, concatenate, padding, dynamic slice/update, and gather contracts. Dynamic indices do not imply dynamic tensor shapes. |
| Numerical and control operations | Restricted dot, reduction, sort, scatter, scan, and conditional/while bodies. These are not blanket implementations of every XLA configuration; unsupported bodies or metadata return errors. |
| Specialized formats | Complex storage uses real component lanes; packed F4 uses byte/nibble addressing. Their admitted operations are narrower than ordinary scalar formats, and neither supports the generic tensor-argument metadata ABI. |
| Extensions | Module-level strategies, per-operation strategies, pure scalar/ordered-fold callbacks, target passes, and optional fusion budgets. |

This is not a complete XLA executor, JAX backend, or device-capability checker.
The selected CubeCL target must support the emitted primitives, address type,
workgroup dimensions, and resource requirements. Compilation here does not
replace target compilation or runtime launch validation.

## Executing a plan

The application owns the CubeCL runtime, allocation handles, transfers, dispatch,
synchronization, and error reporting. It must honor the following contracts:

1. Bind inputs in their declared `BufferRole::Input` order. Use
   `BufferDescriptor::kernel_buffer_geometry` for physical registration: complex
   lanes and packed F4 bytes are not one pointer element per logical element.
2. Allocate each physical `allocation_root` once and honor its lifetime. Private
   storage versions inherit the same allocation without copying; use
   `KernelStep::effective_access` for physical memory effects.
3. Schedule `buffer_initializers` before their first consumer. Upload their exact
   bytes, retain the owner until the transfer is safe, and propagate allocation or
   transfer failures without dispatching dependent work.
4. Visit kernels in schedule order. Respect skipped steps, static launch geometry,
   optional `device_count` windows, binding order, tensor metadata, and read
   dependencies. A device-controlled step must not be mistaken for a skip simply
   because its static `cube_count` is absent.
5. Settle every `checked_completion` record before dependent work or publication.
   The five-word native-endian U32 record is successful only when all words are
   zero. Pending, unknown, partially written, descending-offset, out-of-bounds,
   and counter-overflow records are errors, not permission to continue.
6. Publish outputs through `output_contract`. Retain the whole backing allocation
   for aliases/views and apply element offsets exactly once. Signed affine strides
   and composed maps must be represented faithfully. Static outputs own their
   bytes and need no plan allocation, but uploading them is a runtime decision.

`MemoryRequirements` and `InitializerRequirements` are analytical requirements,
not measured device usage or evidence of successful allocation/upload. Allocator
alignment, hidden workspace, asynchronous overlap, and target limits remain
application responsibilities.

The plan owns kernel artifacts and metadata independently of the source context.
Extension implementations must uphold that ownership: do not retain source IR
pointers inside callbacks that survive compilation.

## Errors and recovery

`ModuleCompileError` is the compilation boundary; `PlanError` and
`KernelDefinitionError` describe plan and kernel failures. Use
`unsupported_kind()` to select an intentional compatibility fallback without
parsing diagnostic text. `None` means the error is not classified as a supported
fallback condition; it must not be silently treated as one.

`ModuleCompileError::Verification` retains the original `pliron::result::Error`
as its standard source. Inspect that payload with `xlahlo::Diagnostic::from_error`
to recover nested native diagnostics. Pliron's own wrapper does not expose every
payload through `std::error::Error::source`, so source-chain traversal alone is
not sufficient. `KernelOverrideError::Lowering` also exposes its backend source.
Extension callbacks that return `String` supply diagnostic text, not typed source
errors.

Compilation discards an unsuccessful in-progress plan. Individual builder and
emitter calls do not promise transactional rollback; discard partially built
artifacts after failure. A public accessor requiring a plan-local buffer/binding
index may panic when given an invalid or foreign identifier; its documentation
states that precondition.

Prefer `KernelStep::define_checked` for recoverable kernel-definition errors.
CubeCL's infallible `CubeKernel::define` adapter must panic if definition fails.
The module compiler preflights definitions, but stateful callbacks can still
invalidate later regeneration; keep their configuration deterministic and fixed.

## Extending lowering

Create a `ModuleCompilerBuilder` and register extensions before calling `build`.
The built-in module strategy has the lowest priority. Higher-priority matches
win; equal highest-priority matches fail rather than depending on registration
order. Matcher errors abort compilation rather than becoming non-matches.

An `OperationLoweringStrategy` receives preplanned operand/result buffers and
appends kernels to the shared schedule. Opt into indexed operands only when the
strategy applies their `OperandView` mappings. Optional `ScalarLoweringBody` and
`OrderedReductionLoweringBody` callbacks must emit their actual pure computation,
own captured state, and provide complete semantic cache keys.

`CubeclTargetPass` callbacks run on each generated kernel in ascending priority,
with registration order breaking ties. Their cache keys must include every
configuration value that affects emitted code. They may be invoked again during
definition regeneration. Likewise, custom `PlannedKernel` metadata must identify
all code-generation choices. Uploaded-data caches additionally need initializer
content keys and the plan's buffer/schedule mapping.

`TargetFusionPolicy` and `FusionBudget` control profitability only after semantic
legality. They cannot authorize unsupported operations, discard explicit strategy
selection, or substitute guessed target capabilities.

## License

Licensed under Apache-2.0. The published crate includes the full text in `LICENSE`.
