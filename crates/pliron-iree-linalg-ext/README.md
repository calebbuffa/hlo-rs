# pliron-iree-linalg-ext

Native [pliron](https://crates.io/crates/pliron) operations for IREE's
[`iree_linalg_ext`](https://iree.dev/reference/mlir-dialects/LinalgExt/)
dialect on tensors.

`iree_linalg_ext` is a separate MLIR dialect, not part of upstream `linalg`.
IREE registers it next to `linalg` for operations that do not fit the
structured-op model, such as sorts, scatters, attention and FFT. It reuses the
[`pliron-linalg`](https://crates.io/crates/pliron-linalg) indexing-map,
iterator-type and integer-array attributes.

- **Sorting and selection**: `sort`, `scan`, `topk`, `topk_v2`, `arg_compare`.
- **Indexed data movement**: `scatter`, `gather`, `map_load`, `map_store`.
- **Grouped contractions**: `group_matmul`, `group_mmt4d`.
- **Attention**: `attention`, `online_attention`, `exp_reduction`, with an
  optional `decomposition_config` dictionary.
- **Region operations**: `custom_op`, plus the body operations `yield` and
  `index`.
- **Signal processing**: `fft`.
- **Quantization**: `quantize_affine`, `dequantize_affine`.
- **Winograd**: `winograd.input_transform`, `winograd.filter_transform`,
  `winograd.output_transform`.
- **Convolution lowering**: `im2col`, with mixed static/dynamic kernel sizes,
  offsets, output sizes and padding.

Operations use destination-passing style on ranked tensors from the
[`stablehlo`](https://crates.io/crates/stablehlo) crate: each result has the
type of its `outs` operand. Upstream also accepts memrefs; this crate models
tensors only. Region bodies are built from
[`pliron-arith`](https://crates.io/crates/pliron-arith) operations.

Mixed static/dynamic `im2col` lists use `ops::MixedIndex`. In storage, a
dynamic entry is `ops::DYNAMIC_INDEX` (`i64::MIN`, equal to MLIR's
`ShapedType::kDynamic`), and its value is the next matching `index` operand.

## Installation

```toml
[dependencies]
pliron-iree-linalg-ext = "0.1"
```

Default features are empty. Enable `mlir-interop` for textual MLIR export.

## Example

Build a sort with a verified scalar comparator region:

```rust
use pliron_iree_linalg_ext::{
    ops::SortOp,
    pliron::{
        basic_block::BasicBlock, builtin::types::FP32Type,
        common_traits::Verify, context::Context, op::Op,
    },
    pliron_arith::{attributes::CmpFPredicate, ops::CmpFOp},
    stablehlo::types::TensorType,
};

let ctx = &mut Context::new();
let ty = TensorType::from_element(ctx, FP32Type::get(ctx).into(), vec![8]);
let block = BasicBlock::new(ctx, None, vec![ty, ty]);
let args: Vec<_> = block.deref(ctx).arguments().collect();
let sort = SortOp::build(
    ctx, vec![args[0]], vec![args[1]], 0,
    Box::new(|ctx, block, args| {
        let cmp = CmpFOp::new(ctx, CmpFPredicate::Olt, args[0], args[1]);
        cmp.get_operation().insert_at_back(block, ctx);
        Ok(vec![cmp.result(ctx)])
    }),
)?;
sort.get_operation().insert_at_back(block, ctx);
sort.verify(ctx)?;
# Ok::<(), pliron_iree_linalg_ext::Error>(())
```

The ordered comparison is suitable for finite values; define an explicit
NaN policy if the frontend needs a total floating-point ordering.

## MLIR interoperability

Enable the `mlir-interop` feature to implement `ToMlirOp` for every operation.
The default export uses upstream custom assembly, including nested regions,
mixed `im2col` index lists, and the dotted Winograd operation names. Use
`mlir::print_generic` for explicit generic-form export with upstream property
names. These are export paths, not an IREE assembly parser.

`attributes::SplitReductionMappingAttr` represents
`#iree_linalg_ext.split_reduction_mapping<N>` and checks mapping-list invariants.
Configuration dictionaries can contain `stablehlo::attributes::OpaqueAttr` for
foreign IREE attributes, such as GPU lowering configurations; their syntax is
validated by the shared MLIR exporter.

Winograd verification does not impose a blanket positive tile/kernel
restriction absent from upstream. Shape arithmetic is widened to avoid integer
overflow, and a zero tile divisor is reported as an error only when computing
a statically known image dimension.

```toml
[dependencies]
pliron-iree-linalg-ext = { version = "0.1", features = ["mlir-interop"] }
```

Export a complete mixed-dialect `stablehlo::ops::ModuleOp` using
`stablehlo::mlir::export(ctx, &module)`, then let IREE parse, verify, and compile
it for the selected target. Generic and custom exporters both reject unknown,
mistyped, or misplaced properties.

## API contract

`new` allocates without verification. Region `build` constructors append
`iree_linalg_ext.yield` and verify the completed operation. Allocation is not
transactional: a failed builder may leave operations in the context.
Operand accessors assume valid arity. Whole-module verification is required
for SSA scope/dominance; runtime-dependent constraints remain the caller's
responsibility. Use `Diagnostic::from_error` for structured dialect errors.

## License

Original code and IREE-derived portions are Apache-2.0; LLVM-derived portions
retain Apache-2.0 with LLVM Exceptions. The package includes `LICENSE`,
`LICENSE-IREE`, `LICENSE-LLVM`, and `NOTICE`.
