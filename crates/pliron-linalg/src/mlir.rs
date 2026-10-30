//! MLIR textual export for `linalg`.
//!
//! Operations whose computation is implied by their name print in MLIR's
//! custom form, because their generic form would require the implicit body
//! region spelled out:
//!
//! ```text
//! %r = linalg.conv_2d_nhwc_hwcf {strides = dense<[2, 2]> : tensor<2xi64>} ins(%x, %w : T, U) outs(%o : V) -> V
//! %r = linalg.matmul indexing_maps = [..] ins(%a, %b : T, U) outs(%c : V) -> V
//! %r = linalg.elementwise <add> ins(%a, %b : T, T) outs(%c : T) -> T
//! %r = linalg.transpose ins(%x : T) outs(%y : U) permutation = [1, 0]
//! %r = linalg.softmax dimension(1) ins(%x : T) outs(%y : T) -> T
//! %r = linalg.pack %s padding_value(%p : f32) inner_dims_pos = [0] inner_tiles = [16] into %d : T -> U
//! ```
//!
//! Region operations (`generic`, `map`, `reduce`, `yield`, `index`) print in
//! generic form, translating typed storage to upstream property names:
//!
//! | storage key | MLIR property | text |
//! |---|---|---|
//! | `linalg_indexing_maps` | `indexing_maps` | `[affine_map<..>, ..]` |
//! | `linalg_iterator_types` | `iterator_types` | `[#linalg.iterator_type<parallel>, ..]` |
//! | `linalg_operand_segment_sizes` | `operandSegmentSizes` | `array<i32: ins, outs>` |
//! | `linalg_dimensions` | `dimensions` | `array<i64: ..>` |
//! | `linalg_index_dim` | `dim` | `N : i64` |
//! | `linalg_doc`, `linalg_library_call` | `doc`, `library_call` | `"..."` |

use crate::{
    attributes::{
        ElementwiseKindAttr, I64ArrayAttr, IndexingMapsAttr, IteratorTypesAttr, TypeFnAttr,
        WinogradFmrAttr,
    },
    ops::*,
};
use pliron::{
    attribute::Attribute,
    builtin::attributes::{OperandSegmentSizesAttr, StringAttr},
    common_traits::Named,
    context::{Context, Ptr},
    derive::op_interface_impl,
    op::Op,
    operation::Operation,
    printable::State,
    result::Result,
    r#type::Typed,
    value::Value,
};
use pliron_arith::mlir::print_generic;
use pliron_mlir_interop::{
    ToMlirOp,
    printers::{
        mlir_dense_array, mlir_string_literal, print_op_location, print_type, print_type_list,
    },
};
use stablehlo::attributes::{AffineMap, IndexAttr};
use std::fmt;

fn untranslatable(message: String) -> pliron::result::Error {
    pliron::input_error_noloc!(pliron_mlir_interop::Error::Untranslatable(message))
}

/// Formats affine maps as an MLIR array: `[affine_map<..>, ..]`.
///
/// # Errors
///
/// Returns an error when a map cannot be printed.
pub fn affine_maps_text(ctx: &Context, maps: &[AffineMap]) -> Result<String> {
    let maps = maps
        .iter()
        .map(|m| stablehlo::mlir::assembly::print_attribute(ctx, m))
        .collect::<Result<Vec<_>>>()?;
    Ok(format!("[{}]", maps.join(", ")))
}

fn list(values: &[i64]) -> String {
    values
        .iter()
        .map(i64::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

/// Translates one `linalg` storage key to an MLIR property of a generic-form
/// operation.
///
/// Returns `Ok(None)` for keys this crate does not own.
///
/// # Errors
///
/// Returns an error when an owned key holds an attribute of the wrong type.
pub fn property(
    ctx: &Context,
    key: &str,
    attr: &dyn Attribute,
) -> Result<Option<(&'static str, Option<String>)>> {
    let wrong = || untranslatable(format!("attribute {key} has an unexpected type"));
    let string = |name: &'static str| -> Result<(&'static str, Option<String>)> {
        let value = attr.downcast_ref::<StringAttr>().ok_or_else(wrong)?;
        Ok((name, Some(mlir_string_literal(value.as_str()))))
    };
    Ok(Some(match key {
        "linalg_indexing_maps" => (
            "indexing_maps",
            Some(affine_maps_text(
                ctx,
                &attr
                    .downcast_ref::<IndexingMapsAttr>()
                    .ok_or_else(wrong)?
                    .maps,
            )?),
        ),
        "linalg_iterator_types" => {
            let values = &attr
                .downcast_ref::<IteratorTypesAttr>()
                .ok_or_else(wrong)?
                .values;
            let text = values
                .iter()
                .map(|v| format!("#linalg.iterator_type<{}>", v.as_str()))
                .collect::<Vec<_>>()
                .join(", ");
            ("iterator_types", Some(format!("[{text}]")))
        }
        "linalg_operand_segment_sizes" => {
            let sizes = &attr
                .downcast_ref::<OperandSegmentSizesAttr>()
                .ok_or_else(wrong)?
                .0;
            ("operandSegmentSizes", Some(mlir_dense_array("i32", sizes)))
        }
        "linalg_dimensions" => {
            let values = &attr
                .downcast_ref::<I64ArrayAttr>()
                .ok_or_else(wrong)?
                .values;
            ("dimensions", Some(mlir_dense_array("i64", values)))
        }
        "linalg_index_dim" => {
            let value = attr.downcast_ref::<IndexAttr>().ok_or_else(wrong)?.value;
            ("dim", Some(format!("{value} : i64")))
        }
        "linalg_doc" => string("doc")?,
        "linalg_library_call" => string("library_call")?,
        _ => return Ok(None),
    }))
}

/// Translates one discardable-dictionary attribute of a custom-form operation
/// to `name = text`.
fn custom_attr(key: &str, attr: &dyn Attribute) -> Result<String> {
    let wrong = || untranslatable(format!("attribute {key} has an unexpected type"));
    let dense = |name: &str| -> Result<String> {
        let values = &attr
            .downcast_ref::<I64ArrayAttr>()
            .ok_or_else(wrong)?
            .values;
        Ok(format!(
            "{name} = dense<[{}]> : tensor<{}xi64>",
            list(values),
            values.len()
        ))
    };
    match key {
        "linalg_strides" => dense("strides"),
        "linalg_dilations" => dense("dilations"),
        "linalg_cast" => {
            let value = attr.downcast_ref::<TypeFnAttr>().ok_or_else(wrong)?.value;
            Ok(format!("cast = #linalg.type_fn<{}>", value.as_str()))
        }
        _ => Err(untranslatable(format!("unmapped attribute {key}"))),
    }
}

fn print_values(ctx: &Context, values: &[Value], f: &mut fmt::Formatter<'_>) -> Result<()> {
    for (i, value) in values.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(f, "%{}", value.unique_name(ctx))?;
    }
    Ok(())
}

fn print_typed(
    ctx: &Context,
    keyword: &str,
    values: &[Value],
    state: &State,
    f: &mut fmt::Formatter<'_>,
) -> Result<()> {
    if values.is_empty() {
        return Ok(());
    }
    write!(f, " {keyword}(")?;
    print_values(ctx, values, f)?;
    write!(f, " : ")?;
    print_type_list(ctx, values.iter().map(|v| v.get_type(ctx)), state, f)?;
    write!(f, ")")?;
    Ok(())
}

fn op_name(ctx: &Context, op: Ptr<Operation>) -> String {
    Operation::get_op_dyn(op, ctx).get_opid().to_string()
}

fn validate_attributes(ctx: &Context, op: Ptr<Operation>) -> Result<()> {
    let name = op_name(ctx, op);
    let mnemonic = name
        .strip_prefix("linalg.")
        .ok_or_else(|| untranslatable(name.clone()))?;
    let window = (mnemonic.starts_with("conv_")
        && !matches!(mnemonic, "conv_1d" | "conv_2d" | "conv_3d"))
        || mnemonic.starts_with("depthwise_conv_")
        || mnemonic.starts_with("pooling_");
    for (key, attr) in &op.deref(ctx).attributes.0 {
        let key = AsRef::<str>::as_ref(key);
        if key == "builtin_given_names" {
            continue;
        }
        let valid = match key {
            "linalg_cast" => {
                matches!(
                    mnemonic,
                    "copy" | "matmul" | "batch_matmul" | "batch_reduce_matmul" | "contract"
                ) && attr.downcast_ref::<TypeFnAttr>().is_some()
            }
            "linalg_strides" | "linalg_dilations" => {
                window && attr.downcast_ref::<I64ArrayAttr>().is_some()
            }
            "linalg_indexing_maps" => {
                matches!(
                    mnemonic,
                    "generic"
                        | "matmul"
                        | "batch_matmul"
                        | "batch_reduce_matmul"
                        | "contract"
                        | "scaled_contract"
                        | "elementwise"
                ) && attr.downcast_ref::<IndexingMapsAttr>().is_some()
            }
            "linalg_iterator_types" => {
                mnemonic == "generic" && attr.downcast_ref::<IteratorTypesAttr>().is_some()
            }
            "linalg_doc" | "linalg_library_call" => {
                mnemonic == "generic" && attr.downcast_ref::<StringAttr>().is_some()
            }
            "linalg_operand_segment_sizes" => {
                matches!(mnemonic, "generic" | "pack" | "unpack")
                    && attr.downcast_ref::<OperandSegmentSizesAttr>().is_some()
            }
            "linalg_dimensions" => {
                matches!(mnemonic, "broadcast" | "reduce")
                    && attr.downcast_ref::<I64ArrayAttr>().is_some()
            }
            "linalg_permutation" => {
                mnemonic == "transpose" && attr.downcast_ref::<I64ArrayAttr>().is_some()
            }
            "linalg_dimension" => {
                mnemonic == "softmax" && attr.downcast_ref::<IndexAttr>().is_some()
            }
            "linalg_index_dim" => mnemonic == "index" && attr.downcast_ref::<IndexAttr>().is_some(),
            "linalg_kind" => {
                mnemonic == "elementwise" && attr.downcast_ref::<ElementwiseKindAttr>().is_some()
            }
            "linalg_fmr" => {
                matches!(
                    mnemonic,
                    "winograd_filter_transform"
                        | "winograd_input_transform"
                        | "winograd_output_transform"
                ) && attr.downcast_ref::<WinogradFmrAttr>().is_some()
            }
            "linalg_inner_dims_pos" | "linalg_outer_dims_perm" | "linalg_static_inner_tiles" => {
                matches!(mnemonic, "pack" | "unpack")
                    && attr.downcast_ref::<I64ArrayAttr>().is_some()
            }
            _ => false,
        };
        if !valid {
            return Err(untranslatable(format!(
                "attribute {key} has the wrong type or owner on {name}"
            )));
        }
    }
    Ok(())
}

/// Layout of a destination-passing operation in custom form.
#[derive(Default)]
struct Custom<'a> {
    /// Text after the mnemonic, before the attribute dictionary.
    prefix: String,
    /// Storage keys printed by `prefix` or `suffix`.
    elided: &'a [&'a str],
    /// Text after `outs(..)`.
    suffix: String,
    /// Whether to print `-> result types`.
    arrow: bool,
}

/// Prints `%r = name prefix {attrs} ins(..) outs(..) suffix -> T` after
/// verifying the operation. The last operand is the `outs` operand.
fn print_custom(
    ctx: &Context,
    op: Ptr<Operation>,
    layout: Custom<'_>,
    state: &State,
    f: &mut fmt::Formatter<'_>,
) -> Result<()> {
    Operation::get_op_dyn(op, ctx).verify(ctx)?;
    let name = op_name(ctx, op);
    let raw = op.deref(ctx);
    let mut entries = raw.attributes.0.iter().collect::<Vec<_>>();
    entries.sort_by(|(a, _), (b, _)| a.as_ref().cmp(b.as_ref()));
    let mut attrs = Vec::new();
    for (key, attr) in entries {
        let key: &str = key.as_ref();
        if key == "builtin_given_names" || layout.elided.contains(&key) {
            continue;
        }
        attrs.push(
            custom_attr(key, &**attr)
                .map_err(|_| untranslatable(format!("unmapped attribute {key} on {name}")))?,
        );
    }
    let operands = raw.operands().collect::<Vec<_>>();
    let results = raw.results().collect::<Vec<_>>();
    let (ins, outs) = operands.split_at(operands.len().saturating_sub(1));
    print_values(ctx, &results, f)?;
    write!(f, " = {name}{}", layout.prefix)?;
    if !attrs.is_empty() {
        write!(f, " {{{}}}", attrs.join(", "))?;
    }
    print_typed(ctx, "ins", ins, state, f)?;
    print_typed(ctx, "outs", outs, state, f)?;
    write!(f, "{}", layout.suffix)?;
    if layout.arrow && !results.is_empty() {
        write!(f, " -> ")?;
        print_type_list(ctx, results.iter().map(|v| v.get_type(ctx)), state, f)?;
    }
    print_op_location(ctx, raw.loc(), state, f)
}

fn named(
    ctx: &Context,
    op: Ptr<Operation>,
    state: &State,
    f: &mut fmt::Formatter<'_>,
) -> Result<()> {
    print_custom(
        ctx,
        op,
        Custom {
            arrow: true,
            ..Custom::default()
        },
        state,
        f,
    )
}

/// Prints `indexing_maps = [..]` when `explicit`.
fn maps_prefix(ctx: &Context, explicit: bool, maps: &[AffineMap]) -> Result<String> {
    Ok(if explicit {
        format!(" indexing_maps = {}", affine_maps_text(ctx, maps)?)
    } else {
        String::new()
    })
}

fn print_tiles(ctx: &Context, tiles: &[TileSize], f: &mut fmt::Formatter<'_>) -> Result<()> {
    write!(f, " inner_tiles = [")?;
    for (i, tile) in tiles.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        match tile {
            TileSize::Static(size) => write!(f, "{size}")?,
            TileSize::Dynamic(value) => write!(f, "%{}", value.unique_name(ctx))?,
        }
    }
    write!(f, "]")?;
    Ok(())
}

/// Prints `linalg.pack` / `linalg.unpack` in custom form after verifying it.
#[allow(clippy::too_many_arguments)]
fn print_relayout(
    ctx: &Context,
    op: Ptr<Operation>,
    source: Value,
    dest: Value,
    padding: Option<Value>,
    outer_dims_perm: &[i64],
    inner_dims_pos: &[i64],
    tiles: &[TileSize],
    state: &State,
    f: &mut fmt::Formatter<'_>,
) -> Result<()> {
    Operation::get_op_dyn(op, ctx).verify(ctx)?;
    let name = op_name(ctx, op);
    let raw = op.deref(ctx);
    const OWNED: [&str; 4] = [
        "linalg_inner_dims_pos",
        "linalg_outer_dims_perm",
        "linalg_static_inner_tiles",
        "linalg_operand_segment_sizes",
    ];
    if let Some((key, _)) = raw.attributes.0.iter().find(|(key, _)| {
        let key: &str = key.as_ref();
        key != "builtin_given_names" && !OWNED.contains(&key)
    }) {
        return Err(untranslatable(format!(
            "unmapped attribute {key} on {name}"
        )));
    }
    let results = raw.results().collect::<Vec<_>>();
    print_values(ctx, &results, f)?;
    write!(f, " = {name} %{}", source.unique_name(ctx))?;
    if let Some(padding) = padding {
        write!(f, " padding_value(%{} : ", padding.unique_name(ctx))?;
        print_type(ctx, padding.get_type(ctx), state, f)?;
        write!(f, ")")?;
    }
    if !outer_dims_perm.is_empty() {
        write!(f, " outer_dims_perm = [{}]", list(outer_dims_perm))?;
    }
    write!(f, " inner_dims_pos = [{}]", list(inner_dims_pos))?;
    print_tiles(ctx, tiles, f)?;
    write!(f, " into %{} : ", dest.unique_name(ctx))?;
    print_type(ctx, source.get_type(ctx), state, f)?;
    write!(f, " -> ")?;
    print_type(ctx, dest.get_type(ctx), state, f)?;
    print_op_location(ctx, raw.loc(), state, f)
}

macro_rules! to_mlir {
    ($($ty:ty => |$this:ident, $ctx:ident, $state:ident, $f:ident| $body:expr;)*) => {$(
        #[op_interface_impl]
        impl ToMlirOp for $ty {
            fn to_mlir(&self, $ctx: &Context, $state: &State, $f: &mut fmt::Formatter<'_>) -> Result<()> {
                let $this = self;
                Operation::get_op_dyn($this.get_operation(), $ctx).verify($ctx)?;
                validate_attributes($ctx, $this.get_operation())?;
                $body
            }
        }
    )*};
}

macro_rules! generic {
    ($($ty:ty),* $(,)?) => {
        to_mlir!($($ty => |op, ctx, state, f| print_generic(ctx, op.get_operation(), state, f, &property);)*);
    };
}

macro_rules! named {
    ($($ty:ty),* $(,)?) => {
        to_mlir!($($ty => |op, ctx, state, f| named(ctx, op.get_operation(), state, f);)*);
    };
}

macro_rules! matmul {
    ($($ty:ty),* $(,)?) => {
        to_mlir!($($ty => |op, ctx, state, f| {
            let prefix = maps_prefix(ctx, op.has_user_maps(ctx), &op.indexing_maps(ctx))?;
            let layout = Custom { prefix, elided: &["linalg_indexing_maps"], arrow: true, ..Custom::default() };
            print_custom(ctx, op.get_operation(), layout, state, f)
        };)*);
    };
}

macro_rules! contract {
    ($($ty:ty),* $(,)?) => {
        to_mlir!($($ty => |op, ctx, state, f| {
            let prefix = maps_prefix(ctx, true, &op.indexing_maps(ctx))?;
            let layout = Custom { prefix, elided: &["linalg_indexing_maps"], arrow: true, ..Custom::default() };
            print_custom(ctx, op.get_operation(), layout, state, f)
        };)*);
    };
}

macro_rules! winograd {
    ($($ty:ty),* $(,)?) => {
        to_mlir!($($ty => |op, ctx, state, f| {
            let fmr = op
                .fmr(ctx)
                .ok_or_else(|| untranslatable("winograd transform without fmr".to_string()))?;
            let layout = Custom {
                prefix: format!(" fmr({})", fmr.as_str()),
                elided: &["linalg_fmr"],
                arrow: true,
                ..Custom::default()
            };
            print_custom(ctx, op.get_operation(), layout, state, f)
        };)*);
    };
}

generic!(GenericOp, MapOp, ReduceOp, YieldOp, IndexOp);

matmul!(MatmulOp, BatchMatmulOp, BatchReduceMatmulOp);

contract!(ContractOp, ScaledContractOp);

winograd!(
    WinogradFilterTransformOp,
    WinogradInputTransformOp,
    WinogradOutputTransformOp
);

to_mlir! {
    ElementwiseOp => |op, ctx, state, f| {
        let kind = op
            .kind(ctx)
            .ok_or_else(|| untranslatable("linalg.elementwise without kind".to_string()))?;
        let maps = maps_prefix(ctx, op.has_user_maps(ctx), &op.indexing_maps(ctx))?;
        let layout = Custom {
            prefix: format!(" <{}>{maps}", kind.as_str()),
            elided: &["linalg_kind", "linalg_indexing_maps"],
            arrow: true,
            ..Custom::default()
        };
        print_custom(ctx, op.get_operation(), layout, state, f)
    };
    TransposeOp => |op, ctx, state, f| {
        let layout = Custom {
            suffix: format!(" permutation = [{}]", list(&op.permutation(ctx))),
            elided: &["linalg_permutation"],
            ..Custom::default()
        };
        print_custom(ctx, op.get_operation(), layout, state, f)
    };
    BroadcastOp => |op, ctx, state, f| {
        let layout = Custom {
            suffix: format!(" dimensions = [{}]", list(&op.dimensions(ctx))),
            elided: &["linalg_dimensions"],
            ..Custom::default()
        };
        print_custom(ctx, op.get_operation(), layout, state, f)
    };
    SoftmaxOp => |op, ctx, state, f| {
        let dimension = op
            .dimension(ctx)
            .ok_or_else(|| untranslatable("linalg.softmax without dimension".to_string()))?;
        let layout = Custom {
            prefix: format!(" dimension({dimension})"),
            elided: &["linalg_dimension"],
            arrow: true,
            ..Custom::default()
        };
        print_custom(ctx, op.get_operation(), layout, state, f)
    };
    PackOp => |op, ctx, state, f| print_relayout(
        ctx,
        op.get_operation(),
        op.source(ctx),
        op.dest(ctx),
        op.padding_value(ctx),
        &op.outer_dims_perm(ctx),
        &op.inner_dims_pos(ctx),
        &op.inner_tiles(ctx),
        state,
        f,
    );
    UnPackOp => |op, ctx, state, f| print_relayout(
        ctx,
        op.get_operation(),
        op.source(ctx),
        op.dest(ctx),
        None,
        &op.outer_dims_perm(ctx),
        &op.inner_dims_pos(ctx),
        &op.inner_tiles(ctx),
        state,
        f,
    );
}

named!(
    CopyOp,
    QuantizedMatmulOp,
    Mmt4DOp,
    BatchMmt4DOp,
    QuantizedBatchMatmulOp,
    MatvecOp,
    VecmatOp,
    BatchMatvecOp,
    BatchVecmatOp,
    DotOp,
    Conv1DOp,
    Conv2DOp,
    Conv3DOp,
    Conv1DNwcWcfOp,
    Conv1DNcwFcwOp,
    Conv2DNhwcHwcfOp,
    Conv2DNhwcFhwcOp,
    Conv2DNhwcHwcfQOp,
    Conv2DNhwcFhwcQOp,
    Conv2DNchwFchwQOp,
    Conv2DNchwFchwOp,
    Conv2DNgchwFgchwOp,
    Conv2DNgchwGfchwOp,
    Conv2DNhwgcGfhwcOp,
    Conv2DNhwgcGfhwcQOp,
    Conv2DNgchwGfchwQOp,
    Conv3DNdhwcDhwcfOp,
    Conv3DNdhwcDhwcfQOp,
    Conv3DNcdhwFcdhwOp,
    DepthwiseConv1DNwcWcOp,
    DepthwiseConv1DNcwCwOp,
    DepthwiseConv1DNwcWcmOp,
    DepthwiseConv2DNhwcHwcOp,
    DepthwiseConv2DNchwChwOp,
    DepthwiseConv2DNhwcHwcQOp,
    DepthwiseConv2DNhwcHwcmOp,
    DepthwiseConv2DNhwcHwcmQOp,
    DepthwiseConv3DNdhwcDhwcOp,
    DepthwiseConv3DNcdhwCdhwOp,
    DepthwiseConv3DNdhwcDhwcmOp,
    PoolingNhwcSumOp,
    PoolingNchwSumOp,
    PoolingNhwcMaxOp,
    PoolingNhwcMaxUnsignedOp,
    PoolingNchwMaxOp,
    PoolingNhwcMinOp,
    PoolingNhwcMinUnsignedOp,
    PoolingNwcSumOp,
    PoolingNcwSumOp,
    PoolingNwcMaxOp,
    PoolingNwcMaxUnsignedOp,
    PoolingNcwMaxOp,
    PoolingNwcMinOp,
    PoolingNwcMinUnsignedOp,
    PoolingNdhwcSumOp,
    PoolingNdhwcMaxOp,
    PoolingNdhwcMinOp,
    FillOp,
    FillRng2DOp,
);
