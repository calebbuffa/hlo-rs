//! MLIR textual export for `iree_linalg_ext`.
//!
//! Operations print in upstream custom form. [`print_generic`] provides an
//! explicit generic-form export, translating storage to upstream properties:
//!
//! | storage key suffix | MLIR property | text |
//! |---|---|---|
//! | `_indexing_maps` | `indexing_maps` | `[affine_map<..>, ..]` |
//! | `_iterator_types` | `iterator_types` | `[#iree_linalg_ext.iterator_type<parallel>, ..]` |
//! | `_operand_segment_sizes` | `operandSegmentSizes` | `array<i32: ..>` |
//! | `_decomposition_config` | `decomposition_config` | `{..}` |
//! | `I64ArrayAttr` keys | same name | `array<i64: ..>` |
//! | `IndexAttr` keys | same name | `N : i64` |
//! | `BoolAttr` keys | same name | `true` / `false` |
//! | `UnitAttr` keys | same name | `unit` |
//! | `iree_im2col_static_output_sizes` | `static_output_sizes` | `[array<i64: ..>, ..]` |
//!
//! Dynamic im2col entries are stored as [`DYNAMIC_INDEX`],
//! which equals MLIR's `ShapedType::kDynamic`, so they print unchanged.

use crate::{attributes::SplitReductionMappingAttr, ops::*};
use pliron::{
    attribute::Attribute,
    builtin::attributes::{BoolAttr, DictAttr, OperandSegmentSizesAttr, UnitAttr, VecAttr},
    context::Context,
    derive::{attr_interface_impl, op_interface_impl},
    op::Op,
    printable::State,
    result::Result,
};
use pliron_arith::mlir::print_generic_named;
use pliron_linalg::{
    attributes::{I64ArrayAttr, IndexingMapsAttr, IteratorTypesAttr},
    mlir::affine_maps_text,
};
use pliron_mlir_builtin::attributes::IndexAttr;
use pliron_mlir_interop::{
    ToMlirAttr, ToMlirOp,
    printers::{mlir_dense_array, print_attr},
};
use std::fmt;

mod custom;

/// Prints an operation in generic form, including upstream Winograd names.
///
/// # Errors
///
/// Returns an error when verification or translation fails.
pub fn print_generic(
    ctx: &Context,
    op: pliron::context::Ptr<pliron::operation::Operation>,
    state: &State,
    f: &mut fmt::Formatter<'_>,
) -> Result<()> {
    let name = custom::upstream_name(ctx, op);
    print_generic_named(ctx, op, &name, state, f, &|ctx, key, attr| {
        property_for(ctx, &name, key, attr)
    })
}

fn untranslatable(message: String) -> pliron::result::Error {
    pliron::input_error_noloc!(pliron_mlir_interop::Error::Untranslatable(message))
}

/// Storage-key prefixes of every operation, longest first so that
/// `iree_topk_v2_` wins over `iree_topk_`.
const PREFIXES: [&str; 23] = [
    "iree_winograd_filter_transform_",
    "iree_winograd_output_transform_",
    "iree_winograd_input_transform_",
    "iree_online_attention_",
    "iree_dequantize_affine_",
    "iree_quantize_affine_",
    "iree_group_mmt4d_",
    "iree_exp_reduction_",
    "iree_arg_compare_",
    "iree_attention_",
    "iree_custom_op_",
    "iree_topk_v2_",
    "iree_scatter_",
    "iree_gather_",
    "iree_im2col_",
    "iree_index_",
    "iree_topk_",
    "iree_scan_",
    "iree_sort_",
    "iree_fft_",
    "iree_map_load_",
    "iree_map_store_",
    "iree_group_matmul_",
];

/// Upstream property names this dialect stores.
const PROPERTIES: [&str; 34] = [
    "indexing_maps",
    "iterator_types",
    "decomposition_config",
    "exp_reduced_operands",
    "dimension",
    "inclusive",
    "is_sorted",
    "dimension_map",
    "unique_indices",
    "transposed",
    "dim",
    "quant_min",
    "quant_max",
    "storage_unsigned",
    "input_unsigned",
    "zp_unsigned",
    "output_tile_size",
    "kernel_size",
    "image_dimensions",
    "kernel_dimensions",
    "strides",
    "dilations",
    "static_kernel_size",
    "static_offsets",
    "static_output_sizes",
    "batch_pos",
    "m_pos",
    "k_pos",
    "input_k_perm",
    "output_perm",
    "static_input_pad_low",
    "static_input_pad_high",
    "static_output_pad_low",
    "static_output_pad_high",
];

fn property_name(key: &str) -> Option<&'static str> {
    let (prefix, rest) = PREFIXES
        .iter()
        .find_map(|p| key.strip_prefix(p).map(|rest| (*p, rest)))?;
    let allowed: &[&str] = match prefix {
        "iree_sort_" => &["dimension", "operand_segment_sizes"],
        "iree_scan_" => &["dimension", "inclusive"],
        "iree_topk_" => &["dimension"],
        "iree_topk_v2_" => &["dimension", "is_sorted", "operand_segment_sizes"],
        "iree_arg_compare_" => &["dimension", "operand_segment_sizes"],
        "iree_scatter_" => &["dimension_map", "unique_indices"],
        "iree_gather_" => &["dimension_map"],
        "iree_group_mmt4d_" => &["transposed"],
        "iree_attention_" | "iree_online_attention_" => &["indexing_maps", "decomposition_config"],
        "iree_exp_reduction_" => &[
            "indexing_maps",
            "iterator_types",
            "exp_reduced_operands",
            "operand_segment_sizes",
        ],
        "iree_custom_op_" => &["indexing_maps", "iterator_types", "operand_segment_sizes"],
        "iree_index_" => &["dim"],
        "iree_fft_" => &["operand_segment_sizes"],
        "iree_quantize_affine_" => &[
            "indexing_maps",
            "quant_min",
            "quant_max",
            "storage_unsigned",
            "zp_unsigned",
        ],
        "iree_dequantize_affine_" => &[
            "indexing_maps",
            "quant_min",
            "quant_max",
            "input_unsigned",
            "zp_unsigned",
        ],
        "iree_winograd_filter_transform_" => {
            &["output_tile_size", "kernel_size", "kernel_dimensions"]
        }
        "iree_winograd_input_transform_" | "iree_winograd_output_transform_" => {
            &["output_tile_size", "kernel_size", "image_dimensions"]
        }
        "iree_im2col_" => &[
            "strides",
            "dilations",
            "static_kernel_size",
            "static_offsets",
            "static_output_sizes",
            "batch_pos",
            "m_pos",
            "k_pos",
            "input_k_perm",
            "output_perm",
            "static_input_pad_low",
            "static_input_pad_high",
            "static_output_pad_low",
            "static_output_pad_high",
            "operand_segment_sizes",
        ],
        _ => &[],
    };
    if !allowed.contains(&rest) {
        return None;
    }
    if rest == "operand_segment_sizes" {
        return Some("operandSegmentSizes");
    }
    PROPERTIES.iter().copied().find(|name| *name == rest)
}

fn property_for(
    ctx: &Context,
    name: &str,
    key: &str,
    attr: &dyn Attribute,
) -> Result<Option<(&'static str, Option<String>)>> {
    let mnemonic = name
        .strip_prefix("iree_linalg_ext.")
        .ok_or_else(|| untranslatable(format!("unexpected operation {name}")))?;
    let prefix = format!("iree_{}_", mnemonic.replace('.', "_"));
    // Prefix comparison alone would also accept topk_v2 properties on topk.
    let owner = PREFIXES.iter().find(|p| key.starts_with(**p));
    if owner.copied() != Some(prefix.as_str()) {
        return Err(untranslatable(format!(
            "attribute {key} is not a property of {name}"
        )));
    }
    property(ctx, key, attr)
}

/// Displays an attribute through `pliron-mlir-interop`.
struct MlirAttr<'a>(&'a Context, &'a dyn Attribute);

impl fmt::Display for MlirAttr<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        print_attr(self.0, self.1, &State::default(), f).map_err(|_| fmt::Error)
    }
}

/// Translates one `iree_linalg_ext` storage key to an MLIR property of a
/// generic-form operation.
///
/// Returns `Ok(None)` for keys this crate does not own.
///
/// # Errors
///
/// Returns an error when an owned key holds an attribute of an unexpected
/// type or one that cannot be printed.
pub fn property(
    ctx: &Context,
    key: &str,
    attr: &dyn Attribute,
) -> Result<Option<(&'static str, Option<String>)>> {
    let Some(name) = property_name(key) else {
        return Ok(None);
    };
    let wrong = || untranslatable(format!("attribute {key} has an unexpected type"));
    let valid_type = match name {
        "indexing_maps" => attr.downcast_ref::<IndexingMapsAttr>().is_some(),
        "iterator_types" => attr.downcast_ref::<IteratorTypesAttr>().is_some(),
        "decomposition_config" => attr.downcast_ref::<DictAttr>().is_some(),
        "operandSegmentSizes" => attr.downcast_ref::<OperandSegmentSizesAttr>().is_some(),
        "dimension" | "dim" | "quant_min" | "quant_max" | "output_tile_size" | "kernel_size" => {
            attr.downcast_ref::<IndexAttr>().is_some()
        }
        "inclusive" | "unique_indices" | "transposed" => attr.downcast_ref::<BoolAttr>().is_some(),
        "is_sorted" | "storage_unsigned" | "input_unsigned" | "zp_unsigned" => {
            attr.downcast_ref::<UnitAttr>().is_some()
        }
        "static_output_sizes" => attr.downcast_ref::<VecAttr>().is_some(),
        _ => attr.downcast_ref::<I64ArrayAttr>().is_some(),
    };
    if !valid_type {
        return Err(wrong());
    }
    let text = if let Some(maps) = attr.downcast_ref::<IndexingMapsAttr>() {
        affine_maps_text(ctx, &maps.maps)?
    } else if let Some(types) = attr.downcast_ref::<IteratorTypesAttr>() {
        let text = types
            .values
            .iter()
            .map(|v| format!("#iree_linalg_ext.iterator_type<{}>", v.as_str()))
            .collect::<Vec<_>>()
            .join(", ");
        format!("[{text}]")
    } else if let Some(sizes) = attr.downcast_ref::<OperandSegmentSizesAttr>() {
        mlir_dense_array("i32", &sizes.0)
    } else if let Some(values) = attr.downcast_ref::<I64ArrayAttr>() {
        mlir_dense_array("i64", &values.values)
    } else if let Some(value) = attr.downcast_ref::<IndexAttr>() {
        format!("{} : i64", value.value)
    } else if let Some(value) = attr.downcast_ref::<BoolAttr>() {
        bool::from(value.clone()).to_string()
    } else if attr.downcast_ref::<UnitAttr>().is_some() {
        "unit".to_owned()
    } else if let Some(nested) = attr.downcast_ref::<VecAttr>() {
        let arrays = nested
            .0
            .iter()
            .map(|a| {
                a.downcast_ref::<I64ArrayAttr>()
                    .map(|v| mlir_dense_array("i64", &v.values))
                    .ok_or_else(wrong)
            })
            .collect::<Result<Vec<_>>>()?;
        format!("[{}]", arrays.join(", "))
    } else if attr.downcast_ref::<DictAttr>().is_some() {
        let mut text = String::new();
        fmt::Write::write_fmt(&mut text, format_args!("{}", MlirAttr(ctx, attr)))
            .map_err(|_| untranslatable(format!("attribute {key} cannot be printed")))?;
        text
    } else {
        return Err(wrong());
    };
    Ok(Some((name, Some(text))))
}

#[attr_interface_impl]
impl ToMlirAttr for SplitReductionMappingAttr {
    fn to_mlir(&self, _ctx: &Context, _state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
        write!(
            f,
            "#iree_linalg_ext.split_reduction_mapping<{}>",
            self.dimension
        )?;
        Ok(())
    }
}

macro_rules! generic {
    ($($ty:ty),* $(,)?) => {$(
        #[op_interface_impl]
        impl ToMlirOp for $ty {
            fn to_mlir(&self, ctx: &Context, state: &State, f: &mut fmt::Formatter<'_>) -> Result<()> {
                custom::print(ctx, self.get_operation(), state, f)
            }
        }
    )*};
}

generic!(
    SortOp,
    ScanOp,
    TopkOp,
    TopkV2Op,
    ArgCompareOp,
    ScatterOp,
    GatherOp,
    MapLoadOp,
    MapStoreOp,
    GroupMatmulOp,
    GroupMmt4dOp,
    AttentionOp,
    OnlineAttentionOp,
    ExpReductionOp,
    CustomOp,
    YieldOp,
    IndexOp,
    FftOp,
    QuantizeAffineOp,
    DequantizeAffineOp,
    Im2colOp,
    WinogradInputTransformOp,
    WinogradFilterTransformOp,
    WinogradOutputTransformOp,
);
