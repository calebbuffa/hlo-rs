use super::{TextError, TextErrorKind, module::AssemblyAttributes};
use crate::{
    attributes::{
        ArrayOfLayoutAttr, DenseArrayAttr, DenseElementsAttr, LayoutAttr, Precision, ScalarBitsAttr,
    },
    ops::*,
    pliron::{
        attribute::{AttrObj, AttributeDict},
        builtin::{
            attributes::{BoolAttr, IntegerAttr, VecAttr},
            types::{IntegerType, Signedness},
        },
        common_traits::Verify,
        context::{Context, Ptr},
        identifier::Identifier,
        op::Op,
        operation::Operation,
        r#type::TypeHandle,
        utils::apint::{APInt, bw},
        value::Value,
    },
    types::{ElementType, TensorType},
};

type Builder = fn(&mut Context, Vec<TypeHandle>, Vec<Value>) -> Ptr<Operation>;
struct Field {
    source: &'static str,
    native: &'static str,
    kind: Kind,
    required: bool,
}
enum Kind {
    I32,
    I64,
    I64Array,
    I64Pairs,
    Bool,
    Unit,
    Groups,
    InfeedLayout,
    Precision,
    Attribute,
}

const RESULT_ACCURACY: &[Field] = &[Field {
    source: "result_accuracy",
    native: "stablehlo_result_accuracy",
    kind: Kind::Attribute,
    required: false,
}];

fn create<T: Op>(
    ctx: &mut Context,
    results: Vec<TypeHandle>,
    operands: Vec<Value>,
) -> Ptr<Operation> {
    Operation::new(ctx, T::get_concrete_op_info(), results, operands, vec![], 0)
}

// Source fields and widths are StablehloOps.td contracts, not native field-name
// transformations. Optional fields remain absent; no defaults are synthesized.
fn schema(name: &str) -> Option<(Builder, &'static [Field])> {
    macro_rules! fields {
        ($($source:literal => $native:literal : $kind:ident $(, $optional:literal)?);* $(;)?) => {{
            const FIELDS: &[Field] = &[$(Field { source: $source, native: $native, kind: Kind::$kind, required: true $(&& !$optional)? }),*];
            FIELDS
        }};
    }
    Some(match name {
        "stablehlo.dot" => (
            create::<DotOp>,
            fields! { "precision_config" => "stablehlo_compute_dot_precision": Precision, true; },
        ),
        "stablehlo.dot_general" => (
            create::<DotGeneralOp>,
            fields! {
                "dot_dimension_numbers" => "stablehlo_dot_dimensions": Attribute;
                "precision_config" => "stablehlo_dot_precision": Precision, true;
                "algorithm" => "stablehlo_dot_algorithm": Attribute, true;
            },
        ),
        "stablehlo.cbrt" => (create::<CbrtOp>, RESULT_ACCURACY),
        "stablehlo.cosine" => (create::<CosineOp>, RESULT_ACCURACY),
        "stablehlo.exp2" => (create::<Exp2Op>, RESULT_ACCURACY),
        "stablehlo.exponential" => (create::<ExponentialOp>, RESULT_ACCURACY),
        "stablehlo.exponential_minus_one" => (create::<ExponentialMinusOneOp>, RESULT_ACCURACY),
        "stablehlo.log" => (create::<LogOp>, RESULT_ACCURACY),
        "stablehlo.log2" => (create::<Log2Op>, RESULT_ACCURACY),
        "stablehlo.log_plus_one" => (create::<LogPlusOneOp>, RESULT_ACCURACY),
        "stablehlo.logistic" => (create::<LogisticOp>, RESULT_ACCURACY),
        "stablehlo.rsqrt" => (create::<RsqrtOp>, RESULT_ACCURACY),
        "stablehlo.sine" => (create::<SineOp>, RESULT_ACCURACY),
        "stablehlo.sqrt" => (create::<SqrtOp>, RESULT_ACCURACY),
        "stablehlo.tan" => (create::<TanOp>, RESULT_ACCURACY),
        "stablehlo.tanh" => (create::<TanhOp>, RESULT_ACCURACY),
        "stablehlo.transpose" => (
            create::<TransposeOp>,
            fields! {
                "permutation" => "stablehlo_transpose_permutation": I64Array;
            },
        ),
        "stablehlo.broadcast_in_dim" => (
            create::<BroadcastInDimOp>,
            fields! {
                "broadcast_dimensions" => "stablehlo_broadcast_dimensions": I64Array;
            },
        ),
        "stablehlo.dynamic_broadcast_in_dim" => (
            create::<DynamicBroadcastInDimOp>,
            fields! {
                "broadcast_dimensions" => "stablehlo_dynamic_broadcast_dimensions": I64Array;
                "known_expanding_dimensions" => "stablehlo_dynamic_broadcast_known_expanding": I64Array, true;
                "known_nonexpanding_dimensions" => "stablehlo_dynamic_broadcast_known_nonexpanding": I64Array, true;
            },
        ),
        "stablehlo.reverse" => (
            create::<ReverseOp>,
            fields! {
                "dimensions" => "stablehlo_reverse_dimensions": I64Array;
            },
        ),
        "stablehlo.slice" => (
            create::<SliceOp>,
            fields! {
                "start_indices" => "stablehlo_slice_starts": I64Array;
                "limit_indices" => "stablehlo_slice_limits": I64Array;
                "strides" => "stablehlo_slice_strides": I64Array;
            },
        ),
        "stablehlo.concatenate" => (
            create::<ConcatenateOp>,
            fields! {
                "dimension" => "stablehlo_concatenate_dimension": I64;
            },
        ),
        "stablehlo.iota" => (
            create::<IotaOp>,
            fields! {
                "iota_dimension" => "stablehlo_iota_dimension": I64;
            },
        ),
        "stablehlo.dynamic_iota" => (
            create::<DynamicIotaOp>,
            fields! {
                "iota_dimension" => "stablehlo_dynamic_iota_dimension": I64;
            },
        ),
        "stablehlo.get_tuple_element" => (
            create::<GetTupleElementOp>,
            fields! {
                "index" => "stablehlo_get_tuple_element_index": I32;
            },
        ),
        "stablehlo.get_dimension_size" => (
            create::<GetDimensionSizeOp>,
            fields! {
                "dimension" => "stablehlo_get_dimension_size_dimension": I64;
            },
        ),
        "stablehlo.set_dimension_size" => (
            create::<SetDimensionSizeOp>,
            fields! {
                "dimension" => "stablehlo_set_dimension_size_dimension": I64;
            },
        ),
        "stablehlo.compare" => (
            create::<CompareOp>,
            fields! {
                "comparison_direction" => "stablehlo_compare_direction": Attribute;
                "compare_type" => "stablehlo_compare_type": Attribute, true;
            },
        ),
        "stablehlo.fft" => (
            create::<FftOp>,
            fields! {
                "fft_type" => "stablehlo_compute_fft_type": Attribute;
                "fft_length" => "stablehlo_compute_fft_length": I64Array;
            },
        ),
        "stablehlo.reduce" => (
            create::<ReduceOp>,
            fields! {
                "dimensions" => "stablehlo_reduce_dimensions": I64Array;
            },
        ),
        "stablehlo.reduce_window" => (
            create::<ReduceWindowOp>,
            fields! {
                "window_dimensions" => "stablehlo_reduction_window": I64Array;
                "window_strides" => "stablehlo_reduction_strides": I64Array, true;
                "base_dilations" => "stablehlo_reduction_base_dilations": I64Array, true;
                "window_dilations" => "stablehlo_reduction_window_dilations": I64Array, true;
                "padding" => "stablehlo_reduction_padding": I64Pairs, true;
            },
        ),
        "stablehlo.map" => (
            create::<MapOp>,
            fields! {
                "dimensions" => "stablehlo_reduction_map_dimensions": I64Array;
            },
        ),
        "stablehlo.sort" => (
            create::<SortOp>,
            fields! {
                "dimension" => "stablehlo_reduction_sort_dimension": I64, true;
                "is_stable" => "stablehlo_reduction_sort_stable": Bool, true;
            },
        ),
        "stablehlo.select_and_scatter" => (
            create::<SelectAndScatterOp>,
            fields! {
                "window_dimensions" => "stablehlo_reduction_select_window": I64Array, true;
                "window_strides" => "stablehlo_reduction_select_strides": I64Array, true;
                "padding" => "stablehlo_reduction_select_padding": I64Pairs, true;
            },
        ),
        "stablehlo.scatter" => (
            create::<ScatterOp>,
            fields! {
                "scatter_dimension_numbers" => "stablehlo_scatter_dimension_numbers": Attribute;
                "indices_are_sorted" => "stablehlo_scatter_indices_are_sorted": Bool, true;
                "unique_indices" => "stablehlo_scatter_unique_indices": Bool, true;
            },
        ),
        "stablehlo.gather" => (
            create::<GatherOp>,
            fields! {
                "dimension_numbers" => "stablehlo_gather_dimension_numbers": Attribute;
                "slice_sizes" => "stablehlo_gather_slice_sizes": I64Array;
                "indices_are_sorted" => "stablehlo_gather_indices_are_sorted": Bool, true;
            },
        ),
        "stablehlo.dynamic_gather" => (
            create::<DynamicGatherOp>,
            fields! {
                "dimension_numbers" => "stablehlo_dynamic_gather_dimension_numbers": Attribute;
                "indices_are_sorted" => "stablehlo_dynamic_gather_indices_are_sorted": Bool, true;
            },
        ),
        "stablehlo.dynamic_slice" => (
            create::<DynamicSliceOp>,
            fields! {
                "slice_sizes" => "stablehlo_dynamic_slice_sizes": I64Array;
            },
        ),
        "stablehlo.pad" => (
            create::<PadOp>,
            fields! {
                "edge_padding_low" => "stablehlo_pad_low": I64Array;
                "edge_padding_high" => "stablehlo_pad_high": I64Array;
                "interior_padding" => "stablehlo_pad_interior": I64Array;
            },
        ),
        "stablehlo.reduce_precision" => (
            create::<ReducePrecisionOp>,
            fields! {
                "exponent_bits" => "stablehlo_exponent_bits": I32;
                "mantissa_bits" => "stablehlo_mantissa_bits": I32;
            },
        ),
        "stablehlo.broadcast" => (
            create::<BroadcastOp>,
            fields! { "broadcast_sizes" => "stablehlo_broadcast_sizes": I64Array; },
        ),
        "stablehlo.einsum" => (
            create::<EinsumOp>,
            fields! { "einsum_config" => "stablehlo_compute_einsum_equation": Attribute; },
        ),
        "stablehlo.unary_einsum" => (
            create::<UnaryEinsumOp>,
            fields! { "einsum_config" => "stablehlo_compute_unary_einsum_equation": Attribute; },
        ),
        "stablehlo.cholesky" => (
            create::<CholeskyOp>,
            fields! { "lower" => "stablehlo_compute_cholesky_lower": Bool, true; },
        ),
        "stablehlo.rng" => (
            create::<RngOp>,
            fields! { "rng_distribution" => "stablehlo_compute_rng_distribution": Attribute; },
        ),
        "stablehlo.rng_bit_generator" => (
            create::<RngBitGeneratorOp>,
            fields! { "rng_algorithm" => "stablehlo_compute_rng_bit_algorithm": Attribute; },
        ),
        "stablehlo.torch_index_select" => (
            create::<TorchIndexSelectOp>,
            fields! {
                "dim" => "stablehlo_torch_index_select_dimension": I64;
                "batch_dims" => "stablehlo_torch_index_select_batch_dims": I64;
            },
        ),
        "stablehlo.infeed" => (
            create::<InfeedOp>,
            fields! {
                "infeed_config" => "stablehlo_infeed_config": Attribute, true;
                "layout" => "stablehlo_infeed_layout": InfeedLayout, true;
            },
        ),
        "stablehlo.outfeed" => (
            create::<OutfeedOp>,
            fields! { "outfeed_config" => "stablehlo_outfeed_config": Attribute, true; },
        ),
        "stablehlo.send" => (
            create::<SendOp>,
            fields! {
                "channel_handle" => "stablehlo_send_channel_handle": Attribute;
                "is_host_transfer" => "stablehlo_send_is_host_transfer": Bool, true;
                "source_target_pairs" => "stablehlo_send_source_target_pairs": I64Pairs, true;
            },
        ),
        "stablehlo.recv" => (
            create::<RecvOp>,
            fields! {
                "channel_handle" => "stablehlo_recv_channel_handle": Attribute;
                "is_host_transfer" => "stablehlo_recv_is_host_transfer": Bool, true;
                "source_target_pairs" => "stablehlo_recv_source_target_pairs": I64Pairs, true;
            },
        ),
        "stablehlo.all_gather" => (
            create::<AllGatherOp>,
            fields! {
                "all_gather_dim" => "stablehlo_all_gather_dimension": I64;
                "replica_groups" => "stablehlo_all_gather_replica_groups": Groups;
                "channel_handle" => "stablehlo_all_gather_channel_handle": Attribute, true;
                "use_global_device_ids" => "stablehlo_all_gather_use_global_device_ids": Unit, true;
            },
        ),
        "stablehlo.all_reduce" => (
            create::<AllReduceOp>,
            fields! {
                "replica_groups" => "stablehlo_all_reduce_replica_groups": Groups;
                "channel_handle" => "stablehlo_all_reduce_channel_handle": Attribute, true;
                "use_global_device_ids" => "stablehlo_all_reduce_use_global_device_ids": Unit, true;
            },
        ),
        "stablehlo.reduce_scatter" => (
            create::<ReduceScatterOp>,
            fields! {
                "scatter_dimension" => "stablehlo_reduce_scatter_dimension": I64;
                "replica_groups" => "stablehlo_reduce_scatter_replica_groups": Groups;
                "channel_handle" => "stablehlo_reduce_scatter_channel_handle": Attribute, true;
                "use_global_device_ids" => "stablehlo_reduce_scatter_use_global_device_ids": Unit, true;
            },
        ),
        "stablehlo.all_to_all" => (
            create::<AllToAllOp>,
            fields! {
                "split_dimension" => "stablehlo_all_to_all_split_dimension": I64;
                "concat_dimension" => "stablehlo_all_to_all_concat_dimension": I64;
                "split_count" => "stablehlo_all_to_all_split_count": I64;
                "replica_groups" => "stablehlo_all_to_all_replica_groups": Groups;
                "channel_handle" => "stablehlo_all_to_all_channel_handle": Attribute, true;
            },
        ),
        "stablehlo.collective_broadcast" => (
            create::<CollectiveBroadcastOp>,
            fields! {
                "replica_groups" => "stablehlo_collective_broadcast_replica_groups": Groups;
                "channel_handle" => "stablehlo_collective_broadcast_channel_handle": Attribute, true;
                "has_dynamic_root" => "stablehlo_collective_broadcast_has_dynamic_root": Unit, true;
            },
        ),
        "stablehlo.collective_reduce" => (
            create::<CollectiveReduceOp>,
            fields! {
                "replica_groups" => "stablehlo_collective_reduce_replica_groups": Groups;
                "channel_handle" => "stablehlo_collective_reduce_channel_handle": Attribute, true;
                "has_dynamic_root" => "stablehlo_collective_reduce_has_dynamic_root": Unit, true;
                "use_global_device_ids" => "stablehlo_collective_reduce_use_global_device_ids": Unit, true;
            },
        ),
        "stablehlo.collective_permute" => (
            create::<CollectivePermuteOp>,
            fields! {
                "source_target_pairs" => "stablehlo_collective_permute_source_target_pairs": I64Pairs;
                "channel_handle" => "stablehlo_collective_permute_channel_handle": Attribute, true;
            },
        ),
        "stablehlo.cross_replica_sum" => (
            create::<CrossReplicaSumOp>,
            fields! { "replica_groups" => "stablehlo_cross_replica_sum_replica_groups": Groups; },
        ),
        "stablehlo.composite" => (
            create::<CompositeOp>,
            fields! {
                "name" => "stablehlo_composite_name": Attribute;
                "version" => "stablehlo_composite_version": I32, true;
                "decomposition" => "stablehlo_composite_decomposition": Attribute;
                "composite_attributes" => "stablehlo_composite_attributes": Attribute, true;
            },
        ),
        "stablehlo.custom_call" => (
            create::<CustomCallOp>,
            fields! {
                "call_target_name" => "stablehlo_custom_call_target": Attribute;
                "has_side_effect" => "stablehlo_custom_call_has_side_effect": Bool, true;
                "api_version" => "stablehlo_custom_call_api_version": I32, true;
                "backend_config" => "stablehlo_custom_call_backend_config": Attribute, true;
                "called_computations" => "stablehlo_custom_call_called_computations": Attribute, true;
                "output_operand_aliases" => "stablehlo_custom_call_output_operand_aliases": Attribute, true;
                "operand_layouts" => "stablehlo_custom_call_operand_layouts": Attribute, true;
                "result_layouts" => "stablehlo_custom_call_result_layouts": Attribute, true;
                "result_tilings" => "stablehlo_custom_call_result_tilings": Attribute, true;
            },
        ),
        "stablehlo.convolution" => (
            create::<ConvolutionOp>,
            fields! {
                "dimension_numbers" => "stablehlo_compute_conv_dims": Attribute;
                "window_strides" => "stablehlo_compute_conv_strides": I64Array, true;
                "padding" => "stablehlo_compute_conv_padding": I64Pairs, true;
                "lhs_dilation" => "stablehlo_compute_conv_lhs_dilation": I64Array, true;
                "rhs_dilation" => "stablehlo_compute_conv_rhs_dilation": I64Array, true;
                "window_reversal" => "stablehlo_compute_conv_reversal": Attribute, true;
                "precision_config" => "stablehlo_compute_conv_precision": Precision, true;
            },
        ),
        "stablehlo.dynamic_conv" => (
            create::<DynamicConvOp>,
            fields! {
                "dimension_numbers" => "stablehlo_compute_dynamic_conv_dims": Attribute;
                "window_strides" => "stablehlo_compute_dynamic_conv_strides": I64Array, true;
                "lhs_dilation" => "stablehlo_compute_dynamic_conv_lhs_dilation": I64Array, true;
                "rhs_dilation" => "stablehlo_compute_dynamic_conv_rhs_dilation": I64Array, true;
                "window_reversal" => "stablehlo_compute_dynamic_conv_reversal": Attribute, true;
                "precision_config" => "stablehlo_compute_dynamic_conv_precision": Precision, true;
            },
        ),
        _ => return None,
    })
}

fn invalid(reason: impl Into<String>) -> TextError {
    TextError {
        offset: 0,
        kind: TextErrorKind::InvalidAttribute(reason.into()),
    }
}

pub(super) fn check_source_fields(name: &str, attributes: &AttributeDict) -> Result<(), TextError> {
    for key in attributes.0.keys() {
        if *key == crate::attributes::DISCARDABLE_ATTRIBUTES {
            continue;
        }
        let key = key.as_ref();
        if is_source_field(name, key) {
            continue;
        }
        return Err(TextError {
            offset: 0,
            kind: TextErrorKind::Unsupported("unmapped inherent operation attribute"),
        });
    }
    Ok(())
}

pub(super) fn is_source_field(name: &str, key: &str) -> bool {
    schema(name).is_some_and(|(_, fields)| fields.iter().any(|field| field.source == key))
        || matches!(
            (name, key),
            ("stablehlo.constant", "value")
                | ("stablehlo.call" | "func.call", "callee")
                | (
                    "stablehlo.convolution" | "stablehlo.dynamic_conv",
                    "feature_group_count" | "batch_group_count" | "window_reversal"
                )
                | (
                    "stablehlo.batch_norm_inference"
                        | "stablehlo.batch_norm_training"
                        | "stablehlo.batch_norm_grad",
                    "epsilon" | "feature_index"
                )
                | (
                    "stablehlo.triangular_solve",
                    "left_side" | "lower" | "unit_diagonal" | "transpose_a"
                )
        )
}

#[cfg(test)]
pub(super) fn source_field_required(name: &str, key: &str) -> Option<bool> {
    schema(name)?
        .1
        .iter()
        .find(|field| field.source == key)
        .map(|field| field.required)
}

#[cfg(test)]
pub(super) fn is_native_field(name: &str, key: &str) -> bool {
    schema(name).is_some_and(|(_, fields)| fields.iter().any(|field| field.native == key))
        || matches!(
            (name, key),
            (
                "stablehlo.constant",
                "stablehlo_constant_bits"
                    | "stablehlo_constant_splat"
                    | "stablehlo_constant_sparse"
            ) | (
                "stablehlo.func",
                "stablehlo_function_name" | "stablehlo_function_signature"
            ) | ("stablehlo.call", "stablehlo_call_callee")
                | ("stablehlo.convolution", "stablehlo_compute_conv_groups")
                | (
                    "stablehlo.dynamic_conv",
                    "stablehlo_compute_dynamic_conv_groups"
                )
                | (
                    "stablehlo.batch_norm_inference",
                    "stablehlo_compute_bn_options"
                )
                | (
                    "stablehlo.batch_norm_training",
                    "stablehlo_compute_bn_training_options"
                )
                | (
                    "stablehlo.batch_norm_grad",
                    "stablehlo_compute_bn_grad_options"
                )
                | (
                    "stablehlo.triangular_solve",
                    "stablehlo_compute_triangular_flags"
                )
                | (
                    "stablehlo.all_to_all",
                    "stablehlo_all_to_all_use_global_device_ids"
                )
        )
}

fn bits(ctx: &Context, value: &AttrObj, width: u32) -> Result<u64, TextError> {
    let scalar = value
        .downcast_ref::<ScalarBitsAttr>()
        .ok_or_else(|| invalid("expected signless integer attribute"))?;
    scalar
        .verify(ctx)
        .map_err(|failure| invalid(failure.to_string()))?;
    let expected = if width == 32 {
        ElementType::I32
    } else {
        ElementType::I64
    };
    if scalar.ty != expected.native(ctx) {
        return Err(invalid("integer attribute has wrong width or signedness"));
    }
    scalar
        .words
        .first()
        .copied()
        .ok_or_else(|| invalid("integer payload is empty"))
}

fn native_integer(ctx: &Context, word: u64, width: u32) -> AttrObj {
    let ty = IntegerType::get(
        ctx,
        width,
        if width == 32 {
            Signedness::Signless
        } else {
            Signedness::Unsigned
        },
    );
    Box::new(IntegerAttr::new(
        ty,
        APInt::from_u64(word, bw(width as usize)),
    ))
}

fn lower(ctx: &Context, field: &Field, value: AttrObj) -> Result<AttrObj, TextError> {
    match field.kind {
        Kind::InfeedLayout => {
            let layouts = value
                .downcast_ref::<VecAttr>()
                .ok_or_else(|| invalid("expected infeed layout array"))?;
            let layouts = layouts
                .0
                .iter()
                .map(|layout| {
                    let dimensions = layout
                        .downcast_ref::<VecAttr>()
                        .ok_or_else(|| invalid("expected infeed dimension array"))?;
                    let minor_to_major = dimensions
                        .0
                        .iter()
                        .map(|dimension| Ok(bits(ctx, dimension, 64)? as i64))
                        .collect::<Result<Vec<_>, TextError>>()?;
                    Ok(LayoutAttr { minor_to_major })
                })
                .collect::<Result<Vec<_>, TextError>>()?;
            Ok(Box::new(ArrayOfLayoutAttr { layouts }))
        }
        Kind::Unit => {
            if value
                .downcast_ref::<pliron::builtin::attributes::UnitAttr>()
                .is_none()
            {
                return Err(invalid("expected unit attribute"));
            }
            Ok(Box::new(BoolAttr::new(true)))
        }
        Kind::Groups => {
            if value
                .downcast_ref::<crate::attributes::ReplicaGroupMeshAxesAttr>()
                .is_some()
            {
                return Ok(value);
            }
            let dense = value
                .downcast_ref::<DenseElementsAttr>()
                .ok_or_else(|| invalid("expected dense i64 replica groups"))?;
            dense
                .verify(ctx)
                .map_err(|error| invalid(error.to_string()))?;
            let ty = dense.ty.deref(ctx);
            let ty = ty
                .downcast_ref::<TensorType>()
                .filter(|ty| ty.element == ElementType::I64.native(ctx) && ty.dimensions.len() == 2)
                .ok_or_else(|| invalid("expected a ranked i64 replica group matrix"))?;
            let columns =
                usize::try_from(ty.dimensions[1]).map_err(|_| invalid("invalid group width"))?;
            let rows =
                usize::try_from(ty.dimensions[0]).map_err(|_| invalid("invalid group count"))?;
            let mut groups = Vec::with_capacity(rows);
            for row in 0..rows {
                let mut values = Vec::with_capacity(columns);
                for column in 0..columns {
                    let index = if dense.is_splat {
                        0
                    } else {
                        row * columns + column
                    };
                    let value = dense
                        .values
                        .0
                        .get(index)
                        .ok_or_else(|| invalid("missing group element"))?;
                    values.push(native_integer(ctx, bits(ctx, value, 64)?, 64));
                }
                groups.push(Box::new(VecAttr::new(values)) as AttrObj);
            }
            Ok(Box::new(VecAttr::new(groups)))
        }
        Kind::Precision => {
            let values = value.downcast_ref::<VecAttr>().ok_or_else(|| {
                invalid("precision configuration must be an array of precision enums")
            })?;
            let values = values
                .0
                .iter()
                .map(|value| {
                    let precision = value.downcast_ref::<Precision>().ok_or_else(|| {
                        invalid("precision configuration requires precision enums")
                    })?;
                    precision
                        .verify(ctx)
                        .map_err(|error| invalid(error.to_string()))?;
                    Ok(native_integer(ctx, *precision as u64, 64))
                })
                .collect::<Result<Vec<_>, TextError>>()?;
            Ok(Box::new(VecAttr::new(values)))
        }
        Kind::Attribute => Ok(value),
        Kind::Bool => {
            if value.downcast_ref::<BoolAttr>().is_some() {
                Ok(value)
            } else {
                Err(invalid("expected boolean attribute"))
            }
        }
        Kind::I32 => Ok(native_integer(ctx, bits(ctx, &value, 32)?, 32)),
        Kind::I64 => Ok(native_integer(ctx, bits(ctx, &value, 64)?, 64)),
        Kind::I64Array => {
            let array = value
                .downcast_ref::<DenseArrayAttr>()
                .filter(|array| array.element == ElementType::I64.native(ctx))
                .ok_or_else(|| invalid("expected dense i64 array"))?;
            let values = array
                .values
                .0
                .iter()
                .map(|value| bits(ctx, value, 64).map(|word| native_integer(ctx, word, 64)))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Box::new(VecAttr::new(values)))
        }
        Kind::I64Pairs => {
            let dense = value
                .downcast_ref::<DenseElementsAttr>()
                .ok_or_else(|| invalid("padding must be a dense i64 matrix"))?;
            dense
                .verify(ctx)
                .map_err(|failure| invalid(failure.to_string()))?;
            let dense_ty = dense.ty.deref(ctx);
            let ty = dense_ty
                .downcast_ref::<TensorType>()
                .ok_or_else(|| invalid("padding must have a ranked tensor type"))?;
            if dense.is_splat
                || ty.element != ElementType::I64.native(ctx)
                || ty.dimensions.len() != 2
                || ty.dimensions[0] <= 0
                || ty.dimensions[1] != 2
            {
                return Err(invalid("padding must be a nonempty [rank, 2] i64 matrix"));
            }
            let values = dense
                .values
                .0
                .iter()
                .map(|value| signed_i64(ctx, value))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Box::new(signed_integer_list(ctx, values)))
        }
    }
}

fn raise_integer(ctx: &Context, value: &AttrObj, width: u32) -> Result<AttrObj, TextError> {
    let integer = value
        .downcast_ref::<IntegerAttr>()
        .ok_or_else(|| invalid("native integer attribute required"))?;
    let value = integer
        .value()
        .to_string_decimal(false)
        .parse::<u64>()
        .map_err(|_| invalid("native integer does not fit source storage"))?;
    if width < 64 && value >> width != 0 {
        return Err(invalid("native integer does not fit source width"));
    }
    Ok(Box::new(ScalarBitsAttr {
        ty: if width == 32 {
            ElementType::I32
        } else {
            ElementType::I64
        }
        .native(ctx),
        words: vec![value],
    }))
}

fn raise(ctx: &Context, field: &Field, value: AttrObj) -> Result<AttrObj, TextError> {
    match field.kind {
        Kind::InfeedLayout => {
            let layouts = value
                .downcast_ref::<ArrayOfLayoutAttr>()
                .ok_or_else(|| invalid("expected native infeed layouts"))?;
            Ok(Box::new(VecAttr::new(
                layouts
                    .layouts
                    .iter()
                    .map(|layout| {
                        Box::new(VecAttr::new(
                            layout
                                .minor_to_major
                                .iter()
                                .map(|dimension| {
                                    Box::new(ScalarBitsAttr {
                                        ty: ElementType::I64.native(ctx),
                                        words: vec![*dimension as u64],
                                    }) as AttrObj
                                })
                                .collect(),
                        )) as AttrObj
                    })
                    .collect(),
            )))
        }
        Kind::Unit => {
            if value.downcast_ref::<BoolAttr>().is_none() {
                return Err(invalid("expected native boolean flag"));
            }
            Ok(Box::new(pliron::builtin::attributes::UnitAttr))
        }
        Kind::Groups => {
            if value
                .downcast_ref::<crate::attributes::ReplicaGroupMeshAxesAttr>()
                .is_some()
            {
                return Ok(value);
            }
            let groups = value
                .downcast_ref::<VecAttr>()
                .ok_or_else(|| invalid("expected native replica groups"))?;
            let rows = groups
                .0
                .iter()
                .map(|row| {
                    row.downcast_ref::<VecAttr>()
                        .ok_or_else(|| invalid("expected replica group row"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let columns = rows.iter().map(|row| row.0.len()).max().unwrap_or(0);
            let mut values = Vec::new();
            for row in &rows {
                for value in &row.0 {
                    values.push(raise_integer(ctx, value, 64)?);
                }
                for _ in row.0.len()..columns {
                    values.push(Box::new(ScalarBitsAttr {
                        ty: ElementType::I64.native(ctx),
                        words: vec![u64::MAX],
                    }) as AttrObj);
                }
            }
            let ty = TensorType::get(
                ctx,
                vec![rows.len() as i64, columns as i64],
                ElementType::I64.native(ctx),
            )
            .into();
            Ok(Box::new(DenseElementsAttr::new(ty, VecAttr::new(values))))
        }
        Kind::Precision => {
            let values = value
                .downcast_ref::<VecAttr>()
                .ok_or_else(|| invalid("native precision integer list required"))?;
            let values = values
                .0
                .iter()
                .map(|value| {
                    let value = raise_integer(ctx, value, 64)?;
                    let precision = bits(ctx, &value, 64)?;
                    let precision = match u8::try_from(precision)
                        .map_err(|_| invalid("unknown native precision enum"))?
                    {
                        0 => Precision::Default,
                        1 => Precision::High,
                        2 => Precision::Highest,
                        _ => return Err(invalid("unknown native precision enum")),
                    };
                    precision
                        .verify(ctx)
                        .map_err(|error| invalid(error.to_string()))?;
                    Ok(Box::new(precision) as AttrObj)
                })
                .collect::<Result<Vec<_>, TextError>>()?;
            Ok(Box::new(VecAttr::new(values)))
        }
        Kind::Attribute => Ok(value),
        Kind::Bool => {
            if value.downcast_ref::<BoolAttr>().is_some() {
                Ok(value)
            } else {
                Err(invalid("expected native boolean attribute"))
            }
        }
        Kind::I32 => raise_integer(ctx, &value, 32),
        Kind::I64 => raise_integer(ctx, &value, 64),
        Kind::I64Array => {
            let array = value
                .downcast_ref::<VecAttr>()
                .ok_or_else(|| invalid("native integer list required"))?;
            let values = array
                .0
                .iter()
                .map(|value| raise_integer(ctx, value, 64))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Box::new(DenseArrayAttr {
                element: ElementType::I64.native(ctx),
                values: VecAttr::new(values),
            }))
        }
        Kind::I64Pairs => {
            let values = value
                .downcast_ref::<VecAttr>()
                .ok_or_else(|| invalid("expected integer padding pairs"))?;
            if values.0.len() % 2 != 0 {
                return Err(invalid("padding must contain low/high pairs"));
            }
            let values = values
                .0
                .iter()
                .map(|value| signed_i64(ctx, value))
                .collect::<Result<Vec<_>, _>>()?;
            let rows = i64::try_from(values.len() / 2)
                .map_err(|_| invalid("padding rank exceeds tensor dimensions"))?;
            let element_type = IntegerType::get(ctx, 64, Signedness::Signless);
            let elements = values
                .into_iter()
                .map(|value| {
                    Box::new(IntegerAttr::new(
                        element_type,
                        APInt::from_i64(value, bw(64)),
                    )) as AttrObj
                })
                .collect();
            let ty: TypeHandle =
                TensorType::get(ctx, vec![rows, 2], ElementType::I64.native(ctx)).into();
            Ok(Box::new(DenseElementsAttr::new(ty, VecAttr::new(elements))))
        }
    }
}

fn signed_i64(ctx: &Context, value: &AttrObj) -> Result<i64, TextError> {
    let integer = value
        .downcast_ref::<IntegerAttr>()
        .ok_or_else(|| invalid("i64 integer element required"))?;
    integer
        .verify(ctx)
        .map_err(|failure| invalid(failure.to_string()))?;
    let ty = integer.get_type().deref(ctx);
    if ty.width() != 64 || ty.is_unsigned() {
        return Err(invalid("padding elements must be signed or signless i64"));
    }
    integer
        .value()
        .to_string_decimal(true)
        .parse::<i64>()
        .map_err(|_| invalid("padding element does not fit i64"))
}

fn signed_integer_list(ctx: &Context, values: impl IntoIterator<Item = i64>) -> VecAttr {
    VecAttr::new(
        values
            .into_iter()
            .map(|value| {
                Box::new(IntegerAttr::new(
                    IntegerType::get(ctx, 64, Signedness::Signed),
                    APInt::from_i64(value, bw(64)),
                )) as AttrObj
            })
            .collect(),
    )
}

pub(super) fn build(
    ctx: &mut Context,
    name: &str,
    results: Vec<TypeHandle>,
    operands: Vec<Value>,
    source: &mut AssemblyAttributes,
) -> Result<Option<Ptr<Operation>>, TextError> {
    let Some((builder, fields)) = schema(name) else {
        return Ok(None);
    };
    let mut attributes = AttributeDict::default();
    for field in fields {
        let Some(value) = source.remove(field.source) else {
            if field.required {
                return Err(invalid(format!("missing {}", field.source)));
            }
            continue;
        };
        attributes
            .0
            .insert(Identifier::new(field.native), lower(ctx, field, value)?);
    }
    let operation = builder(ctx, results, operands);
    operation.deref_mut(ctx).attributes = attributes;
    Ok(Some(operation))
}

pub(super) fn source_attributes(
    ctx: &Context,
    name: &str,
    attributes: &mut AttributeDict,
) -> Result<(), TextError> {
    super::translation::unpack_attributes(ctx, name, attributes)?;
    let Some((_, fields)) = schema(name) else {
        return Ok(());
    };
    for field in fields {
        let Some(value) = attributes.0.remove(&Identifier::new(field.native)) else {
            if field.required {
                return Err(invalid(format!("missing native {}", field.native)));
            }
            continue;
        };
        if matches!(field.kind, Kind::Unit)
            && value
                .downcast_ref::<BoolAttr>()
                .is_some_and(|value| !bool::from(value.clone()))
        {
            continue;
        }
        if attributes
            .0
            .insert(Identifier::new(field.source), raise(ctx, field, value)?)
            .is_some()
        {
            return Err(invalid(format!(
                "native attributes collide with source {}",
                field.source
            )));
        }
    }
    Ok(())
}
