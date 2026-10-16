use crate::Attributes;

use super::{
    BytecodeContainer, BytecodeError, BytecodeErrorKind as Kind, payload::PayloadDecoder,
    primitives::Cursor,
};

pub(super) fn read(
    container: &BytecodeContainer<'_>,
    payloads: &PayloadDecoder<'_, '_>,
    name: &str,
    registered: bool,
    cursor: &mut Cursor<'_>,
) -> Result<Attributes, BytecodeError> {
    let id = cursor.varint()?;
    let entry = usize::try_from(id)
        .ok()
        .and_then(|id| container.properties().get(id))
        .ok_or_else(|| {
            cursor.error(Kind::InvalidIndex {
                table: "property",
                index: id,
            })
        })?;
    let mut cursor = Cursor::new(entry.data, entry.offset);
    if !registered {
        let attribute = payloads.attribute(cursor.varint()?, 0)?;
        cursor.finish()?;
        return match attribute {
            crate::Attribute::Dictionary(values) => Ok(values),
            _ => Err(cursor.error(Kind::InvalidPayload(
                "dictionary properties for unregistered operation",
            ))),
        };
    }
    let fields = match name {
        "builtin.module" => "sym_name? sym_visibility?",
        "func.func" => "arg_attrs? function_type no_inline? res_attrs? sym_name sym_visibility?",
        "func.call" => "arg_attrs? callee no_inline? res_attrs?",
        "func.call_indirect" => "arg_attrs? res_attrs?",
        "func.constant" | "stablehlo.constant" => "value",
        "stablehlo.all_gather" => {
            "all_gather_dim channel_handle? replica_groups use_global_device_ids?"
        }
        "stablehlo.all_reduce" => "channel_handle? replica_groups use_global_device_ids?",
        "stablehlo.all_to_all" => {
            "channel_handle? concat_dimension replica_groups split_count split_dimension"
        }
        "stablehlo.batch_norm_grad"
        | "stablehlo.batch_norm_inference"
        | "stablehlo.batch_norm_training" => "epsilon feature_index",
        "stablehlo.broadcast_in_dim" => "broadcast_dimensions",
        "stablehlo.broadcast" => "broadcast_sizes",
        "stablehlo.cbrt"
        | "stablehlo.cosine"
        | "stablehlo.exponential"
        | "stablehlo.exponential_minus_one"
        | "stablehlo.log"
        | "stablehlo.log_plus_one"
        | "stablehlo.logistic"
        | "stablehlo.rsqrt"
        | "stablehlo.sine"
        | "stablehlo.sqrt"
        | "stablehlo.tan"
        | "stablehlo.tanh" => "result_accuracy?",
        "stablehlo.cholesky" => "lower?",
        "stablehlo.collective_broadcast" => "channel_handle? has_dynamic_root? replica_groups",
        "stablehlo.collective_permute" => "channel_handle? source_target_pairs",
        "stablehlo.collective_reduce" => {
            "channel_handle? has_dynamic_root? replica_groups use_global_device_ids?"
        }
        "stablehlo.compare" => "compare_type? comparison_direction",
        "stablehlo.composite" => "composite_attributes? decomposition name version?",
        "stablehlo.concatenate"
        | "stablehlo.get_dimension_size"
        | "stablehlo.set_dimension_size" => "dimension",
        "stablehlo.convolution" => {
            "batch_group_count dimension_numbers feature_group_count lhs_dilation? padding? precision_config? rhs_dilation? window_reversal? window_strides?"
        }
        "stablehlo.cross-replica-sum" => "replica_groups",
        "stablehlo.custom_call" => {
            "api_version? backend_config? call_target_name called_computations? has_side_effect? operand_layouts? output_operand_aliases? result_layouts? result_tilings?"
        }
        "stablehlo.dot_general" => "algorithm? dot_dimension_numbers precision_config?",
        "stablehlo.dot" => "precision_config?",
        "stablehlo.dynamic_broadcast_in_dim" => {
            "broadcast_dimensions known_expanding_dimensions? known_nonexpanding_dimensions?"
        }
        "stablehlo.dynamic_conv" => {
            "batch_group_count dimension_numbers feature_group_count lhs_dilation? precision_config? rhs_dilation? window_reversal? window_strides?"
        }
        "stablehlo.dynamic_gather" => "dimension_numbers indices_are_sorted?",
        "stablehlo.dynamic_iota" | "stablehlo.iota" => "iota_dimension",
        "stablehlo.dynamic_slice" => "slice_sizes",
        "stablehlo.einsum" | "stablehlo.unary_einsum" => "einsum_config",
        "stablehlo.fft" => "fft_length fft_type",
        "stablehlo.gather" => "dimension_numbers indices_are_sorted? slice_sizes",
        "stablehlo.get_tuple_element" => "index",
        "stablehlo.infeed" => "infeed_config? layout?",
        "stablehlo.map" | "stablehlo.reduce" | "stablehlo.reverse" => "dimensions",
        "stablehlo.outfeed" => "outfeed_config?",
        "stablehlo.pad" => "edge_padding_high edge_padding_low interior_padding",
        "stablehlo.recv" | "stablehlo.send" => {
            "channel_handle is_host_transfer? source_target_pairs?"
        }
        "stablehlo.reduce_precision" => "exponent_bits mantissa_bits",
        "stablehlo.reduce_scatter" => {
            "channel_handle? replica_groups scatter_dimension use_global_device_ids?"
        }
        "stablehlo.reduce_window" => {
            "base_dilations? padding? window_dilations? window_dimensions window_strides?"
        }
        "stablehlo.rng_bit_generator" => "rng_algorithm",
        "stablehlo.rng" => "rng_distribution",
        "stablehlo.scatter" => "indices_are_sorted? scatter_dimension_numbers unique_indices?",
        "stablehlo.select_and_scatter" => "padding? window_dimensions? window_strides?",
        "stablehlo.slice" => "limit_indices start_indices strides",
        "stablehlo.sort" => "dimension? is_stable?",
        "stablehlo.torch_index_select" => "batch_dims dim",
        "stablehlo.transpose" => "permutation",
        "stablehlo.triangular_solve" => "left_side lower transpose_a unit_diagonal",
        _ => return Err(cursor.error(Kind::Unsupported("operation property schema"))),
    };
    let mut values = Attributes::new();
    for field in fields.split_whitespace() {
        let (name, optional) = field
            .strip_suffix('?')
            .map_or((field, false), |name| (name, true));
        let id = cursor.varint()?;
        if optional && id & 1 == 0 {
            continue;
        }
        let attribute = payloads.attribute(if optional { id >> 1 } else { id }, 0)?;
        values.insert(name.to_string(), attribute);
    }
    cursor.finish()?;
    Ok(values)
}
