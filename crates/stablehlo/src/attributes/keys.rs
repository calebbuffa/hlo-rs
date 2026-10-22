//! String constants for StableHLO dictionary keys used in explicit lookups.

/// Attribute name used by `stablehlo.call` for its callee symbol.
pub const CALL_CALLEE: &str = "stablehlo_call_callee";
/// Attribute name for StableHLO discardable operation attributes.
pub const DISCARDABLE_ATTRIBUTES: &str = "stablehlo_discardable_attributes";
/// Attribute name for the `stablehlo.dot_general` algorithm configuration.
pub const DOT_ALGORITHM: &str = "stablehlo_dot_algorithm";
/// Attribute name for `stablehlo.dot_general` precision configuration.
pub const DOT_PRECISION: &str = "stablehlo_dot_precision";
/// Attribute name for `stablehlo.dot` precision configuration.
pub const COMPUTE_DOT_PRECISION: &str = "stablehlo_compute_dot_precision";
/// Attribute name for a custom call's API version.
pub const CUSTOM_CALL_API_VERSION: &str = "stablehlo_custom_call_api_version";
/// Attribute name for a custom call's backend configuration.
pub const CUSTOM_CALL_BACKEND_CONFIG: &str = "stablehlo_custom_call_backend_config";
/// Attribute name for a custom call's called computations.
pub const CUSTOM_CALL_CALLED_COMPUTATIONS: &str = "stablehlo_custom_call_called_computations";
/// Attribute name for a custom call's side-effect flag.
pub const CUSTOM_CALL_HAS_SIDE_EFFECT: &str = "stablehlo_custom_call_has_side_effect";
/// Attribute name for a custom call's operand layouts.
pub const CUSTOM_CALL_OPERAND_LAYOUTS: &str = "stablehlo_custom_call_operand_layouts";
/// Attribute name for a custom call's output-operand aliases.
pub const CUSTOM_CALL_OUTPUT_OPERAND_ALIASES: &str = "stablehlo_custom_call_output_operand_aliases";
/// Attribute name for a custom call's result layouts.
pub const CUSTOM_CALL_RESULT_LAYOUTS: &str = "stablehlo_custom_call_result_layouts";
/// Attribute name for a custom call's target.
pub const CUSTOM_CALL_TARGET: &str = "stablehlo_custom_call_target";
/// Attribute name for dynamic broadcast's known-expanding dimensions.
pub const DYNAMIC_BROADCAST_KNOWN_EXPANDING: &str = "stablehlo_dynamic_broadcast_known_expanding";
/// Attribute name for dynamic broadcast's known-nonexpanding dimensions.
pub const DYNAMIC_BROADCAST_KNOWN_NONEXPANDING: &str =
    "stablehlo_dynamic_broadcast_known_nonexpanding";
/// Attribute name for an elementwise operation's result-accuracy configuration.
pub const RESULT_ACCURACY: &str = "stablehlo_result_accuracy";
