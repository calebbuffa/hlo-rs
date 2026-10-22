// SPDX-License-Identifier: Apache-2.0

//! Attribute-dictionary keys for native HLO instruction and module attributes.
//!
//! Each constant is the serialized `hlo_*` identifier, written once. Several keys hold
//! different attribute types depending on the opcode, so they are plain names rather
//! than type-associated keys.

/// `hlo_aliases`.
pub const ALIASES: &str = "hlo_aliases";

/// `hlo_backend_config`.
pub const BACKEND_CONFIG: &str = "hlo_backend_config";

/// `hlo_batch_groups`.
pub const BATCH_GROUPS: &str = "hlo_batch_groups";

/// `hlo_block_scaling`.
pub const BLOCK_SCALING: &str = "hlo_block_scaling";

/// `hlo_buffer_donors`.
pub const BUFFER_DONORS: &str = "hlo_buffer_donors";

/// `hlo_callees`.
pub const CALLEES: &str = "hlo_callees";

/// `hlo_channel`.
pub const CHANNEL: &str = "hlo_channel";

/// `hlo_constant_folding`.
pub const CONSTANT_FOLDING: &str = "hlo_constant_folding";

/// `hlo_constrain_layout`.
pub const CONSTRAIN_LAYOUT: &str = "hlo_constrain_layout";

/// `hlo_dimensions`.
pub const DIMENSIONS: &str = "hlo_dimensions";

/// `hlo_direction`.
pub const DIRECTION: &str = "hlo_direction";

/// `hlo_domain_entry_sharding`.
pub const DOMAIN_ENTRY_SHARDING: &str = "hlo_domain_entry_sharding";

/// `hlo_domain_exit_sharding`.
pub const DOMAIN_EXIT_SHARDING: &str = "hlo_domain_exit_sharding";

/// `hlo_execution_thread`.
pub const EXECUTION_THREAD: &str = "hlo_execution_thread";

/// `hlo_feature_groups`.
pub const FEATURE_GROUPS: &str = "hlo_feature_groups";

/// `hlo_flags`.
pub const FLAGS: &str = "hlo_flags";

/// `hlo_fusion_parameters`.
pub const FUSION_PARAMETERS: &str = "hlo_fusion_parameters";

/// `hlo_geometry`.
pub const GEOMETRY: &str = "hlo_geometry";

/// `hlo_has_side_effect`.
pub const HAS_SIDE_EFFECT: &str = "hlo_has_side_effect";

/// `hlo_index`.
pub const INDEX: &str = "hlo_index";

/// `hlo_is_composite`.
pub const IS_COMPOSITE: &str = "hlo_is_composite";

/// `hlo_literal`.
pub const LITERAL: &str = "hlo_literal";

/// `hlo_metadata`.
pub const METADATA: &str = "hlo_metadata";

/// `hlo_metadata_payloads`.
pub const METADATA_PAYLOADS: &str = "hlo_metadata_payloads";

/// `hlo_module_metadata`.
pub const MODULE_METADATA: &str = "hlo_module_metadata";

/// `hlo_name`.
pub const NAME: &str = "hlo_name";

/// `hlo_named_sharding`.
pub const NAMED_SHARDING: &str = "hlo_named_sharding";

/// `hlo_operand_layouts`.
pub const OPERAND_LAYOUTS: &str = "hlo_operand_layouts";

/// `hlo_order`.
pub const ORDER: &str = "hlo_order";

/// `hlo_original_value`.
pub const ORIGINAL_VALUE: &str = "hlo_original_value";

/// `hlo_output_operand_aliases`.
pub const OUTPUT_OPERAND_ALIASES: &str = "hlo_output_operand_aliases";

/// `hlo_precision`.
pub const PRECISION: &str = "hlo_precision";

/// `hlo_result_accuracy`.
pub const RESULT_ACCURACY: &str = "hlo_result_accuracy";

/// `hlo_schedule`.
pub const SCHEDULE: &str = "hlo_schedule";

/// `hlo_sharding`.
pub const SHARDING: &str = "hlo_sharding";

/// `hlo_slice_sizes`.
pub const SLICE_SIZES: &str = "hlo_slice_sizes";

/// `hlo_sparsity`.
pub const SPARSITY: &str = "hlo_sparsity";

/// `hlo_statistics_viz`.
pub const STATISTICS_VIZ: &str = "hlo_statistics_viz";

/// `hlo_target`.
pub const TARGET: &str = "hlo_target";

/// `hlo_use_global_device_ids`.
pub const USE_GLOBAL_DEVICE_IDS: &str = "hlo_use_global_device_ids";

/// `hlo_window`.
pub const WINDOW: &str = "hlo_window";

/// `hlo_backend_config_payload`.
pub const BACKEND_CONFIG_PAYLOAD: &str = "hlo_backend_config_payload";

/// `hlo_control_predecessors`.
pub const CONTROL_PREDECESSORS: &str = "hlo_control_predecessors";

/// `hlo_frontend_attributes`.
pub const FRONTEND_ATTRIBUTES: &str = "hlo_frontend_attributes";

/// `hlo_input_output_aliases`.
pub const INPUT_OUTPUT_ALIASES: &str = "hlo_input_output_aliases";
