use super::same;
use pliron::{
    context::{Context, Ptr},
    operation::Operation,
    result::Result,
    r#type::TypeHandle,
};

#[derive(Clone, Copy, Debug)]
/// Execution topology used to validate collective group membership.
pub struct CollectiveContext {
    /// Number of replicated program instances.
    pub replica_count: i64,
    /// Number of partitions per program instance.
    pub partition_count: i64,
}
impl Default for CollectiveContext {
    fn default() -> Self {
        Self {
            replica_count: 1,
            partition_count: 1,
        }
    }
}

#[derive(Clone, Copy, Debug)]
/// Contextual policy for validating device assignments.
pub struct ShardingContext {
    /// Interpret device numbers as SPMD partition IDs.
    pub use_spmd_partitioning: bool,
    /// Validate concrete device IDs against the module topology.
    pub verify_device_numbers: bool,
}
impl Default for ShardingContext {
    fn default() -> Self {
        Self {
            use_spmd_partitioning: false,
            verify_device_numbers: true,
        }
    }
}

/// Caller-supplied policy for a particular native instruction.
pub type InstructionPredicate = fn(&Context, Ptr<Operation>) -> bool;
/// Caller-supplied target allocation size calculation.
pub type ShapeSize = fn(&Context, TypeHandle) -> Result<i64>;

#[derive(Clone, Copy, Debug)]
/// Semantic verification policies; defaults retain layout-insensitive HLO rules.
pub struct VerifyOptions {
    /// Require operation-specific layout and dynamic-shape consistency.
    pub layout_sensitive: bool,
    /// Permit floating-point precision differences where the opcode allows them.
    pub allow_mixed_precision: bool,
    /// Authorize operations that produce unbounded dynamic shapes.
    pub supported_unbounded_dynamic_op: Option<InstructionPredicate>,
    /// Decide which instructions may change operand layouts.
    pub instruction_can_change_layout: Option<InstructionPredicate>,
    /// Check scheduling metadata against instruction names.
    pub verify_instruction_name_unchanged: bool,
    /// Reject array layouts in host memory space 5.
    pub verify_no_host_memory_space: bool,
    /// Require reshape operations to preserve physical element order.
    pub verify_reshape_is_bitcast: bool,
    /// Enable operation-specific bitcast size relaxation.
    pub allow_bitcast_to_have_different_size: bool,
    /// Override target allocation sizes for strict alias/bitcast checks.
    pub shape_size: Option<ShapeSize>,
    /// Require nested to-apply computations to use the caller's thread.
    pub verify_call_nested_computation_thread_name: bool,
    /// Require broadcast dimensions to be strictly increasing.
    pub verify_broadcast_dimensions_order: bool,
    /// Check communication ordering for collective/channel deadlocks.
    pub check_for_collective_deadlocks: bool,
    /// Validate replica-group membership in addition to collective geometry.
    pub check_replica_groups: bool,
    /// Replica/partition topology when not overridden by module configuration.
    pub collective_context: CollectiveContext,
    /// Device-number and SPMD policy for contextual sharding checks.
    pub sharding_context: ShardingContext,
    /// Use caller topology rather than the module's non-default configuration.
    pub ignore_module_config: bool,
}

impl Default for VerifyOptions {
    fn default() -> Self {
        Self {
            layout_sensitive: false,
            allow_mixed_precision: false,
            supported_unbounded_dynamic_op: None,
            instruction_can_change_layout: None,
            verify_instruction_name_unchanged: false,
            verify_no_host_memory_space: false,
            verify_reshape_is_bitcast: false,
            allow_bitcast_to_have_different_size: false,
            shape_size: None,
            verify_call_nested_computation_thread_name: false,
            verify_broadcast_dimensions_order: false,
            check_for_collective_deadlocks: false,
            check_replica_groups: true,
            collective_context: CollectiveContext::default(),
            sharding_context: ShardingContext::default(),
            ignore_module_config: false,
        }
    }
}

impl VerifyOptions {
    pub(crate) fn size_of(self, ctx: &Context, shape: TypeHandle) -> Result<i64> {
        match self.shape_size {
            Some(size) => size(ctx, shape),
            None if crate::shape::has_unbounded(ctx, shape) => Ok(i64::MIN),
            None => crate::shape::byte_size(ctx, shape, -1),
        }
    }

    pub(crate) fn same_shapes(self, ctx: &Context, left: TypeHandle, right: TypeHandle) -> bool {
        if self.layout_sensitive {
            crate::shape::equal_including_layout(ctx, left, right)
        } else {
            same(ctx, left, right)
        }
    }
}
