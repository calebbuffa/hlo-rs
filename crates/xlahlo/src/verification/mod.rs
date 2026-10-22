//! Structural, semantic, and policy-driven verification of native HLO IR.

mod buffers;
pub(crate) mod control;
mod driver;
pub(crate) mod elementwise;
mod layout;
mod metadata;
pub(crate) mod movement;
pub(crate) mod numerical;
mod options;
mod sharding;
mod support;

pub use control::{async_chain_starts, async_wrapped_computations};
pub(crate) use driver::computation;
pub(crate) use driver::{returned, single_block};
pub use driver::{verify, verify_with_options};
pub use options::{
    CollectiveContext, InstructionPredicate, ShapeSize, ShardingContext, VerifyOptions,
};
pub(crate) use support::{
    error, instruction_error, instruction_types, optional_attr, require, same,
    verify_result_accuracy,
};
