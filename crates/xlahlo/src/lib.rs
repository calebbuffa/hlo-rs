#![doc = include_str!("../README.md")]

mod accuracy;
mod error;
pub mod inference;
pub mod interchange;
mod ir;
mod literal;
mod opcode;
mod original_value;
pub mod proto;
mod replica_groups;
mod shape;
mod sharding;
pub mod text;
mod verification;

pub use error::{Error, Result};
pub use ir::{
    ClonedComputation, ComputationId, EffectContext, Instruction, InstructionId, Module,
    ModuleConfig,
};
pub use literal::Literal;
pub use opcode::Opcode;
pub use original_value::OriginalValue;
pub use proto::xla_data::PrimitiveType;
pub use replica_groups::MeshAxesReplicaGroupList;
pub use shape::{
    COMBINE_DIMENSION, Dimension, MATCH_DIMENSION, Shape, UNBOUNDED_SIZE, validate_layout,
};
pub use sharding::{AxisRef, Mesh, NamedSharding, Sharding};
pub use verification::{CollectiveContext, ShardingContext, VerifyOptions};

/// OpenXLA revision used to generate this build's schema and opcode catalog.
pub const XLA_SOURCE_COMMIT: &str = env!("XLA_SOURCE_COMMIT");
