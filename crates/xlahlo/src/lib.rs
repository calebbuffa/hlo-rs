#![doc = include_str!("../README.md")]
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use pliron::{attribute::Attribute, r#type::Type};

pub mod attributes;
pub mod effects;
mod error;
pub mod inference;
pub mod ir;
pub mod literal;
pub mod opcode;
pub mod ops;
pub mod original_value;
pub mod primitive;
pub mod replica_groups;
pub mod shape;
pub mod sharding;
pub mod types;
pub mod verification;

pub use error::Diagnostic;
pub use pliron;

/// Install HLO's typed parser/printer and operation registrations.
pub fn register(ctx: &mut pliron::context::Context) {
    types::TensorType::register(ctx);
    types::TupleType::register(ctx);
    types::TokenType::register(ctx);
    types::OpaqueType::register(ctx);
    types::BufferType::register(ctx);
    macro_rules! attrs { ($($ty:ty),* $(,)?) => { $(<$ty>::register::<$ty>(ctx);)* }; }
    attrs!(
        types::ElementType,
        primitive::PrimitiveType,
        shape::Dimension,
        shape::Tile,
        shape::SplitConfig,
        shape::DimLevelType,
        shape::Layout,
        literal::Literal,
        sharding::ShardingType,
        sharding::ShardGroup,
        sharding::Sharding,
        sharding::MeshAxis,
        sharding::IotaTransform,
        sharding::Mesh,
        sharding::SubAxis,
        sharding::AxisRef,
        sharding::DimensionSharding,
        sharding::ReductionOp,
        sharding::NamedSharding,
        replica_groups::IotaReplicaGroupList,
        replica_groups::MeshAxesReplicaGroupList,
        original_value::OriginalArray,
        original_value::OriginalValueElement,
        original_value::OriginalValue,
        opcode::Opcode,
    );
    attributes::register(ctx);
    ir::register(ctx);
    ops::register(ctx);
}
