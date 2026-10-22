#![doc = include_str!("../README.md")]
#![warn(missing_docs)]
#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used))]

use pliron::{attribute::Attribute, r#type::Type};

pub mod analysis;
pub mod attributes;
pub mod evaluator;
pub mod inference;
pub mod ir;
pub mod literal;
pub mod opcode;
pub mod ops;
pub mod scalar;
pub mod shape;
pub mod sharding;
pub mod types;
pub mod verification;

mod error;

pub use error::{Diagnostic, RecoveryCategory};
pub use pliron;
/// Shared error envelope and result type used by fallible IR APIs.
pub use pliron::result::{Error, Result};

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
        types::PrimitiveType,
        shape::Dimension,
        shape::Tile,
        shape::SplitConfig,
        shape::DimLevelKind,
        shape::Layout,
        literal::Literal,
        sharding::ShardingKind,
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
        sharding::IotaReplicaGroupList,
        sharding::MeshAxesReplicaGroupList,
        ir::OriginalArray,
        ir::OriginalValueElement,
        ir::OriginalValue,
        opcode::Opcode,
    );
    attributes::register(ctx);
    ir::register(ctx);
    ops::register(ctx);
}
