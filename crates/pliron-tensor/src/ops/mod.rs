//! Tensor operations, builders, and verifiers.

use pliron::{context::Context, derive::op_interface, op::Op, result::Result, value::Value};

/// One static integer or dynamic SSA `index` value.
#[derive(Clone, Copy)]
pub enum MixedIndex {
    /// Compile-time value.
    Static(i64),
    /// Runtime index operand.
    Dynamic(Value),
}

/// Marker for a parallel-combining parent of `parallel_insert_slice`.
///
/// A dialect implementing SCF-style parallel regions can implement this on
/// its combining operation, matching MLIR's `InParallelOpInterface`.
#[op_interface]
pub trait InParallelOpInterface {
    /// Verifies the interface's parent-specific contract.
    fn verify(_op: &dyn Op, _ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}

pliron::dict_key!(
    #[doc = "Operand segment storage key."]
    SEGMENTS,
    "tensor_operand_segment_sizes"
);
pliron::dict_key!(
    #[doc = "Static slice offset storage key."]
    OFFSETS,
    "tensor_static_offsets"
);
pliron::dict_key!(
    #[doc = "Static slice size storage key."]
    SIZES,
    "tensor_static_sizes"
);
pliron::dict_key!(
    #[doc = "Static slice stride storage key."]
    STRIDES,
    "tensor_static_strides"
);
pliron::dict_key!(
    #[doc = "Reassociation storage key."]
    REASSOCIATION,
    "tensor_reassociation"
);
pliron::dict_key!(
    #[doc = "Expansion output shape storage key."]
    OUTPUT_SHAPE,
    "tensor_static_output_shape"
);
pliron::dict_key!(
    #[doc = "Low padding storage key."]
    LOW,
    "tensor_static_low"
);
pliron::dict_key!(
    #[doc = "High padding storage key."]
    HIGH,
    "tensor_static_high"
);
pliron::dict_key!(
    #[doc = "Padding folding control storage key."]
    NOFOLD,
    "tensor_nofold"
);
pliron::dict_key!(
    #[doc = "Concatenation dimension storage key."]
    DIM,
    "tensor_concat_dim"
);
pliron::dict_key!(
    #[doc = "Gather dimensions storage key."]
    GATHER_DIMS,
    "tensor_gather_dims"
);
pliron::dict_key!(
    #[doc = "Scatter dimensions storage key."]
    SCATTER_DIMS,
    "tensor_scatter_dims"
);
pliron::dict_key!(
    #[doc = "Index uniqueness promise storage key."]
    UNIQUE,
    "tensor_unique"
);

macro_rules! operation {
    ($ty:ident, $name:literal, $doc:literal) => {
        operation!(
            $ty,
            $name,
            $doc,
            "operands(CharSpace(`,`)) attr_dict `:` typesig"
        );
    };
    ($ty:ident, $name:literal, $doc:literal, $format:literal) => {
        #[doc = $doc]
        #[pliron::derive::pliron_op(name = $name, format = $format)]
        pub struct $ty;
        impl $ty {
            /// Returns the first result, or `None` for a result-less operation.
            pub fn result(&self, ctx: &pliron::context::Context) -> Option<pliron::value::Value> {
                self.get_operation().deref(ctx).results().next()
            }
        }
    };
}

mod regions;
mod reshape;
mod simple;
mod slices;
pub use regions::{BodyBuilder, GenerateOp, PadOp, YieldOp};
pub use reshape::{CollapseShapeOp, ExpandShapeOp, ReshapeOp};
pub use simple::{
    BitcastOp, CastOp, ConcatOp, DimOp, EmptyOp, ExtractOp, FromElementsOp, GatherOp, InsertOp,
    RankOp, ScatterOp, SplatOp,
};
pub use slices::{ExtractSliceOp, InsertSliceOp, ParallelInsertSliceOp, Slice};
