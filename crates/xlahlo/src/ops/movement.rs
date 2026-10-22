//! shape and data-movement operations with explicitly typed geometry.

use crate::attributes::{
    Axes, Callees, GatherDimensionNumbers, Index, IndexFlags, PaddingConfig,
    ScatterDimensionNumbers, ShuffleMode, Slice,
};
use pliron::{context::Context, identifier::Identifier, op::Op, result::Result};

native_instruction!(
    CopyOp,
    "xlahlo.copy",
    Copy,
    crate::verification::movement::instruction,
);
native_instruction!(BroadcastOp, "xlahlo.broadcast", Broadcast, crate::verification::movement::instruction, dimensions: Axes);
native_instruction!(
    ReshapeOp,
    "xlahlo.reshape",
    Reshape,
    crate::verification::movement::instruction,
);

impl ReshapeOp {
    /// Output axis receiving the inferred dynamic factor, or -1 for automatic
    /// inference. The optional dimensions attribute follows `HloReshapeInstruction`.
    pub fn inferred_dimension(&self, ctx: &Context) -> Result<i64> {
        crate::verification::movement::reshape_inferred_dimension(ctx, self.op)
    }

    /// Select an output axis, or -1 for automatic inference. Verification checks
    /// that an explicit axis is valid for the result rank.
    pub fn set_inferred_dimension(&self, ctx: &Context, dimension: i64) {
        let dimensions = if dimension == -1 {
            vec![]
        } else {
            vec![dimension]
        };
        self.op.deref_mut(ctx).attributes.set(
            Identifier::new(crate::attributes::keys::DIMENSIONS),
            Axes(dimensions),
        );
    }
}

native_instruction!(
    DynamicReshapeOp,
    "xlahlo.dynamic_reshape",
    DynamicReshape,
    crate::verification::movement::instruction,
);
native_instruction!(
    BitcastOp,
    "xlahlo.bitcast",
    Bitcast,
    crate::verification::movement::instruction,
);
native_instruction!(TransposeOp, "xlahlo.transpose", Transpose, crate::verification::movement::instruction, dimensions: Axes);
native_instruction!(ReverseOp, "xlahlo.reverse", Reverse, crate::verification::movement::instruction, dimensions: Axes);
native_instruction!(SliceOp, "xlahlo.slice", Slice, crate::verification::movement::instruction, geometry: Slice);
native_instruction!(DynamicSliceOp, "xlahlo.dynamic_slice", DynamicSlice, crate::verification::movement::instruction, slice_sizes: Axes);
native_instruction!(
    DynamicUpdateSliceOp,
    "xlahlo.dynamic_update_slice",
    DynamicUpdateSlice,
    crate::verification::movement::instruction,
);
native_instruction!(ConcatenateOp, "xlahlo.concatenate", Concatenate, crate::verification::movement::instruction, index: Index);
native_instruction!(PadOp, "xlahlo.pad", Pad, crate::verification::movement::instruction, config: PaddingConfig);
native_instruction!(IotaOp, "xlahlo.iota", Iota, crate::verification::movement::instruction, index: Index);
native_instruction!(GatherOp, "xlahlo.gather", Gather, crate::verification::movement::instruction, dimensions: GatherDimensionNumbers, slice_sizes: Axes, flags: IndexFlags);
native_instruction!(ScatterOp, "xlahlo.scatter", Scatter, crate::verification::movement::instruction, dimensions: ScatterDimensionNumbers, callees: Callees, flags: IndexFlags);
native_instruction!(GetDimensionSizeOp, "xlahlo.get_dimension_size", GetDimensionSize, crate::verification::movement::instruction, index: Index);
native_instruction!(SetDimensionSizeOp, "xlahlo.set_dimension_size", SetDimensionSize, crate::verification::movement::instruction, index: Index);
native_instruction!(ShuffleOp, "xlahlo.shuffle", Shuffle, crate::verification::movement::instruction, dimensions: Axes, mode: ShuffleMode);

/// Register all shape and data-movement operations in `ctx`.
pub(crate) fn register(ctx: &mut Context) {
    CopyOp::register(ctx);
    BroadcastOp::register(ctx);
    ReshapeOp::register(ctx);
    DynamicReshapeOp::register(ctx);
    BitcastOp::register(ctx);
    TransposeOp::register(ctx);
    ReverseOp::register(ctx);
    SliceOp::register(ctx);
    DynamicSliceOp::register(ctx);
    DynamicUpdateSliceOp::register(ctx);
    ConcatenateOp::register(ctx);
    PadOp::register(ctx);
    IotaOp::register(ctx);
    GatherOp::register(ctx);
    ScatterOp::register(ctx);
    GetDimensionSizeOp::register(ctx);
    SetDimensionSizeOp::register(ctx);
    ShuffleOp::register(ctx);
}
