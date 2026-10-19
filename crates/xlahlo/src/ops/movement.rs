//! Shape and data-movement operations with explicitly typed geometry.

use crate::attributes::{
    Axes, Callees, GatherDimensionNumbers, Index, IndexFlags, PaddingConfig,
    ScatterDimensionNumbers, ShuffleMode, Slice,
};
use pliron::{context::Context, op::Op};

native_instruction!(
    CopyOp,
    "hlo.copy",
    Copy,
    crate::verification::movement::instruction,
);
native_instruction!(BroadcastOp, "hlo.broadcast", Broadcast, crate::verification::movement::instruction, dimensions: Axes);
native_instruction!(
    ReshapeOp,
    "hlo.reshape",
    Reshape,
    crate::verification::movement::instruction,
);
native_instruction!(DynamicReshapeOp, "hlo.dynamic_reshape", DynamicReshape, crate::verification::movement::instruction, dimensions: Axes);
native_instruction!(
    BitcastOp,
    "hlo.bitcast",
    Bitcast,
    crate::verification::movement::instruction,
);
native_instruction!(TransposeOp, "hlo.transpose", Transpose, crate::verification::movement::instruction, dimensions: Axes);
native_instruction!(ReverseOp, "hlo.reverse", Reverse, crate::verification::movement::instruction, dimensions: Axes);
native_instruction!(SliceOp, "hlo.slice", Slice, crate::verification::movement::instruction, geometry: Slice);
native_instruction!(DynamicSliceOp, "hlo.dynamic_slice", DynamicSlice, crate::verification::movement::instruction, slice_sizes: Axes);
native_instruction!(
    DynamicUpdateSliceOp,
    "hlo.dynamic_update_slice",
    DynamicUpdateSlice,
    crate::verification::movement::instruction,
);
native_instruction!(ConcatenateOp, "hlo.concatenate", Concatenate, crate::verification::movement::instruction, index: Index);
native_instruction!(PadOp, "hlo.pad", Pad, crate::verification::movement::instruction, config: PaddingConfig);
native_instruction!(IotaOp, "hlo.iota", Iota, crate::verification::movement::instruction, index: Index);
native_instruction!(GatherOp, "hlo.gather", Gather, crate::verification::movement::instruction, dimensions: GatherDimensionNumbers, slice_sizes: Axes, flags: IndexFlags);
native_instruction!(ScatterOp, "hlo.scatter", Scatter, crate::verification::movement::instruction, dimensions: ScatterDimensionNumbers, callees: Callees, flags: IndexFlags);
native_instruction!(GetDimensionSizeOp, "hlo.get_dimension_size", GetDimensionSize, crate::verification::movement::instruction, index: Index);
native_instruction!(SetDimensionSizeOp, "hlo.set_dimension_size", SetDimensionSize, crate::verification::movement::instruction, index: Index);
native_instruction!(ShuffleOp, "hlo.shuffle", Shuffle, crate::verification::movement::instruction, dimensions: Axes, mode: ShuffleMode);

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
