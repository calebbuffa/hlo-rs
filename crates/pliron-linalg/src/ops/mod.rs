//! `linalg` operation definitions, builders, and verification.
//!
//! Operations use destination-passing style on ranked tensors: each result has
//! the type of its `outs` operand. `new` constructors allocate operations
//! without verification; `build` constructors on region operations populate
//! and verify a body in one step. Use
//! [`pliron::common_traits::Verify::verify`] before consuming externally
//! supplied or manually edited IR.
//!
//! Coverage follows upstream MLIR: every operation of
//! `LinalgNamedStructuredOps.yaml` ([`named`] module, generated),
//! `LinalgStructuredOps.td`, and `LinalgRelayoutOps.td`. Operations whose
//! scalar computation is implied by their name store no body region.
//!
//! Operand and result accessors assume the operation has its declared arity;
//! malformed IR can panic when an operand or result is missing.

// Keys shared by several operations are declared once: pliron rejects a key
// declared by more than one `#[pliron_op(attributes = ...)]`.
pliron::dict_key!(
    #[doc = "Dictionary key of the shared `linalg_cast` attribute."]
    ATTR_KEY_LINALG_CAST,
    "linalg_cast"
);
pliron::dict_key!(
    #[doc = "Dictionary key of the shared `linalg_dilations` attribute."]
    ATTR_KEY_LINALG_DILATIONS,
    "linalg_dilations"
);
pliron::dict_key!(
    #[doc = "Dictionary key of the shared `linalg_dimensions` attribute."]
    ATTR_KEY_LINALG_DIMENSIONS,
    "linalg_dimensions"
);
pliron::dict_key!(
    #[doc = "Dictionary key of the shared `linalg_fmr` attribute."]
    ATTR_KEY_LINALG_FMR,
    "linalg_fmr"
);
pliron::dict_key!(
    #[doc = "Dictionary key of the shared `linalg_indexing_maps` attribute."]
    ATTR_KEY_LINALG_INDEXING_MAPS,
    "linalg_indexing_maps"
);
pliron::dict_key!(
    #[doc = "Dictionary key of the shared `linalg_inner_dims_pos` attribute."]
    ATTR_KEY_LINALG_INNER_DIMS_POS,
    "linalg_inner_dims_pos"
);
pliron::dict_key!(
    #[doc = "Dictionary key of the shared `linalg_operand_segment_sizes` attribute."]
    ATTR_KEY_LINALG_OPERAND_SEGMENT_SIZES,
    "linalg_operand_segment_sizes"
);
pliron::dict_key!(
    #[doc = "Dictionary key of the shared `linalg_outer_dims_perm` attribute."]
    ATTR_KEY_LINALG_OUTER_DIMS_PERM,
    "linalg_outer_dims_perm"
);
pliron::dict_key!(
    #[doc = "Dictionary key of the shared `linalg_static_inner_tiles` attribute."]
    ATTR_KEY_LINALG_STATIC_INNER_TILES,
    "linalg_static_inner_tiles"
);
pliron::dict_key!(
    #[doc = "Dictionary key of the shared `linalg_strides` attribute."]
    ATTR_KEY_LINALG_STRIDES,
    "linalg_strides"
);

pub mod named;
mod regions;
mod relayout;
mod structured;

pub use named::*;
pub use regions::{BodyBuilder, GenericOp, IndexOp, MapOp, ReduceOp, YieldOp};
pub use relayout::{
    BroadcastOp, DYNAMIC_TILE, PackOp, SoftmaxOp, TileSize, TransposeOp, UnPackOp,
    WinogradFilterTransformOp, WinogradInputTransformOp, WinogradOutputTransformOp,
};
pub use structured::{
    BatchMatmulOp, BatchReduceMatmulOp, ContractOp, ElementwiseOp, MatmulOp, ScaledContractOp,
};
