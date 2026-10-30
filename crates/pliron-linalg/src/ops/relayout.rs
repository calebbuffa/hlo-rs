//! Data-movement and specialized structured operations defined in MLIR's
//! `LinalgStructuredOps.td` and `LinalgRelayoutOps.td`: `transpose`,
//! `broadcast`, `softmax`, the Winograd transforms, `pack`, and `unpack`.
//!
//! These operations carry no body region.

use crate::{
    attributes::{I64ArrayAttr, WinogradFmr, WinogradFmrAttr},
    support::*,
};
use pliron::{
    builtin::attributes::OperandSegmentSizesAttr,
    common_traits::Verify,
    context::{Context, Ptr},
    derive::pliron_op,
    op::Op,
    operation::Operation,
    result::Result,
    r#type::{TypeHandle, Typed},
    value::Value,
};
use pliron_arith::shaped;
use stablehlo::attributes::IndexAttr;

fn allocate<T: Op>(
    ctx: &mut Context,
    make: fn(Ptr<Operation>) -> T,
    operands: Vec<Value>,
    init: usize,
) -> T {
    let result = vec![operands[init].get_type(ctx)];
    make(Operation::new(
        ctx,
        T::get_concrete_op_info(),
        result,
        operands,
        vec![],
        0,
    ))
}

fn operand(ctx: &Context, op: Ptr<Operation>, i: usize) -> Value {
    op.deref(ctx).get_operand(i)
}

/// Verifies a two-operand relayout operation whose operands are ranked tensors.
///
/// Returns each operand's shape and element type.
fn ranked_shapes(
    ctx: &Context,
    op: Ptr<Operation>,
    inputs: usize,
) -> Result<Vec<(Vec<i64>, TypeHandle)>> {
    let operands = verify_dps(ctx, op, inputs)?;
    if operand_types(ctx, op)
        .iter()
        .any(|&ty| ranked(ctx, ty).is_none())
    {
        return Err(fail(ctx, op, "operands must be ranked tensors"));
    }
    Ok(operands)
}

/// Returns whether `values` is a permutation of `0..len`.
fn is_permutation(values: &[i64], len: usize) -> bool {
    let mut seen = vec![false; len];
    values.len() == len
        && values.iter().all(|&v| match usize::try_from(v) {
            Ok(v) if v < len && !seen[v] => {
                seen[v] = true;
                true
            }
            _ => false,
        })
}

/// Permutes the dimensions of `input` into `init`: `init[i] = input[permutation[i]]`.
#[pliron_op(
    name = "linalg.transpose",
    format = "operands(CharSpace(`,`)) attr_dict `:` typesig",
    interfaces = [pliron::builtin::op_interfaces::OneResultInterface],
    attributes = (linalg_permutation: I64ArrayAttr)
)]
pub struct TransposeOp;

impl TransposeOp {
    /// Allocates the operation; the result type is `init`'s type.
    pub fn new(ctx: &mut Context, input: Value, init: Value, permutation: Vec<i64>) -> Self {
        let op = allocate(ctx, |op| Self { op }, vec![input, init], 1);
        op.set_attr_linalg_permutation(ctx, I64ArrayAttr::new(permutation));
        op
    }
    /// Returns the input operand.
    pub fn input(&self, ctx: &Context) -> Value {
        operand(ctx, self.get_operation(), 0)
    }
    /// Returns the `init` operand.
    pub fn init(&self, ctx: &Context) -> Value {
        operand(ctx, self.get_operation(), 1)
    }
    /// Returns the permutation, or an empty list when absent.
    pub fn permutation(&self, ctx: &Context) -> Vec<i64> {
        self.get_attr_linalg_permutation(ctx)
            .map(|a| a.values.clone())
            .unwrap_or_default()
    }
}

impl Verify for TransposeOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        let shapes = ranked_shapes(ctx, op, 1)?;
        let (input, init) = (&shapes[0].0, &shapes[1].0);
        let perm = self.permutation(ctx);
        if input.len() != init.len() || !is_permutation(&perm, input.len()) {
            return Err(fail(
                ctx,
                op,
                "permutation must permute the input rank, equal to the init rank",
            ));
        }
        if perm.iter().zip(init).any(|(&p, &d)| input[p as usize] != d) {
            return Err(fail(
                ctx,
                op,
                "init dimensions must be the permuted input dimensions",
            ));
        }
        Ok(())
    }
}

/// Broadcasts `input` into `init` by adding the listed `init` dimensions.
#[pliron_op(
    name = "linalg.broadcast",
    format = "operands(CharSpace(`,`)) attr_dict `:` typesig",
    interfaces = [pliron::builtin::op_interfaces::OneResultInterface]
)]
pub struct BroadcastOp;
crate::pliron_arith::shared_attr!(
    BroadcastOp,
    super::ATTR_KEY_LINALG_DIMENSIONS,
    I64ArrayAttr,
    get_attr_linalg_dimensions,
    set_attr_linalg_dimensions
);

impl BroadcastOp {
    /// Allocates the operation; the result type is `init`'s type.
    pub fn new(ctx: &mut Context, input: Value, init: Value, dimensions: Vec<i64>) -> Self {
        let op = allocate(ctx, |op| Self { op }, vec![input, init], 1);
        op.set_attr_linalg_dimensions(ctx, I64ArrayAttr::new(dimensions));
        op
    }
    /// Returns the input operand.
    pub fn input(&self, ctx: &Context) -> Value {
        operand(ctx, self.get_operation(), 0)
    }
    /// Returns the `init` operand.
    pub fn init(&self, ctx: &Context) -> Value {
        operand(ctx, self.get_operation(), 1)
    }
    /// Returns the added dimensions, or an empty list when absent.
    pub fn dimensions(&self, ctx: &Context) -> Vec<i64> {
        self.get_attr_linalg_dimensions(ctx)
            .map(|a| a.values.clone())
            .unwrap_or_default()
    }
}

impl Verify for BroadcastOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        let shapes = ranked_shapes(ctx, op, 1)?;
        let (input, init) = (&shapes[0].0, &shapes[1].0);
        let dims = self.dimensions(ctx);
        if input.len() + dims.len() != init.len() {
            return Err(fail(
                ctx,
                op,
                "input rank plus added dimensions must equal the init rank",
            ));
        }
        let mut added = vec![false; init.len()];
        for &d in &dims {
            match usize::try_from(d) {
                Ok(d) if d < init.len() && !added[d] => added[d] = true,
                _ => {
                    return Err(fail(
                        ctx,
                        op,
                        "added dimensions must be unique and in range",
                    ));
                }
            }
        }
        let kept: Vec<i64> = init
            .iter()
            .zip(&added)
            .filter(|(_, a)| !**a)
            .map(|(&d, _)| d)
            .collect();
        if kept != *input {
            return Err(fail(
                ctx,
                op,
                "init dimensions not added must equal the input dimensions",
            ));
        }
        Ok(())
    }
}

/// Softmax of `input` along `dimension`, written to `output`.
#[pliron_op(
    name = "linalg.softmax",
    format = "operands(CharSpace(`,`)) attr_dict `:` typesig",
    interfaces = [pliron::builtin::op_interfaces::OneResultInterface],
    attributes = (linalg_dimension: IndexAttr)
)]
pub struct SoftmaxOp;

impl SoftmaxOp {
    /// Allocates the operation; the result type is `output`'s type.
    pub fn new(ctx: &mut Context, input: Value, output: Value, dimension: i64) -> Self {
        let op = allocate(ctx, |op| Self { op }, vec![input, output], 1);
        op.set_attr_linalg_dimension(ctx, IndexAttr { value: dimension });
        op
    }
    /// Returns the input operand.
    pub fn input(&self, ctx: &Context) -> Value {
        operand(ctx, self.get_operation(), 0)
    }
    /// Returns the `output` operand.
    pub fn output(&self, ctx: &Context) -> Value {
        operand(ctx, self.get_operation(), 1)
    }
    /// Returns the softmax dimension, or `None` when absent.
    pub fn dimension(&self, ctx: &Context) -> Option<i64> {
        self.get_attr_linalg_dimension(ctx).map(|a| a.value)
    }
}

impl Verify for SoftmaxOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        let shapes = ranked_shapes(ctx, op, 1)?;
        if !compatible_shapes(&shapes[0].0, &shapes[1].0) {
            return Err(fail(
                ctx,
                op,
                "output shape must be compatible with the input shape",
            ));
        }
        let dim = self
            .dimension(ctx)
            .ok_or_else(|| fail(ctx, op, "missing dimension"))?;
        if dim < 0 || dim >= shapes[0].0.len() as i64 {
            return Err(fail(ctx, op, "dimension must be in range"));
        }
        Ok(())
    }
}

macro_rules! winograd_op {
    ($struct:ident, $name:literal, $doc:literal, $in_rank:literal, $out_rank:literal, $expected:ident) => {
        #[doc = $doc]
        #[pliron_op(
                            name = $name,
                            format = "operands(CharSpace(`,`)) attr_dict `:` typesig",
                            interfaces = [pliron::builtin::op_interfaces::OneResultInterface]
                        )]
        pub struct $struct;
        crate::pliron_arith::shared_attr!(
            $struct,
            super::ATTR_KEY_LINALG_FMR,
            WinogradFmrAttr,
            get_attr_linalg_fmr,
            set_attr_linalg_fmr
        );

        impl $struct {
            /// Allocates the operation; the result type is `output`'s type.
            pub fn new(ctx: &mut Context, input: Value, output: Value, fmr: WinogradFmr) -> Self {
                let op = allocate(ctx, |op| Self { op }, vec![input, output], 1);
                op.set_attr_linalg_fmr(ctx, WinogradFmrAttr::new(fmr));
                op
            }
            /// Returns the transformed operand.
            pub fn input(&self, ctx: &Context) -> Value {
                operand(ctx, self.get_operation(), 0)
            }
            /// Returns the `output` operand.
            pub fn output(&self, ctx: &Context) -> Value {
                operand(ctx, self.get_operation(), 1)
            }
            /// Returns the `F(m, r)` configuration, or `None` when absent.
            pub fn fmr(&self, ctx: &Context) -> Option<WinogradFmr> {
                self.get_attr_linalg_fmr(ctx).map(|a| a.value)
            }
        }

        impl Verify for $struct {
            fn verify(&self, ctx: &Context) -> Result<()> {
                let op = self.get_operation();
                let shapes = ranked_shapes(ctx, op, 1)?;
                let (input, output) = (&shapes[0].0, &shapes[1].0);
                if input.len() != $in_rank || output.len() != $out_rank {
                    return Err(fail(ctx, op, "unexpected operand ranks"));
                }
                let fmr = self.fmr(ctx).ok_or_else(|| fail(ctx, op, "missing fmr"))?;
                let expected = $expected(input, output, fmr).map_err(|r| fail(ctx, op, r))?;
                if !compatible_shapes(&expected, output) {
                    return Err(fail(
                        ctx,
                        op,
                        "output shape does not match the transformed shape",
                    ));
                }
                Ok(())
            }
        }
    };
}

type Expected = std::result::Result<Vec<i64>, &'static str>;

fn filter_shape(filter: &[i64], _output: &[i64], fmr: WinogradFmr) -> Expected {
    let (m, r) = fmr.m_r();
    let alpha = m + r - 1;
    let (h, w) = (filter[1], filter[2]);
    let valid = |d: i64| d < 0 || d == r || d == 1;
    if !valid(h) || !valid(w) || (h == 1 && w == 1) {
        return Err("filter height and width must each be r or 1, and not both 1");
    }
    let side = |d: i64| {
        if d < 0 {
            -1
        } else if d == r {
            alpha
        } else {
            1
        }
    };
    Ok(vec![side(h), side(w), filter[3], filter[0]])
}

fn input_shape(input: &[i64], output: &[i64], fmr: WinogradFmr) -> Expected {
    let (m, r) = fmr.m_r();
    let alpha = m + r - 1;
    let side = |extent: i64, transform: bool| -> (i64, i64) {
        let alpha_dim = if transform { alpha } else { 1 };
        let tiles = if extent < 0 {
            -1
        } else if transform {
            (extent - (r - 1)) / m
        } else {
            extent
        };
        (alpha_dim, tiles)
    };
    let (ah, th) = side(input[1], output[0] != 1);
    let (aw, tw) = side(input[2], output[1] != 1);
    Ok(vec![ah, aw, th, tw, input[0], input[3]])
}

fn output_shape(value: &[i64], _output: &[i64], fmr: WinogradFmr) -> Expected {
    let (m, r) = fmr.m_r();
    let alpha = m + r - 1;
    let side = |alpha_dim: i64, tiles: i64| -> std::result::Result<i64, &'static str> {
        let transform = alpha_dim != 1;
        if alpha_dim >= 0 && transform && alpha_dim != alpha {
            return Err("value alpha dimensions must be alpha or 1");
        }
        Ok(if tiles < 0 {
            -1
        } else if transform {
            m * tiles
        } else {
            tiles
        })
    };
    let h = side(value[0], value[2])?;
    let w = side(value[1], value[3])?;
    Ok(vec![value[4], h, w, value[5]])
}

winograd_op!(
    WinogradFilterTransformOp,
    "linalg.winograd_filter_transform",
    "Winograd filter transform of an `(F, H, W, C)` filter into `(alphaH, alphaW, C, F)`.",
    4,
    4,
    filter_shape
);
winograd_op!(
    WinogradInputTransformOp,
    "linalg.winograd_input_transform",
    "Winograd input transform of an `(N, H, W, C)` input into `(alphaH, alphaW, tileH, tileW, N, C)`.",
    4,
    6,
    input_shape
);
winograd_op!(
    WinogradOutputTransformOp,
    "linalg.winograd_output_transform",
    "Winograd output transform of `(alphaH, alphaW, tileH, tileW, N, F)` values into `(N, H, W, F)`.",
    6,
    4,
    output_shape
);

/// Inner tile size of `linalg.pack`/`linalg.unpack`.
#[derive(Clone, Copy, Debug)]
pub enum TileSize {
    /// Compile-time tile size.
    Static(i64),
    /// Tile size given by an `index` operand.
    Dynamic(Value),
}

/// Sentinel stored in `static_inner_tiles` for a dynamic tile, as in MLIR.
pub const DYNAMIC_TILE: i64 = i64::MIN;

fn split_tiles(tiles: Vec<TileSize>) -> (Vec<i64>, Vec<Value>) {
    let mut statics = Vec::with_capacity(tiles.len());
    let mut dynamics = Vec::new();
    for tile in tiles {
        match tile {
            TileSize::Static(t) => statics.push(t),
            TileSize::Dynamic(v) => {
                statics.push(DYNAMIC_TILE);
                dynamics.push(v);
            }
        }
    }
    (statics, dynamics)
}

fn mixed_tiles(statics: &[i64], dynamics: &[Value]) -> Vec<TileSize> {
    let mut dynamics = dynamics.iter();
    statics
        .iter()
        .map(|&t| match (t, t == DYNAMIC_TILE) {
            (_, true) => dynamics
                .next()
                .map_or(TileSize::Static(t), |&v| TileSize::Dynamic(v)),
            (t, false) => TileSize::Static(t),
        })
        .collect()
}

/// Shared pack/unpack checks; returns an error reason on failure.
fn verify_relayout(
    unpacked: &[i64],
    packed: &[i64],
    inner_dims_pos: &[i64],
    outer_dims_perm: &[i64],
    tiles: &[i64],
) -> std::result::Result<(), &'static str> {
    let rank = unpacked.len();
    if tiles.iter().any(|&t| t != DYNAMIC_TILE && t <= 0) {
        return Err("inner tile sizes must be positive");
    }
    if tiles.len() != inner_dims_pos.len() || tiles.len() > rank {
        return Err(
            "expected one inner tile per inner dimension position, at most the unpacked rank",
        );
    }
    let mut seen = vec![false; rank];
    for &p in inner_dims_pos {
        match usize::try_from(p) {
            Ok(p) if p < rank && !seen[p] => seen[p] = true,
            _ => return Err("inner_dims_pos must be unique and in range"),
        }
    }
    if !outer_dims_perm.is_empty() && !is_permutation(outer_dims_perm, rank) {
        return Err("outer_dims_perm must be empty or a permutation of the unpacked rank");
    }
    if packed.len() != rank + tiles.len() {
        return Err("packed rank must equal the unpacked rank plus the number of tiles");
    }
    let trailing = &packed[rank..];
    if trailing
        .iter()
        .zip(tiles)
        .any(|(&d, &t)| if t == DYNAMIC_TILE { d >= 0 } else { d != t })
    {
        return Err("trailing packed dimensions must equal the inner tile sizes");
    }
    let mut outer = unpacked.to_vec();
    for (&p, &t) in inner_dims_pos.iter().zip(tiles) {
        let d = &mut outer[p as usize];
        *d = if *d < 0 || t == DYNAMIC_TILE {
            -1
        } else {
            (*d + t - 1) / t
        };
    }
    if !outer_dims_perm.is_empty() {
        outer = outer_dims_perm.iter().map(|&p| outer[p as usize]).collect();
    }
    if !compatible_shapes(&outer, &packed[..rank]) {
        return Err("packed outer dimensions do not match the tiled unpacked shape");
    }
    Ok(())
}

macro_rules! relayout_accessors {
    () => {
        /// Returns the source operand.
        pub fn source(&self, ctx: &Context) -> Value {
            operand(ctx, self.get_operation(), 0)
        }
        /// Returns the destination operand.
        pub fn dest(&self, ctx: &Context) -> Value {
            operand(ctx, self.get_operation(), 1)
        }
        /// Returns the tiled dimensions of the unpacked tensor.
        pub fn inner_dims_pos(&self, ctx: &Context) -> Vec<i64> {
            self.get_attr_linalg_inner_dims_pos(ctx)
                .map(|a| a.values.clone())
                .unwrap_or_default()
        }
        /// Returns the outer-dimension permutation, empty for the identity.
        pub fn outer_dims_perm(&self, ctx: &Context) -> Vec<i64> {
            self.get_attr_linalg_outer_dims_perm(ctx)
                .map(|a| a.values.clone())
                .unwrap_or_default()
        }
        /// Returns the static tile sizes, [`DYNAMIC_TILE`] for dynamic tiles.
        pub fn static_inner_tiles(&self, ctx: &Context) -> Vec<i64> {
            self.get_attr_linalg_static_inner_tiles(ctx)
                .map(|a| a.values.clone())
                .unwrap_or_default()
        }
        /// Returns the dynamic tile operands.
        pub fn dynamic_inner_tiles(&self, ctx: &Context) -> Vec<Value> {
            let sizes = self.get_attr_linalg_operand_segment_sizes(ctx);
            let n = sizes
                .as_deref()
                .and_then(|s| s.0.last().copied())
                .unwrap_or(0) as usize;
            let ir = self.get_operation().deref(ctx);
            let total = ir.get_num_operands();
            ir.operands().skip(total.saturating_sub(n)).collect()
        }
        /// Returns the tile sizes, static or dynamic.
        pub fn inner_tiles(&self, ctx: &Context) -> Vec<TileSize> {
            mixed_tiles(
                &self.static_inner_tiles(ctx),
                &self.dynamic_inner_tiles(ctx),
            )
        }

        fn verify_common(&self, ctx: &Context, pack: bool) -> Result<()> {
            let op = self.get_operation();
            let sizes = self
                .get_attr_linalg_operand_segment_sizes(ctx)
                .ok_or_else(|| fail(ctx, op, "missing operand segment sizes"))?
                .0
                .clone();
            let operands = operand_types(ctx, op);
            let expected_segments = if pack { 4 } else { 3 };
            if sizes.len() != expected_segments
                || sizes[..2] != [1, 1]
                || sizes.iter().map(|&s| s as usize).sum::<usize>() != operands.len()
                || (pack && sizes[2] > 1)
            {
                return Err(fail(
                    ctx,
                    op,
                    "operand segment sizes do not match the operands",
                ));
            }
            {
                let ir = op.deref(ctx);
                if ir.get_num_results() != 1 || ir.num_regions() != 0 {
                    return Err(fail(ctx, op, "expected one result and no regions"));
                }
            }
            verify_tensor_outputs(ctx, op, 1..2)?;
            let (source, source_element) = ranked(ctx, operands[0])
                .ok_or_else(|| fail(ctx, op, "source must be a ranked tensor"))?;
            let (dest, _) = ranked(ctx, operands[1])
                .ok_or_else(|| fail(ctx, op, "dest must be a ranked tensor"))?;
            let dynamic = *sizes.last().unwrap_or(&0) as usize;
            let statics = self.static_inner_tiles(ctx);
            if statics.iter().filter(|&&t| t == DYNAMIC_TILE).count() != dynamic {
                return Err(fail(
                    ctx,
                    op,
                    "dynamic tile count must match the dynamic tile operands",
                ));
            }
            if operands[operands.len() - dynamic..]
                .iter()
                .any(|&ty| !shaped::is_index(ctx, ty))
            {
                return Err(fail(ctx, op, "dynamic tile sizes must be index values"));
            }
            let (unpacked, packed) = if pack {
                (&source, &dest)
            } else {
                (&dest, &source)
            };
            let pos = self.inner_dims_pos(ctx);
            verify_relayout(unpacked, packed, &pos, &self.outer_dims_perm(ctx), &statics)
                .map_err(|r| fail(ctx, op, r))?;
            if pack {
                if sizes[2] == 1 {
                    let padding = operands[2];
                    if padding != source_element {
                        return Err(fail(
                            ctx,
                            op,
                            "padding value must have the source element type",
                        ));
                    }
                } else if pos.iter().zip(&statics).any(|(&p, &t)| {
                    let d = source[p as usize];
                    t != DYNAMIC_TILE && d >= 0 && d % t != 0
                }) {
                    return Err(fail(
                        ctx,
                        op,
                        "only full tiles are supported when padding_value is not set",
                    ));
                }
            }
            Ok(())
        }
    };
}

/// Packs `source` into a tiled `dest` layout.
///
/// `inner_dims_pos[i]` of the source is tiled by `inner_tiles[i]`; tiles
/// become the trailing `dest` dimensions, and the outer dimensions are
/// optionally permuted by `outer_dims_perm`. A `padding_value` fills partial
/// tiles.
#[pliron_op(
    name = "linalg.pack",
    format = "operands(CharSpace(`,`)) attr_dict `:` typesig",
    interfaces = [pliron::builtin::op_interfaces::OneResultInterface]
)]
pub struct PackOp;
crate::pliron_arith::shared_attr!(
    PackOp,
    super::ATTR_KEY_LINALG_INNER_DIMS_POS,
    I64ArrayAttr,
    get_attr_linalg_inner_dims_pos,
    set_attr_linalg_inner_dims_pos
);
crate::pliron_arith::shared_attr!(
    PackOp,
    super::ATTR_KEY_LINALG_OUTER_DIMS_PERM,
    I64ArrayAttr,
    get_attr_linalg_outer_dims_perm,
    set_attr_linalg_outer_dims_perm
);
crate::pliron_arith::shared_attr!(
    PackOp,
    super::ATTR_KEY_LINALG_STATIC_INNER_TILES,
    I64ArrayAttr,
    get_attr_linalg_static_inner_tiles,
    set_attr_linalg_static_inner_tiles
);
crate::pliron_arith::shared_attr!(
    PackOp,
    super::ATTR_KEY_LINALG_OPERAND_SEGMENT_SIZES,
    OperandSegmentSizesAttr,
    get_attr_linalg_operand_segment_sizes,
    set_attr_linalg_operand_segment_sizes
);

impl PackOp {
    /// Allocates the operation; the result type is `dest`'s type.
    pub fn new(
        ctx: &mut Context,
        source: Value,
        dest: Value,
        inner_dims_pos: Vec<i64>,
        inner_tiles: Vec<TileSize>,
        outer_dims_perm: Vec<i64>,
        padding_value: Option<Value>,
    ) -> Self {
        let (statics, dynamics) = split_tiles(inner_tiles);
        let sizes = OperandSegmentSizesAttr(vec![
            1,
            1,
            u32::from(padding_value.is_some()),
            dynamics.len() as u32,
        ]);
        let operands = [source, dest]
            .into_iter()
            .chain(padding_value)
            .chain(dynamics)
            .collect();
        let op = allocate(ctx, |op| Self { op }, operands, 1);
        op.set_attr_linalg_inner_dims_pos(ctx, I64ArrayAttr::new(inner_dims_pos));
        op.set_attr_linalg_outer_dims_perm(ctx, I64ArrayAttr::new(outer_dims_perm));
        op.set_attr_linalg_static_inner_tiles(ctx, I64ArrayAttr::new(statics));
        op.set_attr_linalg_operand_segment_sizes(ctx, sizes);
        op
    }

    relayout_accessors!();

    /// Returns the padding value, if any.
    pub fn padding_value(&self, ctx: &Context) -> Option<Value> {
        let sizes = self.get_attr_linalg_operand_segment_sizes(ctx)?;
        (sizes.0.get(2) == Some(&1)).then(|| operand(ctx, self.get_operation(), 2))
    }
}

impl Verify for PackOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        self.verify_common(ctx, true)
    }
}

/// Unpacks a tiled `source` back into `dest`; the inverse of [`PackOp`].
#[pliron_op(
    name = "linalg.unpack",
    format = "operands(CharSpace(`,`)) attr_dict `:` typesig",
    interfaces = [pliron::builtin::op_interfaces::OneResultInterface]
)]
pub struct UnPackOp;
crate::pliron_arith::shared_attr!(
    UnPackOp,
    super::ATTR_KEY_LINALG_INNER_DIMS_POS,
    I64ArrayAttr,
    get_attr_linalg_inner_dims_pos,
    set_attr_linalg_inner_dims_pos
);
crate::pliron_arith::shared_attr!(
    UnPackOp,
    super::ATTR_KEY_LINALG_OUTER_DIMS_PERM,
    I64ArrayAttr,
    get_attr_linalg_outer_dims_perm,
    set_attr_linalg_outer_dims_perm
);
crate::pliron_arith::shared_attr!(
    UnPackOp,
    super::ATTR_KEY_LINALG_STATIC_INNER_TILES,
    I64ArrayAttr,
    get_attr_linalg_static_inner_tiles,
    set_attr_linalg_static_inner_tiles
);
crate::pliron_arith::shared_attr!(
    UnPackOp,
    super::ATTR_KEY_LINALG_OPERAND_SEGMENT_SIZES,
    OperandSegmentSizesAttr,
    get_attr_linalg_operand_segment_sizes,
    set_attr_linalg_operand_segment_sizes
);

impl UnPackOp {
    /// Allocates the operation; the result type is `dest`'s type.
    pub fn new(
        ctx: &mut Context,
        source: Value,
        dest: Value,
        inner_dims_pos: Vec<i64>,
        inner_tiles: Vec<TileSize>,
        outer_dims_perm: Vec<i64>,
    ) -> Self {
        let (statics, dynamics) = split_tiles(inner_tiles);
        let sizes = OperandSegmentSizesAttr(vec![1, 1, dynamics.len() as u32]);
        let operands = [source, dest].into_iter().chain(dynamics).collect();
        let op = allocate(ctx, |op| Self { op }, operands, 1);
        op.set_attr_linalg_inner_dims_pos(ctx, I64ArrayAttr::new(inner_dims_pos));
        op.set_attr_linalg_outer_dims_perm(ctx, I64ArrayAttr::new(outer_dims_perm));
        op.set_attr_linalg_static_inner_tiles(ctx, I64ArrayAttr::new(statics));
        op.set_attr_linalg_operand_segment_sizes(ctx, sizes);
        op
    }

    relayout_accessors!();
}

impl Verify for UnPackOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        self.verify_common(ctx, false)
    }
}
