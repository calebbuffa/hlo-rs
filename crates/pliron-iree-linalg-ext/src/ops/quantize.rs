//! Affine (uniform) quantization and dequantization.

use crate::support::*;
use pliron::{
    builtin::attributes::UnitAttr,
    common_traits::Verify,
    context::{Context, Ptr},
    derive::pliron_op,
    op::Op,
    operation::Operation,
    result::Result,
    r#type::{TypeHandle, Typed},
    value::Value,
};
use pliron_arith::{ops::ConstantOp, shaped};
use pliron_linalg::attributes::IndexingMapsAttr;
use pliron_mlir_builtin::attributes::{AffineMap, DenseElementsAttr, IndexAttr, ScalarBitsAttr};

/// Inclusive `[min, max]` range of quantized values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantizeBounds {
    /// Smallest quantized value.
    pub min: i64,
    /// Largest quantized value.
    pub max: i64,
}

/// Checks a `scale` or `zero_point` operand against its indexing map.
///
/// `iteration_to_input[d]` is the input dimension that bounds iteration
/// dimension `d`.
fn verify_parameter(
    ctx: &Context,
    op: Ptr<Operation>,
    ty: TypeHandle,
    map: &AffineMap,
    input: &[i64],
    iteration_to_input: &[usize],
) -> Result<()> {
    let Some((dims, _)) = ranked(ctx, ty) else {
        ensure(
            ctx,
            op,
            is_plain_scalar(ctx, ty),
            "quantization parameter must be a ranked tensor or scalar",
        )?;
        return ensure(
            ctx,
            op,
            map.results.is_empty(),
            "expected scalar quantization parameter to have an indexing map with no results",
        );
    };
    ensure(
        ctx,
        op,
        dims.len() == map.results.len(),
        "quantization parameter rank must match the number of results of its indexing map",
    )?;
    let sizes_match = map.results.iter().zip(&dims).all(|(expr, &size)| {
        as_dim(expr).is_some_and(|d| {
            let value = input[iteration_to_input[d]];
            value < 0 || size < 0 || value == size
        })
    });
    ensure(
        ctx,
        op,
        sizes_match,
        "quantization parameter dimension sizes must match the quantized value",
    )
}

/// Verifies the structure shared by quantize and dequantize and returns the
/// input and output element types.
fn verify_affine_quantization(
    ctx: &Context,
    op: Ptr<Operation>,
    maps: Option<&IndexingMapsAttr>,
    zp_unsigned: bool,
) -> Result<(TypeHandle, TypeHandle)> {
    no_regions(ctx, op)?;
    let count = num_operands(ctx, op);
    ensure(
        ctx,
        op,
        (3..=4).contains(&count),
        "expected input, scale, optional zero_point, and output",
    )?;
    verify_outputs(ctx, op, count - 1..count)?;
    let has_zero_point = count == 4;
    let types = operand_types(ctx, op);
    let (input, input_element) = ranked_operand(ctx, op, types[0])?;
    let (output, output_element) = ranked_operand(ctx, op, types[count - 1])?;
    let rank = input.len();
    let maps = maps.map(|m| m.maps.as_slice()).unwrap_or_default();
    ensure(
        ctx,
        op,
        maps.len() == if has_zero_point { 4 } else { 3 },
        "expected one indexing map per operand",
    )?;
    for map in maps {
        ensure(
            ctx,
            op,
            map.dimensions == rank,
            "indexing maps must have as many dimensions as the quantized value's rank",
        )?;
        ensure(
            ctx,
            op,
            map.symbols == 0,
            "indexing maps must have no symbols",
        )?;
        ensure(
            ctx,
            op,
            is_projected_permutation(map),
            "indexing maps must be projected permutations",
        )?;
    }
    let (input_map, output_map) = (&maps[0], &maps[maps.len() - 1]);
    ensure(
        ctx,
        op,
        input_map.results.len() == rank && output_map.results.len() == rank,
        "input and output indexing maps must be permutations",
    )?;
    let mut iteration_to_input = vec![0; rank];
    for (position, expr) in input_map.results.iter().enumerate() {
        if let Some(d) = as_dim(expr) {
            iteration_to_input[d] = position;
        }
    }
    let expected: Vec<i64> = output_map
        .results
        .iter()
        .filter_map(as_dim)
        .map(|d| input[iteration_to_input[d]])
        .collect();
    ensure(
        ctx,
        op,
        output == expected,
        "expected input and output to have the same shape up to their indexing maps",
    )?;
    let scale = element(ctx, op.deref(ctx).get_operand(1));
    ensure(
        ctx,
        op,
        shaped::is_float(ctx, scale),
        "expected scale to have a floating point element type",
    )?;
    let scale_width = shaped::float_width(ctx, scale);
    let mixes = [input_element, output_element].into_iter().any(|t| {
        shaped::is_float(ctx, t) && t != scale && shaped::float_width(ctx, t) == scale_width
    });
    ensure(
        ctx,
        op,
        !mixes,
        "cannot mix equal-width floating point types",
    )?;
    verify_parameter(ctx, op, types[1], &maps[1], &input, &iteration_to_input)?;
    if !has_zero_point {
        ensure(
            ctx,
            op,
            !zp_unsigned,
            "zp_unsigned is only allowed when a zero_point operand is present",
        )?;
        return Ok((input_element, output_element));
    }
    verify_parameter(ctx, op, types[2], &maps[2], &input, &iteration_to_input)?;
    let zero_point = element(ctx, op.deref(ctx).get_operand(2));
    ensure(
        ctx,
        op,
        shaped::signless_width(ctx, zero_point).is_some(),
        "expected zero_point to have a signless integer element type",
    )?;
    Ok((input_element, output_element))
}

/// Returns the representable range of a `width`-bit integer, or `None` at 64 bits or more.
fn storage_range(width: u32, unsigned: bool) -> Option<(i128, i128)> {
    (width < 64).then(|| {
        if unsigned {
            (0, (1_i128 << width) - 1)
        } else {
            (-(1_i128 << (width - 1)), (1_i128 << (width - 1)) - 1)
        }
    })
}

/// Checks that `[min, max]` is ordered and representable in `storage`.
fn verify_range(
    ctx: &Context,
    op: Ptr<Operation>,
    storage: TypeHandle,
    bounds: QuantizeBounds,
    unsigned: bool,
) -> Result<()> {
    ensure(
        ctx,
        op,
        bounds.min <= bounds.max,
        "expected quant_min to not exceed quant_max",
    )?;
    let Some((low, high)) = int_width(ctx, storage).and_then(|w| storage_range(w, unsigned)) else {
        return Ok(());
    };
    ensure(
        ctx,
        op,
        i128::from(bounds.min) >= low && i128::from(bounds.max) <= high,
        "quantization range is not representable in the storage type",
    )
}

/// Returns the integer values of an `arith.constant` defining `value`.
fn constant_integers(ctx: &Context, value: Value) -> Option<Vec<i128>> {
    let constant = Operation::get_op::<ConstantOp>(value.defining_op()?, ctx)?;
    let attr = constant.value(ctx)?;
    let decode = |bits: &ScalarBitsAttr| bits.integer_value(ctx)?.to_i128();
    if let Some(bits) = attr.downcast_ref::<ScalarBitsAttr>() {
        return Some(vec![decode(bits)?]);
    }
    attr.downcast_ref::<DenseElementsAttr>()?
        .values
        .0
        .iter()
        .map(|v| v.downcast_ref::<ScalarBitsAttr>().and_then(decode))
        .collect()
}

fn bounds_attr(value: i64) -> IndexAttr {
    IndexAttr { value }
}

/// Quantizes a floating-point tensor into integer storage.
///
/// `output[out_map(d)] = clamp(round(input[in_map(d)] / scale[scale_map(d)])
/// + zero_point[zp_map(d)], quant_min, quant_max)`. Operands are `input`,
/// `scale`, optional `zero_point`, then `output` (outs); `indexing_maps` has
/// one map per operand. `storage_unsigned` and `zp_unsigned` select unsigned
/// interpretation of the signless output and zero point.
#[pliron_op(
    name = "iree_linalg_ext.quantize_affine",
    format = "operands(CharSpace(`,`)) attr_dict `:` typesig",
    attributes = (
        iree_quantize_affine_indexing_maps: IndexingMapsAttr,
        iree_quantize_affine_quant_min: IndexAttr,
        iree_quantize_affine_quant_max: IndexAttr,
        iree_quantize_affine_storage_unsigned: UnitAttr,
        iree_quantize_affine_zp_unsigned: UnitAttr
    )
)]
pub struct QuantizeAffineOp;

impl QuantizeAffineOp {
    /// Allocates the operation without verifying it.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: &mut Context,
        input: Value,
        scale: Value,
        zero_point: Option<Value>,
        output: Value,
        indexing_maps: Vec<AffineMap>,
        bounds: QuantizeBounds,
        storage_unsigned: bool,
        zp_unsigned: bool,
    ) -> Self {
        let results = vec![output.get_type(ctx)];
        let operands = [input, scale]
            .into_iter()
            .chain(zero_point)
            .chain([output])
            .collect();
        let op = new_plain(ctx, |op| Self { op }, results, operands);
        op.set_attr_iree_quantize_affine_indexing_maps(ctx, IndexingMapsAttr::new(indexing_maps));
        op.set_attr_iree_quantize_affine_quant_min(ctx, bounds_attr(bounds.min));
        op.set_attr_iree_quantize_affine_quant_max(ctx, bounds_attr(bounds.max));
        if storage_unsigned {
            op.set_attr_iree_quantize_affine_storage_unsigned(ctx, UnitAttr::new());
        }
        if zp_unsigned {
            op.set_attr_iree_quantize_affine_zp_unsigned(ctx, UnitAttr::new());
        }
        op
    }

    /// Builds and verifies the operation.
    ///
    /// # Errors
    ///
    /// Returns an error when verification fails.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        ctx: &mut Context,
        input: Value,
        scale: Value,
        zero_point: Option<Value>,
        output: Value,
        indexing_maps: Vec<AffineMap>,
        bounds: QuantizeBounds,
        storage_unsigned: bool,
        zp_unsigned: bool,
    ) -> Result<Self> {
        let op = Self::new(
            ctx,
            input,
            scale,
            zero_point,
            output,
            indexing_maps,
            bounds,
            storage_unsigned,
            zp_unsigned,
        );
        op.verify(ctx)?;
        Ok(op)
    }

    /// Returns whether a zero point operand is present.
    pub fn has_zero_point(&self, ctx: &Context) -> bool {
        num_operands(ctx, self.get_operation()) == 4
    }

    /// Returns the indexing maps, or an empty list when absent.
    pub fn indexing_maps(&self, ctx: &Context) -> Vec<AffineMap> {
        self.get_attr_iree_quantize_affine_indexing_maps(ctx)
            .map(|a| a.maps.clone())
            .unwrap_or_default()
    }

    /// Returns the quantization range, if both bounds are set.
    pub fn bounds(&self, ctx: &Context) -> Option<QuantizeBounds> {
        let min = self.get_attr_iree_quantize_affine_quant_min(ctx)?.value;
        let max = self.get_attr_iree_quantize_affine_quant_max(ctx)?.value;
        Some(QuantizeBounds { min, max })
    }

    /// Returns whether the output storage is interpreted as unsigned.
    pub fn storage_unsigned(&self, ctx: &Context) -> bool {
        self.get_attr_iree_quantize_affine_storage_unsigned(ctx)
            .is_some()
    }

    /// Returns whether the zero point is interpreted as unsigned.
    pub fn zp_unsigned(&self, ctx: &Context) -> bool {
        self.get_attr_iree_quantize_affine_zp_unsigned(ctx)
            .is_some()
    }
}

impl Verify for QuantizeAffineOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        let maps = self.get_attr_iree_quantize_affine_indexing_maps(ctx);
        let (real, storage) =
            verify_affine_quantization(ctx, op, maps.as_deref(), self.zp_unsigned(ctx))?;
        ensure(
            ctx,
            op,
            shaped::is_float(ctx, real),
            "expected input to have a floating point element type",
        )?;
        ensure(
            ctx,
            op,
            shaped::signless_width(ctx, storage).is_some(),
            "expected output to have a signless integer element type",
        )?;
        let bounds = self
            .bounds(ctx)
            .ok_or_else(|| fail(ctx, op, "quant_min and quant_max are required"))?;
        verify_range(ctx, op, storage, bounds, self.storage_unsigned(ctx))
    }
}

/// Dequantizes integer storage into a floating-point tensor.
///
/// `output[out_map(d)] = (extend(input[in_map(d)]) - extend(zero_point[zp_map(d)]))
/// * scale[scale_map(d)]`. Operands are `input`, `scale`, optional
/// `zero_point`, then `output` (outs). `quant_min`/`quant_max` optionally
/// record the input's value range without clamping.
#[pliron_op(
    name = "iree_linalg_ext.dequantize_affine",
    format = "operands(CharSpace(`,`)) attr_dict `:` typesig",
    attributes = (
        iree_dequantize_affine_indexing_maps: IndexingMapsAttr,
        iree_dequantize_affine_quant_min: IndexAttr,
        iree_dequantize_affine_quant_max: IndexAttr,
        iree_dequantize_affine_input_unsigned: UnitAttr,
        iree_dequantize_affine_zp_unsigned: UnitAttr
    )
)]
pub struct DequantizeAffineOp;

impl DequantizeAffineOp {
    /// Allocates the operation without verifying it.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: &mut Context,
        input: Value,
        scale: Value,
        zero_point: Option<Value>,
        output: Value,
        indexing_maps: Vec<AffineMap>,
        bounds: Option<QuantizeBounds>,
        input_unsigned: bool,
        zp_unsigned: bool,
    ) -> Self {
        let results = vec![output.get_type(ctx)];
        let operands = [input, scale]
            .into_iter()
            .chain(zero_point)
            .chain([output])
            .collect();
        let op = new_plain(ctx, |op| Self { op }, results, operands);
        op.set_attr_iree_dequantize_affine_indexing_maps(ctx, IndexingMapsAttr::new(indexing_maps));
        if let Some(bounds) = bounds {
            op.set_attr_iree_dequantize_affine_quant_min(ctx, bounds_attr(bounds.min));
            op.set_attr_iree_dequantize_affine_quant_max(ctx, bounds_attr(bounds.max));
        }
        if input_unsigned {
            op.set_attr_iree_dequantize_affine_input_unsigned(ctx, UnitAttr::new());
        }
        if zp_unsigned {
            op.set_attr_iree_dequantize_affine_zp_unsigned(ctx, UnitAttr::new());
        }
        op
    }

    /// Builds and verifies the operation.
    ///
    /// # Errors
    ///
    /// Returns an error when verification fails.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        ctx: &mut Context,
        input: Value,
        scale: Value,
        zero_point: Option<Value>,
        output: Value,
        indexing_maps: Vec<AffineMap>,
        bounds: Option<QuantizeBounds>,
        input_unsigned: bool,
        zp_unsigned: bool,
    ) -> Result<Self> {
        let op = Self::new(
            ctx,
            input,
            scale,
            zero_point,
            output,
            indexing_maps,
            bounds,
            input_unsigned,
            zp_unsigned,
        );
        op.verify(ctx)?;
        Ok(op)
    }

    /// Returns whether a zero point operand is present.
    pub fn has_zero_point(&self, ctx: &Context) -> bool {
        num_operands(ctx, self.get_operation()) == 4
    }

    /// Returns the indexing maps, or an empty list when absent.
    pub fn indexing_maps(&self, ctx: &Context) -> Vec<AffineMap> {
        self.get_attr_iree_dequantize_affine_indexing_maps(ctx)
            .map(|a| a.maps.clone())
            .unwrap_or_default()
    }

    /// Returns the recorded input range, if both bounds are set.
    pub fn bounds(&self, ctx: &Context) -> Option<QuantizeBounds> {
        let min = self.get_attr_iree_dequantize_affine_quant_min(ctx)?.value;
        let max = self.get_attr_iree_dequantize_affine_quant_max(ctx)?.value;
        Some(QuantizeBounds { min, max })
    }

    /// Returns whether the input storage is interpreted as unsigned.
    pub fn input_unsigned(&self, ctx: &Context) -> bool {
        self.get_attr_iree_dequantize_affine_input_unsigned(ctx)
            .is_some()
    }

    /// Returns whether the zero point is interpreted as unsigned.
    pub fn zp_unsigned(&self, ctx: &Context) -> bool {
        self.get_attr_iree_dequantize_affine_zp_unsigned(ctx)
            .is_some()
    }

    /// Returns the width in which `input - zero_point` is computed:
    /// `max(PowerOf2Ceil(storage_width + 1), 8)`, or `None` above 64 bits.
    pub fn zero_point_difference_width(&self, ctx: &Context) -> Option<u32> {
        let input = self.get_operation().deref(ctx).get_operand(0);
        let width = (int_width(ctx, element(ctx, input))? + 1)
            .next_power_of_two()
            .max(8);
        (width <= 64).then_some(width)
    }
}

impl Verify for DequantizeAffineOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        let maps = self.get_attr_iree_dequantize_affine_indexing_maps(ctx);
        let (storage, real) =
            verify_affine_quantization(ctx, op, maps.as_deref(), self.zp_unsigned(ctx))?;
        let width = shaped::signless_width(ctx, storage).ok_or_else(|| {
            fail(
                ctx,
                op,
                "expected input to have a signless integer element type",
            )
        })?;
        ensure(
            ctx,
            op,
            shaped::is_float(ctx, real),
            "expected output to have a floating point element type",
        )?;
        let unsigned = self.input_unsigned(ctx);
        if self.has_zero_point(ctx) {
            ensure(
                ctx,
                op,
                self.zero_point_difference_width(ctx).is_some(),
                "cannot subtract a zero point without exceeding 64 bits",
            )?;
            let zero_point = op.deref(ctx).get_operand(2);
            let zp_width = int_width(ctx, element(ctx, zero_point)).unwrap_or(64);
            if let (Some(values), Some((low, high))) = (
                constant_integers(ctx, zero_point),
                storage_range(width, unsigned),
            ) {
                let zp_unsigned = self.zp_unsigned(ctx);
                let in_range = values.into_iter().all(|v| {
                    let v = if zp_unsigned && v < 0 && zp_width < 127 {
                        v + (1_i128 << zp_width)
                    } else {
                        v
                    };
                    (low..=high).contains(&v)
                });
                ensure(
                    ctx,
                    op,
                    in_range,
                    "zero point is not representable in the storage type",
                )?;
            }
        }
        let min = self
            .get_attr_iree_dequantize_affine_quant_min(ctx)
            .is_some();
        let max = self
            .get_attr_iree_dequantize_affine_quant_max(ctx)
            .is_some();
        ensure(
            ctx,
            op,
            min == max,
            "expected quant_min and quant_max to be specified together",
        )?;
        match self.bounds(ctx) {
            Some(bounds) => verify_range(ctx, op, storage, bounds, unsigned),
            None => Ok(()),
        }
    }
}
