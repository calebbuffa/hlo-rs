use crate::verification::invalid;
use crate::{attributes::*, types::*};
use pliron::{
    attribute::AttrObj,
    builtin::attributes::{BoolAttr, IntegerAttr, StringAttr, VecAttr},
    builtin::types::{IntegerType, Signedness},
    common_traits::Verify,
    context::Context,
    derive::pliron_op,
    identifier::Identifier,
    op::Op,
    operation::Operation,
    result::Result,
    r#type::{TypeHandle, Typed},
    utils::apint::{APInt, bw},
    value::Value,
};

fn ints(ctx: &Context, v: impl IntoIterator<Item = u64>) -> VecAttr {
    crate::attributes::integer_list(ctx, v)
}
fn read_ints(a: Option<std::cell::Ref<'_, VecAttr>>) -> Option<Vec<u64>> {
    a.as_deref().and_then(crate::attributes::read_integer_list)
}
fn signed_list(ctx: &Context, values: impl IntoIterator<Item = i64>) -> VecAttr {
    VecAttr::new(
        values
            .into_iter()
            .map(|v| {
                Box::new(IntegerAttr::new(
                    IntegerType::get(ctx, 64, Signedness::Signed),
                    APInt::from_i64(v, bw(64)),
                )) as AttrObj
            })
            .collect(),
    )
}
fn read_signed_ints(a: Option<std::cell::Ref<'_, VecAttr>>) -> Option<Vec<i64>> {
    a.as_deref().and_then(|a| {
        a.0.iter()
            .map(|x| {
                x.downcast_ref::<IntegerAttr>()?
                    .value()
                    .to_string_decimal(true)
                    .parse()
                    .ok()
            })
            .collect()
    })
}
fn precision_values(ctx: &Context, values: impl IntoIterator<Item = Precision>) -> VecAttr {
    ints(ctx, values.into_iter().map(|precision| precision as u64))
}
fn read_precision_values(attr: &VecAttr) -> Option<Vec<Precision>> {
    crate::attributes::read_integer_list(attr)?
        .into_iter()
        .map(|value| match value {
            0 => Some(Precision::Default),
            1 => Some(Precision::High),
            2 => Some(Precision::Highest),
            _ => None,
        })
        .collect()
}
fn read_precision_attribute(attr: Option<std::cell::Ref<'_, VecAttr>>) -> Option<Vec<Precision>> {
    read_precision_values(attr.as_deref()?)
}
fn compatible_dim(lhs: i64, rhs: i64) -> bool {
    lhs < 0 || rhs < 0 || lhs == rhs
}
fn parts(ctx: &Context, v: Value) -> Option<(Vec<i64>, TypeHandle)> {
    let ty = v.get_type(ctx);
    let d = ty.deref(ctx);
    let t = d.downcast_ref::<TensorType>()?;
    Some((t.dimensions.clone(), t.element))
}
fn parts_ty(ctx: &Context, ty: TypeHandle) -> Option<(Vec<i64>, TypeHandle)> {
    let d = ty.deref(ctx);
    let t = d.downcast_ref::<TensorType>()?;
    Some((t.dimensions.clone(), t.element))
}
fn quantized_element<'a>(
    ctx: &'a Context,
    element: TypeHandle,
) -> Option<std::cell::Ref<'a, QuantizedElementType>> {
    std::cell::Ref::filter_map(element.deref(ctx), |element| {
        element.downcast_ref::<QuantizedElementType>()
    })
    .ok()
}
fn convolution_element_compatible(
    ctx: &Context,
    lhs: TypeHandle,
    rhs: TypeHandle,
    result: TypeHandle,
    kernel_output: i64,
    output_feature: i64,
) -> bool {
    let (left, right, output) = (
        quantized_element(ctx, lhs),
        quantized_element(ctx, rhs),
        quantized_element(ctx, result),
    );
    match (left, right, output) {
        (None, None, None) => {
            let a = crate::types::primitive_element(ctx, lhs);
            let b = crate::types::primitive_element(ctx, rhs);
            let c = crate::types::primitive_element(ctx, result);
            a == c
                && (a == b
                    || a.is_some_and(|e| e.is_float() && e.bit_width() == Some(8))
                        && b.is_some_and(|e| e.is_float() && e.bit_width() == Some(8)))
        }
        (Some(left), Some(right), Some(output)) => {
            left.storage == right.storage
                && left.expressed == right.expressed
                && left.expressed == output.expressed
                && right
                    .quantization_dimension()
                    .is_none_or(|axis| i64::try_from(axis) == Ok(kernel_output))
                && output
                    .quantization_dimension()
                    .is_none_or(|axis| i64::try_from(axis) == Ok(output_feature))
        }
        (None, Some(right), None) => {
            lhs == right.expressed
                && result == right.expressed
                && right
                    .quantization_dimension()
                    .is_none_or(|axis| i64::try_from(axis) == Ok(kernel_output))
        }
        _ => false,
    }
}

fn dot_element_compatible(
    ctx: &Context,
    lhs: TypeHandle,
    rhs: TypeHandle,
    result: TypeHandle,
    rhs_contract: &[u64],
) -> bool {
    let (left, right, output) = (
        quantized_element(ctx, lhs),
        quantized_element(ctx, rhs),
        quantized_element(ctx, result),
    );
    match (left, right, output) {
        (None, None, None) => lhs == rhs && rhs == result,
        (Some(left), Some(right), Some(output)) => {
            left.quantization_dimension().is_none()
                && left.storage == right.storage
                && left.storage == output.storage
                && left.expressed == right.expressed
                && left.expressed == output.expressed
                && right.zero_points().all(|point| point == 0)
                && right
                    .quantization_dimension()
                    .is_none_or(|axis| !rhs_contract.contains(&(axis as u64)))
                && (right.quantization_dimension().is_some()
                    || output.quantization_dimension().is_none())
        }
        (None, Some(right), None) => {
            lhs == right.expressed
                && result == right.expressed
                && right.zero_points().all(|point| point == 0)
                && right
                    .quantization_dimension()
                    .is_none_or(|axis| !rhs_contract.contains(&(axis as u64)))
        }
        _ => false,
    }
}

/// Contracts and batches tensor axes according to explicit dimension numbers.
#[pliron_op(name="stablehlo.dot_general",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface],attributes=(stablehlo_dot_dimensions:DotDimensionNumbersAttr,stablehlo_dot_precision:VecAttr,stablehlo_dot_algorithm:DotAlgorithmAttr))]
pub struct DotGeneralOp;
impl DotGeneralOp {
    /// Creates a `stablehlo.dot_general` operation with explicit dimension numbers.
    ///
    /// Precision defaults to the StableHLO default and can be set with
    /// [`with_precision`](Self::with_precision).
    pub fn new(
        ctx: &mut Context,
        lhs: Value,
        rhs: Value,
        result: TypeHandle,
        dimensions: DotDimensionNumbersAttr,
    ) -> Self {
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![lhs, rhs],
                vec![],
                0,
            ),
        };
        op.set_attr_stablehlo_dot_dimensions(ctx, dimensions);
        op.set_attr_stablehlo_dot_precision(ctx, precision_values(ctx, []));
        op
    }
    /// Creates a `stablehlo.dot_general` operation with an inferred result shape.
    ///
    /// The caller supplies the result element type because it may differ from
    /// either operand's element type.
    pub fn new_inferred(
        ctx: &mut Context,
        lhs: Value,
        rhs: Value,
        dimensions: DotDimensionNumbersAttr,
        result_element: TypeHandle,
    ) -> Result<Self> {
        let lhs_type = lhs.get_type(ctx);
        let rhs_type = rhs.get_type(ctx);
        let result = crate::inference::infer_dot_general_type(
            ctx,
            lhs_type,
            rhs_type,
            &dimensions,
            result_element,
        )?;
        Ok(Self::new(ctx, lhs, rhs, result, dimensions))
    }
    /// Borrow the batching and contracting dimension numbers, if present.
    pub fn dimension_numbers<'a>(
        &self,
        ctx: &'a Context,
    ) -> Option<std::cell::Ref<'a, DotDimensionNumbersAttr>> {
        self.get_attr_stablehlo_dot_dimensions(ctx)
    }
    /// Return the per-operand precision configuration values.
    pub fn precision(&self, ctx: &Context) -> Option<Vec<Precision>> {
        read_precision_attribute(self.get_attr_stablehlo_dot_precision(ctx))
    }
    /// Sets the per-operand precision configuration during construction.
    pub fn with_precision(self, ctx: &Context, precision: Vec<Precision>) -> Self {
        self.set_attr_stablehlo_dot_precision(ctx, precision_values(ctx, precision));
        self
    }
    /// Attach an optional dot algorithm to this operation.
    pub fn with_algorithm(self, ctx: &Context, algorithm: DotAlgorithmAttr) -> Self {
        self.set_attr_stablehlo_dot_algorithm(ctx, algorithm);
        self
    }
    /// Borrow the optional dot algorithm.
    pub fn algorithm<'a>(&self, ctx: &'a Context) -> Option<std::cell::Ref<'a, DotAlgorithmAttr>> {
        self.get_attr_stablehlo_dot_algorithm(ctx)
    }
}
impl Verify for DotGeneralOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() != 2 || ir.get_num_results() != 1 {
            return invalid(self, ctx, "dot_general arity mismatch");
        };
        let (Some((l, le)), Some((r, re)), Some((o, oe))) = (
            parts(ctx, ir.get_operand(0)),
            parts(ctx, ir.get_operand(1)),
            parts(ctx, ir.get_result(0)),
        ) else {
            return invalid(self, ctx, "dot_general tensors must be ranked");
        };
        let Some(dimensions) = self.get_attr_stablehlo_dot_dimensions(ctx) else {
            return invalid(self, ctx, "dot_general dimensions are required");
        };
        let algorithm = ir
            .attributes
            .0
            .get(&Identifier::new(crate::attributes::keys::DOT_ALGORITHM));
        let algorithm = if let Some(algorithm) = algorithm {
            let Some(algorithm) = algorithm.downcast_ref::<DotAlgorithmAttr>() else {
                return invalid(
                    self,
                    ctx,
                    "dot_general algorithm must use the StableHLO dot_algorithm attribute",
                );
            };
            algorithm.verify(ctx)?;
            Some(algorithm)
        } else {
            None
        };
        let precision_attribute = ir
            .attributes
            .0
            .get(&Identifier::new(crate::attributes::keys::DOT_PRECISION));
        let p = if let Some(precision_attribute) = precision_attribute {
            let Some(precision_attribute) = precision_attribute.downcast_ref::<VecAttr>() else {
                return invalid(
                    self,
                    ctx,
                    "dot_general precision must use an integer-list attribute",
                );
            };
            let Some(precision) = read_precision_values(precision_attribute) else {
                return invalid(self, ctx, "dot_general precision configuration is invalid");
            };
            precision
        } else {
            Vec::new()
        };
        if algorithm.is_some() && p.iter().any(|&precision| precision != Precision::Default) {
            return invalid(
                self,
                ctx,
                "dot_general algorithm requires DEFAULT precision configuration",
            );
        }
        let (Some(lc), Some(rc), Some(lb), Some(rb)) = (
            dimensions
                .lhs_contracting_dimensions
                .iter()
                .map(|&axis| u64::try_from(axis).ok())
                .collect::<Option<Vec<_>>>(),
            dimensions
                .rhs_contracting_dimensions
                .iter()
                .map(|&axis| u64::try_from(axis).ok())
                .collect::<Option<Vec<_>>>(),
            dimensions
                .lhs_batching_dimensions
                .iter()
                .map(|&axis| u64::try_from(axis).ok())
                .collect::<Option<Vec<_>>>(),
            dimensions
                .rhs_batching_dimensions
                .iter()
                .map(|&axis| u64::try_from(axis).ok())
                .collect::<Option<Vec<_>>>(),
        ) else {
            return invalid(self, ctx, "dot_general dimensions must be nonnegative");
        };
        let axes_ok = |rank: usize, a: &[u64], b: &[u64]| {
            let mut seen = vec![false; rank];
            a.iter().chain(b).all(|&x| {
                let i = x as usize;
                if i >= rank || seen[i] {
                    false
                } else {
                    seen[i] = true;
                    true
                }
            })
        };
        if !dot_element_compatible(ctx, le, re, oe, &rc)
            || lc.len() != rc.len()
            || lb.len() != rb.len()
            || p.len() > 2
            || p.len() > 2
            || !axes_ok(l.len(), &lb, &lc)
            || !axes_ok(r.len(), &rb, &rc)
            || lc
                .iter()
                .zip(&rc)
                .any(|(&a, &b)| !compatible_dim(l[a as usize], r[b as usize]))
            || lb
                .iter()
                .zip(&rb)
                .any(|(&a, &b)| !compatible_dim(l[a as usize], r[b as usize]))
            || !crate::verification::compute::validate_dot_general(
                &l,
                &r,
                &o,
                &DotDimensionNumbersAttr {
                    lhs_batching_dimensions: lb.iter().map(|&x| x as i64).collect(),
                    rhs_batching_dimensions: rb.iter().map(|&x| x as i64).collect(),
                    lhs_contracting_dimensions: lc.iter().map(|&x| x as i64).collect(),
                    rhs_contracting_dimensions: rc.iter().map(|&x| x as i64).collect(),
                },
            )
        {
            return invalid(
                self,
                ctx,
                "dot_general dimensions, precision, or elements are invalid",
            );
        };
        let mut shape = Vec::new();
        for (&a, &b) in lb.iter().zip(&rb) {
            shape.push(l[a as usize]);
            let _ = b;
        }
        for (axis, &dimension) in l.iter().enumerate() {
            if !lb.contains(&(axis as u64)) && !lc.contains(&(axis as u64)) {
                shape.push(dimension);
            }
        }
        for (axis, &dimension) in r.iter().enumerate() {
            if !rb.contains(&(axis as u64)) && !rc.contains(&(axis as u64)) {
                shape.push(dimension);
            }
        }
        if shape.len() != o.len()
            || shape
                .iter()
                .zip(&o)
                .any(|(&a, &b)| a >= 0 && b >= 0 && a != b)
        {
            return invalid(self, ctx, "dot_general result shape is invalid");
        };
        Ok(())
    }
}

/// Computes the dot product of rank-one or rank-two tensors.
#[pliron_op(name="stablehlo.dot",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface],attributes=(stablehlo_compute_dot_precision:VecAttr))]
pub struct DotOp;
impl DotOp {
    /// Creates a `stablehlo.dot` operation with the supplied result type. Uses `lhs`, `rhs` as operands. Configures the operation with `precision`.
    pub fn new(
        ctx: &mut Context,
        lhs: Value,
        rhs: Value,
        result: TypeHandle,
        precision: Vec<Precision>,
    ) -> Self {
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![lhs, rhs],
                vec![],
                0,
            ),
        };
        op.set_attr_stablehlo_compute_dot_precision(ctx, precision_values(ctx, precision));
        op
    }
}
impl Verify for DotOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() != 2 || ir.get_num_results() != 1 {
            return invalid(self, ctx, "dot arity mismatch");
        };
        let (Some((a, ae)), Some((b, be)), Some((o, oe))) = (
            parts(ctx, ir.get_operand(0)),
            parts(ctx, ir.get_operand(1)),
            parts(ctx, ir.get_result(0)),
        ) else {
            return invalid(self, ctx, "dot requires ranked tensors");
        };
        let p = if let Some(attribute) = ir.attributes.0.get(&Identifier::new(
            crate::attributes::keys::COMPUTE_DOT_PRECISION,
        )) {
            let Some(precision) = attribute
                .downcast_ref::<VecAttr>()
                .and_then(read_precision_values)
            else {
                return invalid(
                    self,
                    ctx,
                    "dot precision must use an integer-list attribute",
                );
            };
            precision
        } else {
            Vec::new()
        };
        if !(1..=2).contains(&a.len())
            || !(1..=2).contains(&b.len())
            || !dot_element_compatible(ctx, ae, be, oe, &[0])
            || p.len() > 2
            || p.len() > 2
            || !compatible_dim(a[a.len() - 1], b[0])
        {
            return invalid(
                self,
                ctx,
                "dot operand dimensions, precision or elements are invalid",
            );
        };
        let mut s = Vec::new();
        if a.len() == 2 {
            s.push(a[0])
        }
        if b.len() == 2 {
            s.push(b[1])
        }
        if s.len() != o.len() || s.iter().zip(&o).any(|(&x, &y)| x >= 0 && y >= 0 && x != y) {
            return invalid(self, ctx, "dot result shape is invalid");
        };
        Ok(())
    }
}

/// Applies a two-operand Einstein summation specified by an equation.
#[pliron_op(name="stablehlo.einsum",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface],attributes=(stablehlo_compute_einsum_equation:StringAttr))]
pub struct EinsumOp;
impl EinsumOp {
    /// Creates a `stablehlo.einsum` operation with the supplied result type. Uses `lhs`, `rhs` as operands. Configures the operation with `equation`.
    pub fn new(
        ctx: &mut Context,
        lhs: Value,
        rhs: Value,
        result: TypeHandle,
        equation: String,
    ) -> Self {
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![lhs, rhs],
                vec![],
                0,
            ),
        };
        op.set_attr_stablehlo_compute_einsum_equation(ctx, StringAttr::new(equation));
        op
    }
}
impl Verify for EinsumOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() != 2
            || ir.get_num_results() != 1
            || self
                .get_attr_stablehlo_compute_einsum_equation(ctx)
                .is_none()
            || parts(ctx, ir.get_operand(0)).is_none()
            || parts(ctx, ir.get_operand(1)).is_none()
            || parts(ctx, ir.get_result(0)).is_none()
        {
            return invalid(
                self,
                ctx,
                "einsum requires two ranked tensors, a result and an equation",
            );
        };
        Ok(())
    }
}
/// Applies an Einstein summation equation to a single tensor.
#[pliron_op(name="stablehlo.unary_einsum",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface],attributes=(stablehlo_compute_unary_einsum_equation:StringAttr))]
pub struct UnaryEinsumOp;
impl UnaryEinsumOp {
    /// Creates a `stablehlo.unary_einsum` operation with the supplied result type. Uses `input` as its operand. Configures the operation with `equation`.
    pub fn new(ctx: &mut Context, input: Value, result: TypeHandle, equation: String) -> Self {
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![input],
                vec![],
                0,
            ),
        };
        op.set_attr_stablehlo_compute_unary_einsum_equation(ctx, StringAttr::new(equation));
        op
    }
}
impl Verify for UnaryEinsumOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() != 1
            || ir.get_num_results() != 1
            || self
                .get_attr_stablehlo_compute_unary_einsum_equation(ctx)
                .is_none()
            || parts(ctx, ir.get_operand(0)).is_none()
            || parts(ctx, ir.get_result(0)).is_none()
        {
            return invalid(
                self,
                ctx,
                "unary_einsum requires ranked tensors and an equation",
            );
        };
        Ok(())
    }
}

/// Convolves input and kernel tensors using explicit dimension and window options.
#[pliron_op(name="stablehlo.convolution",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface],attributes=(stablehlo_compute_conv_dims:ConvDimensionNumbersAttr,stablehlo_compute_conv_strides:VecAttr,stablehlo_compute_conv_padding:VecAttr,stablehlo_compute_conv_lhs_dilation:VecAttr,stablehlo_compute_conv_rhs_dilation:VecAttr,stablehlo_compute_conv_groups:VecAttr,stablehlo_compute_conv_reversal:VecAttr,stablehlo_compute_conv_precision:VecAttr))]
pub struct ConvolutionOp;

/// Dimension and window settings for a convolution operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvolutionOptions {
    /// Input, kernel, and output batch, feature, and spatial axes.
    pub dimensions: ConvDimensionNumbersAttr,
    /// Window stride for each spatial axis.
    pub strides: Vec<u64>,
    /// Low/high padding pairs, flattened in spatial-axis order.
    pub padding: Vec<i64>,
    /// Dilation applied to the input.
    pub lhs_dilation: Vec<u64>,
    /// Dilation applied to the kernel.
    pub rhs_dilation: Vec<u64>,
    /// Number of feature groups.
    pub feature_group_count: u64,
    /// Number of batch groups.
    pub batch_group_count: u64,
    /// Whether each spatial window axis is reversed.
    pub window_reversal: Vec<bool>,
    /// Precision configuration for the input operands.
    pub precision: Vec<Precision>,
}

impl ConvolutionOp {
    /// Creates a `stablehlo.convolution` operation with explicit options.
    pub fn new(
        ctx: &mut Context,
        lhs: Value,
        rhs: Value,
        result: TypeHandle,
        options: ConvolutionOptions,
    ) -> Self {
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![lhs, rhs],
                vec![],
                0,
            ),
        };
        op.set_attr_stablehlo_compute_conv_dims(ctx, options.dimensions);
        op.set_attr_stablehlo_compute_conv_strides(ctx, ints(ctx, options.strides));
        op.set_attr_stablehlo_compute_conv_padding(ctx, signed_list(ctx, options.padding));
        op.set_attr_stablehlo_compute_conv_lhs_dilation(ctx, ints(ctx, options.lhs_dilation));
        op.set_attr_stablehlo_compute_conv_rhs_dilation(ctx, ints(ctx, options.rhs_dilation));
        op.set_attr_stablehlo_compute_conv_groups(
            ctx,
            ints(
                ctx,
                [options.feature_group_count, options.batch_group_count],
            ),
        );
        op.set_attr_stablehlo_compute_conv_reversal(
            ctx,
            ints(ctx, options.window_reversal.into_iter().map(u64::from)),
        );
        op.set_attr_stablehlo_compute_conv_precision(ctx, precision_values(ctx, options.precision));
        op
    }
}
impl Verify for ConvolutionOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() != 2 || ir.get_num_results() != 1 {
            return invalid(self, ctx, "convolution arity mismatch");
        };
        let (Some((a, ae)), Some((b, be)), Some((o, oe))) = (
            parts(ctx, ir.get_operand(0)),
            parts(ctx, ir.get_operand(1)),
            parts(ctx, ir.get_result(0)),
        ) else {
            return invalid(self, ctx, "convolution requires ranked tensors");
        };
        let Some(d) = self.get_attr_stablehlo_compute_conv_dims(ctx) else {
            return invalid(self, ctx, "convolution dimension numbers missing");
        };
        let (
            Some(strides),
            Some(pad),
            Some(ld),
            Some(rd),
            Some(g),
            Some(reversal),
            Some(precision),
        ) = (
            read_ints(self.get_attr_stablehlo_compute_conv_strides(ctx)),
            read_signed_ints(self.get_attr_stablehlo_compute_conv_padding(ctx)),
            read_ints(self.get_attr_stablehlo_compute_conv_lhs_dilation(ctx)),
            read_ints(self.get_attr_stablehlo_compute_conv_rhs_dilation(ctx)),
            read_ints(self.get_attr_stablehlo_compute_conv_groups(ctx)),
            read_ints(self.get_attr_stablehlo_compute_conv_reversal(ctx)),
            read_precision_attribute(self.get_attr_stablehlo_compute_conv_precision(ctx)),
        )
        else {
            return invalid(self, ctx, "convolution window attributes missing");
        };
        let rank = a.len();
        let valid_perm = |first: i64, second: i64, sp: &[i64]| {
            if sp.len() + 2 != rank {
                return false;
            };
            let mut xs = vec![first, second];
            xs.extend_from_slice(sp);
            let mut seen = vec![false; rank];
            xs.iter().all(|&x| {
                let i = x as usize;
                if i >= rank || seen[i] {
                    false
                } else {
                    seen[i] = true;
                    true
                }
            })
        };
        if rank < 2
            || b.len() != rank
            || o.len() != rank
            || !convolution_element_compatible(
                ctx,
                ae,
                be,
                oe,
                d.kernel_output_feature_dimension,
                d.output_feature_dimension,
            )
            || !crate::verification::compute::valid_convolution_dimensions(rank, &d)
            || !valid_perm(
                d.input_batch_dimension,
                d.input_feature_dimension,
                &d.input_spatial_dimensions,
            )
            || !valid_perm(
                d.kernel_input_feature_dimension,
                d.kernel_output_feature_dimension,
                &d.kernel_spatial_dimensions,
            )
            || !valid_perm(
                d.output_batch_dimension,
                d.output_feature_dimension,
                &d.output_spatial_dimensions,
            )
            || g.len() != 2
            || g[0] == 0
            || g[1] == 0
            || strides.len() != rank - 2
            || pad.len() != 2 * (rank - 2)
            || ld.len() != rank - 2
            || rd.len() != rank - 2
            || strides.contains(&0)
            || ld.contains(&0)
            || rd.contains(&0)
            || reversal.len() != rank - 2
            || reversal.iter().any(|&x| x > 1)
            || precision.len() > 2
            || precision.len() > 2
            || g[0] > 1 && g[1] > 1
            || a[d.input_feature_dimension as usize] >= 0
                && a[d.input_feature_dimension as usize] % (g[0] as i64) != 0
            || b[d.kernel_input_feature_dimension as usize] >= 0
                && b[d.kernel_input_feature_dimension as usize] % (g[0] as i64) != 0
        {
            return invalid(
                self,
                ctx,
                "invalid convolution dimensions, groups, precision, or window attributes",
            );
        };
        if parts(ctx, ir.get_operand(0)).is_none() {
            return invalid(self, ctx, "convolution lhs type invalid");
        }
        if parts(ctx, ir.get_operand(1)).is_none() {
            return invalid(self, ctx, "convolution rhs type invalid");
        }
        let strides_i = strides
            .iter()
            .map(|&x| i64::try_from(x).ok())
            .collect::<Option<Vec<_>>>();
        let lhs_i = ld
            .iter()
            .map(|&x| i64::try_from(x).ok())
            .collect::<Option<Vec<_>>>();
        let rhs_i = rd
            .iter()
            .map(|&x| i64::try_from(x).ok())
            .collect::<Option<Vec<_>>>();
        let padding = pad
            .chunks_exact(2)
            .map(|p| [p[0], p[1]])
            .collect::<Vec<_>>();
        let (Some(strides_i), Some(lhs_i), Some(rhs_i)) = (strides_i, lhs_i, rhs_i) else {
            return invalid(self, ctx, "convolution window values overflow");
        };
        let options = crate::inference::ConvolutionOptions {
            dimensions: &d,
            strides: Some(&strides_i),
            padding: Some(&padding),
            lhs_dilation: Some(&lhs_i),
            rhs_dilation: Some(&rhs_i),
            feature_group_count: g[0] as i64,
            batch_group_count: g[1] as i64,
        };
        let expected = crate::inference::infer_convolution_type(
            ctx,
            ir.get_operand(0).get_type(ctx),
            ir.get_operand(1).get_type(ctx),
            oe,
            options,
        );
        let Some(expected) = expected.ok().and_then(|t| parts_ty(ctx, t)) else {
            return invalid(self, ctx, "convolution shape inference failed");
        };
        if expected.0.len() != o.len()
            || expected
                .0
                .iter()
                .zip(o.iter())
                .any(|(&x, &y)| x >= 0 && y >= 0 && x != y)
        {
            return invalid(self, ctx, "convolution result shape is invalid");
        };
        Ok(())
    }
}

/// Convolves input and kernel tensors with runtime-provided padding.
#[pliron_op(name="stablehlo.dynamic_conv",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface],attributes=(stablehlo_compute_dynamic_conv_dims:ConvDimensionNumbersAttr,stablehlo_compute_dynamic_conv_strides:VecAttr,stablehlo_compute_dynamic_conv_lhs_dilation:VecAttr,stablehlo_compute_dynamic_conv_rhs_dilation:VecAttr,stablehlo_compute_dynamic_conv_groups:VecAttr,stablehlo_compute_dynamic_conv_reversal:VecAttr,stablehlo_compute_dynamic_conv_precision:VecAttr))]
pub struct DynamicConvOp;

/// Optional window settings for a dynamic convolution operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DynamicConvolutionOptions {
    /// Window stride for each spatial axis.
    pub strides: Vec<u64>,
    /// Dilation applied to the input.
    pub lhs_dilation: Vec<u64>,
    /// Dilation applied to the kernel.
    pub rhs_dilation: Vec<u64>,
    /// Number of feature groups.
    pub feature_group_count: u64,
    /// Number of batch groups.
    pub batch_group_count: u64,
    /// Whether each spatial window axis is reversed.
    pub window_reversal: Vec<bool>,
    /// Precision configuration for the input operands.
    pub precision: Vec<Precision>,
}

impl DynamicConvOp {
    /// Creates a `stablehlo.dynamic_conv` operation with the supplied result type. Uses `lhs`, `rhs`, `padding` as operands. Configures the operation with `dimensions`.
    pub fn new(
        ctx: &mut Context,
        lhs: Value,
        rhs: Value,
        padding: Value,
        result: TypeHandle,
        dimensions: ConvDimensionNumbersAttr,
    ) -> Self {
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![lhs, rhs, padding],
                vec![],
                0,
            ),
        };
        op.set_attr_stablehlo_compute_dynamic_conv_dims(ctx, dimensions);
        op.set_attr_stablehlo_compute_dynamic_conv_strides(ctx, ints(ctx, []));
        op.set_attr_stablehlo_compute_dynamic_conv_lhs_dilation(ctx, ints(ctx, []));
        op.set_attr_stablehlo_compute_dynamic_conv_rhs_dilation(ctx, ints(ctx, []));
        op.set_attr_stablehlo_compute_dynamic_conv_groups(ctx, ints(ctx, [1, 1]));
        op.set_attr_stablehlo_compute_dynamic_conv_reversal(ctx, ints(ctx, []));
        op.set_attr_stablehlo_compute_dynamic_conv_precision(ctx, precision_values(ctx, []));
        op
    }
    /// Sets optional stride, dilation, grouping, reversal, and precision settings.
    pub fn with_window_options(self, ctx: &Context, options: DynamicConvolutionOptions) -> Self {
        self.set_attr_stablehlo_compute_dynamic_conv_strides(ctx, ints(ctx, options.strides));
        self.set_attr_stablehlo_compute_dynamic_conv_lhs_dilation(
            ctx,
            ints(ctx, options.lhs_dilation),
        );
        self.set_attr_stablehlo_compute_dynamic_conv_rhs_dilation(
            ctx,
            ints(ctx, options.rhs_dilation),
        );
        self.set_attr_stablehlo_compute_dynamic_conv_groups(
            ctx,
            ints(
                ctx,
                [options.feature_group_count, options.batch_group_count],
            ),
        );
        self.set_attr_stablehlo_compute_dynamic_conv_reversal(
            ctx,
            ints(ctx, options.window_reversal.into_iter().map(u64::from)),
        );
        self.set_attr_stablehlo_compute_dynamic_conv_precision(
            ctx,
            precision_values(ctx, options.precision),
        );
        self
    }
}
impl Verify for DynamicConvOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() != 3 || ir.get_num_results() != 1 {
            return invalid(self, ctx, "dynamic_conv arity mismatch");
        };
        let (Some((lhs, le)), Some((rhs, re)), Some((pad, pe)), Some((out, oe)), Some(dims)) = (
            parts(ctx, ir.get_operand(0)),
            parts(ctx, ir.get_operand(1)),
            parts(ctx, ir.get_operand(2)),
            parts(ctx, ir.get_result(0)),
            self.get_attr_stablehlo_compute_dynamic_conv_dims(ctx),
        ) else {
            return invalid(
                self,
                ctx,
                "dynamic_conv operands, result and dimension numbers are required",
            );
        };
        let rank = lhs.len();
        let padding_type = crate::types::primitive_element(ctx, pe).is_some_and(|e| e.is_integer());
        if rank < 2
            || rhs.len() != rank
            || out.len() != rank
            || !padding_type
            || !convolution_element_compatible(
                ctx,
                le,
                re,
                oe,
                dims.kernel_output_feature_dimension,
                dims.output_feature_dimension,
            )
            || pad != vec![(rank - 2) as i64, 2]
            || !crate::verification::compute::valid_convolution_dimensions(rank, &dims)
        {
            return invalid(
                self,
                ctx,
                "dynamic_conv dimensions, padding shape or elements invalid",
            );
        };
        let (
            Some(strides),
            Some(lhs_dilation),
            Some(rhs_dilation),
            Some(groups),
            Some(reversal),
            Some(precision),
        ) = (
            read_ints(self.get_attr_stablehlo_compute_dynamic_conv_strides(ctx)),
            read_ints(self.get_attr_stablehlo_compute_dynamic_conv_lhs_dilation(ctx)),
            read_ints(self.get_attr_stablehlo_compute_dynamic_conv_rhs_dilation(ctx)),
            read_ints(self.get_attr_stablehlo_compute_dynamic_conv_groups(ctx)),
            read_ints(self.get_attr_stablehlo_compute_dynamic_conv_reversal(ctx)),
            read_precision_attribute(self.get_attr_stablehlo_compute_dynamic_conv_precision(ctx)),
        )
        else {
            return invalid(self, ctx, "dynamic_conv window attributes are missing");
        };
        let spatial_rank = rank - 2;
        let valid_window = |values: &[u64]| {
            values.is_empty() || values.len() == spatial_rank && !values.contains(&0)
        };
        if !valid_window(&strides)
            || !valid_window(&lhs_dilation)
            || !valid_window(&rhs_dilation)
            || groups.len() != 2
            || groups.contains(&0)
            || groups[0] > 1 && groups[1] > 1
            || !reversal.is_empty()
                && (reversal.len() != spatial_rank || reversal.iter().any(|&value| value > 1))
            || precision.len() > 2
            || precision.len() > 2
            || groups.iter().any(|&value| value > i64::MAX as u64)
        {
            return invalid(
                self,
                ctx,
                "dynamic_conv window, groups, or precision invalid",
            );
        }
        let compatible = |a: i64, b: i64| a < 0 || b < 0 || a == b;
        let divide = |size: i64, divisor: u64| -> Option<i64> {
            if size < 0 {
                Some(-1)
            } else if (size as u64).is_multiple_of(divisor) {
                Some(size / divisor as i64)
            } else {
                None
            }
        };
        let input_features = divide(lhs[dims.input_feature_dimension as usize], groups[0]);
        let kernel_features = rhs[dims.kernel_input_feature_dimension as usize];
        let output_features = rhs[dims.kernel_output_feature_dimension as usize];
        let output_batch = divide(lhs[dims.input_batch_dimension as usize], groups[1]);
        if input_features.is_none_or(|size| !compatible(size, kernel_features))
            || divide(output_features, groups[0]).is_none()
            || divide(output_features, groups[1]).is_none()
            || output_batch
                .is_none_or(|size| !compatible(out[dims.output_batch_dimension as usize], size))
            || !compatible(out[dims.output_feature_dimension as usize], output_features)
        {
            return invalid(
                self,
                ctx,
                "dynamic_conv batch, feature, or group dimensions mismatch",
            );
        }
        Ok(())
    }
}

/// Applies a real or complex Fourier transform along the selected trailing axes.
#[pliron_op(name="stablehlo.fft",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface],attributes=(stablehlo_compute_fft_type:FftKind,stablehlo_compute_fft_length:VecAttr))]
pub struct FftOp;
impl FftOp {
    /// Creates a `stablehlo.fft` operation with the supplied result type. Uses `input` as its operand. Configures the operation with `kind`, `length`.
    pub fn new(
        ctx: &mut Context,
        input: Value,
        result: TypeHandle,
        kind: FftKind,
        length: Vec<u64>,
    ) -> Self {
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![input],
                vec![],
                0,
            ),
        };
        op.set_attr_stablehlo_compute_fft_type(ctx, kind);
        op.set_attr_stablehlo_compute_fft_length(ctx, ints(ctx, length));
        op
    }
}
impl Verify for FftOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() != 1 || ir.get_num_results() != 1 {
            return invalid(self, ctx, "fft arity mismatch");
        };
        let (Some((s, e)), Some((o, oe))) =
            (parts(ctx, ir.get_operand(0)), parts(ctx, ir.get_result(0)))
        else {
            return invalid(self, ctx, "fft requires ranked tensors");
        };
        let (Some(k), Some(l)) = (
            self.get_attr_stablehlo_compute_fft_type(ctx),
            read_ints(self.get_attr_stablehlo_compute_fft_length(ctx)),
        ) else {
            return invalid(self, ctx, "fft attributes missing");
        };
        if l.is_empty() || l.len() > 3 || l.len() > s.len() || l.contains(&0) {
            return invalid(self, ctx, "fft type or lengths invalid");
        };
        let Some(lengths) = l
            .iter()
            .map(|&v| i64::try_from(v).ok())
            .collect::<Option<Vec<_>>>()
        else {
            return invalid(self, ctx, "fft lengths overflow");
        };
        let Some(expected) =
            crate::inference::infer_fft_type(ctx, ir.get_operand(0).get_type(ctx), *k, &lengths)
                .ok()
                .and_then(|ty| parts_ty(ctx, ty))
        else {
            return invalid(self, ctx, "fft inference failed");
        };
        if expected.0.len() != o.len()
            || expected
                .0
                .iter()
                .zip(&o)
                .any(|(&a, &b)| a >= 0 && b >= 0 && a != b)
            || expected.1 != oe
        {
            return invalid(self, ctx, "fft result type is invalid");
        };
        let _ = (s, e);
        Ok(())
    }
}

/// Computes a Cholesky factorization of the trailing square matrix dimensions.
#[pliron_op(name="stablehlo.cholesky",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface],attributes=(stablehlo_compute_cholesky_lower:BoolAttr))]
pub struct CholeskyOp;
impl CholeskyOp {
    /// Creates a `stablehlo.cholesky` operation with the supplied result type. Uses `input` as its operand. Configures the operation with `lower`.
    pub fn new(ctx: &mut Context, input: Value, result: TypeHandle, lower: bool) -> Self {
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![input],
                vec![],
                0,
            ),
        };
        op.set_attr_stablehlo_compute_cholesky_lower(ctx, BoolAttr::new(lower));
        op
    }
}
impl Verify for CholeskyOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        let (Some((s, e)), Some((o, oe))) =
            (parts(ctx, ir.get_operand(0)), parts(ctx, ir.get_result(0)))
        else {
            return invalid(self, ctx, "cholesky requires ranked tensors");
        };
        if ir.get_num_operands() != 1
            || ir.get_num_results() != 1
            || s.len() < 2
            || s.len() != o.len()
            || e != oe
            || s.iter()
                .zip(o.iter())
                .any(|(&a, &b)| a >= 0 && b >= 0 && a != b)
            || s[s.len() - 2] >= 0 && s[s.len() - 1] >= 0 && s[s.len() - 2] != s[s.len() - 1]
        {
            return invalid(
                self,
                ctx,
                "cholesky requires square trailing dimensions and matching result",
            );
        };
        Ok(())
    }
}

/// Solves a triangular linear system with configurable side, triangle, and transpose mode.
#[pliron_op(name="stablehlo.triangular_solve",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface],attributes=(stablehlo_compute_triangular_flags:VecAttr))]
pub struct TriangularSolveOp;

/// Transpose applied to the triangular matrix before solving.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransposeMode {
    /// Use the matrix as-is.
    None,
    /// Use the matrix transpose.
    Transpose,
    /// Use the conjugate transpose.
    Adjoint,
}

/// Solve-side, triangle, diagonal, and transpose settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TriangularSolveOptions {
    /// Solve from the left when true, or from the right when false.
    pub left_side: bool,
    /// Use the lower triangle when true, or the upper triangle when false.
    pub lower: bool,
    /// Treat the diagonal as all ones when true.
    pub unit_diagonal: bool,
    /// Matrix transpose mode.
    pub transpose: TransposeMode,
}

impl TriangularSolveOp {
    /// Creates a `stablehlo.triangular_solve` operation with explicit options.
    pub fn new(
        ctx: &mut Context,
        a: Value,
        b: Value,
        result: TypeHandle,
        options: TriangularSolveOptions,
    ) -> Self {
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![a, b],
                vec![],
                0,
            ),
        };
        op.set_attr_stablehlo_compute_triangular_flags(
            ctx,
            ints(
                ctx,
                [
                    u64::from(options.left_side),
                    u64::from(options.lower),
                    u64::from(options.unit_diagonal),
                    match options.transpose {
                        TransposeMode::None => 0,
                        TransposeMode::Transpose => 1,
                        TransposeMode::Adjoint => 2,
                    },
                ],
            ),
        );
        op
    }
}
impl Verify for TriangularSolveOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() != 2 || ir.get_num_results() != 1 {
            return invalid(self, ctx, "triangular_solve arity mismatch");
        };
        let (Some((a, ae)), Some((b, be)), Some((o, oe))) = (
            parts(ctx, ir.get_operand(0)),
            parts(ctx, ir.get_operand(1)),
            parts(ctx, ir.get_result(0)),
        ) else {
            return invalid(self, ctx, "triangular_solve requires ranked tensors");
        };
        let Some(f) = read_ints(self.get_attr_stablehlo_compute_triangular_flags(ctx)) else {
            return invalid(self, ctx, "triangular_solve flags missing");
        };
        if f.len() != 4
            || f[0] > 1
            || f[1] > 1
            || f[2] > 1
            || f[3] > 2
            || a.len() < 2
            || a.len() != b.len()
            || b.len() != o.len()
            || ae != be
            || be != oe
        {
            return invalid(self, ctx, "invalid triangular_solve shapes or flags");
        };
        let Some(expected) = crate::inference::infer_triangular_solve_type(
            ctx,
            ir.get_operand(0).get_type(ctx),
            ir.get_operand(1).get_type(ctx),
            f[0] != 0,
        )
        .ok()
        .and_then(|ty| parts_ty(ctx, ty)) else {
            return invalid(self, ctx, "triangular_solve inference failed");
        };
        if expected.0.len() != o.len()
            || expected.1 != oe
            || expected
                .0
                .iter()
                .zip(&o)
                .any(|(&expected, &actual)| expected >= 0 && actual >= 0 && expected != actual)
        {
            return invalid(self, ctx, "triangular_solve result type mismatch");
        }
        Ok(())
    }
}

/// Normalizes a tensor using supplied per-feature mean and variance statistics.
#[pliron_op(name="stablehlo.batch_norm_inference",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface],attributes=(stablehlo_compute_bn_options:VecAttr))]
pub struct BatchNormInferenceOp;

/// Feature-axis and epsilon settings shared by StableHLO batch normalization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BatchNormOptions {
    /// Axis containing the feature channels.
    pub feature_index: u64,
    /// Positive finite epsilon added to the variance.
    pub epsilon: f64,
}

/// Operands to `stablehlo.batch_norm_inference`.
pub struct BatchNormInferenceOperands {
    /// Input tensor to normalize.
    pub x: Value,
    /// Per-feature scale.
    pub scale: Value,
    /// Per-feature offset.
    pub offset: Value,
    /// Per-feature estimated mean.
    pub mean: Value,
    /// Per-feature estimated variance.
    pub variance: Value,
}

impl BatchNormInferenceOp {
    /// Creates a `stablehlo.batch_norm_inference` operation with named operands and options.
    pub fn new(
        ctx: &mut Context,
        operands: BatchNormInferenceOperands,
        result: TypeHandle,
        options: BatchNormOptions,
    ) -> Self {
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![
                    operands.x,
                    operands.scale,
                    operands.offset,
                    operands.mean,
                    operands.variance,
                ],
                vec![],
                0,
            ),
        };
        op.set_attr_stablehlo_compute_bn_options(
            ctx,
            ints(ctx, [options.feature_index, options.epsilon.to_bits()]),
        );
        op
    }
}
impl Verify for BatchNormInferenceOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        verify_bn(
            self,
            ctx,
            5,
            1,
            false,
            read_ints(self.get_attr_stablehlo_compute_bn_options(ctx)),
        )
    }
}
/// Computes batch normalization outputs and batch statistics from the input batch.
#[pliron_op(name="stablehlo.batch_norm_training",format,attributes=(stablehlo_compute_bn_training_options:VecAttr))]
pub struct BatchNormTrainingOp;
impl BatchNormTrainingOp {
    /// Creates a `stablehlo.batch_norm_training` operation with explicit options.
    pub fn new(
        ctx: &mut Context,
        x: Value,
        scale: Value,
        offset: Value,
        results: Vec<TypeHandle>,
        options: BatchNormOptions,
    ) -> Self {
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                results,
                vec![x, scale, offset],
                vec![],
                0,
            ),
        };
        op.set_attr_stablehlo_compute_bn_training_options(
            ctx,
            ints(ctx, [options.feature_index, options.epsilon.to_bits()]),
        );
        op
    }
}
impl Verify for BatchNormTrainingOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        verify_bn(
            self,
            ctx,
            3,
            3,
            false,
            read_ints(self.get_attr_stablehlo_compute_bn_training_options(ctx)),
        )
    }
}
/// Computes gradients for batch normalization with respect to its inputs and parameters.
#[pliron_op(name="stablehlo.batch_norm_grad",format,attributes=(stablehlo_compute_bn_grad_options:VecAttr))]
pub struct BatchNormGradOp;

/// Operands to `stablehlo.batch_norm_grad`.
pub struct BatchNormGradOperands {
    /// Input tensor whose gradient is computed.
    pub x: Value,
    /// Per-feature scale.
    pub scale: Value,
    /// Per-feature estimated mean.
    pub mean: Value,
    /// Per-feature estimated variance.
    pub variance: Value,
    /// Gradient of the batch-normalization output.
    pub grad: Value,
}

impl BatchNormGradOp {
    /// Creates a `stablehlo.batch_norm_grad` operation with named operands and options.
    pub fn new(
        ctx: &mut Context,
        operands: BatchNormGradOperands,
        results: Vec<TypeHandle>,
        options: BatchNormOptions,
    ) -> Self {
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                results,
                vec![
                    operands.x,
                    operands.scale,
                    operands.mean,
                    operands.variance,
                    operands.grad,
                ],
                vec![],
                0,
            ),
        };
        op.set_attr_stablehlo_compute_bn_grad_options(
            ctx,
            ints(ctx, [options.feature_index, options.epsilon.to_bits()]),
        );
        op
    }
}
impl Verify for BatchNormGradOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        verify_bn(
            self,
            ctx,
            5,
            3,
            true,
            read_ints(self.get_attr_stablehlo_compute_bn_grad_options(ctx)),
        )
    }
}
fn verify_bn(
    op: &dyn Op,
    ctx: &Context,
    operands: usize,
    results: usize,
    has_grad_output: bool,
    options: Option<Vec<u64>>,
) -> Result<()> {
    let ir = op.get_operation().deref(ctx);
    if ir.get_num_operands() != operands || ir.get_num_results() != results {
        return invalid(op, ctx, "batch normalization arity mismatch");
    };
    let Some((shape, element)) = parts(ctx, ir.get_operand(0)) else {
        return invalid(op, ctx, "batch normalization data must be ranked");
    };
    let Some(config) = options else {
        return invalid(op, ctx, "batch normalization options missing");
    };
    if config.len() != 2
        || config[0] as usize >= shape.len()
        || !f64::from_bits(config[1]).is_finite()
        || f64::from_bits(config[1]) <= 0.0
    {
        return invalid(op, ctx, "batch normalization feature or epsilon invalid");
    };
    let feature = shape[config[0] as usize];
    for (i, v) in ir.operands().enumerate().skip(1) {
        let Some((dims, elem)) = parts(ctx, v) else {
            return invalid(op, ctx, "batch normalization operands must be ranked");
        };
        let grad_output = has_grad_output && i == operands - 1;
        if elem != element
            || if grad_output {
                dims.len() != shape.len()
                    || dims
                        .iter()
                        .zip(&shape)
                        .any(|(&a, &b)| a >= 0 && b >= 0 && a != b)
            } else {
                dims.len() != 1 || feature >= 0 && dims[0] >= 0 && dims[0] != feature
            }
        {
            return invalid(
                op,
                ctx,
                "batch normalization operand dimensions or elements are invalid",
            );
        }
    }
    for (i, v) in ir.results().enumerate() {
        let Some((dims, elem)) = parts_ty(ctx, v.get_type(ctx)) else {
            return invalid(op, ctx, "batch normalization results must be ranked");
        };
        let is_data = results == 1 || i == 0;
        if elem != element
            || if is_data {
                dims.len() != shape.len()
                    || dims
                        .iter()
                        .zip(&shape)
                        .any(|(&a, &b)| a >= 0 && b >= 0 && a != b)
            } else {
                dims.len() != 1 || feature >= 0 && dims[0] >= 0 && dims[0] != feature
            }
        {
            return invalid(
                op,
                ctx,
                "batch normalization result dimensions or elements are invalid",
            );
        }
    }
    Ok(())
}

/// Generates random values from the selected distribution using bounds and a runtime shape.
#[pliron_op(name="stablehlo.rng",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface],attributes=(stablehlo_compute_rng_distribution:RngDistributionAttr))]
pub struct RngOp;
impl RngOp {
    /// Creates a `stablehlo.rng` operation with the supplied result type. Uses `lower`, `upper`, `shape` as operands. Configures the operation with `distribution`.
    pub fn new(
        ctx: &mut Context,
        lower: Value,
        upper: Value,
        shape: Value,
        result: TypeHandle,
        distribution: RngDistributionAttr,
    ) -> Self {
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![lower, upper, shape],
                vec![],
                0,
            ),
        };
        op.set_attr_stablehlo_compute_rng_distribution(ctx, distribution);
        op
    }
}
impl Verify for RngOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() != 3 || ir.get_num_results() != 1 {
            return invalid(self, ctx, "rng arity mismatch");
        };
        let (
            Some((lower, le)),
            Some((upper, ue)),
            Some((shape, se)),
            Some((result, re)),
            Some(distribution),
        ) = (
            parts(ctx, ir.get_operand(0)),
            parts(ctx, ir.get_operand(1)),
            parts(ctx, ir.get_operand(2)),
            parts(ctx, ir.get_result(0)),
            self.get_attr_stablehlo_compute_rng_distribution(ctx),
        )
        else {
            return invalid(self, ctx, "rng requires ranked bounds, shape, and result");
        };
        if le != ue
            || le != re
            || shape.len() != 1
            || shape[0] < 0
            || result.len() != shape[0] as usize
            || !crate::types::primitive_element(ctx, se).is_some_and(|e| e.is_integer())
            || matches!(*distribution, RngDistributionAttr::Normal)
                && !crate::types::primitive_element(ctx, re).is_some_and(|e| e.is_float())
            || lower.len() != upper.len()
            || lower
                .iter()
                .zip(&upper)
                .any(|(&a, &b)| a >= 0 && b >= 0 && a != b)
        {
            return invalid(
                self,
                ctx,
                "rng shape, bounds, elements, or distribution invalid",
            );
        };
        Ok(())
    }
}

/// Advances RNG state and returns the updated state together with generated bits.
#[pliron_op(name="stablehlo.rng_bit_generator",format,attributes=(stablehlo_compute_rng_bit_algorithm:RngAlgorithmAttr))]
pub struct RngBitGeneratorOp;
impl RngBitGeneratorOp {
    /// Creates a `stablehlo.rng_bit_generator` operation with the supplied result types. Uses `state` as its operand. Configures the operation with `algorithm`.
    pub fn new(
        ctx: &mut Context,
        state: Value,
        results: Vec<TypeHandle>,
        algorithm: RngAlgorithmAttr,
    ) -> Self {
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                results,
                vec![state],
                vec![],
                0,
            ),
        };
        op.set_attr_stablehlo_compute_rng_bit_algorithm(ctx, algorithm);
        op
    }
}
impl Verify for RngBitGeneratorOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() != 1 || ir.get_num_results() != 2 {
            return invalid(self, ctx, "rng_bit_generator arity mismatch");
        };
        let (
            Some((state, element)),
            Some((state_result, state_element)),
            Some((_, data_element)),
            Some(algorithm),
        ) = (
            parts(ctx, ir.get_operand(0)),
            parts(ctx, ir.get_result(0)),
            parts(ctx, ir.get_result(1)),
            self.get_attr_stablehlo_compute_rng_bit_algorithm(ctx),
        )
        else {
            return invalid(
                self,
                ctx,
                "rng_bit_generator requires ranked state and results plus algorithm",
            );
        };
        if state.len() != 1
            || state != state_result
            || element != state_element
            || crate::types::primitive_element(ctx, element) != Some(ElementType::Ui64)
            || crate::types::primitive_element(ctx, data_element).is_none_or(|e| !e.is_integer())
            || match *algorithm {
                RngAlgorithmAttr::Default => false,
                RngAlgorithmAttr::ThreeFry => state[0] >= 0 && state[0] != 2,
                RngAlgorithmAttr::Philox => state[0] >= 0 && state[0] != 2 && state[0] != 3,
            }
        {
            return invalid(self, ctx, "rng_bit_generator state or result types invalid");
        };
        Ok(())
    }
}
