use crate::verification::invalid;
use crate::{attributes::*, types::*};
use pliron::{
    attribute::AttrObj,
    builtin::attributes::{IntegerAttr, VecAttr},
    builtin::types::{IntegerType, Signedness},
    common_traits::Verify,
    context::Context,
    derive::pliron_op,
    op::Op,
    operation::Operation,
    result::Result,
    r#type::{TypeHandle, Typed},
    utils::apint::{APInt, bw},
    value::Value,
};

/// Broadcasts a tensor by mapping its dimensions into a larger result rank.
#[pliron_op(name="stablehlo.broadcast_in_dim",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface],attributes=(stablehlo_broadcast_dimensions:VecAttr))]
pub struct BroadcastInDimOp;
impl BroadcastInDimOp {
    /// Creates a `stablehlo.broadcast_in_dim` operation with the supplied result type. Uses `input` as its operand. Configures the operation with `dimensions`.
    pub fn new(
        ctx: &mut Context,
        input: Value,
        result: TypeHandle,
        dimensions: Vec<usize>,
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
        let dimensions = integer_list(ctx, dimensions.into_iter().map(|d| d as u64));
        op.set_attr_stablehlo_broadcast_dimensions(ctx, dimensions);
        op
    }
    /// Returns the dimensions encoded by this operation, or `None` if the attribute is missing or malformed.
    pub fn dimensions(&self, ctx: &Context) -> Option<Vec<usize>> {
        read_integer_list(&*self.get_attr_stablehlo_broadcast_dimensions(ctx)?)?
            .into_iter()
            .map(|v| usize::try_from(v).ok())
            .collect()
    }
    /// Broadcasts `input` into `result_shape`, mapping each input axis to `dimensions`.
    ///
    /// The result uses the input element type. Returns an error if `input` is not a ranked tensor;
    /// mapping rank, uniqueness, range, and shape compatibility are checked by operation
    /// verification.
    pub fn new_inferred(
        ctx: &mut Context,
        input: Value,
        result_shape: &[i64],
        dimensions: Vec<usize>,
    ) -> Result<Self> {
        let (_, element) = crate::ops::builder_option(
            shape_elem(ctx, input.get_type(ctx)),
            "stablehlo.broadcast_in_dim",
            "operand must be a supported ranked tensor",
        )?;
        Ok(Self::new(
            ctx,
            input,
            TensorType::get(ctx, result_shape.to_vec(), element).into(),
            dimensions,
        ))
    }
}
impl Verify for BroadcastInDimOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() != 1 || ir.get_num_results() != 1 {
            return invalid(self, ctx, "broadcast arity mismatch");
        }
        let (Some((src, se)), Some((dst, de)), Some(dims)) = (
            shape_elem(ctx, ir.get_operand(0).get_type(ctx)),
            shape_elem(ctx, ir.get_result(0).get_type(ctx)),
            self.dimensions(ctx),
        ) else {
            return invalid(self, ctx, "broadcast types or dimensions are invalid");
        };
        let mut seen = vec![false; dst.len()];
        if se != de
            || dims.len() != src.len()
            || dims
                .iter()
                .any(|&d| d >= dst.len() || std::mem::replace(&mut seen[d], true))
            || dims
                .iter()
                .zip(&src)
                .any(|(&axis, &size)| size >= 0 && size != 1 && dst[axis] >= 0 && size != dst[axis])
        {
            return invalid(self, ctx, "invalid broadcast dimensions or shapes");
        }
        Ok(())
    }
}

/// Changes the shape of an input tensor without changing its element sequence.
#[pliron_op(name="stablehlo.reshape",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct ReshapeOp;
impl ReshapeOp {
    /// Creates a `stablehlo.reshape` operation with the supplied result type. Uses `input` as its operand.
    pub fn new(ctx: &mut Context, input: Value, result: TypeHandle) -> Self {
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![input],
                vec![],
                0,
            ),
        }
    }
    /// Constructs a reshape from `input` to `shape`, retaining the input element type.
    ///
    /// Dimensions may be dynamic (`-1`), but values below `-1` are rejected. Returns an error if
    /// `input` is not ranked or the requested dimensions are malformed. Element-count compatibility
    /// is checked by operation verification, not by this builder.
    pub fn new_inferred(ctx: &mut Context, input: Value, shape: &[i64]) -> Result<Self> {
        let (_, element) = crate::ops::builder_option(
            shape_elem(ctx, input.get_type(ctx)),
            "stablehlo.reshape",
            "operand must be a supported ranked tensor",
        )?;
        if shape.iter().any(|&d| d < -1) {
            return crate::ops::builder_option(
                None,
                "stablehlo.reshape",
                "builder inputs are invalid",
            );
        }
        Ok(Self::new(
            ctx,
            input,
            TensorType::get(ctx, shape.to_vec(), element).into(),
        ))
    }
}
impl Verify for ReshapeOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() != 1 || ir.get_num_results() != 1 {
            return invalid(self, ctx, "reshape arity mismatch");
        }
        let (Some((src, se)), Some((dst, de))) = (
            shape_elem(ctx, ir.get_operand(0).get_type(ctx)),
            shape_elem(ctx, ir.get_result(0).get_type(ctx)),
        ) else {
            return invalid(self, ctx, "reshape types must be ranked tensors");
        };
        if se != de || src.iter().any(|&x| x < -1) || dst.iter().any(|&x| x < -1) {
            return invalid(
                self,
                ctx,
                "reshape must preserve a valid element type and ranked shape",
            );
        }
        let count = |shape: &[i64]| {
            shape.iter().try_fold(1u64, |acc, &dimension| {
                u64::try_from(dimension).ok()?.checked_mul(acc)
            })
        };
        if let (Some(input_count), Some(result_count)) = (count(&src), count(&dst))
            && input_count != result_count
        {
            return invalid(self, ctx, "reshape must preserve element count");
        }
        Ok(())
    }
}

/// Permutes tensor axes without changing the element values.
#[pliron_op(name="stablehlo.transpose",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface],attributes=(stablehlo_transpose_permutation:VecAttr))]
pub struct TransposeOp;
impl TransposeOp {
    /// Reorders the axes of `input` according to `permutation`.
    ///
    /// The result shape is inferred by permuting the input dimensions and retains its element type.
    /// Returns an error if the input is not a ranked tensor or the permutation is not valid for
    /// its rank.
    pub fn new_inferred(ctx: &mut Context, input: Value, permutation: Vec<usize>) -> Result<Self> {
        let ty = crate::ops::builder_result(
            crate::inference::infer_transpose(ctx, input.get_type(ctx), &permutation),
            "stablehlo.transpose",
        )?;
        Ok(Self::new(ctx, input, ty, permutation))
    }
    /// Creates a `stablehlo.transpose` operation with the supplied result type. Uses `input` as its operand. Configures the operation with `permutation`.
    pub fn new(
        ctx: &mut Context,
        input: Value,
        result: TypeHandle,
        permutation: Vec<usize>,
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
        let permutation = integer_list(ctx, permutation.into_iter().map(|d| d as u64));
        op.set_attr_stablehlo_transpose_permutation(ctx, permutation);
        op
    }
    /// Returns the dimension permutation encoded by this operation, if available.
    pub fn permutation(&self, ctx: &Context) -> Option<Vec<usize>> {
        read_integer_list(&*self.get_attr_stablehlo_transpose_permutation(ctx)?)?
            .into_iter()
            .map(|v| usize::try_from(v).ok())
            .collect()
    }
}
impl Verify for TransposeOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() != 1 || ir.get_num_results() != 1 {
            return invalid(self, ctx, "transpose arity mismatch");
        }
        let (Some((src, se)), Some((dst, de)), Some(permutation)) = (
            shape_elem(ctx, ir.get_operand(0).get_type(ctx)),
            shape_elem(ctx, ir.get_result(0).get_type(ctx)),
            self.permutation(ctx),
        ) else {
            return invalid(self, ctx, "transpose types or permutation are invalid");
        };
        let mut seen = vec![false; src.len()];
        let expected =
            crate::inference::infer_transpose(ctx, ir.get_operand(0).get_type(ctx), &permutation)
                .ok();
        if permutation.len() != src.len()
            || dst.len() != src.len()
            || se != de
            || permutation.iter().enumerate().any(|(axis, &d)| {
                d >= src.len()
                    || std::mem::replace(&mut seen[d], true)
                    || src[d] >= 0 && dst[axis] >= 0 && src[d] != dst[axis]
            })
            || expected
                .is_none_or(|ty| !compatible_native_shape(ctx, ty, ir.get_result(0).get_type(ctx)))
        {
            return invalid(
                self,
                ctx,
                "transpose permutation must exactly permute input axes",
            );
        }
        Ok(())
    }
}

/// Adds leading dimensions to a tensor by broadcasting its values.
#[pliron_op(name="stablehlo.broadcast",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface],attributes=(stablehlo_broadcast_sizes:VecAttr))]
pub struct BroadcastOp;
impl BroadcastOp {
    /// Creates a `stablehlo.broadcast` operation with the supplied result type. Uses `input` as its operand. Configures the operation with `sizes`.
    pub fn new(ctx: &mut Context, input: Value, result: TypeHandle, sizes: Vec<i64>) -> Self {
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
        op.set_attr_stablehlo_broadcast_sizes(ctx, signed_list(ctx, sizes));
        op
    }
    /// Broadcasts `input` by prepending the requested `sizes` to its dimensions.
    ///
    /// The inferred result preserves the input element type. Returns an error when the input shape
    /// or requested sizes are unsupported.
    pub fn new_inferred(ctx: &mut Context, input: Value, sizes: Vec<i64>) -> Result<Self> {
        let ty = crate::ops::builder_result(
            crate::inference::infer_broadcast(ctx, input.get_type(ctx), &sizes),
            "stablehlo.broadcast",
        )?;
        Ok(Self::new(ctx, input, ty, sizes))
    }
    /// Returns the slice or broadcast sizes encoded by this operation, if available.
    pub fn sizes(&self, ctx: &Context) -> Option<Vec<i64>> {
        signed_integer_list(&*self.get_attr_stablehlo_broadcast_sizes(ctx)?)
    }
}
impl Verify for BroadcastOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() != 1 || ir.get_num_results() != 1 {
            return invalid(self, ctx, "broadcast arity mismatch");
        }
        let (Some(sizes), Some((src, se)), Some((dst, de))) = (
            self.sizes(ctx),
            shape_elem(ctx, ir.get_operand(0).get_type(ctx)),
            shape_elem(ctx, ir.get_result(0).get_type(ctx)),
        ) else {
            return invalid(self, ctx, "broadcast types or sizes are invalid");
        };
        let expected =
            crate::inference::infer_broadcast(ctx, ir.get_operand(0).get_type(ctx), &sizes).ok();
        if se != de
            || expected
                .is_none_or(|t| !compatible_native_shape(ctx, t, ir.get_result(0).get_type(ctx)))
        {
            return invalid(self, ctx, "broadcast result shape is invalid");
        }
        let _ = (src, dst);
        Ok(())
    }
}

/// Reverses elements along each selected tensor axis.
#[pliron_op(name="stablehlo.reverse",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface],attributes=(stablehlo_reverse_dimensions:VecAttr))]
pub struct ReverseOp;
impl ReverseOp {
    /// Creates a `stablehlo.reverse` operation with the supplied result type. Uses `input` as its operand. Configures the operation with `dimensions`.
    pub fn new(
        ctx: &mut Context,
        input: Value,
        result: TypeHandle,
        dimensions: Vec<usize>,
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
        op.set_attr_stablehlo_reverse_dimensions(
            ctx,
            integer_list(ctx, dimensions.into_iter().map(|x| x as u64)),
        );
        op
    }
    /// Returns the dimensions encoded by this operation, or `None` if the attribute is missing or malformed.
    pub fn dimensions(&self, ctx: &Context) -> Option<Vec<usize>> {
        read_integer_list(&*self.get_attr_stablehlo_reverse_dimensions(ctx)?)?
            .into_iter()
            .map(|x| usize::try_from(x).ok())
            .collect()
    }
    /// Constructs a reverse operation using the input type as its result type.
    ///
    /// Returns `Ok` without checking the axis list. Call [`Verify::verify`] to ensure the axes are
    /// unique and within the input rank.
    pub fn new_inferred(ctx: &mut Context, input: Value, dimensions: Vec<usize>) -> Result<Self> {
        Ok(Self::new(ctx, input, input.get_type(ctx), dimensions))
    }
}
impl Verify for ReverseOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() != 1 || ir.get_num_results() != 1 {
            return invalid(self, ctx, "reverse arity mismatch");
        }
        let (Some((a, ae)), Some((b, be)), Some(d)) = (
            shape_elem(ctx, ir.get_operand(0).get_type(ctx)),
            shape_elem(ctx, ir.get_result(0).get_type(ctx)),
            self.dimensions(ctx),
        ) else {
            return invalid(self, ctx, "reverse types or dimensions are invalid");
        };
        let mut seen = vec![false; a.len()];
        if a != b
            || ae != be
            || d.iter()
                .any(|&x| x >= a.len() || std::mem::replace(&mut seen[x], true))
        {
            return invalid(
                self,
                ctx,
                "reverse dimensions must be unique in-range axes and preserve type",
            );
        }
        Ok(())
    }
}

/// Broadcasts a tensor using a runtime result shape and an explicit axis mapping.
#[pliron_op(name="stablehlo.dynamic_broadcast_in_dim",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface],attributes=(stablehlo_dynamic_broadcast_dimensions:VecAttr,stablehlo_dynamic_broadcast_known_expanding:VecAttr,stablehlo_dynamic_broadcast_known_nonexpanding:VecAttr))]
pub struct DynamicBroadcastInDimOp;
impl DynamicBroadcastInDimOp {
    /// Creates a `stablehlo.dynamic_broadcast_in_dim` operation with the supplied result type. Uses `input`, `output_dimensions` as operands. Configures the operation with `dimensions`, `known_expanding`, `known_nonexpanding`.
    pub fn new(
        ctx: &mut Context,
        input: Value,
        output_dimensions: Value,
        result: TypeHandle,
        dimensions: Vec<usize>,
        known_expanding: Vec<usize>,
        known_nonexpanding: Vec<usize>,
    ) -> Self {
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![input, output_dimensions],
                vec![],
                0,
            ),
        };
        op.set_attr_stablehlo_dynamic_broadcast_dimensions(
            ctx,
            integer_list(ctx, dimensions.into_iter().map(|x| x as u64)),
        );
        op.set_attr_stablehlo_dynamic_broadcast_known_expanding(
            ctx,
            integer_list(ctx, known_expanding.into_iter().map(|x| x as u64)),
        );
        op.set_attr_stablehlo_dynamic_broadcast_known_nonexpanding(
            ctx,
            integer_list(ctx, known_nonexpanding.into_iter().map(|x| x as u64)),
        );
        op
    }
    /// Broadcasts `input` to the runtime shape in `output_dimensions`, mapping input axes through
    /// `dimensions`.
    ///
    /// The output rank is inferred from the length of the runtime dimension vector. Returns an
    /// error for unsupported tensor types, invalid mappings, or an invalid shape vector. The
    /// inferred builder leaves the optional known-expanding and known-nonexpanding axis sets empty.
    pub fn new_inferred(
        ctx: &mut Context,
        input: Value,
        output_dimensions: Value,
        dimensions: Vec<usize>,
    ) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_dynamic_broadcast_in_dim_type(
                ctx,
                input.get_type(ctx),
                output_dimensions.get_type(ctx),
                &dimensions,
            ),
            "stablehlo.dynamic_broadcast_in_dim",
        )?;
        Ok(Self::new(
            ctx,
            input,
            output_dimensions,
            result,
            dimensions,
            vec![],
            vec![],
        ))
    }
}
impl Verify for DynamicBroadcastInDimOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() != 2 || ir.get_num_results() != 1 {
            return invalid(self, ctx, "dynamic_broadcast_in_dim arity mismatch");
        }
        let optional_axes = |name: &'static str| match ir
            .attributes
            .0
            .get(&pliron::identifier::Identifier::new(name))
        {
            None => Some(Vec::new()),
            Some(value) => value.downcast_ref::<VecAttr>().and_then(read_integer_list),
        };
        let (
            Some((src, src_elem)),
            Some((dst, dst_elem)),
            Some((output_shape, dimension_elem)),
            Some(dimensions),
            Some(expanding),
            Some(nonexpanding),
        ) = (
            shape_elem(ctx, ir.get_operand(0).get_type(ctx)),
            shape_elem(ctx, ir.get_result(0).get_type(ctx)),
            shape_elem(ctx, ir.get_operand(1).get_type(ctx)),
            self.get_attr_stablehlo_dynamic_broadcast_dimensions(ctx)
                .as_deref()
                .and_then(read_integer_list),
            optional_axes(crate::attributes::keys::DYNAMIC_BROADCAST_KNOWN_EXPANDING),
            optional_axes(crate::attributes::keys::DYNAMIC_BROADCAST_KNOWN_NONEXPANDING),
        )
        else {
            return invalid(
                self,
                ctx,
                "dynamic broadcast attributes or tensors are invalid",
            );
        };
        if src_elem != dst_elem
            || output_shape != vec![dst.len() as i64]
            || dimension_elem
                .deref(ctx)
                .downcast_ref::<IntegerType>()
                .is_none()
            || dimensions.len() != src.len()
        {
            return invalid(
                self,
                ctx,
                "dynamic broadcast dimension tensor or rank is invalid",
            );
        }
        let mut mapped = vec![false; dst.len()];
        for (&axis, &size) in dimensions.iter().zip(&src) {
            let Ok(axis) = usize::try_from(axis) else {
                return invalid(self, ctx, "dynamic broadcast axis is negative");
            };
            if axis >= dst.len()
                || mapped[axis]
                || size >= 0 && size != 1 && dst[axis] >= 0 && size != dst[axis]
            {
                return invalid(
                    self,
                    ctx,
                    "dynamic broadcast dimension mapping or shape is invalid",
                );
            }
            mapped[axis] = true;
        }
        let validate_axes = |values: &[u64]| {
            let mut seen = vec![false; src.len()];
            values.iter().all(|&v| {
                usize::try_from(v).ok().is_some_and(|axis| {
                    axis < src.len() && !std::mem::replace(&mut seen[axis], true)
                })
            })
        };
        if !validate_axes(&expanding)
            || !validate_axes(&nonexpanding)
            || expanding.iter().any(|axis| nonexpanding.contains(axis))
        {
            return invalid(
                self,
                ctx,
                "dynamic broadcast known-dimension attributes are invalid",
            );
        }
        Ok(())
    }
}

/// Reshapes an input tensor using a runtime-provided result shape.
#[pliron_op(name="stablehlo.dynamic_reshape",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct DynamicReshapeOp;
impl DynamicReshapeOp {
    /// Creates a `stablehlo.dynamic_reshape` operation with the supplied result type. Uses `input`, `output_shape` as operands.
    pub fn new(ctx: &mut Context, input: Value, output_shape: Value, result: TypeHandle) -> Self {
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![input, output_shape],
                vec![],
                0,
            ),
        }
    }
    /// Reshapes `input` to the runtime dimensions provided by `output_shape`.
    ///
    /// The result rank is derived from the length of the shape tensor and the input element type is
    /// preserved. Returns an error when either operand is not a supported ranked tensor or the
    /// result type cannot be inferred.
    pub fn new_inferred(ctx: &mut Context, input: Value, output_shape: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_dynamic_reshape_type(
                ctx,
                input.get_type(ctx),
                output_shape.get_type(ctx),
            ),
            "stablehlo.dynamic_reshape",
        )?;
        Ok(Self::new(ctx, input, output_shape, result))
    }
}
impl Verify for DynamicReshapeOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() != 2 || ir.get_num_results() != 1 {
            return invalid(self, ctx, "dynamic_reshape arity mismatch");
        }
        let (Some((input, e)), Some((shape, shape_element)), Some((out, oe))) = (
            shape_elem(ctx, ir.get_operand(0).get_type(ctx)),
            shape_elem(ctx, ir.get_operand(1).get_type(ctx)),
            shape_elem(ctx, ir.get_result(0).get_type(ctx)),
        ) else {
            return invalid(self, ctx, "dynamic_reshape requires ranked tensors");
        };
        if e != oe
            || shape.len() != 1
            || shape[0] < 0
            || usize::try_from(shape[0]).ok() != Some(out.len())
            || shape_element
                .deref(ctx)
                .downcast_ref::<IntegerType>()
                .is_none()
            || input.iter().any(|&d| d < -1)
            || out.iter().any(|&d| d < -1)
        {
            return invalid(
                self,
                ctx,
                "dynamic_reshape shape tensor, result rank or element type is invalid",
            );
        }
        let input_count = input.iter().try_fold(1u64, |count, &dimension| {
            u64::try_from(dimension).ok()?.checked_mul(count)
        });
        let output_count = out.iter().try_fold(1u64, |count, &dimension| {
            u64::try_from(dimension).ok()?.checked_mul(count)
        });
        if input_count.zip(output_count).is_some_and(|(a, b)| a != b) {
            return invalid(self, ctx, "dynamic_reshape must preserve element count");
        }
        Ok(())
    }
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
fn signed_integer_list(attr: &VecAttr) -> Option<Vec<i64>> {
    attr.0
        .iter()
        .map(|x| {
            x.downcast_ref::<IntegerAttr>()?
                .value()
                .to_string_decimal(true)
                .parse()
                .ok()
        })
        .collect()
}
fn shape_elem(ctx: &Context, ty: TypeHandle) -> Option<(Vec<i64>, TypeHandle)> {
    let ty = ty.deref(ctx);
    let t = ty.downcast_ref::<TensorType>()?;
    Some((t.dimensions.clone(), t.element))
}
fn compatible_native_shape(ctx: &Context, a: TypeHandle, b: TypeHandle) -> bool {
    crate::verification::shape::compatible_shape(ctx, a, b)
}

/// Produces the runtime extent of one axis of a ranked tensor.
#[pliron_op(name="stablehlo.get_dimension_size",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface],attributes=(stablehlo_get_dimension_size_dimension:IntegerAttr))]
pub struct GetDimensionSizeOp;
impl GetDimensionSizeOp {
    /// Creates a `stablehlo.get_dimension_size` operation with the supplied result type. Uses `input` as its operand. Configures the operation with `dimension`.
    pub fn new(ctx: &mut Context, input: Value, result: TypeHandle, dimension: usize) -> Self {
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
        op.set_attr_stablehlo_get_dimension_size_dimension(
            ctx,
            integer_attr(ctx, dimension as u64),
        );
        op
    }
    /// Reads the runtime size of axis `dimension` in `input`.
    ///
    /// The result is a scalar `i32`. Returns an error if `input` is not a ranked tensor or the axis
    /// is outside its rank.
    pub fn new_inferred(ctx: &mut Context, input: Value, dimension: usize) -> Result<Self> {
        let (s, _) = crate::ops::builder_option(
            shape_elem(ctx, input.get_type(ctx)),
            "stablehlo.get_dimension_size",
            "operand must be a supported ranked tensor",
        )?;
        if dimension >= s.len() {
            return crate::ops::builder_option(
                None,
                "stablehlo.get_dimension_size",
                "builder inputs are invalid",
            );
        }
        Ok(Self::new(
            ctx,
            input,
            TensorType::scalar(ctx, ElementType::I32),
            dimension,
        ))
    }
    /// Returns the operation's dimension attribute when it is present and representable.
    pub fn dimension(&self, ctx: &Context) -> Option<usize> {
        usize::try_from(
            self.get_attr_stablehlo_get_dimension_size_dimension(ctx)?
                .value()
                .to_string_decimal(false)
                .parse::<u64>()
                .ok()?,
        )
        .ok()
    }
}
impl Verify for GetDimensionSizeOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() != 1 || ir.get_num_results() != 1 {
            return invalid(self, ctx, "get_dimension_size arity mismatch");
        }
        let (Some((s, _)), Some(d), Some((out, e))) = (
            shape_elem(ctx, ir.get_operand(0).get_type(ctx)),
            self.dimension(ctx),
            shape_elem(ctx, ir.get_result(0).get_type(ctx)),
        ) else {
            return invalid(
                self,
                ctx,
                "get_dimension_size types or dimension are invalid",
            );
        };
        if d >= s.len() || !out.is_empty() || e != ElementType::I32.native(ctx) {
            return invalid(
                self,
                ctx,
                "get_dimension_size requires an in-range dimension and scalar i32 result",
            );
        }
        Ok(())
    }
}

/// Associates a runtime extent with one axis while preserving the tensor values.
#[pliron_op(name="stablehlo.set_dimension_size",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface],attributes=(stablehlo_set_dimension_size_dimension:IntegerAttr))]
pub struct SetDimensionSizeOp;
impl SetDimensionSizeOp {
    /// Creates a `stablehlo.set_dimension_size` operation with the supplied result type. Uses `input`, `size` as operands. Configures the operation with `dimension`.
    pub fn new(
        ctx: &mut Context,
        input: Value,
        size: Value,
        result: TypeHandle,
        dimension: usize,
    ) -> Self {
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![input, size],
                vec![],
                0,
            ),
        };
        op.set_attr_stablehlo_set_dimension_size_dimension(
            ctx,
            integer_attr(ctx, dimension as u64),
        );
        op
    }
    /// Sets the runtime size of axis `dimension` in `input` to the scalar value in `size`.
    ///
    /// The result retains the input element type and rank, with the selected axis made dynamic
    /// where needed. Returns an error for unsupported operand types, a non-scalar size, or an
    /// out-of-range axis.
    pub fn new_inferred(
        ctx: &mut Context,
        input: Value,
        size: Value,
        dimension: usize,
    ) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_set_dimension_size_type(
                ctx,
                input.get_type(ctx),
                size.get_type(ctx),
                dimension,
            ),
            "stablehlo.set_dimension_size",
        )?;
        Ok(Self::new(ctx, input, size, result, dimension))
    }
    /// Returns the operation's dimension attribute when it is present and representable.
    pub fn dimension(&self, ctx: &Context) -> Option<usize> {
        usize::try_from(
            self.get_attr_stablehlo_set_dimension_size_dimension(ctx)?
                .value()
                .to_string_decimal(false)
                .parse::<u64>()
                .ok()?,
        )
        .ok()
    }
}
impl Verify for SetDimensionSizeOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() != 2 || ir.get_num_results() != 1 {
            return invalid(self, ctx, "set_dimension_size arity mismatch");
        }
        let (Some((a, e)), Some((sz, se)), Some((b, be)), Some(d)) = (
            shape_elem(ctx, ir.get_operand(0).get_type(ctx)),
            shape_elem(ctx, ir.get_operand(1).get_type(ctx)),
            shape_elem(ctx, ir.get_result(0).get_type(ctx)),
            self.dimension(ctx),
        ) else {
            return invalid(
                self,
                ctx,
                "set_dimension_size types or dimension are invalid",
            );
        };
        let expected = crate::inference::infer_set_dimension_size_type(
            ctx,
            ir.get_operand(0).get_type(ctx),
            ir.get_operand(1).get_type(ctx),
            d,
        )
        .ok();
        if d >= a.len()
            || expected
                .is_none_or(|ty| !compatible_native_shape(ctx, ty, ir.get_result(0).get_type(ctx)))
            || !sz.is_empty()
            || se != ElementType::I32.native(ctx)
            || a.len() != b.len()
            || e != be
            || a.iter().enumerate().any(|(i, &x)| i != d && b[i] != x)
            || b[d] < -1
        {
            return invalid(
                self,
                ctx,
                "set_dimension_size result must retain rank and alter only selected dimension",
            );
        }
        Ok(())
    }
}
