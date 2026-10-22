use crate::{attributes::*, types::*};
use pliron::{
    builtin::attributes::IntegerAttr,
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

pliron::dict_key!(
    /// Dictionary key used to store the optional StableHLO result-accuracy attribute.
    ATTR_KEY_STABLEHLO_RESULT_ACCURACY,
    "stablehlo_result_accuracy"
);

macro_rules! impl_result_accuracy_accessors {
    ($($op:ty),+ $(,)?) => {
        $(
            impl $op {
                /// Returns the optional result-accuracy attribute when present.
                pub fn result_accuracy<'a>(
                    &self,
                    ctx: &'a Context,
                ) -> Option<core::cell::Ref<'a, ResultAccuracyAttr>> {
                    core::cell::Ref::filter_map(self.get_operation().deref(ctx), |op| {
                        op.attributes
                            .get::<ResultAccuracyAttr>(&ATTR_KEY_STABLEHLO_RESULT_ACCURACY)
                    })
                    .ok()
                }

                /// Sets the result-accuracy attribute on this operation.
                pub fn set_result_accuracy(
                    &self,
                    ctx: &Context,
                    value: ResultAccuracyAttr,
                ) {
                    self.get_operation()
                        .deref_mut(ctx)
                        .attributes
                        .set(ATTR_KEY_STABLEHLO_RESULT_ACCURACY.clone(), value);
                }
            }
        )+
    };
}

/// Converts the input elements to the specified result element type.
#[pliron_op(name="stablehlo.convert",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct ConvertOp;
impl ConvertOp {
    /// Creates a `stablehlo.convert` operation with the supplied result type. Uses `input` as its operand.
    #[inline]
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
    /// Returns the input operand.
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    /// Converts `input` to `to`, preserving its shape and tensor encoding.
    ///
    /// The source and target must be supported primitive element types; index, `tf32`, and
    /// signless integer element types are not supported. Returns an error if type inference fails.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value, to: ElementType) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_convert_type(ctx, input.get_type(ctx), to),
            "stablehlo.convert",
        )?;
        Ok(Self::new(ctx, input, result))
    }
}
impl Verify for ConvertOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_convert(self, ctx)
    }
}

/// Selects elements from the true or false operand according to the predicate.
#[pliron_op(name="stablehlo.select",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct SelectOp;
impl SelectOp {
    /// Returns the predicate operand.
    pub fn predicate(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    /// Returns the value selected when the predicate is true.
    pub fn on_true(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
    /// Returns the value selected when the predicate is false.
    pub fn on_false(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(2)
    }
    /// Selects the corresponding element from `yes` or `no` using the `i1` predicate in `pred`.
    ///
    /// The value operands must have compatible element types and shapes. `pred` must contain `i1`
    /// elements; a scalar predicate is allowed, otherwise its shape must also be compatible with
    /// the value operands. Returns an error when the element types or shapes cannot be reconciled.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, pred: Value, yes: Value, no: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_select_type(
                ctx,
                pred.get_type(ctx),
                yes.get_type(ctx),
                no.get_type(ctx),
            ),
            "stablehlo.select",
        )?;
        Ok(Self::new(ctx, pred, yes, no, result))
    }
    /// Creates a `stablehlo.select` operation with the supplied result type. Uses `pred`, `yes`, `no` as operands.
    #[inline]
    pub fn new(ctx: &mut Context, pred: Value, yes: Value, no: Value, result: TypeHandle) -> Self {
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![pred, yes, no],
                vec![],
                0,
            ),
        }
    }
}
impl Verify for SelectOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_select(self, ctx)
    }
}

/// Compares tensor elements and produces a predicate tensor.
#[pliron_op(name="stablehlo.compare",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface],attributes=(stablehlo_compare_direction:ComparisonDirection,stablehlo_compare_type:ComparisonKind))]
pub struct CompareOp;
impl CompareOp {
    /// Returns the left-hand operand.
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    /// Returns the right-hand operand.
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
    /// Compares corresponding elements of `lhs` and `rhs` using `direction`.
    ///
    /// The result has predicate elements and a shape compatible with both operands (dynamic
    /// dimensions may match known extents). `comparison` selects comparison semantics when
    /// present; valid element domains depend on that kind. Per-axis quantized operands are not
    /// supported. Returns an error for unsupported element types or incompatible shapes.
    #[inline]
    pub fn new_inferred(
        ctx: &mut Context,
        lhs: Value,
        rhs: Value,
        direction: ComparisonDirection,
        comparison: Option<ComparisonKind>,
    ) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_compare_type(
                ctx,
                lhs.get_type(ctx),
                rhs.get_type(ctx),
                comparison,
            ),
            "stablehlo.compare",
        )?;
        Ok(if let Some(comparison) = comparison {
            Self::new(ctx, lhs, rhs, result, direction, comparison)
        } else {
            Self::new_without_comparison_kind(ctx, lhs, rhs, result, direction)
        })
    }
    /// Creates a comparison with the supplied result type and direction, omitting the optional comparison-kind attribute.
    #[inline]
    pub fn new_without_comparison_kind(
        ctx: &mut Context,
        lhs: Value,
        rhs: Value,
        result: TypeHandle,
        direction: ComparisonDirection,
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
        op.set_attr_stablehlo_compare_direction(ctx, direction);
        op
    }
    /// Creates a `stablehlo.compare` operation with the supplied result type. Uses `lhs`, `rhs` as operands. Configures the operation with `direction`, `comparison`.
    #[inline]
    pub fn new(
        ctx: &mut Context,
        lhs: Value,
        rhs: Value,
        result: TypeHandle,
        direction: ComparisonDirection,
        comparison: ComparisonKind,
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
        op.set_attr_stablehlo_compare_direction(ctx, direction);
        op.set_attr_stablehlo_compare_type(ctx, comparison);
        op
    }
    /// Return the comparison direction, if the attribute is present and valid.
    pub fn direction(&self, ctx: &Context) -> Option<ComparisonDirection> {
        self.get_attr_stablehlo_compare_direction(ctx)
            .map(|direction| *direction)
    }
    /// Return the explicit comparison kind; `None` when omitted or invalid.
    pub fn comparison_kind(&self, ctx: &Context) -> Option<ComparisonKind> {
        self.get_attr_stablehlo_compare_type(ctx)
            .map(|comparison| *comparison)
    }
}
impl Verify for CompareOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_compare(
            self,
            ctx,
            self.get_attr_stablehlo_compare_direction(ctx)
                .map(|a| *a as u64),
            self.get_attr_stablehlo_compare_type(ctx).map(|a| *a as u64),
        )
    }
}

/// Clamps each value element between the corresponding minimum and maximum elements.
#[pliron_op(name="stablehlo.clamp",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct ClampOp;
impl ClampOp {
    /// Returns the minimum-bound operand.
    pub fn min(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    /// Returns the value operand.
    pub fn value(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
    /// Returns the maximum-bound operand.
    pub fn max(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(2)
    }
    /// Clamps corresponding elements of `operand` between `min` and `max`.
    ///
    /// Bounds may be scalars or tensors with shapes compatible with `value`; their element types
    /// must be compatible with the value element type. Returns an error if these types or shapes
    /// cannot be reconciled.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, min: Value, value: Value, max: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_clamp_type(
                ctx,
                min.get_type(ctx),
                value.get_type(ctx),
                max.get_type(ctx),
            ),
            "stablehlo.clamp",
        )?;
        Ok(Self::new(ctx, min, value, max, result))
    }
    /// Creates a `stablehlo.clamp` operation with the supplied result type. Uses `min`, `value`, `max` as operands.
    #[inline]
    pub fn new(
        ctx: &mut Context,
        min: Value,
        value: Value,
        max: Value,
        result: TypeHandle,
    ) -> Self {
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![min, value, max],
                vec![],
                0,
            ),
        }
    }
}
impl Verify for ClampOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_clamp(self, ctx)
    }
}

/// Produces the coordinates of a ranked tensor along one selected axis.
#[pliron_op(name="stablehlo.iota",format,interfaces=[pliron::builtin::op_interfaces::NOpdsInterface<0>,pliron::builtin::op_interfaces::OneResultInterface],attributes=(stablehlo_iota_dimension:IntegerAttr))]
pub struct IotaOp;
impl IotaOp {
    /// Returns the operation's dimension attribute when it is present and representable.
    pub fn dimension(&self, ctx: &Context) -> Option<usize> {
        self.get_attr_stablehlo_iota_dimension(ctx)
            .and_then(|a| a.value().to_string_decimal(false).parse::<u64>().ok())
            .and_then(|v| usize::try_from(v).ok())
    }
    /// Creates a `stablehlo.iota` operation with the supplied result type. Configures the operation with `dimension`.
    #[inline]
    pub fn new(ctx: &mut Context, result: TypeHandle, dimension: usize) -> Self {
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![],
                vec![],
                0,
            ),
        };
        let d = integer_attr(ctx, dimension as u64);
        op.set_attr_stablehlo_iota_dimension(ctx, d);
        op
    }
}
impl Verify for IotaOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let dim = self
            .get_attr_stablehlo_iota_dimension(ctx)
            .and_then(|a| a.value().to_string_decimal(false).parse::<u64>().ok())
            .and_then(|v| usize::try_from(v).ok());
        crate::verification::elementwise::verify_iota(self, ctx, dim, false)
    }
}

/// Computes the elementwise absolute value of its input.
#[pliron_op(name="stablehlo.abs",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct AbsOp;
impl AbsOp {
    /// Creates a `stablehlo.abs` operation with the supplied result type. Uses `input` as its operand.
    #[inline]
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
    /// Infers the result type for element-wise absolute value of `operand`.
    ///
    /// Returns an error when the operand type is unsupported.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "abs", &[input.get_type(ctx)]),
            "stablehlo.abs",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    /// Returns the input operand.
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for AbsOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "abs", 1)
    }
}

/// Computes the elementwise cube root of its input.
#[pliron_op(name="stablehlo.cbrt",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct CbrtOp;
impl CbrtOp {
    /// Creates a `stablehlo.cbrt` operation with the supplied result type. Uses `input` as its operand.
    #[inline]
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
    /// Infers the result type for the element-wise cube root of `operand`.
    ///
    /// Returns an error when the operand type is unsupported.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "cbrt", &[input.get_type(ctx)]),
            "stablehlo.cbrt",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    /// Returns the input operand.
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for CbrtOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "cbrt", 1)
    }
}

/// Rounds each input element toward positive infinity.
#[pliron_op(name="stablehlo.ceil",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct CeilOp;
impl CeilOp {
    /// Creates a `stablehlo.ceil` operation with the supplied result type. Uses `input` as its operand.
    #[inline]
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
    /// Infers the result type for element-wise rounding of `operand` toward positive infinity.
    ///
    /// Returns an error when the operand type is unsupported.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "ceil", &[input.get_type(ctx)]),
            "stablehlo.ceil",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    /// Returns the input operand.
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for CeilOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "ceil", 1)
    }
}

/// Counts the leading zero bits in each input element.
#[pliron_op(name="stablehlo.count_leading_zeros",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct CountLeadingZerosOp;
impl CountLeadingZerosOp {
    /// Creates a `stablehlo.count_leading_zeros` operation with the supplied result type. Uses `input` as its operand.
    #[inline]
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
    /// Infers the result type for counting leading zero bits in each integer element of `operand`.
    ///
    /// Returns an error unless the operand has a supported integer element type.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "count_leading_zeros", &[input.get_type(ctx)]),
            "stablehlo.count_leading_zeros",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    /// Returns the input operand.
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for CountLeadingZerosOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "count_leading_zeros", 1)
    }
}

/// Computes the elementwise cosine of its input.
#[pliron_op(name="stablehlo.cosine",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct CosineOp;
impl CosineOp {
    /// Creates a `stablehlo.cosine` operation with the supplied result type. Uses `input` as its operand.
    #[inline]
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
    /// Infers the result type for the element-wise cosine of `operand`.
    ///
    /// Returns an error when the operand type is unsupported.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "cosine", &[input.get_type(ctx)]),
            "stablehlo.cosine",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    /// Returns the input operand.
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for CosineOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "cosine", 1)
    }
}

/// Computes the elementwise natural exponential of its input.
#[pliron_op(name="stablehlo.exponential",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct ExponentialOp;
impl ExponentialOp {
    /// Creates a `stablehlo.exponential` operation with the supplied result type. Uses `input` as its operand.
    #[inline]
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
    /// Infers the result type for the element-wise natural exponential of `operand`.
    ///
    /// Returns an error when the operand type is unsupported.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "exponential", &[input.get_type(ctx)]),
            "stablehlo.exponential",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    /// Returns the input operand.
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for ExponentialOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "exponential", 1)
    }
}

/// Computes the elementwise natural exponential of its input minus one.
#[pliron_op(name="stablehlo.exponential_minus_one",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct ExponentialMinusOneOp;
impl ExponentialMinusOneOp {
    /// Creates a `stablehlo.exponential_minus_one` operation with the supplied result type. Uses `input` as its operand.
    #[inline]
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
    /// Infers the result type for computing `exp(x) - 1` element-wise over `operand`.
    ///
    /// Returns an error when the operand type is unsupported.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(
                ctx,
                "exponential_minus_one",
                &[input.get_type(ctx)],
            ),
            "stablehlo.exponential_minus_one",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    /// Returns the input operand.
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for ExponentialMinusOneOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "exponential_minus_one", 1)
    }
}

/// Raises two to the power of each input element.
#[pliron_op(name="stablehlo.exp2",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct Exp2Op;
impl Exp2Op {
    /// Creates a `stablehlo.exp2` operation with the supplied result type. Uses `input` as its operand.
    #[inline]
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
    /// Infers the result type for computing `2^x` element-wise over `operand`.
    ///
    /// Returns an error when the operand type is unsupported.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "exp2", &[input.get_type(ctx)]),
            "stablehlo.exp2",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    /// Returns the input operand.
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for Exp2Op {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "exp2", 1)
    }
}

/// Rounds each input element toward negative infinity.
#[pliron_op(name="stablehlo.floor",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct FloorOp;
impl FloorOp {
    /// Creates a `stablehlo.floor` operation with the supplied result type. Uses `input` as its operand.
    #[inline]
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
    /// Infers the result type for element-wise rounding of `operand` toward negative infinity.
    ///
    /// Returns an error when the operand type is unsupported.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "floor", &[input.get_type(ctx)]),
            "stablehlo.floor",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    /// Returns the input operand.
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for FloorOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "floor", 1)
    }
}

/// Extracts the imaginary component of each complex input element.
#[pliron_op(name="stablehlo.imag",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct ImagOp;
impl ImagOp {
    /// Creates a `stablehlo.imag` operation with the supplied result type. Uses `input` as its operand.
    #[inline]
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
    /// Infers the result type containing the imaginary component of each element in `operand`.
    ///
    /// Returns an error when the operand does not have a supported complex element type.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "imag", &[input.get_type(ctx)]),
            "stablehlo.imag",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    /// Returns the input operand.
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for ImagOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "imag", 1)
    }
}

/// Tests whether each input element is finite.
#[pliron_op(name="stablehlo.is_finite",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct IsFiniteOp;
impl IsFiniteOp {
    /// Creates a `stablehlo.is_finite` operation with the supplied result type. Uses `input` as its operand.
    #[inline]
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
    /// Infers a predicate result type for testing each `operand` element for finiteness.
    ///
    /// The result preserves the operand shape. Returns an error when the element type is unsupported.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "is_finite", &[input.get_type(ctx)]),
            "stablehlo.is_finite",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    /// Returns the input operand.
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for IsFiniteOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "is_finite", 1)
    }
}

/// Computes the elementwise natural logarithm of its input.
#[pliron_op(name="stablehlo.log",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct LogOp;
impl LogOp {
    /// Creates a `stablehlo.log` operation with the supplied result type. Uses `input` as its operand.
    #[inline]
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
    /// Infers the result type for the element-wise natural logarithm of `operand`.
    ///
    /// Returns an error when the operand type is unsupported.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "log", &[input.get_type(ctx)]),
            "stablehlo.log",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    /// Returns the input operand.
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for LogOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "log", 1)
    }
}

/// Computes the elementwise natural logarithm of one plus its input.
#[pliron_op(name="stablehlo.log_plus_one",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct LogPlusOneOp;
impl LogPlusOneOp {
    /// Creates a `stablehlo.log_plus_one` operation with the supplied result type. Uses `input` as its operand.
    #[inline]
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
    /// Infers the result type for computing `log(1 + x)` element-wise over `operand`.
    ///
    /// Returns an error when the operand type is unsupported.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "log_plus_one", &[input.get_type(ctx)]),
            "stablehlo.log_plus_one",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    /// Returns the input operand.
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for LogPlusOneOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "log_plus_one", 1)
    }
}

/// Computes the elementwise base-two logarithm of its input.
#[pliron_op(name="stablehlo.log2",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct Log2Op;
impl Log2Op {
    /// Creates a `stablehlo.log2` operation with the supplied result type. Uses `input` as its operand.
    #[inline]
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
    /// Infers the result type for the element-wise base-two logarithm of `operand`.
    ///
    /// Returns an error when the operand type is unsupported.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "log2", &[input.get_type(ctx)]),
            "stablehlo.log2",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    /// Returns the input operand.
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for Log2Op {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "log2", 1)
    }
}

/// Computes the elementwise logistic (sigmoid) function.
#[pliron_op(name="stablehlo.logistic",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct LogisticOp;
impl LogisticOp {
    /// Creates a `stablehlo.logistic` operation with the supplied result type. Uses `input` as its operand.
    #[inline]
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
    /// Infers the result type for the element-wise logistic function `1 / (1 + exp(-x))`.
    ///
    /// Returns an error when the operand type is unsupported.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "logistic", &[input.get_type(ctx)]),
            "stablehlo.logistic",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    /// Returns the input operand.
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for LogisticOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "logistic", 1)
    }
}

/// Applies elementwise logical or bitwise negation.
#[pliron_op(name="stablehlo.not",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct NotOp;
impl NotOp {
    /// Creates a `stablehlo.not` operation with the supplied result type. Uses `input` as its operand.
    #[inline]
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
    /// Infers the result type for the element-wise logical or bitwise complement of `operand`.
    ///
    /// Returns an error when the operand type is unsupported.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "not", &[input.get_type(ctx)]),
            "stablehlo.not",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    /// Returns the input operand.
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for NotOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "not", 1)
    }
}

/// Negates each input element.
#[pliron_op(name="stablehlo.negate",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct NegateOp;
impl NegateOp {
    /// Creates a `stablehlo.negate` operation with the supplied result type. Uses `input` as its operand.
    #[inline]
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
    /// Infers the result type for negating each element of `operand`.
    ///
    /// Returns an error when the operand type is unsupported.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "negate", &[input.get_type(ctx)]),
            "stablehlo.negate",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    /// Returns the input operand.
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for NegateOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "negate", 1)
    }
}

/// Counts the set bits in each input element.
#[pliron_op(name="stablehlo.popcnt",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct PopulationCountOp;
impl PopulationCountOp {
    /// Creates a `stablehlo.popcnt` operation with the supplied result type. Uses `input` as its operand.
    #[inline]
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
    /// Infers the result type for counting set bits in each integer element of `operand`.
    ///
    /// Returns an error unless the operand has a supported integer element type.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "popcnt", &[input.get_type(ctx)]),
            "stablehlo.popcnt",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    /// Returns the input operand.
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for PopulationCountOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "popcnt", 1)
    }
}

/// Extracts the real component of each complex input element.
#[pliron_op(name="stablehlo.real",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct RealOp;
impl RealOp {
    /// Creates a `stablehlo.real` operation with the supplied result type. Uses `input` as its operand.
    #[inline]
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
    /// Infers the result type containing the real component of each element in `operand`.
    ///
    /// Returns an error when the operand does not have a supported complex element type.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "real", &[input.get_type(ctx)]),
            "stablehlo.real",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    /// Returns the input operand.
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for RealOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "real", 1)
    }
}

/// Rounds each input element to the nearest integer, with ties away from zero.
#[pliron_op(name="stablehlo.round_nearest_afz",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct RoundNearestAfzOp;
impl RoundNearestAfzOp {
    /// Creates a `stablehlo.round_nearest_afz` operation with the supplied result type. Uses `input` as its operand.
    #[inline]
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
    /// Infers the result type for rounding `operand` to the nearest integer, with ties away from zero.
    ///
    /// Returns an error when the operand type is unsupported.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "round_nearest_afz", &[input.get_type(ctx)]),
            "stablehlo.round_nearest_afz",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    /// Returns the input operand.
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for RoundNearestAfzOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "round_nearest_afz", 1)
    }
}

/// Rounds each input element to the nearest integer, with ties to even.
#[pliron_op(name="stablehlo.round_nearest_even",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct RoundNearestEvenOp;
impl RoundNearestEvenOp {
    /// Creates a `stablehlo.round_nearest_even` operation with the supplied result type. Uses `input` as its operand.
    #[inline]
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
    /// Infers the result type for rounding `operand` to the nearest integer, with ties to even.
    ///
    /// Returns an error when the operand type is unsupported.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "round_nearest_even", &[input.get_type(ctx)]),
            "stablehlo.round_nearest_even",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    /// Returns the input operand.
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for RoundNearestEvenOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "round_nearest_even", 1)
    }
}

/// Computes the elementwise reciprocal square root of its input.
#[pliron_op(name="stablehlo.rsqrt",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct RsqrtOp;
impl RsqrtOp {
    /// Creates a `stablehlo.rsqrt` operation with the supplied result type. Uses `input` as its operand.
    #[inline]
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
    /// Infers the result type for the element-wise reciprocal square root of `operand`.
    ///
    /// Returns an error when the operand type is unsupported.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "rsqrt", &[input.get_type(ctx)]),
            "stablehlo.rsqrt",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    /// Returns the input operand.
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for RsqrtOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "rsqrt", 1)
    }
}

/// Returns the sign of each input element.
#[pliron_op(name="stablehlo.sign",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct SignOp;
impl SignOp {
    /// Creates a `stablehlo.sign` operation with the supplied result type. Uses `input` as its operand.
    #[inline]
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
    /// Infers the result type for the element-wise sign of `operand`.
    ///
    /// Returns an error when the operand type is unsupported.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "sign", &[input.get_type(ctx)]),
            "stablehlo.sign",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    /// Returns the input operand.
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for SignOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "sign", 1)
    }
}

/// Computes the elementwise sine of its input.
#[pliron_op(name="stablehlo.sine",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct SineOp;
impl SineOp {
    /// Creates a `stablehlo.sine` operation with the supplied result type. Uses `input` as its operand.
    #[inline]
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
    /// Infers the result type for the element-wise sine of `operand`.
    ///
    /// Returns an error when the operand type is unsupported.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "sine", &[input.get_type(ctx)]),
            "stablehlo.sine",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    /// Returns the input operand.
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for SineOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "sine", 1)
    }
}

/// Computes the elementwise square root of its input.
#[pliron_op(name="stablehlo.sqrt",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct SqrtOp;
impl SqrtOp {
    /// Creates a `stablehlo.sqrt` operation with the supplied result type. Uses `input` as its operand.
    #[inline]
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
    /// Infers the result type for the element-wise square root of `operand`.
    ///
    /// Returns an error when the operand type is unsupported.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "sqrt", &[input.get_type(ctx)]),
            "stablehlo.sqrt",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    /// Returns the input operand.
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for SqrtOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "sqrt", 1)
    }
}

/// Computes the elementwise tangent of its input.
#[pliron_op(name="stablehlo.tan",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct TanOp;
impl TanOp {
    /// Creates a `stablehlo.tan` operation with the supplied result type. Uses `input` as its operand.
    #[inline]
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
    /// Infers the result type for the element-wise tangent of `operand`.
    ///
    /// Returns an error when the operand type is unsupported.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "tan", &[input.get_type(ctx)]),
            "stablehlo.tan",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    /// Returns the input operand.
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for TanOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "tan", 1)
    }
}

/// Computes the elementwise hyperbolic tangent of its input.
#[pliron_op(name="stablehlo.tanh",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct TanhOp;
impl TanhOp {
    /// Creates a `stablehlo.tanh` operation with the supplied result type. Uses `input` as its operand.
    #[inline]
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
    /// Infers the result type for the element-wise hyperbolic tangent of `operand`.
    ///
    /// Returns an error when the operand type is unsupported.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "tanh", &[input.get_type(ctx)]),
            "stablehlo.tanh",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    /// Returns the input operand.
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for TanhOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "tanh", 1)
    }
}

/// Adds corresponding elements of two operands.
#[pliron_op(name="stablehlo.add",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct AddOp;
impl AddOp {
    /// Creates a `stablehlo.add` operation with the supplied result type. Uses `lhs`, `rhs` as operands.
    #[inline]
    pub fn new(ctx: &mut Context, lhs: Value, rhs: Value, result: TypeHandle) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);
        Operation::push_result(op, ctx, result);
        Operation::push_operand(op, ctx, lhs);
        Operation::push_operand(op, ctx, rhs);
        Self { op }
    }
    /// Adds corresponding elements of `lhs` and `rhs`, inferring a result from compatible tensor types.
    ///
    /// Returns an error when element types or shapes are incompatible.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, lhs: Value, rhs: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(
                ctx,
                "add",
                &[lhs.get_type(ctx), rhs.get_type(ctx)],
            ),
            "stablehlo.add",
        )?;
        Ok(Self::new(ctx, lhs, rhs, result))
    }
    /// Returns the left-hand operand.
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    /// Returns the right-hand operand.
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}
impl Verify for AddOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "add", 2)
    }
}

/// Computes the elementwise two-argument arctangent of its operands.
#[pliron_op(name="stablehlo.atan2",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct Atan2Op;
impl Atan2Op {
    /// Creates a `stablehlo.atan2` operation with the supplied result type. Uses `lhs`, `rhs` as operands.
    #[inline]
    pub fn new(ctx: &mut Context, lhs: Value, rhs: Value, result: TypeHandle) -> Self {
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![lhs, rhs],
                vec![],
                0,
            ),
        }
    }
    /// Infers the result type for the element-wise two-argument arctangent of `lhs` and `rhs`.
    ///
    /// Returns an error when operand element types or shapes are incompatible.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, lhs: Value, rhs: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(
                ctx,
                "atan2",
                &[lhs.get_type(ctx), rhs.get_type(ctx)],
            ),
            "stablehlo.atan2",
        )?;
        Ok(Self::new(ctx, lhs, rhs, result))
    }
    /// Returns the left-hand operand.
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    /// Returns the right-hand operand.
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}
impl Verify for Atan2Op {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "atan2", 2)
    }
}

/// Combines real and imaginary operands into complex values.
#[pliron_op(name="stablehlo.complex",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct ComplexOp;
impl ComplexOp {
    /// Creates a `stablehlo.complex` operation with the supplied result type. Uses `lhs`, `rhs` as operands.
    #[inline]
    pub fn new(ctx: &mut Context, lhs: Value, rhs: Value, result: TypeHandle) -> Self {
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![lhs, rhs],
                vec![],
                0,
            ),
        }
    }
    /// Forms complex elements from the real values in `lhs` and imaginary values in `rhs`.
    ///
    /// Operand types and compatible shapes determine the result complex tensor type. Returns an
    /// error when they cannot be reconciled.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, lhs: Value, rhs: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(
                ctx,
                "complex",
                &[lhs.get_type(ctx), rhs.get_type(ctx)],
            ),
            "stablehlo.complex",
        )?;
        Ok(Self::new(ctx, lhs, rhs, result))
    }
    /// Returns the left-hand operand.
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    /// Returns the right-hand operand.
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}
impl Verify for ComplexOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "complex", 2)
    }
}

/// Divides corresponding elements of the left operand by the right operand.
#[pliron_op(name="stablehlo.divide",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct DivideOp;
impl DivideOp {
    /// Creates a `stablehlo.divide` operation with the supplied result type. Uses `lhs`, `rhs` as operands.
    #[inline]
    pub fn new(ctx: &mut Context, lhs: Value, rhs: Value, result: TypeHandle) -> Self {
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![lhs, rhs],
                vec![],
                0,
            ),
        }
    }
    /// Divides corresponding elements of `lhs` by `rhs`, inferring a result from compatible tensor types.
    ///
    /// Returns an error when element types or shapes are incompatible.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, lhs: Value, rhs: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(
                ctx,
                "divide",
                &[lhs.get_type(ctx), rhs.get_type(ctx)],
            ),
            "stablehlo.divide",
        )?;
        Ok(Self::new(ctx, lhs, rhs, result))
    }
    /// Returns the left-hand operand.
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    /// Returns the right-hand operand.
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}
impl Verify for DivideOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "divide", 2)
    }
}

/// Selects the elementwise maximum of two operands.
#[pliron_op(name="stablehlo.maximum",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct MaximumOp;
impl MaximumOp {
    /// Creates a `stablehlo.maximum` operation with the supplied result type. Uses `lhs`, `rhs` as operands.
    #[inline]
    pub fn new(ctx: &mut Context, lhs: Value, rhs: Value, result: TypeHandle) -> Self {
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![lhs, rhs],
                vec![],
                0,
            ),
        }
    }
    /// Selects the greater of each pair of corresponding elements in `lhs` and `rhs`.
    ///
    /// Returns an error when operand element types or shapes are incompatible.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, lhs: Value, rhs: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(
                ctx,
                "maximum",
                &[lhs.get_type(ctx), rhs.get_type(ctx)],
            ),
            "stablehlo.maximum",
        )?;
        Ok(Self::new(ctx, lhs, rhs, result))
    }
    /// Returns the left-hand operand.
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    /// Returns the right-hand operand.
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}
impl Verify for MaximumOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "maximum", 2)
    }
}

/// Selects the elementwise minimum of two operands.
#[pliron_op(name="stablehlo.minimum",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct MinimumOp;
impl MinimumOp {
    /// Creates a `stablehlo.minimum` operation with the supplied result type. Uses `lhs`, `rhs` as operands.
    #[inline]
    pub fn new(ctx: &mut Context, lhs: Value, rhs: Value, result: TypeHandle) -> Self {
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![lhs, rhs],
                vec![],
                0,
            ),
        }
    }
    /// Selects the lesser of each pair of corresponding elements in `lhs` and `rhs`.
    ///
    /// Returns an error when operand element types or shapes are incompatible.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, lhs: Value, rhs: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(
                ctx,
                "minimum",
                &[lhs.get_type(ctx), rhs.get_type(ctx)],
            ),
            "stablehlo.minimum",
        )?;
        Ok(Self::new(ctx, lhs, rhs, result))
    }
    /// Returns the left-hand operand.
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    /// Returns the right-hand operand.
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}
impl Verify for MinimumOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "minimum", 2)
    }
}

/// Multiplies corresponding elements of two operands.
#[pliron_op(name="stablehlo.multiply",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct MultiplyOp;
impl MultiplyOp {
    /// Creates a `stablehlo.multiply` operation with the supplied result type. Uses `lhs`, `rhs` as operands.
    #[inline]
    pub fn new(ctx: &mut Context, lhs: Value, rhs: Value, result: TypeHandle) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);
        Operation::push_result(op, ctx, result);
        Operation::push_operand(op, ctx, lhs);
        Operation::push_operand(op, ctx, rhs);
        Self { op }
    }
    /// Multiplies corresponding elements of `lhs` and `rhs`, inferring a result from compatible tensor types.
    ///
    /// Returns an error when element types or shapes are incompatible.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, lhs: Value, rhs: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(
                ctx,
                "multiply",
                &[lhs.get_type(ctx), rhs.get_type(ctx)],
            ),
            "stablehlo.multiply",
        )?;
        Ok(Self::new(ctx, lhs, rhs, result))
    }
    /// Returns the left-hand operand.
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    /// Returns the right-hand operand.
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}
impl Verify for MultiplyOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "multiply", 2)
    }
}

/// Raises each element of the left operand to the corresponding power in the right operand.
#[pliron_op(name="stablehlo.power",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct PowerOp;
impl PowerOp {
    /// Creates a `stablehlo.power` operation with the supplied result type. Uses `lhs`, `rhs` as operands.
    #[inline]
    pub fn new(ctx: &mut Context, lhs: Value, rhs: Value, result: TypeHandle) -> Self {
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![lhs, rhs],
                vec![],
                0,
            ),
        }
    }
    /// Raises each element of `lhs` to the corresponding power in `rhs`.
    ///
    /// Returns an error when operand element types or shapes are incompatible.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, lhs: Value, rhs: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(
                ctx,
                "power",
                &[lhs.get_type(ctx), rhs.get_type(ctx)],
            ),
            "stablehlo.power",
        )?;
        Ok(Self::new(ctx, lhs, rhs, result))
    }
    /// Returns the left-hand operand.
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    /// Returns the right-hand operand.
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}
impl Verify for PowerOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "power", 2)
    }
}

/// Computes the elementwise remainder of its operands.
#[pliron_op(name="stablehlo.remainder",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct RemainderOp;
impl RemainderOp {
    /// Creates a `stablehlo.remainder` operation with the supplied result type. Uses `lhs`, `rhs` as operands.
    #[inline]
    pub fn new(ctx: &mut Context, lhs: Value, rhs: Value, result: TypeHandle) -> Self {
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![lhs, rhs],
                vec![],
                0,
            ),
        }
    }
    /// Computes the element-wise remainder of `lhs` divided by `rhs`.
    ///
    /// Returns an error when operand element types or shapes are incompatible.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, lhs: Value, rhs: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(
                ctx,
                "remainder",
                &[lhs.get_type(ctx), rhs.get_type(ctx)],
            ),
            "stablehlo.remainder",
        )?;
        Ok(Self::new(ctx, lhs, rhs, result))
    }
    /// Returns the left-hand operand.
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    /// Returns the right-hand operand.
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}
impl Verify for RemainderOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "remainder", 2)
    }
}

/// Shifts each left operand element left by the corresponding right operand amount.
#[pliron_op(name="stablehlo.shift_left",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct ShiftLeftOp;
impl ShiftLeftOp {
    /// Creates a `stablehlo.shift_left` operation with the supplied result type. Uses `lhs`, `rhs` as operands.
    #[inline]
    pub fn new(ctx: &mut Context, lhs: Value, rhs: Value, result: TypeHandle) -> Self {
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![lhs, rhs],
                vec![],
                0,
            ),
        }
    }
    /// Shifts each integer element of `lhs` left by the corresponding amount in `rhs`.
    ///
    /// Returns an error when operand integer types or shapes are incompatible.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, lhs: Value, rhs: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(
                ctx,
                "shift_left",
                &[lhs.get_type(ctx), rhs.get_type(ctx)],
            ),
            "stablehlo.shift_left",
        )?;
        Ok(Self::new(ctx, lhs, rhs, result))
    }
    /// Returns the left-hand operand.
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    /// Returns the right-hand operand.
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}
impl Verify for ShiftLeftOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "shift_left", 2)
    }
}

/// Performs an arithmetic right shift by the corresponding shift amounts.
#[pliron_op(name="stablehlo.shift_right_arithmetic",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct ShiftRightArithmeticOp;
impl ShiftRightArithmeticOp {
    /// Creates a `stablehlo.shift_right_arithmetic` operation with the supplied result type. Uses `lhs`, `rhs` as operands.
    #[inline]
    pub fn new(ctx: &mut Context, lhs: Value, rhs: Value, result: TypeHandle) -> Self {
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![lhs, rhs],
                vec![],
                0,
            ),
        }
    }
    /// Arithmetically shifts each integer element of `lhs` right by the corresponding amount in `rhs`.
    ///
    /// Returns an error when operand integer types or shapes are incompatible.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, lhs: Value, rhs: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(
                ctx,
                "shift_right_arithmetic",
                &[lhs.get_type(ctx), rhs.get_type(ctx)],
            ),
            "stablehlo.shift_right_arithmetic",
        )?;
        Ok(Self::new(ctx, lhs, rhs, result))
    }
    /// Returns the left-hand operand.
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    /// Returns the right-hand operand.
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}
impl Verify for ShiftRightArithmeticOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "shift_right_arithmetic", 2)
    }
}

/// Performs a logical right shift by the corresponding shift amounts.
#[pliron_op(name="stablehlo.shift_right_logical",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct ShiftRightLogicalOp;
impl ShiftRightLogicalOp {
    /// Creates a `stablehlo.shift_right_logical` operation with the supplied result type. Uses `lhs`, `rhs` as operands.
    #[inline]
    pub fn new(ctx: &mut Context, lhs: Value, rhs: Value, result: TypeHandle) -> Self {
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![lhs, rhs],
                vec![],
                0,
            ),
        }
    }
    /// Logically shifts each integer element of `lhs` right by the corresponding amount in `rhs`.
    ///
    /// Returns an error when operand integer types or shapes are incompatible.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, lhs: Value, rhs: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(
                ctx,
                "shift_right_logical",
                &[lhs.get_type(ctx), rhs.get_type(ctx)],
            ),
            "stablehlo.shift_right_logical",
        )?;
        Ok(Self::new(ctx, lhs, rhs, result))
    }
    /// Returns the left-hand operand.
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    /// Returns the right-hand operand.
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}
impl Verify for ShiftRightLogicalOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "shift_right_logical", 2)
    }
}

/// Subtracts the right operand from the left operand elementwise.
#[pliron_op(name="stablehlo.subtract",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct SubtractOp;
impl SubtractOp {
    /// Creates a `stablehlo.subtract` operation with the supplied result type. Uses `lhs`, `rhs` as operands.
    #[inline]
    pub fn new(ctx: &mut Context, lhs: Value, rhs: Value, result: TypeHandle) -> Self {
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![lhs, rhs],
                vec![],
                0,
            ),
        }
    }
    /// Subtracts corresponding elements of `rhs` from `lhs`, inferring a result from compatible tensor types.
    ///
    /// Returns an error when element types or shapes are incompatible.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, lhs: Value, rhs: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(
                ctx,
                "subtract",
                &[lhs.get_type(ctx), rhs.get_type(ctx)],
            ),
            "stablehlo.subtract",
        )?;
        Ok(Self::new(ctx, lhs, rhs, result))
    }
    /// Returns the left-hand operand.
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    /// Returns the right-hand operand.
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}
impl Verify for SubtractOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "subtract", 2)
    }
}

/// Applies elementwise logical or bitwise conjunction.
#[pliron_op(name="stablehlo.and",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct AndOp;
impl AndOp {
    /// Creates a `stablehlo.and` operation with the supplied result type. Uses `lhs`, `rhs` as operands.
    #[inline]
    pub fn new(ctx: &mut Context, lhs: Value, rhs: Value, result: TypeHandle) -> Self {
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![lhs, rhs],
                vec![],
                0,
            ),
        }
    }
    /// Computes the element-wise logical or bitwise AND of `lhs` and `rhs`.
    ///
    /// Returns an error when operand element types or shapes are incompatible.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, lhs: Value, rhs: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(
                ctx,
                "and",
                &[lhs.get_type(ctx), rhs.get_type(ctx)],
            ),
            "stablehlo.and",
        )?;
        Ok(Self::new(ctx, lhs, rhs, result))
    }
    /// Returns the left-hand operand.
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    /// Returns the right-hand operand.
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}
impl Verify for AndOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "and", 2)
    }
}

/// Applies elementwise logical or bitwise disjunction.
#[pliron_op(name="stablehlo.or",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct OrOp;
impl OrOp {
    /// Creates a `stablehlo.or` operation with the supplied result type. Uses `lhs`, `rhs` as operands.
    #[inline]
    pub fn new(ctx: &mut Context, lhs: Value, rhs: Value, result: TypeHandle) -> Self {
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![lhs, rhs],
                vec![],
                0,
            ),
        }
    }
    /// Computes the element-wise logical or bitwise OR of `lhs` and `rhs`.
    ///
    /// Returns an error when operand element types or shapes are incompatible.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, lhs: Value, rhs: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "or", &[lhs.get_type(ctx), rhs.get_type(ctx)]),
            "stablehlo.or",
        )?;
        Ok(Self::new(ctx, lhs, rhs, result))
    }
    /// Returns the left-hand operand.
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    /// Returns the right-hand operand.
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}
impl Verify for OrOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "or", 2)
    }
}

/// Applies elementwise exclusive-or.
#[pliron_op(name="stablehlo.xor",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct XorOp;
impl XorOp {
    /// Creates a `stablehlo.xor` operation with the supplied result type. Uses `lhs`, `rhs` as operands.
    #[inline]
    pub fn new(ctx: &mut Context, lhs: Value, rhs: Value, result: TypeHandle) -> Self {
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![lhs, rhs],
                vec![],
                0,
            ),
        }
    }
    /// Computes the element-wise logical or bitwise exclusive OR of `lhs` and `rhs`.
    ///
    /// Returns an error when operand element types or shapes are incompatible.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, lhs: Value, rhs: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(
                ctx,
                "xor",
                &[lhs.get_type(ctx), rhs.get_type(ctx)],
            ),
            "stablehlo.xor",
        )?;
        Ok(Self::new(ctx, lhs, rhs, result))
    }
    /// Returns the left-hand operand.
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    /// Returns the right-hand operand.
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}
impl Verify for XorOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "xor", 2)
    }
}

/// Reinterprets the input bit pattern using the requested result element type.
#[pliron_op(name="stablehlo.bitcast_convert",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct BitcastConvertOp;
impl BitcastConvertOp {
    /// Returns the input operand.
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    /// Creates a `stablehlo.bitcast_convert` operation with the supplied result type. Uses `input` as its operand.
    #[inline]
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
    /// Reinterprets the bits of `operand` using `to` as its result element type.
    ///
    /// The inferred shape accounts for the source and target element bit widths. Returns an error
    /// when the conversion is unsupported or shape inference overflows.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value, to: ElementType) -> Result<Self> {
        let template = TensorType::scalar(ctx, to);
        let result = crate::ops::builder_result(
            crate::inference::infer_bitcast_convert_type(ctx, input.get_type(ctx), template),
            "stablehlo.bitcast_convert",
        )?;
        Ok(Self::new(ctx, input, result))
    }
}
impl Verify for BitcastConvertOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_bitcast(self, ctx)
    }
}

/// Reduces the effective exponent and mantissa precision of floating-point elements.
#[pliron_op(name="stablehlo.reduce_precision",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface],attributes=(stablehlo_exponent_bits:IntegerAttr,stablehlo_mantissa_bits:IntegerAttr))]
pub struct ReducePrecisionOp;
impl ReducePrecisionOp {
    /// Returns the input operand.
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    /// Creates a `stablehlo.reduce_precision` operation with the supplied result type. Uses `input` as its operand. Configures the operation with `exponent_bits`, `mantissa_bits`.
    #[inline]
    pub fn new(
        ctx: &mut Context,
        input: Value,
        result: TypeHandle,
        exponent_bits: i32,
        mantissa_bits: i32,
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
        op.set_attr_stablehlo_exponent_bits(ctx, signed_i32_attr(ctx, exponent_bits));
        op.set_attr_stablehlo_mantissa_bits(ctx, signed_i32_attr(ctx, mantissa_bits));
        op
    }
    /// Creates a precision-reduction operation using the input type as its result type.
    ///
    /// The exponent and mantissa widths are stored as attributes; the builder does not validate
    /// their domains. Call [`Verify::verify`] to check the operation.
    #[inline]
    pub fn new_inferred(
        ctx: &mut Context,
        input: Value,
        exponent_bits: i32,
        mantissa_bits: i32,
    ) -> Self {
        let result = input.get_type(ctx);
        Self::new(ctx, input, result, exponent_bits, mantissa_bits)
    }
}
fn signed_i32_attr(ctx: &Context, value: i32) -> IntegerAttr {
    IntegerAttr::new(
        IntegerType::get(ctx, 32, Signedness::Signless),
        APInt::from_i32(value, bw(32)),
    )
}
impl Verify for ReducePrecisionOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_reduce_precision(
            self,
            ctx,
            self.get_attr_stablehlo_exponent_bits(ctx)
                .and_then(|a| a.value().to_string_decimal(true).parse::<i32>().ok()),
            self.get_attr_stablehlo_mantissa_bits(ctx)
                .and_then(|a| a.value().to_string_decimal(true).parse::<i32>().ok()),
        )
    }
}

/// Quantizes floating-point values using the quantization parameters encoded by their element type.
#[pliron_op(name="stablehlo.uniform_quantize",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct UniformQuantizeOp;
impl UniformQuantizeOp {
    /// Returns the input operand.
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    /// Creates a `stablehlo.uniform_quantize` operation with the supplied result type. Uses `input` as its operand.
    #[inline]
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
    /// Quantizes `input` using the quantized element type in `result_template`.
    ///
    /// The input must be a supported floating-point or quantized tensor, and its expressed type
    /// must match the template's expressed type. The result shape comes from `input`, not the
    /// template. Returns an error when these types are incompatible.
    #[inline]
    pub fn new_inferred(
        ctx: &mut Context,
        input: Value,
        result_template: TypeHandle,
    ) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_uniform_quantize_type(
                ctx,
                input.get_type(ctx),
                result_template,
            ),
            "stablehlo.uniform_quantize",
        )?;
        Ok(Self::new(ctx, input, result))
    }
}
impl Verify for UniformQuantizeOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_uniform_quantize(self, ctx)
    }
}

/// Converts uniformly quantized values to floating-point values.
#[pliron_op(name="stablehlo.uniform_dequantize",format,interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct UniformDequantizeOp;
impl UniformDequantizeOp {
    /// Returns the input operand.
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    /// Creates a `stablehlo.uniform_dequantize` operation with the supplied result type. Uses `input` as its operand.
    #[inline]
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
    /// Converts each quantized element of `operand` to its expressed floating-point representation.
    ///
    /// Returns an error when the operand does not have a supported quantized element type.
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_uniform_dequantize_type(ctx, input.get_type(ctx)),
            "stablehlo.uniform_dequantize",
        )?;
        Ok(Self::new(ctx, input, result))
    }
}
impl Verify for UniformDequantizeOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_uniform_dequantize(self, ctx)
    }
}

impl_result_accuracy_accessors!(
    CbrtOp,
    CosineOp,
    ExponentialOp,
    ExponentialMinusOneOp,
    Exp2Op,
    LogOp,
    LogPlusOneOp,
    Log2Op,
    LogisticOp,
    RsqrtOp,
    SineOp,
    SqrtOp,
    TanOp,
    TanhOp,
);
