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

#[pliron_op(name="stablehlo.convert",format="$0",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct ConvertOp;
impl ConvertOp {
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
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
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

#[pliron_op(name="stablehlo.select",format="$0 `,` $1 `,` $2",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct SelectOp;
impl SelectOp {
    pub fn predicate(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    pub fn on_true(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
    pub fn on_false(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(2)
    }
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

#[pliron_op(name="stablehlo.compare",format="$0 `,` $1 attr($stablehlo_compare_direction, $ComparisonDirectionAttr) attr($stablehlo_compare_type, $ComparisonTypeAttr)",interfaces=[pliron::builtin::op_interfaces::OneResultInterface],attributes=(stablehlo_compare_direction:ComparisonDirectionAttr,stablehlo_compare_type:ComparisonTypeAttr))]
pub struct CompareOp;
impl CompareOp {
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
    #[inline]
    pub fn new_inferred(
        ctx: &mut Context,
        lhs: Value,
        rhs: Value,
        direction: CompareDirection,
        comparison: Option<CompareType>,
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
            Self::new_without_type(ctx, lhs, rhs, result, direction)
        })
    }
    #[inline]
    pub fn new_without_type(
        ctx: &mut Context,
        lhs: Value,
        rhs: Value,
        result: TypeHandle,
        direction: CompareDirection,
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
        op.set_attr_stablehlo_compare_direction(ctx, comparison_direction_attr(direction));
        op
    }
    #[inline]
    pub fn new(
        ctx: &mut Context,
        lhs: Value,
        rhs: Value,
        result: TypeHandle,
        direction: CompareDirection,
        comparison: CompareType,
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
        op.set_attr_stablehlo_compare_direction(ctx, comparison_direction_attr(direction));
        op.set_attr_stablehlo_compare_type(ctx, comparison_type_attr(comparison));
        op
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u64)]
pub enum CompareDirection {
    Eq,
    Ne,
    Ge,
    Gt,
    Le,
    Lt,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u64)]
pub enum CompareType {
    NoType = 0,
    Float = 1,
    TotalOrder = 2,
    Signed = 3,
    Unsigned = 4,
    WeakOrder = 5,
}
fn comparison_direction_attr(value: CompareDirection) -> ComparisonDirectionAttr {
    use crate::attributes::ComparisonDirection as D;
    ComparisonDirectionAttr::new(match value {
        CompareDirection::Eq => D::Eq,
        CompareDirection::Ne => D::Ne,
        CompareDirection::Ge => D::Ge,
        CompareDirection::Gt => D::Gt,
        CompareDirection::Le => D::Le,
        CompareDirection::Lt => D::Lt,
    })
}
fn comparison_type_attr(value: CompareType) -> ComparisonTypeAttr {
    use crate::attributes::ComparisonType as T;
    ComparisonTypeAttr::new(match value {
        CompareType::NoType => T::NoType,
        CompareType::Float => T::Float,
        CompareType::TotalOrder => T::TotalOrder,
        CompareType::Signed => T::Signed,
        CompareType::Unsigned => T::Unsigned,
        CompareType::WeakOrder => T::WeakOrder,
    })
}
impl Verify for CompareOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_compare(
            self,
            ctx,
            self.get_attr_stablehlo_compare_direction(ctx)
                .map(|a| u64::from(a.value)),
            self.get_attr_stablehlo_compare_type(ctx)
                .map(|a| u64::from(a.value)),
        )
    }
}

#[pliron_op(name="stablehlo.clamp",format="$0 `,` $1 `,` $2",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct ClampOp;
impl ClampOp {
    pub fn min(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    pub fn value(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
    pub fn max(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(2)
    }
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

#[pliron_op(name="stablehlo.iota",format="attr($stablehlo_iota_dimension, $IntegerAttr)",interfaces=[pliron::builtin::op_interfaces::NOpdsInterface<0>,pliron::builtin::op_interfaces::OneResultInterface],attributes=(stablehlo_iota_dimension:IntegerAttr))]
pub struct IotaOp;
impl IotaOp {
    pub fn dimension(&self, ctx: &Context) -> Option<usize> {
        self.get_attr_stablehlo_iota_dimension(ctx)
            .and_then(|a| a.value().to_string_decimal(false).parse::<u64>().ok())
            .and_then(|v| usize::try_from(v).ok())
    }
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

#[pliron_op(name="stablehlo.abs",format="$0",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct AbsOp;
impl AbsOp {
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
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "abs", &[input.get_type(ctx)]),
            "stablehlo.abs",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for AbsOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "abs", 1)
    }
}

#[pliron_op(name="stablehlo.cbrt",format="$0",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct CbrtOp;
impl CbrtOp {
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
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "cbrt", &[input.get_type(ctx)]),
            "stablehlo.cbrt",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for CbrtOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "cbrt", 1)
    }
}

#[pliron_op(name="stablehlo.ceil",format="$0",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct CeilOp;
impl CeilOp {
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
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "ceil", &[input.get_type(ctx)]),
            "stablehlo.ceil",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for CeilOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "ceil", 1)
    }
}

#[pliron_op(name="stablehlo.count_leading_zeros",format="$0",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct CountLeadingZerosOp;
impl CountLeadingZerosOp {
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
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "count_leading_zeros", &[input.get_type(ctx)]),
            "stablehlo.count_leading_zeros",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for CountLeadingZerosOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "count_leading_zeros", 1)
    }
}

#[pliron_op(name="stablehlo.cosine",format="$0",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct CosineOp;
impl CosineOp {
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
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "cosine", &[input.get_type(ctx)]),
            "stablehlo.cosine",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for CosineOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "cosine", 1)
    }
}

#[pliron_op(name="stablehlo.exponential",format="$0",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct ExponentialOp;
impl ExponentialOp {
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
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "exponential", &[input.get_type(ctx)]),
            "stablehlo.exponential",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for ExponentialOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "exponential", 1)
    }
}

#[pliron_op(name="stablehlo.exponential_minus_one",format="$0",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct ExponentialMinusOneOp;
impl ExponentialMinusOneOp {
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
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for ExponentialMinusOneOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "exponential_minus_one", 1)
    }
}

#[pliron_op(name="stablehlo.floor",format="$0",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct FloorOp;
impl FloorOp {
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
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "floor", &[input.get_type(ctx)]),
            "stablehlo.floor",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for FloorOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "floor", 1)
    }
}

#[pliron_op(name="stablehlo.imag",format="$0",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct ImagOp;
impl ImagOp {
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
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "imag", &[input.get_type(ctx)]),
            "stablehlo.imag",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for ImagOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "imag", 1)
    }
}

#[pliron_op(name="stablehlo.is_finite",format="$0",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct IsFiniteOp;
impl IsFiniteOp {
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
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "is_finite", &[input.get_type(ctx)]),
            "stablehlo.is_finite",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for IsFiniteOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "is_finite", 1)
    }
}

#[pliron_op(name="stablehlo.log",format="$0",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct LogOp;
impl LogOp {
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
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "log", &[input.get_type(ctx)]),
            "stablehlo.log",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for LogOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "log", 1)
    }
}

#[pliron_op(name="stablehlo.log_plus_one",format="$0",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct LogPlusOneOp;
impl LogPlusOneOp {
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
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "log_plus_one", &[input.get_type(ctx)]),
            "stablehlo.log_plus_one",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for LogPlusOneOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "log_plus_one", 1)
    }
}

#[pliron_op(name="stablehlo.logistic",format="$0",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct LogisticOp;
impl LogisticOp {
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
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "logistic", &[input.get_type(ctx)]),
            "stablehlo.logistic",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for LogisticOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "logistic", 1)
    }
}

#[pliron_op(name="stablehlo.not",format="$0",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct NotOp;
impl NotOp {
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
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "not", &[input.get_type(ctx)]),
            "stablehlo.not",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for NotOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "not", 1)
    }
}

#[pliron_op(name="stablehlo.negate",format="$0",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct NegateOp;
impl NegateOp {
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
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "negate", &[input.get_type(ctx)]),
            "stablehlo.negate",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for NegateOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "negate", 1)
    }
}

#[pliron_op(name="stablehlo.popcnt",format="$0",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct PopulationCountOp;
impl PopulationCountOp {
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
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "popcnt", &[input.get_type(ctx)]),
            "stablehlo.popcnt",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for PopulationCountOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "popcnt", 1)
    }
}

#[pliron_op(name="stablehlo.real",format="$0",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct RealOp;
impl RealOp {
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
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "real", &[input.get_type(ctx)]),
            "stablehlo.real",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for RealOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "real", 1)
    }
}

#[pliron_op(name="stablehlo.round_nearest_afz",format="$0",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct RoundNearestAfzOp;
impl RoundNearestAfzOp {
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
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "round_nearest_afz", &[input.get_type(ctx)]),
            "stablehlo.round_nearest_afz",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for RoundNearestAfzOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "round_nearest_afz", 1)
    }
}

#[pliron_op(name="stablehlo.round_nearest_even",format="$0",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct RoundNearestEvenOp;
impl RoundNearestEvenOp {
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
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "round_nearest_even", &[input.get_type(ctx)]),
            "stablehlo.round_nearest_even",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for RoundNearestEvenOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "round_nearest_even", 1)
    }
}

#[pliron_op(name="stablehlo.rsqrt",format="$0",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct RsqrtOp;
impl RsqrtOp {
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
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "rsqrt", &[input.get_type(ctx)]),
            "stablehlo.rsqrt",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for RsqrtOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "rsqrt", 1)
    }
}

#[pliron_op(name="stablehlo.sign",format="$0",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct SignOp;
impl SignOp {
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
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "sign", &[input.get_type(ctx)]),
            "stablehlo.sign",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for SignOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "sign", 1)
    }
}

#[pliron_op(name="stablehlo.sine",format="$0",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct SineOp;
impl SineOp {
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
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "sine", &[input.get_type(ctx)]),
            "stablehlo.sine",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for SineOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "sine", 1)
    }
}

#[pliron_op(name="stablehlo.sqrt",format="$0",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct SqrtOp;
impl SqrtOp {
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
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "sqrt", &[input.get_type(ctx)]),
            "stablehlo.sqrt",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for SqrtOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "sqrt", 1)
    }
}

#[pliron_op(name="stablehlo.tan",format="$0",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct TanOp;
impl TanOp {
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
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "tan", &[input.get_type(ctx)]),
            "stablehlo.tan",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for TanOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "tan", 1)
    }
}

#[pliron_op(name="stablehlo.tanh",format="$0",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct TanhOp;
impl TanhOp {
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
    #[inline]
    pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "tanh", &[input.get_type(ctx)]),
            "stablehlo.tanh",
        )?;
        Ok(Self::new(ctx, input, result))
    }
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for TanhOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "tanh", 1)
    }
}

#[pliron_op(name="stablehlo.add",format="$0 `,` $1",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct AddOp;
impl AddOp {
    #[inline]
    pub fn new(ctx: &mut Context, lhs: Value, rhs: Value, result: TypeHandle) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);
        Operation::push_result(op, ctx, result);
        Operation::push_operand(op, ctx, lhs);
        Operation::push_operand(op, ctx, rhs);
        Self { op }
    }
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
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}
impl Verify for AddOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "add", 2)
    }
}

#[pliron_op(name="stablehlo.atan2",format="$0 `,` $1",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct Atan2Op;
impl Atan2Op {
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
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}
impl Verify for Atan2Op {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "atan2", 2)
    }
}

#[pliron_op(name="stablehlo.complex",format="$0 `,` $1",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct ComplexOp;
impl ComplexOp {
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
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}
impl Verify for ComplexOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "complex", 2)
    }
}

#[pliron_op(name="stablehlo.divide",format="$0 `,` $1",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct DivideOp;
impl DivideOp {
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
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}
impl Verify for DivideOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "divide", 2)
    }
}

#[pliron_op(name="stablehlo.maximum",format="$0 `,` $1",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct MaximumOp;
impl MaximumOp {
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
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}
impl Verify for MaximumOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "maximum", 2)
    }
}

#[pliron_op(name="stablehlo.minimum",format="$0 `,` $1",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct MinimumOp;
impl MinimumOp {
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
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}
impl Verify for MinimumOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "minimum", 2)
    }
}

#[pliron_op(name="stablehlo.multiply",format="$0 `,` $1",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct MultiplyOp;
impl MultiplyOp {
    #[inline]
    pub fn new(ctx: &mut Context, lhs: Value, rhs: Value, result: TypeHandle) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);
        Operation::push_result(op, ctx, result);
        Operation::push_operand(op, ctx, lhs);
        Operation::push_operand(op, ctx, rhs);
        Self { op }
    }
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
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}
impl Verify for MultiplyOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "multiply", 2)
    }
}

#[pliron_op(name="stablehlo.power",format="$0 `,` $1",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct PowerOp;
impl PowerOp {
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
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}
impl Verify for PowerOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "power", 2)
    }
}

#[pliron_op(name="stablehlo.remainder",format="$0 `,` $1",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct RemainderOp;
impl RemainderOp {
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
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}
impl Verify for RemainderOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "remainder", 2)
    }
}

#[pliron_op(name="stablehlo.shift_left",format="$0 `,` $1",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct ShiftLeftOp;
impl ShiftLeftOp {
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
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}
impl Verify for ShiftLeftOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "shift_left", 2)
    }
}

#[pliron_op(name="stablehlo.shift_right_arithmetic",format="$0 `,` $1",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct ShiftRightArithmeticOp;
impl ShiftRightArithmeticOp {
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
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}
impl Verify for ShiftRightArithmeticOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "shift_right_arithmetic", 2)
    }
}

#[pliron_op(name="stablehlo.shift_right_logical",format="$0 `,` $1",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct ShiftRightLogicalOp;
impl ShiftRightLogicalOp {
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
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}
impl Verify for ShiftRightLogicalOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "shift_right_logical", 2)
    }
}

#[pliron_op(name="stablehlo.subtract",format="$0 `,` $1",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct SubtractOp;
impl SubtractOp {
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
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}
impl Verify for SubtractOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "subtract", 2)
    }
}

#[pliron_op(name="stablehlo.and",format="$0 `,` $1",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct AndOp;
impl AndOp {
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
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}
impl Verify for AndOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "and", 2)
    }
}

#[pliron_op(name="stablehlo.or",format="$0 `,` $1",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct OrOp;
impl OrOp {
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
    #[inline]
    pub fn new_inferred(ctx: &mut Context, lhs: Value, rhs: Value) -> Result<Self> {
        let result = crate::ops::builder_result(
            crate::inference::infer_elementwise(ctx, "or", &[lhs.get_type(ctx), rhs.get_type(ctx)]),
            "stablehlo.or",
        )?;
        Ok(Self::new(ctx, lhs, rhs, result))
    }
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}
impl Verify for OrOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "or", 2)
    }
}

#[pliron_op(name="stablehlo.xor",format="$0 `,` $1",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct XorOp;
impl XorOp {
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
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}
impl Verify for XorOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        crate::verification::elementwise::verify_elementwise(self, ctx, "xor", 2)
    }
}

#[pliron_op(name="stablehlo.bitcast_convert",format="$0",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct BitcastConvertOp;
impl BitcastConvertOp {
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
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

#[pliron_op(name="stablehlo.reduce_precision",format="$0 attr($stablehlo_exponent_bits, $IntegerAttr) attr($stablehlo_mantissa_bits, $IntegerAttr)",interfaces=[pliron::builtin::op_interfaces::OneResultInterface],attributes=(stablehlo_exponent_bits:IntegerAttr,stablehlo_mantissa_bits:IntegerAttr))]
pub struct ReducePrecisionOp;
impl ReducePrecisionOp {
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
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

#[pliron_op(name="stablehlo.uniform_quantize",format="$0",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct UniformQuantizeOp;
impl UniformQuantizeOp {
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
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

#[pliron_op(name="stablehlo.uniform_dequantize",format="$0",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct UniformDequantizeOp;
impl UniformDequantizeOp {
    pub fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
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
