//! Private operation definitions and their generated attribute storage.

use crate::Diagnostic;
use pliron::{
    common_traits::Verify,
    context::{Context, Ptr},
    derive::pliron_op,
    op::Op,
    operation::Operation,
    printable::Printable,
    result::Result,
    r#type::{TypeHandle, Typed},
    value::Value,
};
use pliron_arith::{
    attributes::{FastMathFlags, FastMathFlagsAttr},
    shaped::{self, Shaped},
};

pliron::dict_key!(ATTR_KEY_MATH_FASTMATH, "math_fastmath");

/// Verification error carrying the operation's name and source location.
fn fail(ctx: &Context, op: Ptr<Operation>, reason: &'static str) -> pliron::result::Error {
    let location = op.deref(ctx).loc().disp(ctx).to_string();
    Diagnostic::InvalidOperation {
        operation: Operation::get_op_dyn(op, ctx).get_opid().to_string(),
        location,
        reason,
    }
    .into()
}

/// Checks operand and result counts and rejects regions and successors.
fn verify_structure(ctx: &Context, op: Ptr<Operation>, operands: usize) -> Result<()> {
    let (num_operands, num_results, num_regions, num_successors) = {
        let ir = op.deref(ctx);
        (
            ir.get_num_operands(),
            ir.get_num_results(),
            ir.num_regions(),
            ir.get_num_successors(),
        )
    };
    if num_operands != operands || num_results != 1 {
        return Err(fail(ctx, op, "operand or result count mismatch"));
    }
    if num_regions != 0 || num_successors != 0 {
        return Err(fail(
            ctx,
            op,
            "operation must not have regions or successors",
        ));
    }
    Ok(())
}

fn operand_type(ctx: &Context, op: Ptr<Operation>, index: usize) -> TypeHandle {
    op.deref(ctx).get_operand(index).get_type(ctx)
}

fn result_type(ctx: &Context, op: Ptr<Operation>) -> TypeHandle {
    op.deref(ctx).get_result(0).get_type(ctx)
}

fn classify(ctx: &Context, op: Ptr<Operation>, ty: TypeHandle) -> Result<Shaped> {
    shaped::classify(ctx, ty).ok_or_else(|| {
        fail(
            ctx,
            op,
            "type must be a scalar or a tensor or vector of scalars",
        )
    })
}

/// Element-type family accepted by an operation.
#[derive(Clone, Copy)]
enum Domain {
    /// Floating-point elements.
    Float,
    /// Signless integer elements; `math` integer operations reject `index`.
    Integer,
}

fn in_domain(ctx: &Context, element: TypeHandle, domain: Domain) -> bool {
    match domain {
        Domain::Float => shaped::is_float(ctx, element),
        Domain::Integer => shaped::signless_width(ctx, element).is_some(),
    }
}

/// Verifies `SameOperandsAndResultType` over `arity` operands in `domain`.
fn verify_same_type(ctx: &Context, op: Ptr<Operation>, arity: usize, domain: Domain) -> Result<()> {
    verify_structure(ctx, op, arity)?;
    let result = result_type(ctx, op);
    if (0..arity).any(|i| operand_type(ctx, op, i) != result) {
        return Err(fail(ctx, op, "operands and result must have the same type"));
    }
    let shaped = classify(ctx, op, result)?;
    if !in_domain(ctx, shaped.element, domain) {
        return Err(fail(
            ctx,
            op,
            "element type is outside the operation's domain",
        ));
    }
    Ok(())
}

fn new_op<T: Op>(
    ctx: &mut Context,
    make: fn(Ptr<Operation>) -> T,
    result: TypeHandle,
    operands: Vec<Value>,
) -> T {
    let info = T::get_concrete_op_info();
    make(Operation::new(ctx, info, vec![result], operands, vec![], 0))
}

macro_rules! fastmath_accessors {
    ($struct:ident) => {
        pliron_arith::shared_attr!(
            $struct,
            ATTR_KEY_MATH_FASTMATH,
            FastMathFlagsAttr,
            get_attr_math_fastmath,
            set_attr_math_fastmath
        );
        impl $struct {
            /// Returns the fast-math flags, or [`FastMathFlags::NONE`] when absent.
            pub fn fastmath(&self, ctx: &Context) -> FastMathFlags {
                self.get_attr_math_fastmath(ctx)
                    .map(|attr| attr.value)
                    .unwrap_or_default()
            }
            /// Replaces the fast-math flags.
            pub fn set_fastmath(&self, ctx: &Context, flags: FastMathFlags) {
                self.set_attr_math_fastmath(ctx, FastMathFlagsAttr::new(flags));
            }
        }
    };
}

macro_rules! result_accessor {
    ($struct:ident) => {
        impl $struct {
            /// Returns the result.
            pub fn result(&self, ctx: &Context) -> Value {
                self.get_operation().deref(ctx).get_result(0)
            }
        }
    };
}

macro_rules! unary_op {
    ($struct:ident, $name:literal, $doc:literal, Float) => {
        #[doc = $doc]
        ///
        /// Operand and result share one floating-point scalar, tensor, or vector type.
        #[pliron_op(
            name = $name,
            format = "$0 attr_dict `:` typesig",
            interfaces = [pliron::builtin::op_interfaces::OneResultInterface]
        )]
        pub struct $struct;
        unary_op!(@common $struct, Domain::Float);
        fastmath_accessors!($struct);
    };
    ($struct:ident, $name:literal, $doc:literal, Integer) => {
        #[doc = $doc]
        ///
        /// Operand and result share one signless-integer scalar, tensor, or vector type.
        #[pliron_op(
            name = $name,
            format = "$0 attr_dict `:` typesig",
            interfaces = [pliron::builtin::op_interfaces::OneResultInterface]
        )]
        pub struct $struct;
        unary_op!(@common $struct, Domain::Integer);
    };
    (@common $struct:ident, $domain:expr) => {
        impl $struct {
            /// Allocates the operation; the result type is the operand's type.
            pub fn new(ctx: &mut Context, operand: Value) -> Self {
                let ty = operand.get_type(ctx);
                new_op(ctx, |op| Self { op }, ty, vec![operand])
            }
            /// Returns the operand.
            pub fn operand(&self, ctx: &Context) -> Value {
                self.get_operation().deref(ctx).get_operand(0)
            }
        }
        result_accessor!($struct);
        impl Verify for $struct {
            fn verify(&self, ctx: &Context) -> Result<()> {
                verify_same_type(ctx, self.get_operation(), 1, $domain)
            }
        }
    };
}

macro_rules! binary_op {
    ($struct:ident, $name:literal, $doc:literal, Float) => {
        #[doc = $doc]
        ///
        /// Operands and result share one floating-point scalar, tensor, or vector type.
        #[pliron_op(
            name = $name,
            format = "$0 `,` $1 attr_dict `:` typesig",
            interfaces = [pliron::builtin::op_interfaces::OneResultInterface]
        )]
        pub struct $struct;
        binary_op!(@common $struct, Domain::Float);
        fastmath_accessors!($struct);
    };
    ($struct:ident, $name:literal, $doc:literal, Integer) => {
        #[doc = $doc]
        ///
        /// Operands and result share one signless-integer scalar, tensor, or vector type.
        #[pliron_op(
            name = $name,
            format = "$0 `,` $1 attr_dict `:` typesig",
            interfaces = [pliron::builtin::op_interfaces::OneResultInterface]
        )]
        pub struct $struct;
        binary_op!(@common $struct, Domain::Integer);
    };
    (@common $struct:ident, $domain:expr) => {
        impl $struct {
            /// Allocates the operation; the result type is the left operand's type.
            pub fn new(ctx: &mut Context, lhs: Value, rhs: Value) -> Self {
                let ty = lhs.get_type(ctx);
                new_op(ctx, |op| Self { op }, ty, vec![lhs, rhs])
            }
            /// Returns the left operand.
            pub fn lhs(&self, ctx: &Context) -> Value {
                self.get_operation().deref(ctx).get_operand(0)
            }
            /// Returns the right operand.
            pub fn rhs(&self, ctx: &Context) -> Value {
                self.get_operation().deref(ctx).get_operand(1)
            }
        }
        result_accessor!($struct);
        impl Verify for $struct {
            fn verify(&self, ctx: &Context) -> Result<()> {
                verify_same_type(ctx, self.get_operation(), 2, $domain)
            }
        }
    };
}

macro_rules! classification_op {
    ($struct:ident, $name:literal, $doc:literal) => {
        #[doc = $doc]
        ///
        /// The operand is floating-point; the result is `i1` with the operand's shape.
        #[pliron_op(
                            name = $name,
                            format = "$0 attr_dict `:` typesig",
                            interfaces = [pliron::builtin::op_interfaces::OneResultInterface]
                        )]
        pub struct $struct;
        impl $struct {
            /// Allocates the operation with an `i1` result shaped like the operand.
            ///
            /// A non-shaped operand yields a scalar `i1`; call `verify` to reject it.
            pub fn new(ctx: &mut Context, operand: Value) -> Self {
                let ty = operand.get_type(ctx);
                let i1 = shaped::i1(ctx);
                let result = shaped::classify(ctx, ty)
                    .map(|s| shaped::with_element(ctx, &s, i1))
                    .unwrap_or(i1);
                new_op(ctx, |op| Self { op }, result, vec![operand])
            }
            /// Returns the operand.
            pub fn operand(&self, ctx: &Context) -> Value {
                self.get_operation().deref(ctx).get_operand(0)
            }
        }
        result_accessor!($struct);
        fastmath_accessors!($struct);
        impl Verify for $struct {
            fn verify(&self, ctx: &Context) -> Result<()> {
                let op = self.get_operation();
                verify_structure(ctx, op, 1)?;
                let input = classify(ctx, op, operand_type(ctx, op, 0))?;
                if !shaped::is_float(ctx, input.element) {
                    return Err(fail(ctx, op, "operand must be floating-point"));
                }
                let output = classify(ctx, op, result_type(ctx, op))?;
                if !output.same_shape(&input) || output.element != shaped::i1(ctx) {
                    return Err(fail(ctx, op, "result must be i1 with the operand's shape"));
                }
                Ok(())
            }
        }
    };
}

unary_op!(AbsFOp, "math.absf", "Floating-point absolute value.", Float);
unary_op!(
    CeilOp,
    "math.ceil",
    "Rounds toward positive infinity.",
    Float
);
unary_op!(
    FloorOp,
    "math.floor",
    "Rounds toward negative infinity.",
    Float
);
unary_op!(
    RoundOp,
    "math.round",
    "Rounds to nearest, ties away from zero.",
    Float
);
unary_op!(
    RoundEvenOp,
    "math.roundeven",
    "Rounds to nearest, ties to even.",
    Float
);
unary_op!(TruncOp, "math.trunc", "Rounds toward zero.", Float);
unary_op!(ExpOp, "math.exp", "Base-e exponential.", Float);
unary_op!(Exp2Op, "math.exp2", "Base-2 exponential.", Float);
unary_op!(
    ExpM1Op,
    "math.expm1",
    "`exp(x) - 1`, accurate near zero.",
    Float
);
unary_op!(LogOp, "math.log", "Natural logarithm.", Float);
unary_op!(Log2Op, "math.log2", "Base-2 logarithm.", Float);
unary_op!(Log10Op, "math.log10", "Base-10 logarithm.", Float);
unary_op!(
    Log1pOp,
    "math.log1p",
    "`log(1 + x)`, accurate near zero.",
    Float
);
unary_op!(SqrtOp, "math.sqrt", "Square root.", Float);
unary_op!(RsqrtOp, "math.rsqrt", "Reciprocal square root.", Float);
unary_op!(CbrtOp, "math.cbrt", "Cube root.", Float);
unary_op!(SinOp, "math.sin", "Sine.", Float);
unary_op!(CosOp, "math.cos", "Cosine.", Float);
unary_op!(TanOp, "math.tan", "Tangent.", Float);
unary_op!(AsinOp, "math.asin", "Arcsine.", Float);
unary_op!(AcosOp, "math.acos", "Arccosine.", Float);
unary_op!(AtanOp, "math.atan", "Arctangent.", Float);
unary_op!(SinhOp, "math.sinh", "Hyperbolic sine.", Float);
unary_op!(CoshOp, "math.cosh", "Hyperbolic cosine.", Float);
unary_op!(TanhOp, "math.tanh", "Hyperbolic tangent.", Float);
unary_op!(AsinhOp, "math.asinh", "Inverse hyperbolic sine.", Float);
unary_op!(AcoshOp, "math.acosh", "Inverse hyperbolic cosine.", Float);
unary_op!(AtanhOp, "math.atanh", "Inverse hyperbolic tangent.", Float);
unary_op!(ErfOp, "math.erf", "Error function.", Float);
unary_op!(ErfcOp, "math.erfc", "Complementary error function.", Float);

binary_op!(
    Atan2Op,
    "math.atan2",
    "Two-argument arctangent of `lhs / rhs`.",
    Float
);
binary_op!(PowFOp, "math.powf", "Floating-point power.", Float);
binary_op!(
    CopySignOp,
    "math.copysign",
    "Magnitude of `lhs` with the sign of `rhs`.",
    Float
);

unary_op!(
    AbsIOp,
    "math.absi",
    "Integer absolute value; the minimum value wraps.",
    Integer
);
unary_op!(
    CountLeadingZerosOp,
    "math.ctlz",
    "Counts leading zero bits.",
    Integer
);
unary_op!(
    CountTrailingZerosOp,
    "math.cttz",
    "Counts trailing zero bits.",
    Integer
);
unary_op!(CtPopOp, "math.ctpop", "Counts set bits.", Integer);
binary_op!(
    IPowIOp,
    "math.ipowi",
    "Signed integer power; negative exponents follow MLIR semantics.",
    Integer
);

classification_op!(IsNaNOp, "math.isnan", "Tests for NaN.");
classification_op!(
    IsInfOp,
    "math.isinf",
    "Tests for positive or negative infinity."
);
classification_op!(IsFiniteOp, "math.isfinite", "Tests for a finite value.");
classification_op!(
    IsNormalOp,
    "math.isnormal",
    "Tests for a normal (not subnormal) value."
);

/// Fused multiply-add `a * b + c` with a single rounding.
///
/// All operands and the result share one floating-point scalar, tensor, or vector type.
#[pliron_op(
    name = "math.fma",
    format = "$0 `,` $1 `,` $2 attr_dict `:` typesig",
    interfaces = [pliron::builtin::op_interfaces::OneResultInterface]
)]
pub struct FmaOp;

impl FmaOp {
    /// Allocates the operation; the result type is `a`'s type.
    pub fn new(ctx: &mut Context, a: Value, b: Value, c: Value) -> Self {
        let ty = a.get_type(ctx);
        new_op(ctx, |op| Self { op }, ty, vec![a, b, c])
    }
    /// Returns the operand at `index` (`0` = `a`, `1` = `b`, `2` = `c`).
    pub fn operand(&self, ctx: &Context, index: usize) -> Value {
        self.get_operation().deref(ctx).get_operand(index)
    }
}
result_accessor!(FmaOp);
fastmath_accessors!(FmaOp);

impl Verify for FmaOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        verify_same_type(ctx, self.get_operation(), 3, Domain::Float)
    }
}

/// Clamps `value` to `[min, max]`: `maxf(minf(value, max), min)`; poison when `min > max`.
///
/// All operands and the result share one floating-point scalar, tensor, or vector type.
#[pliron_op(
    name = "math.clampf",
    format = "$0 `,` $1 `,` $2 attr_dict `:` typesig",
    interfaces = [pliron::builtin::op_interfaces::OneResultInterface]
)]
pub struct ClampFOp;

impl ClampFOp {
    /// Allocates the operation; the result type is `value`'s type.
    pub fn new(ctx: &mut Context, value: Value, min: Value, max: Value) -> Self {
        let ty = value.get_type(ctx);
        new_op(ctx, |op| Self { op }, ty, vec![value, min, max])
    }
    /// Returns the clamped operand.
    pub fn value(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    /// Returns the lower bound.
    pub fn min(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
    /// Returns the upper bound.
    pub fn max(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(2)
    }
}
result_accessor!(ClampFOp);
fastmath_accessors!(ClampFOp);

impl Verify for ClampFOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        verify_same_type(ctx, self.get_operation(), 3, Domain::Float)
    }
}

/// Computes the sine and cosine of one operand together.
///
/// The operand and both results share one floating-point scalar, tensor, or vector type.
#[pliron_op(
    name = "math.sincos",
    format = "$0 attr_dict `:` typesig",
    interfaces = [pliron::builtin::op_interfaces::NResultsInterface<2>]
)]
pub struct SincosOp;

impl SincosOp {
    /// Allocates the operation; both results take `operand`'s type.
    pub fn new(ctx: &mut Context, operand: Value) -> Self {
        let ty = operand.get_type(ctx);
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![ty, ty],
                vec![operand],
                vec![],
                0,
            ),
        }
    }
    /// Returns the operand.
    pub fn operand(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    /// Returns the sine result.
    pub fn sin(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }
    /// Returns the cosine result.
    pub fn cos(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(1)
    }
}
fastmath_accessors!(SincosOp);

impl Verify for SincosOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        let (operands, results, regions, successors) = {
            let ir = op.deref(ctx);
            (
                ir.get_num_operands(),
                ir.get_num_results(),
                ir.num_regions(),
                ir.get_num_successors(),
            )
        };
        if operands != 1 || results != 2 || regions != 0 || successors != 0 {
            return Err(fail(
                ctx,
                op,
                "expected one operand, two results, and no regions",
            ));
        }
        let ty = operand_type(ctx, op, 0);
        let ir = op.deref(ctx);
        if ir.get_result(0).get_type(ctx) != ty || ir.get_result(1).get_type(ctx) != ty {
            return Err(fail(ctx, op, "operand and results must have the same type"));
        }
        drop(ir);
        let shaped = classify(ctx, op, ty)?;
        if !in_domain(ctx, shaped.element, Domain::Float) {
            return Err(fail(ctx, op, "element type must be floating point"));
        }
        Ok(())
    }
}

/// Floating-point base raised to a signless-integer exponent.
///
/// The base and result share one floating-point type; the exponent has the
/// base's shape with a signless-integer element.
#[pliron_op(
    name = "math.fpowi",
    format = "$0 `,` $1 attr_dict `:` typesig",
    interfaces = [pliron::builtin::op_interfaces::OneResultInterface]
)]
pub struct FPowIOp;

impl FPowIOp {
    /// Allocates the operation; the result type is the base's type.
    pub fn new(ctx: &mut Context, base: Value, exponent: Value) -> Self {
        let ty = base.get_type(ctx);
        new_op(ctx, |op| Self { op }, ty, vec![base, exponent])
    }
    /// Returns the floating-point base.
    pub fn base(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    /// Returns the integer exponent.
    pub fn exponent(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}
result_accessor!(FPowIOp);
fastmath_accessors!(FPowIOp);

impl Verify for FPowIOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        verify_structure(ctx, op, 2)?;
        let result = result_type(ctx, op);
        if operand_type(ctx, op, 0) != result {
            return Err(fail(ctx, op, "base and result must have the same type"));
        }
        let base = classify(ctx, op, result)?;
        if !shaped::is_float(ctx, base.element) {
            return Err(fail(ctx, op, "base must be floating-point"));
        }
        let exponent = classify(ctx, op, operand_type(ctx, op, 1))?;
        if !exponent.same_shape(&base) || shaped::signless_width(ctx, exponent.element).is_none() {
            return Err(fail(
                ctx,
                op,
                "exponent must be a signless integer with the base's shape",
            ));
        }
        Ok(())
    }
}
