//! Private operation definitions and their generated attribute storage.

use crate::{
    Diagnostic,
    attributes::*,
    shaped::{self, Shaped},
};
use pliron::{
    attribute::AttrObj,
    builtin::attributes::BoolAttr,
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
use pliron_mlir_builtin::{
    attributes::{DenseElementsAttr, IndexAttr, ScalarBitsAttr},
    integer::IntegerValue,
    types::ElementType,
};

pliron::dict_key!(ATTR_KEY_ARITH_FASTMATH, "arith_fastmath");
pliron::dict_key!(ATTR_KEY_ARITH_ROUNDINGMODE, "arith_roundingmode");
pliron::dict_key!(ATTR_KEY_ARITH_OVERFLOW_FLAGS, "arith_overflow_flags");
pliron::dict_key!(ATTR_KEY_ARITH_IS_EXACT, "arith_is_exact");
pliron::dict_key!(ATTR_KEY_ARITH_NNEG, "arith_nneg");

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
fn verify_structure(
    ctx: &Context,
    op: Ptr<Operation>,
    operands: usize,
    results: usize,
) -> Result<()> {
    let (num_operands, num_results, num_regions, num_successors) = {
        let ir = op.deref(ctx);
        (
            ir.get_num_operands(),
            ir.get_num_results(),
            ir.num_regions(),
            ir.get_num_successors(),
        )
    };
    if num_operands != operands || num_results != results {
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

/// Element-type family accepted by an operation.
#[derive(Clone, Copy)]
enum Domain {
    /// Floating-point elements.
    Float,
    /// Signless integer or `index` elements.
    Integer,
}

fn in_domain(ctx: &Context, element: TypeHandle, domain: Domain) -> bool {
    match domain {
        Domain::Float => shaped::is_float(ctx, element),
        Domain::Integer => shaped::is_signless_int_or_index(ctx, element),
    }
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

/// Verifies `SameOperandsAndResultType` over `arity` operands in `domain`.
fn verify_same_type(ctx: &Context, op: Ptr<Operation>, arity: usize, domain: Domain) -> Result<()> {
    verify_structure(ctx, op, arity, 1)?;
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

macro_rules! fastmath_accessors {
    ($struct:ident) => {
        crate::shared_attr!(
            $struct,
            ATTR_KEY_ARITH_FASTMATH,
            FastMathFlagsAttr,
            get_attr_arith_fastmath,
            set_attr_arith_fastmath
        );
        impl $struct {
            /// Returns the fast-math flags, or [`FastMathFlags::NONE`] when absent.
            pub fn fastmath(&self, ctx: &Context) -> FastMathFlags {
                self.get_attr_arith_fastmath(ctx)
                    .map(|attr| attr.value)
                    .unwrap_or_default()
            }
            /// Replaces the fast-math flags.
            pub fn set_fastmath(&self, ctx: &Context, flags: FastMathFlags) {
                self.set_attr_arith_fastmath(ctx, FastMathFlagsAttr::new(flags));
            }
        }
    };
}

macro_rules! overflow_accessors {
    ($struct:ident) => {
        crate::shared_attr!(
            $struct,
            ATTR_KEY_ARITH_OVERFLOW_FLAGS,
            IntegerOverflowFlagsAttr,
            get_attr_arith_overflow_flags,
            set_attr_arith_overflow_flags
        );
        impl $struct {
            /// Returns the overflow flags, or [`IntegerOverflowFlags::NONE`] when absent.
            pub fn overflow_flags(&self, ctx: &Context) -> IntegerOverflowFlags {
                self.get_attr_arith_overflow_flags(ctx)
                    .map(|attr| attr.value)
                    .unwrap_or_default()
            }
            /// Replaces the overflow flags.
            pub fn set_overflow_flags(&self, ctx: &Context, flags: IntegerOverflowFlags) {
                self.set_attr_arith_overflow_flags(ctx, IntegerOverflowFlagsAttr::new(flags));
            }
        }
    };
}

macro_rules! rounding_accessors {
    ($struct:ident) => {
        crate::shared_attr!(
            $struct,
            ATTR_KEY_ARITH_ROUNDINGMODE,
            RoundingModeAttr,
            get_attr_arith_roundingmode,
            set_attr_arith_roundingmode
        );
        impl $struct {
            /// Returns the rounding mode, or `None` for the default (to nearest even).
            pub fn rounding_mode(&self, ctx: &Context) -> Option<RoundingMode> {
                self.get_attr_arith_roundingmode(ctx).map(|attr| attr.value)
            }
            /// Replaces the rounding mode.
            pub fn set_rounding_mode(&self, ctx: &Context, mode: RoundingMode) {
                self.set_attr_arith_roundingmode(ctx, RoundingModeAttr::new(mode));
            }
        }
    };
}

/// Accessors for a unit flag stored as a [`BoolAttr`]; absent means `false`.
macro_rules! unit_accessors {
    ($struct:ident, $key:ident, $get:ident, $set:ident, $getter:ident, $setter:ident, $doc:literal) => {
        crate::shared_attr!($struct, $key, BoolAttr, $get, $set);
        impl $struct {
            #[doc = concat!("Returns whether ", $doc, ".")]
            pub fn $getter(&self, ctx: &Context) -> bool {
                self.$get(ctx)
                    .is_some_and(|attr| bool::from((*attr).clone()))
            }
            #[doc = concat!("Sets whether ", $doc, ".")]
            pub fn $setter(&self, ctx: &Context, value: bool) {
                self.$set(ctx, BoolAttr::new(value));
            }
        }
    };
}

macro_rules! nneg_accessors {
    ($struct:ident) => {
        unit_accessors!(
            $struct,
            ATTR_KEY_ARITH_NNEG,
            get_attr_arith_nneg,
            set_attr_arith_nneg,
            is_non_negative,
            set_non_negative,
            "the operand is asserted non-negative (`nneg`; poison otherwise)"
        );
    };
}

macro_rules! binary_common {
    ($struct:ident, $domain:expr) => {
        impl $struct {
            /// Allocates the operation; the result type is the left operand's type.
            ///
            /// Both operands must have the same type. Call `verify` to check it.
            pub fn new(ctx: &mut Context, lhs: Value, rhs: Value) -> Self {
                let ty = lhs.get_type(ctx);
                Self {
                    op: Operation::new(
                        ctx,
                        Self::get_concrete_op_info(),
                        vec![ty],
                        vec![lhs, rhs],
                        vec![],
                        0,
                    ),
                }
            }
            /// Returns the left operand.
            pub fn lhs(&self, ctx: &Context) -> Value {
                self.get_operation().deref(ctx).get_operand(0)
            }
            /// Returns the right operand.
            pub fn rhs(&self, ctx: &Context) -> Value {
                self.get_operation().deref(ctx).get_operand(1)
            }
            /// Returns the result.
            pub fn result(&self, ctx: &Context) -> Value {
                self.get_operation().deref(ctx).get_result(0)
            }
        }
        impl Verify for $struct {
            fn verify(&self, ctx: &Context) -> Result<()> {
                verify_same_type(ctx, self.get_operation(), 2, $domain)
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
        binary_common!($struct, Domain::Float);
        fastmath_accessors!($struct);
    };
    ($struct:ident, $name:literal, $doc:literal, FloatRounding) => {
        #[doc = $doc]
        ///
        /// Operands and result share one floating-point scalar, tensor, or vector type.
        /// An optional rounding mode overrides the default round-to-nearest-even.
        #[pliron_op(
                            name = $name,
                            format = "$0 `,` $1 attr_dict `:` typesig",
                            interfaces = [pliron::builtin::op_interfaces::OneResultInterface]
                        )]
        pub struct $struct;
        binary_common!($struct, Domain::Float);
        fastmath_accessors!($struct);
        rounding_accessors!($struct);
    };
    ($struct:ident, $name:literal, $doc:literal, Exact) => {
        #[doc = $doc]
        ///
        /// Operands and result share one signless-integer or `index` type. With
        /// `exact`, a nonzero discarded remainder (or shifted-out bit) yields poison.
        #[pliron_op(
                            name = $name,
                            format = "$0 `,` $1 attr_dict `:` typesig",
                            interfaces = [pliron::builtin::op_interfaces::OneResultInterface]
                        )]
        pub struct $struct;
        binary_common!($struct, Domain::Integer);
        unit_accessors!(
            $struct,
            ATTR_KEY_ARITH_IS_EXACT,
            get_attr_arith_is_exact,
            set_attr_arith_is_exact,
            is_exact,
            set_exact,
            "the operation is marked `exact`"
        );
    };
    ($struct:ident, $name:literal, $doc:literal, Overflow) => {
        #[doc = $doc]
        ///
        /// Operands and result share one signless-integer or `index` type.
        #[pliron_op(
                            name = $name,
                            format = "$0 `,` $1 attr_dict `:` typesig",
                            interfaces = [pliron::builtin::op_interfaces::OneResultInterface]
                        )]
        pub struct $struct;
        binary_common!($struct, Domain::Integer);
        overflow_accessors!($struct);
    };
    ($struct:ident, $name:literal, $doc:literal, Integer) => {
        #[doc = $doc]
        ///
        /// Operands and result share one signless-integer or `index` type.
        #[pliron_op(
                            name = $name,
                            format = "$0 `,` $1 attr_dict `:` typesig",
                            interfaces = [pliron::builtin::op_interfaces::OneResultInterface]
                        )]
        pub struct $struct;
        binary_common!($struct, Domain::Integer);
    };
}

binary_op!(
    AddFOp,
    "arith.addf",
    "Floating-point addition.",
    FloatRounding
);
binary_op!(
    SubFOp,
    "arith.subf",
    "Floating-point subtraction.",
    FloatRounding
);
binary_op!(
    MulFOp,
    "arith.mulf",
    "Floating-point multiplication.",
    FloatRounding
);
binary_op!(
    DivFOp,
    "arith.divf",
    "Floating-point division.",
    FloatRounding
);
binary_op!(
    RemFOp,
    "arith.remf",
    "Floating-point remainder (C `fmod`).",
    Float
);
binary_op!(
    MaximumNumFOp,
    "arith.maximumnumf",
    "IEEE-754-2019 `maximumNumber`: returns the non-NaN operand and orders `-0.0 < +0.0`.",
    Float
);
binary_op!(
    MinimumNumFOp,
    "arith.minimumnumf",
    "IEEE-754-2019 `minimumNumber`: returns the non-NaN operand and orders `-0.0 < +0.0`.",
    Float
);
binary_op!(
    MaximumFOp,
    "arith.maximumf",
    "IEEE-754 `maximum`: propagates NaN and orders `-0.0 < +0.0`.",
    Float
);
binary_op!(
    MinimumFOp,
    "arith.minimumf",
    "IEEE-754 `minimum`: propagates NaN and orders `-0.0 < +0.0`.",
    Float
);
binary_op!(
    MaxNumFOp,
    "arith.maxnumf",
    "IEEE-754 `maxNum`: returns the non-NaN operand when exactly one is NaN.",
    Float
);
binary_op!(
    MinNumFOp,
    "arith.minnumf",
    "IEEE-754 `minNum`: returns the non-NaN operand when exactly one is NaN.",
    Float
);

binary_op!(AddIOp, "arith.addi", "Wrapping integer addition.", Overflow);
binary_op!(
    SubIOp,
    "arith.subi",
    "Wrapping integer subtraction.",
    Overflow
);
binary_op!(
    MulIOp,
    "arith.muli",
    "Wrapping integer multiplication.",
    Overflow
);
binary_op!(
    ShLIOp,
    "arith.shli",
    "Left shift; shifting by at least the bit width yields poison.",
    Overflow
);
binary_op!(
    DivSIOp,
    "arith.divsi",
    "Signed integer division rounding toward zero.",
    Exact
);
binary_op!(DivUIOp, "arith.divui", "Unsigned integer division.", Exact);
binary_op!(
    CeilDivSIOp,
    "arith.ceildivsi",
    "Signed integer division rounding toward positive infinity.",
    Integer
);
binary_op!(
    CeilDivUIOp,
    "arith.ceildivui",
    "Unsigned integer division rounding toward positive infinity.",
    Integer
);
binary_op!(
    FloorDivSIOp,
    "arith.floordivsi",
    "Signed integer division rounding toward negative infinity.",
    Integer
);
binary_op!(
    RemSIOp,
    "arith.remsi",
    "Signed integer remainder; the result has the sign of the dividend.",
    Integer
);
binary_op!(
    RemUIOp,
    "arith.remui",
    "Unsigned integer remainder.",
    Integer
);
binary_op!(AndIOp, "arith.andi", "Bitwise AND.", Integer);
binary_op!(OrIOp, "arith.ori", "Bitwise OR.", Integer);
binary_op!(XOrIOp, "arith.xori", "Bitwise XOR.", Integer);
binary_op!(
    ShRSIOp,
    "arith.shrsi",
    "Arithmetic (sign-extending) right shift.",
    Exact
);
binary_op!(
    ShRUIOp,
    "arith.shrui",
    "Logical (zero-filling) right shift.",
    Exact
);
binary_op!(MaxSIOp, "arith.maxsi", "Signed integer maximum.", Integer);
binary_op!(MinSIOp, "arith.minsi", "Signed integer minimum.", Integer);
binary_op!(MaxUIOp, "arith.maxui", "Unsigned integer maximum.", Integer);
binary_op!(MinUIOp, "arith.minui", "Unsigned integer minimum.", Integer);

macro_rules! float_unary_op {
    ($struct:ident, $name:literal, $doc:literal) => {
        #[doc = $doc]
        ///
        /// Operand and result share one floating-point scalar, tensor, or vector type.
        #[pliron_op(
                            name = $name,
                            format = "$0 attr_dict `:` typesig",
                            interfaces = [pliron::builtin::op_interfaces::OneResultInterface]
                        )]
        pub struct $struct;
        impl $struct {
            /// Allocates the operation; the result type is the operand's type.
            pub fn new(ctx: &mut Context, operand: Value) -> Self {
                let ty = operand.get_type(ctx);
                Self {
                    op: Operation::new(
                        ctx,
                        Self::get_concrete_op_info(),
                        vec![ty],
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
            /// Returns the result.
            pub fn result(&self, ctx: &Context) -> Value {
                self.get_operation().deref(ctx).get_result(0)
            }
        }
        fastmath_accessors!($struct);
        impl Verify for $struct {
            fn verify(&self, ctx: &Context) -> Result<()> {
                verify_same_type(ctx, self.get_operation(), 1, Domain::Float)
            }
        }
    };
}

float_unary_op!(NegFOp, "arith.negf", "Floating-point negation.");
float_unary_op!(
    FlushDenormalsOp,
    "arith.flush_denormals",
    "Replaces denormal values with a zero of the same sign; other values pass through."
);

/// Verifies an extended binary op: two operands of one integer type and two
/// results. The first result has the operands' type; the second is either an
/// `i1` flag shaped like the operands (`flag_result`) or the operands' type.
fn verify_extended(ctx: &Context, op: Ptr<Operation>, flag_result: bool) -> Result<()> {
    verify_structure(ctx, op, 2, 2)?;
    let lhs = operand_type(ctx, op, 0);
    let (first, second) = {
        let ir = op.deref(ctx);
        (
            ir.get_result(0).get_type(ctx),
            ir.get_result(1).get_type(ctx),
        )
    };
    if operand_type(ctx, op, 1) != lhs || first != lhs {
        return Err(fail(
            ctx,
            op,
            "operands and first result must have the same type",
        ));
    }
    let shaped = classify(ctx, op, lhs)?;
    if !in_domain(ctx, shaped.element, Domain::Integer) {
        return Err(fail(
            ctx,
            op,
            "element type is outside the operation's domain",
        ));
    }
    if flag_result {
        let flag = classify(ctx, op, second)?;
        if flag.element != shaped::i1(ctx) || !flag.same_shape(&shaped) {
            return Err(fail(
                ctx,
                op,
                "second result must be i1 shaped like the operands",
            ));
        }
    } else if second != lhs {
        return Err(fail(ctx, op, "both results must have the operands' type"));
    }
    Ok(())
}

macro_rules! extended_op {
    ($struct:ident, $name:literal, $doc:literal, $first:ident, $second:ident, $flag:literal) => {
        #[doc = $doc]
        #[pliron_op(
                            name = $name,
                            format = "$0 `,` $1 attr_dict `:` typesig",
                            interfaces = [pliron::builtin::op_interfaces::NResultsInterface<2>]
                        )]
        pub struct $struct;
        impl $struct {
            /// Allocates the operation with result types inferred from `lhs`.
            pub fn new(ctx: &mut Context, lhs: Value, rhs: Value) -> Self {
                let ty = lhs.get_type(ctx);
                let second = if $flag {
                    let i1 = shaped::i1(ctx);
                    shaped::classify(ctx, ty)
                        .map(|s| shaped::with_element(ctx, &s, i1))
                        .unwrap_or(i1)
                } else {
                    ty
                };
                Self {
                    op: Operation::new(
                        ctx,
                        Self::get_concrete_op_info(),
                        vec![ty, second],
                        vec![lhs, rhs],
                        vec![],
                        0,
                    ),
                }
            }
            /// Returns the left operand.
            pub fn lhs(&self, ctx: &Context) -> Value {
                self.get_operation().deref(ctx).get_operand(0)
            }
            /// Returns the right operand.
            pub fn rhs(&self, ctx: &Context) -> Value {
                self.get_operation().deref(ctx).get_operand(1)
            }
            /// Returns the first result.
            pub fn $first(&self, ctx: &Context) -> Value {
                self.get_operation().deref(ctx).get_result(0)
            }
            /// Returns the second result.
            pub fn $second(&self, ctx: &Context) -> Value {
                self.get_operation().deref(ctx).get_result(1)
            }
        }
        impl Verify for $struct {
            fn verify(&self, ctx: &Context) -> Result<()> {
                verify_extended(ctx, self.get_operation(), $flag)
            }
        }
    };
}

extended_op!(
    AddUIExtendedOp,
    "arith.addui_extended",
    "Unsigned addition returning the wrapped sum and an `i1` carry (overflow) bit.",
    sum,
    overflow,
    true
);
extended_op!(
    SubUIExtendedOp,
    "arith.subui_extended",
    "Unsigned subtraction returning the wrapped difference and an `i1` borrow bit.",
    diff,
    borrow,
    true
);
extended_op!(
    MulSIExtendedOp,
    "arith.mulsi_extended",
    "Signed full-width multiplication returning the low and high halves of the product.",
    low,
    high,
    false
);
extended_op!(
    MulUIExtendedOp,
    "arith.mului_extended",
    "Unsigned full-width multiplication returning the low and high halves of the product.",
    low,
    high,
    false
);

/// Verifies a comparison: equal operand types in `domain` and an `i1` result of the same shape.
fn verify_compare(ctx: &Context, op: Ptr<Operation>, domain: Domain) -> Result<()> {
    verify_structure(ctx, op, 2, 1)?;
    let lhs = operand_type(ctx, op, 0);
    if operand_type(ctx, op, 1) != lhs {
        return Err(fail(ctx, op, "operands must have the same type"));
    }
    let shaped = classify(ctx, op, lhs)?;
    if !in_domain(ctx, shaped.element, domain) {
        return Err(fail(
            ctx,
            op,
            "element type is outside the operation's domain",
        ));
    }
    let actual = classify(ctx, op, result_type(ctx, op))?;
    if !actual.same_shape(&shaped) || actual.element != shaped::i1(ctx) {
        return Err(fail(ctx, op, "result must be i1 with the operands' shape"));
    }
    Ok(())
}

/// Elementwise floating-point comparison producing `i1`.
#[pliron_op(
    name = "arith.cmpf",
    format = "$0 `,` $1 attr_dict `:` typesig",
    interfaces = [pliron::builtin::op_interfaces::OneResultInterface],
    attributes = (arith_cmpf_predicate: CmpFPredicateAttr)
)]
pub struct CmpFOp;
impl CmpFOp {
    /// Allocates the comparison with an `i1` result shaped like `lhs`.
    ///
    /// When `lhs` is not a recognized scalar or shaped type, a scalar `i1` result is used.
    pub fn new(ctx: &mut Context, predicate: CmpFPredicate, lhs: Value, rhs: Value) -> Self {
        let lhs_ty = lhs.get_type(ctx);
        let i1 = shaped::i1(ctx);
        let ty = shaped::classify(ctx, lhs_ty)
            .map(|s| shaped::with_element(ctx, &s, i1))
            .unwrap_or(i1);
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![ty],
                vec![lhs, rhs],
                vec![],
                0,
            ),
        };
        op.set_attr_arith_cmpf_predicate(ctx, CmpFPredicateAttr::new(predicate));
        op
    }
    /// Returns the comparison predicate, if present.
    pub fn predicate(&self, ctx: &Context) -> Option<CmpFPredicate> {
        self.get_attr_arith_cmpf_predicate(ctx)
            .map(|attr| attr.value)
    }
    /// Returns the left operand.
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    /// Returns the right operand.
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
    /// Returns the `i1` result.
    pub fn result(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }
}
fastmath_accessors!(CmpFOp);
impl Verify for CmpFOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if self.predicate(ctx).is_none() {
            return Err(fail(ctx, self.get_operation(), "missing predicate"));
        }
        verify_compare(ctx, self.get_operation(), Domain::Float)
    }
}

/// Elementwise integer or `index` comparison producing `i1`.
#[pliron_op(
    name = "arith.cmpi",
    format = "$0 `,` $1 attr_dict `:` typesig",
    interfaces = [pliron::builtin::op_interfaces::OneResultInterface],
    attributes = (arith_cmpi_predicate: CmpIPredicateAttr)
)]
pub struct CmpIOp;
impl CmpIOp {
    /// Allocates the comparison with an `i1` result shaped like `lhs`.
    ///
    /// When `lhs` is not a recognized scalar or shaped type, a scalar `i1` result is used.
    pub fn new(ctx: &mut Context, predicate: CmpIPredicate, lhs: Value, rhs: Value) -> Self {
        let lhs_ty = lhs.get_type(ctx);
        let i1 = shaped::i1(ctx);
        let ty = shaped::classify(ctx, lhs_ty)
            .map(|s| shaped::with_element(ctx, &s, i1))
            .unwrap_or(i1);
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![ty],
                vec![lhs, rhs],
                vec![],
                0,
            ),
        };
        op.set_attr_arith_cmpi_predicate(ctx, CmpIPredicateAttr::new(predicate));
        op
    }
    /// Returns the comparison predicate, if present.
    pub fn predicate(&self, ctx: &Context) -> Option<CmpIPredicate> {
        self.get_attr_arith_cmpi_predicate(ctx)
            .map(|attr| attr.value)
    }
    /// Returns the left operand.
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    /// Returns the right operand.
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
    /// Returns the `i1` result.
    pub fn result(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }
}
impl Verify for CmpIOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if self.predicate(ctx).is_none() {
            return Err(fail(ctx, self.get_operation(), "missing predicate"));
        }
        verify_compare(ctx, self.get_operation(), Domain::Integer)
    }
}

/// Chooses between two values with an `i1` condition.
///
/// The condition is either a scalar `i1`, selecting whole values, or an `i1`
/// container with the same shape as the operands, selecting per element.
#[pliron_op(
    name = "arith.select",
    format = "$0 `,` $1 `,` $2 attr_dict `:` typesig",
    interfaces = [pliron::builtin::op_interfaces::OneResultInterface]
)]
pub struct SelectOp;
impl SelectOp {
    /// Allocates the selection; the result type is `true_value`'s type.
    pub fn new(ctx: &mut Context, condition: Value, true_value: Value, false_value: Value) -> Self {
        let ty = true_value.get_type(ctx);
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![ty],
                vec![condition, true_value, false_value],
                vec![],
                0,
            ),
        }
    }
    /// Returns the `i1` condition.
    pub fn condition(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    /// Returns the value chosen where the condition is true.
    pub fn true_value(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
    /// Returns the value chosen where the condition is false.
    pub fn false_value(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(2)
    }
    /// Returns the result.
    pub fn result(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }
}
impl Verify for SelectOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        verify_structure(ctx, op, 3, 1)?;
        let result = result_type(ctx, op);
        if operand_type(ctx, op, 1) != result || operand_type(ctx, op, 2) != result {
            return Err(fail(
                ctx,
                op,
                "selected values and result must have the same type",
            ));
        }
        let condition = operand_type(ctx, op, 0);
        if condition == shaped::i1(ctx) {
            return Ok(());
        }
        let value = classify(ctx, op, result)?;
        let condition = classify(ctx, op, condition)?;
        if condition.element != shaped::i1(ctx) || !condition.same_shape(&value) {
            return Err(fail(
                ctx,
                op,
                "condition must be i1 or an i1 container shaped like the result",
            ));
        }
        Ok(())
    }
}

/// Element conversion rule of a cast operation.
type CastRule = fn(&Context, TypeHandle, TypeHandle) -> bool;

fn float_extend(ctx: &Context, from: TypeHandle, to: TypeHandle) -> bool {
    matches!((shaped::float_width(ctx, from), shaped::float_width(ctx, to)), (Some(a), Some(b)) if b > a)
}
fn float_truncate(ctx: &Context, from: TypeHandle, to: TypeHandle) -> bool {
    matches!((shaped::float_width(ctx, from), shaped::float_width(ctx, to)), (Some(a), Some(b)) if b < a)
}
fn float_convert(ctx: &Context, from: TypeHandle, to: TypeHandle) -> bool {
    from != to
        && matches!((shaped::float_width(ctx, from), shaped::float_width(ctx, to)), (Some(a), Some(b)) if a == b)
}
fn int_extend(ctx: &Context, from: TypeHandle, to: TypeHandle) -> bool {
    matches!((shaped::signless_width(ctx, from), shaped::signless_width(ctx, to)), (Some(a), Some(b)) if b > a)
}
fn int_truncate(ctx: &Context, from: TypeHandle, to: TypeHandle) -> bool {
    matches!((shaped::signless_width(ctx, from), shaped::signless_width(ctx, to)), (Some(a), Some(b)) if b < a)
}
fn int_to_float(ctx: &Context, from: TypeHandle, to: TypeHandle) -> bool {
    shaped::signless_width(ctx, from).is_some() && shaped::is_float(ctx, to)
}
fn float_to_int(ctx: &Context, from: TypeHandle, to: TypeHandle) -> bool {
    shaped::is_float(ctx, from) && shaped::signless_width(ctx, to).is_some()
}
fn index_cast(ctx: &Context, from: TypeHandle, to: TypeHandle) -> bool {
    (shaped::is_index(ctx, from) && shaped::signless_width(ctx, to).is_some())
        || (shaped::signless_width(ctx, from).is_some() && shaped::is_index(ctx, to))
}
fn bit_cast(ctx: &Context, from: TypeHandle, to: TypeHandle) -> bool {
    matches!((shaped::scalar_width(ctx, from), shaped::scalar_width(ctx, to)), (Some(a), Some(b)) if a == b)
}

fn verify_cast(ctx: &Context, op: Ptr<Operation>, rule: CastRule) -> Result<()> {
    verify_structure(ctx, op, 1, 1)?;
    let input = classify(ctx, op, operand_type(ctx, op, 0))?;
    let output = classify(ctx, op, result_type(ctx, op))?;
    if !input.same_shape(&output) {
        return Err(fail(ctx, op, "operand and result shapes must match"));
    }
    if !rule(ctx, input.element, output.element) {
        return Err(fail(
            ctx,
            op,
            "element types are not a valid conversion for this operation",
        ));
    }
    Ok(())
}

macro_rules! cast_op {
    ($struct:ident, $name:literal, $doc:literal, $rule:expr) => {
        #[doc = $doc]
        ///
        /// Operand and result have the same container shape; only the element type changes.
        #[pliron_op(
                            name = $name,
                            format = "$0 attr_dict `:` typesig",
                            interfaces = [pliron::builtin::op_interfaces::OneResultInterface]
                        )]
        pub struct $struct;
        cast_common!($struct, $name, $rule);
    };
}

macro_rules! cast_common {
    ($struct:ident, $name:literal, $rule:expr) => {
        impl $struct {
            /// Allocates the conversion with an explicit result type, without verification.
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
            /// Allocates the conversion to `element`, keeping the input's container shape.
            ///
            /// # Errors
            ///
            /// Returns an error when the input is not a scalar or shaped scalar type.
            pub fn new_inferred(
                ctx: &mut Context,
                input: Value,
                element: TypeHandle,
            ) -> Result<Self> {
                let shaped = shaped::classify(ctx, input.get_type(ctx)).ok_or_else(|| {
                    pliron::result::Error::from(Diagnostic::InvalidOperation {
                        operation: $name.to_owned(),
                        location: "<builder>".to_owned(),
                        reason: "input must be a scalar or a tensor or vector of scalars",
                    })
                })?;
                let ty = shaped::with_element(ctx, &shaped, element);
                Ok(Self::new(ctx, input, ty))
            }
            /// Returns the converted operand.
            pub fn input(&self, ctx: &Context) -> Value {
                self.get_operation().deref(ctx).get_operand(0)
            }
            /// Returns the result.
            pub fn result(&self, ctx: &Context) -> Value {
                self.get_operation().deref(ctx).get_result(0)
            }
        }
        impl Verify for $struct {
            fn verify(&self, ctx: &Context) -> Result<()> {
                verify_cast(ctx, self.get_operation(), $rule)
            }
        }
    };
}

cast_op!(
    ExtFOp,
    "arith.extf",
    "Widens a floating-point value exactly.",
    float_extend
);
fastmath_accessors!(ExtFOp);
cast_op!(
    TruncFOp,
    "arith.truncf",
    "Narrows a floating-point value, rounding to nearest even unless a rounding mode is set.",
    float_truncate
);
fastmath_accessors!(TruncFOp);
rounding_accessors!(TruncFOp);
cast_op!(
    ConvertFOp,
    "arith.convertf",
    "Converts between distinct floating-point types of equal width, such as `f16` and `bf16`.",
    float_convert
);
fastmath_accessors!(ConvertFOp);
rounding_accessors!(ConvertFOp);
cast_op!(
    ExtSIOp,
    "arith.extsi",
    "Sign-extends an integer.",
    int_extend
);
cast_op!(
    ExtUIOp,
    "arith.extui",
    "Zero-extends an integer.",
    int_extend
);
nneg_accessors!(ExtUIOp);
cast_op!(
    TruncIOp,
    "arith.trunci",
    "Truncates an integer to its low-order bits.",
    int_truncate
);
overflow_accessors!(TruncIOp);
cast_op!(
    SIToFPOp,
    "arith.sitofp",
    "Converts a signed integer to floating point.",
    int_to_float
);
cast_op!(
    UIToFPOp,
    "arith.uitofp",
    "Converts an unsigned integer to floating point.",
    int_to_float
);
nneg_accessors!(UIToFPOp);
cast_op!(
    FPToSIOp,
    "arith.fptosi",
    "Converts floating point to a signed integer, rounding toward zero.",
    float_to_int
);
cast_op!(
    FPToUIOp,
    "arith.fptoui",
    "Converts floating point to an unsigned integer, rounding toward zero.",
    float_to_int
);
cast_op!(
    IndexCastOp,
    "arith.index_cast",
    "Converts between `index` and a signless integer, sign-extending or truncating.",
    index_cast
);
cast_op!(
    IndexCastUIOp,
    "arith.index_castui",
    "Converts between `index` and a signless integer, zero-extending or truncating.",
    index_cast
);
nneg_accessors!(IndexCastUIOp);
cast_op!(
    BitcastOp,
    "arith.bitcast",
    "Reinterprets the bits of an integer or floating-point value of equal width.",
    bit_cast
);

/// Verifies a scaling cast: `in` and `scale` are floats of one shape, and the
/// result has that shape with an element accepted by `rule`.
fn verify_scaling(ctx: &Context, op: Ptr<Operation>, rule: CastRule) -> Result<()> {
    verify_structure(ctx, op, 2, 1)?;
    let input = classify(ctx, op, operand_type(ctx, op, 0))?;
    let scale = classify(ctx, op, operand_type(ctx, op, 1))?;
    let output = classify(ctx, op, result_type(ctx, op))?;
    if !input.same_shape(&scale) || !input.same_shape(&output) {
        return Err(fail(ctx, op, "input, scale, and result shapes must match"));
    }
    if !shaped::is_float(ctx, scale.element) || !rule(ctx, input.element, output.element) {
        return Err(fail(
            ctx,
            op,
            "element types are not a valid scaling conversion for this operation",
        ));
    }
    Ok(())
}

macro_rules! scaling_op {
    ($struct:ident, $name:literal, $doc:literal, $rule:expr) => {
        #[doc = $doc]
        ///
        /// Operands are `in` and a per-element `scale` of the same shape (scales
        /// computed per block must be broadcast first), following the OCP MX spec.
        #[pliron_op(
                            name = $name,
                            format = "$0 `,` $1 attr_dict `:` typesig",
                            interfaces = [pliron::builtin::op_interfaces::OneResultInterface]
                        )]
        pub struct $struct;
        impl $struct {
            /// Allocates the conversion with an explicit result type, without verification.
            pub fn new(ctx: &mut Context, input: Value, scale: Value, result: TypeHandle) -> Self {
                Self {
                    op: Operation::new(
                        ctx,
                        Self::get_concrete_op_info(),
                        vec![result],
                        vec![input, scale],
                        vec![],
                        0,
                    ),
                }
            }
            /// Returns the converted operand.
            pub fn input(&self, ctx: &Context) -> Value {
                self.get_operation().deref(ctx).get_operand(0)
            }
            /// Returns the scale operand.
            pub fn scale(&self, ctx: &Context) -> Value {
                self.get_operation().deref(ctx).get_operand(1)
            }
            /// Returns the result.
            pub fn result(&self, ctx: &Context) -> Value {
                self.get_operation().deref(ctx).get_result(0)
            }
        }
        fastmath_accessors!($struct);
        impl Verify for $struct {
            fn verify(&self, ctx: &Context) -> Result<()> {
                verify_scaling(ctx, self.get_operation(), $rule)
            }
        }
    };
}

scaling_op!(
    ScalingExtFOp,
    "arith.scaling_extf",
    "Widens `in` and multiplies it by the widened `scale`.",
    float_extend
);
scaling_op!(
    ScalingTruncFOp,
    "arith.scaling_truncf",
    "Divides `in` by `scale` and narrows the quotient.",
    float_truncate
);
rounding_accessors!(ScalingTruncFOp);

/// Materializes a scalar, tensor, or vector constant.
///
/// The `value` attribute is one of:
/// - [`ScalarBitsAttr`] whose type equals the result type (integer or float scalar),
/// - [`IndexAttr`] with an `index` result,
/// - [`BoolAttr`] with an `i1` result,
/// - [`DenseElementsAttr`] whose shaped type equals the result type.
#[pliron_op(
    name = "arith.constant",
    format = "attr_dict `:` typesig",
    interfaces = [pliron::builtin::op_interfaces::OneResultInterface],
    attributes = (arith_constant_value)
)]
pub struct ConstantOp;
impl ConstantOp {
    /// Allocates a constant from an arbitrary value attribute, without verification.
    pub fn new(ctx: &mut Context, value: AttrObj, result: TypeHandle) -> Self {
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
        op.set_attr_arith_constant_value(ctx, value);
        op
    }
    /// Allocates a scalar integer constant of a signless integer `element`.
    ///
    /// # Errors
    ///
    /// Returns an error when `element` is not an integer type or `value` does not fit it.
    pub fn integer(ctx: &mut Context, element: ElementType, value: IntegerValue) -> Result<Self> {
        let attr = ScalarBitsAttr::integer(ctx, element, value)?;
        let ty = attr.ty;
        Ok(Self::new(ctx, Box::new(attr), ty))
    }
    /// Allocates an `index` constant.
    pub fn index(ctx: &mut Context, value: i64) -> Self {
        let ty = shaped::index(ctx);
        Self::new(ctx, Box::new(IndexAttr { value }), ty)
    }
    /// Allocates an `i1` constant.
    pub fn bool(ctx: &mut Context, value: bool) -> Self {
        let ty = shaped::i1(ctx);
        Self::new(ctx, Box::new(BoolAttr::new(value)), ty)
    }
    /// Allocates an `f32` constant from its exact value.
    pub fn f32(ctx: &mut Context, value: f32) -> Self {
        let attr = ScalarBitsAttr::f32_bits(ctx, value.to_bits());
        let ty = attr.ty;
        Self::new(ctx, Box::new(attr), ty)
    }
    /// Allocates an `f64` constant from its exact value.
    pub fn f64(ctx: &mut Context, value: f64) -> Self {
        let attr = ScalarBitsAttr::float_bits(ctx, value.to_bits());
        let ty = attr.ty;
        Self::new(ctx, Box::new(attr), ty)
    }
    /// Allocates a tensor or vector constant from dense elements.
    pub fn dense(ctx: &mut Context, value: DenseElementsAttr) -> Self {
        let ty = value.ty;
        Self::new(ctx, Box::new(value), ty)
    }
    /// Returns a copy of the value attribute, if present.
    pub fn value(&self, ctx: &Context) -> Option<AttrObj> {
        self.get_attr_arith_constant_value(ctx)
            .map(|attr| (*attr).clone())
    }
    /// Returns the result.
    pub fn result(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }
}
impl Verify for ConstantOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let op = self.get_operation();
        verify_structure(ctx, op, 0, 1)?;
        let result = result_type(ctx, op);
        let Some(value) = self.get_attr_arith_constant_value(ctx) else {
            return Err(fail(ctx, op, "missing value attribute"));
        };
        let matches = if let Some(bits) = value.downcast_ref::<ScalarBitsAttr>() {
            bits.ty == result
        } else if value.downcast_ref::<IndexAttr>().is_some() {
            shaped::is_index(ctx, result)
        } else if value.downcast_ref::<BoolAttr>().is_some() {
            result == shaped::i1(ctx)
        } else if let Some(dense) = value.downcast_ref::<DenseElementsAttr>() {
            dense.ty == result
        } else {
            return Err(fail(ctx, op, "unsupported constant value attribute"));
        };
        if !matches {
            return Err(fail(ctx, op, "value type must equal the result type"));
        }
        Ok(())
    }
}
