//! Private operation definitions and their generated attribute storage.

use crate::{
    Diagnostic,
    attributes::*,
    inference::{self, Domain, Shaped},
};
use pliron::{
    attribute::AttrObj,
    builtin::attributes::{BoolAttr, IntegerAttr, VecAttr},
    common_traits::Verify,
    context::Context,
    derive::pliron_op,
    op::Op,
    operation::Operation,
    printable::Printable,
    result::Result,
    r#type::{TypeHandle, Typed},
    utils::apint::{APInt, bw},
    value::Value,
};
use stablehlo::types::ElementType;

fn invalid(op: &'static str, why: &'static str) -> pliron::result::Error {
    inference::invalid(op, why)
}
/// Verification error carrying the operation's source location.
fn fail(
    ctx: &Context,
    ir: &Operation,
    op: &'static str,
    why: &'static str,
) -> pliron::result::Error {
    inference::invalid_at(op, ir.loc().disp(ctx).to_string(), why)
}
fn domain_error(operation: &'static str) -> pliron::result::Error {
    Diagnostic::InvalidDomain { operation }.into()
}
fn failure_with_source(
    ctx: &Context,
    ir: &Operation,
    operation: &'static str,
    reason: &'static str,
    source: pliron::result::Error,
) -> pliron::result::Error {
    Diagnostic::OperationFailure {
        operation,
        location: ir.loc().disp(ctx).to_string(),
        reason,
        source: Box::new(source),
    }
    .into()
}
fn verify_pure_structure(ir: &Operation, operation: &'static str) -> Result<()> {
    if ir.num_regions() != 0 || ir.get_num_successors() != 0 {
        return Err(invalid(
            operation,
            "pure operation must not have regions or successors",
        ));
    }
    Ok(())
}
/// Arity plus region/successor guard for pure single-result operations.
fn verify_pure_shape(
    ctx: &Context,
    ir: &Operation,
    operation: &'static str,
    operands: usize,
    results: usize,
) -> Result<()> {
    if ir.get_num_operands() != operands || ir.get_num_results() != results {
        return Err(fail(ctx, ir, operation, "operand or result count mismatch"));
    }
    if ir.num_regions() != 0 || ir.get_num_successors() != 0 {
        return Err(fail(
            ctx,
            ir,
            operation,
            "pure operation must not have regions or successors",
        ));
    }
    Ok(())
}
fn make(ctx: &Context, shape: Vec<i64>, el: ElementType) -> TypeHandle {
    inference::ty(ctx, shape, el)
}
fn tensor_info(ctx: &Context, ty: TypeHandle) -> Option<inference::TensorInfo> {
    inference::tensor_info(ctx, ty)
}
fn compatible_tensors(ctx: &Context, lhs: TypeHandle, rhs: TypeHandle) -> bool {
    inference::shaped(ctx, lhs).is_some()
        && inference::shaped(ctx, rhs).is_some()
        && inference::compatible_types(ctx, lhs, rhs)
}
/// Shaped view of a value whose type satisfies an upstream tensor constraint.
fn typed(
    ctx: &Context,
    ty: TypeHandle,
    domain: Domain,
    ranked_only: bool,
    operation: &'static str,
) -> Result<Shaped> {
    match inference::shaped(ctx, ty) {
        Some(s) if inference::hlo_tensor_type(ctx, ty, domain, ranked_only) => Ok(s),
        _ => Err(domain_error(operation)),
    }
}
fn i1(ctx: &Context) -> TypeHandle {
    ElementType::I1.native(ctx)
}

/// Result rule of a CHLO unary elementwise operation.
#[derive(Clone, Copy)]
enum UnaryResult {
    /// `HLO_CompatibleOperandsAndResultType`; result type inferred from the operand.
    Compatible,
    /// `IsInf`-like: same shape and encoding with an `i1` element.
    Pred,
}
fn infer_unary(
    ctx: &Context,
    operand: TypeHandle,
    operation: &'static str,
    domain: Domain,
    rule: UnaryResult,
) -> Result<TypeHandle> {
    typed(ctx, operand, domain, false, operation)?;
    let inferred = match rule {
        UnaryResult::Compatible => inference::most_specific(ctx, &[operand]),
        UnaryResult::Pred => inference::same_shape(ctx, operand, i1(ctx)),
    };
    inferred.ok_or_else(|| invalid(operation, "operand type is not inferable"))
}
fn check_unary(
    ctx: &Context,
    op: &impl Op,
    operation: &'static str,
    domain: Domain,
    rule: UnaryResult,
) -> Result<()> {
    let ir = op.get_operation().deref(ctx);
    verify_pure_shape(ctx, &ir, operation, 1, 1)?;
    let operand_ty = ir.get_operand(0).get_type(ctx);
    let result_ty = ir.get_result(0).get_type(ctx);
    let operand = typed(ctx, operand_ty, domain, false, operation)?;
    let result_domain = match rule {
        UnaryResult::Compatible => domain,
        UnaryResult::Pred => Domain::Pred,
    };
    let result = typed(ctx, result_ty, result_domain, false, operation)?;
    if !inference::compatible_shape(&operand, &result) {
        return Err(fail(
            ctx,
            &ir,
            operation,
            "operand and result shapes must be compatible",
        ));
    }
    let inferred = infer_unary(ctx, operand_ty, operation, domain, rule)?;
    let compatible = match rule {
        UnaryResult::Compatible => inference::compatible_types(ctx, operand_ty, result_ty),
        UnaryResult::Pred => true,
    };
    if !compatible || !inference::compatible_types(ctx, inferred, result_ty) {
        return Err(fail(
            ctx,
            &ir,
            operation,
            "result type is incompatible with the inferred type",
        ));
    }
    Ok(())
}
macro_rules! unary_op {
    ($struct:ident,$name:literal,$doc:literal,$domain:expr,$rule:expr) => {
        #[doc = $doc]
        ///
        /// Applies independently to each tensor element, preserving shape and
        /// encoding. Infinity predicates produce `i1` elements; other operations
        /// preserve the operand element type. Ranked and unranked tensors are accepted.
        #[pliron_op(name=$name,format="$0 attr_dict `:` typesig",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
        pub struct $struct;
        impl $struct {
            /// Allocates an operation with an explicit result type, without verification.
            ///
            /// The result must be compatible with the operand's shape and the
            /// operation's element-type rule. Call `verify` to check that contract.
            pub fn new(ctx: &mut Context, input: Value, result: TypeHandle) -> Self {
                Self { op: Operation::new(ctx, Self::get_concrete_op_info(), vec![result], vec![input], vec![], 0) }
            }
            /// Allocates an operation after inferring its result from the input tensor.
            ///
            /// # Errors
            ///
            /// Returns an error when the operand is not a tensor in this operation's
            /// element domain or its result type cannot be inferred.
            pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
                let ty = infer_unary(ctx, input.get_type(ctx), $name, $domain, $rule)?;
                Ok(Self::new(ctx, input, ty))
            }
            /// Returns the tensor operand to which the elementwise operation applies.
            pub fn operand(&self, ctx: &Context) -> Value { self.get_operation().deref(ctx).get_operand(0) }
            /// Returns the input tensor; an alias of [`Self::operand`].
            pub fn input(&self, ctx: &Context) -> Value { self.operand(ctx) }
            /// Returns the single output tensor.
            pub fn result(&self, ctx: &Context) -> Value { self.get_operation().deref(ctx).get_result(0) }
        }
        impl Verify for $struct {
            fn verify(&self, ctx: &Context) -> Result<()> { check_unary(ctx, self, $name, $domain, $rule) }
        }
    };
}
use UnaryResult::{Compatible as Same, Pred as IsPred};
unary_op!(
    AsinAcosKernelOp,
    "chlo._asin_acos_kernel",
    "Shared complex-valued kernel used when expanding inverse sine and cosine.",
    Domain::Complex,
    Same
);
unary_op!(
    AcosOp,
    "chlo.acos",
    "Inverse cosine of floating-point or complex elements.",
    Domain::FloatOrComplex,
    Same
);
unary_op!(
    AcoshOp,
    "chlo.acosh",
    "Inverse hyperbolic cosine of floating-point or complex elements.",
    Domain::FloatOrComplex,
    Same
);
unary_op!(
    AsinOp,
    "chlo.asin",
    "Inverse sine of floating-point or complex elements.",
    Domain::FloatOrComplex,
    Same
);
unary_op!(
    AsinhOp,
    "chlo.asinh",
    "Inverse hyperbolic sine of floating-point or complex elements.",
    Domain::FloatOrComplex,
    Same
);
unary_op!(
    AtanOp,
    "chlo.atan",
    "Inverse tangent of floating-point or complex elements.",
    Domain::FloatOrComplex,
    Same
);
unary_op!(
    AtanhOp,
    "chlo.atanh",
    "Inverse hyperbolic tangent of floating-point or complex elements.",
    Domain::FloatOrComplex,
    Same
);
unary_op!(
    BesselI1eOp,
    "chlo.bessel_i1e",
    "Exponentially scaled modified Bessel function of the first kind, order one, on floating-point or complex elements.",
    Domain::FloatOrComplex,
    Same
);
unary_op!(
    ConjOp,
    "chlo.conj",
    "Complex conjugation, leaving real floating-point elements unchanged.",
    Domain::FloatOrComplex,
    Same
);
unary_op!(
    CoshOp,
    "chlo.cosh",
    "Hyperbolic cosine of floating-point or complex elements.",
    Domain::FloatOrComplex,
    Same
);
unary_op!(
    SinhOp,
    "chlo.sinh",
    "Hyperbolic sine of floating-point or complex elements.",
    Domain::FloatOrComplex,
    Same
);
unary_op!(
    TanOp,
    "chlo.tan",
    "Tangent of floating-point or complex elements.",
    Domain::FloatOrComplex,
    Same
);
unary_op!(
    DigammaOp,
    "chlo.digamma",
    "Logarithmic derivative of the gamma function on floating-point elements.",
    Domain::Float,
    Same
);
unary_op!(
    ErfOp,
    "chlo.erf",
    "Error function on floating-point elements.",
    Domain::Float,
    Same
);
unary_op!(
    ErfInvOp,
    "chlo.erf_inv",
    "Inverse error function on floating-point elements.",
    Domain::Float,
    Same
);
unary_op!(
    ErfcOp,
    "chlo.erfc",
    "Complementary error function on floating-point elements.",
    Domain::Float,
    Same
);
unary_op!(
    LgammaOp,
    "chlo.lgamma",
    "Logarithm of the absolute value of the gamma function on floating-point elements.",
    Domain::Float,
    Same
);
unary_op!(
    SquareOp,
    "chlo.square",
    "Elementwise squaring of floating-point or complex elements.",
    Domain::FloatOrComplex,
    Same
);
unary_op!(
    IsInfOp,
    "chlo.is_inf",
    "Predicate identifying either signed infinity in floating-point elements.",
    Domain::Float,
    IsPred
);
unary_op!(
    IsNegInfOp,
    "chlo.is_neg_inf",
    "Predicate identifying negative infinity in floating-point elements.",
    Domain::Float,
    IsPred
);
unary_op!(
    IsPosInfOp,
    "chlo.is_pos_inf",
    "Predicate identifying positive infinity in floating-point elements.",
    Domain::Float,
    IsPred
);

/// `HLO_CompatibleOperandsAndResultType` inference over `inferMostSpecificType`.
fn infer_compatible(
    ctx: &Context,
    operands: &[TypeHandle; 2],
    operation: &'static str,
    domain: Domain,
    ranked_only: bool,
) -> Result<TypeHandle> {
    let [lhs, rhs] = operands;
    typed(ctx, *lhs, domain, ranked_only, operation)?;
    typed(ctx, *rhs, domain, ranked_only, operation)?;
    if !inference::compatible_types(ctx, *lhs, *rhs) {
        return Err(invalid(operation, "operand types must be compatible"));
    }
    inference::most_specific(ctx, operands)
        .ok_or_else(|| invalid(operation, "operand shapes are incompatible"))
}
fn check_compatible(
    ctx: &Context,
    op: &impl Op,
    operation: &'static str,
    domain: Domain,
    ranked_only: bool,
) -> Result<()> {
    let ir = op.get_operation().deref(ctx);
    verify_pure_shape(ctx, &ir, operation, 2, 1)?;
    let operands = [
        ir.get_operand(0).get_type(ctx),
        ir.get_operand(1).get_type(ctx),
    ];
    let result = ir.get_result(0).get_type(ctx);
    typed(ctx, result, domain, ranked_only, operation)?;
    let inferred =
        infer_compatible(ctx, &operands, operation, domain, ranked_only).map_err(|error| {
            failure_with_source(ctx, &ir, operation, "operand types are incompatible", error)
        })?;
    if operands
        .iter()
        .chain([&inferred])
        .any(|&ty| !inference::compatible_types(ctx, ty, result))
    {
        return Err(fail(
            ctx,
            &ir,
            operation,
            "operands and result must have compatible types",
        ));
    }
    Ok(())
}
macro_rules! binary_op {
    ($struct:ident,$name:literal,$doc:literal,$domain:expr,$ranked:expr) => {
        #[doc = $doc]
        ///
        /// This operation does not broadcast: operand and result shapes and
        /// element types must be compatible.
        #[pliron_op(name=$name,format="$0 `,` $1 attr_dict `:` typesig",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
        pub struct $struct;
        impl $struct {
            /// Allocates an operation with the supplied result type, without verification.
            ///
            /// Call `verify` to check operand domains and compatibility of all types.
            pub fn new(ctx: &mut Context, lhs: Value, rhs: Value, result: TypeHandle) -> Self {
                Self { op: Operation::new(ctx, Self::get_concrete_op_info(), vec![result], vec![lhs, rhs], vec![], 0) }
            }
            /// Infers the most specific compatible operand type and allocates the operation.
            ///
            /// # Errors
            ///
            /// Returns an error for operands outside the operation's domain, a
            /// rank requirement violation, or incompatible element types or shapes.
            pub fn new_inferred(ctx: &mut Context, lhs: Value, rhs: Value) -> Result<Self> {
                let ty = infer_compatible(ctx, &[lhs.get_type(ctx), rhs.get_type(ctx)], $name, $domain, $ranked)?;
                Ok(Self::new(ctx, lhs, rhs, ty))
            }
            /// Returns the left-hand operand.
            pub fn lhs(&self, ctx: &Context) -> Value { self.get_operation().deref(ctx).get_operand(0) }
            /// Returns the right-hand operand.
            pub fn rhs(&self, ctx: &Context) -> Value { self.get_operation().deref(ctx).get_operand(1) }
        }
        impl Verify for $struct {
            fn verify(&self, ctx: &Context) -> Result<()> { check_compatible(ctx, self, $name, $domain, $ranked) }
        }
    };
}
binary_op!(
    NextAfterOp,
    "chlo.next_after",
    "Next representable floating-point value after `lhs` in the direction of `rhs`; accepts ranked or unranked tensors.",
    Domain::Float,
    false
);
binary_op!(
    PolygammaOp,
    "chlo.polygamma",
    "Polygamma function with floating-point derivative order in `lhs` and argument in `rhs`; accepts ranked or unranked tensors.",
    Domain::Float,
    false
);
binary_op!(
    ZetaOp,
    "chlo.zeta",
    "Hurwitz zeta function with floating-point exponent in `lhs` and shift in `rhs`; accepts ranked or unranked tensors.",
    Domain::Float,
    false
);
binary_op!(
    MulhiOp,
    "chlo.mulhi",
    "High half of an integer multiplication on ranked integer tensors.",
    Domain::Int,
    true
);

/// Element-type trait attached to a broadcasting binary operation.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ElementTrait {
    /// `SameOperandsAndResultElementType` (exact identity).
    Same,
    /// `HLO_CompatibleOperandsAndResultElementType`.
    Compatible,
    None,
}
fn broadcast_dims_attr(
    ctx: &Context,
    ir: &Operation,
    operation: &'static str,
    attr: Option<Option<Vec<i64>>>,
) -> Result<Option<Vec<i64>>> {
    match attr {
        None => Ok(None),
        Some(list) => list
            .map(Some)
            .ok_or_else(|| fail(ctx, ir, operation, "malformed broadcast_dimensions")),
    }
}
/// Upstream `InferBroadcastBinaryOpReturnTypeComponents`.
fn infer_broadcast_binary(
    ctx: &Context,
    lhs: &Shaped,
    rhs: &Shaped,
    dimensions: Option<&[i64]>,
    element: Option<TypeHandle>,
    operation: &'static str,
) -> Result<TypeHandle> {
    if !inference::compatible_elements(ctx, lhs.element, rhs.element) {
        return Err(invalid(operation, "mismatched operand types"));
    }
    Ok(inference::broadcast_type(
        ctx,
        lhs,
        rhs,
        dimensions,
        element.unwrap_or(lhs.element),
    ))
}
#[allow(clippy::too_many_arguments)]
fn check_broadcast(
    ctx: &Context,
    ir: &Operation,
    operation: &'static str,
    operand_domain: Domain,
    result_domain: Domain,
    element_trait: ElementTrait,
    dimensions: Option<&[i64]>,
    element: Option<TypeHandle>,
) -> Result<()> {
    verify_pure_shape(ctx, ir, operation, 2, 1)?;
    let lhs = typed(
        ctx,
        ir.get_operand(0).get_type(ctx),
        operand_domain,
        false,
        operation,
    )?;
    let rhs = typed(
        ctx,
        ir.get_operand(1).get_type(ctx),
        operand_domain,
        false,
        operation,
    )?;
    let result_ty = ir.get_result(0).get_type(ctx);
    let result = typed(ctx, result_ty, result_domain, false, operation)?;
    let elements_ok = match element_trait {
        ElementTrait::Same => lhs.element == rhs.element && lhs.element == result.element,
        ElementTrait::Compatible => {
            inference::compatible_elements(ctx, lhs.element, rhs.element)
                && inference::compatible_elements(ctx, lhs.element, result.element)
        }
        ElementTrait::None => true,
    };
    if !elements_ok {
        return Err(fail(
            ctx,
            ir,
            operation,
            "operand and result element types violate the element type trait",
        ));
    }
    let inferred = infer_broadcast_binary(ctx, &lhs, &rhs, dimensions, element, operation)
        .map_err(|error| {
            failure_with_source(ctx, ir, operation, "mismatched operand types", error)
        })?;
    if !inference::compatible_types(ctx, inferred, result_ty) {
        return Err(fail(
            ctx,
            ir,
            operation,
            "result type is incompatible with the inferred broadcast type",
        ));
    }
    Ok(())
}
macro_rules! broadcast_op {
    ($struct:ident,$name:literal,$doc:literal,$attr:literal,$attr_module:ident,$attr_key:ident,$setter:ident,$getter:ident,$domain:expr,$result_domain:expr,$trait:expr) => {
        #[doc = $doc]
        ///
        /// With no explicit dimensions, operands use trailing-axis NumPy broadcasting.
        /// Explicit dimensions map the lower-rank operand's axes into the higher-rank
        /// operand (the left operand is mapped when ranks are equal). Inference can
        /// yield an unranked result when the broadcast shape cannot be determined.
        ///
        #[doc = concat!("The optional axis mapping is stored as `", $attr, "` with type [`VecAttr`].")]
        // pliron's attribute-table generator stringifies long token paths with
        // embedded newlines, producing invalid Markdown links. Keep these
        // accessors local so documentation uses the actual identifier tokens.
        #[pliron_op(name=$name,format="$0 `,` $1 attr_dict `:` typesig",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
        pub struct $struct;
        mod $attr_module {
            pliron::dict_key!($attr_key, $attr);
        }
        impl $struct {
            #[doc = concat!("Reads `", $attr, "`.")]
            ///
            /// Borrows the explicit broadcast axis mapping attribute.
            ///
            /// Returns `None` when the attribute is absent or is not a `VecAttr`.
            /// The returned borrow holds the containing operation immutably;
            /// drop it before mutating that operation.
            pub fn $getter<'a>(&self, ctx: &'a Context) -> Option<std::cell::Ref<'a, VecAttr>> {
                std::cell::Ref::filter_map(self.op.deref(ctx), |op|
                    op.attributes.get::<VecAttr>(&$attr_module::$attr_key)).ok()
            }
            #[doc = concat!("Replaces `", $attr, "`.")]
            ///
            /// Stores an explicit broadcast axis mapping without validating its entries.
            ///
            /// The list maps lower-rank operand axes into the higher-rank operand;
            /// use `verify` to check the operation after modifying it.
            pub fn $setter(&self, ctx: &Context, value: VecAttr) {
                self.op.deref_mut(ctx).attributes.set($attr_module::$attr_key.clone(), value);
            }
            /// Allocates an operation with the supplied output type, without verification.
            ///
            /// `dimensions` is an optional lower-rank-to-higher-rank axis mapping;
            /// `None` selects NumPy broadcasting. Call `verify` before using the IR.
            pub fn new(ctx: &mut Context, lhs: Value, rhs: Value, result: TypeHandle, dimensions: Option<Vec<i64>>) -> Self {
                let op = Self { op: Operation::new(ctx, Self::get_concrete_op_info(), vec![result], vec![lhs, rhs], vec![], 0) };
                if let Some(d) = dimensions { op.$setter(ctx, signed_list(ctx, &d)); }
                op
            }
            /// Infers the broadcast output type and allocates the operation.
            ///
            /// `None` selects NumPy broadcasting. An indeterminate or incompatible
            /// broadcast shape yields an unranked tensor rather than an error.
            ///
            /// # Errors
            ///
            /// Returns an error for operands outside the operation's element domain
            /// or operands violating its element-type compatibility requirements.
            pub fn new_inferred(ctx: &mut Context, lhs: Value, rhs: Value, dimensions: Option<Vec<i64>>) -> Result<Self> {
                let a = typed(ctx, lhs.get_type(ctx), $domain, false, $name)?;
                let b = typed(ctx, rhs.get_type(ctx), $domain, false, $name)?;
                if $trait == ElementTrait::Same && a.element != b.element {
                    return Err(invalid($name, "operand element types must be identical"));
                }
                let result = infer_broadcast_binary(ctx, &a, &b, dimensions.as_deref(), None, $name)?;
                Ok(Self::new(ctx, lhs, rhs, result, dimensions))
            }
            /// Returns the left-hand tensor operand.
            pub fn lhs(&self, ctx: &Context) -> Value { self.get_operation().deref(ctx).get_operand(0) }
            /// Returns the right-hand tensor operand.
            pub fn rhs(&self, ctx: &Context) -> Value { self.get_operation().deref(ctx).get_operand(1) }
            /// Copies the explicit axis mapping, or returns `None` if absent or malformed.
            ///
            /// An absent mapping selects NumPy broadcasting; an empty mapping is
            /// distinct and explicitly maps a scalar operand.
            pub fn broadcast_dimensions(&self, ctx: &Context) -> Option<Vec<i64>> {
                let a = self.$getter(ctx)?;
                read_signed_list(&a)
            }
        }
        impl Verify for $struct {
            fn verify(&self, ctx: &Context) -> Result<()> {
                let ir = self.get_operation().deref(ctx);
                let dims = broadcast_dims_attr(ctx, &ir, $name, self.$getter(ctx).map(|a| read_signed_list(&a)))?;
                check_broadcast(ctx, &ir, $name, $domain, $result_domain, $trait, dims.as_deref(), None)
            }
        }
    };
}
use ElementTrait::{
    Compatible as CompatibleElements, None as NoElementTrait, Same as SameElements,
};
broadcast_op!(
    BroadcastAddOp,
    "chlo.broadcast_add",
    "Elementwise addition with broadcasting and HLO-compatible element types, including quantized tensors.",
    "chlo_broadcast_add_dimensions",
    broadcast_add_op_attr_names,
    ATTR_KEY_CHLO_BROADCAST_ADD_DIMENSIONS,
    set_attr_chlo_broadcast_add_dimensions,
    get_attr_chlo_broadcast_add_dimensions,
    Domain::Any,
    Domain::Any,
    CompatibleElements
);
broadcast_op!(
    BroadcastAtan2Op,
    "chlo.broadcast_atan2",
    "Quadrant-aware arctangent of `lhs / rhs` with broadcasting and identical operand/result element types.",
    "chlo_broadcast_atan2_dimensions",
    broadcast_atan2_op_attr_names,
    ATTR_KEY_CHLO_BROADCAST_ATAN2_DIMENSIONS,
    set_attr_chlo_broadcast_atan2_dimensions,
    get_attr_chlo_broadcast_atan2_dimensions,
    Domain::Any,
    Domain::Any,
    SameElements
);
broadcast_op!(
    BroadcastDivideOp,
    "chlo.broadcast_divide",
    "Elementwise `lhs / rhs` with broadcasting and identical operand/result element types.",
    "chlo_broadcast_divide_dimensions",
    broadcast_divide_op_attr_names,
    ATTR_KEY_CHLO_BROADCAST_DIVIDE_DIMENSIONS,
    set_attr_chlo_broadcast_divide_dimensions,
    get_attr_chlo_broadcast_divide_dimensions,
    Domain::Any,
    Domain::Any,
    SameElements
);
broadcast_op!(
    BroadcastMaximumOp,
    "chlo.broadcast_maximum",
    "Elementwise maximum with broadcasting and HLO-compatible element types, including quantized tensors.",
    "chlo_broadcast_maximum_dimensions",
    broadcast_maximum_op_attr_names,
    ATTR_KEY_CHLO_BROADCAST_MAXIMUM_DIMENSIONS,
    set_attr_chlo_broadcast_maximum_dimensions,
    get_attr_chlo_broadcast_maximum_dimensions,
    Domain::Any,
    Domain::Any,
    CompatibleElements
);
broadcast_op!(
    BroadcastMinimumOp,
    "chlo.broadcast_minimum",
    "Elementwise minimum with broadcasting and HLO-compatible element types, including quantized tensors.",
    "chlo_broadcast_minimum_dimensions",
    broadcast_minimum_op_attr_names,
    ATTR_KEY_CHLO_BROADCAST_MINIMUM_DIMENSIONS,
    set_attr_chlo_broadcast_minimum_dimensions,
    get_attr_chlo_broadcast_minimum_dimensions,
    Domain::Any,
    Domain::Any,
    CompatibleElements
);
broadcast_op!(
    BroadcastMultiplyOp,
    "chlo.broadcast_multiply",
    "Elementwise multiplication with broadcasting and identical operand/result element types.",
    "chlo_broadcast_multiply_dimensions",
    broadcast_multiply_op_attr_names,
    ATTR_KEY_CHLO_BROADCAST_MULTIPLY_DIMENSIONS,
    set_attr_chlo_broadcast_multiply_dimensions,
    get_attr_chlo_broadcast_multiply_dimensions,
    Domain::Any,
    Domain::Any,
    SameElements
);
broadcast_op!(
    BroadcastNextAfterOp,
    "chlo.broadcast_next_after",
    "Next representable value after `lhs` toward `rhs`, with broadcasting and identical element types.",
    "chlo_broadcast_next_after_dimensions",
    broadcast_next_after_op_attr_names,
    ATTR_KEY_CHLO_BROADCAST_NEXT_AFTER_DIMENSIONS,
    set_attr_chlo_broadcast_next_after_dimensions,
    get_attr_chlo_broadcast_next_after_dimensions,
    Domain::Any,
    Domain::Any,
    SameElements
);
broadcast_op!(
    BroadcastPolygammaOp,
    "chlo.broadcast_polygamma",
    "Polygamma function with order in `lhs`, argument in `rhs`, broadcasting, and identical element types.",
    "chlo_broadcast_polygamma_dimensions",
    broadcast_polygamma_op_attr_names,
    ATTR_KEY_CHLO_BROADCAST_POLYGAMMA_DIMENSIONS,
    set_attr_chlo_broadcast_polygamma_dimensions,
    get_attr_chlo_broadcast_polygamma_dimensions,
    Domain::Any,
    Domain::Any,
    SameElements
);
broadcast_op!(
    BroadcastPowerOp,
    "chlo.broadcast_power",
    "Elementwise `lhs` raised to `rhs`, with broadcasting and identical operand/result element types.",
    "chlo_broadcast_power_dimensions",
    broadcast_power_op_attr_names,
    ATTR_KEY_CHLO_BROADCAST_POWER_DIMENSIONS,
    set_attr_chlo_broadcast_power_dimensions,
    get_attr_chlo_broadcast_power_dimensions,
    Domain::Any,
    Domain::Any,
    SameElements
);
broadcast_op!(
    BroadcastRemainderOp,
    "chlo.broadcast_remainder",
    "Elementwise remainder of `lhs / rhs`, with broadcasting and identical operand/result element types.",
    "chlo_broadcast_remainder_dimensions",
    broadcast_remainder_op_attr_names,
    ATTR_KEY_CHLO_BROADCAST_REMAINDER_DIMENSIONS,
    set_attr_chlo_broadcast_remainder_dimensions,
    get_attr_chlo_broadcast_remainder_dimensions,
    Domain::Any,
    Domain::Any,
    SameElements
);
broadcast_op!(
    BroadcastShiftLeftOp,
    "chlo.broadcast_shift_left",
    "Elementwise left shift of `lhs` by `rhs` bits, with broadcasting and identical element types.",
    "chlo_broadcast_shift_left_dimensions",
    broadcast_shift_left_op_attr_names,
    ATTR_KEY_CHLO_BROADCAST_SHIFT_LEFT_DIMENSIONS,
    set_attr_chlo_broadcast_shift_left_dimensions,
    get_attr_chlo_broadcast_shift_left_dimensions,
    Domain::Any,
    Domain::Any,
    SameElements
);
broadcast_op!(
    BroadcastShiftRightArithmeticOp,
    "chlo.broadcast_shift_right_arithmetic",
    "Sign-extending right shift of `lhs` by `rhs` bits, with broadcasting and identical element types.",
    "chlo_broadcast_shift_right_arithmetic_dimensions",
    broadcast_shift_right_arithmetic_op_attr_names,
    ATTR_KEY_CHLO_BROADCAST_SHIFT_RIGHT_ARITHMETIC_DIMENSIONS,
    set_attr_chlo_broadcast_shift_right_arithmetic_dimensions,
    get_attr_chlo_broadcast_shift_right_arithmetic_dimensions,
    Domain::Any,
    Domain::Any,
    SameElements
);
broadcast_op!(
    BroadcastShiftRightLogicalOp,
    "chlo.broadcast_shift_right_logical",
    "Zero-filling right shift of `lhs` by `rhs` bits, with broadcasting and identical element types.",
    "chlo_broadcast_shift_right_logical_dimensions",
    broadcast_shift_right_logical_op_attr_names,
    ATTR_KEY_CHLO_BROADCAST_SHIFT_RIGHT_LOGICAL_DIMENSIONS,
    set_attr_chlo_broadcast_shift_right_logical_dimensions,
    get_attr_chlo_broadcast_shift_right_logical_dimensions,
    Domain::Any,
    Domain::Any,
    SameElements
);
broadcast_op!(
    BroadcastSubtractOp,
    "chlo.broadcast_subtract",
    "Elementwise `lhs - rhs` with broadcasting and identical operand/result element types.",
    "chlo_broadcast_subtract_dimensions",
    broadcast_subtract_op_attr_names,
    ATTR_KEY_CHLO_BROADCAST_SUBTRACT_DIMENSIONS,
    set_attr_chlo_broadcast_subtract_dimensions,
    get_attr_chlo_broadcast_subtract_dimensions,
    Domain::Any,
    Domain::Any,
    SameElements
);
broadcast_op!(
    BroadcastZetaOp,
    "chlo.broadcast_zeta",
    "Hurwitz zeta function with exponent in `lhs` and shift in `rhs`, broadcasting, and identical floating-point element types.",
    "chlo_broadcast_zeta_dimensions",
    broadcast_zeta_op_attr_names,
    ATTR_KEY_CHLO_BROADCAST_ZETA_DIMENSIONS,
    set_attr_chlo_broadcast_zeta_dimensions,
    get_attr_chlo_broadcast_zeta_dimensions,
    Domain::Float,
    Domain::Float,
    SameElements
);
broadcast_op!(
    BroadcastAndOp,
    "chlo.broadcast_and",
    "Elementwise logical or bitwise conjunction of predicate or integer tensors with broadcasting.",
    "chlo_broadcast_and_dimensions",
    broadcast_and_op_attr_names,
    ATTR_KEY_CHLO_BROADCAST_AND_DIMENSIONS,
    set_attr_chlo_broadcast_and_dimensions,
    get_attr_chlo_broadcast_and_dimensions,
    Domain::PredOrInt,
    Domain::Any,
    NoElementTrait
);
broadcast_op!(
    BroadcastOrOp,
    "chlo.broadcast_or",
    "Elementwise logical or bitwise disjunction of predicate or integer tensors with broadcasting.",
    "chlo_broadcast_or_dimensions",
    broadcast_or_op_attr_names,
    ATTR_KEY_CHLO_BROADCAST_OR_DIMENSIONS,
    set_attr_chlo_broadcast_or_dimensions,
    get_attr_chlo_broadcast_or_dimensions,
    Domain::PredOrInt,
    Domain::Any,
    NoElementTrait
);
broadcast_op!(
    BroadcastXorOp,
    "chlo.broadcast_xor",
    "Elementwise logical or bitwise exclusive disjunction of predicate or integer tensors with broadcasting.",
    "chlo_broadcast_xor_dimensions",
    broadcast_xor_op_attr_names,
    ATTR_KEY_CHLO_BROADCAST_XOR_DIMENSIONS,
    set_attr_chlo_broadcast_xor_dimensions,
    get_attr_chlo_broadcast_xor_dimensions,
    Domain::PredOrInt,
    Domain::Any,
    NoElementTrait
);

/// `complex(element)` restricted to `HLO_Complex`.
fn complex_of(ctx: &Context, element: TypeHandle) -> Option<TypeHandle> {
    let primitive = stablehlo::types::primitive_element(ctx, element)?;
    ElementType::complex(primitive)
        .filter(|c| inference::hlo_complex(*c))
        .map(|c| c.native(ctx))
}

/// Forms complex elements from real `lhs` and imaginary `rhs` components.
///
/// Operands must have matching `f32` or `f64` elements. Shapes use NumPy
/// broadcasting unless an explicit lower-rank-to-higher-rank mapping is supplied.
#[pliron_op(name="chlo.broadcast_complex",format="$0 `,` $1 attr_dict `:` typesig",interfaces=[pliron::builtin::op_interfaces::OneResultInterface],attributes=(chlo_broadcast_complex_dimensions:VecAttr))]
pub struct BroadcastComplexOp;
impl BroadcastComplexOp {
    /// Allocates a complex construction with an explicit output type, without verification.
    ///
    /// `dimensions` maps the lower-rank operand's axes into the higher-rank
    /// operand, or selects NumPy broadcasting when absent.
    pub fn new(
        ctx: &mut Context,
        lhs: Value,
        rhs: Value,
        result: TypeHandle,
        dimensions: Option<Vec<i64>>,
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
        if let Some(d) = dimensions {
            op.set_attr_chlo_broadcast_complex_dimensions(ctx, signed_list(ctx, &d));
        }
        op
    }
    /// Infers a complex output type from the real and imaginary tensors.
    ///
    /// Incompatible or indeterminate broadcast shapes yield an unranked result.
    ///
    /// # Errors
    ///
    /// Returns an error unless both operands have the same floating-point
    /// element type and that type has an HLO complex counterpart (`f32` or `f64`).
    pub fn new_inferred(
        ctx: &mut Context,
        lhs: Value,
        rhs: Value,
        dimensions: Option<Vec<i64>>,
    ) -> Result<Self> {
        const NAME: &str = "chlo.broadcast_complex";
        let a = typed(ctx, lhs.get_type(ctx), Domain::Float, false, NAME)?;
        let b = typed(ctx, rhs.get_type(ctx), Domain::Float, false, NAME)?;
        let element = complex_of(ctx, a.element).ok_or_else(|| domain_error(NAME))?;
        let ty = infer_broadcast_binary(ctx, &a, &b, dimensions.as_deref(), Some(element), NAME)?;
        Ok(Self::new(ctx, lhs, rhs, ty, dimensions))
    }
    /// Returns the real-component tensor.
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    /// Returns the imaginary-component tensor.
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
    /// Copies the explicit axis mapping, or returns `None` if absent or malformed.
    pub fn broadcast_dimensions(&self, ctx: &Context) -> Option<Vec<i64>> {
        let a = self.get_attr_chlo_broadcast_complex_dimensions(ctx)?;
        read_signed_list(&a)
    }
}
impl Verify for BroadcastComplexOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        const NAME: &str = "chlo.broadcast_complex";
        let ir = self.get_operation().deref(ctx);
        verify_pure_shape(ctx, &ir, NAME, 2, 1)?;
        let lhs = typed(
            ctx,
            ir.get_operand(0).get_type(ctx),
            Domain::Float,
            false,
            NAME,
        )?;
        let element = complex_of(ctx, lhs.element)
            .ok_or_else(|| fail(ctx, &ir, NAME, "operand element has no HLO complex type"))?;
        let dims = broadcast_dims_attr(
            ctx,
            &ir,
            NAME,
            self.get_attr_chlo_broadcast_complex_dimensions(ctx)
                .map(|a| read_signed_list(&a)),
        )?;
        check_broadcast(
            ctx,
            &ir,
            NAME,
            Domain::Float,
            Domain::Complex,
            ElementTrait::None,
            dims.as_deref(),
            Some(element),
        )
    }
}

/// Optional configuration for [`BroadcastCompareOp`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BroadcastCompareOptions {
    /// Lower-rank-to-higher-rank axis mapping; `None` selects NumPy broadcasting.
    pub broadcast_dimensions: Option<Vec<i64>>,
    /// Comparison relation to apply.
    pub direction: ComparisonDirection,
    /// Explicit comparison semantics, stored under CHLO's `comparison_type` wire name.
    ///
    /// `None` leaves the choice to the dialect's operand-dependent semantics.
    pub comparison_kind: Option<ComparisonKind>,
}

impl BroadcastCompareOptions {
    /// Creates options for a comparison without explicit broadcast dimensions or kind.
    pub const fn new(direction: ComparisonDirection) -> Self {
        Self {
            broadcast_dimensions: None,
            direction,
            comparison_kind: None,
        }
    }
}

/// Broadcasts two tensors and compares each pair of elements, producing `i1` predicates.
///
/// [`BroadcastCompareOptions`] controls the relation, axis mapping, and optional
/// comparison semantics. The serialized attribute name is `comparison_type`,
/// even though the Rust API calls the setting `comparison_kind`.
#[pliron_op(name="chlo.broadcast_compare",format="$0 `,` $1 attr_dict `:` typesig",interfaces=[pliron::builtin::op_interfaces::OneResultInterface],attributes=(chlo_broadcast_compare_dimensions:VecAttr,chlo_comparison_direction:ComparisonDirectionAttr,chlo_comparison_type:ComparisonKindAttr))]
pub struct BroadcastCompareOp;
impl BroadcastCompareOp {
    /// Allocates a comparison with the supplied predicate tensor type, without verification.
    ///
    /// Stores the required direction and only those optional attributes present
    /// in `options`. Call `verify` to check operand and result compatibility.
    pub fn new(
        ctx: &mut Context,
        lhs: Value,
        rhs: Value,
        result: TypeHandle,
        options: BroadcastCompareOptions,
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
        if let Some(d) = options.broadcast_dimensions {
            op.set_attr_chlo_broadcast_compare_dimensions(ctx, signed_list(ctx, &d));
        }
        op.set_attr_chlo_comparison_direction(ctx, ComparisonDirectionAttr::new(options.direction));
        if let Some(t) = options.comparison_kind {
            op.set_attr_chlo_comparison_type(ctx, ComparisonKindAttr::new(t));
        }
        op
    }
    /// Infers an `i1` tensor with the operands' broadcast shape.
    ///
    /// Incompatible or indeterminate broadcast shapes yield an unranked result.
    /// This checks type inference requirements, not whether the chosen comparison
    /// semantics are meaningful for the operand element type.
    ///
    /// # Errors
    ///
    /// Returns an error for non-HLO tensors or incompatible operand element types.
    pub fn new_inferred(
        ctx: &mut Context,
        lhs: Value,
        rhs: Value,
        options: BroadcastCompareOptions,
    ) -> Result<Self> {
        const NAME: &str = "chlo.broadcast_compare";
        let a = typed(ctx, lhs.get_type(ctx), Domain::Any, false, NAME)?;
        let b = typed(ctx, rhs.get_type(ctx), Domain::Any, false, NAME)?;
        let ty = infer_broadcast_binary(
            ctx,
            &a,
            &b,
            options.broadcast_dimensions.as_deref(),
            Some(i1(ctx)),
            NAME,
        )?;
        Ok(Self::new(ctx, lhs, rhs, ty, options))
    }
    /// Returns the left-hand tensor being compared.
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    /// Returns the right-hand tensor being compared.
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
    /// Copies the explicit axis mapping, or returns `None` if absent or malformed.
    pub fn broadcast_dimensions(&self, ctx: &Context) -> Option<Vec<i64>> {
        let a = self.get_attr_chlo_broadcast_compare_dimensions(ctx)?;
        read_signed_list(&a)
    }
    /// Reads the required comparison relation, or returns `None` if missing or malformed.
    pub fn comparison_direction(&self, ctx: &Context) -> Option<ComparisonDirection> {
        Some(self.get_attr_chlo_comparison_direction(ctx)?.value)
    }
    /// Reads explicit comparison semantics without inventing a default for an absent attribute.
    pub fn comparison_kind(&self, ctx: &Context) -> Option<ComparisonKind> {
        Some(self.get_attr_chlo_comparison_type(ctx)?.value)
    }
}
impl Verify for BroadcastCompareOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        const NAME: &str = "chlo.broadcast_compare";
        let ir = self.get_operation().deref(ctx);
        verify_pure_shape(ctx, &ir, NAME, 2, 1)?;
        if self.comparison_direction(ctx).is_none() {
            return Err(fail(ctx, &ir, NAME, "missing comparison_direction"));
        }
        let dims = broadcast_dims_attr(
            ctx,
            &ir,
            NAME,
            self.get_attr_chlo_broadcast_compare_dimensions(ctx)
                .map(|a| read_signed_list(&a)),
        )?;
        check_broadcast(
            ctx,
            &ir,
            NAME,
            Domain::Any,
            Domain::Pred,
            ElementTrait::None,
            dims.as_deref(),
            Some(i1(ctx)),
        )
    }
}

/// Upstream `BroadcastSelectOp::inferReturnTypeComponents`.
fn infer_select(
    ctx: &Context,
    pred: &Shaped,
    on_true: &Shaped,
    on_false: &Shaped,
    operation: &'static str,
) -> Result<TypeHandle> {
    if on_true.element != on_false.element {
        return Err(invalid(operation, "mismatched operand types"));
    }
    Ok(inference::numpy_broadcast_type(
        ctx,
        &[pred, on_true, on_false],
        on_true.element,
    ))
}

/// Selects elements from two tensors using a broadcast predicate tensor.
///
/// The predicate is a ranked `i1` tensor. Both value tensors have identical
/// element types and all three operands use trailing-axis NumPy broadcasting.
#[pliron_op(name="chlo.broadcast_select",format="$0 `,` $1 `,` $2 attr_dict `:` typesig",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct BroadcastSelectOp;
impl BroadcastSelectOp {
    /// Allocates a selection with the supplied result type, without verification.
    pub fn new(
        ctx: &mut Context,
        pred: Value,
        on_true: Value,
        on_false: Value,
        result: TypeHandle,
    ) -> Self {
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![pred, on_true, on_false],
                vec![],
                0,
            ),
        }
    }
    /// Infers the broadcast output shape and copies the value operands' element type.
    ///
    /// Incompatible or indeterminate broadcast shapes yield an unranked result.
    ///
    /// # Errors
    ///
    /// Returns an error for a nonranked or nonpredicate condition, invalid HLO
    /// value tensors, or different element types on the true and false branches.
    pub fn new_inferred(
        ctx: &mut Context,
        pred: Value,
        on_true: Value,
        on_false: Value,
    ) -> Result<Self> {
        const NAME: &str = "chlo.broadcast_select";
        let p = typed(ctx, pred.get_type(ctx), Domain::Pred, true, NAME)?;
        let t = typed(ctx, on_true.get_type(ctx), Domain::Any, false, NAME)?;
        let f = typed(ctx, on_false.get_type(ctx), Domain::Any, false, NAME)?;
        let ty = infer_select(ctx, &p, &t, &f, NAME)?;
        Ok(Self::new(ctx, pred, on_true, on_false, ty))
    }
    /// Returns the condition tensor whose true elements select [`Self::on_true`].
    pub fn predicate(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    /// Returns the tensor selected where the broadcast predicate is true.
    pub fn on_true(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
    /// Returns the tensor selected where the broadcast predicate is false.
    pub fn on_false(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(2)
    }
}
impl Verify for BroadcastSelectOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        const NAME: &str = "chlo.broadcast_select";
        let ir = self.get_operation().deref(ctx);
        verify_pure_shape(ctx, &ir, NAME, 3, 1)?;
        let p = typed(
            ctx,
            ir.get_operand(0).get_type(ctx),
            Domain::Pred,
            true,
            NAME,
        )?;
        let t = typed(
            ctx,
            ir.get_operand(1).get_type(ctx),
            Domain::Any,
            false,
            NAME,
        )?;
        let f = typed(
            ctx,
            ir.get_operand(2).get_type(ctx),
            Domain::Any,
            false,
            NAME,
        )?;
        let result_ty = ir.get_result(0).get_type(ctx);
        let result = typed(ctx, result_ty, Domain::Any, false, NAME)?;
        let inferred = infer_select(ctx, &p, &t, &f, NAME).map_err(|error| {
            failure_with_source(ctx, &ir, NAME, "mismatched operand types", error)
        })?;
        let inferred = inference::shaped(ctx, inferred)
            .ok_or_else(|| fail(ctx, &ir, NAME, "inferred type is not a tensor"))?;
        if !inference::compatible_shape(&inferred, &result) {
            return Err(fail(
                ctx,
                &ir,
                NAME,
                "result shape is incompatible with the broadcast shape",
            ));
        }
        Ok(())
    }
}

fn constant_word_values(ctx: &Context, ty: TypeHandle, bits: Vec<u64>) -> VecAttr {
    use stablehlo::attributes::ScalarBitsAttr;
    let width = stablehlo::types::primitive_element(ctx, ty).and_then(ElementType::bit_width);
    let Some(width) = width.filter(|width| *width > 0) else {
        return VecAttr::new(vec![Box::new(ScalarBitsAttr { ty, words: bits })]);
    };
    VecAttr::new(
        bits.chunks(width.div_ceil(64) as usize)
            .map(|words| {
                Box::new(ScalarBitsAttr {
                    ty,
                    words: words.to_vec(),
                }) as AttrObj
            })
            .collect(),
    )
}

pub(crate) fn constant_element_value(ctx: &Context, value: &AttrObj) -> AttrObj {
    use pliron::{attribute::attr_cast, builtin::attr_interfaces::FloatAttr};
    if let Some(float) = attr_cast::<dyn FloatAttr>(&**value) {
        let ty = float.get_type(ctx);
        let Some(width) =
            stablehlo::types::primitive_element(ctx, ty).and_then(ElementType::bit_width)
        else {
            return value.clone();
        };
        let bits = float.get_inner().to_bits();
        return Box::new(stablehlo::attributes::ScalarBitsAttr {
            ty,
            words: if width > 64 {
                vec![bits as u64, (bits >> 64) as u64]
            } else {
                vec![bits as u64]
            },
        });
    }
    value.clone()
}

fn constant_elements_value(ctx: &Context, mut value: AttrObj) -> AttrObj {
    use stablehlo::attributes::{DenseElementsAttr, SparseElementsAttr};
    let values = if let Some(dense) = value.downcast_mut::<DenseElementsAttr>() {
        Some(&mut dense.values)
    } else {
        value
            .downcast_mut::<SparseElementsAttr>()
            .map(|sparse| &mut sparse.values)
    };
    if let Some(values) = values {
        for scalar in &mut values.0 {
            *scalar = constant_element_value(ctx, scalar);
        }
    }
    value
}

fn constant_scalar_type(ctx: &Context, value: &AttrObj) -> Result<TypeHandle> {
    use stablehlo::attributes::{ComplexAttr, ScalarBitsAttr};
    value.verify(ctx)?;
    let ty = if let Some(value) = value.downcast_ref::<ScalarBitsAttr>() {
        Some(value.ty)
    } else if let Some(value) = value.downcast_ref::<ComplexAttr>() {
        Some(value.ty)
    } else if value.downcast_ref::<BoolAttr>().is_some() {
        Some(ElementType::I1.native(ctx))
    } else {
        pliron::attribute::attr_type(&**value, ctx)
    }
    .ok_or_else(|| {
        invalid(
            "chlo.constant_like",
            "value must be a typed scalar attribute",
        )
    })?;
    if !inference::element_in(ctx, ty, inference::Domain::Any) {
        return Err(invalid(
            "chlo.constant_like",
            "value type must be an upstream HLO tensor element",
        ));
    }
    ty.deref(ctx).verify(ctx)?;
    Ok(ty)
}

fn constant_like_type(
    ctx: &Context,
    operand: TypeHandle,
    element: TypeHandle,
) -> Result<TypeHandle> {
    operand.deref(ctx).verify(ctx)?;
    if !inference::hlo_tensor_type(ctx, operand, inference::Domain::Any, false) {
        return Err(invalid(
            "chlo.constant_like",
            "operand must be an upstream HLO tensor",
        ));
    }
    inference::same_shape(ctx, operand, element)
        .ok_or_else(|| invalid("chlo.constant_like", "operand must be a tensor"))
}

/// A statically shaped constant tensor backed by a typed dense or sparse elements attribute.
///
/// The payload is stored in `value` and must have exactly the result type.
/// Use [`Self::from_value`] for arbitrary supported elements attributes, or
/// [`Self::new_inferred`] for a dense payload supplied as raw words.
#[pliron_op(name="chlo.constant",format="attr_dict `:` type($0)",interfaces=[pliron::builtin::op_interfaces::NOpdsInterface<0>,pliron::builtin::op_interfaces::OneResultInterface])]
pub struct ConstantOp;
impl ConstantOp {
    /// Allocates a dense constant from raw words and an explicit result type.
    ///
    /// Words encode element bits, not numeric conversions. `splat` stores one
    /// element repeated over the tensor. The sole payload is the typed `value`
    /// attribute; call `verify` to check the static shape, element width, and count.
    pub fn new(ctx: &mut Context, result: TypeHandle, bits: Vec<u64>, splat: bool) -> Self {
        let element = result
            .deref(ctx)
            .downcast_ref::<stablehlo::types::TensorType>()
            .map_or(result, |tensor| tensor.element);
        let element = element
            .deref(ctx)
            .downcast_ref::<stablehlo::types::QuantizedElementType>()
            .map_or(element, |quantized| quantized.storage);
        let value = stablehlo::attributes::DenseElementsAttr {
            ty: result,
            values: constant_word_values(ctx, element, bits),
            is_splat: splat,
        };
        Self::new_with_value(ctx, result, Box::new(value))
    }
    /// Allocates a constant with an explicit type and payload, without verification.
    ///
    /// A supported `value` is a dense or sparse elements attribute with exactly
    /// the result type. Legacy word payloads are normalized by [`Self::set_value`].
    pub fn new_with_value(ctx: &mut Context, result: TypeHandle, value: AttrObj) -> Self {
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
        op.set_value(ctx, value);
        op
    }
    /// Derives the result type from a dense or sparse elements attribute and verifies it.
    ///
    /// # Errors
    ///
    /// Returns an error for unsupported payloads, invalid elements data, or a type
    /// that is not a static ranked HLO tensor in the constant element domain.
    pub fn from_value(ctx: &mut Context, value: AttrObj) -> Result<Self> {
        let ty = Self::elements_type(&value)
            .ok_or_else(|| invalid("chlo.constant", "value must be an elements attribute"))?;
        let op = Self::new_with_value(ctx, ty, value);
        op.verify(ctx)?;
        Ok(op)
    }
    fn elements_type(value: &AttrObj) -> Option<TypeHandle> {
        value
            .downcast_ref::<stablehlo::attributes::DenseElementsAttr>()
            .map(|value| value.ty)
            .or_else(|| {
                value
                    .downcast_ref::<stablehlo::attributes::SparseElementsAttr>()
                    .map(|value| value.ty)
            })
    }
    /// Clones the stored payload attribute, or returns `None` when it is absent.
    ///
    /// This accessor does not validate the payload's kind or type.
    pub fn value(&self, ctx: &Context) -> Option<AttrObj> {
        self.get_operation()
            .deref(ctx)
            .attributes
            .0
            .get(&pliron::ident!("value"))
            .cloned()
    }
    /// Replaces the payload, normalizing supported legacy word representations.
    ///
    /// Does not change the result type or verify payload/type consistency.
    pub fn set_value(&self, ctx: &Context, value: AttrObj) {
        let value = constant_elements_value(ctx, value);
        self.get_operation()
            .deref_mut(ctx)
            .attributes
            .0
            .insert(pliron::ident!("value"), value);
    }
    /// Builds and verifies a dense constant from its static shape, element type, and bits.
    ///
    /// A splat supplies one encoded element; a nonsplat supplies every element
    /// in tensor order. Raw words are interpreted using the element's bit width.
    ///
    /// # Errors
    ///
    /// Returns an error for dynamic dimensions, unsupported element types, or
    /// a payload whose word count or element data does not satisfy the tensor type.
    pub fn new_inferred(
        ctx: &mut Context,
        shape: Vec<i64>,
        element: ElementType,
        bits: Vec<u64>,
        splat: bool,
    ) -> Result<Self> {
        if shape.iter().any(|d| *d < 0) {
            return Err(invalid("chlo.constant", "constant shape must be static"));
        }
        let op = Self::new(ctx, make(ctx, shape, element), bits, splat);
        op.verify(ctx)?;
        Ok(op)
    }
    /// Copies raw words from a dense scalar-bits payload in storage order.
    ///
    /// Returns `None` for a missing payload, sparse elements, or dense elements
    /// not represented by scalar-bits attributes. A splat is not expanded.
    pub fn bits(&self, ctx: &Context) -> Option<Vec<u64>> {
        let value = self.value(ctx)?;
        let dense = value.downcast_ref::<stablehlo::attributes::DenseElementsAttr>()?;
        let mut bits = Vec::new();
        for value in &dense.values.0 {
            if let Some(value) = value.downcast_ref::<stablehlo::attributes::ScalarBitsAttr>() {
                bits.extend(&value.words);
            } else {
                return None;
            }
        }
        Some(bits)
    }
    /// Returns the dense payload's splat flag, or `None` for absent or nondense data.
    pub fn is_splat(&self, ctx: &Context) -> Option<bool> {
        self.value(ctx)?
            .downcast_ref::<stablehlo::attributes::DenseElementsAttr>()
            .map(|value| value.is_splat)
    }
}
impl Verify for ConstantOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() != 0 || ir.get_num_results() != 1 {
            return Err(invalid("chlo.constant", "constant arity mismatch"));
        }
        verify_pure_structure(&ir, "chlo.constant")?;
        let result = ir.get_result(0).get_type(ctx);
        result.deref(ctx).verify(ctx)?;
        if !inference::hlo_tensor_type(ctx, result, inference::Domain::Tensor, true) {
            return Err(invalid(
                "chlo.constant",
                "result must have an upstream static tensor element type",
            ));
        }
        let info = tensor_info(ctx, result)
            .ok_or_else(|| invalid("chlo.constant", "constant result must be ranked"))?;
        if info.dimensions.iter().any(|d| *d < 0) {
            return Err(invalid(
                "chlo.constant",
                "constant result must have static dimensions",
            ));
        }
        let value = self
            .value(ctx)
            .ok_or_else(|| invalid("chlo.constant", "value is missing"))?;
        if Self::elements_type(&value) != Some(result) {
            return Err(invalid(
                "chlo.constant",
                "elements value type must equal result type",
            ));
        }
        value.verify(ctx)
    }
}
#[pliron::derive::op_interface_impl]
impl pliron::opts::constants::ConstFoldInterface for ConstantOp {
    fn check_fold(&self, ctx: &Context, _operands: &[Option<AttrObj>]) -> Vec<Option<AttrObj>> {
        vec![self.verify(ctx).ok().and_then(|()| self.value(ctx))]
    }
    fn fold_in_place(
        &self,
        _ctx: &mut Context,
        _operands: &[Option<AttrObj>],
        _rewriter: &mut dyn pliron::irbuild::rewriter::Rewriter,
    ) -> pliron::irbuild::IRStatus {
        pliron::irbuild::IRStatus::Unchanged
    }
}

/// Repeats a scalar constant over the shape and encoding of an operand tensor.
///
/// Unlike [`ConstantOp`], the output shape may be dynamic or unranked. The
/// output element type comes from the scalar payload, not the shape operand.
#[pliron_op(name="chlo.constant_like",format="$0 attr_dict `:` typesig",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct ConstantLikeOp;
impl ConstantLikeOp {
    /// Allocates a scalar-bits fill with an explicit output type, without verification.
    ///
    /// `value_bits` encodes one scalar of `element`. `operand` contributes the
    /// shape, not element values; `result` must have that shape and the scalar's
    /// element type.
    pub fn new(
        ctx: &mut Context,
        value_bits: Vec<u64>,
        element: ElementType,
        operand: Value,
        result: TypeHandle,
    ) -> Self {
        Self::new_with_value(
            ctx,
            Box::new(stablehlo::attributes::ScalarBitsAttr {
                ty: element.native(ctx),
                words: value_bits,
            }),
            operand,
            result,
        )
    }
    /// Allocates a fill from a scalar attribute and explicit type, without verification.
    ///
    /// The scalar payload must have an accepted HLO element type. Call `verify`
    /// to check its representation and the output's compatibility with `operand`.
    pub fn new_with_value(
        ctx: &mut Context,
        value: AttrObj,
        operand: Value,
        result: TypeHandle,
    ) -> Self {
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![operand],
                vec![],
                0,
            ),
        };
        op.set_value(ctx, value);
        op
    }
    /// Clones the stored scalar payload, or returns `None` if it is absent.
    ///
    /// The attribute's validity is checked by `verify`, not by this accessor.
    pub fn value(&self, ctx: &Context) -> Option<AttrObj> {
        self.get_operation()
            .deref(ctx)
            .attributes
            .0
            .get(&pliron::ident!("value"))
            .cloned()
    }
    /// Replaces the scalar payload without changing or verifying the output type.
    pub fn set_value(&self, ctx: &Context, value: AttrObj) {
        self.get_operation()
            .deref_mut(ctx)
            .attributes
            .0
            .insert(pliron::ident!("value"), value);
    }
    /// Infers the fill type from a scalar payload and shape operand, then verifies it.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid or unsupported scalar payload, a non-HLO
    /// shape operand, or an invalid inferred tensor type.
    pub fn from_value(ctx: &mut Context, value: AttrObj, operand: Value) -> Result<Self> {
        let element = constant_scalar_type(ctx, &value)?;
        let ty = constant_like_type(ctx, operand.get_type(ctx), element)?;
        let op = Self::new_with_value(ctx, value, operand, ty);
        op.verify(ctx)?;
        Ok(op)
    }
    /// Infers and verifies a fill using one raw scalar and the operand's tensor shape.
    ///
    /// # Errors
    ///
    /// Returns the same validation errors as [`Self::from_value`], including
    /// invalid scalar word counts or an unsupported operand/element type.
    pub fn new_inferred(
        ctx: &mut Context,
        value_bits: Vec<u64>,
        element: ElementType,
        operand: Value,
    ) -> Result<Self> {
        Self::from_value(
            ctx,
            Box::new(stablehlo::attributes::ScalarBitsAttr {
                ty: element.native(ctx),
                words: value_bits,
            }),
            operand,
        )
    }
    /// Returns the tensor supplying the fill shape and encoding.
    pub fn operand(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}
impl Verify for ConstantLikeOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_operands() != 1 || ir.get_num_results() != 1 {
            return Err(invalid("chlo.constant_like", "arity mismatch"));
        }
        verify_pure_structure(&ir, "chlo.constant_like")?;
        let value = self
            .value(ctx)
            .ok_or_else(|| invalid("chlo.constant_like", "value is missing"))?;
        let element = constant_scalar_type(ctx, &value)?;
        let expected = constant_like_type(ctx, ir.get_operand(0).get_type(ctx), element)?;
        let actual = ir.get_result(0).get_type(ctx);
        actual.deref(ctx).verify(ctx)?;
        if !inference::hlo_tensor_type(ctx, actual, inference::Domain::Any, false)
            || inference::shaped(ctx, actual).is_none_or(|shaped| shaped.element != element)
            || !inference::compatible_types(ctx, actual, expected)
        {
            return Err(invalid(
                "chlo.constant_like",
                "result must use operand shape and value element type",
            ));
        }
        Ok(())
    }
}
#[pliron::derive::op_interface_impl]
impl pliron::opts::constants::ConstFoldInterface for ConstantLikeOp {
    fn check_fold(&self, ctx: &Context, _operands: &[Option<AttrObj>]) -> Vec<Option<AttrObj>> {
        use stablehlo::{attributes::DenseElementsAttr, types::TensorType};
        let folded = (|| {
            self.verify(ctx).ok()?;
            let operand_ty = self.operand(ctx).get_type(ctx);
            let object = operand_ty.deref(ctx);
            let tensor = object.downcast_ref::<TensorType>()?;
            if tensor.dimensions.iter().any(|dim| *dim < 0) {
                return None;
            }
            let value = self.value(ctx)?;
            let element = constant_scalar_type(ctx, &value).ok()?;
            let value = constant_element_value(ctx, &value);
            let ty = TensorType::get(ctx, tensor.dimensions.clone(), element).into();
            let dense = if tensor.dimensions.contains(&0) {
                DenseElementsAttr::new(ty, VecAttr::new(vec![]))
            } else {
                DenseElementsAttr::splat(ty, value)
            };
            dense.verify(ctx).ok()?;
            Some(Box::new(dense) as AttrObj)
        })();
        vec![folded]
    }
    fn fold_in_place(
        &self,
        ctx: &mut Context,
        operands: &[Option<AttrObj>],
        rewriter: &mut dyn pliron::irbuild::rewriter::Rewriter,
    ) -> pliron::irbuild::IRStatus {
        use pliron::irbuild::IRStatus;
        let Some(Some(mut value)) = self.check_fold(ctx, operands).into_iter().next() else {
            return IRStatus::Unchanged;
        };
        if let Some(dense) = value.downcast_mut::<stablehlo::attributes::DenseElementsAttr>() {
            dense.ty = self.get_operation().deref(ctx).get_result(0).get_type(ctx);
        }
        let Ok(constant) = ConstantOp::from_value(ctx, value) else {
            return IRStatus::Unchanged;
        };
        rewriter.append_operation(ctx, constant.get_operation());
        rewriter.replace_operation(ctx, self.get_operation(), constant.get_operation());
        IRStatus::Changed
    }
}
/// Upstream `hlo::inferTopKOp`: `(values, indices)` types.
fn infer_top_k(ctx: &Context, operand: &Shaped, k: i64) -> Result<(TypeHandle, TypeHandle)> {
    const NAME: &str = "chlo.top_k";
    let i32_ty = ElementType::I32.native(ctx);
    let Some(dims) = &operand.dimensions else {
        return Ok((
            inference::unranked(ctx, operand.element),
            inference::unranked(ctx, i32_ty),
        ));
    };
    let Some(&last) = dims.last() else {
        return Err(invalid(NAME, "operand's rank must be at least 1"));
    };
    if last >= 0 && last < k {
        return Err(invalid(NAME, "operand's last dimension must be at least k"));
    }
    let mut shape = dims.clone();
    let rank = shape.len();
    shape[rank - 1] = if k < 0 { -1 } else { k };
    let encoding = match &operand.bounds {
        Some(bounds) => {
            let mut bounds = bounds.clone();
            if let Some(b) = bounds.last_mut() {
                *b = -1;
            }
            inference::bounds_encoding(bounds)
        }
        None => operand.encoding.clone(),
    };
    Ok((
        inference::shaped_type(ctx, Some(shape.clone()), operand.element, encoding.clone()),
        inference::shaped_type(ctx, Some(shape), i32_ty, encoding),
    ))
}

/// Configuration for [`TopKOp`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TopKOptions {
    /// Number of values to return along the final dimension.
    ///
    /// A negative value produces a dynamic final dimension during type inference;
    /// a known final input dimension must be at least a nonnegative `k`.
    pub k: i64,
    /// Whether equal values retain their original order.
    pub is_stable: bool,
}

impl TopKOptions {
    /// Creates top-k options with the requested count and stability.
    ///
    /// Does not validate `k` against an operand shape.
    pub const fn new(k: i64, is_stable: bool) -> Self {
        Self { k, is_stable }
    }
}

/// Selects the largest `k` values along the final tensor dimension.
///
/// Produces a values tensor followed by an `i32` indices tensor with the same
/// shape. Ranked inputs must have at least one axis. Equal values retain their
/// original order when the stability option is enabled.
#[pliron_op(name="chlo.top_k",format="$0 attr_dict `:` typesig",interfaces=[pliron::builtin::op_interfaces::NResultsInterface<2>],attributes=(chlo_top_k_k:IntegerAttr,chlo_top_k_is_stable:BoolAttr))]
pub struct TopKOp;
impl TopKOp {
    /// Allocates top-k with explicit values and indices types, without verification.
    ///
    /// Result ordering is `(values, indices)`. Call `verify` to check the operand,
    /// count, shape compatibility, and index element type.
    pub fn new(
        ctx: &mut Context,
        operand: Value,
        options: TopKOptions,
        values: TypeHandle,
        indices: TypeHandle,
    ) -> Self {
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![values, indices],
                vec![operand],
                vec![],
                0,
            ),
        };
        op.set_attr_chlo_top_k_k(
            ctx,
            IntegerAttr::new(
                pliron::builtin::types::IntegerType::get(
                    ctx,
                    64,
                    pliron::builtin::types::Signedness::Signless,
                ),
                APInt::from_i64(options.k, bw(64)),
            ),
        );
        op.set_attr_chlo_top_k_is_stable(ctx, BoolAttr::new(options.is_stable));
        op
    }
    /// Infers values and `i32` indices types and allocates top-k.
    ///
    /// Replaces the final dimension by `k`, or by a dynamic dimension for negative
    /// `k`. An unranked operand produces unranked results.
    ///
    /// # Errors
    ///
    /// Returns an error for a non-HLO operand, a ranked scalar, or a known final
    /// dimension smaller than `k`.
    pub fn new_inferred(ctx: &mut Context, operand: Value, options: TopKOptions) -> Result<Self> {
        let input = typed(ctx, operand.get_type(ctx), Domain::Any, false, "chlo.top_k")?;
        let (values, indices) = infer_top_k(ctx, &input, options.k)?;
        Ok(Self::new(ctx, operand, options, values, indices))
    }
    /// Returns the tensor searched along its final dimension.
    pub fn operand(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    /// Returns the selected values tensor (result zero).
    pub fn values(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }
    /// Returns the `i32` source-index tensor (result one).
    pub fn indices(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(1)
    }
    /// `k` as upstream `I64Attr`; `None` unless it is a signless 64-bit integer attribute.
    pub fn k(&self, ctx: &Context) -> Option<i64> {
        let attr = self.get_attr_chlo_top_k_k(ctx)?;
        let ty = attr.get_type();
        let ty = ty.deref(ctx);
        if ty.width() != 64 || ty.signedness() != pliron::builtin::types::Signedness::Signless {
            return None;
        }
        attr.value().to_string_decimal(true).parse().ok()
    }
    /// Returns the stability flag, defaulting to `true` without a typed boolean attribute.
    ///
    /// This convenience accessor is infallible and does not validate the raw
    /// attribute dictionary. Use the generated typed getter when the distinction
    /// between an explicit value and the default matters.
    pub fn is_stable(&self, ctx: &Context) -> bool {
        self.get_attr_chlo_top_k_is_stable(ctx)
            .is_none_or(|v| bool::from((*v).clone()))
    }
}
impl Verify for TopKOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        const NAME: &str = "chlo.top_k";
        let ir = self.get_operation().deref(ctx);
        verify_pure_shape(ctx, &ir, NAME, 1, 2)?;
        let operand = typed(
            ctx,
            ir.get_operand(0).get_type(ctx),
            Domain::Any,
            false,
            NAME,
        )?;
        let values = ir.get_result(0).get_type(ctx);
        let indices = ir.get_result(1).get_type(ctx);
        typed(ctx, values, Domain::Any, false, NAME)?;
        typed(ctx, indices, Domain::Any, false, NAME)?;
        let k = self
            .k(ctx)
            .ok_or_else(|| fail(ctx, &ir, NAME, "k attribute is missing or malformed"))?;
        let (want_values, want_indices) = infer_top_k(ctx, &operand, k).map_err(|error| {
            failure_with_source(ctx, &ir, NAME, "operand is incompatible with k", error)
        })?;
        if !inference::compatible_types(ctx, want_values, values)
            || !inference::compatible_types(ctx, want_indices, indices)
        {
            return Err(fail(
                ctx,
                &ir,
                NAME,
                "result types are incompatible with the inferred top_k types",
            ));
        }
        Ok(())
    }
}

/// Configuration for [`RaggedDotOp`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RaggedDotOptions {
    /// Paired batch/contracting axes plus the left ragged and right group axes.
    pub dimension_numbers: RaggedDotDimensionNumbersAttr,
    /// Precision requests in operand order; ragged dot accepts at most two entries.
    ///
    /// `None` omits the attribute; `Some(vec![])` explicitly stores an empty
    /// configuration. Both use the dialect's default precision.
    pub precision_config: Option<Vec<Precision>>,
}

impl RaggedDotOptions {
    /// Creates ragged-dot options with default precision.
    pub fn new(dimension_numbers: RaggedDotDimensionNumbersAttr) -> Self {
        Self {
            dimension_numbers,
            precision_config: None,
        }
    }

    /// Stores an explicit precision configuration and returns the updated options.
    ///
    /// An empty vector stores an empty attribute rather than omitting it.
    /// The operation, not this method, checks that there are at most two entries.
    #[must_use]
    pub fn with_precision_config(mut self, precision_config: Vec<Precision>) -> Self {
        self.precision_config = Some(precision_config);
        self
    }
}

/// Grouped dot product with a ragged left-hand dimension.
///
/// Dimension numbers determine batching, contraction, and grouping. `group_sizes`
/// is a ranked integer tensor describing the ragged partition. Inference uses the
/// left operand's element type; verification checks output shape compatibility
/// without requiring the explicit output element type to match the operands.
#[pliron_op(name="chlo.ragged_dot",format="$0 `,` $1 `,` $2 attr_dict `:` typesig",interfaces=[pliron::builtin::op_interfaces::OneResultInterface],attributes=(chlo_ragged_dot_dimension_numbers:RaggedDotDimensionNumbersAttr,chlo_precision_config:PrecisionConfigAttr))]
pub struct RaggedDotOp;
impl RaggedDotOp {
    /// Allocates ragged dot with an explicit result type, without verification.
    ///
    /// Stores the dimension numbers and any explicit precision configuration.
    /// Call `verify` to validate the dimension geometry and result compatibility.
    pub fn new(
        ctx: &mut Context,
        lhs: Value,
        rhs: Value,
        group_sizes: Value,
        result: TypeHandle,
        options: RaggedDotOptions,
    ) -> Self {
        let op = Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![result],
                vec![lhs, rhs, group_sizes],
                vec![],
                0,
            ),
        };
        op.set_attr_chlo_ragged_dot_dimension_numbers(ctx, options.dimension_numbers);
        if let Some(values) = options.precision_config {
            op.set_attr_chlo_precision_config(ctx, crate::attributes::precision_config(values));
        }
        op
    }
    /// Infers a ranked result from operand geometry and the dimension numbers.
    ///
    /// The output uses the left operand's element type and no tensor encoding.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid operand domains,
    /// unranked dot operands, invalid or repeated axes, incompatible batch or
    /// contraction sizes, inconsistent ragged/group geometry, invalid group-size
    /// shapes, or a precision configuration with more than two entries.
    pub fn new_inferred(
        ctx: &mut Context,
        lhs: Value,
        rhs: Value,
        group_sizes: Value,
        options: RaggedDotOptions,
    ) -> Result<Self> {
        let (shape, element) = ragged_dot_shape(
            ctx,
            lhs.get_type(ctx),
            rhs.get_type(ctx),
            group_sizes.get_type(ctx),
            &options.dimension_numbers,
            options.precision_config.as_ref().map_or(0, Vec::len),
        )?;
        let result = inference::shaped_type(ctx, Some(shape), element, VecAttr::new(vec![]));
        Ok(Self::new(ctx, lhs, rhs, group_sizes, result, options))
    }
    /// Returns the left-hand tensor, which contains the ragged axis.
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    /// Returns the right-hand tensor, including a group axis when required by the geometry.
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
    /// Returns the integer tensor describing each ragged group's size.
    pub fn group_sizes(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(2)
    }
    /// Clones the dimension mappings, or returns `None` if missing or incorrectly typed.
    pub fn dimension_numbers(&self, ctx: &Context) -> Option<RaggedDotDimensionNumbersAttr> {
        self.get_attr_chlo_ragged_dot_dimension_numbers(ctx)
            .map(|v| (*v).clone())
    }
    /// Clones the explicit precision attribute without applying a default.
    ///
    /// Returns `None` when the attribute is absent or incorrectly typed.
    pub fn precision_config_attr(&self, ctx: &Context) -> Option<PrecisionConfigAttr> {
        self.get_attr_chlo_precision_config(ctx)
            .map(|v| (*v).clone())
    }
    /// Copies explicit precision requests in operand order.
    ///
    /// `None` means no correctly typed attribute was found; `Some(vec![])`
    /// means an empty configuration was explicitly stored.
    pub fn precision_config(&self, ctx: &Context) -> Option<Vec<Precision>> {
        let attr = self.get_attr_chlo_precision_config(ctx)?;
        Some(read_precision_config(&attr))
    }
}
/// Upstream `inferRaggedDotOp`: result dimensions and lhs element type.
fn ragged_dot_shape(
    ctx: &Context,
    lhs: TypeHandle,
    rhs: TypeHandle,
    groups: TypeHandle,
    d: &RaggedDotDimensionNumbersAttr,
    precision_len: usize,
) -> Result<(Vec<i64>, TypeHandle)> {
    const NAME: &str = "chlo.ragged_dot";
    let lhs = typed(ctx, lhs, Domain::Any, false, NAME)?;
    let rhs = typed(ctx, rhs, Domain::Any, false, NAME)?;
    let groups = typed(ctx, groups, Domain::Int, true, NAME)?;
    if precision_len > 2 {
        return Err(invalid(
            NAME,
            "expects precision config to be empty or have <= 2 elements",
        ));
    }
    let (Some(ls), Some(rs), Some(gs)) = (&lhs.dimensions, &rhs.dimensions, &groups.dimensions)
    else {
        return Err(invalid(
            NAME,
            "lhs, rhs and group_sizes must be ranked tensors",
        ));
    };
    let dimensions = d.dimensions();
    let lb = dimensions.lhs_batching_dimensions;
    let rb = dimensions.rhs_batching_dimensions;
    let lc = dimensions.lhs_contracting_dimensions;
    let rc = dimensions.rhs_contracting_dimensions;
    let lr = dimensions.lhs_ragged_dimensions;
    let rg = dimensions.rhs_group_dimensions;
    if lb.len() != rb.len() {
        return Err(invalid(
            NAME,
            "lhs and rhs should have the same number of batching dimensions",
        ));
    }
    if lc.len() != rc.len() {
        return Err(invalid(
            NAME,
            "lhs and rhs should have the same number of contracting dimensions",
        ));
    }
    let distinct = |a: &[i64], b: &[i64]| {
        let all: Vec<i64> = a.iter().chain(b).copied().collect();
        all.iter().enumerate().all(|(i, x)| !all[..i].contains(x))
    };
    if !distinct(&lb, &lc) {
        return Err(invalid(
            NAME,
            "has duplicated dimension from lhs_batching_dimensions and lhs_contracting_dimensions",
        ));
    }
    if !distinct(&rb, &rc) {
        return Err(invalid(
            NAME,
            "has duplicated dimension from rhs_batching_dimensions and rhs_contracting_dimensions",
        ));
    }
    let in_range = |dims: &[i64], rank: usize| -> Result<Vec<usize>> {
        dims.iter()
            .map(|&v| {
                usize::try_from(v)
                    .ok()
                    .filter(|&i| i < rank)
                    .ok_or_else(|| invalid(NAME, "dimension index is out of range"))
            })
            .collect()
    };
    let lb = in_range(&lb, ls.len())?;
    let lc = in_range(&lc, ls.len())?;
    let rb = in_range(&rb, rs.len())?;
    let rc = in_range(&rc, rs.len())?;
    let compatible = |a: i64, b: i64| a < 0 || b < 0 || a == b;
    if lb.iter().zip(&rb).any(|(&a, &b)| !compatible(ls[a], rs[b])) {
        return Err(invalid(
            NAME,
            "batching dimension sizes must match for lhs/rhs",
        ));
    }
    if lc.iter().zip(&rc).any(|(&a, &b)| !compatible(ls[a], rs[b])) {
        return Err(invalid(
            NAME,
            "contracting dimension sizes must match for lhs/rhs",
        ));
    }
    if lr.len() != 1 {
        return Err(invalid(
            NAME,
            "There must be exactly one ragged dimension in the lhs.",
        ));
    }
    let lr = in_range(&lr, ls.len())?[0];
    #[derive(PartialEq)]
    enum Mode {
        NonContracting,
        Contracting,
        Batch,
    }
    let mode = if lb.contains(&lr) {
        Mode::Batch
    } else if lc.contains(&lr) {
        Mode::Contracting
    } else {
        Mode::NonContracting
    };
    let mut prefix: Vec<usize> = lb.clone();
    match mode {
        Mode::Batch => prefix.truncate(lb.iter().position(|&x| x == lr).unwrap_or(lb.len())),
        Mode::Contracting => prefix.extend(lc.iter().take_while(|&&x| x != lr)),
        Mode::NonContracting => {
            prefix.extend((0..lr).filter(|i| !lb.contains(i) && !lc.contains(i)))
        }
    }
    let expected: Vec<i64> = prefix.iter().map(|&i| ls[i]).collect();
    if gs.len() != 1 {
        if gs.len() != expected.len() + 1 {
            return Err(invalid(NAME, "group_sizes has an unexpected rank"));
        }
        if gs[..expected.len()] != expected[..] {
            return Err(invalid(NAME, "group_sizes has an unexpected shape"));
        }
    }
    let n_groups = *gs
        .last()
        .ok_or_else(|| invalid(NAME, "group_sizes must have rank at least one"))?;
    let rg = in_range(&rg, rs.len())?;
    if rg.iter().any(|g| rb.contains(g) || rc.contains(g)) {
        return Err(invalid(
            NAME,
            "rhs_group_dimensions overlap rhs batching or contracting dimensions",
        ));
    }
    match mode {
        Mode::Batch | Mode::Contracting => {
            if !rg.is_empty() {
                return Err(invalid(
                    NAME,
                    "There must be zero group dimensions in the rhs when the ragged dimension is batch or contracting.",
                ));
            }
        }
        Mode::NonContracting => {
            let [g] = rg[..] else {
                return Err(invalid(
                    NAME,
                    "There must be exactly one group dimension in the rhs when the lhs ragged dimension is non-contracting.",
                ));
            };
            if !compatible(n_groups, rs[g]) {
                return Err(invalid(
                    NAME,
                    "rhs group dimension size differs from group_sizes",
                ));
            }
        }
    }
    let mut out = Vec::new();
    if mode == Mode::Contracting {
        out.push(n_groups);
    }
    out.extend(lb.iter().map(|&i| ls[i]));
    out.extend(
        ls.iter()
            .enumerate()
            .filter(|(i, _)| !lb.contains(i) && !lc.contains(i))
            .map(|(_, &d)| d),
    );
    out.extend(
        rs.iter()
            .enumerate()
            .filter(|(i, _)| !rb.contains(i) && !rc.contains(i) && !rg.contains(i))
            .map(|(_, &d)| d),
    );
    Ok((out, lhs.element))
}
impl Verify for RaggedDotOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        const NAME: &str = "chlo.ragged_dot";
        let ir = self.get_operation().deref(ctx);
        verify_pure_shape(ctx, &ir, NAME, 3, 1)?;
        let d = self
            .dimension_numbers(ctx)
            .ok_or_else(|| fail(ctx, &ir, NAME, "ragged_dot_dimension_numbers is missing"))?;
        let precision_len = self
            .precision_config_attr(ctx)
            .map_or(0, |p| p.values.len());
        let (shape, _) = ragged_dot_shape(
            ctx,
            ir.get_operand(0).get_type(ctx),
            ir.get_operand(1).get_type(ctx),
            ir.get_operand(2).get_type(ctx),
            &d,
            precision_len,
        )?;
        let result = typed(
            ctx,
            ir.get_result(0).get_type(ctx),
            Domain::Any,
            false,
            NAME,
        )?;
        let inferred = Shaped {
            dimensions: Some(shape),
            element: result.element,
            encoding: VecAttr::new(vec![]),
            bounds: None,
        };
        if !inference::compatible_shape(&inferred, &result) {
            return Err(fail(
                ctx,
                &ir,
                NAME,
                "inferred shape is incompatible with return type of operation",
            ));
        }
        Ok(())
    }
}

fn scan_slice_type(ctx: &Context, ty: TypeHandle, dimension: usize) -> Result<TypeHandle> {
    use stablehlo::types::TensorType;
    ty.deref(ctx).verify(ctx)?;
    if !inference::hlo_tensor_type(ctx, ty, inference::Domain::Any, true) {
        return Err(invalid(
            "chlo.scan",
            "inputs must be ranked upstream HLO tensors",
        ));
    }
    let object = ty.deref(ctx);
    let tensor = object
        .downcast_ref::<TensorType>()
        .ok_or_else(|| invalid("chlo.scan", "operands must be ranked tensors"))?;
    if dimension >= tensor.rank() {
        return Err(invalid("chlo.scan", "scan dimension is out of bounds"));
    }
    let mut shape = tensor.dimensions.clone();
    shape.remove(dimension);
    let mut encoding = tensor.encoding.clone();
    if let Some(bounds) = tensor.bounds() {
        let mut bounds = bounds.to_vec();
        bounds.remove(dimension);
        encoding = inference::bounds_encoding(bounds);
    }
    Ok(inference::shaped_type(
        ctx,
        Some(shape),
        tensor.element,
        encoding,
    ))
}

fn scan_parameters(
    ctx: &Context,
    inputs: &[Value],
    dimension: u64,
    size: Option<u64>,
) -> Result<(usize, i64, i64)> {
    use stablehlo::types::TensorType;
    let dimension = i64::try_from(dimension)
        .ok()
        .and_then(|dim| usize::try_from(dim).ok())
        .ok_or_else(|| invalid("chlo.scan", "dimension must be a nonnegative signed i64"))?;
    let mut inferred_size = match size {
        Some(size) => i64::try_from(size).map_err(|_| {
            invalid(
                "chlo.scan",
                "scan_dim_size must be a nonnegative signed i64",
            )
        })?,
        None => -1,
    };
    let mut bound = -1;
    for input in inputs {
        let ty = input.get_type(ctx);
        scan_slice_type(ctx, ty, dimension)?;
        let object = ty.deref(ctx);
        let tensor = object
            .downcast_ref::<TensorType>()
            .ok_or_else(|| invalid("chlo.scan", "inputs must be ranked tensors"))?;
        let actual = tensor.dimensions[dimension];
        if actual >= 0 {
            if inferred_size >= 0 && actual != inferred_size {
                return Err(invalid("chlo.scan", "scan dimension sizes differ"));
            }
            inferred_size = actual;
        }
        if let Some(current) = tensor
            .bounds()
            .map(|bounds| bounds[dimension])
            .filter(|bound| *bound >= 0)
        {
            bound = if bound < 0 {
                current
            } else {
                bound.min(current)
            };
        }
    }
    Ok((dimension, inferred_size, bound))
}

fn scan_infer_results(
    ctx: &Context,
    inputs: &[Value],
    ncarry: usize,
    dimension: u64,
    size: Option<u64>,
    returned: &[Value],
) -> Result<Vec<TypeHandle>> {
    use stablehlo::types::TensorType;
    let (dimension, size, bound) = scan_parameters(ctx, inputs, dimension, size)?;
    let noutput = returned
        .len()
        .checked_sub(ncarry)
        .ok_or_else(|| invalid("chlo.scan", "body must return at least one value per carry"))?;
    let mut results = Vec::with_capacity(returned.len());
    for (index, value) in returned.iter().enumerate() {
        let ty = value.get_type(ctx);
        ty.deref(ctx).verify(ctx)?;
        if !inference::hlo_tensor_type(ctx, ty, inference::Domain::Any, true) {
            return Err(invalid(
                "chlo.scan",
                "body returns must be ranked upstream HLO tensors",
            ));
        }
        let object = ty.deref(ctx);
        let tensor = object
            .downcast_ref::<TensorType>()
            .ok_or_else(|| invalid("chlo.scan", "body returns must be ranked tensors"))?;
        if index >= noutput {
            results.push(ty);
            continue;
        }
        if dimension > tensor.rank() {
            return Err(invalid(
                "chlo.scan",
                "scan dimension is out of bounds for body output",
            ));
        }
        let mut shape = tensor.dimensions.clone();
        shape.insert(dimension, size);
        let insert_bound = if size < 0 { bound } else { -1 };
        let encoding = if tensor.bounds().is_some() || insert_bound >= 0 {
            let mut bounds = tensor
                .bounds()
                .map_or_else(|| vec![-1; tensor.rank()], |b| b.to_vec());
            bounds.insert(dimension, insert_bound);
            inference::bounds_encoding(bounds)
        } else {
            tensor.encoding.clone()
        };
        let output = inference::shaped_type(ctx, Some(shape), tensor.element, encoding);
        output.deref(ctx).verify(ctx)?;
        results.push(output);
    }
    Ok(results)
}

fn scan_compatible(ctx: &Context, actual: TypeHandle, expected: TypeHandle) -> bool {
    compatible_tensors(ctx, actual, expected)
}

/// Scans tensor slices while threading loop-carried tensors through an isolated body.
///
/// Operands are ordered as scan inputs followed by initial carries. The body
/// receives sliced inputs followed by carries and terminates with
/// `stablehlo.return`, yielding per-step outputs followed by updated carries.
/// Results are stacked outputs followed by final carries. The body cannot
/// capture values from outside the operation.
#[pliron_op(name="chlo.scan",format="operands(CharSpace(`,`)) attr_dict `{` region($0) `}` `:` typesig",interfaces=[pliron::builtin::op_interfaces::IsolatedFromAboveInterface],attributes=(chlo_scan_dimension:IntegerAttr,chlo_scan_dim_size:IntegerAttr,chlo_scan_is_reverse:BoolAttr,chlo_scan_is_associative:BoolAttr,chlo_scan_operand_segment_sizes:VecAttr,chlo_scan_result_segment_sizes:VecAttr))]
pub struct ScanOp;
impl ScanOp {
    /// Allocates a scan with explicit result types and an unterminated entry block.
    ///
    /// `dimension` is the zero-based scanned axis. `scan_dim_size` supplies an
    /// explicit iteration count when needed; input axis sizes otherwise determine
    /// it. `is_reverse` traverses that axis backwards, while `is_associative`
    /// optionally records whether reassociation is permitted.
    ///
    /// `results` contains stacked output types followed by one carry type per
    /// `init`. The entry block is created with sliced input arguments followed
    /// by carry arguments. If slicing fails, this unchecked constructor keeps the
    /// original input type as a placeholder. Populate the block and its return,
    /// then call `verify`; this constructor performs no validation.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: &mut Context,
        inputs: Vec<Value>,
        inits: Vec<Value>,
        results: Vec<TypeHandle>,
        dimension: u64,
        scan_dim_size: Option<u64>,
        is_reverse: bool,
        is_associative: Option<bool>,
    ) -> Self {
        use pliron::basic_block::BasicBlock;
        let all: Vec<Value> = inputs.iter().chain(&inits).copied().collect();
        let output_count = results.len().saturating_sub(inits.len());
        let result_count = results.len();
        let op = Self {
            op: Operation::new(ctx, Self::get_concrete_op_info(), results, all, vec![], 1),
        };
        let int_ty = pliron::builtin::types::IntegerType::get(
            ctx,
            64,
            pliron::builtin::types::Signedness::Signless,
        );
        op.set_attr_chlo_scan_dimension(
            ctx,
            IntegerAttr::new(int_ty, APInt::from_u64(dimension, bw(64))),
        );
        if let Some(size) = scan_dim_size {
            op.set_attr_chlo_scan_dim_size(
                ctx,
                IntegerAttr::new(int_ty, APInt::from_u64(size, bw(64))),
            );
        }
        op.set_attr_chlo_scan_is_reverse(ctx, BoolAttr::new(is_reverse));
        if let Some(value) = is_associative {
            op.set_attr_chlo_scan_is_associative(ctx, BoolAttr::new(value));
        }
        op.set_attr_chlo_scan_operand_segment_sizes(
            ctx,
            signed_list(ctx, &[inputs.len() as i64, inits.len() as i64]),
        );
        op.set_attr_chlo_scan_result_segment_sizes(
            ctx,
            signed_list(
                ctx,
                &[
                    output_count as i64,
                    result_count.saturating_sub(output_count) as i64,
                ],
            ),
        );
        let mut args: Vec<TypeHandle> = Vec::new();
        for input in &inputs {
            let ty = input.get_type(ctx);
            if let Ok(d) = usize::try_from(dimension)
                && let Ok(sliced) = scan_slice_type(ctx, ty, d)
            {
                args.push(sliced);
                continue;
            }
            args.push(ty);
        }
        args.extend(inits.iter().map(|v| v.get_type(ctx)));
        let block = BasicBlock::new(ctx, None, args);
        block.insert_at_back(op.get_operation().deref(ctx).get_region(0), ctx);
        op
    }
    /// Builds the body before inferring outputs and carries from its return.
    ///
    /// Operands and returned values must be ranked. The closure may adjust the
    /// generated argument types, including to compatible unranked tensor types.
    /// Inputs are sliced along the zero-based `dimension`; `scan_dim_size` may
    /// supply the iteration count and `is_reverse` selects backwards traversal.
    /// `is_associative` records an optional reassociation permission.
    ///
    /// The closure must populate the isolated entry block and terminate it with
    /// `stablehlo.return`, returning per-step outputs followed by one updated
    /// carry per `init`. Inference inserts the scan axis into output shapes and
    /// preserves compatible carry types. The completed operation is verified.
    ///
    /// # Errors
    ///
    /// Propagates closure errors and returns an error for invalid scan parameters,
    /// unsupported tensor types, an invalid body or return, incompatible carries,
    /// or output types that cannot be inferred. Allocation is not transactional:
    /// an error after body construction can leave allocated IR in `ctx`.
    #[allow(clippy::too_many_arguments)]
    pub fn new_inferred(
        ctx: &mut Context,
        inputs: Vec<Value>,
        inits: Vec<Value>,
        dimension: u64,
        scan_dim_size: Option<u64>,
        is_reverse: bool,
        is_associative: Option<bool>,
        build_body: impl FnOnce(
            &mut Context,
            pliron::context::Ptr<pliron::basic_block::BasicBlock>,
        ) -> Result<()>,
    ) -> Result<Self> {
        scan_parameters(ctx, &inputs, dimension, scan_dim_size)?;
        for init in &inits {
            init.get_type(ctx).deref(ctx).verify(ctx)?;
            if !inference::hlo_tensor_type(ctx, init.get_type(ctx), inference::Domain::Any, true) {
                return Err(invalid("chlo.scan", "inits must be ranked tensors"));
            }
        }
        let op = Self::new(
            ctx,
            inputs.clone(),
            inits.clone(),
            vec![],
            dimension,
            scan_dim_size,
            is_reverse,
            is_associative,
        );
        let block = op
            .entry_block(ctx)
            .ok_or_else(|| invalid("chlo.scan", "body block is missing"))?;
        build_body(ctx, block)?;
        let returned = op.body_returns(ctx)?;
        let types = scan_infer_results(
            ctx,
            &inputs,
            inits.len(),
            dimension,
            scan_dim_size,
            &returned,
        )?;
        let noutput = types
            .len()
            .checked_sub(inits.len())
            .ok_or_else(|| invalid("chlo.scan", "body has too few carries"))?;
        for ty in types {
            Operation::push_result(op.get_operation(), ctx, ty);
        }
        op.set_attr_chlo_scan_result_segment_sizes(
            ctx,
            signed_list(ctx, &[noutput as i64, inits.len() as i64]),
        );
        op.verify(ctx)?;
        Ok(op)
    }
    fn body_returns(&self, ctx: &Context) -> Result<Vec<Value>> {
        use pliron::linked_list::ContainsLinkedList;
        let ir = self.get_operation().deref(ctx);
        if ir.regions().count() != 1 {
            return Err(invalid(
                "chlo.scan",
                "scan requires exactly one body region",
            ));
        }
        let blocks: Vec<_> = ir.get_region(0).deref(ctx).iter(ctx).collect();
        if blocks.len() != 1 {
            return Err(invalid("chlo.scan", "scan requires exactly one body block"));
        }
        let last = blocks[0]
            .deref(ctx)
            .iter(ctx)
            .last()
            .ok_or_else(|| invalid("chlo.scan", "body is empty"))?;
        if Operation::get_op_dyn(last, ctx)
            .downcast_ref::<stablehlo::ops::ReturnOp>()
            .is_none()
        {
            return Err(invalid(
                "chlo.scan",
                "body must terminate with stablehlo.return",
            ));
        }
        verify_pure_structure(&last.deref(ctx), "chlo.scan")?;
        Ok(last.deref(ctx).operands().collect())
    }
    /// Reads the scanned axis as an unsigned integer.
    ///
    /// Returns `None` if the attribute is absent, not an integer, or not
    /// representable as `u64`. Full attribute constraints are checked by `verify`.
    pub fn dimension(&self, ctx: &Context) -> Option<u64> {
        self.get_attr_chlo_scan_dimension(ctx)?
            .value()
            .to_string_decimal(false)
            .parse()
            .ok()
    }
    /// Reads the traversal direction, defaulting to `Some(false)` when omitted.
    ///
    /// Returns `None` when an explicitly stored attribute is not a boolean.
    pub fn is_reverse(&self, ctx: &Context) -> Option<bool> {
        if !self
            .get_operation()
            .deref(ctx)
            .attributes
            .0
            .contains_key(&pliron::ident!("chlo_scan_is_reverse"))
        {
            return Some(false);
        }
        self.get_attr_chlo_scan_is_reverse(ctx)
            .map(|v| bool::from((*v).clone()))
    }
    /// Reads the explicit reassociation permission, or `None` if absent or malformed.
    ///
    /// No default permission is synthesized.
    pub fn is_associative(&self, ctx: &Context) -> Option<bool> {
        self.get_attr_chlo_scan_is_associative(ctx)
            .map(|v| bool::from((*v).clone()))
    }
    /// Reads the optional explicit iteration count without inferring it from the inputs.
    ///
    /// Returns `None` for an absent, incorrectly typed, or unrepresentable attribute.
    pub fn scan_dim_size(&self, ctx: &Context) -> Option<u64> {
        self.get_attr_chlo_scan_dim_size(ctx)?
            .value()
            .to_string_decimal(false)
            .parse()
            .ok()
    }
    /// Returns the first block when exactly one body region exists.
    ///
    /// Returns `None` for a missing/multiple region or an empty region. This does
    /// not check that the region has only one block or a valid terminator.
    pub fn entry_block(
        &self,
        ctx: &Context,
    ) -> Option<pliron::context::Ptr<pliron::basic_block::BasicBlock>> {
        use pliron::linked_list::ContainsLinkedList;
        let ir = self.get_operation().deref(ctx);
        if ir.regions().count() != 1 {
            return None;
        }
        ir.get_region(0).deref(ctx).iter(ctx).next()
    }
    /// Copies scan inputs using the first operand-segment count.
    ///
    /// Returns `None` when that count cannot be read as a nonnegative `usize`.
    /// Segment consistency and operand count are checked by `verify`, not here.
    pub fn inputs(&self, ctx: &Context) -> Option<Vec<Value>> {
        let n = read_signed_list(
            self.get_attr_chlo_scan_operand_segment_sizes(ctx)
                .as_deref()?,
        )?;
        let count = usize::try_from(*n.first()?).ok()?;
        Some(
            self.get_operation()
                .deref(ctx)
                .operands()
                .take(count)
                .collect(),
        )
    }
    /// Copies initial carry operands after the scan-input segment.
    ///
    /// Returns `None` when the input segment count cannot be decoded.
    /// This accessor does not validate either segment's declared length.
    pub fn inits(&self, ctx: &Context) -> Option<Vec<Value>> {
        let n = read_signed_list(
            self.get_attr_chlo_scan_operand_segment_sizes(ctx)
                .as_deref()?,
        )?;
        let start = usize::try_from(*n.first()?).ok()?;
        Some(
            self.get_operation()
                .deref(ctx)
                .operands()
                .skip(start)
                .collect(),
        )
    }
}
impl Verify for ScanOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        use pliron::linked_list::ContainsLinkedList;
        let ir = self.get_operation().deref(ctx);
        if ir.get_num_successors() != 0 {
            return Err(invalid("chlo.scan", "scan must not have successors"));
        }
        let operands: Vec<Value> = ir.operands().collect();
        let results: Vec<Value> = ir.results().collect();
        let seg = read_signed_list(
            self.get_attr_chlo_scan_operand_segment_sizes(ctx)
                .as_deref()
                .ok_or_else(|| invalid("chlo.scan", "operand segment sizes missing"))?,
        )
        .ok_or_else(|| invalid("chlo.scan", "operand segment sizes malformed"))?;
        let rseg = read_signed_list(
            self.get_attr_chlo_scan_result_segment_sizes(ctx)
                .as_deref()
                .ok_or_else(|| invalid("chlo.scan", "result segment sizes missing"))?,
        )
        .ok_or_else(|| invalid("chlo.scan", "result segment sizes malformed"))?;
        if seg.len() != 2
            || rseg.len() != 2
            || seg.iter().chain(&rseg).any(|v| *v < 0)
            || seg[0]
                .checked_add(seg[1])
                .and_then(|n| usize::try_from(n).ok())
                != Some(operands.len())
            || rseg[0]
                .checked_add(rseg[1])
                .and_then(|n| usize::try_from(n).ok())
                != Some(results.len())
        {
            return Err(invalid("chlo.scan", "invalid segment sizes"));
        }
        let ninput = seg[0] as usize;
        let ninit = seg[1] as usize;
        if ninput == 0 && rseg[0] == 0 {
            return Err(invalid(
                "chlo.scan",
                "at least one input or output is required",
            ));
        }
        if rseg[1] != ninit as i64 {
            return Err(invalid(
                "chlo.scan",
                "carry result count must match init count",
            ));
        }
        let dimension = self
            .dimension(ctx)
            .ok_or_else(|| invalid("chlo.scan", "dimension is missing or malformed"))?;
        let dimension_attr = self
            .get_attr_chlo_scan_dimension(ctx)
            .ok_or_else(|| invalid("chlo.scan", "dimension is missing or malformed"))?;
        dimension_attr.verify(ctx)?;
        if dimension_attr.get_type().deref(ctx).width() != 64
            || dimension_attr.get_type().deref(ctx).signedness()
                != pliron::builtin::types::Signedness::Signless
        {
            return Err(invalid(
                "chlo.scan",
                "dimension attribute must be signless i64",
            ));
        }
        if ir
            .attributes
            .0
            .contains_key(&pliron::ident!("chlo_scan_dim_size"))
            && self.get_attr_chlo_scan_dim_size(ctx).is_none()
        {
            return Err(invalid("chlo.scan", "scan_dim_size attribute must be i64"));
        }
        if let Some(size) = self.get_attr_chlo_scan_dim_size(ctx) {
            size.verify(ctx)?;
            if size.get_type().deref(ctx).width() != 64
                || size.get_type().deref(ctx).signedness()
                    != pliron::builtin::types::Signedness::Signless
            {
                return Err(invalid(
                    "chlo.scan",
                    "scan_dim_size attribute must be signless i64",
                ));
            }
        }
        for name in [
            pliron::ident!("chlo_scan_is_reverse"),
            pliron::ident!("chlo_scan_is_associative"),
        ] {
            if ir
                .attributes
                .0
                .get(&name)
                .is_some_and(|attr| attr.downcast_ref::<BoolAttr>().is_none())
            {
                return Err(invalid(
                    "chlo.scan",
                    "scan options must be boolean attributes",
                ));
            }
        }
        let dim_size = self.scan_dim_size(ctx);
        let (dim, _, _) = scan_parameters(ctx, &operands[..ninput], dimension, dim_size)?;
        if ir.regions().count() != 1 {
            return Err(invalid(
                "chlo.scan",
                "scan requires exactly one body region",
            ));
        }
        let blocks: Vec<_> = ir.get_region(0).deref(ctx).iter(ctx).collect();
        if blocks.len() != 1 {
            return Err(invalid(
                "chlo.scan",
                "scan body must contain exactly one block",
            ));
        }
        let block = blocks[0].deref(ctx);
        let args: Vec<Value> = block.arguments().collect();
        if args.len() != operands.len() {
            return Err(invalid(
                "chlo.scan",
                "body argument count must match operands",
            ));
        }
        for (i, (operand, arg)) in operands.iter().zip(&args).enumerate() {
            let ty = operand.get_type(ctx);
            ty.deref(ctx).verify(ctx)?;
            arg.get_type(ctx).deref(ctx).verify(ctx)?;
            if !inference::hlo_tensor_type(ctx, ty, inference::Domain::Any, true)
                || !inference::hlo_tensor_type(
                    ctx,
                    arg.get_type(ctx),
                    inference::Domain::Any,
                    false,
                )
            {
                return Err(invalid(
                    "chlo.scan",
                    "operands must be ranked and body arguments must be upstream HLO tensors",
                ));
            }
            let expected = if i < ninput {
                scan_slice_type(ctx, ty, dim)?
            } else {
                ty
            };
            if !scan_compatible(ctx, arg.get_type(ctx), expected) {
                return Err(invalid(
                    "chlo.scan",
                    "body argument type does not match input element slice or carry",
                ));
            }
        }
        let returned = self.body_returns(ctx)?;
        if returned.len() != results.len() {
            return Err(invalid(
                "chlo.scan",
                "body return count must match output and carry segments",
            ));
        }
        let inferred = scan_infer_results(
            ctx,
            &operands[..ninput],
            ninit,
            dimension,
            dim_size,
            &returned,
        )?;
        for (result, expected) in results.iter().zip(inferred) {
            result.get_type(ctx).deref(ctx).verify(ctx)?;
            let actual = result.get_type(ctx);
            let shapes_match = match (
                inference::shaped(ctx, actual),
                inference::shaped(ctx, expected),
            ) {
                (Some(actual), Some(expected)) => inference::compatible_shape(&actual, &expected),
                _ => false,
            };
            if !inference::hlo_tensor_type(ctx, actual, inference::Domain::Any, false)
                || !shapes_match
            {
                return Err(invalid(
                    "chlo.scan",
                    "result shape does not match body return",
                ));
            }
        }
        let mut allowed: std::collections::HashSet<Value> = args.iter().copied().collect();
        let mut pending: Vec<_> = block.iter(ctx).collect();
        let mut body_ops = Vec::new();
        while let Some(body_op) = pending.pop() {
            let body_ir = body_op.deref(ctx);
            allowed.extend(body_ir.results());
            for region in body_ir.regions() {
                for nested in region.deref(ctx).iter(ctx) {
                    allowed.extend(nested.deref(ctx).arguments());
                    pending.extend(nested.deref(ctx).iter(ctx));
                }
            }
            body_ops.push(body_op);
        }
        for body_op in &body_ops {
            if body_op
                .deref(ctx)
                .operands()
                .any(|operand| !allowed.contains(&operand))
            {
                return Err(invalid(
                    "chlo.scan",
                    "body must not capture values defined outside the scan",
                ));
            }
        }
        // Check nested operations as well as the inferred signature: valid types alone
        // do not make an invalid body legal.
        for body_op in block.iter(ctx) {
            body_op.deref(ctx).verify(ctx)?;
        }
        let mut dominance = pliron::graph::dominance::DomInfo::default();
        for body_op in body_ops {
            for operand in body_op.deref(ctx).operands() {
                if !dominance.value_strictly_dominates_op(ctx, operand, body_op) {
                    return Err(invalid(
                        "chlo.scan",
                        "body operand is out of scope or does not dominate its use",
                    ));
                }
            }
        }
        Ok(())
    }
}
