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
    location::Located,
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
    ($struct:ident,$name:literal,$domain:expr,$rule:expr) => {
        #[pliron_op(name=$name,format="$0 attr_dict `:` typesig",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
        pub struct $struct;
        impl $struct {
            pub fn new(ctx: &mut Context, input: Value, result: TypeHandle) -> Self {
                Self { op: Operation::new(ctx, Self::get_concrete_op_info(), vec![result], vec![input], vec![], 0) }
            }
            pub fn new_inferred(ctx: &mut Context, input: Value) -> Result<Self> {
                let ty = infer_unary(ctx, input.get_type(ctx), $name, $domain, $rule)?;
                Ok(Self::new(ctx, input, ty))
            }
            pub fn operand(&self, ctx: &Context) -> Value { self.get_operation().deref(ctx).get_operand(0) }
            pub fn input(&self, ctx: &Context) -> Value { self.operand(ctx) }
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
    Domain::Complex,
    Same
);
unary_op!(AcosOp, "chlo.acos", Domain::FloatOrComplex, Same);
unary_op!(AcoshOp, "chlo.acosh", Domain::FloatOrComplex, Same);
unary_op!(AsinOp, "chlo.asin", Domain::FloatOrComplex, Same);
unary_op!(AsinhOp, "chlo.asinh", Domain::FloatOrComplex, Same);
unary_op!(AtanOp, "chlo.atan", Domain::FloatOrComplex, Same);
unary_op!(AtanhOp, "chlo.atanh", Domain::FloatOrComplex, Same);
unary_op!(BesselI1eOp, "chlo.bessel_i1e", Domain::FloatOrComplex, Same);
unary_op!(ConjOp, "chlo.conj", Domain::FloatOrComplex, Same);
unary_op!(CoshOp, "chlo.cosh", Domain::FloatOrComplex, Same);
unary_op!(SinhOp, "chlo.sinh", Domain::FloatOrComplex, Same);
unary_op!(TanOp, "chlo.tan", Domain::FloatOrComplex, Same);
unary_op!(DigammaOp, "chlo.digamma", Domain::Float, Same);
unary_op!(ErfOp, "chlo.erf", Domain::Float, Same);
unary_op!(ErfInvOp, "chlo.erf_inv", Domain::Float, Same);
unary_op!(ErfcOp, "chlo.erfc", Domain::Float, Same);
unary_op!(LgammaOp, "chlo.lgamma", Domain::Float, Same);
unary_op!(SquareOp, "chlo.square", Domain::FloatOrComplex, Same);
unary_op!(IsInfOp, "chlo.is_inf", Domain::Float, IsPred);
unary_op!(IsNegInfOp, "chlo.is_neg_inf", Domain::Float, IsPred);
unary_op!(IsPosInfOp, "chlo.is_pos_inf", Domain::Float, IsPred);

/// `HLO_CompatibleOperandsAndResultType` inference over `inferMostSpecificType`.
fn infer_compatible(
    ctx: &Context,
    operands: &[TypeHandle],
    operation: &'static str,
    domain: Domain,
    ranked_only: bool,
) -> Result<TypeHandle> {
    for &ty in operands {
        typed(ctx, ty, domain, ranked_only, operation)?;
    }
    if operands
        .iter()
        .any(|&ty| !inference::compatible_types(ctx, operands[0], ty))
    {
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
    let inferred = infer_compatible(ctx, &operands, operation, domain, ranked_only)
        .map_err(|_| fail(ctx, &ir, operation, "operand types are incompatible"))?;
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
    ($struct:ident,$name:literal,$domain:expr,$ranked:expr) => {
        #[pliron_op(name=$name,format="$0 `,` $1 attr_dict `:` typesig",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
        pub struct $struct;
        impl $struct {
            pub fn new(ctx: &mut Context, lhs: Value, rhs: Value, result: TypeHandle) -> Self {
                Self { op: Operation::new(ctx, Self::get_concrete_op_info(), vec![result], vec![lhs, rhs], vec![], 0) }
            }
            pub fn new_inferred(ctx: &mut Context, lhs: Value, rhs: Value) -> Result<Self> {
                let ty = infer_compatible(ctx, &[lhs.get_type(ctx), rhs.get_type(ctx)], $name, $domain, $ranked)?;
                Ok(Self::new(ctx, lhs, rhs, ty))
            }
            pub fn lhs(&self, ctx: &Context) -> Value { self.get_operation().deref(ctx).get_operand(0) }
            pub fn rhs(&self, ctx: &Context) -> Value { self.get_operation().deref(ctx).get_operand(1) }
        }
        impl Verify for $struct {
            fn verify(&self, ctx: &Context) -> Result<()> { check_compatible(ctx, self, $name, $domain, $ranked) }
        }
    };
}
binary_op!(NextAfterOp, "chlo.next_after", Domain::Float, false);
binary_op!(PolygammaOp, "chlo.polygamma", Domain::Float, false);
binary_op!(ZetaOp, "chlo.zeta", Domain::Float, false);
binary_op!(MulhiOp, "chlo.mulhi", Domain::Int, true);

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
        .map_err(|_| fail(ctx, ir, operation, "mismatched operand types"))?;
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
    ($struct:ident,$name:literal,$attr:ident,$setter:ident,$getter:ident,$domain:expr,$result_domain:expr,$trait:expr) => {
        #[pliron_op(name=$name,format="$0 `,` $1 attr_dict `:` typesig",interfaces=[pliron::builtin::op_interfaces::OneResultInterface],attributes=($attr:VecAttr))]
        pub struct $struct;
        impl $struct {
            pub fn new(ctx: &mut Context, lhs: Value, rhs: Value, result: TypeHandle, dimensions: Option<Vec<i64>>) -> Self {
                let op = Self { op: Operation::new(ctx, Self::get_concrete_op_info(), vec![result], vec![lhs, rhs], vec![], 0) };
                if let Some(d) = dimensions { op.$setter(ctx, signed_list(ctx, &d)); }
                op
            }
            pub fn new_inferred(ctx: &mut Context, lhs: Value, rhs: Value, dimensions: Option<Vec<i64>>) -> Result<Self> {
                let a = typed(ctx, lhs.get_type(ctx), $domain, false, $name)?;
                let b = typed(ctx, rhs.get_type(ctx), $domain, false, $name)?;
                if $trait == ElementTrait::Same && a.element != b.element {
                    return Err(invalid($name, "operand element types must be identical"));
                }
                let result = infer_broadcast_binary(ctx, &a, &b, dimensions.as_deref(), None, $name)?;
                Ok(Self::new(ctx, lhs, rhs, result, dimensions))
            }
            pub fn lhs(&self, ctx: &Context) -> Value { self.get_operation().deref(ctx).get_operand(0) }
            pub fn rhs(&self, ctx: &Context) -> Value { self.get_operation().deref(ctx).get_operand(1) }
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
    chlo_broadcast_add_dimensions,
    set_attr_chlo_broadcast_add_dimensions,
    get_attr_chlo_broadcast_add_dimensions,
    Domain::Any,
    Domain::Any,
    CompatibleElements
);
broadcast_op!(
    BroadcastAtan2Op,
    "chlo.broadcast_atan2",
    chlo_broadcast_atan2_dimensions,
    set_attr_chlo_broadcast_atan2_dimensions,
    get_attr_chlo_broadcast_atan2_dimensions,
    Domain::Any,
    Domain::Any,
    SameElements
);
broadcast_op!(
    BroadcastDivideOp,
    "chlo.broadcast_divide",
    chlo_broadcast_divide_dimensions,
    set_attr_chlo_broadcast_divide_dimensions,
    get_attr_chlo_broadcast_divide_dimensions,
    Domain::Any,
    Domain::Any,
    SameElements
);
broadcast_op!(
    BroadcastMaximumOp,
    "chlo.broadcast_maximum",
    chlo_broadcast_maximum_dimensions,
    set_attr_chlo_broadcast_maximum_dimensions,
    get_attr_chlo_broadcast_maximum_dimensions,
    Domain::Any,
    Domain::Any,
    CompatibleElements
);
broadcast_op!(
    BroadcastMinimumOp,
    "chlo.broadcast_minimum",
    chlo_broadcast_minimum_dimensions,
    set_attr_chlo_broadcast_minimum_dimensions,
    get_attr_chlo_broadcast_minimum_dimensions,
    Domain::Any,
    Domain::Any,
    CompatibleElements
);
broadcast_op!(
    BroadcastMultiplyOp,
    "chlo.broadcast_multiply",
    chlo_broadcast_multiply_dimensions,
    set_attr_chlo_broadcast_multiply_dimensions,
    get_attr_chlo_broadcast_multiply_dimensions,
    Domain::Any,
    Domain::Any,
    SameElements
);
broadcast_op!(
    BroadcastNextAfterOp,
    "chlo.broadcast_next_after",
    chlo_broadcast_next_after_dimensions,
    set_attr_chlo_broadcast_next_after_dimensions,
    get_attr_chlo_broadcast_next_after_dimensions,
    Domain::Any,
    Domain::Any,
    SameElements
);
broadcast_op!(
    BroadcastPolygammaOp,
    "chlo.broadcast_polygamma",
    chlo_broadcast_polygamma_dimensions,
    set_attr_chlo_broadcast_polygamma_dimensions,
    get_attr_chlo_broadcast_polygamma_dimensions,
    Domain::Any,
    Domain::Any,
    SameElements
);
broadcast_op!(
    BroadcastPowerOp,
    "chlo.broadcast_power",
    chlo_broadcast_power_dimensions,
    set_attr_chlo_broadcast_power_dimensions,
    get_attr_chlo_broadcast_power_dimensions,
    Domain::Any,
    Domain::Any,
    SameElements
);
broadcast_op!(
    BroadcastRemainderOp,
    "chlo.broadcast_remainder",
    chlo_broadcast_remainder_dimensions,
    set_attr_chlo_broadcast_remainder_dimensions,
    get_attr_chlo_broadcast_remainder_dimensions,
    Domain::Any,
    Domain::Any,
    SameElements
);
broadcast_op!(
    BroadcastShiftLeftOp,
    "chlo.broadcast_shift_left",
    chlo_broadcast_shift_left_dimensions,
    set_attr_chlo_broadcast_shift_left_dimensions,
    get_attr_chlo_broadcast_shift_left_dimensions,
    Domain::Any,
    Domain::Any,
    SameElements
);
broadcast_op!(
    BroadcastShiftRightArithmeticOp,
    "chlo.broadcast_shift_right_arithmetic",
    chlo_broadcast_shift_right_arithmetic_dimensions,
    set_attr_chlo_broadcast_shift_right_arithmetic_dimensions,
    get_attr_chlo_broadcast_shift_right_arithmetic_dimensions,
    Domain::Any,
    Domain::Any,
    SameElements
);
broadcast_op!(
    BroadcastShiftRightLogicalOp,
    "chlo.broadcast_shift_right_logical",
    chlo_broadcast_shift_right_logical_dimensions,
    set_attr_chlo_broadcast_shift_right_logical_dimensions,
    get_attr_chlo_broadcast_shift_right_logical_dimensions,
    Domain::Any,
    Domain::Any,
    SameElements
);
broadcast_op!(
    BroadcastSubtractOp,
    "chlo.broadcast_subtract",
    chlo_broadcast_subtract_dimensions,
    set_attr_chlo_broadcast_subtract_dimensions,
    get_attr_chlo_broadcast_subtract_dimensions,
    Domain::Any,
    Domain::Any,
    SameElements
);
broadcast_op!(
    BroadcastZetaOp,
    "chlo.broadcast_zeta",
    chlo_broadcast_zeta_dimensions,
    set_attr_chlo_broadcast_zeta_dimensions,
    get_attr_chlo_broadcast_zeta_dimensions,
    Domain::Float,
    Domain::Float,
    SameElements
);
broadcast_op!(
    BroadcastAndOp,
    "chlo.broadcast_and",
    chlo_broadcast_and_dimensions,
    set_attr_chlo_broadcast_and_dimensions,
    get_attr_chlo_broadcast_and_dimensions,
    Domain::PredOrInt,
    Domain::Any,
    NoElementTrait
);
broadcast_op!(
    BroadcastOrOp,
    "chlo.broadcast_or",
    chlo_broadcast_or_dimensions,
    set_attr_chlo_broadcast_or_dimensions,
    get_attr_chlo_broadcast_or_dimensions,
    Domain::PredOrInt,
    Domain::Any,
    NoElementTrait
);
broadcast_op!(
    BroadcastXorOp,
    "chlo.broadcast_xor",
    chlo_broadcast_xor_dimensions,
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

#[pliron_op(name="chlo.broadcast_complex",format="$0 `,` $1 attr_dict `:` typesig",interfaces=[pliron::builtin::op_interfaces::OneResultInterface],attributes=(chlo_broadcast_complex_dimensions:VecAttr))]
pub struct BroadcastComplexOp;
impl BroadcastComplexOp {
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
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
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

#[pliron_op(name="chlo.broadcast_compare",format="$0 `,` $1 attr_dict `:` typesig",interfaces=[pliron::builtin::op_interfaces::OneResultInterface],attributes=(chlo_broadcast_compare_dimensions:VecAttr,chlo_comparison_direction:ComparisonDirectionAttr,chlo_comparison_type:ComparisonTypeAttr))]
pub struct BroadcastCompareOp;
impl BroadcastCompareOp {
    pub fn new(
        ctx: &mut Context,
        lhs: Value,
        rhs: Value,
        result: TypeHandle,
        dimensions: Option<Vec<i64>>,
        direction: ComparisonDirection,
        compare_type: Option<ComparisonType>,
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
            op.set_attr_chlo_broadcast_compare_dimensions(ctx, signed_list(ctx, &d));
        }
        op.set_attr_chlo_comparison_direction(ctx, ComparisonDirectionAttr::new(direction));
        if let Some(t) = compare_type {
            op.set_attr_chlo_comparison_type(ctx, ComparisonTypeAttr::new(t));
        }
        op
    }
    pub fn new_inferred(
        ctx: &mut Context,
        lhs: Value,
        rhs: Value,
        dimensions: Option<Vec<i64>>,
        direction: ComparisonDirection,
        compare_type: Option<ComparisonType>,
    ) -> Result<Self> {
        const NAME: &str = "chlo.broadcast_compare";
        let a = typed(ctx, lhs.get_type(ctx), Domain::Any, false, NAME)?;
        let b = typed(ctx, rhs.get_type(ctx), Domain::Any, false, NAME)?;
        let ty = infer_broadcast_binary(ctx, &a, &b, dimensions.as_deref(), Some(i1(ctx)), NAME)?;
        Ok(Self::new(
            ctx,
            lhs,
            rhs,
            ty,
            dimensions,
            direction,
            compare_type,
        ))
    }
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
    pub fn broadcast_dimensions(&self, ctx: &Context) -> Option<Vec<i64>> {
        let a = self.get_attr_chlo_broadcast_compare_dimensions(ctx)?;
        read_signed_list(&a)
    }
    pub fn comparison_direction(&self, ctx: &Context) -> Option<ComparisonDirection> {
        Some(self.get_attr_chlo_comparison_direction(ctx)?.value)
    }
    pub fn comparison_type(&self, ctx: &Context) -> Option<ComparisonType> {
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

#[pliron_op(name="chlo.broadcast_select",format="$0 `,` $1 `,` $2 attr_dict `:` typesig",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct BroadcastSelectOp;
impl BroadcastSelectOp {
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
    pub fn predicate(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    pub fn on_true(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
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
        let inferred = infer_select(ctx, &p, &t, &f, NAME)
            .map_err(|_| fail(ctx, &ir, NAME, "mismatched operand types"))?;
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

fn constant_element_value(ctx: &Context, value: &AttrObj) -> AttrObj {
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

#[pliron_op(name="chlo.constant",format="attr_dict `:` type($0)",interfaces=[pliron::builtin::op_interfaces::NOpdsInterface<0>,pliron::builtin::op_interfaces::OneResultInterface])]
pub struct ConstantOp;
impl ConstantOp {
    /// Compatibility word constructor; the sole stored payload is a typed `value`.
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
    pub fn value(&self, ctx: &Context) -> Option<AttrObj> {
        self.get_operation()
            .deref(ctx)
            .attributes
            .0
            .get(&pliron::ident!("value"))
            .cloned()
    }
    pub fn set_value(&self, ctx: &Context, value: AttrObj) {
        let value = constant_elements_value(ctx, value);
        self.get_operation()
            .deref_mut(ctx)
            .attributes
            .0
            .insert(pliron::ident!("value"), value);
    }
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

#[pliron_op(name="chlo.constant_like",format="$0 attr_dict `:` typesig",interfaces=[pliron::builtin::op_interfaces::OneResultInterface])]
pub struct ConstantLikeOp;
impl ConstantLikeOp {
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
    pub fn value(&self, ctx: &Context) -> Option<AttrObj> {
        self.get_operation()
            .deref(ctx)
            .attributes
            .0
            .get(&pliron::ident!("value"))
            .cloned()
    }
    pub fn set_value(&self, ctx: &Context, value: AttrObj) {
        self.get_operation()
            .deref_mut(ctx)
            .attributes
            .0
            .insert(pliron::ident!("value"), value);
    }
    pub fn from_value(ctx: &mut Context, value: AttrObj, operand: Value) -> Result<Self> {
        let element = constant_scalar_type(ctx, &value)?;
        let ty = constant_like_type(ctx, operand.get_type(ctx), element)?;
        let op = Self::new_with_value(ctx, value, operand, ty);
        op.verify(ctx)?;
        Ok(op)
    }
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

#[pliron_op(name="chlo.top_k",format="$0 attr_dict `:` typesig",interfaces=[pliron::builtin::op_interfaces::NResultsInterface<2>],attributes=(chlo_top_k_k:IntegerAttr,chlo_top_k_is_stable:BoolAttr))]
pub struct TopKOp;
impl TopKOp {
    pub fn new(
        ctx: &mut Context,
        operand: Value,
        k: i64,
        is_stable: bool,
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
                APInt::from_i64(k, bw(64)),
            ),
        );
        op.set_attr_chlo_top_k_is_stable(ctx, BoolAttr::new(is_stable));
        op
    }
    pub fn new_inferred(
        ctx: &mut Context,
        operand: Value,
        k: i64,
        is_stable: bool,
    ) -> Result<Self> {
        let input = typed(ctx, operand.get_type(ctx), Domain::Any, false, "chlo.top_k")?;
        let (values, indices) = infer_top_k(ctx, &input, k)?;
        Ok(Self::new(ctx, operand, k, is_stable, values, indices))
    }
    pub fn operand(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    pub fn values(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_result(0)
    }
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
    /// `is_stable`, defaulting to `true` when the attribute is omitted.
    pub fn is_stable(&self, ctx: &Context) -> Option<bool> {
        Some(
            self.get_attr_chlo_top_k_is_stable(ctx)
                .is_none_or(|v| bool::from((*v).clone())),
        )
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
        let (want_values, want_indices) = infer_top_k(ctx, &operand, k)
            .map_err(|_| fail(ctx, &ir, NAME, "operand is incompatible with k"))?;
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

#[pliron_op(name="chlo.ragged_dot",format="$0 `,` $1 `,` $2 attr_dict `:` typesig",interfaces=[pliron::builtin::op_interfaces::OneResultInterface],attributes=(chlo_ragged_dot_dimension_numbers:RaggedDotDimensionNumbersAttr,chlo_precision_config:PrecisionConfigAttr))]
pub struct RaggedDotOp;
impl RaggedDotOp {
    pub fn new(
        ctx: &mut Context,
        lhs: Value,
        rhs: Value,
        group_sizes: Value,
        result: TypeHandle,
        dimensions: RaggedDotDimensionNumbersAttr,
        precision_config: Option<Vec<Precision>>,
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
        op.set_attr_chlo_ragged_dot_dimension_numbers(ctx, dimensions);
        if let Some(values) = precision_config {
            op.set_attr_chlo_precision_config(ctx, crate::attributes::precision_config(values));
        }
        op
    }
    pub fn new_inferred(
        ctx: &mut Context,
        lhs: Value,
        rhs: Value,
        group_sizes: Value,
        dimensions: RaggedDotDimensionNumbersAttr,
        precision_config: Option<Vec<Precision>>,
    ) -> Result<Self> {
        let (shape, element) = ragged_dot_shape(
            ctx,
            lhs.get_type(ctx),
            rhs.get_type(ctx),
            group_sizes.get_type(ctx),
            &dimensions,
            precision_config.as_ref().map_or(0, Vec::len),
        )?;
        let result = inference::shaped_type(ctx, Some(shape), element, VecAttr::new(vec![]));
        Ok(Self::new(
            ctx,
            lhs,
            rhs,
            group_sizes,
            result,
            dimensions,
            precision_config,
        ))
    }
    pub fn lhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
    pub fn rhs(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
    pub fn group_sizes(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(2)
    }
    pub fn dimension_numbers(&self, ctx: &Context) -> Option<RaggedDotDimensionNumbersAttr> {
        self.get_attr_chlo_ragged_dot_dimension_numbers(ctx)
            .map(|v| (*v).clone())
    }
    pub fn precision_config_attr(&self, ctx: &Context) -> Option<PrecisionConfigAttr> {
        self.get_attr_chlo_precision_config(ctx)
            .map(|v| (*v).clone())
    }
    pub fn precision_config(&self, ctx: &Context) -> Option<Vec<Precision>> {
        read_precision_config(&*self.get_attr_chlo_precision_config(ctx)?)
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
    let (lb, rb, lc, rc, lr, rg) = d.dimensions();
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

#[pliron_op(name="chlo.scan",format="operands(CharSpace(`,`)) attr_dict `{` region($0) `}` `:` typesig",interfaces=[pliron::builtin::op_interfaces::IsolatedFromAboveInterface],attributes=(chlo_scan_dimension:IntegerAttr,chlo_scan_dim_size:IntegerAttr,chlo_scan_is_reverse:BoolAttr,chlo_scan_is_associative:BoolAttr,chlo_scan_operand_segment_sizes:VecAttr,chlo_scan_result_segment_sizes:VecAttr))]
pub struct ScanOp;
impl ScanOp {
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
    pub fn dimension(&self, ctx: &Context) -> Option<u64> {
        self.get_attr_chlo_scan_dimension(ctx)?
            .value()
            .to_string_decimal(false)
            .parse()
            .ok()
    }
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
    pub fn is_associative(&self, ctx: &Context) -> Option<bool> {
        self.get_attr_chlo_scan_is_associative(ctx)
            .map(|v| bool::from((*v).clone()))
    }
    pub fn scan_dim_size(&self, ctx: &Context) -> Option<u64> {
        self.get_attr_chlo_scan_dim_size(ctx)?
            .value()
            .to_string_decimal(false)
            .parse()
            .ok()
    }
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
