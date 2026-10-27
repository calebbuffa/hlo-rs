//! Tensor/broadcast semantics of the CHLO dialect against upstream ChloOps.td / ChloOps.cpp.
use super::{TensorMetadata as Shaped, tensor_metadata};
use clienthlo::{
    attributes::{
        ComparisonDirection, Precision, RaggedDotDimensionNumbersAttr, RaggedDotDimensions,
        read_precision_config,
    },
    inference,
    ops::*,
    pliron::{
        basic_block::BasicBlock,
        builtin::attributes::IntegerAttr,
        builtin::types::IntegerType,
        builtin::types::Signedness,
        common_traits::Verify,
        context::Context,
        op::Op,
        operation::Operation,
        r#type::{TypeHandle, Typed},
        utils::apint::{APInt, bw},
        value::Value,
    },
};
use stablehlo::{
    attributes::TypeExtensionsAttr,
    types::{ElementType, QuantizedElementType, TensorType, UniformQuantizedSpec, tensor},
};

fn arg(ctx: &mut Context, ty: TypeHandle) -> Value {
    BasicBlock::new(ctx, None, vec![ty])
        .deref(ctx)
        .arguments()
        .next()
        .unwrap()
}
fn val(ctx: &mut Context, shape: &[i64], element: ElementType) -> Value {
    let ty = tensor(ctx, shape, element);
    arg(ctx, ty)
}

#[test]
fn broadcast_shapes_preserve_axes_and_reject_incompatible_extents() {
    for (shapes, expected) in [
        (vec![vec![-1, 1, 4], vec![3, 1]], Some(vec![-1, 3, 4])),
        (vec![vec![0], vec![1]], Some(vec![0])),
        (vec![vec![2, 3], vec![4, 3]], None),
    ] {
        let shapes = shapes.iter().map(Vec::as_slice).collect::<Vec<_>>();
        assert_eq!(inference::numpy_broadcast_shape(&shapes), expected);
    }
    for (axes, expected) in [(vec![2, 1], Some(vec![2, 3, 4])), (vec![3, 1], None)] {
        assert_eq!(
            inference::infer_broadcast_shape(&[2, 3, 4], &[4, 3], Some(&axes)),
            expected,
            "axes={axes:?}"
        );
    }
}
fn unranked_ty(ctx: &Context, element: ElementType) -> TypeHandle {
    stablehlo::types::UnrankedTensorType::get(ctx, element.native(ctx)).into()
}
fn unranked(ctx: &mut Context, element: ElementType) -> Value {
    let ty = unranked_ty(ctx, element);
    arg(ctx, ty)
}
fn bounded_ty(ctx: &Context, shape: &[i64], bounds: &[i64], element: ElementType) -> TypeHandle {
    let ty = tensor(ctx, shape, element);
    ty.deref(ctx)
        .downcast_ref::<TensorType>()
        .unwrap()
        .with_encoding(
            ctx,
            Box::new(TypeExtensionsAttr {
                bounds: bounds.to_vec(),
            }),
        )
}
fn bounded(ctx: &mut Context, shape: &[i64], bounds: &[i64], element: ElementType) -> Value {
    let ty = bounded_ty(ctx, shape, bounds, element);
    arg(ctx, ty)
}
fn quantized_element(ctx: &Context, axis: Option<usize>) -> TypeHandle {
    QuantizedElementType::intern(
        ctx,
        UniformQuantizedSpec {
            storage_type: ElementType::I8,
            storage_min: Some(-8),
            storage_max: Some(7),
            expressed_type: ElementType::F32,
            quantization_dimension: axis,
            scales: if axis.is_some() {
                vec![0.25, 0.5]
            } else {
                vec![0.25]
            },
            zero_points: if axis.is_some() { vec![0, 0] } else { vec![0] },
        },
    )
}
fn quantized(ctx: &mut Context, shape: &[i64], axis: Option<usize>) -> Value {
    let element = quantized_element(ctx, axis);
    let ty: TypeHandle = TensorType::get(ctx, shape.to_vec(), element).into();
    arg(ctx, ty)
}
fn result(ctx: &Context, op: &impl Op) -> TypeHandle {
    op.get_operation().deref(ctx).get_result(0).get_type(ctx)
}
fn shaped(ctx: &Context, ty: TypeHandle) -> Shaped {
    tensor_metadata(ctx, ty)
}
fn dims(ctx: &Context, ty: TypeHandle) -> Option<Vec<i64>> {
    shaped(ctx, ty).dimensions
}
fn ty_of(ctx: &Context, v: Value) -> TypeHandle {
    v.get_type(ctx)
}
fn wide(width: u32) -> ElementType {
    ElementType::integer(width, Signedness::Signless).unwrap()
}
fn complex_f16() -> ElementType {
    ElementType::complex(ElementType::F16).unwrap()
}

const FLOATS: &[ElementType] = &[
    ElementType::F4E2M1Fn,
    ElementType::F6E2M3Fn,
    ElementType::F6E3M2Fn,
    ElementType::F8E3M4,
    ElementType::F8E4M3,
    ElementType::F8E4M3Fn,
    ElementType::F8E4M3FnUz,
    ElementType::F8E4M3B11FnUz,
    ElementType::F8E5M2,
    ElementType::F8E5M2FnUz,
    ElementType::F8E8M0FnUz,
    ElementType::F16,
    ElementType::Bf16,
    ElementType::F32,
    ElementType::F64,
];
const COMPLEX: &[ElementType] = &[ElementType::C64, ElementType::C128];
const INTS: &[ElementType] = &[
    ElementType::I2,
    ElementType::I4,
    ElementType::I8,
    ElementType::I16,
    ElementType::I32,
    ElementType::I64,
    ElementType::Ui2,
    ElementType::Ui4,
    ElementType::Ui8,
    ElementType::Ui16,
    ElementType::Ui32,
    ElementType::Ui64,
];
fn non_hlo() -> Vec<ElementType> {
    vec![
        ElementType::Tf32,
        ElementType::Si8,
        ElementType::Si32,
        ElementType::Ui1,
        ElementType::Si1,
        ElementType::Index,
        wide(128),
        wide(3),
        complex_f16(),
    ]
}

#[derive(Clone, Copy, PartialEq)]
enum Family {
    Complex,
    FloatOrComplex,
    Float,
}
fn accepts(family: Family, e: ElementType) -> bool {
    match family {
        Family::Complex => COMPLEX.contains(&e),
        Family::FloatOrComplex => FLOATS.contains(&e) || COMPLEX.contains(&e),
        Family::Float => FLOATS.contains(&e),
    }
}
fn all_elements() -> Vec<ElementType> {
    let mut all: Vec<ElementType> = FLOATS.iter().chain(COMPLEX).chain(INTS).copied().collect();
    all.push(ElementType::I1);
    all.extend(non_hlo());
    all
}

macro_rules! unary_suite {
    ($op:ident, $family:expr, $pred:expr) => {{
        let mut ctx = Context::new();
        let family: Family = $family;
        for e in all_elements() {
            let v = val(&mut ctx, &[2, 3], e);
            let inferred = $op::new_inferred(&mut ctx, v);
            assert_eq!(
                inferred.is_ok(),
                accepts(family, e),
                "{} {:?}",
                stringify!($op),
                e
            );
            if let Ok(op) = inferred {
                op.verify(&ctx).unwrap();
                let out = result(&ctx, &op);
                let want = if $pred {
                    tensor(&ctx, &[2, 3], ElementType::I1)
                } else {
                    ty_of(&ctx, v)
                };
                assert!(out == want, "{} preserves type", stringify!($op));
            }
        }
        // Quantized tensors are outside every unary floating/complex domain.
        let q = quantized(&mut ctx, &[2], None);
        assert!($op::new_inferred(&mut ctx, q).is_err());
        // Unranked operands infer unranked results.
        let ok = if family == Family::Complex {
            ElementType::C64
        } else {
            ElementType::F32
        };
        let u = unranked(&mut ctx, ok);
        let op = $op::new_inferred(&mut ctx, u).unwrap();
        op.verify(&ctx).unwrap();
        assert_eq!(dims(&ctx, result(&ctx, &op)), None);
        // Bounded encodings propagate.
        let b = bounded(&mut ctx, &[-1, 3], &[5, -1], ok);
        let op = $op::new_inferred(&mut ctx, b).unwrap();
        op.verify(&ctx).unwrap();
        assert_eq!(shaped(&ctx, result(&ctx, &op)).bounds, Some(vec![5, -1]));
        // Results: dynamic and unranked are compatible; shape/rank mismatches are not.
        let res_el = if $pred { ElementType::I1 } else { ok };
        let x = val(&mut ctx, &[2, 3], ok);
        let dynamic = tensor(&ctx, &[-1, 3], res_el);
        $op::new(&mut ctx, x, dynamic).verify(&ctx).unwrap();
        let unranked_result = unranked_ty(&ctx, res_el);
        $op::new(&mut ctx, x, unranked_result).verify(&ctx).unwrap();
        let wrong_shape = tensor(&ctx, &[3, 2], res_el);
        assert!($op::new(&mut ctx, x, wrong_shape).verify(&ctx).is_err());
        let wrong_rank = tensor(&ctx, &[6], res_el);
        assert!($op::new(&mut ctx, x, wrong_rank).verify(&ctx).is_err());
        let wrong_element = tensor(&ctx, &[2, 3], ElementType::I32);
        assert!($op::new(&mut ctx, x, wrong_element).verify(&ctx).is_err());
        let small_bound = bounded_ty(&ctx, &[-1, 3], &[1, -1], res_el);
        assert!($op::new(&mut ctx, x, small_bound).verify(&ctx).is_err());
    }};
}

#[test]
fn unary_complex_family() {
    unary_suite!(AsinAcosKernelOp, Family::Complex, false);
}
#[test]
fn unary_float_or_complex_family() {
    unary_suite!(AcosOp, Family::FloatOrComplex, false);
    unary_suite!(AcoshOp, Family::FloatOrComplex, false);
    unary_suite!(AsinOp, Family::FloatOrComplex, false);
    unary_suite!(AsinhOp, Family::FloatOrComplex, false);
    unary_suite!(AtanOp, Family::FloatOrComplex, false);
    unary_suite!(AtanhOp, Family::FloatOrComplex, false);
    unary_suite!(BesselI1eOp, Family::FloatOrComplex, false);
    unary_suite!(ConjOp, Family::FloatOrComplex, false);
    unary_suite!(CoshOp, Family::FloatOrComplex, false);
    unary_suite!(SinhOp, Family::FloatOrComplex, false);
    unary_suite!(TanOp, Family::FloatOrComplex, false);
    unary_suite!(SquareOp, Family::FloatOrComplex, false);
}
#[test]
fn unary_float_family() {
    unary_suite!(DigammaOp, Family::Float, false);
    unary_suite!(ErfOp, Family::Float, false);
    unary_suite!(ErfInvOp, Family::Float, false);
    unary_suite!(ErfcOp, Family::Float, false);
    unary_suite!(LgammaOp, Family::Float, false);
}
#[test]
fn unary_is_inf_family_infers_i1() {
    unary_suite!(IsInfOp, Family::Float, true);
    unary_suite!(IsNegInfOp, Family::Float, true);
    unary_suite!(IsPosInfOp, Family::Float, true);
}

macro_rules! float_binary_suite {
    ($op:ident) => {{
        let mut ctx = Context::new();
        for e in all_elements() {
            let a = val(&mut ctx, &[2], e);
            assert_eq!(
                $op::new_inferred(&mut ctx, a, a).is_ok(),
                FLOATS.contains(&e),
                "{:?}",
                e
            );
        }
        let a = val(&mut ctx, &[-1, 3], ElementType::F32);
        let b = val(&mut ctx, &[2, -1], ElementType::F32);
        let op = $op::new_inferred(&mut ctx, a, b).unwrap();
        op.verify(&ctx).unwrap();
        assert_eq!(dims(&ctx, result(&ctx, &op)), Some(vec![2, 3]));
        let u = unranked(&mut ctx, ElementType::F32);
        let op = $op::new_inferred(&mut ctx, u, b).unwrap();
        op.verify(&ctx).unwrap();
        assert_eq!(dims(&ctx, result(&ctx, &op)), Some(vec![2, -1]));
        let c = val(&mut ctx, &[3], ElementType::F32);
        let d = val(&mut ctx, &[4], ElementType::F32);
        assert!($op::new_inferred(&mut ctx, c, d).is_err());
        let f64v = val(&mut ctx, &[3], ElementType::F64);
        assert!($op::new_inferred(&mut ctx, c, f64v).is_err());
        let bad_result = tensor(&ctx, &[4], ElementType::F32);
        assert!($op::new(&mut ctx, c, c, bad_result).verify(&ctx).is_err());
    }};
}
#[test]
fn compatible_binary_ops() {
    float_binary_suite!(NextAfterOp);
    float_binary_suite!(PolygammaOp);
    float_binary_suite!(ZetaOp);
}
#[test]
fn mulhi_requires_ranked_hlo_integers() {
    let mut ctx = Context::new();
    for e in all_elements() {
        let a = val(&mut ctx, &[2], e);
        assert_eq!(
            MulhiOp::new_inferred(&mut ctx, a, a).is_ok(),
            INTS.contains(&e),
            "{:?}",
            e
        );
    }
    let u = unranked(&mut ctx, ElementType::I32);
    let r = val(&mut ctx, &[2], ElementType::I32);
    assert!(MulhiOp::new_inferred(&mut ctx, u, r).is_err());
    let res = unranked_ty(&ctx, ElementType::I32);
    assert!(MulhiOp::new(&mut ctx, r, r, res).verify(&ctx).is_err());
    let op = MulhiOp::new_inferred(&mut ctx, r, r).unwrap();
    op.verify(&ctx).unwrap();
}

macro_rules! broadcast_suite {
    ($op:ident, $element:expr, $bad:expr) => {{
        let mut ctx = Context::new();
        let element: ElementType = $element;
        let lhs = val(&mut ctx, &[2, 1], element);
        let rhs = val(&mut ctx, &[3], element);
        let op = $op::new_inferred(&mut ctx, lhs, rhs, None).unwrap();
        op.verify(&ctx).unwrap();
        assert_eq!(dims(&ctx, result(&ctx, &op)), Some(vec![2, 3]));
        // size-0 and size-1 broadcasting.
        let z = val(&mut ctx, &[0, 1], element);
        let one = val(&mut ctx, &[1, 4], element);
        let op = $op::new_inferred(&mut ctx, z, one, None).unwrap();
        assert_eq!(dims(&ctx, result(&ctx, &op)), Some(vec![0, 4]));
        // Dynamic dimensions.
        let dy = val(&mut ctx, &[-1, 3], element);
        let op = $op::new_inferred(&mut ctx, dy, rhs, None).unwrap();
        op.verify(&ctx).unwrap();
        assert_eq!(dims(&ctx, result(&ctx, &op)), Some(vec![-1, 3]));
        // Bounded dimensions keep their bounds.
        let b = bounded(&mut ctx, &[-1], &[4], element);
        let s = val(&mut ctx, &[1], element);
        let op = $op::new_inferred(&mut ctx, b, s, None).unwrap();
        op.verify(&ctx).unwrap();
        let out = shaped(&ctx, result(&ctx, &op));
        assert_eq!(out.dimensions, Some(vec![-1]));
        assert_eq!(out.bounds, Some(vec![4]));
        // Explicit broadcast dimensions.
        let big = val(&mut ctx, &[2, 3, 4], element);
        let small = val(&mut ctx, &[3, 4], element);
        let op = $op::new_inferred(&mut ctx, big, small, Some(vec![1, 2])).unwrap();
        op.verify(&ctx).unwrap();
        assert_eq!(op.broadcast_dimensions(&ctx), Some(vec![1, 2]));
        assert_eq!(dims(&ctx, result(&ctx, &op)), Some(vec![2, 3, 4]));
        // Illegal broadcasts fall back to unranked results (upstream).
        for bad_dims in [vec![1, 3], vec![-1, 2], vec![2]] {
            let op = $op::new_inferred(&mut ctx, big, small, Some(bad_dims)).unwrap();
            op.verify(&ctx).unwrap();
            assert_eq!(dims(&ctx, result(&ctx, &op)), None);
        }
        let mismatch = val(&mut ctx, &[4], element);
        let op = $op::new_inferred(&mut ctx, rhs, mismatch, None).unwrap();
        assert_eq!(dims(&ctx, result(&ctx, &op)), None);
        // Unranked operands produce unranked results.
        let u = unranked(&mut ctx, element);
        let op = $op::new_inferred(&mut ctx, u, rhs, None).unwrap();
        op.verify(&ctx).unwrap();
        assert_eq!(dims(&ctx, result(&ctx, &op)), None);
        // Explicit result types: dynamic/unranked compatible, wrong shapes rejected.
        let dyn_res = tensor(&ctx, &[-1, 3], element);
        $op::new(&mut ctx, lhs, rhs, dyn_res, None)
            .verify(&ctx)
            .unwrap();
        let un_res = unranked_ty(&ctx, element);
        $op::new(&mut ctx, lhs, rhs, un_res, None)
            .verify(&ctx)
            .unwrap();
        let bad_res = tensor(&ctx, &[3, 3], element);
        assert!(
            $op::new(&mut ctx, lhs, rhs, bad_res, None)
                .verify(&ctx)
                .is_err()
        );
        // Domain rejections.
        for e in $bad {
            let a = val(&mut ctx, &[2], e);
            assert!($op::new_inferred(&mut ctx, a, a, None).is_err(), "{:?}", e);
        }
    }};
}
fn arithmetic_bad() -> Vec<ElementType> {
    non_hlo()
}
fn logical_bad() -> Vec<ElementType> {
    let mut v = non_hlo();
    v.extend([ElementType::F32, ElementType::C64]);
    v
}
fn zeta_bad() -> Vec<ElementType> {
    let mut v = non_hlo();
    v.extend([ElementType::I32, ElementType::C64, ElementType::I1]);
    v
}

#[test]
fn broadcast_inventory_arithmetic() {
    broadcast_suite!(BroadcastAddOp, ElementType::F32, arithmetic_bad());
    broadcast_suite!(BroadcastAtan2Op, ElementType::F32, arithmetic_bad());
    broadcast_suite!(BroadcastDivideOp, ElementType::I32, arithmetic_bad());
    broadcast_suite!(BroadcastMaximumOp, ElementType::Ui8, arithmetic_bad());
    broadcast_suite!(BroadcastMinimumOp, ElementType::Bf16, arithmetic_bad());
    broadcast_suite!(BroadcastMultiplyOp, ElementType::C64, arithmetic_bad());
    broadcast_suite!(BroadcastNextAfterOp, ElementType::F64, arithmetic_bad());
    broadcast_suite!(BroadcastPolygammaOp, ElementType::F32, arithmetic_bad());
    broadcast_suite!(BroadcastPowerOp, ElementType::C128, arithmetic_bad());
    broadcast_suite!(BroadcastRemainderOp, ElementType::I64, arithmetic_bad());
    broadcast_suite!(BroadcastShiftLeftOp, ElementType::I8, arithmetic_bad());
    broadcast_suite!(
        BroadcastShiftRightArithmeticOp,
        ElementType::I16,
        arithmetic_bad()
    );
    broadcast_suite!(
        BroadcastShiftRightLogicalOp,
        ElementType::Ui32,
        arithmetic_bad()
    );
    broadcast_suite!(BroadcastSubtractOp, ElementType::F16, arithmetic_bad());
    broadcast_suite!(BroadcastZetaOp, ElementType::F32, zeta_bad());
}
#[test]
fn broadcast_inventory_logical() {
    broadcast_suite!(BroadcastAndOp, ElementType::I1, logical_bad());
    broadcast_suite!(BroadcastOrOp, ElementType::I32, logical_bad());
    broadcast_suite!(BroadcastXorOp, ElementType::Ui4, logical_bad());
}

#[test]
fn broadcast_element_traits() {
    let mut ctx = Context::new();
    let f32v = val(&mut ctx, &[2], ElementType::F32);
    let f64v = val(&mut ctx, &[2], ElementType::F64);
    let i32v = val(&mut ctx, &[2], ElementType::I32);
    let i1v = val(&mut ctx, &[2], ElementType::I1);
    let f64_res = tensor(&ctx, &[2], ElementType::F64);
    // Mismatched operand elements are rejected by inference.
    assert!(BroadcastAddOp::new_inferred(&mut ctx, f32v, f64v, None).is_err());
    assert!(BroadcastAtan2Op::new_inferred(&mut ctx, f32v, f64v, None).is_err());
    assert!(BroadcastAndOp::new_inferred(&mut ctx, i32v, i1v, None).is_err());
    // Element traits on explicit results.
    assert!(
        BroadcastAtan2Op::new(&mut ctx, f32v, f32v, f64_res, None)
            .verify(&ctx)
            .is_err()
    );
    assert!(
        BroadcastAddOp::new(&mut ctx, f32v, f32v, f64_res, None)
            .verify(&ctx)
            .is_err()
    );
    let i1_res = tensor(&ctx, &[2], ElementType::I1);
    assert!(
        BroadcastAndOp::new(&mut ctx, i32v, i32v, i1_res, None)
            .verify(&ctx)
            .is_err()
    );
    // Quantized tensors are HLO_AnyTensor.
    let q = quantized(&mut ctx, &[2, 2], None);
    let qa = quantized(&mut ctx, &[2, 2], Some(1));
    BroadcastAddOp::new_inferred(&mut ctx, q, q, None)
        .unwrap()
        .verify(&ctx)
        .unwrap();
    BroadcastMultiplyOp::new_inferred(&mut ctx, q, q, None)
        .unwrap()
        .verify(&ctx)
        .unwrap();
    BroadcastMaximumOp::new_inferred(&mut ctx, qa, qa, None)
        .unwrap()
        .verify(&ctx)
        .unwrap();
    assert!(BroadcastAddOp::new_inferred(&mut ctx, q, qa, None).is_err());
    assert!(BroadcastAndOp::new_inferred(&mut ctx, q, q, None).is_err());
    // A broadcast_dimensions length mismatch is an unranked fallback, not an error.
    let op = BroadcastAddOp::new_inferred(&mut ctx, f32v, f32v, Some(vec![0, 1])).unwrap();
    op.verify(&ctx).unwrap();
    assert_eq!(dims(&ctx, result(&ctx, &op)), None);
}

#[test]
fn broadcast_complex_semantics() {
    let mut ctx = Context::new();
    let f32v = val(&mut ctx, &[2, 1], ElementType::F32);
    let f32w = val(&mut ctx, &[3], ElementType::F32);
    let op = BroadcastComplexOp::new_inferred(&mut ctx, f32v, f32w, None).unwrap();
    op.verify(&ctx).unwrap();
    assert!(result(&ctx, &op) == tensor(&ctx, &[2, 3], ElementType::C64));
    let f64v = val(&mut ctx, &[2], ElementType::F64);
    let op = BroadcastComplexOp::new_inferred(&mut ctx, f64v, f64v, None).unwrap();
    assert!(result(&ctx, &op) == tensor(&ctx, &[2], ElementType::C128));
    for e in [
        ElementType::F16,
        ElementType::Bf16,
        ElementType::I32,
        ElementType::C64,
        ElementType::Tf32,
    ] {
        let v = val(&mut ctx, &[2], e);
        assert!(
            BroadcastComplexOp::new_inferred(&mut ctx, v, v, None).is_err(),
            "{:?}",
            e
        );
    }
    assert!(BroadcastComplexOp::new_inferred(&mut ctx, f32w, f64v, None).is_err());
    let u = unranked(&mut ctx, ElementType::F32);
    let op = BroadcastComplexOp::new_inferred(&mut ctx, u, f32w, None).unwrap();
    op.verify(&ctx).unwrap();
    assert_eq!(dims(&ctx, result(&ctx, &op)), None);
    let wrong = tensor(&ctx, &[2, 3], ElementType::C128);
    assert!(
        BroadcastComplexOp::new(&mut ctx, f32v, f32w, wrong, None)
            .verify(&ctx)
            .is_err()
    );
    let real = tensor(&ctx, &[2, 3], ElementType::F32);
    assert!(
        BroadcastComplexOp::new(&mut ctx, f32v, f32w, real, None)
            .verify(&ctx)
            .is_err()
    );
}

#[test]
fn broadcast_compare_semantics() {
    let mut ctx = Context::new();
    let a = val(&mut ctx, &[2, 1], ElementType::F32);
    let b = val(&mut ctx, &[3], ElementType::F32);
    let op = BroadcastCompareOp::new_inferred(
        &mut ctx,
        a,
        b,
        BroadcastCompareOptions {
            broadcast_dimensions: None,
            direction: ComparisonDirection::Lt,
            comparison_kind: None,
        },
    )
    .unwrap();
    op.verify(&ctx).unwrap();
    assert!(result(&ctx, &op) == tensor(&ctx, &[2, 3], ElementType::I1));
    assert_eq!(op.comparison_direction(&ctx), Some(ComparisonDirection::Lt));
    assert_eq!(op.comparison_kind(&ctx), None);
    // Any HLO tensor, including complex, predicates and quantized.
    for e in [
        ElementType::C128,
        ElementType::I1,
        ElementType::Ui64,
        ElementType::F8E5M2,
    ] {
        let v = val(&mut ctx, &[2], e);
        BroadcastCompareOp::new_inferred(
            &mut ctx,
            v,
            v,
            BroadcastCompareOptions {
                broadcast_dimensions: None,
                direction: ComparisonDirection::Eq,
                comparison_kind: None,
            },
        )
        .unwrap()
        .verify(&ctx)
        .unwrap();
    }
    let q = quantized(&mut ctx, &[2], None);
    BroadcastCompareOp::new_inferred(
        &mut ctx,
        q,
        q,
        BroadcastCompareOptions {
            broadcast_dimensions: None,
            direction: ComparisonDirection::Ne,
            comparison_kind: None,
        },
    )
    .unwrap()
    .verify(&ctx)
    .unwrap();
    for e in non_hlo() {
        let v = val(&mut ctx, &[2], e);
        assert!(
            BroadcastCompareOp::new_inferred(
                &mut ctx,
                v,
                v,
                BroadcastCompareOptions {
                    broadcast_dimensions: None,
                    direction: ComparisonDirection::Eq,
                    comparison_kind: None
                }
            )
            .is_err(),
            "{:?}",
            e
        );
    }
    let i = val(&mut ctx, &[3], ElementType::I32);
    assert!(
        BroadcastCompareOp::new_inferred(
            &mut ctx,
            b,
            i,
            BroadcastCompareOptions {
                broadcast_dimensions: None,
                direction: ComparisonDirection::Eq,
                comparison_kind: None
            }
        )
        .is_err()
    );
    let u = unranked(&mut ctx, ElementType::F32);
    let op = BroadcastCompareOp::new_inferred(
        &mut ctx,
        u,
        b,
        BroadcastCompareOptions {
            broadcast_dimensions: None,
            direction: ComparisonDirection::Ge,
            comparison_kind: None,
        },
    )
    .unwrap();
    op.verify(&ctx).unwrap();
    assert_eq!(dims(&ctx, result(&ctx, &op)), None);
    let bd = bounded(&mut ctx, &[-1], &[4], ElementType::F32);
    let one = val(&mut ctx, &[1], ElementType::F32);
    let op = BroadcastCompareOp::new_inferred(
        &mut ctx,
        bd,
        one,
        BroadcastCompareOptions {
            broadcast_dimensions: None,
            direction: ComparisonDirection::Gt,
            comparison_kind: None,
        },
    )
    .unwrap();
    op.verify(&ctx).unwrap();
    assert_eq!(shaped(&ctx, result(&ctx, &op)).bounds, Some(vec![4]));
    let big = val(&mut ctx, &[2, 3], ElementType::F32);
    let op = BroadcastCompareOp::new_inferred(
        &mut ctx,
        big,
        b,
        BroadcastCompareOptions {
            broadcast_dimensions: Some(vec![1]),
            direction: ComparisonDirection::Le,
            comparison_kind: None,
        },
    )
    .unwrap();
    op.verify(&ctx).unwrap();
    assert_eq!(op.broadcast_dimensions(&ctx), Some(vec![1]));
    let f32_res = tensor(&ctx, &[2, 3], ElementType::F32);
    assert!(
        BroadcastCompareOp::new(
            &mut ctx,
            a,
            b,
            f32_res,
            BroadcastCompareOptions {
                broadcast_dimensions: None,
                direction: ComparisonDirection::Eq,
                comparison_kind: None
            }
        )
        .verify(&ctx)
        .is_err()
    );
    let bad_shape = tensor(&ctx, &[3, 3], ElementType::I1);
    assert!(
        BroadcastCompareOp::new(
            &mut ctx,
            a,
            b,
            bad_shape,
            BroadcastCompareOptions {
                broadcast_dimensions: None,
                direction: ComparisonDirection::Eq,
                comparison_kind: None
            }
        )
        .verify(&ctx)
        .is_err()
    );
}

#[test]
fn broadcast_select_semantics() {
    let mut ctx = Context::new();
    let p = val(&mut ctx, &[2, 1], ElementType::I1);
    let t = val(&mut ctx, &[3], ElementType::F32);
    let f = val(&mut ctx, &[1, 3], ElementType::F32);
    let op = BroadcastSelectOp::new_inferred(&mut ctx, p, t, f).unwrap();
    op.verify(&ctx).unwrap();
    assert!(result(&ctx, &op) == tensor(&ctx, &[2, 3], ElementType::F32));
    let ps = val(&mut ctx, &[], ElementType::I1);
    let op = BroadcastSelectOp::new_inferred(&mut ctx, ps, t, t).unwrap();
    assert_eq!(dims(&ctx, result(&ctx, &op)), Some(vec![3]));
    let q = quantized(&mut ctx, &[3], None);
    BroadcastSelectOp::new_inferred(&mut ctx, ps, q, q)
        .unwrap()
        .verify(&ctx)
        .unwrap();
    let c = val(&mut ctx, &[3], ElementType::C64);
    BroadcastSelectOp::new_inferred(&mut ctx, p, c, c)
        .unwrap()
        .verify(&ctx)
        .unwrap();
    let ip = val(&mut ctx, &[3], ElementType::I32);
    assert!(BroadcastSelectOp::new_inferred(&mut ctx, ip, t, t).is_err());
    let up = unranked(&mut ctx, ElementType::I1);
    assert!(BroadcastSelectOp::new_inferred(&mut ctx, up, t, t).is_err());
    let f64v = val(&mut ctx, &[3], ElementType::F64);
    assert!(BroadcastSelectOp::new_inferred(&mut ctx, p, t, f64v).is_err());
    let w = val(&mut ctx, &[4], ElementType::F32);
    // Upstream getBroadcastType falls back to an unranked result on illegal broadcasts.
    let op = BroadcastSelectOp::new_inferred(&mut ctx, p, t, w).unwrap();
    op.verify(&ctx).unwrap();
    assert_eq!(dims(&ctx, result(&ctx, &op)), None);
    let dy = val(&mut ctx, &[-1, 3], ElementType::F32);
    let op = BroadcastSelectOp::new_inferred(&mut ctx, p, dy, t).unwrap();
    op.verify(&ctx).unwrap();
    assert_eq!(dims(&ctx, result(&ctx, &op)), Some(vec![2, 3]));
    let bad = tensor(&ctx, &[3, 3], ElementType::F32);
    assert!(
        BroadcastSelectOp::new(&mut ctx, p, t, f, bad)
            .verify(&ctx)
            .is_err()
    );
    let dyn_res = tensor(&ctx, &[-1, 3], ElementType::F32);
    BroadcastSelectOp::new(&mut ctx, p, t, f, dyn_res)
        .verify(&ctx)
        .unwrap();
    for e in non_hlo() {
        let v = val(&mut ctx, &[3], e);
        assert!(
            BroadcastSelectOp::new_inferred(&mut ctx, p, v, v).is_err(),
            "{:?}",
            e
        );
    }
}

macro_rules! structural_guards {
    ($ctx:ident, $op:ident, $operands:expr, $results:expr) => {{
        let operands: Vec<Value> = $operands;
        let results: Vec<TypeHandle> = $results;
        let with_region = Operation::new(
            &mut $ctx,
            $op::get_concrete_op_info(),
            results.clone(),
            operands.clone(),
            vec![],
            1,
        );
        let e = Operation::get_op::<$op>(with_region, &$ctx)
            .unwrap()
            .verify(&$ctx);
        assert!(e.is_err(), "{} region", stringify!($op));
        let successor = BasicBlock::new(&mut $ctx, None, vec![]);
        let with_successor = Operation::new(
            &mut $ctx,
            $op::get_concrete_op_info(),
            results.clone(),
            operands.clone(),
            vec![successor],
            0,
        );
        let e = Operation::get_op::<$op>(with_successor, &$ctx)
            .unwrap()
            .verify(&$ctx);
        assert!(e.is_err(), "{} successor", stringify!($op));
        let mut fewer = operands.clone();
        fewer.pop();
        let short = Operation::new(
            &mut $ctx,
            $op::get_concrete_op_info(),
            results.clone(),
            fewer,
            vec![],
            0,
        );
        let e = Operation::get_op::<$op>(short, &$ctx)
            .unwrap()
            .verify(&$ctx);
        assert!(e.is_err(), "{} operand arity", stringify!($op));
        let none = Operation::new(
            &mut $ctx,
            $op::get_concrete_op_info(),
            vec![],
            operands.clone(),
            vec![],
            0,
        );
        let e = Operation::get_op::<$op>(none, &$ctx).unwrap().verify(&$ctx);
        assert!(e.is_err(), "{} result arity", stringify!($op));
    }};
}

#[test]
fn pure_ops_reject_regions_successors_and_bad_arity() {
    let mut ctx = Context::new();
    let f = val(&mut ctx, &[2], ElementType::F32);
    let i = val(&mut ctx, &[2], ElementType::I32);
    let p = val(&mut ctx, &[2], ElementType::I1);
    let ft = tensor(&ctx, &[2], ElementType::F32);
    let it = tensor(&ctx, &[2], ElementType::I32);
    let pt = tensor(&ctx, &[2], ElementType::I1);
    let ct = tensor(&ctx, &[2], ElementType::C64);
    structural_guards!(ctx, ErfOp, vec![f], vec![ft]);
    structural_guards!(ctx, IsInfOp, vec![f], vec![pt]);
    structural_guards!(ctx, AcosOp, vec![f], vec![ft]);
    structural_guards!(ctx, AsinAcosKernelOp, vec![f], vec![ct]);
    structural_guards!(ctx, NextAfterOp, vec![f, f], vec![ft]);
    structural_guards!(ctx, MulhiOp, vec![i, i], vec![it]);
    structural_guards!(ctx, BroadcastAddOp, vec![f, f], vec![ft]);
    structural_guards!(ctx, BroadcastAndOp, vec![p, p], vec![pt]);
    structural_guards!(ctx, BroadcastZetaOp, vec![f, f], vec![ft]);
    structural_guards!(ctx, BroadcastComplexOp, vec![f, f], vec![ct]);
    structural_guards!(ctx, BroadcastSelectOp, vec![p, f, f], vec![ft]);
    structural_guards!(ctx, TopKOp, vec![f], vec![ft, it]);
    let compare = Operation::new(
        &mut ctx,
        BroadcastCompareOp::get_concrete_op_info(),
        vec![pt],
        vec![f, f],
        vec![],
        1,
    );
    assert!(
        Operation::get_op::<BroadcastCompareOp>(compare, &ctx)
            .unwrap()
            .verify(&ctx)
            .is_err()
    );
    // A missing comparison_direction is rejected even with a valid shape.
    let compare = Operation::new(
        &mut ctx,
        BroadcastCompareOp::get_concrete_op_info(),
        vec![pt],
        vec![f, f],
        vec![],
        0,
    );
    assert!(
        Operation::get_op::<BroadcastCompareOp>(compare, &ctx)
            .unwrap()
            .verify(&ctx)
            .is_err()
    );
    let ragged = Operation::new(
        &mut ctx,
        RaggedDotOp::get_concrete_op_info(),
        vec![ft],
        vec![f, f, i],
        vec![],
        0,
    );
    assert!(
        Operation::get_op::<RaggedDotOp>(ragged, &ctx)
            .unwrap()
            .verify(&ctx)
            .is_err()
    );
}

fn top_k_parts(ctx: &Context, op: &TopKOp) -> (Shaped, Shaped) {
    let v = op.values(ctx).get_type(ctx);
    let i = op.indices(ctx).get_type(ctx);
    (shaped(ctx, v), shaped(ctx, i))
}

#[test]
fn top_k_semantics() {
    let mut ctx = Context::new();
    let x = val(&mut ctx, &[3, 8], ElementType::F32);
    let op = TopKOp::new_inferred(&mut ctx, x, TopKOptions::new(2, true)).unwrap();
    op.verify(&ctx).unwrap();
    assert!(op.values(&ctx).get_type(&ctx) == tensor(&ctx, &[3, 2], ElementType::F32));
    assert!(op.indices(&ctx).get_type(&ctx) == tensor(&ctx, &[3, 2], ElementType::I32));
    assert_eq!(op.k(&ctx), Some(2));
    assert!(op.is_stable(&ctx));
    let op = TopKOp::new_inferred(&mut ctx, x, TopKOptions::new(8, false)).unwrap();
    op.verify(&ctx).unwrap();
    assert!(!op.is_stable(&ctx));
    // HLO_AnyTensor: integers, complex, predicates and quantized are allowed.
    for e in [
        ElementType::I8,
        ElementType::Ui16,
        ElementType::C64,
        ElementType::I1,
        ElementType::F8E4M3Fn,
    ] {
        let v = val(&mut ctx, &[4], e);
        TopKOp::new_inferred(&mut ctx, v, TopKOptions::new(1, true))
            .unwrap()
            .verify(&ctx)
            .unwrap();
    }
    let q = quantized(&mut ctx, &[4], None);
    TopKOp::new_inferred(&mut ctx, q, TopKOptions::new(2, true))
        .unwrap()
        .verify(&ctx)
        .unwrap();
    for e in non_hlo() {
        let v = val(&mut ctx, &[4], e);
        assert!(
            TopKOp::new_inferred(&mut ctx, v, TopKOptions::new(1, true)).is_err(),
            "{:?}",
            e
        );
    }
    // Rank 0 and k larger than a static last dimension are errors.
    let s = val(&mut ctx, &[], ElementType::F32);
    assert!(TopKOp::new_inferred(&mut ctx, s, TopKOptions::new(1, true)).is_err());
    assert!(TopKOp::new_inferred(&mut ctx, x, TopKOptions::new(9, true)).is_err());
    // Dynamic last dimension accepts any k.
    let d = val(&mut ctx, &[3, -1], ElementType::F32);
    let op = TopKOp::new_inferred(&mut ctx, d, TopKOptions::new(100, true)).unwrap();
    op.verify(&ctx).unwrap();
    assert_eq!(top_k_parts(&ctx, &op).0.dimensions, Some(vec![3, 100]));
    // Bounded: the last bound is dropped, earlier bounds kept.
    let b = bounded(&mut ctx, &[-1, -1], &[5, 9], ElementType::F32);
    let op = TopKOp::new_inferred(&mut ctx, b, TopKOptions::new(2, true)).unwrap();
    op.verify(&ctx).unwrap();
    let (values, indices) = top_k_parts(&ctx, &op);
    assert_eq!(values.dimensions, Some(vec![-1, 2]));
    assert_eq!(values.bounds, Some(vec![5, -1]));
    assert_eq!(indices.bounds, Some(vec![5, -1]));
    // Unranked operands infer unranked results.
    let u = unranked(&mut ctx, ElementType::F32);
    let op = TopKOp::new_inferred(&mut ctx, u, TopKOptions::new(2, true)).unwrap();
    op.verify(&ctx).unwrap();
    let (values, indices) = top_k_parts(&ctx, &op);
    assert_eq!(values.dimensions, None);
    assert_eq!(indices.dimensions, None);
    // Explicit results: compatible types only.
    let ft = tensor(&ctx, &[3, 2], ElementType::F32);
    let it = tensor(&ctx, &[3, 2], ElementType::I32);
    let dt = tensor(&ctx, &[-1, 2], ElementType::F32);
    let ut = unranked_ty(&ctx, ElementType::I32);
    TopKOp::new(&mut ctx, x, TopKOptions::new(2, true), dt, ut)
        .verify(&ctx)
        .unwrap();
    let i64t = tensor(&ctx, &[3, 2], ElementType::I64);
    assert!(
        TopKOp::new(&mut ctx, x, TopKOptions::new(2, true), ft, i64t)
            .verify(&ctx)
            .is_err()
    );
    let f64t = tensor(&ctx, &[3, 2], ElementType::F64);
    assert!(
        TopKOp::new(&mut ctx, x, TopKOptions::new(2, true), f64t, it)
            .verify(&ctx)
            .is_err()
    );
    let wrong = tensor(&ctx, &[3, 3], ElementType::F32);
    assert!(
        TopKOp::new(&mut ctx, x, TopKOptions::new(2, true), wrong, it)
            .verify(&ctx)
            .is_err()
    );
}

#[test]
fn top_k_is_stable_defaults_to_true_when_omitted() {
    let mut ctx = Context::new();
    let x = val(&mut ctx, &[4], ElementType::F32);
    let op = TopKOp::new_inferred(&mut ctx, x, TopKOptions::new(2, false)).unwrap();
    op.get_operation()
        .deref_mut(&ctx)
        .attributes
        .0
        .retain(|key, _| !key.to_string().contains("is_stable"));
    assert!(op.is_stable(&ctx));
    op.verify(&ctx).unwrap();
}

#[allow(clippy::too_many_arguments)]
fn ragged(
    lb: &[i64],
    rb: &[i64],
    lc: &[i64],
    rc: &[i64],
    lr: &[i64],
    rg: &[i64],
) -> RaggedDotDimensionNumbersAttr {
    RaggedDotDimensionNumbersAttr::new(RaggedDotDimensions {
        lhs_batching_dimensions: lb.to_vec(),
        rhs_batching_dimensions: rb.to_vec(),
        lhs_contracting_dimensions: lc.to_vec(),
        rhs_contracting_dimensions: rc.to_vec(),
        lhs_ragged_dimensions: lr.to_vec(),
        rhs_group_dimensions: rg.to_vec(),
    })
}

#[test]
fn ragged_dot_modes() {
    let mut ctx = Context::new();
    // Non-contracting ragged dimension: [m,k] x [g,k,n] -> [m,n].
    let lhs = val(&mut ctx, &[11, 5], ElementType::F32);
    let rhs = val(&mut ctx, &[3, 5, 7], ElementType::F32);
    let groups = val(&mut ctx, &[3], ElementType::I64);
    let d = ragged(&[], &[], &[1], &[1], &[0], &[0]);
    let precision = Some(vec![Precision::Default, Precision::Highest]);
    let op = RaggedDotOp::new_inferred(
        &mut ctx,
        lhs,
        rhs,
        groups,
        RaggedDotOptions {
            dimension_numbers: d,
            precision_config: precision.clone(),
        },
    )
    .unwrap();
    op.verify(&ctx).unwrap();
    assert!(result(&ctx, &op) == tensor(&ctx, &[11, 7], ElementType::F32));
    assert_eq!(op.precision_config(&ctx), precision);
    let attr = op.precision_config_attr(&ctx).unwrap();
    assert_eq!(Some(read_precision_config(&attr)), precision);
    assert_eq!(
        op.dimension_numbers(&ctx)
            .unwrap()
            .dimensions()
            .lhs_ragged_dimensions,
        vec![0]
    );
    // Batched non-contracting: group_sizes may carry the batch prefix.
    let lhs_b = val(&mut ctx, &[2, 11, 5], ElementType::F32);
    let rhs_b = val(&mut ctx, &[2, 3, 5, 7], ElementType::F32);
    let groups_b = val(&mut ctx, &[2, 3], ElementType::Ui32);
    let d = ragged(&[0], &[0], &[2], &[2], &[1], &[1]);
    let op = RaggedDotOp::new_inferred(
        &mut ctx,
        lhs_b,
        rhs_b,
        groups_b,
        RaggedDotOptions {
            dimension_numbers: d.clone(),
            precision_config: None,
        },
    )
    .unwrap();
    op.verify(&ctx).unwrap();
    assert_eq!(dims(&ctx, result(&ctx, &op)), Some(vec![2, 11, 7]));
    assert_eq!(op.precision_config(&ctx), None);
    let bad_prefix = val(&mut ctx, &[4, 3], ElementType::I32);
    assert!(
        RaggedDotOp::new_inferred(
            &mut ctx,
            lhs_b,
            rhs_b,
            bad_prefix,
            RaggedDotOptions {
                dimension_numbers: d.clone(),
                precision_config: None
            }
        )
        .is_err()
    );
    let dyn_prefix = val(&mut ctx, &[-1, 3], ElementType::I32);
    assert!(
        RaggedDotOp::new_inferred(
            &mut ctx,
            lhs_b,
            rhs_b,
            dyn_prefix,
            RaggedDotOptions {
                dimension_numbers: d.clone(),
                precision_config: None
            }
        )
        .is_err()
    );
    let bad_rank = val(&mut ctx, &[2, 1, 3], ElementType::I32);
    assert!(
        RaggedDotOp::new_inferred(
            &mut ctx,
            lhs_b,
            rhs_b,
            bad_rank,
            RaggedDotOptions {
                dimension_numbers: d,
                precision_config: None
            }
        )
        .is_err()
    );
    // Contracting ragged dimension: [m,k] x [k,n] -> [g,m,n].
    let rhs_c = val(&mut ctx, &[5, 7], ElementType::F32);
    let d = ragged(&[], &[], &[1], &[0], &[1], &[]);
    let op = RaggedDotOp::new_inferred(
        &mut ctx,
        lhs,
        rhs_c,
        groups,
        RaggedDotOptions {
            dimension_numbers: d,
            precision_config: None,
        },
    )
    .unwrap();
    op.verify(&ctx).unwrap();
    assert_eq!(dims(&ctx, result(&ctx, &op)), Some(vec![3, 11, 7]));
    // Batch ragged dimension: [b,m,k] x [b,k,n] -> [b,m,n].
    let lhs_t = val(&mut ctx, &[4, 11, 5], ElementType::Bf16);
    let rhs_t = val(&mut ctx, &[4, 5, 7], ElementType::Bf16);
    let d = ragged(&[0], &[0], &[2], &[1], &[0], &[]);
    let op = RaggedDotOp::new_inferred(
        &mut ctx,
        lhs_t,
        rhs_t,
        groups,
        RaggedDotOptions {
            dimension_numbers: d,
            precision_config: None,
        },
    )
    .unwrap();
    op.verify(&ctx).unwrap();
    assert!(result(&ctx, &op) == tensor(&ctx, &[4, 11, 7], ElementType::Bf16));
    // Dynamic dimensions flow through.
    let lhs_d = val(&mut ctx, &[-1, 5], ElementType::F32);
    let rhs_d = val(&mut ctx, &[-1, -1, 7], ElementType::F32);
    let d = ragged(&[], &[], &[1], &[1], &[0], &[0]);
    let op = RaggedDotOp::new_inferred(
        &mut ctx,
        lhs_d,
        rhs_d,
        groups,
        RaggedDotOptions {
            dimension_numbers: d,
            precision_config: None,
        },
    )
    .unwrap();
    op.verify(&ctx).unwrap();
    assert_eq!(dims(&ctx, result(&ctx, &op)), Some(vec![-1, 7]));
    // Quantized and integer operands are HLO_AnyTensor.
    let q = quantized(&mut ctx, &[11, 5], None);
    let d = ragged(&[], &[], &[1], &[1], &[0], &[0]);
    RaggedDotOp::new_inferred(
        &mut ctx,
        q,
        rhs,
        groups,
        RaggedDotOptions {
            dimension_numbers: d,
            precision_config: None,
        },
    )
    .unwrap()
    .verify(&ctx)
    .unwrap();
}

#[test]
fn ragged_dot_constraints() {
    let mut ctx = Context::new();
    let lhs = val(&mut ctx, &[11, 5], ElementType::F32);
    let rhs = val(&mut ctx, &[3, 5, 7], ElementType::F32);
    let groups = val(&mut ctx, &[3], ElementType::I64);
    let good = ragged(&[], &[], &[1], &[1], &[0], &[0]);
    let check = |ctx: &mut Context,
                 l: Value,
                 r: Value,
                 g: Value,
                 d: RaggedDotDimensionNumbersAttr,
                 p: Option<Vec<Precision>>| {
        RaggedDotOp::new_inferred(
            ctx,
            l,
            r,
            g,
            RaggedDotOptions {
                dimension_numbers: d,
                precision_config: p,
            },
        )
        .is_ok()
    };
    assert!(check(&mut ctx, lhs, rhs, groups, good.clone(), None));
    assert!(check(
        &mut ctx,
        lhs,
        rhs,
        groups,
        good.clone(),
        Some(vec![])
    ));
    assert!(check(
        &mut ctx,
        lhs,
        rhs,
        groups,
        good.clone(),
        Some(vec![Precision::High])
    ));
    assert!(!check(
        &mut ctx,
        lhs,
        rhs,
        groups,
        good.clone(),
        Some(vec![Precision::High; 3])
    ));
    let cases = [
        ("batch count", ragged(&[0], &[], &[1], &[1], &[0], &[0])),
        ("contracting count", ragged(&[], &[], &[1], &[], &[0], &[0])),
        ("lhs duplicate", ragged(&[1], &[0], &[1], &[1], &[0], &[2])),
        ("lhs out of range", ragged(&[], &[], &[2], &[1], &[0], &[0])),
        ("rhs out of range", ragged(&[], &[], &[1], &[3], &[0], &[0])),
        ("negative", ragged(&[], &[], &[-1], &[1], &[0], &[0])),
        ("two ragged", ragged(&[], &[], &[1], &[1], &[0, 1], &[0])),
        ("no ragged", ragged(&[], &[], &[1], &[1], &[], &[0])),
        (
            "ragged out of range",
            ragged(&[], &[], &[1], &[1], &[2], &[0]),
        ),
        ("missing group dim", ragged(&[], &[], &[1], &[1], &[0], &[])),
        (
            "two group dims",
            ragged(&[], &[], &[1], &[1], &[0], &[0, 2]),
        ),
        (
            "group overlaps contracting",
            ragged(&[], &[], &[1], &[1], &[0], &[1]),
        ),
        (
            "group out of range",
            ragged(&[], &[], &[1], &[1], &[0], &[3]),
        ),
        (
            "group size mismatch",
            ragged(&[], &[], &[1], &[1], &[0], &[2]),
        ),
    ];
    for (name, d) in cases {
        assert!(!check(&mut ctx, lhs, rhs, groups, d, None), "{}", name);
    }
    // Contracting size mismatch.
    let rhs_bad = val(&mut ctx, &[3, 6, 7], ElementType::F32);
    assert!(!check(&mut ctx, lhs, rhs_bad, groups, good.clone(), None));
    // Group dims are forbidden in contracting and batch modes.
    let rhs_c = val(&mut ctx, &[3, 5, 7], ElementType::F32);
    let contracting = ragged(&[], &[], &[1], &[1], &[1], &[0]);
    assert!(!check(&mut ctx, lhs, rhs_c, groups, contracting, None));
    let lhs_t = val(&mut ctx, &[4, 11, 5], ElementType::F32);
    let rhs_t = val(&mut ctx, &[4, 3, 5, 7], ElementType::F32);
    let batch = ragged(&[0], &[0], &[2], &[2], &[0], &[1]);
    assert!(!check(&mut ctx, lhs_t, rhs_t, groups, batch, None));
    // Batch size mismatch.
    let rhs_bm = val(&mut ctx, &[5, 3, 5, 7], ElementType::F32);
    let batched = ragged(&[0], &[0], &[2], &[2], &[1], &[1]);
    assert!(!check(&mut ctx, lhs_t, rhs_bm, groups, batched, None));
    // group_sizes must be a ranked HLO integer tensor.
    for e in [
        ElementType::F32,
        ElementType::I1,
        ElementType::Si32,
        ElementType::Tf32,
    ] {
        let g = val(&mut ctx, &[3], e);
        assert!(!check(&mut ctx, lhs, rhs, g, good.clone(), None), "{:?}", e);
    }
    let gu = unranked(&mut ctx, ElementType::I32);
    assert!(!check(&mut ctx, lhs, rhs, gu, good.clone(), None));
    let gs = val(&mut ctx, &[], ElementType::I32);
    assert!(!check(&mut ctx, lhs, rhs, gs, good.clone(), None));
    // Unranked or non-HLO operands.
    let lu = unranked(&mut ctx, ElementType::F32);
    assert!(!check(&mut ctx, lu, rhs, groups, good.clone(), None));
    let lt = val(&mut ctx, &[11, 5], ElementType::Tf32);
    assert!(!check(&mut ctx, lt, rhs, groups, good.clone(), None));
    // Explicit result shape compatibility.
    let dyn_res = tensor(&ctx, &[-1, 7], ElementType::F32);
    RaggedDotOp::new(
        &mut ctx,
        lhs,
        rhs,
        groups,
        dyn_res,
        RaggedDotOptions {
            dimension_numbers: good.clone(),
            precision_config: None,
        },
    )
    .verify(&ctx)
    .unwrap();
    let bad_res = tensor(&ctx, &[11, 8], ElementType::F32);
    assert!(
        RaggedDotOp::new(
            &mut ctx,
            lhs,
            rhs,
            groups,
            bad_res,
            RaggedDotOptions {
                dimension_numbers: good.clone(),
                precision_config: None
            }
        )
        .verify(&ctx)
        .is_err()
    );
    let bad_p = RaggedDotOp::new(
        &mut ctx,
        lhs,
        rhs,
        groups,
        dyn_res,
        RaggedDotOptions {
            dimension_numbers: good,
            precision_config: Some(vec![Precision::Default; 3]),
        },
    );
    assert!(bad_p.verify(&ctx).is_err());
}

fn int_attr(ctx: &mut Context, width: u32, signedness: Signedness, v: i64) -> IntegerAttr {
    let ty = IntegerType::get(ctx, width, signedness);
    IntegerAttr::new(ty, APInt::from_i64(v, bw(width as usize)))
}

#[test]
fn top_k_verifier_requires_signless_i64_k() {
    let mut ctx = Context::new();
    let x = val(&mut ctx, &[3, 8], ElementType::F32);
    let op = TopKOp::new_inferred(&mut ctx, x, TopKOptions::new(2, true)).unwrap();
    op.verify(&ctx).unwrap();
    for (width, signedness) in [
        (64, Signedness::Signed),
        (64, Signedness::Unsigned),
        (32, Signedness::Signless),
        (8, Signedness::Signless),
    ] {
        let attr = int_attr(&mut ctx, width, signedness, 2);
        op.set_attr_chlo_top_k_k(&ctx, attr);
        assert_eq!(op.k(&ctx), None, "{width} {signedness:?}");
        assert!(op.verify(&ctx).is_err(), "{width} {signedness:?}");
    }
    let attr = int_attr(&mut ctx, 64, Signedness::Signless, 2);
    op.set_attr_chlo_top_k_k(&ctx, attr);
    assert_eq!(op.k(&ctx), Some(2));
    op.verify(&ctx).unwrap();
    // A missing k is rejected by the verifier too.
    op.get_operation()
        .deref_mut(&ctx)
        .attributes
        .0
        .retain(|key, _| !key.to_string().ends_with("top_k_k"));
    assert_eq!(op.k(&ctx), None);
    assert!(op.verify(&ctx).is_err());
}

/// Upstream `inferRaggedDotOp` (ChloOps.cpp) emplaces `ShapedTypeComponents(dims)` built
/// from `inferRaggedDotOutputDimensions` only: no encoding/bounds are propagated, and the
/// verifier checks `verifyCompatibleShape` against the result shape only.
#[test]
fn ragged_dot_inference_drops_operand_encodings() {
    let mut ctx = Context::new();
    let lhs = bounded(&mut ctx, &[-1, 5], &[11, -1], ElementType::F32);
    let rhs = bounded(&mut ctx, &[3, 5, -1], &[-1, -1, 7], ElementType::F32);
    let groups = val(&mut ctx, &[3], ElementType::I64);
    let d = ragged(&[], &[], &[1], &[1], &[0], &[0]);
    let op = RaggedDotOp::new_inferred(
        &mut ctx,
        lhs,
        rhs,
        groups,
        RaggedDotOptions {
            dimension_numbers: d.clone(),
            precision_config: None,
        },
    )
    .unwrap();
    op.verify(&ctx).unwrap();
    let out = shaped(&ctx, result(&ctx, &op));
    assert_eq!(out.dimensions, Some(vec![-1, -1]));
    assert_eq!(out.bounds, None);
    assert!(result(&ctx, &op) == tensor(&ctx, &[-1, -1], ElementType::F32));
    // Shape-only verification: bounded or differently-typed results with compatible
    // shapes are accepted, as upstream.
    let bounded_res = bounded_ty(&ctx, &[-1, 7], &[11, -1], ElementType::F32);
    RaggedDotOp::new(
        &mut ctx,
        lhs,
        rhs,
        groups,
        bounded_res,
        RaggedDotOptions {
            dimension_numbers: d.clone(),
            precision_config: None,
        },
    )
    .verify(&ctx)
    .unwrap();
    let other_element = tensor(&ctx, &[4, 7], ElementType::F64);
    RaggedDotOp::new(
        &mut ctx,
        lhs,
        rhs,
        groups,
        other_element,
        RaggedDotOptions {
            dimension_numbers: d.clone(),
            precision_config: None,
        },
    )
    .verify(&ctx)
    .unwrap();
    let wrong_rank = tensor(&ctx, &[4], ElementType::F32);
    assert!(
        RaggedDotOp::new(
            &mut ctx,
            lhs,
            rhs,
            groups,
            wrong_rank,
            RaggedDotOptions {
                dimension_numbers: d,
                precision_config: None
            }
        )
        .verify(&ctx)
        .is_err()
    );
}

/// `getBroadcastTypeWithBroadcastDimensions` and the non-bounded numpy fallback build
/// `{shape, elementType}` without an encoding; only the all-static-or-bounded numpy path
/// keeps bounds.
#[test]
fn broadcast_encoding_paths_follow_upstream() {
    let mut ctx = Context::new();
    let b = bounded(&mut ctx, &[2, -1], &[-1, 4], ElementType::F32);
    let s = val(&mut ctx, &[4], ElementType::F32);
    let op = BroadcastAddOp::new_inferred(&mut ctx, b, s, Some(vec![1])).unwrap();
    op.verify(&ctx).unwrap();
    let out = shaped(&ctx, result(&ctx, &op));
    assert_eq!(out.dimensions, Some(vec![2, 4]));
    assert_eq!(out.bounds, None);
    // Mixed bounded and unbounded dynamic operands use the lenient MLIR rule.
    let u = val(&mut ctx, &[-1, 1], ElementType::F32);
    let b2 = bounded(&mut ctx, &[-1], &[4], ElementType::F32);
    let op = BroadcastAddOp::new_inferred(&mut ctx, u, b2, None).unwrap();
    op.verify(&ctx).unwrap();
    let out = shaped(&ctx, result(&ctx, &op));
    assert_eq!(out.dimensions, Some(vec![-1, -1]));
    assert_eq!(out.bounds, None);
    let d = val(&mut ctx, &[-1, -1], ElementType::F32);
    let s3 = val(&mut ctx, &[1, 3], ElementType::F32);
    let op = BroadcastAddOp::new_inferred(&mut ctx, d, s3, None).unwrap();
    assert_eq!(dims(&ctx, result(&ctx, &op)), Some(vec![-1, 3]));
    let d1 = val(&mut ctx, &[-1], ElementType::F32);
    let s5 = val(&mut ctx, &[2, 5], ElementType::F32);
    let op = BroadcastAddOp::new_inferred(&mut ctx, d1, s5, None).unwrap();
    assert_eq!(dims(&ctx, result(&ctx, &op)), Some(vec![2, 5]));
}
