use pliron::{
    attribute::AttrObj,
    common_traits::Verify,
    context::Context,
    op::Op,
    r#type::{TypeHandle, Typed},
    value::Value,
};
use stablehlo::{attributes::*, ops::*, types::*};

fn value(ctx: &mut Context, shape: &[i64], element: ElementType, bit: u64) -> Value {
    let ty = tensor(ctx, shape, element);
    constant(ctx, ty, bit)
}
fn constant(ctx: &mut Context, ty: TypeHandle, bit: u64) -> Value {
    let op = ConstantOp::new(ctx, ty, vec![bit], true);
    op.verify(ctx).unwrap();
    op.get_operation().deref(ctx).get_result(0)
}

#[test]
fn constants_and_iotas_verify_native_types() {
    let mut ctx = Context::new();
    let scalar = tensor(&ctx, &[2, 3], ElementType::I32);
    let c = ConstantOp::new(&mut ctx, scalar, vec![7], true);
    assert_eq!(c.bits(&ctx), Some(vec![7]));
    assert!(c.verify(&ctx).is_ok());
    let iota_type = tensor(&ctx, &[4], ElementType::F32);
    assert!(IotaOp::new(&mut ctx, iota_type, 0).verify(&ctx).is_ok());
    let sizes = value(&mut ctx, &[1], ElementType::I32, 4);
    let dynamic_type = tensor(&ctx, &[4], ElementType::I32);
    let dynamic = DynamicIotaOp::new(&mut ctx, sizes, dynamic_type, 0);
    assert!(dynamic.verify(&ctx).is_ok());
}

#[test]
fn every_unary_operation_builds_and_verifies() {
    let mut ctx = Context::new();
    let f = value(&mut ctx, &[2], ElementType::F32, 0);
    let i = value(&mut ctx, &[2], ElementType::I32, 0);
    macro_rules! ok {
        ($op:ident, $value:expr) => {
            assert!(
                $op::new_inferred(&mut ctx, $value)
                    .unwrap()
                    .verify(&ctx)
                    .is_ok(),
                stringify!($op)
            );
        };
    }
    ok!(AbsOp, f);
    ok!(CbrtOp, f);
    ok!(CeilOp, f);
    ok!(CosineOp, f);
    ok!(ExponentialOp, f);
    ok!(ExponentialMinusOneOp, f);
    ok!(FloorOp, f);
    ok!(ImagOp, f);
    ok!(IsFiniteOp, f);
    ok!(LogOp, f);
    ok!(LogPlusOneOp, f);
    ok!(LogisticOp, f);
    ok!(NegateOp, f);
    ok!(RealOp, f);
    ok!(RoundNearestAfzOp, f);
    ok!(RoundNearestEvenOp, f);
    ok!(RsqrtOp, f);
    ok!(SignOp, f);
    ok!(SineOp, f);
    ok!(SqrtOp, f);
    ok!(TanOp, f);
    ok!(TanhOp, f);
    ok!(CountLeadingZerosOp, i);
    ok!(PopulationCountOp, i);
    ok!(NotOp, i);
}

#[test]
fn every_binary_operation_builds_and_verifies() {
    let mut ctx = Context::new();
    let f = value(&mut ctx, &[2], ElementType::F32, 0);
    let i = value(&mut ctx, &[2], ElementType::I32, 0);
    macro_rules! ok {
        ($op:ident, $a:expr, $b:expr) => {
            assert!(
                $op::new_inferred(&mut ctx, $a, $b)
                    .unwrap()
                    .verify(&ctx)
                    .is_ok(),
                stringify!($op)
            );
        };
    }
    ok!(AddOp, f, f);
    ok!(Atan2Op, f, f);
    ok!(DivideOp, f, f);
    ok!(MaximumOp, f, f);
    ok!(MinimumOp, f, f);
    ok!(MultiplyOp, f, f);
    ok!(PowerOp, f, f);
    ok!(RemainderOp, f, f);
    ok!(SubtractOp, f, f);
    ok!(ComplexOp, f, f);
    ok!(ShiftLeftOp, i, i);
    ok!(ShiftRightArithmeticOp, i, i);
    ok!(ShiftRightLogicalOp, i, i);
    ok!(AndOp, i, i);
    ok!(OrOp, i, i);
    ok!(XorOp, i, i);
}

#[test]
fn conversion_comparison_select_and_clamp_verify() {
    let mut ctx = Context::new();
    let f = value(&mut ctx, &[3], ElementType::F32, 0);
    let i = value(&mut ctx, &[3], ElementType::I32, 0);
    assert!(
        ConvertOp::new_inferred(&mut ctx, f, ElementType::I32)
            .unwrap()
            .verify(&ctx)
            .is_ok()
    );
    let cmp = CompareOp::new_inferred(
        &mut ctx,
        i,
        i,
        CompareDirection::Lt,
        Some(CompareType::Signed),
    )
    .unwrap();
    assert!(cmp.verify(&ctx).is_ok());
    let pred = cmp.get_operation().deref(&ctx).get_result(0);
    assert!(
        SelectOp::new_inferred(&mut ctx, pred, f, f)
            .unwrap()
            .verify(&ctx)
            .is_ok()
    );
    assert!(
        ClampOp::new_inferred(&mut ctx, f, f, f)
            .unwrap()
            .verify(&ctx)
            .is_ok()
    );
}

#[test]
fn bitcast_and_reduce_precision_verify() {
    let mut ctx = Context::new();
    let i = value(&mut ctx, &[2], ElementType::I32, 0);
    assert!(
        BitcastConvertOp::new_inferred(&mut ctx, i, ElementType::F32)
            .unwrap()
            .verify(&ctx)
            .is_ok()
    );
    let f = value(&mut ctx, &[2], ElementType::F32, 0);
    assert!(
        ReducePrecisionOp::new_inferred(&mut ctx, f, 8, 23)
            .verify(&ctx)
            .is_ok()
    );
}

#[test]
fn uniform_quantize_and_dequantize_verify() {
    let mut ctx = Context::new();
    let input = value(&mut ctx, &[2], ElementType::F32, 0);
    let qelement = QuantizedElementType::intern(
        &ctx,
        UniformQuantizedSpec {
            storage_type: ElementType::I8,
            storage_min: None,
            storage_max: None,
            expressed_type: ElementType::F32,
            quantization_dimension: None,
            scales: vec![0.5],
            zero_points: vec![0],
        },
    );
    let qtype = TensorType::from_element(&ctx, qelement, vec![2]);
    let quantized = UniformQuantizeOp::new(&mut ctx, input, qtype);
    assert!(quantized.verify(&ctx).is_ok());
    let qvalue = quantized.get_operation().deref(&ctx).get_result(0);
    assert!(
        UniformDequantizeOp::new_inferred(&mut ctx, qvalue)
            .unwrap()
            .verify(&ctx)
            .is_ok()
    );
}

#[test]
fn elementwise_rejects_incompatible_element_domains() {
    let mut ctx = Context::new();
    let f = value(&mut ctx, &[2], ElementType::F32, 0);
    let i = value(&mut ctx, &[2], ElementType::I32, 0);
    assert!(AddOp::new_inferred(&mut ctx, f, i).is_err());
    let result_type = tensor(&ctx, &[2], ElementType::F32);
    let invalid = AddOp::new(&mut ctx, f, i, result_type);
    assert!(invalid.verify(&ctx).is_err());
}

#[test]
fn constant_rejects_mismatched_counts_and_dynamic_shapes() {
    let mut ctx = Context::new();
    let ty = tensor(&ctx, &[2], ElementType::I32);
    assert!(
        ConstantOp::new(&mut ctx, ty, vec![1, 2], false)
            .verify(&ctx)
            .is_ok()
    );
    assert!(
        ConstantOp::new(&mut ctx, ty, vec![1], false)
            .verify(&ctx)
            .is_err()
    );
    let dynamic = tensor(&ctx, &[-1], ElementType::I32);
    assert!(
        ConstantOp::new(&mut ctx, dynamic, vec![1], true)
            .verify(&ctx)
            .is_err()
    );
}

#[test]
fn native_inference_helpers_preserve_shape_and_apply_element_rules() {
    let ctx = Context::new();
    let complex = TensorType::from_element(&ctx, ElementType::C64.native(&ctx), vec![-1, 2]);
    let real = stablehlo::inference::infer_abs_type(&ctx, complex).unwrap();
    let real_ty = real.deref(&ctx);
    let real_tensor = real_ty.downcast_ref::<TensorType>().unwrap();
    assert_eq!(real_tensor.dimensions, vec![-1, 2]);
    assert_eq!(
        primitive_element(&ctx, real_tensor.element),
        Some(ElementType::F32)
    );
    let rhs = tensor(&ctx, &[3, 2], ElementType::F32);
    let complex_result = stablehlo::inference::infer_complex_type(&ctx, rhs, rhs).unwrap();
    assert_eq!(
        tensor_parts(&ctx, complex_result).unwrap(),
        (vec![3, 2], ElementType::C64)
    );
    let i = tensor(&ctx, &[3], ElementType::I32);
    let compare =
        stablehlo::inference::infer_compare_type(&ctx, i, i, Some(CompareType::Signed)).unwrap();
    assert_eq!(
        tensor_parts(&ctx, compare).unwrap(),
        (vec![3], ElementType::I1)
    );
    let pred = tensor(&ctx, &[], ElementType::I1);
    let selected = stablehlo::inference::infer_select_type(&ctx, pred, i, i).unwrap();
    assert_eq!(
        tensor_parts(&ctx, selected).unwrap(),
        (vec![3], ElementType::I32)
    );
    let clamped = stablehlo::inference::infer_clamp_type(&ctx, i, i, i).unwrap();
    assert_eq!(
        tensor_parts(&ctx, clamped).unwrap(),
        (vec![3], ElementType::I32)
    );
    assert!(
        stablehlo::inference::infer_abs_type(&ctx, tensor(&ctx, &[3], ElementType::I1)).is_err()
    );
    assert!(
        stablehlo::inference::infer_compare_type(&ctx, i, i, Some(CompareType::Float)).is_err()
    );
}

#[test]
fn conversion_and_bitcast_inference_reject_invalid_domains() {
    let ctx = Context::new();
    let input = tensor(&ctx, &[2, 4], ElementType::I32);
    let converted =
        stablehlo::inference::infer_convert_type(&ctx, input, ElementType::F64).unwrap();
    assert_eq!(
        tensor_parts(&ctx, converted).unwrap(),
        (vec![2, 4], ElementType::F64)
    );
    let template = tensor(&ctx, &[999], ElementType::I8);
    let bitcast = stablehlo::inference::infer_bitcast_convert_type(&ctx, input, template).unwrap();
    assert_eq!(
        tensor_parts(&ctx, bitcast).unwrap(),
        (vec![2, 4, 4], ElementType::I8)
    );
    let invalid = tensor(&ctx, &[2, 3], ElementType::F32);
    let wide_template = tensor(&ctx, &[99], ElementType::F64);
    assert!(
        stablehlo::inference::infer_bitcast_convert_type(&ctx, invalid, wide_template).is_err()
    );
    assert!(stablehlo::inference::infer_convert_type(&ctx, input, ElementType::Index).is_err());
}

#[test]
fn quantization_inference_uses_operand_shape_and_expressed_type() {
    let ctx = Context::new();
    let input = tensor(&ctx, &[2], ElementType::F32);
    let qelement = QuantizedElementType::intern(
        &ctx,
        UniformQuantizedSpec {
            storage_type: ElementType::I8,
            storage_min: None,
            storage_max: None,
            expressed_type: ElementType::F32,
            quantization_dimension: None,
            scales: vec![0.5],
            zero_points: vec![0],
        },
    );
    let template = TensorType::from_element(&ctx, qelement, vec![99]);
    let quantized =
        stablehlo::inference::infer_uniform_quantize_type(&ctx, input, template).unwrap();
    let quant_ty = quantized.deref(&ctx);
    let quant_tensor = quant_ty.downcast_ref::<TensorType>().unwrap();
    assert_eq!(quant_tensor.dimensions, vec![2]);
    let expressed = stablehlo::inference::infer_uniform_dequantize_type(&ctx, quantized).unwrap();
    assert_eq!(
        tensor_parts(&ctx, expressed).unwrap(),
        (vec![2], ElementType::F32)
    );
    assert!(
        stablehlo::inference::infer_uniform_quantize_type(
            &ctx,
            tensor(&ctx, &[2], ElementType::I32),
            template
        )
        .is_err()
    );
}

#[test]
fn compare_modes_follow_integer_float_and_complex_domains() {
    let mut ctx = Context::new();
    for (element, allowed) in [
        (
            ElementType::I32,
            vec![
                CompareType::NoType,
                CompareType::Signed,
                CompareType::Unsigned,
            ],
        ),
        (
            ElementType::F32,
            vec![
                CompareType::Float,
                CompareType::TotalOrder,
                CompareType::WeakOrder,
            ],
        ),
        (ElementType::C64, vec![CompareType::Float]),
    ] {
        let input = value(&mut ctx, &[2], element, 0);
        for mode in [
            CompareType::NoType,
            CompareType::Float,
            CompareType::TotalOrder,
            CompareType::Signed,
            CompareType::Unsigned,
            CompareType::WeakOrder,
        ] {
            let result = tensor(&ctx, &[2], ElementType::I1);
            let op = CompareOp::new(&mut ctx, input, input, result, CompareDirection::Eq, mode);
            assert_eq!(
                op.verify(&ctx).is_ok(),
                allowed.contains(&mode),
                "{element:?}: {mode:?}"
            );
        }
    }
    let i = value(&mut ctx, &[2], ElementType::I32, 0);
    let pred_result = tensor(&ctx, &[2], ElementType::I1);
    assert!(
        CompareOp::new_without_type(&mut ctx, i, i, pred_result, CompareDirection::Eq)
            .verify(&ctx)
            .is_ok()
    );
}

#[test]
fn constant_domain_and_reduce_precision_attributes_are_checked() {
    let mut ctx = Context::new();
    for element in [ElementType::Index, ElementType::Tf32, ElementType::Si8] {
        let ty = tensor(&ctx, &[2], element);
        assert!(
            ConstantOp::new(&mut ctx, ty, vec![0], true)
                .verify(&ctx)
                .is_err(),
            "{element:?}"
        );
    }
    let f = value(&mut ctx, &[2], ElementType::F32, 0);
    let same = tensor(&ctx, &[2], ElementType::F32);
    let different = tensor(&ctx, &[3], ElementType::F32);
    assert!(
        ReducePrecisionOp::new(&mut ctx, f, same, 8, 0)
            .verify(&ctx)
            .is_ok()
    );
    assert!(
        ReducePrecisionOp::new(&mut ctx, f, same, 0, 23)
            .verify(&ctx)
            .is_err()
    );
    assert!(
        ReducePrecisionOp::new(&mut ctx, f, same, 8, -1)
            .verify(&ctx)
            .is_err()
    );
    assert!(
        ReducePrecisionOp::new(&mut ctx, f, different, 8, 23)
            .verify(&ctx)
            .is_err()
    );
}

#[test]
fn select_and_clamp_reject_invalid_shapes_and_elements() {
    let mut ctx = Context::new();
    let f = value(&mut ctx, &[2], ElementType::F32, 0);
    let other = value(&mut ctx, &[3], ElementType::F32, 0);
    let i = value(&mut ctx, &[2], ElementType::I32, 0);
    let wrong_pred = value(&mut ctx, &[2], ElementType::I32, 0);
    let result = tensor(&ctx, &[2], ElementType::F32);
    assert!(
        SelectOp::new(&mut ctx, wrong_pred, f, f, result)
            .verify(&ctx)
            .is_err()
    );
    assert!(SelectOp::new_inferred(&mut ctx, wrong_pred, f, f).is_err());
    assert!(
        ClampOp::new(&mut ctx, other, f, f, result)
            .verify(&ctx)
            .is_err()
    );
    assert!(
        ClampOp::new(&mut ctx, i, f, f, result)
            .verify(&ctx)
            .is_err()
    );
}

#[test]
fn elementwise_inference_intersects_bounds_and_verification_checks_witnesses() {
    let mut ctx = Context::new();
    let bounded = |ctx: &Context, bound: i64| {
        let tensor = TensorType::from_element(ctx, ElementType::F32.native(ctx), vec![-1]);
        tensor
            .deref(ctx)
            .downcast_ref::<TensorType>()
            .unwrap()
            .with_encoding(
                ctx,
                Box::new(TypeExtensionsAttr {
                    bounds: vec![bound],
                }),
            )
    };
    let lhs_ty = bounded(&ctx, 8);
    let rhs_ty = bounded(&ctx, 5);
    let lhs = ConstantOp::new(&mut ctx, lhs_ty, vec![0], true)
        .get_operation()
        .deref(&ctx)
        .get_result(0);
    let rhs = ConstantOp::new(&mut ctx, rhs_ty, vec![0], true)
        .get_operation()
        .deref(&ctx)
        .get_result(0);
    let add = AddOp::new_inferred(&mut ctx, lhs, rhs).unwrap();
    let result = add.get_operation().deref(&ctx).get_result(0).get_type(&ctx);
    let bounds = result
        .deref(&ctx)
        .downcast_ref::<TensorType>()
        .unwrap()
        .bounds()
        .map(<[i64]>::to_vec);
    assert_eq!(bounds, Some(vec![5]));
    assert!(add.verify(&ctx).is_ok());

    let static_ty = tensor(&ctx, &[9], ElementType::F32);
    let static_value = ConstantOp::new(&mut ctx, static_ty, vec![0], true)
        .get_operation()
        .deref(&ctx)
        .get_result(0);
    assert!(AddOp::new_inferred(&mut ctx, lhs, static_value).is_err());

    let too_narrow_result = bounded(&ctx, 4);
    assert!(
        AddOp::new(&mut ctx, lhs, rhs, too_narrow_result)
            .verify(&ctx)
            .is_ok()
    );
    let incompatible_static = tensor(&ctx, &[10], ElementType::F32);
    assert!(
        AddOp::new(&mut ctx, lhs, static_value, incompatible_static)
            .verify(&ctx)
            .is_err()
    );
}

#[test]
fn dynamic_iota_constant_shape_values_match_result_extents() {
    let mut ctx = Context::new();
    let shape_ty = tensor(&ctx, &[2], ElementType::I32);
    let matching = ConstantOp::new(&mut ctx, shape_ty, vec![2, 3], false)
        .get_operation()
        .deref(&ctx)
        .get_result(0);
    let mismatch = ConstantOp::new(&mut ctx, shape_ty, vec![2, 4], false)
        .get_operation()
        .deref(&ctx)
        .get_result(0);
    let result = tensor(&ctx, &[2, 3], ElementType::F32);
    assert!(
        DynamicIotaOp::new(&mut ctx, matching, result, 1)
            .verify(&ctx)
            .is_ok()
    );
    assert!(
        DynamicIotaOp::new(&mut ctx, mismatch, result, 1)
            .verify(&ctx)
            .is_err()
    );
}

#[test]
fn bitcast_inference_projects_bounds_and_uses_quantized_storage_width() {
    let ctx = Context::new();
    let input = TensorType::from_element(&ctx, ElementType::I32.native(&ctx), vec![-1]);
    let input = input
        .deref(&ctx)
        .downcast_ref::<TensorType>()
        .unwrap()
        .with_encoding(&ctx, Box::new(TypeExtensionsAttr { bounds: vec![7] }));
    let output = tensor(&ctx, &[99], ElementType::I8);
    let inferred = stablehlo::inference::infer_bitcast_convert_type(&ctx, input, output).unwrap();
    let (dimensions, bounds) = {
        let ty = inferred.deref(&ctx);
        let inferred = ty.downcast_ref::<TensorType>().unwrap();
        (
            inferred.dimensions.clone(),
            inferred.bounds().map(<[i64]>::to_vec),
        )
    };
    assert_eq!(dimensions, vec![-1, 4]);
    assert_eq!(bounds, Some(vec![7, -1]));

    let q = QuantizedElementType::intern(
        &ctx,
        UniformQuantizedSpec {
            storage_type: ElementType::I8,
            storage_min: None,
            storage_max: None,
            expressed_type: ElementType::F32,
            quantization_dimension: None,
            scales: vec![0.5],
            zero_points: vec![0],
        },
    );
    let qinput = TensorType::from_element(&ctx, q, vec![2]);
    let i8_output = tensor(&ctx, &[9], ElementType::I8);
    let inferred =
        stablehlo::inference::infer_bitcast_convert_type(&ctx, qinput, i8_output).unwrap();
    let dimensions = {
        let ty = inferred.deref(&ctx);
        ty.downcast_ref::<TensorType>().unwrap().dimensions.clone()
    };
    assert_eq!(dimensions, vec![2]);
}

#[test]
fn per_axis_quantization_follows_each_operation_schema_domain() {
    let mut ctx = Context::new();
    let per_axis = QuantizedElementType::intern(
        &ctx,
        UniformQuantizedSpec {
            storage_type: ElementType::I8,
            storage_min: None,
            storage_max: None,
            expressed_type: ElementType::F32,
            quantization_dimension: Some(0),
            scales: vec![0.25, 0.5],
            zero_points: vec![0, 1],
        },
    );
    let qtype = TensorType::from_element(&ctx, per_axis, vec![2]);
    let q = ConstantOp::new(&mut ctx, qtype, vec![0], true)
        .get_operation()
        .deref(&ctx)
        .get_result(0);
    let predicate = value(&mut ctx, &[], ElementType::I1, 1);
    assert!(AddOp::new_inferred(&mut ctx, q, q).is_ok());
    assert!(MaximumOp::new_inferred(&mut ctx, q, q).is_err());
    assert!(SelectOp::new_inferred(&mut ctx, predicate, q, q).is_err());
    assert!(ClampOp::new_inferred(&mut ctx, q, q, q).is_err());
    assert!(CompareOp::new_inferred(&mut ctx, q, q, CompareDirection::Eq, None).is_err());
}

#[test]
fn integer_domains_distinguish_predicates_from_hlo_integer_ops() {
    let mut ctx = Context::new();
    let predicate = value(&mut ctx, &[2], ElementType::I1, 0);
    assert!(NotOp::new_inferred(&mut ctx, predicate).is_ok());
    assert!(CountLeadingZerosOp::new_inferred(&mut ctx, predicate).is_err());
    assert!(ShiftLeftOp::new_inferred(&mut ctx, predicate, predicate).is_err());
    let unsupported_type = tensor(&ctx, &[2], ElementType::Ui1);
    assert!(
        ConstantOp::new(&mut ctx, unsupported_type, vec![0], true)
            .verify(&ctx)
            .is_err()
    );
    let integer = value(&mut ctx, &[2], ElementType::I2, 0);
    assert!(AddOp::new_inferred(&mut ctx, integer, integer).is_ok());
}

#[test]
fn sparse_constants_verify_native_typed_values_and_indices() {
    let mut ctx = Context::new();
    let ty = tensor(&ctx, &[2], ElementType::I32);
    let scalar = Box::new(ScalarBitsAttr {
        ty: ElementType::I32.native(&ctx),
        words: vec![7],
    }) as AttrObj;
    let valid = ConstantOp::new_sparse(&mut ctx, ty, vec![vec![0]], vec![scalar.clone()]);
    assert!(valid.verify(&ctx).is_ok());
    let invalid = ConstantOp::new_sparse(&mut ctx, ty, vec![vec![2]], vec![scalar]);
    assert!(invalid.verify(&ctx).is_err());
}

#[test]
fn dense_constants_validate_payload_word_counts_and_high_bits() {
    let mut ctx = Context::new();
    let i1 = tensor(&ctx, &[1], ElementType::I1);
    assert!(
        ConstantOp::new(&mut ctx, i1, vec![1], true)
            .verify(&ctx)
            .is_ok()
    );
    assert!(
        ConstantOp::new(&mut ctx, i1, vec![2], true)
            .verify(&ctx)
            .is_err()
    );

    let signed = tensor(&ctx, &[1], ElementType::I8);
    assert!(
        ConstantOp::new(&mut ctx, signed, vec![0xff], true)
            .verify(&ctx)
            .is_ok()
    );
    assert!(
        ConstantOp::new(&mut ctx, signed, vec![0x1ff], true)
            .verify(&ctx)
            .is_err()
    );

    let complex = tensor(&ctx, &[1], ElementType::C128);
    assert!(
        ConstantOp::new(&mut ctx, complex, vec![0, 0], true)
            .verify(&ctx)
            .is_ok()
    );
    assert!(
        ConstantOp::new(&mut ctx, complex, vec![0], true)
            .verify(&ctx)
            .is_err()
    );
}
