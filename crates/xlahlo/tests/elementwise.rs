use pliron::{
    common_traits::Verify, context::Context, identifier::Identifier, op::Op, r#type::TypeHandle,
};
use xlahlo::{
    Diagnostic,
    attributes::{ComparisonOrder, Direction, ReducePrecision, ResultAccuracy},
    inference::elementwise as inference,
    opcode::Opcode as O,
    ops::{
        HloOpInterface,
        control::ParameterOp,
        elementwise::{
            AbsOp, AddOp, BitcastConvertOp, ClampOp, CompareOp, ComplexOp, ExpOp,
            ReducePrecisionOp, RoundNearestAfzOp, SelectOp, StochasticConvertOp,
        },
    },
    shape::{Dimension, Layout, Shape},
    types::{ElementType as E, tensor},
    verification::VerifyOptions,
};

fn assert_diagnostic<T: std::fmt::Debug>(
    result: xlahlo::pliron::result::Result<T>,
    expected: impl FnOnce(&Diagnostic) -> bool,
) {
    let error = result.unwrap_err();
    let diagnostic = error
        .err
        .downcast_ref::<Diagnostic>()
        .expect("HLO error should carry a typed diagnostic");
    assert!(expected(diagnostic), "unexpected diagnostic: {diagnostic}");
}

fn array(ctx: &Context, element: E, dimensions: &[i64]) -> TypeHandle {
    Shape::array(ctx, element, dimensions.to_vec()).unwrap()
}

fn with_layout(
    ctx: &Context,
    element: E,
    dimensions: &[i64],
    minor_to_major: &[i64],
) -> TypeHandle {
    Shape::with_layout(
        ctx,
        array(ctx, element, dimensions),
        Layout {
            minor_to_major: minor_to_major.to_vec(),
            ..Layout::default()
        },
    )
    .unwrap()
}

#[test]
fn unary_inference_covers_every_catalog_unary_opcode() {
    let ctx = Context::new();
    let integral = array(&ctx, E::S32, &[2]);
    let float = array(&ctx, E::F32, &[2]);
    let complex = array(&ctx, E::C64, &[2]);
    for opcode in [
        O::Abs,
        O::Acos,
        O::Acosh,
        O::Asin,
        O::Asinh,
        O::Atanh,
        O::Cbrt,
        O::Ceil,
        O::Clz,
        O::Cos,
        O::Cosh,
        O::Erf,
        O::Exp,
        O::Exp2,
        O::Expm1,
        O::Floor,
        O::Imag,
        O::IsFinite,
        O::Log,
        O::Log1p,
        O::Log2,
        O::Logistic,
        O::Negate,
        O::Not,
        O::PopulationCount,
        O::Real,
        O::RoundNearestAfz,
        O::RoundNearestEven,
        O::Rsqrt,
        O::Sign,
        O::Sin,
        O::Sinh,
        O::Sqrt,
        O::Tan,
        O::Tanh,
    ] {
        let input = if matches!(opcode, O::Clz | O::Negate | O::Not | O::PopulationCount) {
            integral
        } else {
            float
        };
        let result = inference::infer_unary(&ctx, opcode, input)
            .unwrap_or_else(|error| panic!("{opcode}: {error}"));
        let actual = tensor(&ctx, result).unwrap().element;
        let expected = if opcode == O::IsFinite {
            E::Pred
        } else if matches!(opcode, O::Clz | O::Negate | O::Not | O::PopulationCount) {
            E::S32
        } else {
            E::F32
        };
        assert_eq!(actual, expected, "{opcode}");
    }
    for opcode in [
        O::Abs,
        O::Acos,
        O::Acosh,
        O::Asin,
        O::Asinh,
        O::Atanh,
        O::Cos,
        O::Cosh,
        O::Exp,
        O::Exp2,
        O::Expm1,
        O::Imag,
        O::Log,
        O::Log1p,
        O::Log2,
        O::Logistic,
        O::Negate,
        O::Real,
        O::Rsqrt,
        O::Sign,
        O::Sin,
        O::Sinh,
        O::Sqrt,
        O::Tan,
        O::Tanh,
    ] {
        assert!(
            inference::infer_unary(&ctx, opcode, complex).is_ok(),
            "{opcode}"
        );
    }
    assert_eq!(
        tensor(&ctx, inference::infer_unary(&ctx, O::Abs, complex).unwrap())
            .unwrap()
            .element,
        E::F32
    );
    assert_diagnostic(
        inference::infer_unary(&ctx, O::Ceil, integral),
        |diagnostic| matches!(diagnostic, Diagnostic::Inference { .. }),
    );
}

#[test]
fn binary_inference_covers_opcode_domains_and_precision_promotion() {
    let ctx = Context::new();
    let f32s = array(&ctx, E::F32, &[2]);
    let f16s = array(&ctx, E::F16, &[2]);
    for opcode in [
        O::Add,
        O::Atan2,
        O::Compare,
        O::Divide,
        O::Maximum,
        O::Minimum,
        O::Multiply,
        O::Power,
        O::Remainder,
        O::Subtract,
    ] {
        let result = inference::infer_binary(&ctx, opcode, f16s, f32s).unwrap();
        assert_eq!(
            tensor(&ctx, result).unwrap().element,
            if opcode == O::Compare {
                E::Pred
            } else {
                E::F32
            },
            "{opcode}"
        );
    }
    let s32s = array(&ctx, E::S32, &[2]);
    for opcode in [
        O::And,
        O::Mulhi,
        O::Or,
        O::ShiftLeft,
        O::ShiftRightArithmetic,
        O::ShiftRightLogical,
        O::Xor,
    ] {
        assert!(
            inference::infer_binary(&ctx, opcode, s32s, s32s).is_ok(),
            "{opcode}"
        );
    }
    let predicates = array(&ctx, E::Pred, &[2]);
    for opcode in [O::And, O::Or, O::Xor] {
        assert!(
            inference::infer_binary(&ctx, opcode, predicates, predicates).is_ok(),
            "{opcode}"
        );
    }
    let scalar = array(&ctx, E::F32, &[]);
    assert_eq!(
        inference::infer_scalar_broadcast(&ctx, &[scalar, f32s]).unwrap(),
        Some(f32s)
    );
    assert!(inference::infer_scalar_broadcast(&ctx, &[f32s, array(&ctx, E::F32, &[3])]).is_err());
    let mapped = inference::infer_binary_in_dim(
        &ctx,
        O::Add,
        array(&ctx, E::F32, &[2]),
        array(&ctx, E::F32, &[3, 2]),
        &[1],
    )
    .unwrap();
    assert_eq!(tensor(&ctx, mapped).unwrap().dimensions, [3, 2]);
    assert!(
        inference::infer_binary_in_dim(
            &ctx,
            O::Add,
            array(&ctx, E::F32, &[2]),
            array(&ctx, E::F32, &[3, 2]),
            &[],
        )
        .is_err()
    );
    for dimensions in [&[-1][..], &[3][..], &[1, 0][..]] {
        let error = inference::infer_binary_in_dim(
            &ctx,
            O::Add,
            array(&ctx, E::F32, &[2, 2]),
            array(&ctx, E::F32, &[3, 2, 2]),
            dimensions,
        );
        assert_diagnostic(error, |diagnostic| {
            matches!(diagnostic, Diagnostic::Inference { .. })
        });
    }
    assert_eq!(
        tensor(
            &ctx,
            inference::infer_binary(&ctx, O::Complex, f32s, array(&ctx, E::F32, &[2])).unwrap()
        )
        .unwrap()
        .element,
        E::C64
    );
    for opcode in [O::Power, O::Divide, O::Remainder, O::Subtract] {
        assert!(
            inference::infer_binary(&ctx, opcode, predicates, predicates).is_err(),
            "{opcode}"
        );
    }
    assert!(inference::infer_binary(&ctx, O::And, f32s, f32s).is_err());
    assert!(inference::infer_binary(&ctx, O::Complex, s32s, f32s).is_err());
}

#[test]
fn elementwise_inference_preserves_dynamic_shapes_and_checks_special_conversions() {
    let ctx = Context::new();
    let dynamic = Shape::dynamic_array(
        &ctx,
        E::F32,
        vec![Dimension::Bounded(5), Dimension::Static(2)],
    )
    .unwrap();
    let same = inference::infer_binary(&ctx, O::Add, dynamic, dynamic).unwrap();
    assert_eq!(
        tensor(&ctx, same).unwrap().dynamic_dimensions,
        [true, false]
    );
    let nondefault_layout = with_layout(&ctx, E::F32, &[2, 3], &[0, 1]);
    let layout_preserved =
        inference::infer_binary(&ctx, O::Add, nondefault_layout, nondefault_layout).unwrap();
    assert_eq!(
        Shape::layout(&ctx, layout_preserved)
            .unwrap()
            .unwrap()
            .minor_to_major,
        [0, 1]
    );
    let singleton = array(&ctx, E::F32, &[1, 4]);
    let wider = array(&ctx, E::F32, &[3, 4]);
    assert_eq!(
        Shape::layout(
            &ctx,
            inference::infer_binary(&ctx, O::Add, singleton, wider).unwrap()
        )
        .unwrap()
        .unwrap()
        .minor_to_major,
        [1, 0]
    );

    let f32s = array(&ctx, E::F32, &[2]);
    assert_eq!(
        tensor(
            &ctx,
            inference::infer_select(&ctx, array(&ctx, E::Pred, &[]), f32s, f32s).unwrap()
        )
        .unwrap()
        .element,
        E::F32
    );
    let unbounded_predicate =
        Shape::dynamic_array(&ctx, E::Pred, vec![Dimension::Unbounded]).unwrap();
    let static_value = array(&ctx, E::F32, &[4]);
    assert_eq!(
        Shape::dimension(
            &ctx,
            inference::infer_select(&ctx, unbounded_predicate, static_value, static_value).unwrap(),
            0
        )
        .unwrap(),
        Dimension::Unbounded
    );
    assert!(inference::infer_clamp(&ctx, array(&ctx, E::F32, &[1]), f32s, f32s).is_err());
    let scalar = array(&ctx, E::F32, &[]);
    assert!(inference::infer_clamp(&ctx, scalar, f32s, scalar).is_ok());
    assert!(inference::infer_select(&ctx, f32s, f32s, f32s).is_err());
    assert!(
        inference::infer_select(
            &ctx,
            array(&ctx, E::Pred, &[2]),
            f32s,
            array(&ctx, E::F32, &[3])
        )
        .is_err()
    );
    assert_eq!(
        tensor(&ctx, inference::infer_convert(&ctx, f32s, E::F16).unwrap())
            .unwrap()
            .element,
        E::F16
    );
    assert!(inference::infer_reduce_precision(&ctx, f32s, 0, -1).is_err());
    assert!(inference::infer_reduce_precision(&ctx, array(&ctx, E::S32, &[2]), 8, 2).is_err());
    assert!(
        inference::infer_stochastic_convert(&ctx, f32s, array(&ctx, E::U32, &[2]), E::F16).is_ok()
    );
    assert!(
        inference::infer_stochastic_convert(&ctx, f32s, array(&ctx, E::S32, &[2]), E::F16).is_err()
    );
    let custom_input = with_layout(&ctx, E::F32, &[2, 3], &[0, 1]);
    let default_random = array(&ctx, E::U32, &[2, 3]);
    assert!(
        inference::infer_stochastic_convert(&ctx, custom_input, default_random, E::F16).is_err()
    );
    let bounded_input = Shape::dynamic_array(&ctx, E::F32, vec![Dimension::Bounded(2)]).unwrap();
    assert!(
        inference::infer_stochastic_convert(&ctx, bounded_input, array(&ctx, E::U32, &[2]), E::F16)
            .is_err()
    );
    let packed = array(&ctx, E::S32, &[2]);
    let unpacked = inference::infer_bitcast_convert(&ctx, packed, E::S8).unwrap();
    assert_eq!(tensor(&ctx, unpacked).unwrap().dimensions, [2, 4]);
    assert!(inference::infer_bitcast_convert(&ctx, array(&ctx, E::F32, &[3]), E::F64).is_err());
}

#[test]
fn native_elementwise_ops_verify_types_and_required_attributes() {
    let mut ctx = Context::new();
    xlahlo::register(&mut ctx);
    let f32s = array(&ctx, E::F32, &[2]);
    let parameter = ParameterOp::new(&mut ctx, f32s, vec![], xlahlo::attributes::Index(0));
    let value = parameter.result(&ctx);
    let good = AbsOp::new(&mut ctx, f32s, vec![value]);
    assert!(good.verify(&ctx).is_ok());
    let missing_unary_operand = AbsOp::new(&mut ctx, f32s, vec![]);
    assert!(missing_unary_operand.verify(&ctx).is_err());
    let missing_binary_operands = AddOp::new(&mut ctx, f32s, vec![value]);
    assert!(missing_binary_operands.verify(&ctx).is_err());
    let missing_ternary_operands = SelectOp::new(&mut ctx, f32s, vec![value, value]);
    assert!(missing_ternary_operands.verify(&ctx).is_err());
    let nondefault_input = with_layout(&ctx, E::F32, &[2, 3], &[0, 1]);
    let parameter_with_layout = ParameterOp::new(
        &mut ctx,
        nondefault_input,
        vec![],
        xlahlo::attributes::Index(3),
    );
    let layout_value = parameter_with_layout.result(&ctx);
    let default_output = array(&ctx, E::F32, &[2, 3]);
    let layout_sensitive_abs = AbsOp::new(&mut ctx, default_output, vec![layout_value]);
    assert!(layout_sensitive_abs.verify(&ctx).is_ok());
    let options = VerifyOptions {
        layout_sensitive: true,
        ..Default::default()
    };
    assert!(
        layout_sensitive_abs
            .verify_with_options(&ctx, options)
            .is_err()
    );
    let wrong_unary_result = array(&ctx, E::S32, &[2]);
    let bad = AbsOp::new(&mut ctx, wrong_unary_result, vec![value]);
    assert!(bad.verify(&ctx).is_err());
    let predicate_result = array(&ctx, E::Pred, &[2]);
    let compare = CompareOp::new(
        &mut ctx,
        predicate_result,
        vec![value, value],
        Direction::Eq,
        ComparisonOrder::Total,
    );
    assert!(compare.verify(&ctx).is_ok());
    let s32_type = array(&ctx, E::S32, &[2]);
    let integer_parameter =
        ParameterOp::new(&mut ctx, s32_type, vec![], xlahlo::attributes::Index(4));
    let integer_value = integer_parameter.result(&ctx);
    let partial_integer_compare = CompareOp::new(
        &mut ctx,
        predicate_result,
        vec![integer_value, integer_value],
        Direction::Lt,
        ComparisonOrder::Partial,
    );
    assert!(partial_integer_compare.verify(&ctx).is_err());
    let complex_result = array(&ctx, E::C64, &[2]);
    let complex = ComplexOp::new(&mut ctx, complex_result, vec![value, value]);
    assert!(complex.verify(&ctx).is_ok());
    let predicate_type = array(&ctx, E::Pred, &[2]);
    let pred = ParameterOp::new(
        &mut ctx,
        predicate_type,
        vec![],
        xlahlo::attributes::Index(1),
    );
    let predicate = pred.result(&ctx);
    let select = SelectOp::new(&mut ctx, f32s, vec![predicate, value, value]);
    assert!(select.verify(&ctx).is_ok());
    let clamp = ClampOp::new(&mut ctx, f32s, vec![value, value, value]);
    assert!(clamp.verify(&ctx).is_ok());
    let round = RoundNearestAfzOp::new(&mut ctx, f32s, vec![value]);
    assert!(round.verify(&ctx).is_ok());
    let accurate_exp = ExpOp::new(&mut ctx, f32s, vec![value]);
    accurate_exp.get_operation().deref_mut(&ctx).attributes.set(
        Identifier::new("hlo_result_accuracy"),
        ResultAccuracy::tolerance(1e-4, 1e-4, 2).unwrap(),
    );
    assert!(accurate_exp.verify(&ctx).is_ok());
    let invalid_accuracy_exp = ExpOp::new(&mut ctx, f32s, vec![value]);
    invalid_accuracy_exp
        .get_operation()
        .deref_mut(&ctx)
        .attributes
        .set(
            Identifier::new("hlo_result_accuracy"),
            ResultAccuracy::Tolerance {
                atol_bits: (-1.0_f64).to_bits(),
                rtol_bits: 0.0_f64.to_bits(),
                ulps: 0,
            },
        );
    assert!(invalid_accuracy_exp.verify(&ctx).is_err());
    let bitcast_result = array(&ctx, E::S32, &[2, 4]);
    let bitcast = BitcastConvertOp::new(&mut ctx, bitcast_result, vec![value]);
    assert!(
        bitcast.verify(&ctx).is_err(),
        "bitcast result geometry checked"
    );
    let precision = ReducePrecisionOp::new(
        &mut ctx,
        f32s,
        vec![value],
        ReducePrecision {
            exponent_bits: 8,
            mantissa_bits: 10,
        },
    );
    assert!(precision.verify(&ctx).is_ok());
    let random_type = array(&ctx, E::U32, &[2]);
    let random = ParameterOp::new(&mut ctx, random_type, vec![], xlahlo::attributes::Index(2));
    let converted_type = array(&ctx, E::F16, &[2]);
    let random_value = random.result(&ctx);
    let stochastic = StochasticConvertOp::new(&mut ctx, converted_type, vec![value, random_value]);
    assert!(stochastic.verify(&ctx).is_ok());
    let add = AddOp::new(&mut ctx, f32s, vec![value, value]);
    assert!(add.verify(&ctx).is_ok());
    let f16_type = array(&ctx, E::F16, &[2]);
    let f16_parameter = ParameterOp::new(&mut ctx, f16_type, vec![], xlahlo::attributes::Index(5));
    let f16_value = f16_parameter.result(&ctx);
    let mixed_add = AddOp::new(&mut ctx, f32s, vec![f16_value, value]);
    assert!(mixed_add.verify(&ctx).is_err());
    let mixed_options = VerifyOptions {
        allow_mixed_precision: true,
        ..Default::default()
    };
    assert!(mixed_add.verify_with_options(&ctx, mixed_options).is_ok());
    let wrong_add_result = array(&ctx, E::F32, &[3]);
    let bad_add = AddOp::new(&mut ctx, wrong_add_result, vec![value, value]);
    assert!(bad_add.verify(&ctx).is_err());
    let bad_accuracy = AddOp::new(&mut ctx, f32s, vec![value, value]);
    bad_accuracy.get_operation().deref_mut(&ctx).attributes.set(
        Identifier::new("hlo_result_accuracy"),
        ResultAccuracy::Default,
    );
    assert!(bad_accuracy.verify(&ctx).is_err());
}

#[test]
fn elementwise_verification_requires_typed_compare_and_precision_attributes() {
    let mut ctx = Context::new();
    xlahlo::register(&mut ctx);
    let float_type = array(&ctx, E::F32, &[2]);
    let predicate_type = array(&ctx, E::Pred, &[2]);
    let parameter = ParameterOp::new(&mut ctx, float_type, vec![], xlahlo::attributes::Index(0));
    let value = parameter.result(&ctx);

    let missing_direction = CompareOp::new(
        &mut ctx,
        predicate_type,
        vec![value, value],
        Direction::Eq,
        ComparisonOrder::Total,
    );
    missing_direction
        .get_operation()
        .deref_mut(&ctx)
        .attributes
        .0
        .remove(&Identifier::new("hlo_direction"));
    assert_diagnostic(missing_direction.verify(&ctx), |diagnostic| {
        matches!(
            diagnostic,
            Diagnostic::InvalidAttribute {
                attribute: "hlo_direction",
                ..
            }
        )
    });

    let ill_typed_order = CompareOp::new(
        &mut ctx,
        predicate_type,
        vec![value, value],
        Direction::Eq,
        ComparisonOrder::Total,
    );
    ill_typed_order
        .get_operation()
        .deref_mut(&ctx)
        .attributes
        .set(Identifier::new("hlo_order"), Direction::Eq);
    assert_diagnostic(ill_typed_order.verify(&ctx), |diagnostic| {
        matches!(
            diagnostic,
            Diagnostic::InvalidAttribute {
                attribute: "hlo_order",
                ..
            }
        )
    });

    let missing_precision = ReducePrecisionOp::new(
        &mut ctx,
        float_type,
        vec![value],
        ReducePrecision {
            exponent_bits: 8,
            mantissa_bits: 10,
        },
    );
    missing_precision
        .get_operation()
        .deref_mut(&ctx)
        .attributes
        .0
        .remove(&Identifier::new("hlo_precision"));
    assert_diagnostic(missing_precision.verify(&ctx), |diagnostic| {
        matches!(
            diagnostic,
            Diagnostic::InvalidAttribute {
                attribute: "hlo_precision",
                ..
            }
        )
    });
}
