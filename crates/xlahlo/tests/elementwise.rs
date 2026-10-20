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
        control::{DomainOp, ParameterOp},
        elementwise::{
            AbsOp, AddOp, BitcastConvertOp, ClampOp, CompareOp, ComplexOp, ConvertOp, ExpOp,
            ReducePrecisionOp, RoundNearestAfzOp, SelectOp, StochasticConvertOp,
        },
        movement::CopyOp,
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

#[derive(Clone, Copy)]
enum UnaryDomain {
    Floating,
    FloatingOrComplex,
    SignedFloatingOrComplex,
    Integral,
    Numeric,
    PredicateOrIntegral,
}

impl UnaryDomain {
    fn accepts(self, element: E) -> bool {
        match self {
            Self::Floating => element.is_float(),
            Self::FloatingOrComplex => element.is_float() || element.is_complex(),
            Self::SignedFloatingOrComplex => {
                element.is_signed() || element.is_float() || element.is_complex()
            }
            Self::Integral => element.is_integer(),
            Self::Numeric => element.is_integer() || element.is_float() || element.is_complex(),
            Self::PredicateOrIntegral => element == E::Pred || element.is_integer(),
        }
    }
}

#[derive(Clone, Copy)]
enum BinaryDomain {
    Any,
    NonPredicate,
    FloatingOrComplex,
    PredicateOrIntegral,
    ComplexComponents,
}

impl BinaryDomain {
    fn accepts(self, element: E) -> bool {
        match self {
            Self::Any => true,
            Self::NonPredicate => element != E::Pred,
            Self::FloatingOrComplex => element.is_float() || element.is_complex(),
            Self::PredicateOrIntegral => element == E::Pred || element.is_integer(),
            Self::ComplexComponents => matches!(element, E::F32 | E::F64),
        }
    }
}

fn openxla_precision_key(element: E) -> (i32, i32, u32) {
    match element {
        E::F16 => (16, 11, 16),
        E::BF16 => (128, 8, 16),
        E::F32 => (128, 24, 32),
        E::F64 => (1024, 53, 64),
        E::F4E2M1FN => (3, 2, 4),
        E::F6E3M2FN => (5, 3, 6),
        E::F6E2M3FN => (3, 4, 6),
        E::F8E3M4 => (4, 5, 8),
        E::F8E4M3 => (8, 4, 8),
        E::F8E4M3FN => (9, 4, 8),
        E::F8E4M3B11FNUZ => (5, 4, 8),
        E::F8E5M2 | E::F8E5M2FNUZ => (16, 3, 8),
        E::F8E4M3FNUZ => (8, 4, 8),
        E::F8E8M0FNU => (128, 1, 8),
        _ => panic!("not a floating-point type: {element:?}"),
    }
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
fn dtype_domain_matrix_matches_pinned_openxla_for_all_unary_and_binary_ops() {
    let ctx = Context::new();
    let unary = [
        (O::Abs, UnaryDomain::SignedFloatingOrComplex),
        (O::Acos, UnaryDomain::FloatingOrComplex),
        (O::Acosh, UnaryDomain::FloatingOrComplex),
        (O::Asin, UnaryDomain::FloatingOrComplex),
        (O::Asinh, UnaryDomain::FloatingOrComplex),
        (O::Atanh, UnaryDomain::FloatingOrComplex),
        (O::Cbrt, UnaryDomain::Floating),
        (O::Ceil, UnaryDomain::Floating),
        (O::Clz, UnaryDomain::Integral),
        (O::Cos, UnaryDomain::FloatingOrComplex),
        (O::Cosh, UnaryDomain::FloatingOrComplex),
        (O::Erf, UnaryDomain::Floating),
        (O::Exp, UnaryDomain::FloatingOrComplex),
        (O::Exp2, UnaryDomain::FloatingOrComplex),
        (O::Expm1, UnaryDomain::FloatingOrComplex),
        (O::Floor, UnaryDomain::Floating),
        (O::Imag, UnaryDomain::FloatingOrComplex),
        (O::IsFinite, UnaryDomain::Floating),
        (O::Log, UnaryDomain::FloatingOrComplex),
        (O::Log1p, UnaryDomain::FloatingOrComplex),
        (O::Log2, UnaryDomain::FloatingOrComplex),
        (O::Logistic, UnaryDomain::FloatingOrComplex),
        (O::Negate, UnaryDomain::Numeric),
        (O::Not, UnaryDomain::PredicateOrIntegral),
        (O::PopulationCount, UnaryDomain::Integral),
        (O::Real, UnaryDomain::FloatingOrComplex),
        (O::RoundNearestAfz, UnaryDomain::Floating),
        (O::RoundNearestEven, UnaryDomain::Floating),
        (O::Rsqrt, UnaryDomain::FloatingOrComplex),
        (O::Sign, UnaryDomain::SignedFloatingOrComplex),
        (O::Sin, UnaryDomain::FloatingOrComplex),
        (O::Sinh, UnaryDomain::FloatingOrComplex),
        (O::Sqrt, UnaryDomain::FloatingOrComplex),
        (O::Tan, UnaryDomain::FloatingOrComplex),
        (O::Tanh, UnaryDomain::FloatingOrComplex),
    ];
    let binary = [
        (O::Add, BinaryDomain::Any),
        (O::And, BinaryDomain::PredicateOrIntegral),
        (O::Atan2, BinaryDomain::FloatingOrComplex),
        (O::Complex, BinaryDomain::ComplexComponents),
        (O::Compare, BinaryDomain::Any),
        (O::Divide, BinaryDomain::NonPredicate),
        (O::Maximum, BinaryDomain::Any),
        (O::Minimum, BinaryDomain::Any),
        (O::Mulhi, BinaryDomain::Any),
        (O::Multiply, BinaryDomain::Any),
        (O::Or, BinaryDomain::PredicateOrIntegral),
        (O::Power, BinaryDomain::NonPredicate),
        (O::Remainder, BinaryDomain::NonPredicate),
        (O::ShiftLeft, BinaryDomain::NonPredicate),
        (O::ShiftRightArithmetic, BinaryDomain::NonPredicate),
        (O::ShiftRightLogical, BinaryDomain::NonPredicate),
        (O::Subtract, BinaryDomain::NonPredicate),
        (O::Xor, BinaryDomain::PredicateOrIntegral),
    ];
    assert_eq!(unary.len() + binary.len() + 6, 59);

    let dimensions = [2];
    for (opcode, domain) in unary {
        for &element in E::ALL {
            let shape = array(&ctx, element, &dimensions);
            let inferred = inference::infer_unary(&ctx, opcode, shape);
            assert_eq!(
                inferred.is_ok(),
                domain.accepts(element),
                "{opcode} with {element:?}"
            );
            if let Ok(inferred) = inferred {
                let expected = match (opcode, element) {
                    (O::IsFinite, _) => E::Pred,
                    (O::Abs | O::Real | O::Imag, E::C64) => E::F32,
                    (O::Abs | O::Real | O::Imag, E::C128) => E::F64,
                    _ => element,
                };
                assert_eq!(
                    tensor(&ctx, inferred).unwrap().element,
                    expected,
                    "{opcode} with {element:?}"
                );
            }
        }
    }
    for (opcode, domain) in binary {
        for &element in E::ALL {
            let shape = array(&ctx, element, &dimensions);
            let inferred = inference::infer_binary(&ctx, opcode, shape, shape);
            assert_eq!(
                inferred.is_ok(),
                domain.accepts(element),
                "{opcode} with {element:?}"
            );
            if let Ok(inferred) = inferred {
                let expected = match (opcode, element) {
                    (O::Compare, _) => E::Pred,
                    (O::Complex, E::F32) => E::C64,
                    (O::Complex, E::F64) => E::C128,
                    _ => element,
                };
                assert_eq!(
                    tensor(&ctx, inferred).unwrap().element,
                    expected,
                    "{opcode} with {element:?}"
                );
            }
        }
    }

    for &source_element in E::ALL {
        let source = array(&ctx, source_element, &dimensions);
        for &target_element in E::ALL {
            assert!(
                inference::infer_convert(&ctx, source, target_element).is_ok(),
                "convert {source_element:?} -> {target_element:?}"
            );
            assert!(
                inference::infer_clamp(&ctx, source, source, source).is_ok(),
                "clamp with {source_element:?}"
            );
            assert!(
                inference::infer_select(&ctx, array(&ctx, E::Pred, &dimensions), source, source)
                    .is_ok(),
                "select with {source_element:?}"
            );

            let input_bits = source_element.storage_bit_width();
            let output_bits = target_element.storage_bit_width();
            let complex_compatible = source_element.is_complex() == target_element.is_complex();
            let dims = if input_bits < output_bits {
                vec![2, i64::from(output_bits / input_bits)]
            } else {
                vec![2]
            };
            let source = array(&ctx, source_element, &dims);
            let result = inference::infer_bitcast_convert(&ctx, source, target_element);
            assert_eq!(
                result.is_ok(),
                complex_compatible,
                "bitcast-convert {source_element:?} -> {target_element:?}"
            );
            if let Ok(result) = result {
                let expected_dims = if input_bits > output_bits {
                    vec![2, i64::from(input_bits / output_bits)]
                } else if input_bits < output_bits {
                    vec![2]
                } else {
                    dims
                };
                assert_eq!(
                    tensor(&ctx, result).unwrap().dimensions,
                    expected_dims,
                    "bitcast-convert {source_element:?} -> {target_element:?}"
                );
            }
        }
    }

    let unsigned_integers = E::ALL
        .iter()
        .copied()
        .filter(|element| element.is_integer() && !element.is_signed())
        .collect::<Vec<_>>();
    for &input_element in E::ALL {
        let input = array(&ctx, input_element, &dimensions);
        assert_eq!(
            inference::infer_reduce_precision(&ctx, input, 1, 0).is_ok(),
            input_element.is_float(),
            "reduce-precision with {input_element:?}"
        );
        for &random_element in &unsigned_integers {
            let random = array(&ctx, random_element, &dimensions);
            assert_eq!(
                inference::infer_stochastic_convert(&ctx, input, random, E::F16).is_ok(),
                input_element.is_float() && input_element.bit_width() == random_element.bit_width(),
                "stochastic-convert {input_element:?} with {random_element:?}"
            );
        }
    }
    for (exponent_bits, mantissa_bits, succeeds) in
        [(1, 0, true), (1, 1, true), (0, 0, false), (1, -1, false)]
    {
        assert_eq!(
            inference::infer_reduce_precision(
                &ctx,
                array(&ctx, E::F32, &dimensions),
                exponent_bits,
                mantissa_bits,
            )
            .is_ok(),
            succeeds,
            "reduce-precision {exponent_bits}/{mantissa_bits}"
        );
    }
}

#[test]
fn mixed_floating_elementwise_promotion_matches_openxla_precision_ordering() {
    let ctx = Context::new();
    let float_types = E::ALL
        .iter()
        .copied()
        .filter(|element| element.is_float())
        .collect::<Vec<_>>();
    let predicate = array(&ctx, E::Pred, &[2]);
    for &lhs_element in &float_types {
        for &rhs_element in &float_types {
            let lhs_key = openxla_precision_key(lhs_element);
            let rhs_key = openxla_precision_key(rhs_element);
            if lhs_element != rhs_element && lhs_key == rhs_key {
                // Pinned HigherPrecisionType CHECK_EQs when two distinct
                // formats tie in every ordering property.
                continue;
            }
            let expected = if lhs_key >= rhs_key {
                lhs_element
            } else {
                rhs_element
            };
            let lhs = array(&ctx, lhs_element, &[2]);
            let rhs = array(&ctx, rhs_element, &[2]);
            for (left, right) in [(lhs, rhs), (rhs, lhs)] {
                assert_eq!(
                    tensor(
                        &ctx,
                        inference::infer_binary(&ctx, O::Add, left, right).unwrap()
                    )
                    .unwrap()
                    .element,
                    expected,
                    "add {lhs_element:?} + {rhs_element:?}"
                );
                assert_eq!(
                    tensor(
                        &ctx,
                        inference::infer_binary(&ctx, O::Compare, left, right).unwrap()
                    )
                    .unwrap()
                    .element,
                    E::Pred,
                    "compare {lhs_element:?} with {rhs_element:?}"
                );
                assert_eq!(
                    tensor(
                        &ctx,
                        inference::infer_select(&ctx, predicate, left, right).unwrap()
                    )
                    .unwrap()
                    .element,
                    expected,
                    "select {lhs_element:?} / {rhs_element:?}"
                );
            }
        }
    }
}

#[test]
fn mixed_floating_elementwise_verification_obeys_precision_policy() {
    let mut ctx = Context::new();
    xlahlo::register(&mut ctx);
    let float_types = E::ALL
        .iter()
        .copied()
        .filter(|element| element.is_float())
        .collect::<Vec<_>>();
    let operands = float_types
        .iter()
        .enumerate()
        .map(|(index, &element)| {
            let operand_type = array(&ctx, element, &[2]);
            let parameter = ParameterOp::new(
                &mut ctx,
                operand_type,
                vec![],
                xlahlo::attributes::Index(index as u32),
            );
            (element, parameter.result(&ctx))
        })
        .collect::<Vec<_>>();
    let mixed_options = VerifyOptions {
        allow_mixed_precision: true,
        ..Default::default()
    };

    for &(lhs_element, lhs) in &operands {
        for &(rhs_element, rhs) in &operands {
            let lhs_key = openxla_precision_key(lhs_element);
            let rhs_key = openxla_precision_key(rhs_element);
            if lhs_element != rhs_element && lhs_key == rhs_key {
                continue;
            }
            if lhs_element == rhs_element {
                continue;
            }
            let expected_element = if lhs_key >= rhs_key {
                lhs_element
            } else {
                rhs_element
            };
            let result_type = array(&ctx, expected_element, &[2]);
            let add = AddOp::new(&mut ctx, result_type, vec![lhs, rhs]);
            assert_diagnostic(add.verify(&ctx), |diagnostic| {
                matches!(
                    diagnostic,
                    Diagnostic::Verification { message }
                        if message.contains("mixed floating operand precision is disabled")
                )
            });
            assert!(
                add.verify_with_options(&ctx, mixed_options).is_ok(),
                "Add {lhs_element:?} + {rhs_element:?} with mixed precision enabled"
            );
        }
    }
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
    let dynamic_unary = inference::infer_unary(&ctx, O::Abs, dynamic).unwrap();
    assert_eq!(
        tensor(&ctx, dynamic_unary).unwrap().dynamic_dimensions,
        [true, false]
    );
    let unbounded = Shape::dynamic_array(&ctx, E::F32, vec![Dimension::Unbounded]).unwrap();
    let bounded_static = array(&ctx, E::F32, &[4]);
    for (lhs, rhs) in [(unbounded, bounded_static), (bounded_static, unbounded)] {
        let inferred = inference::infer_binary(&ctx, O::Add, lhs, rhs).unwrap();
        assert_eq!(
            tensor(&ctx, inferred).unwrap().dimensions,
            [4],
            "a static dimension resolves the unbounded operand extent"
        );
        assert_eq!(tensor(&ctx, inferred).unwrap().dynamic_dimensions, [false]);
    }
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
    let custom_predicate = with_layout(&ctx, E::Pred, &[2, 3], &[0, 1]);
    assert_diagnostic(
        inference::infer_select(&ctx, custom_predicate, nondefault_layout, wider),
        |diagnostic| matches!(diagnostic, Diagnostic::Inference { .. }),
    );
    let custom_select =
        inference::infer_select(&ctx, custom_predicate, nondefault_layout, nondefault_layout)
            .unwrap();
    assert_eq!(
        Shape::layout(&ctx, custom_select)
            .unwrap()
            .unwrap()
            .minor_to_major,
        [0, 1],
        "select retains the non-scalar predicate layout"
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
    let six_bit_input = array(&ctx, E::F6E3M2FN, &[2]);
    assert_diagnostic(
        inference::infer_stochastic_convert(&ctx, six_bit_input, array(&ctx, E::U8, &[2]), E::F16),
        |diagnostic| matches!(diagnostic, Diagnostic::Inference { .. }),
    );
    assert!(
        inference::infer_stochastic_convert(
            &ctx,
            array(&ctx, E::BF16, &[2]),
            array(&ctx, E::U16, &[2]),
            E::F16,
        )
        .is_ok()
    );
    let packed = array(&ctx, E::S32, &[2]);
    let unpacked = inference::infer_bitcast_convert(&ctx, packed, E::S8).unwrap();
    assert_eq!(tensor(&ctx, unpacked).unwrap().dimensions, [2, 4]);
    assert!(inference::infer_bitcast_convert(&ctx, array(&ctx, E::F32, &[3]), E::F64).is_err());
}

#[test]
fn select_and_clamp_dynamic_shapes_match_pinned_openxla_cases() {
    let ctx = Context::new();
    let pred_scalar = array(&ctx, E::Pred, &[]);
    let pred_two = array(&ctx, E::Pred, &[2]);
    let pred_unbounded = Shape::dynamic_array(&ctx, E::Pred, vec![Dimension::Unbounded]).unwrap();
    let float_scalar = array(&ctx, E::F32, &[]);
    let float_two = array(&ctx, E::F32, &[2]);
    let float_bounded = Shape::dynamic_array(&ctx, E::F32, vec![Dimension::Bounded(2)]).unwrap();
    let float_unbounded = Shape::dynamic_array(&ctx, E::F32, vec![Dimension::Unbounded]).unwrap();

    let select_cases = [
        (
            pred_scalar,
            float_unbounded,
            float_two,
            Dimension::Static(2),
        ),
        (
            pred_scalar,
            float_unbounded,
            float_bounded,
            Dimension::Bounded(2),
        ),
        (pred_two, float_unbounded, float_two, Dimension::Static(2)),
        (
            pred_unbounded,
            float_two,
            float_unbounded,
            Dimension::Static(2),
        ),
        (
            pred_unbounded,
            float_bounded,
            float_unbounded,
            Dimension::Bounded(2),
        ),
    ];
    for (predicate, on_true, on_false, expected_dimension) in select_cases {
        let result = inference::infer_select(&ctx, predicate, on_true, on_false).unwrap();
        assert_eq!(
            Shape::dimension(&ctx, result, 0).unwrap(),
            expected_dimension
        );
    }

    let clamp_cases = [
        (
            float_scalar,
            float_unbounded,
            float_scalar,
            Dimension::Unbounded,
        ),
        (
            float_scalar,
            float_unbounded,
            float_two,
            Dimension::Unbounded,
        ),
        (
            float_scalar,
            float_unbounded,
            float_bounded,
            Dimension::Unbounded,
        ),
        (float_two, float_unbounded, float_two, Dimension::Unbounded),
        (float_unbounded, float_two, float_two, Dimension::Static(2)),
        (
            float_unbounded,
            float_bounded,
            float_unbounded,
            Dimension::Bounded(2),
        ),
        (
            float_bounded,
            float_unbounded,
            float_bounded,
            Dimension::Unbounded,
        ),
        (
            float_unbounded,
            float_unbounded,
            float_unbounded,
            Dimension::Unbounded,
        ),
    ];
    for (minimum, operand, maximum, expected_dimension) in clamp_cases {
        let result = inference::infer_clamp(&ctx, minimum, operand, maximum).unwrap();
        assert_eq!(
            Shape::dimension(&ctx, result, 0).unwrap(),
            expected_dimension
        );
    }
}

#[test]
fn every_native_elementwise_op_verifies_a_pinned_valid_operand_contract() {
    use xlahlo::ops::elementwise::*;

    let mut ctx = Context::new();
    xlahlo::register(&mut ctx);
    let f32_type = array(&ctx, E::F32, &[2]);
    let s32_type = array(&ctx, E::S32, &[2]);
    let pred_type = array(&ctx, E::Pred, &[2]);
    let c64_type = array(&ctx, E::C64, &[2]);
    let f16_type = array(&ctx, E::F16, &[2]);
    let u32_type = array(&ctx, E::U32, &[2]);
    let f32_parameter = ParameterOp::new(&mut ctx, f32_type, vec![], xlahlo::attributes::Index(0));
    let s32_parameter = ParameterOp::new(&mut ctx, s32_type, vec![], xlahlo::attributes::Index(1));
    let pred_parameter =
        ParameterOp::new(&mut ctx, pred_type, vec![], xlahlo::attributes::Index(2));
    let u32_parameter = ParameterOp::new(&mut ctx, u32_type, vec![], xlahlo::attributes::Index(3));
    let f32_value = f32_parameter.result(&ctx);
    let s32_value = s32_parameter.result(&ctx);
    let pred_value = pred_parameter.result(&ctx);
    let u32_value = u32_parameter.result(&ctx);

    macro_rules! unary {
        ($op:ty, $result:expr, $input:expr) => {{
            let operation = <$op>::new(&mut ctx, $result, vec![$input]);
            assert!(operation.verify(&ctx).is_ok(), stringify!($op));
            for accuracy in [
                ResultAccuracy::Default,
                ResultAccuracy::Highest,
                ResultAccuracy::tolerance(0.0, 0.0, 0).unwrap(),
            ] {
                operation
                    .get_operation()
                    .deref_mut(&ctx)
                    .attributes
                    .set(Identifier::new("hlo_result_accuracy"), accuracy);
                assert_eq!(
                    operation.verify(&ctx).is_ok(),
                    accuracy == ResultAccuracy::Default || <$op>::OPCODE.supports_result_accuracy(),
                    "{} accuracy {accuracy:?}",
                    stringify!($op),
                );
            }
        }};
    }
    unary!(AbsOp, f32_type, f32_value);
    unary!(AcosOp, f32_type, f32_value);
    unary!(AcoshOp, f32_type, f32_value);
    unary!(AsinOp, f32_type, f32_value);
    unary!(AsinhOp, f32_type, f32_value);
    unary!(AtanhOp, f32_type, f32_value);
    unary!(CbrtOp, f32_type, f32_value);
    unary!(CeilOp, f32_type, f32_value);
    unary!(ClzOp, s32_type, s32_value);
    unary!(CosOp, f32_type, f32_value);
    unary!(CoshOp, f32_type, f32_value);
    unary!(ErfOp, f32_type, f32_value);
    unary!(ExpOp, f32_type, f32_value);
    unary!(Exp2Op, f32_type, f32_value);
    unary!(Expm1Op, f32_type, f32_value);
    unary!(FloorOp, f32_type, f32_value);
    unary!(ImagOp, f32_type, f32_value);
    unary!(IsFiniteOp, pred_type, f32_value);
    unary!(LogOp, f32_type, f32_value);
    unary!(Log1pOp, f32_type, f32_value);
    unary!(Log2Op, f32_type, f32_value);
    unary!(LogisticOp, f32_type, f32_value);
    unary!(NegateOp, s32_type, s32_value);
    unary!(NotOp, s32_type, s32_value);
    unary!(PopulationCountOp, s32_type, s32_value);
    unary!(RealOp, f32_type, f32_value);
    unary!(RoundNearestAfzOp, f32_type, f32_value);
    unary!(RoundNearestEvenOp, f32_type, f32_value);
    unary!(RsqrtOp, f32_type, f32_value);
    unary!(SignOp, f32_type, f32_value);
    unary!(SinOp, f32_type, f32_value);
    unary!(SinhOp, f32_type, f32_value);
    unary!(SqrtOp, f32_type, f32_value);
    unary!(TanOp, f32_type, f32_value);
    unary!(TanhOp, f32_type, f32_value);

    macro_rules! binary {
        ($op:ty, $result:expr, $lhs:expr, $rhs:expr) => {{
            let operation = <$op>::new(&mut ctx, $result, vec![$lhs, $rhs]);
            assert!(operation.verify(&ctx).is_ok(), stringify!($op));
        }};
    }
    binary!(AddOp, f32_type, f32_value, f32_value);
    binary!(AndOp, s32_type, s32_value, s32_value);
    binary!(Atan2Op, f32_type, f32_value, f32_value);
    binary!(DivideOp, f32_type, f32_value, f32_value);
    binary!(MaximumOp, f32_type, f32_value, f32_value);
    binary!(MinimumOp, f32_type, f32_value, f32_value);
    binary!(MulhiOp, s32_type, s32_value, s32_value);
    binary!(MultiplyOp, f32_type, f32_value, f32_value);
    binary!(OrOp, s32_type, s32_value, s32_value);
    binary!(PowerOp, f32_type, f32_value, f32_value);
    binary!(RemainderOp, f32_type, f32_value, f32_value);
    binary!(ShiftLeftOp, s32_type, s32_value, s32_value);
    binary!(ShiftRightArithmeticOp, s32_type, s32_value, s32_value);
    binary!(ShiftRightLogicalOp, s32_type, s32_value, s32_value);
    binary!(SubtractOp, f32_type, f32_value, f32_value);
    binary!(XorOp, s32_type, s32_value, s32_value);

    assert!(
        CompareOp::new(
            &mut ctx,
            pred_type,
            vec![f32_value, f32_value],
            Direction::Eq,
            ComparisonOrder::Partial,
        )
        .verify(&ctx)
        .is_ok()
    );
    assert!(
        ComplexOp::new(&mut ctx, c64_type, vec![f32_value, f32_value])
            .verify(&ctx)
            .is_ok()
    );
    assert!(
        ClampOp::new(&mut ctx, f32_type, vec![f32_value, f32_value, f32_value])
            .verify(&ctx)
            .is_ok()
    );
    assert!(
        SelectOp::new(&mut ctx, f32_type, vec![pred_value, f32_value, f32_value])
            .verify(&ctx)
            .is_ok()
    );
    assert!(
        ConvertOp::new(&mut ctx, f16_type, vec![f32_value])
            .verify(&ctx)
            .is_ok()
    );
    assert!(
        BitcastConvertOp::new(&mut ctx, f32_type, vec![f32_value])
            .verify(&ctx)
            .is_ok()
    );
    assert!(
        StochasticConvertOp::new(&mut ctx, f16_type, vec![f32_value, u32_value])
            .verify(&ctx)
            .is_ok()
    );
    assert!(
        ReducePrecisionOp::new(
            &mut ctx,
            f32_type,
            vec![f32_value],
            ReducePrecision {
                exponent_bits: 8,
                mantissa_bits: 10,
            },
        )
        .verify(&ctx)
        .is_ok()
    );
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
    assert_diagnostic(invalid_accuracy_exp.verify(&ctx), |diagnostic| {
        matches!(
            diagnostic,
            Diagnostic::InvalidAttribute {
                attribute: "result_accuracy",
                ..
            }
        )
    });
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
    let default_accuracy = AddOp::new(&mut ctx, f32s, vec![value, value]);
    default_accuracy
        .get_operation()
        .deref_mut(&ctx)
        .attributes
        .set(
            Identifier::new("hlo_result_accuracy"),
            ResultAccuracy::Default,
        );
    default_accuracy.verify(&ctx).unwrap();
}

#[test]
fn result_accuracy_defaults_and_unary_restrictions_match_pinned_verifier() {
    let mut ctx = Context::new();
    xlahlo::register(&mut ctx);
    let ty = array(&ctx, E::F32, &[2]);
    let parameter = ParameterOp::new(&mut ctx, ty, vec![], xlahlo::attributes::Index(0));
    let value = parameter.result(&ctx);
    for accuracy in [
        ResultAccuracy::Default,
        ResultAccuracy::Highest,
        ResultAccuracy::tolerance(0.0, -0.0, 0).unwrap(),
        ResultAccuracy::tolerance(0.01, 0.02, 3).unwrap(),
    ] {
        let abs = AbsOp::new(&mut ctx, ty, vec![value]);
        let convert = ConvertOp::new(&mut ctx, ty, vec![value]);
        let copy = CopyOp::new(&mut ctx, ty, vec![value]);
        let domain = DomainOp::new(&mut ctx, ty, vec![value]);
        let reduce_precision = ReducePrecisionOp::new(
            &mut ctx,
            ty,
            vec![value],
            ReducePrecision {
                exponent_bits: 8,
                mantissa_bits: 10,
            },
        );
        let exp = ExpOp::new(&mut ctx, ty, vec![value]);
        let add = AddOp::new(&mut ctx, ty, vec![value, value]);
        let bitcast = BitcastConvertOp::new(&mut ctx, ty, vec![value]);
        let restricted = [
            abs.get_operation(),
            convert.get_operation(),
            copy.get_operation(),
            domain.get_operation(),
            reduce_precision.get_operation(),
        ];
        let unrestricted = [
            exp.get_operation(),
            add.get_operation(),
            bitcast.get_operation(),
        ];
        for operation in restricted.into_iter().chain(unrestricted) {
            operation
                .deref_mut(&ctx)
                .attributes
                .set(Identifier::new("hlo_result_accuracy"), accuracy);
        }
        for operation in restricted {
            assert_eq!(
                pliron::operation::Operation::get_op_dyn(operation, &ctx)
                    .verify(&ctx)
                    .is_ok(),
                accuracy == ResultAccuracy::Default,
                "unsupported unary accuracy: {accuracy:?}",
            );
        }
        for operation in unrestricted {
            pliron::operation::Operation::get_op_dyn(operation, &ctx)
                .verify(&ctx)
                .unwrap();
        }
        for operation in restricted.into_iter().chain(unrestricted) {
            for invalid in [
                ResultAccuracy::Tolerance {
                    atol_bits: f64::NAN.to_bits(),
                    rtol_bits: 0.0_f64.to_bits(),
                    ulps: 0,
                },
                ResultAccuracy::Tolerance {
                    atol_bits: 0.0_f64.to_bits(),
                    rtol_bits: (-1.0_f64).to_bits(),
                    ulps: 0,
                },
                ResultAccuracy::Tolerance {
                    atol_bits: 0.0_f64.to_bits(),
                    rtol_bits: 0.0_f64.to_bits(),
                    ulps: -1,
                },
            ] {
                operation
                    .deref_mut(&ctx)
                    .attributes
                    .set(Identifier::new("hlo_result_accuracy"), invalid);
                assert_diagnostic(
                    pliron::operation::Operation::get_op_dyn(operation, &ctx).verify(&ctx),
                    |diagnostic| {
                        matches!(
                            diagnostic,
                            Diagnostic::InvalidAttribute {
                                attribute: "result_accuracy",
                                ..
                            }
                        )
                    },
                );
            }
            operation.deref_mut(&ctx).attributes.set(
                Identifier::new("hlo_result_accuracy"),
                pliron::builtin::attributes::StringAttr::new("default".into()),
            );
            assert_diagnostic(
                pliron::operation::Operation::get_op_dyn(operation, &ctx).verify(&ctx),
                |diagnostic| {
                    matches!(
                        diagnostic,
                        Diagnostic::InvalidAttribute {
                            attribute: "hlo_result_accuracy",
                            ..
                        }
                    )
                },
            );
        }
    }
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
