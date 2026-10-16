use stablehlo::{
    Attribute, Attributes, BuildError, ElementType, Operation, StableHloAttribute, StableHloOp,
    TensorType, Type,
};

mod common;
use common::computation;

fn tensor(element: ElementType) -> Type {
    TensorType::static_shape(element, [2]).into()
}

#[test]
fn custom_call_api_version_uses_the_upstream_integer_enum_representation() {
    for (version, accepted) in [
        (-1, false),
        (0, true),
        (1, true),
        (2, true),
        (3, true),
        (4, true),
        (5, false),
        (i64::MAX, false),
    ] {
        assert_eq!(
            computation(
                StableHloOp::CustomCall,
                vec![],
                vec![],
                [
                    (
                        "call_target_name".into(),
                        Attribute::String("target".into())
                    ),
                    (
                        "api_version".into(),
                        Attribute::Integer {
                            ty: ElementType::I32,
                            value: i128::from(version)
                        }
                    ),
                ]
                .into()
            )
            .is_ok(),
            accepted,
            "version: {version}"
        );
    }
}
#[test]
fn unary_element_domains_follow_base_td_not_rust_numeric_categories() {
    use ElementType::*;
    let elements = [
        I1,
        I2,
        I4,
        I8,
        I16,
        I32,
        I64,
        Ui2,
        Ui4,
        Ui8,
        Ui16,
        Ui32,
        Ui64,
        Si2,
        Si4,
        Si8,
        Si16,
        Si32,
        Si64,
        Index,
        Tf32,
        F16,
        Bf16,
        F32,
        F64,
        F4E2M1Fn,
        F6E2M3Fn,
        F6E3M2Fn,
        F8E3M4,
        F8E4M3,
        F8E4M3Fn,
        F8E4M3FnUz,
        F8E4M3B11FnUz,
        F8E5M2,
        F8E5M2FnUz,
        F8E8M0FnUz,
        C64,
        C128,
    ];
    let signed = [I2, I4, I8, I16, I32, I64];
    let unsigned = [Ui2, Ui4, Ui8, Ui16, Ui32, Ui64];
    let operations = [
        StableHloOp::CountLeadingZeros,
        StableHloOp::PopulationCount,
        StableHloOp::Not,
        StableHloOp::Negate,
        StableHloOp::Sign,
        StableHloOp::Ceil,
        StableHloOp::Floor,
        StableHloOp::RoundNearestAfz,
        StableHloOp::RoundNearestEven,
        StableHloOp::Cbrt,
        StableHloOp::Cosine,
        StableHloOp::Exponential,
        StableHloOp::ExponentialMinusOne,
        StableHloOp::Log,
        StableHloOp::LogPlusOne,
        StableHloOp::Logistic,
        StableHloOp::Rsqrt,
        StableHloOp::Sine,
        StableHloOp::Sqrt,
        StableHloOp::Tan,
        StableHloOp::Tanh,
    ];
    for opcode in operations {
        for element in elements {
            let integer = signed.contains(&element) || unsigned.contains(&element);
            let float = element.is_float() && element != Tf32;
            let complex = matches!(element, C64 | C128);
            let expected = match opcode {
                StableHloOp::CountLeadingZeros | StableHloOp::PopulationCount => integer,
                StableHloOp::Not => integer || element == I1,
                StableHloOp::Negate => integer || float || complex,
                StableHloOp::Sign => signed.contains(&element) || float || complex,
                StableHloOp::Ceil
                | StableHloOp::Floor
                | StableHloOp::RoundNearestAfz
                | StableHloOp::RoundNearestEven => float,
                _ => float || complex,
            };
            let result = computation(
                opcode,
                vec![tensor(element)],
                vec![tensor(element)],
                Attributes::new(),
            );
            assert_eq!(
                result.is_ok(),
                expected,
                "{opcode:?} {element:?}: {result:?}"
            );
        }
    }
}

#[test]
fn bitwise_operations_require_matching_integer_or_predicate_types() {
    for opcode in [StableHloOp::And, StableHloOp::Or, StableHloOp::Xor] {
        for element in [ElementType::I1, ElementType::I8, ElementType::Ui64] {
            assert!(
                computation(
                    opcode,
                    vec![tensor(element), tensor(element)],
                    vec![tensor(element)],
                    Attributes::new(),
                )
                .is_ok()
            );
        }
        for element in [
            ElementType::F32,
            ElementType::C64,
            ElementType::Index,
            ElementType::Si32,
        ] {
            assert!(
                computation(
                    opcode,
                    vec![tensor(element), tensor(element)],
                    vec![tensor(element)],
                    Attributes::new(),
                )
                .is_err(),
                "{opcode:?} accepted {element:?}"
            );
        }
        assert!(matches!(
            computation(
                opcode,
                vec![tensor(ElementType::I8), tensor(ElementType::I32)],
                vec![tensor(ElementType::I8)],
                Attributes::new()
            ),
            Err(BuildError::IncompatibleOperands { .. })
        ));
        assert!(matches!(
            computation(
                opcode,
                vec![tensor(ElementType::I8), tensor(ElementType::I8)],
                vec![tensor(ElementType::I32)],
                Attributes::new()
            ),
            Err(BuildError::InvalidResultType { .. })
        ));
    }
}

#[test]
fn shifts_and_transcendentals_reject_otherwise_shape_compatible_elements() {
    for opcode in [
        StableHloOp::ShiftLeft,
        StableHloOp::ShiftRightArithmetic,
        StableHloOp::ShiftRightLogical,
        StableHloOp::Atan2,
    ] {
        for element in [ElementType::I1, ElementType::Index, ElementType::Si32] {
            assert!(
                computation(
                    opcode,
                    vec![tensor(element), tensor(element)],
                    vec![tensor(element)],
                    Attributes::new(),
                )
                .is_err()
            );
        }
    }
    assert!(
        computation(
            StableHloOp::Atan2,
            vec![tensor(ElementType::I32), tensor(ElementType::I32)],
            vec![tensor(ElementType::I32)],
            Attributes::new(),
        )
        .is_err()
    );
}

#[test]
fn tensor_operations_do_not_accept_tokens_tuples_or_buffers() {
    for opcode in [
        StableHloOp::Add,
        StableHloOp::Negate,
        StableHloOp::Fft,
        StableHloOp::Dot,
    ] {
        for ty in [
            Type::Token,
            Type::Tuple(vec![tensor(ElementType::F32)]),
            Type::Buffer(stablehlo::BufferType::new(ElementType::F32, vec![Some(2)])),
        ] {
            let count = Operation::stablehlo(opcode, Attributes::new())
                .expected_arity()
                .unwrap();
            assert!(
                computation(opcode, vec![ty.clone(); count], vec![ty], Attributes::new()).is_err()
            );
        }
    }
}

#[test]
fn compare_requires_a_recognized_direction_and_dialect() {
    for attribute in [
        None,
        Some(Attribute::String("EQ".to_owned())),
        Some(Attribute::Enum {
            dialect: "stablehlo".to_owned(),
            name: "comparison_direction".to_owned(),
            value: "INVALID".to_owned(),
        }),
        Some(Attribute::Enum {
            dialect: "other".to_owned(),
            name: "comparison_direction".to_owned(),
            value: "EQ".to_owned(),
        }),
    ] {
        let attributes = attribute
            .map(|value| [("comparison_direction".to_owned(), value)].into())
            .unwrap_or_default();
        assert!(matches!(
            computation(
                StableHloOp::Compare,
                vec![tensor(ElementType::F32), tensor(ElementType::F32)],
                vec![tensor(ElementType::I1)],
                attributes
            ),
            Err(BuildError::MissingOrInvalidAttribute {
                attribute: "comparison_direction",
                ..
            })
        ));
    }
    for value in ["EQ", "NE", "GE", "GT", "LE", "LT"] {
        assert!(
            computation(
                StableHloOp::Compare,
                vec![tensor(ElementType::F32), tensor(ElementType::F32)],
                vec![tensor(ElementType::I1)],
                [(
                    "comparison_direction".to_owned(),
                    Attribute::Enum {
                        dialect: "stablehlo".to_owned(),
                        name: "comparison_direction".to_owned(),
                        value: value.to_owned(),
                    }
                )]
                .into(),
            )
            .is_ok()
        );
    }
}

#[test]
fn optional_attributes_are_not_silently_ignored_when_malformed() {
    assert!(matches!(
        computation(
            StableHloOp::Cosine,
            vec![tensor(ElementType::F32)],
            vec![tensor(ElementType::F32)],
            [(
                "result_accuracy".to_owned(),
                Attribute::String("DEFAULT".to_owned())
            )]
            .into()
        ),
        Err(BuildError::MissingOrInvalidAttribute {
            attribute: "result_accuracy",
            ..
        })
    ));
    assert!(matches!(
        computation(
            StableHloOp::Cholesky,
            vec![tensor(ElementType::F32)],
            vec![tensor(ElementType::F32)],
            [("lower".to_owned(), Attribute::integer(1))].into()
        ),
        Err(BuildError::MissingOrInvalidAttribute {
            attribute: "lower",
            ..
        })
    ));
}

#[test]
fn communication_operations_return_tokens_and_require_a_token_operand() {
    assert!(
        computation(
            StableHloOp::Outfeed,
            vec![tensor(ElementType::F32), Type::Token],
            vec![Type::Token],
            Attributes::new(),
        )
        .is_ok()
    );
    assert!(
        computation(
            StableHloOp::Outfeed,
            vec![tensor(ElementType::F32)],
            vec![Type::Token],
            Attributes::new(),
        )
        .is_err()
    );
    assert!(matches!(
        computation(
            StableHloOp::Outfeed,
            vec![Type::Token],
            vec![],
            Attributes::new()
        ),
        Err(BuildError::InvalidResultArity {
            expected: 1,
            actual: 0,
            ..
        })
    ));
    let channel = Attribute::StableHlo(Box::new(StableHloAttribute::ChannelHandle(
        stablehlo::ChannelHandle {
            handle: 1,
            channel_type: 1,
        },
    )));
    assert!(
        computation(
            StableHloOp::Send,
            vec![Type::Token],
            vec![Type::Token],
            [("channel_handle".to_owned(), channel)].into(),
        )
        .is_ok()
    );
}

#[test]
fn rng_has_three_operands_and_a_required_distribution() {
    let scalar = Type::from(TensorType::scalar(ElementType::F32));
    let shape = Type::from(TensorType::static_shape(ElementType::I64, [1]));
    let distribution = [(
        "rng_distribution".to_owned(),
        Attribute::Enum {
            dialect: "stablehlo".to_owned(),
            name: "rng_distribution".to_owned(),
            value: "UNIFORM".to_owned(),
        },
    )]
    .into();
    assert!(
        computation(
            StableHloOp::Rng,
            vec![scalar.clone(), scalar.clone(), shape],
            vec![tensor(ElementType::F32)],
            distribution,
        )
        .is_ok()
    );
    assert!(matches!(
        computation(
            StableHloOp::Rng,
            vec![scalar.clone(), scalar],
            vec![tensor(ElementType::F32)],
            Attributes::new()
        ),
        Err(BuildError::InvalidArity {
            expected: 3,
            actual: 2,
            ..
        })
    ));
}

#[test]
fn dimension_tensors_do_not_accept_explicitly_signed_mlir_integer_types() {
    assert!(
        computation(
            StableHloOp::DynamicReshape,
            vec![
                tensor(ElementType::F32),
                TensorType::static_shape(ElementType::Si64, [1]).into()
            ],
            vec![tensor(ElementType::F32)],
            Attributes::new(),
        )
        .is_err()
    );
}

#[test]
fn static_reshape_requires_static_or_single_bounded_result_dimensions() {
    let dynamic = Type::from(TensorType::new(ElementType::F32, vec![None]));
    assert!(
        computation(
            StableHloOp::Reshape,
            vec![tensor(ElementType::F32)],
            vec![dynamic],
            Attributes::new(),
        )
        .is_err()
    );
    let bounded = TensorType::new(ElementType::F32, vec![None]).with_encoding(
        Attribute::StableHlo(Box::new(StableHloAttribute::TypeExtensions(
            stablehlo::TypeExtensions { bounds: vec![4] },
        ))),
    );
    assert!(
        computation(
            StableHloOp::Reshape,
            vec![tensor(ElementType::F32)],
            vec![bounded.into()],
            Attributes::new(),
        )
        .is_ok()
    );
}
