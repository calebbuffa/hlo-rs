use stablehlo::text::{parse_module, parse_type};
use stablehlo::{
    Attribute, ElementType, InferenceError, TensorType, Type, infer_abs_type, infer_complex_type,
    infer_imag_type, infer_is_finite_type, infer_real_type,
};

fn ty(source: &str) -> Type {
    parse_type(source).unwrap()
}

#[test]
fn abs_preserves_real_and_quantized_metadata_and_unwraps_complex_elements() {
    for source in [
        "tensor<2xi2>",
        "tensor<i4>",
        "tensor<?xi64, #stablehlo.bounds<5>>",
        "tensor<?xf32, #stablehlo.bounds<7>>",
        "tensor<!quant.uniform<i8:f32, 0.25:1>>",
        "tensor<?x!quant.uniform<i8:f32, 0.5>, #stablehlo.bounds<8>>",
    ] {
        let input = ty(source);
        assert_eq!(infer_abs_type(&input).unwrap(), input);
    }
    for (input, expected) in [
        ("tensor<complex<f32>>", "tensor<f32>"),
        ("tensor<0xcomplex<f64>>", "tensor<0xf64>"),
        (
            "tensor<?x2xcomplex<f32>, #stablehlo.bounds<5, ?>>",
            "tensor<?x2xf32, #stablehlo.bounds<5, ?>>",
        ),
    ] {
        assert_eq!(infer_abs_type(&ty(input)).unwrap(), ty(expected));
    }
}

#[test]
fn real_and_imag_use_the_same_precision_shape_and_encoding_rules() {
    for (input, expected) in [
        ("tensor<complex<f32>>", "tensor<f32>"),
        ("tensor<2xcomplex<f64>>", "tensor<2xf64>"),
        (
            "tensor<?xcomplex<f64>, #stablehlo.bounds<8>>",
            "tensor<?xf64, #stablehlo.bounds<8>>",
        ),
        ("tensor<2xbf16>", "tensor<2xbf16>"),
        ("tensor<2xf8E4M3FN>", "tensor<2xf8E4M3FN>"),
    ] {
        let expected = ty(expected);
        assert_eq!(infer_real_type(&ty(input)).unwrap(), expected);
        assert_eq!(infer_imag_type(&ty(input)).unwrap(), expected);
    }
    let input: Type = TensorType::new(ElementType::C64, vec![None])
        .with_encoding(Attribute::String("consumer encoding".into()))
        .into();
    let expected: Type = TensorType::new(ElementType::F32, vec![None])
        .with_encoding(Attribute::String("consumer encoding".into()))
        .into();
    assert_eq!(infer_real_type(&input).unwrap(), expected);
    assert_eq!(infer_imag_type(&input).unwrap(), expected);
}

#[test]
fn finite_predicate_inference_preserves_quantized_input_bounds() {
    for (input, expected) in [
        ("tensor<f32>", "tensor<i1>"),
        ("tensor<0xf16>", "tensor<0xi1>"),
        (
            "tensor<?xf32, #stablehlo.bounds<3>>",
            "tensor<?xi1, #stablehlo.bounds<3>>",
        ),
        (
            "tensor<?x!quant.uniform<i8:f32, 0.25>, #stablehlo.bounds<3>>",
            "tensor<?xi1, #stablehlo.bounds<3>>",
        ),
    ] {
        assert_eq!(infer_is_finite_type(&ty(input)).unwrap(), ty(expected));
    }
    assert!(
        parse_module(
            r#"
        module { func.func @main(%x: tensor<?x!quant.uniform<i8:f32, 0.5>, #stablehlo.bounds<3>>)
          -> tensor<4xi1> {
          %r = "stablehlo.is_finite"(%x) :
            (tensor<?x!quant.uniform<i8:f32, 0.5>, #stablehlo.bounds<3>>) -> tensor<4xi1>
          return %r : tensor<4xi1>
        } }
    "#
        )
        .is_err()
    );
    parse_module(
        r#"
        module { func.func @main(%x: tensor<?x!quant.uniform<i8:f32, 0.5>, #stablehlo.bounds<3>>)
          -> tensor<3xi1> {
          %r = "stablehlo.is_finite"(%x) :
            (tensor<?x!quant.uniform<i8:f32, 0.5>, #stablehlo.bounds<3>>) -> tensor<3xi1>
          return %r : tensor<3xi1>
        } }
    "#,
    )
    .unwrap();
}

#[test]
fn complex_inference_preserves_lhs_without_refining_from_rhs() {
    let lhs = ty("tensor<?x2xf32, #stablehlo.bounds<5, ?>>");
    let rhs = ty("tensor<3x2xf32>");
    assert_eq!(
        infer_complex_type(&lhs, &rhs).unwrap(),
        ty("tensor<?x2xcomplex<f32>, #stablehlo.bounds<5, ?>>")
    );
    assert_eq!(
        infer_complex_type(&rhs, &lhs).unwrap(),
        ty("tensor<3x2xcomplex<f32>>")
    );
    assert_eq!(
        infer_complex_type(&ty("tensor<f64>"), &ty("tensor<f64>")).unwrap(),
        ty("tensor<complex<f64>>")
    );
    for rhs in [
        "tensor<6x2xf32>",
        "tensor<3x2xf64>",
        "tensor<3xf32>",
        "tensor<3x3xf32>",
    ] {
        assert!(infer_complex_type(&lhs, &ty(rhs)).is_err(), "{rhs}");
    }
}

#[test]
fn complex_verification_checks_rhs_against_declared_result_too() {
    for (rhs, result, accepted) in [
        ("tensor<3xf32>", "tensor<3xcomplex<f32>>", true),
        ("tensor<3xf32>", "tensor<4xcomplex<f32>>", false),
        (
            "tensor<4xf32>",
            "tensor<?xcomplex<f32>, #stablehlo.bounds<3>>",
            false,
        ),
    ] {
        let source = format!(
            r#"
          module {{ func.func @main(%a: tensor<?xf32>, %b: {rhs}) -> {result} {{
            %r = "stablehlo.complex"(%a, %b) : (tensor<?xf32>, {rhs}) -> {result}
            return %r : {result}
          }} }}
        "#
        );
        assert_eq!(parse_module(&source).is_ok(), accepted, "{source}");
    }
}

#[test]
fn public_helpers_reuse_operation_schema_domains_and_reject_invalid_types() {
    for source in ["tensor<ui8>", "tensor<i1>", "tensor<si8>", "tensor<tf32>"] {
        assert!(infer_abs_type(&ty(source)).is_err(), "{source}");
    }
    for source in [
        "tensor<i32>",
        "tensor<tf32>",
        "tensor<!quant.uniform<i8:f32, 0.5>>",
    ] {
        assert!(infer_real_type(&ty(source)).is_err(), "{source}");
        assert!(infer_imag_type(&ty(source)).is_err(), "{source}");
    }
    for source in [
        "tensor<i1>",
        "tensor<i32>",
        "tensor<complex<f32>>",
        "tensor<tf32>",
    ] {
        assert!(infer_is_finite_type(&ty(source)).is_err(), "{source}");
    }
    for invalid in [
        Type::Token,
        Type::Tuple(vec![]),
        ty("!quant.uniform<i8:f32, 0.25>"),
        ty("tensor<2x!quant.uniform<i8:f32:0, {0.25, 0.5}>>"),
    ] {
        assert!(infer_abs_type(&invalid).is_err());
        assert!(infer_real_type(&invalid).is_err());
        assert!(infer_imag_type(&invalid).is_err());
        assert!(infer_is_finite_type(&invalid).is_err());
        assert!(infer_complex_type(&invalid, &invalid).is_err());
    }
    let invalid: Type = TensorType::new(ElementType::F32, vec![Some(u64::MAX)]).into();
    assert!(matches!(
        infer_abs_type(&invalid),
        Err(InferenceError::InvalidType { index: 0, .. })
    ));
}
