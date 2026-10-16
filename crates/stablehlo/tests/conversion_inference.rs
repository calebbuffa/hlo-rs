use stablehlo::text::{parse_module, parse_type, print_module};
use stablehlo::{ElementType as E, TensorType, Type, infer_convert_type};

fn ty(source: &str) -> Type {
    parse_type(source).unwrap()
}

fn module(input: &str, output: &str) -> bool {
    let source = format!(
        "func.func @convert(%x:{input})->{output} {{ \
         %r=\"stablehlo.convert\"(%x):({input})->{output} \
         return %r:{output} }}"
    );
    let Ok(module) = parse_module(&source) else {
        return false;
    };
    assert_eq!(
        parse_module(&print_module(&module).unwrap()).unwrap(),
        module
    );
    true
}

#[test]
fn conversion_retains_shape_bounds_and_encoding_without_refinement() {
    for (input, element, expected) in [
        ("tensor<f32>", E::I1, "tensor<i1>"),
        ("tensor<0x3xi8>", E::C128, "tensor<0x3xcomplex<f64>>"),
        (
            "tensor<?x2xf32,#stablehlo.bounds<9,?>>",
            E::F16,
            "tensor<?x2xf16,#stablehlo.bounds<9,?>>",
        ),
        (
            "tensor<4xf32,#sparse_tensor.encoding<{map=(i)->(i:compressed)}>>",
            E::F64,
            "tensor<4xf64,#sparse_tensor.encoding<{map=(i)->(i:compressed)}>>",
        ),
    ] {
        let result = infer_convert_type(&ty(input), element).unwrap();
        assert_eq!(result, ty(expected));
        result.verify().unwrap();
    }
    assert!(module("tensor<3xf32>", "tensor<?xi32>"));
    assert!(module(
        "tensor<?xf32,#stablehlo.bounds<9>>",
        "tensor<3xi32>"
    ));
}

#[test]
fn generated_conversion_matrix_covers_all_ordinary_stablehlo_element_pairs() {
    let elements = [
        E::I1,
        E::I2,
        E::I4,
        E::I8,
        E::I16,
        E::I32,
        E::I64,
        E::Ui2,
        E::Ui4,
        E::Ui8,
        E::Ui16,
        E::Ui32,
        E::Ui64,
        E::F4E2M1Fn,
        E::F6E2M3Fn,
        E::F6E3M2Fn,
        E::F16,
        E::Bf16,
        E::F32,
        E::F64,
        E::F8E3M4,
        E::F8E4M3,
        E::F8E4M3Fn,
        E::F8E4M3FnUz,
        E::F8E4M3B11FnUz,
        E::F8E5M2,
        E::F8E5M2FnUz,
        E::F8E8M0FnUz,
        E::C64,
        E::C128,
    ];
    for input in elements {
        let operand: Type = TensorType::new(input, vec![None, Some(0)]).into();
        for output in elements {
            let expected: Type = TensorType::new(output, vec![None, Some(0)]).into();
            assert_eq!(infer_convert_type(&operand, output).unwrap(), expected);
        }
    }
}

#[test]
fn conversion_rejects_non_hlo_elements_invalid_metadata_and_nontensor_inputs() {
    for input in [
        Type::Token,
        Type::Tuple(vec![]),
        ty("tensor<*xf32>"),
        ty("memref<4xf32>"),
        ty("tensor<4x!quant.uniform<i8:f32,0.5>>"),
        TensorType::new(E::F32, vec![Some(i64::MAX as u64 + 1)]).into(),
    ] {
        assert!(infer_convert_type(&input, E::F64).is_err(), "{input:?}");
    }
    for element in [
        E::Index,
        E::Tf32,
        E::Si2,
        E::Si4,
        E::Si8,
        E::Si16,
        E::Si32,
        E::Si64,
    ] {
        let input: Type = TensorType::scalar(element).into();
        assert!(infer_convert_type(&input, E::F32).is_err());
        assert!(infer_convert_type(&ty("tensor<f32>"), element).is_err());
    }
    for (input, output) in [
        ("tensor<3xf32>", "tensor<4xi32>"),
        ("tensor<?xf32,#stablehlo.bounds<3>>", "tensor<4xi32>"),
        ("tensor<4xf32>", "tensor<?xi32,#stablehlo.bounds<3>>"),
        ("tensor<4xf32>", "tensor<2x2xi32>"),
        ("tensor<4x!quant.uniform<i8:f32,0.5>>", "tensor<4xi32>"),
        ("tensor<4xf32>", "tensor<4xtf32>"),
    ] {
        assert!(!module(input, output), "{input} -> {output}");
    }
}

#[test]
fn retained_sparse_numeric_metadata_is_revalidated_but_declared_encodings_are_independent() {
    let input =
        "tensor<4xf32,#sparse_tensor.encoding<{map=(i)->(i:compressed),explicitVal=1.0:f32}>>";
    assert!(infer_convert_type(&ty(input), E::F64).is_err());
    assert!(infer_convert_type(&ty(input), E::I1).is_err());
    assert_eq!(infer_convert_type(&ty(input), E::F32).unwrap(), ty(input));
    // A declared dense result need not inherit the inference witness's encoding.
    assert!(module(input, "tensor<4xf64>"));
    assert!(module(input, "tensor<4xi1>"));
}
