use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use stablehlo::text::{
    TextErrorKind, parse_attribute, parse_module, parse_type, print_attribute, print_module,
    print_type,
};
use stablehlo::{
    Attribute, ElementType, QuantizedElementType, QuantizedTensorType, Type, TypeError,
};

fn element(axis: Option<usize>, scales: Vec<f64>, zeros: Vec<i128>) -> QuantizedElementType {
    QuantizedElementType::new(
        ElementType::I8,
        None,
        None,
        ElementType::F32,
        axis,
        scales,
        zeros,
    )
}

fn hash(value: &impl Hash) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

#[test]
fn standalone_uniform_elements_round_trip_independently_of_tensor_shape() {
    for source in [
        "!quant.uniform<i8:f32, 0.25>",
        "!quant.uniform<ui8:f64, 0.125:255>",
        "!quant.uniform<i8<-127:127>:f32, 0.5:-127>",
        "!quant.uniform<i8:f32:7, {0.25, 0.5:1}>",
        "!quant.uniform<i4:f16:0, {1.0:-8, 2.0:7}>",
        "!quant.uniform<i8:f32, 0x3FD0000000000000:0>",
    ] {
        let ty = parse_type(source).unwrap_or_else(|error| panic!("{source}: {error}"));
        assert!(matches!(&ty, Type::QuantizedElement(_)));
        assert!(!ty.is_value_type());
        let printed = print_type(&ty).unwrap();
        assert_eq!(parse_type(&printed).unwrap(), ty, "{printed}");
        let attribute = Attribute::Type(Box::new(ty));
        assert_eq!(
            parse_attribute(&print_attribute(&attribute).unwrap()).unwrap(),
            attribute
        );
    }
}

#[test]
fn tensor_and_standalone_elements_use_one_metadata_value_and_verifier() {
    let metadata = element(Some(1), vec![0.25, 0.5], vec![-1, 1]);
    metadata.verify().unwrap();
    let tensor = QuantizedTensorType::from_element(vec![Some(3), Some(2)], metadata.clone());
    tensor.verify().unwrap();
    assert_eq!(tensor.element(), &metadata);
    assert_eq!(tensor.storage_type(), metadata.storage_type());
    assert_eq!(tensor.storage_range(), metadata.storage_range());
    assert_eq!(tensor.expressed_type(), metadata.expressed_type());
    assert_eq!(
        tensor.quantization_dimension(),
        metadata.quantization_dimension()
    );
    assert!(tensor.scales().eq(metadata.scales()));
    assert_eq!(tensor.zero_points(), metadata.zero_points());
    let old_constructor = QuantizedTensorType::new(
        vec![Some(3), Some(2)],
        ElementType::I8,
        None,
        None,
        ElementType::F32,
        Some(1),
        vec![0.25, 0.5],
        vec![-1, 1],
    );
    assert_eq!(tensor, old_constructor);
    assert_eq!(hash(&tensor), hash(&old_constructor));
    let printed = print_type(&Type::QuantizedTensor(tensor.clone())).unwrap();
    assert_eq!(parse_type(&printed).unwrap(), Type::QuantizedTensor(tensor));
}

#[test]
fn per_axis_element_constraints_are_separate_from_tensor_axis_constraints() {
    let metadata = element(Some(2), vec![0.25, 0.5], vec![0, 1]);
    metadata.verify().unwrap();
    assert_eq!(
        QuantizedTensorType::from_element(vec![Some(2)], metadata.clone()).verify(),
        Err(TypeError::QuantizationDimensionOutOfRange {
            dimension: 2,
            rank: 1
        }),
    );
    assert_eq!(
        QuantizedTensorType::from_element(vec![Some(3), Some(3), Some(3)], metadata.clone())
            .verify(),
        Err(TypeError::QuantizationAxisSizeMismatch {
            dimension: 2,
            axis_size: Some(3),
            parameter_count: 2,
        }),
    );
    QuantizedTensorType::from_element(vec![Some(3), Some(3), None], metadata.clone())
        .verify()
        .unwrap();
    QuantizedTensorType::from_element(vec![Some(3), Some(3), Some(2)], metadata)
        .verify()
        .unwrap();
}

#[test]
fn invalid_metadata_is_rejected_by_elements_tensors_and_text_printers() {
    let mut invalid = vec![
        element(None, vec![], vec![]),
        element(None, vec![0.25], vec![]),
        element(None, vec![0.25, 0.5], vec![0, 1]),
        element(None, vec![0.0], vec![0]),
        element(None, vec![-0.0], vec![0]),
        element(None, vec![-0.25], vec![0]),
        element(None, vec![f64::INFINITY], vec![0]),
        element(None, vec![f64::NAN], vec![0]),
        element(None, vec![0.25], vec![128]),
    ];
    for (storage, min, max, expressed) in [
        (ElementType::F32, None, None, ElementType::F32),
        (ElementType::I8, None, None, ElementType::I32),
        (ElementType::I8, Some(-129), None, ElementType::F32),
        (ElementType::I8, None, Some(128), ElementType::F32),
        (ElementType::I8, Some(2), Some(2), ElementType::F32),
        (ElementType::I8, Some(2), Some(1), ElementType::F32),
    ] {
        invalid.push(QuantizedElementType::new(
            storage,
            min,
            max,
            expressed,
            None,
            vec![1.0],
            vec![0],
        ));
    }
    for metadata in invalid {
        let expected = metadata.verify().unwrap_err();
        let tensor = QuantizedTensorType::from_element(vec![Some(1)], metadata.clone());
        assert_eq!(tensor.verify(), Err(expected.clone()));
        assert_eq!(
            Type::QuantizedElement(metadata.clone()).verify(),
            Err(expected.clone())
        );
        assert_eq!(
            print_type(&Type::QuantizedElement(metadata))
                .unwrap_err()
                .kind,
            TextErrorKind::InvalidType(expected),
        );
    }
}

#[test]
fn standalone_elements_cannot_be_used_as_stablehlo_values_or_wrong_components() {
    for source in [
        "tuple<!quant.uniform<i8:f32, 0.5>>",
        "!stablehlo.future<!quant.uniform<i8:f32, 0.5>>",
        "(!quant.uniform<i8:f32, 0.5>) -> ()",
        "memref<2x!quant.uniform<i8:f32, 0.5>>",
        "complex<!quant.uniform<i8:f32, 0.5>>",
        "tensor<2xtensor<f32>>",
        "tensor<2x!stablehlo.token>",
        "tensor<!quant.uniform<i8:f32:0, {0.5}>>",
        "tensor<2x!quant.uniform<i8:f32:0, {0.5}>>",
    ] {
        assert!(parse_type(source).is_err(), "{source}");
    }
    assert!(
        parse_module(
            "!q = !quant.uniform<i8:f32, 0.5> module { func.func @bad(%x: !q) { return } }"
        )
        .is_err()
    );
}

#[test]
fn quantized_element_aliases_resolve_through_chains_and_keep_bounds_and_literals() {
    let source = r#"
      !i = i8
      !f = f32
      !per_tensor = !quant.uniform<!i:!f, 0.25:1>
      !per_axis = !quant.uniform<i8<-127:127>:f32:0, {0.25, 0.5:1}>
      !alias = !per_axis
      #bounds = #stablehlo.bounds<5>
      !bounded = tensor<?x!alias, #bounds>
      #metadata = {consumer.quantization = [!per_tensor, !alias]}
      module attributes #metadata {
        func.func @main(%x: !bounded) -> !bounded {
          return %x : !bounded
        }
        func.func @constant() -> tensor<2x!per_tensor> {
          %c = "stablehlo.constant"() {value = dense<[1, 2]> : tensor<2xi8>}
            : () -> tensor<2x!per_tensor>
          return %c : tensor<2x!per_tensor>
        }
      }
    "#;
    let module = parse_module(source).unwrap();
    let printed = print_module(&module).unwrap();
    assert_eq!(parse_module(&printed).unwrap(), module);
    let Type::QuantizedTensor(tensor) = module.computations()[0].instructions()[0].ty().unwrap()
    else {
        panic!("expected quantized tensor");
    };
    assert_eq!(
        tensor.element(),
        &QuantizedElementType::new(
            ElementType::I8,
            Some(-127),
            Some(127),
            ElementType::F32,
            Some(0),
            vec![0.25, 0.5],
            vec![0, 1],
        )
    );
}

#[test]
fn quantized_element_aliases_do_not_bypass_rank_checks_or_expansion_limits() {
    for source in [
        "!q = !quant.uniform<i8:f32:2, {0.5}> module { func.func private @f(tensor<1x!q>) }",
        "!q = !quant.uniform<i8:f32:0, {0.5}> module { func.func private @f(tensor<2x!q>) }",
        "!q = !quant.uniform<i8:f32, 0.5> module { func.func private @f(memref<2x!q>) }",
        "!q = !quant.uniform<i8:f32, 0.5> module { func.func private @f(tensor<2xcomplex<!q>>) }",
    ] {
        assert!(parse_module(source).is_err(), "{source}");
    }
    let mut chain = String::from("!q0 = !quant.uniform<i8:f32, 0.5>\n");
    for index in 1..140 {
        chain.push_str(&format!("!q{index} = !q{}\n", index - 1));
    }
    chain.push_str("module {}");
    assert_eq!(
        parse_module(&chain).unwrap_err().kind,
        TextErrorKind::NestingLimit
    );
}

#[test]
fn metadata_equality_and_hashing_preserve_scale_bits() {
    let a = element(None, vec![0.5], vec![0]);
    let b = element(None, vec![f64::from_bits(0.5_f64.to_bits() + 1)], vec![0]);
    assert_ne!(a, b);
    assert_eq!(hash(&a), hash(&a.clone()));
    let a = element(None, vec![f64::from_bits(0x7ff8_0000_0000_0001)], vec![0]);
    let b = element(None, vec![f64::from_bits(0x7ff8_0000_0000_0002)], vec![0]);
    assert_ne!(a, b);
    assert_eq!(a, a.clone());
}
