use stablehlo::{
    Attribute, BufferLayout, BufferType, ElementType, FunctionType, QuantizedTensorType,
    StableHloAttribute, TensorType, Type, TypeError, TypeExtensions,
};

#[test]
fn element_storage_widths_include_packed_formats_and_target_dependent_index() {
    for (element, width) in [
        (ElementType::I1, 1),
        (ElementType::I2, 2),
        (ElementType::Ui4, 4),
        (ElementType::F4E2M1Fn, 4),
        (ElementType::F6E3M2Fn, 6),
        (ElementType::F8E4M3Fn, 8),
        (ElementType::Bf16, 16),
        (ElementType::F32, 32),
        (ElementType::Tf32, 19),
        (ElementType::F64, 64),
        (ElementType::C64, 64),
        (ElementType::C128, 128),
    ] {
        assert_eq!(element.bit_width(), Some(width), "{element:?}");
    }
    assert_eq!(ElementType::Index.bit_width(), None);
}
#[test]
fn quantized_per_tensor_type_obeys_spec_constraints() {
    let ty = QuantizedTensorType::new(
        vec![Some(2), Some(3)],
        ElementType::Si8,
        None,
        None,
        ElementType::F32,
        None,
        vec![0.5],
        vec![0],
    );
    assert_eq!(ty.verify(), Ok(()));
}

#[test]
fn quantized_per_axis_type_checks_dimension_and_parameter_count() {
    let ty = QuantizedTensorType::new(
        vec![Some(2), Some(3)],
        ElementType::Ui8,
        Some(0),
        Some(127),
        ElementType::F32,
        Some(1),
        vec![0.5, 0.25, 0.125],
        vec![0, 1, 2],
    );
    assert_eq!(ty.verify(), Ok(()));

    let invalid = QuantizedTensorType::new(
        vec![Some(2), Some(3)],
        ElementType::Ui8,
        None,
        None,
        ElementType::F32,
        Some(1),
        vec![0.5],
        vec![0],
    );
    assert_eq!(
        invalid.verify(),
        Err(TypeError::QuantizationAxisSizeMismatch {
            dimension: 1,
            axis_size: Some(3),
            parameter_count: 1,
        })
    );
}

#[test]
fn quantized_type_rejects_invalid_scales_and_supports_uint64_range() {
    let invalid = QuantizedTensorType::new(
        vec![Some(1)],
        ElementType::Ui64,
        None,
        None,
        ElementType::F32,
        None,
        vec![f64::NAN],
        vec![0],
    );
    assert_eq!(invalid.verify(), Err(TypeError::InvalidQuantizationScale));

    let valid = QuantizedTensorType::new(
        vec![Some(1)],
        ElementType::Ui64,
        None,
        None,
        ElementType::F32,
        None,
        vec![1.0],
        vec![u64::MAX as i128],
    );
    assert_eq!(valid.verify(), Ok(()));
}

#[test]
fn function_and_tuple_types_only_contain_value_types() {
    let tensor = Type::from(TensorType::static_shape(ElementType::F32, [2, 3]));
    let function = Type::Function(FunctionType::new(
        vec![tensor.clone()],
        vec![Type::Tuple(vec![tensor])],
    ));
    assert_eq!(function.verify(), Ok(()));

    let invalid_tuple = Type::Tuple(vec![Type::Element(ElementType::F32)]);
    assert_eq!(
        invalid_tuple.verify(),
        Err(TypeError::ExpectedValueType {
            context: "tuple element",
            index: 0,
        })
    );
}

#[test]
fn tensor_type_extensions_enforce_rank_and_dynamic_dimension_bounds() {
    let encoding = Attribute::StableHlo(Box::new(StableHloAttribute::TypeExtensions(
        TypeExtensions {
            bounds: vec![-1, 8],
        },
    )));
    let valid = Type::from(
        TensorType::new(ElementType::F32, vec![None, None]).with_encoding(encoding.clone()),
    );
    assert_eq!(valid.verify(), Ok(()));

    let wrong_rank =
        Type::from(TensorType::new(ElementType::F32, vec![None]).with_encoding(encoding.clone()));
    assert_eq!(
        wrong_rank.verify(),
        Err(TypeError::EncodingBoundsRankMismatch { bounds: 2, rank: 1 })
    );

    let static_encoding = Attribute::StableHlo(Box::new(StableHloAttribute::TypeExtensions(
        TypeExtensions { bounds: vec![5, 8] },
    )));
    let static_dimension = Type::from(
        TensorType::static_shape(ElementType::F32, [4, 8]).with_encoding(static_encoding),
    );
    assert_eq!(
        static_dimension.verify(),
        Err(TypeError::StaticDimensionHasBound {
            dimension: 0,
            bound: 5,
        })
    );
}

#[test]
fn type_extensions_accept_the_dynamic_sentinel_but_reject_lower_values() {
    let dynamic_bound = StableHloAttribute::TypeExtensions(TypeExtensions { bounds: vec![-1] });
    assert_eq!(dynamic_bound.verify(), Ok(()));

    let invalid_bound = StableHloAttribute::TypeExtensions(TypeExtensions { bounds: vec![-2] });
    assert_eq!(
        invalid_bound.verify(),
        Err(stablehlo::AttributeError::InvalidDimensionBound {
            index: 0,
            bound: -2,
        })
    );
}

#[test]
fn tensor_dimensions_fit_the_stablehlo_int64_shape_domain() {
    let oversized = Type::from(TensorType::static_shape(
        ElementType::F32,
        [i64::MAX as u64 + 1],
    ));
    assert_eq!(
        oversized.verify(),
        Err(TypeError::DimensionTooLarge {
            dimension: 0,
            size: i64::MAX as u64 + 1,
        })
    );
}

#[test]
fn strided_buffer_layout_rank_matches_the_buffer_rank() {
    let buffer = Type::Buffer(
        BufferType::new(ElementType::F32, vec![Some(2), Some(3)]).with_layout(
            BufferLayout::Strided {
                offset: Some(0),
                strides: vec![Some(3)],
            },
        ),
    );
    assert_eq!(
        buffer.verify(),
        Err(TypeError::BufferLayoutRankMismatch {
            strides: 1,
            rank: 2,
        })
    );
}
