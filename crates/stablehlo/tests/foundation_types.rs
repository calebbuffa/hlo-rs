use pliron::{common_traits::Verify, context::Context, r#type::TypeHandle};
use stablehlo::pliron::builtin::{
    attributes::{BoolAttr, VecAttr},
    types::Signedness as IntegerSignedness,
};
use stablehlo::{attributes::*, types::*};
fn verifies(ctx: &Context, ty: TypeHandle) -> bool {
    ty.deref(ctx).verify(ctx).is_ok()
}

#[test]
fn scalar_integer_widths_signedness_and_complex_components_are_canonical() {
    let ctx = Context::new();
    for signedness in [
        IntegerSignedness::Signless,
        IntegerSignedness::Signed,
        IntegerSignedness::Unsigned,
    ] {
        for width in [
            0,
            1,
            2,
            4,
            8,
            16,
            32,
            64,
            65,
            127,
            128,
            129,
            256,
            IntegerElement::MAX_WIDTH,
        ] {
            let element = ElementType::integer(width, signedness).unwrap();
            assert_eq!(element.integer_parts(), Some((width, signedness)));
            assert_eq!(element.bit_width(), Some(width));
            assert_eq!(primitive_element(&ctx, element.native(&ctx)), Some(element));
            let complex = ElementType::complex(element).unwrap();
            assert_eq!(complex.complex_component(), Some(element));
            assert!(verifies(&ctx, complex.native(&ctx)));
        }
        assert!(ElementType::integer(IntegerElement::MAX_WIDTH + 1, signedness).is_none());
    }
    assert!(ElementType::complex(ElementType::Index).is_none());
}

#[test]
fn every_named_scalar_format_round_trips_native_type_identity() {
    let ctx = Context::new();
    use ElementType::*;
    for element in [
        Index,
        I1,
        I2,
        I4,
        I8,
        I16,
        I32,
        I64,
        Si1,
        Si2,
        Si4,
        Si8,
        Si16,
        Si32,
        Si64,
        Ui1,
        Ui2,
        Ui4,
        Ui8,
        Ui16,
        Ui32,
        Ui64,
        F4E2M1Fn,
        F6E2M3Fn,
        F6E3M2Fn,
        F16,
        Bf16,
        F32,
        F64,
        Tf32,
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
    ] {
        let ty = element.native(&ctx);
        assert_eq!(primitive_element(&ctx, ty), Some(element));
        assert!(verifies(&ctx, ty));
        assert!(verifies(
            &ctx,
            TensorType::static_shape(&ctx, element, [0, 2, 3])
        ));
    }
}

#[test]
fn bounded_dimensions_match_rank_and_only_bound_dynamic_extents() {
    let ctx = Context::new();
    for (shape, bounds, accepted) in [
        (vec![-1, 3], vec![8, -1], true),
        (vec![-1, 3], vec![-1, -1], true),
        (vec![-1, 3], vec![8, 3], false),
        (vec![-1, 3], vec![8], false),
        (vec![-1, 3], vec![-2, -1], false),
        (vec![0], vec![-1], true),
    ] {
        let ty = TensorType::from_element(&ctx, ElementType::F32.native(&ctx), shape);
        let ty = ty.deref(&ctx);
        let tensor = ty.downcast_ref::<TensorType>().unwrap();
        let encoded = tensor.with_encoding(&ctx, Box::new(TypeExtensionsAttr { bounds }));
        assert_eq!(verifies(&ctx, encoded), accepted);
    }
    assert!(!verifies(
        &ctx,
        TensorType::from_element(&ctx, ElementType::F32.native(&ctx), vec![-2])
    ));
}

#[test]
fn uniform_quantization_storage_scale_and_axis_rules() {
    let ctx = Context::new();
    for (storage, expressed, scale, accepted) in [
        (ElementType::I8, ElementType::F32, 0.5, true),
        (ElementType::Ui8, ElementType::F32, 0.5, true),
        (ElementType::I64, ElementType::F32, 0.5, false),
        (ElementType::I8, ElementType::I32, 0.5, false),
        (ElementType::I8, ElementType::F32, 0.0, false),
        (ElementType::I8, ElementType::F32, -0.5, false),
        (ElementType::I8, ElementType::F32, f64::INFINITY, false),
        (ElementType::I8, ElementType::F32, f64::NAN, true),
        (ElementType::I8, ElementType::F16, 65504.0, true),
        (ElementType::I8, ElementType::F16, 65505.0, false),
        (ElementType::I8, ElementType::F16, 2f64.powi(-24), true),
        (ElementType::I8, ElementType::F16, 2f64.powi(-25), false),
    ] {
        let ty = QuantizedElementType::intern(
            &ctx,
            UniformQuantizedSpec {
                storage_type: storage,
                storage_min: None,
                storage_max: None,
                expressed_type: expressed,
                quantization_dimension: None,
                scales: vec![scale],
                zero_points: vec![0],
            },
        );
        assert_eq!(verifies(&ctx, ty), accepted);
    }
    let a = QuantizedElementType::intern(
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
    let b = QuantizedElementType::intern(
        &ctx,
        UniformQuantizedSpec {
            storage_type: ElementType::I8,
            storage_min: Some(-128),
            storage_max: Some(127),
            expressed_type: ElementType::F32,
            quantization_dimension: None,
            scales: vec![0.5],
            zero_points: vec![0],
        },
    );
    assert_eq!(a, b);
    let axis = QuantizedElementType::intern(
        &ctx,
        UniformQuantizedSpec {
            storage_type: ElementType::I8,
            storage_min: None,
            storage_max: None,
            expressed_type: ElementType::F32,
            quantization_dimension: Some(1),
            scales: vec![0.25, 0.5],
            zero_points: vec![0, 1],
        },
    );
    assert!(verifies(
        &ctx,
        TensorType::from_element(&ctx, axis, vec![3, 2])
    ));
    assert!(!verifies(
        &ctx,
        TensorType::from_element(&ctx, axis, vec![3, 3])
    ));
    assert!(!verifies(
        &ctx,
        TensorType::from_element(&ctx, axis, vec![3])
    ));
    assert!(verifies(
        &ctx,
        TensorType::from_element(&ctx, axis, vec![3, -1])
    ));
}

#[test]
fn calibrated_any_and_quantile_metadata_enforce_valid_ranges() {
    let ctx = Context::new();
    for (min, max, accepted) in [
        (0.0, 1.0, true),
        (1.0, 1.0, false),
        (2.0, 1.0, false),
        (f64::NAN, 1.0, true),
    ] {
        assert_eq!(
            verifies(
                &ctx,
                CalibratedQuantizedType::intern(&ctx, ElementType::F32, min, max)
            ),
            accepted
        );
    }
    assert!(!verifies(
        &ctx,
        CalibratedQuantizedType::intern(&ctx, ElementType::I32, 0.0, 1.0)
    ));
    assert!(verifies(
        &ctx,
        AnyQuantizedType::intern(&ctx, ElementType::I8, None, None, None)
    ));
    assert!(!verifies(
        &ctx,
        AnyQuantizedType::intern(
            &ctx,
            ElementType::I8,
            Some(2),
            Some(1),
            Some(ElementType::F32)
        )
    ));
    assert!(verifies(
        &ctx,
        QuantileType::intern(
            &ctx,
            ElementType::I2,
            ElementType::F32,
            vec![0.0, 0.25, 0.5, 1.0],
            None
        )
    ));
    assert!(!verifies(
        &ctx,
        QuantileType::intern(
            &ctx,
            ElementType::I2,
            ElementType::F32,
            vec![0.0, 0.25],
            None
        )
    ));
    assert!(!verifies(
        &ctx,
        QuantileType::intern(
            &ctx,
            ElementType::I2,
            ElementType::F32,
            vec![0.0, 0.25, f64::NAN, 1.0],
            None
        )
    ));
}

#[test]
fn vector_scalable_dimensions_and_nested_tensor_elements_are_verified() {
    let ctx = Context::new();
    assert!(verifies(
        &ctx,
        VectorType::new_scalable(&ctx, ElementType::F32, vec![2, 3], vec![false, true])
    ));
    assert!(!verifies(
        &ctx,
        VectorType::new_scalable(&ctx, ElementType::F32, vec![2, 3], vec![false])
    ));
    assert!(!verifies(
        &ctx,
        VectorType::fixed(&ctx, ElementType::C64, vec![2])
    ));
    assert!(!verifies(
        &ctx,
        VectorType::fixed(&ctx, ElementType::F32, vec![0])
    ));
    let token: TypeHandle = TokenType::get(&ctx).into();
    assert!(verifies(
        &ctx,
        TensorType::from_element(&ctx, token, vec![2])
    ));
    let vector = VectorType::fixed(&ctx, ElementType::F32, vec![2]);
    assert!(verifies(
        &ctx,
        TensorType::from_element(&ctx, vector, vec![3])
    ));
    let scalar = ElementType::F32.native(&ctx);
    assert!(!verifies(&ctx, TupleType::get(&ctx, vec![scalar]).into()));
}

#[test]
fn subchannel_parameter_tables_normalize_splats_and_check_widths() {
    let ctx = Context::new();
    let ty = SubchannelQuantizedType::intern(
        &ctx,
        SubchannelQuantizedSpec {
            storage: ElementType::I8,
            range: (None, None),
            expressed: ElementType::F32,
            quantized_dimensions: vec![0],
            block_sizes: vec![2],
            parameter_shape: vec![2],
            scales: vec![0x3f00_0000, 0x3f00_0000],
            zero_points: vec![0, 0],
        },
    );
    assert!(verifies(&ctx, ty));
    let t = ty.deref(&ctx);
    let t = t.downcast_ref::<SubchannelQuantizedType>().unwrap();
    assert_eq!(t.scales(), &[0x3f00_0000]);
    let bad = SubchannelQuantizedType::intern(
        &ctx,
        SubchannelQuantizedSpec {
            storage: ElementType::I8,
            range: (None, None),
            expressed: ElementType::F32,
            quantized_dimensions: vec![0],
            block_sizes: vec![0],
            parameter_shape: vec![2],
            scales: vec![0x3f00_0000],
            zero_points: vec![0],
        },
    );
    assert!(!verifies(&ctx, bad));
    let bad = SubchannelQuantizedType::intern(
        &ctx,
        SubchannelQuantizedSpec {
            storage: ElementType::I8,
            range: (None, None),
            expressed: ElementType::F32,
            quantized_dimensions: vec![0],
            block_sizes: vec![2],
            parameter_shape: vec![2],
            scales: vec![1u64 << 32],
            zero_points: vec![0],
        },
    );
    assert!(!verifies(&ctx, bad));
}

#[test]
fn deep_tuples_and_nested_buffers_verify_without_recursive_stack_growth() {
    let ctx = Context::new();
    let mut tuple: TypeHandle = TokenType::get(&ctx).into();
    let mut buffer = ElementType::F32.native(&ctx);
    for _ in 0..4096 {
        tuple = TupleType::get(&ctx, vec![tuple]).into();
        buffer = BufferType::without_layout(&ctx, buffer, vec![2]);
    }
    assert!(verifies(&ctx, tuple));
    assert!(verifies(&ctx, buffer));
}
#[test]
fn function_signatures_reject_non_value_types() {
    let ctx = Context::new();
    let tensor = tensor(&ctx, &[2], ElementType::F32);
    assert!(verifies(
        &ctx,
        FunctionType::intern(&ctx, vec![tensor], vec![])
    ));
    assert!(!verifies(
        &ctx,
        FunctionType::intern(&ctx, vec![ElementType::F32.native(&ctx)], vec![])
    ));
}
#[test]
fn buffer_layout_rank_and_default_memory_spaces_are_canonical() {
    let ctx = Context::new();
    let buffer = BufferType::without_layout(&ctx, ElementType::F32.native(&ctx), vec![2]);
    let ty = buffer.deref(&ctx);
    let ty = ty.downcast_ref::<BufferType>().unwrap();
    assert_eq!(
        ty.with_memory_space(&ctx, Box::new(BoolAttr::new(false))),
        buffer
    );
    let map = AffineMap {
        dimensions: 1,
        symbols: 0,
        results: vec![AffineExpr::Dimension(0)],
    };
    assert_eq!(ty.with_layout(&ctx, Box::new(map)), buffer);
    let map = AffineMap {
        dimensions: 2,
        symbols: 0,
        results: vec![AffineExpr::Dimension(0)],
    };
    assert!(!verifies(&ctx, ty.with_layout(&ctx, Box::new(map))));
    assert!(!verifies(
        &ctx,
        ty.with_layout(
            &ctx,
            Box::new(StridedLayoutAttr {
                offset: vec![0],
                strides: vec![]
            })
        )
    ));
}

#[test]
fn invalid_raw_complex_components_and_extreme_zero_sized_tensors_are_checked() {
    let ctx = Context::new();
    let invalid = FloatType::get(&ctx, 255).into();
    assert!(!verifies(&ctx, ComplexType::get(&ctx, invalid).into()));
    let ty = TensorType::static_shape(&ctx, ElementType::F32, [i64::MAX, i64::MAX, 0]);
    assert!(verifies(&ctx, ty));
    let attr = DenseElementsAttr::new(ty, VecAttr::new(vec![]));
    assert!(attr.verify(&ctx).is_ok());
}
