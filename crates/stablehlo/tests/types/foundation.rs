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
fn scalar_formats_have_canonical_spellings_and_native_identity() {
    let ctx = Context::new();
    for (element, spelling) in [
        (ElementType::Index, "index"),
        (ElementType::I1, "i1"),
        (ElementType::I2, "i2"),
        (ElementType::I4, "i4"),
        (ElementType::I8, "i8"),
        (ElementType::I16, "i16"),
        (ElementType::I32, "i32"),
        (ElementType::I64, "i64"),
        (ElementType::Si1, "si1"),
        (ElementType::Si2, "si2"),
        (ElementType::Si4, "si4"),
        (ElementType::Si8, "si8"),
        (ElementType::Si16, "si16"),
        (ElementType::Si32, "si32"),
        (ElementType::Si64, "si64"),
        (ElementType::Ui1, "ui1"),
        (ElementType::Ui2, "ui2"),
        (ElementType::Ui4, "ui4"),
        (ElementType::Ui8, "ui8"),
        (ElementType::Ui16, "ui16"),
        (ElementType::Ui32, "ui32"),
        (ElementType::Ui64, "ui64"),
        (ElementType::F4E2M1Fn, "f4E2M1FN"),
        (ElementType::F6E2M3Fn, "f6E2M3FN"),
        (ElementType::F6E3M2Fn, "f6E3M2FN"),
        (ElementType::F16, "f16"),
        (ElementType::Bf16, "bf16"),
        (ElementType::F32, "f32"),
        (ElementType::F64, "f64"),
        (ElementType::Tf32, "tf32"),
        (ElementType::F8E3M4, "f8E3M4"),
        (ElementType::F8E4M3, "f8E4M3"),
        (ElementType::F8E4M3Fn, "f8E4M3FN"),
        (ElementType::F8E4M3FnUz, "f8E4M3FNUZ"),
        (ElementType::F8E4M3B11FnUz, "f8E4M3B11FNUZ"),
        (ElementType::F8E5M2, "f8E5M2"),
        (ElementType::F8E5M2FnUz, "f8E5M2FNUZ"),
        (ElementType::F8E8M0FnUz, "f8E8M0FNU"),
        (ElementType::C64, "complex<f32>"),
        (ElementType::C128, "complex<f64>"),
        (
            ElementType::integer(129, IntegerSignedness::Signed).unwrap(),
            "si129",
        ),
        (
            ElementType::integer(65, IntegerSignedness::Unsigned).unwrap(),
            "ui65",
        ),
        (
            ElementType::complex(ElementType::I32).unwrap(),
            "complex<i32>",
        ),
        (
            ElementType::complex(ElementType::F16).unwrap(),
            "complex<f16>",
        ),
        (
            ElementType::complex(ElementType::integer(129, IntegerSignedness::Signed).unwrap())
                .unwrap(),
            "complex<si129>",
        ),
    ] {
        assert_eq!(element.to_string(), spelling);
        let ty = element.native(&ctx);
        assert_eq!(primitive_element(&ctx, ty), Some(element));
        assert!(verifies(&ctx, ty));
        assert!(verifies(
            &ctx,
            TensorType::static_shape(&ctx, element, [0, 2, 3])
        ));
    }
    assert_eq!(
        format!(
            "{:?}",
            ElementType::integer(129, IntegerSignedness::Signed).unwrap()
        ),
        "Integer(IntegerElement { width: 129, signedness: Signed })"
    );
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
        results: vec![AffineExpr::dimension(0)],
    };
    assert_eq!(ty.with_layout(&ctx, Box::new(map)), buffer);
    let map = AffineMap {
        dimensions: 2,
        symbols: 0,
        results: vec![AffineExpr::dimension(0)],
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
