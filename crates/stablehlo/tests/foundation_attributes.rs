use pliron::{common_traits::Verify, context::Context};
use stablehlo::pliron::builtin::attributes::{StringAttr, VecAttr};
use stablehlo::{attributes::*, integer::*, sparse::*, types::*};

#[test]
fn dimension_number_attributes_reject_negative_fields_without_op_specific_constraints() {
    let ctx = Context::new();
    let mut dot = DotDimensionNumbersAttr {
        lhs_batching_dimensions: vec![0],
        rhs_batching_dimensions: vec![0],
        lhs_contracting_dimensions: vec![2],
        rhs_contracting_dimensions: vec![1],
    };
    assert!(dot.verify(&ctx).is_ok());
    dot.lhs_contracting_dimensions[0] = -1;
    assert!(dot.verify(&ctx).is_err());
    let mut gather = GatherDimensionNumbersAttr {
        offset_dims: vec![1],
        collapsed_slice_dims: vec![0],
        operand_batching_dims: vec![],
        start_indices_batching_dims: vec![],
        start_index_map: vec![0],
        index_vector_dim: 1,
    };
    assert!(gather.verify(&ctx).is_ok());
    gather.index_vector_dim = -1;
    assert!(gather.verify(&ctx).is_err());
    let mut scatter = ScatterDimensionNumbersAttr {
        update_window_dims: vec![1],
        inserted_window_dims: vec![0],
        input_batching_dims: vec![],
        scatter_indices_batching_dims: vec![],
        scatter_dims_to_operand_dims: vec![0],
        index_vector_dim: 1,
    };
    assert!(scatter.verify(&ctx).is_ok());
    scatter.input_batching_dims.push(-1);
    assert!(scatter.verify(&ctx).is_err());
    let mut conv = ConvDimensionNumbersAttr {
        input_batch_dimension: 0,
        input_feature_dimension: 1,
        input_spatial_dimensions: vec![2],
        kernel_input_feature_dimension: 1,
        kernel_output_feature_dimension: 0,
        kernel_spatial_dimensions: vec![2],
        output_batch_dimension: 0,
        output_feature_dimension: 1,
        output_spatial_dimensions: vec![2],
    };
    assert!(conv.verify(&ctx).is_ok());
    conv.output_feature_dimension = -1;
    assert!(conv.verify(&ctx).is_err());
}
#[test]
fn enum_attributes_accept_only_known_values() {
    let ctx = Context::new();
    for value in 0..=6 {
        assert_eq!(
            ComparisonDirectionAttr { value }.verify(&ctx).is_ok(),
            value <= 5
        );
        assert_eq!(
            ComparisonTypeAttr { value }.verify(&ctx).is_ok(),
            value <= 5
        );
    }
    for value in 0..=3 {
        assert_eq!(PrecisionAttr { value }.verify(&ctx).is_ok(), value <= 2);
        assert_eq!(RngAlgorithmAttr { value }.verify(&ctx).is_ok(), value <= 2);
        assert_eq!(
            ResultAccuracyModeAttr { value }.verify(&ctx).is_ok(),
            value <= 2
        );
        assert_eq!(
            RngDistributionAttr { value }.verify(&ctx).is_ok(),
            value <= 1
        );
    }
    for value in 0..=4 {
        assert_eq!(FftTypeAttr { value }.verify(&ctx).is_ok(), value <= 3);
    }
    assert!(
        EnumAttr {
            dialect: StringAttr::new("stablehlo".into()),
            name: StringAttr::new("precision".into()),
            value: StringAttr::new("HIGH".into())
        }
        .verify(&ctx)
        .is_ok()
    );
    assert!(
        EnumAttr {
            dialect: StringAttr::new("stablehlo".into()),
            name: StringAttr::new("precision".into()),
            value: StringAttr::new("INVALID".into())
        }
        .verify(&ctx)
        .is_err()
    );
}
#[test]
fn result_accuracy_preserves_signed_zero_nan_and_mode_rules() {
    let ctx = Context::new();
    for (mode, atol, rtol, ulps, accepted) in [
        (ResultAccuracyMode::Default, 0.0, 0.0, 0, true),
        (ResultAccuracyMode::Highest, 0.0, 0.0, 0, true),
        (ResultAccuracyMode::Default, 0.1, 0.0, 0, false),
        (ResultAccuracyMode::Tolerance, 0.1, 0.2, 1, true),
        (ResultAccuracyMode::Tolerance, -0.0, 0.0, 0, false),
        (ResultAccuracyMode::Tolerance, 0.0, 0.0, -1, false),
        (
            ResultAccuracyMode::Tolerance,
            f64::INFINITY,
            f64::NAN,
            0,
            true,
        ),
    ] {
        assert_eq!(
            ResultAccuracyAttr::new(atol, rtol, ulps, mode)
                .verify(&ctx)
                .is_ok(),
            accepted
        );
    }
}
#[test]
fn dot_algorithm_type_and_operation_tables_are_checked() {
    let ctx = Context::new();
    for (lhs, rhs, accumulation, operations, accepted) in [
        (
            ElementType::F16,
            ElementType::F16,
            ElementType::F32,
            1,
            true,
        ),
        (
            ElementType::Bf16,
            ElementType::Bf16,
            ElementType::F32,
            6,
            true,
        ),
        (
            ElementType::Tf32,
            ElementType::Tf32,
            ElementType::F32,
            3,
            true,
        ),
        (
            ElementType::F32,
            ElementType::F32,
            ElementType::F32,
            2,
            false,
        ),
        (
            ElementType::F16,
            ElementType::F32,
            ElementType::F32,
            1,
            false,
        ),
    ] {
        let algorithm = DotAlgorithmAttr {
            lhs_precision_type: lhs.native(&ctx),
            rhs_precision_type: rhs.native(&ctx),
            accumulation_type: accumulation.native(&ctx),
            lhs_component_count: 1,
            rhs_component_count: 1,
            num_primitive_operations: operations,
            allow_imprecise_accumulation: false,
        };
        assert_eq!(algorithm.verify(&ctx).is_ok(), accepted);
    }
}
fn axis(name: &str) -> AxisRefAttr {
    AxisRefAttr {
        name: StringAttr::new(name.into()),
        sub_axis_info: vec![],
    }
}
fn mesh() -> MeshAttr {
    MeshAttr {
        axes: vec![
            MeshAxisAttr {
                name: StringAttr::new("x".into()),
                size: 2,
            },
            MeshAxisAttr {
                name: StringAttr::new("y".into()),
                size: 3,
            },
        ],
        device_ids: vec![],
    }
}
#[test]
fn mesh_replica_group_order_and_explicit_device_ids_are_preserved() {
    let ctx = Context::new();
    let mut mesh = mesh();
    assert_eq!(
        mesh.replica_groups(&ctx, &[axis("y")]).unwrap(),
        vec![vec![0, 1, 2], vec![3, 4, 5]]
    );
    assert_eq!(
        mesh.replica_groups(&ctx, &[axis("x")]).unwrap(),
        vec![vec![0, 3], vec![1, 4], vec![2, 5]]
    );
    assert_eq!(
        mesh.replica_group_size(&ctx, &[axis("x"), axis("y")])
            .unwrap(),
        6
    );
    assert!(mesh.replica_groups(&ctx, &[axis("missing")]).is_err());
    assert!(mesh.replica_groups(&ctx, &[axis("x"), axis("x")]).is_err());
    mesh.device_ids = vec![IndexTensorAttr {
        values: vec![5, 4, 3, 2, 1, 0],
    }];
    assert_eq!(
        mesh.replica_groups(&ctx, &[axis("y")]).unwrap(),
        vec![vec![5, 4, 3], vec![2, 1, 0]]
    );
    mesh.device_ids[0].values = vec![0, 0, 1, 2, 3, 4];
    assert!(mesh.replica_groups(&ctx, &[axis("x")]).is_err());
}
#[test]
fn mesh_subaxes_validate_divisibility_and_overlap() {
    let ctx = Context::new();
    let mesh = MeshAttr {
        axes: vec![MeshAxisAttr {
            name: StringAttr::new("x".into()),
            size: 8,
        }],
        device_ids: vec![],
    };
    let sub = AxisRefAttr {
        name: StringAttr::new("x".into()),
        sub_axis_info: vec![SubAxisInfoAttr {
            pre_size: 2,
            size: 2,
        }],
    };
    assert_eq!(
        mesh.replica_groups(&ctx, std::slice::from_ref(&sub))
            .unwrap(),
        vec![vec![0, 2], vec![1, 3], vec![4, 6], vec![5, 7]]
    );
    assert!(mesh.replica_groups(&ctx, &[axis("x"), sub]).is_err());
    let bad = AxisRefAttr {
        name: StringAttr::new("x".into()),
        sub_axis_info: vec![SubAxisInfoAttr {
            pre_size: 3,
            size: 2,
        }],
    };
    assert!(mesh.replica_groups(&ctx, &[bad]).is_err());
}
#[test]
fn dense_payloads_preserve_bits_and_check_counts_and_high_bits() {
    let ctx = Context::new();
    let ty = tensor(&ctx, &[2], ElementType::F32);
    let bits = ScalarBitsAttr::f32_bits(&ctx, 0x8000_0000);
    let attr = DenseElementsAttr::splat(ty, Box::new(bits.clone()));
    assert!(attr.verify(&ctx).is_ok());
    let attr = DenseElementsAttr::new(ty, VecAttr::new(vec![]));
    assert!(attr.verify(&ctx).is_err());
    assert!(
        ScalarBitsAttr {
            ty: ElementType::F32.native(&ctx),
            words: vec![1 << 32]
        }
        .verify(&ctx)
        .is_err()
    );
    assert!(
        DenseArrayAttr::bools(&ctx, vec![true, false])
            .verify(&ctx)
            .is_ok()
    );
    assert!(
        DenseArrayAttr::integers(&ctx, vec![-1, 0, 1])
            .verify(&ctx)
            .is_ok()
    );
    let wrong = SparseElementsAttr {
        ty,
        indices: vec![IndexTensorAttr { values: vec![2] }],
        values: VecAttr::new(vec![Box::new(bits)]),
    };
    assert!(wrong.verify(&ctx).is_err());
}
#[test]
fn affine_product_divisor_signed_rounding_and_argument_rules() {
    use AffineBinaryOp::*;
    let dim = AffineExpr::Dimension(0);
    let invalid = AffineExpr::binary(Mul, dim.clone(), dim.clone());
    let error = invalid
        .validate(1, 0)
        .expect_err("non-affine product must fail");
    assert!(matches!(
        error.err.downcast_ref::<stablehlo::Diagnostic>(),
        Some(stablehlo::Diagnostic::NonAffineProduct)
    ));
    assert!(
        AffineExpr::binary(FloorDiv, AffineExpr::Constant(3), dim.clone())
            .validate(1, 0)
            .is_err()
    );
    for (op, expected) in [(FloorDiv, -3), (CeilDiv, -2), (Mod, 2)] {
        assert_eq!(
            AffineExpr::binary(op, dim.clone(), AffineExpr::Constant(3))
                .evaluate_at(&[-7], &[])
                .unwrap(),
            expected
        );
    }
    let error = AffineExpr::binary(FloorDiv, dim.clone(), AffineExpr::Constant(0))
        .evaluate_at(&[1], &[])
        .expect_err("division by zero must fail");
    assert!(matches!(
        error.err.downcast_ref::<stablehlo::Diagnostic>(),
        Some(stablehlo::Diagnostic::DivisionByZero)
    ));
    let map = AffineMap {
        dimensions: 1,
        symbols: 0,
        results: vec![dim],
    };
    assert_eq!(map.evaluate(&[9], &[]).unwrap(), vec![9]);
    assert!(map.evaluate(&[], &[]).is_err());
}
#[test]
fn sparse_map_inverse_levels_and_tensor_dtype_rules() {
    let ctx = Context::new();
    let map = AffineMap {
        dimensions: 2,
        symbols: 0,
        results: vec![AffineExpr::Dimension(1), AffineExpr::Dimension(0)],
    };
    let mut encoding = SparseTensorEncoding::new(
        &ctx,
        vec![
            SparseLevelType::new(SparseLevelFormat::Dense),
            SparseLevelType::new(SparseLevelFormat::Compressed),
        ],
        map,
    )
    .unwrap();
    assert!(encoding.verify_tensor(&ctx, 2, ElementType::F32).is_ok());
    assert!(encoding.verify_tensor(&ctx, 1, ElementType::F32).is_err());
    assert!(encoding.verify_tensor(&ctx, 2, ElementType::I1).is_err());
    encoding.position_width = 7;
    assert!(encoding.verify(&ctx).is_err());
    assert!(
        SparseLevelType::new(SparseLevelFormat::Structured { n: 3, m: 2 })
            .verify(&ctx)
            .is_err()
    );
    assert!(
        SparseTensorDimSlice {
            offset: vec![0],
            size: vec![0],
            stride: vec![1]
        }
        .verify(&ctx)
        .is_err()
    );
}

#[test]
fn dense_short_payloads_require_explicit_splats_and_quantized_ranges_are_checked() {
    let ctx = Context::new();
    let ty = tensor(&ctx, &[2], ElementType::F32);
    assert!(
        DenseElementsAttr::new(
            ty,
            VecAttr::new(vec![Box::new(ScalarBitsAttr::f32_bits(&ctx, 0))])
        )
        .verify(&ctx)
        .is_err()
    );
    let quantized = QuantizedElementType::intern(
        &ctx,
        UniformQuantizedSpec {
            storage_type: ElementType::I8,
            storage_min: Some(-10),
            storage_max: Some(10),
            expressed_type: ElementType::F32,
            quantization_dimension: None,
            scales: vec![0.5],
            zero_points: vec![0],
        },
    );
    let ty = TensorType::from_element(&ctx, quantized, vec![1]);
    for (value, accepted) in [(-10, true), (10, true), (11, false)] {
        let bits =
            ScalarBitsAttr::integer(&ctx, ElementType::I8, IntegerValue::from(value)).unwrap();
        assert_eq!(
            DenseElementsAttr::new(ty, VecAttr::new(vec![Box::new(bits)]))
                .verify(&ctx)
                .is_ok(),
            accepted
        );
    }
}

#[test]
fn standalone_complex_metadata_requires_matching_floating_components() {
    let ctx = Context::new();
    let value = ComplexAttr {
        ty: ElementType::C64.native(&ctx),
        real: ScalarBitsAttr::f32_bits(&ctx, 0x7fc0_1234),
        imaginary: ScalarBitsAttr::f32_bits(&ctx, 0x8000_0000),
    };
    assert!(value.verify(&ctx).is_ok());
    let mut wrong = value.clone();
    wrong.imaginary = ScalarBitsAttr::float_bits(&ctx, 0);
    assert!(wrong.verify(&ctx).is_err());
    let component = ElementType::I32;
    let complex = ElementType::complex(component).unwrap();
    let real = ScalarBitsAttr::integer(&ctx, component, IntegerValue::from(0)).unwrap();
    let value = ComplexAttr {
        ty: complex.native(&ctx),
        real: real.clone(),
        imaginary: real,
    };
    assert!(value.verify(&ctx).is_err());
}
