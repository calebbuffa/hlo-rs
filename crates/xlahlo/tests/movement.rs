use xlahlo::{
    Diagnostic,
    attributes::{
        Axes, Callees, GatherDimensionNumbers, Index, IndexFlags, PaddingConfig,
        PaddingConfigDimension, ScatterDimensionNumbers, ShuffleMode, Slice,
    },
    inference::{Signature, movement as infer},
    literal::Literal,
    ops::{
        ComputationOp, HloOpInterface, ReturnOp,
        control::{ConstantOp, ParameterOp, TupleOp},
        movement::{
            BitcastOp, BroadcastOp, ConcatenateOp, CopyOp, DynamicReshapeOp, DynamicSliceOp,
            DynamicUpdateSliceOp, GatherOp, GetDimensionSizeOp, IotaOp, PadOp, ReshapeOp,
            ReverseOp, ScatterOp, SetDimensionSizeOp, ShuffleOp, SliceOp, TransposeOp,
        },
    },
    pliron::{
        basic_block::BasicBlock,
        builtin::{ops::ModuleOp, types::FunctionType},
        common_traits::Verify,
        context::Context,
        identifier::Identifier,
        op::Op,
        operation::Operation,
        result::Result,
        r#type::{Type, TypeHandle},
        value::Value,
    },
    shape::{Dimension, Layout, Shape},
    types::{ElementType as E, TensorType},
    verification::{VerifyOptions, verify, verify_with_options},
};

#[test]
fn dimension_size_inference_preserves_s32_boundary_for_every_primitive() {
    let ctx = Context::new();
    for &element in E::ALL {
        for dimension in [
            Dimension::Static(0),
            Dimension::Static(i64::from(i32::MAX)),
            Dimension::Bounded(0),
            Dimension::Bounded(i64::from(i32::MAX)),
            Dimension::Unbounded,
        ] {
            let input = Shape::dynamic_array(&ctx, element, vec![dimension]).unwrap();
            let result = infer::infer_get_dimension_size(&ctx, input, 0).unwrap();
            assert_eq!(Shape::element_type(&ctx, result).unwrap(), E::S32);
            assert_eq!(Shape::rank(&ctx, result).unwrap(), 0);
            assert!(infer::infer_get_dimension_size(&ctx, input, 1).is_err());
        }
        for dimension in [
            Dimension::Static(i64::from(i32::MAX) + 1),
            Dimension::Bounded(i64::from(i32::MAX) + 1),
        ] {
            let input = Shape::dynamic_array(&ctx, element, vec![dimension]).unwrap();
            let error = infer::infer_get_dimension_size(&ctx, input, 0).unwrap_err();
            assert!(error.to_string().contains("dimension exceeds S32 limit"));
        }
    }
}

#[test]
fn layout_sensitive_copy_checks_host_offloading_direction_and_precision() {
    let mut ctx = Context::new();
    let device = tensor(&ctx, E::F32, &[2, 3]);
    let mut host_layout = Layout::descending(2);
    host_layout.memory_space = 5;
    let host = Shape::with_layout(&ctx, device, host_layout.clone()).unwrap();
    let mut transposed_layout = Layout::descending(2);
    transposed_layout.minor_to_major = vec![0, 1];
    let transposed_device = Shape::with_layout(&ctx, device, transposed_layout.clone()).unwrap();
    transposed_layout.memory_space = 5;
    let transposed_host = Shape::with_layout(&ctx, device, transposed_layout).unwrap();
    let narrow_device = tensor(&ctx, E::BF16, &[2, 3]);
    let narrow_host = Shape::with_layout(&ctx, narrow_device, host_layout).unwrap();
    let options = VerifyOptions {
        layout_sensitive: true,
        allow_mixed_precision: true,
        ..Default::default()
    };
    for (input, output, accepted) in [
        (device, host, true),
        (host, device, true),
        (device, transposed_host, true),
        (host, transposed_device, false),
        (host, narrow_device, false),
        (device, narrow_host, false),
    ] {
        let operand = value(&mut ctx, input, 0);
        let copy = CopyOp::new(&mut ctx, output, vec![operand]);
        assert_eq!(copy.verify_with_options(&ctx, options).is_ok(), accepted);
    }
    let operand = value(&mut ctx, device, 0);
    CopyOp::new(&mut ctx, host, vec![operand])
        .verify(&ctx)
        .unwrap();
}

#[test]
fn static_slice_propagates_only_bounded_dynamism_and_eliminates_unit_extents() {
    let mut ctx = Context::new();
    for input_dimension in [
        Dimension::Static(6),
        Dimension::Bounded(6),
        Dimension::Unbounded,
    ] {
        let input_type = Shape::dynamic_array(&ctx, E::F32, vec![input_dimension]).unwrap();
        let input = value(&mut ctx, input_type, 0);
        for (limit, stride, extent) in [(0, 1, 0), (1, 1, 1), (5, 2, 3), (6, 1, 6)] {
            let geometry = Slice {
                starts: Axes(vec![0]),
                limits: Axes(vec![limit]),
                strides: Axes(vec![stride]),
            };
            let expected_dimension = if input_dimension == Dimension::Bounded(6) && extent != 1 {
                Dimension::Bounded(extent)
            } else {
                Dimension::Static(extent)
            };
            let inferred = infer::infer_slice(&ctx, input_type, &geometry).unwrap();
            assert_eq!(
                Shape::dimension(&ctx, inferred, 0).unwrap(),
                expected_dimension
            );
            let expected = Shape::dynamic_array(&ctx, E::F32, vec![expected_dimension]).unwrap();
            SliceOp::new(&mut ctx, expected, vec![input], geometry)
                .verify(&ctx)
                .unwrap();
        }
    }
}

#[test]
fn concatenate_precision_uses_each_operand_against_the_first_operand() {
    let mut ctx = Context::new();
    for (elements, expected_element) in [
        (vec![E::BF16, E::F32], E::F32),
        (vec![E::F32, E::BF16], E::F32),
        (vec![E::F16, E::BF16], E::BF16),
        (vec![E::BF16, E::F16], E::BF16),
        (vec![E::BF16, E::F64, E::F32], E::F32),
        (vec![E::BF16, E::F32, E::F64], E::F64),
    ] {
        let inputs = elements
            .iter()
            .map(|&element| tensor(&ctx, element, &[2]))
            .collect::<Vec<_>>();
        let inferred = infer::infer_concatenate(&ctx, &inputs, 0).unwrap();
        assert_eq!(
            Shape::element_type(&ctx, inferred).unwrap(),
            expected_element
        );
        let expected = tensor(&ctx, expected_element, &[2 * inputs.len() as i64]);
        let operands = inputs
            .iter()
            .enumerate()
            .map(|(index, &ty)| value(&mut ctx, ty, index.try_into().unwrap()))
            .collect();
        let concat = ConcatenateOp::new(&mut ctx, expected, operands, Index(0));
        assert!(concat.verify(&ctx).is_err());
        concat
            .verify_with_options(
                &ctx,
                VerifyOptions {
                    allow_mixed_precision: true,
                    ..Default::default()
                },
            )
            .unwrap();
    }
}

#[test]
fn dynamic_slice_accepts_vector_or_homogeneous_scalar_indices_and_preserves_full_bounds() {
    let mut ctx = Context::new();
    let index = scalar(&ctx, E::S32);
    let vector = tensor(&ctx, E::S32, &[1]);
    for input_dimension in [
        Dimension::Static(6),
        Dimension::Bounded(6),
        Dimension::Unbounded,
    ] {
        let input_type = Shape::dynamic_array(&ctx, E::F32, vec![input_dimension]).unwrap();
        let input = value(&mut ctx, input_type, 0);
        for size in [0, 1, 3, 6] {
            let expected_dimension = if input_dimension == Dimension::Bounded(6) && size == 6 {
                Dimension::Bounded(size)
            } else {
                Dimension::Static(size)
            };
            for indices in [&[index][..], &[vector][..]] {
                let inferred =
                    infer::infer_dynamic_slice(&ctx, input_type, indices, &Axes(vec![size]))
                        .unwrap();
                assert_eq!(
                    Shape::dimension(&ctx, inferred, 0).unwrap(),
                    expected_dimension
                );
                let expected =
                    Shape::dynamic_array(&ctx, E::F32, vec![expected_dimension]).unwrap();
                let start = value(&mut ctx, indices[0], 1);
                DynamicSliceOp::new(&mut ctx, expected, vec![input, start], Axes(vec![size]))
                    .verify(&ctx)
                    .unwrap();
            }
        }
    }
    let matrix = tensor(&ctx, E::F32, &[6, 6]);
    let s64_index = scalar(&ctx, E::S64);
    let float_index = scalar(&ctx, E::F32);
    let wrong_vector = tensor(&ctx, E::S32, &[1]);
    for indices in [
        vec![index],
        vec![index, s64_index],
        vec![float_index; 2],
        vec![wrong_vector],
    ] {
        assert!(infer::infer_dynamic_slice(&ctx, matrix, &indices, &Axes(vec![2, 2])).is_err());
        assert!(infer::infer_dynamic_update_slice(&ctx, matrix, matrix, &indices).is_err());
    }
    let scalar_input = scalar(&ctx, E::F32);
    let sliced = infer::infer_dynamic_slice(&ctx, scalar_input, &[], &Axes(vec![])).unwrap();
    assert!(Shape::dimensions(&ctx, sliced).unwrap().is_empty());
    assert_eq!(Shape::element_type(&ctx, sliced).unwrap(), E::F32);
    assert_eq!(
        infer::infer_dynamic_update_slice(&ctx, scalar_input, scalar_input, &[]).unwrap(),
        scalar_input
    );
}

#[test]
fn dynamic_update_slice_only_propagates_update_dynamism_for_full_dimensions() {
    let mut ctx = Context::new();
    let index_type = scalar(&ctx, E::S32);
    for (input_dimension, update_dimension, expected_dimension) in [
        (
            Dimension::Static(6),
            Dimension::Bounded(6),
            Dimension::Bounded(6),
        ),
        (
            Dimension::Static(6),
            Dimension::Bounded(3),
            Dimension::Static(6),
        ),
        (
            Dimension::Bounded(6),
            Dimension::Static(3),
            Dimension::Bounded(6),
        ),
        (
            Dimension::Unbounded,
            Dimension::Bounded(3),
            Dimension::Unbounded,
        ),
        (
            Dimension::Unbounded,
            Dimension::Unbounded,
            Dimension::Unbounded,
        ),
    ] {
        let array = Shape::dynamic_array(&ctx, E::F32, vec![input_dimension]).unwrap();
        let mut layout = Layout::descending(1);
        layout.memory_space = 7;
        let array = Shape::with_layout(&ctx, array, layout).unwrap();
        let update = Shape::dynamic_array(&ctx, E::F32, vec![update_dimension]).unwrap();
        for input in [array, Shape::buffer(&ctx, array).unwrap()] {
            let inferred =
                infer::infer_dynamic_update_slice(&ctx, input, update, &[index_type]).unwrap();
            let inferred_array = Shape::array_view(&ctx, inferred).unwrap();
            assert_eq!(
                Shape::dimension(&ctx, inferred_array, 0).unwrap(),
                expected_dimension
            );
            assert_eq!(
                Shape::layout(&ctx, inferred_array)
                    .unwrap()
                    .unwrap()
                    .memory_space,
                7
            );
            assert_eq!(
                Shape::is_array(&ctx, inferred),
                Shape::is_array(&ctx, input)
            );
            let operand = value(&mut ctx, input, 0);
            let update_value = value(&mut ctx, update, 1);
            let index = value(&mut ctx, index_type, 2);
            DynamicUpdateSliceOp::new(&mut ctx, inferred, vec![operand, update_value, index])
                .verify(&ctx)
                .unwrap();
        }
    }
}

#[test]
fn malformed_variadic_operand_lists_return_typed_errors() {
    let mut ctx = Context::new();
    let ty = tensor(&ctx, E::F32, &[2]);
    let errors = [
        ConcatenateOp::new(&mut ctx, ty, vec![], Index(0)).verify(&ctx),
        DynamicReshapeOp::new(&mut ctx, ty, vec![]).verify(&ctx),
        DynamicSliceOp::new(&mut ctx, ty, vec![], Axes(vec![1])).verify(&ctx),
        DynamicUpdateSliceOp::new(&mut ctx, ty, vec![]).verify(&ctx),
    ];
    for result in errors {
        let error = result.expect_err("missing data operands must not panic or succeed");
        assert!(error.err.downcast_ref::<Diagnostic>().is_some());
    }
}

#[test]
fn missing_and_wrongly_typed_movement_attributes_are_typed_errors() {
    let mut ctx = Context::new();
    let ty = tensor(&ctx, E::F32, &[2]);
    let input = value(&mut ctx, ty, 0);
    let operation = Operation::new(
        &mut ctx,
        BroadcastOp::get_concrete_op_info(),
        vec![ty],
        vec![input],
        vec![],
        0,
    );
    let broadcast = Operation::get_op::<BroadcastOp>(operation, &ctx).unwrap();
    let missing = broadcast.verify(&ctx).unwrap_err();
    assert!(matches!(
        missing.err.downcast_ref::<Diagnostic>(),
        Some(Diagnostic::InvalidAttribute {
            attribute: "hlo_dimensions",
            ..
        })
    ));
    operation
        .deref_mut(&ctx)
        .attributes
        .set(Identifier::new("hlo_dimensions"), Index(0));
    let wrong_type = broadcast.verify(&ctx).unwrap_err();
    assert!(matches!(
        wrong_type.err.downcast_ref::<Diagnostic>(),
        Some(Diagnostic::InvalidAttribute {
            attribute: "hlo_dimensions",
            ..
        })
    ));
}

#[test]
fn malformed_copy_origins_and_copy_cycles_fail_bitcast_verification() {
    let mut ctx = Context::new();
    let host = tensor(&ctx, E::F32, &[2]);
    let mut device_layout = Layout::descending(1);
    device_layout.memory_space = 5;
    let device = Shape::with_layout(&ctx, host, device_layout).unwrap();
    let strict = VerifyOptions {
        layout_sensitive: true,
        ..Default::default()
    };
    let malformed_copy = CopyOp::new(&mut ctx, host, vec![]);
    let malformed_value = malformed_copy.result(&ctx);
    let bitcast = BitcastOp::new(&mut ctx, device, vec![malformed_value]);
    let error = bitcast.verify_with_options(&ctx, strict).unwrap_err();
    assert!(error.err.downcast_ref::<Diagnostic>().is_some());

    let parameter = value(&mut ctx, host, 0);
    let first = CopyOp::new(&mut ctx, host, vec![parameter]);
    let first_value = first.result(&ctx);
    let second = CopyOp::new(&mut ctx, host, vec![first_value]);
    Operation::replace_operand(first.get_operation(), &ctx, 0, second.result(&ctx));
    let bitcast = BitcastOp::new(&mut ctx, device, vec![first_value]);
    let error = bitcast.verify_with_options(&ctx, strict).unwrap_err();
    assert!(matches!(
        error.err.downcast_ref::<Diagnostic>(),
        Some(Diagnostic::Verification { message }) if message.contains("cycle")
    ));
}

#[test]
fn reshape_rejects_invalid_inferred_axes_for_static_and_dynamic_inputs() {
    let ctx = Context::new();
    let inputs = [
        tensor(&ctx, E::F32, &[6]),
        Shape::dynamic_array(&ctx, E::F32, vec![Dimension::Bounded(6)]).unwrap(),
    ];
    for input in inputs {
        let error = infer::infer_reshape_with_dimension(&ctx, input, &[2, 3], Some(usize::MAX))
            .unwrap_err();
        assert!(matches!(
            error.err.downcast_ref::<Diagnostic>(),
            Some(Diagnostic::Inference { message }) if message.contains("output rank 2")
        ));
    }
}

#[test]
fn movement_overflow_and_invalid_index_vector_return_typed_diagnostics() {
    let ctx = Context::new();
    let large = tensor(&ctx, E::U8, &[i64::MAX]);
    let error = infer::infer_concatenate(&ctx, &[large, large], 0).unwrap_err();
    assert!(matches!(
        error.err.downcast_ref::<Diagnostic>(),
        Some(Diagnostic::Inference { message }) if message.contains("dimension 0") && message.contains("overflows")
    ));
    let config = PaddingConfig(vec![PaddingConfigDimension {
        edge_padding_low: 1,
        edge_padding_high: 0,
        interior_padding: 0,
    }]);
    let error = infer::infer_pad(&ctx, large, scalar(&ctx, E::U8), &config).unwrap_err();
    assert!(matches!(
        error.err.downcast_ref::<Diagnostic>(),
        Some(Diagnostic::Inference { message }) if message.contains("axis 0") && message.contains("overflows")
    ));
    let input = tensor(&ctx, E::F32, &[4]);
    let indices = tensor(&ctx, E::S32, &[1]);
    let dimensions = GatherDimensionNumbers {
        offset_dims: Axes(vec![]),
        collapsed_slice_dims: Axes(vec![0]),
        operand_batching_dims: Axes(vec![]),
        start_indices_batching_dims: Axes(vec![]),
        start_index_map: Axes(vec![0]),
        index_vector_dim: -1,
    };
    let error = infer::infer_gather(&ctx, input, indices, &dimensions, &Axes(vec![1])).unwrap_err();
    assert!(matches!(
        error.err.downcast_ref::<Diagnostic>(),
        Some(Diagnostic::InvalidAttribute { attribute: "index_vector_dim", message }) if message.contains("-1")
    ));
}

#[test]
fn malformed_tensor_metadata_is_rejected_before_movement_indexing() {
    let ctx = Context::new();
    let malformed: TypeHandle = TensorType::instantiate(
        TensorType {
            element: E::F32,
            dimensions: vec![2, 3],
            dynamic_dimensions: vec![false],
            layout: vec![],
        },
        &ctx,
    )
    .into();
    let errors = [
        infer::infer_transpose(&ctx, malformed, &Axes(vec![1, 0])),
        infer::infer_reshape(&ctx, malformed, &[6]),
        infer::infer_slice(
            &ctx,
            malformed,
            &Slice {
                starts: Axes(vec![0, 0]),
                limits: Axes(vec![2, 3]),
                strides: Axes(vec![1, 1]),
            },
        ),
    ];
    for result in errors {
        let error = result.expect_err("malformed dynamic flags must not reach axis indexing");
        assert!(matches!(
            error.err.downcast_ref::<Diagnostic>(),
            Some(Diagnostic::InvalidShape { .. })
        ));
    }
}

#[test]
fn dynamic_reshape_inference_and_instruction_size_contracts_are_distinct() {
    let mut ctx = Context::new();
    let input_type = tensor(&ctx, E::F32, &[6]);
    let bounds = vec![Dimension::Static(2), Dimension::Bounded(3)];
    let output = Shape::dynamic_array(&ctx, E::F32, bounds.clone()).unwrap();
    let s32_type = scalar(&ctx, E::S32);
    let f64_type = scalar(&ctx, E::F64);
    for sizes in [vec![], vec![f64_type], vec![s32_type; 3]] {
        assert!(Shape::compatible(
            &ctx,
            infer::infer_dynamic_reshape(&ctx, input_type, &sizes, bounds.clone()).unwrap(),
            output,
        ));
    }
    let input = value(&mut ctx, input_type, 0);
    let size = value(&mut ctx, s32_type, 1);
    let float_size = value(&mut ctx, f64_type, 2);
    DynamicReshapeOp::new(&mut ctx, output, vec![input, size, size])
        .verify(&ctx)
        .unwrap();
    assert!(
        DynamicReshapeOp::new(&mut ctx, output, vec![input, size])
            .verify(&ctx)
            .is_err()
    );
    assert!(
        DynamicReshapeOp::new(&mut ctx, output, vec![input, size, float_size],)
            .verify(&ctx)
            .is_err()
    );
    let vector_type = tensor(&ctx, E::S32, &[2]);
    let vector_size = value(&mut ctx, vector_type, 3);
    DynamicReshapeOp::new(&mut ctx, output, vec![input, vector_size, size])
        .verify(&ctx)
        .unwrap();
    let wrong_element = tensor(&ctx, E::S32, &[2, 3]);
    assert!(
        DynamicReshapeOp::new(&mut ctx, wrong_element, vec![input, size, size],)
            .verify(&ctx)
            .is_err()
    );
}

#[test]
fn dynamic_reshape_rejects_unbounded_element_counts_without_guessing() {
    let ctx = Context::new();
    let unbounded = Shape::dynamic_array(&ctx, E::F32, vec![Dimension::Unbounded]).unwrap();
    let input = tensor(&ctx, E::F32, &[6]);
    for result in [
        infer::infer_dynamic_reshape(&ctx, unbounded, &[], vec![Dimension::Static(6)]),
        infer::infer_dynamic_reshape(&ctx, input, &[], vec![Dimension::Unbounded]),
    ] {
        let error = result.unwrap_err();
        assert!(matches!(
            error.err.downcast_ref::<Diagnostic>(),
            Some(Diagnostic::Inference { message }) if message.contains("unbounded")
        ));
    }
}

#[test]
fn reshape_zero_element_factor_groups_preserve_supported_dynamism() {
    let ctx = Context::new();
    for (input, output, expected) in [
        (
            vec![Dimension::Bounded(0)],
            vec![0, 3],
            vec![Dimension::Bounded(0), Dimension::Static(3)],
        ),
        (
            vec![Dimension::Static(0), Dimension::Bounded(6)],
            vec![0],
            vec![Dimension::Bounded(0)],
        ),
        (
            vec![Dimension::Bounded(0), Dimension::Static(6)],
            vec![0, 6],
            vec![Dimension::Bounded(0), Dimension::Static(6)],
        ),
        (
            vec![Dimension::Bounded(1)],
            vec![1, 1],
            vec![Dimension::Static(1), Dimension::Bounded(1)],
        ),
    ] {
        let input = Shape::dynamic_array(&ctx, E::F32, input).unwrap();
        let result = infer::infer_reshape(&ctx, input, &output).unwrap();
        for (axis, expected) in expected.into_iter().enumerate() {
            assert_eq!(Shape::dimension(&ctx, result, axis).unwrap(), expected);
        }
    }
    let combined_and_split = Shape::dynamic_array(
        &ctx,
        E::F32,
        vec![Dimension::Static(0), Dimension::Bounded(6)],
    )
    .unwrap();
    assert!(infer::infer_reshape(&ctx, combined_and_split, &[0, 3]).is_err());
}

#[test]
fn gather_batching_preserves_unsorted_index_batch_mapping_and_dynamic_bounds() {
    let ctx = Context::new();
    let input = tensor(&ctx, E::F32, &[100, 64, 5, 48]);
    let indices = Shape::dynamic_array(
        &ctx,
        E::S64,
        vec![
            Dimension::Bounded(5),
            Dimension::Static(100),
            Dimension::Static(32),
        ],
    )
    .unwrap();
    let dimensions = GatherDimensionNumbers {
        offset_dims: Axes(vec![3]),
        collapsed_slice_dims: Axes(vec![1]),
        operand_batching_dims: Axes(vec![0, 2]),
        start_indices_batching_dims: Axes(vec![1, 0]),
        start_index_map: Axes(vec![1]),
        index_vector_dim: 3,
    };
    let result =
        infer::infer_gather(&ctx, input, indices, &dimensions, &Axes(vec![1, 1, 1, 8])).unwrap();
    assert_eq!(
        Shape::dimensions(&ctx, result).unwrap().as_ref(),
        [5, 100, 32, 8]
    );
    assert_eq!(
        Shape::dimension(&ctx, result, 0).unwrap(),
        Dimension::Bounded(5)
    );
    let mut repeated = dimensions;
    repeated.start_indices_batching_dims = Axes(vec![1, 1]);
    assert!(
        infer::infer_gather(&ctx, input, indices, &repeated, &Axes(vec![1, 1, 1, 8]),).is_err()
    );
}

#[test]
fn scatter_batching_checks_update_windows_and_index_batches() {
    let ctx = Context::new();
    let input = tensor(&ctx, E::F32, &[100, 64, 48]);
    let indices = tensor(&ctx, E::S64, &[100, 32]);
    let dimensions = ScatterDimensionNumbers {
        update_window_dims: Axes(vec![1]),
        inserted_window_dims: Axes(vec![2]),
        input_batching_dims: Axes(vec![0]),
        scatter_indices_batching_dims: Axes(vec![0]),
        scatter_dims_to_operand_dims: Axes(vec![1]),
        index_vector_dim: 2,
    };
    let updates = tensor(&ctx, E::F32, &[100, 64, 32]);
    assert_eq!(
        infer::infer_scatter(&ctx, input, indices, updates, &dimensions).unwrap(),
        input
    );
    for shape in [vec![100, 65, 32], vec![100, 64, 31]] {
        let updates = tensor(&ctx, E::F32, &shape);
        assert!(infer::infer_scatter(&ctx, input, indices, updates, &dimensions).is_err());
    }
}

#[test]
fn batching_metadata_does_not_add_unrequested_shape_or_map_constraints() {
    let ctx = Context::new();
    let input = tensor(&ctx, E::F32, &[7, 64, 9, 48]);
    let indices = tensor(&ctx, E::S64, &[5, 100, 32]);
    let mut dimensions = GatherDimensionNumbers {
        offset_dims: Axes(vec![3]),
        collapsed_slice_dims: Axes(vec![1]),
        operand_batching_dims: Axes(vec![0, 2]),
        start_indices_batching_dims: Axes(vec![1, 0]),
        start_index_map: Axes(vec![0]),
        index_vector_dim: 3,
    };
    for index_batching in [vec![1, 0], vec![3, 0]] {
        dimensions.start_indices_batching_dims = Axes(index_batching);
        let result =
            infer::infer_gather(&ctx, input, indices, &dimensions, &Axes(vec![1, 1, 1, 8]))
                .unwrap();
        assert_eq!(
            Shape::dimensions(&ctx, result).unwrap().as_ref(),
            [5, 100, 32, 8]
        );
    }
    dimensions.start_indices_batching_dims = Axes(vec![0]);
    assert!(
        infer::infer_gather(&ctx, input, indices, &dimensions, &Axes(vec![1, 1, 1, 8]),).is_err()
    );

    let input = tensor(&ctx, E::F32, &[7, 64, 48]);
    let indices = tensor(&ctx, E::S64, &[100, 32]);
    let updates = tensor(&ctx, E::F32, &[100, 64, 32]);
    let mut dimensions = ScatterDimensionNumbers {
        update_window_dims: Axes(vec![1]),
        inserted_window_dims: Axes(vec![2]),
        input_batching_dims: Axes(vec![0]),
        scatter_indices_batching_dims: Axes(vec![0]),
        scatter_dims_to_operand_dims: Axes(vec![0]),
        index_vector_dim: 2,
    };
    for index_batching in [0, 2] {
        dimensions.scatter_indices_batching_dims = Axes(vec![index_batching]);
        assert_eq!(
            infer::infer_scatter(&ctx, input, indices, updates, &dimensions).unwrap(),
            input
        );
    }
    dimensions.scatter_indices_batching_dims = Axes(vec![]);
    assert!(infer::infer_scatter(&ctx, input, indices, updates, &dimensions).is_err());
}

#[test]
fn scatter_reducer_uses_accumulators_first_and_exact_accumulator_result_types() {
    let mut ctx = Context::new();
    xlahlo::register(&mut ctx);
    let f32_input = tensor(&ctx, E::F32, &[5, 4]);
    let s32_input = tensor(&ctx, E::S32, &[5, 4]);
    let bf16_updates = tensor(&ctx, E::BF16, &[3, 4]);
    let s32_updates = tensor(&ctx, E::S32, &[3, 4]);
    let indices = tensor(&ctx, E::S32, &[3, 1]);
    let f32_scalar = scalar(&ctx, E::F32);
    let bf16_scalar = scalar(&ctx, E::BF16);
    let s32_scalar = scalar(&ctx, E::S32);
    let result = Shape::tuple(&ctx, vec![f32_input, s32_input]).unwrap();
    let reducer_result = Shape::tuple(&ctx, vec![f32_scalar, s32_scalar]).unwrap();
    let dimensions = ScatterDimensionNumbers {
        update_window_dims: Axes(vec![1]),
        inserted_window_dims: Axes(vec![0]),
        input_batching_dims: Axes(vec![]),
        scatter_indices_batching_dims: Axes(vec![]),
        scatter_dims_to_operand_dims: Axes(vec![0]),
        index_vector_dim: 1,
    };
    let options = VerifyOptions {
        allow_mixed_precision: true,
        ..Default::default()
    };
    let (module, _) = scatter_graph(
        &mut ctx,
        vec![f32_input, s32_input, indices, bf16_updates, s32_updates],
        result,
        dimensions.clone(),
        &(
            vec![f32_scalar, s32_scalar, bf16_scalar, s32_scalar],
            reducer_result,
        ),
    );
    verify_with_options(&ctx, module.get_operation(), options).unwrap();
    let (reversed, _) = scatter_graph(
        &mut ctx,
        vec![f32_input, s32_input, indices, bf16_updates, s32_updates],
        result,
        dimensions.clone(),
        &(
            vec![bf16_scalar, s32_scalar, f32_scalar, s32_scalar],
            reducer_result,
        ),
    );
    assert!(verify_with_options(&ctx, reversed.get_operation(), options).is_err());
    let tuple_scalar = Shape::tuple(&ctx, vec![f32_scalar]).unwrap();
    let (single, _) = scatter_graph(
        &mut ctx,
        vec![f32_input, indices, bf16_updates],
        f32_input,
        dimensions,
        &(vec![f32_scalar, bf16_scalar], tuple_scalar),
    );
    verify_with_options(&ctx, single.get_operation(), options).unwrap();
}

fn tensor(ctx: &Context, element: E, dims: &[i64]) -> TypeHandle {
    TensorType::get(ctx, element, dims.to_vec()).unwrap()
}

#[test]
fn shape_preserving_movement_inference_accepts_every_primitive_element_format() {
    let ctx = Context::new();
    for &element in E::ALL {
        let matrix = tensor(&ctx, element, &[2, 3]);
        let vector = tensor(&ctx, element, &[3]);
        let results = [
            infer::infer_copy(&ctx, matrix),
            infer::infer_reshape(&ctx, matrix, &[6]),
            infer::infer_transpose(&ctx, matrix, &Axes(vec![1, 0])),
            infer::infer_reverse(&ctx, matrix, &Axes(vec![0, 1])),
            infer::infer_slice(
                &ctx,
                matrix,
                &Slice {
                    starts: Axes(vec![0, 0]),
                    limits: Axes(vec![2, 3]),
                    strides: Axes(vec![1, 1]),
                },
            ),
            infer::infer_concatenate(&ctx, &[matrix, matrix], 0),
            infer::infer_broadcast(&ctx, vector, tensor(&ctx, element, &[2, 3]), &Axes(vec![1])),
        ];
        for result in results {
            let result =
                result.unwrap_or_else(|error| panic!("movement rejected {element:?}: {error}"));
            assert_eq!(Shape::element_type(&ctx, result).unwrap(), element);
        }
    }
}

fn scalar(ctx: &Context, element: E) -> TypeHandle {
    Shape::scalar(ctx, element).unwrap()
}

fn value(ctx: &mut Context, ty: TypeHandle, index: u32) -> Value {
    ParameterOp::new(ctx, ty, vec![], Index(index)).result(ctx)
}

fn flags() -> IndexFlags {
    IndexFlags {
        sorted: false,
        unique: false,
    }
}

fn scatter_graph(
    ctx: &mut Context,
    inputs: Vec<TypeHandle>,
    output: TypeHandle,
    dimensions: ScatterDimensionNumbers,
    reducer: &Signature,
) -> (ModuleOp, ScatterOp) {
    let module = ModuleOp::new(ctx, Identifier::new("scatter_test"));
    xlahlo::ir::set_entry(ctx, module.get_operation(), Identifier::new("main")).unwrap();
    let container = module
        .get_operation()
        .deref(ctx)
        .get_region(0)
        .deref(ctx)
        .get_entry_block()
        .unwrap();
    let reducer_signature = FunctionType::get(ctx, reducer.0.clone(), vec![reducer.1]).into();
    let reducer_op = ComputationOp::new(ctx, Identifier::new("reduce"), reducer_signature);
    reducer_op.get_operation().insert_at_back(container, ctx);
    let reducer_block = BasicBlock::new(ctx, None, vec![]);
    reducer_block.insert_at_back(reducer_op.body(ctx), ctx);
    for (index, &ty) in reducer.0.iter().enumerate() {
        ParameterOp::new(ctx, ty, vec![], Index(index.try_into().unwrap()))
            .get_operation()
            .insert_at_back(reducer_block, ctx);
    }
    let result_types = if Shape::is_tuple(ctx, reducer.1) {
        Shape::children(ctx, reducer.1).to_vec()
    } else {
        vec![reducer.1]
    };
    let results = result_types
        .into_iter()
        .map(|ty| {
            let literal = Literal::from_bits(ctx, ty, &[0]).unwrap();
            let constant = ConstantOp::new(ctx, ty, vec![], literal);
            constant.get_operation().insert_at_back(reducer_block, ctx);
            constant.result(ctx)
        })
        .collect::<Vec<_>>();
    let reducer_result = if Shape::is_tuple(ctx, reducer.1) {
        let tuple = TupleOp::new(ctx, reducer.1, results);
        tuple.get_operation().insert_at_back(reducer_block, ctx);
        tuple.result(ctx)
    } else {
        results[0]
    };
    ReturnOp::new(ctx, reducer_result)
        .get_operation()
        .insert_at_back(reducer_block, ctx);

    let signature = FunctionType::get(ctx, inputs.clone(), vec![output]).into();
    let main = ComputationOp::new(ctx, Identifier::new("main"), signature);
    main.get_operation().insert_at_back(container, ctx);
    let block = BasicBlock::new(ctx, None, vec![]);
    block.insert_at_back(main.body(ctx), ctx);
    let operands = inputs
        .into_iter()
        .enumerate()
        .map(|(index, ty)| {
            let parameter = ParameterOp::new(ctx, ty, vec![], Index(index.try_into().unwrap()));
            parameter.get_operation().insert_at_back(block, ctx);
            parameter.result(ctx)
        })
        .collect();
    let scatter = ScatterOp::new(
        ctx,
        output,
        operands,
        dimensions,
        Callees(vec![Identifier::new("reduce")]),
        flags(),
    );
    scatter.get_operation().insert_at_back(block, ctx);
    ReturnOp::new(ctx, scatter.result(ctx))
        .get_operation()
        .insert_at_back(block, ctx);
    (module, scatter)
}

#[test]
fn broadcast_checks_mapping_and_order_policy() {
    let mut ctx = Context::new();
    let one_dim = tensor(&ctx, E::F32, &[2]);
    let two_dim = tensor(&ctx, E::F32, &[2, 3]);
    let input = value(&mut ctx, one_dim, 0);
    let broadcast = BroadcastOp::new(&mut ctx, two_dim, vec![input], Axes(vec![0]));
    broadcast.verify(&ctx).unwrap();
    assert_eq!(
        infer::infer_broadcast(&ctx, one_dim, two_dim, &Axes(vec![0])).unwrap(),
        two_dim
    );
    assert!(infer::infer_broadcast(&ctx, one_dim, two_dim, &Axes(vec![1])).is_err());

    let unsorted_input_type = tensor(&ctx, E::F32, &[2, 3]);
    let unsorted_output_type = tensor(&ctx, E::F32, &[3, 2]);
    let unsorted_input = value(&mut ctx, unsorted_input_type, 1);
    let unsorted = BroadcastOp::new(
        &mut ctx,
        unsorted_output_type,
        vec![unsorted_input],
        Axes(vec![1, 0]),
    );
    unsorted.verify(&ctx).unwrap();
    assert!(
        unsorted
            .verify_with_options(
                &ctx,
                VerifyOptions {
                    verify_broadcast_dimensions_order: true,
                    ..Default::default()
                },
            )
            .is_err()
    );
}

#[test]
fn reshape_and_dynamic_reshape_check_element_counts() {
    let mut ctx = Context::new();
    let unsorted_input_type = tensor(&ctx, E::F32, &[2, 3]);
    let reshaped_type = tensor(&ctx, E::F32, &[6]);
    let reshape_input = value(&mut ctx, unsorted_input_type, 2);
    let reshape = ReshapeOp::new(&mut ctx, reshaped_type, vec![reshape_input]);
    reshape.verify(&ctx).unwrap();
    let bad_reshape_input = value(&mut ctx, unsorted_input_type, 3);
    let wrong_reshape_type = tensor(&ctx, E::F32, &[5]);
    let wrong_reshape = ReshapeOp::new(&mut ctx, wrong_reshape_type, vec![bad_reshape_input]);
    assert!(wrong_reshape.verify(&ctx).is_err());
    let dynamic_output = Shape::dynamic_array(&ctx, E::F32, vec![Dimension::Bounded(6)]).unwrap();
    let dynamic_input = value(&mut ctx, unsorted_input_type, 4);
    let dynamic_size_type = scalar(&ctx, E::S32);
    let dynamic_size = value(&mut ctx, dynamic_size_type, 5);
    let dynamic_reshape =
        DynamicReshapeOp::new(&mut ctx, dynamic_output, vec![dynamic_input, dynamic_size]);
    dynamic_reshape.verify(&ctx).unwrap();
    let wrong_dynamic_output = tensor(&ctx, E::F32, &[7]);
    let wrong_dynamic = DynamicReshapeOp::new(
        &mut ctx,
        wrong_dynamic_output,
        vec![dynamic_input, dynamic_size],
    );
    assert!(wrong_dynamic.verify(&ctx).is_err());
}

#[test]
fn bitcast_applies_layout_and_shape_size_policies() {
    let mut ctx = Context::new();
    let f32_two = tensor(&ctx, E::F32, &[2]);
    let u8_ninety_nine = tensor(&ctx, E::U8, &[99]);
    let bitcast_input = value(&mut ctx, f32_two, 6);
    let bitcast = BitcastOp::new(&mut ctx, u8_ninety_nine, vec![bitcast_input]);
    bitcast.verify(&ctx).unwrap();
    assert!(
        bitcast
            .verify_with_options(
                &ctx,
                VerifyOptions {
                    layout_sensitive: true,
                    ..Default::default()
                },
            )
            .is_err()
    );
    assert!(
        bitcast
            .verify_with_options(
                &ctx,
                VerifyOptions {
                    layout_sensitive: true,
                    allow_bitcast_to_have_different_size: true,
                    ..Default::default()
                },
            )
            .is_err()
    );
    fn adjusted_size(ctx: &Context, ty: TypeHandle) -> Result<i64> {
        Ok(if Shape::element_type(ctx, ty)? == E::F32 {
            16
        } else {
            8
        })
    }
    let same_data_output = tensor(&ctx, E::U8, &[8]);
    let same_data_input = value(&mut ctx, f32_two, 9);
    let same_data_bitcast = BitcastOp::new(&mut ctx, same_data_output, vec![same_data_input]);
    assert!(
        same_data_bitcast
            .verify_with_options(
                &ctx,
                VerifyOptions {
                    layout_sensitive: true,
                    shape_size: Some(adjusted_size),
                    ..Default::default()
                },
            )
            .is_err()
    );
    assert!(
        same_data_bitcast
            .verify_with_options(
                &ctx,
                VerifyOptions {
                    layout_sensitive: true,
                    allow_bitcast_to_have_different_size: true,
                    shape_size: Some(adjusted_size),
                    ..Default::default()
                },
            )
            .is_ok()
    );
}

#[test]
fn bitcast_size_relaxation_accepts_equal_data_with_different_tail_padding() {
    fn allocation_size(ctx: &xlahlo::pliron::context::Context, shape: TypeHandle) -> Result<i64> {
        let count = Shape::element_count(ctx, shape)?;
        let layout = Shape::layout(ctx, shape)?.unwrap();
        let alignment = layout.tail_padding_alignment_in_elements.max(1);
        let elements = (count / alignment + i64::from(count % alignment != 0)) * alignment;
        let element_bits = if layout.element_size_in_bits == 0 {
            i64::from(Shape::element_type(ctx, shape)?.bit_width())
        } else {
            layout.element_size_in_bits
        };
        let bits = elements * element_bits;
        Ok(bits / 8 + i64::from(bits % 8 != 0))
    }

    let mut ctx = Context::new();
    let input = Shape::array(&ctx, E::F32, vec![2]).unwrap();
    let mut input_layout = Layout::descending(1);
    input_layout.tail_padding_alignment_in_elements = 4;
    let input = Shape::with_layout(&ctx, input, input_layout).unwrap();
    let output = Shape::array(&ctx, E::U8, vec![8]).unwrap();
    let output = Shape::with_default_layout(&ctx, output).unwrap();
    assert_eq!(Shape::array_data_size(&ctx, input).unwrap(), 8);
    assert_eq!(Shape::array_data_size(&ctx, output).unwrap(), 8);
    let value = value(&mut ctx, input, 0);
    let bitcast = BitcastOp::new(&mut ctx, output, vec![value]);
    let options = VerifyOptions {
        layout_sensitive: true,
        shape_size: Some(allocation_size),
        ..Default::default()
    };
    assert!(bitcast.verify_with_options(&ctx, options).is_err());
    assert!(
        bitcast
            .verify_with_options(
                &ctx,
                VerifyOptions {
                    allow_bitcast_to_have_different_size: true,
                    ..options
                },
            )
            .is_ok()
    );
}

#[test]
fn transpose_and_reverse_validate_axes() {
    let mut ctx = Context::new();
    let unsorted_input_type = tensor(&ctx, E::F32, &[2, 3]);
    let transposed_type = tensor(&ctx, E::F32, &[3, 2]);
    let transpose_input = value(&mut ctx, unsorted_input_type, 7);
    let transpose = TransposeOp::new(
        &mut ctx,
        transposed_type,
        vec![transpose_input],
        Axes(vec![1, 0]),
    );
    transpose.verify(&ctx).unwrap();
    assert!(infer::infer_transpose(&ctx, unsorted_input_type, &Axes(vec![0, 0])).is_err());
    let reverse_input = value(&mut ctx, unsorted_input_type, 8);
    let reverse = ReverseOp::new(
        &mut ctx,
        unsorted_input_type,
        vec![reverse_input],
        Axes(vec![1]),
    );
    reverse.verify(&ctx).unwrap();
    assert!(infer::infer_reverse(&ctx, unsorted_input_type, &Axes(vec![2])).is_err());
}

#[test]
fn slice_and_dynamic_slice_validate_geometry_and_indices() {
    let mut ctx = Context::new();
    let input_type = tensor(&ctx, E::F32, &[4, 5]);
    let index_type = scalar(&ctx, E::S32);
    let input = value(&mut ctx, input_type, 0);
    let geometry = Slice {
        starts: Axes(vec![1, 1]),
        limits: Axes(vec![4, 5]),
        strides: Axes(vec![2, 2]),
    };
    let slice_result = tensor(&ctx, E::F32, &[2, 2]);
    SliceOp::new(&mut ctx, slice_result, vec![input], geometry)
        .verify(&ctx)
        .unwrap();
    let invalid_slice = Slice {
        starts: Axes(vec![0, 0]),
        limits: Axes(vec![5, 5]),
        strides: Axes(vec![1, 1]),
    };
    assert!(infer::infer_slice(&ctx, input_type, &invalid_slice).is_err());

    let dynamic_result = tensor(&ctx, E::F32, &[2, 3]);
    let index0 = value(&mut ctx, index_type, 1);
    let index1 = value(&mut ctx, index_type, 2);
    let dynamic_slice = DynamicSliceOp::new(
        &mut ctx,
        dynamic_result,
        vec![input, index0, index1],
        Axes(vec![2, 3]),
    );
    dynamic_slice.verify(&ctx).unwrap();
    let float_index_type = scalar(&ctx, E::F32);
    let float_index = value(&mut ctx, float_index_type, 3);
    let index2 = value(&mut ctx, index_type, 4);
    let invalid_dynamic = DynamicSliceOp::new(
        &mut ctx,
        dynamic_result,
        vec![input, float_index, index2],
        Axes(vec![2, 3]),
    );
    assert!(invalid_dynamic.verify(&ctx).is_err());
}

#[test]
fn dynamic_update_slice_checks_update_bounds() {
    let mut ctx = Context::new();
    let input_type = tensor(&ctx, E::F32, &[4, 5]);
    let input = value(&mut ctx, input_type, 0);
    let index_type = scalar(&ctx, E::S32);
    let update_type = tensor(&ctx, E::F32, &[2, 3]);
    let update = value(&mut ctx, update_type, 5);
    let index3 = value(&mut ctx, index_type, 6);
    let index4 = value(&mut ctx, index_type, 7);
    let dynamic_update =
        DynamicUpdateSliceOp::new(&mut ctx, input_type, vec![input, update, index3, index4]);
    dynamic_update.verify(&ctx).unwrap();
    let oversize_type = tensor(&ctx, E::F32, &[5, 3]);
    let oversize_update = value(&mut ctx, oversize_type, 8);
    let index5 = value(&mut ctx, index_type, 9);
    let index6 = value(&mut ctx, index_type, 10);
    let invalid_update = DynamicUpdateSliceOp::new(
        &mut ctx,
        input_type,
        vec![input, oversize_update, index5, index6],
    );
    assert!(invalid_update.verify(&ctx).is_err());
}

#[test]
fn concatenate_checks_non_axis_dimensions() {
    let mut ctx = Context::new();
    let lhs = tensor(&ctx, E::F32, &[2, 3]);
    let rhs = tensor(&ctx, E::F32, &[4, 3]);
    let concatenated = tensor(&ctx, E::F32, &[6, 3]);
    let concat_lhs = value(&mut ctx, lhs, 11);
    let concat_rhs = value(&mut ctx, rhs, 12);
    ConcatenateOp::new(
        &mut ctx,
        concatenated,
        vec![concat_lhs, concat_rhs],
        Index(0),
    )
    .verify(&ctx)
    .unwrap();
    assert!(infer::infer_concatenate(&ctx, &[lhs, rhs], 1).is_err());
}

#[test]
fn pad_checks_edge_and_interior_geometry() {
    let mut ctx = Context::new();
    let input_type = tensor(&ctx, E::F32, &[4, 5]);
    let input = value(&mut ctx, input_type, 0);
    let lhs = tensor(&ctx, E::F32, &[2, 3]);
    let pad_config = PaddingConfig(vec![
        PaddingConfigDimension {
            edge_padding_low: 1,
            edge_padding_high: 1,
            interior_padding: 0,
        },
        PaddingConfigDimension {
            edge_padding_low: 0,
            edge_padding_high: 0,
            interior_padding: 0,
        },
    ]);
    let padded = tensor(&ctx, E::F32, &[6, 5]);
    let pad_value_type = scalar(&ctx, E::F32);
    let pad_value = value(&mut ctx, pad_value_type, 13);
    PadOp::new(&mut ctx, padded, vec![input, pad_value], pad_config)
        .verify(&ctx)
        .unwrap();
    let invalid_pad = PaddingConfig(vec![
        PaddingConfigDimension {
            edge_padding_low: 0,
            edge_padding_high: 0,
            interior_padding: -1,
        },
        PaddingConfigDimension {
            edge_padding_low: 0,
            edge_padding_high: 0,
            interior_padding: 0,
        },
    ]);
    let pad_scalar = scalar(&ctx, E::F32);
    assert!(infer::infer_pad(&ctx, lhs, pad_scalar, &invalid_pad).is_err());
}

#[test]
fn iota_requires_a_non_scalar_numeric_result() {
    let mut ctx = Context::new();
    let iota_result = tensor(&ctx, E::S32, &[2, 3]);
    IotaOp::new(&mut ctx, iota_result, vec![], Index(1))
        .verify(&ctx)
        .unwrap();
    let iota_scalar = scalar(&ctx, E::S32);
    assert!(infer::infer_iota(&ctx, iota_scalar, 0).is_err());
}

#[test]
fn gather_checks_index_and_slice_partitions() {
    let mut ctx = Context::new();
    let input = tensor(&ctx, E::F32, &[5, 4]);
    let indices = tensor(&ctx, E::S32, &[3, 1]);
    let gather_dimensions = GatherDimensionNumbers {
        offset_dims: Axes(vec![1]),
        collapsed_slice_dims: Axes(vec![0]),
        operand_batching_dims: Axes(vec![]),
        start_indices_batching_dims: Axes(vec![]),
        start_index_map: Axes(vec![0]),
        index_vector_dim: 1,
    };
    let gather_result = tensor(&ctx, E::F32, &[3, 4]);
    let gather_input = value(&mut ctx, input, 0);
    let gather_indices = value(&mut ctx, indices, 1);
    GatherOp::new(
        &mut ctx,
        gather_result,
        vec![gather_input, gather_indices],
        gather_dimensions.clone(),
        Axes(vec![1, 4]),
        flags(),
    )
    .verify(&ctx)
    .unwrap();
    assert!(
        infer::infer_gather(&ctx, input, indices, &gather_dimensions, &Axes(vec![1, 5])).is_err()
    );
}

#[test]
fn scatter_graph_requires_a_scalar_reducer() {
    let mut ctx = Context::new();
    xlahlo::register(&mut ctx);
    let input = tensor(&ctx, E::F32, &[5, 4]);
    let indices = tensor(&ctx, E::S32, &[3, 1]);
    let updates = tensor(&ctx, E::F32, &[3, 4]);
    let scatter_dimensions = ScatterDimensionNumbers {
        update_window_dims: Axes(vec![1]),
        inserted_window_dims: Axes(vec![0]),
        input_batching_dims: Axes(vec![]),
        scatter_indices_batching_dims: Axes(vec![]),
        scatter_dims_to_operand_dims: Axes(vec![0]),
        index_vector_dim: 1,
    };
    let scatter_input = value(&mut ctx, input, 2);
    let scatter_indices = value(&mut ctx, indices, 3);
    let scatter_updates = value(&mut ctx, updates, 4);
    let scatter = ScatterOp::new(
        &mut ctx,
        input,
        vec![scatter_input, scatter_indices, scatter_updates],
        scatter_dimensions.clone(),
        Callees(vec![Identifier::new("reduce")]),
        flags(),
    );
    scatter.verify(&ctx).unwrap();
    let scalar_f32 = scalar(&ctx, E::F32);
    infer::infer_variadic_scatter(&ctx, &[input], indices, &[updates], &scatter_dimensions)
        .unwrap();
    let (module, graph_scatter) = scatter_graph(
        &mut ctx,
        vec![input, indices, updates],
        input,
        scatter_dimensions.clone(),
        &(vec![scalar_f32, scalar_f32], scalar_f32),
    );
    verify(&ctx, module.get_operation()).unwrap();
    graph_scatter
        .get_operation()
        .deref_mut(&ctx)
        .attributes
        .set(Identifier::new("hlo_callees"), Callees(vec![]));
    assert!(verify(&ctx, module.get_operation()).is_err());
    let invalid_scatter_input = value(&mut ctx, input, 5);
    let invalid_scatter_indices = value(&mut ctx, indices, 6);
    let invalid_scatter = ScatterOp::new(
        &mut ctx,
        input,
        vec![invalid_scatter_input, invalid_scatter_indices],
        scatter_dimensions,
        Callees(vec![]),
        flags(),
    );
    assert!(invalid_scatter.verify(&ctx).is_err());
}

#[test]
fn copy_preserves_shape() {
    let mut ctx = Context::new();
    let input = tensor(&ctx, E::F32, &[5, 4]);
    let copy_input = value(&mut ctx, input, 7);
    CopyOp::new(&mut ctx, input, vec![copy_input])
        .verify(&ctx)
        .unwrap();
    let wrong_copy_input_type = tensor(&ctx, E::F32, &[5, 4]);
    let wrong_copy_output_type = tensor(&ctx, E::F32, &[6, 4]);
    let wrong_copy_input = value(&mut ctx, wrong_copy_input_type, 12);
    assert!(
        CopyOp::new(&mut ctx, wrong_copy_output_type, vec![wrong_copy_input])
            .verify(&ctx)
            .is_err()
    );
}

#[test]
fn dimension_size_operations_check_axes_and_scalar_types() {
    let mut ctx = Context::new();
    let input = tensor(&ctx, E::F32, &[5, 4]);
    let dimension_size = scalar(&ctx, E::S32);
    let dimension_input = value(&mut ctx, input, 8);
    GetDimensionSizeOp::new(&mut ctx, dimension_size, vec![dimension_input], Index(1))
        .verify(&ctx)
        .unwrap();
    let bad_dimension_input = value(&mut ctx, input, 13);
    assert!(
        GetDimensionSizeOp::new(
            &mut ctx,
            dimension_size,
            vec![bad_dimension_input],
            Index(2)
        )
        .verify(&ctx)
        .is_err()
    );
    assert!(infer::infer_get_dimension_size(&ctx, input, 2).is_err());
    let bounded = Shape::dynamic_array(
        &ctx,
        E::F32,
        vec![Dimension::Static(5), Dimension::Bounded(4)],
    )
    .unwrap();
    let set_input = value(&mut ctx, input, 9);
    let set_size_type = scalar(&ctx, E::S32);
    let set_size = value(&mut ctx, set_size_type, 10);
    SetDimensionSizeOp::new(&mut ctx, bounded, vec![set_input, set_size], Index(1))
        .verify(&ctx)
        .unwrap();
    let invalid_set_size = scalar(&ctx, E::F32);
    assert!(infer::infer_set_dimension_size(&ctx, input, invalid_set_size, 1).is_err());
}

#[test]
fn shuffle_checks_axes_and_shift_count() {
    let mut ctx = Context::new();
    let input = tensor(&ctx, E::F32, &[5, 4]);
    let shuffle_input = value(&mut ctx, input, 11);
    let shuffle = ShuffleOp::new(
        &mut ctx,
        input,
        vec![shuffle_input],
        Axes(vec![1]),
        ShuffleMode {
            shifts: Axes(vec![2]),
        },
    );
    shuffle.verify(&ctx).unwrap();
    assert!(
        infer::infer_shuffle(
            &ctx,
            input,
            &Axes(vec![1]),
            &ShuffleMode {
                shifts: Axes(vec![])
            }
        )
        .is_err()
    );
    let bad_shuffle_input = value(&mut ctx, input, 14);
    assert!(
        ShuffleOp::new(
            &mut ctx,
            input,
            vec![bad_shuffle_input],
            Axes(vec![2]),
            ShuffleMode {
                shifts: Axes(vec![0]),
            },
        )
        .verify(&ctx)
        .is_err()
    );
}

#[test]
fn dynamic_flags_buffer_layout_and_bounded_inference_are_preserved() {
    let ctx = Context::new();
    let bounded = Shape::dynamic_array(
        &ctx,
        E::F32,
        vec![Dimension::Static(2), Dimension::Bounded(3)],
    )
    .unwrap();
    let size = Shape::scalar(&ctx, E::S32).unwrap();
    let updated = infer::infer_dynamic_update_slice(&ctx, bounded, bounded, &[size, size]).unwrap();
    assert_eq!(
        Shape::dimension(&ctx, updated, 1).unwrap(),
        Dimension::Bounded(3)
    );

    let buffered = Shape::buffer(&ctx, bounded).unwrap();
    let dynamic = infer::infer_dynamic_reshape(
        &ctx,
        buffered,
        &[Shape::token(&ctx)],
        vec![Dimension::Bounded(6)],
    )
    .unwrap();
    assert_eq!(Shape::rank(&ctx, dynamic).unwrap(), 1);
    let buffer_update =
        infer::infer_dynamic_update_slice(&ctx, buffered, buffered, &[size, size]).unwrap();
    assert!(Shape::is_array_or_buffer(&ctx, buffer_update));
    assert_eq!(
        Shape::dimension(&ctx, Shape::array_view(&ctx, buffer_update).unwrap(), 1).unwrap(),
        Dimension::Bounded(3)
    );

    let nondefault = TensorType::with_layout(&ctx, E::F32, vec![2, 3], vec![0, 1]).unwrap();
    let transposed = infer::infer_transpose(&ctx, nondefault, &Axes(vec![1, 0])).unwrap();
    assert_eq!(
        Shape::layout(&ctx, transposed)
            .unwrap()
            .unwrap()
            .minor_to_major,
        [1, 0]
    );
    assert!(
        infer::infer_dynamic_slice(
            &ctx,
            bounded,
            &[Shape::scalar(&ctx, E::F32).unwrap(); 2],
            &Axes(vec![1, 2]),
        )
        .is_err()
    );
}

#[test]
fn broadcast_inference_checks_bounded_and_unbounded_dimensions() {
    let ctx = Context::new();
    let bounded = Shape::dynamic_array(
        &ctx,
        E::F32,
        vec![Dimension::Bounded(4), Dimension::Static(3)],
    )
    .unwrap();
    let static_output = Shape::dynamic_array(
        &ctx,
        E::U8,
        vec![Dimension::Bounded(4), Dimension::Static(3)],
    )
    .unwrap();
    assert_eq!(
        infer::infer_broadcast(&ctx, bounded, static_output, &Axes(vec![0, 1])).unwrap(),
        static_output,
    );
    assert!(
        infer::infer_broadcast(
            &ctx,
            bounded,
            Shape::dynamic_array(
                &ctx,
                E::F32,
                vec![Dimension::Unbounded, Dimension::Static(3)]
            )
            .unwrap(),
            &Axes(vec![0, 1]),
        )
        .is_err()
    );
}

#[test]
fn static_reshape_reports_unbounded_results_as_inference_errors_and_recovers() {
    let ctx = Context::new();
    let unbounded_size = xlahlo::shape::UNBOUNDED_SIZE;
    for input in [
        tensor(&ctx, E::F32, &[6]),
        Shape::dynamic_array(&ctx, E::F32, vec![Dimension::Bounded(6)]).unwrap(),
        Shape::dynamic_array(&ctx, E::F32, vec![Dimension::Unbounded]).unwrap(),
    ] {
        let before = Shape::dimension(&ctx, input, 0).unwrap();
        for dimensions in [
            vec![unbounded_size],
            vec![2, unbounded_size],
            vec![unbounded_size, 0],
        ] {
            for result in [
                infer::infer_reshape(&ctx, input, &dimensions),
                infer::infer_static_reshape(&ctx, input, &dimensions),
                infer::infer_reshape_with_dimension(&ctx, input, &dimensions, Some(0)),
            ] {
                let error = result.unwrap_err();
                let Some(Diagnostic::Inference { message }) =
                    error.err.downcast_ref::<Diagnostic>()
                else {
                    panic!("expected a reshape inference error, got {error:?}");
                };
                assert_eq!(
                    message,
                    "reshaping with an unbounded result shape is not supported"
                );
                assert_eq!(Shape::dimension(&ctx, input, 0).unwrap(), before);
            }
        }
        let recovered = infer::infer_reshape(&ctx, input, &[2, 3]).unwrap();
        assert_eq!(Shape::dimensions(&ctx, recovered).unwrap().as_ref(), [2, 3]);
        assert!(Shape::layout(&ctx, recovered).unwrap().is_some());
        assert_eq!(Shape::dimension(&ctx, input, 0).unwrap(), before);
    }
}

#[test]
fn reshape_propagates_dynamic_factors_and_validates_inferred_axis() {
    let ctx = Context::new();
    for (input_dims, output_dims, inferred, expected) in [
        (
            vec![Dimension::Bounded(6)],
            vec![2, 3],
            None,
            vec![Dimension::Bounded(2), Dimension::Static(3)],
        ),
        (
            vec![Dimension::Bounded(6)],
            vec![2, 3],
            Some(1),
            vec![Dimension::Static(2), Dimension::Bounded(3)],
        ),
        (
            vec![Dimension::Static(2), Dimension::Bounded(3)],
            vec![6],
            None,
            vec![Dimension::Bounded(6)],
        ),
        (
            vec![Dimension::Bounded(2), Dimension::Bounded(3)],
            vec![6],
            None,
            vec![Dimension::Bounded(6)],
        ),
        (
            vec![Dimension::Bounded(2), Dimension::Static(3)],
            vec![2, 3],
            None,
            vec![Dimension::Bounded(2), Dimension::Static(3)],
        ),
    ] {
        let input = Shape::dynamic_array(&ctx, E::F32, input_dims).unwrap();
        let actual =
            infer::infer_reshape_with_dimension(&ctx, input, &output_dims, inferred).unwrap();
        for (axis, expected) in expected.into_iter().enumerate() {
            assert_eq!(Shape::dimension(&ctx, actual, axis).unwrap(), expected);
        }
        assert!(Shape::layout(&ctx, actual).unwrap().is_some());
    }
    let unbounded = Shape::dynamic_array(&ctx, E::F32, vec![Dimension::Unbounded]).unwrap();
    let static_reshape = infer::infer_static_reshape(&ctx, unbounded, &[2, 3]).unwrap();
    assert_eq!(
        Shape::dimensions(&ctx, static_reshape).unwrap().as_ref(),
        [2, 3]
    );
    assert!(
        infer::infer_static_reshape(
            &ctx,
            Shape::dynamic_array(
                &ctx,
                E::F32,
                vec![Dimension::Unbounded, Dimension::Bounded(4)],
            )
            .unwrap(),
            &[3],
        )
        .is_err()
    );
    assert!(
        infer::infer_static_reshape(
            &ctx,
            Shape::dynamic_array(&ctx, E::F32, vec![Dimension::Bounded(6)]).unwrap(),
            &[7],
        )
        .is_err()
    );
    assert!(
        infer::infer_reshape_with_dimension(
            &ctx,
            Shape::dynamic_array(&ctx, E::F32, vec![Dimension::Bounded(6)]).unwrap(),
            &[2, 3],
            Some(3),
        )
        .is_err()
    );
}

#[test]
fn pad_inference_checks_extent_combinations_and_dynamic_bounds() {
    let ctx = Context::new();
    let unbounded = Shape::dynamic_array(&ctx, E::F32, vec![Dimension::Unbounded]).unwrap();
    let value = scalar(&ctx, E::F32);
    for size in 0..5 {
        let input = tensor(&ctx, E::F32, &[size]);
        for low in -3..4 {
            for high in -3..4 {
                for interior in 0..4 {
                    let expected = size + low + high + (size - 1).max(0) * interior;
                    let config = PaddingConfig(vec![PaddingConfigDimension {
                        edge_padding_low: low,
                        edge_padding_high: high,
                        interior_padding: interior,
                    }]);
                    let result = infer::infer_pad(&ctx, input, value, &config);
                    if expected < 0 {
                        assert!(result.is_err());
                    } else {
                        assert_eq!(
                            Shape::dimensions(&ctx, result.unwrap()).unwrap().as_ref(),
                            [expected]
                        );
                    }
                }
            }
        }
    }
    let dynamic_input = Shape::dynamic_array(&ctx, E::F32, vec![Dimension::Bounded(4)]).unwrap();
    let dynamic_pad = infer::infer_pad(
        &ctx,
        dynamic_input,
        value,
        &PaddingConfig(vec![PaddingConfigDimension {
            edge_padding_low: 1,
            edge_padding_high: 2,
            interior_padding: 1,
        }]),
    )
    .unwrap();
    assert_eq!(
        Shape::dimension(&ctx, dynamic_pad, 0).unwrap(),
        Dimension::Bounded(10)
    );
    let unbounded_pad = infer::infer_pad(
        &ctx,
        unbounded,
        value,
        &PaddingConfig(vec![PaddingConfigDimension {
            edge_padding_low: 1,
            edge_padding_high: 2,
            interior_padding: 1,
        }]),
    )
    .unwrap();
    assert_eq!(
        Shape::dimension(&ctx, unbounded_pad, 0).unwrap(),
        Dimension::Unbounded
    );
}

#[test]
fn concatenate_inference_checks_extent_combinations_and_dynamic_bounds() {
    let ctx = Context::new();
    for left in 0..5 {
        for right in 0..5 {
            let a = tensor(&ctx, E::F32, &[2, left]);
            let b = tensor(&ctx, E::F32, &[2, right]);
            let joined = infer::infer_concatenate(&ctx, &[a, b], 1).unwrap();
            assert_eq!(
                Shape::dimensions(&ctx, joined).unwrap().as_ref(),
                [2, left + right]
            );
        }
    }
    let mixed = infer::infer_concatenate(
        &ctx,
        &[
            Shape::dynamic_array(
                &ctx,
                E::F32,
                vec![Dimension::Unbounded, Dimension::Bounded(3)],
            )
            .unwrap(),
            tensor(&ctx, E::F32, &[2, 4]),
        ],
        1,
    )
    .unwrap();
    assert_eq!(
        Shape::dimension(&ctx, mixed, 0).unwrap(),
        Dimension::Static(2)
    );
    assert_eq!(
        Shape::dimension(&ctx, mixed, 1).unwrap(),
        Dimension::Bounded(7)
    );
}

#[test]
fn dynamic_reshape_pad_and_indexing_validate_bounds() {
    let ctx = Context::new();
    let input = tensor(&ctx, E::F32, &[6]);
    let scalar_size = scalar(&ctx, E::F64);
    let bounds = vec![Dimension::Bounded(6)];
    for size in [
        scalar_size,
        Shape::token(&ctx),
        Shape::buffer(&ctx, scalar(&ctx, E::F64)).unwrap(),
        tensor(&ctx, E::S32, &[1]),
    ] {
        assert!(Shape::compatible(
            &ctx,
            infer::infer_dynamic_reshape(&ctx, input, &[size], bounds.clone()).unwrap(),
            Shape::dynamic_array(&ctx, E::F32, bounds.clone()).unwrap()
        ));
    }
    assert!(
        infer::infer_dynamic_reshape(&ctx, input, &[tensor(&ctx, E::F64, &[1])], bounds.clone())
            .is_err()
    );
    assert!(
        infer::infer_dynamic_reshape(
            &ctx,
            input,
            &[scalar(&ctx, E::S32)],
            vec![Dimension::Bounded(7)]
        )
        .is_err()
    );
    let buffered_input = Shape::buffer(&ctx, input).unwrap();
    let reshaped =
        infer::infer_dynamic_reshape(&ctx, buffered_input, &[], vec![Dimension::Static(6)])
            .unwrap();
    assert!(Shape::is_array(&ctx, reshaped));

    let unbounded = Shape::dynamic_array(&ctx, E::F32, vec![Dimension::Unbounded]).unwrap();
    let unbounded_reshape = infer::infer_static_reshape(&ctx, unbounded, &[3]).unwrap();
    assert!(Shape::layout(&ctx, unbounded_reshape).unwrap().is_some());

    let data = tensor(&ctx, E::F32, &[2]);
    let pad_value = Shape::buffer(&ctx, scalar(&ctx, E::F32)).unwrap();
    let config = PaddingConfig(vec![PaddingConfigDimension {
        edge_padding_low: 1,
        edge_padding_high: 2,
        interior_padding: 1,
    }]);
    assert_eq!(
        Shape::dimensions(
            &ctx,
            infer::infer_pad(&ctx, data, pad_value, &config).unwrap()
        )
        .unwrap()
        .as_ref(),
        [6]
    );
    let invalid_config = PaddingConfig(vec![
        PaddingConfigDimension {
            edge_padding_low: 0,
            edge_padding_high: i64::MAX,
            interior_padding: 0,
        },
        PaddingConfigDimension {
            edge_padding_low: 0,
            edge_padding_high: 0,
            interior_padding: -1,
        },
    ]);
    let two_dimensional = tensor(&ctx, E::F32, &[2, 2]);
    let error =
        infer::infer_pad(&ctx, two_dimensional, scalar(&ctx, E::F32), &invalid_config).unwrap_err();
    assert!(error.to_string().contains("negative interior padding"));

    let gather_dimensions = GatherDimensionNumbers {
        offset_dims: Axes(vec![1]),
        collapsed_slice_dims: Axes(vec![0]),
        operand_batching_dims: Axes(vec![]),
        start_indices_batching_dims: Axes(vec![]),
        start_index_map: Axes(vec![0]),
        index_vector_dim: 1,
    };
    let gather_input = Shape::dynamic_array(
        &ctx,
        E::F32,
        vec![Dimension::Static(5), Dimension::Bounded(4)],
    )
    .unwrap();
    let gather_indices = Shape::dynamic_array(&ctx, E::U32, vec![Dimension::Bounded(3)]).unwrap();
    let gathered = infer::infer_gather(
        &ctx,
        gather_input,
        gather_indices,
        &gather_dimensions,
        &Axes(vec![1, 4]),
    )
    .unwrap();
    assert_eq!(
        Shape::dimension(&ctx, gathered, 0).unwrap(),
        Dimension::Bounded(3)
    );
    assert_eq!(
        Shape::dimension(&ctx, gathered, 1).unwrap(),
        Dimension::Bounded(4)
    );
    assert!(
        infer::infer_gather(
            &ctx,
            gather_input,
            gather_indices,
            &GatherDimensionNumbers {
                start_index_map: Axes(vec![0, 0]),
                ..gather_dimensions.clone()
            },
            &Axes(vec![1, 4])
        )
        .is_err()
    );

    let unbounded_indices = Shape::dynamic_array(
        &ctx,
        E::S32,
        vec![Dimension::Unbounded, Dimension::Static(1)],
    )
    .unwrap();
    let update = Shape::dynamic_array(
        &ctx,
        E::F32,
        vec![Dimension::Unbounded, Dimension::Static(4)],
    )
    .unwrap();
    let scatter_dimensions = ScatterDimensionNumbers {
        update_window_dims: Axes(vec![1]),
        inserted_window_dims: Axes(vec![0]),
        input_batching_dims: Axes(vec![]),
        scatter_indices_batching_dims: Axes(vec![]),
        scatter_dims_to_operand_dims: Axes(vec![0]),
        index_vector_dim: 1,
    };
    let scattered = infer::infer_scatter(
        &ctx,
        tensor(&ctx, E::F32, &[5, 4]),
        unbounded_indices,
        update,
        &scatter_dimensions,
    )
    .unwrap();
    assert_eq!(Shape::dimensions(&ctx, scattered).unwrap().as_ref(), [5, 4]);
}

#[test]
fn variadic_scatter_graph_checks_reducer_signature() {
    let mut ctx = Context::new();
    xlahlo::register(&mut ctx);
    let f32_input = tensor(&ctx, E::F32, &[5, 4]);
    let s32_input = tensor(&ctx, E::S32, &[5, 4]);
    let indices = tensor(&ctx, E::S32, &[3, 1]);
    let f32_updates = tensor(&ctx, E::F32, &[3, 4]);
    let s32_updates = tensor(&ctx, E::S32, &[3, 4]);
    let dimensions = ScatterDimensionNumbers {
        update_window_dims: Axes(vec![1]),
        inserted_window_dims: Axes(vec![0]),
        input_batching_dims: Axes(vec![]),
        scatter_indices_batching_dims: Axes(vec![]),
        scatter_dims_to_operand_dims: Axes(vec![0]),
        index_vector_dim: 1,
    };
    let result = infer::infer_variadic_scatter(
        &ctx,
        &[f32_input, s32_input],
        indices,
        &[f32_updates, s32_updates],
        &dimensions,
    )
    .unwrap();
    assert!(Shape::is_tuple(&ctx, result));

    let scalar_f32 = scalar(&ctx, E::F32);
    let scalar_s32 = scalar(&ctx, E::S32);
    let reducer = (
        vec![scalar_f32, scalar_s32, scalar_f32, scalar_s32],
        Shape::tuple(&ctx, vec![scalar_f32, scalar_s32]).unwrap(),
    );
    let inputs = vec![f32_input, s32_input, indices, f32_updates, s32_updates];
    let (module, _) = scatter_graph(
        &mut ctx,
        inputs.clone(),
        result,
        dimensions.clone(),
        &reducer,
    );
    verify(&ctx, module.get_operation()).unwrap();
    let (invalid_module, _) = scatter_graph(
        &mut ctx,
        inputs,
        result,
        dimensions,
        &(vec![scalar_f32; 4], reducer.1),
    );
    assert!(verify(&ctx, invalid_module.get_operation()).is_err());
}

#[test]
fn movement_verifiers_apply_element_type_and_mixed_precision_policies() {
    let mut ctx = Context::new();
    let input = tensor(&ctx, E::F32, &[2]);
    let target = tensor(&ctx, E::S32, &[2, 3]);
    let broadcast_input = value(&mut ctx, input, 5);
    let broadcast = BroadcastOp::new(&mut ctx, target, vec![broadcast_input], Axes(vec![0]));
    assert!(broadcast.verify(&ctx).is_err());

    let pad_data_type = tensor(&ctx, E::F32, &[2]);
    let pad_output_type = tensor(&ctx, E::F32, &[4]);
    let pad_data = value(&mut ctx, pad_data_type, 6);
    let narrow_pad_type = Shape::buffer(&ctx, scalar(&ctx, E::BF16)).unwrap();
    let narrow_pad_value = value(&mut ctx, narrow_pad_type, 7);
    let pad = PadOp::new(
        &mut ctx,
        pad_output_type,
        vec![pad_data, narrow_pad_value],
        PaddingConfig(vec![PaddingConfigDimension {
            edge_padding_low: 1,
            edge_padding_high: 1,
            interior_padding: 0,
        }]),
    );
    assert!(pad.verify(&ctx).is_err());
    assert!(
        pad.verify_with_options(
            &ctx,
            VerifyOptions {
                allow_mixed_precision: true,
                ..Default::default()
            },
        )
        .is_ok()
    );

    let concat_left_type = tensor(&ctx, E::F32, &[2]);
    let concat_right_type = tensor(&ctx, E::BF16, &[2]);
    let concat_output_type = tensor(&ctx, E::F32, &[4]);
    let concat_left = value(&mut ctx, concat_left_type, 8);
    let concat_right = value(&mut ctx, concat_right_type, 9);
    let concat = ConcatenateOp::new(
        &mut ctx,
        concat_output_type,
        vec![concat_left, concat_right],
        Index(0),
    );
    assert!(concat.verify(&ctx).is_err());
    assert!(
        concat
            .verify_with_options(
                &ctx,
                VerifyOptions {
                    allow_mixed_precision: true,
                    ..Default::default()
                },
            )
            .is_ok()
    );

    let update_input_type = tensor(&ctx, E::F32, &[4]);
    let narrow_update_type = tensor(&ctx, E::BF16, &[2]);
    let update_index_type = scalar(&ctx, E::S32);
    let update_input = value(&mut ctx, update_input_type, 10);
    let narrow_update = value(&mut ctx, narrow_update_type, 11);
    let update_index = value(&mut ctx, update_index_type, 12);
    let dynamic_update = DynamicUpdateSliceOp::new(
        &mut ctx,
        update_input_type,
        vec![update_input, narrow_update, update_index],
    );
    assert!(dynamic_update.verify(&ctx).is_err());
    assert!(
        dynamic_update
            .verify_with_options(
                &ctx,
                VerifyOptions {
                    allow_mixed_precision: true,
                    ..Default::default()
                },
            )
            .is_ok()
    );

    assert!(
        infer::infer_shuffle(
            &ctx,
            input,
            &Axes(vec![0]),
            &ShuffleMode {
                shifts: Axes(vec![0, 1])
            }
        )
        .is_err()
    );
}

#[test]
fn bitcast_host_device_transitions_require_a_constant_origin() {
    let mut ctx = Context::new();
    let mut host_layout = Layout::descending(1);
    host_layout.memory_space = 0;
    let input_type = Shape::with_layout(
        &ctx,
        TensorType::get(&ctx, E::F32, vec![2]).unwrap(),
        host_layout,
    )
    .unwrap();
    let mut device_layout = Layout::descending(1);
    device_layout.memory_space = 5;
    let output_type = Shape::with_layout(
        &ctx,
        TensorType::get(&ctx, E::U8, vec![8]).unwrap(),
        device_layout,
    )
    .unwrap();
    let strict = VerifyOptions {
        layout_sensitive: true,
        ..Default::default()
    };

    let parameter = value(&mut ctx, input_type, 0);
    let parameter_bitcast = BitcastOp::new(&mut ctx, output_type, vec![parameter]);
    assert!(parameter_bitcast.verify_with_options(&ctx, strict).is_err());

    let literal = Literal::from_bits(&ctx, input_type, &[0; 2]).unwrap();
    let constant = ConstantOp::new(&mut ctx, input_type, vec![], literal);
    let constant_value = constant.result(&ctx);
    let direct_bitcast = BitcastOp::new(&mut ctx, output_type, vec![constant_value]);
    direct_bitcast.verify_with_options(&ctx, strict).unwrap();

    let copied_constant = CopyOp::new(&mut ctx, input_type, vec![constant_value]);
    let copied_value = copied_constant.result(&ctx);
    let copied_bitcast = BitcastOp::new(&mut ctx, output_type, vec![copied_value]);
    copied_bitcast.verify_with_options(&ctx, strict).unwrap();
}
