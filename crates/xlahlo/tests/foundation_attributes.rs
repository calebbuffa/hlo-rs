use xlahlo::{
    Diagnostic,
    attributes::*,
    literal::Literal,
    original_value::{OriginalArray, OriginalValue, OriginalValueElement},
    pliron::{
        attribute::{AttrObj, Attribute},
        common_traits::Verify,
        context::Context,
        parsable::{Parsable, parse_from_str},
        printable::Printable,
        r#type::TypeHandle,
    },
    shape::{Dimension, Layout, Shape, Tile},
    sharding::{Mesh, MeshAxis, Sharding},
    types::ElementType,
};

#[test]
fn diagnostic_payloads_survive_native_verify_boundaries_and_allow_recovery() {
    let ctx = Context::new();
    let bad = ResultAccuracy::Tolerance {
        atol_bits: f64::NAN.to_bits(),
        rtol_bits: 0,
        ulps: -1,
    };
    for error in [bad.validate().unwrap_err(), bad.verify(&ctx).unwrap_err()] {
        assert!(matches!(
            error.err.downcast_ref::<Diagnostic>(),
            Some(Diagnostic::InvalidAttribute { .. })
        ));
        assert!(error.to_string().contains("accuracy tolerances"));
    }
    assert!(ResultAccuracy::tolerance(0.1, 0.01, 1).is_ok());
}

fn roundtrip<T: Attribute + PartialEq + std::fmt::Debug + Clone>(ctx: &mut Context, value: T) {
    value.verify(ctx).unwrap();
    let attr: AttrObj = Box::new(value.clone());
    let text = attr.disp(ctx).to_string();
    let parsed =
        parse_from_str(AttrObj::parser(()), ctx, &text).unwrap_or_else(|e| panic!("{text}: {e}"));
    parsed.verify(ctx).unwrap();
    assert_eq!(parsed.downcast_ref::<T>().unwrap(), &value, "{text}");
}

#[test]
fn instruction_attribute_registration_includes_indexing_and_control_records() {
    let mut ctx = Context::new();
    xlahlo::register(&mut ctx);
    roundtrip(&mut ctx, Axes(vec![0, 2]));
    roundtrip(&mut ctx, Index(3));
    roundtrip(&mut ctx, Direction::Eq);
    roundtrip(&mut ctx, ComparisonOrder::Total);
    roundtrip(&mut ctx, Precision::High);
    roundtrip(&mut ctx, FusionKind::Loop);
    roundtrip(
        &mut ctx,
        Slice {
            starts: Axes(vec![0]),
            limits: Axes(vec![3]),
            strides: Axes(vec![1]),
        },
    );
    roundtrip(&mut ctx, Callees(vec![]));
    roundtrip(
        &mut ctx,
        IndexFlags {
            sorted: true,
            unique: false,
        },
    );
}

#[test]
fn geometry_precision_comparison_and_frontend_attributes_are_native_roundtrippable() {
    let mut ctx = Context::new();
    xlahlo::register(&mut ctx);
    let dot = DotDimensionNumbers {
        lhs_contracting_dimensions: Axes(vec![1]),
        rhs_contracting_dimensions: Axes(vec![0]),
        lhs_batch_dimensions: Axes(vec![]),
        rhs_batch_dimensions: Axes(vec![]),
    };
    roundtrip(&mut ctx, dot.clone());
    roundtrip(
        &mut ctx,
        RaggedDotDimensionNumbers {
            dot_dimension_numbers: dot,
            lhs_ragged_dimensions: Axes(vec![0]),
            rhs_group_dimensions: Axes(vec![2]),
        },
    );
    roundtrip(
        &mut ctx,
        Window(vec![WindowDimension {
            size: 3,
            stride: 2,
            padding_low: -1,
            padding_high: 4,
            window_dilation: 2,
            base_dilation: 1,
            window_reversal: true,
        }]),
    );
    roundtrip(
        &mut ctx,
        PaddingConfig(vec![PaddingConfigDimension {
            edge_padding_low: -1,
            edge_padding_high: 3,
            interior_padding: 2,
        }]),
    );
    roundtrip(
        &mut ctx,
        ConvolutionDimensionNumbers {
            input_batch_dimension: 0,
            input_feature_dimension: 3,
            input_spatial_dimensions: Axes(vec![1, 2]),
            kernel_input_feature_dimension: 2,
            kernel_output_feature_dimension: 3,
            kernel_spatial_dimensions: Axes(vec![0, 1]),
            output_batch_dimension: 0,
            output_feature_dimension: 3,
            output_spatial_dimensions: Axes(vec![1, 2]),
        },
    );
    roundtrip(
        &mut ctx,
        GatherDimensionNumbers {
            offset_dims: Axes(vec![1]),
            collapsed_slice_dims: Axes(vec![0]),
            start_index_map: Axes(vec![0]),
            index_vector_dim: 1,
            operand_batching_dims: Axes(vec![]),
            start_indices_batching_dims: Axes(vec![]),
        },
    );
    roundtrip(
        &mut ctx,
        ScatterDimensionNumbers {
            update_window_dims: Axes(vec![1]),
            inserted_window_dims: Axes(vec![0]),
            scatter_dims_to_operand_dims: Axes(vec![0]),
            index_vector_dim: 1,
            input_batching_dims: Axes(vec![]),
            scatter_indices_batching_dims: Axes(vec![]),
        },
    );
    roundtrip(
        &mut ctx,
        Slice {
            starts: Axes(vec![0]),
            limits: Axes(vec![7]),
            strides: Axes(vec![2]),
        },
    );
    roundtrip(
        &mut ctx,
        PrecisionConfig {
            operand_precision: OperandPrecision(vec![Precision::High, Precision::Highest]),
            algorithm: Algorithm::DotBF16BF16F32X6,
        },
    );
    roundtrip(
        &mut ctx,
        Comparison {
            direction: Direction::Lt,
            comparison_type: ComparisonType::TotalOrder,
        },
    );
    let attrs = FrontendAttributes::new([
        ("z".into(), "quoted \"\\\n".into()),
        ("a".into(), "first".into()),
        ("a".into(), "last".into()),
    ]);
    assert_eq!(attrs.get("a"), Some("last"));
    assert_eq!(attrs.get("missing"), None);
    roundtrip(&mut ctx, attrs);
    assert_eq!(Direction::Lt.inverse(), Direction::Ge);
    assert_eq!(Direction::Lt.converse(), Direction::Gt);
}

#[test]
fn numerical_collective_and_alias_attributes_roundtrip() {
    let mut ctx = Context::new();
    xlahlo::register(&mut ctx);
    roundtrip(&mut ctx, FftType::IRFFT);
    roundtrip(&mut ctx, RandomDistribution::Normal);
    roundtrip(&mut ctx, RandomAlgorithm::ThreeFry);
    roundtrip(
        &mut ctx,
        TriangularSolveOptions {
            left_side: true,
            lower: true,
            unit_diagonal: false,
            transpose_a: Transpose::Adjoint,
        },
    );
    roundtrip(&mut ctx, CholeskyOptions { lower: true });
    roundtrip(
        &mut ctx,
        ShuffleMode {
            shifts: Axes(vec![-3, 7]),
        },
    );
    roundtrip(
        &mut ctx,
        CubScanOptions {
            vector_length: 1,
            row_length: 2,
            column_length: 3,
            kind: CubScanKind::Sum,
            is_reverse: true,
        },
    );
    roundtrip(&mut ctx, SortOptions { descending: true });
    roundtrip(&mut ctx, PaddingType::Same);
    roundtrip(&mut ctx, TriState::True);
    roundtrip(&mut ctx, ConvolutionKind::Dgrad);
    roundtrip(&mut ctx, CustomCallSchedule::Earliest);
    roundtrip(&mut ctx, CustomCallApiVersion::TypedFfi);
    roundtrip(&mut ctx, CollectiveOpGroupMode::FlattenedID);
    roundtrip(&mut ctx, ChannelHandle { handle: 2, kind: 1 });
    roundtrip(&mut ctx, Payload::Value(Bytes(vec![0, 255])));
    roundtrip(
        &mut ctx,
        SliceDimensions {
            start: 1,
            limit: 3,
            stride: 2,
        },
    );
    roundtrip(
        &mut ctx,
        ReducePrecision {
            exponent_bits: 4,
            mantissa_bits: 3,
        },
    );
    roundtrip(
        &mut ctx,
        BatchNormOptions {
            epsilon_bits: 0.001f32.to_bits(),
            feature_index: 1,
        },
    );
    roundtrip(
        &mut ctx,
        WhileLoopBackendConfig {
            known_trip_count: OptionalInteger::Value(0),
            known_init_step: KnownInitSteps(vec![KnownInitStep { init: 3, step: 2 }]),
            known_induction_variable: OptionalInteger::Value(0),
            dynamic_variables: DynamicVariables(vec![DynamicVariable {
                tuple_index: 1,
                init: OptionalInteger::Absent,
                step: OptionalInteger::Value(1),
            }]),
        },
    );
    roundtrip(
        &mut ctx,
        SourceTargetPairs(vec![SourceTarget {
            source: 2,
            target: 3,
        }]),
    );
    roundtrip(
        &mut ctx,
        ReplicaGroups(vec![ReplicaGroup(vec![0, 2]), ReplicaGroup(vec![1, 3])]),
    );
    roundtrip(&mut ctx, ParameterReplication(vec![true, false]));
    roundtrip(
        &mut ctx,
        OutputOperandAliases(vec![OutputOperandAliasing {
            output_shape_index: Axes(vec![0]),
            operand_index: 1,
            operand_shape_index: Axes(vec![]),
        }]),
    );
    let sparsity = TensorSparsityConfig {
        num_non_zero: 2,
        block_size: 4,
        dimension: 1,
        stride: 1,
        idx: 2,
    };
    roundtrip(
        &mut ctx,
        SparsityConfig {
            lhs: sparsity.clone().into(),
            rhs: sparsity.into(),
        },
    );
    let scaling = TensorBlockScalingConfig {
        scale_idx: 2,
        zero_idx: OptionalInteger::Value(3),
        strides: Axes(vec![1, 1]),
        steps: Axes(vec![32, 1]),
    };
    roundtrip(
        &mut ctx,
        BlockScalingConfig {
            lhs: scaling.clone().into(),
            rhs: scaling.into(),
        },
    );
}

#[test]
fn metadata_retains_exact_numeric_bits_and_optional_payload_identity() {
    let mut ctx = Context::new();
    xlahlo::register(&mut ctx);
    roundtrip(
        &mut ctx,
        OpMetadata {
            op_type: "matmul".into(),
            op_name: "test".into(),
            source_file: "model.rs".into(),
            source_line: 3,
            profile_info: ProfileInfos(vec![ProfileInfo {
                profile_type: Axes(vec![1, 3]),
                relative_speedup_bits: f64::NAN.to_bits(),
                profile_source: 2,
                compilation_event: 1,
                profile_generation_strategy: 3,
            }]),
            metadata_payload: Payload::Id(0),
            ..OpMetadata::default()
        },
    );
    roundtrip(
        &mut ctx,
        StatisticsViz {
            stat_index_to_visualize: 0,
            statistics: Statistics(vec![Statistic {
                stat_name: "cost".into(),
                stat_val_bits: (-0.0f64).to_bits(),
            }]),
        },
    );
}

#[test]
fn sparsity_and_block_scaling_preserve_absent_and_single_sided_configs() {
    let mut ctx = Context::new();
    xlahlo::register(&mut ctx);
    let sparsity = TensorSparsityConfig {
        num_non_zero: 2,
        block_size: 4,
        dimension: 1,
        stride: 1,
        idx: 2,
    };
    for (lhs, rhs) in [(false, false), (true, false), (false, true), (true, true)] {
        let config = SparsityConfig {
            lhs: if lhs {
                sparsity.clone().into()
            } else {
                OptionalTensorSparsityConfig::Absent
            },
            rhs: if rhs {
                sparsity.clone().into()
            } else {
                OptionalTensorSparsityConfig::Absent
            },
        };
        assert_eq!(config.lhs.is_some(), lhs);
        assert_eq!(config.rhs.is_some(), rhs);
        roundtrip(&mut ctx, config);
        for zero_idx in [
            OptionalInteger::Absent,
            OptionalInteger::Value(0),
            OptionalInteger::Value(3),
        ] {
            let scaling = TensorBlockScalingConfig {
                scale_idx: 2,
                zero_idx,
                strides: Axes(vec![1]),
                steps: Axes(vec![32]),
            };
            let config = BlockScalingConfig {
                lhs: if lhs {
                    scaling.clone().into()
                } else {
                    OptionalTensorBlockScalingConfig::Absent
                },
                rhs: if rhs {
                    scaling.clone().into()
                } else {
                    OptionalTensorBlockScalingConfig::Absent
                },
            };
            assert_eq!(config.lhs.is_some(), lhs);
            assert_eq!(config.rhs.is_some(), rhs);
            if let Some(side) = config.lhs.as_ref() {
                assert_eq!(side.zero_idx, zero_idx);
            }
            if let Some(side) = config.rhs.as_ref() {
                assert_eq!(side.zero_idx, zero_idx);
            }
            roundtrip(&mut ctx, config);
        }
    }
}

#[test]
fn accuracy_equality_uses_native_mode_defaults_and_numeric_tolerances() {
    let mut ctx = Context::new();
    xlahlo::register(&mut ctx);
    let absent = ResultAccuracy::Default;
    let explicit_default = ResultAccuracy::Default;
    let highest = ResultAccuracy::Highest;
    let tolerance = ResultAccuracy::tolerance(0.0, 0.0, 0).unwrap();
    let negative_zero = ResultAccuracy::tolerance(-0.0, -0.0, 0).unwrap();
    assert!(absent.equal(&explicit_default));
    assert!(!absent.equal(&highest));
    assert!(!absent.equal(&tolerance));
    assert!(tolerance.equal(&negative_zero));
    assert!(ResultAccuracy::tolerance(-1.0, 0.0, 0).is_err());
    assert!(ResultAccuracy::tolerance(0.0, f64::NAN, 0).is_err());
    assert!(ResultAccuracy::tolerance(0.0, 0.0, -1).is_err());
    roundtrip(&mut ctx, negative_zero);
    roundtrip(&mut ctx, highest);
}

#[test]
fn original_values_use_last_write_wins_tree_semantics_and_verify_leaf_indices() {
    let mut ctx = Context::new();
    xlahlo::register(&mut ctx);
    let a = OriginalArray {
        instruction_name: "arg".into(),
        shape_index: vec![],
    };
    let value = OriginalValue {
        elements: vec![
            OriginalValueElement {
                shape_index: vec![],
                original_array: vec![a.clone()],
            },
            OriginalValueElement {
                shape_index: vec![0],
                original_array: vec![a],
            },
            OriginalValueElement {
                shape_index: vec![1],
                original_array: vec![],
            },
        ],
        ..OriginalValue::default()
    };
    assert_eq!(value.leaves().unwrap().len(), 2);
    let array = Shape::scalar(&ctx, ElementType::F32).unwrap();
    let shape = Shape::tuple(&ctx, vec![array, Shape::token(&ctx)]).unwrap();
    value.verify_shape(&ctx, shape).unwrap();
    assert!(value.verify_shape(&ctx, array).is_err());
    assert!(value.equal(&value.normalized().unwrap()).unwrap());
    roundtrip(&mut ctx, value);
    let synthetic = OriginalValue {
        call_hierarchy: vec![String::new()],
        ..OriginalValue::default()
    };
    assert!(synthetic.is_synthetic_call());
    synthetic.verify_shape(&ctx, array).unwrap();
    roundtrip(&mut ctx, synthetic);
}

#[test]
fn native_types_layouts_shardings_and_exact_literals_roundtrip_in_pliron() {
    let mut ctx = Context::new();
    xlahlo::register(&mut ctx);
    let array = Shape::array(&ctx, ElementType::C128, vec![2]).unwrap();
    let literal = Literal::from_bits(&ctx, array, &[u128::MAX, 1u128 << 127]).unwrap();
    roundtrip(&mut ctx, literal);
    roundtrip(
        &mut ctx,
        Layout {
            minor_to_major: vec![1, 0],
            tiles: vec![Tile {
                dimensions: vec![2, 4],
            }],
            memory_space: 3,
            ..Layout::default()
        },
    );
    roundtrip(
        &mut ctx,
        Sharding::tuple(vec![Sharding::replicated(), Sharding::maximal(3)]),
    );
    roundtrip(
        &mut ctx,
        Mesh::new(
            vec![MeshAxis {
                name: "x".into(),
                size: 3,
            }],
            vec![],
            vec![],
        )
        .unwrap(),
    );
    let dynamic = Shape::dynamic_array(
        &ctx,
        ElementType::F6E2M3FN,
        vec![Dimension::Bounded(7), Dimension::Unbounded],
    )
    .unwrap();
    let token = Shape::token(&ctx);
    let buffer = Shape::buffer(&ctx, array).unwrap();
    let tuple = Shape::tuple(&ctx, vec![dynamic, token, buffer, Shape::opaque(&ctx)]).unwrap();
    for ty in [dynamic, token, buffer, tuple] {
        let text = ty.disp(&ctx).to_string();
        let parsed = parse_from_str(TypeHandle::parser(()), &mut ctx, &text).unwrap();
        parsed.deref(&ctx).verify(&ctx).unwrap();
        assert_eq!(parsed, ty);
    }
}
