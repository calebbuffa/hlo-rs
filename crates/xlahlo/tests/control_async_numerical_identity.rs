#[path = "control_support/mod.rs"]
mod support;

use support::*;
use xlahlo::{
    attributes::{
        Algorithm, Axes, BatchNormOptions, BlockScalingConfig, ConvolutionDimensionNumbers,
        ConvolutionKind, DotDimensionNumbers, OpMetadata, OperandPrecision, OptionalInteger,
        OptionalTensorBlockScalingConfig, OptionalTensorSparsityConfig, Precision, PrecisionConfig,
        RaggedDotDimensionNumbers, ResultAccuracy, SparsityConfig, TensorBlockScalingConfig,
        TensorSparsityConfig, TriState, Window,
    },
    opcode::Opcode,
    ops::{
        control::{AsyncDoneOp, AsyncStartOp, DomainOp},
        elementwise::AddOp,
        movement::{DynamicReshapeOp, ReshapeOp},
        numerical::{
            BatchNormGradOp, BatchNormInferenceOp, BatchNormTrainingOp, ConvolutionOp, DotOp,
            NumericalInteger, RaggedDotOp, ScaledDotOp, ScanOp,
        },
    },
    pliron::{
        builtin::ops::ModuleOp,
        context::{Context, Ptr},
        irfmt::parsers::spaced,
        op::Op,
        operation::Operation,
        parsable::parse_from_str,
        printable::Printable,
        r#type::TypeHandle,
        value::Value,
    },
    shape::Shape,
    sharding::Sharding,
    types::ElementType,
    verification,
};

#[test]
fn dynamic_reshape_identity_uses_size_operands_not_redundant_bounds_metadata() {
    for swap_sizes in [false, true] {
        let mut ctx = context();
        let input = shape(&ctx, ElementType::F32, &[6]);
        let size = shape(&ctx, ElementType::S32, &[]);
        let output = shape(&ctx, ElementType::F32, &[2, 3]);
        let mut second = false;
        let (module, a, b) = pair(
            &mut ctx,
            vec![input, size, size],
            output,
            |ctx, mut args| {
                if second && swap_sizes {
                    args.swap(1, 2);
                }
                second = true;
                DynamicReshapeOp::new(ctx, output, args).get_operation()
            },
        );
        b.deref_mut(&ctx)
            .attributes
            .0
            .remove(&xlahlo::pliron::identifier::Identifier::new(
                "hlo_dimensions",
            ));
        Operation::get_op_dyn(b, &ctx).verify(&ctx).unwrap();
        for bounds in [vec![2, 3], vec![17], vec![]] {
            attr(&ctx, a, "hlo_dimensions", Axes(bounds));
            Operation::get_op_dyn(a, &ctx).verify(&ctx).unwrap();
            let text = module.get_operation().disp(&ctx).to_string();
            assert_eq!(
                verification::verify(&ctx, module.get_operation()).is_ok(),
                !swap_sizes
            );
            assert_eq!(module.get_operation().disp(&ctx).to_string(), text);
            let mut fresh = context();
            let parsed =
                parse_from_str(spaced(Operation::top_level_parser()), &mut fresh, &text).unwrap();
            assert_eq!(verification::verify(&fresh, parsed).is_ok(), !swap_sizes);
        }
    }
}

#[test]
fn domain_identity_compares_normalized_sharding_constraints() {
    let mut ctx = context();
    let array = shape(&ctx, ElementType::F32, &[2]);
    let tuple = Shape::tuple(&ctx, vec![array, array]).unwrap();
    let (module, a, b) = pair(&mut ctx, vec![tuple], tuple, |ctx, args| {
        DomainOp::new(ctx, tuple, args).get_operation()
    });
    verification::verify(&ctx, module.get_operation()).unwrap();
    for key in ["hlo_domain_entry_sharding", "hlo_domain_exit_sharding"] {
        attr(&ctx, a, key, Sharding::replicated());
        assert!(verification::verify(&ctx, module.get_operation()).is_err());
        attr(
            &ctx,
            b,
            key,
            Sharding::tuple(vec![Sharding::replicated(), Sharding::replicated()]),
        );
        Operation::get_op_dyn(a, &ctx).verify(&ctx).unwrap();
        Operation::get_op_dyn(b, &ctx).verify(&ctx).unwrap();
        verification::verify(&ctx, module.get_operation()).unwrap();
        attr(
            &ctx,
            a,
            key,
            Sharding::tuple(vec![Sharding::replicated(), Sharding::replicated()]),
        );
        attr(&ctx, b, key, Sharding::replicated());
        verification::verify(&ctx, module.get_operation()).unwrap();
        attr(&ctx, a, key, Sharding::replicated());
        let mut annotated = Sharding::replicated();
        annotated.metadata.0.push(OpMetadata {
            op_name: "diagnostic-only".into(),
            ..OpMetadata::default()
        });
        attr(
            &ctx,
            b,
            key,
            Sharding::tuple(vec![annotated, Sharding::replicated()]),
        );
        verification::verify(&ctx, module.get_operation()).unwrap();
        attr(
            &ctx,
            b,
            key,
            Sharding::tuple(vec![Sharding::replicated(), Sharding::maximal(0)]),
        );
        let invalid = module.get_operation().disp(&ctx).to_string();
        assert!(verification::verify(&ctx, module.get_operation()).is_err());
        assert_eq!(module.get_operation().disp(&ctx).to_string(), invalid);
        attr(
            &ctx,
            b,
            key,
            Sharding::tuple(vec![Sharding::replicated(), Sharding::replicated()]),
        );
        verification::verify(&ctx, module.get_operation()).unwrap();
    }
    let text = module.get_operation().disp(&ctx).to_string();
    let mut fresh = context();
    let parsed = parse_from_str(spaced(Operation::top_level_parser()), &mut fresh, &text).unwrap();
    verification::verify(&fresh, parsed).unwrap();
}

#[test]
fn scan_identity_ignores_unowned_cub_metadata_but_preserves_associativity() {
    let mut ctx = context();
    let array = shape(&ctx, ElementType::F32, &[2, 2]);
    let slice = shape(&ctx, ElementType::F32, &[2]);
    let (body, block, args) = computation(&mut ctx, "scan_body", vec![slice], slice);
    ret(&mut ctx, block, args[0]);
    let (module, a, b) = pair(&mut ctx, vec![array], array, |ctx, args| {
        ScanOp::new(
            ctx,
            array,
            args,
            Axes(vec![0]),
            NumericalInteger(0),
            TriState::False,
            callees(&["scan_body"]),
        )
        .get_operation()
    });
    let module_block = module
        .get_operation()
        .deref(&ctx)
        .get_region(0)
        .deref(&ctx)
        .get_entry_block()
        .unwrap();
    body.get_operation().insert_at_back(module_block, &ctx);
    verification::verify(&ctx, module.get_operation()).unwrap();
    attr(&ctx, a, "hlo_cub_scan", Axes(vec![17]));
    attr(&ctx, b, "hlo_cub_scan", Axes(vec![23]));
    verification::verify(&ctx, module.get_operation()).unwrap();
    attr(&ctx, b, "hlo_is_associative", TriState::True);
    assert!(verification::verify(&ctx, module.get_operation()).is_err());
    attr(&ctx, a, "hlo_is_associative", TriState::True);
    verification::verify(&ctx, module.get_operation()).unwrap();
}

#[test]
fn wrapped_identity_compares_accuracy_outside_supported_unary_opcodes() {
    for opcode in [Opcode::Add, Opcode::Dot, Opcode::Reshape] {
        let mut ctx = context();
        let array = shape(&ctx, ElementType::F32, &[2, 2]);
        let inputs = if opcode == Opcode::Reshape {
            vec![array]
        } else {
            vec![array, array]
        };
        let (module, a, b) = pair(&mut ctx, inputs, array, |ctx, args| match opcode {
            Opcode::Add => AddOp::new(ctx, array, args).get_operation(),
            Opcode::Dot => contraction(ctx, args, array, opcode),
            Opcode::Reshape => ReshapeOp::new(ctx, array, args).get_operation(),
            _ => unreachable!(),
        });
        attr(&ctx, b, "hlo_result_accuracy", ResultAccuracy::Default);
        verification::verify(&ctx, module.get_operation()).unwrap();
        for accuracy in [
            ResultAccuracy::Highest,
            ResultAccuracy::tolerance(0.0, 0.0, 0).unwrap(),
            ResultAccuracy::tolerance(0.1, 0.2, 3).unwrap(),
        ] {
            attr(&ctx, b, "hlo_result_accuracy", accuracy);
            Operation::get_op_dyn(b, &ctx).verify(&ctx).unwrap();
            let invalid = module.get_operation().disp(&ctx).to_string();
            assert!(verification::verify(&ctx, module.get_operation()).is_err());
            assert_eq!(module.get_operation().disp(&ctx).to_string(), invalid);
            attr(&ctx, a, "hlo_result_accuracy", accuracy);
            verification::verify(&ctx, module.get_operation()).unwrap();
            attr(&ctx, a, "hlo_result_accuracy", ResultAccuracy::Default);
        }
        attr(
            &ctx,
            a,
            "hlo_result_accuracy",
            ResultAccuracy::tolerance(0.0, -0.0, 0).unwrap(),
        );
        attr(
            &ctx,
            b,
            "hlo_result_accuracy",
            ResultAccuracy::tolerance(-0.0, 0.0, 0).unwrap(),
        );
        verification::verify(&ctx, module.get_operation()).unwrap();
        let text = module.get_operation().disp(&ctx).to_string();
        let mut fresh = context();
        let parsed =
            parse_from_str(spaced(Operation::top_level_parser()), &mut fresh, &text).unwrap();
        verification::verify(&fresh, parsed).unwrap();
    }
}

#[test]
fn reshape_identity_normalizes_default_axis_and_distinguishes_explicit_axes() {
    let mut ctx = context();
    let input = shape(&ctx, ElementType::F32, &[6]);
    let output = shape(&ctx, ElementType::F32, &[2, 3]);
    let (module, a, b) = pair(&mut ctx, vec![input], output, |ctx, args| {
        ReshapeOp::new(ctx, output, args).get_operation()
    });
    let a = Operation::get_op::<ReshapeOp>(a, &ctx).unwrap();
    let b = Operation::get_op::<ReshapeOp>(b, &ctx).unwrap();
    for default in [vec![], vec![-1]] {
        attr(&ctx, b.get_operation(), "hlo_dimensions", Axes(default));
        verification::verify(&ctx, module.get_operation()).unwrap();
    }

    for (first, second, equal) in [(-1, 0, false), (0, 1, false), (1, 1, true), (0, 0, true)] {
        a.set_inferred_dimension(&ctx, first);
        b.set_inferred_dimension(&ctx, second);
        Operation::get_op_dyn(a.get_operation(), &ctx)
            .verify(&ctx)
            .unwrap();
        Operation::get_op_dyn(b.get_operation(), &ctx)
            .verify(&ctx)
            .unwrap();
        assert_eq!(
            verification::verify(&ctx, module.get_operation()).is_ok(),
            equal
        );
    }
}

#[test]
fn dot_identity_uses_canonical_dimensions_and_precision_not_unowned_lookalike_keys() {
    use xlahlo::pliron::builtin::attributes::StringAttr;
    let mut ctx = context();
    let array = shape(&ctx, ElementType::F32, &[2, 2]);
    let (module, a, b) = pair(&mut ctx, vec![array, array], array, |ctx, args| {
        contraction(ctx, args, array, Opcode::Dot)
    });
    for key in ["hlo_lhs_precision", "hlo_rhs_precision"] {
        attr(&ctx, a, key, StringAttr::new("unowned metadata".into()));
        attr(&ctx, b, key, Axes(vec![7]));
    }
    attr(&ctx, a, "hlo_channel", OptionalInteger::Value(17));
    attr(&ctx, b, "hlo_channel", OptionalInteger::Value(23));
    verification::verify(&ctx, module.get_operation()).unwrap();
    let dimensions = DotDimensionNumbers {
        lhs_contracting_dimensions: Axes(vec![0]),
        rhs_contracting_dimensions: Axes(vec![1]),
        lhs_batch_dimensions: Axes(vec![]),
        rhs_batch_dimensions: Axes(vec![]),
    };
    attr(&ctx, b, "hlo_dimensions", dimensions.clone());
    Operation::get_op_dyn(b, &ctx).verify(&ctx).unwrap();
    assert!(verification::verify(&ctx, module.get_operation()).is_err());
    attr(&ctx, a, "hlo_dimensions", dimensions);
    verification::verify(&ctx, module.get_operation()).unwrap();
    let precision = PrecisionConfig {
        operand_precision: OperandPrecision(vec![Precision::Highest, Precision::Default]),
        algorithm: Algorithm::Unset,
    };
    attr(&ctx, b, "hlo_precision", precision.clone());
    Operation::get_op_dyn(b, &ctx).verify(&ctx).unwrap();
    assert!(verification::verify(&ctx, module.get_operation()).is_err());
    attr(&ctx, a, "hlo_precision", precision);
    verification::verify(&ctx, module.get_operation()).unwrap();
}

fn pair(
    ctx: &mut Context,
    inputs: Vec<TypeHandle>,
    output: TypeHandle,
    mut build: impl FnMut(&mut Context, Vec<Value>) -> Ptr<Operation>,
) -> (ModuleOp, Ptr<Operation>, Ptr<Operation>) {
    let (first, block, args) = computation(ctx, "first", inputs.clone(), output);
    let a = build(ctx, args);
    a.insert_at_back(block, ctx);
    let value = a.deref(ctx).get_result(0);
    ret(ctx, block, value);
    let (second, block, args) = computation(ctx, "second", inputs.clone(), output);
    let b = build(ctx, args);
    b.insert_at_back(block, ctx);
    let value = b.deref(ctx).get_result(0);
    ret(ctx, block, value);
    let (main, block, args) = computation(ctx, "main", inputs.clone(), output);
    let state = shape(ctx, ElementType::U32, &[]);
    let async_context =
        Shape::tuple(ctx, vec![Shape::tuple(ctx, inputs).unwrap(), output, state]).unwrap();
    let start = AsyncStartOp::new(ctx, async_context, args, callees(&["first"]));
    start.get_operation().insert_at_back(block, ctx);
    let value = start.result(ctx);
    let done = AsyncDoneOp::new(ctx, output, vec![value]);
    attr(
        ctx,
        done.get_operation(),
        "hlo_callees",
        callees(&["second"]),
    );
    done.get_operation().insert_at_back(block, ctx);
    let value = done.result(ctx);
    ret(ctx, block, value);
    (module(ctx, &[first, second, main], "main"), a, b)
}

fn contraction(
    ctx: &mut Context,
    operands: Vec<Value>,
    output: TypeHandle,
    kind: Opcode,
) -> Ptr<Operation> {
    let precision = PrecisionConfig {
        operand_precision: OperandPrecision(vec![Precision::Default; 2]),
        algorithm: Algorithm::Unset,
    };
    if kind == Opcode::Dot {
        return DotOp::new(
            ctx,
            output,
            operands,
            DotDimensionNumbers {
                lhs_contracting_dimensions: Axes(vec![1]),
                rhs_contracting_dimensions: Axes(vec![0]),
                lhs_batch_dimensions: Axes(vec![]),
                rhs_batch_dimensions: Axes(vec![]),
            },
            precision,
        )
        .get_operation();
    }
    assert_eq!(kind, Opcode::Convolution);
    ConvolutionOp::new(
        ctx,
        output,
        operands,
        ConvolutionDimensionNumbers {
            input_batch_dimension: 0,
            input_feature_dimension: 1,
            input_spatial_dimensions: Axes(vec![]),
            kernel_input_feature_dimension: 0,
            kernel_output_feature_dimension: 1,
            kernel_spatial_dimensions: Axes(vec![]),
            output_batch_dimension: 0,
            output_feature_dimension: 1,
            output_spatial_dimensions: Axes(vec![]),
        },
        Window(vec![]),
        NumericalInteger(1),
        NumericalInteger(1),
        ConvolutionKind::Fprop,
        precision,
    )
    .get_operation()
}

#[test]
fn contraction_identity_normalizes_empty_sparsity_and_scaling_configs() {
    for kind in [Opcode::Dot, Opcode::Convolution] {
        let mut ctx = context();
        let lhs = shape(&ctx, ElementType::F32, &[2, 3]);
        let rhs = shape(&ctx, ElementType::F32, &[3, 4]);
        let output = shape(&ctx, ElementType::F32, &[2, 4]);
        let (module, _, b) = pair(&mut ctx, vec![lhs, rhs], output, |ctx, args| {
            contraction(ctx, args, output, kind)
        });
        attr(&ctx, b, "hlo_sparsity", SparsityConfig::default());
        attr(&ctx, b, "hlo_block_scaling", BlockScalingConfig::default());
        verification::verify(&ctx, module.get_operation()).unwrap();
    }
}

#[test]
fn ragged_and_scaled_dot_identity_compares_only_owned_contraction_metadata() {
    for kind in [Opcode::RaggedDot, Opcode::ScaledDot] {
        let mut ctx = context();
        let data = shape(&ctx, ElementType::F32, &[2, 2]);
        let groups = shape(&ctx, ElementType::S32, &[3]);
        let scale = shape(&ctx, ElementType::F32, &[1, 1]);
        let inputs = if kind == Opcode::RaggedDot {
            vec![data, data, groups]
        } else {
            vec![data, data, scale, scale]
        };
        let output = if kind == Opcode::RaggedDot {
            shape(&ctx, ElementType::F32, &[3, 2, 2])
        } else {
            data
        };
        let dimensions = DotDimensionNumbers {
            lhs_contracting_dimensions: Axes(vec![1]),
            rhs_contracting_dimensions: Axes(vec![0]),
            lhs_batch_dimensions: Axes(vec![]),
            rhs_batch_dimensions: Axes(vec![]),
        };
        let ragged_dimensions = RaggedDotDimensionNumbers {
            dot_dimension_numbers: dimensions.clone(),
            lhs_ragged_dimensions: Axes(vec![1]),
            rhs_group_dimensions: Axes(vec![]),
        };
        let precision = PrecisionConfig {
            operand_precision: OperandPrecision(vec![Precision::Default; 2]),
            algorithm: Algorithm::Unset,
        };
        let (module, _, b) = pair(&mut ctx, inputs, output, |ctx, args| {
            if kind == Opcode::RaggedDot {
                RaggedDotOp::new(
                    ctx,
                    output,
                    args,
                    ragged_dimensions.clone(),
                    precision.clone(),
                )
                .get_operation()
            } else {
                ScaledDotOp::new(ctx, output, args, dimensions.clone(), precision.clone())
                    .get_operation()
            }
        });
        verification::verify(&ctx, module.get_operation()).unwrap();
        attr(&ctx, b, "hlo_sparsity", SparsityConfig::default());
        attr(&ctx, b, "hlo_block_scaling", BlockScalingConfig::default());
        verification::verify(&ctx, module.get_operation()).unwrap();
        attr(
            &ctx,
            b,
            "hlo_sparsity",
            SparsityConfig {
                lhs: OptionalTensorSparsityConfig::Present(TensorSparsityConfig {
                    num_non_zero: 1,
                    block_size: 2,
                    dimension: 1,
                    stride: 1,
                    idx: 2,
                }),
                rhs: OptionalTensorSparsityConfig::Absent,
            },
        );
        verification::verify(&ctx, module.get_operation()).unwrap();
        attr(
            &ctx,
            b,
            "hlo_precision",
            PrecisionConfig {
                operand_precision: OperandPrecision(vec![Precision::High; 2]),
                algorithm: Algorithm::Unset,
            },
        );
        Operation::get_op_dyn(b, &ctx).verify(&ctx).unwrap();
        assert!(verification::verify(&ctx, module.get_operation()).is_err());
        attr(&ctx, b, "hlo_precision", precision);
        verification::verify(&ctx, module.get_operation()).unwrap();

        let other_dimensions = DotDimensionNumbers {
            lhs_contracting_dimensions: Axes(vec![0]),
            rhs_contracting_dimensions: Axes(vec![1]),
            lhs_batch_dimensions: Axes(vec![]),
            rhs_batch_dimensions: Axes(vec![]),
        };
        if kind == Opcode::RaggedDot {
            attr(
                &ctx,
                b,
                "hlo_dimensions",
                RaggedDotDimensionNumbers {
                    dot_dimension_numbers: other_dimensions,
                    lhs_ragged_dimensions: Axes(vec![0]),
                    rhs_group_dimensions: Axes(vec![]),
                },
            );
        } else {
            attr(&ctx, b, "hlo_dimensions", other_dimensions);
        }
        Operation::get_op_dyn(b, &ctx).verify(&ctx).unwrap();
        assert!(verification::verify(&ctx, module.get_operation()).is_err());
    }
}

#[test]
fn contraction_identity_checks_block_scaling_and_sparsity_semantics() {
    for kind in [Opcode::Dot, Opcode::Convolution] {
        for sparsity in [false, true] {
            let mut ctx = context();
            let lhs = shape(&ctx, ElementType::F32, &[2, 3]);
            let rhs = shape(&ctx, ElementType::F32, &[3, 4]);
            let aux = shape(&ctx, ElementType::U16, &[]);
            let output = shape(&ctx, ElementType::F32, &[2, 4]);
            let (module, a, b) = pair(&mut ctx, vec![lhs, rhs, aux], output, |ctx, args| {
                contraction(ctx, args, output, kind)
            });
            if sparsity {
                let config = SparsityConfig {
                    lhs: OptionalTensorSparsityConfig::Present(TensorSparsityConfig {
                        num_non_zero: 1,
                        block_size: 1,
                        dimension: 1,
                        stride: 1,
                        idx: 2,
                    }),
                    rhs: OptionalTensorSparsityConfig::Absent,
                };
                attr(&ctx, a, "hlo_sparsity", config.clone());
                attr(&ctx, b, "hlo_sparsity", config.clone());
                verification::verify(&ctx, module.get_operation()).unwrap();
                let mut different = config.clone();
                if let OptionalTensorSparsityConfig::Present(side) = &mut different.lhs {
                    side.stride = 2;
                }
                attr(&ctx, b, "hlo_sparsity", different);
                Operation::get_op_dyn(b, &ctx).verify(&ctx).unwrap();
                assert!(verification::verify(&ctx, module.get_operation()).is_err());
                attr(&ctx, b, "hlo_sparsity", config);
            } else {
                let config = BlockScalingConfig {
                    lhs: OptionalTensorBlockScalingConfig::Present(TensorBlockScalingConfig {
                        scale_idx: 2,
                        zero_idx: OptionalInteger::Absent,
                        strides: Axes(vec![1]),
                        steps: Axes(vec![1]),
                    }),
                    rhs: OptionalTensorBlockScalingConfig::Absent,
                };
                attr(&ctx, a, "hlo_block_scaling", config.clone());
                attr(&ctx, b, "hlo_block_scaling", config.clone());
                verification::verify(&ctx, module.get_operation()).unwrap();
                let mut different = config.clone();
                if let OptionalTensorBlockScalingConfig::Present(side) = &mut different.lhs {
                    side.steps = Axes(vec![2]);
                }
                attr(&ctx, b, "hlo_block_scaling", different);
                Operation::get_op_dyn(b, &ctx).verify(&ctx).unwrap();
                assert!(verification::verify(&ctx, module.get_operation()).is_err());
                attr(&ctx, b, "hlo_block_scaling", config);
            }
            verification::verify(&ctx, module.get_operation()).unwrap();
        }
    }
}

#[test]
fn batch_norm_identity_compares_epsilon_as_ieee_float_not_encoding_bits() {
    for kind in [
        Opcode::BatchNormTraining,
        Opcode::BatchNormInference,
        Opcode::BatchNormGrad,
    ] {
        for (a_epsilon, b_epsilon, equal) in [
            (0.0f32.to_bits(), (-0.0f32).to_bits(), true),
            (0.001f32.to_bits(), 0.001f32.to_bits(), true),
            (0.001f32.to_bits(), 0.002f32.to_bits(), false),
            (f32::INFINITY.to_bits(), f32::INFINITY.to_bits(), true),
            (0x7fc0_0001, 0x7fc0_0001, false),
            (0x7f80_0001, 0x7f80_0001, false),
        ] {
            let mut ctx = context();
            let data = shape(&ctx, ElementType::F32, &[3, 3]);
            let scale = shape(&ctx, ElementType::F32, &[3]);
            let output = if kind == Opcode::BatchNormInference {
                data
            } else {
                Shape::tuple(&ctx, vec![data, scale, scale]).unwrap()
            };
            let inputs = match kind {
                Opcode::BatchNormTraining => vec![data, scale, scale],
                Opcode::BatchNormInference => vec![data, scale, scale, scale, scale],
                Opcode::BatchNormGrad => vec![data, scale, scale, scale, data],
                _ => unreachable!(),
            };
            let (module, a, b) = pair(&mut ctx, inputs, output, |ctx, args| {
                let options = BatchNormOptions {
                    feature_index: 1,
                    epsilon_bits: a_epsilon,
                };
                match kind {
                    Opcode::BatchNormTraining => {
                        BatchNormTrainingOp::new(ctx, output, args, options).get_operation()
                    }
                    Opcode::BatchNormInference => {
                        BatchNormInferenceOp::new(ctx, output, args, options).get_operation()
                    }
                    Opcode::BatchNormGrad => {
                        BatchNormGradOp::new(ctx, output, args, options).get_operation()
                    }
                    _ => unreachable!(),
                }
            });
            Operation::get_op_dyn(a, &ctx).verify(&ctx).unwrap();
            Operation::get_op_dyn(b, &ctx).verify(&ctx).unwrap();
            attr(
                &ctx,
                b,
                "hlo_options",
                BatchNormOptions {
                    feature_index: 1,
                    epsilon_bits: b_epsilon,
                },
            );
            assert_eq!(
                verification::verify(&ctx, module.get_operation()).is_ok(),
                equal
            );
            attr(
                &ctx,
                a,
                "hlo_options",
                BatchNormOptions {
                    feature_index: 1,
                    epsilon_bits: 0.001f32.to_bits(),
                },
            );
            attr(
                &ctx,
                b,
                "hlo_options",
                BatchNormOptions {
                    feature_index: 1,
                    epsilon_bits: 0.001f32.to_bits(),
                },
            );
            verification::verify(&ctx, module.get_operation()).unwrap();
            attr(
                &ctx,
                b,
                "hlo_options",
                BatchNormOptions {
                    feature_index: 0,
                    epsilon_bits: 0.001f32.to_bits(),
                },
            );
            Operation::get_op_dyn(b, &ctx).verify(&ctx).unwrap();
            assert!(verification::verify(&ctx, module.get_operation()).is_err());
        }
    }
}
