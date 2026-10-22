use crate::support::{computation, context, module, ret};
use xlahlo::{
    attributes::{
        Axes, Bytes, Callees, Index, OpMetadata, OutputOperandAliases, OutputOperandAliasing,
        Payload,
    },
    ir::{
        AliasKind, BufferDonorEntry, BufferDonors, ControlPredecessors, DebugOptions,
        EntryComputationLayout, InputOutputAliases, InstructionName, InstructionSequence,
        MetadataPayloads, ModuleConfig, Schedule,
    },
    ops::{
        control::{
            CallOp, ConstantOp, CustomCallOp, GetTupleElementOp, ParameterOp, TupleOp, WhileOp,
        },
        movement::CopyOp,
    },
    pliron::{
        builtin::{
            attributes::{StringAttr, TypeAttr},
            ops::ModuleOp,
            types::FunctionType,
        },
        common_traits::Verify,
        context::Context,
        identifier::Identifier,
        irfmt::parsers::spaced,
        op::Op,
        operation::Operation,
        parsable::parse_from_str,
        printable::Printable,
        r#type::{Type, TypeHandle},
    },
    types::{ElementType, TensorType, TupleType},
};

#[test]
fn layout_insensitive_tuple_and_gte_preserve_shape_compatibility() {
    let mut ctx = context();
    let input = TensorType::with_layout(&ctx, ElementType::F32, vec![2, 3], vec![0, 1]).unwrap();
    let output = TensorType::with_layout(&ctx, ElementType::F32, vec![2, 3], vec![1, 0]).unwrap();
    let p = ParameterOp::new(&mut ctx, input, vec![], Index(0)).result(&ctx);
    let tuple_type = TupleType::get(&ctx, vec![output]).unwrap();
    let tuple = TupleOp::new(&mut ctx, tuple_type, vec![p]);
    tuple.verify(&ctx).unwrap();
    let tuple_value = tuple.result(&ctx);
    let gte = GetTupleElementOp::new(&mut ctx, input, vec![tuple_value], Index(0));
    gte.verify(&ctx).unwrap();
    let incorrect = TensorType::get(&ctx, ElementType::F32, vec![3, 2]).unwrap();
    let invalid = GetTupleElementOp::new(&mut ctx, incorrect, vec![tuple_value], Index(0));
    assert!(invalid.verify(&ctx).is_err());
}

#[test]
fn layout_insensitive_calls_and_while_use_body_then_condition() {
    let mut ctx = context();
    let input = TensorType::with_layout(&ctx, ElementType::F32, vec![2, 3], vec![0, 1]).unwrap();
    let output = TensorType::with_layout(&ctx, ElementType::F32, vec![2, 3], vec![1, 0]).unwrap();
    let pred = TensorType::get(&ctx, ElementType::Pred, vec![]).unwrap();
    let (module, container) = module(&mut ctx);
    let (_, block, parameters) = computation(&mut ctx, container, "body", vec![input], input);
    ret(&mut ctx, block, parameters[0]);
    let (_, block, _) = computation(&mut ctx, container, "condition", vec![output], pred);
    let literal = xlahlo::literal::Literal::from_bits(&ctx, pred, &[0]).unwrap();
    let false_ = ConstantOp::new(&mut ctx, pred, vec![], literal);
    false_.get_operation().insert_at_back(block, &ctx);
    let false_value = false_.result(&ctx);
    ret(&mut ctx, block, false_value);
    let (_, block, parameters) = computation(&mut ctx, container, "main", vec![input], output);
    let call = CallOp::new(&mut ctx, output, parameters, Callees(vec!["body".into()]));
    call.get_operation().insert_at_back(block, &ctx);
    let call_value = call.result(&ctx);
    let while_ = WhileOp::new(
        &mut ctx,
        input,
        vec![call_value],
        Callees(vec!["body".into(), "condition".into()]),
    );
    while_.get_operation().insert_at_back(block, &ctx);
    let result = while_.result(&ctx);
    ret(&mut ctx, block, result);
    xlahlo::verification::verify(&ctx, module.get_operation()).unwrap();
    assert!(
        xlahlo::verification::verify_with_options(
            &ctx,
            module.get_operation(),
            xlahlo::verification::VerifyOptions {
                layout_sensitive: true,
                ..Default::default()
            },
        )
        .is_err()
    );
    while_.get_operation().deref_mut(&ctx).attributes.set(
        Identifier::new("hlo_callees"),
        Callees(vec!["condition".into(), "body".into()]),
    );
    assert!(xlahlo::verification::verify(&ctx, module.get_operation()).is_err());
}

#[test]
fn declaration_order_does_not_limit_callee_resolution() {
    let mut ctx = context();
    let ty = TensorType::get(&ctx, ElementType::S32, vec![]).unwrap();
    let (module, container) = module(&mut ctx);
    xlahlo::ir::set_entry(&ctx, module.get_operation(), Identifier::new("caller")).unwrap();
    let (_, block, parameters) = computation(&mut ctx, container, "caller", vec![ty], ty);
    let call = CallOp::new(&mut ctx, ty, parameters, Callees(vec!["later".into()]));
    call.get_operation().insert_at_back(block, &ctx);
    let result = call.result(&ctx);
    ret(&mut ctx, block, result);
    let (_, block, parameters) = computation(&mut ctx, container, "later", vec![ty], ty);
    ret(&mut ctx, block, parameters[0]);
    xlahlo::verification::verify(&ctx, module.get_operation()).unwrap();
}

#[test]
fn duplicate_symbols_and_parameter_numbers_are_rejected() {
    let mut ctx = context();
    let ty = TensorType::get(&ctx, ElementType::S32, vec![]).unwrap();
    let (module, container) = module(&mut ctx);
    let (_, block, parameters) = computation(&mut ctx, container, "same", vec![ty], ty);
    ret(&mut ctx, block, parameters[0]);
    let (_, block, parameters) = computation(&mut ctx, container, "same", vec![ty], ty);
    ret(&mut ctx, block, parameters[0]);
    assert!(xlahlo::verification::verify(&ctx, module.get_operation()).is_err());

    let (module, container) = self::module(&mut ctx);
    xlahlo::ir::set_entry(&ctx, module.get_operation(), Identifier::new("unique")).unwrap();
    let (_, block, parameters) = computation(&mut ctx, container, "unique", vec![ty, ty], ty);
    let first = parameters[0].defining_entity();
    let xlahlo::pliron::value::DefiningEntity::Op(first) = first else {
        unreachable!()
    };
    first
        .deref_mut(&ctx)
        .attributes
        .set(Identifier::new("hlo_index"), Index(1));
    ret(&mut ctx, block, parameters[0]);
    assert!(xlahlo::verification::verify(&ctx, module.get_operation()).is_err());
}

fn alias_fixture(ctx: &mut Context, input: TypeHandle, result: TypeHandle) -> ModuleOp {
    let (module, container) = module(ctx);
    let (_, block, parameters) = computation(ctx, container, "main", vec![input, result], result);
    ret(ctx, block, parameters[1]);
    module
}

fn alias(kind: xlahlo::ir::AliasKind) -> xlahlo::ir::AliasEntry {
    xlahlo::ir::AliasEntry {
        output_shape_index: xlahlo::attributes::Axes(vec![]),
        parameter_number: 0,
        parameter_shape_index: xlahlo::attributes::Axes(vec![]),
        kind,
    }
}

#[test]
fn module_aliases_preserve_default_and_strict_size_policies() {
    let mut ctx = context();
    let input = TensorType::get(&ctx, ElementType::F32, vec![2]).unwrap();
    let output = TensorType::get(&ctx, ElementType::U8, vec![99]).unwrap();
    let module = alias_fixture(&mut ctx, input, output);
    let set = |value| {
        module
            .get_operation()
            .deref_mut(&ctx)
            .attributes
            .set(Identifier::new("hlo_input_output_aliases"), value)
    };
    set(InputOutputAliases(vec![alias(AliasKind::MayAlias)]));
    xlahlo::verification::verify(&ctx, module.get_operation()).unwrap();
    assert!(
        xlahlo::verification::verify_with_options(
            &ctx,
            module.get_operation(),
            xlahlo::verification::VerifyOptions {
                layout_sensitive: true,
                ..Default::default()
            }
        )
        .is_err()
    );
    set(InputOutputAliases(vec![alias(AliasKind::MustAlias)]));
    assert!(xlahlo::verification::verify(&ctx, module.get_operation()).is_err());
}

#[test]
fn duplicate_aliases_invalid_paths_and_donor_conflicts_are_rejected() {
    let mut ctx = context();
    let ty = TensorType::get(&ctx, ElementType::F32, vec![2]).unwrap();
    let module = alias_fixture(&mut ctx, ty, ty);
    let set = |value| {
        module
            .get_operation()
            .deref_mut(&ctx)
            .attributes
            .set(Identifier::new("hlo_input_output_aliases"), value)
    };
    let valid = alias(AliasKind::MustAlias);
    set(InputOutputAliases(vec![valid.clone()]));
    xlahlo::verification::verify(&ctx, module.get_operation()).unwrap();
    set(InputOutputAliases(vec![valid.clone(), valid.clone()]));
    assert!(xlahlo::verification::verify(&ctx, module.get_operation()).is_err());
    let mut invalid = valid.clone();
    invalid.parameter_number = 2;
    set(InputOutputAliases(vec![invalid]));
    assert!(xlahlo::verification::verify(&ctx, module.get_operation()).is_err());
    let mut invalid = valid.clone();
    invalid.output_shape_index.0.push(-1);
    set(InputOutputAliases(vec![invalid]));
    assert!(xlahlo::verification::verify(&ctx, module.get_operation()).is_err());
    set(InputOutputAliases(vec![valid]));
    module.get_operation().deref_mut(&ctx).attributes.set(
        Identifier::new("hlo_buffer_donors"),
        BufferDonors(vec![BufferDonorEntry {
            parameter_number: 0,
            parameter_shape_index: xlahlo::attributes::Axes(vec![]),
        }]),
    );
    assert!(xlahlo::verification::verify(&ctx, module.get_operation()).is_err());
    set(InputOutputAliases(vec![]));
    xlahlo::verification::verify(&ctx, module.get_operation()).unwrap();
}

#[test]
fn module_entry_and_host_program_shape_are_checked() {
    let mut ctx = context();
    let ty = TensorType::get(&ctx, ElementType::S32, vec![]).unwrap();
    let module = alias_fixture(&mut ctx, ty, ty);
    let wrong = TensorType::get(&ctx, ElementType::S32, vec![1]).unwrap();
    let declared = FunctionType::get(&ctx, vec![ty, ty], vec![wrong]).into();
    module.get_operation().deref_mut(&ctx).attributes.set(
        Identifier::new("hlo_host_program_shape"),
        TypeAttr::new(declared),
    );
    assert!(xlahlo::verification::verify(&ctx, module.get_operation()).is_err());
    let declared = FunctionType::get(&ctx, vec![ty, ty], vec![ty]).into();
    module.get_operation().deref_mut(&ctx).attributes.set(
        Identifier::new("hlo_host_program_shape"),
        TypeAttr::new(declared),
    );
    xlahlo::verification::verify(&ctx, module.get_operation()).unwrap();
    xlahlo::ir::set_entry(&ctx, module.get_operation(), Identifier::new("missing")).unwrap();
    assert!(xlahlo::verification::verify(&ctx, module.get_operation()).is_err());
}

#[test]
fn schedules_cover_live_computations_and_respect_control_dependencies() {
    let mut ctx = context();
    let ty = TensorType::get(&ctx, ElementType::S32, vec![]).unwrap();
    let (module, container) = module(&mut ctx);
    let (_, block, parameters) = computation(&mut ctx, container, "main", vec![ty, ty], ty);
    let definitions: Vec<_> = parameters
        .iter()
        .map(|value| {
            let xlahlo::pliron::value::DefiningEntity::Op(op) = value.defining_entity() else {
                unreachable!()
            };
            op
        })
        .collect();
    for (&op, name) in definitions.iter().zip(["first", "second"]) {
        op.deref_mut(&ctx).attributes.set(
            Identifier::new("hlo_name"),
            InstructionName(name.to_owned()),
        );
    }

    definitions[1].deref_mut(&ctx).attributes.set(
        Identifier::new("hlo_control_predecessors"),
        ControlPredecessors(vec!["first".to_owned()]),
    );
    ret(&mut ctx, block, parameters[1]);
    let schedule = |names: &[&str]| {
        module.get_operation().deref_mut(&ctx).attributes.set(
            Identifier::new("hlo_schedule"),
            Schedule(vec![InstructionSequence {
                computation: "main".into(),
                instructions: names.iter().map(|s| (*s).to_owned()).collect(),
            }]),
        );
    };
    schedule(&["first", "second"]);
    xlahlo::verification::verify(&ctx, module.get_operation()).unwrap();
    for invalid in [
        vec!["second", "first"],
        vec!["first"],
        vec!["first", "first"],
        vec!["missing", "second"],
    ] {
        schedule(&invalid);
        assert!(xlahlo::verification::verify(&ctx, module.get_operation()).is_err());
    }
    schedule(&["first", "second"]);
    definitions[0].deref_mut(&ctx).attributes.set(
        Identifier::new("hlo_control_predecessors"),
        ControlPredecessors(vec!["second".to_owned()]),
    );
    assert!(xlahlo::verification::verify(&ctx, module.get_operation()).is_err());
}

#[test]
fn scheduling_names_preserve_clone_and_rematerialization_exceptions() {
    let mut ctx = context();
    let ty = TensorType::get(&ctx, ElementType::S32, vec![]).unwrap();
    let (_, container) = module(&mut ctx);
    let (comp, block, parameters) = computation(&mut ctx, container, "main", vec![ty], ty);
    ret(&mut ctx, block, parameters[0]);
    let xlahlo::pliron::value::DefiningEntity::Op(parameter) = parameters[0].defining_entity()
    else {
        unreachable!()
    };
    parameter.deref_mut(&ctx).attributes.set(
        Identifier::new("hlo_metadata"),
        OpMetadata {
            scheduling_name: "original".to_owned(),
            ..Default::default()
        },
    );
    let options = xlahlo::verification::VerifyOptions {
        verify_instruction_name_unchanged: true,
        ..Default::default()
    };
    let name = |name: &str| {
        parameter.deref_mut(&ctx).attributes.set(
            Identifier::new("hlo_name"),
            InstructionName(name.to_owned()),
        )
    };
    name("changed");
    xlahlo::verification::verify(&ctx, comp.get_operation()).unwrap();
    assert!(
        xlahlo::verification::verify_with_options(&ctx, comp.get_operation(), options).is_err()
    );
    for allowed in ["original", "original.clone.1", "original.remat.2"] {
        name(allowed);
        xlahlo::verification::verify_with_options(&ctx, comp.get_operation(), options).unwrap();
    }
}

#[test]
fn native_ssa_buffer_values_have_at_most_one_writer() {
    for writers in 0..=2 {
        let mut ctx = context();
        let array = TensorType::get(&ctx, ElementType::F32, vec![4]).unwrap();
        let buffer = xlahlo::shape::buffer(&ctx, array).unwrap();
        let (_, container) = module(&mut ctx);
        let (comp, block, _) = computation(&mut ctx, container, "main", vec![], buffer);
        let create = CustomCallOp::new(&mut ctx, buffer, vec![], StringAttr::from("CreateBuffer"));
        create.get_operation().insert_at_back(block, &ctx);
        let original = create.result(&ctx);
        let mut root = original;
        for _ in 0..writers {
            let update =
                CustomCallOp::new(&mut ctx, buffer, vec![original], StringAttr::from("Update"));
            update.get_operation().deref_mut(&ctx).attributes.set(
                Identifier::new("hlo_aliases"),
                OutputOperandAliases(vec![OutputOperandAliasing {
                    output_shape_index: Axes(vec![]),
                    operand_index: 0,
                    operand_shape_index: Axes(vec![]),
                }]),
            );
            update.get_operation().insert_at_back(block, &ctx);
            root = update.result(&ctx);
        }

        ret(&mut ctx, block, root);
        assert_eq!(
            xlahlo::verification::verify(&ctx, comp.get_operation()).is_ok(),
            writers <= 1,
            "writer count {writers}"
        );
    }
}

#[test]
fn common_layout_callback_and_host_memory_policy_are_enforced() {
    let mut ctx = context();
    let input = TensorType::with_layout(&ctx, ElementType::F32, vec![2, 3], vec![0, 1]).unwrap();
    let output = TensorType::with_layout(&ctx, ElementType::F32, vec![2, 3], vec![1, 0]).unwrap();
    let (_, container) = module(&mut ctx);
    let (comp, block, parameters) = computation(&mut ctx, container, "main", vec![input], output);
    let copy = CopyOp::new(&mut ctx, output, parameters);
    copy.get_operation().insert_at_back(block, &ctx);
    let result = copy.result(&ctx);
    ret(&mut ctx, block, result);
    xlahlo::verification::verify(&ctx, comp.get_operation()).unwrap();
    let options = xlahlo::verification::VerifyOptions {
        instruction_can_change_layout: Some(|_, _| false),
        ..Default::default()
    };
    assert!(
        xlahlo::verification::verify_with_options(&ctx, comp.get_operation(), options).is_err()
    );
    let mut layout = xlahlo::shape::layout(&ctx, input).unwrap().unwrap().clone();
    layout.memory_space = 5;
    let host = xlahlo::shape::with_layout(&ctx, input, layout).unwrap();
    let (_, container) = module(&mut ctx);
    let (comp, block, parameters) = computation(&mut ctx, container, "main", vec![host], host);
    ret(&mut ctx, block, parameters[0]);
    xlahlo::verification::verify(&ctx, comp.get_operation()).unwrap();
    assert!(
        xlahlo::verification::verify_with_options(
            &ctx,
            comp.get_operation(),
            xlahlo::verification::VerifyOptions {
                verify_no_host_memory_space: true,
                ..Default::default()
            }
        )
        .is_err()
    );
}

#[test]
fn unbounded_shapes_require_instruction_specific_authorization() {
    let mut ctx = context();
    let one = TensorType::get(&ctx, ElementType::F32, vec![1]).unwrap();
    let mut shape = xlahlo::types::tensor(&ctx, one).unwrap().clone();
    shape.dimensions[0] = i64::MIN;
    shape.dynamic_dimensions[0] = true;
    let unbounded = Type::instantiate(shape, &ctx).into();
    let (_, container) = module(&mut ctx);
    let (comp, block, parameters) =
        computation(&mut ctx, container, "main", vec![unbounded], unbounded);
    ret(&mut ctx, block, parameters[0]);
    assert!(xlahlo::verification::verify(&ctx, comp.get_operation()).is_err());
    xlahlo::verification::verify_with_options(
        &ctx,
        comp.get_operation(),
        xlahlo::verification::VerifyOptions {
            supported_unbounded_dynamic_op: Some(|_, _| true),
            ..Default::default()
        },
    )
    .unwrap();
}

#[test]
fn metadata_payload_references_are_checked_against_native_module_storage() {
    let mut ctx = context();
    let ty = TensorType::get(&ctx, ElementType::S32, vec![]).unwrap();
    let (module, container) = module(&mut ctx);
    let (_, block, parameters) = computation(&mut ctx, container, "main", vec![ty], ty);
    ret(&mut ctx, block, parameters[0]);
    let xlahlo::pliron::value::DefiningEntity::Op(parameter) = parameters[0].defining_entity()
    else {
        unreachable!()
    };
    let payload = |id| {
        parameter.deref_mut(&ctx).attributes.set(
            Identifier::new("hlo_metadata"),
            OpMetadata {
                metadata_payload: Payload::Id(id),
                ..Default::default()
            },
        )
    };
    payload(0);
    assert!(xlahlo::verification::verify(&ctx, module.get_operation()).is_err());
    module.get_operation().deref_mut(&ctx).attributes.set(
        Identifier::new("hlo_metadata_payloads"),
        MetadataPayloads(vec![Bytes(vec![1, 2, 3])]),
    );
    xlahlo::verification::verify(&ctx, module.get_operation()).unwrap();
    for invalid in [-1, 1, i64::MAX] {
        payload(invalid);
        assert!(xlahlo::verification::verify(&ctx, module.get_operation()).is_err());
    }
}

#[test]
fn module_configuration_overrides_context_unless_explicitly_ignored() {
    let mut ctx = context();
    let ty = TensorType::get(&ctx, ElementType::S32, vec![]).unwrap();
    let module = alias_fixture(&mut ctx, ty, ty);
    let configuration = xlahlo::ir::ModuleConfig {
        num_partitions: 0,
        use_spmd_partitioning: true,
        ..Default::default()
    };
    xlahlo::ir::set_config(&ctx, module.get_operation(), configuration.clone()).unwrap();
    assert_eq!(
        xlahlo::ir::config(&ctx, module.get_operation()).unwrap(),
        configuration
    );
    assert!(xlahlo::verification::verify(&ctx, module.get_operation()).is_err());
    xlahlo::verification::verify_with_options(
        &ctx,
        module.get_operation(),
        xlahlo::verification::VerifyOptions {
            ignore_module_config: true,
            ..Default::default()
        },
    )
    .unwrap();
}

#[test]
fn module_configuration_defaults_and_channel_id_sensitivity_match_xla() {
    let default = ModuleConfig::default();
    assert!(default.debug_options.xla_backend_extra_options.is_empty());
    assert!(default.entry_computation_layout.is_none());
    assert!(default.channel_id_sensitive());

    let mut ctx = context();
    let ty = TensorType::get(&ctx, ElementType::S32, vec![]).unwrap();
    let module = alias_fixture(&mut ctx, ty, ty);
    xlahlo::ir::set_config(&ctx, module.get_operation(), default).unwrap();
    let assembly = module.get_operation().disp(&ctx).to_string();
    let mut parsed = context();
    let root = parse_from_str(
        spaced(Operation::top_level_parser()),
        &mut parsed,
        &assembly,
    )
    .unwrap();
    let recovered = xlahlo::ir::config(&parsed, root).unwrap();
    assert!(recovered.debug_options.xla_backend_extra_options.is_empty());
    assert!(recovered.entry_computation_layout.is_none());

    for (replica_count, num_partitions) in [(1, 1), (4, 8)] {
        assert!(
            ModuleConfig {
                replica_count,
                num_partitions,
                use_spmd_partitioning: false,
                ..Default::default()
            }
            .channel_id_sensitive()
        );
        assert!(
            !ModuleConfig {
                replica_count,
                num_partitions,
                use_spmd_partitioning: true,
                ..Default::default()
            }
            .channel_id_sensitive()
        );
    }
}

#[test]
fn backend_extra_option_aliases_and_entry_layout_round_trip_through_module_ir() {
    let mut ctx = context();
    let parameter =
        TensorType::with_layout(&ctx, ElementType::F32, vec![2, 3], vec![0, 1]).unwrap();
    let result = TensorType::with_layout(&ctx, ElementType::F32, vec![2, 3], vec![1, 0]).unwrap();
    let module = alias_fixture(&mut ctx, parameter, result);
    let options = [
        "xla_disable_while_loop_dce",
        "xla.disable_while_loop_dce",
        "xla_preserve_tuple_indices",
        "xla.preserve_tuple_indices",
    ];
    let backend_options = options
        .into_iter()
        .map(|key| (key.to_owned(), "true".to_owned()))
        .collect();
    let configuration = ModuleConfig {
        debug_options: DebugOptions {
            xla_backend_extra_options: backend_options,
        },
        entry_computation_layout: Some(EntryComputationLayout::new(
            vec![parameter, result],
            result,
        )),
        ..Default::default()
    };
    xlahlo::ir::set_config(&ctx, module.get_operation(), configuration.clone()).unwrap();
    assert_eq!(
        xlahlo::ir::config(&ctx, module.get_operation()).unwrap(),
        configuration.clone()
    );
    xlahlo::verification::verify(&ctx, module.get_operation()).unwrap();
    for alias in options {
        assert_eq!(
            configuration
                .debug_options
                .xla_backend_extra_options
                .get(alias)
                .map(String::as_str),
            Some("true")
        );
    }

    let assembly = module.get_operation().disp(&ctx).to_string();
    let mut parsed = context();
    let root = parse_from_str(
        spaced(Operation::top_level_parser()),
        &mut parsed,
        &assembly,
    )
    .unwrap();
    let recovered = xlahlo::ir::config(&parsed, root).unwrap();
    assert_eq!(
        recovered.debug_options.xla_backend_extra_options,
        configuration.debug_options.xla_backend_extra_options
    );
    let recovered_layout = recovered.entry_computation_layout.unwrap();
    assert_eq!(recovered_layout.parameter_shapes.len(), 2);
    let source_layout = configuration.entry_computation_layout.unwrap();
    for (source, recovered) in source_layout
        .parameter_shapes
        .iter()
        .chain(std::iter::once(&source_layout.result_shape))
        .zip(
            recovered_layout
                .parameter_shapes
                .iter()
                .chain(std::iter::once(&recovered_layout.result_shape)),
        )
    {
        let source_layout = xlahlo::shape::layout(&ctx, *source)
            .unwrap()
            .map(|layout| (*layout).clone());
        let recovered_layout = xlahlo::shape::layout(&parsed, *recovered)
            .unwrap()
            .map(|layout| (*layout).clone());
        assert_eq!(source_layout, recovered_layout);
    }
    xlahlo::verification::verify(&parsed, root).unwrap();
}

#[test]
fn entry_computation_layout_verification_rejects_mismatches_and_recovers() {
    let mut ctx = context();
    let ty = TensorType::get(&ctx, ElementType::S32, vec![2]).unwrap();
    let module = alias_fixture(&mut ctx, ty, ty);
    let config = |parameters, result| ModuleConfig {
        entry_computation_layout: Some(EntryComputationLayout::new(parameters, result)),
        ..Default::default()
    };

    xlahlo::ir::set_config(&ctx, module.get_operation(), config(vec![ty], ty)).unwrap();
    assert!(xlahlo::verification::verify(&ctx, module.get_operation()).is_err());

    let wrong = TensorType::get(&ctx, ElementType::S32, vec![3]).unwrap();
    xlahlo::ir::set_config(&ctx, module.get_operation(), config(vec![wrong, ty], ty)).unwrap();
    assert!(xlahlo::verification::verify(&ctx, module.get_operation()).is_err());

    xlahlo::ir::set_config(&ctx, module.get_operation(), config(vec![ty, ty], wrong)).unwrap();
    assert!(xlahlo::verification::verify(&ctx, module.get_operation()).is_err());

    xlahlo::ir::set_config(&ctx, module.get_operation(), config(vec![ty, ty], ty)).unwrap();
    xlahlo::verification::verify(&ctx, module.get_operation()).unwrap();

    let parameter =
        TensorType::with_layout(&ctx, ElementType::F32, vec![2, 3], vec![0, 1]).unwrap();
    let result = TensorType::with_layout(&ctx, ElementType::F32, vec![2, 3], vec![1, 0]).unwrap();
    let alternate =
        TensorType::with_layout(&ctx, ElementType::F32, vec![2, 3], vec![1, 0]).unwrap();
    let layout_module = alias_fixture(&mut ctx, parameter, result);
    xlahlo::ir::set_config(
        &ctx,
        layout_module.get_operation(),
        config(vec![alternate, result], result),
    )
    .unwrap();
    assert!(xlahlo::verification::verify(&ctx, layout_module.get_operation()).is_err());
    xlahlo::ir::set_config(
        &ctx,
        layout_module.get_operation(),
        config(vec![parameter, result], result),
    )
    .unwrap();
    xlahlo::verification::verify(&ctx, layout_module.get_operation()).unwrap();
}
