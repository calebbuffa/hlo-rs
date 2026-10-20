#[path = "control_support/mod.rs"]
mod support;
use support::*;
use xlahlo::{
    attributes::FrontendAttributes,
    ir::{self, FusionParameter, FusionParameters},
    ops::control::{FusionOp, ParameterOp},
    ops::elementwise::NegateOp,
    pliron::{
        basic_block::BasicBlock, common_traits::Verify, identifier::Identifier,
        linked_list::ContainsLinkedList, op::Op, parsable::Parsable, printable::Printable,
    },
    types::ElementType::F32,
};

fn fusion_with_two_parameters(
    ctx: &mut xlahlo::pliron::context::Context,
) -> (FusionOp, BasicBlockPtr) {
    let ty = shape(ctx, F32, &[]);
    let p0 = ParameterOp::new(ctx, ty, vec![], xlahlo::attributes::Index(0));
    let p1 = ParameterOp::new(ctx, ty, vec![], xlahlo::attributes::Index(1));
    let fusion = FusionOp::new(
        ctx,
        ty,
        vec![p0.result(ctx), p1.result(ctx)],
        xlahlo::attributes::FusionKind::Loop,
    );
    let block = BasicBlock::new(ctx, None, vec![ty, ty]);
    block.insert_at_back(fusion.body(ctx), ctx);
    let root = block.deref(ctx).get_argument(0);
    let neg = NegateOp::new(ctx, ty, vec![root]);
    neg.get_operation().insert_at_back(block, ctx);
    let ret = xlahlo::ops::ReturnOp::new(ctx, neg.result(ctx));
    ret.get_operation().insert_at_back(block, ctx);
    (fusion, block)
}

type BasicBlockPtr = xlahlo::pliron::context::Ptr<xlahlo::pliron::basic_block::BasicBlock>;

#[test]
fn fusion_edits_invalidate_enclosing_order_but_preserve_it_on_noops_and_errors() {
    use xlahlo::{
        ir::instruction_order::{InstructionOrder, OrderPermutation, set_order},
        ops::elementwise::AddOp,
        pliron::{operation::Operation, parsable::parse_from_str},
    };

    let mut ctx = context();
    let ty = shape(&ctx, F32, &[]);
    let (owner, outer, arguments) = computation(&mut ctx, "main", vec![ty, ty], ty);
    for (index, argument) in arguments.iter().enumerate() {
        attr(
            &ctx,
            argument.defining_op().unwrap(),
            "hlo_name",
            ir::InstructionName(format!("outer.{index}")),
        );
    }
    let fusion = FusionOp::new(
        &mut ctx,
        ty,
        arguments,
        xlahlo::attributes::FusionKind::Loop,
    );
    fusion.get_operation().insert_at_back(outer, &ctx);
    attr(
        &ctx,
        fusion.get_operation(),
        "hlo_name",
        ir::InstructionName("fusion".into()),
    );
    let body = BasicBlock::new(&mut ctx, None, vec![ty, ty]);
    body.insert_at_back(fusion.body(&ctx), &ctx);
    let left = body.deref(&ctx).get_argument(0);
    let right = body.deref(&ctx).get_argument(1);
    let add = AddOp::new(&mut ctx, ty, vec![left, right]);
    add.get_operation().insert_at_back(body, &ctx);
    attr(
        &ctx,
        add.get_operation(),
        "hlo_name",
        ir::InstructionName("add".into()),
    );
    let result = add.result(&ctx);
    ret(&mut ctx, body, result);
    let result = fusion.result(&ctx);
    ret(&mut ctx, outer, result);
    let module = module(&mut ctx, &[owner], "main");
    let mut parameters = FusionParameters(vec![
        FusionParameter {
            name: Some("left".into()),
            ..Default::default()
        },
        FusionParameter {
            name: Some("right".into()),
            ..Default::default()
        },
    ]);
    ir::set_fusion_parameters(&ctx, body, parameters.clone()).unwrap();
    let order = InstructionOrder {
        blocks: vec![OrderPermutation {
            subject: "fusion".into(),
            members: vec!["right".into(), "left".into(), "add".into()],
        }],
        ..Default::default()
    };
    let key = Identifier::new("hlo_instruction_order");
    set_order(&ctx, owner.get_operation(), order.clone()).unwrap();
    xlahlo::verification::verify(&ctx, module.get_operation()).unwrap();

    ir::set_fusion_parameters(&ctx, body, parameters.clone()).unwrap();
    parameters.0[0].frontend_attributes = FrontendAttributes::new([("keep".into(), "yes".into())]);
    ir::set_fusion_parameters(&ctx, body, parameters.clone()).unwrap();
    assert_eq!(
        owner
            .get_operation()
            .deref(&ctx)
            .attributes
            .get::<InstructionOrder>(&key),
        Some(&order)
    );
    let before = module.get_operation().disp(&ctx).to_string();
    let mut invalid = parameters.clone();
    invalid.0[1].name = Some("left".into());
    assert!(ir::set_fusion_parameters(&ctx, body, invalid).is_err());
    assert!(ir::remove_fusion_parameter(&ctx, body, 1).is_err());
    assert!(ir::add_fusion_control_dependency(&ctx, body, "missing", "add").is_err());
    assert!(ir::remove_fusion_control_dependency(&ctx, body, "left", "add").is_err());
    assert_eq!(module.get_operation().disp(&ctx).to_string(), before);

    ir::add_fusion_control_dependency(&ctx, body, "left", "right").unwrap();
    assert!(
        !owner
            .get_operation()
            .deref(&ctx)
            .attributes
            .0
            .contains_key(&key)
    );
    set_order(&ctx, owner.get_operation(), order.clone()).unwrap();
    ir::remove_fusion_control_dependency(&ctx, body, "left", "right").unwrap();
    assert!(
        !owner
            .get_operation()
            .deref(&ctx)
            .attributes
            .0
            .contains_key(&key)
    );
    set_order(&ctx, owner.get_operation(), order.clone()).unwrap();
    ir::add_fusion_control_dependency(&ctx, body, "right", "add").unwrap();
    assert!(
        !owner
            .get_operation()
            .deref(&ctx)
            .attributes
            .0
            .contains_key(&key)
    );
    set_order(&ctx, owner.get_operation(), order.clone()).unwrap();
    ir::remove_fusion_control_dependency(&ctx, body, "right", "add").unwrap();
    assert!(
        !owner
            .get_operation()
            .deref(&ctx)
            .attributes
            .0
            .contains_key(&key)
    );

    set_order(&ctx, owner.get_operation(), order).unwrap();
    parameters.0[1].name = Some("renamed".into());
    ir::set_fusion_parameters(&ctx, body, parameters.clone()).unwrap();
    assert!(
        !owner
            .get_operation()
            .deref(&ctx)
            .attributes
            .0
            .contains_key(&key)
    );
    set_order(&ctx, owner.get_operation(), InstructionOrder::default()).unwrap();
    Operation::replace_operand(add.get_operation(), &ctx, 1, left);
    ir::remove_fusion_parameter(&ctx, body, 1).unwrap();
    assert!(
        !owner
            .get_operation()
            .deref(&ctx)
            .attributes
            .0
            .contains_key(&key)
    );
    assert_eq!(
        ir::fusion_parameters(&ctx, body).unwrap().0,
        vec![parameters.0.remove(0)]
    );
    xlahlo::verification::verify(&ctx, module.get_operation()).unwrap();
    let text = module.get_operation().disp(&ctx).to_string();
    let mut fresh = context();
    let parsed = parse_from_str(Operation::top_level_parser(), &mut fresh, &text).unwrap();
    xlahlo::verification::verify(&fresh, parsed).unwrap();
    Operation::erase(parsed, &mut fresh);
    Operation::erase(module.get_operation(), &mut ctx);
    assert!(fresh.is_ir_empty());
    assert!(ctx.is_ir_empty());
}

#[test]
fn fusion_parameter_metadata_roundtrips_with_frontend_attributes_and_controls() {
    let mut ctx = context();
    let parameter = FusionParameter {
        name: Some("arg.0".to_owned()),
        frontend_attributes: FrontendAttributes::new([(
            "xla_allow_dce_side_effecting_op".to_owned(),
            "false".to_owned(),
        )]),
        control_predecessors: vec!["before".to_owned()],
        ..Default::default()
    };
    parameter.verify(&ctx).unwrap();
    let text = parameter.disp(&ctx).to_string();
    let parsed =
        xlahlo::pliron::parsable::parse_from_str(FusionParameter::parser(()), &mut ctx, &text)
            .unwrap_or_else(|error| panic!("{text}: {error}"));
    assert_eq!(parsed, parameter);

    let parameters = FusionParameters(vec![parameter]);
    let text = parameters.disp(&ctx).to_string();
    let parsed =
        xlahlo::pliron::parsable::parse_from_str(FusionParameters::parser(()), &mut ctx, &text)
            .unwrap_or_else(|error| panic!("{text}: {error}"));
    assert_eq!(parsed, parameters);
}

#[test]
fn old_positional_assembly_defaults_annotations_and_new_form_preserves_presence() {
    let mut ctx = context();
    for name in [None, Some("legacy".to_owned())] {
        let parameter = FusionParameter {
            name,
            ..Default::default()
        };
        let frontend = parameter.frontend_attributes.disp(&ctx).to_string();
        let name = parameter
            .name
            .as_ref()
            .map(|name| name.disp(&ctx).to_string())
            .unwrap_or_default();
        let old = format!("<{name}, {frontend}, [] >");
        let parsed =
            xlahlo::pliron::parsable::parse_from_str(FusionParameter::parser(()), &mut ctx, &old)
                .unwrap_or_else(|error| panic!("{old}: {error}"));
        assert_eq!(parsed, parameter);
        let mut annotated = parameter;
        annotated.metadata = Some(xlahlo::attributes::OpMetadata::default());
        annotated.sharding = Some(xlahlo::sharding::Sharding::replicated());
        annotated.original_value = Some(xlahlo::original_value::OriginalValue::default());
        let text = annotated.disp(&ctx).to_string();
        let parsed =
            xlahlo::pliron::parsable::parse_from_str(FusionParameter::parser(()), &mut ctx, &text)
                .unwrap_or_else(|error| panic!("{text}: {error}"));
        assert_eq!(parsed, annotated);
        assert_ne!(parsed.metadata, None);
    }
}

#[test]
fn typed_parameter_annotations_roundtrip_and_validate_exact_argument_shapes() {
    use xlahlo::original_value::{OriginalArray, OriginalValue, OriginalValueElement};
    let mut ctx = context();
    let (fusion, block) = fusion_with_two_parameters(&mut ctx);
    let neg = block.deref(&ctx).iter(&ctx).next().unwrap();
    let operands = vec![
        block.deref(&ctx).get_argument(0),
        block.deref(&ctx).get_argument(1),
    ];
    let ty = shape(&ctx, F32, &[]);
    let add = xlahlo::ops::elementwise::AddOp::new(&mut ctx, ty, operands);
    add.get_operation().insert_before(&ctx, neg);
    xlahlo::pliron::operation::Operation::replace_operand(neg, &ctx, 0, add.result(&ctx));
    let parameter = FusionParameter {
        metadata: Some(xlahlo::attributes::OpMetadata {
            op_name: "source".into(),
            source_file: "model.rs".into(),
            source_line: 7,
            ..Default::default()
        }),
        sharding: Some(xlahlo::sharding::Sharding::replicated()),
        original_value: Some(OriginalValue {
            elements: vec![OriginalValueElement {
                shape_index: vec![],
                original_array: vec![OriginalArray {
                    instruction_name: "source".into(),
                    shape_index: vec![],
                }],
            }],
            ..Default::default()
        }),
        ..Default::default()
    };
    let text = parameter.disp(&ctx).to_string();
    let parsed =
        xlahlo::pliron::parsable::parse_from_str(FusionParameter::parser(()), &mut ctx, &text)
            .unwrap_or_else(|error| panic!("{text}: {error}"));
    assert_eq!(parsed, parameter);
    let records = FusionParameters(vec![parameter.clone(), FusionParameter::default()]);
    ir::set_fusion_parameters(&ctx, block, records.clone()).unwrap();
    fusion.verify(&ctx).unwrap();
    let mut invalid = records.clone();
    invalid.0[0].original_value.as_mut().unwrap().elements[0].shape_index = vec![0];
    assert!(ir::set_fusion_parameters(&ctx, block, invalid).is_err());
    assert_eq!(ir::fusion_parameters(&ctx, block).unwrap(), records);
    let mut invalid = records.clone();
    invalid.0[0].sharding = Some(xlahlo::sharding::Sharding::tuple(vec![
        xlahlo::sharding::Sharding::replicated(),
    ]));
    assert!(ir::set_fusion_parameters(&ctx, block, invalid).is_err());
    assert_eq!(ir::fusion_parameters(&ctx, block).unwrap(), records);
}

#[test]
fn removing_unused_fusion_parameter_atomically_removes_operand_and_metadata() {
    let mut ctx = context();
    let (fusion, block) = fusion_with_two_parameters(&mut ctx);
    ir::set_fusion_parameters(
        &ctx,
        block,
        FusionParameters(vec![
            FusionParameter {
                name: Some("arg.0".into()),
                ..Default::default()
            },
            FusionParameter {
                name: Some("arg.1".into()),
                frontend_attributes: FrontendAttributes::new([("origin".into(), "dead".into())]),
                ..Default::default()
            },
        ]),
    )
    .unwrap();

    let removed = ir::remove_fusion_parameter(&ctx, block, 1).unwrap();
    assert_eq!(removed.name.as_deref(), Some("arg.1"));
    assert_eq!(removed.frontend_attributes.get("origin"), Some("dead"));
    assert_eq!(block.deref(&ctx).get_num_arguments(), 1);
    assert_eq!(fusion.get_operation().deref(&ctx).get_num_operands(), 1);
    assert_eq!(
        ir::fusion_parameters(&ctx, block).unwrap().0,
        vec![FusionParameter {
            name: Some("arg.0".into()),
            ..Default::default()
        }]
    );
    fusion.verify(&ctx).unwrap();
}

#[test]
fn parameter_removal_refuses_used_or_control_connected_parameters_without_mutation() {
    let mut ctx = context();
    let (fusion, block) = fusion_with_two_parameters(&mut ctx);
    ir::set_fusion_parameters(
        &ctx,
        block,
        FusionParameters(vec![
            FusionParameter {
                name: Some("arg.0".into()),
                ..Default::default()
            },
            FusionParameter {
                name: Some("arg.1".into()),
                ..Default::default()
            },
        ]),
    )
    .unwrap();
    let error = ir::remove_fusion_parameter(&ctx, block, 0).unwrap_err();
    assert!(error.to_string().contains("still used"));
    assert_eq!(block.deref(&ctx).get_num_arguments(), 2);
    assert_eq!(fusion.get_operation().deref(&ctx).get_num_operands(), 2);

    ir::add_fusion_control_dependency(&ctx, block, "arg.0", "arg.1").unwrap();
    assert!(ir::parameter_has_control_dependency(&ctx, block, 0).unwrap());
    assert!(ir::parameter_has_control_dependency(&ctx, block, 1).unwrap());
    let error = ir::remove_fusion_parameter(&ctx, block, 1).unwrap_err();
    assert!(error.to_string().contains("control dependency"));
    assert_eq!(block.deref(&ctx).get_num_arguments(), 2);
    assert_eq!(fusion.get_operation().deref(&ctx).get_num_operands(), 2);
}

#[test]
fn parameter_to_instruction_control_edges_use_native_instruction_predecessors() {
    let mut ctx = context();
    let (fusion, block) = fusion_with_two_parameters(&mut ctx);
    ir::set_fusion_parameters(
        &ctx,
        block,
        FusionParameters(vec![
            FusionParameter {
                name: Some("arg.0".into()),
                ..Default::default()
            },
            FusionParameter {
                name: Some("arg.1".into()),
                ..Default::default()
            },
        ]),
    )
    .unwrap();
    let operation = block.deref(&ctx).iter(&ctx).next().unwrap();
    operation.deref_mut(&ctx).attributes.set(
        Identifier::new("hlo_name"),
        xlahlo::ir::InstructionName("neg".into()),
    );
    ir::add_fusion_control_dependency(&ctx, block, "arg.1", "neg").unwrap();
    assert!(ir::parameter_has_control_dependency(&ctx, block, 1).unwrap());

    let predecessors = operation
        .deref(&ctx)
        .attributes
        .get::<ir::ControlPredecessors>(&Identifier::new("hlo_control_predecessors"))
        .cloned()
        .unwrap();
    assert_eq!(predecessors.0, ["arg.1"]);
    fusion.verify(&ctx).unwrap();
}

#[test]
fn removing_control_edges_preserves_remaining_parameter_metadata() {
    let mut ctx = context();
    let (fusion, block) = fusion_with_two_parameters(&mut ctx);
    ir::set_fusion_parameters(
        &ctx,
        block,
        FusionParameters(vec![
            FusionParameter {
                name: Some("arg.0".into()),
                ..Default::default()
            },
            FusionParameter {
                name: Some("arg.1".into()),
                frontend_attributes: FrontendAttributes::new([("keep".into(), "yes".into())]),
                ..Default::default()
            },
        ]),
    )
    .unwrap();
    let operation = block.deref(&ctx).iter(&ctx).next().unwrap();
    operation.deref_mut(&ctx).attributes.set(
        Identifier::new("hlo_name"),
        ir::InstructionName("neg".into()),
    );
    ir::add_fusion_control_dependency(&ctx, block, "arg.0", "arg.1").unwrap();
    ir::add_fusion_control_dependency(&ctx, block, "arg.1", "neg").unwrap();
    fusion.verify(&ctx).unwrap();

    ir::remove_fusion_control_dependency(&ctx, block, "arg.0", "arg.1").unwrap();
    let parameter = ir::fusion_parameters(&ctx, block).unwrap().0.remove(1);
    assert_eq!(parameter.name.as_deref(), Some("arg.1"));
    assert_eq!(parameter.frontend_attributes.get("keep"), Some("yes"));
    assert!(parameter.control_predecessors.is_empty());

    ir::remove_fusion_control_dependency(&ctx, block, "arg.1", "neg").unwrap();
    let operation_predecessors = operation
        .deref(&ctx)
        .attributes
        .get::<ir::ControlPredecessors>(&Identifier::new("hlo_control_predecessors"))
        .unwrap()
        .clone();
    assert!(operation_predecessors.0.is_empty());
}

#[test]
fn fusion_control_verification_rejects_unknown_endpoints_and_cycles() {
    let mut ctx = context();
    let (_fusion, block) = fusion_with_two_parameters(&mut ctx);
    ir::set_fusion_parameters(
        &ctx,
        block,
        FusionParameters(vec![
            FusionParameter {
                name: Some("arg.0".into()),
                ..Default::default()
            },
            FusionParameter {
                name: Some("arg.1".into()),
                ..Default::default()
            },
        ]),
    )
    .unwrap();
    let operation = block.deref(&ctx).iter(&ctx).next().unwrap();
    operation.deref_mut(&ctx).attributes.set(
        Identifier::new("hlo_name"),
        ir::InstructionName("neg".into()),
    );
    operation.deref_mut(&ctx).attributes.set(
        Identifier::new("hlo_control_predecessors"),
        ir::ControlPredecessors(vec!["missing".into()]),
    );
    assert!(
        ir::verify_fusion_parameters(&ctx, block)
            .unwrap_err()
            .to_string()
            .contains("unknown fusion control predecessor")
    );

    operation.deref_mut(&ctx).attributes.set(
        Identifier::new("hlo_control_predecessors"),
        ir::ControlPredecessors::default(),
    );
    ir::add_fusion_control_dependency(&ctx, block, "neg", "arg.0").unwrap();
    assert!(
        ir::verify_fusion_parameters(&ctx, block)
            .unwrap_err()
            .to_string()
            .contains("cycle")
    );
}

#[test]
fn mismatched_fusion_arguments_and_operands_are_not_partially_rewritten() {
    let mut ctx = context();
    let ty = shape(&ctx, F32, &[]);
    let parameter = ParameterOp::new(&mut ctx, ty, vec![], xlahlo::attributes::Index(0));
    let operand = parameter.result(&ctx);
    let fusion = FusionOp::new(
        &mut ctx,
        ty,
        vec![operand, operand],
        xlahlo::attributes::FusionKind::Loop,
    );
    let block = BasicBlock::new(&mut ctx, None, vec![ty]);
    block.insert_at_back(fusion.body(&ctx), &ctx);
    let root = block.deref(&ctx).get_argument(0);
    let ret = xlahlo::ops::ReturnOp::new(&mut ctx, root);
    ret.get_operation().insert_at_back(block, &ctx);

    let error = ir::remove_fusion_parameter(&ctx, block, 0).unwrap_err();
    assert!(error.to_string().contains("not aligned"));
    assert_eq!(block.deref(&ctx).get_num_arguments(), 1);
    assert_eq!(fusion.get_operation().deref(&ctx).get_num_operands(), 2);
}
