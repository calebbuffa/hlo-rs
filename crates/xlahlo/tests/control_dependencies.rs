#[path = "control_support/mod.rs"]
mod support;
use support::*;
use xlahlo::{
    attributes::{Bytes, FusionKind, Payload},
    ir::{
        BackendConfigPayload, ControlPredecessors, InstructionName, copy_all_control_deps_from,
        drop_all_control_deps,
    },
    ops::{control::FusionOp, elementwise::NegateOp},
    pliron::{
        basic_block::BasicBlock,
        builtin::{attributes::StringAttr, ops::ModuleOp},
        context::{Context, Ptr},
        identifier::Identifier,
        irfmt::parsers::spaced,
        linked_list::ContainsLinkedList,
        op::Op,
        operation::Operation,
        parsable::parse_from_str,
        printable::Printable,
        value::DefiningEntity,
    },
    types::ElementType,
    verification,
};

struct Fixture {
    module: ModuleOp,
    nodes: Vec<Ptr<Operation>>,
}

fn fixture(ctx: &mut Context) -> Fixture {
    let ty = shape(ctx, ElementType::F32, &[]);
    let (computation, block, args) = computation(ctx, "main", vec![ty], ty);
    let mut nodes = Vec::new();
    for name in ["a", "b", "destination", "source", "target", "unrelated"] {
        let instruction = NegateOp::new(ctx, ty, args.clone());
        let operation = instruction.get_operation();
        attr(ctx, operation, "hlo_name", InstructionName(name.into()));
        operation.insert_at_back(block, ctx);
        nodes.push(operation);
    }
    for (node, predecessors) in [
        (2, vec!["b"]),
        (3, vec!["a"]),
        (4, vec!["source", "destination", "unrelated"]),
    ] {
        set_predecessors(ctx, nodes[node], &predecessors);
    }
    let root = nodes[4].deref(ctx).get_result(0);
    ret(ctx, block, root);
    Fixture {
        module: module(ctx, &[computation], "main"),
        nodes,
    }
}

fn set_predecessors(ctx: &Context, operation: Ptr<Operation>, names: &[&str]) {
    attr(
        ctx,
        operation,
        "hlo_control_predecessors",
        ControlPredecessors(names.iter().map(|name| (*name).into()).collect()),
    );
}

fn predecessors(ctx: &Context, operation: Ptr<Operation>) -> Vec<String> {
    operation
        .deref(ctx)
        .attributes
        .get::<ControlPredecessors>(&Identifier::new("hlo_control_predecessors"))
        .map(|predecessors| predecessors.0.clone())
        .unwrap_or_default()
}

#[test]
fn copy_and_drop_relay_ordered_edges_idempotently_and_preserve_other_metadata() {
    let mut ctx = context();
    let fixture = fixture(&mut ctx);
    let destination = fixture.nodes[2];
    let source = fixture.nodes[3];
    let target = fixture.nodes[4];
    let payload = BackendConfigPayload(Payload::Value(Bytes(vec![0, 255])));
    attr(
        &ctx,
        destination,
        "hlo_backend_config_payload",
        payload.clone(),
    );
    attr(
        &ctx,
        source,
        "hlo_backend_config",
        StringAttr::new("source-only".into()),
    );
    verification::verify(&ctx, fixture.module.get_operation()).unwrap();
    let operand = destination.deref(&ctx).get_operand(0);
    copy_all_control_deps_from(&ctx, destination, source).unwrap();
    assert_eq!(predecessors(&ctx, destination), ["b", "a"]);
    assert_eq!(predecessors(&ctx, source), ["a"]);
    assert_eq!(
        predecessors(&ctx, target),
        ["source", "destination", "unrelated"],
    );
    let copied = fixture.module.get_operation().disp(&ctx).to_string();
    copy_all_control_deps_from(&ctx, destination, source).unwrap();
    copy_all_control_deps_from(&ctx, source, source).unwrap();
    assert_eq!(
        fixture.module.get_operation().disp(&ctx).to_string(),
        copied
    );
    drop_all_control_deps(&ctx, source).unwrap();
    assert!(predecessors(&ctx, source).is_empty());
    assert_eq!(predecessors(&ctx, destination), ["b", "a"]);
    assert_eq!(predecessors(&ctx, target), ["destination", "unrelated"]);
    assert_eq!(destination.deref(&ctx).get_operand(0), operand);
    assert_eq!(
        destination
            .deref(&ctx)
            .attributes
            .get::<BackendConfigPayload>(&Identifier::new("hlo_backend_config_payload")),
        Some(&payload),
    );
    assert!(
        destination
            .deref(&ctx)
            .attributes
            .0
            .get(&Identifier::new("hlo_backend_config"))
            .is_none(),
    );
    let dropped = fixture.module.get_operation().disp(&ctx).to_string();
    drop_all_control_deps(&ctx, source).unwrap();
    assert_eq!(
        fixture.module.get_operation().disp(&ctx).to_string(),
        dropped
    );
    verification::verify(&ctx, fixture.module.get_operation()).unwrap();
    let mut fresh = context();
    let parsed =
        parse_from_str(spaced(Operation::top_level_parser()), &mut fresh, &dropped).unwrap();
    verification::verify(&fresh, parsed).unwrap();
}

#[test]
fn copying_outgoing_edges_names_an_unnamed_destination_without_collisions() {
    let mut ctx = context();
    let fixture = fixture(&mut ctx);
    let destination = fixture.nodes[2];
    let source = fixture.nodes[3];
    let target = fixture.nodes[4];
    destination
        .deref_mut(&ctx)
        .attributes
        .0
        .remove(&Identifier::new("hlo_name"));
    attr(
        &ctx,
        fixture.nodes[0],
        "hlo_name",
        InstructionName("control_relay.0".into()),
    );
    set_predecessors(&ctx, source, &["control_relay.0"]);
    set_predecessors(&ctx, target, &["source"]);
    verification::verify(&ctx, fixture.module.get_operation()).unwrap();
    copy_all_control_deps_from(&ctx, destination, source).unwrap();
    assert_eq!(
        destination
            .deref(&ctx)
            .attributes
            .get::<InstructionName>(&Identifier::new("hlo_name"))
            .unwrap()
            .0,
        "control_relay.1",
    );
    assert_eq!(predecessors(&ctx, destination), ["b", "control_relay.0"]);
    assert_eq!(predecessors(&ctx, target), ["source", "control_relay.1"]);
    drop_all_control_deps(&ctx, source).unwrap();
    verification::verify(&ctx, fixture.module.get_operation()).unwrap();
}

#[test]
fn generated_control_names_avoid_other_computations_and_inline_parameters() {
    let mut ctx = context();
    let fixture = fixture(&mut ctx);
    let destination = fixture.nodes[2];
    let source = fixture.nodes[3];
    let target = fixture.nodes[4];
    destination
        .deref_mut(&ctx)
        .attributes
        .0
        .remove(&Identifier::new("hlo_name"));
    set_predecessors(&ctx, target, &["source"]);

    let ty = shape(&ctx, ElementType::F32, &[]);
    let (other, block, args) = computation(&mut ctx, "other", vec![ty], ty);
    let negate = NegateOp::new(&mut ctx, ty, args);
    attr(
        &ctx,
        negate.get_operation(),
        "hlo_name",
        InstructionName("control_relay.0".into()),
    );
    negate.get_operation().insert_at_back(block, &ctx);
    let value = negate.result(&ctx);
    let fusion = FusionOp::new(&mut ctx, ty, vec![value], FusionKind::Loop);
    fusion.get_operation().insert_at_back(block, &ctx);
    let body = BasicBlock::new(&mut ctx, None, vec![ty]);
    body.insert_at_back(fusion.body(&ctx), &ctx);
    xlahlo::ir::set_fusion_parameters(
        &ctx,
        body,
        xlahlo::ir::FusionParameters(vec![xlahlo::ir::FusionParameter {
            name: Some("control_relay.1".into()),
            ..Default::default()
        }]),
    )
    .unwrap();
    let value = body.deref(&ctx).get_argument(0);
    ret(&mut ctx, body, value);
    let value = fusion.result(&ctx);
    ret(&mut ctx, block, value);
    let module_block = fixture
        .module
        .get_operation()
        .deref(&ctx)
        .get_region(0)
        .deref(&ctx)
        .get_entry_block()
        .unwrap();
    other.get_operation().insert_at_back(module_block, &ctx);
    verification::verify(&ctx, fixture.module.get_operation()).unwrap();
    let name_key = Identifier::new("hlo_name");
    let saved_name = negate
        .get_operation()
        .deref(&ctx)
        .attributes
        .0
        .get(&name_key)
        .unwrap()
        .clone();
    attr(
        &ctx,
        negate.get_operation(),
        "hlo_name",
        StringAttr::new("wrong".into()),
    );
    let before = fixture.module.get_operation().disp(&ctx).to_string();
    for _ in 0..2 {
        assert!(copy_all_control_deps_from(&ctx, destination, source).is_err());
        assert_eq!(
            fixture.module.get_operation().disp(&ctx).to_string(),
            before
        );
    }
    negate
        .get_operation()
        .deref_mut(&ctx)
        .attributes
        .0
        .insert(name_key, saved_name);
    let parameter_key = Identifier::new("hlo_fusion_parameters");
    let saved_parameters = body
        .deref(&ctx)
        .attributes
        .0
        .get(&parameter_key)
        .unwrap()
        .clone();
    body.deref_mut(&ctx)
        .attributes
        .set(parameter_key.clone(), StringAttr::new("wrong".into()));
    let before = fixture.module.get_operation().disp(&ctx).to_string();
    for _ in 0..2 {
        assert!(copy_all_control_deps_from(&ctx, destination, source).is_err());
        assert_eq!(
            fixture.module.get_operation().disp(&ctx).to_string(),
            before
        );
    }
    body.deref_mut(&ctx)
        .attributes
        .0
        .insert(parameter_key, saved_parameters);
    copy_all_control_deps_from(&ctx, destination, source).unwrap();
    assert_eq!(
        destination
            .deref(&ctx)
            .attributes
            .get::<InstructionName>(&Identifier::new("hlo_name"))
            .unwrap()
            .0,
        "control_relay.2",
    );
    assert_eq!(predecessors(&ctx, target), ["source", "control_relay.2"]);
    verification::verify(&ctx, fixture.module.get_operation()).unwrap();
    let text = fixture.module.get_operation().disp(&ctx).to_string();
    copy_all_control_deps_from(&ctx, destination, source).unwrap();
    assert_eq!(fixture.module.get_operation().disp(&ctx).to_string(), text);
    let mut fresh = context();
    let parsed = parse_from_str(spaced(Operation::top_level_parser()), &mut fresh, &text).unwrap();
    verification::verify(&fresh, parsed).unwrap();
    Operation::erase(parsed, &mut fresh);
    Operation::erase(fixture.module.get_operation(), &mut ctx);
    assert!(fresh.is_ir_empty());
    assert!(ctx.is_ir_empty());
}

#[test]
fn control_edits_invalidate_logical_order_only_after_successful_changes() {
    use xlahlo::ir::instruction_order::{InstructionOrder, OrderPermutation, set_order};

    let mut ctx = context();
    let fixture = fixture(&mut ctx);
    let destination = fixture.nodes[2];
    let source = fixture.nodes[3];
    let target = fixture.nodes[4];
    let block = source.deref(&ctx).get_parent_block().unwrap();
    let owner = block.deref(&ctx).get_parent_op(&ctx).unwrap();
    let parameter = block.deref(&ctx).iter(&ctx).next().unwrap();
    attr(
        &ctx,
        parameter,
        "hlo_name",
        InstructionName("argument".into()),
    );
    let key = Identifier::new("hlo_instruction_order");
    let order = InstructionOrder {
        control_successors: vec![OrderPermutation {
            subject: "source".into(),
            members: vec!["target".into()],
        }],
        ..Default::default()
    };
    set_order(&ctx, owner, order.clone()).unwrap();
    verification::verify(&ctx, fixture.module.get_operation()).unwrap();
    let before = fixture.module.get_operation().disp(&ctx).to_string();
    copy_all_control_deps_from(&ctx, source, source).unwrap();
    assert_eq!(
        fixture.module.get_operation().disp(&ctx).to_string(),
        before
    );

    set_predecessors(&ctx, target, &["missing"]);
    let malformed = fixture.module.get_operation().disp(&ctx).to_string();
    assert!(copy_all_control_deps_from(&ctx, destination, source).is_err());
    assert!(drop_all_control_deps(&ctx, source).is_err());
    assert_eq!(
        fixture.module.get_operation().disp(&ctx).to_string(),
        malformed
    );
    assert_eq!(
        owner.deref(&ctx).attributes.get::<InstructionOrder>(&key),
        Some(&order)
    );
    set_predecessors(&ctx, target, &["source", "destination", "unrelated"]);

    copy_all_control_deps_from(&ctx, destination, source).unwrap();
    assert!(!owner.deref(&ctx).attributes.0.contains_key(&key));
    set_order(&ctx, owner, order.clone()).unwrap();
    let copied = fixture.module.get_operation().disp(&ctx).to_string();
    copy_all_control_deps_from(&ctx, destination, source).unwrap();
    assert_eq!(
        fixture.module.get_operation().disp(&ctx).to_string(),
        copied
    );
    drop_all_control_deps(&ctx, source).unwrap();
    assert!(!owner.deref(&ctx).attributes.0.contains_key(&key));

    set_order(&ctx, owner, InstructionOrder::default()).unwrap();
    let dropped = fixture.module.get_operation().disp(&ctx).to_string();
    drop_all_control_deps(&ctx, source).unwrap();
    assert_eq!(
        fixture.module.get_operation().disp(&ctx).to_string(),
        dropped
    );
    verification::verify(&ctx, fixture.module.get_operation()).unwrap();
    let mut fresh = context();
    let parsed =
        parse_from_str(spaced(Operation::top_level_parser()), &mut fresh, &dropped).unwrap();
    verification::verify(&fresh, parsed).unwrap();
    Operation::erase(parsed, &mut fresh);
    Operation::erase(fixture.module.get_operation(), &mut ctx);
    assert!(fresh.is_ir_empty());
    assert!(ctx.is_ir_empty());
}

#[test]
fn malformed_graphs_and_cross_parent_endpoints_fail_before_any_mutation() {
    let mut ctx = context();
    let fixture = fixture(&mut ctx);
    let destination = fixture.nodes[2];
    let source = fixture.nodes[3];
    for key in ["hlo_name", "hlo_control_predecessors"] {
        let saved = fixture.nodes[4]
            .deref(&ctx)
            .attributes
            .0
            .get(&Identifier::new(key))
            .unwrap()
            .clone();
        attr(&ctx, fixture.nodes[4], key, StringAttr::new("wrong".into()));
        let before = fixture.module.get_operation().disp(&ctx).to_string();
        for _ in 0..2 {
            assert!(copy_all_control_deps_from(&ctx, destination, source).is_err());
            assert!(drop_all_control_deps(&ctx, source).is_err());
            assert_eq!(
                fixture.module.get_operation().disp(&ctx).to_string(),
                before
            );
        }
        fixture.nodes[4]
            .deref_mut(&ctx)
            .attributes
            .0
            .insert(Identifier::new(key), saved);
    }
    set_predecessors(&ctx, fixture.nodes[4], &["missing"]);
    let before = fixture.module.get_operation().disp(&ctx).to_string();
    assert!(copy_all_control_deps_from(&ctx, destination, source).is_err());
    assert!(drop_all_control_deps(&ctx, source).is_err());
    assert_eq!(
        fixture.module.get_operation().disp(&ctx).to_string(),
        before
    );
    set_predecessors(&ctx, fixture.nodes[4], &["source"]);
    let ty = shape(&ctx, ElementType::F32, &[]);
    let (_, _, arguments) = computation(&mut ctx, "other", vec![ty], ty);
    let DefiningEntity::Op(other) = arguments[0].defining_entity() else {
        panic!("parameter is an instruction");
    };
    let before = fixture.module.get_operation().disp(&ctx).to_string();
    assert!(copy_all_control_deps_from(&ctx, other, source).is_err());
    assert_eq!(
        fixture.module.get_operation().disp(&ctx).to_string(),
        before
    );
    copy_all_control_deps_from(&ctx, destination, source).unwrap();
    drop_all_control_deps(&ctx, source).unwrap();
    verification::verify(&ctx, fixture.module.get_operation()).unwrap();
}

#[test]
fn low_level_copy_leaves_cycle_policy_to_the_module_verifier() {
    let mut ctx = context();
    let fixture = fixture(&mut ctx);
    let destination = fixture.nodes[2];
    let source = fixture.nodes[3];
    set_predecessors(&ctx, source, &["destination"]);
    verification::verify(&ctx, fixture.module.get_operation()).unwrap();
    copy_all_control_deps_from(&ctx, destination, source).unwrap();
    assert_eq!(predecessors(&ctx, destination), ["b", "destination"]);
    assert!(verification::verify(&ctx, fixture.module.get_operation()).is_err());
    drop_all_control_deps(&ctx, destination).unwrap();
    verification::verify(&ctx, fixture.module.get_operation()).unwrap();
}

#[test]
fn incoming_only_copy_and_empty_drop_do_not_invent_instruction_names_or_edges() {
    let mut ctx = context();
    let fixture = fixture(&mut ctx);
    let destination = fixture.nodes[2];
    let source = fixture.nodes[3];
    destination
        .deref_mut(&ctx)
        .attributes
        .0
        .remove(&Identifier::new("hlo_name"));
    set_predecessors(&ctx, fixture.nodes[4], &["unrelated"]);
    copy_all_control_deps_from(&ctx, destination, source).unwrap();
    assert_eq!(predecessors(&ctx, destination), ["b", "a"]);
    assert!(
        !destination
            .deref(&ctx)
            .attributes
            .0
            .contains_key(&Identifier::new("hlo_name")),
    );
    let DefiningEntity::Op(empty) = destination.deref(&ctx).get_operand(0).defining_entity() else {
        panic!("parameter is an instruction");
    };
    let before = fixture.module.get_operation().disp(&ctx).to_string();
    copy_all_control_deps_from(&ctx, destination, empty).unwrap();
    assert_eq!(
        fixture.module.get_operation().disp(&ctx).to_string(),
        before
    );
    drop_all_control_deps(&ctx, empty).unwrap();
    assert_eq!(
        fixture.module.get_operation().disp(&ctx).to_string(),
        before
    );
    drop_all_control_deps(&ctx, fixture.nodes[5]).unwrap();
    assert!(!predecessors(&ctx, fixture.nodes[4]).contains(&"unrelated".into()));
    assert!(
        !empty
            .deref(&ctx)
            .attributes
            .0
            .contains_key(&Identifier::new("hlo_control_predecessors")),
    );
    verification::verify(&ctx, fixture.module.get_operation()).unwrap();
}
