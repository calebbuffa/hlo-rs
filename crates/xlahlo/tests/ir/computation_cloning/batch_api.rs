use crate::support::{computation, context, module, ret};
use std::collections::HashSet;
use xlahlo::{
    attributes::{Callees, Index},
    ir::{
        ComputationClone, ComputationCloneRequest, ComputationReplacements, InstructionName,
        InstructionSequence, Schedule, clone_computations_with_replacements,
    },
    ops::{
        ComputationOp, ReturnOp,
        control::{AsyncDoneOp, AsyncStartOp, ParameterOp},
        elementwise::AddOp,
    },
    pliron::{
        basic_block::BasicBlock,
        context::{Context, Ptr},
        identifier::Identifier,
        linked_list::ContainsLinkedList,
        op::Op,
        operation::Operation,
        printable::Printable,
        r#type::TypeHandle,
    },
    shape,
    types::ElementType,
    verification::{VerifyOptions, verify},
};

fn name(ctx: &Context, operation: Ptr<Operation>, text: &str) {
    operation
        .deref_mut(ctx)
        .attributes
        .set(Identifier::new("hlo_name"), InstructionName(text.into()));
}

fn identity(
    ctx: &mut Context,
    container: Ptr<BasicBlock>,
    symbol: &'static str,
    scalar: TypeHandle,
) -> (ComputationOp, Ptr<Operation>) {
    let (source, block, inputs) = computation(ctx, container, symbol, vec![scalar], scalar);
    let add = AddOp::new(ctx, scalar, vec![inputs[0], inputs[0]]);
    add.get_operation().insert_at_back(block, ctx);
    ret(ctx, block, add.result(ctx));
    (source, inputs[0].defining_op().unwrap())
}

fn member_names(ctx: &Context, outputs: &[ComputationClone]) -> Vec<String> {
    let mut names = Vec::new();
    for output in outputs {
        for computation in std::iter::once(&output.computation).chain(&output.embedded_computations)
        {
            for block in computation.body(ctx).deref(ctx).iter(ctx) {
                for operation in block.deref(ctx).iter(ctx) {
                    if Operation::is_op::<ReturnOp>(operation, ctx) {
                        continue;
                    }
                    names.push(
                        operation
                            .deref(ctx)
                            .attributes
                            .get::<InstructionName>(&Identifier::new("hlo_name"))
                            .unwrap()
                            .0
                            .clone(),
                    );
                }
            }
        }
    }
    names
}

#[test]
fn unnamed_batch_reserves_names_and_preserves_sources_and_templates() {
    for verification in [None, Some(VerifyOptions::default())] {
        let ctx = &mut context();
        let (module, container) = module(ctx);
        let scalar = shape::scalar(ctx, ElementType::F32).unwrap();
        let (main, main_parameter) = identity(ctx, container, "main", scalar);
        let (second, _) = identity(ctx, container, "second", scalar);
        let before = module.disp(ctx).to_string();
        let template = ParameterOp::new(ctx, scalar, vec![], Index(0));
        let template_before = template.disp(ctx).to_string();
        let mut first_spec = ComputationReplacements::default();
        first_spec
            .instructions
            .insert(main_parameter, Some(template.get_operation()));
        let second_spec = ComputationReplacements::default();
        let outputs = clone_computations_with_replacements(
            ctx,
            &[
                ComputationCloneRequest {
                    source: &main,
                    name: "copy".into(),
                    replacements: &first_spec,
                },
                ComputationCloneRequest {
                    source: &second,
                    name: "copy".into(),
                    replacements: &second_spec,
                },
            ],
            verification,
        )
        .unwrap();
        assert_eq!(outputs.len(), 2);
        assert_ne!(
            outputs[0]
                .computation
                .get_attr_hlo_computation_name(ctx)
                .unwrap()
                .as_ref(),
            outputs[1]
                .computation
                .get_attr_hlo_computation_name(ctx)
                .unwrap()
                .as_ref()
        );
        let names = member_names(ctx, &outputs);
        assert_eq!(names.len(), 4);
        assert_eq!(names.iter().collect::<HashSet<_>>().len(), names.len());
        assert!(
            outputs[0]
                .mapping
                .lookup_op(template.get_operation())
                .is_some()
        );
        assert!(outputs[0].mapping.lookup_op(main_parameter).is_none());
        assert_eq!(module.disp(ctx).to_string(), before);
        assert_eq!(template.disp(ctx).to_string(), template_before);
        for output in &outputs {
            assert!(
                output
                    .computation
                    .get_operation()
                    .deref(ctx)
                    .get_parent_block()
                    .is_none()
            );
        }
        assert!(ComputationClone::dispose_all(outputs, ctx).is_ok());
        Operation::erase(template.get_operation(), ctx);
        Operation::erase(module.get_operation(), ctx);
        assert!(ctx.is_ir_empty());
    }
}

#[test]
fn partial_attachment_disposal_returns_all_owners_without_erasure() {
    let ctx = &mut context();
    let (module, container) = module(ctx);
    let scalar = shape::scalar(ctx, ElementType::F32).unwrap();
    let (main, _) = identity(ctx, container, "main", scalar);
    let (second, _) = identity(ctx, container, "second", scalar);
    let spec = ComputationReplacements::default();
    let outputs = clone_computations_with_replacements(
        ctx,
        &[
            ComputationCloneRequest {
                source: &main,
                name: "first_copy".into(),
                replacements: &spec,
            },
            ComputationCloneRequest {
                source: &second,
                name: "second_copy".into(),
                replacements: &spec,
            },
        ],
        None,
    )
    .unwrap();
    let attached = outputs[0].computation.get_operation();
    let detached = outputs[1].computation.get_operation();
    attached.insert_at_back(container, ctx);
    let before = module.disp(ctx).to_string();
    let detached_before = detached.disp(ctx).to_string();
    let (outputs, _) = ComputationClone::dispose_all(outputs, ctx).err().unwrap();
    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].computation.get_operation(), attached);
    assert_eq!(outputs[1].computation.get_operation(), detached);
    assert_eq!(module.disp(ctx).to_string(), before);
    assert_eq!(detached.disp(ctx).to_string(), detached_before);
    assert!(detached.deref(ctx).get_parent_block().is_none());
    attached.unlink(ctx);
    assert!(ComputationClone::dispose_all(outputs, ctx).is_ok());
    Operation::erase(module.get_operation(), ctx);
    assert!(ctx.is_ir_empty());
}

#[test]
fn late_clone_failure_disposes_prior_allocations_and_preserves_operand_uses() {
    let ctx = &mut context();
    let (module, container) = module(ctx);
    let scalar = shape::scalar(ctx, ElementType::F32).unwrap();
    let (main, parameter) = identity(ctx, container, "main", scalar);
    let (second, second_parameter) = identity(ctx, container, "second", scalar);
    let spec = ComputationReplacements::default();
    let mut invalid = ComputationReplacements::default();
    invalid.instructions.insert(second_parameter, None);
    let before = module.disp(ctx).to_string();
    let uses = parameter.deref(ctx).get_result(0).uses(ctx).len();
    assert!(
        clone_computations_with_replacements(
            ctx,
            &[
                ComputationCloneRequest {
                    source: &main,
                    name: "valid_copy".into(),
                    replacements: &spec
                },
                ComputationCloneRequest {
                    source: &second,
                    name: "invalid_copy".into(),
                    replacements: &invalid
                },
            ],
            None
        )
        .is_err()
    );
    assert_eq!(module.disp(ctx).to_string(), before);
    assert_eq!(parameter.deref(ctx).get_result(0).uses(ctx).len(), uses);
    Operation::erase(module.get_operation(), ctx);
    assert!(ctx.is_ir_empty(), "failed batch leaked detached IR");
}

#[test]
fn duplicate_sources_cross_module_sources_and_callbacks_fail_before_publication() {
    let ctx = &mut context();
    let (first_module, container) = module(ctx);
    let scalar = shape::scalar(ctx, ElementType::F32).unwrap();
    let (first, _) = identity(ctx, container, "main", scalar);
    let (second_module, container) = module(ctx);
    let (second, _) = identity(ctx, container, "main", scalar);
    let spec = ComputationReplacements::default();
    for second in [&first, &second] {
        assert!(
            clone_computations_with_replacements(
                ctx,
                &[
                    ComputationCloneRequest {
                        source: &first,
                        name: "copy".into(),
                        replacements: &spec
                    },
                    ComputationCloneRequest {
                        source: second,
                        name: "other".into(),
                        replacements: &spec
                    },
                ],
                None
            )
            .is_err()
        );
    }
    fn callback(_: &Context, _: Ptr<Operation>) -> bool {
        panic!("must not invoke callback")
    }
    let policy = VerifyOptions {
        instruction_can_change_layout: Some(callback),
        ..Default::default()
    };
    let failure = clone_computations_with_replacements(ctx, &[], Some(policy))
        .err()
        .unwrap();
    assert!(
        failure
            .to_string()
            .contains("instruction_can_change_layout")
    );
    assert!(
        clone_computations_with_replacements(ctx, &[], None)
            .unwrap()
            .is_empty()
    );
    Operation::erase(first_module.get_operation(), ctx);
    Operation::erase(second_module.get_operation(), ctx);
    assert!(ctx.is_ir_empty());
}

fn async_source(
    ctx: &mut Context,
    container: Ptr<BasicBlock>,
    symbol: &'static str,
    scalar: TypeHandle,
    asynchronous: TypeHandle,
) -> (ComputationOp, Ptr<Operation>) {
    let (source, block, inputs) = computation(ctx, container, symbol, vec![scalar], scalar);
    name(
        ctx,
        inputs[0].defining_op().unwrap(),
        &format!("{symbol}_p"),
    );
    let start = AsyncStartOp::new(
        ctx,
        asynchronous,
        vec![inputs[0]],
        Callees(vec!["work".into()]),
    );
    name(ctx, start.get_operation(), &format!("{symbol}_start"));
    start.get_operation().insert_at_back(block, ctx);
    let done = AsyncDoneOp::new(ctx, scalar, vec![start.result(ctx)]);
    name(ctx, done.get_operation(), &format!("{symbol}_done"));
    done.get_operation().insert_at_back(block, ctx);
    ret(ctx, block, done.result(ctx));
    (source, start.get_operation())
}

#[test]
fn async_dependencies_and_schedules_are_shared_collectively_without_publication() {
    let ctx = &mut context();
    let (module, container) = module(ctx);
    let scalar = shape::scalar(ctx, ElementType::F32).unwrap();
    let (work, block, inputs) = computation(ctx, container, "work", vec![scalar], scalar);
    name(ctx, inputs[0].defining_op().unwrap(), "work_p");
    ret(ctx, block, inputs[0]);
    let state = shape::scalar(ctx, ElementType::S32).unwrap();
    let operands = shape::tuple(ctx, vec![scalar]).unwrap();
    let asynchronous = shape::tuple(ctx, vec![operands, scalar, state]).unwrap();
    let (main, main_start) = async_source(ctx, container, "main", scalar, asynchronous);
    let (other, other_start) = async_source(ctx, container, "other", scalar, asynchronous);
    let original = Schedule(vec![
        InstructionSequence {
            computation: "work".into(),
            instructions: vec!["work_p".into()],
        },
        InstructionSequence {
            computation: "main".into(),
            instructions: vec!["main_p".into(), "main_start".into(), "main_done".into()],
        },
        InstructionSequence {
            computation: "other".into(),
            instructions: vec!["other_p".into(), "other_start".into(), "other_done".into()],
        },
    ]);
    module
        .get_operation()
        .deref_mut(ctx)
        .attributes
        .set(Identifier::new("hlo_schedule"), original.clone());
    verify(ctx, module.get_operation()).unwrap();
    let before = module.disp(ctx).to_string();
    let spec = ComputationReplacements::default();
    let outputs = clone_computations_with_replacements(
        ctx,
        &[
            ComputationCloneRequest {
                source: &main,
                name: "main_copy".into(),
                replacements: &spec,
            },
            ComputationCloneRequest {
                source: &other,
                name: "other_copy".into(),
                replacements: &spec,
            },
        ],
        Some(VerifyOptions::default()),
    )
    .unwrap();
    assert_eq!(outputs[0].embedded_computations.len(), 1);
    assert!(outputs[1].embedded_computations.is_empty());
    assert_eq!(outputs[0].embedded_schedules.len(), 1);
    assert!(outputs[1].embedded_schedules.is_empty());
    let wrapped = outputs[0].embedded_computations[0];
    assert_ne!(wrapped.get_operation(), work.get_operation());
    let symbol = wrapped.get_attr_hlo_computation_name(ctx).unwrap().clone();
    assert_eq!(outputs[0].embedded_schedules[0].computation, symbol);
    for (output, source_start) in outputs.iter().zip([main_start, other_start]) {
        let start = output.mapping.lookup_op(source_start).unwrap();
        let callees = start
            .deref(ctx)
            .attributes
            .get::<Callees>(&Identifier::new("hlo_callees"))
            .unwrap()
            .clone();
        assert_eq!(callees.0, vec![symbol.clone()]);
    }
    let names = member_names(ctx, &outputs);
    assert_eq!(names.len(), names.iter().collect::<HashSet<_>>().len());
    assert_eq!(module.disp(ctx).to_string(), before);
    // A partially attached embedded dependency must preserve every batch owner.
    wrapped.get_operation().insert_at_back(container, ctx);
    let (outputs, _) = ComputationClone::dispose_all(outputs, ctx).err().unwrap();
    assert_eq!(outputs.len(), 2);
    assert_eq!(
        outputs[0].embedded_computations[0].get_operation(),
        wrapped.get_operation()
    );
    let mut published_schedule = original.clone();
    for output in &outputs {
        output
            .computation
            .get_operation()
            .insert_at_back(container, ctx);
        let mut instructions = Vec::new();
        for block in output.computation.body(ctx).deref(ctx).iter(ctx) {
            for operation in block.deref(ctx).iter(ctx) {
                if !Operation::is_op::<ReturnOp>(operation, ctx) {
                    instructions.push(
                        operation
                            .deref(ctx)
                            .attributes
                            .get::<InstructionName>(&Identifier::new("hlo_name"))
                            .unwrap()
                            .0
                            .clone(),
                    );
                }
            }
        }
        published_schedule.0.push(InstructionSequence {
            computation: output
                .computation
                .get_attr_hlo_computation_name(ctx)
                .unwrap()
                .clone(),
            instructions,
        });
        published_schedule
            .0
            .extend(output.embedded_schedules.iter().cloned());
    }
    module
        .get_operation()
        .deref_mut(ctx)
        .attributes
        .set(Identifier::new("hlo_schedule"), published_schedule);
    verify(ctx, module.get_operation()).unwrap();
    for output in &outputs {
        output.computation.get_operation().unlink(ctx);
        for embedded in &output.embedded_computations {
            embedded.get_operation().unlink(ctx);
        }
    }
    module
        .get_operation()
        .deref_mut(ctx)
        .attributes
        .set(Identifier::new("hlo_schedule"), original);
    assert!(ComputationClone::dispose_all(outputs, ctx).is_ok());
    assert_eq!(module.disp(ctx).to_string(), before);
    Operation::erase(module.get_operation(), ctx);
    assert!(ctx.is_ir_empty());
}

#[test]
fn failure_after_async_dependency_allocation_erases_the_entire_owned_graph() {
    let ctx = &mut context();
    let (module, container) = module(ctx);
    let scalar = shape::scalar(ctx, ElementType::F32).unwrap();
    let (_, block, inputs) = computation(ctx, container, "work", vec![scalar], scalar);
    ret(ctx, block, inputs[0]);
    let state = shape::scalar(ctx, ElementType::S32).unwrap();
    let operands = shape::tuple(ctx, vec![scalar]).unwrap();
    let asynchronous = shape::tuple(ctx, vec![operands, scalar, state]).unwrap();
    let (main, _) = async_source(ctx, container, "main", scalar, asynchronous);
    let (other, parameter) = identity(ctx, container, "other", scalar);
    let spec = ComputationReplacements::default();
    let mut invalid = ComputationReplacements::default();
    invalid.instructions.insert(parameter, None);
    let before = module.disp(ctx).to_string();
    assert!(
        clone_computations_with_replacements(
            ctx,
            &[
                ComputationCloneRequest {
                    source: &main,
                    name: "async_copy".into(),
                    replacements: &spec
                },
                ComputationCloneRequest {
                    source: &other,
                    name: "invalid_copy".into(),
                    replacements: &invalid
                },
            ],
            None
        )
        .is_err()
    );
    assert_eq!(module.disp(ctx).to_string(), before);
    Operation::erase(module.get_operation(), ctx);
    assert!(ctx.is_ir_empty());
}
