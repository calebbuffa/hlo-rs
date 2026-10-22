//! Private implementation of the public detached computation-cloning API.
//!
//! Based on OpenXLA's computation cloning in `hlo_computation.cc`,
//! `hlo_computation.h`, `hlo_instruction.cc` and `hlo_instructions.cc`.
//! Replacement lookup occurs once at each operand edge, not recursively.
//! An omitted instruction must have no surviving data use. External operand
//! dependencies are copied, never captured through same-context clone fallback.
//!
//! A replacement's attributes, including OriginalValue, belong to that
//! replacement, not the replaced instruction. shape-index rebasing is the
//! caller's responsibility (as in RemoveDeadTupleIndices). The original module,
//! templates, module schedule and configuration are never edited. The caller
//! owns the returned detached computation and must verify its eventual complete
//! module with its original VerifyOptions after publishing/remapping callsites.
//! Passing no VerifyOptions is a raw structural clone, not implicit verification;
//! upstream can clone intermediate computations that are not yet semantically valid.
//! OriginalValue is copied without validating its tree against the shape, matching
//! hlo_original_value_util.cc:30-58; explicit verification checks that contract.
//!
//! The public facade owns disposal and rejects unsupported verifier callbacks.
//! Scheduled/context-sharing companions remain test-only until a consumer needs
//! them; this is not a while-state transformation or caller publication API.
use super::{
    ControlPredecessors, FusionParameters, InstructionName, InstructionSequence, Schedule,
    instruction_order::{self, InstructionOrder, OrderPermutation},
};
use crate::{
    attributes::{Callees, Index, ResultAccuracy},
    ir::OriginalValue,
    opcode::Opcode,
    ops::{ComputationOp, HloOpInterface, ReturnOp, control::TupleOp},
    types::TupleType,
    verification::{VerifyOptions, error, optional_attr, require},
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{attributes::TypeAttr, ops::ModuleOp, types::FunctionType},
    common_traits::Verify,
    context::{Context, Ptr},
    identifier::Identifier,
    irbuild::{
        cloning::{IrMapping, clone_operation},
        listener::DummyListener,
        rewriter::IRRewriter,
    },
    linked_list::ContainsLinkedList,
    location::Located,
    op::{Op, op_cast},
    operation::Operation,
    result::Result,
    r#type::TypeHandle,
    value::{DefiningEntity, Value},
};
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Clone, Debug, Default)]
pub(crate) enum CloneRoot {
    #[default]
    Keep,
    Instruction(Ptr<Operation>),
    Tuple(Vec<Ptr<Operation>>),
}

/// Detached templates have no native name scope. Explicit edges supply the
/// pointer identity that upstream stores on HloInstruction, including outgoing
/// template edges; existing attached control edges use their actual block scope.
#[derive(Default)]
pub(crate) struct Replacements {
    pub instructions: HashMap<Ptr<Operation>, Option<Ptr<Operation>>>,
    pub extra_parameters: Vec<Ptr<Operation>>,
    pub template_control_edges: Vec<(Ptr<Operation>, Ptr<Operation>)>,
    pub root: CloneRoot,
}

pub(crate) struct ComputationClone {
    pub computation: ComputationOp,
    /// AsyncStart owns a new wrapped computation even in the same module.
    /// Publish these alongside `computation`; all remain detached on success.
    pub embedded_computations: Vec<ComputationOp>,
    /// Pinned AsyncStart cloning gives a scheduled wrapped clone a trivial
    /// postorder schedule. Apply these only when publishing the embedded clones.
    pub embedded_schedules: Vec<InstructionSequence>,
    /// Pinned HloCloneContext maps selected templates, not replaced originals.
    pub mapping: IrMapping,
}

/// Cached native handles must outlive this context. Mappings and reserved names
/// are committed only on success; the caller still owns every returned clone.
#[derive(Clone)]
pub(crate) struct ComputationCloneContext {
    pub(in crate::ir::clone_with_replacements) destination: Option<Ptr<Operation>>,
    pub(in crate::ir::clone_with_replacements) suffix: String,
    pub(in crate::ir::clone_with_replacements) computations:
        HashMap<Ptr<Operation>, Ptr<Operation>>,
    pub(in crate::ir::clone_with_replacements) instructions:
        HashMap<Ptr<Operation>, Ptr<Operation>>,
    pub(in crate::ir::clone_with_replacements) computations_by_name:
        HashMap<String, Ptr<Operation>>,
    pub(in crate::ir::clone_with_replacements) instructions_by_name:
        HashMap<String, Ptr<Operation>>,
    pub(in crate::ir::clone_with_replacements) instruction_names: HashSet<String>,
    pub(in crate::ir::clone_with_replacements) computation_names: HashSet<String>,
    pub(in crate::ir::clone_with_replacements) schedules: Vec<InstructionSequence>,
    pub(in crate::ir::clone_with_replacements) in_progress: HashSet<Ptr<Operation>>,
}

impl ComputationCloneContext {
    pub(crate) fn new(destination: Option<Ptr<Operation>>, suffix: String) -> Self {
        Self {
            destination,
            suffix,
            computations: HashMap::new(),
            instructions: HashMap::new(),
            computations_by_name: HashMap::new(),
            instructions_by_name: HashMap::new(),
            instruction_names: HashSet::new(),
            computation_names: HashSet::new(),
            schedules: Vec::new(),
            in_progress: HashSet::new(),
        }
    }

    pub(crate) fn computation(&self, source: Ptr<Operation>) -> Option<Ptr<Operation>> {
        self.computations.get(&source).copied()
    }

    #[cfg(test)]
    pub(crate) fn instruction(&self, source: Ptr<Operation>) -> Option<Ptr<Operation>> {
        self.instructions.get(&source).copied()
    }

    #[cfg(test)]
    pub(crate) fn computation_by_name(&self, name: &str) -> Option<Ptr<Operation>> {
        self.computations_by_name.get(name).copied()
    }

    #[cfg(test)]
    pub(crate) fn instruction_by_name(&self, name: &str) -> Option<Ptr<Operation>> {
        self.instructions_by_name.get(name).copied()
    }
}

pub(super) fn owning_module(
    ctx: &Context,
    mut operation: Ptr<Operation>,
) -> Option<Ptr<Operation>> {
    loop {
        if Operation::is_op::<ModuleOp>(operation, ctx) {
            return Some(operation);
        }
        operation = operation
            .deref(ctx)
            .get_parent_block()?
            .deref(ctx)
            .get_parent_op(ctx)?;
    }
}

mod build;
mod cloning;
mod metadata;
mod planning;
mod schedule;
use build::*;
pub(crate) use cloning::*;
use metadata::*;
use planning::*;
use schedule::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        attributes::{Bytes, Callees, FrontendAttributes, Payload},
        ir::{BackendConfigPayload, ExecutionThread},
        ops::{
            control::{CallOp, ParameterOp},
            elementwise::{AddOp, ExpOp, NegateOp},
        },
        types::{ElementType, TensorType},
    };
    use pliron::{attribute::Attribute, printable::Printable};

    fn context() -> Context {
        let mut ctx = Context::new();
        crate::register(&mut ctx);
        instruction_order::register(&mut ctx);
        ctx
    }

    fn scalar(ctx: &Context) -> TypeHandle {
        TensorType::get(ctx, ElementType::F32, vec![]).unwrap()
    }

    fn attribute<T: Attribute>(
        ctx: &Context,
        operation: Ptr<Operation>,
        name: &'static str,
        value: T,
    ) {
        operation
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new(name), value);
    }

    fn computation(
        ctx: &mut Context,
        name: &'static str,
        parameter_count: u32,
    ) -> (ComputationOp, Ptr<BasicBlock>, Vec<Ptr<Operation>>) {
        let ty = scalar(ctx);
        let signature: TypeHandle =
            FunctionType::get(ctx, vec![ty; parameter_count as usize], vec![ty]).into();
        let computation = ComputationOp::new(ctx, Identifier::new(name), signature);
        let block = BasicBlock::new(ctx, None, vec![]);
        block.insert_at_back(computation.body(ctx), ctx);
        let parameters = (0..parameter_count)
            .map(|index| {
                let parameter = ParameterOp::new(ctx, ty, vec![], Index(index)).get_operation();
                parameter.insert_at_back(block, ctx);
                parameter
            })
            .collect();
        (computation, block, parameters)
    }

    fn returned(ctx: &mut Context, block: Ptr<BasicBlock>, root: Ptr<Operation>) {
        let value = root.deref(ctx).get_result(0);
        ReturnOp::new(ctx, value)
            .get_operation()
            .insert_at_back(block, ctx);
    }

    fn clone(
        ctx: &mut Context,
        source: &ComputationOp,
        replacements: &Replacements,
    ) -> Result<ComputationClone> {
        clone_with_replacements(
            ctx,
            source,
            "cloned".into(),
            replacements,
            Some(VerifyOptions::default()),
        )
    }

    fn assert_closed(ctx: &Context, computation: &ComputationOp) {
        let block = body(ctx, computation).unwrap();
        let instructions: HashSet<_> = block.deref(ctx).iter(ctx).collect();
        for instruction in &instructions {
            for operand in instruction.deref(ctx).operands() {
                assert!(instructions.contains(&operand_definition(ctx, operand).unwrap()));
            }
        }
    }

    mod cloning {
        use super::*;

        #[test]
        fn unchanged_clone_preserves_attributes_and_does_not_capture_source_values() {
            let mut ctx = context();
            let (source, block, parameters) = computation(&mut ctx, "source", 1);
            let ty = scalar(&ctx);
            let parameter = parameters[0].deref(&ctx).get_result(0);
            let root = NegateOp::new(&mut ctx, ty, vec![parameter]).get_operation();
            root.insert_at_back(block, &ctx);
            attribute(
                &ctx,
                source.get_operation(),
                "hlo_execution_thread",
                ExecutionThread("worker".into()),
            );
            attribute(
                &ctx,
                root,
                "hlo_frontend_attributes",
                FrontendAttributes::new([("key".into(), "value".into())]),
            );
            attribute(
                &ctx,
                root,
                "hlo_backend_config_payload",
                BackendConfigPayload(Payload::Value(Bytes(vec![1, 2, 3]))),
            );
            returned(&mut ctx, block, root);
            let before = source.get_operation().disp(&ctx).to_string();
            let output = clone(&mut ctx, &source, &Replacements::default()).unwrap();
            assert_eq!(before, source.get_operation().disp(&ctx).to_string());
            assert_closed(&ctx, &output.computation);
            let cloned_root = output.mapping.lookup_op(root).unwrap();
            let mut cloned_attributes = cloned_root.deref(&ctx).attributes.clone();
            cloned_attributes.0.remove(&Identifier::new("hlo_name"));
            assert_eq!(root.deref(&ctx).attributes, cloned_attributes);
            assert_eq!(
                optional_attr::<ExecutionThread>(
                    &ctx,
                    output.computation.get_operation(),
                    "hlo_execution_thread"
                )
                .unwrap()
                .unwrap()
                .0,
                "worker",
            );
            crate::verification::verify(&ctx, output.computation.get_operation()).unwrap();
        }

        #[test]
        fn omissions_replacements_extra_parameters_and_root_substitution() {
            let mut ctx = context();
            let (source, block, parameters) = computation(&mut ctx, "source", 2);
            returned(&mut ctx, block, parameters[0]);
            let ty = scalar(&ctx);
            let replacement = ParameterOp::new(&mut ctx, ty, vec![], Index(0)).get_operation();
            let extra = ParameterOp::new(&mut ctx, ty, vec![], Index(1)).get_operation();
            let replacements = Replacements {
                instructions: HashMap::from([
                    (parameters[0], Some(replacement)),
                    (parameters[1], None),
                ]),
                extra_parameters: vec![extra],
                root: CloneRoot::Instruction(parameters[0]),
                ..Default::default()
            };
            let mut context = ComputationCloneContext::new(None, "clone".into());
            let output = clone_in_context(
                &mut ctx,
                &source,
                "cloned".into(),
                &replacements,
                Some(VerifyOptions::default()),
                &mut context,
            )
            .unwrap();
            assert!(output.mapping.lookup_op(parameters[1]).is_none());
            assert!(output.mapping.lookup_op(parameters[0]).is_none());
            assert!(output.mapping.lookup_op(replacement).is_some());
            assert!(output.mapping.lookup_op(extra).is_some());
            assert!(context.instruction(extra).is_none());
            assert_eq!(
                context.instruction(replacement),
                output.mapping.lookup_op(replacement)
            );
            assert_closed(&ctx, &output.computation);
            crate::verification::verify(&ctx, output.computation.get_operation()).unwrap();
            Operation::erase(output.computation.get_operation(), &mut ctx);
            Operation::erase(replacement, &mut ctx);
            Operation::erase(extra, &mut ctx);
        }

        #[test]
        fn attached_extra_is_independent_of_operand_closure_and_shared_context() {
            let mut ctx = context();
            let (source, block, parameters) = computation(&mut ctx, "source", 1);
            returned(&mut ctx, block, parameters[0]);
            let (external, external_block, external_parameters) =
                computation(&mut ctx, "external", 2);
            let extra = external_parameters[1];
            returned(&mut ctx, external_block, extra);
            attribute(
                &ctx,
                external_parameters[0],
                "hlo_name",
                InstructionName("external_input".into()),
            );
            attribute(
                &ctx,
                extra,
                "hlo_control_predecessors",
                ControlPredecessors(vec!["external_input".into()]),
            );
            let source_before = source.get_operation().disp(&ctx).to_string();
            let external_before = external.get_operation().disp(&ctx).to_string();
            let uses_before = extra.deref(&ctx).get_result(0).num_uses(&ctx);
            let replacements = Replacements {
                extra_parameters: vec![extra],
                ..Default::default()
            };
            let mut shared = ComputationCloneContext::new(None, "clone".into());
            let output = clone_in_context(
                &mut ctx,
                &source,
                "cloned".into(),
                &replacements,
                Some(VerifyOptions::default()),
                &mut shared,
            )
            .unwrap();
            assert!(output.mapping.lookup_op(extra).is_some());
            assert!(shared.instruction(extra).is_none());
            assert_closed(&ctx, &output.computation);
            crate::verification::verify(&ctx, output.computation.get_operation()).unwrap();
            Operation::erase(output.computation.get_operation(), &mut ctx);
            let ty = scalar(&ctx);
            let value = extra.deref(&ctx).get_result(0);
            let template = NegateOp::new(&mut ctx, ty, vec![value]).get_operation();
            let replacements = Replacements {
                extra_parameters: vec![extra],
                instructions: HashMap::from([(parameters[0], Some(template))]),
                ..Default::default()
            };
            let mut fresh = ComputationCloneContext::new(None, "clone".into());
            // The extra and DFS dependency are separate parameter occurrences.
            // Both carry index 1, so upstream's computation builder rejects them;
            // return an error before allocation rather than deduplicating them.
            let error = clone_in_context(
                &mut ctx,
                &source,
                "invalid".into(),
                &replacements,
                None,
                &mut fresh,
            )
            .err()
            .unwrap();
            assert!(
                error
                    .to_string()
                    .contains("duplicate clone parameter index")
            );
            assert!(fresh.instructions.is_empty());
            assert!(fresh.computations.is_empty());
            assert!(fresh.instruction_names.is_empty());
            assert!(fresh.computation_names.is_empty());
            assert!(fresh.schedules.is_empty());
            Operation::erase(template, &mut ctx);
            assert_eq!(uses_before, extra.deref(&ctx).get_result(0).num_uses(&ctx));
            assert_eq!(source_before, source.get_operation().disp(&ctx).to_string());
            assert_eq!(
                external_before,
                external.get_operation().disp(&ctx).to_string()
            );
        }

        #[test]
        fn external_attached_operand_closure_is_copied_without_editing_its_owner() {
            let mut ctx = context();
            let (source, block, parameters) = computation(&mut ctx, "source", 1);
            returned(&mut ctx, block, parameters[0]);
            let (external, external_block, external_parameters) =
                computation(&mut ctx, "external", 1);
            let ty = scalar(&ctx);
            let value = external_parameters[0].deref(&ctx).get_result(0);
            let dependency = NegateOp::new(&mut ctx, ty, vec![value]).get_operation();
            dependency.insert_at_back(external_block, &ctx);
            returned(&mut ctx, external_block, dependency);
            let dependency_value = dependency.deref(&ctx).get_result(0);
            let replacement = NegateOp::new(&mut ctx, ty, vec![dependency_value]).get_operation();
            let before = external.get_operation().disp(&ctx).to_string();
            let replacements = Replacements {
                instructions: HashMap::from([(parameters[0], Some(replacement))]),
                ..Default::default()
            };
            let output = clone(&mut ctx, &source, &replacements).unwrap();
            assert_eq!(before, external.get_operation().disp(&ctx).to_string());
            assert!(output.mapping.lookup_op(dependency).is_some());
            assert!(output.mapping.lookup_op(external_parameters[0]).is_some());
            assert_closed(&ctx, &output.computation);
            let logical =
                instruction_order::instruction_order(&ctx, output.computation.get_operation())
                    .unwrap();
            // Pinned storage puts mapped replacement first and new nonparameters
            // last, despite the reverse physical SSA dependency order.
            assert_eq!(logical[0], output.mapping.lookup_op(replacement).unwrap());
            assert_eq!(
                logical.last(),
                output.mapping.lookup_op(dependency).as_ref()
            );
            crate::verification::verify(&ctx, output.computation.get_operation()).unwrap();
            Operation::erase(output.computation.get_operation(), &mut ctx);
            let external_replacement =
                ParameterOp::new(&mut ctx, ty, vec![], Index(0)).get_operation();
            let mut replacements = replacements;
            replacements
                .instructions
                .insert(external_parameters[0], Some(external_replacement));
            let output = clone(&mut ctx, &source, &replacements).unwrap();
            assert!(output.mapping.lookup_op(external_parameters[0]).is_none());
            assert!(output.mapping.lookup_op(external_replacement).is_some());
            assert_eq!(before, external.get_operation().disp(&ctx).to_string());
            assert_closed(&ctx, &output.computation);
            crate::verification::verify(&ctx, output.computation.get_operation()).unwrap();
            Operation::erase(output.computation.get_operation(), &mut ctx);
            Operation::erase(external_replacement, &mut ctx);
            Operation::erase(replacement, &mut ctx);
        }

        #[test]
        fn replacement_lookup_is_once_per_operand_edge() {
            let mut ctx = context();
            let (source, block, parameters) = computation(&mut ctx, "source", 1);
            let ty = scalar(&ctx);
            let parameter = parameters[0].deref(&ctx).get_result(0);
            let old = NegateOp::new(&mut ctx, ty, vec![parameter]).get_operation();
            old.insert_at_back(block, &ctx);
            returned(&mut ctx, block, old);
            let replacement = NegateOp::new(&mut ctx, ty, vec![parameter]).get_operation();
            let new_parameter = ParameterOp::new(&mut ctx, ty, vec![], Index(0)).get_operation();
            let replacements = Replacements {
                instructions: HashMap::from([
                    (old, Some(replacement)),
                    (parameters[0], Some(new_parameter)),
                ]),
                ..Default::default()
            };
            let output = clone(&mut ctx, &source, &replacements).unwrap();
            assert_eq!(
                output
                    .mapping
                    .lookup_op(replacement)
                    .unwrap()
                    .deref(&ctx)
                    .get_operand(0),
                output
                    .mapping
                    .lookup_op(new_parameter)
                    .unwrap()
                    .deref(&ctx)
                    .get_result(0),
            );
            assert_closed(&ctx, &output.computation);
            Operation::erase(output.computation.get_operation(), &mut ctx);
            Operation::erase(replacement, &mut ctx);
            Operation::erase(new_parameter, &mut ctx);
        }

        #[test]
        fn tuple_root_supports_empty_and_repeated_elements() {
            for elements in [vec![], vec![0], vec![1, 0, 1]] {
                let mut ctx = context();
                let (source, block, parameters) = computation(&mut ctx, "source", 2);
                returned(&mut ctx, block, parameters[0]);
                let replacements = Replacements {
                    root: CloneRoot::Tuple(elements.iter().map(|&i| parameters[i]).collect()),
                    ..Default::default()
                };
                let output = clone(&mut ctx, &source, &replacements).unwrap();
                let block = body(&ctx, &output.computation).unwrap();
                let instructions: Vec<_> = block.deref(&ctx).iter(&ctx).collect();
                let tuple = instructions[instructions.len() - 2];
                assert_eq!(opcode(&ctx, tuple).unwrap(), Opcode::Tuple);
                assert_eq!(tuple.deref(&ctx).get_num_operands(), elements.len());
                assert_closed(&ctx, &output.computation);
                crate::verification::verify(&ctx, output.computation.get_operation()).unwrap();
            }
        }

        #[test]
        fn omitted_used_operand_and_omitted_root_fail_before_source_changes() {
            for omit_root in [false, true] {
                let mut ctx = context();
                let (source, block, parameters) = computation(&mut ctx, "source", 1);
                let ty = scalar(&ctx);
                let value = parameters[0].deref(&ctx).get_result(0);
                let root = NegateOp::new(&mut ctx, ty, vec![value]).get_operation();
                root.insert_at_back(block, &ctx);
                returned(&mut ctx, block, root);
                let before = source.get_operation().disp(&ctx).to_string();
                let replacements = Replacements {
                    instructions: HashMap::from([(
                        if omit_root { root } else { parameters[0] },
                        None,
                    )]),
                    ..Default::default()
                };
                assert!(clone(&mut ctx, &source, &replacements).is_err());
                assert_eq!(before, source.get_operation().disp(&ctx).to_string());
            }
        }

        #[test]
        fn controls_follow_selected_predecessor_and_replaced_successor_identity() {
            let mut ctx = context();
            let (source, block, parameters) = computation(&mut ctx, "source", 2);
            attribute(
                &ctx,
                parameters[0],
                "hlo_name",
                InstructionName("left".into()),
            );
            attribute(
                &ctx,
                parameters[1],
                "hlo_name",
                InstructionName("right".into()),
            );
            attribute(
                &ctx,
                parameters[1],
                "hlo_control_predecessors",
                ControlPredecessors(vec!["left".into()]),
            );
            returned(&mut ctx, block, parameters[1]);
            let ty = scalar(&ctx);
            let replacement = ParameterOp::new(&mut ctx, ty, vec![], Index(1)).get_operation();
            attribute(
                &ctx,
                replacement,
                "hlo_name",
                InstructionName("right".into()),
            );
            let replacements = Replacements {
                instructions: HashMap::from([(parameters[1], Some(replacement))]),
                ..Default::default()
            };
            let output = clone(&mut ctx, &source, &replacements).unwrap();
            assert_eq!(
                optional_attr::<ControlPredecessors>(
                    &ctx,
                    output.mapping.lookup_op(replacement).unwrap(),
                    "hlo_control_predecessors"
                )
                .unwrap()
                .unwrap()
                .0,
                vec!["left"],
            );
            assert!(replacement.deref(&ctx).get_parent_block().is_none());
            Operation::erase(output.computation.get_operation(), &mut ctx);
            Operation::erase(replacement, &mut ctx);
        }

        #[test]
        fn replacing_control_source_does_not_inherit_its_outgoing_edges() {
            let mut ctx = context();
            let (source, block, parameters) = computation(&mut ctx, "source", 2);
            attribute(
                &ctx,
                parameters[0],
                "hlo_name",
                InstructionName("left".into()),
            );
            attribute(
                &ctx,
                parameters[1],
                "hlo_control_predecessors",
                ControlPredecessors(vec!["left".into()]),
            );
            returned(&mut ctx, block, parameters[1]);
            let ty = scalar(&ctx);
            let replacement = ParameterOp::new(&mut ctx, ty, vec![], Index(0)).get_operation();
            attribute(
                &ctx,
                replacement,
                "hlo_name",
                InstructionName("left".into()),
            );
            let mut replacements = Replacements {
                instructions: HashMap::from([(parameters[0], Some(replacement))]),
                ..Default::default()
            };
            for explicit_edge in [false, true] {
                if explicit_edge {
                    replacements
                        .template_control_edges
                        .push((replacement, parameters[1]));
                }
                let output = clone(&mut ctx, &source, &replacements).unwrap();
                let incoming = optional_attr::<ControlPredecessors>(
                    &ctx,
                    output.mapping.lookup_op(parameters[1]).unwrap(),
                    "hlo_control_predecessors",
                )
                .unwrap()
                .unwrap()
                .0
                .clone();
                assert_eq!(incoming.len(), usize::from(explicit_edge));
                crate::verification::verify(&ctx, output.computation.get_operation()).unwrap();
                Operation::erase(output.computation.get_operation(), &mut ctx);
            }
            Operation::erase(replacement, &mut ctx);
        }
    }

    mod metadata {
        use super::*;

        #[test]
        fn output_names_are_uniquified_but_duplicate_parameter_indices_fail() {
            for duplicate_name in [false, true] {
                let mut ctx = context();
                let (source, block, parameters) = computation(&mut ctx, "source", 2);
                returned(&mut ctx, block, parameters[0]);
                attribute(
                    &ctx,
                    parameters[0],
                    "hlo_name",
                    InstructionName("left".into()),
                );
                let ty = scalar(&ctx);
                let replacement = ParameterOp::new(
                    &mut ctx,
                    ty,
                    vec![],
                    Index(if duplicate_name { 1 } else { 0 }),
                )
                .get_operation();
                if duplicate_name {
                    attribute(
                        &ctx,
                        replacement,
                        "hlo_name",
                        InstructionName("left".into()),
                    );
                }
                let before = source.get_operation().disp(&ctx).to_string();
                let replacements = Replacements {
                    instructions: HashMap::from([(parameters[1], Some(replacement))]),
                    ..Default::default()
                };
                let result = clone(&mut ctx, &source, &replacements);
                if duplicate_name {
                    let output = result.unwrap();
                    let original = output.mapping.lookup_op(parameters[0]).unwrap();
                    let replaced = output.mapping.lookup_op(replacement).unwrap();
                    assert_ne!(
                        optional_attr::<InstructionName>(&ctx, original, "hlo_name")
                            .unwrap()
                            .unwrap()
                            .0,
                        optional_attr::<InstructionName>(&ctx, replaced, "hlo_name")
                            .unwrap()
                            .unwrap()
                            .0,
                    );
                    Operation::erase(output.computation.get_operation(), &mut ctx);
                } else {
                    assert!(result.is_err());
                }
                assert_eq!(before, source.get_operation().disp(&ctx).to_string());
                Operation::erase(replacement, &mut ctx);
            }
        }

        #[test]
        fn local_verification_error_drops_all_staging_operand_uses() {
            let mut ctx = context();
            let (source, block, parameters) = computation(&mut ctx, "source", 1);
            returned(&mut ctx, block, parameters[0]);
            let ty = scalar(&ctx);
            let replacement = AddOp::new(&mut ctx, ty, vec![]).get_operation();
            let replacements = Replacements {
                instructions: HashMap::from([(parameters[0], Some(replacement))]),
                ..Default::default()
            };
            let before = source.get_operation().disp(&ctx).to_string();
            assert!(clone(&mut ctx, &source, &replacements).is_err());
            assert_eq!(before, source.get_operation().disp(&ctx).to_string());
            // Erasing the template would assert if a failed clone retained its result.
            Operation::erase(replacement, &mut ctx);
            crate::verification::verify(&ctx, source.get_operation()).unwrap();
        }

        #[test]
        fn unary_accuracy_is_preserved_but_nonunary_accuracy_is_reset() {
            let mut ctx = context();
            let (source, block, parameters) = computation(&mut ctx, "source", 1);
            let ty = scalar(&ctx);
            let value = parameters[0].deref(&ctx).get_result(0);
            let unary = ExpOp::new(&mut ctx, ty, vec![value]).get_operation();
            unary.insert_at_back(block, &ctx);
            let unary_result = unary.deref(&ctx).get_result(0);
            let binary = AddOp::new(&mut ctx, ty, vec![value, unary_result]).get_operation();
            binary.insert_at_back(block, &ctx);
            for operation in [unary, binary] {
                attribute(
                    &ctx,
                    operation,
                    "hlo_result_accuracy",
                    ResultAccuracy::Highest,
                );
            }
            returned(&mut ctx, block, binary);
            let output = clone(&mut ctx, &source, &Replacements::default()).unwrap();
            for (operation, expected) in [
                (unary, ResultAccuracy::Highest),
                (binary, ResultAccuracy::Default),
            ] {
                assert_eq!(
                    *optional_attr::<ResultAccuracy>(
                        &ctx,
                        output.mapping.lookup_op(operation).unwrap(),
                        "hlo_result_accuracy"
                    )
                    .unwrap()
                    .unwrap(),
                    expected,
                );
            }
            crate::verification::verify(&ctx, output.computation.get_operation()).unwrap();
        }

        #[test]
        fn named_callees_are_verified_in_an_isolated_complete_module_context() {
            let mut ctx = context();
            let (callee, block, parameters) = computation(&mut ctx, "callee", 1);
            returned(&mut ctx, block, parameters[0]);
            let (source, block, parameters) = computation(&mut ctx, "source", 1);
            let ty = scalar(&ctx);
            let value = parameters[0].deref(&ctx).get_result(0);
            let call = CallOp::new(&mut ctx, ty, vec![value], Callees(vec!["callee".into()]))
                .get_operation();
            attribute(&ctx, call, "hlo_name", InstructionName("call".into()));
            call.insert_at_back(block, &ctx);
            returned(&mut ctx, block, call);
            let module = ModuleOp::new(&mut ctx, Identifier::new("test"));
            let block = module
                .get_operation()
                .deref(&ctx)
                .get_region(0)
                .deref(&ctx)
                .get_entry_block()
                .unwrap();
            callee.get_operation().insert_at_back(block, &ctx);
            source.get_operation().insert_at_back(block, &ctx);
            crate::ir::set_entry(&ctx, module.get_operation(), Identifier::new("source")).unwrap();
            crate::verification::verify(&ctx, module.get_operation()).unwrap();
            let before = module.get_operation().disp(&ctx).to_string();
            let output = clone(&mut ctx, &source, &Replacements::default()).unwrap();
            assert_eq!(before, module.get_operation().disp(&ctx).to_string());
            assert!(
                output
                    .computation
                    .get_operation()
                    .deref(&ctx)
                    .get_parent_block()
                    .is_none()
            );
            output
                .computation
                .get_operation()
                .insert_at_back(block, &ctx);
            crate::verification::verify(&ctx, module.get_operation()).unwrap();
        }

        #[test]
        fn provenance_belongs_to_selected_template_and_shape_rebasing_is_explicit() {
            use crate::ir::{OriginalArray, OriginalValueElement};
            let mut ctx = context();
            let scalar = scalar(&ctx);
            let tuple = TupleType::get(&ctx, vec![scalar, scalar]).unwrap();
            let signature = FunctionType::get(&ctx, vec![tuple], vec![tuple]).into();
            let source = ComputationOp::new(&mut ctx, Identifier::new("source"), signature);
            let block = BasicBlock::new(&mut ctx, None, vec![]);
            block.insert_at_back(source.body(&ctx), &ctx);
            let parameter = ParameterOp::new(&mut ctx, tuple, vec![], Index(0)).get_operation();
            parameter.insert_at_back(block, &ctx);
            let provenance = |paths: Vec<Vec<i64>>| OriginalValue {
                elements: paths
                    .into_iter()
                    .map(|path| OriginalValueElement {
                        shape_index: path.clone(),
                        original_array: vec![OriginalArray {
                            instruction_name: "original_source".into(),
                            shape_index: path,
                        }],
                    })
                    .collect(),
                ..Default::default()
            };
            let original_value = provenance(vec![vec![0], vec![1]]);
            attribute(
                &ctx,
                parameter,
                "hlo_original_value",
                original_value.clone(),
            );
            returned(&mut ctx, block, parameter);
            let template = ParameterOp::new(&mut ctx, scalar, vec![], Index(0)).get_operation();
            attribute(&ctx, template, "hlo_original_value", original_value.clone());
            let replacements = Replacements {
                instructions: HashMap::from([(parameter, Some(template))]),
                ..Default::default()
            };
            assert!(clone(&mut ctx, &source, &replacements).is_err());
            let raw = clone_with_replacements(&mut ctx, &source, "raw".into(), &replacements, None)
                .unwrap();
            assert_eq!(
                optional_attr::<OriginalValue>(
                    &ctx,
                    raw.mapping.lookup_op(template).unwrap(),
                    "hlo_original_value",
                )
                .unwrap()
                .as_deref(),
                Some(&original_value),
            );
            assert!(crate::verification::verify(&ctx, raw.computation.get_operation()).is_err());
            Operation::erase(raw.computation.get_operation(), &mut ctx);
            let rebased = provenance(vec![vec![]]);
            attribute(&ctx, template, "hlo_original_value", rebased.clone());
            let output = clone(&mut ctx, &source, &replacements).unwrap();
            assert_eq!(
                *optional_attr::<OriginalValue>(
                    &ctx,
                    output.mapping.lookup_op(template).unwrap(),
                    "hlo_original_value"
                )
                .unwrap()
                .unwrap(),
                rebased,
            );
            assert_eq!(
                optional_attr::<OriginalValue>(&ctx, parameter, "hlo_original_value")
                    .unwrap()
                    .unwrap()
                    .elements
                    .len(),
                2,
            );
            crate::verification::verify(&ctx, output.computation.get_operation()).unwrap();
            Operation::erase(output.computation.get_operation(), &mut ctx);
            Operation::erase(template, &mut ctx);
        }

        #[test]
        fn async_start_clones_wrapped_ownership_instead_of_sharing_original() {
            use crate::ops::control::{AsyncDoneOp, AsyncStartOp, AsyncUpdateOp};
            let mut ctx = context();
            let (work, block, parameters) = computation(&mut ctx, "work", 1);
            returned(&mut ctx, block, parameters[0]);
            let (source, block, parameters) = computation(&mut ctx, "source", 1);
            let ty = scalar(&ctx);
            let state = TensorType::get(&ctx, ElementType::S32, vec![]).unwrap();
            let inputs = TupleType::get(&ctx, vec![ty]).unwrap();
            let async_type = TupleType::get(&ctx, vec![inputs, ty, state]).unwrap();
            let input = parameters[0].deref(&ctx).get_result(0);
            let start = AsyncStartOp::new(
                &mut ctx,
                async_type,
                vec![input],
                Callees(vec!["work".into()]),
            );
            attribute(
                &ctx,
                start.get_operation(),
                "hlo_name",
                InstructionName("async_start".into()),
            );
            start.get_operation().insert_at_back(block, &ctx);
            let value = start.result(&ctx);
            let update = AsyncUpdateOp::new(&mut ctx, async_type, vec![value]);
            update.get_operation().insert_at_back(block, &ctx);
            let value = update.result(&ctx);
            let done = AsyncDoneOp::new(&mut ctx, ty, vec![value]);
            attribute(
                &ctx,
                done.get_operation(),
                "hlo_name",
                InstructionName("async_done".into()),
            );
            done.get_operation().insert_at_back(block, &ctx);
            returned(&mut ctx, block, done.get_operation());
            let module = ModuleOp::new(&mut ctx, Identifier::new("test"));
            let block = module
                .get_operation()
                .deref(&ctx)
                .get_region(0)
                .deref(&ctx)
                .get_entry_block()
                .unwrap();
            for computation in [&work, &source] {
                computation.get_operation().insert_at_back(block, &ctx);
            }
            crate::ir::set_entry(&ctx, module.get_operation(), Identifier::new("source")).unwrap();
            crate::verification::verify(&ctx, module.get_operation()).unwrap();
            let before = module.get_operation().disp(&ctx).to_string();
            let output = clone(&mut ctx, &source, &Replacements::default()).unwrap();
            assert_eq!(output.embedded_computations.len(), 1);
            assert_eq!(before, module.get_operation().disp(&ctx).to_string());
            let cloned_start = output.mapping.lookup_op(start.get_operation()).unwrap();
            assert_ne!(
                optional_attr::<Callees>(&ctx, cloned_start, "hlo_callees")
                    .unwrap()
                    .unwrap()
                    .0[0],
                "work",
            );
            for embedded in &output.embedded_computations {
                embedded.get_operation().insert_at_back(block, &ctx);
            }
            output
                .computation
                .get_operation()
                .insert_at_back(block, &ctx);
            crate::verification::verify(&ctx, module.get_operation()).unwrap();
            let second = clone(&mut ctx, &source, &Replacements::default()).unwrap();
            for original in [start.get_operation(), done.get_operation()] {
                let first_name = optional_attr::<InstructionName>(
                    &ctx,
                    output.mapping.lookup_op(original).unwrap(),
                    "hlo_name",
                )
                .unwrap()
                .unwrap()
                .0
                .clone();
                let second_name = optional_attr::<InstructionName>(
                    &ctx,
                    second.mapping.lookup_op(original).unwrap(),
                    "hlo_name",
                )
                .unwrap()
                .unwrap()
                .0
                .clone();
                assert_ne!(first_name, second_name);
            }
            assert_ne!(
                output.embedded_computations[0].get_operation(),
                second.embedded_computations[0].get_operation(),
            );
            for computation in &second.embedded_computations {
                computation.get_operation().insert_at_back(block, &ctx);
            }
            second
                .computation
                .get_operation()
                .insert_at_back(block, &ctx);
            crate::verification::verify(&ctx, module.get_operation()).unwrap();
        }
    }

    mod ordering {
        use super::*;

        #[test]
        fn mapped_order_matrix_is_stable_with_aliases_omissions_and_unmapped_members() {
            let mut ctx = context();
            let (_, _, parameters) = computation(&mut ctx, "ordering", 5);
            let values: Vec<_> = parameters
                .iter()
                .map(|parameter| parameter.deref(&ctx).get_result(0))
                .collect();
            for (preferred, expected) in [
                (vec![], vec![0, 1, 2, 3]),
                (vec![3, 1, 0, 2], vec![3, 1, 0, 2]),
                (vec![2, 2, 0, 0], vec![2, 0, 1, 3]),
                (vec![4, 2, 4, 0], vec![2, 0, 1, 3]),
                (vec![3, 4, 1, 3, 2], vec![3, 1, 2, 0]),
            ] {
                assert_eq!(
                    mapped_order(
                        &values[..4],
                        preferred.into_iter().map(|index| values[index])
                    ),
                    expected
                        .into_iter()
                        .map(|index| values[index])
                        .collect::<Vec<_>>(),
                );
            }
        }

        #[test]
        fn clone_preserves_explicit_source_user_and_successor_permutations() {
            let mut ctx = context();
            let (source, block, parameters) = computation(&mut ctx, "source", 1);
            let ty = scalar(&ctx);
            let input = parameters[0].deref(&ctx).get_result(0);
            let first = AddOp::new(&mut ctx, ty, vec![input, input]).get_operation();
            first.insert_at_back(block, &ctx);
            let second = NegateOp::new(&mut ctx, ty, vec![input]).get_operation();
            second.insert_at_back(block, &ctx);
            for (operation, name) in [(parameters[0], "p"), (first, "first"), (second, "second")] {
                attribute(&ctx, operation, "hlo_name", InstructionName(name.into()));
            }
            for operation in [first, second] {
                attribute(
                    &ctx,
                    operation,
                    "hlo_control_predecessors",
                    ControlPredecessors(vec!["p".into()]),
                );
            }
            returned(&mut ctx, block, second);
            instruction_order::set_order(
                &ctx,
                source.get_operation(),
                InstructionOrder {
                    users: vec![OrderPermutation {
                        subject: "p".into(),
                        members: vec!["second".into(), "first".into()],
                    }],
                    control_successors: vec![OrderPermutation {
                        subject: "p".into(),
                        members: vec!["second".into(), "first".into()],
                    }],
                    ..Default::default()
                },
            )
            .unwrap();
            let before = source.get_operation().disp(&ctx).to_string();
            let output = clone(&mut ctx, &source, &Replacements::default()).unwrap();
            let parameter = output
                .mapping
                .lookup_op(parameters[0])
                .unwrap()
                .deref(&ctx)
                .get_result(0);
            let expected: Vec<_> = [second, first]
                .into_iter()
                .map(|operation| {
                    output
                        .mapping
                        .lookup_op(operation)
                        .unwrap()
                        .deref(&ctx)
                        .get_result(0)
                })
                .collect();
            assert_eq!(instruction_order::users(&ctx, parameter).unwrap(), expected);
            assert_eq!(
                instruction_order::control_successors(&ctx, parameter).unwrap(),
                expected
            );
            assert_eq!(before, source.get_operation().disp(&ctx).to_string());
            instruction_order::verify_order(&ctx, output.computation.get_operation()).unwrap();
            crate::verification::verify(&ctx, output.computation.get_operation()).unwrap();
        }

        #[test]
        fn omitted_control_successors_are_filtered_and_unresolved_edges_error() {
            let mut ctx = context();
            let (source, block, parameters) = computation(&mut ctx, "source", 2);
            attribute(&ctx, parameters[0], "hlo_name", InstructionName("p".into()));
            attribute(
                &ctx,
                parameters[1],
                "hlo_name",
                InstructionName("unused".into()),
            );
            attribute(
                &ctx,
                parameters[1],
                "hlo_control_predecessors",
                ControlPredecessors(vec!["p".into()]),
            );
            returned(&mut ctx, block, parameters[0]);
            let before = source.get_operation().disp(&ctx).to_string();
            let replacements = Replacements {
                instructions: HashMap::from([(parameters[1], None)]),
                ..Default::default()
            };
            let output = clone(&mut ctx, &source, &replacements).unwrap();
            let parameter = output
                .mapping
                .lookup_op(parameters[0])
                .unwrap()
                .deref(&ctx)
                .get_result(0);
            assert!(
                instruction_order::control_successors(&ctx, parameter)
                    .unwrap()
                    .is_empty()
            );
            assert!(output.mapping.lookup_op(parameters[1]).is_none());
            assert_eq!(before, source.get_operation().disp(&ctx).to_string());
            Operation::erase(output.computation.get_operation(), &mut ctx);
            attribute(
                &ctx,
                parameters[1],
                "hlo_control_predecessors",
                ControlPredecessors(vec!["missing".into()]),
            );
            let malformed = source.get_operation().disp(&ctx).to_string();
            // Raw cloning must still reject ambiguous/unresolved symbolic graph
            // references, independent of optional semantic verification.
            assert!(
                clone_with_replacements(
                    &mut ctx,
                    &source,
                    "raw".into(),
                    &Replacements::default(),
                    None,
                )
                .is_err()
            );
            assert_eq!(malformed, source.get_operation().disp(&ctx).to_string());
        }

        #[test]
        fn cross_module_context_clones_foreign_callees_once_and_commits_mappings_on_success() {
            let mut ctx = context();
            let (callee, block, parameters) = computation(&mut ctx, "callee", 1);
            returned(&mut ctx, block, parameters[0]);
            let (source, block, parameters) = computation(&mut ctx, "source", 1);
            let ty = scalar(&ctx);
            let input = parameters[0].deref(&ctx).get_result(0);
            let call = CallOp::new(&mut ctx, ty, vec![input], Callees(vec!["callee".into()]))
                .get_operation();
            attribute(
                &ctx,
                call,
                "hlo_name",
                InstructionName("foreign_call".into()),
            );
            call.insert_at_back(block, &ctx);
            returned(&mut ctx, block, call);
            let source_module = ModuleOp::new(&mut ctx, Identifier::new("foreign"));
            let source_block = source_module
                .get_operation()
                .deref(&ctx)
                .get_region(0)
                .deref(&ctx)
                .get_entry_block()
                .unwrap();
            callee.get_operation().insert_at_back(source_block, &ctx);
            source.get_operation().insert_at_back(source_block, &ctx);
            crate::ir::set_entry(
                &ctx,
                source_module.get_operation(),
                Identifier::new("source"),
            )
            .unwrap();
            let (entry, block, parameters) = computation(&mut ctx, "entry", 1);
            returned(&mut ctx, block, parameters[0]);
            let destination = ModuleOp::new(&mut ctx, Identifier::new("destination"));
            let destination_block = destination
                .get_operation()
                .deref(&ctx)
                .get_region(0)
                .deref(&ctx)
                .get_entry_block()
                .unwrap();
            entry
                .get_operation()
                .insert_at_back(destination_block, &ctx);
            crate::ir::set_entry(&ctx, destination.get_operation(), Identifier::new("entry"))
                .unwrap();
            let before_source = source_module.get_operation().disp(&ctx).to_string();
            let before_destination = destination.get_operation().disp(&ctx).to_string();
            let mut context =
                ComputationCloneContext::new(Some(destination.get_operation()), "clone".into());
            let first = clone_in_context(
                &mut ctx,
                &source,
                "copy".into(),
                &Replacements::default(),
                Some(VerifyOptions::default()),
                &mut context,
            )
            .unwrap();
            assert_eq!(first.embedded_computations.len(), 1);
            assert_eq!(context.instruction(call), first.mapping.lookup_op(call));
            assert_eq!(
                context.instruction_by_name("foreign_call"),
                first.mapping.lookup_op(call)
            );
            assert_eq!(
                context.computation_by_name("source"),
                Some(first.computation.get_operation())
            );
            let shared_callee = context.computation(callee.get_operation()).unwrap();
            let second = clone_in_context(
                &mut ctx,
                &source,
                "copy".into(),
                &Replacements::default(),
                Some(VerifyOptions::default()),
                &mut context,
            )
            .unwrap();
            assert!(second.embedded_computations.is_empty());
            assert_eq!(
                context.computation(callee.get_operation()),
                Some(shared_callee)
            );
            assert_eq!(context.instruction(call), second.mapping.lookup_op(call));
            assert_eq!(
                context.instruction_by_name("foreign_call"),
                second.mapping.lookup_op(call)
            );
            assert_eq!(
                context.computation_by_name("source"),
                Some(second.computation.get_operation())
            );
            let names = context.computation_names.clone();
            let mappings = context.computations.clone();
            let invalid = Replacements {
                instructions: HashMap::from([(call, None)]),
                ..Default::default()
            };
            assert!(
                clone_in_context(
                    &mut ctx,
                    &source,
                    "failed".into(),
                    &invalid,
                    None,
                    &mut context,
                )
                .is_err()
            );
            assert_eq!(context.computation_names, names);
            assert_eq!(context.computations, mappings);
            assert_eq!(
                before_source,
                source_module.get_operation().disp(&ctx).to_string()
            );
            assert_eq!(
                before_destination,
                destination.get_operation().disp(&ctx).to_string()
            );
            for computation in &first.embedded_computations {
                computation
                    .get_operation()
                    .insert_at_back(destination_block, &ctx);
            }
            for output in [&first, &second] {
                output
                    .computation
                    .get_operation()
                    .insert_at_back(destination_block, &ctx);
            }
            crate::verification::verify(&ctx, destination.get_operation()).unwrap();
            crate::verification::verify(&ctx, source_module.get_operation()).unwrap();
        }

        #[test]
        fn inline_fusion_clones_block_arguments_controls_and_nested_accuracy() {
            use crate::{
                attributes::{FusionKind, OpMetadata},
                ir::FusionParameter,
                ir::{OriginalArray, OriginalValueElement},
                ops::control::FusionOp,
                sharding::Sharding,
            };
            let mut ctx = context();
            let (source, block, parameters) = computation(&mut ctx, "source", 1);
            let ty = scalar(&ctx);
            let input = parameters[0].deref(&ctx).get_result(0);
            let fusion = FusionOp::new(&mut ctx, ty, vec![input], FusionKind::Loop);
            fusion.get_operation().insert_at_back(block, &ctx);
            let fusion_block = BasicBlock::new(&mut ctx, None, vec![ty]);
            fusion_block.insert_at_back(fusion.body(&ctx), &ctx);
            let argument = fusion_block.deref(&ctx).get_argument(0);
            let root = ExpOp::new(&mut ctx, ty, vec![argument]).get_operation();
            root.insert_at_back(fusion_block, &ctx);
            attribute(&ctx, root, "hlo_name", InstructionName("inside".into()));
            attribute(&ctx, root, "hlo_result_accuracy", ResultAccuracy::Highest);
            attribute(
                &ctx,
                root,
                "hlo_control_predecessors",
                ControlPredecessors(vec!["argument".into()]),
            );
            crate::ir::set_fusion_parameters(
                &ctx,
                fusion_block,
                FusionParameters(vec![FusionParameter {
                    name: Some("argument".into()),
                    frontend_attributes: FrontendAttributes::new([(
                        "annotation".into(),
                        "preserved".into(),
                    )]),
                    metadata: Some(OpMetadata {
                        op_name: "original_argument".into(),
                        source_file: "clone_fixture".into(),
                        source_line: 7,
                        ..Default::default()
                    }),
                    sharding: Some(Sharding::replicated()),
                    original_value: Some(OriginalValue {
                        elements: vec![OriginalValueElement {
                            shape_index: vec![],
                            original_array: vec![OriginalArray {
                                instruction_name: "original_argument".into(),
                                shape_index: vec![],
                            }],
                        }],
                        ..Default::default()
                    }),
                    ..Default::default()
                }]),
            )
            .unwrap();
            returned(&mut ctx, fusion_block, root);
            returned(&mut ctx, block, fusion.get_operation());
            attribute(
                &ctx,
                fusion.get_operation(),
                "hlo_name",
                InstructionName("fusion".into()),
            );
            instruction_order::set_order(
                &ctx,
                source.get_operation(),
                InstructionOrder {
                    blocks: vec![OrderPermutation {
                        subject: "fusion".into(),
                        members: vec!["inside".into(), "argument".into()],
                    }],
                    ..Default::default()
                },
            )
            .unwrap();
            let before = source.get_operation().disp(&ctx).to_string();
            let output = clone(&mut ctx, &source, &Replacements::default()).unwrap();
            let cloned_block = output.mapping.lookup_block(fusion_block).unwrap();
            let metadata = crate::ir::fusion_parameters(&ctx, cloned_block).unwrap();
            let mut expected = crate::ir::fusion_parameters(&ctx, fusion_block).unwrap();
            expected.0[0].name.clone_from(&metadata.0[0].name);
            assert_eq!(metadata, expected);
            let cloned_root = output.mapping.lookup_op(root).unwrap();
            assert_eq!(
                instruction_order::block_value_order(&ctx, cloned_block).unwrap(),
                vec![
                    cloned_root.deref(&ctx).get_result(0),
                    cloned_block.deref(&ctx).get_argument(0)
                ]
            );
            assert_eq!(
                optional_attr::<ControlPredecessors>(&ctx, cloned_root, "hlo_control_predecessors")
                    .unwrap()
                    .unwrap()
                    .0,
                vec![metadata.0[0].name.clone().unwrap()],
            );
            assert_eq!(
                *optional_attr::<ResultAccuracy>(&ctx, cloned_root, "hlo_result_accuracy")
                    .unwrap()
                    .unwrap(),
                ResultAccuracy::Highest,
            );
            assert_eq!(
                cloned_root.deref(&ctx).get_operand(0),
                cloned_block.deref(&ctx).get_argument(0)
            );
            assert_eq!(before, source.get_operation().disp(&ctx).to_string());
            crate::verification::verify(&ctx, output.computation.get_operation()).unwrap();
            Operation::erase(output.computation.get_operation(), &mut ctx);
            let mut invalid = crate::ir::fusion_parameters(&ctx, fusion_block).unwrap();
            invalid.0[0].original_value.as_mut().unwrap().elements[0].shape_index = vec![0];
            fusion_block
                .deref_mut(&ctx)
                .attributes
                .set(Identifier::new("hlo_fusion_parameters"), invalid.clone());
            let before = source.get_operation().disp(&ctx).to_string();
            assert!(clone(&mut ctx, &source, &Replacements::default()).is_err());
            let raw = clone_with_replacements(
                &mut ctx,
                &source,
                "raw_fusion".into(),
                &Replacements::default(),
                None,
            )
            .unwrap();
            let raw_block = raw.mapping.lookup_block(fusion_block).unwrap();
            assert_eq!(
                crate::ir::fusion_parameters(&ctx, raw_block).unwrap().0[0].original_value,
                invalid.0[0].original_value,
            );
            assert!(crate::verification::verify(&ctx, raw.computation.get_operation()).is_err());
            assert_eq!(before, source.get_operation().disp(&ctx).to_string());
            Operation::erase(raw.computation.get_operation(), &mut ctx);
        }
    }

    mod scheduling {
        use super::*;

        #[test]
        fn scheduled_companion_maps_order_and_inserts_extra_parameters_after_existing_ones() {
            let mut ctx = context();
            let (source, block, parameters) = computation(&mut ctx, "source", 1);
            let ty = scalar(&ctx);
            let input = parameters[0].deref(&ctx).get_result(0);
            let root = NegateOp::new(&mut ctx, ty, vec![input]).get_operation();
            root.insert_at_back(block, &ctx);
            attribute(
                &ctx,
                parameters[0],
                "hlo_name",
                InstructionName("input".into()),
            );
            attribute(&ctx, root, "hlo_name", InstructionName("root".into()));
            returned(&mut ctx, block, root);
            let module = ModuleOp::new(&mut ctx, Identifier::new("test"));
            let block = module
                .get_operation()
                .deref(&ctx)
                .get_region(0)
                .deref(&ctx)
                .get_entry_block()
                .unwrap();
            source.get_operation().insert_at_back(block, &ctx);
            crate::ir::set_entry(&ctx, module.get_operation(), Identifier::new("source")).unwrap();
            attribute(
                &ctx,
                module.get_operation(),
                "hlo_schedule",
                Schedule(vec![InstructionSequence {
                    computation: "source".into(),
                    instructions: vec!["input".into(), "root".into()],
                }]),
            );
            let extra = ParameterOp::new(&mut ctx, ty, vec![], Index(1)).get_operation();
            let replacements = Replacements {
                extra_parameters: vec![extra],
                ..Default::default()
            };
            let before = module.get_operation().disp(&ctx).to_string();
            let output = clone_with_schedule_and_replacements(
                &mut ctx,
                &source,
                "cloned".into(),
                &replacements,
                Some(VerifyOptions::default()),
            )
            .unwrap();
            assert_eq!(before, module.get_operation().disp(&ctx).to_string());
            let mapped_name = |original| {
                optional_attr::<InstructionName>(
                    &ctx,
                    output.cloned.mapping.lookup_op(original).unwrap(),
                    "hlo_name",
                )
                .unwrap()
                .unwrap()
                .0
                .clone()
            };
            assert_eq!(
                output.sequence.instructions,
                vec![
                    mapped_name(parameters[0]),
                    mapped_name(extra),
                    mapped_name(root)
                ],
            );
            output
                .cloned
                .computation
                .get_operation()
                .insert_at_back(block, &ctx);
            let schedule = Schedule(vec![
                InstructionSequence {
                    computation: "source".into(),
                    instructions: vec!["input".into(), "root".into()],
                },
                output.sequence,
            ]);
            attribute(&ctx, module.get_operation(), "hlo_schedule", schedule);
            crate::verification::verify(&ctx, module.get_operation()).unwrap();
            Operation::erase(extra, &mut ctx);
        }

        #[test]
        fn raw_schedule_preserves_mapped_order_and_verified_failure_rolls_back_for_recovery() {
            let mut ctx = context();
            let (source, block, parameters) = computation(&mut ctx, "source", 1);
            let ty = scalar(&ctx);
            let input = parameters[0].deref(&ctx).get_result(0);
            let root = NegateOp::new(&mut ctx, ty, vec![input]).get_operation();
            root.insert_at_back(block, &ctx);
            returned(&mut ctx, block, root);
            attribute(
                &ctx,
                parameters[0],
                "hlo_name",
                InstructionName("input".into()),
            );
            attribute(&ctx, root, "hlo_name", InstructionName("root".into()));
            let module = ModuleOp::new(&mut ctx, Identifier::new("test"));
            let module_block = module
                .get_operation()
                .deref(&ctx)
                .get_region(0)
                .deref(&ctx)
                .get_entry_block()
                .unwrap();
            source.get_operation().insert_at_back(module_block, &ctx);
            crate::ir::set_entry(&ctx, module.get_operation(), Identifier::new("source")).unwrap();
            attribute(
                &ctx,
                module.get_operation(),
                "hlo_schedule",
                Schedule(vec![InstructionSequence {
                    computation: "source".into(),
                    instructions: vec!["input".into(), "root".into()],
                }]),
            );
            crate::verification::verify(&ctx, module.get_operation()).unwrap();
            let dependency = NegateOp::new(&mut ctx, ty, vec![input]).get_operation();
            let dependency_value = dependency.deref(&ctx).get_result(0);
            let replacement = NegateOp::new(&mut ctx, ty, vec![dependency_value]).get_operation();
            let replacements = Replacements {
                instructions: HashMap::from([(root, Some(replacement))]),
                root: CloneRoot::Tuple(vec![root]),
                ..Default::default()
            };
            let before = module.get_operation().disp(&ctx).to_string();
            let mut raw_sequences = Vec::new();
            for _ in 0..2 {
                let raw = clone_with_schedule_and_replacements(
                    &mut ctx,
                    &source,
                    "cloned".into(),
                    &replacements,
                    None,
                )
                .unwrap();
                let mapped: Vec<_> = [parameters[0], replacement, dependency]
                    .into_iter()
                    .map(|original| {
                        instruction_name(&ctx, raw.cloned.mapping.lookup_op(original).unwrap())
                            .unwrap()
                    })
                    .collect();
                assert_eq!(&raw.sequence.instructions[..3], mapped);
                assert_eq!(raw.sequence.instructions.len(), 4);
                assert_ne!(
                    raw.sequence.instructions,
                    sequence(
                        &ctx,
                        &raw.cloned.computation,
                        &topological_order(
                            &ctx,
                            &ordered_instructions(&ctx, &raw.cloned.computation).unwrap()
                        )
                        .unwrap(),
                    )
                    .unwrap()
                    .instructions
                );
                assert_closed(&ctx, &raw.cloned.computation);
                raw_sequences.push(raw.sequence);
                Operation::erase(raw.cloned.computation.get_operation(), &mut ctx);
            }
            assert_eq!(raw_sequences[0], raw_sequences[1]);
            let mut shared =
                ComputationCloneContext::new(Some(module.get_operation()), "clone".into());
            for _ in 0..2 {
                assert!(
                    clone_with_schedule_in_context(
                        &mut ctx,
                        &source,
                        "cloned".into(),
                        &replacements,
                        Some(VerifyOptions::default()),
                        &mut shared,
                    )
                    .is_err()
                );
                assert!(shared.computations.is_empty());
                assert!(shared.instructions.is_empty());
                assert!(shared.computations_by_name.is_empty());
                assert!(shared.instructions_by_name.is_empty());
                assert!(shared.instruction_names.is_empty());
                assert!(shared.computation_names.is_empty());
                assert!(shared.schedules.is_empty());
                assert!(shared.in_progress.is_empty());
                assert_eq!(before, module.get_operation().disp(&ctx).to_string());
            }
            let mut fresh =
                ComputationCloneContext::new(Some(module.get_operation()), "clone".into());
            let mut recovered_sequences = Vec::new();
            for context in [&mut shared, &mut fresh] {
                let recovered = clone_with_schedule_in_context(
                    &mut ctx,
                    &source,
                    "cloned".into(),
                    &Replacements::default(),
                    Some(VerifyOptions::default()),
                    context,
                )
                .unwrap();
                assert_eq!(
                    context.instruction(root),
                    recovered.cloned.mapping.lookup_op(root)
                );
                assert_eq!(context.schedules, vec![recovered.sequence.clone()]);
                recovered_sequences.push(recovered.sequence);
                Operation::erase(recovered.cloned.computation.get_operation(), &mut ctx);
            }
            assert_eq!(recovered_sequences[0], recovered_sequences[1]);
            assert_eq!(before, module.get_operation().disp(&ctx).to_string());
            Operation::erase(replacement, &mut ctx);
            Operation::erase(dependency, &mut ctx);
        }

        #[test]
        fn raw_clone_accepts_semantically_intermediate_ir_without_a_hidden_verifier() {
            let mut ctx = context();
            let (source, block, _) = computation(&mut ctx, "source", 0);
            let ty = scalar(&ctx);
            let intermediate = AddOp::new(&mut ctx, ty, vec![]).get_operation();
            intermediate.insert_at_back(block, &ctx);
            returned(&mut ctx, block, intermediate);
            assert!(crate::verification::verify(&ctx, source.get_operation()).is_err());
            let before = source.get_operation().disp(&ctx).to_string();
            let output = clone_with_replacements(
                &mut ctx,
                &source,
                "cloned".into(),
                &Replacements::default(),
                None,
            )
            .unwrap();
            assert_eq!(before, source.get_operation().disp(&ctx).to_string());
            assert!(crate::verification::verify(&ctx, output.computation.get_operation()).is_err());
            assert_closed(&ctx, &output.computation);
            assert!(clone(&mut ctx, &source, &Replacements::default()).is_err());
        }
    }
}
