//! Bounded exact/upper analysis, source metadata and checked While recurrences.
//!
//! Based on OpenXLA's `hlo/analysis/while_loop_analysis.cc`.
//! Range metadata is not an execution proof. Checked recurrences are separate
//! outputs. Analysis performs no annotation or rewriting, and provides no
//! known-range input or general dataflow. All evaluation uses the same
//! evaluator and owned partial-value representation.
mod affine;
mod auxiliary;
mod dependency;
mod host_offload;
mod proofs;
mod range;
mod trip_count;
pub use affine::SourceAffine;
pub use auxiliary::SourceAuxiliaryInduction;
pub use host_offload::SourceDynamicVariable;
pub use proofs::{ProvenDynamicVariable, ProvenPrimaryInduction};
pub use range::SourceLoopRange;

use crate::{
    attributes::{Callees, Direction, Index},
    evaluator::{DependencyMode, EvaluatedValue, EvaluationError, HloEvaluator, Request},
    literal::semantics as literal_semantics,
    opcode::Opcode,
    ops::{ComputationOp, HloOpInterface, ReturnOp, control::FusionOp},
    scalar::semantics::Scalar,
    shape,
    verification::{self, optional_attr},
};
use pliron::{
    attribute::Attribute,
    basic_block::BasicBlock,
    builtin::ops::ModuleOp,
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    op::op_cast,
    operation::Operation,
    result::Result,
    r#type::Typed,
    value::{DefiningEntity, Value},
};
use std::collections::{HashMap, HashSet};

/// One source-pattern analysis request for a particular While result.
///
/// The evaluator fallback tries candidate trip counts from zero through the
/// configured budget, inclusive. This bounds only that fallback search; it
/// does not cap other evaluator work.
#[derive(Clone, Copy, Debug)]
pub struct WhileLoopRequest {
    /// Result value of the native While operation to analyze.
    pub value: Value,
    /// Largest fallback trip count to test; must be in `0..i64::MAX`.
    pub max_brute_force_iters: i64,
}

impl WhileLoopRequest {
    /// Create a request with the default fallback budget of 128 iterations.
    pub fn new(value: Value) -> Self {
        Self {
            value,
            max_brute_force_iters: 128,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Phase of evaluation associated with a reported decline.
pub enum EvaluationStage {
    /// Evaluation of the loop's initial carried value.
    InitialValue,
    /// Evaluation of the loop condition.
    Condition,
    /// Evaluation of a carried-value update.
    Update,
    /// Evaluation used to derive a non-exact upper bound.
    UpperBound,
}

/// Expected evaluator limitations are retained, not silently replaced by zero.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeclineReason {
    /// The evaluator could not determine a value from the available inputs.
    UnknownValue,
    /// Evaluation stopped because a dependency was not available at the
    /// evaluator's configured dependency depth.
    ShallowDependency,
    /// The operation's opcode is outside the evaluator's supported subset.
    Unsupported {
        /// The opcode whose evaluator implementation is not supported.
        opcode: Opcode,
    },
    /// Evaluation of this opcode declined for the reported value-specific
    /// reason.
    UnsupportedValue {
        /// The opcode whose value could not be evaluated.
        opcode: Opcode,
        /// Details supplied by the evaluator for this unsupported value.
        message: String,
    },
}

/// A supported-analysis limitation encountered while evaluating a loop fact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvaluationDecline {
    /// The phase in which evaluation declined.
    pub stage: EvaluationStage,
    /// Why the evaluator did not produce a value at that phase.
    pub reason: DeclineReason,
}

/// None means a source pattern or supported evaluation could not prove a fact.
/// Counts are nonnegative; upper_bound is exact when exact_trip_count is known.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WhileLoopFacts {
    /// Tuple element selected as the structural induction variable, if found.
    pub induction_index: Option<usize>,
    /// Evaluated exact number of body executions, when established.
    pub exact_trip_count: Option<i64>,
    /// Nonnegative upper bound on body executions, if established. This equals
    /// `exact_trip_count` whenever an exact count is available.
    pub upper_bound: Option<i64>,
    /// Empty-known-map source-matcher metadata, never used to infer counts.
    pub source_range: Option<SourceLoopRange>,
    /// Structural extraction candidates, not affine or execution proofs.
    pub source_auxiliary_inductions: Vec<SourceAuxiliaryInduction>,
    /// Host-pattern slice indices, without a loop-carried-origin proof.
    pub source_dynamic_variables: Vec<SourceDynamicVariable>,
    /// Independently checked finite induction, never inferred from range alone.
    pub proven_primary_induction: Option<ProvenPrimaryInduction>,
    /// Typed, genuine loop-carried recurrences over the known exact count.
    pub proven_dynamic_variables: Vec<ProvenDynamicVariable>,
    /// Expected evaluator limitations, in encounter order. These are not
    /// substitutes for malformed-IR or kernel errors, which are returned.
    pub evaluation_declines: Vec<EvaluationDecline>,
}

/// Analyze without mutating IR or allocating unrelated tuple array payloads.
///
/// Malformed IR, kernel errors, effects, recursion, and loop-limit errors
/// propagate. Only explicitly classified unknown/unsupported evaluations
/// become declines. Source patterns do not inherit the execution parser's
/// broader induction shortcuts or unique-caller constant inference.
pub fn analyze(
    ctx: &Context,
    module: Ptr<Operation>,
    request: WhileLoopRequest,
) -> Result<WhileLoopFacts> {
    if request.max_brute_force_iters < 0 || request.max_brute_force_iters == i64::MAX {
        return Err(invalid("fallback budget must be in 0..i64::MAX"));
    }
    verification::verify(ctx, module)?;
    let view = ModuleView::new(ctx, module)?;
    view.check_value(request.value)?;
    let operation = request
        .value
        .defining_op()
        .ok_or_else(|| invalid("analysis requires a While result"))?;
    if opcode(ctx, operation)? != Opcode::While {
        return Err(invalid("analysis requires a While result"));
    }
    let parts = Parts::new(&view, operation)?;
    let induction_index = dependency::induction_index(&view, &parts)?;
    let mut analysis = Analysis {
        view,
        evaluator: HloEvaluator::new(ctx, module)?.with_max_loop_iterations(0),
        declines: Vec::new(),
    };
    let exact_trip_count = if let Some(index) = induction_index {
        analysis.exact(&parts, index, request.max_brute_force_iters)?
    } else {
        None
    };
    let upper_bound = if exact_trip_count.is_some() {
        exact_trip_count
    } else {
        analysis.upper(&parts)?
    };
    // Exact and range use the same traced, shallow initial-value request.
    let initial_value_declined = analysis
        .declines
        .iter()
        .any(|decline| decline.stage == EvaluationStage::InitialValue);
    let source_range = match induction_index {
        Some(index) if !initial_value_declined => analysis.source_range(&parts, index)?,
        _ => None,
    };
    let source_auxiliary_inductions = auxiliary::candidates(&analysis.view, &parts)?;
    let source_dynamic_variables = host_offload::dynamic_variables(&analysis.view, &parts)?;
    let proven_primary_induction = proofs::primary(
        &analysis.view,
        &parts,
        induction_index,
        exact_trip_count,
        source_range,
    )?;
    let proven_dynamic_variables = proofs::dynamics(
        &analysis.view,
        &parts,
        exact_trip_count,
        &source_dynamic_variables,
        proven_primary_induction,
    )?;
    Ok(WhileLoopFacts {
        induction_index,
        exact_trip_count,
        upper_bound,
        source_range,
        source_auxiliary_inductions,
        source_dynamic_variables,
        proven_primary_induction,
        proven_dynamic_variables,
        evaluation_declines: analysis.declines,
    })
}

fn invalid(message: &str) -> pliron::result::Error {
    pliron::verify_error_noloc!(EvaluationError::Malformed {
        message: message.into(),
    })
}

fn opcode(ctx: &Context, op: Ptr<Operation>) -> Result<Opcode> {
    let dynamic = Operation::get_op_dyn(op, ctx);
    op_cast::<dyn HloOpInterface>(&*dynamic)
        .map(|hlo| hlo.opcode())
        .ok_or_else(|| invalid("analysis encountered a non-HLO instruction"))
}

fn attr<T: Attribute + Clone>(ctx: &Context, op: Ptr<Operation>, key: &'static str) -> Result<T> {
    optional_attr::<T>(ctx, op, key)?
        .map(|value| value.clone())
        .ok_or_else(|| invalid("analysis instruction is missing a typed attribute"))
}

fn inputs(ctx: &Context, op: Ptr<Operation>) -> Vec<Value> {
    op.deref(ctx).operands().collect()
}

fn defining(value: Value) -> Result<Ptr<Operation>> {
    value
        .defining_op()
        .ok_or_else(|| invalid("analysis expected an instruction result"))
}

fn operand(ctx: &Context, value: Value, index: usize) -> Result<Value> {
    defining(value)?
        .deref(ctx)
        .operands()
        .nth(index)
        .ok_or_else(|| invalid("analysis operand index is invalid"))
}

fn block(ctx: &Context, owner: Ptr<Operation>) -> Result<Ptr<BasicBlock>> {
    let regions: Vec<_> = owner.deref(ctx).regions().collect();
    if regions.len() != 1 {
        return Err(invalid("analysis computation requires one region"));
    }
    let blocks: Vec<_> = regions[0].deref(ctx).iter(ctx).collect();
    if blocks.len() != 1 {
        return Err(invalid("analysis computation requires one block"));
    }
    Ok(blocks[0])
}

fn root(ctx: &Context, owner: Ptr<Operation>) -> Result<Value> {
    for op in block(ctx, owner)?.deref(ctx).iter(ctx) {
        if Operation::is_op::<ReturnOp>(op, ctx) {
            return Ok(op.deref(ctx).get_operand(0));
        }
    }
    Err(invalid("analysis computation has no return"))
}

fn parameter(ctx: &Context, owner: Ptr<Operation>, index: usize) -> Result<Value> {
    let block = block(ctx, owner)?;
    if Operation::is_op::<FusionOp>(owner, ctx) {
        return Ok(block.deref(ctx).get_argument(index));
    }
    for op in block.deref(ctx).iter(ctx) {
        if Operation::is_op::<crate::ops::control::ParameterOp>(op, ctx)
            && attr::<Index>(ctx, op, "hlo_index")?.0 as usize == index
        {
            return Ok(op.deref(ctx).get_result(0));
        }
    }
    Err(invalid("analysis computation has no requested parameter"))
}

struct ModuleView<'a> {
    ctx: &'a Context,
    module: Ptr<Operation>,
    callees: HashMap<crate::ir::ComputationName, Ptr<Operation>>,
}

impl<'a> ModuleView<'a> {
    fn new(ctx: &'a Context, module: Ptr<Operation>) -> Result<Self> {
        if !Operation::is_op::<ModuleOp>(module, ctx) {
            return Err(invalid("analysis requires a module"));
        }
        let mut callees = HashMap::new();
        for op in block(ctx, module)?.deref(ctx).iter(ctx) {
            if let Some(computation) = Operation::get_op::<ComputationOp>(op, ctx) {
                let name = computation
                    .get_attr_hlo_computation_name(ctx)
                    .ok_or_else(|| invalid("analysis computation has no name"))?;
                callees.insert(name.clone(), op);
            }
        }
        Ok(Self {
            ctx,
            module,
            callees,
        })
    }

    fn resolve(&self, name: &crate::ir::ComputationName) -> Result<Ptr<Operation>> {
        self.callees
            .get(name)
            .copied()
            .ok_or_else(|| invalid("analysis callee is unresolved"))
    }

    fn callee(&self, op: Ptr<Operation>) -> Result<Ptr<Operation>> {
        if opcode(self.ctx, op)? == Opcode::Fusion {
            return Ok(op);
        }
        let names = attr::<Callees>(self.ctx, op, "hlo_callees")?;
        if names.0.len() != 1 {
            return Err(invalid("analysis Call requires one callee"));
        }
        self.resolve(&names.0[0])
    }

    fn check_value(&self, value: Value) -> Result<()> {
        let block = match value.defining_entity() {
            DefiningEntity::Op(op) => op.deref(self.ctx).get_parent_block(),
            DefiningEntity::Block(block) => Some(block),
        }
        .ok_or_else(|| invalid("analysis value is detached"))?;
        let mut owner = block.deref(self.ctx).get_parent_op(self.ctx);
        while let Some(op) = owner {
            if op == self.module {
                return Ok(());
            }
            owner = op.deref(self.ctx).get_parent_op(self.ctx);
        }
        Err(invalid("analysis value is outside the supplied module"))
    }
}

struct Parts {
    input: Value,
    condition: Ptr<Operation>,
    body_root: Value,
    condition_root: Value,
    body_parameter: Value,
    condition_parameter: Value,
}

impl Parts {
    fn new(view: &ModuleView<'_>, op: Ptr<Operation>) -> Result<Self> {
        let ctx = view.ctx;
        let names = attr::<Callees>(ctx, op, "hlo_callees")?;
        if names.0.len() != 2 {
            return Err(invalid("While analysis requires body and condition"));
        }
        let body = view.resolve(&names.0[0])?;
        let condition = view.resolve(&names.0[1])?;
        Ok(Self {
            input: op.deref(ctx).get_operand(0),
            condition,
            body_root: root(ctx, body)?,
            condition_root: root(ctx, condition)?,
            body_parameter: parameter(ctx, body, 0)?,
            condition_parameter: parameter(ctx, condition, 0)?,
        })
    }
}

fn is_opcode(ctx: &Context, value: Value, expected: Opcode) -> Result<bool> {
    value
        .defining_op()
        .map(|op| opcode(ctx, op).map(|actual| actual == expected))
        .transpose()
        .map(|actual| actual.unwrap_or(false))
}

fn trace(ctx: &Context, mut value: Value) -> Result<Value> {
    let mut visited = HashSet::new();
    while visited.insert(value) {
        let Some(op) = value.defining_op() else {
            return Ok(value);
        };
        match opcode(ctx, op)? {
            Opcode::Copy | Opcode::CopyStart | Opcode::CopyDone => {
                value = op.deref(ctx).get_operand(0);
            }
            Opcode::GetTupleElement => {
                let tuple = op.deref(ctx).get_operand(0);
                if !is_opcode(ctx, tuple, Opcode::Tuple)? {
                    return Ok(value);
                }
                let index = attr::<Index>(ctx, op, "hlo_index")?.0 as usize;
                value = operand(ctx, tuple, index)?;
            }
            _ => return Ok(value),
        }
    }
    Err(invalid("cycle while tracing analysis source"))
}

fn unique_gte(ctx: &Context, parameter: Value, index: usize) -> Result<Option<Value>> {
    let users: HashSet<_> = parameter
        .uses(ctx)
        .iter()
        .map(|use_| use_.user_op())
        .collect();
    let mut found = None;
    for user in users {
        if Operation::is_op::<ReturnOp>(user, ctx) {
            continue;
        }
        if opcode(ctx, user)? == Opcode::GetTupleElement
            && attr::<Index>(ctx, user, "hlo_index")?.0 as usize == index
        {
            if found.is_some() {
                return Ok(None);
            }
            found = Some(user.deref(ctx).get_result(0));
        }
    }
    Ok(found)
}

fn nonconstant_operand(ctx: &Context, value: Value) -> Result<Option<Value>> {
    let Some(op) = value.defining_op() else {
        return Ok(None);
    };
    let mut found = None;
    for value in inputs(ctx, op) {
        if !is_opcode(ctx, value, Opcode::Constant)? {
            if found.is_some_and(|found| found != value) {
                return Ok(None);
            }
            found = Some(value);
        }
    }
    Ok(found)
}

fn integer(ctx: &Context, value: &EvaluatedValue) -> Result<Option<i64>> {
    let ty = value.shape();
    if !value.is_known()
        || !shape::is_array(ctx, ty)
        || shape::is_dynamic(ctx, ty)
        || shape::element_count(ctx, ty)? != 1
    {
        return Ok(None);
    }
    let element = shape::element_type(ctx, ty)?;
    if !element.is_integer() {
        return Ok(None);
    }
    let literal = value.clone().into_literal(ctx)?;
    let scalar = Scalar::read(ctx, &literal, &vec![0; shape::rank(ctx, ty)?])?;
    if !element.is_signed() && scalar.bits() > i64::MAX as u128 {
        return Ok(None);
    }
    literal_semantics::scalar_integer(ctx, &literal).map(Some)
}

struct Analysis<'a> {
    view: ModuleView<'a>,
    evaluator: HloEvaluator<'a>,
    declines: Vec<EvaluationDecline>,
}

#[cfg(test)]
mod tests {
    //! Shared source-analysis fixtures and responsibility-specific regressions.
    use super::trip_count::checked_count;

    mod induction {
        use super::*;

        #[test]
        fn exact_bound_trivial_trip_count_and_negative_zero_nonunit_adaptations() {
            for direction in [Direction::Lt, Direction::Le] {
                for init in [-10, -5, -2, 0, 1, 5] {
                    for bound in [-3, 0, 2, 10, 42] {
                        for step in [1, 2, 5] {
                            let mut ctx = context();
                            let f = fixture(
                                &mut ctx,
                                Options {
                                    init,
                                    bound,
                                    step,
                                    direction,
                                    ..Default::default()
                                },
                            );
                            let mut current = init;
                            let mut expected = 0;
                            while if direction == Direction::Lt {
                                current < bound
                            } else {
                                current <= bound
                            } {
                                expected += 1;
                                current += step;
                            }
                            let result = facts(&ctx, &f, 0);
                            assert_eq!(result.induction_index, Some(0));
                            assert_eq!(result.exact_trip_count, Some(expected));
                            assert_eq!(result.upper_bound, Some(expected));
                        }
                    }
                }
            }
        }

        #[test]
        fn pinned_u32_overflow_is_not_a_fabricated_analytic_trip_count() {
            let mut ctx = context();
            let f = fixture(
                &mut ctx,
                Options {
                    element: ElementType::U32,
                    init: 2_147_483_648,
                    step: 2_147_483_648,
                    bound: 3_221_225_472,
                    ..Default::default()
                },
            );
            assert_eq!(
                checked_count(
                    2_147_483_648,
                    3_221_225_472,
                    2_147_483_648,
                    Direction::Lt,
                    ElementType::U32
                ),
                None
            );
            assert_eq!(facts(&ctx, &f, 128).exact_trip_count, None);
        }

        #[test]
        fn checked_distance_and_final_step_do_not_wrap_host_i64() {
            assert_eq!(
                checked_count(i64::MIN, i64::MAX, 1, Direction::Lt, ElementType::S64),
                None
            );
            assert_eq!(
                checked_count(0, i64::MAX, 1, Direction::Le, ElementType::S64),
                None
            );
            assert_eq!(
                checked_count(126, 127, 2, Direction::Lt, ElementType::S8),
                None
            );
            assert_eq!(
                checked_count(7, 7, 1, Direction::Lt, ElementType::S4),
                Some(0)
            );
        }

        #[test]
        fn indvar_initialization_not_constant_copy_and_gte_tuple_chain_adaptations() {
            for initialization in [Init::Copy, Init::TupleProjection, Init::Add] {
                let mut ctx = context();
                let f = fixture(
                    &mut ctx,
                    Options {
                        initialization,
                        ..Default::default()
                    },
                );
                assert_eq!(facts(&ctx, &f, 128).exact_trip_count, Some(5));
            }
        }

        #[test]
        fn evaluate_indvar_init_is_shallow_not_recursive_constant_folding() {
            let mut ctx = context();
            let f = fixture(
                &mut ctx,
                Options {
                    initialization: Init::NestedAdd,
                    ..Default::default()
                },
            );
            let result = facts(&ctx, &f, 128);
            assert_eq!(result.induction_index, Some(0));
            assert_eq!(result.exact_trip_count, None);
            assert_eq!(
                result.evaluation_declines,
                [EvaluationDecline {
                    stage: EvaluationStage::InitialValue,
                    reason: DeclineReason::ShallowDependency,
                }]
            );
        }

        #[test]
        fn source_parameter_init_does_not_use_execution_parser_unique_caller_inference() {
            let mut ctx = context();
            let f = fixture(
                &mut ctx,
                Options {
                    initialization: Init::Parameter,
                    ..Default::default()
                },
            );
            let shape = f.while_value.get_type(&ctx);
            let (_, block, _) = computation(&mut ctx, f.module, "outer", vec![], shape);
            let zero = constant(&mut ctx, block, ElementType::S64, 0);
            let call = CallOp::new(
                &mut ctx,
                shape,
                vec![zero, zero],
                Callees(vec!["main".into()]),
            )
            .get_operation();
            let call = append(&ctx, block, call);
            finish(&mut ctx, block, call);
            crate::ir::set_entry(&ctx, f.module, Identifier::new("outer")).unwrap();
            let replacement = EvaluatedValue::known(
                &ctx,
                literal_semantics::integer_scalar(&ctx, ElementType::S64, 0).unwrap(),
            )
            .unwrap();
            let explicit = HloEvaluator::new(&ctx, f.module)
                .unwrap()
                .evaluate(Request::Instruction {
                    value: f.initial_parameter,
                    substitutions: &[(f.initial_parameter, replacement)],
                    dependency_mode: DependencyMode::Shallow,
                })
                .unwrap();
            assert_eq!(integer(&ctx, &explicit).unwrap(), Some(0));
            let result = facts(&ctx, &f, 128);
            assert_eq!(result.exact_trip_count, None);
            assert_eq!(
                result.evaluation_declines[0].reason,
                DeclineReason::ShallowDependency
            );
        }

        #[test]
        fn update_on_indvar_copy_success_adaptation() {
            for update in [Update::CopyOperand, Update::CopyResult] {
                let mut ctx = context();
                let f = fixture(
                    &mut ctx,
                    Options {
                        update,
                        ..Default::default()
                    },
                );
                assert_eq!(facts(&ctx, &f, 0).exact_trip_count, Some(5));
            }
        }

        #[test]
        fn fused_update_and_fused_condition_op_adaptations() {
            for update_expr in [Expr::Call, Expr::Fusion] {
                let mut ctx = context();
                let f = fixture(
                    &mut ctx,
                    Options {
                        update_expr,
                        ..Default::default()
                    },
                );
                let result = facts(&ctx, &f, 128);
                assert_eq!(result.induction_index, Some(0));
                assert_eq!(result.exact_trip_count, Some(5));
            }
            for condition_expr in [Expr::Call, Expr::Fusion] {
                let mut ctx = context();
                let f = fixture(
                    &mut ctx,
                    Options {
                        condition_expr,
                        ..Default::default()
                    },
                );
                let result = facts(&ctx, &f, 128);
                assert_eq!(result.induction_index, Some(0));
                assert_eq!(result.exact_trip_count, Some(5));
            }
        }

        #[test]
        fn nested_fusion_has_qualified_scalar_dependency_provenance() {
            let mut ctx = context();
            let f = fixture(
                &mut ctx,
                Options {
                    update_expr: Expr::NestedFusion,
                    ..Default::default()
                },
            );
            assert_eq!(facts(&ctx, &f, 128).induction_index, Some(0));
            assert_eq!(facts(&ctx, &f, 128).exact_trip_count, Some(5));
        }

        #[test]
        fn loop_fusion_for_loop_variable_source_pattern_finds_index_without_faking_shallow_init() {
            let mut ctx = context();
            let f = fixture(
                &mut ctx,
                Options {
                    initialization: Init::FusionOfAdd,
                    update: Update::ChainedFusion,
                    update_expr: Expr::Fusion,
                    condition_expr: Expr::Fusion,
                    bound: 7,
                    ..Default::default()
                },
            );
            let result = facts(&ctx, &f, 128);
            assert_eq!(result.induction_index, Some(0));
            assert_eq!(result.exact_trip_count, None);
            assert_eq!(
                result.evaluation_declines,
                [EvaluationDecline {
                    stage: EvaluationStage::InitialValue,
                    reason: DeclineReason::ShallowDependency,
                }]
            );
        }

        #[test]
        fn multiple_operations_update_condition_constant_nonconstant_adaptations() {
            for (update, condition) in [
                (Update::NestedAdd, Condition::Direct),
                (Update::ComputedStep, Condition::Direct),
                (Update::Add, Condition::NestedIv),
                (Update::Add, Condition::ComputedBound),
            ] {
                let mut ctx = context();
                let f = fixture(
                    &mut ctx,
                    Options {
                        update,
                        condition,
                        ..Default::default()
                    },
                );
                let result = facts(&ctx, &f, 128);
                assert_eq!(result.induction_index, Some(0));
                assert_eq!(result.exact_trip_count, None);
            }
        }

        #[test]
        fn intermediate_tuple_non_scalar_operand_rejects_index_even_if_not_selected() {
            let mut ctx = context();
            let f = fixture(
                &mut ctx,
                Options {
                    update: Update::IntermediateTuple,
                    huge_payload: true,
                    ..Default::default()
                },
            );
            assert_eq!(facts(&ctx, &f, 128).induction_index, None);
        }

        #[test]
        fn custom_call_update_is_not_scalar_dependency_admission() {
            let mut ctx = context();
            let f = fixture(
                &mut ctx,
                Options {
                    update: Update::CustomCall,
                    ..Default::default()
                },
            );
            assert_eq!(facts(&ctx, &f, 128).induction_index, None);
        }

        #[test]
        fn condition_with_two_unknown_tuple_indices_has_no_unique_induction_variable() {
            let mut ctx = context();
            let f = fixture(
                &mut ctx,
                Options {
                    condition: Condition::OtherSlot,
                    ..Default::default()
                },
            );
            assert_eq!(facts(&ctx, &f, 128).induction_index, None);
        }

        #[test]
        fn duplicate_gte_identity_declines_fast_and_brute_force_counts() {
            for (duplicate_body_gte, duplicate_condition_gte) in [(true, false), (false, true)] {
                let mut ctx = context();
                let f = fixture(
                    &mut ctx,
                    Options {
                        duplicate_body_gte,
                        duplicate_condition_gte,
                        ..Default::default()
                    },
                );
                let result = facts(&ctx, &f, 128);
                assert_eq!(result.induction_index, Some(0));
                assert_eq!(result.exact_trip_count, None);
            }
        }
    }
    mod fallback {
        use super::*;

        #[test]
        fn analytic_count_is_not_capped_by_fallback_budget() {
            let mut ctx = context();
            let f = fixture(
                &mut ctx,
                Options {
                    bound: 1_000_000,
                    ..Default::default()
                },
            );
            assert_eq!(facts(&ctx, &f, 0).exact_trip_count, Some(1_000_000));
        }

        #[test]
        fn pinned_s32_overflow_rejects_trivial_count_but_typed_fallback_finds_seven() {
            let mut ctx = context();
            let f = fixture(
                &mut ctx,
                Options {
                    element: ElementType::S32,
                    init: 1_073_741_824,
                    step: 1_342_177_280,
                    bound: 1_879_048_192,
                    ..Default::default()
                },
            );
            assert_eq!(
                checked_count(
                    1_073_741_824,
                    1_879_048_192,
                    1_342_177_280,
                    Direction::Lt,
                    ElementType::S32
                ),
                None
            );
            assert_eq!(facts(&ctx, &f, 128).exact_trip_count, Some(7));
        }

        #[test]
        fn trivial_induction_step_requires_positive_add_but_fallback_keeps_typed_math() {
            for (step, update, init, bound, direction, expected) in [
                (1, Update::Subtract, 5, 0, Direction::Gt, Some(5)),
                (0, Update::Add, 0, 5, Direction::Lt, None),
                (2, Update::Multiply, 1, 16, Direction::Lt, Some(4)),
                (-1, Update::Add, 5, 0, Direction::Gt, Some(5)),
            ] {
                let mut ctx = context();
                let f = fixture(
                    &mut ctx,
                    Options {
                        step,
                        update,
                        init,
                        bound,
                        direction,
                        ..Default::default()
                    },
                );
                assert_eq!(facts(&ctx, &f, 128).exact_trip_count, expected);
            }
        }

        #[test]
        fn multi_output_fusion_index_traces_selected_output_but_fallback_stays_source_restricted() {
            for (update_expr, condition_expr) in [
                (Expr::MultiOutputFusion, Expr::Direct),
                (Expr::Direct, Expr::MultiOutputFusion),
            ] {
                let mut ctx = context();
                let f = fixture(
                    &mut ctx,
                    Options {
                        update_expr,
                        condition_expr,
                        ..Default::default()
                    },
                );
                let result = facts(&ctx, &f, 128);
                assert_eq!(result.induction_index, Some(0));
                assert_eq!(result.exact_trip_count, None);
            }
        }

        #[test]
        fn get_indvar_index_when_parameter_is_copied_does_not_broaden_direct_gte_fallback() {
            let mut ctx = context();
            let f = fixture(
                &mut ctx,
                Options {
                    condition: Condition::CopyParameter,
                    ..Default::default()
                },
            );
            let result = facts(&ctx, &f, 128);
            assert_eq!(result.induction_index, Some(0));
            assert_eq!(result.exact_trip_count, None);
        }

        #[test]
        fn repeated_same_nonconstant_operand_counts_once_and_uses_scalar_fallback() {
            let mut ctx = context();
            let f = fixture(
                &mut ctx,
                Options {
                    init: 1,
                    bound: 10,
                    update: Update::Double,
                    ..Default::default()
                },
            );
            assert_eq!(facts(&ctx, &f, 128).exact_trip_count, Some(4));
        }

        #[test]
        fn swapped_comparison_is_not_analytic_but_exact_fallback_can_evaluate_it() {
            let mut ctx = context();
            let f = fixture(
                &mut ctx,
                Options {
                    condition: Condition::Swapped,
                    direction: Direction::Gt,
                    ..Default::default()
                },
            );
            assert_eq!(facts(&ctx, &f, 128).exact_trip_count, Some(5));
            assert_eq!(facts(&ctx, &f, 0).exact_trip_count, None);
        }

        #[test]
        fn fallback_budget_is_inclusive_at_128_not_a_loop_trip_heuristic() {
            for (bound, budget, expected) in [
                (0, 0, Some(0)),
                (1, 0, None),
                (127, 128, Some(127)),
                (128, 128, Some(128)),
                (129, 128, None),
                (129, 129, Some(129)),
            ] {
                let mut ctx = context();
                let f = fixture(
                    &mut ctx,
                    Options {
                        bound,
                        direction: Direction::Ne,
                        ..Default::default()
                    },
                );
                assert_eq!(facts(&ctx, &f, budget).exact_trip_count, expected);
            }
        }

        #[test]
        fn final_budget_true_condition_still_evaluates_update_and_propagates_effect_error() {
            let mut ctx = context();
            let f = fixture(
                &mut ctx,
                Options {
                    direction: Direction::Ne,
                    update_control_effect: true,
                    ..Default::default()
                },
            );
            let error = analyze(
                &ctx,
                f.module,
                WhileLoopRequest {
                    value: f.while_value,
                    max_brute_force_iters: 0,
                },
            )
            .unwrap_err();
            assert!(matches!(
                error.err.downcast_ref::<EvaluationError>(),
                Some(EvaluationError::Effectful {
                    opcode: Opcode::CustomCall
                })
            ));
        }

        #[test]
        fn initially_false_fallback_does_not_execute_control_effect_update() {
            let mut ctx = context();
            let f = fixture(
                &mut ctx,
                Options {
                    bound: 0,
                    direction: Direction::Ne,
                    update_control_effect: true,
                    ..Default::default()
                },
            );
            assert_eq!(facts(&ctx, &f, 0).exact_trip_count, Some(0));
        }

        #[test]
        fn qualified_float_fallback_evaluates_the_exact_trip_count() {
            let mut ctx = context();
            let f = fixture(
                &mut ctx,
                Options {
                    element: ElementType::F32,
                    ..Default::default()
                },
            );
            let result = facts(&ctx, &f, 128);
            assert_eq!(result.induction_index, Some(0));
            assert_eq!(result.exact_trip_count, Some(5));
            assert!(result.evaluation_declines.is_empty());
        }
    }
    mod upper_bound {
        use super::*;

        #[test]
        fn single_iteration_upper_bound_adaptation_uses_owned_partial_tuple() {
            let mut ctx = context();
            let f = fixture(
                &mut ctx,
                Options {
                    initialization: Init::Parameter,
                    update: Update::Constant,
                    step: 10,
                    huge_payload: true,
                    ..Default::default()
                },
            );
            let result = facts(&ctx, &f, 128);
            assert_eq!(result.induction_index, None);
            assert_eq!(result.exact_trip_count, None);
            assert_eq!(result.upper_bound, Some(1));
        }

        #[test]
        fn no_upper_bound_adaptation_when_constant_update_keeps_condition_true() {
            let mut ctx = context();
            let f = fixture(
                &mut ctx,
                Options {
                    update: Update::Constant,
                    step: 0,
                    ..Default::default()
                },
            );
            assert_eq!(facts(&ctx, &f, 128).upper_bound, None);
        }

        #[test]
        fn upper_bound_requires_exactly_one_parameter_gte_user() {
            let mut ctx = context();
            let f = fixture(
                &mut ctx,
                Options {
                    update: Update::Constant,
                    step: 10,
                    duplicate_condition_gte: true,
                    ..Default::default()
                },
            );
            assert_eq!(facts(&ctx, &f, 128).upper_bound, None);
        }

        #[test]
        fn upper_bound_computation_scan_does_not_hide_unused_executed_effects() {
            let mut ctx = context();
            let f = fixture(
                &mut ctx,
                Options {
                    update: Update::Constant,
                    step: 10,
                    condition_effect: true,
                    ..Default::default()
                },
            );
            let error = analyze(&ctx, f.module, WhileLoopRequest::new(f.while_value)).unwrap_err();
            assert!(matches!(
                error.err.downcast_ref::<EvaluationError>(),
                Some(EvaluationError::Effectful {
                    opcode: Opcode::CustomCall
                })
            ));
        }

        #[test]
        fn upper_bound_resolves_call_and_inline_fusion_with_isolated_arguments() {
            for condition_expr in [
                Expr::Call,
                Expr::Fusion,
                Expr::NestedFusion,
                Expr::MultiOutputFusion,
            ] {
                let mut ctx = context();
                let f = fixture(
                    &mut ctx,
                    Options {
                        update: Update::Constant,
                        step: 10,
                        condition_expr,
                        ..Default::default()
                    },
                );
                assert_eq!(facts(&ctx, &f, 128).upper_bound, Some(1));
            }
        }
    }
    mod admission {
        use super::*;

        #[test]
        fn requests_reject_invalid_budget_wrong_value_and_outside_module() {
            let mut ctx = context();
            let f = fixture(&mut ctx, Options::default());
            for budget in [-1, i64::MAX] {
                let error = analyze(
                    &ctx,
                    f.module,
                    WhileLoopRequest {
                        value: f.while_value,
                        max_brute_force_iters: budget,
                    },
                )
                .unwrap_err();
                assert!(matches!(
                    error.err.downcast_ref::<EvaluationError>(),
                    Some(EvaluationError::Malformed { .. })
                ));
            }
            assert!(analyze(&ctx, f.module, WhileLoopRequest::new(f.init)).is_err());
            let other = fixture(&mut ctx, Options::default());
            assert!(analyze(&ctx, f.module, WhileLoopRequest::new(other.while_value)).is_err());
            assert_eq!(
                WhileLoopRequest::new(f.while_value).max_brute_force_iters,
                128
            );
        }

        #[test]
        fn malformed_evaluator_request_is_not_caught_as_an_analysis_decline() {
            let mut ctx = context();
            let f = fixture(&mut ctx, Options::default());
            let view = ModuleView::new(&ctx, f.module).unwrap();
            let mut analysis = Analysis {
                view,
                evaluator: HloEvaluator::new(&ctx, f.module).unwrap(),
                declines: vec![],
            };
            let known = EvaluatedValue::known(
                &ctx,
                literal_semantics::integer_scalar(&ctx, ElementType::S64, 0).unwrap(),
            )
            .unwrap();
            let error = analysis
                .instruction(
                    f.init,
                    &[(f.init, known.clone()), (f.init, known)],
                    EvaluationStage::InitialValue,
                )
                .unwrap_err();
            assert!(matches!(
                error.err.downcast_ref::<EvaluationError>(),
                Some(EvaluationError::Malformed { .. })
            ));
            assert!(analysis.declines.is_empty());
        }

        #[test]
        fn unsupported_only_is_classified_and_request_state_does_not_survive() {
            let mut ctx = context();
            let f = fixture(
                &mut ctx,
                Options {
                    update: Update::CustomCall,
                    ..Default::default()
                },
            );
            let update = root(&ctx, f.body)
                .unwrap()
                .defining_op()
                .unwrap()
                .deref(&ctx)
                .get_operand(0);
            let gte = update.defining_op().unwrap().deref(&ctx).get_operand(0);
            let mut analysis = Analysis {
                view: ModuleView::new(&ctx, f.module).unwrap(),
                evaluator: HloEvaluator::new(&ctx, f.module)
                    .unwrap()
                    .with_max_loop_iterations(0),
                declines: vec![],
            };
            let known = EvaluatedValue::known(
                &ctx,
                literal_semantics::integer_scalar(&ctx, ElementType::S64, 3).unwrap(),
            )
            .unwrap();
            assert!(
                analysis
                    .instruction(update, &[(gte, known)], EvaluationStage::Update)
                    .unwrap()
                    .is_none()
            );
            assert!(matches!(
                analysis.declines[0].reason,
                DeclineReason::Unsupported {
                    opcode: Opcode::CustomCall
                }
            ));
            let result = analysis.instruction(gte, &[], EvaluationStage::Update);
            assert!(result.unwrap().is_none());
        }

        #[test]
        fn analysis_is_read_only_and_repeated_calls_have_no_stale_substitutions() {
            let mut ctx = context();
            let f = fixture(
                &mut ctx,
                Options {
                    init: 1,
                    update: Update::Double,
                    bound: 10,
                    ..Default::default()
                },
            );
            let before = format!("{}", f.module.deref(&ctx).disp(&ctx));
            for _ in 0..3 {
                assert_eq!(facts(&ctx, &f, 128).exact_trip_count, Some(4));
            }
            assert_eq!(format!("{}", f.module.deref(&ctx).disp(&ctx)), before);
            assert_eq!(
                root(&ctx, f.condition).unwrap().get_type(&ctx),
                shape::scalar(&ctx, ElementType::Pred).unwrap()
            );
        }

        #[test]
        fn dynamic_and_nonscalar_induction_values_do_not_enter_integer_analysis() {
            let ctx = context();
            let shape = shape::dynamic_array(
                &ctx,
                ElementType::S64,
                vec![crate::shape::Dimension::Bounded(1)],
            )
            .unwrap();
            let value = EvaluatedValue::unknown(&ctx, shape).unwrap();
            assert_eq!(integer(&ctx, &value).unwrap(), None);
            let vector = shape::array(&ctx, ElementType::S64, vec![2]).unwrap();
            let literal = Literal::from_bits(&ctx, vector, &[0, 1]).unwrap();
            let value = EvaluatedValue::known(&ctx, literal).unwrap();
            assert_eq!(integer(&ctx, &value).unwrap(), None);
            let pred = shape::scalar(&ctx, ElementType::Pred).unwrap();
            let value =
                EvaluatedValue::known(&ctx, Literal::from_bits(&ctx, pred, &[1]).unwrap()).unwrap();
            assert_eq!(integer(&ctx, &value).unwrap(), None);
            assert!(matches!(value.state(), ValueState::Known(_)));
        }

        #[test]
        fn nontuple_and_dynamic_while_source_patterns_decline_without_allocation() {
            let mut ctx = context();
            let pred = shape::scalar(&ctx, ElementType::Pred).unwrap();
            let dynamic = shape::dynamic_array(
                &ctx,
                ElementType::S64,
                vec![crate::shape::Dimension::Bounded(1)],
            )
            .unwrap();
            for shape in [pred, dynamic] {
                let module = ModuleOp::new(&mut ctx, Identifier::new("nontuple")).get_operation();
                crate::ir::set_entry(&ctx, module, Identifier::new("main")).unwrap();
                let (_, block, args) = computation(&mut ctx, module, "body", vec![shape], shape);
                finish(&mut ctx, block, args[0]);
                let (_, block, _) = computation(&mut ctx, module, "condition", vec![shape], pred);
                let literal = Literal::from_bits(&ctx, pred, &[0]).unwrap();
                let constant = ConstantOp::new(&mut ctx, pred, vec![], literal).get_operation();
                let constant = append(&ctx, block, constant);
                finish(&mut ctx, block, constant);
                let (_, block, args) = computation(&mut ctx, module, "main", vec![shape], shape);
                let while_op = WhileOp::new(
                    &mut ctx,
                    shape,
                    vec![args[0]],
                    Callees(vec!["body".into(), "condition".into()]),
                )
                .get_operation();
                let while_value = append(&ctx, block, while_op);
                finish(&mut ctx, block, while_value);
                let result = analyze(&ctx, module, WhileLoopRequest::new(while_value)).unwrap();
                assert_eq!(result.induction_index, None);
                assert_eq!(result.exact_trip_count, None);
                assert_eq!(result.upper_bound, None);
            }
        }
    }
    use super::*;
    use pliron::identifier::Identifier;

    mod metadata_tests {
        //! Structural auxiliary, affine, host-offload, and checked-proof fixtures.
        //! Qualified source metadata and separately checked recurrence tests.

        mod auxiliary_tests {
            //! Auxiliary induction source-candidate discovery.
            use super::*;

            #[test]

            fn upstream_aiv_no_chain_and_multichain() {
                for chain in [
                    vec![Opcode::Add],
                    vec![Opcode::Add, Opcode::Subtract, Opcode::Multiply],
                    vec![Opcode::Add, Opcode::Divide],
                ] {
                    let mut ctx = context();

                    let f = fixture(&mut ctx, Options::default());

                    update_chain(&mut ctx, &f, &chain);

                    assert_eq!(aux(&ctx, &f), [0]);
                }
            }

            #[test]

            fn upstream_no_aiv_repeated_nonconstant_occurrences_and_two_variable_chain() {
                for update in [Update::Double, Update::ComputedStep] {
                    let mut ctx = context();

                    let f = fixture(
                        &mut ctx,
                        Options {
                            update,

                            ..Default::default()
                        },
                    );

                    assert!(aux(&ctx, &f).is_empty());
                }

                let mut ctx = context();

                let f = fixture(&mut ctx, Options::default());

                let (_, p) = parts(&ctx, &f);

                let update = operand(&ctx, p.body_root, 0).unwrap();

                let iv = operand(&ctx, update, 0).unwrap();

                let data = operand(&ctx, p.body_root, 1).unwrap();

                let add = math_before(&mut ctx, p.body_root, Opcode::Add, iv, data);

                let sub = math_before(&mut ctx, p.body_root, Opcode::Subtract, add, data);

                Operation::replace_operand(p.body_root.defining_op().unwrap(), &ctx, 0, sub);

                assert!(aux(&ctx, &f).is_empty());
            }

            #[test]

            fn auxiliary_duplicates_identity_and_direct_constant_decline() {
                for options in [
                    Options {
                        duplicate_body_gte: true,

                        ..Default::default()
                    },
                    Options {
                        update: Update::Constant,

                        ..Default::default()
                    },
                ] {
                    let mut ctx = context();

                    let f = fixture(&mut ctx, options);

                    assert!(aux(&ctx, &f).is_empty());
                }

                let mut ctx = context();

                let f = fixture(&mut ctx, Options::default());

                let (_, p) = parts(&ctx, &f);

                let update = operand(&ctx, p.body_root, 0).unwrap();

                let iv = operand(&ctx, update, 0).unwrap();

                Operation::replace_operand(p.body_root.defining_op().unwrap(), &ctx, 0, iv);

                assert!(aux(&ctx, &f).is_empty());
            }

            #[test]

            fn auxiliary_does_not_trace_copy_call_or_fusion() {
                for options in [
                    Options {
                        update: Update::CopyOperand,

                        ..Default::default()
                    },
                    Options {
                        update: Update::CopyResult,

                        ..Default::default()
                    },
                    Options {
                        update_expr: Expr::Call,

                        ..Default::default()
                    },
                    Options {
                        update_expr: Expr::Fusion,

                        ..Default::default()
                    },
                ] {
                    let mut ctx = context();

                    let f = fixture(&mut ctx, options);

                    assert!(aux(&ctx, &f).is_empty());
                }
            }

            #[test]

            fn auxiliary_float_nonlinear_and_effect_control_are_only_source_candidates() {
                let mut ctx = context();

                let f = fixture(
                    &mut ctx,
                    Options {
                        element: ElementType::F32,

                        update: Update::Multiply,

                        update_control_effect: true,

                        ..Default::default()
                    },
                );

                assert_eq!(aux(&ctx, &f), [0]);

                assert_eq!(resolved(&ctx, &f, 0, None), None);
            }

            #[test]

            fn auxiliary_arrays_are_source_candidates_without_affine_or_execution_claim() {
                let mut ctx = context();

                let f = host_fixture(&mut ctx, HostPath::Host, false, false);

                let (_, p) = parts(&ctx, &f);

                let parameter = p.body_parameter;

                let buffer_type = shape::subshape(&ctx, parameter.get_type(&ctx), &[2]).unwrap();

                let extraction = block(&ctx, f.body)
                    .unwrap()
                    .deref(&ctx)
                    .iter(&ctx)
                    .find(|&op| {
                        opcode(&ctx, op).ok() == Some(Opcode::GetTupleElement)
                            && attr::<Index>(&ctx, op, "hlo_index").ok() == Some(Index(2))
                    })
                    .unwrap()
                    .deref(&ctx)
                    .get_result(0);

                let literal = Literal::from_bits(&ctx, buffer_type, &[0; 16]).unwrap();

                let constant =
                    ConstantOp::new(&mut ctx, buffer_type, vec![], literal).get_operation();

                let constant = before(&ctx, p.body_root, constant);

                let update = math_before(&mut ctx, p.body_root, Opcode::Add, extraction, constant);

                Operation::replace_operand(p.body_root.defining_op().unwrap(), &ctx, 2, update);

                assert_eq!(aux(&ctx, &f), [0, 1, 2]);

                assert_eq!(resolved(&ctx, &f, 2, None), None);
            }

            #[test]

            fn auxiliary_occurrence_rule_is_not_exact_fallback_unique_value_rule() {
                let mut ctx = context();

                let f = fixture(
                    &mut ctx,
                    Options {
                        init: 1,

                        update: Update::Double,

                        ..Default::default()
                    },
                );

                assert_eq!(facts(&ctx, &f, 128).exact_trip_count, Some(3));

                assert!(aux(&ctx, &f).is_empty());
            }
        }
        mod affine_tests {
            //! Structural affine coefficient and overflow matching.
            use super::*;

            #[test]

            fn affine_lagged_delta_and_final_add_overflow_use_checked_host_arithmetic() {
                for (init, delta, primary, expected) in [
                    (
                        6,
                        3,
                        affine::PrimaryAffine {
                            tuple_index: 0,

                            init: 5,

                            step: 2,
                        },
                        Some(affine::SourceAffine { init: 6, step: 2 }),
                    ),
                    (
                        0,
                        1,
                        affine::PrimaryAffine {
                            tuple_index: 0,

                            init: i64::MAX,

                            step: 0,
                        },
                        None,
                    ),
                ] {
                    let mut ctx = context();

                    let f = fixture(&mut ctx, Options::default());

                    shadow(&mut ctx, &f, init, delta);

                    assert_eq!(resolved(&ctx, &f, 1, Some(primary)), expected);
                }
            }

            #[test]

            fn affine_self_positive_zero_negative_and_constant_first() {
                for step in [-3, 0, 7] {
                    let mut ctx = context();

                    let f = fixture(
                        &mut ctx,
                        Options {
                            init: 5,

                            step,

                            ..Default::default()
                        },
                    );

                    assert_eq!(
                        resolved(&ctx, &f, 0, None),
                        Some(affine::SourceAffine {
                            init: 5,

                            step: step as i64
                        })
                    );
                }

                let mut ctx = context();

                let f = fixture(&mut ctx, Options::default());

                let (_, p) = parts(&ctx, &f);

                let update = operand(&ctx, p.body_root, 0)
                    .unwrap()
                    .defining_op()
                    .unwrap();

                let lhs = update.deref(&ctx).get_operand(0);

                let rhs = update.deref(&ctx).get_operand(1);

                Operation::replace_operand(update, &ctx, 0, rhs);

                Operation::replace_operand(update, &ctx, 1, lhs);

                assert_eq!(
                    resolved(&ctx, &f, 0, None),
                    Some(affine::SourceAffine { init: 0, step: 1 })
                );
            }

            #[test]

            fn affine_subtraction_orientation_and_two_variable_terms() {
                let mut ctx = context();

                let f = fixture(
                    &mut ctx,
                    Options {
                        update: Update::Subtract,

                        step: 3,

                        ..Default::default()
                    },
                );

                assert_eq!(
                    resolved(&ctx, &f, 0, None),
                    Some(affine::SourceAffine { init: 0, step: -3 })
                );

                let (_, p) = parts(&ctx, &f);

                let update = operand(&ctx, p.body_root, 0)
                    .unwrap()
                    .defining_op()
                    .unwrap();

                let iv = update.deref(&ctx).get_operand(0);

                let c = update.deref(&ctx).get_operand(1);

                Operation::replace_operand(update, &ctx, 0, c);

                Operation::replace_operand(update, &ctx, 1, iv);

                assert_eq!(resolved(&ctx, &f, 0, None), None);

                let mut ctx = context();

                let f = fixture(
                    &mut ctx,
                    Options {
                        update: Update::Double,

                        ..Default::default()
                    },
                );

                assert_eq!(resolved(&ctx, &f, 0, None), None);
            }

            #[test]

            fn affine_nested_offsets_and_checked_overflow() {
                for (first, second, expected) in
                    [(2, 3, Some(5)), (i64::MAX, 1, None), (i64::MIN, -1, None)]
                {
                    let mut ctx = context();

                    let f = fixture(&mut ctx, Options::default());

                    let (_, p) = parts(&ctx, &f);

                    let old = operand(&ctx, p.body_root, 0).unwrap();

                    let iv = operand(&ctx, old, 0).unwrap();

                    let c1 = c_before(&mut ctx, p.body_root, first);

                    let c2 = c_before(&mut ctx, p.body_root, second);

                    let constants = math_before(&mut ctx, p.body_root, Opcode::Add, c1, c2);

                    let update = math_before(&mut ctx, p.body_root, Opcode::Add, iv, constants);

                    Operation::replace_operand(p.body_root.defining_op().unwrap(), &ctx, 0, update);

                    assert_eq!(
                        resolved(&ctx, &f, 0, None),
                        expected.map(|step| affine::SourceAffine { init: 0, step })
                    );
                }
            }

            #[test]

            fn affine_requires_direct_constant_init_and_structural_update() {
                for options in [
                    Options {
                        initialization: Init::Copy,

                        ..Default::default()
                    },
                    Options {
                        initialization: Init::Add,

                        ..Default::default()
                    },
                    Options {
                        initialization: Init::Parameter,

                        ..Default::default()
                    },
                    Options {
                        update: Update::CopyResult,

                        ..Default::default()
                    },
                    Options {
                        update: Update::Multiply,

                        ..Default::default()
                    },
                    Options {
                        update_expr: Expr::Fusion,

                        ..Default::default()
                    },
                    Options {
                        update: Update::Constant,

                        ..Default::default()
                    },
                ] {
                    let mut ctx = context();

                    let f = fixture(&mut ctx, options);

                    assert_eq!(resolved(&ctx, &f, 0, None), None);
                }
            }

            #[test]

            fn affine_lagged_helper_requires_explicit_trusted_primary_and_consistent_init() {
                let mut ctx = context();

                let f = fixture(
                    &mut ctx,
                    Options {
                        init: 1,

                        ..Default::default()
                    },
                );

                shadow(&mut ctx, &f, 0, 0);

                assert_eq!(resolved(&ctx, &f, 1, None), None);

                let primary = affine::PrimaryAffine {
                    tuple_index: 0,

                    init: 1,

                    step: 1,
                };

                assert_eq!(
                    resolved(&ctx, &f, 1, Some(primary)),
                    Some(affine::SourceAffine { init: 0, step: 1 })
                );

                assert_eq!(
                    resolved(
                        &ctx,
                        &f,
                        1,
                        Some(affine::PrimaryAffine {
                            tuple_index: 1,

                            ..primary
                        })
                    ),
                    None
                );

                assert_eq!(
                    resolved(
                        &ctx,
                        &f,
                        1,
                        Some(affine::PrimaryAffine { init: 2, ..primary })
                    ),
                    None
                );
            }

            #[test]

            fn affine_lagged_checked_derived_init_overflow_declines() {
                let mut ctx = context();

                let f = fixture(&mut ctx, Options::default());

                shadow(&mut ctx, &f, 0, 0);

                assert_eq!(
                    resolved(
                        &ctx,
                        &f,
                        1,
                        Some(affine::PrimaryAffine {
                            tuple_index: 0,

                            init: i64::MIN,

                            step: 1,
                        })
                    ),
                    None
                );
            }

            #[test]

            fn upstream_lagged_primary_copy_remains_unresolved_in_initial_integration() {
                let mut ctx = context();

                let f = host_fixture(&mut ctx, HostPath::Device, false, true);

                let found = dynamics(&ctx, &f);

                assert_eq!(found[1].tuple_index, 1);

                assert_eq!(found[1].affine, None);

                assert_eq!(
                    resolved(
                        &ctx,
                        &f,
                        1,
                        Some(affine::PrimaryAffine {
                            tuple_index: 0,

                            init: 1,

                            step: 1,
                        })
                    ),
                    Some(affine::SourceAffine { init: 0, step: 1 })
                );
            }

            #[test]

            fn malformed_direct_constant_attribute_propagates() {
                let mut ctx = context();

                let f = fixture(&mut ctx, Options::default());

                f.init
                    .defining_op()
                    .unwrap()
                    .deref_mut(&ctx)
                    .attributes
                    .set(Identifier::new("hlo_literal"), Index(7));

                let (view, parts) = parts(&ctx, &f);

                assert!(affine::resolve(&view, &parts, 0, None).is_err());
            }

            #[test]

            fn affine_self_narrow_wrap_is_metadata_not_typed_iteration_proof() {
                let mut ctx = context();

                let f = fixture(
                    &mut ctx,
                    Options {
                        element: ElementType::S8,

                        init: 127,

                        step: 1,

                        bound: 127,

                        direction: Direction::Le,

                        ..Default::default()
                    },
                );

                assert_eq!(
                    resolved(&ctx, &f, 0, None),
                    Some(affine::SourceAffine { init: 127, step: 1 })
                );
            }

            #[test]

            fn affine_subtraction_offset_overflow_declines() {
                let mut ctx = context();

                let f = fixture(&mut ctx, Options::default());

                let (_, p) = parts(&ctx, &f);

                let old = operand(&ctx, p.body_root, 0).unwrap();

                let iv = operand(&ctx, old, 0).unwrap();

                let minimum = c_before(&mut ctx, p.body_root, i64::MIN);

                let one = c_before(&mut ctx, p.body_root, 1);

                let constants = math_before(&mut ctx, p.body_root, Opcode::Subtract, minimum, one);

                let update = math_before(&mut ctx, p.body_root, Opcode::Add, iv, constants);

                Operation::replace_operand(p.body_root.defining_op().unwrap(), &ctx, 0, update);

                assert_eq!(resolved(&ctx, &f, 0, None), None);
            }
        }
        mod host_offload_tests {
            //! Host-offload dynamic-index source matching.
            use super::*;

            #[test]

            fn upstream_host_gate_both_directions_and_forward_transparent_ops() {
                for path in [
                    HostPath::Host,
                    HostPath::Device,
                    HostPath::Reshape,
                    HostPath::Broadcast,
                    HostPath::Transpose,
                ] {
                    let mut ctx = context();

                    let f = host_fixture(&mut ctx, path, false, false);

                    let found = dynamics(&ctx, &f);

                    assert_eq!(
                        found.iter().map(|v| v.tuple_index).collect::<Vec<_>>(),
                        [0, 1]
                    );

                    assert_eq!(
                        found[1].affine,
                        Some(affine::SourceAffine { init: 5, step: 1 })
                    );
                }
            }

            #[test]

            fn upstream_host_gate_absent_wrong_target_or_copy_barrier_yields_no_indices() {
                for path in [HostPath::None, HostPath::Wrong, HostPath::Copy] {
                    let mut ctx = context();

                    let f = host_fixture(&mut ctx, path, false, false);

                    assert!(dynamics(&ctx, &f).is_empty());
                }
            }

            #[test]

            fn host_foreign_gte_index_is_metadata_not_body_origin_proof() {
                let mut ctx = context();

                let f = host_fixture(&mut ctx, HostPath::Host, true, false);

                let found = dynamics(&ctx, &f);

                assert_eq!(
                    found.iter().map(|v| v.tuple_index).collect::<Vec<_>>(),
                    [0, 3]
                );

                assert_eq!(found[1].affine, None);
            }

            #[test]

            fn metadata_queries_do_not_mutate_ir_or_run_host_calls() {
                let mut ctx = context();

                let f = host_fixture(&mut ctx, HostPath::Host, false, false);

                let before = f.module.disp(&ctx).to_string();

                assert_eq!(aux(&ctx, &f), [0, 1]);

                assert_eq!(dynamics(&ctx, &f).len(), 2);

                assert_eq!(f.module.disp(&ctx).to_string(), before);
            }

            #[test]

            fn host_backward_reshape_is_transparent_but_copy_is_a_barrier() {
                for transparent in [true, false] {
                    let mut ctx = context();

                    let f = host_fixture(&mut ctx, HostPath::Device, false, false);

                    let (_, p) = parts(&ctx, &f);

                    let update = operand(&ctx, p.body_root, 2).unwrap();

                    let moved = operand(&ctx, update, 1).unwrap();

                    let original = operand(&ctx, moved, 0).unwrap();

                    let ty = original.get_type(&ctx);

                    let op = if transparent {
                        ReshapeOp::new(&mut ctx, ty, vec![original]).get_operation()
                    } else {
                        CopyOp::new(&mut ctx, ty, vec![original]).get_operation()
                    };

                    let operand = before(&ctx, moved, op);

                    Operation::replace_operand(moved.defining_op().unwrap(), &ctx, 0, operand);

                    assert_eq!(dynamics(&ctx, &f).is_empty(), !transparent);
                }
            }

            #[test]

            fn host_index_copy_and_arithmetic_union_all_operands() {
                for code in [
                    Opcode::Add,
                    Opcode::Subtract,
                    Opcode::Multiply,
                    Opcode::Divide,
                ] {
                    let mut ctx = context();

                    let f = host_fixture(&mut ctx, HostPath::Host, false, false);

                    let (_, p) = parts(&ctx, &f);

                    let instructions: Vec<_> = block(&ctx, f.body)
                        .unwrap()
                        .deref(&ctx)
                        .iter(&ctx)
                        .collect();

                    let slice = instructions
                        .iter()
                        .copied()
                        .find(|&op| opcode(&ctx, op).ok() == Some(Opcode::DynamicSlice))
                        .unwrap();

                    let anchor = slice.deref(&ctx).get_result(0);

                    let iv = slice.deref(&ctx).get_operand(1);

                    let counter = slice.deref(&ctx).get_operand(2);

                    let ty = iv.get_type(&ctx);

                    let copied = CopyOp::new(&mut ctx, ty, vec![iv]).get_operation();

                    let copied = before(&ctx, anchor, copied);

                    let expr = math_before(&mut ctx, anchor, code, copied, counter);

                    Operation::replace_operand(slice, &ctx, 1, expr);

                    Operation::replace_operand(slice, &ctx, 2, expr);

                    assert_eq!(
                        dynamics(&ctx, &f)
                            .iter()
                            .map(|v| v.tuple_index)
                            .collect::<Vec<_>>(),
                        [0, 1]
                    );

                    assert_eq!(
                        dependency::induction_index(&ModuleView::new(&ctx, f.module).unwrap(), &p)
                            .unwrap(),
                        Some(0)
                    );
                }
            }

            #[test]

            fn host_gate_scans_unrelated_slices_and_sorts_foreign_indices() {
                let mut ctx = context();

                let f = host_fixture(&mut ctx, HostPath::Host, false, false);

                let (_, p) = parts(&ctx, &f);

                let next = operand(&ctx, p.body_root, 0).unwrap();

                let iv = operand(&ctx, next, 0).unwrap();

                let updated = operand(&ctx, p.body_root, 2).unwrap();

                let base = operand(&ctx, updated, 0).unwrap();

                let scalar = iv.get_type(&ctx);

                let tuple_type = shape::tuple(&ctx, vec![scalar; 4]).unwrap();

                let tuple = TupleOp::new(&mut ctx, tuple_type, vec![iv; 4]).get_operation();

                let tuple = before(&ctx, p.body_root, tuple);

                let gte =
                    GetTupleElementOp::new(&mut ctx, scalar, vec![tuple], Index(3)).get_operation();

                let foreign = before(&ctx, p.body_root, gte);

                let tile = shape::array(&ctx, ElementType::F32, vec![1, 1]).unwrap();

                let slice =
                    DynamicSliceOp::new(&mut ctx, tile, vec![base, foreign, iv], Axes(vec![1, 1]))
                        .get_operation();

                before(&ctx, p.body_root, slice);

                assert_eq!(
                    dynamics(&ctx, &f)
                        .iter()
                        .map(|v| v.tuple_index)
                        .collect::<Vec<_>>(),
                    [0, 1, 3]
                );
            }

            #[test]

            fn host_malformed_target_attribute_propagates() {
                let mut ctx = context();

                let f = host_fixture(&mut ctx, HostPath::Host, false, false);

                let call = block(&ctx, f.body)
                    .unwrap()
                    .deref(&ctx)
                    .iter(&ctx)
                    .find(|&op| opcode(&ctx, op).ok() == Some(Opcode::CustomCall))
                    .unwrap();

                call.deref_mut(&ctx)
                    .attributes
                    .set(Identifier::new("hlo_target"), Index(0));

                let (view, p) = parts(&ctx, &f);

                assert!(host_offload::dynamic_variables(&view, &p).is_err());
            }

            #[test]

            fn host_effect_flags_do_not_execute_or_change_structural_discovery() {
                let mut ctx = context();

                let f = host_fixture(&mut ctx, HostPath::Device, false, false);

                let call = block(&ctx, f.body)
                    .unwrap()
                    .deref(&ctx)
                    .iter(&ctx)
                    .find(|&op| opcode(&ctx, op).ok() == Some(Opcode::CustomCall))
                    .unwrap();

                call.deref_mut(&ctx)
                    .attributes
                    .set(Identifier::new("hlo_has_side_effect"), Bools(vec![true]));

                let before = f.module.disp(&ctx).to_string();

                assert_eq!(dynamics(&ctx, &f).len(), 2);

                assert_eq!(f.module.disp(&ctx).to_string(), before);
            }

            #[test]

            fn host_gate_inside_named_call_is_not_a_direct_body_host_gate() {
                let mut ctx = context();

                let f = host_fixture(&mut ctx, HostPath::None, false, false);

                let (_, p) = parts(&ctx, &f);

                let updated = operand(&ctx, p.body_root, 2).unwrap();

                let data = operand(&ctx, updated, 0).unwrap();

                let iv = operand(&ctx, updated, 2).unwrap();

                let counter = operand(&ctx, updated, 3).unwrap();

                let buffer = data.get_type(&ctx);

                let scalar = iv.get_type(&ctx);

                let tile = shape::array(&ctx, ElementType::F32, vec![1, 1]).unwrap();

                let (_, bb, args) = computation(
                    &mut ctx,
                    f.module,
                    "host_nested",
                    vec![buffer, scalar, scalar],
                    buffer,
                );

                let call = CustomCallOp::new(
                    &mut ctx,
                    buffer,
                    vec![args[0]],
                    StringAttr::new("MoveToHost".into()),
                )
                .get_operation();

                let moved = append(&ctx, bb, call);

                let slice = DynamicSliceOp::new(
                    &mut ctx,
                    tile,
                    vec![args[0], args[1], args[2]],
                    Axes(vec![1, 1]),
                )
                .get_operation();

                let slice = append(&ctx, bb, slice);

                let update = DynamicUpdateSliceOp::new(
                    &mut ctx,
                    buffer,
                    vec![moved, slice, args[1], args[2]],
                )
                .get_operation();

                let update = append(&ctx, bb, update);

                finish(&mut ctx, bb, update);

                let call = CallOp::new(
                    &mut ctx,
                    buffer,
                    vec![data, iv, counter],
                    Callees(vec!["host_nested".into()]),
                )
                .get_operation();

                let result = before(&ctx, p.body_root, call);

                Operation::replace_operand(p.body_root.defining_op().unwrap(), &ctx, 2, result);

                assert!(dynamics(&ctx, &f).is_empty());
            }
        }
        use pliron::identifier::Identifier;

        mod proofs_tests {
            //! Checked primary and dynamic recurrence proofs using the metadata fixtures.

            use super::*;

            use crate::analysis::while_loop::proofs;

            use pliron::identifier::Identifier;

            fn primary_proof(ctx: &Context, f: &Fixture) -> Option<proofs::ProvenPrimaryInduction> {
                let facts = facts(ctx, f, 128);

                let (view, parts) = parts(ctx, f);

                proofs::primary(
                    &view,
                    &parts,
                    facts.induction_index,
                    facts.exact_trip_count,
                    facts.source_range,
                )
                .unwrap()
            }

            fn dynamic_proofs(ctx: &Context, f: &Fixture) -> Vec<proofs::ProvenDynamicVariable> {
                let facts = facts(ctx, f, 128);

                let primary = primary_proof(ctx, f);

                let (view, parts) = parts(ctx, f);

                proofs::dynamics(
                    &view,
                    &parts,
                    facts.exact_trip_count,
                    &facts.source_dynamic_variables,
                    primary,
                )
                .unwrap()
            }

            #[test]

            fn primary_direct_positive_lt_le_typed_finite_agrees_with_exact() {
                for (init, bound, step, direction, count, final_value) in [
                    (0, 5, 1, Direction::Lt, 5, 5),
                    (2, 10, 3, Direction::Le, 3, 11),
                    (1, 10_000, 2, Direction::Lt, 5_000, 10_001),
                ] {
                    let mut ctx = context();

                    let f = fixture(
                        &mut ctx,
                        Options {
                            init,

                            bound,

                            step,

                            direction,

                            ..Default::default()
                        },
                    );

                    let proof = primary_proof(&ctx, &f).unwrap();

                    assert_eq!(proof.tuple_index(), 0);

                    assert_eq!(proof.init(), init as i64);

                    assert_eq!(proof.step(), step as i64);

                    assert_eq!(proof.trip_count(), count);

                    assert_eq!(proof.final_value().bits(), final_value);
                }
            }

            #[test]

            fn primary_quirky_source_orientations_remain_metadata_only() {
                for (direction, condition) in [
                    (Direction::Gt, Condition::Direct),
                    (Direction::Ge, Condition::Direct),
                    (Direction::Lt, Condition::Swapped),
                    (Direction::Le, Condition::Swapped),
                ] {
                    let mut ctx = context();

                    let f = fixture(
                        &mut ctx,
                        Options {
                            direction,

                            condition,

                            ..Default::default()
                        },
                    );

                    assert!(facts(&ctx, &f, 128).source_range.is_some());

                    assert!(primary_proof(&ctx, &f).is_none());
                }
            }

            #[test]

            fn primary_rejects_negative_reverse_zero_count_copy_and_fallback_wrap() {
                for options in [
                    Options {
                        init: -3,

                        ..Default::default()
                    },
                    Options {
                        init: 9,

                        bound: 2,

                        ..Default::default()
                    },
                    Options {
                        init: 5,

                        bound: 5,

                        ..Default::default()
                    },
                    Options {
                        update: Update::CopyResult,

                        ..Default::default()
                    },
                    Options {
                        update: Update::CopyOperand,

                        ..Default::default()
                    },
                    Options {
                        element: ElementType::S32,

                        init: 1_073_741_824,

                        step: 1_342_177_280,

                        bound: 1_879_048_192,

                        ..Default::default()
                    },
                ] {
                    let mut ctx = context();

                    let f = fixture(&mut ctx, options);

                    assert!(primary_proof(&ctx, &f).is_none());
                }
            }

            #[test]

            fn primary_requires_exact_agreement_without_rerunning_evaluator() {
                let mut ctx = context();

                let f = fixture(&mut ctx, Options::default());

                let facts = facts(&ctx, &f, 128);

                let (view, parts) = parts(&ctx, &f);

                assert!(
                    proofs::primary(&view, &parts, Some(0), Some(4), facts.source_range)
                        .unwrap()
                        .is_none()
                );

                assert!(
                    proofs::primary(&view, &parts, Some(0), None, facts.source_range)
                        .unwrap()
                        .is_none()
                );
            }

            #[test]

            fn proved_primary_typed_narrow_post_loop_overflow_declines() {
                let mut ctx = context();

                let f = fixture(
                    &mut ctx,
                    Options {
                        element: ElementType::S8,

                        init: 120,

                        bound: 127,

                        step: 7,

                        direction: Direction::Le,

                        ..Default::default()
                    },
                );

                assert!(primary_proof(&ctx, &f).is_none());
            }

            #[test]

            fn self_dynamic_and_lagged_primary_copy_use_genuine_origins() {
                for lagged in [false, true] {
                    let mut ctx = context();

                    let f = host_fixture(&mut ctx, HostPath::Device, false, lagged);

                    assert!(primary_proof(&ctx, &f).is_some());

                    let proofs = dynamic_proofs(&ctx, &f);

                    assert_eq!(
                        proofs.iter().map(|p| p.tuple_index()).collect::<Vec<_>>(),
                        [0, 1]
                    );

                    assert_eq!(proofs[1].init(), if lagged { 0 } else { 5 });

                    assert_eq!(proofs[1].step(), 1);

                    if lagged {
                        assert_eq!(
                            facts(&ctx, &f, 128).source_dynamic_variables[1].affine,
                            None
                        );
                    }
                }
            }

            fn set_counter_step(ctx: &mut Context, f: &Fixture, step: i64) {
                let (_, p) = parts(ctx, f);

                let update = operand(ctx, p.body_root, 1).unwrap();

                let replacement = c_before(ctx, update, step);

                Operation::replace_operand(update.defining_op().unwrap(), ctx, 1, replacement);
            }

            #[test]

            fn self_dynamic_negative_and_zero_steps_fit_the_complete_exact_count() {
                for step in [-2, 0, 3] {
                    let mut ctx = context();

                    let f = host_fixture(&mut ctx, HostPath::Device, false, false);

                    set_counter_step(&mut ctx, &f, step);

                    let proofs = dynamic_proofs(&ctx, &f);

                    assert_eq!(proofs[1].init(), 5);

                    assert_eq!(proofs[1].step(), step);
                }
            }

            #[test]

            fn self_dynamic_can_use_exact_fallback_without_a_primary_proof() {
                let mut ctx = context();

                let f = host_fixture(&mut ctx, HostPath::Device, false, false);

                let (_, p) = parts(&ctx, &f);

                let update = operand(&ctx, p.body_root, 0).unwrap();

                let iv = operand(&ctx, update, 0).unwrap();

                let one = operand(&ctx, update, 1).unwrap();

                let sub = math_before(&mut ctx, p.body_root, Opcode::Subtract, iv, one);

                Operation::replace_operand(p.body_root.defining_op().unwrap(), &ctx, 0, sub);

                // A direct initially false condition proves zero without executing this update.

                let cond = p.condition_root.defining_op().unwrap();

                cond.deref_mut(&ctx)
                    .attributes
                    .set(Identifier::new("hlo_direction"), Direction::Gt);

                assert_eq!(facts(&ctx, &f, 128).exact_trip_count, Some(0));

                assert!(primary_proof(&ctx, &f).is_none());

                assert_eq!(
                    dynamic_proofs(&ctx, &f)
                        .iter()
                        .map(|p| p.tuple_index())
                        .collect::<Vec<_>>(),
                    [0, 1]
                );
            }

            #[test]

            fn dynamic_state_overflow_and_unknown_count_decline_without_erasing_sources() {
                let mut ctx = context();

                let f = host_fixture(&mut ctx, HostPath::Device, false, false);

                set_counter_step(&mut ctx, &f, i64::MAX);

                let old = facts(&ctx, &f, 128);

                assert_eq!(
                    dynamic_proofs(&ctx, &f)
                        .iter()
                        .map(|p| p.tuple_index())
                        .collect::<Vec<_>>(),
                    [0]
                );

                let (view, p) = parts(&ctx, &f);

                assert!(
                    proofs::dynamics(&view, &p, None, &old.source_dynamic_variables, None)
                        .unwrap()
                        .is_empty()
                );

                assert_eq!(facts(&ctx, &f, 128), old);
            }

            #[test]

            fn foreign_index_gte_is_not_promoted_to_a_body_origin_proof() {
                let mut ctx = context();

                let f = host_fixture(&mut ctx, HostPath::Host, true, false);

                assert_eq!(
                    dynamic_proofs(&ctx, &f)
                        .iter()
                        .map(|p| p.tuple_index())
                        .collect::<Vec<_>>(),
                    [0]
                );
            }

            #[test]

            fn lagged_coefficients_require_the_new_primary_not_source_range() {
                let mut ctx = context();

                let f = host_fixture(&mut ctx, HostPath::Device, false, true);

                let facts = facts(&ctx, &f, 128);

                let (view, p) = parts(&ctx, &f);

                let proofs = proofs::dynamics(
                    &view,
                    &p,
                    facts.exact_trip_count,
                    &facts.source_dynamic_variables,
                    None,
                )
                .unwrap();

                assert_eq!(
                    proofs.iter().map(|p| p.tuple_index()).collect::<Vec<_>>(),
                    [0]
                );
            }

            #[test]

            fn mismatched_lagged_initial_value_is_not_proved() {
                let mut ctx = context();

                let f = host_fixture(&mut ctx, HostPath::Device, false, true);

                let (_, p) = parts(&ctx, &f);

                let wrong = c_before(&mut ctx, p.input, 7);

                Operation::replace_operand(p.input.defining_op().unwrap(), &ctx, 1, wrong);

                assert_eq!(
                    dynamic_proofs(&ctx, &f)
                        .iter()
                        .map(|p| p.tuple_index())
                        .collect::<Vec<_>>(),
                    [0]
                );
            }

            #[test]

            fn same_width_modular_wrap_cancellation_is_a_positive_dynamic_proof() {
                let mut ctx = context();

                let f = host_fixture(&mut ctx, HostPath::Device, false, false);

                let (_, p) = parts(&ctx, &f);

                let old = operand(&ctx, p.body_root, 1).unwrap();

                let counter = operand(&ctx, old, 0).unwrap();

                let max = c_before(&mut ctx, p.body_root, i64::MAX);

                let add = math_before(&mut ctx, p.body_root, Opcode::Add, counter, max);

                let sub = math_before(&mut ctx, p.body_root, Opcode::Subtract, add, max);

                Operation::replace_operand(p.body_root.defining_op().unwrap(), &ctx, 1, sub);

                let values = dynamic_proofs(&ctx, &f);

                assert_eq!(values[1].init(), 5);

                assert_eq!(values[1].step(), 0);
            }

            #[test]

            fn non_add_sub_dynamic_patterns_decline() {
                for code in [Opcode::Multiply, Opcode::Divide] {
                    let mut ctx = context();

                    let f = host_fixture(&mut ctx, HostPath::Device, false, false);

                    let (_, p) = parts(&ctx, &f);

                    let old = operand(&ctx, p.body_root, 1).unwrap();

                    let counter = operand(&ctx, old, 0).unwrap();

                    let one = c_before(&mut ctx, p.body_root, 1);

                    let update = math_before(&mut ctx, p.body_root, code, counter, one);

                    Operation::replace_operand(p.body_root.defining_op().unwrap(), &ctx, 1, update);

                    assert_eq!(
                        dynamic_proofs(&ctx, &f)
                            .iter()
                            .map(|p| p.tuple_index())
                            .collect::<Vec<_>>(),
                        [0]
                    );
                }
            }

            #[test]

            fn proof_queries_leave_source_fields_declines_and_ir_unchanged() {
                let mut ctx = context();

                let f = host_fixture(&mut ctx, HostPath::Host, false, false);

                let before = f.module.disp(&ctx).to_string();

                let old = facts(&ctx, &f, 128);

                primary_proof(&ctx, &f);

                dynamic_proofs(&ctx, &f);

                assert_eq!(facts(&ctx, &f, 128), old);

                assert_eq!(f.module.disp(&ctx).to_string(), before);
            }

            #[test]

            fn executed_body_effects_reject_proofs_without_changing_source_metadata() {
                let mut ctx = context();

                let f = host_fixture(&mut ctx, HostPath::Host, false, false);

                let call = block(&ctx, f.body)
                    .unwrap()
                    .deref(&ctx)
                    .iter(&ctx)
                    .find(|&op| opcode(&ctx, op).ok() == Some(Opcode::CustomCall))
                    .unwrap();

                call.deref_mut(&ctx)
                    .attributes
                    .set(Identifier::new("hlo_has_side_effect"), Bools(vec![true]));

                assert!(facts(&ctx, &f, 128).source_range.is_some());

                assert!(primary_proof(&ctx, &f).is_none());

                assert!(dynamic_proofs(&ctx, &f).is_empty());
            }

            #[test]

            fn control_dependencies_are_not_bypassed_by_a_primary_proof() {
                let mut ctx = context();

                let f = fixture(&mut ctx, Options::default());

                let (_, p) = parts(&ctx, &f);

                let update = operand(&ctx, p.body_root, 0).unwrap();

                let step = operand(&ctx, update, 1).unwrap();

                let name =
                    attr::<InstructionName>(&ctx, step.defining_op().unwrap(), "hlo_name").unwrap();

                update
                    .defining_op()
                    .unwrap()
                    .deref_mut(&ctx)
                    .attributes
                    .set(
                        Identifier::new("hlo_control_predecessors"),
                        ControlPredecessors(vec![name.0]),
                    );

                assert!(primary_proof(&ctx, &f).is_none());
            }

            #[test]

            fn qualified_scalar_s8_wrap_cancellation_agrees_with_ring_proof() {
                use crate::scalar::semantics::{Scalar, binary};

                let i = Scalar::from_bits(ElementType::S8, 5).unwrap();

                let c = Scalar::from_bits(ElementType::S8, 127).unwrap();

                let wrapped = binary(Opcode::Add, i, c).unwrap();

                assert_eq!(wrapped.bits(), 132);

                assert_eq!(binary(Opcode::Subtract, wrapped, c).unwrap(), i);
            }

            #[test]

            fn foreign_gte_with_an_existing_tuple_index_is_still_not_a_genuine_origin() {
                let mut ctx = context();

                let f = host_fixture(&mut ctx, HostPath::Host, true, false);

                let (_, p) = parts(&ctx, &f);

                let foreign = block(&ctx, f.body)
                    .unwrap()
                    .deref(&ctx)
                    .iter(&ctx)
                    .find(|&op| {
                        opcode(&ctx, op).ok() == Some(Opcode::GetTupleElement)
                            && attr::<Index>(&ctx, op, "hlo_index").is_ok_and(|index| index.0 == 3)
                    })
                    .unwrap();

                foreign
                    .deref_mut(&ctx)
                    .attributes
                    .set(Identifier::new("hlo_index"), Index(1));

                assert_ne!(foreign.deref(&ctx).get_operand(0), p.body_parameter);

                assert!(
                    facts(&ctx, &f, 128)
                        .source_dynamic_variables
                        .iter()
                        .any(|candidate| candidate.tuple_index == 1)
                );

                assert_eq!(
                    dynamic_proofs(&ctx, &f)
                        .iter()
                        .map(|p| p.tuple_index())
                        .collect::<Vec<_>>(),
                    [0]
                );
            }

            #[test]

            fn copy_and_narrow_convert_roundtrips_do_not_prove_dynamic_recurrences() {
                use crate::ops::elementwise::ConvertOp;

                for copy in [false, true] {
                    let mut ctx = context();

                    let f = host_fixture(&mut ctx, HostPath::Device, false, false);

                    let (_, p) = parts(&ctx, &f);

                    let old = operand(&ctx, p.body_root, 1).unwrap();

                    let counter = operand(&ctx, old, 0).unwrap();

                    let wide = counter.get_type(&ctx);

                    let replacement = if copy {
                        let op = CopyOp::new(&mut ctx, wide, vec![counter]).get_operation();

                        before(&ctx, p.body_root, op)
                    } else {
                        let narrow = shape::scalar(&ctx, ElementType::S8).unwrap();

                        let op = ConvertOp::new(&mut ctx, narrow, vec![counter]).get_operation();

                        let narrowed = before(&ctx, p.body_root, op);

                        let op = ConvertOp::new(&mut ctx, wide, vec![narrowed]).get_operation();

                        before(&ctx, p.body_root, op)
                    };

                    Operation::replace_operand(
                        p.body_root.defining_op().unwrap(),
                        &ctx,
                        1,
                        replacement,
                    );

                    assert_eq!(
                        dynamic_proofs(&ctx, &f)
                            .iter()
                            .map(|p| p.tuple_index())
                            .collect::<Vec<_>>(),
                        [0]
                    );
                }
            }

            #[test]

            fn nonzero_exact_fallback_proves_self_recurrence_without_a_primary() {
                let mut ctx = context();

                let f = host_fixture(&mut ctx, HostPath::Device, false, false);

                let (_, p) = parts(&ctx, &f);

                let old = operand(&ctx, p.body_root, 0).unwrap();

                let iv = operand(&ctx, old, 0).unwrap();

                let one = operand(&ctx, old, 1).unwrap();

                let update = math_before(&mut ctx, p.body_root, Opcode::Subtract, iv, one);

                Operation::replace_operand(p.body_root.defining_op().unwrap(), &ctx, 0, update);

                let bound = c_before(&mut ctx, p.condition_root, 0);

                let compare = p.condition_root.defining_op().unwrap();

                Operation::replace_operand(compare, &ctx, 1, bound);

                compare
                    .deref_mut(&ctx)
                    .attributes
                    .set(Identifier::new("hlo_direction"), Direction::Gt);

                assert_eq!(facts(&ctx, &f, 128).exact_trip_count, Some(1));

                assert!(primary_proof(&ctx, &f).is_none());

                let values = dynamic_proofs(&ctx, &f);

                assert_eq!(
                    values.iter().map(|p| p.tuple_index()).collect::<Vec<_>>(),
                    [0, 1]
                );

                assert_eq!(values[0].step(), -1);

                assert_eq!(values[1].init(), 5);

                assert_eq!(values[1].step(), 1);
            }
        }
        use super::*;

        use crate::{
            analysis::while_loop::{affine, auxiliary, host_offload},
            attributes::Axes,
            ops::{
                elementwise::DivideOp,
                movement::{
                    BroadcastOp, CopyOp, DynamicSliceOp, DynamicUpdateSliceOp, ReshapeOp,
                    TransposeOp,
                },
            },
        };

        fn parts<'a>(ctx: &'a Context, f: &Fixture) -> (ModuleView<'a>, Parts) {
            let view = ModuleView::new(ctx, f.module).unwrap();

            let parts = Parts::new(&view, f.while_value.defining_op().unwrap()).unwrap();

            (view, parts)
        }

        fn aux(ctx: &Context, f: &Fixture) -> Vec<usize> {
            verification::verify(ctx, f.module).unwrap();

            let (view, parts) = parts(ctx, f);

            auxiliary::candidates(&view, &parts)
                .unwrap()
                .iter()
                .map(|item| item.tuple_index)
                .collect()
        }

        fn resolved(
            ctx: &Context,

            f: &Fixture,

            index: usize,

            primary: Option<affine::PrimaryAffine>,
        ) -> Option<affine::SourceAffine> {
            verification::verify(ctx, f.module).unwrap();

            let (view, parts) = parts(ctx, f);

            affine::resolve(&view, &parts, index, primary).unwrap()
        }

        fn before(ctx: &Context, anchor: Value, op: Ptr<Operation>) -> Value {
            static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

            let index = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

            op.deref_mut(ctx).attributes.set(
                Identifier::new("hlo_name"),
                InstructionName(format!("metadata_{index}")),
            );

            op.insert_before(ctx, anchor.defining_op().unwrap());

            op.deref(ctx).get_result(0)
        }

        fn c_before(ctx: &mut Context, anchor: Value, value: i64) -> Value {
            let literal =
                literal_semantics::integer_scalar(ctx, ElementType::S64, i128::from(value))
                    .unwrap();

            let op = ConstantOp::new(ctx, literal.shape, vec![], literal).get_operation();

            before(ctx, anchor, op)
        }

        fn math_before(
            ctx: &mut Context,
            anchor: Value,
            code: Opcode,
            lhs: Value,
            rhs: Value,
        ) -> Value {
            let ty = lhs.get_type(ctx);

            let args = vec![lhs, rhs];

            let op = match code {
                Opcode::Add => AddOp::new(ctx, ty, args).get_operation(),

                Opcode::Subtract => SubtractOp::new(ctx, ty, args).get_operation(),

                Opcode::Multiply => MultiplyOp::new(ctx, ty, args).get_operation(),

                Opcode::Divide => DivideOp::new(ctx, ty, args).get_operation(),

                _ => panic!("invalid metadata fixture operation"),
            };

            before(ctx, anchor, op)
        }

        fn update_chain(ctx: &mut Context, f: &Fixture, codes: &[Opcode]) {
            let (_, p) = parts(ctx, f);

            let original = operand(ctx, p.body_root, 0).unwrap();

            let original_op = original.defining_op().unwrap();

            let iv = original_op.deref(ctx).get_operand(0);

            let mut value = iv;

            for &code in codes {
                let c = c_before(ctx, p.body_root, 42);

                value = math_before(ctx, p.body_root, code, value, c);
            }

            Operation::replace_operand(p.body_root.defining_op().unwrap(), ctx, 0, value);
        }

        fn shadow(ctx: &mut Context, f: &Fixture, initial: i64, delta: i64) {
            let (_, p) = parts(ctx, f);

            let initial_value = c_before(ctx, p.input, initial);

            Operation::replace_operand(p.input.defining_op().unwrap(), ctx, 1, initial_value);

            let update = operand(ctx, p.body_root, 0).unwrap();

            let iv = operand(ctx, update, 0).unwrap();

            let delta = c_before(ctx, p.body_root, delta);

            let shadow = math_before(ctx, p.body_root, Opcode::Add, iv, delta);

            Operation::replace_operand(p.body_root.defining_op().unwrap(), ctx, 1, shadow);
        }

        #[derive(Clone, Copy)]

        enum HostPath {
            None,

            Host,

            Device,

            Reshape,

            Broadcast,

            Transpose,

            Copy,

            Wrong,
        }

        fn host_fixture(
            ctx: &mut Context,
            path: HostPath,
            foreign: bool,
            shadow_primary: bool,
        ) -> Fixture {
            let module = ModuleOp::new(ctx, Identifier::new("host_metadata")).get_operation();

            crate::ir::set_entry(ctx, module, Identifier::new("main")).unwrap();

            let scalar = shape::scalar(ctx, ElementType::S64).unwrap();

            let pred = shape::scalar(ctx, ElementType::Pred).unwrap();

            let buffer = shape::array(ctx, ElementType::F32, vec![4, 4]).unwrap();

            let tile = shape::array(ctx, ElementType::F32, vec![1, 1]).unwrap();

            let state = shape::tuple(ctx, vec![scalar, scalar, buffer]).unwrap();

            let (body, bb, args) = computation(ctx, module, "body", vec![state], state);

            let iv = gte(ctx, bb, args[0], 0);

            let counter = gte(ctx, bb, args[0], 1);

            let data = gte(ctx, bb, args[0], 2);

            let one = constant(ctx, bb, ElementType::S64, 1);

            let next = binary(ctx, bb, Opcode::Add, iv, one, Direction::Lt);

            let counter_next = if shadow_primary {
                iv
            } else {
                binary(ctx, bb, Opcode::Add, counter, one, Direction::Lt)
            };

            let index = if foreign {
                let t = tuple(ctx, bb, vec![iv, counter, iv, counter]);

                gte(ctx, bb, t, 3)
            } else {
                counter
            };

            let slice = DynamicSliceOp::new(ctx, tile, vec![data, iv, index], Axes(vec![1, 1]))
                .get_operation();

            let sliced = append(ctx, bb, slice);

            let target = match path {
                HostPath::Device => "MoveToDevice",

                HostPath::Wrong => "NotMoveToHost",

                _ => "MoveToHost",
            };

            let moved = if matches!(path, HostPath::None) {
                data
            } else {
                let input = if matches!(path, HostPath::Device) {
                    sliced
                } else {
                    data
                };

                let ty = input.get_type(ctx);

                let op = CustomCallOp::new(ctx, ty, vec![input], StringAttr::new(target.into()))
                    .get_operation();

                append(ctx, bb, op)
            };

            let moved = match path {
                HostPath::Reshape => {
                    let op = ReshapeOp::new(ctx, buffer, vec![moved]).get_operation();

                    append(ctx, bb, op)
                }

                HostPath::Broadcast => {
                    let op = BroadcastOp::new(ctx, buffer, vec![moved], Axes(vec![0, 1]))
                        .get_operation();

                    append(ctx, bb, op)
                }

                HostPath::Transpose => {
                    let op = TransposeOp::new(ctx, buffer, vec![moved], Axes(vec![1, 0]))
                        .get_operation();

                    append(ctx, bb, op)
                }

                HostPath::Copy => copy(ctx, bb, moved),

                _ => moved,
            };

            let (base, update) = if matches!(path, HostPath::Device) {
                (data, moved)
            } else {
                (moved, sliced)
            };

            let op = DynamicUpdateSliceOp::new(ctx, buffer, vec![base, update, iv, index])
                .get_operation();

            let updated = append(ctx, bb, op);

            let root = tuple(ctx, bb, vec![next, counter_next, updated]);

            finish(ctx, bb, root);

            let (condition, bb, args) = computation(ctx, module, "condition", vec![state], pred);

            let i = gte(ctx, bb, args[0], 0);

            let limit = constant(ctx, bb, ElementType::S64, 10);

            let cmp = binary(ctx, bb, Opcode::Compare, i, limit, Direction::Lt);

            finish(ctx, bb, cmp);

            let (_, bb, args) = computation(ctx, module, "main", vec![buffer], state);

            let init = constant(ctx, bb, ElementType::S64, 1);

            let counter_init = constant(
                ctx,
                bb,
                ElementType::S64,
                if shadow_primary { 0 } else { 5 },
            );

            let input = tuple(ctx, bb, vec![init, counter_init, args[0]]);

            let op = WhileOp::new(
                ctx,
                state,
                vec![input],
                Callees(vec!["body".into(), "condition".into()]),
            )
            .get_operation();

            let while_value = append(ctx, bb, op);

            finish(ctx, bb, while_value);

            verification::verify(ctx, module).unwrap();

            Fixture {
                module,

                while_value,

                init,

                initial_parameter: args[0],

                body,

                condition,
            }
        }

        fn dynamics(ctx: &Context, f: &Fixture) -> Vec<host_offload::SourceDynamicVariable> {
            verification::verify(ctx, f.module).unwrap();

            let (view, parts) = parts(ctx, f);

            host_offload::dynamic_variables(&view, &parts).unwrap()
        }
    }
    mod range_tests {
        //! Source loop-range matching and endpoint arithmetic using the analysis fixtures.
        use super::*;
        use crate::analysis::while_loop::range::SourceLoopRange;
        use pliron::identifier::Identifier;

        fn matched(
            ctx: &Context,
            fixture: &Fixture,
        ) -> (Option<SourceLoopRange>, Vec<EvaluationDecline>) {
            verification::verify(ctx, fixture.module).unwrap();
            let view = ModuleView::new(ctx, fixture.module).unwrap();
            let parts = Parts::new(&view, fixture.while_value.defining_op().unwrap()).unwrap();
            let index = dependency::induction_index(&view, &parts).unwrap();
            let mut analysis = Analysis {
                view,
                evaluator: HloEvaluator::new(ctx, fixture.module)
                    .unwrap()
                    .with_max_loop_iterations(0),
                declines: Vec::new(),
            };
            let range = index.and_then(|index| analysis.source_range(&parts, index).unwrap());
            (range, analysis.declines)
        }

        fn signed(value: Scalar) -> i64 {
            let width = value.element().bit_width();
            let bits = value.bits() as u64;
            if value.element().is_signed() {
                ((bits << (64 - width)) as i64) >> (64 - width)
            } else {
                bits as i64
            }
        }

        fn assert_range(ctx: &Context, f: &Fixture, start: i64, end: i64, step: i64) {
            let (range, declines) = matched(ctx, f);
            let range = range.unwrap();
            assert!(declines.is_empty());
            assert_eq!(signed(range.start), start);
            assert_eq!(signed(range.inclusive_end), end);
            assert_eq!(range.step.element(), ElementType::S64);
            assert_eq!(signed(range.step), step);
        }

        #[test]
        fn upstream_exact_bound_trivial_range_eight_cases() {
            for (direction, bound, step, end) in [
                (Direction::Lt, 42, 1, 41),
                (Direction::Lt, 42, 2, 40),
                (Direction::Lt, 42, 5, 40),
                (Direction::Lt, 40, 5, 35),
                (Direction::Le, 42, 1, 42),
                (Direction::Le, 42, 2, 42),
                (Direction::Le, 42, 5, 40),
                (Direction::Le, 40, 5, 40),
            ] {
                let mut ctx = context();
                let f = fixture(
                    &mut ctx,
                    Options {
                        bound,
                        step,
                        direction,
                        ..Default::default()
                    },
                );
                assert_range(&ctx, &f, 0, end, step as i64);
            }
        }

        #[test]
        fn upstream_trivial_loop_induction_step_admission_and_declines() {
            for step in [1, 7] {
                let mut ctx = context();
                let f = fixture(
                    &mut ctx,
                    Options {
                        step,
                        bound: 42,
                        ..Default::default()
                    },
                );
                let range = matched(&ctx, &f).0.unwrap();
                assert_eq!(signed(range.step), step as i64);
            }
            for options in [
                Options {
                    step: 0,
                    ..Default::default()
                },
                Options {
                    step: -2,
                    ..Default::default()
                },
                Options {
                    step: 2,
                    update: Update::Multiply,
                    ..Default::default()
                },
                Options {
                    element: ElementType::F32,
                    ..Default::default()
                },
                Options {
                    update: Update::Double,
                    ..Default::default()
                },
            ] {
                let mut ctx = context();
                let f = fixture(&mut ctx, options);
                assert!(matched(&ctx, &f).0.is_none());
            }
        }

        #[test]
        fn upstream_simple_arithmetic_initialization_has_inclusive_range() {
            let mut ctx = context();
            let f = fixture(
                &mut ctx,
                Options {
                    init: 1,
                    bound: 10,
                    direction: Direction::Le,
                    initialization: Init::Add,
                    ..Default::default()
                },
            );
            let add = f.init.defining_op().unwrap();
            let one = add.deref(&ctx).get_operand(0);
            Operation::replace_operand(add, &ctx, 1, one);
            assert_range(&ctx, &f, 2, 10, 1);
        }

        #[test]
        fn upstream_s32_and_u32_overflow_decline_ranges_without_erasing_exact_seven() {
            for (element, init, step, bound) in [
                (
                    ElementType::S32,
                    1_073_741_824,
                    1_342_177_280,
                    1_879_048_192,
                ),
                (
                    ElementType::U32,
                    2_147_483_648,
                    2_147_483_648,
                    3_221_225_472,
                ),
            ] {
                let mut ctx = context();
                let f = fixture(
                    &mut ctx,
                    Options {
                        element,
                        init,
                        step,
                        bound,
                        ..Default::default()
                    },
                );
                assert!(matched(&ctx, &f).0.is_none());
                assert_eq!(
                    facts(&ctx, &f, 128).exact_trip_count,
                    (element == ElementType::S32).then_some(7)
                );
            }
        }

        #[test]
        fn negative_alignment_matches_truncating_remainder_and_preserves_reverse_endpoints() {
            for (init, bound, step, direction, end) in [
                (-7, 7, 5, Direction::Lt, 3),
                (-7, -1, 5, Direction::Le, -2),
                (3, -1, 5, Direction::Le, -2),
                (8, 2, 3, Direction::Lt, -1),
            ] {
                let mut ctx = context();
                let f = fixture(
                    &mut ctx,
                    Options {
                        init,
                        bound,
                        step,
                        direction,
                        ..Default::default()
                    },
                );
                assert_range(&ctx, &f, init as i64, end, step as i64);
            }
        }

        #[test]
        fn source_orientation_quirks_are_metadata_not_execution_counts() {
            for direction in [Direction::Lt, Direction::Le, Direction::Gt, Direction::Ge] {
                for condition in [Condition::Direct, Condition::Swapped] {
                    let mut ctx = context();
                    let f = fixture(
                        &mut ctx,
                        Options {
                            direction,
                            condition,
                            ..Default::default()
                        },
                    );
                    let end = if matches!(direction, Direction::Lt | Direction::Gt) {
                        4
                    } else {
                        5
                    };
                    assert_range(&ctx, &f, 0, end, 1);
                    let before = facts(&ctx, &f, 0);
                    matched(&ctx, &f);
                    assert_eq!(facts(&ctx, &f, 0), before);
                }
            }
        }

        #[test]
        fn source_endpoint_formats_and_unsigned_negative_end_encoding_are_preserved() {
            for element in [
                ElementType::S8,
                ElementType::U8,
                ElementType::S32,
                ElementType::U64,
            ] {
                let mut ctx = context();
                let f = fixture(
                    &mut ctx,
                    Options {
                        element,
                        bound: 0,
                        ..Default::default()
                    },
                );
                let range = matched(&ctx, &f).0.unwrap();
                assert_eq!(range.start.element(), element);
                assert_eq!(range.inclusive_end.element(), element);
                assert_eq!(
                    range.inclusive_end.bits(),
                    (1u128 << element.bit_width()) - 1
                );
                assert_eq!(range.step.element(), ElementType::S64);
            }
        }

        #[test]
        fn checked_bound_alignment_and_post_loop_overflow_decline() {
            for options in [
                Options {
                    init: i64::MIN as i128,
                    bound: i64::MIN as i128,
                    ..Default::default()
                },
                Options {
                    init: 0,
                    bound: i64::MIN as i128,
                    step: 3,
                    direction: Direction::Le,
                    ..Default::default()
                },
                Options {
                    bound: i64::MAX as i128,
                    direction: Direction::Le,
                    ..Default::default()
                },
                Options {
                    element: ElementType::S8,
                    bound: 127,
                    direction: Direction::Le,
                    ..Default::default()
                },
                Options {
                    element: ElementType::U8,
                    bound: 255,
                    direction: Direction::Le,
                    ..Default::default()
                },
                Options {
                    element: ElementType::U64,
                    init: i64::MAX as i128 + 1,
                    ..Default::default()
                },
            ] {
                let mut ctx = context();
                let f = fixture(&mut ctx, options);
                assert!(matched(&ctx, &f).0.is_none());
            }
        }

        #[test]
        fn copies_and_shallow_init_keep_range_stricter_than_existing_exact_proof() {
            for initialization in [Init::Copy, Init::TupleProjection, Init::Add] {
                let mut ctx = context();
                let f = fixture(
                    &mut ctx,
                    Options {
                        initialization,
                        ..Default::default()
                    },
                );
                assert_range(&ctx, &f, 0, 4, 1);
            }
            for update in [Update::CopyOperand, Update::CopyResult] {
                let mut ctx = context();
                let f = fixture(
                    &mut ctx,
                    Options {
                        update,
                        ..Default::default()
                    },
                );
                assert_eq!(facts(&ctx, &f, 128).exact_trip_count, Some(5));
                assert!(matched(&ctx, &f).0.is_none());
            }
            let mut ctx = context();
            let f = fixture(
                &mut ctx,
                Options {
                    initialization: Init::NestedAdd,
                    ..Default::default()
                },
            );
            let (range, declines) = matched(&ctx, &f);
            assert!(range.is_none());
            assert!(matches!(
                declines.as_slice(),
                [EvaluationDecline {
                    stage: EvaluationStage::InitialValue,
                    reason: DeclineReason::ShallowDependency,
                }]
            ));
        }

        #[test]
        fn index_provenance_does_not_expand_direct_range_patterns() {
            for form in [
                Expr::Call,
                Expr::Fusion,
                Expr::NestedFusion,
                Expr::MultiOutputFusion,
            ] {
                for in_update in [false, true] {
                    let mut ctx = context();
                    let f = fixture(
                        &mut ctx,
                        Options {
                            update_expr: if in_update { form } else { Expr::Direct },
                            condition_expr: if in_update { Expr::Direct } else { form },
                            ..Default::default()
                        },
                    );
                    assert_eq!(facts(&ctx, &f, 128).induction_index, Some(0));
                    assert!(matched(&ctx, &f).0.is_none());
                }
            }
        }

        #[test]
        fn duplicate_indices_and_nonconstant_bounds_are_not_range_inputs() {
            for options in [
                Options {
                    duplicate_body_gte: true,
                    ..Default::default()
                },
                Options {
                    duplicate_condition_gte: true,
                    ..Default::default()
                },
                Options {
                    condition: Condition::OtherSlot,
                    ..Default::default()
                },
                Options {
                    condition: Condition::ComputedBound,
                    ..Default::default()
                },
                Options {
                    update: Update::ComputedStep,
                    ..Default::default()
                },
                Options {
                    huge_payload: true,
                    update: Update::IntermediateTuple,
                    ..Default::default()
                },
            ] {
                let mut ctx = context();
                let f = fixture(&mut ctx, options);
                assert!(matched(&ctx, &f).0.is_none());
            }
        }

        #[test]
        fn range_metadata_does_not_execute_body_or_condition_or_allocate_payloads() {
            let mut ctx = context();
            let f = fixture(
                &mut ctx,
                Options {
                    huge_payload: true,
                    condition_effect: true,
                    update_control_effect: true,
                    ..Default::default()
                },
            );
            let before = f.module.disp(&ctx).to_string();
            assert_range(&ctx, &f, 0, 4, 1);
            assert_eq!(f.module.disp(&ctx).to_string(), before);
        }

        #[test]
        fn comparison_equality_and_inequality_do_not_match_ranges() {
            for direction in [Direction::Eq, Direction::Ne] {
                let mut ctx = context();
                let f = fixture(
                    &mut ctx,
                    Options {
                        direction,
                        ..Default::default()
                    },
                );
                assert!(matched(&ctx, &f).0.is_none());
            }
        }

        #[test]
        fn add_any_order_accepts_constant_first_without_changing_metadata() {
            let mut ctx = context();
            let f = fixture(&mut ctx, Options::default());
            let parts = Parts::new(
                &ModuleView::new(&ctx, f.module).unwrap(),
                f.while_value.defining_op().unwrap(),
            )
            .unwrap();
            let update = operand(&ctx, parts.body_root, 0)
                .unwrap()
                .defining_op()
                .unwrap();
            let iv = update.deref(&ctx).get_operand(0);
            let constant = update.deref(&ctx).get_operand(1);
            Operation::replace_operand(update, &ctx, 0, constant);
            Operation::replace_operand(update, &ctx, 1, iv);
            assert_range(&ctx, &f, 0, 4, 1);
        }

        #[test]
        fn widened_remainder_difference_avoids_signed_overflow_without_wrapping_endpoint() {
            let mut ctx = context();
            let f = fixture(
                &mut ctx,
                Options {
                    init: i64::MAX as i128 - 1,
                    bound: -(i64::MAX as i128) + 1,
                    step: i64::MAX as i128,
                    direction: Direction::Le,
                    ..Default::default()
                },
            );
            assert_range(&ctx, &f, i64::MAX - 1, i64::MIN, i64::MAX);
        }

        #[test]
        fn initial_evaluation_errors_propagate_instead_of_becoming_absent_metadata() {
            let mut ctx = context();
            let f = fixture(
                &mut ctx,
                Options {
                    initialization: Init::Add,
                    ..Default::default()
                },
            );
            let initial = f.init.defining_op().unwrap();
            let ty = f.init.get_type(&ctx);
            let effect = CustomCallOp::new(
                &mut ctx,
                ty,
                vec![],
                StringAttr::new("range_init_effect".into()),
            )
            .get_operation();
            effect.deref_mut(&ctx).attributes.set(
                Identifier::new("hlo_name"),
                InstructionName("range_init_effect".into()),
            );
            effect
                .deref_mut(&ctx)
                .attributes
                .set(Identifier::new("hlo_has_side_effect"), Bools(vec![true]));
            effect.insert_before(&ctx, initial);
            initial.deref_mut(&ctx).attributes.set(
                Identifier::new("hlo_control_predecessors"),
                ControlPredecessors(vec!["range_init_effect".into()]),
            );
            verification::verify(&ctx, f.module).unwrap();
            let view = ModuleView::new(&ctx, f.module).unwrap();
            let parts = Parts::new(&view, f.while_value.defining_op().unwrap()).unwrap();
            let mut analysis = Analysis {
                view,
                evaluator: HloEvaluator::new(&ctx, f.module)
                    .unwrap()
                    .with_max_loop_iterations(0),
                declines: Vec::new(),
            };
            let error = analysis.source_range(&parts, 0).unwrap_err();
            assert!(matches!(
                error.err.downcast_ref::<EvaluationError>(),
                Some(EvaluationError::Effectful {
                    opcode: Opcode::CustomCall
                })
            ));
            assert!(analysis.declines.is_empty());
        }
    }
    use crate::{
        attributes::{Bools, ComparisonOrder, FusionKind},
        evaluator::ValueState,
        ir::{ControlPredecessors, InstructionName},
        literal::Literal,
        ops::{
            control::{
                CallOp, ConstantOp, CustomCallOp, GetTupleElementOp, ParameterOp, TupleOp, WhileOp,
            },
            elementwise::{AddOp, CompareOp, MultiplyOp, SubtractOp},
        },
        types::ElementType,
    };
    use pliron::{
        builtin::{attributes::StringAttr, types::FunctionType},
        op::Op,
        printable::Printable,
        r#type::TypeHandle,
    };

    fn context() -> Context {
        let mut ctx = Context::new();
        crate::register(&mut ctx);
        ctx
    }

    fn append(ctx: &Context, block: Ptr<BasicBlock>, op: Ptr<Operation>) -> Value {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let index = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        op.deref_mut(ctx).attributes.set(
            Identifier::new("hlo_name"),
            InstructionName(format!("analysis.{index}")),
        );
        op.insert_at_back(block, ctx);
        op.deref(ctx).get_result(0)
    }

    fn finish(ctx: &mut Context, block: Ptr<BasicBlock>, value: Value) {
        ReturnOp::new(ctx, value)
            .get_operation()
            .insert_at_back(block, ctx);
    }

    fn computation(
        ctx: &mut Context,
        module: Ptr<Operation>,
        name: &'static str,
        inputs: Vec<TypeHandle>,
        output: TypeHandle,
    ) -> (Ptr<Operation>, Ptr<BasicBlock>, Vec<Value>) {
        let signature = FunctionType::get(ctx, inputs.clone(), vec![output]).into();
        let comp = ComputationOp::new(ctx, Identifier::new(name), signature);
        let module_block = super::block(ctx, module).unwrap();
        comp.get_operation().insert_at_back(module_block, ctx);
        let block = BasicBlock::new(ctx, None, vec![]);
        block.insert_at_back(comp.body(ctx), ctx);
        let parameters = inputs
            .into_iter()
            .enumerate()
            .map(|(index, ty)| {
                let param = ParameterOp::new(ctx, ty, vec![], Index(index as u32)).get_operation();
                append(ctx, block, param)
            })
            .collect();
        (comp.get_operation(), block, parameters)
    }

    fn constant(
        ctx: &mut Context,
        block: Ptr<BasicBlock>,
        element: ElementType,
        value: i128,
    ) -> Value {
        let literal = if element.is_integer() {
            literal_semantics::integer_scalar(ctx, element, value).unwrap()
        } else {
            let bits = match (element, value) {
                (ElementType::F32, 0) => 0,
                (ElementType::F32, 1) => 0x3f80_0000,
                (ElementType::F32, 5) => 0x40a0_0000,
                _ => panic!("test fixture has no such floating constant"),
            };
            let ty = shape::scalar(ctx, element).unwrap();
            Literal::from_bits(ctx, ty, &[bits]).unwrap()
        };
        let op = ConstantOp::new(ctx, literal.shape, vec![], literal).get_operation();
        append(ctx, block, op)
    }

    fn tuple(ctx: &mut Context, block: Ptr<BasicBlock>, values: Vec<Value>) -> Value {
        let shape = shape::tuple(
            ctx,
            values.iter().map(|value| value.get_type(ctx)).collect(),
        )
        .unwrap();
        let op = TupleOp::new(ctx, shape, values).get_operation();
        append(ctx, block, op)
    }

    fn gte(ctx: &mut Context, block: Ptr<BasicBlock>, value: Value, index: usize) -> Value {
        let shape = shape::subshape(ctx, value.get_type(ctx), &[index]).unwrap();
        let op =
            GetTupleElementOp::new(ctx, shape, vec![value], Index(index as u32)).get_operation();
        append(ctx, block, op)
    }

    fn copy(ctx: &mut Context, block: Ptr<BasicBlock>, value: Value) -> Value {
        let ty = value.get_type(ctx);
        let op = crate::ops::movement::CopyOp::new(ctx, ty, vec![value]).get_operation();
        append(ctx, block, op)
    }

    fn binary(
        ctx: &mut Context,
        block: Ptr<BasicBlock>,
        opcode: Opcode,
        lhs: Value,
        rhs: Value,
        direction: Direction,
    ) -> Value {
        let ty = lhs.get_type(ctx);
        let args = vec![lhs, rhs];
        let op = match opcode {
            Opcode::Add => AddOp::new(ctx, ty, args).get_operation(),
            Opcode::Subtract => SubtractOp::new(ctx, ty, args).get_operation(),
            Opcode::Multiply => MultiplyOp::new(ctx, ty, args).get_operation(),
            Opcode::Compare => {
                let element = shape::element_type(ctx, ty).unwrap();
                let order = if element.is_integer() || element == ElementType::Pred {
                    ComparisonOrder::Total
                } else {
                    ComparisonOrder::Partial
                };
                let pred = shape::scalar(ctx, ElementType::Pred).unwrap();
                CompareOp::new(ctx, pred, args, direction, order).get_operation()
            }
            _ => panic!("unsupported test-fixture opcode"),
        };
        append(ctx, block, op)
    }

    #[derive(Clone, Copy)]
    enum Expr {
        Direct,
        Call,
        Fusion,
        MultiOutputFusion,
        NestedFusion,
    }

    #[allow(clippy::too_many_arguments)]
    fn expression(
        ctx: &mut Context,
        module: Ptr<Operation>,
        block: Ptr<BasicBlock>,
        inputs: [Value; 2],
        op: Opcode,
        direction: Direction,
        form: Expr,
        name: &'static str,
    ) -> Value {
        if matches!(form, Expr::Direct) {
            return binary(ctx, block, op, inputs[0], inputs[1], direction);
        }
        let scalar = if op == Opcode::Compare {
            shape::scalar(ctx, ElementType::Pred).unwrap()
        } else {
            inputs[0].get_type(ctx)
        };
        if matches!(form, Expr::Call) {
            let input_types = inputs.iter().map(|value| value.get_type(ctx)).collect();
            let (_, inner, parameters) = computation(ctx, module, name, input_types, scalar);
            let result = binary(ctx, inner, op, parameters[0], parameters[1], direction);
            finish(ctx, inner, result);
            let call = CallOp::new(ctx, scalar, inputs.to_vec(), Callees(vec![name.into()]))
                .get_operation();
            return append(ctx, block, call);
        }
        let output = if matches!(form, Expr::MultiOutputFusion) {
            shape::tuple(ctx, vec![scalar, scalar]).unwrap()
        } else {
            scalar
        };
        let fusion = FusionOp::new(ctx, output, inputs.to_vec(), FusionKind::Loop);
        let input_types = inputs.iter().map(|value| value.get_type(ctx)).collect();
        let inner = BasicBlock::new(ctx, None, input_types);
        inner.insert_at_back(fusion.body(ctx), ctx);
        let args = [
            inner.deref(ctx).get_argument(0),
            inner.deref(ctx).get_argument(1),
        ];
        let result = if matches!(form, Expr::NestedFusion) {
            expression(ctx, module, inner, args, op, direction, Expr::Fusion, name)
        } else {
            binary(ctx, inner, op, args[0], args[1], direction)
        };
        let result = if matches!(form, Expr::MultiOutputFusion) {
            tuple(ctx, inner, vec![result, result])
        } else {
            result
        };
        finish(ctx, inner, result);
        let result = append(ctx, block, fusion.get_operation());
        if matches!(form, Expr::MultiOutputFusion) {
            gte(ctx, block, result, 0)
        } else {
            result
        }
    }

    #[derive(Clone, Copy)]
    enum Init {
        Constant,
        Copy,
        TupleProjection,
        Add,
        NestedAdd,
        Parameter,
        FusionOfAdd,
    }

    #[derive(Clone, Copy)]
    enum Update {
        Add,
        Subtract,
        Multiply,
        Double,
        CopyOperand,
        CopyResult,
        NestedAdd,
        ComputedStep,
        Constant,
        CustomCall,
        IntermediateTuple,
        ChainedFusion,
    }

    #[derive(Clone, Copy)]
    enum Condition {
        Direct,
        ComputedBound,
        NestedIv,
        CopyParameter,
        OtherSlot,
        Swapped,
    }

    struct Options {
        element: ElementType,
        init: i128,
        bound: i128,
        step: i128,
        direction: Direction,
        initialization: Init,
        update: Update,
        condition: Condition,
        update_expr: Expr,
        condition_expr: Expr,
        huge_payload: bool,
        duplicate_body_gte: bool,
        duplicate_condition_gte: bool,
        condition_effect: bool,
        update_control_effect: bool,
    }

    impl Default for Options {
        fn default() -> Self {
            Self {
                element: ElementType::S64,
                init: 0,
                bound: 5,
                step: 1,
                direction: Direction::Lt,
                initialization: Init::Constant,
                update: Update::Add,
                condition: Condition::Direct,
                update_expr: Expr::Direct,
                condition_expr: Expr::Direct,
                huge_payload: false,
                duplicate_body_gte: false,
                duplicate_condition_gte: false,
                condition_effect: false,
                update_control_effect: false,
            }
        }
    }

    struct Fixture {
        module: Ptr<Operation>,
        while_value: Value,
        init: Value,
        initial_parameter: Value,
        body: Ptr<Operation>,
        condition: Ptr<Operation>,
    }

    fn effect(ctx: &mut Context, block: Ptr<BasicBlock>, input: Value) -> Value {
        let ty = input.get_type(ctx);
        let op = CustomCallOp::new(ctx, ty, vec![input], StringAttr::new("effect".into()))
            .get_operation();
        op.deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_has_side_effect"), Bools(vec![true]));
        append(ctx, block, op)
    }

    fn fixture(ctx: &mut Context, options: Options) -> Fixture {
        let module = ModuleOp::new(ctx, Identifier::new("while_analysis")).get_operation();
        crate::ir::set_entry(ctx, module, Identifier::new("main")).unwrap();
        let scalar = shape::scalar(ctx, options.element).unwrap();
        let pred = shape::scalar(ctx, ElementType::Pred).unwrap();
        let payload = if options.huge_payload {
            shape::array(ctx, ElementType::F32, vec![1_000_000_000]).unwrap()
        } else {
            scalar
        };
        let state = shape::tuple(ctx, vec![scalar, payload]).unwrap();
        let (body, block, args) = computation(ctx, module, "body", vec![state], state);
        let iv = gte(ctx, block, args[0], 0);
        let data = gte(ctx, block, args[0], 1);
        if options.duplicate_body_gte {
            gte(ctx, block, args[0], 0);
        }
        let step = constant(ctx, block, options.element, options.step);
        let update = match options.update {
            Update::Constant => step,
            Update::Double => binary(ctx, block, Opcode::Add, iv, iv, options.direction),
            Update::CustomCall => {
                let op =
                    CustomCallOp::new(ctx, scalar, vec![iv], StringAttr::new("unsupported".into()))
                        .get_operation();
                append(ctx, block, op)
            }
            _ => {
                let iv = match options.update {
                    Update::CopyOperand => copy(ctx, block, iv),
                    Update::NestedAdd => {
                        binary(ctx, block, Opcode::Add, iv, step, options.direction)
                    }
                    Update::IntermediateTuple => {
                        let intermediate = tuple(ctx, block, vec![iv, data]);
                        gte(ctx, block, intermediate, 0)
                    }
                    Update::ChainedFusion => expression(
                        ctx,
                        module,
                        block,
                        [iv, step],
                        Opcode::Add,
                        options.direction,
                        Expr::Fusion,
                        "first_update",
                    ),
                    _ => iv,
                };
                let step = if matches!(options.update, Update::ComputedStep) {
                    binary(ctx, block, Opcode::Add, step, step, options.direction)
                } else {
                    step
                };
                let opcode = match options.update {
                    Update::Subtract => Opcode::Subtract,
                    Update::Multiply => Opcode::Multiply,
                    _ => Opcode::Add,
                };
                let result = expression(
                    ctx,
                    module,
                    block,
                    [iv, step],
                    opcode,
                    options.direction,
                    options.update_expr,
                    "update_fn",
                );
                if matches!(options.update, Update::CopyResult) {
                    copy(ctx, block, result)
                } else {
                    result
                }
            }
        };
        if options.update_control_effect {
            let effect = effect(ctx, block, iv);
            let name =
                attr::<InstructionName>(ctx, effect.defining_op().unwrap(), "hlo_name").unwrap();
            update.defining_op().unwrap().deref_mut(ctx).attributes.set(
                Identifier::new("hlo_control_predecessors"),
                ControlPredecessors(vec![name.0]),
            );
        }
        let body_root = tuple(ctx, block, vec![update, data]);
        finish(ctx, block, body_root);

        let (condition, block, args) = computation(ctx, module, "condition", vec![state], pred);
        let parameter = if matches!(options.condition, Condition::CopyParameter) {
            copy(ctx, block, args[0])
        } else {
            args[0]
        };
        let iv = gte(ctx, block, parameter, 0);
        if options.duplicate_condition_gte {
            gte(ctx, block, parameter, 0);
        }
        let bound = constant(ctx, block, options.element, options.bound);
        let (iv, bound) = match options.condition {
            Condition::ComputedBound => {
                let zero = constant(ctx, block, options.element, 0);
                (
                    iv,
                    binary(ctx, block, Opcode::Add, bound, zero, options.direction),
                )
            }
            Condition::NestedIv => {
                let one = constant(ctx, block, options.element, 1);
                (
                    binary(ctx, block, Opcode::Add, iv, one, options.direction),
                    bound,
                )
            }
            Condition::OtherSlot => (iv, gte(ctx, block, parameter, 1)),
            Condition::Swapped => (bound, iv),
            _ => (iv, bound),
        };
        let result = expression(
            ctx,
            module,
            block,
            [iv, bound],
            Opcode::Compare,
            options.direction,
            options.condition_expr,
            "condition_fn",
        );
        if options.condition_effect {
            effect(ctx, block, iv);
        }
        finish(ctx, block, result);

        let (_, block, args) = computation(ctx, module, "main", vec![scalar, payload], state);
        let initial_parameter = args[0];
        let init = constant(ctx, block, options.element, options.init);
        let init = match options.initialization {
            Init::Constant => init,
            Init::Copy => copy(ctx, block, init),
            Init::TupleProjection => {
                let input = tuple(ctx, block, vec![init]);
                gte(ctx, block, input, 0)
            }
            Init::Parameter => initial_parameter,
            Init::Add | Init::NestedAdd | Init::FusionOfAdd => {
                let zero = constant(ctx, block, options.element, 0);
                let first = binary(ctx, block, Opcode::Add, init, zero, options.direction);
                if matches!(options.initialization, Init::NestedAdd) {
                    binary(ctx, block, Opcode::Add, first, zero, options.direction)
                } else if matches!(options.initialization, Init::FusionOfAdd) {
                    expression(
                        ctx,
                        module,
                        block,
                        [first, zero],
                        Opcode::Add,
                        options.direction,
                        Expr::Fusion,
                        "init_fn",
                    )
                } else {
                    first
                }
            }
        };
        let input = tuple(ctx, block, vec![init, args[1]]);
        let while_op = WhileOp::new(
            ctx,
            state,
            vec![input],
            Callees(vec!["body".into(), "condition".into()]),
        )
        .get_operation();
        let while_value = append(ctx, block, while_op);
        finish(ctx, block, while_value);
        verification::verify(ctx, module).unwrap();
        Fixture {
            module,
            while_value,
            init,
            initial_parameter,
            body,
            condition,
        }
    }

    fn facts(ctx: &Context, f: &Fixture, budget: i64) -> WhileLoopFacts {
        analyze(
            ctx,
            f.module,
            WhileLoopRequest {
                value: f.while_value,
                max_brute_force_iters: budget,
            },
        )
        .unwrap()
    }
}
