//! Pure native evaluation with explicit partial-value and error semantics.
//!
//! Based on OpenXLA's `hlo_evaluator.{h,cc}` (request state,
//! substitutions, tuple paths, embedded evaluators, layout postprocessing).
//! Evaluation supports a subset of HLO operations and rejects unsupported
//! operations with [`EvaluationError`]. Unknown inputs are represented by a
//! partial value tree, never by fabricated zero-filled literals. Use
//! [`EvaluatedValue::into_literal`] when a completely known result is required.
//! Each request owns its dependency stack, substitutions, and memoized values.
mod bitcast;
mod dispatch;
mod dynamic_indexed;
mod frame;
mod numerical;
mod reduce;
mod reduce_window;
mod session;
mod value;
mod while_loop;

pub use value::{EvaluatedValue, ValueState};

use crate::{
    analysis::effects,
    attributes::{Axes, Callees, ComparisonOrder, Direction, Index, Slice as SliceGeometry},
    ir::{ControlPredecessors, InstructionName},
    literal::{
        Literal,
        semantics::{self as literal_semantics, movement},
    },
    opcode::Opcode,
    ops::{ComputationOp, HloOpInterface, control::FusionOp},
    scalar::semantics::{self as scalar_semantics, EvaluationFloatContext, Scalar},
    shape::{self, Layout},
    verification::{self, optional_attr},
};
use frame::{Frame, Key};
use pliron::{
    attribute::Attribute,
    builtin::ops::ModuleOp,
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    op::op_cast,
    operation::Operation,
    pass::{Analysis, AnalysisManager},
    result::Result,
    r#type::{TypeHandle, Typed},
    value::{DefiningEntity, Value},
};
use std::collections::HashMap;

/// Recoverable evaluator categories, carried by the standard pliron Error.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum EvaluationError {
    /// A complete result was requested for an unknown or unvisited array.
    #[error("evaluation needs a known value, but an array is unknown or undetermined")]
    UnknownValue,
    /// Loop execution reached its configured safety limit.
    #[error("loop exceeded its configured iteration limit ({limit})")]
    LoopIterationLimit {
        /// Maximum number of iterations allowed by the evaluation session.
        limit: i64,
    },
    /// Shallow evaluation encountered an operand that requires recursive execution.
    #[error("shallow evaluation requires direct constants or supplied operand values")]
    ShallowDependency,
    /// The operation has no supported native evaluator implementation.
    #[error("native evaluator foundation does not implement {opcode}")]
    Unsupported {
        /// Operation that cannot be evaluated.
        opcode: Opcode,
    },
    /// The operation is known, but its operand format or value is unsupported.
    #[error("native evaluator foundation does not implement {opcode} for this value: {message}")]
    UnsupportedValue {
        /// Operation whose value domain is unsupported.
        opcode: Opcode,
        /// Explanation of the unsupported operand or result configuration.
        message: String,
    },
    /// Evaluation needs a simulated floating-point context that was not supplied.
    #[error("evaluating {opcode} requires an explicit floating-point context")]
    MissingFloatContext {
        /// Operation requiring an explicit floating-point context.
        opcode: Opcode,
    },
    /// Pure evaluation cannot execute an operation with observable side effects.
    #[error("pure evaluation cannot execute effectful instruction {opcode}")]
    Effectful {
        /// Effectful operation rejected by pure evaluation.
        opcode: Opcode,
    },
    /// A computation call or data dependency would recurse into an active frame.
    #[error("recursive computation or data dependency during pure evaluation")]
    Recursion,
    /// The request, graph, or evaluation frame violates structural requirements.
    #[error("malformed evaluation request or frame: {message}")]
    Malformed {
        /// Explanation of the invalid request or frame.
        message: String,
    },
}

impl EvaluationError {
    /// Returns the action needed before this evaluation can be retried.
    pub fn recovery_category(&self) -> crate::RecoveryCategory {
        use crate::RecoveryCategory;
        match self {
            Self::UnknownValue | Self::ShallowDependency | Self::MissingFloatContext { .. } => {
                RecoveryCategory::MissingInput
            }
            Self::LoopIterationLimit { .. } => RecoveryCategory::ResourceLimit,
            Self::Unsupported { .. } | Self::UnsupportedValue { .. } | Self::Effectful { .. } => {
                RecoveryCategory::Unsupported
            }
            Self::Recursion | Self::Malformed { .. } => RecoveryCategory::InvalidInput,
        }
    }
}

impl From<EvaluationError> for pliron::result::Error {
    fn from(error: EvaluationError) -> Self {
        pliron::verify_error_noloc!(error)
    }
}

fn failure(error: EvaluationError) -> pliron::result::Error {
    error.into()
}

fn malformed(message: impl Into<String>) -> pliron::result::Error {
    failure(EvaluationError::Malformed {
        message: message.into(),
    })
}

fn unsupported_value(opcode: Opcode, message: &str) -> pliron::result::Error {
    failure(EvaluationError::UnsupportedValue {
        opcode,
        message: message.into(),
    })
}

/// Policy for resolving an instruction's operand dependencies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DependencyMode {
    /// Accepts only direct constants, substitutions, or bound frame arguments.
    Shallow,
    /// Evaluates supported dependency instructions recursively.
    Recursive,
}

/// One request model for both instruction and computation evaluation.
///
/// Substitutions apply only to values in the instruction's isolated frame, take
/// precedence over constants/bindings, and are owned by fresh request state.
/// Computation arguments may contain partial tuple values.
pub enum Request<'a> {
    /// Evaluates one SSA value, optionally replacing dependencies in its frame.
    Instruction {
        /// SSA result to evaluate in its enclosing isolated computation.
        value: Value,
        /// Frame-local value replacements; duplicate keys are rejected.
        substitutions: &'a [(Value, EvaluatedValue)],
        /// Whether operand producers may be evaluated recursively.
        dependency_mode: DependencyMode,
    },
    /// Evaluates an entire computation using the supplied parameter values.
    Computation {
        /// Computation operation belonging to the evaluator's module.
        callee: Ptr<Operation>,
        /// Parameter values in parameter-number order; tuple leaves may be unknown.
        args: &'a [EvaluatedValue],
    },
}

/// Reusable immutable-context evaluator handle. No result or analysis survives
/// a request, so changes made between requests cannot leave stale memo entries.
pub struct HloEvaluator<'a> {
    ctx: &'a Context,
    module: Ptr<Operation>,
    max_loop_iterations: i64,
    float_context: Option<EvaluationFloatContext>,
}

impl<'a> HloEvaluator<'a> {
    /// Creates an evaluator bound to a native module and its owning context.
    ///
    /// Returns an error if `module` is not a builtin module operation. Full
    /// graph verification is deferred until each call to [`Self::evaluate`].
    /// No floating-point context is assumed; use [`Self::with_float_context`]
    /// before evaluating operations that require one.
    pub fn new(ctx: &'a Context, module: Ptr<Operation>) -> Result<Self> {
        if !Operation::is_op::<ModuleOp>(module, ctx) {
            return Err(malformed("evaluator requires a native module"));
        }
        Ok(Self {
            ctx,
            module,
            max_loop_iterations: -1,
            float_context: None,
        })
    }

    /// Set the explicit floating-point context used by supported rounding ops.
    ///
    /// RoundNearestEven requires this context; the evaluator never assumes an
    /// ambient or default rounding mode. The context is copied into each
    /// request-local session and does not affect other evaluator handles.
    pub fn with_float_context(mut self, context: EvaluationFloatContext) -> Self {
        self.float_context = Some(context);
        self
    }

    /// Validate the module and evaluate using a fresh stack, frames and memo.
    ///
    /// Constants and raw movement preserve all formats. Numerical
    /// evaluation delegates to the qualified scalar kernel; unsupported
    /// opcode/format combinations remain typed errors.
    /// Reduce and ReduceWindow invoke their reducers through the same
    /// request-local session and support bounded dynamic runtime sizes.
    /// Unbounded output allocation remains unsupported.
    /// RoundNearestEven is supported only when this evaluator has an explicit
    /// floating-point context from [`Self::with_float_context`]; other
    /// rounding modes are rejected by the scalar kernel.
    /// Conditional evaluates only its selector and selected argument/callee;
    /// unselected computations and arguments are never executed.
    pub fn evaluate(&self, request: Request<'_>) -> Result<EvaluatedValue> {
        verification::verify(self.ctx, self.module)?;
        let mut session = Session::new(self.ctx, self.module)?;
        session.while_state = while_loop::WhileState::new(self.max_loop_iterations);
        session.float_context = self.float_context;
        match request {
            Request::Instruction {
                value,
                substitutions,
                dependency_mode,
            } => {
                let owner = session.owner(value)?;
                let mut frame = Frame::new(self.ctx, owner)?;
                for (key, replacement) in substitutions {
                    if !frame.members.contains(key) {
                        return Err(malformed("substitution value is outside the request frame"));
                    }
                    let replacement = replacement
                        .clone()
                        .retarget(self.ctx, key.get_type(self.ctx))?;
                    if frame.substitutions.insert(*key, replacement).is_some() {
                        return Err(malformed("duplicate substitution value"));
                    }
                }
                session.stack.push(owner);
                session.eval(&mut frame, value, &[], dependency_mode)
            }
            Request::Computation { callee, args } => session.computation(callee, args, &[]),
        }
    }
}

fn opcode(ctx: &Context, operation: Ptr<Operation>) -> Result<Opcode> {
    let dynamic = Operation::get_op_dyn(operation, ctx);
    op_cast::<dyn HloOpInterface>(&*dynamic)
        .map(|hlo| hlo.opcode())
        .ok_or_else(|| malformed("evaluation encountered a non-HLO instruction"))
}

fn attr<T: Attribute + Clone>(
    ctx: &Context,
    operation: Ptr<Operation>,
    key: &'static str,
) -> Result<T> {
    optional_attr::<T>(ctx, operation, key)?
        .map(|value| value.clone())
        .ok_or_else(|| malformed(format!("instruction has no typed {key} attribute")))
}

struct Session<'a> {
    ctx: &'a Context,
    module: Ptr<Operation>,
    callees: HashMap<crate::ir::ComputationName, Ptr<Operation>>,
    channel_sensitive: bool,
    stack: Vec<Ptr<Operation>>,
    effects: effects::SideEffects,
    while_state: while_loop::WhileState,
    float_context: Option<EvaluationFloatContext>,
}

#[cfg(test)]
mod tests {
    //! Shared evaluator fixtures, request semantics, and opcode-family regressions.
    #[path = "../native_bitcast_tests.rs"]
    mod bitcast;

    #[path = "../native_dynamic_indexed_tests.rs"]
    mod dynamic_indexed;

    #[path = "../native_reduce_window_tests.rs"]
    mod reduce_window;

    #[path = "../native_reduce_tests.rs"]
    mod reduce;

    mod requests {
        use super::*;

        #[test]
        fn upstream_substitutions_override_intermediate_in_shallow_request() {
            let mut ctx = context();
            let f = math(&mut ctx);
            let substitutions = [
                (f.parameter, known(&ctx, integer(&ctx, ElementType::S64, 3))),
                (f.square, known(&ctx, integer(&ctx, ElementType::S64, 40))),
            ];
            assert_eq!(
                read_i64(
                    &ctx,
                    evaluate(
                        &ctx,
                        f.module,
                        f.sum,
                        &substitutions,
                        DependencyMode::Shallow
                    )
                    .unwrap()
                ),
                43
            );
        }

        #[test]
        fn upstream_request_reuse_does_not_cache_substitutions_or_results() {
            let mut ctx = context();
            let f = math(&mut ctx);
            let evaluator = HloEvaluator::new(&ctx, f.module).unwrap();
            for (input, expected) in [(3, 12), (5, 30), (-2, 2)] {
                let substitutions = [(
                    f.parameter,
                    known(&ctx, integer(&ctx, ElementType::S64, input)),
                )];
                let result = evaluator
                    .evaluate(Request::Instruction {
                        value: f.sum,
                        substitutions: &substitutions,
                        dependency_mode: DependencyMode::Recursive,
                    })
                    .unwrap();
                assert_eq!(read_i64(&ctx, result), expected);
            }
        }

        #[test]
        fn upstream_deep_substitution_is_used_by_recursive_dependencies() {
            let mut ctx = context();
            let f = math(&mut ctx);
            let substitutions = [(f.parameter, known(&ctx, integer(&ctx, ElementType::S64, 4)))];
            assert_eq!(
                read_i64(
                    &ctx,
                    evaluate(
                        &ctx,
                        f.module,
                        f.sum,
                        &substitutions,
                        DependencyMode::Recursive
                    )
                    .unwrap()
                ),
                20
            );
        }

        #[test]
        fn upstream_substituted_root_needs_no_operand_evaluation() {
            let mut ctx = context();
            let f = math(&mut ctx);
            let substitutions = [(f.sum, known(&ctx, integer(&ctx, ElementType::S64, 91)))];
            assert_eq!(
                read_i64(
                    &ctx,
                    evaluate(
                        &ctx,
                        f.module,
                        f.sum,
                        &substitutions,
                        DependencyMode::Shallow
                    )
                    .unwrap()
                ),
                91
            );
        }

        #[test]
        fn shallow_nonconstant_dependency_has_typed_precondition_failure() {
            let mut ctx = context();
            let f = math(&mut ctx);
            let substitutions = [(f.parameter, known(&ctx, integer(&ctx, ElementType::S64, 2)))];
            let error = evaluate(
                &ctx,
                f.module,
                f.sum,
                &substitutions,
                DependencyMode::Shallow,
            )
            .unwrap_err();
            assert!(matches!(
                error.err.downcast_ref::<EvaluationError>(),
                Some(EvaluationError::ShallowDependency)
            ));
        }

        #[test]
        fn recursive_unbound_parameter_returns_unknown_not_zero() {
            let mut ctx = context();
            let f = math(&mut ctx);
            let result = evaluate(&ctx, f.module, f.sum, &[], DependencyMode::Recursive).unwrap();
            assert!(matches!(result.state(), ValueState::Unknown));
            assert!(result.is_determined());
            let error = result.into_literal(&ctx).unwrap_err();
            assert!(matches!(
                error.err.downcast_ref::<EvaluationError>(),
                Some(EvaluationError::UnknownValue)
            ));
        }

        #[test]
        fn unknown_and_undetermined_tuple_leaves_are_distinct_and_owned() {
            let ctx = context();
            let scalar = shape::scalar(&ctx, ElementType::S32).unwrap();
            let token = Literal::token(&ctx).shape;
            let tuple = shape::tuple(&ctx, vec![scalar, token]).unwrap();
            let unknown = EvaluatedValue::unknown(&ctx, tuple).unwrap();
            let undetermined = EvaluatedValue::undetermined(&ctx, tuple).unwrap();
            assert!(unknown.is_determined());
            assert!(!undetermined.is_determined());
            assert!(!unknown.is_known());
            assert!(unknown.subvalue(&[1]).unwrap().is_known());
            assert!(matches!(
                undetermined.subvalue(&[0]).unwrap().state(),
                ValueState::Undetermined
            ));
            assert!(unknown.subvalue(&[2]).is_err());
            assert!(unknown.subvalue(&[0, 0]).is_err());
        }

        #[test]
        fn empty_tuple_and_token_materialize_without_array_payload() {
            let ctx = context();
            let empty = shape::tuple(&ctx, vec![]).unwrap();
            let value = EvaluatedValue::unknown(&ctx, empty).unwrap();
            assert!(value.is_known());
            assert!(value.into_literal(&ctx).unwrap().tuple_literals.is_empty());
            let token = known(&ctx, Literal::token(&ctx));
            assert!(token.is_known());
            assert!(token.into_literal(&ctx).unwrap().storage.is_empty());
        }

        #[test]
        fn upstream_partial_tuple_projection_skips_unknown_siblings() {
            let mut ctx = context();
            let module = module(&mut ctx);
            let ty = shape::scalar(&ctx, ElementType::S64).unwrap();
            let tuple_ty = shape::tuple(&ctx, vec![ty, ty]).unwrap();
            let (_, block, p) = computation(&mut ctx, module, "main", &[ty], ty);
            let literal = integer(&ctx, ElementType::S64, 17);
            let c = constant(&mut ctx, block, literal, "constant");
            let tuple = TupleOp::new(&mut ctx, tuple_ty, vec![p[0], c]).get_operation();
            let tuple = append(&ctx, block, tuple, "tuple");
            let gte = GetTupleElementOp::new(&mut ctx, ty, vec![tuple], Index(1)).get_operation();
            let gte = append(&ctx, block, gte, "gte");
            finish(&mut ctx, block, gte);
            assert_eq!(
                read_i64(
                    &ctx,
                    evaluate(&ctx, module, gte, &[], DependencyMode::Recursive).unwrap()
                ),
                17
            );
            let partial = evaluate(&ctx, module, tuple, &[], DependencyMode::Recursive).unwrap();
            assert!(matches!(
                partial.subvalue(&[0]).unwrap().state(),
                ValueState::Unknown
            ));
            assert_eq!(read_i64(&ctx, partial.subvalue(&[1]).unwrap().clone()), 17);
        }

        #[test]
        fn nested_partial_tuple_projects_through_copy_without_unrelated_evaluation() {
            let mut ctx = context();
            let module = module(&mut ctx);
            let ty = shape::scalar(&ctx, ElementType::S64).unwrap();
            let inner_ty = shape::tuple(&ctx, vec![ty, ty]).unwrap();
            let outer_ty = shape::tuple(&ctx, vec![ty, inner_ty]).unwrap();
            let (_, block, p) = computation(&mut ctx, module, "main", &[ty], ty);
            let literal = integer(&ctx, ElementType::S64, 23);
            let c = constant(&mut ctx, block, literal, "c");
            let inner = TupleOp::new(&mut ctx, inner_ty, vec![p[0], c]).get_operation();
            let inner = append(&ctx, block, inner, "inner");
            let outer = TupleOp::new(&mut ctx, outer_ty, vec![p[0], inner]).get_operation();
            let outer = append(&ctx, block, outer, "outer");
            let copy = CopyOp::new(&mut ctx, outer_ty, vec![outer]).get_operation();
            let copy = append(&ctx, block, copy, "copy");
            let a =
                GetTupleElementOp::new(&mut ctx, inner_ty, vec![copy], Index(1)).get_operation();
            let a = append(&ctx, block, a, "a");
            let b = GetTupleElementOp::new(&mut ctx, ty, vec![a], Index(1)).get_operation();
            let b = append(&ctx, block, b, "b");
            finish(&mut ctx, block, b);
            assert_eq!(
                read_i64(
                    &ctx,
                    evaluate(&ctx, module, b, &[], DependencyMode::Recursive).unwrap()
                ),
                23
            );
        }

        #[test]
        fn substitution_of_constant_takes_precedence_over_intrinsic_literal() {
            let mut ctx = context();
            let module = module(&mut ctx);
            let literal = integer(&ctx, ElementType::S64, 2);
            let (_, block, _) = computation(&mut ctx, module, "main", &[], literal.shape);
            let c = constant(&mut ctx, block, literal, "c");
            finish(&mut ctx, block, c);
            let substitutions = [(c, known(&ctx, integer(&ctx, ElementType::S64, 8)))];
            assert_eq!(
                read_i64(
                    &ctx,
                    evaluate(&ctx, module, c, &substitutions, DependencyMode::Shallow).unwrap()
                ),
                8
            );
        }

        #[test]
        fn duplicate_and_wrong_shape_substitutions_fail_explicitly() {
            let mut ctx = context();
            let f = math(&mut ctx);
            let replacement = known(&ctx, integer(&ctx, ElementType::S64, 1));
            assert!(
                evaluate(
                    &ctx,
                    f.module,
                    f.sum,
                    &[
                        (f.parameter, replacement.clone()),
                        (f.parameter, replacement)
                    ],
                    DependencyMode::Recursive
                )
                .is_err()
            );
            let wrong = known(&ctx, integer(&ctx, ElementType::U64, 1));
            assert!(
                evaluate(
                    &ctx,
                    f.module,
                    f.sum,
                    &[(f.parameter, wrong)],
                    DependencyMode::Recursive
                )
                .is_err()
            );
        }

        #[test]
        fn computation_rejects_missing_and_wrongly_typed_arguments() {
            let mut ctx = context();
            let f = math(&mut ctx);
            let evaluator = HloEvaluator::new(&ctx, f.module).unwrap();
            assert!(
                evaluator
                    .evaluate(Request::Computation {
                        callee: f.computation,
                        args: &[]
                    })
                    .is_err()
            );
            let wrong = known(&ctx, integer(&ctx, ElementType::S32, 1));
            assert!(
                evaluator
                    .evaluate(Request::Computation {
                        callee: f.computation,
                        args: &[wrong]
                    })
                    .is_err()
            );
        }

        #[test]
        fn while_initial_false_returns_initial_value_without_body_execution() {
            let mut ctx = context();
            let module = module(&mut ctx);
            let ty = shape::scalar(&ctx, ElementType::S64).unwrap();
            let pred = shape::scalar(&ctx, ElementType::Pred).unwrap();
            let (_, body, p) = computation(&mut ctx, module, "body", &[ty], ty);
            finish(&mut ctx, body, p[0]);
            let (_, cond, _) = computation(&mut ctx, module, "condition", &[ty], pred);
            let condition_literal = Literal::from_bits(&ctx, pred, &[0]).unwrap();
            let c = constant(&mut ctx, cond, condition_literal, "false");
            finish(&mut ctx, cond, c);
            let (_, block, p) = computation(&mut ctx, module, "main", &[ty], ty);
            let loop_op = WhileOp::new(
                &mut ctx,
                ty,
                vec![p[0]],
                Callees(vec!["body".into(), "condition".into()]),
            )
            .get_operation();
            let root = append(&ctx, block, loop_op, "while");
            finish(&mut ctx, block, root);
            let substitutions = [(p[0], known(&ctx, integer(&ctx, ElementType::S64, 0)))];
            let result = evaluate(
                &ctx,
                module,
                root,
                &substitutions,
                DependencyMode::Recursive,
            )
            .unwrap();
            assert_eq!(read_i64(&ctx, result), 0);
        }

        #[test]
        fn repeated_evaluation_preserves_ir_names_controls_and_all_metadata() {
            let mut ctx = context();
            let f = math(&mut ctx);
            f.square
                .defining_op()
                .unwrap()
                .deref_mut(&ctx)
                .attributes
                .set(
                    Identifier::new("hlo_control_predecessors"),
                    ControlPredecessors(vec!["main.p0".into()]),
                );
            f.module.deref_mut(&ctx).attributes.set(
                Identifier::new("hlo_schedule"),
                crate::ir::Schedule(vec![crate::ir::InstructionSequence {
                    computation: "main".into(),
                    instructions: vec!["main.p0".into(), "square".into(), "sum".into()],
                }]),
            );
            let before = format!("{}", f.module.disp(&ctx));
            let operands = f
                .sum
                .defining_op()
                .unwrap()
                .deref(&ctx)
                .operands()
                .collect::<Vec<_>>();
            let substitutions = [(f.parameter, known(&ctx, integer(&ctx, ElementType::S64, 6)))];
            for _ in 0..3 {
                assert_eq!(
                    read_i64(
                        &ctx,
                        evaluate(
                            &ctx,
                            f.module,
                            f.sum,
                            &substitutions,
                            DependencyMode::Recursive
                        )
                        .unwrap()
                    ),
                    42
                );
            }
            assert_eq!(format!("{}", f.module.disp(&ctx)), before);
            assert_eq!(
                f.sum
                    .defining_op()
                    .unwrap()
                    .deref(&ctx)
                    .operands()
                    .collect::<Vec<_>>(),
                operands
            );
        }

        #[test]
        fn request_reuse_after_literal_attribute_change_has_no_persistent_cache() {
            let mut ctx = context();
            let module = module(&mut ctx);
            let ty = shape::scalar(&ctx, ElementType::S64).unwrap();
            let (_, block, _) = computation(&mut ctx, module, "main", &[], ty);
            let literal = integer(&ctx, ElementType::S64, 1);
            let root = constant(&mut ctx, block, literal, "constant");
            finish(&mut ctx, block, root);
            let evaluator = HloEvaluator::new(&ctx, module).unwrap();
            for expected in [2, 5, 9] {
                root.defining_op().unwrap().deref_mut(&ctx).attributes.set(
                    Identifier::new("hlo_literal"),
                    integer(&ctx, ElementType::S64, expected),
                );
                let result = evaluator
                    .evaluate(Request::Instruction {
                        value: root,
                        substitutions: &[],
                        dependency_mode: DependencyMode::Shallow,
                    })
                    .unwrap();
                assert_eq!(read_i64(&ctx, result), expected as i64);
            }
        }

        #[test]
        fn tuple_path_memo_keeps_two_projected_leaves_separate_in_one_request() {
            let mut ctx = context();
            let module = module(&mut ctx);
            let ty = shape::scalar(&ctx, ElementType::S64).unwrap();
            let tuple_ty = shape::tuple(&ctx, vec![ty, ty]).unwrap();
            let (_, block, _) = computation(&mut ctx, module, "main", &[], ty);
            let a_literal = integer(&ctx, ElementType::S64, 7);
            let b_literal = integer(&ctx, ElementType::S64, 11);
            let a = constant(&mut ctx, block, a_literal, "a");
            let b = constant(&mut ctx, block, b_literal, "b");
            let tuple = TupleOp::new(&mut ctx, tuple_ty, vec![a, b]).get_operation();
            let tuple = append(&ctx, block, tuple, "tuple");
            let x = GetTupleElementOp::new(&mut ctx, ty, vec![tuple], Index(0)).get_operation();
            let x = append(&ctx, block, x, "x");
            let y = GetTupleElementOp::new(&mut ctx, ty, vec![tuple], Index(1)).get_operation();
            let y = append(&ctx, block, y, "y");
            let add = AddOp::new(&mut ctx, ty, vec![x, y]).get_operation();
            let root = append(&ctx, block, add, "add");
            finish(&mut ctx, block, root);
            assert_eq!(
                read_i64(
                    &ctx,
                    evaluate(&ctx, module, root, &[], DependencyMode::Recursive).unwrap()
                ),
                18
            );
        }

        #[test]
        fn owned_evaluation_result_does_not_alias_substitution_literal() {
            let mut ctx = context();
            let f = math(&mut ctx);
            let replacement = known(&ctx, integer(&ctx, ElementType::S64, 12));
            let result = evaluate(
                &ctx,
                f.module,
                f.parameter,
                &[(f.parameter, replacement.clone())],
                DependencyMode::Shallow,
            )
            .unwrap();
            let mut literal = result.into_literal(&ctx).unwrap();
            literal.storage.fill(0);
            assert_eq!(read_i64(&ctx, replacement), 12);
        }
    }
    mod calls {
        use super::*;

        #[test]
        fn named_call_frame_binds_parameters_and_computation_request_uses_same_engine() {
            let mut ctx = context();
            let module = module(&mut ctx);
            let ty = shape::scalar(&ctx, ElementType::S64).unwrap();
            let (callee, body, p) = computation(&mut ctx, module, "double", &[ty], ty);
            let sum = AddOp::new(&mut ctx, ty, vec![p[0], p[0]]).get_operation();
            let sum = append(&ctx, body, sum, "double.sum");
            finish(&mut ctx, body, sum);
            let (_, block, args) = computation(&mut ctx, module, "main", &[ty], ty);
            let call = CallOp::new(&mut ctx, ty, vec![args[0]], Callees(vec!["double".into()]))
                .get_operation();
            let call = append(&ctx, block, call, "call");
            finish(&mut ctx, block, call);
            let argument = known(&ctx, integer(&ctx, ElementType::S64, 19));
            assert_eq!(
                read_i64(
                    &ctx,
                    evaluate(
                        &ctx,
                        module,
                        call,
                        &[(args[0], argument.clone())],
                        DependencyMode::Recursive
                    )
                    .unwrap()
                ),
                38
            );
            let evaluator = HloEvaluator::new(&ctx, module).unwrap();
            assert_eq!(
                read_i64(
                    &ctx,
                    evaluator
                        .evaluate(Request::Computation {
                            callee,
                            args: &[argument]
                        })
                        .unwrap()
                ),
                38
            );
        }

        #[test]
        fn sibling_calls_do_not_share_argument_memo_entries() {
            let mut ctx = context();
            let module = module(&mut ctx);
            let ty = shape::scalar(&ctx, ElementType::S64).unwrap();
            let (_, body, p) = computation(&mut ctx, module, "identity", &[ty], ty);
            finish(&mut ctx, body, p[0]);
            let (_, block, p) = computation(&mut ctx, module, "main", &[ty, ty], ty);
            let a = CallOp::new(&mut ctx, ty, vec![p[0]], Callees(vec!["identity".into()]))
                .get_operation();
            let a = append(&ctx, block, a, "a");
            let b = CallOp::new(&mut ctx, ty, vec![p[1]], Callees(vec!["identity".into()]))
                .get_operation();
            let b = append(&ctx, block, b, "b");
            let sum = AddOp::new(&mut ctx, ty, vec![a, b]).get_operation();
            let sum = append(&ctx, block, sum, "sum");
            finish(&mut ctx, block, sum);
            let substitutions = [
                (p[0], known(&ctx, integer(&ctx, ElementType::S64, 7))),
                (p[1], known(&ctx, integer(&ctx, ElementType::S64, 11))),
            ];
            assert_eq!(
                read_i64(
                    &ctx,
                    evaluate(&ctx, module, sum, &substitutions, DependencyMode::Recursive).unwrap()
                ),
                18
            );
        }

        #[test]
        fn named_callee_values_cannot_be_injected_into_caller_substitution_frame() {
            let mut ctx = context();
            let module = module(&mut ctx);
            let ty = shape::scalar(&ctx, ElementType::S64).unwrap();
            let (_, body, p) = computation(&mut ctx, module, "identity", &[ty], ty);
            finish(&mut ctx, body, p[0]);
            let (_, block, a) = computation(&mut ctx, module, "main", &[ty], ty);
            let call = CallOp::new(&mut ctx, ty, vec![a[0]], Callees(vec!["identity".into()]))
                .get_operation();
            let call = append(&ctx, block, call, "call");
            finish(&mut ctx, block, call);
            let substitutions = [(p[0], known(&ctx, integer(&ctx, ElementType::S64, 7)))];
            assert!(
                evaluate(
                    &ctx,
                    module,
                    call,
                    &substitutions,
                    DependencyMode::Recursive
                )
                .is_err()
            );
        }

        #[test]
        fn inline_fusion_binds_block_arguments_without_parameter_instruction_aliases() {
            let mut ctx = context();
            let module = module(&mut ctx);
            let ty = shape::scalar(&ctx, ElementType::S64).unwrap();
            let (_, block, p) = computation(&mut ctx, module, "main", &[ty], ty);
            let fusion = FusionOp::new(&mut ctx, ty, vec![p[0]], FusionKind::Loop);
            let inner = BasicBlock::new(&mut ctx, None, vec![ty]);
            inner.insert_at_back(fusion.body(&ctx), &ctx);
            let arg = inner.deref(&ctx).get_argument(0);
            let add = AddOp::new(&mut ctx, ty, vec![arg, arg]).get_operation();
            let add = append(&ctx, inner, add, "inner.add");
            finish(&mut ctx, inner, add);
            let fusion = append(&ctx, block, fusion.get_operation(), "fusion");
            finish(&mut ctx, block, fusion);
            let substitutions = [(p[0], known(&ctx, integer(&ctx, ElementType::S64, 13)))];
            assert_eq!(
                read_i64(
                    &ctx,
                    evaluate(
                        &ctx,
                        module,
                        fusion,
                        &substitutions,
                        DependencyMode::Recursive
                    )
                    .unwrap()
                ),
                26
            );
        }

        #[test]
        fn fusion_block_argument_request_without_binding_is_unknown_not_captured() {
            let mut ctx = context();
            let module = module(&mut ctx);
            let ty = shape::scalar(&ctx, ElementType::S64).unwrap();
            let (_, block, p) = computation(&mut ctx, module, "main", &[ty], ty);
            let fusion = FusionOp::new(&mut ctx, ty, vec![p[0]], FusionKind::Loop);
            let inner = BasicBlock::new(&mut ctx, None, vec![ty]);
            inner.insert_at_back(fusion.body(&ctx), &ctx);
            let arg = inner.deref(&ctx).get_argument(0);
            finish(&mut ctx, inner, arg);
            let result = append(&ctx, block, fusion.get_operation(), "fusion");
            finish(&mut ctx, block, result);
            assert!(matches!(
                evaluate(&ctx, module, arg, &[], DependencyMode::Recursive)
                    .unwrap()
                    .state(),
                ValueState::Unknown
            ));
        }

        #[test]
        fn conditional_predicate_selects_first_for_true_without_executing_effectful_other_branch() {
            let mut ctx = context();
            let (module, root, p) = conditional_fixture(&mut ctx, true, true);
            let pred = Literal::from_bits(&ctx, p[0].get_type(&ctx), &[1]).unwrap();
            let substitutions = [
                (p[0], known(&ctx, pred)),
                (p[1], known(&ctx, integer(&ctx, ElementType::S64, 31))),
            ];
            assert_eq!(
                read_i64(
                    &ctx,
                    evaluate(
                        &ctx,
                        module,
                        root,
                        &substitutions,
                        DependencyMode::Recursive
                    )
                    .unwrap()
                ),
                31
            );
        }

        #[test]
        fn selected_effectful_conditional_branch_has_typed_effect_error() {
            let mut ctx = context();
            let (module, root, p) = conditional_fixture(&mut ctx, true, true);
            let pred = Literal::from_bits(&ctx, p[0].get_type(&ctx), &[0]).unwrap();
            let substitutions = [
                (p[0], known(&ctx, pred)),
                (p[2], known(&ctx, integer(&ctx, ElementType::S64, 31))),
            ];
            let error = evaluate(
                &ctx,
                module,
                root,
                &substitutions,
                DependencyMode::Recursive,
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
        fn indexed_conditional_out_of_range_and_negative_select_last_branch() {
            let mut ctx = context();
            let (module, root, p) = conditional_fixture(&mut ctx, false, false);
            for index in [-1, 2, 100] {
                let substitutions = [
                    (p[0], known(&ctx, integer(&ctx, ElementType::S32, index))),
                    (p[2], known(&ctx, integer(&ctx, ElementType::S64, 41))),
                ];
                assert_eq!(
                    read_i64(
                        &ctx,
                        evaluate(
                            &ctx,
                            module,
                            root,
                            &substitutions,
                            DependencyMode::Recursive
                        )
                        .unwrap()
                    ),
                    41
                );
            }
        }

        #[test]
        fn unknown_conditional_selector_stays_unknown_without_executing_any_branch() {
            let mut ctx = context();
            let (module, root, _) = conditional_fixture(&mut ctx, true, true);
            assert!(matches!(
                evaluate(&ctx, module, root, &[], DependencyMode::Recursive)
                    .unwrap()
                    .state(),
                ValueState::Unknown
            ));
        }

        #[test]
        fn computation_request_does_not_silently_drop_unused_effects() {
            let mut ctx = context();
            let module = module(&mut ctx);
            let ty = shape::scalar(&ctx, ElementType::S64).unwrap();
            let (comp, block, p) = computation(&mut ctx, module, "main", &[ty], ty);
            let effect =
                CustomCallOp::new(&mut ctx, ty, vec![p[0]], StringAttr::new("effect".into()))
                    .get_operation();
            effect
                .deref_mut(&ctx)
                .attributes
                .set(Identifier::new("hlo_has_side_effect"), Bools(vec![true]));
            append(&ctx, block, effect, "unused.effect");
            finish(&mut ctx, block, p[0]);
            let args = [known(&ctx, integer(&ctx, ElementType::S64, 9))];
            let error = HloEvaluator::new(&ctx, module)
                .unwrap()
                .evaluate(Request::Computation {
                    callee: comp,
                    args: &args,
                })
                .unwrap_err();
            assert!(matches!(
                error.err.downcast_ref::<EvaluationError>(),
                Some(EvaluationError::Effectful { .. })
            ));
        }

        #[test]
        fn partial_call_argument_tuple_can_forward_known_leaf() {
            let mut ctx = context();
            let module = module(&mut ctx);
            let ty = shape::scalar(&ctx, ElementType::S64).unwrap();
            let tuple_ty = shape::tuple(&ctx, vec![ty, ty]).unwrap();
            let (callee, body, p) = computation(&mut ctx, module, "project", &[tuple_ty], ty);
            let gte = GetTupleElementOp::new(&mut ctx, ty, vec![p[0]], Index(1)).get_operation();
            let root = append(&ctx, body, gte, "gte");
            finish(&mut ctx, body, root);
            let (_, main, p) = computation(&mut ctx, module, "main", &[tuple_ty], ty);
            let call = CallOp::new(&mut ctx, ty, vec![p[0]], Callees(vec!["project".into()]))
                .get_operation();
            let call = append(&ctx, main, call, "call");
            finish(&mut ctx, main, call);
            let argument = EvaluatedValue::tuple(
                &ctx,
                tuple_ty,
                vec![
                    EvaluatedValue::unknown(&ctx, ty).unwrap(),
                    known(&ctx, integer(&ctx, ElementType::S64, 71)),
                ],
            )
            .unwrap();
            let result = HloEvaluator::new(&ctx, module)
                .unwrap()
                .evaluate(Request::Computation {
                    callee,
                    args: &[argument],
                })
                .unwrap();
            assert_eq!(read_i64(&ctx, result), 71);
        }
    }
    mod admission {
        use super::*;

        #[test]
        fn effectful_control_predecessor_is_not_ignored() {
            let mut ctx = context();
            let module = module(&mut ctx);
            let ty = shape::scalar(&ctx, ElementType::S64).unwrap();
            let (_, block, p) = computation(&mut ctx, module, "main", &[ty], ty);
            let effect =
                CustomCallOp::new(&mut ctx, ty, vec![p[0]], StringAttr::new("effect".into()))
                    .get_operation();
            effect
                .deref_mut(&ctx)
                .attributes
                .set(Identifier::new("hlo_has_side_effect"), Bools(vec![true]));
            append(&ctx, block, effect, "effect");
            let copy = CopyOp::new(&mut ctx, ty, vec![p[0]]).get_operation();
            copy.deref_mut(&ctx).attributes.set(
                Identifier::new("hlo_control_predecessors"),
                ControlPredecessors(vec!["effect".into()]),
            );
            let copy = append(&ctx, block, copy, "copy");
            finish(&mut ctx, block, copy);
            let substitutions = [(p[0], known(&ctx, integer(&ctx, ElementType::S64, 1)))];
            let error = evaluate(
                &ctx,
                module,
                copy,
                &substitutions,
                DependencyMode::Recursive,
            )
            .unwrap_err();
            assert!(matches!(
                error.err.downcast_ref::<EvaluationError>(),
                Some(EvaluationError::Effectful { .. })
            ));
        }

        #[test]
        fn instruction_substitution_can_supply_effectful_result_without_executing_it() {
            let mut ctx = context();
            let (module, root, _) = conditional_fixture(&mut ctx, true, true);
            let substitutions = [(root, known(&ctx, integer(&ctx, ElementType::S64, 44)))];
            assert_eq!(
                read_i64(
                    &ctx,
                    evaluate(&ctx, module, root, &substitutions, DependencyMode::Shallow).unwrap()
                ),
                44
            );
        }

        #[test]
        fn malformed_instruction_fails_native_verification_before_evaluation() {
            let mut ctx = context();
            let f = math(&mut ctx);
            let add = f.sum.defining_op().unwrap();
            add.deref_mut(&ctx).attributes.set(
                Identifier::new("hlo_control_predecessors"),
                ControlPredecessors(vec!["missing".into()]),
            );
            assert!(evaluate(&ctx, f.module, f.sum, &[], DependencyMode::Recursive).is_err());
        }

        #[test]
        fn detached_and_other_module_values_are_rejected() {
            let mut ctx = context();
            let f = math(&mut ctx);
            let ty = shape::scalar(&ctx, ElementType::S64).unwrap();
            let detached = ParameterOp::new(&mut ctx, ty, vec![], Index(0)).result(&ctx);
            assert!(evaluate(&ctx, f.module, detached, &[], DependencyMode::Recursive).is_err());
            let other = math(&mut ctx);
            assert!(evaluate(&ctx, f.module, other.sum, &[], DependencyMode::Recursive).is_err());
        }

        #[test]
        fn independent_frame_stack_guard_returns_typed_recursion() {
            let mut ctx = context();
            let f = math(&mut ctx);
            let mut session = Session::new(&ctx, f.module).unwrap();
            session.stack.push(f.computation);
            let args = [known(&ctx, integer(&ctx, ElementType::S64, 1))];
            let error = session.computation(f.computation, &args, &[]).unwrap_err();
            assert!(matches!(
                error.err.downcast_ref::<EvaluationError>(),
                Some(EvaluationError::Recursion)
            ));
        }

        #[test]
        fn independent_value_cycle_guard_is_path_specific_and_recoverable() {
            let mut ctx = context();
            let f = math(&mut ctx);
            let mut session = Session::new(&ctx, f.module).unwrap();
            let mut frame = Frame::new(&ctx, f.computation).unwrap();
            frame.active.insert(Key {
                value: f.sum,
                path: vec![],
            });
            let error = session
                .eval(&mut frame, f.sum, &[], DependencyMode::Recursive)
                .unwrap_err();
            assert!(matches!(
                error.err.downcast_ref::<EvaluationError>(),
                Some(EvaluationError::Recursion)
            ));
        }

        #[test]
        fn computation_recursion_is_detected_without_unbounded_host_recursion() {
            let mut ctx = context();
            let module = module(&mut ctx);
            let ty = shape::scalar(&ctx, ElementType::S64).unwrap();
            let (comp, block, p) = computation(&mut ctx, module, "main", &[ty], ty);
            let call =
                CallOp::new(&mut ctx, ty, vec![p[0]], Callees(vec!["main".into()])).get_operation();
            let root = append(&ctx, block, call, "recursive");
            finish(&mut ctx, block, root);
            let arg = known(&ctx, integer(&ctx, ElementType::S64, 1));
            let result = HloEvaluator::new(&ctx, module)
                .unwrap()
                .evaluate(Request::Computation {
                    callee: comp,
                    args: &[arg],
                });
            // The graph verifier may reject a recursive call graph before the engine's
            // independent frame-stack guard; either boundary must be recoverable.
            assert!(result.is_err());
        }
    }
    mod movement {
        use super::*;

        #[test]
        fn constant_all_32_formats_preserves_exact_raw_storage_without_float_evaluation() {
            for &element in ElementType::ALL {
                let mut ctx = context();
                let module = module(&mut ctx);
                let ty = shape::scalar(&ctx, element).unwrap();
                let bits = if element == ElementType::Pred {
                    1
                } else if element.bit_width() == 128 {
                    u128::MAX
                } else {
                    (1u128 << element.bit_width()) - 1
                };
                let literal = Literal::from_bits(&ctx, ty, &[bits]).unwrap();
                let expected = literal.clone();
                let (_, block, _) = computation(&mut ctx, module, "main", &[], ty);
                let c = constant(&mut ctx, block, literal, "c");
                finish(&mut ctx, block, c);
                let result = evaluate(&ctx, module, c, &[], DependencyMode::Shallow)
                    .unwrap()
                    .into_literal(&ctx)
                    .unwrap();
                assert_eq!(result.storage, expected.storage);
            }
        }

        #[test]
        fn output_layout_is_declared_layout_not_first_operand_layout() {
            let mut ctx = context();
            let module = module(&mut ctx);
            let base = shape::array(&ctx, ElementType::S32, vec![2, 2]).unwrap();
            let input_ty = shape::with_default_layout(&ctx, base).unwrap();
            let target = shape::with_layout(
                &ctx,
                base,
                Layout {
                    minor_to_major: vec![0, 1],
                    ..Layout::descending(2)
                },
            )
            .unwrap();
            let (_, block, _) = computation(&mut ctx, module, "main", &[], target);
            let a_literal = Literal::from_bits(&ctx, input_ty, &[1, 2, 3, 4]).unwrap();
            let b_literal = Literal::from_bits(&ctx, input_ty, &[10, 20, 30, 40]).unwrap();
            let a = constant(&mut ctx, block, a_literal, "a");
            let b = constant(&mut ctx, block, b_literal, "b");
            let add = AddOp::new(&mut ctx, target, vec![a, b]).get_operation();
            let add = append(&ctx, block, add, "add");
            finish(&mut ctx, block, add);
            let result = evaluate(&ctx, module, add, &[], DependencyMode::Shallow)
                .unwrap()
                .into_literal(&ctx)
                .unwrap();
            assert_eq!(result.shape, target);
            assert_eq!(result.bits(&ctx).unwrap(), vec![11, 33, 22, 44]);
        }

        #[test]
        fn reshape_transpose_broadcast_chain_reuses_raw_movement_and_declared_layout() {
            let mut ctx = context();
            let module = module(&mut ctx);
            let input = shape::array(&ctx, ElementType::F32, vec![2, 2]).unwrap();
            let vector = shape::array(&ctx, ElementType::F32, vec![4]).unwrap();
            let output = shape::array(&ctx, ElementType::F32, vec![2, 4]).unwrap();
            let (_, block, _) = computation(&mut ctx, module, "main", &[], output);
            let literal = Literal::from_bits(
                &ctx,
                input,
                &[0x7fa0_0001, 0x8000_0000, 0x3f80_0000, 0xffc0_0123],
            )
            .unwrap();
            let c = constant(&mut ctx, block, literal, "c");
            let t = TransposeOp::new(&mut ctx, input, vec![c], Axes(vec![1, 0])).get_operation();
            let t = append(&ctx, block, t, "transpose");
            let r = ReshapeOp::new(&mut ctx, vector, vec![t]).get_operation();
            let r = append(&ctx, block, r, "reshape");
            let b = BroadcastOp::new(&mut ctx, output, vec![r], Axes(vec![1])).get_operation();
            let b = append(&ctx, block, b, "broadcast");
            finish(&mut ctx, block, b);
            let result = evaluate(&ctx, module, b, &[], DependencyMode::Recursive)
                .unwrap()
                .into_literal(&ctx)
                .unwrap();
            assert_eq!(
                result.bits(&ctx).unwrap(),
                vec![
                    0x7fa0_0001,
                    0x3f80_0000,
                    0x8000_0000,
                    0xffc0_0123,
                    0x7fa0_0001,
                    0x3f80_0000,
                    0x8000_0000,
                    0xffc0_0123
                ]
            );
        }

        #[test]
        fn scalar_broadcast_select_condition_preserves_nondefault_result_layout() {
            let mut ctx = context();
            let module = module(&mut ctx);
            let pred = shape::scalar(&ctx, ElementType::Pred).unwrap();
            let base = shape::array(&ctx, ElementType::S32, vec![2, 2]).unwrap();
            let target = shape::with_layout(
                &ctx,
                base,
                Layout {
                    minor_to_major: vec![0, 1],
                    ..Layout::descending(2)
                },
            )
            .unwrap();
            let (_, block, _) = computation(&mut ctx, module, "main", &[], target);
            let a_literal = Literal::from_bits(&ctx, base, &[1, 2, 3, 4]).unwrap();
            let b_literal = Literal::from_bits(&ctx, base, &[10, 20, 30, 40]).unwrap();
            let predicate = Literal::from_bits(&ctx, pred, &[1]).unwrap();
            let a = constant(&mut ctx, block, a_literal, "a");
            let b = constant(&mut ctx, block, b_literal, "b");
            let p = constant(&mut ctx, block, predicate, "p");
            let op = SelectOp::new(&mut ctx, target, vec![p, a, b]).get_operation();
            let root = append(&ctx, block, op, "select");
            finish(&mut ctx, block, root);
            let result = evaluate(&ctx, module, root, &[], DependencyMode::Recursive)
                .unwrap()
                .into_literal(&ctx)
                .unwrap();
            assert_eq!(result.shape, target);
            assert_eq!(result.bits(&ctx).unwrap(), vec![1, 3, 2, 4]);
        }
    }
    mod numerical {
        use super::*;

        fn rounding_fixture(ctx: &mut Context) -> (Ptr<Operation>, Value, Value, TypeHandle) {
            let module = module(ctx);
            let ty = shape::array(ctx, ElementType::F32, vec![4]).unwrap();
            let (_, block, parameters) = computation(ctx, module, "main", &[ty], ty);
            let round = RoundNearestEvenOp::new(ctx, ty, vec![parameters[0]]).get_operation();
            let root = append(ctx, block, round, "round");
            finish(ctx, block, root);
            (module, root, parameters[0], ty)
        }

        fn float_context(
            rounding: scalar_semantics::EvaluationRoundingMode,
        ) -> EvaluationFloatContext {
            EvaluationFloatContext {
                rounding,
                flush_to_zero: false,
                denormals_are_zero: false,
            }
        }

        fn rounding_result(
            ctx: &Context,
            module: Ptr<Operation>,
            root: Value,
            parameter: Value,
            ty: TypeHandle,
            bits: &[u128],
            context: Option<EvaluationFloatContext>,
        ) -> pliron::result::Result<EvaluatedValue> {
            let input = known(ctx, Literal::from_bits(ctx, ty, bits)?);
            let evaluator = match context {
                Some(context) => HloEvaluator::new(ctx, module)?.with_float_context(context),
                None => HloEvaluator::new(ctx, module)?,
            };
            evaluator.evaluate(Request::Instruction {
                value: root,
                substitutions: &[(parameter, input)],
                dependency_mode: DependencyMode::Recursive,
            })
        }

        #[test]
        fn round_nearest_even_requires_context_and_preserves_raw_results() {
            use scalar_semantics::EvaluationRoundingMode::*;

            let mut ctx = context();
            let (module, root, parameter, ty) = rounding_fixture(&mut ctx);
            let ties_zero_nan = [
                0x4020_0000, // 2.5 -> 2
                0xc020_0000, // -2.5 -> -2
                0xbf00_0000, // -0.5 -> -0
                0x7f80_1234, // signaling NaN -> quiet NaN
            ];
            let missing = rounding_result(&ctx, module, root, parameter, ty, &ties_zero_nan, None)
                .unwrap_err();
            assert!(matches!(
                missing.err.downcast_ref::<EvaluationError>(),
                Some(EvaluationError::MissingFloatContext {
                    opcode: Opcode::RoundNearestEven
                })
            ));

            let context = float_context(NearestEven);
            let result = rounding_result(
                &ctx,
                module,
                root,
                parameter,
                ty,
                &ties_zero_nan,
                Some(context),
            )
            .unwrap()
            .into_literal(&ctx)
            .unwrap();
            assert_eq!(
                result.bits(&ctx).unwrap(),
                [0x4000_0000, 0xc000_0000, 0x8000_0000, 0x7fc0_1234]
            );
        }

        #[test]
        fn round_nearest_even_rejects_non_nearest_context_without_leaking_state() {
            use scalar_semantics::EvaluationRoundingMode::*;

            let mut ctx = context();
            let (module, root, parameter, ty) = rounding_fixture(&mut ctx);
            let bits = [0x3f00_0000; 4];
            let nearest = HloEvaluator::new(&ctx, module)
                .unwrap()
                .with_float_context(float_context(NearestEven));
            let toward_zero = HloEvaluator::new(&ctx, module)
                .unwrap()
                .with_float_context(float_context(TowardZero));
            for rounding in [TowardZero, TowardPositive, TowardNegative] {
                let error = rounding_result(
                    &ctx,
                    module,
                    root,
                    parameter,
                    ty,
                    &bits,
                    Some(float_context(rounding)),
                )
                .unwrap_err();
                assert!(matches!(
                    error.err.downcast_ref::<crate::Diagnostic>(),
                    Some(crate::Diagnostic::InvalidShape { .. })
                ));
            }

            for (input_bits, expected) in [
                ([0x3f00_0000; 4], [0x0000_0000; 4]),
                ([0x3fc0_0000; 4], [0x4000_0000; 4]),
            ] {
                let input = known(&ctx, Literal::from_bits(&ctx, ty, &input_bits).unwrap());
                let result = nearest
                    .evaluate(Request::Instruction {
                        value: root,
                        substitutions: &[(parameter, input)],
                        dependency_mode: DependencyMode::Recursive,
                    })
                    .unwrap()
                    .into_literal(&ctx)
                    .unwrap();
                assert_eq!(result.bits(&ctx).unwrap(), expected);
            }
            let toward_zero_error = toward_zero
                .evaluate(Request::Instruction {
                    value: root,
                    substitutions: &[(
                        parameter,
                        known(&ctx, Literal::from_bits(&ctx, ty, &bits).unwrap()),
                    )],
                    dependency_mode: DependencyMode::Recursive,
                })
                .unwrap_err();
            assert!(matches!(
                toward_zero_error.err.downcast_ref::<crate::Diagnostic>(),
                Some(crate::Diagnostic::InvalidShape { .. })
            ));
            let result = nearest
                .evaluate(Request::Instruction {
                    value: root,
                    substitutions: &[(
                        parameter,
                        known(&ctx, Literal::from_bits(&ctx, ty, &bits).unwrap()),
                    )],
                    dependency_mode: DependencyMode::Recursive,
                })
                .unwrap()
                .into_literal(&ctx)
                .unwrap();
            assert_eq!(result.bits(&ctx).unwrap(), [0x0000_0000; 4]);
        }

        #[test]
        fn round_nearest_even_context_reaches_nested_calls() {
            use scalar_semantics::EvaluationRoundingMode::NearestEven;

            let mut ctx = context();
            let module = module(&mut ctx);
            let ty = shape::scalar(&ctx, ElementType::F32).unwrap();
            let (_, callee_block, callee_parameters) =
                computation(&mut ctx, module, "rounder", &[ty], ty);
            let round =
                RoundNearestEvenOp::new(&mut ctx, ty, vec![callee_parameters[0]]).get_operation();
            let callee_root = append(&ctx, callee_block, round, "round");
            finish(&mut ctx, callee_block, callee_root);

            let (_, caller_block, caller_parameters) =
                computation(&mut ctx, module, "main", &[ty], ty);
            let call = CallOp::new(
                &mut ctx,
                ty,
                vec![caller_parameters[0]],
                Callees(vec!["rounder".into()]),
            )
            .get_operation();
            let root = append(&ctx, caller_block, call, "call");
            finish(&mut ctx, caller_block, root);
            verification::verify(&ctx, module).unwrap();

            let argument = known(&ctx, Literal::from_bits(&ctx, ty, &[0x3f00_0000]).unwrap());
            let result = HloEvaluator::new(&ctx, module)
                .unwrap()
                .with_float_context(float_context(NearestEven))
                .evaluate(Request::Instruction {
                    value: root,
                    substitutions: &[(caller_parameters[0], argument)],
                    dependency_mode: DependencyMode::Recursive,
                })
                .unwrap()
                .into_literal(&ctx)
                .unwrap();
            assert_eq!(result.bits(&ctx).unwrap(), [0x0000_0000]);
        }

        #[test]
        fn u64_arithmetic_and_compare_do_not_round_through_host_float() {
            let mut ctx = context();
            let module = module(&mut ctx);
            let ty = shape::scalar(&ctx, ElementType::U64).unwrap();
            let pred = shape::scalar(&ctx, ElementType::Pred).unwrap();
            let (_, block, _) = computation(&mut ctx, module, "main", &[], pred);
            let a_literal = integer(&ctx, ElementType::U64, 9_007_199_254_740_992);
            let b_literal = integer(&ctx, ElementType::U64, 1);
            let a = constant(&mut ctx, block, a_literal, "a");
            let one = constant(&mut ctx, block, b_literal, "one");
            let sum = AddOp::new(&mut ctx, ty, vec![a, one]).get_operation();
            let sum = append(&ctx, block, sum, "sum");
            let compare = CompareOp::new(
                &mut ctx,
                pred,
                vec![a, sum],
                Direction::Lt,
                ComparisonOrder::Total,
            )
            .get_operation();
            let compare = append(&ctx, block, compare, "compare");
            finish(&mut ctx, block, compare);
            let result = evaluate(&ctx, module, compare, &[], DependencyMode::Recursive)
                .unwrap()
                .into_literal(&ctx)
                .unwrap();
            assert!(literal_semantics::scalar_predicate(&ctx, &result).unwrap());
        }

        #[test]
        fn static_empty_supported_arithmetic_produces_empty_not_one_zero_element() {
            let mut ctx = context();
            let module = module(&mut ctx);
            let ty = shape::array(&ctx, ElementType::S64, vec![0, 2]).unwrap();
            let (_, block, p) = computation(&mut ctx, module, "main", &[ty], ty);
            let add = AddOp::new(&mut ctx, ty, vec![p[0], p[0]]).get_operation();
            let add = append(&ctx, block, add, "add");
            finish(&mut ctx, block, add);
            let substitutions = [(
                p[0],
                known(&ctx, Literal::from_bits(&ctx, ty, &[]).unwrap()),
            )];
            let result = evaluate(&ctx, module, add, &substitutions, DependencyMode::Recursive)
                .unwrap()
                .into_literal(&ctx)
                .unwrap();
            assert!(result.storage.is_empty());
        }

        #[test]
        fn empty_subbyte_mulhi_does_not_sneak_past_unsupported_type_admission() {
            let mut ctx = context();
            let module = module(&mut ctx);
            let ty = shape::array(&ctx, ElementType::S4, vec![0]).unwrap();
            let (_, block, p) = computation(&mut ctx, module, "main", &[ty], ty);
            let mul = MulhiOp::new(&mut ctx, ty, vec![p[0], p[0]]).get_operation();
            let mul = append(&ctx, block, mul, "mulhi");
            finish(&mut ctx, block, mul);
            let substitutions = [(
                p[0],
                known(&ctx, Literal::from_bits(&ctx, ty, &[]).unwrap()),
            )];
            assert!(
                evaluate(&ctx, module, mul, &substitutions, DependencyMode::Recursive).is_err()
            );
        }

        #[test]
        fn low_precision_f4_f8_arithmetic_preserves_each_format_encoding() {
            for (element, one, two) in [
                (ElementType::F4E2M1FN, 0x2, 0x4),
                (ElementType::F8E5M2, 0x3c, 0x40),
                (ElementType::F8E4M3, 0x38, 0x40),
                (ElementType::F8E4M3FN, 0x38, 0x40),
                (ElementType::F8E4M3B11FNUZ, 0x60, 0x68),
                (ElementType::F8E3M4, 0x30, 0x40),
                (ElementType::F8E5M2FNUZ, 0x44, 0x48),
                (ElementType::F8E4M3FNUZ, 0x48, 0x50),
                (ElementType::F8E8M0FNU, 0x7f, 0x80),
            ] {
                let mut ctx = context();
                let module = module(&mut ctx);
                let ty = shape::array(&ctx, element, vec![1]).unwrap();
                let (_, block, _) = computation(&mut ctx, module, "main", &[], ty);
                let literal = Literal::from_bits(&ctx, ty, &[one]).unwrap();
                let a = constant(&mut ctx, block, literal.clone(), "a");
                let b = constant(&mut ctx, block, literal, "b");
                let add = AddOp::new(&mut ctx, ty, vec![a, b]).get_operation();
                let add = append(&ctx, block, add, "add");
                finish(&mut ctx, block, add);
                let result = evaluate(&ctx, module, add, &[], DependencyMode::Recursive)
                    .unwrap()
                    .into_literal(&ctx)
                    .unwrap();
                assert_eq!(result.bits(&ctx).unwrap(), [two], "{element:?}");
            }
        }

        #[test]
        fn floating_signed_zero_nan_and_unimplemented_math_keep_typed_contracts() {
            let mut ctx = context();
            let module_a = module(&mut ctx);
            let f8 = shape::array(&ctx, ElementType::F8E5M2, vec![2]).unwrap();
            let pred = shape::array(&ctx, ElementType::Pred, vec![2]).unwrap();
            let (_, block, _) = computation(&mut ctx, module_a, "main", &[], pred);
            let input_literal = Literal::from_bits(&ctx, f8, &[0x80, 0x7d]).unwrap();
            let input = constant(&mut ctx, block, input_literal, "input");
            let finite = IsFiniteOp::new(&mut ctx, pred, vec![input]).get_operation();
            let finite = append(&ctx, block, finite, "finite");
            finish(&mut ctx, block, finite);
            let result = evaluate(&ctx, module_a, finite, &[], DependencyMode::Recursive)
                .unwrap()
                .into_literal(&ctx)
                .unwrap();
            assert_eq!(result.bits(&ctx).unwrap(), [1, 0]);

            let mut ctx = context();
            let module_b = module(&mut ctx);
            let ty = shape::scalar(&ctx, ElementType::F8E5M2).unwrap();
            let (_, block, _) = computation(&mut ctx, module_b, "main", &[], ty);
            let input_literal = Literal::from_bits(&ctx, ty, &[0x80]).unwrap();
            let input = constant(&mut ctx, block, input_literal, "negative_zero");
            let negate = NegateOp::new(&mut ctx, ty, vec![input]).get_operation();
            let negate = append(&ctx, block, negate, "negate");
            finish(&mut ctx, block, negate);
            let result = evaluate(&ctx, module_b, negate, &[], DependencyMode::Recursive)
                .unwrap()
                .into_literal(&ctx)
                .unwrap();
            assert_eq!(result.bits(&ctx).unwrap(), [0]);

            let mut ctx = context();
            let module_c = module(&mut ctx);
            let ty = shape::scalar(&ctx, ElementType::F32).unwrap();
            let (_, block, _) = computation(&mut ctx, module_c, "main", &[], ty);
            let a_literal = Literal::from_bits(&ctx, ty, &[0x3f80_0000]).unwrap();
            let b_literal = Literal::from_bits(&ctx, ty, &[0x4000_0000]).unwrap();
            let a = constant(&mut ctx, block, a_literal, "one");
            let b = constant(&mut ctx, block, b_literal, "two");
            let power =
                crate::ops::elementwise::PowerOp::new(&mut ctx, ty, vec![a, b]).get_operation();
            let power = append(&ctx, block, power, "power");
            finish(&mut ctx, block, power);
            let error =
                evaluate(&ctx, module_c, power, &[], DependencyMode::Recursive).unwrap_err();
            assert!(matches!(
                error.err.downcast_ref::<EvaluationError>(),
                Some(EvaluationError::UnsupportedValue {
                    opcode: Opcode::Power,
                    ..
                })
            ));
        }

        #[test]
        fn complex_construction_projection_cast_arithmetic_and_equality_are_supported() {
            let mut ctx = context();
            let module = module(&mut ctx);
            let f32_ty = shape::scalar(&ctx, ElementType::F32).unwrap();
            let f64_ty = shape::scalar(&ctx, ElementType::F64).unwrap();
            let c64_ty = shape::scalar(&ctx, ElementType::C64).unwrap();
            let c128_ty = shape::scalar(&ctx, ElementType::C128).unwrap();
            let pred_ty = shape::scalar(&ctx, ElementType::Pred).unwrap();
            let output_ty = shape::tuple(
                &ctx,
                vec![
                    c64_ty, f32_ty, f32_ty, c128_ty, c128_ty, pred_ty, c64_ty, c128_ty, f64_ty,
                    f64_ty,
                ],
            )
            .unwrap();
            let (_, block, _) = computation(&mut ctx, module, "main", &[], output_ty);
            let real_literal = Literal::from_bits(&ctx, f32_ty, &[0x3f80_0000]).unwrap();
            let imag_literal = Literal::from_bits(&ctx, f32_ty, &[0xc000_0000]).unwrap();
            let real = constant(&mut ctx, block, real_literal, "real");
            let imag = constant(&mut ctx, block, imag_literal, "imag");
            let complex = ComplexOp::new(&mut ctx, c64_ty, vec![real, imag]).get_operation();
            let complex = append(&ctx, block, complex, "complex");
            let real_part = RealOp::new(&mut ctx, f32_ty, vec![complex]).get_operation();
            let real_part = append(&ctx, block, real_part, "real_part");
            let imag_part = ImagOp::new(&mut ctx, f32_ty, vec![complex]).get_operation();
            let imag_part = append(&ctx, block, imag_part, "imag_part");
            let widened = ConvertOp::new(&mut ctx, c128_ty, vec![complex]).get_operation();
            let widened = append(&ctx, block, widened, "widened");
            let same = CompareOp::new(
                &mut ctx,
                pred_ty,
                vec![complex, complex],
                Direction::Eq,
                ComparisonOrder::Partial,
            )
            .get_operation();
            let same = append(&ctx, block, same, "same");
            let other_literal = Literal::from_bits(&ctx, c128_ty, &[0]).unwrap();
            let other = constant(&mut ctx, block, other_literal, "other");
            let sum = AddOp::new(&mut ctx, c128_ty, vec![widened, other]).get_operation();
            let sum = append(&ctx, block, sum, "sum");
            let c64_sum = AddOp::new(&mut ctx, c64_ty, vec![complex, complex]).get_operation();
            let c64_sum = append(&ctx, block, c64_sum, "c64_sum");
            let real64_literal =
                Literal::from_bits(&ctx, f64_ty, &[0x3ff0_0000_0000_0000]).unwrap();
            let imag64_literal =
                Literal::from_bits(&ctx, f64_ty, &[0xc000_0000_0000_0000]).unwrap();
            let real64 = constant(&mut ctx, block, real64_literal, "real64");
            let imag64 = constant(&mut ctx, block, imag64_literal, "imag64");
            let c128 = ComplexOp::new(&mut ctx, c128_ty, vec![real64, imag64]).get_operation();
            let c128 = append(&ctx, block, c128, "c128");
            let real64_part = RealOp::new(&mut ctx, f64_ty, vec![c128]).get_operation();
            let real64_part = append(&ctx, block, real64_part, "real64_part");
            let imag64_part = ImagOp::new(&mut ctx, f64_ty, vec![c128]).get_operation();
            let imag64_part = append(&ctx, block, imag64_part, "imag64_part");
            let tuple = TupleOp::new(
                &mut ctx,
                output_ty,
                vec![
                    complex,
                    real_part,
                    imag_part,
                    widened,
                    sum,
                    same,
                    c64_sum,
                    c128,
                    real64_part,
                    imag64_part,
                ],
            )
            .get_operation();
            let tuple = append(&ctx, block, tuple, "result");
            finish(&mut ctx, block, tuple);
            let result = evaluate(&ctx, module, tuple, &[], DependencyMode::Recursive)
                .unwrap()
                .into_literal(&ctx)
                .unwrap();
            assert_eq!(
                result.sub_literal(&ctx, &[0]).unwrap().bits(&ctx).unwrap(),
                [0xc000_0000_3f80_0000]
            );
            assert_eq!(
                result.sub_literal(&ctx, &[1]).unwrap().bits(&ctx).unwrap(),
                [0x3f80_0000]
            );
            assert_eq!(
                result.sub_literal(&ctx, &[2]).unwrap().bits(&ctx).unwrap(),
                [0xc000_0000]
            );
            assert_eq!(
                result.sub_literal(&ctx, &[3]).unwrap().bits(&ctx).unwrap(),
                [0xc000_0000_0000_0000_3ff0_0000_0000_0000]
            );
            assert_eq!(
                result.sub_literal(&ctx, &[4]).unwrap().bits(&ctx).unwrap(),
                [0xc000_0000_0000_0000_3ff0_0000_0000_0000]
            );
            assert_eq!(
                result.sub_literal(&ctx, &[5]).unwrap().bits(&ctx).unwrap(),
                [1]
            );
            assert_eq!(
                result.sub_literal(&ctx, &[6]).unwrap().bits(&ctx).unwrap(),
                [0xc080_0000_4000_0000]
            );
            assert_eq!(
                result.sub_literal(&ctx, &[7]).unwrap().bits(&ctx).unwrap(),
                [0xc000_0000_0000_0000_3ff0_0000_0000_0000]
            );
            assert_eq!(
                result.sub_literal(&ctx, &[8]).unwrap().bits(&ctx).unwrap(),
                [0x3ff0_0000_0000_0000]
            );
            assert_eq!(
                result.sub_literal(&ctx, &[9]).unwrap().bits(&ctx).unwrap(),
                [0xc000_0000_0000_0000]
            );
        }

        #[test]
        fn dynamic_arithmetic_is_explicitly_incomplete_but_raw_copy_preserves_sizes() {
            let mut ctx = context();
            let module = module(&mut ctx);
            let ty =
                shape::dynamic_array(&ctx, ElementType::S32, vec![Dimension::Bounded(2)]).unwrap();
            let (_, block, p) = computation(&mut ctx, module, "main", &[ty], ty);
            let add = AddOp::new(&mut ctx, ty, vec![p[0], p[0]]).get_operation();
            let add = append(&ctx, block, add, "add");
            let copy = CopyOp::new(&mut ctx, ty, vec![p[0]]).get_operation();
            let copy = append(&ctx, block, copy, "copy");
            finish(&mut ctx, block, copy);
            let literal = Literal::from_dynamic_storage_bytes(
                &ctx,
                ty,
                vec![1, 0, 0, 0, 0xff, 0xee, 0xdd, 0xcc],
                vec![1],
            )
            .unwrap();
            let substitutions = [(p[0], known(&ctx, literal.clone()))];
            assert!(
                evaluate(&ctx, module, add, &substitutions, DependencyMode::Recursive).is_err()
            );
            let result = evaluate(
                &ctx,
                module,
                copy,
                &substitutions,
                DependencyMode::Recursive,
            )
            .unwrap()
            .into_literal(&ctx)
            .unwrap();
            assert_eq!(result.storage, literal.storage);
            assert_eq!(result.dynamic_sizes, vec![1]);
        }

        #[test]
        fn integer_convert_and_scalar_select_use_checked_qualified_kernel() {
            let mut ctx = context();
            let module = module(&mut ctx);
            let u8_ty = shape::scalar(&ctx, ElementType::U8).unwrap();
            let pred = shape::scalar(&ctx, ElementType::Pred).unwrap();
            let (_, block, _) = computation(&mut ctx, module, "main", &[], u8_ty);
            let input_literal = integer(&ctx, ElementType::S32, -1);
            let input = constant(&mut ctx, block, input_literal, "input");
            let convert = ConvertOp::new(&mut ctx, u8_ty, vec![input]).get_operation();
            let convert = append(&ctx, block, convert, "convert");
            let false_literal = integer(&ctx, ElementType::U8, 0);
            let condition_literal = Literal::from_bits(&ctx, pred, &[1]).unwrap();
            let false_value = constant(&mut ctx, block, false_literal, "false");
            let condition = constant(&mut ctx, block, condition_literal, "pred");
            let select = SelectOp::new(&mut ctx, u8_ty, vec![condition, convert, false_value])
                .get_operation();
            let select = append(&ctx, block, select, "select");
            finish(&mut ctx, block, select);
            assert_eq!(
                read_i64(
                    &ctx,
                    evaluate(&ctx, module, select, &[], DependencyMode::Recursive).unwrap()
                ),
                255
            );
        }

        #[test]
        fn tensor_bitcast_split_is_explicitly_unsupported_not_elementwise_truncated() {
            let mut ctx = context();
            let module = module(&mut ctx);
            let input = shape::scalar(&ctx, ElementType::U16).unwrap();
            let output = shape::array(&ctx, ElementType::U8, vec![2]).unwrap();
            let (_, block, p) = computation(&mut ctx, module, "main", &[input], output);
            let cast = BitcastConvertOp::new(&mut ctx, output, vec![p[0]]).get_operation();
            let cast = append(&ctx, block, cast, "cast");
            finish(&mut ctx, block, cast);
            let substitutions = [(p[0], known(&ctx, integer(&ctx, ElementType::U16, 0x1234)))];
            assert!(
                evaluate(
                    &ctx,
                    module,
                    cast,
                    &substitutions,
                    DependencyMode::Recursive
                )
                .is_err()
            );
        }

        #[test]
        fn transpose_all_32_formats_retains_payloads_without_numeric_dispatch() {
            for &element in ElementType::ALL {
                let mut ctx = context();
                let module = module(&mut ctx);
                let input = shape::array(&ctx, element, vec![1, 2]).unwrap();
                let output = shape::array(&ctx, element, vec![2, 1]).unwrap();
                let (_, block, _) = computation(&mut ctx, module, "main", &[], output);
                let bits = if element == ElementType::Pred {
                    1
                } else if element.bit_width() == 128 {
                    u128::MAX
                } else {
                    (1u128 << element.bit_width()) - 1
                };
                let literal = Literal::from_bits(&ctx, input, &[0, bits]).unwrap();
                let c = constant(&mut ctx, block, literal, "c");
                let transpose =
                    TransposeOp::new(&mut ctx, output, vec![c], Axes(vec![1, 0])).get_operation();
                let root = append(&ctx, block, transpose, "transpose");
                finish(&mut ctx, block, root);
                let result = evaluate(&ctx, module, root, &[], DependencyMode::Recursive)
                    .unwrap()
                    .into_literal(&ctx)
                    .unwrap();
                assert_eq!(result.bits(&ctx).unwrap(), vec![0, bits]);
            }
        }

        #[test]
        fn unknown_numeric_dependency_propagates_unknown_without_allocating_result() {
            let mut ctx = context();
            let f = math(&mut ctx);
            let result = evaluate(&ctx, f.module, f.sum, &[], DependencyMode::Recursive).unwrap();
            assert_eq!(result.shape(), f.sum.get_type(&ctx));
            assert!(matches!(result.state(), ValueState::Unknown));
        }

        #[test]
        fn kernel_arithmetic_matches_scalar_for_each_integer_format() {
            for &element in ElementType::ALL {
                if !element.is_integer() {
                    continue;
                }
                let mut ctx = context();
                let module = module(&mut ctx);
                let ty = shape::scalar(&ctx, element).unwrap();
                let (_, block, p) = computation(&mut ctx, module, "main", &[ty, ty], ty);
                let add = AddOp::new(&mut ctx, ty, vec![p[0], p[1]]).get_operation();
                let root = append(&ctx, block, add, "add");
                finish(&mut ctx, block, root);
                let a =
                    Scalar::from_bits(element, u128::from(u64::MAX >> (64 - element.bit_width())))
                        .unwrap();
                let b = Scalar::from_bits(element, 1).unwrap();
                let substitutions = [
                    (p[0], known(&ctx, a.to_literal(&ctx).unwrap())),
                    (p[1], known(&ctx, b.to_literal(&ctx).unwrap())),
                ];
                let result = evaluate(
                    &ctx,
                    module,
                    root,
                    &substitutions,
                    DependencyMode::Recursive,
                )
                .unwrap()
                .into_literal(&ctx)
                .unwrap();
                assert_eq!(
                    Scalar::read(&ctx, &result, &[]).unwrap(),
                    scalar_semantics::binary(Opcode::Add, a, b).unwrap()
                );
            }
        }
    }
    use super::*;
    use pliron::identifier::Identifier;

    mod indexed_movement {
        use super::*;

        use crate::{
            attributes::Slice,
            literal::semantics::read_element,
            ops::movement::{ConcatenateOp, ReverseOp, SliceOp},
        };

        fn geometry(starts: &[i64], limits: &[i64], strides: &[i64]) -> Slice {
            Slice {
                starts: Axes(starts.to_vec()),

                limits: Axes(limits.to_vec()),

                strides: Axes(strides.to_vec()),
            }
        }

        #[test]

        fn reverse_and_slice_all_32_formats_preserve_payloads_across_declared_layouts() {
            for &element in ElementType::ALL {
                let mut ctx = context();

                let module = module(&mut ctx);

                let source_shape = shape::array(&ctx, element, vec![3, 4]).unwrap();

                let source_shape = shape::with_layout(
                    &ctx,
                    source_shape,
                    Layout {
                        minor_to_major: vec![0, 1],

                        ..Default::default()
                    },
                )
                .unwrap();

                let reversed_shape = shape::array(&ctx, element, vec![3, 4]).unwrap();

                let result_shape = shape::array(&ctx, element, vec![2, 2]).unwrap();

                let result_shape = shape::with_layout(
                    &ctx,
                    result_shape,
                    Layout {
                        minor_to_major: vec![0, 1],

                        ..Default::default()
                    },
                )
                .unwrap();

                let (_, block, _) = computation(&mut ctx, module, "main", &[], result_shape);

                let width = element.bit_width().div_ceil(8) as usize;

                let bytes = (0..12)
                    .flat_map(|index| {
                        (0..width).map(move |byte| {
                            if element == ElementType::Pred {
                                (index % 2) as u8
                            } else {
                                (0x80 + index * 17 + byte) as u8
                            }
                        })
                    })
                    .collect::<Vec<_>>();

                let source = Literal::from_storage_bytes(&ctx, source_shape, bytes).unwrap();

                let input = constant(&mut ctx, block, source.clone(), "input");

                let reverse =
                    ReverseOp::new(&mut ctx, reversed_shape, vec![input], Axes(vec![0, 1]));

                let reverse = append(&ctx, block, reverse.get_operation(), "reverse");

                let slice = SliceOp::new(
                    &mut ctx,
                    result_shape,
                    vec![reverse],
                    geometry(&[0, 1], &[3, 4], &[2, 2]),
                );

                let slice = append(&ctx, block, slice.get_operation(), "slice");

                finish(&mut ctx, block, slice);

                let before = module.disp(&ctx).to_string();

                let result = evaluate(&ctx, module, slice, &[], DependencyMode::Recursive)
                    .unwrap()
                    .into_literal(&ctx)
                    .unwrap();

                assert_eq!(result.shape, result_shape);

                for i in 0..2 {
                    for j in 0..2 {
                        assert_eq!(
                            read_element(&ctx, &result, &[i, j]).unwrap(),
                            read_element(&ctx, &source, &[2 - 2 * i, 2 - 2 * j]).unwrap(),
                            "{element:?}"
                        );
                    }
                }

                assert_eq!(module.disp(&ctx).to_string(), before);
            }
        }

        #[test]

        fn indexed_scalar_and_empty_results_preserve_owned_raw_values() {
            for dimensions in [vec![], vec![0, 3]] {
                let mut ctx = context();

                let module = module(&mut ctx);

                let shape = shape::array(&ctx, ElementType::C128, dimensions.clone()).unwrap();

                let (_, block, _) = computation(&mut ctx, module, "main", &[], shape);

                let source = Literal::from_bits(
                    &ctx,
                    shape,
                    if dimensions.is_empty() {
                        &[u128::MAX]
                    } else {
                        &[]
                    },
                )
                .unwrap();

                let input = constant(&mut ctx, block, source.clone(), "input");

                let axes = Axes((0..dimensions.len()).map(|axis| axis as i64).collect());

                let reverse = ReverseOp::new(&mut ctx, shape, vec![input], axes);

                let reverse = append(&ctx, block, reverse.get_operation(), "reverse");

                let slice = SliceOp::new(
                    &mut ctx,
                    shape,
                    vec![reverse],
                    geometry(
                        &vec![0; dimensions.len()],
                        &dimensions,
                        &vec![1; dimensions.len()],
                    ),
                );

                let slice = append(&ctx, block, slice.get_operation(), "slice");

                finish(&mut ctx, block, slice);

                assert_eq!(
                    evaluate(&ctx, module, slice, &[], DependencyMode::Recursive)
                        .unwrap()
                        .into_literal(&ctx)
                        .unwrap(),
                    source
                );
            }
        }

        #[test]

        fn indexed_unknown_operand_stays_unknown_without_zero_literal() {
            let mut ctx = context();

            let module = module(&mut ctx);

            let shape = shape::array(&ctx, ElementType::F32, vec![3]).unwrap();

            let (_, block, parameters) = computation(&mut ctx, module, "main", &[shape], shape);

            let reverse = ReverseOp::new(&mut ctx, shape, vec![parameters[0]], Axes(vec![0]));

            let reverse = append(&ctx, block, reverse.get_operation(), "reverse");

            finish(&mut ctx, block, reverse);

            let result = evaluate(&ctx, module, reverse, &[], DependencyMode::Recursive).unwrap();

            assert_eq!(result.state(), &ValueState::Unknown);

            assert!(
                result
                    .into_literal(&ctx)
                    .unwrap_err()
                    .err
                    .downcast_ref::<EvaluationError>()
                    .is_some_and(|error| matches!(error, EvaluationError::UnknownValue))
            );
        }

        #[test]

        fn indexed_dynamic_input_reports_typed_unsupported_runtime_policy() {
            let mut ctx = context();

            let module = module(&mut ctx);

            let shape =
                shape::dynamic_array(&ctx, ElementType::U8, vec![Dimension::Bounded(3)]).unwrap();

            let (_, block, parameters) = computation(&mut ctx, module, "main", &[shape], shape);

            let reverse = ReverseOp::new(&mut ctx, shape, vec![parameters[0]], Axes(vec![0]));

            let reverse = append(&ctx, block, reverse.get_operation(), "reverse");

            finish(&mut ctx, block, reverse);

            let input =
                Literal::from_dynamic_storage_bytes(&ctx, shape, vec![1, 2, 3], vec![1]).unwrap();

            let error = evaluate(
                &ctx,
                module,
                reverse,
                &[(parameters[0], known(&ctx, input))],
                DependencyMode::Recursive,
            )
            .unwrap_err();

            assert!(matches!(
                error.err.downcast_ref::<EvaluationError>(),
                Some(EvaluationError::UnsupportedValue {
                    opcode: Opcode::Reverse,

                    ..
                })
            ));
        }

        #[test]

        fn concatenate_all_32_formats_honors_declared_layout_and_raw_operand_order() {
            for &element in ElementType::ALL {
                let mut ctx = context();

                let module = module(&mut ctx);

                let input_shape = shape::array(&ctx, element, vec![2, 3]).unwrap();

                let output_shape = shape::array(&ctx, element, vec![2, 6]).unwrap();

                let output_shape = shape::with_layout(
                    &ctx,
                    output_shape,
                    Layout {
                        minor_to_major: vec![0, 1],

                        ..Default::default()
                    },
                )
                .unwrap();

                let (_, block, _) = computation(&mut ctx, module, "main", &[], output_shape);

                let width = element.bit_width().div_ceil(8) as usize;

                let bytes = (0..6)
                    .flat_map(|index| {
                        (0..width).map(move |byte| {
                            if element == ElementType::Pred {
                                (index % 2) as u8
                            } else {
                                (0x80 + index * 17 + byte) as u8
                            }
                        })
                    })
                    .collect::<Vec<_>>();

                let source = Literal::from_storage_bytes(&ctx, input_shape, bytes).unwrap();

                let input = constant(&mut ctx, block, source.clone(), "input");

                let op = ConcatenateOp::new(&mut ctx, output_shape, vec![input, input], Index(1));

                let result = append(&ctx, block, op.get_operation(), "concatenate");

                finish(&mut ctx, block, result);

                let literal = evaluate(&ctx, module, result, &[], DependencyMode::Recursive)
                    .unwrap()
                    .into_literal(&ctx)
                    .unwrap();

                assert_eq!(literal.shape, output_shape);

                for i in 0..2 {
                    for j in 0..6 {
                        assert_eq!(
                            read_element(&ctx, &literal, &[i, j]).unwrap(),
                            read_element(&ctx, &source, &[i, j % 3]).unwrap(),
                            "{element:?}"
                        );
                    }
                }
            }
        }

        #[test]

        fn concatenate_unknown_operand_propagates_without_evaluating_fake_data() {
            let mut ctx = context();

            let module = module(&mut ctx);

            let input_shape = shape::array(&ctx, ElementType::C128, vec![2]).unwrap();

            let output_shape = shape::array(&ctx, ElementType::C128, vec![4]).unwrap();

            let (_, block, inputs) =
                computation(&mut ctx, module, "main", &[input_shape], output_shape);

            let op =
                ConcatenateOp::new(&mut ctx, output_shape, vec![inputs[0], inputs[0]], Index(0));

            let result = append(&ctx, block, op.get_operation(), "concatenate");

            finish(&mut ctx, block, result);

            let value = evaluate(&ctx, module, result, &[], DependencyMode::Recursive).unwrap();

            assert_eq!(value.shape(), output_shape);

            assert_eq!(value.state(), &ValueState::Unknown);

            assert!(value.into_literal(&ctx).is_err());
        }
    }
    mod padding {
        use super::*;
        use crate::{
            attributes::{PaddingConfig, PaddingConfigDimension},
            literal::semantics::read_element,
            ops::movement::PadOp,
        };

        fn config(axes: &[(i64, i64, i64)]) -> PaddingConfig {
            PaddingConfig(
                axes.iter()
                    .map(|&(low, high, interior)| PaddingConfigDimension {
                        edge_padding_low: low,
                        edge_padding_high: high,
                        interior_padding: interior,
                    })
                    .collect::<Vec<_>>(),
            )
        }

        #[test]
        fn pad_all_32_formats_copies_raw_values_and_retargets_declared_layout() {
            for &element in ElementType::ALL {
                let mut ctx = context();
                let module = module(&mut ctx);
                let input_shape = shape::array(&ctx, element, vec![2, 3]).unwrap();
                let scalar_shape = shape::scalar(&ctx, element).unwrap();
                let output_shape = shape::with_layout(
                    &ctx,
                    shape::array(&ctx, element, vec![4, 8]).unwrap(),
                    Layout {
                        minor_to_major: vec![0, 1],
                        memory_space: 7,
                        ..Default::default()
                    },
                )
                .unwrap();
                let (_, block, _) = computation(&mut ctx, module, "main", &[], output_shape);
                let width = element.bit_width().div_ceil(8) as usize;
                let source = Literal::from_storage_bytes(
                    &ctx,
                    input_shape,
                    (0..6 * width)
                        .map(|index| {
                            if element == ElementType::Pred {
                                (index % 2) as u8
                            } else {
                                (0x80 + index) as u8
                            }
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap();
                let value = Literal::from_storage_bytes(
                    &ctx,
                    scalar_shape,
                    (0..width)
                        .map(|index| {
                            if element == ElementType::Pred {
                                1
                            } else {
                                (0xf0 + index) as u8
                            }
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap();
                let input = constant(&mut ctx, block, source.clone(), "input");
                let padding = constant(&mut ctx, block, value.clone(), "padding_value");
                let op = PadOp::new(
                    &mut ctx,
                    output_shape,
                    vec![input, padding],
                    config(&[(1, 0, 1), (-1, 2, 2)]),
                );
                let result = append(&ctx, block, op.get_operation(), "pad");
                finish(&mut ctx, block, result);
                let before = module.disp(&ctx).to_string();
                let output = evaluate(&ctx, module, result, &[], DependencyMode::Recursive)
                    .unwrap()
                    .into_literal(&ctx)
                    .unwrap();
                assert_eq!(output.shape, output_shape);
                for i in 0..4 {
                    for j in 0..8 {
                        let expected = if matches!(i, 1 | 3) && matches!(j, 2 | 5) {
                            read_element(&ctx, &source, &[(i - 1) / 2, (j + 1) / 3]).unwrap()
                        } else {
                            read_element(&ctx, &value, &[]).unwrap()
                        };
                        assert_eq!(
                            read_element(&ctx, &output, &[i, j]).unwrap(),
                            expected,
                            "{element:?} [{i},{j}]"
                        );
                    }
                }
                assert_eq!(module.disp(&ctx).to_string(), before);
            }
        }

        #[test]
        fn pad_missing_operand_is_unknown_recursive_and_a_typed_shallow_dependency() {
            for mode in [DependencyMode::Recursive, DependencyMode::Shallow] {
                for missing in [0, 1] {
                    let mut ctx = context();
                    let module = module(&mut ctx);
                    let input_shape = shape::array(&ctx, ElementType::C128, vec![2]).unwrap();
                    let scalar_shape = shape::scalar(&ctx, ElementType::C128).unwrap();
                    let output_shape = shape::array(&ctx, ElementType::C128, vec![4]).unwrap();
                    let (_, block, inputs) = computation(
                        &mut ctx,
                        module,
                        "main",
                        &[input_shape, scalar_shape],
                        output_shape,
                    );
                    let op =
                        PadOp::new(&mut ctx, output_shape, inputs.clone(), config(&[(1, 1, 0)]));
                    let result = append(&ctx, block, op.get_operation(), "pad");
                    finish(&mut ctx, block, result);
                    let supplied = 1 - missing;
                    let literal = if supplied == 0 {
                        Literal::from_bits(&ctx, input_shape, &[1, u128::MAX]).unwrap()
                    } else {
                        Literal::from_bits(&ctx, scalar_shape, &[u128::MAX]).unwrap()
                    };
                    let output = evaluate(
                        &ctx,
                        module,
                        result,
                        &[(inputs[supplied], known(&ctx, literal))],
                        mode,
                    );
                    if mode == DependencyMode::Shallow {
                        assert!(matches!(
                            output.unwrap_err().err.downcast_ref::<EvaluationError>(),
                            Some(EvaluationError::ShallowDependency)
                        ));
                        continue;
                    }
                    let output = output.unwrap();
                    assert_eq!(output.shape(), output_shape);
                    assert_eq!(output.state(), &ValueState::Unknown);
                    assert!(output.into_literal(&ctx).is_err());
                }
            }
        }

        #[test]
        fn pad_known_dynamic_input_reports_typed_unsupported_instead_of_guessing_sizes() {
            let mut ctx = context();
            let module = module(&mut ctx);
            let input_shape =
                shape::dynamic_array(&ctx, ElementType::U8, vec![Dimension::Bounded(2)]).unwrap();
            let scalar_shape = shape::scalar(&ctx, ElementType::U8).unwrap();
            let output_shape =
                shape::dynamic_array(&ctx, ElementType::U8, vec![Dimension::Bounded(4)]).unwrap();
            let (_, block, inputs) = computation(
                &mut ctx,
                module,
                "main",
                &[input_shape, scalar_shape],
                output_shape,
            );
            let op = PadOp::new(&mut ctx, output_shape, inputs.clone(), config(&[(1, 1, 0)]));
            let result = append(&ctx, block, op.get_operation(), "pad");
            finish(&mut ctx, block, result);
            let source =
                Literal::from_dynamic_storage_bytes(&ctx, input_shape, vec![1, 2], vec![1])
                    .unwrap();
            let value = Literal::from_bits(&ctx, scalar_shape, &[0]).unwrap();
            let error = evaluate(
                &ctx,
                module,
                result,
                &[
                    (inputs[0], known(&ctx, source)),
                    (inputs[1], known(&ctx, value)),
                ],
                DependencyMode::Recursive,
            )
            .unwrap_err();
            assert!(matches!(
                error.err.downcast_ref::<EvaluationError>(),
                Some(EvaluationError::UnsupportedValue {
                    opcode: Opcode::Pad,
                    ..
                })
            ));
        }
    }
    mod selection {
        use super::*;
        use crate::literal::semantics::read_element;

        #[test]
        fn select_all_32_formats_preserves_raw_storage_across_layouts_and_predicate_forms() {
            for &element in ElementType::ALL {
                for scalar_condition in [None, Some(0), Some(1)] {
                    let mut ctx = context();
                    let module = module(&mut ctx);
                    let shape = shape::array(&ctx, element, vec![2, 3]).unwrap();
                    let column = shape::with_layout(
                        &ctx,
                        shape,
                        Layout {
                            minor_to_major: vec![0, 1],
                            memory_space: 7,
                            ..Default::default()
                        },
                    )
                    .unwrap();
                    let (_, block, _) = computation(&mut ctx, module, "main", &[], column);
                    let width = element.bit_width().div_ceil(8) as usize;
                    let data = |seed: usize| {
                        (0..6 * width)
                            .map(|index| {
                                if element == ElementType::Pred {
                                    ((seed + index) % 2) as u8
                                } else {
                                    (seed + index) as u8
                                }
                            })
                            .collect::<Vec<_>>()
                    };
                    let on_true = Literal::from_storage_bytes(&ctx, column, data(0x80)).unwrap();
                    let on_false = Literal::from_storage_bytes(&ctx, shape, data(0x21)).unwrap();
                    let predicate = if let Some(condition) = scalar_condition {
                        Literal::from_bits(
                            &ctx,
                            shape::scalar(&ctx, ElementType::Pred).unwrap(),
                            &[condition],
                        )
                        .unwrap()
                    } else {
                        Literal::from_bits(
                            &ctx,
                            shape::array(&ctx, ElementType::Pred, vec![2, 3]).unwrap(),
                            &[1, 0, 1, 0, 0, 1],
                        )
                        .unwrap()
                    };
                    let pred = constant(&mut ctx, block, predicate.clone(), "predicate");
                    let left = constant(&mut ctx, block, on_true.clone(), "on_true");
                    let right = constant(&mut ctx, block, on_false.clone(), "on_false");
                    let op = SelectOp::new(&mut ctx, column, vec![pred, left, right]);
                    let result = append(&ctx, block, op.get_operation(), "select");
                    finish(&mut ctx, block, result);
                    let before = module.disp(&ctx).to_string();
                    let output = evaluate(&ctx, module, result, &[], DependencyMode::Recursive)
                        .unwrap()
                        .into_literal(&ctx)
                        .unwrap();
                    assert_eq!(output.shape, column);
                    for i in 0..2 {
                        for j in 0..3 {
                            let index: &[usize] = if scalar_condition.is_some() {
                                &[]
                            } else {
                                &[i, j]
                            };
                            let selected = if read_element(&ctx, &predicate, index).unwrap()[0] != 0
                            {
                                &on_true
                            } else {
                                &on_false
                            };
                            assert_eq!(
                                read_element(&ctx, &output, &[i, j]).unwrap(),
                                read_element(&ctx, selected, &[i, j]).unwrap(),
                                "{element:?} {scalar_condition:?} [{i},{j}]"
                            );
                        }
                    }
                    assert_eq!(module.disp(&ctx).to_string(), before);
                }
            }
        }

        #[test]
        fn select_evaluates_both_data_dependencies_instead_of_becoming_lazy_conditional() {
            for mode in [DependencyMode::Recursive, DependencyMode::Shallow] {
                for missing in 0..3 {
                    let mut ctx = context();
                    let module = module(&mut ctx);
                    let pred_shape = shape::scalar(&ctx, ElementType::Pred).unwrap();
                    let shape = shape::array(&ctx, ElementType::F32, vec![2]).unwrap();
                    let (_, block, args) =
                        computation(&mut ctx, module, "main", &[pred_shape, shape, shape], shape);
                    let op = SelectOp::new(&mut ctx, shape, args.clone());
                    let result = append(&ctx, block, op.get_operation(), "select");
                    finish(&mut ctx, block, result);
                    let literals = [
                        Literal::from_bits(&ctx, pred_shape, &[1]).unwrap(),
                        Literal::from_bits(&ctx, shape, &[0x7f80_1234, 0x8000_0000]).unwrap(),
                        Literal::from_bits(&ctx, shape, &[0xff80_4321, 0]).unwrap(),
                    ];
                    let substitutions = args
                        .iter()
                        .zip(literals)
                        .enumerate()
                        .filter(|(index, _)| *index != missing)
                        .map(|(_, (&value, literal))| (value, known(&ctx, literal)))
                        .collect::<Vec<_>>();
                    let output = evaluate(&ctx, module, result, &substitutions, mode);
                    if mode == DependencyMode::Shallow {
                        assert!(matches!(
                            output.unwrap_err().err.downcast_ref::<EvaluationError>(),
                            Some(EvaluationError::ShallowDependency)
                        ));
                    } else {
                        assert_eq!(output.unwrap().state(), &ValueState::Unknown);
                    }
                }
            }
        }

        #[test]
        fn select_known_dynamic_inputs_have_an_explicit_unsupported_runtime_policy() {
            let mut ctx = context();
            let module = module(&mut ctx);
            let pred_shape = shape::scalar(&ctx, ElementType::Pred).unwrap();
            let shape =
                shape::dynamic_array(&ctx, ElementType::U8, vec![Dimension::Bounded(2)]).unwrap();
            let (_, block, args) =
                computation(&mut ctx, module, "main", &[pred_shape, shape, shape], shape);
            let op = SelectOp::new(&mut ctx, shape, args.clone());
            let result = append(&ctx, block, op.get_operation(), "select");
            finish(&mut ctx, block, result);
            let predicate = Literal::from_bits(&ctx, pred_shape, &[1]).unwrap();
            let data =
                Literal::from_dynamic_storage_bytes(&ctx, shape, vec![1, 2], vec![1]).unwrap();
            let error = evaluate(
                &ctx,
                module,
                result,
                &[
                    (args[0], known(&ctx, predicate)),
                    (args[1], known(&ctx, data.clone())),
                    (args[2], known(&ctx, data)),
                ],
                DependencyMode::Recursive,
            )
            .unwrap_err();
            assert!(matches!(
                error.err.downcast_ref::<EvaluationError>(),
                Some(EvaluationError::UnsupportedValue {
                    opcode: Opcode::Select,
                    ..
                })
            ));
        }
    }
    use crate::{
        attributes::{Bools, FusionKind},
        ir::{ControlPredecessors, InstructionName},
        ops::{
            ReturnOp,
            control::{
                CallOp, ConditionalOp, ConstantOp, CustomCallOp, FusionOp, GetTupleElementOp,
                ParameterOp, TupleOp, WhileOp,
            },
            elementwise::{
                AddOp, BitcastConvertOp, CompareOp, ComplexOp, ConvertOp, ImagOp, IsFiniteOp,
                MulhiOp, MultiplyOp, NegateOp, RealOp, RoundNearestEvenOp, SelectOp,
            },
            movement::{BroadcastOp, CopyOp, ReshapeOp, TransposeOp},
        },
        shape::Dimension,
        types::ElementType,
    };
    use pliron::{
        basic_block::BasicBlock,
        builtin::{attributes::StringAttr, types::FunctionType},
        op::Op,
        printable::Printable,
    };

    fn context() -> Context {
        let mut ctx = Context::new();
        crate::register(&mut ctx);
        ctx
    }

    fn module(ctx: &mut Context) -> Ptr<Operation> {
        let module = ModuleOp::new(ctx, Identifier::new("evaluation")).get_operation();
        crate::ir::set_entry(ctx, module, Identifier::new("main")).unwrap();
        module
    }

    fn computation(
        ctx: &mut Context,
        module: Ptr<Operation>,
        name: &'static str,
        inputs: &[TypeHandle],
        output: TypeHandle,
    ) -> (Ptr<Operation>, Ptr<BasicBlock>, Vec<Value>) {
        let signature = FunctionType::get(ctx, inputs.to_vec(), vec![output]).into();
        let comp = ComputationOp::new(ctx, Identifier::new(name), signature);
        let container = module
            .deref(ctx)
            .get_region(0)
            .deref(ctx)
            .get_entry_block()
            .unwrap();
        comp.get_operation().insert_at_back(container, ctx);
        let block = BasicBlock::new(ctx, None, vec![]);
        block.insert_at_back(comp.body(ctx), ctx);
        let parameters = inputs
            .iter()
            .enumerate()
            .map(|(index, &ty)| {
                let op = ParameterOp::new(ctx, ty, vec![], Index(index as u32));
                op.get_operation().insert_at_back(block, ctx);
                named(ctx, op.get_operation(), &format!("{name}.p{index}"));
                op.result(ctx)
            })
            .collect();
        (comp.get_operation(), block, parameters)
    }

    fn named(ctx: &Context, op: Ptr<Operation>, name: &str) {
        op.deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_name"), InstructionName(name.into()));
    }

    fn append(ctx: &Context, block: Ptr<BasicBlock>, op: Ptr<Operation>, name: &str) -> Value {
        named(ctx, op, name);
        op.insert_at_back(block, ctx);
        op.deref(ctx).get_result(0)
    }

    fn finish(ctx: &mut Context, block: Ptr<BasicBlock>, value: Value) {
        ReturnOp::new(ctx, value)
            .get_operation()
            .insert_at_back(block, ctx);
    }

    fn constant(ctx: &mut Context, block: Ptr<BasicBlock>, literal: Literal, name: &str) -> Value {
        let op = ConstantOp::new(ctx, literal.shape, vec![], literal).get_operation();
        append(ctx, block, op, name)
    }

    fn integer(ctx: &Context, element: ElementType, value: i128) -> Literal {
        literal_semantics::integer_scalar(ctx, element, value).unwrap()
    }

    fn known(ctx: &Context, literal: Literal) -> EvaluatedValue {
        EvaluatedValue::known(ctx, literal).unwrap()
    }

    fn evaluate(
        ctx: &Context,
        module: Ptr<Operation>,
        value: Value,
        substitutions: &[(Value, EvaluatedValue)],
        mode: DependencyMode,
    ) -> Result<EvaluatedValue> {
        HloEvaluator::new(ctx, module)?.evaluate(Request::Instruction {
            value,
            substitutions,
            dependency_mode: mode,
        })
    }

    fn read_i64(ctx: &Context, value: EvaluatedValue) -> i64 {
        literal_semantics::scalar_integer(ctx, &value.into_literal(ctx).unwrap()).unwrap()
    }

    struct Math {
        module: Ptr<Operation>,
        computation: Ptr<Operation>,
        parameter: Value,
        square: Value,
        sum: Value,
    }

    fn math(ctx: &mut Context) -> Math {
        let module = module(ctx);
        let ty = shape::scalar(ctx, ElementType::S64).unwrap();
        let (computation, block, parameters) = computation(ctx, module, "main", &[ty], ty);
        let p = parameters[0];
        let mul = MultiplyOp::new(ctx, ty, vec![p, p]).get_operation();
        let square = append(ctx, block, mul, "square");
        let add = AddOp::new(ctx, ty, vec![p, square]).get_operation();
        let sum = append(ctx, block, add, "sum");
        finish(ctx, block, sum);
        verification::verify(ctx, module).unwrap();
        Math {
            module,
            computation,
            parameter: p,
            square,
            sum,
        }
    }

    fn conditional_fixture(
        ctx: &mut Context,
        predicate: bool,
        bad_branch: bool,
    ) -> (Ptr<Operation>, Value, Vec<Value>) {
        let module = module(ctx);
        let ty = shape::scalar(ctx, ElementType::S64).unwrap();
        for name in ["first", "last"] {
            let (_, block, p) = computation(ctx, module, name, &[ty], ty);
            if name == "last" && bad_branch {
                let custom =
                    CustomCallOp::new(ctx, ty, vec![p[0]], StringAttr::new("effect".into()))
                        .get_operation();
                custom
                    .deref_mut(ctx)
                    .attributes
                    .set(Identifier::new("hlo_has_side_effect"), Bools(vec![true]));
                let custom = append(ctx, block, custom, "effect");
                finish(ctx, block, custom);
            } else {
                finish(ctx, block, p[0]);
            }
        }
        let selector_ty = shape::scalar(
            ctx,
            if predicate {
                ElementType::Pred
            } else {
                ElementType::S32
            },
        )
        .unwrap();
        let (_, block, p) = computation(ctx, module, "main", &[selector_ty, ty, ty], ty);
        let op = ConditionalOp::new(
            ctx,
            ty,
            p.clone(),
            Callees(vec!["first".into(), "last".into()]),
        )
        .get_operation();
        let root = append(ctx, block, op, "conditional");
        finish(ctx, block, root);
        verification::verify(ctx, module).unwrap();
        (module, root, p)
    }
}
