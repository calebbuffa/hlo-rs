//! While execution and request-local induction shortcuts based on OpenXLA's HLO evaluator.
//!
//! Uses the evaluator's shared Session, configured iteration limit, and native
//! condition/body frames; `parse` owns the existing induction proof matcher.
//! This module neither implements trip-count analysis nor annotates the IR.
mod parse;

use super::*;
use std::collections::HashSet;

pub(super) struct WhileState {
    pub limit: i64,
    parsing: HashSet<Value>,
}

impl WhileState {
    pub fn new(limit: i64) -> Self {
        Self {
            limit,
            parsing: HashSet::new(),
        }
    }
}

impl HloEvaluator<'_> {
    /// Bound condition attempts using the pinned pre-condition guard.
    ///
    /// The default is -1; every negative value is unlimited. A limit of zero
    /// permits the first condition and, if true, one body execution before
    /// attempting an induction-variable shortcut or returning a limit error.
    /// Nested calls, fusion and loops inherit the setting without a global
    /// iteration budget. Unlimited loops are not guaranteed to terminate.
    pub fn with_max_loop_iterations(mut self, limit: i64) -> Self {
        self.max_loop_iterations = limit;
        self
    }
}

impl Session<'_> {
    pub(super) fn evaluate_while(
        &mut self,
        frame: &mut Frame,
        operation: Ptr<Operation>,
        path: &[usize],
        mode: DependencyMode,
    ) -> Result<EvaluatedValue> {
        let (input, body, condition) = self.while_parts(operation)?;
        let mut carried = self.operand(frame, input, &[], mode)?;
        let result_type = operation.deref(self.ctx).get_result(0).get_type(self.ctx);
        carried = carried.retarget(self.ctx, result_type)?;
        if !carried.is_known() {
            if path.len() == 1
                && let Some(result) =
                    self.while_induction_result(frame, operation, Some(path[0]))?
            {
                return Ok(result.subvalue(path)?.clone());
            }
            return EvaluatedValue::unknown(
                self.ctx,
                shape::subshape(self.ctx, result_type, path)?,
            );
        }

        let limit = self.while_state.limit;
        let mut attempts = 0u64;
        loop {
            // The pinned post-increment guard precedes even the final false
            // condition. In particular, limit zero still permits one body.
            if limit >= 0 && attempts > limit as u64 {
                if let Some(result) = self.while_induction_result(frame, operation, None)? {
                    return Ok(result.subvalue(path)?.clone());
                }
                return Err(failure(EvaluationError::LoopIterationLimit { limit }));
            }
            if limit >= 0 {
                attempts += 1;
            }
            let predicate = self
                .computation(condition, &[carried.clone()], &[])?
                .into_literal(self.ctx)?;
            if !literal_semantics::scalar_predicate(self.ctx, &predicate)? {
                return Ok(carried.subvalue(path)?.clone());
            }
            carried = self
                .computation(body, &[carried], &[])?
                .retarget(self.ctx, result_type)?;
            if !carried.is_known() {
                return Err(failure(EvaluationError::UnknownValue));
            }
        }
    }

    fn while_parts(
        &self,
        operation: Ptr<Operation>,
    ) -> Result<(Value, Ptr<Operation>, Ptr<Operation>)> {
        let names = attr::<Callees>(self.ctx, operation, "hlo_callees")?;
        let operands: Vec<_> = operation.deref(self.ctx).operands().collect();
        if names.0.len() != 2 || operands.len() != 1 {
            return Err(malformed("While requires one operand, body and condition"));
        }
        Ok((
            operands[0],
            self.resolve(&names.0[0])?,
            self.resolve(&names.0[1])?,
        ))
    }

    fn while_induction_result(
        &mut self,
        frame: &mut Frame,
        operation: Ptr<Operation>,
        requested_index: Option<usize>,
    ) -> Result<Option<EvaluatedValue>> {
        let value = operation.deref(self.ctx).get_result(0);
        if !self.while_state.parsing.insert(value) {
            return Ok(None);
        }
        let result = (|| {
            let (input, body, condition) = self.while_parts(operation)?;
            // A shortcut must not hide even an unused effect in a bypassed
            // computation. Selected-path purity analysis is not implemented.
            if self.effects.has_side_effect(body)? || self.effects.has_side_effect(condition)? {
                return Ok(None);
            }
            let initial = self.eval(frame, input, &[], DependencyMode::Recursive)?;
            let mut parser = parse::Parser::new(self, frame);
            let Some(proof) = parser.prove(operation, &initial)? else {
                return Ok(None);
            };
            if requested_index.is_some_and(|index| index != proof.index()) {
                return Ok(None);
            }
            proof.materialize(parser.context()).map(Some)
        })();
        self.while_state.parsing.remove(&value);
        result
    }
}

#[cfg(test)]
mod tests {
    //! Shared While execution fixtures and limit, induction, frame, and payload regressions.
    mod limits {
        use super::*;

        #[test]
        fn max_zero_initially_false_returns_complete_input_without_body_execution() {
            let mut ctx = context();
            let f = fixture(&mut ctx, ElementType::S64, 10, 3, 1, Direction::Lt);
            f.effect
                .deref_mut(&ctx)
                .attributes
                .set(Identifier::new("hlo_has_side_effect"), Bools(vec![true]));
            let result = complete(&ctx, &f, 0).unwrap();
            assert!(result.is_known());
            assert_eq!(
                read_integer(&ctx, result.subvalue(&[0]).unwrap().clone()),
                10
            );
            assert_eq!(
                read_integer(&ctx, result.subvalue(&[3]).unwrap().clone()),
                19
            );
        }

        #[test]
        fn max_zero_true_executes_body_before_attempting_induction_shortcut() {
            let mut ctx = context();
            let f = fixture(&mut ctx, ElementType::S64, 0, 9, 1, Direction::Lt);
            f.effect
                .deref_mut(&ctx)
                .attributes
                .set(Identifier::new("hlo_has_side_effect"), Bools(vec![true]));
            let error = complete(&ctx, &f, 0).unwrap_err();
            assert!(matches!(
                error.err.downcast_ref::<EvaluationError>(),
                Some(EvaluationError::Effectful {
                    opcode: Opcode::CustomCall
                })
            ));
        }

        #[test]
        fn max_zero_finite_shortcut_returns_only_induction_leaf_as_known() {
            let mut ctx = context();
            let f = fixture(&mut ctx, ElementType::S64, 0, 9, 1, Direction::Lt);
            let result = complete(&ctx, &f, 0).unwrap();
            assert_eq!(
                read_integer(&ctx, result.subvalue(&[0]).unwrap().clone()),
                9
            );
            for index in 1..4 {
                assert!(matches!(
                    result.subvalue(&[index]).unwrap().state(),
                    ValueState::Unknown
                ));
            }
            assert!(result.into_literal(&ctx).is_err());
        }

        #[test]
        fn limit_guard_precedes_final_false_condition_and_discards_known_siblings_on_shortcut() {
            let mut ctx = context();
            let f = fixture(&mut ctx, ElementType::S64, 0, 1, 1, Direction::Lt);
            assert!(!complete(&ctx, &f, 0).unwrap().is_known());
            assert!(complete(&ctx, &f, 1).unwrap().is_known());
        }

        #[test]
        fn zero_step_true_loop_hits_distinct_limit_error_not_success_last_state() {
            let mut ctx = context();
            let f = fixture(&mut ctx, ElementType::S64, 0, 1, 0, Direction::Lt);
            assert_limit(complete(&ctx, &f, 0), 0);
            assert_limit(complete(&ctx, &f, 1), 1);
        }

        #[test]
        fn all_negative_limits_are_unlimited_and_keep_complete_body_result() {
            let mut ctx = context();
            let f = fixture(&mut ctx, ElementType::S64, 0, 3, 1, Direction::Lt);
            for limit in [-1, -2, i64::MIN] {
                let result = complete(&ctx, &f, limit).unwrap();
                assert!(result.is_known());
                assert_eq!(
                    read_integer(&ctx, result.subvalue(&[0]).unwrap().clone()),
                    3
                );
            }
        }

        #[test]
        fn default_engine_limit_is_unlimited_not_zero() {
            let mut ctx = context();
            let f = fixture(&mut ctx, ElementType::S64, 0, 3, 1, Direction::Lt);
            let substitutions = [(f.parameter, known_integer(&ctx, f.element, 19))];
            let result = HloEvaluator::new(&ctx, f.module)
                .unwrap()
                .evaluate(Request::Instruction {
                    value: f.loop_value,
                    substitutions: &substitutions,
                    dependency_mode: DependencyMode::Recursive,
                })
                .unwrap();
            assert!(result.is_known());
            assert_eq!(
                read_integer(&ctx, result.subvalue(&[0]).unwrap().clone()),
                3
            );
        }

        #[test]
        fn narrow_wrap_declined_proof_falls_back_to_execution_then_limit_error() {
            let mut ctx = context();
            let f = fixture(&mut ctx, ElementType::S8, 126, 127, 2, Direction::Lt);
            assert_limit(complete(&ctx, &f, 0), 0);
        }
    }
    mod induction {
        use super::*;

        #[test]
        fn upstream_evaluate_while_induction_var_with_nonunit_step_and_nonzero_init() {
            let mut ctx = context();
            let f = fixture(&mut ctx, ElementType::S32, 3, 23, 4, Direction::Lt);
            assert_eq!(
                read_integer(&ctx, evaluate(&ctx, &f, f.projection, -1, &[]).unwrap()),
                23
            );
        }

        #[test]
        fn upstream_loop_bound_defined_outside_of_condition_uses_unchanged_tuple_leaf() {
            let mut ctx = context();
            let f = fixture(&mut ctx, ElementType::S64, 0, 5, 1, Direction::Lt);
            assert_eq!(
                read_integer(&ctx, evaluate(&ctx, &f, f.projection, 0, &[]).unwrap()),
                5
            );
        }

        #[test]
        fn upstream_step_size_not_one_handles_final_overshoot() {
            let mut ctx = context();
            let f = fixture(&mut ctx, ElementType::S64, 0, 10, 4, Direction::Lt);
            assert_eq!(
                read_integer(&ctx, evaluate(&ctx, &f, f.projection, 0, &[]).unwrap()),
                12
            );
        }

        #[test]
        fn partial_input_non_induction_projection_stays_unknown() {
            let mut ctx = context();
            let f = fixture(&mut ctx, ElementType::S64, 0, 5, 1, Direction::Lt);
            let mut session = Session::new(&ctx, f.module).unwrap();
            let mut frame = Frame::new(&ctx, f.owner).unwrap();
            let result = session
                .evaluate_while(
                    &mut frame,
                    f.loop_value.defining_op().unwrap(),
                    &[3],
                    DependencyMode::Recursive,
                )
                .unwrap();
            assert!(matches!(result.state(), ValueState::Unknown));
        }

        #[test]
        fn partial_input_whole_tuple_does_not_claim_induction_only_projection_success() {
            let mut ctx = context();
            let f = fixture(&mut ctx, ElementType::S64, 0, 5, 1, Direction::Lt);
            let result = evaluate(&ctx, &f, f.loop_value, 0, &[]).unwrap();
            for index in 0..4 {
                assert!(matches!(
                    result.subvalue(&[index]).unwrap().state(),
                    ValueState::Unknown
                ));
            }
        }

        #[test]
        fn undetermined_initial_payload_does_not_become_zero_or_block_proven_iv_projection() {
            let mut ctx = context();
            let f = fixture(&mut ctx, ElementType::S64, 0, 5, 1, Direction::Lt);
            let unvisited = EvaluatedValue::undetermined(&ctx, f.parameter.get_type(&ctx)).unwrap();
            let substitutions = [(f.parameter, unvisited)];
            assert_eq!(
                read_integer(
                    &ctx,
                    evaluate(&ctx, &f, f.projection, 0, &substitutions).unwrap()
                ),
                5
            );
        }

        #[test]
        fn all_14_integer_formats_execute_finite_loop_with_complete_known_state() {
            let mut ctx = context();
            for &element in ElementType::ALL {
                if element.is_integer() {
                    let (init, bound, step, direction) = if element == ElementType::S1 {
                        (-1, 0, 1, Direction::Lt)
                    } else {
                        (0, 1, 1, Direction::Lt)
                    };
                    // S1 has no mathematical +1: its significant one encodes -1.
                    if element == ElementType::S1 {
                        let f = fixture(&mut ctx, element, 0, -1, -1, Direction::Gt);
                        let result = complete(&ctx, &f, -1).unwrap();
                        assert_eq!(
                            read_integer(&ctx, result.subvalue(&[0]).unwrap().clone()),
                            -1
                        );
                    } else {
                        let f = fixture(&mut ctx, element, init, bound, step, direction);
                        let result = complete(&ctx, &f, -1).unwrap();
                        assert_eq!(
                            read_integer(&ctx, result.subvalue(&[0]).unwrap().clone()),
                            1
                        );
                    }
                }
            }
        }

        #[test]
        fn checked_formula_covers_directions_inclusive_bounds_and_zero_trip_cases() {
            for (init, bound, step, direction, expected) in [
                (3, 23, 4, Direction::Lt, 23),
                (3, 23, 4, Direction::Le, 27),
                (23, 3, -4, Direction::Gt, 3),
                (23, 3, -4, Direction::Ge, -1),
                (3, 0, 4, Direction::Lt, 3),
                (0, 3, -4, Direction::Gt, 0),
                (3, 3, 4, Direction::Lt, 3),
                (3, 3, -4, Direction::Gt, 3),
            ] {
                assert_eq!(
                    parse::checked_final(init, bound, step, direction, ElementType::S64),
                    Some(expected)
                );
            }
        }

        #[test]
        fn checked_formula_declines_overflow_and_narrow_wrap_instead_of_fabricating_constant() {
            for (init, bound, step, direction, element) in [
                (126, 127, 2, Direction::Lt, ElementType::S8),
                (i64::MAX - 1, i64::MAX, 2, Direction::Lt, ElementType::S64),
                (0, i64::MAX, 1, Direction::Le, ElementType::S64),
                (0, 4, 0, Direction::Lt, ElementType::S64),
                (0, 4, -1, Direction::Lt, ElementType::S64),
                (0, 4, 1, Direction::Eq, ElementType::S64),
            ] {
                assert_eq!(
                    parse::checked_final(init, bound, step, direction, element),
                    None
                );
            }
        }

        #[test]
        fn unknown_initial_iv_does_not_infer_loop_trip_or_default_zero() {
            let mut ctx = context();
            let f = fixture(&mut ctx, ElementType::S64, 0, 5, 1, Direction::Lt);
            let unknown = EvaluatedValue::unknown(&ctx, f.initial.get_type(&ctx)).unwrap();
            let result = evaluate(&ctx, &f, f.projection, 0, &[(f.initial, unknown)]).unwrap();
            assert!(matches!(result.state(), ValueState::Unknown));
        }

        #[test]
        fn while_evaluation_preserves_printed_ir_names_and_metadata() {
            let mut ctx = context();
            let f = fixture(&mut ctx, ElementType::S64, 0, 5, 1, Direction::Lt);
            let before = format!("{}", f.module.deref(&ctx).disp(&ctx));
            for limit in [-1, 0, 3, 5] {
                let _ = complete(&ctx, &f, limit).unwrap();
            }
            assert_eq!(format!("{}", f.module.deref(&ctx).disp(&ctx)), before);
        }

        #[test]
        fn upstream_loop_bound_computed_outside_condition_uses_qualified_integer_math() {
            let mut ctx = context();
            let f = fixture_form(
                &mut ctx,
                ElementType::S64,
                0,
                5,
                1,
                Direction::Lt,
                Form::ComputedBound,
            );
            assert_eq!(
                read_integer(&ctx, evaluate(&ctx, &f, f.projection, 0, &[]).unwrap()),
                5
            );
        }

        #[test]
        fn descending_loop_and_inclusive_comparison_compute_exact_final_iv() {
            let mut ctx = context();
            for (direction, expected) in [(Direction::Gt, 3), (Direction::Ge, -1)] {
                let f = fixture(&mut ctx, ElementType::S64, 23, 3, -4, direction);
                assert_eq!(
                    read_integer(&ctx, evaluate(&ctx, &f, f.projection, 0, &[]).unwrap()),
                    expected
                );
            }
        }

        #[test]
        fn modified_bound_slot_is_not_mistaken_for_unchanged_invariant() {
            let mut ctx = context();
            let f = fixture_form(
                &mut ctx,
                ElementType::S64,
                0,
                5,
                1,
                Direction::Lt,
                Form::ChangingBound,
            );
            let result = evaluate(&ctx, &f, f.projection, 0, &[]).unwrap();
            assert!(matches!(result.state(), ValueState::Unknown));
            assert_limit(complete(&ctx, &f, 0), 0);
        }
    }
    mod substitutions {
        use super::*;

        #[test]
        fn authoritative_initial_constant_substitution_changes_proof() {
            let mut ctx = context();
            let f = fixture(&mut ctx, ElementType::S64, 0, 10, 4, Direction::Lt);
            let substitutions = [(f.initial, known_integer(&ctx, f.element, 3))];
            assert_eq!(
                read_integer(
                    &ctx,
                    evaluate(&ctx, &f, f.projection, 0, &substitutions).unwrap()
                ),
                11
            );
        }

        #[test]
        fn authoritative_bound_constant_substitution_changes_proof() {
            let mut ctx = context();
            let f = fixture(&mut ctx, ElementType::S64, 0, 10, 4, Direction::Lt);
            let substitutions = [(f.bound, known_integer(&ctx, f.element, 5))];
            assert_eq!(
                read_integer(
                    &ctx,
                    evaluate(&ctx, &f, f.projection, 0, &substitutions).unwrap()
                ),
                8
            );
        }

        #[test]
        fn authoritative_step_constant_substitution_changes_proof() {
            let mut ctx = context();
            let f = fixture(&mut ctx, ElementType::S64, 0, 10, 4, Direction::Lt);
            let substitutions = [(f.step, known_integer(&ctx, f.element, 3))];
            assert_eq!(
                read_integer(
                    &ctx,
                    evaluate(&ctx, &f, f.projection, 0, &substitutions).unwrap()
                ),
                12
            );
        }

        #[test]
        fn replacing_whole_initial_tuple_declines_original_source_pattern() {
            let mut ctx = context();
            let f = fixture(&mut ctx, ElementType::S64, 0, 10, 4, Direction::Lt);
            let replacement = EvaluatedValue::unknown(&ctx, f.input.get_type(&ctx)).unwrap();
            let result = evaluate(&ctx, &f, f.projection, 0, &[(f.input, replacement)]).unwrap();
            assert!(matches!(result.state(), ValueState::Unknown));
        }

        #[test]
        fn authoritative_while_replacement_precedes_execution_effects_and_parser() {
            let mut ctx = context();
            let f = fixture(&mut ctx, ElementType::S64, 0, 1, 0, Direction::Lt);
            f.effect
                .deref_mut(&ctx)
                .attributes
                .set(Identifier::new("hlo_has_side_effect"), Bools(vec![true]));
            let tuple = Literal::tuple(
                &ctx,
                (0..4)
                    .map(|_| literal_semantics::integer_scalar(&ctx, f.element, 77).unwrap())
                    .collect(),
            )
            .unwrap();
            let replacement = EvaluatedValue::known(&ctx, tuple).unwrap();
            assert_eq!(
                read_integer(
                    &ctx,
                    evaluate(&ctx, &f, f.projection, 0, &[(f.loop_value, replacement)]).unwrap()
                ),
                77
            );
        }

        #[test]
        fn repeated_requests_do_not_reuse_proof_after_substitution_changes() {
            let mut ctx = context();
            let f = fixture(&mut ctx, ElementType::S64, 0, 5, 1, Direction::Lt);
            for bound in [7, 3, 9, 1] {
                let substitutions = [(f.bound, known_integer(&ctx, f.element, bound))];
                assert_eq!(
                    read_integer(
                        &ctx,
                        evaluate(&ctx, &f, f.projection, 0, &substitutions).unwrap()
                    ),
                    bound as i64
                );
            }
        }

        #[test]
        fn actual_frame_substitution_overrides_unique_caller_source_constant() {
            let mut ctx = context();
            let f = fixture_form(
                &mut ctx,
                ElementType::S64,
                0,
                0,
                1,
                Direction::Lt,
                Form::DynamicBound,
            );
            wrapper(&mut ctx, &f, "outer", 7);
            crate::ir::set_entry(&ctx, f.module, Identifier::new("outer")).unwrap();
            let substitutions = [(f.parameter, known_integer(&ctx, f.element, 11))];
            assert_eq!(
                read_integer(
                    &ctx,
                    evaluate(&ctx, &f, f.projection, 0, &substitutions).unwrap()
                ),
                11
            );
        }

        #[test]
        fn callee_source_values_cannot_be_used_as_root_frame_substitutions() {
            let mut ctx = context();
            let f = fixture(&mut ctx, ElementType::S64, 0, 5, 1, Direction::Lt);
            for callee in [f.body, f.condition] {
                let root = Frame::new(&ctx, callee).unwrap().root;
                let replacement = EvaluatedValue::unknown(&ctx, root.get_type(&ctx)).unwrap();
                let error =
                    evaluate(&ctx, &f, f.projection, 0, &[(root, replacement)]).unwrap_err();
                assert!(matches!(
                    error.err.downcast_ref::<EvaluationError>(),
                    Some(EvaluationError::Malformed { .. })
                ));
            }
        }
    }
    mod frames {
        use super::*;

        #[test]
        fn upstream_loop_bound_defined_inside_condition_is_parsed_without_caller_guessing() {
            let mut ctx = context();
            let f = fixture_form(
                &mut ctx,
                ElementType::S64,
                0,
                5,
                1,
                Direction::Lt,
                Form::Inside,
            );
            assert_eq!(
                read_integer(&ctx, evaluate(&ctx, &f, f.projection, 0, &[]).unwrap()),
                5
            );
        }

        #[test]
        fn upstream_recursive_condition_call_binds_the_actual_symbolic_tuple() {
            let mut ctx = context();
            let f = fixture_form(
                &mut ctx,
                ElementType::S64,
                3,
                23,
                4,
                Direction::Lt,
                Form::Call,
            );
            assert_eq!(
                read_integer(&ctx, evaluate(&ctx, &f, f.projection, 0, &[]).unwrap()),
                23
            );
        }

        #[test]
        fn upstream_recursive_condition_get_tuple_element_preserves_path_and_frame_provenance() {
            let mut ctx = context();
            let f = fixture_form(
                &mut ctx,
                ElementType::S64,
                3,
                23,
                4,
                Direction::Lt,
                Form::CallTuple,
            );
            assert_eq!(
                read_integer(&ctx, evaluate(&ctx, &f, f.projection, 0, &[]).unwrap()),
                23
            );
        }

        #[test]
        fn upstream_copied_loop_condition_keeps_the_comparison_pattern() {
            let mut ctx = context();
            let f = fixture_form(
                &mut ctx,
                ElementType::S64,
                0,
                5,
                1,
                Direction::Lt,
                Form::Copy,
            );
            assert_eq!(
                read_integer(&ctx, evaluate(&ctx, &f, f.projection, 0, &[]).unwrap()),
                5
            );
        }

        #[test]
        fn inline_fusion_condition_uses_block_arguments_without_crossing_proof_frames() {
            let mut ctx = context();
            let f = fixture_form(
                &mut ctx,
                ElementType::S64,
                3,
                23,
                4,
                Direction::Lt,
                Form::Fusion,
            );
            assert_eq!(
                read_integer(&ctx, evaluate(&ctx, &f, f.projection, 0, &[]).unwrap()),
                23
            );
            assert!(complete(&ctx, &f, -1).unwrap().is_known());
        }

        #[test]
        fn swapped_condition_operands_reverse_comparison_direction() {
            let mut ctx = context();
            let f = fixture_form(
                &mut ctx,
                ElementType::S64,
                3,
                23,
                4,
                Direction::Gt,
                Form::Swapped,
            );
            assert_eq!(
                read_integer(&ctx, evaluate(&ctx, &f, f.projection, 0, &[]).unwrap()),
                23
            );
        }

        #[test]
        fn upstream_dynamic_loop_bound_remains_unknown_without_caller_evidence() {
            let mut ctx = context();
            let f = fixture_form(
                &mut ctx,
                ElementType::S64,
                0,
                5,
                1,
                Direction::Lt,
                Form::DynamicBound,
            );
            let result = evaluate(&ctx, &f, f.projection, 0, &[]).unwrap();
            assert!(matches!(result.state(), ValueState::Unknown));
        }

        #[test]
        fn unique_caller_resolves_dynamic_bound_symbolically_without_global_parameter_inference() {
            let mut ctx = context();
            let f = fixture_form(
                &mut ctx,
                ElementType::S64,
                0,
                0,
                1,
                Direction::Lt,
                Form::DynamicBound,
            );
            wrapper(&mut ctx, &f, "outer", 7);
            crate::ir::set_entry(&ctx, f.module, Identifier::new("outer")).unwrap();
            assert_eq!(
                read_integer(&ctx, evaluate(&ctx, &f, f.projection, 0, &[]).unwrap()),
                7
            );
            let direct_parameter = evaluate(&ctx, &f, f.parameter, 0, &[]).unwrap();
            assert!(matches!(direct_parameter.state(), ValueState::Unknown));
        }

        #[test]
        fn multiple_callers_decline_symbolic_bound_proof_even_if_one_has_known_arguments() {
            let mut ctx = context();
            let f = fixture_form(
                &mut ctx,
                ElementType::S64,
                0,
                0,
                1,
                Direction::Lt,
                Form::DynamicBound,
            );
            wrapper(&mut ctx, &f, "outer", 7);
            wrapper(&mut ctx, &f, "other", 9);
            crate::ir::set_entry(&ctx, f.module, Identifier::new("outer")).unwrap();
            assert!(matches!(
                evaluate(&ctx, &f, f.projection, 0, &[]).unwrap().state(),
                ValueState::Unknown
            ));
        }

        #[test]
        fn named_call_embedded_while_inherits_max_zero_instead_of_defaulting_unlimited() {
            let mut ctx = context();
            let f = fixture(&mut ctx, ElementType::S64, 0, 1, 0, Direction::Lt);
            let outer = wrapper(&mut ctx, &f, "outer", 7);
            crate::ir::set_entry(&ctx, f.module, Identifier::new("outer")).unwrap();
            let args = [known_integer(&ctx, f.element, 0)];
            let result = HloEvaluator::new(&ctx, f.module)
                .unwrap()
                .with_max_loop_iterations(0)
                .evaluate(Request::Computation {
                    callee: outer,
                    args: &args,
                });
            assert_limit(result, 0);
        }

        #[test]
        fn inline_fusion_embedded_while_inherits_limit_and_uses_fresh_call_frames() {
            let mut ctx = context();
            let f = fixture(&mut ctx, ElementType::S64, 0, 1, 0, Direction::Lt);
            let ty = f.parameter.get_type(&ctx);
            let (owner, block, argument) = computation(&mut ctx, f.module, "outer", ty, ty);
            let fusion = FusionOp::new(&mut ctx, ty, vec![argument], FusionKind::Loop);
            let inner = BasicBlock::new(&mut ctx, None, vec![ty]);
            inner.insert_at_back(fusion.body(&ctx), &ctx);
            let argument = inner.deref(&ctx).get_argument(0);
            let call = CallOp::new(&mut ctx, ty, vec![argument], Callees(vec!["main".into()]))
                .get_operation();
            let call = append(&ctx, inner, call, "inner.call");
            finish(&mut ctx, inner, call);
            let root = append(&ctx, block, fusion.get_operation(), "fusion");
            finish(&mut ctx, block, root);
            crate::ir::set_entry(&ctx, f.module, Identifier::new("outer")).unwrap();
            let args = [known_integer(&ctx, f.element, 0)];
            assert_limit(
                HloEvaluator::new(&ctx, f.module)
                    .unwrap()
                    .with_max_loop_iterations(0)
                    .evaluate(Request::Computation {
                        callee: owner,
                        args: &args,
                    }),
                0,
            );
        }

        #[test]
        fn conditional_body_executes_only_selected_callee_effects_during_real_iteration() {
            let mut ctx = context();
            let f = conditional_body_loop(&mut ctx, true);
            let result =
                evaluate_simple(&ctx, &f, known_integer(&ctx, ElementType::S64, 0), 1).unwrap();
            assert_eq!(read_integer(&ctx, result), 1);
        }

        #[test]
        fn conditional_body_rejects_effect_in_selected_callee_without_evaluating_pure_shortcut() {
            let mut ctx = context();
            let f = conditional_body_loop(&mut ctx, false);
            let error =
                evaluate_simple(&ctx, &f, known_integer(&ctx, ElementType::S64, 0), 1).unwrap_err();
            assert!(matches!(
                error.err.downcast_ref::<EvaluationError>(),
                Some(EvaluationError::Effectful {
                    opcode: Opcode::CustomCall
                })
            ));
        }

        #[test]
        fn unique_while_caller_resolves_only_structurally_unchanged_tuple_leaf() {
            let mut ctx = context();
            let f = fixture_form(
                &mut ctx,
                ElementType::S64,
                0,
                0,
                1,
                Direction::Lt,
                Form::DynamicBound,
            );
            enclosing_while_caller(&mut ctx, &f, false);
            assert_eq!(
                read_integer(&ctx, evaluate(&ctx, &f, f.projection, 0, &[]).unwrap()),
                7
            );
            let unknown = EvaluatedValue::unknown(&ctx, f.parameter.get_type(&ctx)).unwrap();
            assert!(matches!(
                evaluate(&ctx, &f, f.projection, 0, &[(f.parameter, unknown)])
                    .unwrap()
                    .state(),
                ValueState::Unknown
            ));
            let replacement = known_integer(&ctx, f.element, 11);
            assert_eq!(
                read_integer(
                    &ctx,
                    evaluate(&ctx, &f, f.projection, 0, &[(f.parameter, replacement)]).unwrap()
                ),
                11
            );
        }

        #[test]
        fn changed_leaf_of_unique_while_caller_is_not_treated_as_constant_init() {
            let mut ctx = context();
            let f = fixture_form(
                &mut ctx,
                ElementType::S64,
                0,
                0,
                1,
                Direction::Lt,
                Form::DynamicBound,
            );
            enclosing_while_caller(&mut ctx, &f, true);
            assert!(matches!(
                evaluate(&ctx, &f, f.projection, 0, &[]).unwrap().state(),
                ValueState::Unknown
            ));
        }
    }
    mod carried_values {
        use super::*;

        #[test]
        fn max_zero_boolean_true_runs_one_body_but_does_not_recheck_new_false_condition() {
            let mut ctx = context();
            let shape = shape::scalar(&ctx, ElementType::Pred).unwrap();
            let false_literal = Literal::from_bits(&ctx, shape, &[0]).unwrap();
            let f = simple_loop(&mut ctx, shape, None, Some(false_literal), false);
            let input = EvaluatedValue::known(&ctx, Literal::from_bits(&ctx, shape, &[1]).unwrap())
                .unwrap();
            assert_limit(evaluate_simple(&ctx, &f, input.clone(), 0), 0);
            let result = evaluate_simple(&ctx, &f, input, 1)
                .unwrap()
                .into_literal(&ctx)
                .unwrap();
            assert!(!literal_semantics::scalar_predicate(&ctx, &result).unwrap());
        }

        #[test]
        fn max_zero_boolean_false_returns_before_body_without_integer_iv_fabrication() {
            let mut ctx = context();
            let shape = shape::scalar(&ctx, ElementType::Pred).unwrap();
            let true_literal = Literal::from_bits(&ctx, shape, &[1]).unwrap();
            let f = simple_loop(&mut ctx, shape, None, Some(true_literal), false);
            let input = EvaluatedValue::known(&ctx, Literal::from_bits(&ctx, shape, &[0]).unwrap())
                .unwrap();
            let result = evaluate_simple(&ctx, &f, input, 0)
                .unwrap()
                .into_literal(&ctx)
                .unwrap();
            assert!(!literal_semantics::scalar_predicate(&ctx, &result).unwrap());
        }

        #[test]
        fn unknown_boolean_initial_value_is_not_assumed_false() {
            let mut ctx = context();
            let shape = shape::scalar(&ctx, ElementType::Pred).unwrap();
            let f = simple_loop(&mut ctx, shape, None, None, false);
            let input = EvaluatedValue::unknown(&ctx, shape).unwrap();
            let result = evaluate_simple(&ctx, &f, input, 0).unwrap();
            assert!(matches!(result.state(), ValueState::Unknown));
            let error = result.into_literal(&ctx).unwrap_err();
            assert!(matches!(
                error.err.downcast_ref::<EvaluationError>(),
                Some(EvaluationError::UnknownValue)
            ));
        }

        #[test]
        fn raw_dynamic_carried_array_preserves_layout_runtime_size_and_inactive_bytes() {
            let mut ctx = context();
            let shape = shape::dynamic_array(
                &ctx,
                ElementType::S4,
                vec![crate::shape::Dimension::Bounded(3)],
            )
            .unwrap();
            let literal =
                Literal::from_dynamic_storage_bytes(&ctx, shape, vec![0xff, 0xaa, 0xbb], vec![1])
                    .unwrap();
            let f = simple_loop(&mut ctx, shape, Some(false), None, false);
            let input = EvaluatedValue::known(&ctx, literal.clone()).unwrap();
            let result = evaluate_simple(&ctx, &f, input, 0)
                .unwrap()
                .into_literal(&ctx)
                .unwrap();
            assert_eq!(result, literal);
        }

        #[test]
        fn dynamic_integer_body_math_is_explicitly_unsupported_not_blanket_loop_rejection() {
            let mut ctx = context();
            let shape = shape::dynamic_array(
                &ctx,
                ElementType::S64,
                vec![crate::shape::Dimension::Bounded(1)],
            )
            .unwrap();
            let literal =
                Literal::from_dynamic_storage_bytes(&ctx, shape, vec![0; 8], vec![1]).unwrap();
            let f = simple_loop(&mut ctx, shape, Some(true), None, true);
            let error = evaluate_simple(&ctx, &f, EvaluatedValue::known(&ctx, literal).unwrap(), 0)
                .unwrap_err();
            assert!(matches!(
                error.err.downcast_ref::<EvaluationError>(),
                Some(EvaluationError::UnsupportedValue {
                    opcode: Opcode::Add,
                    ..
                })
            ));
        }

        #[test]
        fn executed_floating_body_math_reaches_the_configured_iteration_limit() {
            let mut ctx = context();
            let shape = shape::scalar(&ctx, ElementType::F32).unwrap();
            let f = simple_loop(&mut ctx, shape, Some(true), None, true);
            let input = EvaluatedValue::known(&ctx, Literal::from_bits(&ctx, shape, &[0]).unwrap())
                .unwrap();
            let error = evaluate_simple(&ctx, &f, input, 0).unwrap_err();
            assert!(matches!(
                error.err.downcast_ref::<EvaluationError>(),
                Some(EvaluationError::LoopIterationLimit { limit: 0 })
            ));
        }

        #[test]
        fn initial_false_does_not_execute_unsupported_floating_body_math() {
            let mut ctx = context();
            let shape = shape::scalar(&ctx, ElementType::F32).unwrap();
            let f = simple_loop(&mut ctx, shape, Some(false), None, true);
            let literal = Literal::from_bits(&ctx, shape, &[0x7f80_0042]).unwrap();
            let input = EvaluatedValue::known(&ctx, literal.clone()).unwrap();
            assert_eq!(
                evaluate_simple(&ctx, &f, input, 0)
                    .unwrap()
                    .into_literal(&ctx)
                    .unwrap(),
                literal
            );
        }

        #[test]
        fn tuple_with_token_and_dynamic_raw_payload_is_owned_and_complete_on_initial_false() {
            let mut ctx = context();
            let shape = shape::dynamic_array(
                &ctx,
                ElementType::U8,
                vec![crate::shape::Dimension::Bounded(3)],
            )
            .unwrap();
            let array =
                Literal::from_dynamic_storage_bytes(&ctx, shape, vec![1, 2, 3], vec![1]).unwrap();
            let tuple = Literal::tuple(&ctx, vec![Literal::token(&ctx), array]).unwrap();
            let f = simple_loop(&mut ctx, tuple.shape, Some(false), None, false);
            let result = evaluate_simple(
                &ctx,
                &f,
                EvaluatedValue::known(&ctx, tuple.clone()).unwrap(),
                0,
            )
            .unwrap();
            assert!(matches!(
                result.subvalue(&[0]).unwrap().state(),
                ValueState::Token
            ));
            assert_eq!(result.into_literal(&ctx).unwrap(), tuple);
        }

        #[test]
        fn empty_tuple_state_does_not_allocate_fake_array_or_require_iv() {
            let mut ctx = context();
            let tuple = Literal::tuple(&ctx, vec![]).unwrap();
            let f = simple_loop(&mut ctx, tuple.shape, Some(false), None, false);
            let result = evaluate_simple(
                &ctx,
                &f,
                EvaluatedValue::known(&ctx, tuple.clone()).unwrap(),
                0,
            )
            .unwrap();
            assert!(result.is_known());
            assert_eq!(result.into_literal(&ctx).unwrap(), tuple);
        }

        #[test]
        fn u64_above_i64_max_declines_static_proof_but_executes_before_limit() {
            let mut ctx = context();
            let f = fixture(
                &mut ctx,
                ElementType::U64,
                i128::from(i64::MAX) + 1,
                i128::from(u64::MAX),
                1,
                Direction::Lt,
            );
            assert_limit(complete(&ctx, &f, 0), 0);
            assert!(matches!(
                evaluate(&ctx, &f, f.projection, 0, &[]).unwrap().state(),
                ValueState::Unknown
            ));
        }

        #[test]
        fn unknown_nested_tuple_path_is_owned_unknown_not_an_induction_shortcut_or_zero() {
            let mut ctx = context();
            let integer = shape::scalar(&ctx, ElementType::S64).unwrap();
            let inner = shape::tuple(&ctx, vec![integer]).unwrap();
            let shape = shape::tuple(&ctx, vec![inner, integer]).unwrap();
            let f = simple_loop(&mut ctx, shape, Some(false), None, false);
            let mut session = Session::new(&ctx, f.module).unwrap();
            let owner = session.owner(f.root).unwrap();
            let mut frame = Frame::new(&ctx, owner).unwrap();
            frame.bindings.insert(
                f.parameter,
                EvaluatedValue::undetermined(&ctx, shape).unwrap(),
            );
            let result = session
                .evaluate_while(
                    &mut frame,
                    f.root.defining_op().unwrap(),
                    &[0, 0],
                    DependencyMode::Recursive,
                )
                .unwrap();
            assert_eq!(result.shape(), integer);
            assert!(matches!(result.state(), ValueState::Unknown));
        }
    }
    use super::*;
    use crate::{
        attributes::{Bools, FusionKind},
        ops::{
            ReturnOp,
            control::{
                CallOp, ConditionalOp, ConstantOp, CustomCallOp, FusionOp, GetTupleElementOp,
                ParameterOp, TupleOp, WhileOp,
            },
            elementwise::{AddOp, CompareOp},
        },
        types::ElementType,
    };
    use pliron::identifier::Identifier;
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

    fn append(ctx: &Context, block: Ptr<BasicBlock>, op: Ptr<Operation>, name: &str) -> Value {
        static NEXT_NAME: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let suffix = NEXT_NAME.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        op.deref_mut(ctx).attributes.set(
            Identifier::new("hlo_name"),
            InstructionName(format!("{name}.{suffix}")),
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
        input: TypeHandle,
        output: TypeHandle,
    ) -> (Ptr<Operation>, Ptr<BasicBlock>, Value) {
        let signature = FunctionType::get(ctx, vec![input], vec![output]).into();
        let computation = ComputationOp::new(ctx, Identifier::new(name), signature);
        let parent = module
            .deref(ctx)
            .get_region(0)
            .deref(ctx)
            .get_entry_block()
            .unwrap();
        computation.get_operation().insert_at_back(parent, ctx);
        let block = BasicBlock::new(ctx, None, vec![]);
        block.insert_at_back(computation.body(ctx), ctx);
        let parameter = ParameterOp::new(ctx, input, vec![], Index(0));
        let p = append(ctx, block, parameter.get_operation(), "parameter");
        (computation.get_operation(), block, p)
    }

    fn constant(ctx: &mut Context, block: Ptr<BasicBlock>, literal: Literal, name: &str) -> Value {
        let op = ConstantOp::new(ctx, literal.shape, vec![], literal).get_operation();
        append(ctx, block, op, name)
    }

    fn integer(
        ctx: &mut Context,
        block: Ptr<BasicBlock>,
        element: ElementType,
        value: i128,
        name: &str,
    ) -> Value {
        let literal = literal_semantics::integer_scalar(ctx, element, value).unwrap();
        constant(ctx, block, literal, name)
    }

    fn project(
        ctx: &mut Context,
        block: Ptr<BasicBlock>,
        tuple: Value,
        index: usize,
        name: &str,
    ) -> Value {
        let ty = shape::subshape(ctx, tuple.get_type(ctx), &[index]).unwrap();
        let op = GetTupleElementOp::new(ctx, ty, vec![tuple], Index(index as u32)).get_operation();
        append(ctx, block, op, name)
    }

    struct Fixture {
        module: Ptr<Operation>,
        owner: Ptr<Operation>,
        body: Ptr<Operation>,
        condition: Ptr<Operation>,
        parameter: Value,
        input: Value,
        initial: Value,
        bound: Value,
        step: Value,
        loop_value: Value,
        projection: Value,
        effect: Ptr<Operation>,
        element: ElementType,
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Form {
        Carried,
        Inside,
        ComputedBound,
        DynamicBound,
        Copy,
        Call,
        CallTuple,
        Fusion,
        Swapped,
        ChangingBound,
    }

    fn fixture(
        ctx: &mut Context,
        element: ElementType,
        init: i128,
        bound: i128,
        step: i128,
        direction: Direction,
    ) -> Fixture {
        fixture_form(ctx, element, init, bound, step, direction, Form::Carried)
    }

    fn fixture_form(
        ctx: &mut Context,
        element: ElementType,
        init: i128,
        bound: i128,
        step: i128,
        direction: Direction,
        form: Form,
    ) -> Fixture {
        let module = ModuleOp::new(ctx, Identifier::new("while_evaluation")).get_operation();
        crate::ir::set_entry(ctx, module, Identifier::new("main")).unwrap();
        let ty = shape::scalar(ctx, element).unwrap();
        let pred = shape::scalar(ctx, ElementType::Pred).unwrap();
        let tuple = shape::tuple(ctx, vec![ty; 4]).unwrap();
        let (body, block, p) = computation(ctx, module, "body", tuple, tuple);
        let iv = project(ctx, block, p, 0, "body.iv");
        let b = project(ctx, block, p, 1, "body.bound");
        let s = project(ctx, block, p, 2, "body.step");
        let payload = project(ctx, block, p, 3, "body.payload");
        let add = AddOp::new(ctx, ty, vec![iv, s]).get_operation();
        let next = append(ctx, block, add, "body.next");
        let effect = CustomCallOp::new(ctx, ty, vec![payload], StringAttr::new("unused".into()))
            .get_operation();
        effect
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_has_side_effect"), Bools(vec![false]));
        append(ctx, block, effect, "body.effect");
        let b = if form == Form::ChangingBound { next } else { b };
        let root = TupleOp::new(ctx, tuple, vec![next, b, s, payload]).get_operation();
        let root = append(ctx, block, root, "body.root");
        finish(ctx, block, root);

        if matches!(form, Form::Call | Form::CallTuple) {
            let output = if form == Form::CallTuple {
                shape::tuple(ctx, vec![pred]).unwrap()
            } else {
                pred
            };
            let (_, block, p) = computation(ctx, module, "inner", tuple, output);
            let result = predicate(ctx, block, p, pred, bound, direction, Form::Carried);
            let result = if form == Form::CallTuple {
                let op = TupleOp::new(ctx, output, vec![result]).get_operation();
                append(ctx, block, op, "inner.tuple")
            } else {
                result
            };
            finish(ctx, block, result);
        }
        let (condition, block, p) = computation(ctx, module, "condition", tuple, pred);
        let result = match form {
            Form::Call | Form::CallTuple => {
                let output = if form == Form::CallTuple {
                    shape::tuple(ctx, vec![pred]).unwrap()
                } else {
                    pred
                };
                let call = CallOp::new(ctx, output, vec![p], Callees(vec!["inner".into()]))
                    .get_operation();
                let call = append(ctx, block, call, "condition.call");
                if form == Form::CallTuple {
                    project(ctx, block, call, 0, "condition.extract")
                } else {
                    call
                }
            }
            Form::Fusion => {
                let fusion = FusionOp::new(ctx, pred, vec![p], FusionKind::Loop);
                let inner = BasicBlock::new(ctx, None, vec![tuple]);
                inner.insert_at_back(fusion.body(ctx), ctx);
                let argument = inner.deref(ctx).get_argument(0);
                let result = predicate(ctx, inner, argument, pred, bound, direction, Form::Carried);
                finish(ctx, inner, result);
                append(ctx, block, fusion.get_operation(), "condition.fusion")
            }
            _ => predicate(ctx, block, p, pred, bound, direction, form),
        };
        finish(ctx, block, result);

        let (owner, block, parameter) = computation(ctx, module, "main", ty, ty);
        let initial = integer(ctx, block, element, init, "initial");
        let bound = if form == Form::DynamicBound {
            parameter
        } else {
            let b = integer(ctx, block, element, bound, "bound");
            if form == Form::ComputedBound {
                let zero = integer(ctx, block, element, 0, "zero");
                let op = AddOp::new(ctx, ty, vec![b, zero]).get_operation();
                append(ctx, block, op, "computed.bound")
            } else {
                b
            }
        };
        let step = integer(ctx, block, element, step, "step");
        let input = TupleOp::new(ctx, tuple, vec![initial, bound, step, parameter]).get_operation();
        let input = append(ctx, block, input, "input");
        let loop_value = WhileOp::new(
            ctx,
            tuple,
            vec![input],
            Callees(vec!["body".into(), "condition".into()]),
        )
        .get_operation();
        let loop_value = append(ctx, block, loop_value, "while");
        let projection = project(ctx, block, loop_value, 0, "final.iv");
        finish(ctx, block, projection);
        verification::verify(ctx, module).unwrap();
        Fixture {
            module,
            owner,
            body,
            condition,
            parameter,
            input,
            initial,
            bound,
            step,
            loop_value,
            projection,
            effect,
            element,
        }
    }

    fn predicate(
        ctx: &mut Context,
        block: Ptr<BasicBlock>,
        parameter: Value,
        pred: TypeHandle,
        bound: i128,
        direction: Direction,
        form: Form,
    ) -> Value {
        let iv = project(ctx, block, parameter, 0, "condition.iv");
        let b = if form == Form::Inside {
            let element = shape::element_type(ctx, iv.get_type(ctx)).unwrap();
            integer(ctx, block, element, bound, "condition.bound")
        } else {
            project(ctx, block, parameter, 1, "condition.bound")
        };
        let operands = if form == Form::Swapped {
            vec![b, iv]
        } else {
            vec![iv, b]
        };
        let op =
            CompareOp::new(ctx, pred, operands, direction, ComparisonOrder::Total).get_operation();
        let result = append(ctx, block, op, "condition.compare");
        if form == Form::Copy {
            let op = crate::ops::movement::CopyOp::new(ctx, pred, vec![result]).get_operation();
            append(ctx, block, op, "condition.copy")
        } else {
            result
        }
    }

    fn known_integer(ctx: &Context, element: ElementType, value: i128) -> EvaluatedValue {
        EvaluatedValue::known(
            ctx,
            literal_semantics::integer_scalar(ctx, element, value).unwrap(),
        )
        .unwrap()
    }

    fn evaluate(
        ctx: &Context,
        f: &Fixture,
        value: Value,
        limit: i64,
        substitutions: &[(Value, EvaluatedValue)],
    ) -> Result<EvaluatedValue> {
        HloEvaluator::new(ctx, f.module)?
            .with_max_loop_iterations(limit)
            .evaluate(Request::Instruction {
                value,
                substitutions,
                dependency_mode: DependencyMode::Recursive,
            })
    }

    fn complete(ctx: &Context, f: &Fixture, limit: i64) -> Result<EvaluatedValue> {
        evaluate(
            ctx,
            f,
            f.loop_value,
            limit,
            &[(
                f.parameter,
                known_integer(
                    ctx,
                    f.element,
                    if f.element.bit_width() < 6 { 0 } else { 19 },
                ),
            )],
        )
    }

    fn read_integer(ctx: &Context, value: EvaluatedValue) -> i64 {
        literal_semantics::scalar_integer(ctx, &value.into_literal(ctx).unwrap()).unwrap()
    }

    fn assert_limit(result: Result<EvaluatedValue>, expected: i64) {
        let error = result.unwrap_err();
        assert!(matches!(
            error.err.downcast_ref::<EvaluationError>(),
            Some(EvaluationError::LoopIterationLimit { limit }) if *limit == expected
        ));
    }

    fn wrapper(ctx: &mut Context, f: &Fixture, name: &'static str, value: i128) -> Ptr<Operation> {
        let ty = f.parameter.get_type(ctx);
        let (owner, block, _) = computation(ctx, f.module, name, ty, ty);
        let argument = integer(ctx, block, f.element, value, "argument");
        let call =
            CallOp::new(ctx, ty, vec![argument], Callees(vec!["main".into()])).get_operation();
        let call = append(ctx, block, call, "call");
        finish(ctx, block, call);
        owner
    }

    struct SimpleLoop {
        module: Ptr<Operation>,
        root: Value,
        parameter: Value,
    }

    fn simple_loop(
        ctx: &mut Context,
        shape: TypeHandle,
        condition: Option<bool>,
        body_value: Option<Literal>,
        add_body: bool,
    ) -> SimpleLoop {
        let module = ModuleOp::new(ctx, Identifier::new("simple_while")).get_operation();
        crate::ir::set_entry(ctx, module, Identifier::new("main")).unwrap();
        let pred = shape::scalar(ctx, ElementType::Pred).unwrap();
        let (_, block, parameter) = computation(ctx, module, "body", shape, shape);
        let result = if let Some(literal) = body_value {
            constant(ctx, block, literal, "body.constant")
        } else if add_body {
            let add = AddOp::new(ctx, shape, vec![parameter, parameter]).get_operation();
            append(ctx, block, add, "body.add")
        } else {
            parameter
        };
        finish(ctx, block, result);
        let (_, block, parameter) = computation(ctx, module, "condition", shape, pred);
        let result = if let Some(value) = condition {
            let literal = Literal::from_bits(ctx, pred, &[u128::from(value)]).unwrap();
            constant(ctx, block, literal, "condition.constant")
        } else {
            parameter
        };
        finish(ctx, block, result);
        let (_, block, parameter) = computation(ctx, module, "main", shape, shape);
        let root = WhileOp::new(
            ctx,
            shape,
            vec![parameter],
            Callees(vec!["body".into(), "condition".into()]),
        )
        .get_operation();
        let root = append(ctx, block, root, "while");
        finish(ctx, block, root);
        verification::verify(ctx, module).unwrap();
        SimpleLoop {
            module,
            root,
            parameter,
        }
    }

    fn evaluate_simple(
        ctx: &Context,
        f: &SimpleLoop,
        input: EvaluatedValue,
        limit: i64,
    ) -> Result<EvaluatedValue> {
        HloEvaluator::new(ctx, f.module)?
            .with_max_loop_iterations(limit)
            .evaluate(Request::Instruction {
                value: f.root,
                substitutions: &[(f.parameter, input)],
                dependency_mode: DependencyMode::Recursive,
            })
    }

    fn conditional_body_loop(ctx: &mut Context, select_good: bool) -> SimpleLoop {
        let module = ModuleOp::new(ctx, Identifier::new("conditional_body")).get_operation();
        crate::ir::set_entry(ctx, module, Identifier::new("main")).unwrap();
        let ty = shape::scalar(ctx, ElementType::S64).unwrap();
        let pred = shape::scalar(ctx, ElementType::Pred).unwrap();
        let (_, block, _) = computation(ctx, module, "good", ty, ty);
        let one = integer(ctx, block, ElementType::S64, 1, "one");
        finish(ctx, block, one);
        let (_, block, parameter) = computation(ctx, module, "bad", ty, ty);
        let effect = CustomCallOp::new(ctx, ty, vec![parameter], StringAttr::new("effect".into()))
            .get_operation();
        effect
            .deref_mut(ctx)
            .attributes
            .set(Identifier::new("hlo_has_side_effect"), Bools(vec![true]));
        let effect = append(ctx, block, effect, "effect");
        finish(ctx, block, effect);
        let (_, block, parameter) = computation(ctx, module, "body", ty, ty);
        let selector_literal = Literal::from_bits(ctx, pred, &[u128::from(select_good)]).unwrap();
        let selector = constant(ctx, block, selector_literal, "selector");
        let result = ConditionalOp::new(
            ctx,
            ty,
            vec![selector, parameter, parameter],
            Callees(vec!["good".into(), "bad".into()]),
        )
        .get_operation();
        let result = append(ctx, block, result, "conditional");
        finish(ctx, block, result);
        let (_, block, parameter) = computation(ctx, module, "condition", ty, pred);
        let one = integer(ctx, block, ElementType::S64, 1, "one");
        let result = CompareOp::new(
            ctx,
            pred,
            vec![parameter, one],
            Direction::Lt,
            ComparisonOrder::Total,
        )
        .get_operation();
        let result = append(ctx, block, result, "compare");
        finish(ctx, block, result);
        let (_, block, parameter) = computation(ctx, module, "main", ty, ty);
        let root = WhileOp::new(
            ctx,
            ty,
            vec![parameter],
            Callees(vec!["body".into(), "condition".into()]),
        )
        .get_operation();
        let root = append(ctx, block, root, "while");
        finish(ctx, block, root);
        verification::verify(ctx, module).unwrap();
        SimpleLoop {
            module,
            root,
            parameter,
        }
    }

    fn enclosing_while_caller(ctx: &mut Context, f: &Fixture, changes_bound: bool) {
        let ty = f.parameter.get_type(ctx);
        let pred = shape::scalar(ctx, ElementType::Pred).unwrap();
        let state = shape::tuple(ctx, vec![ty, ty]).unwrap();
        let (_, block, p) = computation(ctx, f.module, "outer_body", state, state);
        let iv = project(ctx, block, p, 0, "outer.iv");
        let bound = project(ctx, block, p, 1, "outer.bound");
        let call = CallOp::new(ctx, ty, vec![bound], Callees(vec!["main".into()])).get_operation();
        append(ctx, block, call, "inner.loop.call");
        let one = integer(ctx, block, f.element, 1, "one");
        let next = AddOp::new(ctx, ty, vec![iv, one]).get_operation();
        let next = append(ctx, block, next, "outer.next");
        let bound = if changes_bound {
            let add = AddOp::new(ctx, ty, vec![bound, one]).get_operation();
            append(ctx, block, add, "outer.changed.bound")
        } else {
            bound
        };
        let root = TupleOp::new(ctx, state, vec![next, bound]).get_operation();
        let root = append(ctx, block, root, "outer.body.root");
        finish(ctx, block, root);

        let (_, block, _) = computation(ctx, f.module, "outer_condition", state, pred);
        let literal = Literal::from_bits(ctx, pred, &[0]).unwrap();
        let condition = constant(ctx, block, literal, "false");
        finish(ctx, block, condition);

        let (_, block, _) = computation(ctx, f.module, "outer", ty, ty);
        let zero = integer(ctx, block, f.element, 0, "zero");
        let seven = integer(ctx, block, f.element, 7, "seven");
        let initial = TupleOp::new(ctx, state, vec![zero, seven]).get_operation();
        let initial = append(ctx, block, initial, "outer.initial");
        let loop_value = WhileOp::new(
            ctx,
            state,
            vec![initial],
            Callees(vec!["outer_body".into(), "outer_condition".into()]),
        )
        .get_operation();
        let loop_value = append(ctx, block, loop_value, "outer.while");
        let root = project(ctx, block, loop_value, 0, "outer.final.iv");
        finish(ctx, block, root);
        crate::ir::set_entry(ctx, f.module, Identifier::new("outer")).unwrap();
        verification::verify(ctx, f.module).unwrap();
    }
}
