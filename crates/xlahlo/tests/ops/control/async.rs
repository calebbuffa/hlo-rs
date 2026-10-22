use crate::control_support::*;
use xlahlo::{
    attributes::*,
    inference::control as i,
    ops::{
        HloOpInterface,
        control::*,
        elementwise::{AddOp, NegateOp, SinOp},
    },
    pliron::{
        builtin::attributes::{StringAttr, TypeAttr},
        common_traits::Verify,
        op::Op,
    },
    shape,
    types::ElementType::*,
    verification::{self, VerifyOptions},
};

#[test]
fn copy_context_allows_memory_space_changes_but_not_physical_transposes() {
    let mut ctx = context();
    let input = shape(&ctx, F32, &[2, 3]);
    let context = i::infer_copy_start(&ctx, input).unwrap();
    let p = param(&mut ctx, input, 0);
    let pv = p.result(&ctx);
    let start = CopyStartOp::new(&mut ctx, context, vec![pv]);
    start.verify(&ctx).unwrap();
    let sv = start.result(&ctx);
    CopyDoneOp::new(&mut ctx, input, vec![sv])
        .verify(&ctx)
        .unwrap();
    let mut layout = shape::layout(&ctx, input).unwrap().unwrap().clone();
    layout.memory_space = 5;
    let destination = shape::with_layout(&ctx, input, layout.clone()).unwrap();
    let new_context = shape::tuple(&ctx, vec![destination, input, shape(&ctx, U32, &[])]).unwrap();
    let context_p = param(&mut ctx, new_context, 1);
    let cv = context_p.result(&ctx);
    CopyDoneOp::new(&mut ctx, destination, vec![cv])
        .verify_with_options(
            &ctx,
            VerifyOptions {
                layout_sensitive: true,
                ..Default::default()
            },
        )
        .unwrap();
    layout.minor_to_major.reverse();
    let transpose = shape::with_layout(&ctx, input, layout).unwrap();
    let bad = shape::tuple(&ctx, vec![transpose, input, shape(&ctx, U32, &[])]).unwrap();
    let bp = param(&mut ctx, bad, 1);
    let bv = bp.result(&ctx);
    let done = CopyDoneOp::new(&mut ctx, transpose, vec![bv]);
    done.verify(&ctx).unwrap();
    assert!(
        done.verify_with_options(
            &ctx,
            VerifyOptions {
                layout_sensitive: true,
                ..Default::default()
            }
        )
        .is_err()
    );
}
#[test]
fn copy_pair_requires_one_real_user_and_the_done_requires_a_start() {
    for mode in 0..4 {
        let mut ctx = context();
        let f = shape(&ctx, F32, &[2, 3]);
        let context = i::infer_copy_start(&ctx, f).unwrap();
        let (comp, block, args) = computation(
            &mut ctx,
            "main",
            vec![f],
            if mode == 2 { context } else { f },
        );
        let start = CopyStartOp::new(&mut ctx, context, vec![args[0]]);
        start.get_operation().insert_at_back(block, &ctx);
        let sv = start.result(&ctx);
        if mode == 1 {
            let extra = CopyDoneOp::new(&mut ctx, f, vec![sv]);
            extra.get_operation().insert_at_back(block, &ctx);
        }
        let root = if mode == 2 {
            let barrier = OptimizationBarrierOp::new(&mut ctx, context, vec![sv]);
            barrier.get_operation().insert_at_back(block, &ctx);
            barrier.result(&ctx)
        } else {
            let done = CopyDoneOp::new(&mut ctx, f, vec![if mode == 3 { args[0] } else { sv }]);
            done.get_operation().insert_at_back(block, &ctx);
            done.result(&ctx)
        };
        ret(&mut ctx, block, root);
        assert_eq!(
            verification::verify(&ctx, comp.get_operation()).is_ok(),
            matches!(mode, 0 | 2)
        );
    }
}
#[test]
fn channel_pairing_checks_users_ids_flags_and_done_producers() {
    for edit in 0..5 {
        let mut ctx = context();
        let f = shape(&ctx, F32, &[2]);
        let token = shape::token(&ctx);
        let context = i::infer_send(&ctx, f).unwrap();
        let (comp, block, args) = computation(&mut ctx, "main", vec![f], token);
        let all = AfterAllOp::new(&mut ctx, token, vec![]);
        all.get_operation().insert_at_back(block, &ctx);
        let tv = all.result(&ctx);
        let start = SendOp::new(
            &mut ctx,
            context,
            vec![args[0], tv],
            OptionalInteger::Value(1),
            Bools(vec![false]),
        );
        start.get_operation().insert_at_back(block, &ctx);
        let sv = start.result(&ctx);
        let done = SendDoneOp::new(
            &mut ctx,
            token,
            vec![if edit == 3 { args[0] } else { sv }],
            OptionalInteger::Value(if edit == 1 { 2 } else { 1 }),
            Bools(vec![edit == 2]),
        );
        done.get_operation().insert_at_back(block, &ctx);
        let dv = done.result(&ctx);
        if edit == 4 {
            let extra = SendDoneOp::new(
                &mut ctx,
                token,
                vec![sv],
                OptionalInteger::Value(1),
                Bools(vec![false]),
            );
            extra.get_operation().insert_at_back(block, &ctx);
        }
        ret(&mut ctx, block, dv);
        assert_eq!(
            verification::verify(&ctx, comp.get_operation()).is_ok(),
            edit == 0
        );
    }
}
#[test]
fn receive_context_and_feed_host_shapes_are_checked() {
    let mut ctx = context();
    let f = shape(&ctx, F32, &[2]);
    let token = shape::token(&ctx);
    let context = i::infer_recv(&ctx, f).unwrap();
    let all = AfterAllOp::new(&mut ctx, token, vec![]);
    let tv = all.result(&ctx);
    let recv = RecvOp::new(
        &mut ctx,
        context,
        vec![tv],
        OptionalInteger::Value(2),
        Bools(vec![false]),
    );
    recv.verify(&ctx).unwrap();
    let done_ty = i::infer_recv_done(&ctx, context).unwrap();
    let rv = recv.result(&ctx);
    RecvDoneOp::new(
        &mut ctx,
        done_ty,
        vec![rv],
        OptionalInteger::Value(2),
        Bools(vec![false]),
    )
    .verify(&ctx)
    .unwrap();
    InfeedOp::new(&mut ctx, done_ty, vec![tv])
        .verify(&ctx)
        .unwrap();
    let p = param(&mut ctx, f, 0);
    let pv = p.result(&ctx);
    assert!(
        InfeedOp::new(&mut ctx, done_ty, vec![pv])
            .verify(&ctx)
            .is_err()
    );
    let out = OutfeedOp::new(&mut ctx, token, vec![pv, tv]);
    assert!(out.verify(&ctx).is_err());
    attr(
        &ctx,
        out.get_operation(),
        "hlo_outfeed_shape",
        TypeAttr::new(f),
    );
    out.verify(&ctx).unwrap();
    attr(
        &ctx,
        out.get_operation(),
        "hlo_outfeed_shape",
        TypeAttr::new(shape(&ctx, F32, &[3])),
    );
    assert!(out.verify(&ctx).is_err());
}
#[test]
fn host_transfers_sharing_a_channel_compare_operand_shapes_and_frontend_attributes() {
    for changed in [false, true] {
        let mut ctx = context();
        let f = shape(&ctx, F32, &[2]);
        let token = shape::token(&ctx);
        let tuple = shape::tuple(&ctx, vec![token, token]).unwrap();
        let context = i::infer_send(&ctx, f).unwrap();
        let (comp, block, args) = computation(&mut ctx, "main", vec![f, f], tuple);
        let all = AfterAllOp::new(&mut ctx, token, vec![]);
        all.get_operation().insert_at_back(block, &ctx);
        let tv = all.result(&ctx);
        let mut dones = Vec::new();
        for (index, &value) in args.iter().enumerate() {
            let start = SendOp::new(
                &mut ctx,
                context,
                vec![value, tv],
                OptionalInteger::Value(1),
                Bools(vec![true]),
            );
            start.get_operation().insert_at_back(block, &ctx);
            if changed && index == 1 {
                attr(
                    &ctx,
                    start.get_operation(),
                    "hlo_frontend_attributes",
                    FrontendAttributes::new([("key".into(), "different".into())]),
                );
            }
            let sv = start.result(&ctx);
            let done = SendDoneOp::new(
                &mut ctx,
                token,
                vec![sv],
                OptionalInteger::Value(1),
                Bools(vec![true]),
            );
            done.get_operation().insert_at_back(block, &ctx);
            dones.push(done.result(&ctx));
        }
        let root = TupleOp::new(&mut ctx, tuple, dones);
        root.get_operation().insert_at_back(block, &ctx);
        let rv = root.result(&ctx);
        ret(&mut ctx, block, rv);
        assert_eq!(
            verification::verify(&ctx, comp.get_operation()).is_ok(),
            !changed
        );
    }
}
#[test]
fn generic_async_chains_require_binding_single_done_and_matching_threads() {
    for mode in 0..5 {
        let mut ctx = context();
        let f = shape(&ctx, F32, &[]);
        let state = shape(&ctx, S32, &[]);
        let context =
            shape::tuple(&ctx, vec![shape::tuple(&ctx, vec![f]).unwrap(), f, state]).unwrap();
        let (work, wb, args) = computation(&mut ctx, "work", vec![f], f);
        ret(&mut ctx, wb, args[0]);
        let (main, block, args) = computation(&mut ctx, "main", vec![f], f);
        let start = AsyncStartOp::new(&mut ctx, context, vec![args[0]], callees(&["work"]));
        start.get_operation().insert_at_back(block, &ctx);
        let sv = start.result(&ctx);
        if mode == 2 {
            attr(
                &ctx,
                start.get_operation(),
                "hlo_async_execution_thread",
                StringAttr::new("worker".into()),
            );
        }
        let update = AsyncUpdateOp::new(&mut ctx, context, vec![sv]);
        update.get_operation().insert_at_back(block, &ctx);
        let uv = update.result(&ctx);
        if mode == 3 {
            attr(
                &ctx,
                update.get_operation(),
                "hlo_async_execution_thread",
                StringAttr::new("worker".into()),
            );
        }
        let done = AsyncDoneOp::new(&mut ctx, f, vec![if mode == 4 { args[0] } else { uv }]);
        done.get_operation().insert_at_back(block, &ctx);
        let dv = done.result(&ctx);
        if mode == 1 {
            let duplicate = AsyncDoneOp::new(&mut ctx, f, vec![uv]);
            duplicate.get_operation().insert_at_back(block, &ctx);
        }
        ret(&mut ctx, block, dv);
        let m = module(&mut ctx, &[work, main], "main");
        assert_eq!(
            verification::verify(&ctx, m.get_operation()).is_ok(),
            mode == 0,
            "mode {mode}"
        );
    }
}
#[test]
fn async_update_extends_partial_bindings_and_done_requires_full_parameters() {
    let mut ctx = context();
    let f = shape(&ctx, F32, &[]);
    let empty = shape::tuple(&ctx, vec![]).unwrap();
    let state = shape(&ctx, S32, &[]);
    let partial = shape::tuple(
        &ctx,
        vec![shape::tuple(&ctx, vec![f]).unwrap(), empty, state],
    )
    .unwrap();
    let full = shape::tuple(
        &ctx,
        vec![shape::tuple(&ctx, vec![f, f]).unwrap(), f, state],
    )
    .unwrap();
    let (work, wb, args) = computation(&mut ctx, "work", vec![f, f], f);
    let add = AddOp::new(&mut ctx, f, args);
    add.get_operation().insert_at_back(wb, &ctx);
    let av = add.result(&ctx);
    ret(&mut ctx, wb, av);
    let (main, block, args) = computation(&mut ctx, "main", vec![f, f], f);
    let start = AsyncStartOp::new(&mut ctx, partial, vec![args[0]], callees(&["work"]));
    start.get_operation().insert_at_back(block, &ctx);
    let sv = start.result(&ctx);
    let update = AsyncUpdateOp::new(&mut ctx, full, vec![sv, args[1]]);
    update.get_operation().insert_at_back(block, &ctx);
    let uv = update.result(&ctx);
    let done = AsyncDoneOp::new(&mut ctx, f, vec![uv]);
    done.get_operation().insert_at_back(block, &ctx);
    let dv = done.result(&ctx);
    ret(&mut ctx, block, dv);
    let m = module(&mut ctx, &[work, main], "main");
    verification::verify(&ctx, m.get_operation()).unwrap();
    let bad = AsyncUpdateOp::new(&mut ctx, full, vec![sv]);
    assert!(bad.verify(&ctx).is_err());
}
#[test]
fn generic_async_rejects_main_thread_extra_work_and_non_async_owners() {
    let mut ctx = context();
    let f = shape(&ctx, F32, &[]);
    let state = shape(&ctx, S32, &[]);
    let context = shape::tuple(&ctx, vec![shape::tuple(&ctx, vec![f]).unwrap(), f, state]).unwrap();
    let (work, wb, args) = computation(&mut ctx, "work", vec![f], f);
    let unused = NegateOp::new(&mut ctx, f, vec![args[0]]);
    unused.get_operation().insert_at_back(wb, &ctx);
    ret(&mut ctx, wb, args[0]);
    let (main, block, args) = computation(&mut ctx, "main", vec![f], f);
    let start = AsyncStartOp::new(&mut ctx, context, vec![args[0]], callees(&["work"]));
    start.get_operation().insert_at_back(block, &ctx);
    let sv = start.result(&ctx);
    let done = AsyncDoneOp::new(&mut ctx, f, vec![sv]);
    done.get_operation().insert_at_back(block, &ctx);
    let dv = done.result(&ctx);
    ret(&mut ctx, block, dv);
    let m = module(&mut ctx, &[work, main], "main");
    assert!(verification::verify(&ctx, m.get_operation()).is_err());
    attr(
        &ctx,
        start.get_operation(),
        "hlo_frontend_attributes",
        FrontendAttributes::new([("_collectives_group".into(), "1".into())]),
    );
    verification::verify(&ctx, m.get_operation()).unwrap();
}
#[test]
fn send_recv_deadlock_policy_rejects_cycles_and_unmatched_pairs() {
    for mode in 0..3 {
        let mut ctx = context();
        let f = shape(&ctx, F32, &[2]);
        let token = shape::token(&ctx);
        let context = i::infer_send(&ctx, f).unwrap();
        let recv_done_ty = i::infer_recv_done(&ctx, context).unwrap();
        let (comp, block, args) = computation(&mut ctx, "main", vec![f], token);
        let all = AfterAllOp::new(&mut ctx, token, vec![]);
        all.get_operation().insert_at_back(block, &ctx);
        let tv = all.result(&ctx);
        let send = SendOp::new(
            &mut ctx,
            context,
            vec![args[0], tv],
            OptionalInteger::Value(1),
            Bools(vec![false]),
        );
        send.get_operation().insert_at_back(block, &ctx);
        let sv = send.result(&ctx);
        let send_done = SendDoneOp::new(
            &mut ctx,
            token,
            vec![sv],
            OptionalInteger::Value(1),
            Bools(vec![false]),
        );
        send_done.get_operation().insert_at_back(block, &ctx);
        let pairs = SourceTargetPairs(vec![SourceTarget {
            source: 0,
            target: 1,
        }]);
        attr(
            &ctx,
            send.get_operation(),
            "hlo_source_target_pairs",
            pairs.clone(),
        );
        if mode != 2 {
            let recv = RecvOp::new(
                &mut ctx,
                context,
                vec![tv],
                OptionalInteger::Value(2),
                Bools(vec![false]),
            );
            recv.get_operation().insert_at_back(block, &ctx);
            let rv = recv.result(&ctx);
            attr(&ctx, recv.get_operation(), "hlo_source_target_pairs", pairs);
            let done = RecvDoneOp::new(
                &mut ctx,
                recv_done_ty,
                vec![rv],
                OptionalInteger::Value(2),
                Bools(vec![false]),
            );
            done.get_operation().insert_at_back(block, &ctx);
        }

        if mode == 1 {
            attr(
                &ctx,
                send.get_operation(),
                "hlo_source_target_pairs",
                SourceTargetPairs(vec![
                    SourceTarget {
                        source: 0,
                        target: 1,
                    },
                    SourceTarget {
                        source: 1,
                        target: 0,
                    },
                ]),
            );
        }
        let root = send_done.result(&ctx);
        ret(&mut ctx, block, root);
        verification::verify(&ctx, comp.get_operation()).unwrap();
        assert_eq!(
            verification::verify_with_options(
                &ctx,
                comp.get_operation(),
                VerifyOptions {
                    check_for_collective_deadlocks: true,
                    ..Default::default()
                }
            )
            .is_ok(),
            mode == 0
        );
    }
}

#[test]
fn wrapped_computation_equality_ignores_irrelevant_opcode_fields_but_preserves_accuracy() {
    for mode in 0..3 {
        let mut ctx = context();
        let f = shape(&ctx, F32, &[]);
        let state = shape(&ctx, S32, &[]);
        let context =
            shape::tuple(&ctx, vec![shape::tuple(&ctx, vec![f]).unwrap(), f, state]).unwrap();
        let (first, fb, args) = computation(&mut ctx, "work1", vec![f], f);
        let sine = SinOp::new(&mut ctx, f, vec![args[0]]);
        sine.get_operation().insert_at_back(fb, &ctx);
        let sv = sine.result(&ctx);
        ret(&mut ctx, fb, sv);
        let (second, sb, args) = computation(&mut ctx, "work2", vec![f], f);
        let sine = SinOp::new(&mut ctx, f, vec![args[0]]);
        sine.get_operation().insert_at_back(sb, &ctx);
        let sv = sine.result(&ctx);
        ret(&mut ctx, sb, sv);
        attr(&ctx, sine.get_operation(), "hlo_dimensions", Axes(vec![91]));
        if mode > 0 {
            attr(
                &ctx,
                sine.get_operation(),
                "hlo_result_accuracy",
                if mode == 1 {
                    ResultAccuracy::Default
                } else {
                    ResultAccuracy::Highest
                },
            );
        }
        let (main, block, args) = computation(&mut ctx, "main", vec![f], f);
        let start = AsyncStartOp::new(&mut ctx, context, vec![args[0]], callees(&["work1"]));
        start.get_operation().insert_at_back(block, &ctx);
        let sv = start.result(&ctx);
        let update = AsyncUpdateOp::new(&mut ctx, context, vec![sv]);
        update.get_operation().insert_at_back(block, &ctx);
        let uv = update.result(&ctx);
        attr(
            &ctx,
            update.get_operation(),
            "hlo_callees",
            callees(&["work2"]),
        );
        let done = AsyncDoneOp::new(&mut ctx, f, vec![uv]);
        done.get_operation().insert_at_back(block, &ctx);
        let dv = done.result(&ctx);
        ret(&mut ctx, block, dv);
        let m = module(&mut ctx, &[first, second, main], "main");
        assert_eq!(
            verification::verify(&ctx, m.get_operation()).is_ok(),
            mode < 2
        );
    }
}

#[test]
fn async_aliases_use_full_wrapped_parameters_and_update_context_child_two() {
    for mode in 0..3 {
        let mut ctx = context();
        let f = shape(&ctx, F32, &[]);
        let state = shape(&ctx, S32, &[]);
        let context =
            shape::tuple(&ctx, vec![shape::tuple(&ctx, vec![f]).unwrap(), f, state]).unwrap();
        let (work, wb, args) = computation(&mut ctx, "work", vec![f], f);
        ret(&mut ctx, wb, args[0]);
        let (main, block, args) = computation(&mut ctx, "main", vec![f], f);
        let start = AsyncStartOp::new(&mut ctx, context, vec![args[0]], callees(&["work"]));
        start.get_operation().insert_at_back(block, &ctx);
        let sv = start.result(&ctx);
        attr(
            &ctx,
            start.get_operation(),
            "hlo_aliases",
            OutputOperandAliases(vec![OutputOperandAliasing {
                output_shape_index: Axes(if mode == 1 { vec![] } else { vec![0, 0] }),
                operand_index: 0,
                operand_shape_index: Axes(vec![]),
            }]),
        );
        let update = AsyncUpdateOp::new(&mut ctx, context, vec![sv]);
        update.get_operation().insert_at_back(block, &ctx);
        let uv = update.result(&ctx);
        attr(
            &ctx,
            update.get_operation(),
            "hlo_aliases",
            OutputOperandAliases(vec![OutputOperandAliasing {
                output_shape_index: Axes(vec![if mode == 2 { 1 } else { 2 }]),
                operand_index: 0,
                operand_shape_index: Axes(vec![if mode == 2 { 1 } else { 2 }]),
            }]),
        );
        let done = AsyncDoneOp::new(&mut ctx, f, vec![uv]);
        done.get_operation().insert_at_back(block, &ctx);
        let dv = done.result(&ctx);
        ret(&mut ctx, block, dv);
        let m = module(&mut ctx, &[work, main], "main");
        assert_eq!(
            verification::verify(&ctx, m.get_operation()).is_ok(),
            mode == 0
        );
    }
}
