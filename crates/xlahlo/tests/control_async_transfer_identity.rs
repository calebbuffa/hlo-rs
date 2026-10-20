#[path = "control_support/mod.rs"]
mod support;

use support::*;
use xlahlo::{
    attributes::{
        Bools, Bytes, FrontendAttributes, OptionalInteger, Payload, ResultAccuracy, SourceTarget,
        SourceTargetPairs, StringPair,
    },
    inference::control::{infer_recv, infer_recv_done, infer_send},
    ir::{self, BackendConfigPayload, ExecutionThread, MetadataPayloads},
    ops::control::{AsyncDoneOp, AsyncStartOp, RecvDoneOp, RecvOp, SendDoneOp, SendOp},
    pliron::{
        builtin::{attributes::StringAttr, ops::ModuleOp},
        context::{Context, Ptr},
        irfmt::parsers::spaced,
        op::Op,
        operation::Operation,
        parsable::parse_from_str,
        printable::Printable,
    },
    shape::Shape,
    types::ElementType,
    verification,
};

fn pair(
    ctx: &mut Context,
    send: bool,
    complete: bool,
) -> (ModuleOp, Ptr<Operation>, Ptr<Operation>) {
    let array = shape(ctx, ElementType::F32, &[2]);
    let token = Shape::token(ctx);
    let inputs = if send {
        vec![array, token]
    } else {
        vec![token]
    };
    let transfer_context = if send {
        infer_send(ctx, array).unwrap()
    } else {
        infer_recv(ctx, array).unwrap()
    };
    let output = if !complete {
        transfer_context
    } else if send {
        token
    } else {
        infer_recv_done(ctx, transfer_context).unwrap()
    };
    let mut computations = vec![];
    let mut operations = vec![];
    for name in ["first", "second"] {
        let (computation, block, args) = computation(ctx, name, inputs.clone(), output);
        attr(
            ctx,
            computation.get_operation(),
            "hlo_execution_thread",
            ExecutionThread("transfer".into()),
        );
        let operation = if send {
            SendOp::new(
                ctx,
                transfer_context,
                args,
                OptionalInteger::Value(7),
                Bools(vec![false]),
            )
            .get_operation()
        } else {
            RecvOp::new(
                ctx,
                transfer_context,
                args,
                OptionalInteger::Value(7),
                Bools(vec![false]),
            )
            .get_operation()
        };
        operation.insert_at_back(block, ctx);
        let result = operation.deref(ctx).get_result(0);
        let operation = if !complete {
            operation
        } else {
            let done = if send {
                SendDoneOp::new(
                    ctx,
                    output,
                    vec![result],
                    OptionalInteger::Value(7),
                    Bools(vec![false]),
                )
                .get_operation()
            } else {
                RecvDoneOp::new(
                    ctx,
                    output,
                    vec![result],
                    OptionalInteger::Value(7),
                    Bools(vec![false]),
                )
                .get_operation()
            };
            done.insert_at_back(block, ctx);
            done
        };
        let result = operation.deref(ctx).get_result(0);
        ret(ctx, block, result);
        computations.push(computation);
        operations.push(operation);
    }
    let (main, block, args) = computation(ctx, "main", inputs.clone(), output);
    let inputs = Shape::tuple(ctx, inputs).unwrap();
    let state = shape(ctx, ElementType::U32, &[]);
    let context = Shape::tuple(ctx, vec![inputs, output, state]).unwrap();
    let start = AsyncStartOp::new(ctx, context, args, callees(&["first"]));
    attr(
        ctx,
        start.get_operation(),
        "hlo_async_execution_thread",
        StringAttr::new("transfer".into()),
    );
    start.get_operation().insert_at_back(block, ctx);
    let result = start.result(ctx);
    let done = AsyncDoneOp::new(ctx, output, vec![result]);
    attr(
        ctx,
        done.get_operation(),
        "hlo_callees",
        callees(&["second"]),
    );
    done.get_operation().insert_at_back(block, ctx);
    let result = done.result(ctx);
    ret(ctx, block, result);
    computations.push(main);
    (
        module(ctx, &computations, "main"),
        operations[0],
        operations[1],
    )
}

fn frontend(pairs: &[(&str, &str)]) -> FrontendAttributes {
    FrontendAttributes(
        pairs
            .iter()
            .map(|&(key, value)| StringPair {
                key: key.into(),
                value: value.into(),
            })
            .collect(),
    )
}

#[test]
fn wrapped_transfers_compare_frontend_maps_not_record_order_or_empty_presence() {
    for (send, complete) in [(false, false), (false, true), (true, false), (true, true)] {
        let mut ctx = context();
        let (module, first, second) = pair(&mut ctx, send, complete);
        verification::verify(&ctx, module.get_operation()).unwrap();
        attr(&ctx, second, "hlo_frontend_attributes", frontend(&[]));
        verification::verify(&ctx, module.get_operation()).unwrap();
        attr(
            &ctx,
            first,
            "hlo_frontend_attributes",
            frontend(&[("a", "old"), ("z", "last"), ("a", "new")]),
        );
        attr(
            &ctx,
            second,
            "hlo_frontend_attributes",
            frontend(&[("z", "last"), ("a", "new")]),
        );
        verification::verify(&ctx, module.get_operation()).unwrap();
        attr(
            &ctx,
            second,
            "hlo_frontend_attributes",
            frontend(&[("a", "different"), ("z", "last")]),
        );
        assert!(verification::verify(&ctx, module.get_operation()).is_err());
        attr(
            &ctx,
            second,
            "hlo_frontend_attributes",
            frontend(&[("a", "new"), ("z", "last")]),
        );
        verification::verify(&ctx, module.get_operation()).unwrap();
    }
}

#[test]
fn wrapped_transfers_ignore_schedule_pair_metadata_but_preserve_channel_identity() {
    for send in [false, true] {
        let mut ctx = context();
        let (module, first, second) = pair(&mut ctx, send, false);
        attr(
            &ctx,
            first,
            "hlo_source_target_pairs",
            SourceTargetPairs(vec![]),
        );
        attr(
            &ctx,
            second,
            "hlo_source_target_pairs",
            SourceTargetPairs(vec![SourceTarget {
                source: 0,
                target: 1,
            }]),
        );
        verification::verify(&ctx, module.get_operation()).unwrap();
        attr(&ctx, second, "hlo_channel", OptionalInteger::Value(8));
        assert!(verification::verify(&ctx, module.get_operation()).is_err());
        attr(&ctx, second, "hlo_channel", OptionalInteger::Value(7));
        verification::verify(&ctx, module.get_operation()).unwrap();
    }
}

#[test]
fn transfer_frontend_identity_roundtrips_and_typed_failures_do_not_mutate_source() {
    let mut ctx = context();
    let (module, first, second) = pair(&mut ctx, true, false);
    attr(
        &ctx,
        first,
        "hlo_frontend_attributes",
        frontend(&[("key", "shadowed"), ("key", "value\0\"\\")]),
    );
    attr(
        &ctx,
        second,
        "hlo_frontend_attributes",
        frontend(&[("key", "value\0\"\\")]),
    );
    verification::verify(&ctx, module.get_operation()).unwrap();
    let text = module.get_operation().disp(&ctx).to_string();
    let mut fresh = context();
    let parsed = parse_from_str(spaced(Operation::top_level_parser()), &mut fresh, &text).unwrap();
    verification::verify(&fresh, parsed).unwrap();
    attr(
        &ctx,
        second,
        "hlo_frontend_attributes",
        StringAttr::new("not a frontend map".into()),
    );
    let malformed = module.get_operation().disp(&ctx).to_string();
    for _ in 0..2 {
        assert!(verification::verify(&ctx, module.get_operation()).is_err());
        assert_eq!(module.get_operation().disp(&ctx).to_string(), malformed);
    }
    attr(
        &ctx,
        second,
        "hlo_frontend_attributes",
        frontend(&[("key", "value\0\"\\")]),
    );
    verification::verify(&ctx, module.get_operation()).unwrap();
    assert_eq!(module.get_operation().disp(&ctx).to_string(), text);
}

#[test]
fn host_transfers_sharing_a_channel_use_the_same_frontend_map_equivalence() {
    for send in [false, true] {
        let mut ctx = context();
        let (module, first, second) = pair(&mut ctx, send, false);
        for operation in [first, second] {
            attr(&ctx, operation, "hlo_host_transfer", Bools(vec![true]));
        }
        attr(&ctx, second, "hlo_frontend_attributes", frontend(&[]));
        verification::verify(&ctx, module.get_operation()).unwrap();
        attr(
            &ctx,
            first,
            "hlo_frontend_attributes",
            frontend(&[("key", "old"), ("other", "value"), ("key", "current")]),
        );
        attr(
            &ctx,
            second,
            "hlo_frontend_attributes",
            frontend(&[("key", "current"), ("other", "value")]),
        );
        verification::verify(&ctx, module.get_operation()).unwrap();
        attr(
            &ctx,
            second,
            "hlo_frontend_attributes",
            frontend(&[("key", "current"), ("other", "changed")]),
        );
        assert!(verification::verify(&ctx, module.get_operation()).is_err());
    }
}

#[test]
fn host_transfer_identity_normalizes_default_and_numeric_result_accuracy() {
    for send in [false, true] {
        let mut ctx = context();
        let (module, first, second) = pair(&mut ctx, send, false);
        for operation in [first, second] {
            attr(&ctx, operation, "hlo_host_transfer", Bools(vec![true]));
        }
        attr(&ctx, second, "hlo_result_accuracy", ResultAccuracy::Default);
        verification::verify(&ctx, module.get_operation()).unwrap();
        attr(&ctx, first, "hlo_result_accuracy", ResultAccuracy::Default);
        verification::verify(&ctx, module.get_operation()).unwrap();
        let text = module.get_operation().disp(&ctx).to_string();
        let mut fresh = context();
        let parsed =
            parse_from_str(spaced(Operation::top_level_parser()), &mut fresh, &text).unwrap();
        verification::verify(&fresh, parsed).unwrap();
        attr(&ctx, second, "hlo_result_accuracy", ResultAccuracy::Highest);
        let invalid = module.get_operation().disp(&ctx).to_string();
        for _ in 0..2 {
            assert!(verification::verify(&ctx, module.get_operation()).is_err());
            assert_eq!(module.get_operation().disp(&ctx).to_string(), invalid);
        }
        attr(&ctx, first, "hlo_result_accuracy", ResultAccuracy::Highest);
        verification::verify(&ctx, module.get_operation()).unwrap();
        attr(
            &ctx,
            first,
            "hlo_result_accuracy",
            ResultAccuracy::tolerance(0.0, -0.0, 0).unwrap(),
        );
        attr(
            &ctx,
            second,
            "hlo_result_accuracy",
            ResultAccuracy::tolerance(-0.0, 0.0, 0).unwrap(),
        );
        verification::verify(&ctx, module.get_operation()).unwrap();
        for accuracy in [
            ResultAccuracy::tolerance(1.0, 0.0, 0).unwrap(),
            ResultAccuracy::tolerance(0.0, 1.0, 0).unwrap(),
            ResultAccuracy::tolerance(0.0, 0.0, 1).unwrap(),
            ResultAccuracy::Default,
        ] {
            attr(&ctx, second, "hlo_result_accuracy", accuracy);
            let invalid = module.get_operation().disp(&ctx).to_string();
            assert!(verification::verify(&ctx, module.get_operation()).is_err());
            assert_eq!(module.get_operation().disp(&ctx).to_string(), invalid);
        }
        attr(
            &ctx,
            second,
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
fn host_transfer_identity_compares_resolved_backend_bytes() {
    for send in [false, true] {
        let mut ctx = context();
        let (module, first, second) = pair(&mut ctx, send, false);
        for operation in [first, second] {
            attr(&ctx, operation, "hlo_host_transfer", Bools(vec![true]));
        }
        attr(
            &ctx,
            second,
            "hlo_backend_config",
            StringAttr::new(String::new()),
        );
        verification::verify(&ctx, module.get_operation()).unwrap();
        attr(
            &ctx,
            module.get_operation(),
            "hlo_metadata_payloads",
            MetadataPayloads(vec![
                Bytes(vec![0xff, 0, 1]),
                Bytes(vec![0xff, 0, 1]),
                Bytes(vec![0xff, 0, 2]),
            ]),
        );
        for (operation, id, legacy) in [(first, 0, "old-a"), (second, 1, "old-b")] {
            attr(
                &ctx,
                operation,
                "hlo_backend_config",
                StringAttr::new(legacy.into()),
            );
            ir::set_backend_config_payload(&ctx, operation, BackendConfigPayload(Payload::Id(id)))
                .unwrap();
        }
        verification::verify(&ctx, module.get_operation()).unwrap();
        let text = module.get_operation().disp(&ctx).to_string();
        let mut fresh = context();
        let parsed =
            parse_from_str(spaced(Operation::top_level_parser()), &mut fresh, &text).unwrap();
        verification::verify(&fresh, parsed).unwrap();
        for id in [2, -1, 3] {
            ir::set_backend_config_payload(&ctx, second, BackendConfigPayload(Payload::Id(id)))
                .unwrap();
            let invalid = module.get_operation().disp(&ctx).to_string();
            for _ in 0..2 {
                assert!(verification::verify(&ctx, module.get_operation()).is_err());
                assert_eq!(module.get_operation().disp(&ctx).to_string(), invalid);
            }
        }
        ir::set_backend_config_payload(&ctx, second, BackendConfigPayload(Payload::Id(1))).unwrap();
        verification::verify(&ctx, module.get_operation()).unwrap();
        assert_eq!(module.get_operation().disp(&ctx).to_string(), text);
    }
}
