#[path = "control_support/mod.rs"]
mod support;

use support::*;
use xlahlo::{
    Diagnostic,
    inference::control::infer_rng_get_and_update_state,
    ops::{
        ComputationOp,
        control::{AsyncDoneOp, AsyncStartOp, RngGetAndUpdateStateOp},
    },
    pliron::{
        builtin::attributes::StringAttr,
        common_traits::Verify,
        context::{Context, Ptr},
        irbuild::{
            cloning::{IrMapping, clone_operation},
            listener::DummyListener,
            rewriter::IRRewriter,
        },
        irfmt::parsers::spaced,
        linked_list::ContainsLinkedList,
        op::Op,
        operation::Operation,
        parsable::parse_from_str,
        printable::Printable,
    },
    shape::Shape,
    types::ElementType,
    verification,
};

fn state_body(ctx: &mut Context, name: &'static str) -> (ComputationOp, RngGetAndUpdateStateOp) {
    let state = infer_rng_get_and_update_state(ctx).unwrap();
    let (comp, block, _) = computation(ctx, name, vec![], state);
    let rng = RngGetAndUpdateStateOp::new(ctx, state, vec![]);
    rng.get_operation().insert_at_back(block, ctx);
    let value = rng.result(ctx);
    ret(ctx, block, value);
    (comp, rng)
}

#[test]
fn rng_state_delta_preserves_signed_int64_metadata_in_assembly_and_clones() {
    for delta in [i64::MIN, -1, 0, 1, i64::MAX] {
        let mut ctx = context();
        let (comp, rng) = state_body(&mut ctx, "main");
        let root = module(&mut ctx, &[comp], "main").get_operation();
        assert_eq!(rng.delta(&ctx).unwrap(), 0);
        rng.set_delta(&ctx, delta);
        assert_eq!(rng.delta(&ctx).unwrap(), delta);
        verification::verify(&ctx, root).unwrap();

        let mut fresh = context();
        let parsed = parse_from_str(
            spaced(Operation::top_level_parser()),
            &mut fresh,
            &root.disp(&ctx).to_string(),
        )
        .unwrap();
        verification::verify(&fresh, parsed).unwrap();
        let comp = parsed
            .deref(&fresh)
            .get_region(0)
            .deref(&fresh)
            .get_entry_block()
            .unwrap()
            .deref(&fresh)
            .iter(&fresh)
            .next()
            .unwrap();
        let parsed_rng = comp
            .deref(&fresh)
            .get_region(0)
            .deref(&fresh)
            .get_entry_block()
            .unwrap()
            .deref(&fresh)
            .iter(&fresh)
            .find_map(|op| Operation::get_op::<RngGetAndUpdateStateOp>(op, &fresh))
            .unwrap();
        assert_eq!(parsed_rng.delta(&fresh).unwrap(), delta);

        let mut mapping = IrMapping::new();
        let clone = clone_operation(
            root,
            &mut ctx,
            &mut IRRewriter::<DummyListener>::default(),
            &mut mapping,
        );
        verification::verify(&ctx, clone).unwrap();
        let cloned_rng = Operation::get_op::<RngGetAndUpdateStateOp>(
            mapping.lookup_op(rng.get_operation()).unwrap(),
            &ctx,
        )
        .unwrap();
        assert_eq!(cloned_rng.delta(&ctx).unwrap(), delta);
        cloned_rng.set_delta(&ctx, 42);
        assert_eq!(rng.delta(&ctx).unwrap(), delta);
        rng.set_delta(&ctx, 0);
        verification::verify(&ctx, root).unwrap();
    }
}

#[test]
fn malformed_rng_state_delta_reports_typed_errors_and_recovers() {
    let mut ctx = context();
    let (comp, rng) = state_body(&mut ctx, "main");
    let root = module(&mut ctx, &[comp], "main").get_operation();
    attr(
        &ctx,
        rng.get_operation(),
        "hlo_delta",
        StringAttr::new("1".into()),
    );
    let error = rng.delta(&ctx).unwrap_err();
    assert!(matches!(
        error.err.downcast_ref::<Diagnostic>(),
        Some(Diagnostic::InvalidAttribute {
            attribute: "hlo_delta",
            ..
        })
    ));
    assert!(rng.verify(&ctx).is_err());
    assert!(verification::verify(&ctx, root).is_err());
    rng.set_delta(&ctx, i64::MIN);
    verification::verify(&ctx, root).unwrap();
}

#[test]
fn wrapped_rng_state_identity_normalizes_zero_and_compares_signed_deltas() {
    let mut ctx = context();
    let (first, a) = state_body(&mut ctx, "first");
    let (second, b) = state_body(&mut ctx, "second");
    let state = infer_rng_get_and_update_state(&ctx).unwrap();
    let (main, block, _) = computation(&mut ctx, "main", vec![], state);
    let async_context = Shape::tuple(
        &ctx,
        vec![
            Shape::tuple(&ctx, vec![]).unwrap(),
            state,
            shape(&ctx, ElementType::U32, &[]),
        ],
    )
    .unwrap();
    let start = AsyncStartOp::new(&mut ctx, async_context, vec![], callees(&["first"]));
    start.get_operation().insert_at_back(block, &ctx);
    let value = start.result(&ctx);
    let done = AsyncDoneOp::new(&mut ctx, state, vec![value]);
    attr(
        &ctx,
        done.get_operation(),
        "hlo_callees",
        callees(&["second"]),
    );
    done.get_operation().insert_at_back(block, &ctx);
    let value = done.result(&ctx);
    ret(&mut ctx, block, value);
    let root: Ptr<Operation> = module(&mut ctx, &[first, second, main], "main").get_operation();
    verification::verify(&ctx, root).unwrap();
    b.set_delta(&ctx, 0);
    verification::verify(&ctx, root).unwrap();
    for (left, right, equal) in [
        (0, 1, false),
        (-1, 0, false),
        (-1, -1, true),
        (i64::MIN, i64::MIN, true),
        (i64::MIN, i64::MAX, false),
        (i64::MAX, i64::MAX, true),
    ] {
        a.set_delta(&ctx, left);
        b.set_delta(&ctx, right);
        a.verify(&ctx).unwrap();
        b.verify(&ctx).unwrap();
        assert_eq!(verification::verify(&ctx, root).is_ok(), equal);
    }
}
