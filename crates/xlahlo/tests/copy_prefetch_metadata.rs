#[path = "control_support/mod.rs"]
mod support;

use support::*;
use xlahlo::{
    Diagnostic,
    attributes::OptionalInteger,
    inference::control::infer_copy_start,
    ir::ExecutionThread,
    ops::control::{AsyncDoneOp, AsyncStartOp, CopyDoneOp, CopyStartOp},
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

fn copy_body(ctx: &mut Context) -> (Ptr<Operation>, CopyStartOp) {
    let input = shape(ctx, ElementType::F32, &[2, 3]);
    let output = infer_copy_start(ctx, input).unwrap();
    let (main, block, args) = computation(ctx, "main", vec![input], input);
    let start = CopyStartOp::new(ctx, output, args);
    start.get_operation().insert_at_back(block, ctx);
    let source = start.result(ctx);
    let done = CopyDoneOp::new(ctx, input, vec![source]);
    done.get_operation().insert_at_back(block, ctx);
    let value = done.result(ctx);
    ret(ctx, block, value);
    let root = module(ctx, &[main], "main");
    (root.get_operation(), start)
}

#[test]
fn copy_prefetch_preserves_optional_int32_values_through_assembly_and_cloning() {
    for index in [
        None,
        Some(0),
        Some(17),
        Some(-1),
        Some(i32::MIN),
        Some(i32::MAX),
    ] {
        let mut ctx = context();
        let (root, start) = copy_body(&mut ctx);
        assert_eq!(start.cross_program_prefetch_index(&ctx).unwrap(), None);
        start.set_cross_program_prefetch_index(&ctx, index);
        assert_eq!(start.cross_program_prefetch_index(&ctx).unwrap(), index);
        verification::verify(&ctx, root).unwrap();

        let mut parsed_ctx = context();
        let parsed = parse_from_str(
            spaced(Operation::top_level_parser()),
            &mut parsed_ctx,
            &root.disp(&ctx).to_string(),
        )
        .unwrap();
        verification::verify(&parsed_ctx, parsed).unwrap();
        let main = parsed
            .deref(&parsed_ctx)
            .get_region(0)
            .deref(&parsed_ctx)
            .get_entry_block()
            .unwrap()
            .deref(&parsed_ctx)
            .iter(&parsed_ctx)
            .next()
            .unwrap();
        let block = main
            .deref(&parsed_ctx)
            .get_region(0)
            .deref(&parsed_ctx)
            .get_entry_block()
            .unwrap();
        let parsed_start = block
            .deref(&parsed_ctx)
            .iter(&parsed_ctx)
            .find_map(|op| Operation::get_op::<CopyStartOp>(op, &parsed_ctx))
            .unwrap();
        assert_eq!(
            parsed_start
                .cross_program_prefetch_index(&parsed_ctx)
                .unwrap(),
            index
        );

        let mut mapping = IrMapping::new();
        let cloned = clone_operation(
            root,
            &mut ctx,
            &mut IRRewriter::<DummyListener>::default(),
            &mut mapping,
        );
        verification::verify(&ctx, cloned).unwrap();
        let cloned_start = Operation::get_op::<CopyStartOp>(
            mapping.lookup_op(start.get_operation()).unwrap(),
            &ctx,
        )
        .unwrap();
        assert_eq!(
            cloned_start.cross_program_prefetch_index(&ctx).unwrap(),
            index
        );
        cloned_start.set_cross_program_prefetch_index(&ctx, Some(42));
        assert_eq!(start.cross_program_prefetch_index(&ctx).unwrap(), index);
        start.set_cross_program_prefetch_index(&ctx, None);
        assert_eq!(start.cross_program_prefetch_index(&ctx).unwrap(), None);
        verification::verify(&ctx, root).unwrap();
    }
}

#[test]
fn malformed_copy_prefetch_metadata_fails_with_typed_errors_and_recovers() {
    let mut ctx = context();
    let (root, start) = copy_body(&mut ctx);
    attr(
        &ctx,
        start.get_operation(),
        "hlo_cross_program_prefetch_index",
        StringAttr::new("0".into()),
    );
    assert!(start.verify(&ctx).is_err());
    for raw in [
        i64::MIN,
        i64::from(i32::MIN) - 1,
        i64::from(i32::MAX) + 1,
        i64::MAX,
    ] {
        attr(
            &ctx,
            start.get_operation(),
            "hlo_cross_program_prefetch_index",
            OptionalInteger::Value(raw),
        );
        let error = start.cross_program_prefetch_index(&ctx).unwrap_err();
        assert!(matches!(
            error.err.downcast_ref::<Diagnostic>(),
            Some(Diagnostic::InvalidAttribute {
                attribute: "hlo_cross_program_prefetch_index",
                ..
            })
        ));
        assert!(start.verify(&ctx).is_err());
        assert!(verification::verify(&ctx, root).is_err());
        start.set_cross_program_prefetch_index(&ctx, Some(0));
        verification::verify(&ctx, root).unwrap();
    }
}

#[test]
fn wrapped_copy_identity_normalizes_absence_but_distinguishes_zero_and_signed_indices() {
    let mut ctx = context();
    let data = shape(&ctx, ElementType::F32, &[2]);
    let copy_context = infer_copy_start(&ctx, data).unwrap();
    let mut computations = vec![];
    let mut starts = vec![];
    for name in ["first", "second"] {
        let (comp, block, args) = computation(&mut ctx, name, vec![data], data);
        attr(
            &ctx,
            comp.get_operation(),
            "hlo_execution_thread",
            ExecutionThread("copy_worker".into()),
        );
        let start = CopyStartOp::new(&mut ctx, copy_context, args);
        start.get_operation().insert_at_back(block, &ctx);
        let source = start.result(&ctx);
        let done = CopyDoneOp::new(&mut ctx, data, vec![source]);
        done.get_operation().insert_at_back(block, &ctx);
        let value = done.result(&ctx);
        ret(&mut ctx, block, value);
        computations.push(comp);
        starts.push(start);
    }
    let (main, block, args) = computation(&mut ctx, "main", vec![data], data);
    let state = shape(&ctx, ElementType::U32, &[]);
    let async_context = Shape::tuple(
        &ctx,
        vec![Shape::tuple(&ctx, vec![data]).unwrap(), data, state],
    )
    .unwrap();
    let start = AsyncStartOp::new(&mut ctx, async_context, args, callees(&["first"]));
    attr(
        &ctx,
        start.get_operation(),
        "hlo_async_execution_thread",
        StringAttr::new("copy_worker".into()),
    );
    start.get_operation().insert_at_back(block, &ctx);
    let value = start.result(&ctx);
    let done = AsyncDoneOp::new(&mut ctx, data, vec![value]);
    attr(
        &ctx,
        done.get_operation(),
        "hlo_callees",
        callees(&["second"]),
    );
    done.get_operation().insert_at_back(block, &ctx);
    let value = done.result(&ctx);
    ret(&mut ctx, block, value);
    computations.push(main);
    let root = module(&mut ctx, &computations, "main");
    starts[1].set_cross_program_prefetch_index(&ctx, None);
    verification::verify(&ctx, root.get_operation()).unwrap();
    for (first, second, equal) in [
        (None, Some(0), false),
        (Some(-1), None, false),
        (Some(0), Some(0), true),
        (Some(1), Some(2), false),
        (Some(i32::MIN), Some(i32::MIN), true),
        (Some(i32::MAX), Some(i32::MAX), true),
    ] {
        starts[0].set_cross_program_prefetch_index(&ctx, first);
        starts[1].set_cross_program_prefetch_index(&ctx, second);
        starts[0].verify(&ctx).unwrap();
        starts[1].verify(&ctx).unwrap();
        assert_eq!(
            verification::verify(&ctx, root.get_operation()).is_ok(),
            equal
        );
    }
}
