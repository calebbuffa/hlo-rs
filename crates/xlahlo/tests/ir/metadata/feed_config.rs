use crate::control_support::*;
use xlahlo::{
    Diagnostic,
    attributes::Index,
    ops::{
        ComputationOp,
        control::{AsyncDoneOp, AsyncStartOp, InfeedOp, OutfeedOp},
    },
    pliron::{
        builtin::attributes::TypeAttr,
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
        r#type::TypeHandle,
    },
    shape,
    types::ElementType,
    verification,
};

fn feed_signature(ctx: &Context, infeed: bool) -> (Vec<TypeHandle>, TypeHandle) {
    let data = shape(ctx, ElementType::F32, &[2, 3]);
    let token = shape::token(ctx);
    if infeed {
        (vec![token], shape::tuple(ctx, vec![data, token]).unwrap())
    } else {
        (vec![data, token], token)
    }
}

fn feed_body(
    ctx: &mut Context,
    name: &'static str,
    infeed: bool,
) -> (ComputationOp, Ptr<Operation>) {
    let (parameters, result) = feed_signature(ctx, infeed);
    let (comp, block, args) = computation(ctx, name, parameters, result);
    let (feed, value) = if infeed {
        let feed = InfeedOp::new(ctx, result, args);
        (feed.get_operation(), feed.result(ctx))
    } else {
        let data = shape(ctx, ElementType::F32, &[2, 3]);
        let feed = OutfeedOp::new(ctx, result, args);
        attr(
            ctx,
            feed.get_operation(),
            "hlo_outfeed_shape",
            TypeAttr::new(data),
        );
        (feed.get_operation(), feed.result(ctx))
    };
    feed.insert_at_back(block, ctx);
    ret(ctx, block, value);
    (comp, feed)
}

fn config(ctx: &Context, feed: Ptr<Operation>) -> xlahlo::pliron::result::Result<String> {
    if let Some(feed) = Operation::get_op::<InfeedOp>(feed, ctx) {
        feed.infeed_config(ctx)
    } else {
        Operation::get_op::<OutfeedOp>(feed, ctx)
            .unwrap()
            .outfeed_config(ctx)
    }
}

fn set_config(ctx: &Context, feed: Ptr<Operation>, config: &str) {
    if let Some(feed) = Operation::get_op::<InfeedOp>(feed, ctx) {
        feed.set_infeed_config(ctx, config.into());
    } else {
        Operation::get_op::<OutfeedOp>(feed, ctx)
            .unwrap()
            .set_outfeed_config(ctx, config.into());
    }
}

#[test]
fn opaque_feed_configs_preserve_defaults_assembly_and_independent_clones() {
    for infeed in [true, false] {
        for expected in ["", "host=0", "opaque \"quoted\"\\value\n\t\0\u{3bb}"] {
            let mut ctx = context();
            let (comp, feed) = feed_body(&mut ctx, "main", infeed);
            let root = module(&mut ctx, &[comp], "main").get_operation();
            assert_eq!(config(&ctx, feed).unwrap(), "");
            set_config(&ctx, feed, expected);
            verification::verify(&ctx, root).unwrap();
            assert_eq!(config(&ctx, feed).unwrap(), expected);

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
            let parsed_feed = block
                .deref(&parsed_ctx)
                .iter(&parsed_ctx)
                .find(|&op| {
                    Operation::get_op::<InfeedOp>(op, &parsed_ctx).is_some()
                        || Operation::get_op::<OutfeedOp>(op, &parsed_ctx).is_some()
                })
                .unwrap();
            assert_eq!(config(&parsed_ctx, parsed_feed).unwrap(), expected);

            let mut mapping = IrMapping::new();
            let cloned = clone_operation(
                root,
                &mut ctx,
                &mut IRRewriter::<DummyListener>::default(),
                &mut mapping,
            );
            verification::verify(&ctx, cloned).unwrap();
            let cloned_feed = mapping.lookup_op(feed).unwrap();
            assert_eq!(config(&ctx, cloned_feed).unwrap(), expected);
            set_config(&ctx, cloned_feed, "changed");
            assert_eq!(config(&ctx, feed).unwrap(), expected);
            set_config(&ctx, feed, "");
            assert_eq!(config(&ctx, feed).unwrap(), "");
            verification::verify(&ctx, root).unwrap();
        }
    }
}

#[test]
fn feed_config_wrong_types_report_typed_errors_and_recover() {
    for infeed in [true, false] {
        let mut ctx = context();
        let (comp, feed) = feed_body(&mut ctx, "main", infeed);
        let root = module(&mut ctx, &[comp], "main").get_operation();
        let key = if infeed {
            "hlo_infeed_config"
        } else {
            "hlo_outfeed_config"
        };
        attr(&ctx, feed, key, Index(0));
        let error = config(&ctx, feed).unwrap_err();
        assert!(matches!(
            error.err.downcast_ref::<Diagnostic>(),
            Some(Diagnostic::InvalidAttribute { attribute, .. }) if *attribute == key
        ));
        assert!(Operation::get_op_dyn(feed, &ctx).verify(&ctx).is_err());
        assert!(verification::verify(&ctx, root).is_err());
        set_config(&ctx, feed, "repaired");
        verification::verify(&ctx, root).unwrap();
    }
}

#[test]
fn wrapped_feed_identity_normalizes_empty_defaults_and_compares_opaque_configs() {
    for infeed in [true, false] {
        let mut ctx = context();
        let (first, a) = feed_body(&mut ctx, "first", infeed);
        let (second, b) = feed_body(&mut ctx, "second", infeed);
        let (parameters, result) = feed_signature(&ctx, infeed);
        let state = shape(&ctx, ElementType::U32, &[]);
        let async_context = shape::tuple(
            &ctx,
            vec![
                shape::tuple(&ctx, parameters.clone()).unwrap(),
                result,
                state,
            ],
        )
        .unwrap();
        let (main, block, args) = computation(&mut ctx, "main", parameters, result);
        let start = AsyncStartOp::new(&mut ctx, async_context, args, callees(&["first"]));
        start.get_operation().insert_at_back(block, &ctx);
        let value = start.result(&ctx);
        let done = AsyncDoneOp::new(&mut ctx, result, vec![value]);
        attr(
            &ctx,
            done.get_operation(),
            "hlo_callees",
            callees(&["second"]),
        );
        done.get_operation().insert_at_back(block, &ctx);
        let value = done.result(&ctx);
        ret(&mut ctx, block, value);
        let root = module(&mut ctx, &[first, second, main], "main").get_operation();

        verification::verify(&ctx, root).unwrap();
        set_config(&ctx, b, "");
        verification::verify(&ctx, root).unwrap();
        for (left, right, equal) in [
            ("", "host=0", false),
            ("host=0", "host=0", true),
            ("host=0", "host=1", false),
            ("host=0", "host=0 ", false),
            ("opaque\n\0", "opaque\n\0", true),
        ] {
            set_config(&ctx, a, left);
            set_config(&ctx, b, right);
            Operation::get_op_dyn(a, &ctx).verify(&ctx).unwrap();
            Operation::get_op_dyn(b, &ctx).verify(&ctx).unwrap();
            assert_eq!(verification::verify(&ctx, root).is_ok(), equal);
        }
    }
}
