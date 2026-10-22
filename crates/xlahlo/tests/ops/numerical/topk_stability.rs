use crate::support::{computation, context, module, ret};
use xlahlo::{
    Diagnostic,
    attributes::SortOptions,
    inference::numerical::infer_top_k,
    ops::numerical::{NumericalInteger, TopKOp},
    pliron::{
        builtin::attributes::StringAttr,
        common_traits::Verify,
        identifier::Identifier,
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
    shape,
    types::ElementType,
    verification,
};

#[test]
fn topk_stability_is_independent_of_k_and_largest_and_survives_assembly_and_cloning() {
    for largest in [false, true] {
        for stable in [None, Some(false), Some(true)] {
            let mut ctx = context();
            let data = shape::array(&ctx, ElementType::F32, vec![2, 5]).unwrap();
            let output = infer_top_k(&ctx, data, 3).unwrap();
            let (module, container) = module(&mut ctx);
            let (_, block, args) = computation(&mut ctx, container, "main", vec![data], output);
            let topk = TopKOp::new(
                &mut ctx,
                output,
                args,
                NumericalInteger(3),
                SortOptions {
                    descending: largest,
                },
            );
            if let Some(stable) = stable {
                topk.set_is_stable(&ctx, stable);
            }
            assert_eq!(topk.is_stable(&ctx).unwrap(), stable.unwrap_or(false));
            topk.get_operation().insert_at_back(block, &ctx);
            let value = topk.result(&ctx);
            ret(&mut ctx, block, value);
            verification::verify(&ctx, module.get_operation()).unwrap();
            let mut parsed_ctx = context();
            let parsed = parse_from_str(
                spaced(Operation::top_level_parser()),
                &mut parsed_ctx,
                &module.get_operation().disp(&ctx).to_string(),
            )
            .unwrap();
            verification::verify(&parsed_ctx, parsed).unwrap();
            let container = parsed.deref(&parsed_ctx).get_region(0);
            let main = container
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
            let parsed_topk = block
                .deref(&parsed_ctx)
                .iter(&parsed_ctx)
                .find_map(|op| Operation::get_op::<TopKOp>(op, &parsed_ctx))
                .unwrap();
            assert_eq!(
                parsed_topk.is_stable(&parsed_ctx).unwrap(),
                stable.unwrap_or(false)
            );
            assert_eq!(
                parsed_topk.options(&parsed_ctx).unwrap().descending,
                largest
            );
            assert_eq!(parsed_topk.k(&parsed_ctx).unwrap().0, 3);
            let mut mapping = IrMapping::new();
            let cloned = clone_operation(
                module.get_operation(),
                &mut ctx,
                &mut IRRewriter::<DummyListener>::default(),
                &mut mapping,
            );
            verification::verify(&ctx, cloned).unwrap();
            let cloned_topk =
                Operation::get_op::<TopKOp>(mapping.lookup_op(topk.get_operation()).unwrap(), &ctx)
                    .unwrap();
            assert_eq!(
                cloned_topk.is_stable(&ctx).unwrap(),
                stable.unwrap_or(false)
            );
            cloned_topk.set_is_stable(&ctx, !stable.unwrap_or(false));
            assert_eq!(topk.is_stable(&ctx).unwrap(), stable.unwrap_or(false));
            assert_eq!(cloned_topk.options(&ctx).unwrap().descending, largest);
            assert_eq!(cloned_topk.k(&ctx).unwrap().0, 3);
        }
    }
}

#[test]
fn malformed_topk_stability_fails_typed_verification_and_can_be_repaired() {
    let mut ctx = context();
    let data = shape::array(&ctx, ElementType::F32, vec![5]).unwrap();
    let output = infer_top_k(&ctx, data, 2).unwrap();
    let (module, container) = module(&mut ctx);
    let (_, block, args) = computation(&mut ctx, container, "main", vec![data], output);
    let topk = TopKOp::new(
        &mut ctx,
        output,
        args,
        NumericalInteger(2),
        SortOptions { descending: false },
    );
    topk.get_operation().insert_at_back(block, &ctx);
    let value = topk.result(&ctx);
    ret(&mut ctx, block, value);
    topk.get_operation().deref_mut(&ctx).attributes.set(
        Identifier::new("hlo_is_stable"),
        StringAttr::new("false".into()),
    );
    assert!(matches!(
        topk.verify(&ctx)
            .unwrap_err()
            .err
            .downcast_ref::<Diagnostic>(),
        Some(Diagnostic::InvalidAttribute {
            attribute: "hlo_is_stable",
            ..
        }),
    ));
    assert!(topk.is_stable(&ctx).is_err());
    assert!(verification::verify(&ctx, module.get_operation()).is_err());
    for stable in [true, false] {
        topk.set_is_stable(&ctx, stable);
        assert_eq!(topk.is_stable(&ctx).unwrap(), stable);
        verification::verify(&ctx, module.get_operation()).unwrap();
    }
}
