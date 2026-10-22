use crate::control_support::*;
use xlahlo::{
    Diagnostic,
    attributes::Axes,
    inference::movement::infer_reshape_with_dimension,
    ops::movement::ReshapeOp,
    pliron::{
        builtin::attributes::StringAttr,
        common_traits::Verify,
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
        r#type::Typed,
    },
    shape::{self, Dimension},
    types::ElementType,
    verification,
};

#[test]
fn reshape_inferred_axis_controls_dynamic_factor_and_survives_assembly_and_cloning() {
    for inferred in [None, Some(-1), Some(0), Some(1)] {
        let mut ctx = context();
        let input =
            shape::dynamic_array(&ctx, ElementType::F32, vec![Dimension::Bounded(6)]).unwrap();
        let axis = inferred
            .filter(|&axis| axis != -1)
            .map(|axis| usize::try_from(axis).unwrap());
        let output = infer_reshape_with_dimension(&ctx, input, &[2, 3], axis).unwrap();
        assert_eq!(
            shape::dimension(&ctx, output, 1).unwrap(),
            if axis == Some(1) {
                Dimension::Bounded(3)
            } else {
                Dimension::Static(3)
            }
        );
        let (main, block, args) = computation(&mut ctx, "main", vec![input], output);
        let reshape = ReshapeOp::new(&mut ctx, output, args);
        if let Some(inferred) = inferred {
            reshape.set_inferred_dimension(&ctx, inferred);
        }
        assert_eq!(
            reshape.inferred_dimension(&ctx).unwrap(),
            inferred.unwrap_or(-1)
        );
        reshape.get_operation().insert_at_back(block, &ctx);
        let value = reshape.result(&ctx);
        ret(&mut ctx, block, value);
        let module = module(&mut ctx, &[main], "main");
        verification::verify(&ctx, module.get_operation()).unwrap();

        let mut parsed_ctx = context();
        let parsed = parse_from_str(
            spaced(Operation::top_level_parser()),
            &mut parsed_ctx,
            &module.get_operation().disp(&ctx).to_string(),
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
        let parsed_reshape = block
            .deref(&parsed_ctx)
            .iter(&parsed_ctx)
            .find_map(|op| Operation::get_op::<ReshapeOp>(op, &parsed_ctx))
            .unwrap();
        assert_eq!(
            parsed_reshape.inferred_dimension(&parsed_ctx).unwrap(),
            inferred.unwrap_or(-1)
        );
        assert_eq!(
            shape::dimension(
                &parsed_ctx,
                parsed_reshape.result(&parsed_ctx).get_type(&parsed_ctx),
                1
            )
            .unwrap(),
            shape::dimension(&ctx, output, 1).unwrap()
        );

        let mut mapping = IrMapping::new();
        let cloned = clone_operation(
            module.get_operation(),
            &mut ctx,
            &mut IRRewriter::<DummyListener>::default(),
            &mut mapping,
        );
        verification::verify(&ctx, cloned).unwrap();
        let cloned_reshape = Operation::get_op::<ReshapeOp>(
            mapping.lookup_op(reshape.get_operation()).unwrap(),
            &ctx,
        )
        .unwrap();
        assert_eq!(
            cloned_reshape.inferred_dimension(&ctx).unwrap(),
            inferred.unwrap_or(-1)
        );
        cloned_reshape.set_inferred_dimension(&ctx, 1);
        assert_eq!(
            reshape.inferred_dimension(&ctx).unwrap(),
            inferred.unwrap_or(-1)
        );
    }
}

#[test]
fn malformed_reshape_inferred_axes_are_rejected_and_repairable() {
    let mut ctx = context();
    let input = shape(&ctx, ElementType::F32, &[6]);
    let output = shape(&ctx, ElementType::F32, &[2, 3]);
    let (main, block, args) = computation(&mut ctx, "main", vec![input], output);
    let reshape = ReshapeOp::new(&mut ctx, output, args);
    reshape.get_operation().insert_at_back(block, &ctx);
    let value = reshape.result(&ctx);
    ret(&mut ctx, block, value);
    let module = module(&mut ctx, &[main], "main");
    attr(
        &ctx,
        reshape.get_operation(),
        "hlo_dimensions",
        StringAttr::new("1".into()),
    );
    assert!(matches!(
        reshape
            .inferred_dimension(&ctx)
            .unwrap_err()
            .err
            .downcast_ref::<Diagnostic>(),
        Some(Diagnostic::InvalidAttribute {
            attribute: "hlo_dimensions",
            ..
        })
    ));
    assert!(reshape.verify(&ctx).is_err());
    for dimensions in [vec![-2], vec![0, 1], vec![2], vec![i64::MAX]] {
        attr(
            &ctx,
            reshape.get_operation(),
            "hlo_dimensions",
            Axes(dimensions),
        );
        assert!(reshape.verify(&ctx).is_err());
        assert!(verification::verify(&ctx, module.get_operation()).is_err());
        reshape.set_inferred_dimension(&ctx, -1);
        verification::verify(&ctx, module.get_operation()).unwrap();
    }
    attr(
        &ctx,
        reshape.get_operation(),
        "hlo_dimensions",
        Axes(vec![-1]),
    );
    assert_eq!(reshape.inferred_dimension(&ctx).unwrap(), -1);
    verification::verify(&ctx, module.get_operation()).unwrap();
}
