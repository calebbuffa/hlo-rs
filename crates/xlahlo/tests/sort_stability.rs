mod support;

use support::{computation, context, module, ret};
use xlahlo::{
    Diagnostic,
    attributes::{Axes, Callees, ComparisonOrder, Direction},
    ops::{elementwise::CompareOp, numerical::SortOp},
    pliron::{
        builtin::{attributes::StringAttr, ops::ModuleOp},
        common_traits::Verify,
        context::{Context, Ptr},
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
    shape::Shape,
    types::ElementType,
    verification,
};

fn graph(ctx: &mut Context) -> (ModuleOp, SortOp) {
    let (module, container) = module(ctx);
    let scalar = Shape::scalar(ctx, ElementType::F32).unwrap();
    let predicate = Shape::scalar(ctx, ElementType::Pred).unwrap();
    let (_, block, args) = computation(ctx, container, "compare", vec![scalar, scalar], predicate);
    let comparison = CompareOp::new(ctx, predicate, args, Direction::Lt, ComparisonOrder::Total);
    comparison.get_operation().insert_at_back(block, ctx);
    ret(ctx, block, comparison.result(ctx));
    let data = Shape::array(ctx, ElementType::F32, vec![4]).unwrap();
    let (_, block, args) = computation(ctx, container, "main", vec![data], data);
    let sort = SortOp::new(
        ctx,
        data,
        args,
        Axes(vec![0]),
        Callees(vec![Identifier::new("compare")]),
    );
    sort.get_operation().insert_at_back(block, ctx);
    ret(ctx, block, sort.result(ctx));
    (module, sort)
}

#[test]
fn malformed_sort_stability_is_a_typed_verification_error() {
    let mut ctx = context();
    let (module, sort) = graph(&mut ctx);
    verification::verify(&ctx, module.get_operation()).unwrap();
    sort.get_operation().deref_mut(&ctx).attributes.set(
        Identifier::new("hlo_is_stable"),
        StringAttr::new("true".into()),
    );
    let error = sort.verify(&ctx).unwrap_err();
    assert!(matches!(
        error.err.downcast_ref::<Diagnostic>(),
        Some(Diagnostic::InvalidAttribute {
            attribute: "hlo_is_stable",
            ..
        })
    ));
    assert!(verification::verify(&ctx, module.get_operation()).is_err());
    assert!(sort.is_stable(&ctx).is_err());
    sort.set_is_stable(&ctx, false);
    verification::verify(&ctx, module.get_operation()).unwrap();
    assert!(!sort.is_stable(&ctx).unwrap());
}

#[test]
fn sort_stability_defaults_false_and_updates_without_changing_the_sort_contract() {
    let mut ctx = context();
    let (module, sort) = graph(&mut ctx);
    assert!(!sort.is_stable(&ctx).unwrap());
    let operands = sort
        .get_operation()
        .deref(&ctx)
        .operands()
        .collect::<Vec<_>>();
    let dimensions = sort.dimensions(&ctx).unwrap().clone();
    let callees = sort.callees(&ctx).unwrap().clone();
    for stable in [true, false, true] {
        sort.set_is_stable(&ctx, stable);
        assert_eq!(sort.is_stable(&ctx).unwrap(), stable);
        assert_eq!(*sort.dimensions(&ctx).unwrap(), dimensions);
        assert_eq!(*sort.callees(&ctx).unwrap(), callees);
        assert_eq!(
            sort.get_operation()
                .deref(&ctx)
                .operands()
                .collect::<Vec<_>>(),
            operands,
        );
        verification::verify(&ctx, module.get_operation()).unwrap();
    }
}

fn find_sort(ctx: &Context, root: Ptr<Operation>) -> SortOp {
    let mut pending = vec![root];
    while let Some(operation) = pending.pop() {
        if let Some(sort) = Operation::get_op::<SortOp>(operation, ctx) {
            return sort;
        }
        for region in operation.deref(ctx).regions() {
            for block in region.deref(ctx).iter(ctx) {
                pending.extend(block.deref(ctx).iter(ctx));
            }
        }
    }
    panic!("sort missing from test module");
}

#[test]
fn sort_stability_survives_native_assembly_and_cloning() {
    for stable in [None, Some(false), Some(true)] {
        let mut source = context();
        let (module, sort) = graph(&mut source);
        if let Some(stable) = stable {
            sort.set_is_stable(&source, stable);
        }
        let expected = stable.unwrap_or(false);
        let text = module.get_operation().disp(&source).to_string();
        let mut parsed_context = context();
        let parsed = parse_from_str(
            spaced(Operation::top_level_parser()),
            &mut parsed_context,
            &text,
        )
        .unwrap();
        verification::verify(&parsed_context, parsed).unwrap();
        assert_eq!(
            find_sort(&parsed_context, parsed)
                .is_stable(&parsed_context)
                .unwrap(),
            expected,
        );
        let mut mapping = IrMapping::new();
        let cloned = clone_operation(
            module.get_operation(),
            &mut source,
            &mut IRRewriter::<DummyListener>::default(),
            &mut mapping,
        );
        verification::verify(&source, cloned).unwrap();
        let cloned_sort =
            Operation::get_op::<SortOp>(mapping.lookup_op(sort.get_operation()).unwrap(), &source)
                .unwrap();
        assert_eq!(cloned_sort.is_stable(&source).unwrap(), expected);
        cloned_sort.set_is_stable(&source, !expected);
        assert_eq!(sort.is_stable(&source).unwrap(), expected);
    }
}
