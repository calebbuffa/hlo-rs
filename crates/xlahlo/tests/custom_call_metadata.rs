#[path = "control_support/mod.rs"]
mod support;

use support::*;
use xlahlo::{
    Diagnostic,
    attributes::{
        Algorithm, Axes, Bools, CustomCallApiVersion, CustomCallSchedule, OperandPrecision,
        OutputOperandAliases, OutputOperandAliasing, PaddingType, Precision, PrecisionConfig,
        Window,
    },
    ops::control::{AsyncDoneOp, AsyncStartOp, CustomCallOp, ShapeList},
    pliron::{
        builtin::attributes::StringAttr,
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
    shape::{Layout, Shape},
    types::ElementType,
    verification,
};

fn pair(ctx: &mut Context) -> (Ptr<Operation>, CustomCallOp, CustomCallOp) {
    let data = Shape::with_layout(
        ctx,
        shape(ctx, ElementType::F32, &[4]),
        Layout::descending(1),
    )
    .unwrap();
    let mut computations = vec![];
    let mut calls = vec![];
    for name in ["first", "second"] {
        let (comp, block, args) = computation(ctx, name, vec![data], data);
        let call = CustomCallOp::new(ctx, data, args, StringAttr::new("opaque".into()));
        call.get_operation().insert_at_back(block, ctx);
        let value = call.result(ctx);
        ret(ctx, block, value);
        computations.push(comp);
        calls.push(call);
    }
    let (main, block, args) = computation(ctx, "main", vec![data], data);
    let async_context = Shape::tuple(
        ctx,
        vec![
            Shape::tuple(ctx, vec![data]).unwrap(),
            data,
            shape(ctx, ElementType::U32, &[]),
        ],
    )
    .unwrap();
    let start = AsyncStartOp::new(ctx, async_context, args, callees(&["first"]));
    start.get_operation().insert_at_back(block, ctx);
    let value = start.result(ctx);
    let done = AsyncDoneOp::new(ctx, data, vec![value]);
    attr(
        ctx,
        done.get_operation(),
        "hlo_callees",
        callees(&["second"]),
    );
    done.get_operation().insert_at_back(block, ctx);
    let value = done.result(ctx);
    ret(ctx, block, value);
    computations.push(main);
    (
        module(ctx, &computations, "main").get_operation(),
        calls[0],
        calls[1],
    )
}

#[test]
fn custom_call_defaults_and_explicit_enum_values_have_normalized_identity() {
    let mut ctx = context();
    let (root, a, b) = pair(&mut ctx);
    assert_eq!(
        a.api_version(&ctx).unwrap(),
        CustomCallApiVersion::Unspecified
    );
    assert_eq!(a.schedule(&ctx).unwrap(), CustomCallSchedule::None);
    assert_eq!(a.padding_type(&ctx).unwrap(), PaddingType::Invalid);
    b.set_api_version(&ctx, CustomCallApiVersion::Unspecified);
    b.set_schedule(&ctx, CustomCallSchedule::None);
    b.set_padding_type(&ctx, PaddingType::Invalid);
    attr(
        &ctx,
        b.get_operation(),
        "hlo_aliases",
        OutputOperandAliases(vec![]),
    );
    attr(
        &ctx,
        b.get_operation(),
        "hlo_precision",
        PrecisionConfig {
            operand_precision: OperandPrecision(vec![]),
            algorithm: Algorithm::Unset,
        },
    );
    verification::verify(&ctx, root).unwrap();
    for version in [
        CustomCallApiVersion::Original,
        CustomCallApiVersion::StatusReturning,
        CustomCallApiVersion::StatusReturningUnified,
        CustomCallApiVersion::TypedFfi,
        CustomCallApiVersion::Unspecified,
    ] {
        b.set_api_version(&ctx, version);
        b.verify(&ctx).unwrap();
        assert_eq!(
            verification::verify(&ctx, root).is_ok(),
            a.api_version(&ctx).unwrap() == version
        );
        a.set_api_version(&ctx, version);
        verification::verify(&ctx, root).unwrap();
    }
    for schedule in [
        CustomCallSchedule::Latest,
        CustomCallSchedule::Earliest,
        CustomCallSchedule::None,
    ] {
        b.set_schedule(&ctx, schedule);
        b.verify(&ctx).unwrap();
        assert_eq!(
            verification::verify(&ctx, root).is_ok(),
            a.schedule(&ctx).unwrap() == schedule
        );
        a.set_schedule(&ctx, schedule);
        verification::verify(&ctx, root).unwrap();
    }
    for padding in [PaddingType::Same, PaddingType::Valid, PaddingType::Invalid] {
        b.set_padding_type(&ctx, padding);
        b.verify(&ctx).unwrap();
        assert_eq!(
            verification::verify(&ctx, root).is_ok(),
            a.padding_type(&ctx).unwrap() == padding
        );
        a.set_padding_type(&ctx, padding);
        verification::verify(&ctx, root).unwrap();
    }
}

#[test]
fn custom_call_enums_survive_fresh_assembly_and_independent_cloning() {
    let mut ctx = context();
    let (root, a, b) = pair(&mut ctx);
    for call in [a, b] {
        call.set_api_version(&ctx, CustomCallApiVersion::TypedFfi);
        call.set_schedule(&ctx, CustomCallSchedule::Earliest);
        call.set_padding_type(&ctx, PaddingType::Invalid);
    }
    verification::verify(&ctx, root).unwrap();
    let mut fresh = context();
    let parsed = parse_from_str(
        spaced(Operation::top_level_parser()),
        &mut fresh,
        &root.disp(&ctx).to_string(),
    )
    .unwrap();
    verification::verify(&fresh, parsed).unwrap();
    let mut work = vec![parsed];
    let mut count = 0;
    while let Some(op) = work.pop() {
        if let Some(call) = Operation::get_op::<CustomCallOp>(op, &fresh) {
            assert_eq!(
                call.api_version(&fresh).unwrap(),
                CustomCallApiVersion::TypedFfi
            );
            assert_eq!(call.schedule(&fresh).unwrap(), CustomCallSchedule::Earliest);
            assert_eq!(call.padding_type(&fresh).unwrap(), PaddingType::Invalid);
            count += 1;
        }
        for region in op.deref(&fresh).regions() {
            for block in region.deref(&fresh).iter(&fresh) {
                work.extend(block.deref(&fresh).iter(&fresh));
            }
        }
    }
    assert_eq!(count, 2);
    let mut mapping = IrMapping::new();
    let cloned = clone_operation(
        root,
        &mut ctx,
        &mut IRRewriter::<DummyListener>::default(),
        &mut mapping,
    );
    verification::verify(&ctx, cloned).unwrap();
    let cloned_a =
        Operation::get_op::<CustomCallOp>(mapping.lookup_op(a.get_operation()).unwrap(), &ctx)
            .unwrap();
    assert_eq!(
        cloned_a.api_version(&ctx).unwrap(),
        CustomCallApiVersion::TypedFfi
    );
    assert_eq!(
        cloned_a.schedule(&ctx).unwrap(),
        CustomCallSchedule::Earliest
    );
    assert_eq!(cloned_a.padding_type(&ctx).unwrap(), PaddingType::Invalid);
    cloned_a.set_api_version(&ctx, CustomCallApiVersion::Original);
    cloned_a.set_schedule(&ctx, CustomCallSchedule::Latest);
    cloned_a.set_padding_type(&ctx, PaddingType::Same);
    assert_eq!(a.api_version(&ctx).unwrap(), CustomCallApiVersion::TypedFfi);
    assert_eq!(a.schedule(&ctx).unwrap(), CustomCallSchedule::Earliest);
    assert_eq!(a.padding_type(&ctx).unwrap(), PaddingType::Invalid);
    verification::verify(&ctx, root).unwrap();
}

#[test]
fn custom_call_metadata_rejects_wrong_types_and_recovers() {
    let mut ctx = context();
    let (root, a, _) = pair(&mut ctx);
    for key in [
        "hlo_api_version",
        "hlo_schedule",
        "hlo_padding_type",
        "hlo_window",
        "hlo_dimensions",
        "hlo_precision",
        "hlo_feature_groups",
        "hlo_batch_groups",
    ] {
        attr(
            &ctx,
            a.get_operation(),
            key,
            StringAttr::new("invalid".into()),
        );
        assert!(a.verify(&ctx).is_err(), "{key}");
        assert!(verification::verify(&ctx, root).is_err(), "{key}");
        a.get_operation()
            .deref_mut(&ctx)
            .attributes
            .0
            .remove(&Identifier::new(key));
        verification::verify(&ctx, root).unwrap();
    }
    for (key, getter) in [
        ("hlo_api_version", a.api_version(&ctx).map(|_| ())),
        ("hlo_schedule", a.schedule(&ctx).map(|_| ())),
        ("hlo_padding_type", a.padding_type(&ctx).map(|_| ())),
    ] {
        getter.unwrap();
        attr(
            &ctx,
            a.get_operation(),
            key,
            StringAttr::new("invalid".into()),
        );
        let error = match key {
            "hlo_api_version" => a.api_version(&ctx).map(|_| ()),
            "hlo_schedule" => a.schedule(&ctx).map(|_| ()),
            _ => a.padding_type(&ctx).map(|_| ()),
        }
        .unwrap_err();
        assert!(
            matches!(error.err.downcast_ref::<Diagnostic>(), Some(Diagnostic::InvalidAttribute { attribute, .. }) if *attribute == key)
        );
        a.get_operation()
            .deref_mut(&ctx)
            .attributes
            .0
            .remove(&Identifier::new(key));
    }
}

#[test]
fn custom_call_identity_preserves_message_presence_and_precision_but_ignores_unused_layouts() {
    let mut ctx = context();
    let (root, a, b) = pair(&mut ctx);
    attr(&ctx, b.get_operation(), "hlo_window", Window(vec![]));
    b.verify(&ctx).unwrap();
    assert!(verification::verify(&ctx, root).is_err());
    attr(&ctx, a.get_operation(), "hlo_window", Window(vec![]));
    verification::verify(&ctx, root).unwrap();
    let precise = PrecisionConfig {
        operand_precision: OperandPrecision(vec![Precision::High]),
        algorithm: Algorithm::Unset,
    };
    attr(&ctx, b.get_operation(), "hlo_precision", precise.clone());
    b.verify(&ctx).unwrap();
    assert!(verification::verify(&ctx, root).is_err());
    attr(&ctx, a.get_operation(), "hlo_precision", precise);
    verification::verify(&ctx, root).unwrap();
    attr(
        &ctx,
        b.get_operation(),
        "hlo_operand_layouts",
        ShapeList(vec![]),
    );
    verification::verify(&ctx, root).unwrap();
    let aliases = OutputOperandAliases(vec![OutputOperandAliasing {
        output_shape_index: Axes(vec![]),
        operand_index: 0,
        operand_shape_index: Axes(vec![]),
    }]);
    attr(&ctx, b.get_operation(), "hlo_aliases", aliases.clone());
    b.verify(&ctx).unwrap();
    assert!(verification::verify(&ctx, root).is_err());
    attr(&ctx, a.get_operation(), "hlo_aliases", aliases);
    verification::verify(&ctx, root).unwrap();
}

#[test]
fn constrained_custom_call_identity_compares_declared_operand_layouts() {
    let mut ctx = context();
    let (root, a, b) = pair(&mut ctx);
    let data = Shape::with_layout(
        &ctx,
        shape(&ctx, ElementType::F32, &[4]),
        Layout::descending(1),
    )
    .unwrap();
    for call in [a, b] {
        attr(
            &ctx,
            call.get_operation(),
            "hlo_constrain_layout",
            Bools(vec![true]),
        );
        attr(
            &ctx,
            call.get_operation(),
            "hlo_operand_layouts",
            ShapeList(vec![data]),
        );
    }
    verification::verify(&ctx, root).unwrap();
    let other = Shape::with_layout(
        &ctx,
        data,
        Layout {
            memory_space: 5,
            ..Layout::descending(1)
        },
    )
    .unwrap();
    attr(
        &ctx,
        b.get_operation(),
        "hlo_operand_layouts",
        ShapeList(vec![other]),
    );
    b.verify(&ctx).unwrap();
    assert!(verification::verify(&ctx, root).is_err());
    attr(
        &ctx,
        a.get_operation(),
        "hlo_operand_layouts",
        ShapeList(vec![other]),
    );
    verification::verify(&ctx, root).unwrap();
}
