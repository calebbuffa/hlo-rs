use super::*;
use crate::{
    ops::{
        control::TupleOp,
        movement::{BitcastOp, CopyOp},
    },
    shape::{Dimension, Layout},
};

struct BitcastFixture {
    module: Ptr<Operation>,
    root: Value,
    parameter: Value,
}

fn make_fixture(ctx: &mut Context, input: TypeHandle, target: TypeHandle) -> BitcastFixture {
    let module = module(ctx);
    let tuple = shape::tuple(ctx, vec![target, target]).unwrap();
    let (_, block, parameters) = computation(ctx, module, "main", &[input], tuple);
    let bitcast = BitcastOp::new(ctx, target, vec![parameters[0]]);
    let bitcast = append(ctx, block, bitcast.get_operation(), "bitcast");
    let first = CopyOp::new(ctx, target, vec![bitcast]).get_operation();
    let first = append(ctx, block, first, "first_copy");
    let second = CopyOp::new(ctx, target, vec![bitcast]).get_operation();
    let second = append(ctx, block, second, "second_copy");
    let root = TupleOp::new(ctx, tuple, vec![first, second]).get_operation();
    let root = append(ctx, block, root, "results");
    finish(ctx, block, root);
    BitcastFixture {
        module,
        root,
        parameter: parameters[0],
    }
}

fn evaluate(
    ctx: &mut Context,
    fixture: &BitcastFixture,
    value: EvaluatedValue,
) -> Result<EvaluatedValue> {
    super::evaluate(
        ctx,
        fixture.module,
        fixture.root,
        &[(fixture.parameter, value)],
        DependencyMode::Recursive,
    )
}

fn context() -> Context {
    let mut ctx = Context::new();
    crate::register(&mut ctx);
    ctx
}

#[test]
fn bitcast_copies_physical_bytes_and_preserves_declared_layout() {
    let mut ctx = context();
    let input = shape::array(&ctx, ElementType::F32, vec![2, 2]).unwrap();
    let input = shape::with_layout(
        &ctx,
        input,
        Layout {
            minor_to_major: vec![0, 1],
            ..Layout::descending(2)
        },
    )
    .unwrap();
    let target = shape::array(&ctx, ElementType::U8, vec![4, 4]).unwrap();
    let output_layout = Layout {
        minor_to_major: vec![0, 1],
        ..Layout::descending(2)
    };
    let target = shape::with_layout(&ctx, target, output_layout.clone()).unwrap();
    let fixture = make_fixture(&mut ctx, input, target);
    let bytes = (0u8..16).collect::<Vec<_>>();
    let value = EvaluatedValue::known(
        &ctx,
        Literal::from_storage_bytes(&ctx, input, bytes.clone()).unwrap(),
    )
    .unwrap();
    let output = evaluate(&mut ctx, &fixture, value)
        .unwrap()
        .into_literal(&ctx)
        .unwrap();
    let first = output.sub_literal(&ctx, &[0]).unwrap();
    let second = output.sub_literal(&ctx, &[1]).unwrap();
    assert_eq!(first.storage, bytes);
    assert_eq!(second.storage, first.storage);
    assert_eq!(
        shape::layout(&ctx, first.shape)
            .unwrap()
            .unwrap()
            .minor_to_major,
        output_layout.minor_to_major
    );
}

#[test]
fn bitcast_allows_smaller_output_and_rejects_larger_output() {
    let mut ctx = context();
    let input = shape::array(&ctx, ElementType::F32, vec![2]).unwrap();
    let smaller = shape::array(&ctx, ElementType::U8, vec![4]).unwrap();
    let smaller = shape::with_default_layout(&ctx, smaller).unwrap();
    let fixture = make_fixture(&mut ctx, input, smaller);
    let bytes = vec![0x7f, 0xa1, 0x00, 0x80, 0x12, 0x34, 0x56, 0x78];
    let value = EvaluatedValue::known(
        &ctx,
        Literal::from_storage_bytes(&ctx, input, bytes.clone()).unwrap(),
    )
    .unwrap();
    let output = evaluate(&mut ctx, &fixture, value)
        .unwrap()
        .into_literal(&ctx)
        .unwrap();
    assert_eq!(output.sub_literal(&ctx, &[0]).unwrap().storage, &bytes[..4]);

    let larger = shape::array(&ctx, ElementType::U8, vec![12]).unwrap();
    let larger = shape::with_default_layout(&ctx, larger).unwrap();
    let fixture = make_fixture(&mut ctx, input, larger);
    let value = EvaluatedValue::known(
        &ctx,
        Literal::from_storage_bytes(&ctx, input, bytes).unwrap(),
    )
    .unwrap();
    assert!(evaluate(&mut ctx, &fixture, value).is_err());
}

#[test]
fn bitcast_scalar_without_layout_and_zero_bound_arrays_are_supported() {
    let mut ctx = context();
    let input = shape::scalar(&ctx, ElementType::F64).unwrap();
    let target = shape::scalar(&ctx, ElementType::C64).unwrap();
    let fixture = make_fixture(&mut ctx, input, target);
    let bytes = 0x7ff8_1234_dead_beefu64.to_le_bytes().to_vec();
    let value = EvaluatedValue::known(
        &ctx,
        Literal::from_storage_bytes(&ctx, input, bytes.clone()).unwrap(),
    )
    .unwrap();
    let output = evaluate(&mut ctx, &fixture, value)
        .unwrap()
        .into_literal(&ctx)
        .unwrap();
    let output = output.sub_literal(&ctx, &[0]).unwrap();
    assert_eq!(output.storage, bytes);
    assert!(shape::layout(&ctx, output.shape).unwrap().is_some());

    let input = shape::array(&ctx, ElementType::F64, vec![2]).unwrap();
    let target = shape::scalar(&ctx, ElementType::C128).unwrap();
    let fixture = make_fixture(&mut ctx, input, target);
    let bytes = (0u8..16).collect::<Vec<_>>();
    let value = EvaluatedValue::known(
        &ctx,
        Literal::from_storage_bytes(&ctx, input, bytes.clone()).unwrap(),
    )
    .unwrap();
    let output = evaluate(&mut ctx, &fixture, value)
        .unwrap()
        .into_literal(&ctx)
        .unwrap();
    assert_eq!(output.sub_literal(&ctx, &[0]).unwrap().storage, bytes);

    let empty_input = shape::array(&ctx, ElementType::U8, vec![0]).unwrap();
    let empty_target = shape::array(&ctx, ElementType::F32, vec![0]).unwrap();
    let empty_target = shape::with_default_layout(&ctx, empty_target).unwrap();
    let fixture = make_fixture(&mut ctx, empty_input, empty_target);
    let value = EvaluatedValue::known(
        &ctx,
        Literal::from_storage_bytes(&ctx, empty_input, Vec::new()).unwrap(),
    )
    .unwrap();
    let output = evaluate(&mut ctx, &fixture, value)
        .unwrap()
        .into_literal(&ctx)
        .unwrap();
    assert!(output.sub_literal(&ctx, &[0]).unwrap().storage.is_empty());
}

#[test]
fn bitcast_static_output_copies_full_bounded_allocation_but_dynamic_output_is_unsupported() {
    let mut ctx = context();
    let input = shape::dynamic_array(&ctx, ElementType::U8, vec![Dimension::Bounded(4)]).unwrap();
    let target = shape::array(&ctx, ElementType::U8, vec![4]).unwrap();
    let target = shape::with_default_layout(&ctx, target).unwrap();
    let fixture = make_fixture(&mut ctx, input, target);
    let bytes = vec![1, 2, 3, 4];
    let value = EvaluatedValue::known(
        &ctx,
        Literal::from_dynamic_storage_bytes(&ctx, input, bytes.clone(), vec![2]).unwrap(),
    )
    .unwrap();
    let output = evaluate(&mut ctx, &fixture, value)
        .unwrap()
        .into_literal(&ctx)
        .unwrap();
    let first = output.sub_literal(&ctx, &[0]).unwrap();
    assert_eq!(first.storage, bytes);
    assert!(first.dynamic_sizes.is_empty());

    let dynamic_target =
        shape::dynamic_array(&ctx, ElementType::U8, vec![Dimension::Bounded(4)]).unwrap();
    let dynamic_target = shape::with_default_layout(&ctx, dynamic_target).unwrap();
    let fixture = make_fixture(&mut ctx, input, dynamic_target);
    let value = EvaluatedValue::known(
        &ctx,
        Literal::from_dynamic_storage_bytes(&ctx, input, vec![1, 2, 3, 4], vec![2]).unwrap(),
    )
    .unwrap();
    let error = evaluate(&mut ctx, &fixture, value).unwrap_err();
    assert!(matches!(
        error.err.downcast_ref::<EvaluationError>(),
        Some(EvaluationError::UnsupportedValue {
            opcode: Opcode::Bitcast,
            ..
        })
    ));
}

#[test]
fn bitcast_preserves_subbyte_storage_as_one_byte_per_element() {
    let mut ctx = context();
    let input = shape::array(&ctx, ElementType::S4, vec![4]).unwrap();
    let target = shape::array(&ctx, ElementType::U8, vec![4]).unwrap();
    let target = shape::with_default_layout(&ctx, target).unwrap();
    let fixture = make_fixture(&mut ctx, input, target);
    let bytes = vec![0xa1, 0xb2, 0xc3, 0xd4];
    let value = EvaluatedValue::known(
        &ctx,
        Literal::from_storage_bytes(&ctx, input, bytes.clone()).unwrap(),
    )
    .unwrap();
    let output = evaluate(&mut ctx, &fixture, value)
        .unwrap()
        .into_literal(&ctx)
        .unwrap();
    assert_eq!(output.sub_literal(&ctx, &[0]).unwrap().storage, bytes);
}

#[test]
fn bitcast_rejects_layoutless_non_scalar_and_malformed_operand_count() {
    let mut ctx = context();
    let input = shape::array(&ctx, ElementType::F32, vec![2]).unwrap();
    let target =
        shape::without_layout(&ctx, shape::array(&ctx, ElementType::U8, vec![8]).unwrap()).unwrap();
    let fixture = make_fixture(&mut ctx, input, target);
    let value = EvaluatedValue::known(
        &ctx,
        Literal::from_bits(&ctx, input, &[0x3f80_0000, 0x4000_0000]).unwrap(),
    )
    .unwrap();
    let error = evaluate(&mut ctx, &fixture, value).unwrap_err();
    assert!(matches!(
        error.err.downcast_ref::<EvaluationError>(),
        Some(EvaluationError::UnsupportedValue {
            opcode: Opcode::Bitcast,
            ..
        })
    ));

    let module = module(&mut ctx);
    let tuple = shape::tuple(&ctx, vec![target, target]).unwrap();
    let (_, block, _) = computation(&mut ctx, module, "main", &[], tuple);
    let operation = BitcastOp::new(&mut ctx, target, Vec::new());
    let root = append(&ctx, block, operation.get_operation(), "malformed_bitcast");
    finish(&mut ctx, block, root);
    assert!(
        super::evaluate(&ctx, module, root, &[], DependencyMode::Recursive).is_err(),
        "malformed Bitcast operand count must be rejected"
    );
}

#[test]
fn bitcast_unknown_input_remains_unknown_for_both_tuple_consumers() {
    let mut ctx = context();
    let input = shape::array(&ctx, ElementType::F32, vec![2]).unwrap();
    let target = shape::array(&ctx, ElementType::U8, vec![8]).unwrap();
    let target = shape::with_default_layout(&ctx, target).unwrap();
    let fixture = make_fixture(&mut ctx, input, target);
    let unknown = EvaluatedValue::unknown(&ctx, input).unwrap();
    let output = evaluate(&mut ctx, &fixture, unknown).unwrap();
    assert!(!output.is_known());
    assert!(matches!(
        output.subvalue(&[0]).unwrap().state(),
        ValueState::Unknown
    ));
    assert!(matches!(
        output.subvalue(&[1]).unwrap().state(),
        ValueState::Unknown
    ));
}
