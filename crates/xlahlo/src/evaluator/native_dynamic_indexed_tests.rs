use super::*;
use crate::{
    attributes::Axes,
    inference::movement::{infer_dynamic_slice, infer_dynamic_update_slice},
    ops::{
        control::TupleOp,
        movement::{CopyOp, DynamicSliceOp, DynamicUpdateSliceOp},
    },
    shape::Dimension,
    types::ElementType,
};

struct DynamicFixture {
    module: Ptr<Operation>,
    root: Value,
    parameters: Vec<Value>,
}

fn dynamic_slice_fixture(
    ctx: &mut Context,
    input: TypeHandle,
    indices: &[TypeHandle],
    sizes: &[i64],
) -> DynamicFixture {
    let module = module(ctx);
    let mut signature = vec![input];
    signature.extend_from_slice(indices);
    let target = infer_dynamic_slice(ctx, input, indices, &Axes(sizes.to_vec())).unwrap();
    let (_, block, parameters) = computation(ctx, module, "main", &signature, target);
    let operation = DynamicSliceOp::new(ctx, target, parameters.clone(), Axes(sizes.to_vec()));
    let root = append(ctx, block, operation.get_operation(), "dynamic_slice");
    finish(ctx, block, root);
    DynamicFixture {
        module,
        root,
        parameters,
    }
}

fn dynamic_update_fixture(
    ctx: &mut Context,
    input: TypeHandle,
    update: TypeHandle,
    indices: &[TypeHandle],
) -> DynamicFixture {
    let module = module(ctx);
    let mut signature = vec![input, update];
    signature.extend_from_slice(indices);
    let target = infer_dynamic_update_slice(ctx, input, update, indices).unwrap();
    let tuple_target = shape::tuple(ctx, vec![target, target]).unwrap();
    let (_, block, parameters) = computation(ctx, module, "main", &signature, tuple_target);
    let operation = DynamicUpdateSliceOp::new(ctx, target, parameters.clone());
    let updated = append(ctx, block, operation.get_operation(), "dynamic_update");
    let first_copy = CopyOp::new(ctx, target, vec![updated]).get_operation();
    let first_copy = append(ctx, block, first_copy, "first_copy");
    let second_copy = CopyOp::new(ctx, target, vec![updated]).get_operation();
    let second_copy = append(ctx, block, second_copy, "second_copy");
    let root = TupleOp::new(ctx, tuple_target, vec![first_copy, second_copy]).get_operation();
    let root = append(ctx, block, root, "results");
    finish(ctx, block, root);
    DynamicFixture {
        module,
        root,
        parameters,
    }
}

fn evaluate_fixture(
    ctx: &Context,
    fixture: &DynamicFixture,
    values: Vec<Literal>,
) -> EvaluatedValue {
    let substitutions = fixture
        .parameters
        .iter()
        .copied()
        .zip(values.into_iter().map(|literal| known(ctx, literal)))
        .collect::<Vec<_>>();
    super::evaluate(
        ctx,
        fixture.module,
        fixture.root,
        &substitutions,
        DependencyMode::Recursive,
    )
    .unwrap()
}

fn scalar_index(ctx: &Context, element: ElementType, bits: u128) -> Literal {
    Literal::from_bits(ctx, shape::scalar(ctx, element).unwrap(), &[bits]).unwrap()
}

#[test]
fn bounded_dynamic_slice_clamps_negative_and_unsigned_large_indices() {
    let mut ctx = context();
    let input = shape::dynamic_array(&ctx, ElementType::F32, vec![Dimension::Bounded(8)]).unwrap();
    let signed = shape::scalar(&ctx, ElementType::S64).unwrap();
    let fixture = dynamic_slice_fixture(&mut ctx, input, &[signed], &[3]);
    let source_bits = [
        0x7fa0_0001u32,
        0x8000_0000,
        0x3f80_0000,
        0xffc0_0123,
        0x7f80_0000,
        0xdead_beef,
        0x1357_9bdf,
        0x2468_ace0,
    ];
    let source = Literal::from_dynamic_storage_bytes(
        &ctx,
        input,
        source_bits
            .iter()
            .flat_map(|bits| bits.to_le_bytes())
            .collect(),
        vec![5],
    )
    .unwrap();
    let result = evaluate_fixture(
        &ctx,
        &fixture,
        vec![
            source.clone(),
            scalar_index(&ctx, ElementType::S64, (-12i64) as u64 as u128),
        ],
    )
    .into_literal(&ctx)
    .unwrap();
    assert_eq!(
        result.storage,
        source.storage[..12].to_vec(),
        "negative signed starts clamp to zero"
    );

    let unsigned = shape::scalar(&ctx, ElementType::U64).unwrap();
    let fixture = dynamic_slice_fixture(&mut ctx, input, &[unsigned], &[3]);
    let result = evaluate_fixture(
        &ctx,
        &fixture,
        vec![
            source,
            scalar_index(&ctx, ElementType::U64, u64::MAX as u128),
        ],
    )
    .into_literal(&ctx)
    .unwrap();
    assert_eq!(
        result.storage,
        [0x3f80_0000u32, 0xffc0_0123, 0x7f80_0000]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>(),
        "unsigned oversized starts clamp to the last valid active window without reading inactive storage"
    );
}

#[test]
fn bounded_dynamic_slice_propagates_active_extent_for_full_bound_slices() {
    let mut ctx = context();
    let input = shape::dynamic_array(&ctx, ElementType::S32, vec![Dimension::Bounded(6)]).unwrap();
    let index = shape::array(&ctx, ElementType::U64, vec![1]).unwrap();
    let fixture = dynamic_slice_fixture(&mut ctx, input, &[index], &[6]);
    let source = Literal::from_dynamic_storage_bytes(
        &ctx,
        input,
        (10i32..16).flat_map(i32::to_le_bytes).collect(),
        vec![3],
    )
    .unwrap();
    let indices = Literal::from_bits(&ctx, index, &[u64::MAX as u128]).unwrap();
    let output = evaluate_fixture(&ctx, &fixture, vec![source, indices])
        .into_literal(&ctx)
        .unwrap();
    assert_eq!(output.dynamic_sizes, [3]);
    assert_eq!(
        crate::literal::semantics::geometry(&ctx, &output)
            .unwrap()
            .dimensions(),
        [3]
    );
    assert_eq!(output.bits(&ctx).unwrap(), [10, 11, 12, 0, 0, 0]);
}

#[test]
fn dynamic_slice_handles_scalar_zero_bound_and_nondefault_physical_layout() {
    let mut ctx = context();
    let scalar = shape::scalar(&ctx, ElementType::C64).unwrap();
    let scalar_fixture = dynamic_slice_fixture(&mut ctx, scalar, &[], &[]);
    let scalar_input = Literal::from_bits(&ctx, scalar, &[0x7fc0_1234_8000_0000]).unwrap();
    let scalar_output = evaluate_fixture(&ctx, &scalar_fixture, vec![scalar_input.clone()])
        .into_literal(&ctx)
        .unwrap();
    assert_eq!(scalar_output.storage, scalar_input.storage);

    let empty = shape::dynamic_array(&ctx, ElementType::S32, vec![Dimension::Bounded(0)]).unwrap();
    let index = shape::scalar(&ctx, ElementType::S32).unwrap();
    let empty_fixture = dynamic_slice_fixture(&mut ctx, empty, &[index], &[0]);
    let empty_input = Literal::from_dynamic_storage_bytes(&ctx, empty, vec![], vec![0]).unwrap();
    let empty_output = evaluate_fixture(
        &ctx,
        &empty_fixture,
        vec![empty_input, scalar_index(&ctx, ElementType::S32, 0)],
    )
    .into_literal(&ctx)
    .unwrap();
    assert!(empty_output.storage.is_empty());

    let input = shape::dynamic_array(
        &ctx,
        ElementType::S32,
        vec![Dimension::Bounded(2), Dimension::Bounded(3)],
    )
    .unwrap();
    let input = shape::with_layout(
        &ctx,
        input,
        crate::shape::Layout {
            minor_to_major: vec![0, 1],
            ..crate::shape::Layout::descending(2)
        },
    )
    .unwrap();
    let indices = [
        shape::scalar(&ctx, ElementType::S32).unwrap(),
        shape::scalar(&ctx, ElementType::S32).unwrap(),
    ];
    let fixture = dynamic_slice_fixture(&mut ctx, input, &indices, &[2, 2]);
    let source = Literal::from_dynamic_storage_bytes(
        &ctx,
        input,
        [1i32, 4, 2, 5, 3, 6]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect(),
        vec![2, 3],
    )
    .unwrap();
    let output = evaluate_fixture(
        &ctx,
        &fixture,
        vec![
            source,
            scalar_index(&ctx, ElementType::S32, 0),
            scalar_index(&ctx, ElementType::S32, 1),
        ],
    )
    .into_literal(&ctx)
    .unwrap();
    assert_eq!(output.bits(&ctx).unwrap(), [2, 3, 5, 6]);
}

#[test]
fn dynamic_update_clamps_to_active_extent_preserves_raw_bytes_and_shares_result() {
    let mut ctx = context();
    let input = shape::dynamic_array(&ctx, ElementType::S4, vec![Dimension::Bounded(6)]).unwrap();
    let update = shape::array(&ctx, ElementType::S4, vec![2]).unwrap();
    let index = shape::scalar(&ctx, ElementType::S64).unwrap();
    let fixture = dynamic_update_fixture(&mut ctx, input, update, &[index]);
    let source = Literal::from_dynamic_storage_bytes(
        &ctx,
        input,
        vec![0xa1, 0xb2, 0xc3, 0xd4, 0x55, 0x66],
        vec![4],
    )
    .unwrap();
    let replacement = Literal::from_storage_bytes(&ctx, update, vec![0x7f, 0x80]).unwrap();
    let output = evaluate_fixture(
        &ctx,
        &fixture,
        vec![
            source.clone(),
            replacement.clone(),
            scalar_index(&ctx, ElementType::S64, (-9i64) as u64 as u128),
        ],
    )
    .into_literal(&ctx)
    .unwrap();
    let first = output.sub_literal(&ctx, &[0]).unwrap();
    let second = output.sub_literal(&ctx, &[1]).unwrap();
    assert_eq!(first.storage, [0x7f, 0x80, 0xc3, 0xd4, 0x55, 0x66]);
    assert_eq!(second.storage, first.storage);
    assert_eq!(first.dynamic_sizes, [4]);
    assert_eq!(source.storage, [0xa1, 0xb2, 0xc3, 0xd4, 0x55, 0x66]);

    let unsigned = shape::scalar(&ctx, ElementType::U64).unwrap();
    let fixture = dynamic_update_fixture(&mut ctx, input, update, &[unsigned]);
    let output = evaluate_fixture(
        &ctx,
        &fixture,
        vec![
            source,
            replacement,
            scalar_index(&ctx, ElementType::U64, u64::MAX as u128),
        ],
    )
    .into_literal(&ctx)
    .unwrap();
    assert_eq!(
        output.sub_literal(&ctx, &[0]).unwrap().storage,
        [0xa1, 0xb2, 0x7f, 0x80, 0x55, 0x66]
    );
}

#[test]
fn dynamic_update_relayouts_from_update_view_to_operand_view() {
    let mut ctx = context();
    let input = shape::dynamic_array(
        &ctx,
        ElementType::S32,
        vec![Dimension::Bounded(2), Dimension::Bounded(3)],
    )
    .unwrap();
    let input = shape::with_layout(
        &ctx,
        input,
        crate::shape::Layout {
            minor_to_major: vec![0, 1],
            ..crate::shape::Layout::descending(2)
        },
    )
    .unwrap();
    let update = shape::array(&ctx, ElementType::S32, vec![1, 2]).unwrap();
    let indices = [
        shape::scalar(&ctx, ElementType::S32).unwrap(),
        shape::scalar(&ctx, ElementType::S32).unwrap(),
    ];
    let fixture = dynamic_update_fixture(&mut ctx, input, update, &indices);
    let source = Literal::from_dynamic_storage_bytes(
        &ctx,
        input,
        [1i32, 4, 2, 5, 3, 6]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect(),
        vec![2, 3],
    )
    .unwrap();
    let replacement = Literal::from_bits(&ctx, update, &[9, 8]).unwrap();
    let output = evaluate_fixture(
        &ctx,
        &fixture,
        vec![
            source,
            replacement,
            scalar_index(&ctx, ElementType::S32, 1),
            scalar_index(&ctx, ElementType::S32, 1),
        ],
    )
    .into_literal(&ctx)
    .unwrap();
    assert_eq!(
        output.sub_literal(&ctx, &[0]).unwrap().bits(&ctx).unwrap(),
        [1, 4, 2, 9, 3, 8]
    );
}

#[test]
fn dynamic_indexed_movement_returns_typed_unknown_for_unknown_inputs() {
    let mut ctx = context();
    let input = shape::dynamic_array(&ctx, ElementType::S32, vec![Dimension::Bounded(4)]).unwrap();
    let index = shape::scalar(&ctx, ElementType::S32).unwrap();
    let fixture = dynamic_slice_fixture(&mut ctx, input, &[index], &[2]);
    let substitutions = [
        (
            fixture.parameters[0],
            EvaluatedValue::unknown(&ctx, input).unwrap(),
        ),
        (
            fixture.parameters[1],
            known(&ctx, scalar_index(&ctx, ElementType::S32, 0)),
        ),
    ];
    let output = super::evaluate(
        &ctx,
        fixture.module,
        fixture.root,
        &substitutions,
        DependencyMode::Recursive,
    )
    .unwrap();
    assert!(matches!(output.state(), ValueState::Unknown));

    let update = shape::array(&ctx, ElementType::S32, vec![1]).unwrap();
    let update_fixture = dynamic_update_fixture(&mut ctx, input, update, &[index]);
    let known_input = Literal::from_dynamic_storage_bytes(
        &ctx,
        input,
        [1i32, 2, 3, 4]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect(),
        vec![4],
    )
    .unwrap();
    let unknown_update = [
        (update_fixture.parameters[0], known(&ctx, known_input)),
        (
            update_fixture.parameters[1],
            EvaluatedValue::unknown(&ctx, update).unwrap(),
        ),
        (
            update_fixture.parameters[2],
            known(&ctx, scalar_index(&ctx, ElementType::S32, 0)),
        ),
    ];
    let output = super::evaluate(
        &ctx,
        update_fixture.module,
        update_fixture.root,
        &unknown_update,
        DependencyMode::Recursive,
    )
    .unwrap();
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
