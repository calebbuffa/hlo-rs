use crate::support::{computation, context, module, ret};
use xlahlo::{
    attributes::{Axes, Callees, Index},
    ir::{ComputationCloneRoot, ComputationReplacements, clone_computation_with_replacements},
    literal::Literal,
    ops::{
        control::{CallOp, ConditionalOp, ConstantOp, GetTupleElementOp, ParameterOp, TupleOp},
        elementwise::AddOp,
        numerical::SortOp,
    },
    pliron::{
        context::Context,
        identifier::Identifier,
        linked_list::ContainsLinkedList,
        op::Op,
        operation::Operation,
        printable::Printable,
        r#type::{TypeHandle, Typed},
    },
    shape,
    types::ElementType,
    verification::{VerifyOptions, verify},
};

fn parameter(ctx: &mut Context, ty: TypeHandle, index: u32) -> ParameterOp {
    ParameterOp::new(ctx, ty, vec![], Index(index))
}

#[test]
fn comparator_parameter_pair_compaction_uses_detached_templates_and_preserves_shared_source() {
    let ctx = &mut context();
    let scalar = shape::array(ctx, ElementType::F32, vec![]).unwrap();
    let pred = shape::array(ctx, ElementType::Pred, vec![]).unwrap();
    let array = shape::array(ctx, ElementType::F32, vec![8]).unwrap();
    let pair = shape::tuple(ctx, vec![array, array]).unwrap();
    let (module, container) = module(ctx);
    let (source, block, parameters) = computation(ctx, container, "compare", vec![scalar; 4], pred);
    let literal = Literal::from_bits(ctx, pred, &[0]).unwrap();
    let result = ConstantOp::new(ctx, pred, vec![], literal);
    result.get_operation().insert_at_back(block, ctx);
    let value = result.result(ctx);
    ret(ctx, block, value);
    let (_, block, parameters_main) = computation(ctx, container, "main", vec![array; 2], pair);
    let sort = SortOp::new(
        ctx,
        pair,
        parameters_main.clone(),
        Axes(vec![0]),
        Callees(vec!["compare".into()]),
    );
    sort.get_operation().insert_at_back(block, ctx);
    let value = sort.result(ctx);
    ret(ctx, block, value);
    let before = source.disp(ctx).to_string();
    let left = parameter(ctx, scalar, 0);
    let right = parameter(ctx, scalar, 1);
    let template_before = (left.disp(ctx).to_string(), right.disp(ctx).to_string());
    let mut replacements = ComputationReplacements::default();
    for value in &parameters[..2] {
        replacements
            .instructions
            .insert(value.defining_op().unwrap(), None);
    }
    replacements.instructions.insert(
        parameters[2].defining_op().unwrap(),
        Some(left.get_operation()),
    );
    replacements.instructions.insert(
        parameters[3].defining_op().unwrap(),
        Some(right.get_operation()),
    );
    let cloned = clone_computation_with_replacements(
        ctx,
        &source,
        Identifier::new("compare_compact"),
        &replacements,
        Some(VerifyOptions::default()),
    )
    .unwrap();
    assert!(
        cloned
            .computation
            .get_operation()
            .deref(ctx)
            .get_parent_block()
            .is_none()
    );
    let left_clone = cloned.mapping.lookup_op(left.get_operation()).unwrap();
    let right_clone = cloned.mapping.lookup_op(right.get_operation()).unwrap();
    assert_eq!(
        Operation::get_op::<ParameterOp>(left_clone, ctx)
            .unwrap()
            .index(ctx)
            .unwrap()
            .0,
        0
    );
    assert_eq!(
        Operation::get_op::<ParameterOp>(right_clone, ctx)
            .unwrap()
            .index(ctx)
            .unwrap()
            .0,
        1
    );
    assert!(
        cloned
            .mapping
            .lookup_op(parameters[2].defining_op().unwrap())
            .is_none()
    );
    assert!(cloned.embedded_computations.is_empty());
    assert!(cloned.embedded_schedules.is_empty());
    assert_eq!(source.disp(ctx).to_string(), before);
    assert_eq!(
        (left.disp(ctx).to_string(), right.disp(ctx).to_string()),
        template_before
    );
    let symbol = cloned
        .computation
        .get_attr_hlo_computation_name(ctx)
        .unwrap()
        .clone();
    cloned
        .computation
        .get_operation()
        .insert_at_back(container, ctx);
    let ret = block.deref(ctx).iter(ctx).last().unwrap();
    let compact_sort = SortOp::new(
        ctx,
        array,
        vec![parameters_main[1]],
        Axes(vec![0]),
        Callees(vec![symbol]),
    );
    compact_sort.get_operation().insert_before(ctx, ret);
    verify(ctx, module.get_operation()).unwrap();
    let published = cloned.computation.get_operation();
    let (owned, _) = cloned.dispose(ctx).err().unwrap();
    assert_eq!(owned.computation.get_operation(), published);
    assert_eq!(published.deref(ctx).get_parent_block(), Some(container));
    verify(ctx, module.get_operation()).unwrap();
    Operation::erase(left.get_operation(), ctx);
    Operation::erase(right.get_operation(), ctx);
}

#[test]
fn tuple_output_compaction_preserves_selected_mapping_and_caller_reconstruction() {
    let ctx = &mut context();
    let scalar = shape::array(ctx, ElementType::F32, vec![]).unwrap();
    let pair = shape::tuple(ctx, vec![scalar, scalar]).unwrap();
    let single = shape::tuple(ctx, vec![scalar]).unwrap();
    let (module, container) = module(ctx);
    let (source, block, parameters) = computation(ctx, container, "callee", vec![scalar; 2], pair);
    let add = AddOp::new(ctx, scalar, vec![parameters[1], parameters[1]]);
    add.get_operation().insert_at_back(block, ctx);
    let add_value = add.result(ctx);
    let tuple = TupleOp::new(ctx, pair, vec![parameters[0], add_value]);
    tuple.get_operation().insert_at_back(block, ctx);
    let root = tuple.result(ctx);
    ret(ctx, block, root);
    let (_, main, inputs) = computation(ctx, container, "main", vec![scalar; 2], pair);
    let old_call = CallOp::new(ctx, pair, inputs.clone(), Callees(vec!["callee".into()]));
    old_call.get_operation().insert_at_back(main, ctx);
    let value = old_call.result(ctx);
    ret(ctx, main, value);
    let source_before = source.disp(ctx).to_string();
    let retained = parameter(ctx, scalar, 0);
    let mut replacements = ComputationReplacements {
        root: ComputationCloneRoot::Tuple(vec![add.get_operation()]),
        ..Default::default()
    };
    replacements
        .instructions
        .insert(parameters[0].defining_op().unwrap(), None);
    replacements.instructions.insert(
        parameters[1].defining_op().unwrap(),
        Some(retained.get_operation()),
    );
    replacements
        .instructions
        .insert(tuple.get_operation(), None);
    let cloned = clone_computation_with_replacements(
        ctx,
        &source,
        Identifier::new("callee_compact"),
        &replacements,
        Some(Default::default()),
    )
    .unwrap();
    assert!(cloned.mapping.lookup_op(retained.get_operation()).is_some());
    assert!(cloned.mapping.lookup_op(add.get_operation()).is_some());
    assert_eq!(source_before, source.disp(ctx).to_string());
    let symbol = cloned
        .computation
        .get_attr_hlo_computation_name(ctx)
        .unwrap()
        .clone();
    cloned
        .computation
        .get_operation()
        .insert_at_back(container, ctx);
    let terminator = main.deref(ctx).iter(ctx).last().unwrap();
    let new_call = CallOp::new(ctx, single, vec![inputs[1]], Callees(vec![symbol]));
    new_call.get_operation().insert_before(ctx, terminator);
    let value = new_call.result(ctx);
    let projection = GetTupleElementOp::new(ctx, scalar, vec![value], Index(0));
    projection.get_operation().insert_before(ctx, terminator);
    let result = projection.result(ctx);
    let reconstructed = TupleOp::new(ctx, pair, vec![inputs[0], result]);
    reconstructed.get_operation().insert_before(ctx, terminator);
    let result = reconstructed.result(ctx);
    xlahlo::ir::replace_all_uses_with(ctx, old_call.get_operation(), result, None).unwrap();
    assert_eq!(terminator.deref(ctx).get_operand(0), result);
    assert_eq!(
        old_call.get_operation().deref(ctx).get_parent_block(),
        Some(main)
    );
    verify(ctx, module.get_operation()).unwrap();
    Operation::erase(retained.get_operation(), ctx);
}

#[test]
fn mixed_caller_shared_branch_clone_accepts_tuple_parameter_template_and_tuple_root() {
    let ctx = &mut context();
    let scalar = shape::array(ctx, ElementType::F32, vec![]).unwrap();
    let tuple_ty = shape::tuple(ctx, vec![scalar]).unwrap();
    let pred = shape::array(ctx, ElementType::Pred, vec![]).unwrap();
    let (module, container) = module(ctx);
    let (source, block, parameters) = computation(ctx, container, "branch", vec![scalar], scalar);
    let add = AddOp::new(ctx, scalar, vec![parameters[0], parameters[0]]);
    add.get_operation().insert_at_back(block, ctx);
    let value = add.result(ctx);
    ret(ctx, block, value);
    let (_, main, inputs) = computation(ctx, container, "main", vec![scalar], scalar);
    let call = CallOp::new(ctx, scalar, inputs.clone(), Callees(vec!["branch".into()]));
    call.get_operation().insert_at_back(main, ctx);
    let literal = Literal::from_bits(ctx, pred, &[0]).unwrap();
    let selector = ConstantOp::new(ctx, pred, vec![], literal);
    selector.get_operation().insert_at_back(main, ctx);
    let selector_value = selector.result(ctx);
    let conditional = ConditionalOp::new(
        ctx,
        scalar,
        vec![selector_value, inputs[0], inputs[0]],
        Callees(vec!["branch".into(), "branch".into()]),
    );
    conditional.get_operation().insert_at_back(main, ctx);
    let value = conditional.result(ctx);
    ret(ctx, main, value);
    verify(ctx, module.get_operation()).unwrap();
    let before = source.disp(ctx).to_string();
    let tuple_parameter = parameter(ctx, tuple_ty, 0);
    let input = tuple_parameter.result(ctx);
    let projection = GetTupleElementOp::new(ctx, scalar, vec![input], Index(0));
    let mut replacements = ComputationReplacements {
        root: ComputationCloneRoot::Tuple(vec![add.get_operation()]),
        ..Default::default()
    };
    replacements.instructions.insert(
        parameters[0].defining_op().unwrap(),
        Some(projection.get_operation()),
    );
    let cloned = clone_computation_with_replacements(
        ctx,
        &source,
        Identifier::new("branch_tuple"),
        &replacements,
        Some(Default::default()),
    )
    .unwrap();
    let projected = cloned
        .mapping
        .lookup_op(projection.get_operation())
        .unwrap();
    let cloned_parameter = cloned
        .mapping
        .lookup_op(tuple_parameter.get_operation())
        .unwrap();
    assert_eq!(
        projected.deref(ctx).get_operand(0).defining_op(),
        Some(cloned_parameter)
    );
    assert_eq!(
        cloned_parameter.deref(ctx).get_result(0).get_type(ctx),
        tuple_ty
    );
    assert_eq!(before, source.disp(ctx).to_string());
    verify(ctx, module.get_operation()).unwrap();
    assert!(cloned.dispose(ctx).is_ok());
    Operation::erase(projection.get_operation(), ctx);
    Operation::erase(tuple_parameter.get_operation(), ctx);
}
