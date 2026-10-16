use xlahlo::{
    Dimension as D, Instruction, Module, Opcode as O, PrimitiveType as P, Shape, inference,
};

fn graph() -> (
    Module,
    xlahlo::ComputationId,
    xlahlo::InstructionId,
    xlahlo::InstructionId,
) {
    let mut m = Module::new("editing");
    let c = m.add_computation("main").unwrap();
    let shape = Shape::array(P::F32, vec![2]).unwrap();
    let p = m
        .add_instruction(
            c,
            Instruction::new("p", O::Parameter, shape.clone()),
            &[],
            &[],
        )
        .unwrap();
    let n = m
        .add_instruction(c, Instruction::new("n", O::Negate, shape), &[p], &[])
        .unwrap();
    m.set_root(c, n).unwrap();
    m.set_entry(c).unwrap();
    (m, c, p, n)
}

#[test]
fn computation_clone_remaps_data_control_and_roots() {
    let (mut m, c, p, n) = graph();
    m.add_control_dependency(p, n).unwrap();
    let clone = m.clone_computation(c, "copy").unwrap();
    let new_p = clone.instructions[&p];
    let new_n = clone.instructions[&n];
    assert_eq!(m.root(clone.computation).unwrap(), new_n);
    assert_eq!(m.operands(new_n).unwrap(), [new_p]);
    assert_eq!(
        m.instruction(new_n).unwrap().control_predecessor_ids,
        [m.instruction(new_p).unwrap().id]
    );
    m.remove_control_dependency(p, n).unwrap();
    assert_eq!(
        m.instruction(new_n).unwrap().control_predecessor_ids.len(),
        1
    );
    m.verify(Default::default()).unwrap();
}

#[test]
fn typed_payload_edit_does_not_allow_reference_or_identity_corruption() {
    let (mut m, _, _, n) = graph();
    let before = m.to_proto().unwrap();
    assert!(
        m.edit_instruction_fields(n, |i| {
            i.id = 9000;
            Ok(())
        })
        .is_err()
    );
    assert_eq!(m.to_proto().unwrap(), before);
    m.edit_instruction_fields(n, |i| {
        i.opcode = "abs".into();
        Ok(())
    })
    .unwrap();
    m.verify(Default::default()).unwrap();
    assert_eq!(m.instruction(n).unwrap().opcode, "abs");
}

#[test]
fn scalar_and_degenerate_binary_broadcasting_follow_upstream_geometry() {
    let a = Shape::array(P::F32, vec![2, 1]).unwrap();
    let b = Shape::array(P::F32, vec![1, 3]).unwrap();
    assert_eq!(
        inference::infer_binary(O::Add, &a, &b)
            .unwrap()
            .dimensions(),
        [2, 3]
    );
    let scalar = Shape::scalar(P::F32).unwrap();
    assert_eq!(
        inference::infer_binary(O::Add, &a, &scalar)
            .unwrap()
            .dimensions(),
        [2, 1]
    );
    let vector = Shape::array(P::F32, vec![3]).unwrap();
    assert!(inference::infer_binary(O::Add, &a, &vector).is_err());
    assert_eq!(
        inference::infer_binary_in_dim(O::Add, &a, &vector, &[1])
            .unwrap()
            .dimensions(),
        [2, 3]
    );
    assert!(inference::infer_binary_in_dim(O::Add, &a, &b, &[1, 0]).is_err());
}

#[test]
fn broadcast_dynamic_slice_and_update_validate_indices_and_bounds() {
    let input = Shape::dynamic_array(P::F32, vec![D::Bounded(4), D::Static(5)]).unwrap();
    let indices = vec![Shape::scalar(P::S32).unwrap(); 2];
    let full = inference::infer_dynamic_slice(&input, &indices, &[4, 5]).unwrap();
    assert_eq!(full.dimension(0).unwrap(), D::Bounded(4));
    let partial = inference::infer_dynamic_slice(&input, &indices, &[2, 3]).unwrap();
    assert_eq!(partial.dimension(0).unwrap(), D::Static(2));
    assert!(inference::infer_dynamic_slice(&input, &indices, &[5, 3]).is_err());
    assert!(inference::infer_dynamic_slice(&input, &[], &[2, 3]).is_err());
    assert!(
        inference::infer_dynamic_slice(&input, &vec![Shape::scalar(P::F32).unwrap(); 2], &[2, 3])
            .is_err()
    );
    let updated = inference::infer_dynamic_update_slice(&input, &partial, &indices).unwrap();
    assert_eq!(updated, input);
    let wrong = Shape::array(P::S32, vec![2, 3]).unwrap();
    assert!(inference::infer_dynamic_update_slice(&input, &wrong, &indices).is_err());
}

#[test]
fn bitcast_conversion_tracks_storage_width_and_round_trips_layout() {
    let input = Shape::array(P::F32, vec![2, 3])
        .unwrap()
        .with_default_layout()
        .unwrap();
    let split = inference::infer_bitcast_convert(&input, P::U8).unwrap();
    assert_eq!(split.dimensions(), [2, 3, 4]);
    assert_eq!(split.layout().unwrap().minor_to_major, [2, 1, 0]);
    assert_eq!(
        inference::infer_bitcast_convert(&split, P::F32).unwrap(),
        input
    );
    assert!(inference::infer_bitcast_convert(&input, P::C64).is_err());
    assert!(inference::infer_bitcast_convert(&input, P::F64).is_err());
    assert!(inference::infer_bitcast_convert(&input, P::F6E2M3FN).is_err());
}

#[test]
fn inference_materializes_default_layouts_only_when_upstream_constructs_new_shapes() {
    let input = Shape::array(P::F32, vec![2, 3]).unwrap();
    let expected = Shape::array(P::F32, vec![2, 3])
        .unwrap()
        .with_default_layout()
        .unwrap();
    assert_eq!(
        inference::infer_slice(&input, &[0, 0], &[2, 3], &[1, 1]).unwrap(),
        expected
    );
    assert_eq!(
        inference::infer_static_reshape(&input, &[2, 3]).unwrap(),
        expected
    );
    assert_eq!(
        inference::infer_concatenate(std::slice::from_ref(&input), 0).unwrap(),
        expected
    );
    let indices = vec![Shape::scalar(P::S32).unwrap(); 2];
    assert_eq!(
        inference::infer_dynamic_slice(&input, &indices, &[2, 3]).unwrap(),
        expected
    );
    assert_eq!(inference::infer_unary(O::Negate, &input).unwrap(), input);
    assert_eq!(
        inference::infer_binary(O::Add, &input, &input).unwrap(),
        input
    );
    let degenerate = Shape::array(P::F32, vec![1, 3]).unwrap();
    assert_eq!(
        inference::infer_binary(O::Add, &input, &degenerate).unwrap(),
        expected
    );
}

#[test]
fn dimension_size_inference_checks_i32_limits_and_scalar_size_domain() {
    let input = Shape::array(P::F32, vec![3, 4]).unwrap();
    let size = Shape::scalar(P::S32).unwrap();
    assert_eq!(
        inference::infer_get_dimension_size(&input, 1).unwrap(),
        size.clone().with_default_layout().unwrap()
    );
    let result = inference::infer_set_dimension_size(&input, &size, 1).unwrap();
    assert_eq!(result.dimension(1).unwrap(), D::Bounded(4));
    assert!(
        inference::infer_set_dimension_size(&input, &Shape::scalar(P::S64).unwrap(), 1).is_err()
    );
    assert!(inference::infer_get_dimension_size(&input, -1).is_err());
    let huge = Shape::array(P::U8, vec![i64::from(i32::MAX) + 1]).unwrap();
    assert!(inference::infer_get_dimension_size(&huge, 0).is_err());
}
