use xlahlo::{Module, PrimitiveType as P, Shape, VerifyOptions, inference, text};

fn strict() -> VerifyOptions {
    VerifyOptions {
        layout_sensitive: true,
        ..Default::default()
    }
}
const CHAIN: &str = "HloModule m ENTRY main { p = f32[2] parameter(0) b = b(f32[2]) custom-call(p), custom_call_target=\"Pin\", output_to_operand_aliasing={{}: (0,{})} w = b(f32[2]) custom-call(b), custom_call_target=\"writer\", output_to_operand_aliasing={{}: (0,{})} ROOT u = f32[2] custom-call(w), custom_call_target=\"Unpin\", output_to_operand_aliasing={{}: (0,{})} }";

#[test]
fn buffer_shapes_and_pin_writer_unpin_chain_round_trip() {
    let shape = Shape::buffer(
        Shape::array(P::F32, vec![2])
            .unwrap()
            .with_default_layout()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        text::parse_shape(&text::print_shape(&shape).unwrap()).unwrap(),
        shape
    );
    assert!(text::parse_shape("b((f32[2],s32[]))").is_err());
    let module = text::parse_module(CHAIN).unwrap();
    module.verify(strict()).unwrap();
    let printed = text::print_module(&module).unwrap();
    let parsed = text::parse_module(&printed).unwrap();
    parsed.verify(strict()).unwrap();
    assert_eq!(printed, text::print_module(&parsed).unwrap());
    let created = CHAIN.replace(
        "custom-call(p), custom_call_target=\"Pin\", output_to_operand_aliasing={{}: (0,{})}",
        "custom-call(), custom_call_target=\"CreateBuffer\"",
    );
    text::parse_module(&created)
        .unwrap()
        .verify(strict())
        .unwrap();
}

#[test]
fn pin_unpin_and_create_buffer_validate_arity_shapes_dynamic_flags_and_aliases() {
    for source in [
        CHAIN.replace("custom-call(p),", "custom-call(),"),
        CHAIN.replace("b = b(f32[2])", "b = b(f32[3])"),
        CHAIN.replace("b = b(f32[2])", "b = f32[2]"),
        CHAIN.replace("p = f32[2]", "p = f32[<=2]"),
        CHAIN.replace(", output_to_operand_aliasing={{}: (0,{})}", ""),
        CHAIN.replace(
            "custom_call_target=\"Pin\"",
            "custom_call_target=\"CreateBuffer\"",
        ),
        CHAIN.replace("u = f32[2]", "u = b(f32[2])"),
    ] {
        let module = text::parse_module(&source).unwrap();
        assert!(module.verify(Default::default()).is_err(), "{source}");
    }
}

#[test]
fn every_nested_buffer_operand_and_result_requires_its_own_alias() {
    let source = "HloModule m ENTRY main { b = b(f32[2]) custom-call(), custom_call_target=\"CreateBuffer\" t = (b(f32[2])) tuple(b) ROOT w = (b(f32[2])) custom-call(t), custom_call_target=\"writer\", output_to_operand_aliasing={{0}: (0,{0})} }";
    text::parse_module(source)
        .unwrap()
        .verify(strict())
        .unwrap();
    for aliases in ["{}", "{{}: (0,{})}", "{{0}: (0,{0}),{0}: (0,{0})}"] {
        let changed = source.replace("{{0}: (0,{0})}", aliases);
        assert!(
            text::parse_module(&changed)
                .unwrap()
                .verify(strict())
                .is_err()
        );
    }
    let changed=source.replace("{{0}: (0,{0})}","{}").replace("custom_call_target=\"writer\"","custom_call_target=\"writer\", frontend_attributes={xla_skip_custom_call_alias_shape_check=\"true\"}");
    assert!(
        text::parse_module(&changed)
            .unwrap()
            .verify(strict())
            .is_err()
    );
}

#[test]
fn multiple_writers_and_unpins_are_rejected_through_tuple_get_element_paths() {
    let source=CHAIN.replace("ROOT u =","extra = b(f32[2]) custom-call(b), custom_call_target=\"writer2\", output_to_operand_aliasing={{}: (0,{})} ROOT u =");
    assert_eq!(
        text::parse_module(&source)
            .unwrap()
            .verify(strict())
            .unwrap_err()
            .category,
        "buffer"
    );
    let source = "HloModule m ENTRY main { b = b(f32[2]) custom-call(), custom_call_target=\"CreateBuffer\" t = (b(f32[2])) tuple(b) g = b(f32[2]) get-tuple-element(t), index=0 ROOT w = b(f32[2]) custom-call(g), custom_call_target=\"writer\", output_to_operand_aliasing={{}: (0,{})} }";
    text::parse_module(source)
        .unwrap()
        .verify(strict())
        .unwrap();
    let changed=source.replace("ROOT w =","g2 = b(f32[2]) get-tuple-element(t), index=0 extra = f32[2] custom-call(g2), custom_call_target=\"Unpin\", output_to_operand_aliasing={{}: (0,{})} ROOT w =");
    assert_eq!(
        text::parse_module(&changed)
            .unwrap()
            .verify(strict())
            .unwrap_err()
            .category,
        "buffer"
    );
}

#[test]
fn buffers_cannot_enter_as_parameters_or_flow_through_ordinary_operations() {
    let source = "HloModule m ENTRY main { ROOT p = b(f32[2]) parameter(0) }";
    assert_eq!(
        text::parse_module(source)
            .unwrap()
            .verify(strict())
            .unwrap_err()
            .category,
        "buffer"
    );
    let source = "HloModule m ENTRY main { b = b(f32[2]) custom-call(), custom_call_target=\"CreateBuffer\" ROOT c = b(f32[2]) copy(b) }";
    assert_eq!(
        text::parse_module(source)
            .unwrap()
            .verify(strict())
            .unwrap_err()
            .category,
        "buffer"
    );
}

#[test]
fn dynamic_update_slice_preserves_buffers_and_enforces_unique_writers() {
    let source = "HloModule m ENTRY main { b = b(f32[2]) custom-call(), custom_call_target=\"CreateBuffer\" p = f32[1] parameter(0) i = s32[] constant(0) d = b(f32[2]) dynamic-update-slice(b,p,i) ROOT u = f32[2] custom-call(d), custom_call_target=\"Unpin\", output_to_operand_aliasing={{}: (0,{})} }";
    let module = text::parse_module(source).unwrap();
    module.verify(strict()).unwrap();
    let shape = Shape::buffer(Shape::array(P::F32, vec![2]).unwrap()).unwrap();
    let inferred = inference::infer_dynamic_update_slice(
        &shape,
        &Shape::array(P::F32, vec![1]).unwrap(),
        &[Shape::scalar(P::S32).unwrap()],
    )
    .unwrap();
    assert_eq!(inferred.element_type(), P::BUFFER);
    let changed = source.replace(
        "ROOT u =",
        "extra = b(f32[2]) dynamic-update-slice(b,p,i) ROOT u =",
    );
    assert_eq!(
        text::parse_module(&changed)
            .unwrap()
            .verify(strict())
            .unwrap_err()
            .category,
        "buffer"
    );
    let changed = source.replace("d = b(f32[2])", "d = f32[2]");
    assert!(
        text::parse_module(&changed)
            .unwrap()
            .verify(strict())
            .is_err()
    );
}

#[test]
fn buffer_while_state_and_stream_annotated_calls_use_contextual_rules() {
    let source = "HloModule m body { ROOT p = b(f32[2]) parameter(0) } cond { p = b(f32[2]) parameter(0) ROOT c = pred[] constant(false) } ENTRY main { b = b(f32[2]) custom-call(), custom_call_target=\"CreateBuffer\" w = b(f32[2]) while(b), body=body, condition=cond ROOT u = f32[2] custom-call(w), custom_call_target=\"Unpin\", output_to_operand_aliasing={{}: (0,{})} }";
    text::parse_module(source)
        .unwrap()
        .verify(strict())
        .unwrap();
    let source = "HloModule m body { ROOT p = b(f32[2]) parameter(0) } ENTRY main { b = b(f32[2]) custom-call(), custom_call_target=\"CreateBuffer\" ROOT c = b(f32[2]) call(b), to_apply=body, frontend_attributes={_xla_stream_annotation=\"1\"} }";
    text::parse_module(source)
        .unwrap()
        .verify(strict())
        .unwrap();
    assert_eq!(
        text::parse_module(
            &source.replace(", frontend_attributes={_xla_stream_annotation=\"1\"}", "")
        )
        .unwrap()
        .verify(strict())
        .unwrap_err()
        .category,
        "buffer"
    );
}

#[test]
fn scalar_buffer_aliases_may_change_memory_space_but_pin_must_match_layout() {
    let source = "HloModule m ENTRY main { b = b(f32[]) custom-call(), custom_call_target=\"CreateBuffer\" ROOT w = b(f32[]) custom-call(b), custom_call_target=\"writer\", output_to_operand_aliasing={{}: (0,{})} }";
    let mut wire = text::parse_module(source).unwrap().to_proto().unwrap();
    let writer = wire.computations[0]
        .instructions
        .iter_mut()
        .find(|i| i.name == "w")
        .unwrap();
    writer.shape.as_mut().unwrap().tuple_shapes[0]
        .layout
        .as_mut()
        .unwrap()
        .memory_space = 5;
    Module::from_proto(wire).unwrap().verify(strict()).unwrap();
    let mut wire = text::parse_module(CHAIN).unwrap().to_proto().unwrap();
    let pin = wire.computations[0]
        .instructions
        .iter_mut()
        .find(|i| i.name == "b")
        .unwrap();
    pin.shape.as_mut().unwrap().tuple_shapes[0]
        .layout
        .as_mut()
        .unwrap()
        .minor_to_major
        .clear();
    pin.shape.as_mut().unwrap().tuple_shapes[0]
        .layout
        .as_mut()
        .unwrap()
        .memory_space = 1;
    assert_eq!(
        Module::from_proto(wire)
            .unwrap()
            .verify(strict())
            .unwrap_err()
            .category,
        "layout"
    );
}

#[test]
fn array_only_inference_guards_do_not_treat_buffers_as_plain_arrays() {
    let array = text::parse_shape("f32[2,3]").unwrap();
    let buffer = Shape::buffer(array.clone()).unwrap();
    let index = Shape::scalar(P::S32).unwrap();
    assert!(inference::infer_reshape(&array, &[6], None).is_ok());
    assert!(inference::infer_transpose(&array, &[1, 0]).is_ok());
    assert!(inference::infer_reverse(&array, &[0]).is_ok());
    assert!(inference::infer_slice(&array, &[0, 0], &[2, 3], &[1, 1]).is_ok());
    assert!(
        inference::infer_dynamic_slice(&array, &[index.clone(), index.clone()], &[1, 2]).is_ok()
    );
    assert!(inference::infer_reshape(&buffer, &[6], None).is_err());
    assert!(inference::infer_transpose(&buffer, &[1, 0]).is_err());
    assert!(inference::infer_reverse(&buffer, &[0]).is_err());
    assert!(inference::infer_slice(&buffer, &[0, 0], &[2, 3], &[1, 1]).is_err());
    assert!(inference::infer_dynamic_slice(&buffer, &[index.clone(), index], &[1, 2]).is_err());
    assert!(inference::infer_convert(&buffer, P::F64).is_err());
    assert!(inference::infer_bitcast_convert(&buffer, P::U32).is_err());
}

#[test]
fn dynamic_update_slice_accepts_only_the_data_operand_as_a_buffer() {
    let array = text::parse_shape("f32[<=4]{0:T(2)}").unwrap();
    let buffer = Shape::buffer(array.clone()).unwrap();
    let update = text::parse_shape("f32[2]").unwrap();
    let indices = [Shape::scalar(P::S32).unwrap()];
    let expected = inference::infer_dynamic_update_slice(&array, &update, &indices).unwrap();
    let result = inference::infer_dynamic_update_slice(&buffer, &update, &indices).unwrap();
    assert!(result.equal_including_layout(&Shape::buffer(expected).unwrap()));
    let update = Shape::buffer(update).unwrap();
    assert!(inference::infer_dynamic_update_slice(&array, &update, &indices).is_err());
    assert!(inference::infer_dynamic_update_slice(&buffer, &update, &indices).is_err());
}

#[test]
fn scalar_slice_indices_can_be_buffers_but_legacy_vectors_cannot() {
    let array = text::parse_shape("f32[4,6]").unwrap();
    let update = text::parse_shape("f32[1,2]").unwrap();
    let scalar = text::parse_shape("s32[]").unwrap();
    let buffer = Shape::buffer(scalar.clone()).unwrap();
    for indices in [
        vec![scalar.clone(), scalar.clone()],
        vec![buffer.clone(), buffer.clone()],
        vec![text::parse_shape("s32[2]").unwrap()],
    ] {
        assert!(inference::infer_dynamic_slice(&array, &indices, &[1, 2]).is_ok());
        assert!(inference::infer_dynamic_update_slice(&array, &update, &indices).is_ok());
    }
    for indices in [
        vec![scalar.clone(), buffer.clone()],
        vec![buffer, scalar],
        vec![text::parse_shape("b(s32[2])").unwrap()],
        vec![text::parse_shape("b(f32[])").unwrap(); 2],
    ] {
        assert!(inference::infer_dynamic_slice(&array, &indices, &[1, 2]).is_err());
        assert!(inference::infer_dynamic_update_slice(&array, &update, &indices).is_err());
    }
}

#[test]
fn dimension_size_inference_reads_and_updates_the_array_under_a_buffer() {
    let input = text::parse_shape("b(f32[4,<=6]{1,0:T(2,2)})").unwrap();
    let size = text::parse_shape("b(s32[])").unwrap();
    assert!(
        inference::infer_get_dimension_size(&input, 1)
            .unwrap()
            .equal_including_layout(&text::parse_shape("s32[]").unwrap())
    );
    let mut wire = input.clone().into_proto();
    wire.special_fields
        .mut_unknown_fields()
        .add_varint(1000, 77);
    let input = Shape::from_proto(wire).unwrap();
    let output = inference::infer_set_dimension_size(&input, &size, 0).unwrap();
    assert!(
        output.equal_including_layout(&text::parse_shape("b(f32[<=4,<=6]{1,0:T(2,2)})").unwrap())
    );
    assert_eq!(
        input.as_proto().special_fields,
        output.as_proto().special_fields
    );
    assert!(inference::infer_set_dimension_size(&input.array_view().unwrap(), &size, 0).is_ok());
    for size in ["b(f32[])", "b(s32[1])"] {
        assert!(
            inference::infer_set_dimension_size(&input, &text::parse_shape(size).unwrap(), 0)
                .is_err()
        );
    }
    for dimension in [-1, 2] {
        assert!(inference::infer_get_dimension_size(&input, dimension).is_err());
        assert!(inference::infer_set_dimension_size(&input, &size, dimension).is_err());
    }
    let huge = Shape::buffer(Shape::array(P::F32, vec![i64::from(i32::MAX) + 1]).unwrap()).unwrap();
    assert!(inference::infer_get_dimension_size(&huge, 0).is_err());
    assert!(inference::infer_set_dimension_size(&huge, &size, 0).is_err());
}

#[test]
fn reduce_precision_inference_preserves_the_buffer_and_dynamic_reshape_does_not() {
    let input = text::parse_shape("b(f32[<=6]{0:T(2)})").unwrap();
    assert_eq!(
        inference::infer_reduce_precision(&input, 8, 10).unwrap(),
        input
    );
    assert!(inference::infer_reduce_precision(&input, 0, 10).is_err());
    assert!(inference::infer_reduce_precision(&input, 8, -1).is_err());
    assert!(
        inference::infer_reduce_precision(&text::parse_shape("b(s32[6])").unwrap(), 8, 10).is_err()
    );
    for size in ["s32[]", "b(s32[2])", "b(f32[])"] {
        let output = inference::infer_dynamic_reshape(
            &input,
            &[text::parse_shape(size).unwrap()],
            vec![xlahlo::Dimension::Bounded(2), xlahlo::Dimension::Static(3)],
        )
        .unwrap();
        assert!(output.equal_including_layout(&text::parse_shape("f32[<=2,3]").unwrap()));
    }

    assert!(
        inference::infer_dynamic_reshape(
            &input,
            &[text::parse_shape("b(f32[2])").unwrap()],
            vec![xlahlo::Dimension::Static(6)],
        )
        .is_err()
    );
    assert!(
        inference::infer_dynamic_reshape(&input, &[], vec![xlahlo::Dimension::Static(7)]).is_err()
    );
}

#[test]
fn conditional_dynamic_union_treats_buffers_as_leaves_not_array_subshapes() {
    use xlahlo::proto::xla_data::ProgramShapeProto;
    let scalar = Shape::scalar(P::F32).unwrap();
    let selector = Shape::scalar(P::PRED).unwrap();
    for (first, second, expected) in [
        ("b(f32[2])", "b(f32[<=2])", "b(f32[2])"),
        ("b(f32[<=2])", "b(f32[2])", "b(f32[<=2])"),
        (
            "(b(f32[2]),f32[2])",
            "(b(f32[<=2]),f32[<=2])",
            "(b(f32[2]),f32[<=2])",
        ),
    ] {
        let branches = [first, second].map(|result| ProgramShapeProto {
            parameters: vec![scalar.clone().into_proto()],
            result: Some(text::parse_shape(result).unwrap().into_proto()).into(),
            ..Default::default()
        });
        let result =
            inference::infer_conditional(&selector, &[scalar.clone(), scalar.clone()], &branches)
                .unwrap();
        assert!(result.equal_including_layout(&text::parse_shape(expected).unwrap()));
        assert!(
            inference::infer_conditional(
                &Shape::buffer(selector.clone()).unwrap(),
                &[scalar.clone(), scalar.clone()],
                &branches,
            )
            .is_err()
        );
    }
}

#[test]
fn padding_values_may_be_scalar_buffers_but_data_must_be_a_plain_array() {
    use xlahlo::proto::xla_data::{PaddingConfig, padding_config::PaddingConfigDimension};
    let data = text::parse_shape("f32[<=4]").unwrap();
    let value = text::parse_shape("b(f64[])").unwrap();
    let config = PaddingConfig {
        dimensions: vec![PaddingConfigDimension {
            edge_padding_low: 1,
            edge_padding_high: 2,
            interior_padding: 1,
            ..Default::default()
        }],
        ..Default::default()
    };
    let result = inference::infer_pad(&data, &value, &config).unwrap();
    assert!(result.equal_including_layout(&text::parse_shape("f64[<=10]").unwrap()));
    assert!(inference::infer_pad(&Shape::buffer(data.clone()).unwrap(), &value, &config).is_err());
    for value in ["b(f32[1])", "b(s32[])"] {
        assert!(inference::infer_pad(&data, &text::parse_shape(value).unwrap(), &config).is_err());
    }
}

#[test]
fn map_first_operand_and_scalar_signature_follow_distinct_buffer_guards() {
    use xlahlo::proto::xla_data::ProgramShapeProto;
    fn callee(parameters: &[&str]) -> ProgramShapeProto {
        ProgramShapeProto {
            parameters: parameters
                .iter()
                .map(|s| text::parse_shape(s).unwrap().into_proto())
                .collect(),
            result: Some(text::parse_shape("b(f64[])").unwrap().into_proto()).into(),
            ..Default::default()
        }
    }
    for (operands, parameters, result) in [
        (vec!["b(f32[<=4])"], vec!["b(f32[])"], "f64[<=4]"),
        (
            vec!["b(f32[4])", "f16[]"],
            vec!["b(f32[])", "b(f16[])"],
            "f64[4]",
        ),
        (vec!["b(f32[])", "f16[4]"], vec!["f32[]", "f16[]"], "f64[4]"),
        (vec!["f32[4]"], vec!["b(f32[])"], "f64[4]"),
    ] {
        let operands = operands
            .iter()
            .map(|s| text::parse_shape(s).unwrap())
            .collect::<Vec<_>>();
        assert!(
            inference::infer_map(&operands, &callee(&parameters), &[0])
                .unwrap()
                .equal_including_layout(&text::parse_shape(result).unwrap())
        );
    }
    for operands in [["b(f32[4])", "f32[4]"], ["f32[4]", "b(f32[])"]] {
        let operands = operands
            .iter()
            .map(|s| text::parse_shape(s).unwrap())
            .collect::<Vec<_>>();
        assert!(inference::infer_map(&operands, &callee(&["f32[]", "f32[]"]), &[0]).is_err());
    }
    let operands = [
        text::parse_shape("b(s32[])").unwrap(),
        text::parse_shape("f32[]").unwrap(),
    ];
    assert!(inference::infer_map(&operands, &callee(&["s32[]", "f32[]"]), &[]).is_err());
}

#[test]
fn matrix_decompositions_and_complex_fft_preserve_buffers_without_converting_types() {
    use xlahlo::proto::xla_data::{
        FftType, TriangularSolveOptions, triangular_solve_options::Transpose,
    };
    let a = text::parse_shape("f32[2,3,3]{2,1,0:T(2,2)}").unwrap();
    let b = text::parse_shape("f32[2,3,4]{2,1,0}").unwrap();
    let options = TriangularSolveOptions {
        left_side: true,
        transpose_a: Transpose::NO_TRANSPOSE.into(),
        ..Default::default()
    };
    for a in [a.clone(), Shape::buffer(a).unwrap()] {
        assert_eq!(inference::infer_cholesky(&a).unwrap(), a);
        for b in [b.clone(), Shape::buffer(b.clone()).unwrap()] {
            assert_eq!(
                inference::infer_triangular_solve(&a, &b, &options).unwrap(),
                b
            );
        }
    }
    for shape in ["b(f32[2,3])", "b(s32[3,3])", "b(f32[3])"] {
        assert!(inference::infer_cholesky(&text::parse_shape(shape).unwrap()).is_err());
    }
    let input = text::parse_shape("b(c64[<=2,8]{1,0:T(2,2)})").unwrap();
    for kind in [FftType::FFT, FftType::IFFT] {
        assert_eq!(inference::infer_fft(&input, kind, &[8]).unwrap(), input);
        assert!(inference::infer_fft(&input, kind, &[]).is_err());
        assert!(inference::infer_fft(&input, kind, &[2, 8, 8]).is_err());
    }
    assert!(inference::infer_fft(&input, FftType::IRFFT, &[14]).is_err());
    assert!(
        inference::infer_fft(
            &text::parse_shape("b(f32[8])").unwrap(),
            FftType::RFFT,
            &[8],
        )
        .is_err()
    );
}
