use pliron::{
    basic_block::BasicBlock,
    builtin::attributes::{BoolAttr, StringAttr, VecAttr},
    common_traits::Verify,
    context::Context,
    identifier::Identifier,
    op::Op,
    printable::Printable,
};
use stablehlo::{
    attributes::{ArrayOfLayoutAttr, LayoutAttr, OutputOperandAliasAttr},
    ops::CustomCallOp,
    types::{BufferType, ElementType, TokenType, TupleType, UnrankedTensorType, tensor},
};

fn layouts(values: &[&[i64]]) -> ArrayOfLayoutAttr {
    ArrayOfLayoutAttr {
        layouts: values
            .iter()
            .map(|values| LayoutAttr {
                minor_to_major: values.to_vec(),
            })
            .collect(),
    }
}

#[test]
fn layout_presence_is_independent_of_operand_and_result_count() {
    let mut ctx = Context::new();
    let ty = tensor(&ctx, &[2, 3], ElementType::F32);
    for input_count in 0..=1 {
        for output_count in 0..=1 {
            let block = BasicBlock::new(&mut ctx, None, vec![ty; input_count]);
            let operands = block.deref(&ctx).arguments().collect();
            let call = CustomCallOp::new(
                &mut ctx,
                "external".into(),
                operands,
                vec![ty; output_count],
                1,
            );
            assert!(
                call.get_attr_stablehlo_custom_call_operand_layouts(&ctx)
                    .is_none()
            );
            assert!(
                call.get_attr_stablehlo_custom_call_result_layouts(&ctx)
                    .is_none()
            );
            call.verify(&ctx).unwrap();
            call.set_attr_stablehlo_custom_call_operand_layouts(
                &ctx,
                layouts(&vec![&[0, 1][..]; input_count]),
            );
            assert!(call.verify(&ctx).is_err());
            call.set_attr_stablehlo_custom_call_result_layouts(
                &ctx,
                layouts(&vec![&[1, 0][..]; output_count]),
            );
            call.verify(&ctx).unwrap();
            call.set_attr_stablehlo_custom_call_operand_layouts(&ctx, layouts(&[]));
            call.set_attr_stablehlo_custom_call_result_layouts(&ctx, layouts(&[]));
            assert_eq!(
                call.verify(&ctx).is_ok(),
                input_count == 0 && output_count == 0,
            );
        }
    }
}

#[test]
fn constrained_tokens_buffers_and_unranked_tensors_follow_source_layout_rules() {
    let mut ctx = Context::new();
    let token = TokenType::get(&ctx).into();
    let buffer = BufferType::without_layout(&ctx, ElementType::F32.native(&ctx), vec![2, 3]);
    let unranked = UnrankedTensorType::get(&ctx, ElementType::F32.native(&ctx)).into();
    for (ty, layout) in [
        (token, vec![]),
        (buffer, vec![1, 0]),
        (unranked, vec![8, 8, 2]),
    ] {
        let block = BasicBlock::new(&mut ctx, None, vec![ty]);
        let input = block.deref(&ctx).get_argument(0);
        let call = CustomCallOp::new(&mut ctx, "external".into(), vec![input], vec![ty], 1);
        if ty == buffer {
            call.set_attr_stablehlo_custom_call_output_operand_aliases(
                &ctx,
                VecAttr::new(vec![
                    OutputOperandAliasAttr {
                        output_tuple_indices: vec![],
                        operand_index: 0,
                        operand_tuple_indices: vec![],
                    }
                    .into(),
                ]),
            );
        }
        call.set_attr_stablehlo_custom_call_operand_layouts(&ctx, layouts(&[&layout]));
        call.set_attr_stablehlo_custom_call_result_layouts(&ctx, layouts(&[&layout]));
        call.verify(&ctx).unwrap();
        if ty == token {
            call.set_attr_stablehlo_custom_call_result_layouts(&ctx, layouts(&[&[0]]));
            assert!(call.verify(&ctx).is_err());
        }
    }
}

#[test]
fn malformed_layouts_fail_without_mutation_and_can_be_repaired() {
    let mut ctx = Context::new();
    let ty = tensor(&ctx, &[2, 3], ElementType::F32);
    let call = CustomCallOp::new(&mut ctx, "external".into(), vec![], vec![ty], 1);
    call.set_attr_stablehlo_custom_call_operand_layouts(&ctx, layouts(&[]));
    for layout in [&[][..], &[0, 0], &[0, 2], &[-1, 0], &[0, 1, 2]] {
        call.set_attr_stablehlo_custom_call_result_layouts(&ctx, layouts(&[layout]));
        let before = call.get_operation().disp(&ctx).to_string();
        for _ in 0..2 {
            assert!(call.verify(&ctx).is_err());
            assert_eq!(call.get_operation().disp(&ctx).to_string(), before);
        }
    }
    for key in [
        stablehlo::attributes::keys::CUSTOM_CALL_OPERAND_LAYOUTS,
        stablehlo::attributes::keys::CUSTOM_CALL_RESULT_LAYOUTS,
    ] {
        call.set_attr_stablehlo_custom_call_operand_layouts(&ctx, layouts(&[]));
        call.set_attr_stablehlo_custom_call_result_layouts(&ctx, layouts(&[&[1, 0]]));
        call.get_operation()
            .deref_mut(&ctx)
            .attributes
            .set(Identifier::new(key), StringAttr::new("wrong".into()));
        assert!(call.verify(&ctx).is_err());
    }
    call.set_attr_stablehlo_custom_call_operand_layouts(&ctx, layouts(&[]));
    call.set_attr_stablehlo_custom_call_result_layouts(&ctx, layouts(&[&[1, 0]]));
    call.verify(&ctx).unwrap();
}

#[test]
fn singleton_tuple_result_layouts_apply_to_immediate_children_only() {
    let mut ctx = Context::new();
    let array = tensor(&ctx, &[2, 3], ElementType::F32);
    let token = TokenType::get(&ctx).into();
    let tuple = TupleType::get(&ctx, vec![array, token]).into();
    let call = CustomCallOp::new(&mut ctx, "external".into(), vec![], vec![tuple], 1);
    call.set_attr_stablehlo_custom_call_operand_layouts(&ctx, layouts(&[]));
    call.set_attr_stablehlo_custom_call_result_layouts(&ctx, layouts(&[&[1, 0], &[]]));
    call.verify(&ctx).unwrap();
    let nested = TupleType::get(&ctx, vec![tuple]).into();
    let call = CustomCallOp::new(&mut ctx, "external".into(), vec![], vec![nested], 1);
    call.set_attr_stablehlo_custom_call_operand_layouts(&ctx, layouts(&[]));
    call.set_attr_stablehlo_custom_call_result_layouts(&ctx, layouts(&[&[]]));
    assert!(call.verify(&ctx).is_err());
}

#[test]
fn malformed_effect_and_alias_metadata_never_disappears_or_indexes_out_of_bounds() {
    let mut ctx = Context::new();
    let ty = tensor(&ctx, &[2], ElementType::F32);
    let block = BasicBlock::new(&mut ctx, None, vec![ty]);
    let input = block.deref(&ctx).get_argument(0);
    let call = CustomCallOp::new(&mut ctx, "external".into(), vec![input], vec![ty], 1);
    for operand_index in [-1, 1, i64::MAX] {
        call.set_attr_stablehlo_custom_call_output_operand_aliases(
            &ctx,
            VecAttr::new(vec![Box::new(OutputOperandAliasAttr {
                output_tuple_indices: vec![],
                operand_index,
                operand_tuple_indices: vec![],
            })]),
        );
        assert!(call.verify(&ctx).is_err());
    }
    call.set_attr_stablehlo_custom_call_output_operand_aliases(&ctx, VecAttr::new(vec![]));
    for key in [
        stablehlo::attributes::keys::CUSTOM_CALL_HAS_SIDE_EFFECT,
        stablehlo::attributes::keys::CUSTOM_CALL_OUTPUT_OPERAND_ALIASES,
    ] {
        call.get_operation()
            .deref_mut(&ctx)
            .attributes
            .set(Identifier::new(key), StringAttr::new("wrong".into()));
        let before = call.get_operation().disp(&ctx).to_string();
        for _ in 0..2 {
            assert!(call.verify(&ctx).is_err());
            assert_eq!(call.get_operation().disp(&ctx).to_string(), before);
        }
        call.set_attr_stablehlo_custom_call_has_side_effect(&ctx, BoolAttr::new(false));
        call.set_attr_stablehlo_custom_call_output_operand_aliases(&ctx, VecAttr::new(vec![]));
        call.verify(&ctx).unwrap();
    }
    call.get_operation()
        .deref_mut(&ctx)
        .attributes
        .0
        .remove(&Identifier::new(
            stablehlo::attributes::keys::CUSTOM_CALL_HAS_SIDE_EFFECT,
        ));
    call.get_operation()
        .deref_mut(&ctx)
        .attributes
        .0
        .remove(&Identifier::new(
            stablehlo::attributes::keys::CUSTOM_CALL_OUTPUT_OPERAND_ALIASES,
        ));
    call.verify(&ctx).unwrap();
}

#[test]
fn omitted_api_version_defaults_to_original_without_hiding_malformed_metadata() {
    use pliron::{attribute::AttrObj, builtin::attributes::DictAttr};
    let mut ctx = Context::new();
    let call = CustomCallOp::new(&mut ctx, "external".into(), vec![], vec![], 1);
    let key = Identifier::new(stablehlo::attributes::keys::CUSTOM_CALL_API_VERSION);
    call.get_operation()
        .deref_mut(&ctx)
        .attributes
        .0
        .remove(&key);
    assert_eq!(call.api_version(&ctx).unwrap(), 1);
    call.verify(&ctx).unwrap();
    call.set_backend_config(&mut ctx, Box::new(StringAttr::new("legacy".into())));
    call.verify(&ctx).unwrap();
    call.set_backend_config(&mut ctx, Box::new(DictAttr::new(vec![])));
    assert!(call.verify(&ctx).is_err());
    for value in 0..=5 {
        let other = CustomCallOp::new(&mut ctx, "external".into(), vec![], vec![], value);
        if value <= 4 {
            assert_eq!(other.api_version(&ctx).unwrap(), value);
            other.verify(&ctx).unwrap();
        } else {
            assert!(other.api_version(&ctx).is_err());
            assert!(other.verify(&ctx).is_err());
        }
    }
    let malformed: Vec<AttrObj> = vec![
        Box::new(StringAttr::new("1".into())),
        Box::new(BoolAttr::new(true)),
    ];
    for attribute in malformed {
        call.get_operation()
            .deref_mut(&ctx)
            .attributes
            .0
            .insert(key.clone(), attribute);
        let before = call.get_operation().disp(&ctx).to_string();
        for _ in 0..2 {
            assert!(call.api_version(&ctx).is_err());
            assert!(call.verify(&ctx).is_err());
            assert_eq!(call.get_operation().disp(&ctx).to_string(), before);
        }
    }
    call.get_operation()
        .deref_mut(&ctx)
        .attributes
        .0
        .remove(&key);
    call.set_backend_config(&mut ctx, Box::new(StringAttr::new("repaired".into())));
    call.verify(&ctx).unwrap();
}

#[test]
fn repeated_alias_entries_preserve_the_ordered_source_array() {
    let mut ctx = Context::new();
    let ty = tensor(&ctx, &[2], ElementType::F32);
    let block = BasicBlock::new(&mut ctx, None, vec![ty, ty]);
    let inputs = block.deref(&ctx).arguments().collect();
    let call = CustomCallOp::new(&mut ctx, "external".into(), inputs, vec![ty], 1);
    call.set_attr_stablehlo_custom_call_output_operand_aliases(
        &ctx,
        VecAttr::new(
            [1, 0, 1]
                .into_iter()
                .map(|operand_index| {
                    Box::new(OutputOperandAliasAttr {
                        output_tuple_indices: vec![],
                        operand_index,
                        operand_tuple_indices: vec![],
                    }) as pliron::attribute::AttrObj
                })
                .collect(),
        ),
    );
    call.verify(&ctx).unwrap();
    let aliases = call
        .get_attr_stablehlo_custom_call_output_operand_aliases(&ctx)
        .unwrap();
    assert_eq!(
        aliases
            .0
            .iter()
            .map(|alias| alias
                .downcast_ref::<OutputOperandAliasAttr>()
                .unwrap()
                .operand_index)
            .collect::<Vec<_>>(),
        [1, 0, 1],
    );
}

#[test]
fn target_names_are_opaque_strings_not_backend_validity_filters() {
    let mut ctx = Context::new();
    for target in ["", "ordinary", "quoted \"target\"\n\0", "domain::target"] {
        let call = CustomCallOp::new(&mut ctx, target.into(), vec![], vec![], 1);
        call.verify(&ctx).unwrap();
        assert_eq!(
            call.get_attr_stablehlo_custom_call_target(&ctx)
                .unwrap()
                .as_str(),
            target,
        );
        call.get_operation()
            .deref_mut(&ctx)
            .attributes
            .0
            .remove(&Identifier::new(
                stablehlo::attributes::keys::CUSTOM_CALL_TARGET,
            ));
        assert!(call.verify(&ctx).is_err());
        call.get_operation().deref_mut(&ctx).attributes.set(
            Identifier::new(stablehlo::attributes::keys::CUSTOM_CALL_TARGET),
            BoolAttr::new(false),
        );
        assert!(call.verify(&ctx).is_err());
    }
}
