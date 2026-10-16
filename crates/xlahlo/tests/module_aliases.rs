use xlahlo::{Module, Shape, VerifyOptions, text};

fn strict() -> VerifyOptions {
    VerifyOptions {
        layout_sensitive: true,
        ..Default::default()
    }
}

#[test]
fn module_aliases_check_sizes_not_types_and_only_in_layout_sensitive_mode() {
    let source = "HloModule m, input_output_alias={{}: (0,{})} ENTRY main { p = f32[2] parameter(0) ROOT r = s32[2] convert(p) }";
    let module = text::parse_module(source).unwrap();
    module.verify(strict()).unwrap();
    let config = module.as_proto().input_output_alias.as_ref().unwrap();
    assert_eq!(
        config.entries[0].kind.enum_value().unwrap(),
        xlahlo::proto::hlo::Kind::MAY_ALIAS
    );
    let mut wire = module.to_proto().unwrap();
    wire.computations[0]
        .instructions
        .iter_mut()
        .find(|i| i.opcode == "convert")
        .unwrap()
        .shape
        .as_mut()
        .unwrap()
        .element_type = xlahlo::PrimitiveType::S64.into();
    let module = Module::from_proto(wire).unwrap();
    module.verify(Default::default()).unwrap();
    assert_eq!(module.verify(strict()).unwrap_err().category, "alias");
    fn target_size(_: &Shape) -> xlahlo::Result<i64> {
        Ok(16)
    }
    module
        .verify(VerifyOptions {
            shape_size: Some(target_size),
            ..strict()
        })
        .unwrap();
}

#[test]
fn nested_module_alias_and_donor_indices_round_trip() {
    let source = "HloModule m, input_output_alias={{0}: (0,{0},must-alias)}, buffer_donor={(0,{1})} ENTRY main { p = (f32[2],s32[]) parameter(0) ROOT r = (f32[2],s32[]) opt-barrier(p) }";
    let module = text::parse_module(source).unwrap();
    module.verify(strict()).unwrap();
    let printed = text::print_module(&module).unwrap();
    let reparsed = text::parse_module(&printed).unwrap();
    reparsed.verify(strict()).unwrap();
    assert_eq!(
        reparsed.as_proto().input_output_alias,
        module.as_proto().input_output_alias
    );
    assert_eq!(
        reparsed.as_proto().buffer_donor,
        module.as_proto().buffer_donor
    );
    for source in [
        source.replace("{0}: (0,{0}", "{9}: (0,{0}"),
        source.replace("{0}: (0,{0}", "{0}: (1,{0}"),
        source.replace("{0}: (0,{0}", "{0}: (0,{-1}"),
        source.replace("buffer_donor={(0,{1})}", "buffer_donor={(0,{0})}"),
        source.replace("buffer_donor={(0,{1})}", "buffer_donor={(0,{})}"),
    ] {
        assert!(
            text::parse_module(&source)
                .unwrap()
                .verify(Default::default())
                .is_err(),
            "{source}"
        );
    }
}

#[test]
fn aliases_cannot_reuse_an_input_or_alias_tuple_buffers() {
    let source = "HloModule m, input_output_alias={{0}: (0,{}),{1}: (0,{})} ENTRY main { p = f32[2] parameter(0) ROOT r = (f32[2],f32[2]) tuple(p,p) }";
    assert_eq!(
        text::parse_module(source)
            .unwrap()
            .verify(Default::default())
            .unwrap_err()
            .category,
        "alias"
    );
    let source = "HloModule m, input_output_alias={{}: (0,{})} ENTRY main { ROOT p = (f32[2]) parameter(0) }";
    assert!(
        text::parse_module(source)
            .unwrap()
            .verify(Default::default())
            .is_err()
    );
    let source =
        "HloModule m, input_output_alias={{}: (0,{})} ENTRY main { ROOT p = f32[2] parameter(0) }";
    let mut wire = text::parse_module(source).unwrap().to_proto().unwrap();
    let duplicate = wire.input_output_alias.as_ref().unwrap().entries[0].clone();
    wire.input_output_alias
        .as_mut()
        .unwrap()
        .entries
        .push(duplicate);
    assert!(
        Module::from_proto(wire)
            .unwrap()
            .verify(Default::default())
            .is_err()
    );
}

#[test]
fn donor_set_semantics_and_transactional_metadata_updates_are_explicit() {
    let source =
        "HloModule m, buffer_donor={(0,{}),(0,{})} ENTRY main { ROOT p = f32[2] parameter(0) }";
    let mut module = text::parse_module(source).unwrap();
    assert_eq!(
        module
            .as_proto()
            .buffer_donor
            .as_ref()
            .unwrap()
            .entries
            .len(),
        1
    );
    module.verify(strict()).unwrap();
    let donor = module.as_proto().buffer_donor.as_ref().unwrap().clone();
    let alias = text::parse_module(
        "HloModule m, input_output_alias={{}: (0,{})} ENTRY main { ROOT p = f32[2] parameter(0) }",
    )
    .unwrap()
    .as_proto()
    .input_output_alias
    .as_ref()
    .unwrap()
    .clone();
    let before = module.to_proto().unwrap();
    assert!(
        module
            .edit_verified(strict(), |m| m
                .set_input_output_aliases(Some(alias.clone())))
            .is_err()
    );
    assert_eq!(before, module.to_proto().unwrap());
    module
        .edit_verified(strict(), |m| {
            m.set_buffer_donors(None)?;
            m.set_input_output_aliases(Some(alias))
        })
        .unwrap();
    assert!(
        module
            .edit_verified(strict(), |m| m.set_buffer_donors(Some(donor)))
            .is_err()
    );
    module.verify(strict()).unwrap();
}

#[test]
fn prefetch_metadata_has_no_extra_default_verifier_constraints() {
    let mut wire = text::parse_module("HloModule m ENTRY main { ROOT p = f32[2] parameter(0) }")
        .unwrap()
        .to_proto()
        .unwrap();
    wire.cross_program_prefetches
        .push(xlahlo::proto::hlo::CrossProgramPrefetch {
            parameter: 999,
            index: vec![-1],
            offset: -4,
            ..Default::default()
        });
    let module = Module::from_proto(wire).unwrap();
    module.verify(strict()).unwrap();
    assert_eq!(
        text::print_module(&module).unwrap_err().category,
        "unsupported"
    );
}
