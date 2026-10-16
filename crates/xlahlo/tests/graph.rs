use xlahlo::{
    Instruction, InstructionId, Module, Opcode, PrimitiveType as P, Shape, VerifyOptions, text,
};

fn example() -> (
    Module,
    xlahlo::ComputationId,
    InstructionId,
    InstructionId,
    InstructionId,
) {
    let mut module = Module::new("example");
    let computation = module.add_computation("main").unwrap();
    let shape = Shape::array(P::F32, vec![2]).unwrap();
    let a = module
        .add_instruction(
            computation,
            Instruction::new("a", Opcode::Parameter, shape.clone()),
            &[],
            &[],
        )
        .unwrap();
    let b = module
        .add_instruction(
            computation,
            Instruction::new("b", Opcode::Parameter, shape.clone())
                .with_fields(|i| i.parameter_number = 1)
                .unwrap(),
            &[],
            &[],
        )
        .unwrap();
    let add = module
        .add_instruction(
            computation,
            Instruction::new("add", Opcode::Add, shape),
            &[a, b],
            &[],
        )
        .unwrap();
    module.set_root(computation, add).unwrap();
    module.set_entry(computation).unwrap();
    module.verify(Default::default()).unwrap();
    (module, computation, a, b, add)
}

#[test]
fn binary_metadata_survives_graph_import_edits_forks_and_export() {
    let (module, _, _, _, _) = example();
    let mut wire = module.to_proto().unwrap();
    wire.pjrt_id = Some(0);
    wire.backend_config = vec![0, 255, 1];
    wire.special_fields.mut_unknown_fields().add_varint(900, 42);
    let computation = &mut wire.computations[0];
    computation.backend_config = vec![255, 0];
    computation
        .special_fields
        .mut_unknown_fields()
        .add_varint(901, 43);
    let root = computation
        .instructions
        .iter_mut()
        .find(|i| i.opcode == "add")
        .unwrap();
    root.backend_config = vec![0, 1, 255];
    root.special_fields.mut_unknown_fields().add_varint(902, 44);
    root.shape
        .as_mut()
        .unwrap()
        .special_fields
        .mut_unknown_fields()
        .add_varint(903, 45);
    let mut metadata = xlahlo::proto::xla_data::OpMetadata::new();
    metadata.op_name = "original_add".into();
    metadata.source_file = "origin.rs".into();
    metadata.source_line = 17;
    root.metadata = Some(metadata).into();
    let original_root = root.clone();
    let original_computation = computation.clone();
    let original_module = wire.clone();
    let mut imported = Module::from_proto(wire).unwrap();
    let root = imported.root(imported.entry().unwrap()).unwrap();
    imported
        .edit_verified(Default::default(), |staged| {
            staged.edit_instruction_fields(root, |i| {
                i.opcode = "multiply".into();
                Ok(())
            })
        })
        .unwrap();
    let encoded =
        xlahlo::interchange::encode(&imported.fork().unwrap().to_proto().unwrap()).unwrap();
    let decoded: xlahlo::proto::hlo::HloModuleProto =
        xlahlo::interchange::decode(&encoded, Default::default()).unwrap();
    assert_eq!(decoded.pjrt_id, Some(0));
    assert_eq!(decoded.backend_config, original_module.backend_config);
    assert_eq!(
        decoded.special_fields.unknown_fields(),
        original_module.special_fields.unknown_fields()
    );
    let computation = &decoded.computations[0];
    assert_eq!(
        computation.backend_config,
        original_computation.backend_config
    );
    assert_eq!(
        computation.special_fields.unknown_fields(),
        original_computation.special_fields.unknown_fields()
    );
    let root = computation
        .instructions
        .iter()
        .find(|i| i.opcode == "multiply")
        .unwrap();
    assert_eq!(root.backend_config, original_root.backend_config);
    assert_eq!(root.metadata, original_root.metadata);
    assert_eq!(root.shape, original_root.shape);
    assert_eq!(
        root.special_fields.unknown_fields(),
        original_root.special_fields.unknown_fields()
    );
}

#[test]
fn public_in_memory_construction_verifies_and_exports() {
    let (module, computation, a, b, add) = example();
    assert_eq!(module.entry().unwrap(), computation);
    assert_eq!(module.root(computation).unwrap(), add);
    assert_eq!(module.operands(add).unwrap(), [a, b]);
    assert_eq!(module.users(a).unwrap(), [add]);
    let printed = text::print_module(&module).unwrap();
    let parsed = text::parse_module_with_options(
        &printed,
        text::ParseOptions {
            fill_missing_layouts: false,
            ..Default::default()
        },
    )
    .unwrap();
    parsed.verify(VerifyOptions::default()).unwrap();
    assert_eq!(text::print_module(&parsed).unwrap(), printed);
}

#[test]
fn operand_repetition_and_failed_cycles_preserve_graph() {
    let (mut module, _, a, b, add) = example();
    module.replace_operand(add, 1, a).unwrap();
    assert_eq!(module.users(a).unwrap(), [add]);
    assert!(module.users(b).unwrap().is_empty());
    let before = module.to_proto().unwrap();
    let error = module.replace_operand(add, 0, add).unwrap_err();
    assert_eq!(error.category, "cycle");
    assert_eq!(module.to_proto().unwrap(), before);
    module.replace_all_uses(a, b).unwrap();
    assert_eq!(module.operands(add).unwrap(), [b, b]);
    assert!(module.users(a).unwrap().is_empty());
}

#[test]
fn handles_are_owner_scoped_and_deletions_do_not_reuse_identity() {
    let (mut module, computation, _, _, _) = example();
    let fork = module.fork().unwrap();
    assert!(fork.computation(computation).is_err());
    let spare = module
        .add_instruction(
            computation,
            Instruction::new("spare", Opcode::Constant, Shape::scalar(P::F32).unwrap()),
            &[],
            &[],
        )
        .unwrap();
    module.remove_instruction(spare).unwrap();
    assert!(module.instruction(spare).is_err());
    let next = module
        .add_instruction(
            computation,
            Instruction::new("next", Opcode::Constant, Shape::scalar(P::F32).unwrap()),
            &[],
            &[],
        )
        .unwrap();
    assert_ne!(spare, next);
}

#[test]
fn coordinated_failure_rolls_back_and_escaped_handles_stay_invalid() {
    let (mut module, computation, a, _, add) = example();
    let before = module.to_proto().unwrap();
    let mut escaped = None;
    let error = module
        .edit(|staged| {
            escaped = Some(staged.add_instruction(
                computation,
                Instruction::new(
                    "escaped",
                    Opcode::Negate,
                    Shape::array(P::F32, vec![2]).unwrap(),
                ),
                &[a],
                &[],
            )?);
            staged.add_control_dependency(add, a)
        })
        .unwrap_err();
    assert_eq!(error.category, "cycle");
    assert_eq!(module.to_proto().unwrap(), before);
    let escaped = escaped.unwrap();
    assert!(module.instruction(escaped).is_err());
    let later = module
        .add_instruction(
            computation,
            Instruction::new(
                "later",
                Opcode::Negate,
                Shape::array(P::F32, vec![2]).unwrap(),
            ),
            &[a],
            &[],
        )
        .unwrap();
    assert_ne!(escaped, later);
}

#[test]
fn coordinated_shape_rewrite_does_not_require_valid_intermediate_semantics() {
    let (mut module, _, a, b, add) = example();
    module
        .edit(|staged| {
            let shape = Shape::array(P::F64, vec![3]).unwrap();
            staged.set_shape(a, shape.clone())?;
            staged.set_shape(b, shape.clone())?;
            staged.set_shape(add, shape)?;
            staged.verify(Default::default())
        })
        .unwrap();
    assert_eq!(module.shape(add).unwrap().element_type(), P::F64);
}

#[test]
fn domain_passthrough_is_verified() {
    let (mut module, computation, a, _, _) = example();
    let domain = module
        .add_instruction(
            computation,
            Instruction::new("domain", Opcode::Domain, module.shape(a).unwrap()),
            &[a],
            &[],
        )
        .unwrap();
    module.set_root(computation, domain).unwrap();
    module.verify(Default::default()).unwrap();
}

#[test]
fn names_arities_roots_and_dangling_edges_are_checked() {
    let (mut module, computation, a, _, add) = example();
    assert!(
        module
            .add_instruction(
                computation,
                Instruction::new("arity", Opcode::Add, module.shape(a).unwrap()),
                &[a],
                &[]
            )
            .is_err()
    );
    assert!(module.add_computation("main").is_err());
    assert!(module.remove_instruction(add).is_err());
    let mut wire = module.to_proto().unwrap();
    wire.computations[0]
        .instructions
        .last_mut()
        .unwrap()
        .operand_ids
        .push(10000);
    assert_eq!(Module::from_proto(wire).unwrap_err().category, "reference");
}
