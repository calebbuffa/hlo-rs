use xlahlo::{
    Module, Opcode, VerifyOptions,
    proto::xla_data::{OpSharding, op_sharding::Type},
    text,
};

fn domain() -> xlahlo::proto::hlo::HloModuleProto {
    let mut wire = text::parse_module(
        "HloModule m ENTRY main { a = f32[2] parameter(0) ROOT r = f32[2] copy(a) }",
    )
    .unwrap()
    .to_proto()
    .unwrap();
    wire.computations[0].instructions[1].opcode = Opcode::Domain.name().into();
    wire
}

#[test]
fn domain_imports_optional_shardings_without_validating_against_its_shape() {
    let mut wire = domain();
    Module::from_proto(wire.clone())
        .unwrap()
        .verify(Default::default())
        .unwrap();
    wire.computations[0].instructions[1].domain_entry_sharding = Some(OpSharding {
        type_: Type::OTHER.into(),
        tile_assignment_dimensions: vec![2, 2],
        tile_assignment_devices: vec![0, 1, 2, 3],
        ..Default::default()
    })
    .into();
    wire.computations[0].instructions[1].domain_exit_sharding = Some(OpSharding {
        type_: Type::MAXIMAL.into(),
        tile_assignment_devices: vec![-1],
        ..Default::default()
    })
    .into();
    let module = Module::from_proto(wire.clone()).unwrap();
    module.verify(Default::default()).unwrap();
    assert_eq!(module.to_proto().unwrap(), wire);
    wire.computations[0].instructions[1]
        .domain_entry_sharding
        .as_mut()
        .unwrap()
        .tile_assignment_devices
        .pop();
    assert!(Module::from_proto(wire).is_err());
}

#[test]
fn domain_uses_unary_shape_and_optional_mixed_float_policy() {
    let mut wire = domain();
    wire.computations[0].instructions[1]
        .shape
        .as_mut()
        .unwrap()
        .element_type = xlahlo::PrimitiveType::F16.into();
    let module = Module::from_proto(wire.clone()).unwrap();
    assert!(module.verify(Default::default()).is_err());
    module
        .verify(VerifyOptions {
            allow_mixed_precision: true,
            ..Default::default()
        })
        .unwrap();
    wire.computations[0].instructions[1]
        .shape
        .as_mut()
        .unwrap()
        .dimensions[0] = 3;
    assert!(
        Module::from_proto(wire)
            .unwrap()
            .verify(VerifyOptions {
                allow_mixed_precision: true,
                ..Default::default()
            })
            .is_err()
    );
}
