use xlahlo::{Sharding, text};

#[test]
fn legacy_assembly_round_trips_leaf_modes_flat_tuples_subgroups_and_metadata() {
    for source in [
        "{}",
        "{replicated}",
        "{manual}",
        "{unknown}",
        "{unreduced}",
        "{maximal device=7 shard_as -3}",
        "{{replicated},{maximal device=1},{manual}}",
        "{devices=[2,2]0,1,3,2 last_tile_dim_replicate shard_like 7}",
        "{devices=[2,2,2]<=[8] last_tile_dims={manual,replicated}}",
        "{devices=[2,3]<=[3,2]T(1,0)}",
        "{replicated metadata={op_type=\"sdy::reduction_op\" op_name=\"MAX\"}}",
        "{replicated metadata={{op_name=\"left\"},{op_name=\"right\" source_file=\"a.rs\" source_line=3}}}",
    ] {
        let original =
            text::parse_sharding(source).unwrap_or_else(|error| panic!("{source}: {error}"));
        let printed =
            text::print_sharding(&original).unwrap_or_else(|error| panic!("{source}: {error}"));
        let reparsed =
            text::parse_sharding(&printed).unwrap_or_else(|error| panic!("{printed}: {error}"));
        assert!(
            original.equal_semantics(&reparsed).unwrap(),
            "{source} -> {printed}"
        );
    }
}

#[test]
fn named_assembly_round_trips_meshes_axes_open_dimensions_and_reductions() {
    for source in [
        "{single_device[device_id=7]}",
        "{maximal_mesh[device_id=3], metadata={op_name=\"one\"}}",
        "{mesh[], replicated}",
        "{mesh['x'=4,'y'=2], replicated}",
        "{mesh['x'=4,'y'=2], manual}",
        "{mesh['x'=4,'y'=2], unreduced=max}",
        "{mesh['x'=4,'y'=2], [{'x'},{'y',?}]}",
        "{mesh['x'=8], [{'x':(1)2}], replicated={'x':(2)4}}",
        "{mesh['x'=8], [{'x':(1)2}], unreduced=min{'x':(2)4}}",
        "{mesh['x'=8], [{'0':(1)2}], manual={'0':(2)4}}",
        "{mesh['x'=2,'y'=3], device_ids=([3,2]T(1,0)), [{'x'}], replicated={'y'}}",
        "{mesh['x'=2,'y'=2], device_ids=(3,2,1,0), [{'y'},{'x'}]}",
    ] {
        let original =
            text::parse_sharding(source).unwrap_or_else(|error| panic!("{source}: {error}"));
        let printed =
            text::print_sharding(&original).unwrap_or_else(|error| panic!("{source}: {error}"));
        let reparsed =
            text::parse_sharding(&printed).unwrap_or_else(|error| panic!("{printed}: {error}"));
        assert!(
            original.equal_semantics(&reparsed).unwrap(),
            "{source} -> {printed}"
        );
    }
}

#[test]
fn malformed_sharding_assembly_is_rejected() {
    for source in [
        "{replicated device=1}",
        "{manual device=1}",
        "{maximal}",
        "{device=1}",
        "{devices=[1]0}",
        "{devices=[]0,1}",
        "{devices=[2]0,1,2}",
        "{devices=[2]<=[1,2]}",
        "{devices=[2]<=[1,2]T(0,0)}",
        "{devices=[2,2]<=[4] last_tile_dim_replicate last_tile_dims={manual}}",
        "{{}}",
        "{{{replicated}}}",
        "{replicated} trailing",
        "{mesh['x'=2,'x'=2], replicated}",
        "{mesh['x'=2], [{'missing'}]}",
        "{mesh['x'=2], [{'x','x'}]}",
        "{mesh['x'=4], [{'x':(1)4}]}",
        "{mesh['x'=4], [{?,'x'}]}",
        "{mesh['x'=2], device_ids=(0,1), replicated}",
        "{replicated metadata={op_name=\"a\" op_name=\"b\"}}",
        "{replicated metadata={profile_type={999}}}",
    ] {
        assert!(text::parse_sharding(source).is_err(), "{source}");
    }
}

#[test]
fn domain_and_instruction_shardings_survive_module_assembly() {
    let source = "HloModule m ENTRY main { a = f32[2] parameter(0), sharding={devices=[2]0,1} ROOT r = f32[2] domain(a), domain={kind=\"sharding\", entry={manual}, exit={mesh['x'=2], [{'x'}]}} }";
    let module = text::parse_module(source).unwrap();
    module.verify(Default::default()).unwrap();
    let domain = module.as_proto().computations[0]
        .instructions
        .iter()
        .find(|i| i.opcode == "domain")
        .unwrap();
    assert!(
        domain
            .domain_entry_sharding
            .as_ref()
            .unwrap()
            .named_sharding
            .is_some()
    );
    assert_eq!(
        domain
            .domain_exit_sharding
            .as_ref()
            .unwrap()
            .type_
            .enum_value()
            .unwrap(),
        xlahlo::proto::xla_data::op_sharding::Type::MANUAL
    );
    let printed = text::print_module(&module).unwrap();
    let reparsed = text::parse_module(&printed).unwrap();
    reparsed.verify(Default::default()).unwrap();
    assert!(printed.contains("domain={kind=\"sharding\""));
    for source in [
        "HloModule m ENTRY main { a = f32[] parameter(0) ROOT d = f32[] domain(a) }",
        "HloModule m ENTRY main { a = f32[] parameter(0) ROOT d = f32[] domain(a), domain={kind=\"foo\",entry={manual},exit={manual}} }",
        "HloModule m ENTRY main { a = f32[] parameter(0) ROOT d = f32[] domain(a), domain={kind=\"sharding\",entry={manual}} }",
    ] {
        assert!(text::parse_module(source).is_err());
    }
}

#[test]
fn unrepresentable_and_unknown_sharding_wire_fields_are_not_silently_printed() {
    let mut wire = text::parse_sharding("{replicated}").unwrap().into_proto();
    wire.special_fields.mut_unknown_fields().add_varint(7000, 1);
    assert!(text::print_sharding(&Sharding::from_proto(wire).unwrap()).is_err());
    let mut wire = text::parse_sharding("{replicated}").unwrap().into_proto();
    wire.tile_assignment_devices = vec![9];
    assert!(text::print_sharding(&Sharding::from_proto(wire).unwrap()).is_err());
}
