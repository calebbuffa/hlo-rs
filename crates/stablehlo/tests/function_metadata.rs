use stablehlo::{
    Attribute, Attributes, AxisRef, BuildError, Computation, ElementType, Instruction, Mesh,
    MeshAxis, MeshReference, Module, Operation, ReplicaGroupMeshAxes, StableHloAttribute,
    StableHloOp, TensorType, Type, ValueId,
};

mod common;
use common::computation;

fn function(name: &str) -> Computation {
    Computation::try_new(
        name,
        vec![Instruction::new(
            Operation::parameter(0, "input"),
            vec![],
            vec![],
            ValueId(0),
            TensorType::scalar(ElementType::F32).into(),
        )],
        ValueId(0),
    )
    .unwrap()
}

fn metadata() -> Attributes {
    [
        ("sym_visibility".into(), Attribute::String("private".into())),
        (
            "arg_attrs".into(),
            Attribute::Array(vec![Attribute::Dictionary(
                [("consumer.name".into(), Attribute::String("input".into()))].into(),
            )]),
        ),
        (
            "res_attrs".into(),
            Attribute::Array(vec![Attribute::Dictionary(Attributes::new())]),
        ),
    ]
    .into()
}

#[test]
fn function_attributes_preserve_visibility_and_signature_metadata() {
    for visibility in ["public", "private", "nested"] {
        let mut function = function("function");
        let mut attributes = metadata();
        attributes.insert(
            "sym_visibility".into(),
            Attribute::String(visibility.into()),
        );
        function.set_attributes(attributes.clone()).unwrap();
        assert_eq!(function.attributes(), &attributes);
        let mut module = Module::new("module");
        module.add_computation(function);
        assert!(module.verify().is_ok());
    }
}

#[test]
fn invalid_function_metadata_is_rejected_atomically() {
    let mut function = function("function");
    function.set_attributes(metadata()).unwrap();
    for (name, value) in [
        ("sym_name", Attribute::String("different".into())),
        ("function_type", Attribute::Type(Box::new(Type::Token))),
        ("sym_visibility", Attribute::String("internal".into())),
        ("sym_visibility", Attribute::Bool(true)),
        ("arg_attrs", Attribute::Array(vec![])),
        ("arg_attrs", Attribute::Array(vec![Attribute::Bool(true)])),
        ("res_attrs", Attribute::Dictionary(Attributes::new())),
        ("res_attrs", Attribute::Array(vec![])),
    ] {
        let mut attributes = metadata();
        attributes.insert(name.into(), value);
        assert!(matches!(
            function.set_attributes(attributes),
            Err(BuildError::InvalidFunctionAttribute(_))
        ));
        assert_eq!(function.attributes(), &metadata());
    }
}

#[test]
fn body_replacement_preserves_valid_function_signature_metadata() {
    let mut function = function("function");
    function.set_attributes(metadata()).unwrap();
    let original = function.clone();
    let instructions = [ElementType::F32, ElementType::F32]
        .into_iter()
        .enumerate()
        .map(|(index, element)| {
            Instruction::new(
                Operation::parameter(index as u32, format!("arg{index}")),
                vec![],
                vec![],
                ValueId(index as u32),
                TensorType::scalar(element).into(),
            )
        })
        .collect();
    assert!(function.replace_body(instructions, ValueId(1)).is_err());
    assert_eq!(function, original);
}

#[test]
fn module_attributes_validate_recursively_and_cannot_override_the_module_name() {
    let mut module = Module::new("module");
    let attributes: Attributes = [("stablehlo.num_replicas".into(), Attribute::integer(4))].into();
    module.set_attributes(attributes.clone()).unwrap();
    assert_eq!(module.attributes(), &attributes);
    assert!(module.verify().is_ok());
    assert!(
        module
            .set_attributes([("sym_name".into(), Attribute::String("different".into()))].into())
            .is_err()
    );
    assert_eq!(module.attributes(), &attributes);
}

#[test]
fn mesh_resolution_applies_inside_capturing_control_flow_regions() {
    let input: Type = TensorType::static_shape(ElementType::F32, [3]).into();
    let output: Type = TensorType::static_shape(ElementType::F32, [6]).into();
    let predicate: Type = TensorType::scalar(ElementType::I1).into();
    let gather = gather().instructions().last().unwrap().operation().clone();
    let branches = [100, 101]
        .into_iter()
        .map(|id| {
            stablehlo::Region::new(vec![stablehlo::Block::new(
                vec![],
                vec![
                    Instruction::new(
                        gather.clone(),
                        vec![ValueId(1)],
                        vec![],
                        ValueId(id),
                        output.clone(),
                    ),
                    Instruction::with_results(
                        Operation::stablehlo(StableHloOp::Return, Attributes::new()),
                        vec![ValueId(id)],
                        vec![],
                        vec![],
                    ),
                ],
            )])
        })
        .collect();
    let main = Computation::try_new(
        "main",
        vec![
            Instruction::new(
                Operation::parameter(0, "predicate"),
                vec![],
                vec![],
                ValueId(0),
                predicate,
            ),
            Instruction::new(
                Operation::parameter(1, "input"),
                vec![],
                vec![],
                ValueId(1),
                input,
            ),
            Instruction::new(
                Operation::stablehlo(StableHloOp::If, Attributes::new()),
                vec![ValueId(0)],
                vec![],
                ValueId(2),
                output,
            )
            .with_regions(branches),
        ],
        ValueId(2),
    )
    .unwrap();
    for (size, accepted) in [(2, true), (3, false)] {
        let mut definition = function("mesh");
        definition
            .set_attributes(
                [(
                    "stablehlo.mesh".into(),
                    Attribute::StableHlo(Box::new(StableHloAttribute::Mesh(Mesh {
                        axes: vec![MeshAxis {
                            name: "x".into(),
                            size,
                        }],
                        device_ids: None,
                    }))),
                )]
                .into(),
            )
            .unwrap();
        let mut module = Module::new("module");
        module.add_computation(main.clone());
        module.add_computation(definition);
        assert_eq!(module.verify().is_ok(), accepted);
    }
}

fn gather() -> Computation {
    computation(
        StableHloOp::AllGather,
        vec![TensorType::static_shape(ElementType::F32, [3]).into()],
        vec![TensorType::static_shape(ElementType::F32, [6]).into()],
        [
            ("all_gather_dim".into(), Attribute::integer(0)),
            (
                "replica_groups".into(),
                Attribute::StableHlo(Box::new(StableHloAttribute::ReplicaGroupMeshAxes(
                    ReplicaGroupMeshAxes {
                        mesh: MeshReference::Symbol("mesh".into()),
                        axes: vec![AxisRef {
                            name: "x".into(),
                            sub_axis_info: None,
                        }],
                    },
                ))),
            ),
        ]
        .into(),
    )
    .unwrap()
}

#[test]
fn module_resolves_mesh_symbols_before_checking_collective_cardinality() {
    for name in ["mesh", "stablehlo.mesh"] {
        for (size, accepted) in [(2, true), (3, false)] {
            let mut definition = function("mesh");
            definition
                .set_attributes(
                    [(
                        name.into(),
                        Attribute::StableHlo(Box::new(StableHloAttribute::Mesh(Mesh {
                            axes: vec![MeshAxis {
                                name: "x".into(),
                                size,
                            }],
                            device_ids: None,
                        }))),
                    )]
                    .into(),
                )
                .unwrap();
            let mut module = Module::new("module");
            module.add_computation(gather());
            module.add_computation(definition);
            assert_eq!(module.verify().is_ok(), accepted);
        }
    }
}

#[test]
fn mesh_symbols_must_exist_and_define_mesh_metadata() {
    let mut module = Module::new("module");
    module.add_computation(gather());
    assert!(
        matches!(module.verify(), Err(BuildError::UnknownComputationName { name, .. }) if name == "mesh")
    );
    module.add_computation(function("mesh"));
    assert_eq!(
        module.verify(),
        Err(BuildError::InvalidMeshSymbol("mesh".into()))
    );
    module.computations_mut()[1]
        .set_attributes([("mesh".into(), Attribute::String("not a mesh".into()))].into())
        .unwrap();
    assert_eq!(
        module.verify(),
        Err(BuildError::InvalidMeshSymbol("mesh".into()))
    );
}
