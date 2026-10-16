use stablehlo::{
    Attribute, Attributes, Block, BuildError, Computation, ElementType, FunctionType, Instruction,
    Module, Operation, OperationResult, Region, StableHloOp, TensorType, Type, ValueId,
};

fn scalar(element: ElementType) -> Type {
    TensorType::scalar(element).into()
}

fn parameter() -> Instruction {
    Instruction::new(
        Operation::parameter(0, "x"),
        vec![],
        vec![],
        ValueId(0),
        scalar(ElementType::F32),
    )
}

fn caller(callee: &str, result: Type) -> Computation {
    Computation::try_new(
        "caller",
        vec![
            parameter(),
            Instruction::new(
                Operation::call(callee),
                vec![ValueId(0)],
                vec![],
                ValueId(1),
                result,
            ),
        ],
        ValueId(1),
    )
    .unwrap()
}

fn declaration(name: &str) -> Computation {
    let ty = scalar(ElementType::F32);
    Computation::try_declare(name, FunctionType::new(vec![ty.clone()], vec![ty])).unwrap()
}

#[test]
fn external_declarations_preserve_signatures_and_verify_in_the_same_module_api() {
    let external = declaration("external");
    assert!(external.is_declaration());
    assert!(external.instructions().is_empty());
    assert!(external.roots().is_empty());
    assert_eq!(
        external.function_type(),
        FunctionType::new(
            vec![scalar(ElementType::F32)],
            vec![scalar(ElementType::F32)]
        )
    );
    let mut module = Module::new("module");
    module.add_computation(caller("external", scalar(ElementType::F32)));
    module.add_computation(external);
    module.verify().unwrap();
}

#[test]
fn direct_calls_resolve_forward_references_definitions_and_recursive_symbols() {
    for definition in [
        Computation::try_new("target", vec![parameter()], ValueId(0)).unwrap(),
        {
            let mut recursive = caller("target", scalar(ElementType::F32));
            // Build recursion through the same public constructors.
            recursive =
                Computation::try_new("target", recursive.instructions().to_vec(), ValueId(1))
                    .unwrap();
            recursive
        },
    ] {
        let mut module = Module::new("module");
        module.add_computation(caller("target", scalar(ElementType::F32)));
        module.add_computation(definition);
        module.verify().unwrap();
    }
}

#[test]
fn unresolved_and_mismatched_calls_fail_module_verification() {
    let mut module = Module::new("module");
    module.add_computation(caller("external", scalar(ElementType::F32)));
    assert!(matches!(
        module.verify(),
        Err(BuildError::UnknownComputationName {
            operation: "func.call",
            ..
        })
    ));
    module.add_computation(
        Computation::try_declare(
            "external",
            FunctionType::new(
                vec![scalar(ElementType::I32)],
                vec![scalar(ElementType::F32)],
            ),
        )
        .unwrap(),
    );
    assert!(matches!(
        module.verify(),
        Err(BuildError::InvalidComputationSignature {
            operation: "func.call",
            ..
        })
    ));
    let mut module = Module::new("module");
    module.add_computation(caller("external", scalar(ElementType::I32)));
    module.add_computation(declaration("external"));
    assert!(matches!(
        module.verify(),
        Err(BuildError::InvalidComputationSignature {
            operation: "func.call",
            ..
        })
    ));
}

#[test]
fn call_attribute_and_region_contracts_are_checked_before_symbol_resolution() {
    for attributes in [
        Attributes::new(),
        [("callee".into(), Attribute::String("target".into()))].into(),
        [("callee".into(), Attribute::SymbolRef(vec![]))].into(),
        [("callee".into(), Attribute::SymbolRef(vec!["".into()]))].into(),
        [(
            "callee".into(),
            Attribute::SymbolRef(vec!["module".into(), "target".into()]),
        )]
        .into(),
        [
            ("callee".into(), Attribute::SymbolRef(vec!["target".into()])),
            ("arg_attrs".into(), Attribute::Array(vec![])),
        ]
        .into(),
        [
            ("callee".into(), Attribute::SymbolRef(vec!["target".into()])),
            (
                "res_attrs".into(),
                Attribute::Array(vec![Attribute::Bool(true)]),
            ),
        ]
        .into(),
    ] {
        assert!(
            Computation::try_new(
                "caller",
                vec![
                    parameter(),
                    Instruction::new(
                        Operation::Call { attributes },
                        vec![ValueId(0)],
                        vec![],
                        ValueId(1),
                        scalar(ElementType::F32)
                    )
                ],
                ValueId(1)
            )
            .is_err()
        );
    }
    let call = Instruction::new(
        Operation::call("target"),
        vec![ValueId(0)],
        vec![],
        ValueId(1),
        scalar(ElementType::F32),
    )
    .with_regions(vec![Region::default()]);
    assert!(matches!(
        Computation::try_new("caller", vec![parameter(), call], ValueId(1)),
        Err(BuildError::InvalidRegionCount { .. })
    ));
    assert!(!Operation::call("target").is_pure());
}

#[test]
fn declarations_validate_signature_metadata_and_updates_atomically() {
    assert!(Computation::try_declare("", FunctionType::new(vec![], vec![])).is_err());
    assert!(matches!(
        Computation::try_declare(
            "bad",
            FunctionType::new(vec![Type::Element(ElementType::F32)], vec![])
        ),
        Err(BuildError::InvalidFunctionType(_))
    ));
    let mut function = declaration("external");
    let original = function.clone();
    for attributes in [
        Attributes::new(),
        [("sym_visibility".into(), Attribute::String("public".into()))].into(),
        [
            ("sym_visibility".into(), Attribute::String("private".into())),
            ("arg_attrs".into(), Attribute::Array(vec![])),
        ]
        .into(),
    ] {
        assert!(function.set_attributes(attributes).is_err());
        assert_eq!(function, original);
    }
    function
        .set_attributes(
            [
                ("sym_visibility".into(), Attribute::String("nested".into())),
                (
                    "arg_attrs".into(),
                    Attribute::Array(vec![Attribute::Dictionary(Attributes::new())]),
                ),
                (
                    "res_attrs".into(),
                    Attribute::Array(vec![Attribute::Dictionary(Attributes::new())]),
                ),
            ]
            .into(),
        )
        .unwrap();
}

#[test]
fn declaration_body_installation_preserves_the_declared_signature_atomically() {
    let mut function = declaration("external");
    let original = function.clone();
    assert!(function.replace_body_with_roots(vec![], vec![]).is_err());
    assert_eq!(function, original);
    assert!(
        function
            .replace_body(vec![parameter()], ValueId(20))
            .is_err()
    );
    assert_eq!(function, original);
    function
        .replace_body(vec![parameter()], ValueId(0))
        .unwrap();
    assert!(!function.is_declaration());
    assert_eq!(function.function_type(), original.function_type());
    assert_eq!(function.attributes(), original.attributes());
}

#[test]
fn empty_void_definitions_and_zero_or_multiple_result_calls_are_supported() {
    let empty = Computation::try_new_with_roots("empty", vec![], vec![]).unwrap();
    assert!(!empty.is_declaration());
    assert_eq!(empty.function_type(), FunctionType::new(vec![], vec![]));
    for types in [vec![], vec![Type::Token, scalar(ElementType::F32)]] {
        let roots = (0..types.len())
            .map(|index| ValueId(index as u32))
            .collect::<Vec<_>>();
        let caller = Computation::try_new_with_roots(
            "caller",
            vec![Instruction::with_results(
                Operation::call("target"),
                vec![],
                vec![],
                roots
                    .iter()
                    .copied()
                    .zip(types.clone())
                    .map(|(value, ty)| OperationResult::new(value, ty))
                    .collect(),
            )],
            roots,
        )
        .unwrap();
        let target = Computation::try_declare("target", FunctionType::new(vec![], types)).unwrap();
        let mut module = Module::new("module");
        module.add_computation(caller);
        module.add_computation(target);
        module.add_computation(empty.clone());
        module.verify().unwrap();
    }
}

#[test]
fn calls_inside_nested_regions_are_resolved_and_signature_checked() {
    for (name, accepted) in [("external", true), ("missing", false)] {
        let branches = [100, 101]
            .into_iter()
            .map(|id| {
                Region::new(vec![Block::new(
                    vec![],
                    vec![
                        Instruction::new(
                            Operation::call(name),
                            vec![ValueId(0)],
                            vec![],
                            ValueId(id),
                            scalar(ElementType::F32),
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
                parameter(),
                Instruction::new(
                    Operation::parameter(1, "predicate"),
                    vec![],
                    vec![],
                    ValueId(1),
                    scalar(ElementType::I1),
                ),
                Instruction::new(
                    Operation::stablehlo(StableHloOp::If, Attributes::new()),
                    vec![ValueId(1)],
                    vec![],
                    ValueId(2),
                    scalar(ElementType::F32),
                )
                .with_regions(branches),
            ],
            ValueId(2),
        )
        .unwrap();
        let mut module = Module::new("module");
        module.add_computation(main);
        module.add_computation(declaration("external"));
        assert_eq!(module.verify().is_ok(), accepted);
    }
}
