use stablehlo::{
    Attribute, Attributes, BuildError, Computation, ElementType, Instruction, Module, Operation,
    StableHloOp, TensorType, Type, ValueId,
};

mod common;
use common::computation;

fn scalar(element: ElementType) -> Type {
    TensorType::scalar(element).into()
}

fn composite_attributes(name: &str) -> Attributes {
    [
        ("name".into(), Attribute::String(name.into())),
        (
            "composite_attributes".into(),
            Attribute::Dictionary(Attributes::new()),
        ),
        (
            "decomposition".into(),
            Attribute::SymbolRef(vec!["decomposition".into()]),
        ),
    ]
    .into()
}

fn decomposition(ty: Type) -> Computation {
    Computation::try_new(
        "decomposition",
        vec![Instruction::new(
            Operation::parameter(0, "arg"),
            vec![],
            vec![],
            ValueId(0),
            ty,
        )],
        ValueId(0),
    )
    .unwrap()
}

#[test]
fn composite_names_follow_the_namespaced_identifier_grammar() {
    for name in ["vendor.op", "vendor_2.op$1", "A.b.c", "a.0", "a.$"] {
        assert!(
            computation(
                StableHloOp::Composite,
                vec![scalar(ElementType::F32)],
                vec![scalar(ElementType::F32)],
                composite_attributes(name),
            )
            .is_ok(),
            "{name}"
        );
    }
    for name in [
        "",
        "op",
        ".op",
        "1vendor.op",
        "_vendor.op",
        "vendor..op",
        "vendor.",
        "vendor-op.foo",
        "vendor.op/name",
        "v$.op",
    ] {
        assert!(
            matches!(
                computation(
                    StableHloOp::Composite,
                    vec![scalar(ElementType::F32)],
                    vec![scalar(ElementType::F32)],
                    composite_attributes(name),
                ),
                Err(BuildError::MissingOrInvalidAttribute {
                    attribute: "name",
                    ..
                })
            ),
            "{name}"
        );
    }
}

#[test]
fn composite_requires_exact_decomposition_operand_and_result_signatures() {
    let f32_ty = scalar(ElementType::F32);
    for (inputs, outputs, accepted) in [
        (vec![f32_ty.clone()], vec![f32_ty.clone()], true),
        (vec![], vec![f32_ty.clone()], false),
        (vec![scalar(ElementType::I32)], vec![f32_ty.clone()], false),
        (vec![f32_ty.clone()], vec![scalar(ElementType::I32)], false),
        (
            vec![f32_ty.clone(), f32_ty.clone()],
            vec![f32_ty.clone()],
            false,
        ),
        (vec![f32_ty.clone()], vec![], false),
        (
            vec![TensorType::new(ElementType::F32, vec![None]).into()],
            vec![f32_ty.clone()],
            false,
        ),
    ] {
        let composite = computation(
            StableHloOp::Composite,
            inputs,
            outputs,
            composite_attributes("vendor.operation"),
        )
        .unwrap();
        let mut module = Module::new("module");
        module.add_computation(composite);
        module.add_computation(decomposition(f32_ty.clone()));
        let verified = module.verify();
        assert_eq!(verified.is_ok(), accepted);
        if !accepted {
            assert!(matches!(
                verified,
                Err(BuildError::InvalidComputationSignature { .. })
            ));
        }
    }
}

#[test]
fn composite_resolves_symbols_independently_of_definition_order() {
    let ty = scalar(ElementType::F32);
    let composite = computation(
        StableHloOp::Composite,
        vec![ty.clone()],
        vec![ty.clone()],
        composite_attributes("vendor.operation"),
    )
    .unwrap();
    for computations in [
        vec![composite.clone(), decomposition(ty.clone())],
        vec![decomposition(ty.clone()), composite.clone()],
    ] {
        let mut module = Module::new("module");
        for computation in computations {
            module.add_computation(computation);
        }
        assert!(module.verify().is_ok());
    }
    let mut module = Module::new("module");
    module.add_computation(composite);
    assert!(matches!(
        module.verify(),
        Err(BuildError::UnknownComputationName { .. })
    ));
}
