use stablehlo::{
    Attribute, Attributes, Block, ElementType, Operation, Region, StableHloAttribute, StableHloOp,
    TensorType, Type, TypeExtensions, UnrankedTensorType,
};

mod common;
use common::computation_with_regions;

#[test]
fn every_operation_handles_missing_attributes_regions_and_wrong_type_domains_without_panicking() {
    let types = [
        Type::Token,
        Type::Tuple(vec![]),
        Type::Element(ElementType::F32),
        UnrankedTensorType::Plain(ElementType::F32).into(),
        TensorType::scalar(ElementType::F32).into(),
        TensorType::scalar(ElementType::I1).into(),
        TensorType::static_shape(ElementType::I64, [2]).into(),
        TensorType::new(ElementType::F32, vec![None, Some(2)]).into(),
    ];
    for &opcode in StableHloOp::ALL {
        let operation = Operation::stablehlo(opcode, Attributes::new());
        let operands = operation.expected_arity().unwrap_or(1);
        let results = operation.expected_result_arity().unwrap_or(1);
        for input in &types {
            for output in &types {
                for regions in [vec![], vec![Region::new(vec![Block::default()])]] {
                    let outcome = std::panic::catch_unwind(|| {
                        computation_with_regions(
                            opcode,
                            vec![input.clone(); operands],
                            vec![output.clone(); results],
                            Attributes::new(),
                            regions,
                        )
                    });
                    assert!(
                        outcome.is_ok(),
                        "{opcode:?}, input {input:?}, output {output:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn malformed_result_encodings_are_rejected_before_shape_rules() {
    let output = TensorType::new(ElementType::F32, vec![None])
        .with_encoding(Attribute::StableHlo(Box::new(
            StableHloAttribute::TypeExtensions(TypeExtensions { bounds: vec![] }),
        )))
        .into();
    let outcome = computation_with_regions(
        StableHloOp::Negate,
        vec![TensorType::static_shape(ElementType::F32, [3]).into()],
        vec![output],
        Attributes::new(),
        vec![],
    );
    assert!(matches!(
        outcome,
        Err(stablehlo::BuildError::InvalidType(
            stablehlo::TypeError::EncodingBoundsRankMismatch { .. },
        ))
    ));
}
