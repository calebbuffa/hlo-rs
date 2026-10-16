use std::collections::HashSet;

use stablehlo::{
    Operation, STABLEHLO_CURRENT_VERSION, STABLEHLO_MINIMUM_VERSION, STABLEHLO_SOURCE_COMMIT,
    StableHloOp,
};

#[test]
fn pinned_stablehlo_operation_catalog_has_unique_mnemonics() {
    assert_eq!(StableHloOp::ALL.len(), 118);
    let mnemonics = StableHloOp::ALL
        .iter()
        .map(|operation| operation.mnemonic())
        .collect::<HashSet<_>>();
    assert_eq!(mnemonics.len(), StableHloOp::ALL.len());
    assert!(mnemonics.contains("add"));
    assert!(mnemonics.contains("while"));
    assert!(mnemonics.contains("all_reduce"));
    assert!(mnemonics.contains("dynamic_conv"));
}

#[test]
fn pinned_revision_matches_the_reference_checkout() {
    assert_eq!(
        STABLEHLO_SOURCE_COMMIT,
        "25605c9f48a9aec305876d60d26b028398d298d2"
    );
    assert_eq!(STABLEHLO_CURRENT_VERSION, "1.22.0");
    assert_eq!(STABLEHLO_MINIMUM_VERSION, "0.9.0");
}

#[test]
fn operation_cardinality_metadata_matches_representative_ods_signatures() {
    let cases = [
        (StableHloOp::Constant, Some(0), Some(1)),
        (StableHloOp::DynamicIota, Some(1), Some(1)),
        (StableHloOp::If, Some(1), None),
        (StableHloOp::While, None, None),
        (StableHloOp::ReduceScatter, Some(1), Some(1)),
        (StableHloOp::BatchNormGrad, Some(5), Some(3)),
        (StableHloOp::BatchNormInference, Some(5), Some(1)),
        (StableHloOp::BatchNormTraining, Some(3), Some(3)),
        (StableHloOp::Rng, Some(3), Some(1)),
        (StableHloOp::Outfeed, None, Some(1)),
        (StableHloOp::Send, None, Some(1)),
        (StableHloOp::DynamicPad, Some(5), Some(1)),
        (StableHloOp::DynamicGather, Some(3), Some(1)),
        (StableHloOp::DynamicConv, Some(3), Some(1)),
        (StableHloOp::TorchIndexSelect, Some(2), Some(1)),
        (StableHloOp::OptimizationBarrier, None, None),
    ];
    for (opcode, operands, results) in cases {
        let operation = Operation::stablehlo(opcode, Default::default());
        assert_eq!(operation.expected_arity(), operands, "{opcode:?} operands");
        assert_eq!(
            operation.expected_result_arity(),
            results,
            "{opcode:?} results"
        );
    }
}
