use stablehlo::{
    Attributes, BuildError, Computation, Instruction, Operation, OperationResult, Region,
    StableHloOp, Type, ValueId,
};

#[allow(dead_code)]
pub fn computation(
    opcode: StableHloOp,
    operands: Vec<Type>,
    results: Vec<Type>,
    attributes: Attributes,
) -> Result<Computation, BuildError> {
    computation_with_regions(opcode, operands, results, attributes, vec![])
}

pub fn computation_with_regions(
    opcode: StableHloOp,
    operands: Vec<Type>,
    results: Vec<Type>,
    attributes: Attributes,
    regions: Vec<Region>,
) -> Result<Computation, BuildError> {
    let count = u32::try_from(operands.len()).unwrap();
    let mut instructions = operands
        .into_iter()
        .enumerate()
        .map(|(index, ty)| {
            let id = u32::try_from(index).unwrap();
            Instruction::new(
                Operation::parameter(id, format!("arg{id}")),
                vec![],
                vec![],
                ValueId(id),
                ty,
            )
        })
        .collect::<Vec<_>>();
    let roots = (0..results.len())
        .map(|index| ValueId(count + u32::try_from(index).unwrap()))
        .collect::<Vec<_>>();
    instructions.push(
        Instruction::with_results(
            Operation::stablehlo(opcode, attributes),
            (0..count).map(ValueId).collect(),
            vec![],
            roots
                .iter()
                .copied()
                .zip(results)
                .map(|(id, ty)| OperationResult::new(id, ty))
                .collect(),
        )
        .with_regions(regions),
    );
    Computation::try_new_with_roots("schema_test", instructions, roots)
}
