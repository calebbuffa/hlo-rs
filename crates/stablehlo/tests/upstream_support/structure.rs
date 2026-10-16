use std::collections::HashMap;

use stablehlo::{Instruction, Location, Module, Operation, ValueId};

fn location(value: Option<&Location>) -> Option<&Location> {
    value.filter(|value| !matches!(value, Location::Unknown))
}

/// Ignores SSA spelling and explicit-vs-implicit unknown locations, not math,
/// types, attributes, region structure, function identity or operand ordering.
pub fn assert_equivalent(expected: &Module, actual: &Module) {
    assert_eq!(expected.name(), actual.name());
    assert_eq!(expected.attributes(), actual.attributes());
    assert_eq!(expected.entry(), actual.entry());
    assert_eq!(location(expected.location()), location(actual.location()));
    assert_eq!(expected.computations().len(), actual.computations().len());
    for (expected, actual) in expected.computations().iter().zip(actual.computations()) {
        assert_eq!(expected.name(), actual.name());
        assert_eq!(expected.attributes(), actual.attributes());
        assert_eq!(expected.function_type(), actual.function_type());
        assert_eq!(expected.is_declaration(), actual.is_declaration());
        assert_eq!(location(expected.location()), location(actual.location()));
        assert_eq!(
            location(expected.return_location()),
            location(actual.return_location())
        );
        let mut values = HashMap::new();
        instructions(expected.instructions(), actual.instructions(), &mut values);
        let roots = expected
            .roots()
            .iter()
            .map(|id| values[id])
            .collect::<Vec<_>>();
        assert_eq!(roots, actual.roots());
    }
}

fn instructions(
    expected: &[Instruction],
    actual: &[Instruction],
    values: &mut HashMap<ValueId, ValueId>,
) {
    assert_eq!(expected.len(), actual.len());
    for (expected, actual) in expected.iter().zip(actual) {
        match (expected.operation(), actual.operation()) {
            (
                Operation::Parameter { number: left, .. },
                Operation::Parameter { number: right, .. },
            ) => {
                assert_eq!(left, right);
            }
            (left, right) => assert_eq!(left, right),
        }
        assert_eq!(expected.called_computations(), actual.called_computations());
        assert_eq!(location(expected.location()), location(actual.location()));
        assert_eq!(expected.results().len(), actual.results().len());
        let operands = expected
            .operands()
            .iter()
            .map(|id| values[id])
            .collect::<Vec<_>>();
        assert_eq!(operands, actual.operands());
        assert_eq!(expected.regions().len(), actual.regions().len());
        for (expected, actual) in expected.regions().iter().zip(actual.regions()) {
            assert_eq!(expected.blocks().len(), actual.blocks().len());
            for (expected, actual) in expected.blocks().iter().zip(actual.blocks()) {
                let mut nested = values.clone();
                assert_eq!(expected.arguments().len(), actual.arguments().len());
                for (expected, actual) in expected.arguments().iter().zip(actual.arguments()) {
                    assert_eq!(expected.ty(), actual.ty());
                    assert_eq!(location(expected.location()), location(actual.location()));
                    nested.insert(expected.value(), actual.value());
                }
                instructions(expected.instructions(), actual.instructions(), &mut nested);
            }
        }
        for (expected, actual) in expected.results().iter().zip(actual.results()) {
            assert_eq!(expected.ty(), actual.ty());
            values.insert(expected.value(), actual.value());
        }
    }
}
