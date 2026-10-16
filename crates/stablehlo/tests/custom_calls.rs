use stablehlo::{
    ArrayOfLayout, Attribute, Attributes, BufferType, ElementType, Layout, OutputOperandAlias,
    StableHloAttribute, StableHloOp, TensorType, Type,
};

mod common;
use common::computation;

fn tensor() -> Type {
    TensorType::static_shape(ElementType::F32, [2, 3]).into()
}

fn buffer() -> Type {
    Type::Buffer(BufferType::new(ElementType::F32, vec![Some(2), Some(3)]))
}

fn attributes(target: &str) -> Attributes {
    [("call_target_name".into(), Attribute::String(target.into()))].into()
}

fn layouts(layouts: &[&[i64]]) -> Attribute {
    ArrayOfLayout(layouts.iter().map(|axes| Layout(axes.to_vec())).collect()).into()
}

fn aliases(values: &[(i64, &[i64], &[i64])]) -> Attribute {
    Attribute::Array(
        values
            .iter()
            .map(|(operand_index, operand, result)| {
                Attribute::StableHlo(Box::new(StableHloAttribute::OutputOperandAlias(
                    OutputOperandAlias {
                        output_tuple_indices: result.to_vec(),
                        operand_index: *operand_index,
                        operand_tuple_indices: operand.to_vec(),
                    },
                )))
            })
            .collect(),
    )
}

#[test]
fn custom_call_backend_configuration_matches_the_selected_abi() {
    for (version, config, accepted) in [
        (1, Attribute::String("config".into()), true),
        (1, Attribute::Dictionary(Default::default()), false),
        (4, Attribute::Dictionary(Default::default()), true),
        (4, Attribute::String(String::new()), false),
    ] {
        let mut attributes = attributes("target");
        attributes.insert("api_version".into(), Attribute::i32(version));
        attributes.insert("backend_config".into(), config);
        assert_eq!(
            computation(StableHloOp::CustomCall, vec![], vec![], attributes).is_ok(),
            accepted
        );
    }
}

#[test]
fn custom_call_layouts_must_be_paired_complete_rank_permutations() {
    for (input, output, accepted) in [
        (Some(layouts(&[&[1, 0]])), Some(layouts(&[&[0, 1]])), true),
        (Some(layouts(&[&[1, 0]])), None, false),
        (None, Some(layouts(&[&[1, 0]])), false),
        (Some(layouts(&[&[1, 1]])), Some(layouts(&[&[1, 0]])), false),
        (Some(layouts(&[&[2, 0]])), Some(layouts(&[&[1, 0]])), false),
        (Some(layouts(&[])), Some(layouts(&[&[1, 0]])), false),
    ] {
        let mut attributes = attributes("target");
        if let Some(layout) = input {
            attributes.insert("operand_layouts".into(), layout);
        }
        if let Some(layout) = output {
            attributes.insert("result_layouts".into(), layout);
        }
        assert_eq!(
            computation(
                StableHloOp::CustomCall,
                vec![tensor()],
                vec![tensor()],
                attributes
            )
            .is_ok(),
            accepted
        );
    }
}

#[test]
fn custom_call_tuple_results_expand_once_for_layouts_and_tokens_have_empty_layouts() {
    let mut attributes = attributes("target");
    attributes.insert("operand_layouts".into(), layouts(&[&[]]));
    attributes.insert("result_layouts".into(), layouts(&[&[1, 0], &[]]));
    assert!(
        computation(
            StableHloOp::CustomCall,
            vec![Type::Token],
            vec![Type::Tuple(vec![tensor(), Type::Token])],
            attributes.clone()
        )
        .is_ok()
    );
    attributes.insert("operand_layouts".into(), layouts(&[&[0]]));
    assert!(
        computation(
            StableHloOp::CustomCall,
            vec![Type::Token],
            vec![Type::Tuple(vec![tensor(), Type::Token])],
            attributes
        )
        .is_err()
    );
}

#[test]
fn custom_call_aliases_resolve_nested_operand_and_result_paths() {
    let mut attributes = attributes("target");
    attributes.insert(
        "output_operand_aliases".into(),
        aliases(&[(0, &[1, 0], &[0])]),
    );
    let input = Type::Tuple(vec![Type::Token, Type::Tuple(vec![tensor()])]);
    assert!(
        computation(
            StableHloOp::CustomCall,
            vec![input.clone()],
            vec![tensor(), Type::Token],
            attributes.clone()
        )
        .is_ok()
    );
    for (index, operand, result) in [
        (1, vec![1, 0], vec![0]),
        (0, vec![1, 1], vec![0]),
        (0, vec![1, 0], vec![1]),
        (0, vec![1, 0], vec![2]),
    ] {
        attributes.insert(
            "output_operand_aliases".into(),
            aliases(&[(index, &operand, &result)]),
        );
        assert!(
            computation(
                StableHloOp::CustomCall,
                vec![input.clone()],
                vec![tensor(), Type::Token],
                attributes.clone()
            )
            .is_err()
        );
    }
}

#[test]
fn custom_call_every_buffer_result_requires_a_unique_buffer_alias() {
    assert!(
        computation(
            StableHloOp::CustomCall,
            vec![buffer()],
            vec![buffer()],
            attributes("target")
        )
        .is_err()
    );
    let mut attributes = attributes("target");
    attributes.insert("output_operand_aliases".into(), aliases(&[(0, &[], &[])]));
    assert!(
        computation(
            StableHloOp::CustomCall,
            vec![buffer()],
            vec![buffer()],
            attributes.clone()
        )
        .is_ok()
    );
    attributes.insert(
        "output_operand_aliases".into(),
        aliases(&[(0, &[], &[]), (0, &[], &[])]),
    );
    assert!(
        computation(
            StableHloOp::CustomCall,
            vec![buffer()],
            vec![buffer()],
            attributes
        )
        .is_err()
    );
}

#[test]
fn custom_call_buffer_paths_include_nested_tuple_results() {
    let output = Type::Tuple(vec![Type::Token, Type::Tuple(vec![buffer()])]);
    let mut attributes = attributes("target");
    attributes.insert(
        "output_operand_aliases".into(),
        aliases(&[(0, &[], &[1, 0])]),
    );
    assert!(
        computation(
            StableHloOp::CustomCall,
            vec![buffer()],
            vec![output.clone()],
            attributes.clone()
        )
        .is_ok()
    );
    attributes.insert("output_operand_aliases".into(), aliases(&[]));
    assert!(
        computation(
            StableHloOp::CustomCall,
            vec![buffer()],
            vec![output],
            attributes
        )
        .is_err()
    );
}

#[test]
fn buffer_intrinsics_have_reference_defined_type_signatures() {
    for (target, inputs, outputs, accepted) in [
        ("CreateBuffer", vec![], vec![buffer()], true),
        ("CreateBuffer", vec![tensor()], vec![buffer()], false),
        ("CreateBuffer", vec![], vec![tensor()], false),
        ("Pin", vec![tensor()], vec![buffer()], true),
        ("Pin", vec![buffer()], vec![tensor()], false),
        ("Unpin", vec![buffer()], vec![tensor()], true),
        ("Unpin", vec![tensor()], vec![buffer()], false),
        ("Unpin", vec![], vec![], false),
    ] {
        assert_eq!(
            computation(StableHloOp::CustomCall, inputs, outputs, attributes(target)).is_ok(),
            accepted,
            "{target}"
        );
    }
}
