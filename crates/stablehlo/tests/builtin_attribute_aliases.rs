use stablehlo::{
    ArrayOfLayout, ArrayOfTiling, Attribute, Attributes, ElementType, IndexTensor, Layout,
    Precision, StableHloOp, TensorType, Tiling, Type,
    text::{parse_attribute, print_attribute},
};

mod common;
use common::computation;

#[test]
fn ergonomic_builders_produce_the_canonical_builtin_attributes() {
    for (attribute, assembly) in [
        (
            Attribute::precision_config(vec![Precision::Default, Precision::Highest]),
            "[#stablehlo<precision DEFAULT>, #stablehlo<precision HIGHEST>]",
        ),
        (
            IndexTensor(vec![2, 4]).into(),
            "dense<[2, 4]> : tensor<2xindex>",
        ),
        (Layout(vec![1, 0]).into(), "dense<[1, 0]> : tensor<2xindex>"),
        (
            ArrayOfLayout(vec![Layout(vec![]), Layout(vec![1, 0])]).into(),
            "[dense<> : tensor<0xindex>, dense<[1, 0]> : tensor<2xindex>]",
        ),
        (
            Tiling(vec![IndexTensor(vec![2, 4])]).into(),
            "[dense<[2, 4]> : tensor<2xindex>]",
        ),
        (
            ArrayOfTiling(vec![Tiling(vec![IndexTensor(vec![2, 4])])]).into(),
            "[[dense<[2, 4]> : tensor<2xindex>]]",
        ),
    ] {
        attribute.verify().unwrap();
        assert_eq!(attribute, parse_attribute(assembly).unwrap());
        assert_eq!(
            parse_attribute(&print_attribute(&attribute).unwrap()).unwrap(),
            attribute
        );
    }
}

#[test]
fn parsed_precision_configs_are_directly_usable_by_operations() {
    for (assembly, accepted) in [
        ("[]", true),
        ("[#stablehlo<precision HIGH>]", true),
        (
            "[#stablehlo<precision HIGH>, #stablehlo<precision HIGHEST>]",
            true,
        ),
        (
            "[#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]",
            false,
        ),
        ("[\"DEFAULT\"]", false),
        ("[#stablehlo<rng_algorithm DEFAULT>]", false),
    ] {
        let vector = TensorType::static_shape(ElementType::F32, [2]).into();
        let scalar = TensorType::static_shape(ElementType::F32, []).into();
        assert_eq!(
            computation(
                StableHloOp::Dot,
                vec![vector; 2],
                vec![scalar],
                [(
                    "precision_config".into(),
                    parse_attribute(assembly).unwrap()
                )]
                .into()
            )
            .is_ok(),
            accepted,
            "{assembly}"
        );
    }
}

#[test]
fn parsed_layouts_require_dense_rank_one_index_tensors() {
    for (assembly, accepted) in [
        ("[dense<[1, 0]> : tensor<2xindex>]", true),
        ("[dense<[1, 0]> : tensor<2xi64>]", false),
        ("[dense<[[1, 0]]> : tensor<1x2xindex>]", false),
        ("[dense<[1, 1]> : tensor<2xindex>]", false),
        ("[array<i64: 1, 0>]", false),
        ("[dense<[-1, 0]> : tensor<2xindex>]", false),
    ] {
        let tensor: Type = TensorType::static_shape(ElementType::F32, [2, 3]).into();
        let layout = parse_attribute(assembly).unwrap();
        let attributes = [
            (
                "call_target_name".into(),
                Attribute::String("target".into()),
            ),
            ("operand_layouts".into(), layout.clone()),
            ("result_layouts".into(), layout),
        ]
        .into();
        assert_eq!(
            computation(
                StableHloOp::CustomCall,
                vec![tensor.clone()],
                vec![tensor],
                attributes
            )
            .is_ok(),
            accepted,
            "{assembly}"
        );
    }
}

#[test]
fn parsed_tilings_preserve_both_array_levels() {
    for (assembly, accepted) in [
        ("[]", true),
        ("[[]]", true),
        (
            "[[dense<[2, 4]> : tensor<2xindex>, dense<[1, 8]> : tensor<2xindex>]]",
            true,
        ),
        ("[dense<[2, 4]> : tensor<2xindex>]", false),
        ("[[dense<[2, 4]> : tensor<2xi64>]]", false),
        ("[[dense<[[2, 4]]> : tensor<1x2xindex>]]", false),
    ] {
        let attributes: Attributes = [
            (
                "call_target_name".into(),
                Attribute::String("target".into()),
            ),
            ("result_tilings".into(), parse_attribute(assembly).unwrap()),
        ]
        .into();
        assert_eq!(
            computation(StableHloOp::CustomCall, vec![], vec![], attributes).is_ok(),
            accepted,
            "{assembly}"
        );
    }
}

#[test]
fn generic_dialect_assembly_wraps_structured_attributes_without_changing_them() {
    for payload in [
        "channel_handle<handle = 1, type = 2>",
        "bounds<2, ?>",
        "result_accuracy_mode<DEFAULT>",
        "result_accuracy<mode = #stablehlo<result_accuracy_mode<DEFAULT>>>",
        "dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>",
    ] {
        let generic = parse_attribute(&format!("#stablehlo<{payload}>")).unwrap();
        let pretty = parse_attribute(&format!("#stablehlo.{payload}")).unwrap();
        assert_eq!(generic, pretty);
        assert_eq!(
            parse_attribute(&print_attribute(&generic).unwrap()).unwrap(),
            generic
        );
    }
    for assembly in [
        "#stablehlo<precision DEFAULT",
        "#stablehlo<precision DEFAULT>>",
        "#stablehlo<precision DEFAULT, HIGH>",
        "#stablehlo<result_accuracy_mode DEFAULT>",
        "#stablehlo.result_accuracy<mode = #stablehlo<precision DEFAULT>>",
        "#stablehlo.result_accuracy<mode = #stablehlo.result_accuracy_mode <DEFAULT>>",
    ] {
        assert!(parse_attribute(assembly).is_err(), "{assembly}");
    }
}
