use stablehlo::text::{parse_module, parse_type, print_module, print_type};
use stablehlo::{
    ArrayOfLayout, Attribute, Attributes, ElementType, Layout, StableHloOp, Type,
    UnrankedTensorType, infer_least_specific_type, infer_most_specific_type,
};

mod common;
use common::computation;

fn ty(source: &str) -> Type {
    parse_type(source).unwrap()
}

fn call_attributes() -> Attributes {
    [(
        "call_target_name".into(),
        Attribute::String("external".into()),
    )]
    .into()
}

fn custom_call(input: Type, output: Type, attributes: Attributes) -> bool {
    computation(
        StableHloOp::CustomCall,
        vec![input],
        vec![output],
        attributes,
    )
    .is_ok()
}

#[test]
fn unranked_types_round_trip_plain_and_quantized_elements() {
    for source in [
        "tensor<*xi1>",
        "tensor<*xi4>",
        "tensor<*xui8>",
        "tensor<*xi64>",
        "tensor<*xf4E2M1FN>",
        "tensor<*xf8E5M2>",
        "tensor<*xbf16>",
        "tensor<*xf32>",
        "tensor<*xf64>",
        "tensor<*xcomplex<f32>>",
        "tensor<*xcomplex<f64>>",
        "tensor<*x!quant.uniform<i8:f32, 0.25:1>>",
        "tensor<*x!quant.uniform<i8:f32:12, {0.25:1, 0.5:2}>>",
    ] {
        let tensor = ty(source);
        assert!(matches!(tensor, Type::UnrankedTensor(_)), "{source}");
        tensor.verify().unwrap();
        assert_eq!(parse_type(&print_type(&tensor).unwrap()).unwrap(), tensor);
        assert!(
            custom_call(tensor.clone(), tensor, call_attributes()),
            "{source}"
        );
    }
    assert_eq!(
        ty("tensor<*xf32>"),
        Type::from(UnrankedTensorType::Plain(ElementType::F32))
    );
    let Type::QuantizedElement(element) = ty("!quant.uniform<i8:f32:12, {0.25:1, 0.5:2}>") else {
        panic!("expected canonical quantized element")
    };
    assert_eq!(
        ty("tensor<*x!quant.uniform<i8:f32:12, {0.25:1, 0.5:2}>>"),
        Type::from(UnrankedTensorType::Quantized(element))
    );
}

#[test]
fn unranked_tensors_reject_shapes_encodings_and_invalid_elements() {
    for source in [
        "tensor<*>",
        "tensor<*x>",
        "tensor<*x2xf32>",
        "tensor<?x*xf32>",
        "tensor<*xtensor<f32>>",
        "tensor<*x!stablehlo.token>",
        "tensor<*xf32, #stablehlo.bounds<>>",
        "tensor<*x!quant.uniform<i8:f32, 0.0>>",
        "tensor<*x!quant.uniform<i8:f32:0, {0.5:128}>>",
    ] {
        assert!(parse_type(source).is_err(), "{source}");
    }
}

#[test]
fn unranked_alias_chains_keep_the_shared_nesting_budget() {
    let mut source = String::from("!t0 = tensor<*xf32>\n");
    for index in 1..140 {
        source.push_str(&format!("!t{index} = !t{}\n", index - 1));
    }
    source.push_str("module {}");
    assert_eq!(
        parse_module(&source).unwrap_err().kind,
        stablehlo::text::TextErrorKind::NestingLimit
    );
}

#[test]
fn unranked_assembly_aliases_and_custom_calls_preserve_function_signatures() {
    let source = r#"
      !element = f32
      !plain = tensor<*x!element>
      !quant = !quant.uniform<i8:f32, 0.5>
      !unranked = tensor<*x!quant>
      func.func private @external(!plain) -> !plain
      func.func @main(%a: !plain, %q: !unranked) -> (!plain, !unranked) {
        %b = func.call @external(%a) : (!plain) -> !plain
        %c:2 = "stablehlo.custom_call"(%b, %q) {call_target_name = "external"}
          : (!plain, !unranked) -> (!plain, !unranked)
        return %c#0, %c#1 : !plain, !unranked
      }
    "#;
    let module = parse_module(source).unwrap();
    assert_eq!(
        parse_module(&print_module(&module).unwrap()).unwrap(),
        module
    );
}

#[test]
fn custom_call_tuples_allow_unranked_but_ranked_operation_domains_do_not() {
    let unranked = ty("tensor<*xf32>");
    let nested = Type::Tuple(vec![Type::Tuple(vec![unranked.clone()]), Type::Token]);
    assert!(custom_call(nested.clone(), nested, call_attributes()));
    for opcode in [
        StableHloOp::Add,
        StableHloOp::Multiply,
        StableHloOp::Compare,
    ] {
        assert!(
            computation(
                opcode,
                vec![unranked.clone(), unranked.clone()],
                vec![unranked.clone()],
                Attributes::new()
            )
            .is_err()
        );
    }
    assert!(
        computation(
            StableHloOp::Abs,
            vec![unranked.clone()],
            vec![unranked.clone()],
            Attributes::new()
        )
        .is_err()
    );
    assert!(
        computation(
            StableHloOp::Tuple,
            vec![unranked.clone()],
            vec![Type::Tuple(vec![unranked.clone()])],
            Attributes::new()
        )
        .is_err()
    );
    assert!(Type::Future(vec![unranked]).verify().is_err());
}

#[test]
fn custom_call_unranked_element_domains_still_apply() {
    for source in [
        "tensor<*xindex>",
        "tensor<*xsi32>",
        "tensor<*xtf32>",
        "tensor<*x!quant.uniform<i64:f32, 0.5>>",
    ] {
        let input = ty(source);
        assert!(
            !custom_call(input.clone(), input, call_attributes()),
            "{source}"
        );
    }
}

#[test]
fn unranked_layouts_have_no_rank_to_check_but_counts_and_ranked_layouts_are_checked() {
    let unranked = ty("tensor<*xf32>");
    for axes in [vec![], vec![0], vec![1, 0], vec![2, 0, 1]] {
        let layout: Attribute = ArrayOfLayout(vec![Layout(axes)]).into();
        let mut attributes = call_attributes();
        attributes.insert("operand_layouts".into(), layout.clone());
        attributes.insert("result_layouts".into(), layout);
        assert!(custom_call(
            unranked.clone(),
            unranked.clone(),
            attributes.clone()
        ));
        attributes.insert("result_layouts".into(), ArrayOfLayout(vec![]).into());
        assert!(!custom_call(unranked.clone(), unranked.clone(), attributes));
    }
    let mut attributes = call_attributes();
    attributes.insert(
        "operand_layouts".into(),
        ArrayOfLayout(vec![Layout(vec![1, 0])]).into(),
    );
    attributes.insert(
        "result_layouts".into(),
        ArrayOfLayout(vec![Layout(vec![1, 0])]).into(),
    );
    assert!(custom_call(
        unranked.clone(),
        ty("tensor<2x3xf32>"),
        attributes.clone()
    ));
    assert!(!custom_call(
        unranked.clone(),
        ty("tensor<2xf32>"),
        attributes.clone()
    ));
    attributes.remove("result_layouts");
    assert!(!custom_call(unranked.clone(), unranked, attributes));
}

#[test]
fn rank_refinement_and_join_are_order_independent_and_preserve_ranked_bounds() {
    let unranked = ty("tensor<*xf32>");
    let ranked = ty("tensor<?x2xf32, #stablehlo.bounds<5, ?>>");
    for inputs in [
        vec![unranked.clone(), ranked.clone()],
        vec![ranked.clone(), unranked.clone()],
    ] {
        assert_eq!(infer_most_specific_type(&inputs).unwrap(), ranked);
        assert_eq!(infer_least_specific_type(&inputs).unwrap(), unranked);
    }
    let inputs = vec![unranked.clone(), ranked, ty("tensor<3x2xf32>")];
    assert_eq!(
        infer_most_specific_type(&inputs).unwrap(),
        ty("tensor<3x2xf32>")
    );
    assert_eq!(infer_least_specific_type(&inputs).unwrap(), unranked);
    assert!(
        infer_most_specific_type(&[
            ty("tensor<*xf32>"),
            ty("tensor<2xf32>"),
            ty("tensor<3xf32>"),
        ])
        .is_err()
    );
    assert!(
        infer_most_specific_type(&[
            ty("tensor<*xf32>"),
            ty("tensor<?xf32, #stablehlo.bounds<2>>"),
            ty("tensor<3xf32>"),
        ])
        .is_err()
    );
}

#[test]
fn unranked_refinement_rejects_element_conflicts_and_recurses_through_tuples() {
    for (lhs, rhs) in [
        ("tensor<*xf32>", "tensor<2xf64>"),
        ("tensor<*xf32>", "tensor<*xf64>"),
        ("tensor<*xf32>", "!stablehlo.token"),
        ("tensor<*x!quant.uniform<i8:f32, 0.5>>", "tensor<2xi8>"),
        (
            "tensor<*x!quant.uniform<i8:f32, 0.5>>",
            "tensor<*x!quant.uniform<ui8:f32, 0.5>>",
        ),
    ] {
        let inputs = [ty(lhs), ty(rhs)];
        assert!(infer_most_specific_type(&inputs).is_err(), "{lhs}, {rhs}");
        assert!(infer_least_specific_type(&inputs).is_err(), "{lhs}, {rhs}");
    }
    let lhs = ty("tuple<tensor<*x!quant.uniform<i8:f32, 0.5:1>>, !stablehlo.token>");
    let rhs = ty("tuple<tensor<2x!quant.uniform<i8:f32, 0.25:2>>, !stablehlo.token>");
    assert_eq!(
        infer_most_specific_type(&[lhs.clone(), rhs.clone()]).unwrap(),
        rhs
    );
    assert_eq!(infer_least_specific_type(&[lhs.clone(), rhs]).unwrap(), lhs);
}
