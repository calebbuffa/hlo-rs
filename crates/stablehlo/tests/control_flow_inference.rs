use stablehlo::text::{parse_module, parse_type, print_module};
use stablehlo::{
    InferenceError, Type, infer_case_types, infer_get_tuple_element_type, infer_if_types,
    infer_optimization_barrier_types, infer_tuple_type, infer_while_types,
};

mod upstream_support;

fn ty(source: &str) -> Type {
    parse_type(source).unwrap()
}

fn branch_source(branches: &[&str], output: &str, is_if: bool) -> String {
    let selector = if is_if { "tensor<i1>" } else { "tensor<i32>" };
    let parameters = branches
        .iter()
        .enumerate()
        .map(|(i, ty)| format!(",%x{i}:{ty}"))
        .collect::<String>();
    let regions = branches
        .iter()
        .enumerate()
        .map(|(i, ty)| format!("{{\"stablehlo.return\"(%x{i}):({ty})->()}}"))
        .collect::<Vec<_>>()
        .join(",");
    let name = if is_if { "if" } else { "case" };
    format!(
        "func.func @branch(%p:{selector}{parameters})->{output} {{ \
         %r=\"stablehlo.{name}\"(%p)({regions}):({selector})->{output} \
         return %r:{output} }}"
    )
}

fn branch_module(branches: &[&str], output: &str, is_if: bool) -> bool {
    let source = branch_source(branches, output, is_if);
    let Ok(module) = parse_module(&source) else {
        return false;
    };
    assert_eq!(
        parse_module(&print_module(&module).unwrap()).unwrap(),
        module
    );
    true
}

#[test]
#[ignore = "requires STABLEHLO_UPSTREAM_PYTHON with pinned jaxlib"]
fn conversion_and_conditional_signatures_roundtrip_through_independent_upstream() {
    use upstream_support::UpstreamResult;
    let mut sources = Vec::new();
    for left in [
        "tensor<0xf32>",
        "tensor<2xf32>",
        "tensor<5xf32>",
        "tensor<?xf32>",
        "tensor<?xf32,#stablehlo.bounds<0>>",
        "tensor<?xf32,#stablehlo.bounds<2>>",
        "tensor<?xf32,#stablehlo.bounds<5>>",
    ] {
        for right in [
            "tensor<0xf32>",
            "tensor<2xf32>",
            "tensor<5xf32>",
            "tensor<?xf32>",
            "tensor<?xf32,#stablehlo.bounds<0>>",
            "tensor<?xf32,#stablehlo.bounds<2>>",
            "tensor<?xf32,#stablehlo.bounds<5>>",
        ] {
            let Ok(inferred) = infer_if_types(&ty("tensor<i1>"), &[ty(left)], &[ty(right)]) else {
                continue;
            };
            let output = stablehlo::text::print_type(&inferred[0]).unwrap();
            for is_if in [true, false] {
                sources.push(branch_source(&[left, right], &output, is_if));
            }
        }
    }
    for (input, output) in [
        ("tensor<3xf32>", "tensor<?xi32>"),
        ("tensor<?xf32,#stablehlo.bounds<9>>", "tensor<3xi32>"),
        ("tensor<0x3xi8>", "tensor<0x3xcomplex<f64>>"),
        ("tensor<3xcomplex<f32>>", "tensor<3xi1>"),
    ] {
        sources.push(format!(
            "func.func @convert(%x:{input})->{output} {{ \
             %r=\"stablehlo.convert\"(%x):({input})->{output} \
             return %r:{output} }}"
        ));
    }
    let emitted = sources
        .iter()
        .map(|s| print_module(&parse_module(s).unwrap()).unwrap())
        .collect::<Vec<_>>();
    for (source, result) in emitted.iter().zip(upstream_support::verify(&emitted)) {
        let text = match result {
            UpstreamResult::Accepted(text) => text,
            UpstreamResult::Rejected(error) => panic!("upstream rejected {source}: {error}"),
        };
        parse_module(&text).unwrap_or_else(|error| panic!("{error}\n{text}"));
    }
    let invalid = [
        branch_source(&["tensor<3xf32>", "tensor<4xf32>"], "tensor<?xf32>", true),
        branch_source(&["tensor<3xf32>", "tensor<3xi32>"], "tensor<?xf32>", false),
        branch_source(&["tensor<3xf32>", "tensor<3xf32>"], "tensor<4xf32>", true),
        "func.func @convert(%x:tensor<3xf32>)->tensor<4xi32> { \
         %r=\"stablehlo.convert\"(%x):(tensor<3xf32>)->tensor<4xi32> \
         return %r:tensor<4xi32> }"
            .into(),
    ];
    for (source, result) in invalid.iter().zip(upstream_support::verify(&invalid)) {
        assert!(matches!(result, UpstreamResult::Rejected(_)), "{source}");
        assert!(parse_module(source).is_err(), "{source}");
    }
    eprintln!(
        "{} conversion/conditional modules roundtrip through independent StableHLO 1.20.0; four malformed signatures rejected by both parsers. Not target-1.22 or execution-semantic conformance.",
        emitted.len()
    );
}

#[test]
fn if_and_case_join_shapes_and_bounds_without_refining_branch_metadata() {
    let predicate = ty("tensor<i1>");
    let index = ty("tensor<i32>");
    let left = ty("tensor<3x2xf32>");
    let right = ty("tensor<?x2xf32,#stablehlo.bounds<5,?>>");
    let expected = vec![right.clone(), Type::Token];
    let first = vec![left.clone(), Type::Token];
    let second = vec![right.clone(), Type::Token];
    assert_eq!(
        infer_if_types(&predicate, &first, &second).unwrap(),
        expected
    );
    assert_eq!(
        infer_case_types(&index, &[first.clone(), second.clone(), first]).unwrap(),
        expected
    );
    assert_eq!(
        infer_case_types(&index, std::slice::from_ref(&second)).unwrap(),
        second
    );
    assert_eq!(infer_if_types(&predicate, &[], &[]).unwrap(), vec![]);
    assert_eq!(infer_case_types(&index, &[vec![]]).unwrap(), vec![]);
    for is_if in [true, false] {
        assert!(branch_module(
            &["tensor<3x2xf32>", "tensor<?x2xf32,#stablehlo.bounds<5,?>>"],
            "tensor<?x2xf32,#stablehlo.bounds<5,?>>",
            is_if
        ));
        assert!(!branch_module(
            &["tensor<3xf32>", "tensor<4xf32>"],
            "tensor<?xf32>",
            is_if
        ));
    }
}

#[test]
fn conditional_inference_preserves_first_branch_quantization_and_sparse_metadata() {
    let first = ty("tensor<3x!quant.uniform<i8:f32,0.25:1>>");
    let second = ty("tensor<3x!quant.uniform<i8:f32,0.5:2>>");
    assert_eq!(
        infer_if_types(&ty("tensor<i1>"), std::slice::from_ref(&first), &[second]).unwrap(),
        vec![first]
    );
    let sparse = ty("tensor<3xf32,#sparse_tensor.encoding<{map=(i)->(i:compressed)}>>");
    let dense = ty("tensor<3xf32>");
    assert_eq!(
        infer_if_types(
            &ty("tensor<i1>"),
            std::slice::from_ref(&sparse),
            std::slice::from_ref(&dense)
        )
        .unwrap(),
        vec![sparse.clone()]
    );
    assert_eq!(
        infer_if_types(&ty("tensor<i1>"), std::slice::from_ref(&dense), &[sparse]).unwrap(),
        vec![dense]
    );
}

#[test]
fn conditional_inference_rejects_bad_selectors_arities_domains_and_shapes() {
    assert_eq!(
        infer_case_types(&ty("tensor<i32>"), &[]),
        Err(InferenceError::EmptyBranches)
    );
    assert_eq!(
        infer_if_types(&ty("tensor<i1>"), &[Type::Token], &[]),
        Err(InferenceError::BranchResultCountMismatch {
            branch: 1,
            expected: 1,
            actual: 0,
        })
    );
    for selector in ["tensor<1xi1>", "tensor<i32>", "tensor<ui32>", "tensor<f32>"] {
        assert!(infer_if_types(&ty(selector), &[], &[]).is_err());
    }
    for selector in ["tensor<1xi32>", "tensor<i1>", "tensor<i64>", "tensor<ui32>"] {
        assert!(infer_case_types(&ty(selector), &[vec![]]).is_err());
    }
    for (left, right) in [
        (ty("tensor<3xf32>"), ty("tensor<4xf32>")),
        (ty("tensor<3xf32>"), ty("tensor<3xi32>")),
        (
            ty("tensor<4xf32>"),
            ty("tensor<?xf32,#stablehlo.bounds<3>>"),
        ),
        (ty("tensor<3xf32>"), ty("tensor<3x1xf32>")),
        (Type::Token, ty("tensor<f32>")),
        (Type::Tuple(vec![]), Type::Tuple(vec![])),
        (ty("memref<3xf32>"), ty("memref<3xf32>")),
        (ty("tensor<*xf32>"), ty("tensor<*xf32>")),
    ] {
        assert!(infer_if_types(&ty("tensor<i1>"), &[left], &[right]).is_err());
    }
}

#[test]
fn tuple_construction_and_projection_preserve_nested_value_metadata() {
    let operands = vec![
        Type::Token,
        ty("tensor<?xf32,#stablehlo.bounds<7>>"),
        ty("memref<4xf32,strided<[2],offset:1>>"),
        Type::Tuple(vec![ty("tensor<2x!quant.uniform<i8:f32,0.5>>")]),
    ];
    let result = infer_tuple_type(&operands).unwrap();
    assert_eq!(result, Type::Tuple(operands.clone()));
    for (i, expected) in operands.iter().enumerate() {
        assert_eq!(
            infer_get_tuple_element_type(&result, i as i32).unwrap(),
            *expected
        );
    }
    assert_eq!(infer_tuple_type(&[]).unwrap(), Type::Tuple(vec![]));
    for index in [i32::MIN, -1, 4, i32::MAX] {
        assert_eq!(
            infer_get_tuple_element_type(&result, index),
            Err(InferenceError::TupleIndexOutOfBounds { index, size: 4 })
        );
    }
    assert!(infer_get_tuple_element_type(&Type::Token, 0).is_err());
    for invalid in [
        ty("tensor<*xf32>"),
        Type::Element(stablehlo::ElementType::F32),
    ] {
        assert!(infer_tuple_type(&[invalid]).is_err());
    }
}

#[test]
fn while_and_barrier_inference_are_identity_rules_with_distinct_operand_domains() {
    let tensors = vec![
        ty("tensor<?xf32,#stablehlo.bounds<7>>"),
        ty("tensor<3x!quant.uniform<i8:f32,0.5>>"),
        Type::Token,
    ];
    assert_eq!(infer_while_types(&tensors).unwrap(), tensors);
    assert_eq!(infer_optimization_barrier_types(&tensors).unwrap(), tensors);
    let buffer = ty("memref<4xf32>");
    assert_eq!(
        infer_while_types(std::slice::from_ref(&buffer)).unwrap(),
        vec![buffer.clone()]
    );
    assert!(infer_optimization_barrier_types(&[buffer]).is_err());
    for operands in [
        vec![Type::Tuple(vec![])],
        vec![ty("tensor<*xf32>")],
        vec![Type::Element(stablehlo::ElementType::F32)],
    ] {
        assert!(infer_while_types(&operands).is_err());
        assert!(infer_optimization_barrier_types(&operands).is_err());
    }
    assert_eq!(infer_while_types(&[]).unwrap(), vec![]);
    assert_eq!(infer_optimization_barrier_types(&[]).unwrap(), vec![]);
}

#[test]
fn generated_branch_joins_agree_with_extent_compatibility_and_bound_union() {
    let witnesses = [
        (Some(0), None),
        (Some(2), None),
        (Some(5), None),
        (None, None),
        (None, Some(0)),
        (None, Some(2)),
        (None, Some(5)),
    ];
    let source = |size: Option<u64>, bound: Option<u64>| {
        let size = size.map_or_else(|| "?".into(), |size| size.to_string());
        let encoding =
            bound.map_or_else(String::new, |bound| format!(",#stablehlo.bounds<{bound}>"));
        format!("tensor<{size}xf32{encoding}>")
    };
    for (left_size, left_bound) in witnesses {
        for (right_size, right_bound) in witnesses {
            let left = source(left_size, left_bound);
            let right = source(right_size, right_bound);
            let compatible = !matches!((left_size,right_size), (Some(l),Some(r)) if l != r)
                && !matches!((left_size,right_bound), (Some(l),Some(r)) if l > r)
                && !matches!((right_size,left_bound), (Some(r),Some(l)) if r > l);
            let inferred = infer_if_types(&ty("tensor<i1>"), &[ty(&left)], &[ty(&right)]);
            assert_eq!(inferred.is_ok(), compatible, "{left}, {right}");
            let case = infer_case_types(&ty("tensor<i32>"), &[vec![ty(&left)], vec![ty(&right)]]);
            assert_eq!(inferred, case);
            let output = if let Ok(inferred) = inferred {
                let size = if left_size == right_size {
                    left_size
                } else {
                    None
                };
                let bound = if size.is_none() {
                    left_size
                        .or(left_bound)
                        .zip(right_size.or(right_bound))
                        .map(|(l, r)| l.max(r))
                } else {
                    None
                };
                let output = source(size, bound);
                assert_eq!(inferred, vec![ty(&output)]);
                output
            } else {
                "tensor<?xf32>".into()
            };
            for is_if in [true, false] {
                assert_eq!(
                    branch_module(&[&left, &right], &output, is_if),
                    compatible,
                    "{left}, {right} -> {output}"
                );
            }
        }
    }
}
