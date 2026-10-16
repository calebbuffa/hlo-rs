use stablehlo::text::{parse_attribute, parse_module, print_attribute, print_module};
use stablehlo::{
    AffineBinaryOp as Op, AffineConstraint, AffineError, AffineExpr as Expr, AffineMap, Attribute,
    IntegerSet,
};

mod upstream_support;

fn map(text: &str) -> AffineMap {
    let Attribute::AffineMap(map) = parse_attribute(text).unwrap() else {
        panic!("expected map")
    };
    map
}

fn set(text: &str) -> IntegerSet {
    let Attribute::IntegerSet(set) = parse_attribute(text).unwrap() else {
        panic!("expected set")
    };
    set
}

#[test]
fn maps_have_positional_dimensions_symbols_and_structured_expression_trees() {
    let value = map(
        "affine_map<(row, col)[stride] -> (col, row * stride + col, row floordiv 4, row mod 4)>",
    );
    assert_eq!((value.dimensions, value.symbols), (2, 1));
    assert_eq!(value.results[0], Expr::Dimension(1));
    assert_eq!(value.evaluate(&[11, 2], &[8]).unwrap(), [2, 90, 2, 3]);
    let attribute = Attribute::AffineMap(value);
    assert_eq!(
        parse_attribute(&print_attribute(&attribute).unwrap()).unwrap(),
        attribute
    );
}

#[test]
fn map_grammar_supports_empty_domains_ranges_comments_keywords_and_hex_constants() {
    assert_eq!(map("affine_map<() -> ()>").evaluate(&[], &[]).unwrap(), []);
    assert_eq!(
        map("affine_map<()[] -> (0xA, -0xA)>")
            .evaluate(&[], &[])
            .unwrap(),
        [10, -10]
    );
    assert_eq!(
        map("affine_map<()[width] -> (width ceildiv 4)>")
            .evaluate(&[], &[9])
            .unwrap(),
        [3]
    );
    assert_eq!(
        map("affine_map<(mod, floor, i32)[ceildiv] -> (mod + floor + i32 + ceildiv)>")
            .evaluate(&[1, 2, 3], &[4])
            .unwrap(),
        [10]
    );
    assert_eq!(
        map("affine_map<(row) // domain\n -> (row // result\n + 2)>")
            .evaluate(&[3], &[])
            .unwrap(),
        [5]
    );
    assert_eq!(
        map("affine_map<(floordiv_value)[modulus] -> (floordiv_value mod modulus)>")
            .evaluate(&[11], &[4])
            .unwrap(),
        [3]
    );
}

#[test]
fn precedence_left_associativity_parentheses_and_unary_negation_are_preserved() {
    let value = map(
        "affine_map<(x)[s] -> (x + 2 * 3, x * 2 + 3, x - 2 - 3, x floordiv 2 * 3, x * 2 floordiv 3, x mod 5 mod 3, -x * 2 + 3, -(x + 2), --x, x * (s + 1))>",
    );
    assert_eq!(
        value.evaluate(&[11], &[4]).unwrap(),
        [17, 25, 6, 15, 7, 1, -19, -13, 11, 55]
    );
    let attribute = Attribute::AffineMap(value);
    assert_eq!(
        parse_attribute(&print_attribute(&attribute).unwrap()).unwrap(),
        attribute
    );
}

#[test]
fn floor_ceil_and_modulo_match_mathematical_division_for_both_signs() {
    let value = map("affine_map<(x)[s] -> (x floordiv s, x ceildiv s, x mod s)>");
    for x in -32_i64..=32 {
        for s in -9_i64..=9 {
            if s == 0 {
                assert_eq!(value.evaluate(&[x], &[s]), Err(AffineError::DivisionByZero));
                continue;
            }
            let expected_floor = (-40..=40)
                .find(|&q| {
                    if s > 0 {
                        q * s <= x && (q + 1) * s > x
                    } else {
                        q * s >= x && (q + 1) * s < x
                    }
                })
                .unwrap();
            let expected_ceil = if x == expected_floor * s {
                expected_floor
            } else {
                expected_floor + 1
            };
            assert_eq!(
                value.evaluate(&[x], &[s]).unwrap(),
                [expected_floor, expected_ceil, x - expected_floor * s],
                "{x} {s}"
            );
        }
    }
}

#[test]
fn symbolic_products_and_divisors_are_accepted_but_dimension_products_are_not() {
    let value = map(
        "affine_map<(x)[a,b] -> (x * (a * b), x floordiv (a + b), x ceildiv (a * b), x mod (a - b))>",
    );
    assert_eq!(value.evaluate(&[23], &[4, 2]).unwrap(), [184, 3, 3, 1]);
    for source in [
        "affine_map<(x,y) -> (x*y)>",
        "affine_map<(x,y) -> (x floordiv y)>",
        "affine_map<(x,y) -> (x ceildiv (y+1))>",
        "affine_map<(x,y) -> (x mod (y*2))>",
        "affine_map<(x,y)[s] -> ((x*s)*y)>",
    ] {
        assert!(parse_attribute(source).is_err(), "{source}");
    }
}

#[test]
fn integer_sets_normalize_relations_and_test_all_constraints() {
    let value = set("affine_set<(x,y)[n] : (0 <= x, x < = n - 1, y == x + 2, y mod 2 >= 0)>");
    for x in -2..=8 {
        for y in -2..=10 {
            assert_eq!(
                value.contains(&[x, y], &[5]).unwrap(),
                (0..5).contains(&x) && y == x + 2
            );
        }
    }
    let attribute = Attribute::IntegerSet(value);
    assert_eq!(
        parse_attribute(&print_attribute(&attribute).unwrap()).unwrap(),
        attribute
    );
    let empty = set("affine_set<(x)[s] : ()>");
    assert_eq!(
        empty.constraints,
        [AffineConstraint {
            expression: Expr::Constant(0),
            equality: true
        }]
    );
    assert!(empty.contains(&[-123], &[0]).unwrap());
    let undefined = set("affine_set<() : (-1 >= 0, 1 floordiv 0 == 0)>");
    assert_eq!(
        undefined.contains(&[], &[]),
        Err(AffineError::DivisionByZero)
    );
}

#[test]
fn invalid_bindings_relations_and_trailing_syntax_are_rejected() {
    for source in [
        "affine_map<(x,x) -> (x)>",
        "affine_map<(x)[x] -> (x)>",
        "affine_map<(x)[s,s] -> (x)>",
        "affine_map<(x) -> (y)>",
        "affine_map<(0) -> (0)>",
        "affine_map<(x,) -> (x)>",
        "affine_map<(x) -> (x,)>",
        "affine_map<(x) -> ()> garbage",
        "affine_map<(x) -> (+x)>",
        "affine_map<(x) -> (x+)>",
        "affine_map<(x) -> (x**2)>",
        "affine_map<(x) -> (())>",
        "affine_map<(x) -> (x / 2)>",
        "affine_map<(x) -> (x floordiv_value)>",
        "affine_map<(x) -> (x mod2)>",
        "affine_map<(x) -> (x >= 0)>",
        "affine_map<(x) : (x >= 0)>",
        "affine_set<(x) -> (x)>",
        "affine_set<(x) : (x > 0)>",
        "affine_set<(x) : (x < 0)>",
        "affine_set<(x) : (x = 0)>",
        "affine_set<(x) : (x != 0)>",
        "affine_set<(x) : (x >= 0,)>",
        "affine_map<(x) -> (%x)>",
        "affine_map<() -> (9223372036854775809)>",
        "affine_map<() -> (-9223372036854775809)>",
    ] {
        assert!(parse_attribute(source).is_err(), "{source}");
    }
}

#[test]
fn signed_minimum_constants_roundtrip_including_upstream_printer_magnitudes() {
    let value = map("affine_map<(x) -> (-9223372036854775808, x - 9223372036854775808)>");
    assert_eq!(value.evaluate(&[0], &[]).unwrap(), [i64::MIN, i64::MIN]);
    let attribute = Attribute::AffineMap(value);
    assert_eq!(
        parse_attribute(&print_attribute(&attribute).unwrap()).unwrap(),
        attribute
    );
}

#[test]
fn programmatic_validation_and_evaluation_return_explicit_errors() {
    let invalid = AffineMap {
        dimensions: 1,
        symbols: 1,
        results: vec![Expr::Dimension(1)],
    };
    assert_eq!(
        invalid.verify(),
        Err(AffineError::DimensionOutOfBounds {
            position: 1,
            dimensions: 1
        })
    );
    assert!(print_attribute(&Attribute::AffineMap(invalid)).is_err());
    assert_eq!(
        Expr::Symbol(1).verify(1, 1),
        Err(AffineError::SymbolOutOfBounds {
            position: 1,
            symbols: 1
        })
    );
    assert_eq!(
        Expr::binary(Op::Mul, Expr::Dimension(0), Expr::Dimension(1)).verify(2, 0),
        Err(AffineError::NonAffineProduct)
    );
    for op in [Op::FloorDiv, Op::CeilDiv, Op::Mod] {
        assert_eq!(
            Expr::binary(op, Expr::Constant(1), Expr::Dimension(0)).verify(1, 0),
            Err(AffineError::NonSymbolicDivisor { op })
        );
    }
    let value = map("affine_map<(x)[s] -> (x+s)>");
    assert_eq!(
        value.evaluate(&[], &[1]),
        Err(AffineError::ArgumentCount {
            symbols: false,
            expected: 1,
            actual: 0
        })
    );
    assert_eq!(
        value.evaluate(&[1], &[]),
        Err(AffineError::ArgumentCount {
            symbols: true,
            expected: 1,
            actual: 0
        })
    );
    assert_eq!(
        value.evaluate(&[i64::MAX], &[1]),
        Err(AffineError::ArithmeticOverflow)
    );
    let product = map("affine_map<()[x] -> (x*x*x)>");
    assert_eq!(
        product.evaluate(&[], &[i64::MAX]),
        Err(AffineError::ArithmeticOverflow)
    );
    let restored = map("affine_map<()[x] -> ((x*x) floordiv x)>");
    assert_eq!(restored.evaluate(&[], &[i64::MAX]).unwrap(), [i64::MAX]);
}

#[test]
fn affine_metadata_survives_aliases_nested_attributes_locations_and_modules() {
    let source = r#"
        #map = affine_map<(row)[stride] -> (row * stride)>
        #set = affine_set<(row) : (row >= 0)>
        #attrs = {map = #map, set = #set}
        module attributes {test.maps = [#map, #attrs]} {
            func.func @identity(%x: tensor<f32>) -> tensor<f32> attributes {test.set = #set} {
                return %x : tensor<f32> loc(fused<#attrs>["source"])
            }
        }
    "#;
    let module = parse_module(source).unwrap();
    let emitted = print_module(&module).unwrap();
    let reparsed = parse_module(&emitted).unwrap();
    assert_eq!(module, reparsed);
    assert!(emitted.contains("affine_map<"));
    assert!(emitted.contains("affine_set<"));
}

#[test]
fn affine_buffer_layouts_keep_maps_and_memory_spaces_and_validate_domain_rank() {
    use stablehlo::text::{parse_type, print_type};
    for source in [
        "memref<3x4xf32, affine_map<(row,col) -> (col,row)>>",
        "memref<?x4xf32, affine_map<(row,col)[stride,offset] -> (row*stride+col+offset)>, 3>",
        "memref<16xf32, affine_map<(x) -> (x floordiv 4, x mod 4)>>",
        "memref<f32, affine_map<()[offset] -> (offset)>>",
        "memref<f32, affine_map<() -> ()>>",
        "memref<3x4xf32, affine_map<(row,col) -> ()>>",
    ] {
        let ty = parse_type(source).unwrap();
        assert_eq!(parse_type(&print_type(&ty).unwrap()).unwrap(), ty);
        let stablehlo::Type::Buffer(buffer) = ty else {
            unreachable!()
        };
        assert!(matches!(
            buffer.layout(),
            Some(stablehlo::BufferLayout::Affine(_))
        ));
    }
    let module=parse_module("#layout = affine_map<(x,y) -> (y,x)>\n!buffer = memref<3x4xf32, #layout, 2>\nfunc.func private @external(!buffer) -> !buffer").unwrap();
    assert_eq!(
        parse_module(&print_module(&module).unwrap()).unwrap(),
        module
    );
    for source in [
        "memref<3x4xf32, affine_map<(x) -> (x)>>",
        "memref<3xf32, affine_map<(x,y) -> (x,y)>>",
        "memref<f32, affine_map<(x) -> (0)>>",
    ] {
        assert!(parse_type(source).is_err(), "{source}");
    }
    let invalid = stablehlo::BufferType::new(stablehlo::ElementType::F32, vec![Some(3)])
        .with_layout(stablehlo::BufferLayout::Affine(AffineMap {
            dimensions: 1,
            symbols: 0,
            results: vec![Expr::Symbol(0)],
        }));
    assert!(stablehlo::Type::Buffer(invalid).verify().is_err());
}

#[test]
fn tensor_and_quantized_encodings_cannot_hide_invalid_affine_attributes() {
    let invalid = Attribute::AffineMap(AffineMap {
        dimensions: 1,
        symbols: 0,
        results: vec![Expr::Dimension(1)],
    });
    let plain: stablehlo::Type =
        stablehlo::TensorType::static_shape(stablehlo::ElementType::F32, [4])
            .with_encoding(invalid.clone())
            .into();
    assert!(plain.verify().is_err());
    let stablehlo::Type::QuantizedTensor(quantized) =
        stablehlo::text::parse_type("tensor<4x!quant.uniform<i8:f32, 0.25:0>>").unwrap()
    else {
        unreachable!()
    };
    assert!(
        stablehlo::Type::QuantizedTensor(quantized.with_encoding(invalid))
            .verify()
            .is_err()
    );
}

#[test]
fn parser_and_programmatic_trees_enforce_nesting_limits() {
    for expression in [
        format!("{}x{}", "(".repeat(150), ")".repeat(150)),
        format!("{}x", "-".repeat(150)),
        std::iter::repeat_n("x", 150).collect::<Vec<_>>().join("+"),
    ] {
        assert!(parse_attribute(&format!("affine_map<(x) -> ({expression})>")).is_err());
    }
    let mut expression = Expr::Dimension(0);
    for _ in 0..150 {
        expression = Expr::binary(Op::Add, expression, Expr::Constant(1));
    }
    let value = Attribute::AffineMap(AffineMap {
        dimensions: 1,
        symbols: 0,
        results: vec![expression],
    });
    assert!(value.verify().is_err());
    assert!(print_attribute(&value).is_err());
    let enormous = Attribute::AffineMap(AffineMap {
        dimensions: usize::MAX,
        symbols: 1,
        results: vec![],
    });
    assert_eq!(
        print_attribute(&enormous).unwrap_err().kind,
        stablehlo::text::TextErrorKind::ResourceLimit
    );
}

#[test]
fn affine_expression_parser_and_printer_handle_deep_trees_iteratively() {
    let nested = format!("{}x{}", "(".repeat(100), ")".repeat(100));
    let parsed = Attribute::AffineMap(map(&format!("affine_map<(x) -> ({nested})>")));
    let printed = print_attribute(&parsed).unwrap();
    assert_eq!(parse_attribute(&printed).unwrap(), parsed);

    let mut expression = Expr::Dimension(0);
    for _ in 0..100 {
        expression = Expr::binary(Op::Add, expression, Expr::Constant(1));
    }
    let value = Attribute::AffineMap(AffineMap {
        dimensions: 1,
        symbols: 0,
        results: vec![expression],
    });
    let printed = print_attribute(&value).unwrap();
    assert_eq!(parse_attribute(&printed).unwrap(), value);
}

#[test]
#[ignore = "requires STABLEHLO_UPSTREAM_PYTHON with pinned jaxlib"]
fn affine_maps_and_sets_roundtrip_through_independent_upstream_with_matching_values() {
    use upstream_support::UpstreamResult;
    let mut originals = Vec::new();
    for lhs in ["x", "s", "x+s", "x-s", "-x", "x*3"] {
        for rhs in ["1", "2", "-3", "s", "s+1"] {
            for op in ["+", "-", "*", "floordiv", "ceildiv", "mod"] {
                let source = format!(
                    "module attributes {{test.map = affine_map<(x)[s] -> (({lhs}) {op} ({rhs}))>, test.set = affine_set<(x)[s] : (x >= -2, x <= s + 3, x mod 2 == 0)>}} {{}}"
                );
                originals.push(parse_module(&source).unwrap());
            }
        }
    }
    let emitted = originals
        .iter()
        .map(|module| print_module(module).unwrap())
        .collect::<Vec<_>>();
    let results = upstream_support::verify(&emitted);
    for ((original, result), source) in originals.iter().zip(results).zip(&emitted) {
        let text = match result {
            UpstreamResult::Accepted(text) => text,
            UpstreamResult::Rejected(error) => panic!("upstream rejected {source}: {error}"),
        };
        let returned = parse_module(&text).unwrap_or_else(|error| panic!("{error}\n{text}"));
        let Attribute::AffineMap(before) = &original.attributes()["test.map"] else {
            unreachable!()
        };
        let Attribute::AffineMap(after) = &returned.attributes()["test.map"] else {
            unreachable!()
        };
        let Attribute::IntegerSet(before_set) = &original.attributes()["test.set"] else {
            unreachable!()
        };
        let Attribute::IntegerSet(after_set) = &returned.attributes()["test.set"] else {
            unreachable!()
        };
        for x in -5..=5 {
            for s in 1..=4 {
                assert_eq!(
                    before.evaluate(&[x], &[s]),
                    after.evaluate(&[x], &[s]),
                    "{source}\n{text}"
                );
                assert_eq!(
                    before_set.contains(&[x], &[s]),
                    after_set.contains(&[x], &[s]),
                    "{source}\n{text}"
                );
            }
        }
    }
    let invalid = [
        "affine_map<(x,x) -> (x)>",
        "affine_map<(x)[x] -> (x)>",
        "affine_map<(x) -> (y)>",
        "affine_map<(x,y) -> (x*y)>",
        "affine_map<(x,y) -> (x floordiv y)>",
        "affine_map<(x,y) -> (x ceildiv y)>",
        "affine_map<(x,y) -> (x mod y)>",
        "affine_set<(x) : (x > 0)>",
        "affine_set<(x) : (x = 0)>",
        "affine_map<() -> (9223372036854775809)>",
    ];
    let invalid_modules = invalid
        .iter()
        .map(|value| format!("module attributes {{test.value = {value}}} {{}}"))
        .collect::<Vec<_>>();
    for (source, result) in invalid_modules
        .iter()
        .zip(upstream_support::verify(&invalid_modules))
    {
        assert!(
            matches!(result, UpstreamResult::Rejected(_)),
            "upstream accepted {source}"
        );
        assert!(parse_module(source).is_err(), "Rust accepted {source}");
    }
    let layouts = [
        "memref<3x4xf32, affine_map<(x,y) -> (y,x)>>",
        "memref<3x4xf32, affine_map<(x,y) -> (x,y)>>",
        "memref<?x4xf32, affine_map<(x,y)[s,t] -> (x*s+y+t)>, 3>",
        "memref<16xf32, affine_map<(x) -> (x floordiv 4, x mod 4)>>",
        "memref<f32, affine_map<()[s] -> (s)>>",
        "memref<f32, affine_map<() -> ()>>",
        "memref<3x4xf32, affine_map<(x,y) -> ()>>",
    ];
    let sources = layouts
        .iter()
        .map(|layout| format!("module attributes {{test.buffer = {layout}}} {{}}"))
        .collect::<Vec<_>>();
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
    eprintln!(
        "180 affine map/set modules accepted and reprinted by independent MLIR; finite-coordinate evaluations agree after Rust reparsing. Seven affine buffer layouts roundtrip. 10 malformed modules rejected by both parsers. Not exhaustive affine or StableHLO execution conformance."
    );
}

#[test]
fn generated_expression_roundtrips_preserve_small_coordinate_semantics() {
    for lhs in ["x", "s", "x+s", "x-s", "-x", "x*3"] {
        for rhs in ["1", "2", "-3", "s", "s+1"] {
            for op in ["+", "-", "*", "floordiv", "ceildiv", "mod"] {
                let value = map(&format!("affine_map<(x)[s] -> (({lhs}) {op} ({rhs}))>"));
                let printed = print_attribute(&Attribute::AffineMap(value.clone())).unwrap();
                let reparsed = map(&printed);
                assert_eq!(value, reparsed, "{printed}");
                for x in -5..=5 {
                    for s in -3..=3 {
                        assert_eq!(value.evaluate(&[x], &[s]), reparsed.evaluate(&[x], &[s]));
                    }
                }
            }
        }
    }
}
