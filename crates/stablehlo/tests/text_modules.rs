use stablehlo::{
    Attribute, Attributes, Block, BlockArgument, Computation, ElementType, FunctionType,
    Instruction, Module, Operation, Region, StableHloOp, TensorType, Type, ValueId,
    text::{TextErrorKind, parse_module, print_module},
};

fn scalar(element: ElementType) -> Type {
    TensorType::scalar(element).into()
}

fn round_trip(module: &Module) -> Module {
    let text = print_module(module).unwrap();
    let parsed = parse_module(&text).unwrap_or_else(|error| panic!("{text}\n{error}"));
    parsed.verify().unwrap();
    let canonical = print_module(&parsed).unwrap();
    assert_eq!(
        print_module(&parse_module(&canonical).unwrap()).unwrap(),
        canonical
    );
    parsed
}

#[test]
fn empty_modules_named_modules_and_consumer_entry_metadata_round_trip() {
    for name in ["", "module", "module with spaces"] {
        let original = Module::new(name);
        assert_eq!(round_trip(&original), original);
    }
    let mut original = Module::new("named");
    original
        .set_attributes(
            [(
                "consumer.info".into(),
                Attribute::String("preserved".into()),
            )]
            .into(),
        )
        .unwrap();
    let function = Computation::try_new_with_roots("empty", vec![], vec![]).unwrap();
    let entry = original.add_computation(function);
    original.set_entry(entry).unwrap();
    assert_eq!(round_trip(&original), original);
}

#[test]
fn generic_module_fixture_defines_typed_ssa_and_preserves_function_metadata() {
    let input = r#"
      "builtin.module"() ({
        "func.func"() ({
          ^entry(%x: tensor<2xf32>, %y: tensor<2xf32>):
            %sum = "stablehlo.add"(%x, %y) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
            "func.return"(%sum) : (tensor<2xf32>) -> ()
        }) {sym_name = "add", function_type = (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>,
            sym_visibility = "public", arg_attrs = [{consumer.name = "x"}, {}],
            res_attrs = [{}]} : () -> ()
      }) {sym_name = "example"} : () -> ()
    "#;
    let module = parse_module(input).unwrap();
    assert_eq!(module.name(), "example");
    let function = &module.computations()[0];
    assert_eq!(function.name(), "add");
    assert_eq!(function.instructions()[2].operation().name(), "add");
    assert_eq!(
        function.attributes()["sym_visibility"],
        Attribute::String("public".into())
    );
    assert_eq!(round_trip(&module), module);
}

#[test]
fn declarations_and_multiple_result_calls_round_trip_without_parallel_ir() {
    let input = r#"
      "builtin.module"() ({
        "func.func"() ({
          %pair:2 = "func.call"() {callee = @external} : () -> (tensor<f32>, !stablehlo.token)
          "func.return"(%pair#0, %pair#1) : (tensor<f32>, !stablehlo.token) -> ()
        }) {sym_name = "main", function_type = () -> (tensor<f32>, !stablehlo.token)} : () -> ()
        "func.func"() ({}) {sym_name = "external", sym_visibility = "private",
            function_type = () -> (tensor<f32>, !stablehlo.token)} : () -> ()
      }) {} : () -> ()
    "#;
    let module = parse_module(input).unwrap();
    assert!(module.computations()[1].is_declaration());
    assert_eq!(module.computations()[0].roots(), &[ValueId(0), ValueId(1)]);
    assert_eq!(round_trip(&module), module);
}

#[test]
fn properties_and_discardable_attributes_share_the_canonical_operation_record() {
    let input = r#"
      "builtin.module"() ({
        "func.func"() <{sym_name = "constant", function_type = () -> tensor<i32>}> ({
          %c = "stablehlo.constant"() <{value = dense<1> : tensor<i32>}>
              {consumer.note = "constant"} : () -> tensor<i32>
          "func.return"(%c) : (tensor<i32>) -> ()
        }) : () -> ()
      }) : () -> ()
    "#;
    let module = parse_module(input).unwrap();
    assert_eq!(
        module.computations()[0].instructions()[0]
            .operation()
            .attributes()
            .unwrap()["consumer.note"],
        Attribute::String("constant".into())
    );
    assert_eq!(round_trip(&module), module);
}

#[test]
fn nested_regions_preserve_block_arguments_capture_and_result_visibility() {
    let ty = scalar(ElementType::F32);
    let reducer = Region::new(vec![Block::new(
        vec![
            BlockArgument::new(ValueId(100), ty.clone()),
            BlockArgument::new(ValueId(101), ty.clone()),
        ],
        vec![
            Instruction::new(
                Operation::stablehlo(StableHloOp::Add, Attributes::new()),
                vec![ValueId(100), ValueId(101)],
                vec![],
                ValueId(102),
                ty.clone(),
            ),
            Instruction::with_results(
                Operation::stablehlo(StableHloOp::Return, Attributes::new()),
                vec![ValueId(102)],
                vec![],
                vec![],
            ),
        ],
    )]);
    let function = Computation::try_new(
        "reduce",
        vec![
            Instruction::new(
                Operation::parameter(0, "input"),
                vec![],
                vec![],
                ValueId(0),
                TensorType::static_shape(ElementType::F32, [4]).into(),
            ),
            Instruction::new(
                Operation::parameter(1, "init"),
                vec![],
                vec![],
                ValueId(1),
                ty.clone(),
            ),
            Instruction::new(
                Operation::stablehlo(
                    StableHloOp::Reduce,
                    [("dimensions".into(), Attribute::integers(vec![0]))].into(),
                ),
                vec![ValueId(0), ValueId(1)],
                vec![],
                ValueId(2),
                ty,
            )
            .with_regions(vec![reducer]),
        ],
        ValueId(2),
    )
    .unwrap();
    let mut module = Module::new("reduce");
    module.add_computation(function);
    let parsed = round_trip(&module);
    let function = &parsed.computations()[0];
    assert_eq!(
        function.function_type(),
        module.computations()[0].function_type()
    );
    assert_eq!(
        function.instructions()[2].regions()[0].blocks()[0]
            .arguments()
            .len(),
        2
    );
}

#[test]
fn zero_result_function_calls_and_empty_void_definitions_remain_distinct_from_declarations() {
    let input = r#"
      "builtin.module"() ({
        "func.func"() ({
          ^entry:
            "func.call"() {callee = @empty} : () -> ()
            "func.return"() : () -> ()
        }) {sym_name = "caller", function_type = () -> ()} : () -> ()
        "func.func"() ({
          ^entry:
            "func.return"() : () -> ()
        }) {sym_name = "empty", function_type = () -> ()} : () -> ()
      }) : () -> ()
    "#;
    let module = parse_module(input).unwrap();
    assert!(!module.computations()[1].is_declaration());
    assert_eq!(round_trip(&module), module);
}

#[test]
fn generic_module_parser_rejects_invalid_ssa_types_symbols_and_trailing_input() {
    let input = r#"
      "builtin.module"() ({
        "func.func"() ({
          ^entry(%x: tensor<f32>):
            %result = "stablehlo.negate"(%x) : (tensor<f32>) -> tensor<f32>
            "func.return"(%result) : (tensor<f32>) -> ()
        }) {sym_name = "main", function_type = (tensor<f32>) -> tensor<f32>} : () -> ()
      }) : () -> ()
    "#;
    for invalid in [
        input.replace("\"stablehlo.negate\"(%x)", "\"stablehlo.negate\"(%missing)"),
        input.replace("\"stablehlo.negate\"(%x)", "\"stablehlo.negate\"(%result)"),
        input.replace("%result =", "%x ="),
        input.replace(
            "(tensor<f32>) -> tensor<f32>\n",
            "(tensor<i32>) -> tensor<f32>\n",
        ),
        input.replace("\"func.return\"(%result)", "\"func.return\"(%unknown)"),
        input.replace("\"stablehlo.negate\"", "\"vendor.unknown\""),
        input.replace("sym_name = \"main\"", "sym_name = \"\""),
        input.replace("\"func.return\"(%result) : (tensor<f32>) -> ()", ""),
        format!("{input} garbage"),
    ] {
        assert!(parse_module(&invalid).is_err(), "{invalid}");
    }
}

#[test]
fn unsupported_model_metadata_is_rejected_instead_of_dropped() {
    let mut module = Module::new("module");
    module
        .set_attributes(
            [(
                "stablehlo_rust.entry".into(),
                Attribute::String("collision".into()),
            )]
            .into(),
        )
        .unwrap();
    assert!(matches!(
        print_module(&module).unwrap_err().kind,
        TextErrorKind::DuplicateField(_)
    ));
    for name in ["two words", "1abc"] {
        let function = Computation::try_new(
            "main",
            vec![Instruction::new(
                Operation::parameter(0, name),
                vec![],
                vec![],
                ValueId(0),
                Type::Token,
            )],
            ValueId(0),
        )
        .unwrap();
        let mut module = Module::new("module");
        module.add_computation(function);
        assert!(matches!(
            print_module(&module).unwrap_err().kind,
            TextErrorKind::Unsupported(_)
        ));
    }
    let function = Computation::try_new(
        "main",
        vec![Instruction::new(
            Operation::parameter(0, "arg"),
            vec![],
            vec![0.into()],
            ValueId(0),
            Type::Token,
        )],
        ValueId(0),
    )
    .unwrap();
    let mut module = Module::new("module");
    module.add_computation(function);
    assert!(matches!(
        print_module(&module).unwrap_err().kind,
        TextErrorKind::Unsupported(_)
    ));
}

#[test]
fn generic_assembly_keeps_byte_strings_dense_literals_and_function_signatures() {
    let mut function = Computation::try_declare(
        "external name",
        FunctionType::new(vec![scalar(ElementType::F32)], vec![Type::Token]),
    )
    .unwrap();
    function
        .set_attributes(
            [
                ("sym_visibility".into(), Attribute::String("private".into())),
                (
                    "consumer.bytes".into(),
                    Attribute::Bytes(vec![0xff, 0, b'"']),
                ),
            ]
            .into(),
        )
        .unwrap();
    let mut module = Module::new("module");
    module.add_computation(function);
    assert_eq!(round_trip(&module), module);
}

#[test]
fn standard_module_and_function_syntax_accepts_generic_stablehlo_bodies() {
    let input = r#"
        module @example attributes {consumer.note = "preserved"} {
          func.func private @external(tensor<2xf32>) -> tensor<2xf32>
          func.func @main(%x: tensor<2xf32> {consumer.input = "x"})
              -> (tensor<2xf32> {consumer.output = "y"})
              attributes {consumer.tag = "entry"} {
            %y = func.call @external(%x) : (tensor<2xf32>) -> tensor<2xf32>
            %z = "stablehlo.add"(%x, %y) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
            return %z : tensor<2xf32>
          }
        }
    "#;
    let module = parse_module(input).unwrap();
    assert_eq!(module.name(), "example");
    assert_eq!(module.computations().len(), 2);
    assert!(module.computations()[0].is_declaration());
    let function = &module.computations()[1];
    assert_eq!(
        function.attributes()["arg_attrs"],
        Attribute::Array(vec![Attribute::Dictionary(
            [("consumer.input".into(), Attribute::String("x".into()))].into()
        )])
    );
    assert_eq!(round_trip(&module), module);
}

#[test]
fn standard_void_functions_quoted_symbols_and_multiresult_calls_round_trip() {
    let input = r#"
      module {
        func.func private @"external name"() -> (!stablehlo.token, tensor<f32>)
        func.func @main() -> (!stablehlo.token, tensor<f32>) {
          %pair:2 = func.call @"external name"() : () -> (!stablehlo.token, tensor<f32>)
          func.return %pair#0, %pair#1 : !stablehlo.token, tensor<f32>
        }
        func.func @empty() {
          return
        }
      }
    "#;
    let module = parse_module(input).unwrap();
    assert_eq!(round_trip(&module), module);
}

#[test]
fn standard_functions_can_contain_generic_regions_with_custom_stablehlo_returns() {
    let input = r#"
      module {
        func.func @choose(%predicate: tensor<i1>, %x: tensor<f32>) -> tensor<f32> {
          %result = "stablehlo.if"(%predicate) ({
            stablehlo.return %x : tensor<f32>
          }, {
            stablehlo.return %x : tensor<f32>
          }) : (tensor<i1>) -> tensor<f32>
          return %result : tensor<f32>
        }
      }
    "#;
    round_trip(&parse_module(input).unwrap());
}

#[test]
fn malformed_standard_function_signatures_and_metadata_are_rejected() {
    for input in [
        "module { func.func @public_declaration() }",
        "module { func.func private @external(%x: tensor<f32>, tensor<f32>) }",
        "module { func.func @main(tensor<f32>) { return } }",
        "module { func.func @main() {} }",
        "module { func.func @main() attributes {sym_name = \"collision\"} {return} }",
        "module { func.func @main() attributes {sym_visibility = \"private\"} {return} }",
        "module @one attributes {sym_name = \"two\"} {}",
        "module { func.func @main() { %x = func.call @foo() : () -> () return } }",
        "module { func.func @main() -> tensor<f32> { return } }",
        "module { func.func @main() { return %missing : tensor<f32> } }",
    ] {
        assert!(parse_module(input).is_err(), "{input}");
    }
}

#[test]
fn module_parsing_bounds_nesting_and_result_expansion_before_constructing_ir() {
    let mut nested = String::new();
    for _ in 0..256 {
        nested.push_str("\"stablehlo.composite\"() ({");
    }
    for _ in 0..256 {
        nested.push_str("}) : () -> ()");
    }
    assert_eq!(
        parse_module(&nested).unwrap_err().kind,
        TextErrorKind::NestingLimit
    );
    assert_eq!(
        parse_module("%many:1048577 = \"func.call\"() : () -> ()")
            .unwrap_err()
            .kind,
        TextErrorKind::ResourceLimit
    );
    let valid = r#"module { func.func @empty() {return} }"#;
    for (index, _) in valid.char_indices().skip(1) {
        assert!(parse_module(&valid[..index]).is_err(), "truncation {index}");
    }
}
