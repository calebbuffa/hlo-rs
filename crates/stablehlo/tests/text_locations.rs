use stablehlo::text::{
    TextErrorKind, parse_attribute, parse_module, print_attribute, print_module,
};
use stablehlo::{
    Attribute, Computation, ElementType, FunctionType, Instruction, Location, Module, Operation,
    TensorType, ValueId,
};

fn round_trip_attribute(source: &str) -> Attribute {
    let attribute = parse_attribute(source).unwrap_or_else(|error| panic!("{source}: {error}"));
    let printed = print_attribute(&attribute).unwrap();
    assert_eq!(parse_attribute(&printed).unwrap(), attribute, "{printed}");
    attribute
}

fn round_trip_module(source: &str) -> Module {
    let module = parse_module(source).unwrap_or_else(|error| panic!("{source}: {error}"));
    let printed = print_module(&module).unwrap();
    let reparsed = parse_module(&printed).unwrap_or_else(|error| panic!("{printed}: {error}"));
    assert_eq!(reparsed, module);
    assert_eq!(print_module(&reparsed).unwrap(), printed);
    module
}

#[test]
fn builtin_location_forms_preserve_structure_and_byte_strings() {
    for source in [
        "loc(unknown)",
        "loc(\"name\")",
        "loc(\"name\"(unknown))",
        "loc(\"file.rs\":0)",
        "loc(\"file.rs\":0:4294967295)",
        "loc(\"file.rs\":1:2 to 3:4)",
        "loc(\"file.rs\":1:2 to :4)",
        "loc(\"file.rs\":0x10:0x20)",
        "loc(callsite(\"callee\" at \"caller\"))",
        "loc(callsite(\"callee\"(\"c.rs\":1:2) at unknown))",
        "loc(fused[])",
        "loc(fused[unknown, \"a\", \"b\":1])",
        "loc(fused<\"metadata\">[\"x\":1:2])",
        "loc(fused<{consumer.tag = [1 : i32, loc(unknown)]}>[])",
        "loc\n// comment\n(fused[unknown])",
        r#"loc("\FF\00\22\5C"("\FE":1:2))"#,
    ] {
        assert!(matches!(
            round_trip_attribute(source),
            Attribute::Location(_)
        ));
    }
    assert_eq!(
        round_trip_attribute("loc(\"file.rs\":1:2 to :4)"),
        Attribute::Location(Box::new(Location::FileRange {
            file: b"file.rs".to_vec(),
            start_line: 1,
            start_column: 2,
            end_line: None,
            end_column: 4,
        }))
    );
}

#[test]
fn module_function_argument_instruction_and_return_locations_are_not_attributes() {
    let module = round_trip_module(
        r#"
        module @located {
          func.func @main(%x: tensor<f32> {consumer.tag = "input"} loc("argument"))
            -> tensor<f32> {
            %y = stablehlo.negate %x : tensor<f32> loc("negation"("file.rs":3:7))
            return %y : tensor<f32> loc("return")
          } loc("function")
          func.func private @external(tensor<f32>) -> tensor<f32> loc("declaration")
        } loc("module")
        "#,
    );
    assert!(module.location().is_some());
    assert!(module.attributes().is_empty());
    let function = &module.computations()[0];
    assert!(function.location().is_some());
    assert!(function.return_location().is_some());
    assert!(
        function
            .instructions()
            .iter()
            .all(|op| op.location().is_some())
    );
    assert!(
        function.instructions()[1]
            .operation()
            .attributes()
            .unwrap()
            .is_empty()
    );
    assert!(module.computations()[1].location().is_some());
}

#[test]
fn locations_round_trip_on_generic_operations_and_nested_block_arguments() {
    let module = round_trip_module(
        r#"
        module {
          func.func @main(%x: tensor<i32> loc("input")) -> tensor<i32> {
            %r = "stablehlo.while"(%x) ({
              ^condition(%arg: tensor<i32> loc("condition argument")):
                %false = stablehlo.constant dense<false> : tensor<i1> loc("false")
                "stablehlo.return"(%false) : (tensor<i1>) -> () loc("condition return")
            }, {
              ^body(%arg: tensor<i32> loc("body argument")):
                stablehlo.return %arg : tensor<i32> loc("body return")
            }) : (tensor<i32>) -> tensor<i32> loc(callsite("loop" at "main"))
            "func.return"(%r) : (tensor<i32>) -> () loc("function return")
          }
        }
        "#,
    );
    let instruction = &module.computations()[0].instructions()[1];
    for region in instruction.regions() {
        let block = &region.blocks()[0];
        assert!(block.arguments()[0].location().is_some());
        assert!(
            block
                .instructions()
                .iter()
                .all(|op| op.location().is_some())
        );
    }
}

#[test]
fn location_aliases_are_accepted_in_metadata_and_nested_locations() {
    round_trip_module(
        r#"
        #file = loc("file.rs":1:2)
        #named = loc("definition"(#file))
        #fused = loc(fused<{consumer.tag = "source"}>[#named, #file])
        module attributes {consumer.location = #fused} {
          func.func @main(%x: tensor<f32> loc(#named)) -> tensor<f32> {
            %y = stablehlo.negate %x : tensor<f32> loc(callsite(#named at #file))
            return %y : tensor<f32> loc(#fused)
          } loc(#named)
        } loc(#file)
        "#,
    );
}

#[test]
fn trailing_operation_and_argument_locations_resolve_forward_aliases() {
    round_trip_module(
        r#"
        module {
          func.func @main(%x: tensor<f32> loc(#argument)) -> tensor<f32> {
            %y = func.call @identity(%x) : (tensor<f32>) -> tensor<f32> loc(#call)
            return %y : tensor<f32> loc(#ret)
          } loc(#function)
          func.func @identity(%x: tensor<f32>) -> tensor<f32> {
            return %x : tensor<f32>
          }
        } loc(#module)
        #argument = loc("argument")
        #call = loc("call")
        #ret = loc("return")
        #function = loc("function")
        #module = loc("module")
        "#,
    );
}

#[test]
fn full_reduce_preserves_argument_and_region_locations() {
    let module = round_trip_module(
        r#"
        module {
          func.func @main(%x: tensor<2xf32>, %init: tensor<f32>) -> tensor<f32> {
            %r = stablehlo.reduce (%x init: %init) across dimensions = [0]
              : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
              reducer(%lhs: tensor<f32> loc(#lhs), %rhs: tensor<f32> loc("rhs")) {
                %sum = stablehlo.add %lhs, %rhs : tensor<f32> loc("add")
                stablehlo.return %sum : tensor<f32> loc("region return")
              } loc("reduce")
            return %r : tensor<f32>
          }
        }
        #lhs = loc("lhs")
        "#,
    );
    let reduce = &module.computations()[0].instructions()[2];
    assert!(
        reduce.regions()[0].blocks()[0]
            .arguments()
            .iter()
            .all(|arg| arg.location().is_some())
    );
}

#[test]
fn compact_reduce_propagates_its_explicit_location_to_synthesized_ir() {
    let module = round_trip_module(
        r#"
        module {
          func.func @main(%x: tensor<2xf32>, %init: tensor<f32>) -> tensor<f32> {
            %r = stablehlo.reduce (%x init: %init) applies stablehlo.add across dimensions = [0]
              : (tensor<2xf32>, tensor<f32>) -> tensor<f32> loc(#reduce)
            return %r : tensor<f32>
          }
        }
        #reduce = loc("reduction")
        "#,
    );
    let reduce = &module.computations()[0].instructions()[2];
    let block = &reduce.regions()[0].blocks()[0];
    for argument in block.arguments() {
        assert_eq!(argument.location(), reduce.location());
    }
    for instruction in block.instructions() {
        assert_eq!(instruction.location(), reduce.location());
    }
}

#[test]
fn invalid_location_instances_coordinates_and_token_boundaries_are_errors() {
    for source in [
        "loc()",
        "loc(name)",
        "loc(unknownsuffix)",
        "loc(\"file\":-1:1)",
        "loc(\"file\":+1:1)",
        "loc(\"file\":4294967296:1)",
        "loc(\"file\":1:4294967296)",
        "loc(\"file\":1:2 to 3)",
        "loc(\"file\":1:2 to3:4)",
        "loc(\"file\":1:2 to :+4)",
        "loc(callsite(unknown atunknown))",
        "loc(callsite(unknown unknown))",
        "loc(fused[unknown,])",
        "loc(fused<true>[true])",
        "loc(fused<true[]])",
        "loc(loc(unknown))",
    ] {
        assert!(parse_attribute(source).is_err(), "{source}");
    }
}

#[test]
fn undefined_nonlocation_and_nested_forward_aliases_fail_explicitly() {
    for source in [
        "module {} loc(#missing)",
        "module {} loc(#a) #a = true",
        "#a = loc(#missing) module {}",
        "module attributes {consumer.loc = loc(#a)} {} #a = loc(unknown)",
        "module {} loc(callsite(#a at unknown)) #a = loc(unknown)",
        "#a = true module {} loc(fused[#a])",
        "module {} loc(#stablehlo.bounds)",
    ] {
        let error = parse_module(source).unwrap_err();
        assert!(error.offset <= source.len(), "{error}");
    }
    assert!(matches!(
        parse_module("module {} loc(#missing)").unwrap_err().kind,
        TextErrorKind::UnknownAlias(_)
    ));
}

#[test]
fn location_nesting_and_alias_expansion_are_bounded() {
    let source = format!("loc({}unknown{})", "\"n\"(".repeat(140), ")".repeat(140));
    assert_eq!(
        parse_attribute(&source).unwrap_err().kind,
        TextErrorKind::NestingLimit
    );
    let mut source = String::from("#a0 = loc(\"base\")\n");
    for index in 1..30 {
        source.push_str(&format!(
            "#a{index} = loc(fused[#a{}, #a{}])\n",
            index - 1,
            index - 1
        ));
    }
    source.push_str("module {}");
    assert_eq!(
        parse_module(&source).unwrap_err().kind,
        TextErrorKind::ResourceLimit
    );
}

#[test]
fn deeply_nested_locations_parse_verify_and_print_iteratively() {
    let source = format!("loc({}unknown{})", "\"n\"(".repeat(120), ")".repeat(120));
    let location = parse_attribute(&source).unwrap();
    let printed = print_attribute(&location).unwrap();
    assert_eq!(parse_attribute(&printed).unwrap(), location);

    let mut location = Location::Unknown;
    for _ in 0..120 {
        location = Location::Name {
            name: b"n".to_vec(),
            child: Some(Box::new(location)),
        };
    }
    let attribute = Attribute::Location(Box::new(location));
    let printed = print_attribute(&attribute).unwrap();
    assert_eq!(parse_attribute(&printed).unwrap(), attribute);

    let mut nested_attribute = Attribute::Unit;
    for _ in 0..120 {
        nested_attribute = Attribute::Location(Box::new(Location::Fused {
            locations: Vec::new(),
            metadata: Some(Box::new(nested_attribute)),
        }));
    }
    nested_attribute.verify().unwrap();
}

#[test]
fn authored_ir_locations_are_verified_and_return_metadata_is_not_discarded() {
    let ty = TensorType::scalar(ElementType::F32).into();
    let parameter = Instruction::new(Operation::parameter(0, "x"), vec![], vec![], ValueId(0), ty)
        .with_location(Location::Unknown);
    let mut function = Computation::try_new("main", vec![parameter], ValueId(0)).unwrap();
    function.set_location(Location::Unknown).unwrap();
    function
        .set_return_location(Location::Name {
            name: b"return".to_vec(),
            child: None,
        })
        .unwrap();
    let mut module = Module::new("authored");
    module.set_location(Location::Unknown).unwrap();
    module.add_computation(function);
    assert_eq!(
        parse_module(&print_module(&module).unwrap()).unwrap(),
        module
    );
    let bad = Location::Fused {
        locations: vec![],
        metadata: Some(Box::new(Attribute::Integer {
            ty: ElementType::F32,
            value: 1,
        })),
    };
    assert!(module.set_location(bad.clone()).is_err());
    assert_eq!(module.location(), Some(&Location::Unknown));
    assert!(
        module.computations_mut()[0]
            .set_return_location(bad)
            .is_err()
    );
    let mut declaration =
        Computation::try_declare("external", FunctionType::new(vec![], vec![])).unwrap();
    assert!(declaration.set_return_location(Location::Unknown).is_err());
    assert!(declaration.return_location().is_none());
}
