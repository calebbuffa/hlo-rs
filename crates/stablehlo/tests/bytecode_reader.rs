use stablehlo::{
    bytecode::{
        BytecodeContainer, BytecodeErrorKind as Kind, SectionId, read_module, write_module,
    },
    text::parse_module,
};

mod reference_support;
#[path = "upstream_support/structure.rs"]
mod structure;

const ADD: &str = "module @math { func.func @main(%a:tensor<2xf32>,%b:tensor<2xf32>)->tensor<2xf32> { \
    %r=stablehlo.add %a,%b:tensor<2xf32> return %r:tensor<2xf32> } }";

#[test]
fn rust_bytecode_roundtrips_modules_declarations_tokens_quantization_and_metadata() {
    for source in [
        ADD,
        "module {}",
        "module { func.func private @external(tensor<f32>)->tensor<f32> }",
        "module { func.func @token(%arg:!stablehlo.token)->!stablehlo.token { return %arg:!stablehlo.token } }",
        "module { func.func @quant(%arg:tensor<4x!quant.uniform<i8:f32, 0.25:0>>)\
         ->tensor<4x!quant.uniform<i8:f32, 0.25:0>> { return %arg:tensor<4x!quant.uniform<i8:f32, 0.25:0>> } }",
        "module @meta attributes {test.bytes = \"\\FF\\00\", test.version = 42 : i64} { \
         func.func @main(%a:tensor<2xf32> loc(\"input.rs\":2:3))->tensor<2xf32> { \
         %r=stablehlo.negate %a:tensor<2xf32> loc(\"negate\") return %r:tensor<2xf32> } } loc(\"module.rs\":1:1)",
    ] {
        let expected = parse_module(source).unwrap();
        for version in 0..=6 {
            let bytes = write_module(&expected, version).unwrap();
            let actual =
                read_module(&bytes).unwrap_or_else(|error| panic!("v{version}: {error}\n{source}"));
            structure::assert_equivalent(&expected, &actual);
            assert_eq!(write_module(&actual, version).unwrap(), bytes);
        }
    }
}

#[test]
fn captured_values_sibling_regions_nested_results_and_repeated_operands_keep_identity() {
    let source = "module { func.func @main(%pred:tensor<i1>,%x:tensor<i32>)->tensor<i32> { \
        %r=\"stablehlo.if\"(%pred) ({ \
            %a=stablehlo.add %x,%x:tensor<i32> \
            %b=stablehlo.multiply %a,%x:tensor<i32> stablehlo.return %b:tensor<i32> \
        }, { \
            %c=stablehlo.subtract %x,%x:tensor<i32> stablehlo.return %c:tensor<i32> \
        }):(tensor<i1>)->tensor<i32> \
        %out=stablehlo.add %r,%x:tensor<i32> return %out:tensor<i32> } }";
    let expected = parse_module(source).unwrap();
    for version in 0..=6 {
        let bytes = write_module(&expected, version).unwrap();
        let actual = read_module(&bytes).unwrap();
        structure::assert_equivalent(&expected, &actual);
    }
}

#[test]
fn malformed_ir_masks_names_locations_and_payloads_have_structured_errors() {
    let module = parse_module(ADD).unwrap();
    let original = write_module(&module, 6).unwrap();
    let container = BytecodeContainer::parse(&original).unwrap();
    let ir = container.section(SectionId::Ir).unwrap().offset;
    let mut bytes = original.clone();
    bytes[ir + 2] = 0xff;
    assert_eq!(
        read_module(&bytes).unwrap_err().kind,
        Kind::InvalidOperationMask(0xff)
    );
    bytes = original.clone();
    bytes[ir + 1] = 0xff;
    assert!(matches!(
        read_module(&bytes).unwrap_err().kind,
        Kind::InvalidIndex {
            table: "operation",
            ..
        }
    ));
    bytes = original.clone();
    bytes[ir] = 1;
    assert!(matches!(
        read_module(&bytes).unwrap_err().kind,
        Kind::InvalidPayload(_)
    ));
    let location = container
        .attributes()
        .iter()
        .find(|entry| entry.data == b"loc(unknown)\0")
        .unwrap();
    bytes = original.clone();
    bytes[location.offset] = 0xff;
    assert_eq!(read_module(&bytes).unwrap_err().kind, Kind::InvalidUtf8);
    bytes = original.clone();
    bytes[location.offset + location.data.len() - 1] = b'x';
    assert!(matches!(
        read_module(&bytes).unwrap_err().kind,
        Kind::InvalidPayload(_)
    ));
    let ty = &container.types()[0];
    bytes = original.clone();
    bytes[ty.offset] = b'?';
    assert!(matches!(
        read_module(&bytes).unwrap_err().kind,
        Kind::Assembly(_)
    ));
}

#[test]
fn malformed_input_truncation_and_mutation_never_bypasses_verification_or_panics() {
    let module = parse_module(ADD).unwrap();
    for version in 0..=6 {
        let original = write_module(&module, version).unwrap();
        for length in 0..original.len() {
            assert!(
                read_module(&original[..length]).is_err(),
                "v{version}, prefix{length}"
            );
        }
        for index in 0..original.len() {
            for byte in [0, 1, 0x7f, 0xff] {
                let mut bytes = original.clone();
                bytes[index] = byte;
                if let Ok(module) = read_module(&bytes) {
                    module.verify().unwrap();
                }
            }
        }
    }
}

#[test]
fn writer_rejects_reserved_entry_metadata_instead_of_overwriting_it() {
    let mut module = parse_module(ADD).unwrap();
    let mut attributes = module.attributes().clone();
    attributes.insert(
        "stablehlo_rust.entry".into(),
        stablehlo::Attribute::String("discarded".into()),
    );
    module.set_attributes(attributes).unwrap();
    assert!(matches!(
        write_module(&module, 6).unwrap_err().kind,
        Kind::InvalidPayload(_)
    ));
}

#[test]
#[ignore = "requires STABLEHLO_REFERENCE_DIR"]
fn all_accepted_reference_fixtures_roundtrip_in_pure_rust_at_every_bytecode_version() {
    let source = reference_support::pinned_operation_fixtures();
    let modules = source
        .split("// -----")
        .enumerate()
        .filter(|(_, chunk)| !chunk.contains("expected-error"))
        .filter_map(|(index, chunk)| parse_module(chunk).ok().map(|module| (index + 1, module)))
        .collect::<Vec<_>>();
    assert_eq!(modules.len(), 195);
    for (fixture, expected) in modules {
        for version in 0..=6 {
            let bytes = write_module(&expected, version).unwrap();
            let actual = read_module(&bytes)
                .unwrap_or_else(|error| panic!("fixture{fixture}/v{version}: {error}"));
            structure::assert_equivalent(&expected, &actual);
        }
    }
    eprintln!(
        "195 reference modules roundtripped in pure Rust across all seven MLIR bytecode versions, preserving SSA relationships, types, attributes, nested regions and locations."
    );
}
