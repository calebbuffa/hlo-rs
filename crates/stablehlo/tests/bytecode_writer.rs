use stablehlo::{
    bytecode::{BytecodeContainer, BytecodeErrorKind, write_module},
    text::parse_module,
};

mod reference_support;
#[path = "upstream_support/structure.rs"]
mod structure;
mod upstream_support;

const ADD: &str = "module @math { func.func @main(%a:tensor<2xf32>,%b:tensor<2xf32>)->tensor<2xf32> { \
    %r=stablehlo.add %a,%b:tensor<2xf32> return %r:tensor<2xf32> } }";

#[test]
fn writer_emits_checked_deterministic_containers_for_every_supported_format() {
    for source in [
        ADD,
        "module {}",
        "module { func.func private @external(tensor<f32>)->tensor<f32> }",
        "module { func.func @token(%arg:!stablehlo.token)->!stablehlo.token { return %arg:!stablehlo.token } }",
    ] {
        let module = parse_module(source).unwrap();
        for version in 0..=6 {
            let bytes = write_module(&module, version).unwrap();
            assert_eq!(write_module(&module, version).unwrap(), bytes);
            let container = BytecodeContainer::parse(&bytes).unwrap();
            assert_eq!(container.version(), version);
            assert_eq!(container.producer(), b"stablehlo-rust");
            assert!(
                container
                    .attributes()
                    .iter()
                    .all(|entry| !entry.custom_encoding)
            );
            assert!(container.types().iter().all(|entry| !entry.custom_encoding));
            assert!(
                container
                    .operation_names()
                    .iter()
                    .any(|op| op.name == b"module")
            );
        }
    }
}

#[test]
fn writer_rejects_unsupported_format_versions() {
    let module = parse_module(ADD).unwrap();
    for version in [7, 128, u64::MAX] {
        assert_eq!(
            write_module(&module, version).unwrap_err().kind,
            BytecodeErrorKind::UnsupportedVersion(version)
        );
    }
}

#[test]
#[ignore = "requires STABLEHLO_UPSTREAM_NATIVE"]
fn native_reads_rust_written_bytecode_for_all_format_versions() {
    use upstream_support::UpstreamResult;
    let modules = [
        ADD,
        "module {}",
        "module { func.func private @external(tensor<f32>)->tensor<f32> }",
        "module { func.func @token(%arg:!stablehlo.token)->!stablehlo.token { return %arg:!stablehlo.token } }",
        "module { func.func @quant(%arg:tensor<4x!quant.uniform<i8:f32, 0.25:0>>)\
         ->tensor<4x!quant.uniform<i8:f32, 0.25:0>> { return %arg:tensor<4x!quant.uniform<i8:f32, 0.25:0>> } }",
    ].map(|source| parse_module(source).unwrap());
    let mut cases = Vec::new();
    let mut bytes = Vec::new();
    for (index, module) in modules.iter().enumerate() {
        for version in 0..=6 {
            cases.push((index, version));
            bytes.push(write_module(module, version).unwrap());
        }
    }
    for ((index, version), result) in cases
        .iter()
        .zip(upstream_support::verify_native_bytecode(&bytes))
    {
        match result {
            UpstreamResult::Accepted(text) => {
                let reparsed = parse_module(&text)
                    .unwrap_or_else(|error| panic!("case{index}/v{version}: {error}\n{text}"));
                structure::assert_equivalent(&modules[*index], &reparsed);
            }
            UpstreamResult::Rejected(error) => panic!("case{index}/v{version}: {error}"),
        }
    }
}

#[test]
#[ignore = "requires STABLEHLO_UPSTREAM_NATIVE and STABLEHLO_REFERENCE_DIR"]
fn all_rust_accepted_reference_fixtures_emit_native_readable_bytecode() {
    use upstream_support::UpstreamResult;
    let source = reference_support::pinned_operation_fixtures();
    let modules = source
        .split("// -----")
        .enumerate()
        .filter(|(_, chunk)| !chunk.contains("expected-error"))
        .filter_map(|(index, chunk)| parse_module(chunk).ok().map(|module| (index + 1, module)))
        .collect::<Vec<_>>();
    assert_eq!(modules.len(), 195);
    let mut cases = Vec::new();
    let mut bytes = Vec::new();
    for (fixture, module) in &modules {
        for version in 0..=6 {
            cases.push((*fixture, version));
            bytes.push(
                write_module(module, version)
                    .unwrap_or_else(|error| panic!("fixture{fixture}/v{version}: {error}")),
            );
        }
    }
    let mut failures = Vec::new();
    for ((fixture, version), result) in cases
        .iter()
        .zip(upstream_support::verify_native_bytecode(&bytes))
    {
        match result {
            UpstreamResult::Accepted(text) => match parse_module(&text) {
                Ok(actual) => {
                    let expected = &modules.iter().find(|(id, _)| id == fixture).unwrap().1;
                    structure::assert_equivalent(expected, &actual);
                }
                Err(error) => {
                    failures.push(format!("fixture{fixture}/v{version}: Rust rejected native assembly: {error}\n{text}"));
                }
            },
            UpstreamResult::Rejected(error) => {
                failures.push(format!("fixture{fixture}/v{version}: {error}"))
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n\n")
    );
    eprintln!(
        "195 reference modules serialized in Rust at all seven MLIR bytecode versions (1365 modules), verified natively, and reparsed in Rust."
    );
}
