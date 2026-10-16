//! Independent pinned and older-version parser/verifier evidence, not full conformance.

use stablehlo::text::{parse_module, print_module};
mod reference_support;
mod upstream_support;
use upstream_support::UpstreamResult;

fn diagnostic_matches(text: &str, pattern: &str, regex: bool) -> bool {
    if !regex {
        return text.contains(pattern);
    }
    let mut remaining = text;
    for literal in pattern.split("{{.*}}") {
        assert!(
            !literal.contains("{{") && !literal.contains("}}"),
            "unsupported diagnostic pattern: {pattern}"
        );
        let Some(position) = remaining.find(literal) else {
            return false;
        };
        remaining = &remaining[position + literal.len()..];
    }
    true
}

#[test]
fn diagnostic_patterns_preserve_literal_text_and_ordered_wildcard_fragments() {
    assert!(diagnostic_matches(
        "error: operand #0 must be tensor<f32>",
        "operand #0",
        false
    ));
    assert!(!diagnostic_matches(
        "error: operand #0",
        "operand #1",
        false
    ));
    assert!(diagnostic_matches("error: a.[b]", "a.[b]", false));
    assert!(!diagnostic_matches("error: axb", "a.[b]", false));
    assert!(diagnostic_matches(
        "error: #0 must be 0D tensor of f32, but got 'tensor<1xf32>'",
        "#0 must be 0D tensor of {{.*}}, but got 'tensor<1xf32>'",
        true
    ));
    assert!(diagnostic_matches(
        "a-first b-middle c-last",
        "a{{.*}}b{{.*}}c",
        true
    ));
    assert!(diagnostic_matches("ab", "a{{.*}}b", true));
    assert!(!diagnostic_matches("b-then a", "a{{.*}}b", true));
}

#[test]
#[should_panic(expected = "unsupported diagnostic pattern")]
fn diagnostic_patterns_do_not_silently_accept_unimplemented_regex_forms() {
    diagnostic_matches("error: value 12", "value {{[0-9]+}}", true);
}

#[test]
#[ignore = "requires STABLEHLO_UPSTREAM_PYTHON with pinned jaxlib and STABLEHLO_REFERENCE_DIR"]
fn rust_emitted_reference_fixtures_pass_independent_upstream_parsing_and_verification() {
    let source = reference_support::pinned_operation_fixtures();
    let mut modules = Vec::new();
    for (index, chunk) in source.split("// -----").enumerate() {
        if chunk.contains("expected-error") {
            continue;
        }
        if let Ok(module) = parse_module(chunk) {
            modules.push((index + 1, print_module(&module).unwrap()));
        }
    }
    assert_eq!(
        modules.len(),
        195,
        "run the separate pinned acceptance audit first"
    );
    let results = upstream_support::verify(
        &modules
            .iter()
            .map(|(_, source)| source.clone())
            .collect::<Vec<_>>(),
    );
    let mut failures = Vec::new();
    let mut unavailable_quantization = 0;
    let mut unsupported_weakorder = 0;
    let mut unavailable_sparse = 0;
    for ((fixture, emitted), result) in modules.iter().zip(results) {
        match result {
            UpstreamResult::Accepted(text) => {
                if let Err(error) = parse_module(&text) {
                    failures.push(format!(
                        "chunk {fixture}: Rust rejected upstream assembly: {error}\n{text}"
                    ));
                }
            }
            UpstreamResult::Rejected(text)
                if emitted.contains("!quant.uniform<")
                    && text.contains("type created with unregistered dialect") =>
            {
                unavailable_quantization += 1;
            }
            UpstreamResult::Rejected(text)
                if emitted.contains("#sparse_tensor.encoding")
                    && text.contains("attribute created with unregistered dialect") =>
            {
                unavailable_sparse += 1;
            }
            UpstreamResult::Rejected(text)
                if emitted.contains("WEAKORDER")
                    && text
                        .contains("expected ::mlir::stablehlo::ComparisonType to be one of:") =>
            {
                unsupported_weakorder += 1;
            }
            UpstreamResult::Rejected(text) => failures.push(format!(
                "chunk {fixture}: upstream rejected Rust assembly:\n{text}"
            )),
        }
    }
    assert_eq!(
        (
            unavailable_quantization,
            unsupported_weakorder,
            unavailable_sparse
        ),
        (13, 1, 1),
        "tool/version exceptions changed: audit before updating expectations"
    );
    assert!(
        failures.is_empty(),
        "{} upstream interoperability failures:\n{}",
        failures.len(),
        failures.join("\n\n")
    );
    eprintln!(
        "180 Rust-emitted positive fixture chunks parsed and verified by independent StableHLO 1.20.0; upstream assembly reparsed by Rust. 13 quantized chunks require unavailable Quant dialect registration, one sparse chunk requires unavailable SparseTensor registration, and one WEAKORDER chunk requires newer StableHLO. This does not cover 1.22-only behavior or execution semantics."
    );
}

#[test]
#[ignore = "requires STABLEHLO_UPSTREAM_PYTHON with pinned jaxlib and STABLEHLO_REFERENCE_DIR"]
fn diagnostic_fixtures_match_upstream_rejections_with_explicit_tool_limits() {
    let source = reference_support::pinned_operation_fixtures();
    let modules = source
        .split("// -----")
        .enumerate()
        .filter(|(_, chunk)| chunk.contains("expected-error"))
        .map(|(index, chunk)| (index + 1, chunk.to_owned()))
        .collect::<Vec<_>>();
    assert_eq!(modules.len(), 414);
    let results = upstream_support::verify(
        &modules
            .iter()
            .map(|(_, source)| source.clone())
            .collect::<Vec<_>>(),
    );
    let mut accepted = Vec::new();
    let mut unavailable_quantization = 0;
    let mut unsupported_weakorder = 0;
    let mut matched = 0;
    let mut mismatches = Vec::new();
    for ((fixture, source), result) in modules.iter().zip(results) {
        let text = match result {
            UpstreamResult::Accepted(text) => {
                accepted.push(format!("chunk {fixture} accepted upstream:\n{text}"));
                continue;
            }

            UpstreamResult::Rejected(text) => text,
        };
        if source.contains("!quant.uniform<")
            && text.contains("type created with unregistered dialect")
        {
            unavailable_quantization += 1;
            continue;
        }
        if source.contains("WEAKORDER")
            && text.contains("expected ::mlir::stablehlo::ComparisonType to be one of:")
        {
            unsupported_weakorder += 1;
            continue;
        }
        let patterns = source
            .lines()
            .filter(|line| line.contains("expected-error"))
            .map(|line| {
                let (_, pattern) = line.split_once("{{").expect("expected diagnostic text");
                let (pattern, _) = pattern.rsplit_once("}}").expect("closed diagnostic text");
                (line.contains("expected-error-re"), pattern)
            })
            .collect::<Vec<_>>();
        assert!(
            !patterns.is_empty(),
            "chunk {fixture} has no expected diagnostic pattern"
        );
        let matches = patterns
            .iter()
            .all(|&(regex, pattern)| diagnostic_matches(&text, pattern, regex));
        if matches {
            matched += 1;
        } else {
            mismatches.push(format!("chunk {fixture}, patterns {patterns:?}:\n{text}"));
        }
    }
    assert!(
        accepted.is_empty(),
        "{}\n{}",
        accepted.len(),
        accepted.join("\n\n")
    );
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n\n"));
    assert_eq!(
        (matched, unavailable_quantization, unsupported_weakorder),
        (403, 10, 1),
        "negative diagnostic classifications changed: audit before updating expectations"
    );
    eprintln!(
        "403 diagnostic-bearing pinned chunks rejected with every expected diagnostic pattern present in independent StableHLO 1.20.0 output; 10 unavailable Quant dialect and one unsupported WEAKORDER exceptions. This is not Rust diagnostic equivalence, source-location matching, target-1.22 coverage or execution validation."
    );
}

#[test]
#[ignore = "requires STABLEHLO_UPSTREAM_NATIVE and STABLEHLO_REFERENCE_DIR"]
fn rust_emitted_reference_fixtures_pass_matching_native_upstream() {
    let source = reference_support::pinned_operation_fixtures();
    let modules = source
        .split("// -----")
        .enumerate()
        .filter(|(_, chunk)| !chunk.contains("expected-error"))
        .filter_map(|(index, chunk)| {
            parse_module(chunk)
                .ok()
                .map(|module| (index + 1, print_module(&module).unwrap()))
        })
        .collect::<Vec<_>>();
    assert_eq!(modules.len(), 195);
    let results = upstream_support::verify_native(
        &modules
            .iter()
            .map(|(_, text)| text.clone())
            .collect::<Vec<_>>(),
    );
    let mut failures = Vec::new();
    for ((fixture, _), result) in modules.iter().zip(results) {
        match result {
            UpstreamResult::Accepted(text) => {
                if let Err(error) = parse_module(&text) {
                    failures.push(format!(
                        "chunk {fixture}: Rust rejected native assembly: {error}\n{text}"
                    ));
                }
            }
            UpstreamResult::Rejected(error) => failures.push(format!(
                "chunk {fixture}: native upstream rejected Rust assembly:\n{error}"
            )),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
    eprintln!(
        "195 Rust-emitted fixtures verified with matching native StableHLO 1.22.0 and reparsed in Rust; no registration/version exceptions."
    );
}

#[test]
#[ignore = "requires STABLEHLO_UPSTREAM_NATIVE and STABLEHLO_REFERENCE_DIR"]
fn diagnostic_fixtures_match_matching_native_upstream_without_tool_exemptions() {
    let source = reference_support::pinned_operation_fixtures();
    let modules = source
        .split("// -----")
        .enumerate()
        .filter(|(_, chunk)| chunk.contains("expected-error"))
        .map(|(index, chunk)| (index + 1, chunk.to_owned()))
        .collect::<Vec<_>>();
    assert_eq!(modules.len(), 414);
    let results = upstream_support::verify_native(
        &modules
            .iter()
            .map(|(_, text)| text.clone())
            .collect::<Vec<_>>(),
    );
    let mut failures = Vec::new();
    for ((fixture, source), result) in modules.iter().zip(results) {
        let text = match result {
            UpstreamResult::Accepted(text) => {
                failures.push(format!("chunk {fixture} accepted upstream:\n{text}"));
                continue;
            }
            UpstreamResult::Rejected(text) => text,
        };
        let patterns = source
            .lines()
            .filter(|line| line.contains("expected-error"))
            .map(|line| {
                let (_, pattern) = line.split_once("{{").expect("expected diagnostic text");
                let (pattern, _) = pattern.rsplit_once("}}").expect("closed diagnostic text");
                (line.contains("expected-error-re"), pattern)
            })
            .collect::<Vec<_>>();
        assert!(!patterns.is_empty());
        if !patterns
            .iter()
            .all(|&(regex, pattern)| diagnostic_matches(&text, pattern, regex))
        {
            failures.push(format!("chunk {fixture}, patterns {patterns:?}:\n{text}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
    eprintln!(
        "414 diagnostic-bearing fixtures rejected by matching native StableHLO 1.22.0 with every expected pattern; no registration/version exceptions."
    );
}

#[test]
#[ignore = "requires STABLEHLO_UPSTREAM_NATIVE"]
fn native_validator_registers_supporting_dialects_and_rejects_unknown_dialects() {
    let modules = [
        "module { func.func @quant(%a:tensor<4x!quant.uniform<i8:f32, 0.25:0>>) \
         -> tensor<4x!quant.uniform<i8:f32, 0.25:0>> { return %a : \
         tensor<4x!quant.uniform<i8:f32, 0.25:0>> } }",
        "module { func.func @sparse(%a:tensor<4xf32, #sparse_tensor.encoding<\
         {map=(i)->(i:compressed)}>>) -> tensor<4xf32, #sparse_tensor.encoding<\
         {map=(i)->(i:compressed)}>> { return %a : tensor<4xf32, \
         #sparse_tensor.encoding<{map=(i)->(i:compressed)}>> } }",
        "module { func.func @complex(%re:f32,%im:f32)->complex<f32> { \
         %c=complex.create %re,%im:complex<f32> return %c:complex<f32> } }",
        "module { \"unregistered.operation\"() : () -> () }",
    ]
    .map(str::to_owned);
    let results = upstream_support::verify_native(&modules);
    for (index, result) in results.iter().take(3).enumerate() {
        match result {
            UpstreamResult::Accepted(_) => {}
            UpstreamResult::Rejected(error) => panic!("dialect probe {index}: {error}"),
        }
    }
    match &results[3] {
        UpstreamResult::Accepted(text) => panic!("unregistered dialect accepted: {text}"),
        UpstreamResult::Rejected(error) => assert!(error.contains("unregistered")),
    }
}
