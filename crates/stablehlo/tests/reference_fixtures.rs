//! Fixture classification, not upstream execution or diagnostic equivalence.

use stablehlo::text::{parse_module, print_module};

mod reference_support;

#[test]
#[ignore = "requires git and STABLEHLO_REFERENCE_DIR pointing to the pinned reference checkout"]
fn pinned_operation_fixtures_have_no_unclassified_acceptance_regressions() {
    let source = reference_support::pinned_operation_fixtures();
    let exceptions = [
        (
            "all_gather_variadic",
            "collective dimension 1 has no positive extent satisfying its group-size geometry",
        ),
        (
            "create_sort_integer_key_float_payload",
            "assembly is not implemented for this operation",
        ),
        (
            "create_sort_float_key_integer_payload",
            "assembly is not implemented for this operation",
        ),
        ("convolution_operand_element_type_i4", "invalid result type"),
    ];
    let mut seen = [false; 4];
    let mut accepted_positive = 0;
    let mut rejected_diagnostic = 0;
    for (index, chunk) in source.split("// -----").enumerate() {
        let diagnostic = chunk.contains("expected-error");
        let name = chunk
            .lines()
            .find_map(|line| line.trim().strip_prefix("func.func @")?.split('(').next());
        let exception = exceptions
            .iter()
            .position(|(label, _)| Some(*label) == name);
        match parse_module(chunk) {
            Ok(module) => {
                assert!(
                    !diagnostic,
                    "accepted diagnostic-bearing chunk {}",
                    index + 1
                );
                assert!(
                    exception.is_none(),
                    "resolved exception {name:?}: update the audit ledger"
                );
                let canonical = print_module(&module).unwrap();
                assert_eq!(
                    parse_module(&canonical).unwrap(),
                    module,
                    "canonical round-trip changed chunk {}",
                    index + 1
                );
                accepted_positive += 1;
            }
            Err(_) if diagnostic => rejected_diagnostic += 1,
            Err(error) => {
                let exception = exception.unwrap_or_else(|| {
                    panic!(
                        "unclassified positive rejection in chunk {} ({name:?}): {error}",
                        index + 1
                    )
                });
                assert!(
                    error.to_string().contains(exceptions[exception].1),
                    "exception {name:?} failed for a different reason: {error}"
                );
                assert!(!seen[exception], "duplicate exception {name:?}");
                seen[exception] = true;
                eprintln!("known positive rejection {name:?}: {error}");
            }
        }
    }
    assert!(
        seen.into_iter().all(|seen| seen),
        "missing documented fixture exceptions"
    );
    assert_eq!((accepted_positive, rejected_diagnostic), (195, 414));
    eprintln!(
        "Pinned fixture classification: {accepted_positive} positive round-trips, \
         {} classified positive rejections, {rejected_diagnostic} diagnostic-bearing rejections; \
         no upstream executable or exact diagnostic comparison.",
        exceptions.len()
    );
}
