use clienthlo::{
    Diagnostic, Error, RecoveryCategory, Result,
    attributes::{ComparisonDirection, EnumParseError, EnumValueError},
};

#[test]
fn verification_retains_inference_details_and_allows_repair() {
    use clienthlo::{
        ops::{TopKOp, TopKOptions},
        pliron::{basic_block::BasicBlock, common_traits::Verify, context::Context},
    };
    use stablehlo::types::{ElementType, tensor};

    let mut ctx = Context::new();
    let input_type = tensor(&ctx, &[4], ElementType::F32);
    let block = BasicBlock::new(&mut ctx, None, vec![input_type]);
    let input = block.deref(&ctx).arguments().next().unwrap();
    let values = tensor(&ctx, &[2], ElementType::F32);
    let indices = tensor(&ctx, &[2], ElementType::I32);
    let invalid = TopKOp::new(&mut ctx, input, TopKOptions::new(8, true), values, indices);
    let error = invalid.verify(&ctx).unwrap_err();
    let diagnostic = Diagnostic::from_error(&error).unwrap();
    assert!(matches!(
        diagnostic,
        Diagnostic::OperationFailure {
            operation: "chlo.top_k",
            ..
        }
    ));
    let source = std::error::Error::source(diagnostic)
        .unwrap()
        .downcast_ref::<Error>()
        .unwrap();
    assert!(Diagnostic::from_error(source).is_some());
    let repaired = TopKOp::new(&mut ctx, input, TopKOptions::new(2, true), values, indices);
    repaired.verify(&ctx).unwrap();
}

#[test]
fn enum_errors_use_the_shared_result_boundary() {
    fn direction(text: &str) -> Result<ComparisonDirection> {
        Ok(text.parse()?)
    }
    fn encoded(value: u64) -> Result<ComparisonDirection> {
        Ok(value.try_into()?)
    }
    let error = direction("invalid").unwrap_err();
    let details = error.err.downcast_ref::<EnumParseError>().unwrap();
    assert_eq!(details.enum_name(), "comparison_direction");
    assert_eq!(details.spelling(), "invalid");
    assert!(Diagnostic::from_error(&error).is_none());
    let error = encoded(u64::MAX).unwrap_err();
    let details = error.err.downcast_ref::<EnumValueError>().unwrap();
    assert_eq!(details.enum_name, "comparison_direction");
    assert_eq!(details.value, u64::MAX);
}

#[test]
fn diagnostics_remain_accessible_through_framework_and_standard_wrappers() {
    #[derive(Debug, thiserror::Error)]
    #[error("consumer context: {0}")]
    struct Context(#[source] Error);

    let original: Error = Diagnostic::InvalidDomain {
        operation: "chlo.broadcast_add",
    }
    .into();
    let error = clienthlo::pliron::arg_error_noloc!(Context(original));
    let diagnostic = Diagnostic::from_error(&error).unwrap();
    assert!(matches!(
        diagnostic,
        Diagnostic::InvalidDomain {
            operation: "chlo.broadcast_add"
        }
    ));
    assert_eq!(
        diagnostic.recovery_category(),
        RecoveryCategory::InvalidInput
    );
}
