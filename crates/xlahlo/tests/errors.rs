use xlahlo::{
    Diagnostic, Error, RecoveryCategory, Result,
    attributes::{ChannelKind, ChannelKindValueError, EnumParseError, FftKind},
    evaluator::EvaluationError,
    types::{
        ElementType, ElementTypeParseError, PrimitiveType, PrimitiveTypeNameError,
        PrimitiveTypeValueError,
    },
};

#[test]
fn leaf_errors_propagate_without_string_conversion() {
    fn element(text: &str) -> Result<ElementType> {
        Ok(text.parse()?)
    }
    fn primitive(text: &str) -> Result<PrimitiveType> {
        Ok(text.parse()?)
    }
    fn primitive_id(value: i32) -> Result<PrimitiveType> {
        Ok(value.try_into()?)
    }
    fn channel(value: i64) -> Result<ChannelKind> {
        Ok(value.try_into()?)
    }
    fn fft_kind(text: &str) -> Result<FftKind> {
        Ok(text.parse()?)
    }
    let error = element("tuple").unwrap_err();
    assert_eq!(
        error
            .err
            .downcast_ref::<ElementTypeParseError>()
            .unwrap()
            .spelling(),
        "tuple"
    );
    let error = primitive("unknown").unwrap_err();
    assert_eq!(
        error
            .err
            .downcast_ref::<PrimitiveTypeNameError>()
            .unwrap()
            .spelling(),
        "unknown"
    );
    let error = primitive_id(i32::MAX).unwrap_err();
    assert_eq!(
        error
            .err
            .downcast_ref::<PrimitiveTypeValueError>()
            .unwrap()
            .value(),
        i32::MAX
    );
    let error = channel(i64::MAX).unwrap_err();
    assert_eq!(
        error
            .err
            .downcast_ref::<ChannelKindValueError>()
            .unwrap()
            .value(),
        i64::MAX
    );
    let error = fft_kind("invalid").unwrap_err();
    assert_eq!(
        error
            .err
            .downcast_ref::<EnumParseError>()
            .unwrap()
            .spelling(),
        "invalid"
    );
}

#[test]
fn recovery_guidance_survives_wrapping_and_evaluation_conversion() {
    #[derive(Debug, thiserror::Error)]
    #[error("consumer context: {0}")]
    struct Context(#[source] Error);

    let original: Error = Diagnostic::UnsupportedCallback {
        field: "shape_size",
    }
    .into();
    let error = xlahlo::pliron::arg_error_noloc!(Context(original));
    assert_eq!(
        Diagnostic::from_error(&error).unwrap().recovery_category(),
        RecoveryCategory::Unsupported
    );
    let error: Error = Diagnostic::Operation {
        operation: "example".into(),
        source: Box::new(error),
    }
    .into();
    assert_eq!(
        Diagnostic::from_error(&error).unwrap().recovery_category(),
        RecoveryCategory::Unsupported
    );
    for (failure, category) in [
        (
            EvaluationError::UnknownValue,
            RecoveryCategory::MissingInput,
        ),
        (
            EvaluationError::LoopIterationLimit { limit: 2 },
            RecoveryCategory::ResourceLimit,
        ),
        (EvaluationError::Recursion, RecoveryCategory::InvalidInput),
    ] {
        let error: Error = failure.into();
        assert_eq!(
            error
                .err
                .downcast_ref::<EvaluationError>()
                .unwrap()
                .recovery_category(),
            category
        );
    }
}
