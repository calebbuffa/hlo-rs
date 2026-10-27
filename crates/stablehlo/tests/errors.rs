use stablehlo::{
    Diagnostic, Error, RecoveryCategory, Result,
    attributes::{EnumParseError, Precision},
    integer::{IntegerParseError, IntegerValue},
};

#[test]
fn inference_propagates_the_original_type_validation_failure() {
    use stablehlo::{
        inference::{infer_elementwise, infer_tuple_type},
        pliron::{context::Context, r#type::TypeHandle},
        types::{ElementType, TensorType},
    };

    let ctx = Context::new();
    let malformed: TypeHandle =
        TensorType::get(&ctx, vec![-2], ElementType::F32.native(&ctx)).into();
    let expected = malformed.deref(&ctx).verify(&ctx).unwrap_err();
    let tuple_error = infer_tuple_type(&ctx, &[malformed]).unwrap_err();
    let elementwise_error = infer_elementwise(&ctx, "add", &[malformed]).unwrap_err();
    assert_eq!(tuple_error.to_string(), expected.to_string());
    assert_eq!(elementwise_error.to_string(), expected.to_string());
    let repaired = TensorType::get(&ctx, vec![2], ElementType::F32.native(&ctx)).into();
    infer_tuple_type(&ctx, &[repaired]).unwrap();
    infer_elementwise(&ctx, "add", &[repaired]).unwrap();
}

#[test]
fn parser_errors_propagate_without_losing_their_payloads() {
    fn precision(text: &str) -> Result<Precision> {
        Ok(text.parse()?)
    }
    fn integer(text: &str) -> Result<IntegerValue> {
        Ok(text.parse()?)
    }
    let error = precision("invalid").unwrap_err();
    let details = error.err.downcast_ref::<EnumParseError>().unwrap();
    assert_eq!(details.enum_name(), "precision");
    assert_eq!(details.spelling(), "invalid");
    assert!(Diagnostic::from_error(&error).is_none());
    let error = integer("not an integer").unwrap_err();
    assert!(error.err.downcast_ref::<IntegerParseError>().is_some());
}

#[test]
fn diagnostics_remain_accessible_through_framework_and_standard_wrappers() {
    #[derive(Debug, thiserror::Error)]
    #[error("consumer context: {0}")]
    struct Context(#[source] Error);

    let original: Error = Diagnostic::UnsupportedType.into();
    let error = stablehlo::pliron::arg_error_noloc!(Context(original));
    let diagnostic = Diagnostic::from_error(&error).unwrap();
    assert!(matches!(diagnostic, Diagnostic::UnsupportedType));
    assert_eq!(
        diagnostic.recovery_category(),
        RecoveryCategory::Unsupported
    );
    assert!(std::error::Error::source(error.err.as_ref()).is_some());
}
