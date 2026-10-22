use clienthlo::{
    attributes::{
        ComparisonDirection, ComparisonDirectionAttr, ComparisonKind, ComparisonKindAttr,
        Precision, PrecisionAttr, RaggedDotDimensionNumbersAttr, RaggedDotDimensions,
        precision_config, read_precision_config,
    },
    pliron::{
        attribute::{AttrObj, Attribute},
        common_traits::Verify,
        context::Context,
        parsable::{Parsable, parse_from_str},
        printable::Printable,
    },
};

fn roundtrip<T: Attribute + PartialEq + std::fmt::Debug + Clone>(ctx: &mut Context, value: T) {
    value.verify(ctx).unwrap();
    let attr: AttrObj = Box::new(value.clone());
    let text = attr.disp(ctx).to_string();
    let parsed = parse_from_str(AttrObj::parser(()), ctx, &text)
        .unwrap_or_else(|error| panic!("{text}: {error}"));
    parsed.verify(ctx).unwrap();
    assert_eq!(parsed.downcast_ref::<T>().unwrap(), &value, "{text}");
    assert_eq!(parsed.disp(ctx).to_string(), text);
}

fn text<T: Attribute + 'static>(value: T, ctx: &Context) -> String {
    let attr: AttrObj = Box::new(value);
    attr.disp(ctx).to_string()
}

#[test]
fn chlo_enum_attributes_use_upstream_symbols_and_roundtrip() {
    let mut ctx = Context::new();
    for (value, spelling) in [
        (ComparisonDirection::Eq, "EQ"),
        (ComparisonDirection::Ne, "NE"),
        (ComparisonDirection::Ge, "GE"),
        (ComparisonDirection::Gt, "GT"),
        (ComparisonDirection::Le, "LE"),
        (ComparisonDirection::Lt, "LT"),
    ] {
        let attr = ComparisonDirectionAttr::new(value);
        assert_eq!(value.as_str(), spelling);
        assert_eq!(spelling.parse::<ComparisonDirection>(), Ok(value));
        assert_eq!(attr.numeric_value(), u64::from(value));
        assert_eq!(
            text(attr, &ctx),
            format!("chlo.comparison_direction {spelling}")
        );
        roundtrip(&mut ctx, attr);
    }
    for (value, spelling) in [
        (ComparisonKind::NoType, "NOTYPE"),
        (ComparisonKind::Float, "FLOAT"),
        (ComparisonKind::TotalOrder, "TOTALORDER"),
        (ComparisonKind::Signed, "SIGNED"),
        (ComparisonKind::Unsigned, "UNSIGNED"),
        (ComparisonKind::WeakOrder, "WEAKORDER"),
    ] {
        let attr = ComparisonKindAttr::new(value);
        assert_eq!(value.as_str(), spelling);
        assert_eq!(spelling.parse::<ComparisonKind>(), Ok(value));
        assert_eq!(text(attr, &ctx), format!("chlo.comparison_type {spelling}"));
        roundtrip(&mut ctx, attr);
    }
    for (value, spelling) in [
        (Precision::Default, "DEFAULT"),
        (Precision::High, "HIGH"),
        (Precision::Highest, "HIGHEST"),
    ] {
        let attr = PrecisionAttr::new(value);
        assert_eq!(value.as_str(), spelling);
        assert_eq!(spelling.parse::<Precision>(), Ok(value));
        assert_eq!(text(attr, &ctx), format!("chlo.precision {spelling}"));
        roundtrip(&mut ctx, attr);
    }
}

#[test]
fn ragged_dot_attribute_uses_named_upstream_fields_and_roundtrips() {
    let mut ctx = Context::new();
    let attr = RaggedDotDimensionNumbersAttr::new(RaggedDotDimensions {
        lhs_batching_dimensions: vec![0, 2],
        rhs_batching_dimensions: vec![1],
        lhs_contracting_dimensions: vec![3],
        rhs_contracting_dimensions: vec![0],
        lhs_ragged_dimensions: vec![2],
        rhs_group_dimensions: vec![],
    });
    assert_eq!(
        attr.dimensions(),
        RaggedDotDimensions {
            lhs_batching_dimensions: vec![0, 2],
            rhs_batching_dimensions: vec![1],
            lhs_contracting_dimensions: vec![3],
            rhs_contracting_dimensions: vec![0],
            lhs_ragged_dimensions: vec![2],
            rhs_group_dimensions: vec![],
        }
    );
    let rendered = text(attr.clone(), &ctx);
    assert_eq!(
        rendered,
        "chlo.ragged_dot <lhs_batching_dimensions=[0, 2],rhs_batching_dimensions=[1],lhs_contracting_dimensions=[3],rhs_contracting_dimensions=[0],lhs_ragged_dimensions=[2]>"
    );
    roundtrip(&mut ctx, attr);
    let empty = RaggedDotDimensionNumbersAttr::new(RaggedDotDimensions {
        lhs_batching_dimensions: vec![],
        rhs_batching_dimensions: vec![],
        lhs_contracting_dimensions: vec![],
        rhs_contracting_dimensions: vec![],
        lhs_ragged_dimensions: vec![],
        rhs_group_dimensions: vec![],
    });
    assert_eq!(text(empty.clone(), &ctx), "chlo.ragged_dot <>");
    roundtrip(&mut ctx, empty);

    for (input, expected) in [
        (
            "chlo.ragged_dot <>",
            RaggedDotDimensionNumbersAttr::new(RaggedDotDimensions {
                lhs_batching_dimensions: vec![],
                rhs_batching_dimensions: vec![],
                lhs_contracting_dimensions: vec![],
                rhs_contracting_dimensions: vec![],
                lhs_ragged_dimensions: vec![],
                rhs_group_dimensions: vec![],
            }),
        ),
        (
            "chlo.ragged_dot <rhs_group_dimensions = [3], lhs_ragged_dimensions = [1]>",
            RaggedDotDimensionNumbersAttr::new(RaggedDotDimensions {
                lhs_batching_dimensions: vec![],
                rhs_batching_dimensions: vec![],
                lhs_contracting_dimensions: vec![],
                rhs_contracting_dimensions: vec![],
                lhs_ragged_dimensions: vec![1],
                rhs_group_dimensions: vec![3],
            }),
        ),
        (
            "chlo.ragged_dot <lhs_batching_dimensions = []>",
            RaggedDotDimensionNumbersAttr::new(RaggedDotDimensions {
                lhs_batching_dimensions: vec![],
                rhs_batching_dimensions: vec![],
                lhs_contracting_dimensions: vec![],
                rhs_contracting_dimensions: vec![],
                lhs_ragged_dimensions: vec![],
                rhs_group_dimensions: vec![],
            }),
        ),
    ] {
        let parsed = parse_from_str(AttrObj::parser(()), &mut ctx, input).unwrap();
        assert_eq!(
            parsed.downcast_ref::<RaggedDotDimensionNumbersAttr>(),
            Some(&expected)
        );
        assert_eq!(parsed.disp(&ctx).to_string(), text(expected, &ctx));
    }

    assert!(
        parse_from_str(
            AttrObj::parser(()),
            &mut ctx,
            "chlo.ragged_dot <lhs_batching_dimensions = [0], lhs_batching_dimensions = [1]>"
        )
        .is_err()
    );
}

#[test]
fn enum_values_convert_losslessly_to_stablehlo_and_precision_config_is_typed() {
    for direction in [
        ComparisonDirection::Eq,
        ComparisonDirection::Ne,
        ComparisonDirection::Ge,
        ComparisonDirection::Gt,
        ComparisonDirection::Le,
        ComparisonDirection::Lt,
    ] {
        let stable: stablehlo::attributes::ComparisonDirection = direction.into();
        assert_eq!(ComparisonDirection::from(stable), direction);
    }

    for comparison_kind in [
        ComparisonKind::NoType,
        ComparisonKind::Float,
        ComparisonKind::TotalOrder,
        ComparisonKind::Signed,
        ComparisonKind::Unsigned,
        ComparisonKind::WeakOrder,
    ] {
        let stable: stablehlo::attributes::ComparisonKind = comparison_kind.into();
        assert_eq!(ComparisonKind::from(stable), comparison_kind);
    }

    let config = precision_config([Precision::Default, Precision::High, Precision::Highest]);
    assert_eq!(
        read_precision_config(&config),
        vec![Precision::Default, Precision::High, Precision::Highest]
    );
    let empty = precision_config([]);
    assert_eq!(read_precision_config(&empty), vec![]);
    assert_eq!(
        text(config.clone(), &Context::new()),
        "chlo.precision_config [DEFAULT, HIGH, HIGHEST]"
    );
    roundtrip(&mut Context::new(), config);
}

#[test]
fn chlo_enums_convert_to_and_from_numeric_storage_without_loss() {
    for (value, numeric) in [
        (ComparisonDirection::Eq, 0),
        (ComparisonDirection::Ne, 1),
        (ComparisonDirection::Ge, 2),
        (ComparisonDirection::Gt, 3),
        (ComparisonDirection::Le, 4),
        (ComparisonDirection::Lt, 5),
    ] {
        assert_eq!(u64::from(value), numeric);
        assert_eq!(ComparisonDirection::try_from(numeric), Ok(value));
    }
    let error = ComparisonDirection::try_from(6_u64).unwrap_err();
    assert_eq!(error.enum_name, "comparison_direction");
    assert_eq!(error.value, 6);
    assert_eq!(
        error.to_string(),
        "unknown comparison_direction numeric encoding 6"
    );
    let _: &dyn std::error::Error = &error;

    for (value, numeric) in [
        (ComparisonKind::NoType, 0),
        (ComparisonKind::Float, 1),
        (ComparisonKind::TotalOrder, 2),
        (ComparisonKind::Signed, 3),
        (ComparisonKind::Unsigned, 4),
        (ComparisonKind::WeakOrder, 5),
    ] {
        assert_eq!(u64::from(value), numeric);
        assert_eq!(ComparisonKind::try_from(numeric), Ok(value));
    }
    let error = ComparisonKind::try_from(6_u64).unwrap_err();
    assert_eq!(error.enum_name, "comparison_type");
    assert_eq!(error.value, 6);

    for (value, numeric) in [
        (Precision::Default, 0),
        (Precision::High, 1),
        (Precision::Highest, 2),
    ] {
        assert_eq!(u64::from(value), numeric);
        assert_eq!(Precision::try_from(numeric), Ok(value));
    }
    let error = Precision::try_from(3_u64).unwrap_err();
    assert_eq!(error.enum_name, "precision");
    assert_eq!(error.value, 3);
}

#[test]
fn ragged_dot_attribute_rejects_negative_dimensions_with_structured_diagnostic() {
    let ctx = Context::new();
    let attr = RaggedDotDimensionNumbersAttr::new(RaggedDotDimensions {
        lhs_batching_dimensions: vec![],
        rhs_batching_dimensions: vec![],
        lhs_contracting_dimensions: vec![-1],
        rhs_contracting_dimensions: vec![],
        lhs_ragged_dimensions: vec![],
        rhs_group_dimensions: vec![],
    });
    let error = attr.verify(&ctx).unwrap_err();
    assert!(matches!(
        error.err.downcast_ref::<clienthlo::Diagnostic>(),
        Some(clienthlo::Diagnostic::InvalidOperation {
            operation,
            reason: "dimensions must be non-negative",
            ..
        }) if operation == "CHLO ragged-dot dimension numbers"
    ));
}

#[test]
fn enum_attribute_parser_rejects_non_upstream_spellings() {
    let mut ctx = Context::new();
    assert!("eq".parse::<ComparisonDirection>().is_err());
    assert!("NO_TYPE".parse::<ComparisonKind>().is_err());
    assert!("default".parse::<Precision>().is_err());
    assert!(
        parse_from_str(
            AttrObj::parser(()),
            &mut ctx,
            "chlo.comparison_direction eq"
        )
        .is_err()
    );
    assert!(parse_from_str(AttrObj::parser(()), &mut ctx, "chlo.precision 3").is_err());
    assert!(
        parse_from_str(
            AttrObj::parser(()),
            &mut ctx,
            "chlo.precision_config [DEFAULT, 1]"
        )
        .is_err()
    );
    assert!(
        parse_from_str(
            AttrObj::parser(()),
            &mut ctx,
            "chlo.precision_config [DEFAULT, UNKNOWN]"
        )
        .is_err()
    );
    assert!(
        parse_from_str(
            AttrObj::parser(()),
            &mut ctx,
            "chlo.precision_config [chlo.precision HIGH]"
        )
        .is_err()
    );
}

#[test]
fn enum_display_uses_canonical_symbols_and_parse_errors_preserve_input() {
    assert_eq!(ComparisonDirection::Eq.to_string(), "EQ");
    assert_eq!(ComparisonKind::NoType.to_string(), "NOTYPE");
    assert_eq!(Precision::Highest.to_string(), "HIGHEST");
    let error = " default ".parse::<Precision>().unwrap_err();
    assert_eq!(error.enum_name(), "precision");
    assert_eq!(error.spelling(), " default ");
    let _: &dyn std::error::Error = &error;
}
