use clienthlo::{
    attributes::{
        ComparisonDirection, ComparisonDirectionAttr, ComparisonType, ComparisonTypeAttr,
        Precision, PrecisionAttr, RaggedDotDimensionNumbersAttr, precision_config,
        read_precision_config,
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
        assert_eq!(attr.numeric_value(), value as u64);
        assert_eq!(
            text(attr, &ctx),
            format!("chlo.comparison_direction {spelling}")
        );
        roundtrip(&mut ctx, attr);
    }
    for (value, spelling) in [
        (ComparisonType::NoType, "NOTYPE"),
        (ComparisonType::Float, "FLOAT"),
        (ComparisonType::TotalOrder, "TOTALORDER"),
        (ComparisonType::Signed, "SIGNED"),
        (ComparisonType::Unsigned, "UNSIGNED"),
        (ComparisonType::WeakOrder, "WEAKORDER"),
    ] {
        let attr = ComparisonTypeAttr::new(value);
        assert_eq!(text(attr, &ctx), format!("chlo.comparison_type {spelling}"));
        roundtrip(&mut ctx, attr);
    }
    for (value, spelling) in [
        (Precision::Default, "DEFAULT"),
        (Precision::High, "HIGH"),
        (Precision::Highest, "HIGHEST"),
    ] {
        let attr = PrecisionAttr::new(value);
        assert_eq!(text(attr, &ctx), format!("chlo.precision {spelling}"));
        roundtrip(&mut ctx, attr);
    }
}

#[test]
fn ragged_dot_attribute_uses_named_upstream_fields_and_roundtrips() {
    let mut ctx = Context::new();
    let attr = RaggedDotDimensionNumbersAttr::new(
        &ctx,
        vec![0, 2],
        vec![1],
        vec![3],
        vec![0],
        vec![2],
        vec![],
    );
    assert_eq!(
        attr.dimensions(),
        (vec![0, 2], vec![1], vec![3], vec![0], vec![2], vec![])
    );
    let rendered = text(attr.clone(), &ctx);
    assert_eq!(
        rendered,
        "chlo.ragged_dot <lhs_batching_dimensions=[0, 2],rhs_batching_dimensions=[1],lhs_contracting_dimensions=[3],rhs_contracting_dimensions=[0],lhs_ragged_dimensions=[2]>"
    );
    roundtrip(&mut ctx, attr);
    let empty =
        RaggedDotDimensionNumbersAttr::new(&ctx, vec![], vec![], vec![], vec![], vec![], vec![]);
    assert_eq!(text(empty.clone(), &ctx), "chlo.ragged_dot <>");
    roundtrip(&mut ctx, empty);

    for (input, expected) in [
        (
            "chlo.ragged_dot <>",
            RaggedDotDimensionNumbersAttr::new(
                &ctx,
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
            ),
        ),
        (
            "chlo.ragged_dot <rhs_group_dimensions = [3], lhs_ragged_dimensions = [1]>",
            RaggedDotDimensionNumbersAttr::new(
                &ctx,
                vec![],
                vec![],
                vec![],
                vec![],
                vec![1],
                vec![3],
            ),
        ),
        (
            "chlo.ragged_dot <lhs_batching_dimensions = []>",
            RaggedDotDimensionNumbersAttr::new(
                &ctx,
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
                vec![],
            ),
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
    for comparison_type in [
        ComparisonType::NoType,
        ComparisonType::Float,
        ComparisonType::TotalOrder,
        ComparisonType::Signed,
        ComparisonType::Unsigned,
        ComparisonType::WeakOrder,
    ] {
        let stable: stablehlo::attributes::ComparisonType = comparison_type.into();
        assert_eq!(ComparisonType::from(stable), comparison_type);
    }

    let config = precision_config([Precision::Default, Precision::High, Precision::Highest]);
    assert_eq!(
        read_precision_config(&config),
        Some(vec![
            Precision::Default,
            Precision::High,
            Precision::Highest
        ])
    );
    let empty = precision_config([]);
    assert_eq!(read_precision_config(&empty), Some(vec![]));
    assert_eq!(
        text(config.clone(), &Context::new()),
        "chlo.precision_config [DEFAULT, HIGH, HIGHEST]"
    );
    roundtrip(&mut Context::new(), config);
}

#[test]
fn ragged_dot_attribute_rejects_negative_dimensions_with_structured_diagnostic() {
    let ctx = Context::new();
    let attr =
        RaggedDotDimensionNumbersAttr::new(&ctx, vec![], vec![], vec![-1], vec![], vec![], vec![]);
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
