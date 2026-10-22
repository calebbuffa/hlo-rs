use pliron::builtin::types::Signedness as IntegerSignedness;
use stablehlo::integer::IntegerValue;

#[test]
fn decimal_and_apint_word_conversions_preserve_wide_values() {
    let unsigned = IntegerValue::parse_decimal("340282366920938463463374607431768211455");
    assert!(unsigned.is_some());
    if let Some(unsigned) = unsigned {
        assert_eq!(
            unsigned.to_string(),
            "340282366920938463463374607431768211455"
        );
        assert_eq!(
            u128::from(unsigned.to_twos_complement_words(128)[0]),
            u64::MAX as u128
        );
        assert_eq!(
            IntegerValue::from_twos_complement_words(
                &unsigned.to_twos_complement_words(128),
                128,
                false
            ),
            unsigned
        );
        assert_eq!(unsigned.to_i128(), None);
    }

    let below_i128_min = IntegerValue::parse_decimal("-170141183460469231731687303715884105729");
    assert!(below_i128_min.is_some());
    if let Some(below_i128_min) = below_i128_min {
        assert!(below_i128_min.fits_width(129, IntegerSignedness::Signed));
        assert!(!below_i128_min.fits_width(128, IntegerSignedness::Signed));
        let words = below_i128_min.to_twos_complement_words(129);
        assert_eq!(
            IntegerValue::from_twos_complement_words(&words, 129, true),
            below_i128_min
        );
        let (seven, eight) = (IntegerValue::from(7_i128), IntegerValue::from(8_i128));
        let (minus_seven, minus_eight) = (IntegerValue::from(-7_i128), IntegerValue::from(-8_i128));
        assert!(eight > seven);
        assert!(minus_eight < minus_seven);
    }
}

#[test]
fn primitive_conversions_cover_full_signed_and_unsigned_ranges() {
    let max_u128 = IntegerValue::from(u128::MAX);
    assert_eq!(max_u128.to_u128(), Some(u128::MAX));
    assert_eq!(u128::try_from(&max_u128).ok(), Some(u128::MAX));
    assert!(i128::try_from(&max_u128).is_err());

    let min_i128 = IntegerValue::from(i128::MIN);
    assert_eq!(i128::try_from(&min_i128).ok(), Some(i128::MIN));
    assert!(u128::try_from(&min_i128).is_err());

    assert_eq!(
        i8::try_from(IntegerValue::from(i8::MIN)).ok(),
        Some(i8::MIN)
    );
    assert_eq!(
        u8::try_from(IntegerValue::from(u8::MAX)).ok(),
        Some(u8::MAX)
    );
    assert!(i8::try_from(IntegerValue::from(i8::MAX as i16 + 1)).is_err());
    assert!(u8::try_from(IntegerValue::from(-1_i8)).is_err());
}

#[test]
fn integer_value_supports_standard_parse_and_negation_traits() {
    let parsed = "+340282366920938463463374607431768211455"
        .parse::<IntegerValue>()
        .expect("valid decimal should parse");
    assert_eq!(parsed, IntegerValue::from(u128::MAX));
    assert!("-".parse::<IntegerValue>().is_err());
    assert_eq!(-IntegerValue::from(0_i8), IntegerValue::ZERO);
    assert_eq!(
        (-IntegerValue::from(i128::MIN)).to_string(),
        "170141183460469231731687303715884105728"
    );
    assert_eq!(
        -IntegerValue::from(u128::MAX),
        IntegerValue::parse_decimal("-340282366920938463463374607431768211455")
            .expect("negative wide decimal should parse")
    );
}
